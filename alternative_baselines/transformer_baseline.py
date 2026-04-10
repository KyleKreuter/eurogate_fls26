"""Transformer baseline for Reefer Peak Load Challenge.

Upgrades over the first transformer baseline:
- validation split + early stopping + LR scheduler
- lower default LR and longer training
- scheduled-sampling style robustness for one-step training
- richer autoregressive observed channels (lag/rolling stats)
- optional log1p target transform
- p90 residual calibration on validation history

Run:
    uv run python lightgbm/transformer_baseline.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Allow importing project modules when this file is executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from lightgbm.baseline import (  # type: ignore
        BASELINE_OUT,
        REEFER_CSV,
        TARGET_COL,
        TARGET_CSV,
        load_hourly_total,
    )
except ImportError:
    from baseline import (  # type: ignore
        BASELINE_OUT,
        REEFER_CSV,
        TARGET_COL,
        TARGET_CSV,
        load_hourly_total,
    )

from time_series_transformer.model import HTTransformerRegressor


TRANSFORMER_OUT = BASELINE_OUT.parent / "transformer_baseline.csv"


@dataclass
class TrainConfig:
    seq_len: int = 168
    batch_size: int = 160
    epochs: int = 16
    lr: float = 2e-5
    weight_decay: float = 8e-4
    quantile_alpha: float = 0.9

    val_ratio: float = 0.25
    min_val_samples: int = 168
    early_stopping_patience: int = 3
    scheduler_patience: int = 1
    scheduler_factor: float = 0.3
    min_lr: float = 5e-6

    scheduled_sampling_start: float = 0.02
    scheduled_sampling_end: float = 0.10
    autoreg_train_fraction: float = 0.35
    max_train_rollout_steps: int = 2
    rollout_scale_steps: int = 240

    # Add mild noise to observed channels during training for robustness.
    obs_noise_std: float = 0.01

    use_log_target: bool = True
    peak_weight: float = 1.0
    peak_quantile: float = 0.85

    # Stabilize point forecast with persistence blend.
    point_blend_model: float = 0.08
    point_blend_lag24: float = 0.72
    point_blend_lag168: float = 0.20
    point_peak_trigger_quantile: float = 0.85
    point_peak_model_boost: float = 0.25
    point_bias_calibration_quantile: float = 0.5

    # Fast path: skip separate p90 model and use calibrated residual offset.
    train_p90_model: bool = False

    # p90 = blend * p90_model + (1 - blend) * (point + calibrated_offset)
    p90_blend: float = 0.5
    p90_calibration_quantile: float = 0.92
    p90_vol_mult: float = 0.06
    p90_min_margin_kw: float = 4.0

    # Transformer capacity (point model / MAE model)
    t2v_dim: int = 8
    point_model_dim: int = 160
    point_num_heads: int = 16
    point_num_layers: int = 8
    point_dropout: float = 0.2
    point_dim_feedforward: int = 512

    # Transformer capacity (p90 / peak model)
    p90_model_dim: int = 96
    p90_num_heads: int = 4
    p90_num_layers: int = 4
    p90_dropout: float = 0.2
    p90_dim_feedforward: int = 256

    device: str = "auto"


@dataclass
class ObsNormStats:
    mean: torch.Tensor
    std: torch.Tensor


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _print_progress(label: str, current: int, total: int) -> None:
    """Render a simple in-place progress indicator in the terminal."""
    total = max(total, 1)
    current = min(max(current, 0), total)
    width = 28
    ratio = current / total
    filled = int(width * ratio)
    bar = "#" * filled + "." * (width - filled)
    end = "\n" if current >= total else "\r"
    print(
        f"[{label}] [{bar}] {current}/{total} ({ratio * 100:5.1f}%)",
        end=end,
        flush=True,
    )


def _normalized_time_features(
    ts: pd.Series,
    rollout_steps: np.ndarray | None = None,
    rollout_scale_steps: int = 1,
) -> np.ndarray:
    """Build normalized timestamp features for Time2Vec.

    Base order:
    - hour_norm
    - dow_norm
    - month_norm
    - dayofyear_norm

    Optional:
    - rollout_norm (normalized count of self-generated forecast steps)
    """
    hour_norm = ts.dt.hour.to_numpy(dtype=np.float32) / 23.0
    dow_norm = ts.dt.dayofweek.to_numpy(dtype=np.float32) / 6.0
    month_norm = (ts.dt.month.to_numpy(dtype=np.float32) - 1.0) / 11.0
    dayofyear_norm = (ts.dt.dayofyear.to_numpy(dtype=np.float32) - 1.0) / 365.0

    feats = np.stack([hour_norm, dow_norm, month_norm, dayofyear_norm], axis=1)
    if rollout_steps is not None:
        rollout_steps = np.asarray(rollout_steps, dtype=np.float32).reshape(-1)
        if rollout_steps.shape[0] != feats.shape[0]:
            raise ValueError("rollout_steps length must match timestamp length")
        scale = float(max(1, rollout_scale_steps))
        rollout_norm = np.clip(rollout_steps / scale, 0.0, 1.0)
        feats = np.concatenate([feats, rollout_norm[:, None]], axis=1)
    return np.clip(feats, 0.0, 1.0).astype(np.float32, copy=False)


def _target_transform(y: np.ndarray, use_log_target: bool) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if not use_log_target:
        return y
    return np.log1p(np.clip(y, a_min=0.0, a_max=None)).astype(np.float32, copy=False)


def _target_inverse(y_t: np.ndarray, use_log_target: bool) -> np.ndarray:
    y_t = np.asarray(y_t, dtype=np.float64)
    if not use_log_target:
        return y_t
    return np.expm1(y_t)


def _build_hourly_context_features(csv_path: Path) -> pd.DataFrame:
    """Build hourly context aggregates from raw reefer data.

    These are merged on timestamp and used only as historical observed channels
    during autoregressive forecasting.
    """
    usecols = [
        "EventTime",
        "container_uuid",
        "container_visit_uuid",
        "HardwareType",
        "TemperatureSetPoint",
        "TemperatureAmbient",
        "TemperatureReturn",
        "RemperatureSupply",
        "ContainerSize",
        "stack_tier",
    ]
    raw = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        usecols=lambda c: c in set(usecols),
        low_memory=False,
    )

    raw["EventTime"] = pd.to_datetime(raw["EventTime"], utc=True)
    raw["ts"] = raw["EventTime"].dt.floor("1h")

    if "container_uuid" in raw.columns:
        container_key = raw["container_uuid"].astype("string")
    else:
        container_key = pd.Series(pd.NA, index=raw.index, dtype="string")

    if "container_visit_uuid" in raw.columns:
        visit_key = raw["container_visit_uuid"].astype("string")
    else:
        visit_key = pd.Series(pd.NA, index=raw.index, dtype="string")

    raw["container_key"] = container_key.fillna(visit_key)

    for col in [
        "TemperatureSetPoint",
        "TemperatureAmbient",
        "TemperatureReturn",
        "RemperatureSupply",
        "stack_tier",
    ]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    if "ContainerSize" in raw.columns:
        size_num = pd.to_numeric(raw["ContainerSize"], errors="coerce")
        raw["is_size_40"] = (size_num >= 40).astype(np.float32)
    else:
        raw["is_size_40"] = 0.0

    raw["temp_gap_ambient_setpoint"] = raw.get("TemperatureAmbient") - raw.get(
        "TemperatureSetPoint"
    )
    raw["temp_gap_return_supply"] = raw.get("TemperatureReturn") - raw.get(
        "RemperatureSupply"
    )

    agg = (
        raw.groupby("ts", sort=True)
        .agg(
            total_container_count=("container_key", "nunique"),
            active_visit_count=("container_visit_uuid", "nunique"),
            hardware_type_count=("HardwareType", "nunique"),
            temp_gap_ambient_setpoint_mean=("temp_gap_ambient_setpoint", "mean"),
            temp_gap_return_supply_mean=("temp_gap_return_supply", "mean"),
            stack_tier_mean=("stack_tier", "mean"),
            container_size_40_share=("is_size_40", "mean"),
        )
        .reset_index()
    )

    for c in agg.columns:
        if c != "ts":
            agg[c] = (
                agg[c]
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .fillna(0.0)
                .astype(np.float32)
            )

    return agg


def _normalize_point_blend_weights(cfg: TrainConfig) -> tuple[float, float, float]:
    w_model = float(max(cfg.point_blend_model, 0.0))
    w_lag24 = float(max(cfg.point_blend_lag24, 0.0))
    w_lag168 = float(max(cfg.point_blend_lag168, 0.0))
    s = w_model + w_lag24 + w_lag168
    if s <= 1e-12:
        return 1.0, 0.0, 0.0
    return w_model / s, w_lag24 / s, w_lag168 / s


def _blend_point_with_persistence(
    *,
    model_pred_raw: np.ndarray,
    history_raw: list[float],
    cfg: TrainConfig,
    peak_threshold_kw: float | None = None,
) -> np.ndarray:
    """Blend model point prediction with lag-24/lag-168 persistence recursively."""
    w_model_base, w_lag24_base, w_lag168_base = _normalize_point_blend_weights(cfg)
    hist = [float(v) for v in history_raw]
    out: list[float] = []

    for p in np.asarray(model_pred_raw, dtype=np.float64):
        last = hist[-1] if hist else float(p)
        lag24 = hist[-24] if len(hist) >= 24 else last
        lag168 = hist[-168] if len(hist) >= 168 else lag24

        w_model = w_model_base
        w_lag24 = w_lag24_base
        w_lag168 = w_lag168_base
        if peak_threshold_kw is not None and max(lag24, lag168) >= peak_threshold_kw:
            w_model = min(0.85, w_model_base + max(0.0, cfg.point_peak_model_boost))
            rem = max(0.0, 1.0 - w_model)
            lag_sum = w_lag24_base + w_lag168_base
            if lag_sum > 1e-12:
                w_lag24 = rem * (w_lag24_base / lag_sum)
                w_lag168 = rem * (w_lag168_base / lag_sum)
            else:
                w_lag24 = rem
                w_lag168 = 0.0

        blend = w_model * float(p) + w_lag24 * lag24 + w_lag168 * lag168
        blend = max(0.0, blend)
        out.append(blend)
        hist.append(blend)

    return np.asarray(out, dtype=np.float64)


def _build_p90_from_point(
    *,
    point_raw: np.ndarray,
    history_raw: list[float],
    base_offset_kw: float,
    cfg: TrainConfig,
) -> np.ndarray:
    """Construct p90 from point forecast with calibrated + volatility margin."""
    hist = [float(v) for v in history_raw]
    out: list[float] = []

    for p in np.asarray(point_raw, dtype=np.float64):
        if len(hist) >= 24:
            vol = float(np.std(np.asarray(hist[-24:], dtype=np.float64)))
        elif hist:
            vol = float(np.std(np.asarray(hist, dtype=np.float64)))
        else:
            vol = 0.0

        margin = max(cfg.p90_min_margin_kw, base_offset_kw + cfg.p90_vol_mult * vol)
        q = max(float(p), float(p) + margin)
        out.append(q)
        hist.append(float(p))

    return np.asarray(out, dtype=np.float64)


def _build_observed_features(y_t: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    """Build observed channels using a strict 24-hour lag window.

    Base channels:
    - y_t
    - lag1
    - lag24
    - rolling_mean_24_prev
    - rolling_std_24_prev
    - delta1 (y_t - lag1)
    - context channels (hourly aggregates)
    """
    y_t = np.asarray(y_t, dtype=np.float32)
    ctx = np.asarray(ctx, dtype=np.float32)
    if ctx.ndim != 2 or ctx.shape[0] != y_t.shape[0]:
        raise ValueError("ctx must have shape (len(y_t), n_features)")

    n = y_t.shape[0]
    idx = np.arange(n, dtype=np.int64)

    lag1 = y_t[np.maximum(idx - 1, 0)]
    lag24 = y_t[np.maximum(idx - 24, 0)]

    roll_mean = np.empty(n, dtype=np.float32)
    roll_std = np.empty(n, dtype=np.float32)
    for i in range(n):
        start = max(0, i - 24)
        window = y_t[start:i]
        if window.size == 0:
            roll_mean[i] = y_t[i]
            roll_std[i] = 0.0
        else:
            roll_mean[i] = float(window.mean())
            roll_std[i] = float(window.std())

    delta1 = y_t - lag1

    base = np.stack([y_t, lag1, lag24, roll_mean, roll_std, delta1], axis=1)
    return np.concatenate([base, ctx], axis=1).astype(np.float32, copy=False)


def _build_training_tensors(
    train_df: pd.DataFrame,
    seq_len: int,
    use_log_target: bool,
    rollout_scale_steps: int,
    context_cols: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Create one-step autoregressive samples with richer observed channels."""
    y_raw = train_df[TARGET_COL].to_numpy(dtype=np.float32)
    y_t = _target_transform(y_raw, use_log_target=use_log_target)
    tfx = _normalized_time_features(train_df["ts"])
    tfy = _normalized_time_features(
        train_df["ts"],
        rollout_steps=np.zeros(len(train_df), dtype=np.float32),
        rollout_scale_steps=rollout_scale_steps,
    )
    ctx = train_df[context_cols].to_numpy(dtype=np.float32, copy=False)
    obs = _build_observed_features(y_t, ctx)

    n = len(train_df)
    if n <= seq_len:
        raise ValueError(f"Not enough training rows ({n}) for seq_len={seq_len}.")

    src_list: list[np.ndarray] = []
    tfy_list: list[np.ndarray] = []
    y_list: list[np.float32] = []
    target_indices: list[int] = []

    for i in range(seq_len, n):
        src_obs = obs[i - seq_len : i]
        src_t = tfx[i - seq_len : i]
        src = np.concatenate([src_obs, src_t], axis=1)

        src_list.append(src)
        tfy_list.append(tfy[i])
        y_list.append(y_t[i])
        target_indices.append(i)

    src_np = np.stack(src_list, axis=0).astype(np.float32, copy=False)
    tfy_np = np.stack(tfy_list, axis=0).astype(np.float32, copy=False)
    y_np = np.asarray(y_list, dtype=np.float32)
    idx_np = np.asarray(target_indices, dtype=np.int64)

    return (
        torch.from_numpy(src_np),
        torch.from_numpy(tfy_np),
        torch.from_numpy(y_np),
        idx_np,
        y_raw,
    )


def _compute_obs_stats(src_train: torch.Tensor, obs_dim: int) -> ObsNormStats:
    obs = src_train[..., :obs_dim].reshape(-1, obs_dim)
    mean = obs.mean(dim=0)
    std = obs.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return ObsNormStats(mean=mean, std=std)


def _refresh_last_observed_row_features(src_obs: torch.Tensor, obs_dim: int) -> None:
    """Refresh lag/rolling derived channels on the last sequence row in-place.

    Expected observed channel layout when obs_dim >= 6:
    [y_t, lag1, lag24, roll_mean_24_prev, roll_std_24_prev, delta1]

    This keeps self-conditioned training states internally consistent after
    replacing the latest y_t with model predictions.
    """
    if obs_dim < 6:
        return

    # src_obs shape: (B, seq_len, obs_dim)
    bsz, seq_len, _ = src_obs.shape
    if seq_len < 2:
        return

    for b in range(bsz):
        y_seq = src_obs[b, :, 0]

        last = y_seq[-1]
        lag1 = y_seq[-2] if seq_len >= 2 else last
        lag24 = y_seq[-24] if seq_len >= 24 else lag1

        prev = y_seq[:-1]
        if prev.numel() == 0:
            roll_mean = last
            roll_std = torch.zeros_like(last)
        else:
            prev24 = prev[-24:] if prev.numel() >= 24 else prev
            roll_mean = prev24.mean()
            roll_std = prev24.std(unbiased=False)

        src_obs[b, -1, 1] = lag1
        src_obs[b, -1, 2] = lag24
        src_obs[b, -1, 3] = roll_mean
        src_obs[b, -1, 4] = roll_std
        src_obs[b, -1, 5] = last - lag1


def _normalize_src(
    src: torch.Tensor, obs_stats: ObsNormStats, obs_dim: int
) -> torch.Tensor:
    src_n = src.clone()
    src_n[..., :obs_dim] = (
        src_n[..., :obs_dim] - obs_stats.mean.view(1, 1, -1)
    ) / obs_stats.std.view(1, 1, -1)
    return src_n


def _loss_fn(
    pred: torch.Tensor,
    y_true: torch.Tensor,
    quantile_alpha: float | None,
    peak_weight: float,
    peak_threshold_t: float,
) -> torch.Tensor:
    if quantile_alpha is None:
        abs_err = torch.abs(y_true - pred)
        if peak_weight > 1.0:
            weights = torch.where(
                y_true >= peak_threshold_t,
                torch.full_like(y_true, peak_weight),
                torch.ones_like(y_true),
            )
            return (abs_err * weights).mean() / weights.mean()
        return abs_err.mean()

    err = y_true - pred
    return torch.maximum(quantile_alpha * err, (quantile_alpha - 1.0) * err).mean()


def _evaluate_loader(
    *,
    model: nn.Module,
    loader: DataLoader,
    obs_stats: ObsNormStats,
    obs_dim: int,
    quantile_alpha: float | None,
    peak_weight: float,
    peak_threshold_t: float,
    device: torch.device,
) -> float:
    model.eval()
    running = 0.0
    count = 0
    with torch.no_grad():
        for src, tfy, y_true in loader:
            src = src.to(device)
            tfy = tfy.to(device)
            y_true = y_true.to(device)

            src_n = _normalize_src(src, obs_stats=obs_stats, obs_dim=obs_dim)
            pred = model(src_n, tfy)
            loss = _loss_fn(
                pred,
                y_true,
                quantile_alpha=quantile_alpha,
                peak_weight=peak_weight,
                peak_threshold_t=peak_threshold_t,
            )

            bsz = y_true.shape[0]
            running += float(loss.detach().cpu()) * bsz
            count += bsz

    return running / max(count, 1)


def _train_one_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    obs_stats: ObsNormStats,
    obs_dim: int,
    cfg: TrainConfig,
    quantile_alpha: float | None,
    peak_threshold_t: float,
    device: torch.device,
    label: str,
) -> nn.Module:
    model = model.to(device)
    obs_stats_dev = ObsNormStats(
        mean=obs_stats.mean.to(device),
        std=obs_stats.std.to(device),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.min_lr,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        count = 0

        if cfg.epochs <= 1:
            ss_prob = cfg.scheduled_sampling_end
        else:
            frac = epoch / float(cfg.epochs - 1)
            ss_prob = (
                cfg.scheduled_sampling_start
                + (cfg.scheduled_sampling_end - cfg.scheduled_sampling_start) * frac
            )

        total_batches = len(train_loader)
        progress_step = max(1, total_batches // 25)
        _print_progress(f"{label} e{epoch + 1:02d}", 0, total_batches)

        for batch_idx, (src, tfy, y_true) in enumerate(train_loader, start=1):
            src = src.to(device)
            tfy = tfy.to(device)
            y_true = y_true.to(device)

            src_n = _normalize_src(src, obs_stats=obs_stats_dev, obs_dim=obs_dim)
            if cfg.obs_noise_std > 0.0:
                src_n[..., :obs_dim] = src_n[..., :obs_dim] + (
                    torch.randn_like(src_n[..., :obs_dim]) * cfg.obs_noise_std
                )
            pred = model(src_n, tfy)

            # Scheduled-sampling-lite: replace the latest observed value in part
            # of the batch with the model's own prediction and train again.
            if ss_prob > 0.0:
                use_pred = torch.rand(y_true.shape[0], device=device) < (
                    ss_prob * cfg.autoreg_train_fraction
                )
                if bool(use_pred.any()):
                    src_roll = src.clone()
                    tfy_roll = tfy.clone()

                    max_roll = max(1, int(cfg.max_train_rollout_steps))
                    n_sel = int(use_pred.sum().item())
                    rollout_k = torch.zeros(
                        y_true.shape[0], dtype=torch.long, device=device
                    )
                    rollout_k[use_pred] = torch.randint(
                        low=1,
                        high=max_roll + 1,
                        size=(n_sel,),
                        device=device,
                    )

                    # Build a self-conditioned rollout state without tracking grads,
                    # then run one differentiable forward on that final state.
                    with torch.no_grad():
                        pred_roll_ng = pred.detach()
                        for step in range(1, max_roll + 1):
                            active = rollout_k >= step
                            if not bool(active.any()):
                                break

                            src_roll[active, -1, 0] = pred_roll_ng[active]
                            _refresh_last_observed_row_features(
                                src_roll[active, :, :obs_dim], obs_dim=obs_dim
                            )
                            tfy_roll[active, -1] = min(
                                1.0,
                                float(step) / float(max(1, cfg.rollout_scale_steps)),
                            )

                            src_roll_n_ng = _normalize_src(
                                src_roll, obs_stats=obs_stats_dev, obs_dim=obs_dim
                            )
                            pred_roll_ng = model(src_roll_n_ng, tfy_roll).detach()

                    src_roll_n = _normalize_src(
                        src_roll, obs_stats=obs_stats_dev, obs_dim=obs_dim
                    )
                    pred_roll = model(src_roll_n, tfy_roll)

                    pred = 0.5 * (pred + pred_roll)

            loss = _loss_fn(
                pred,
                y_true,
                quantile_alpha=quantile_alpha,
                peak_weight=cfg.peak_weight if quantile_alpha is None else 1.0,
                peak_threshold_t=peak_threshold_t,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bsz = y_true.shape[0]
            running += float(loss.detach().cpu()) * bsz
            count += bsz

            if batch_idx % progress_step == 0 or batch_idx == total_batches:
                _print_progress(f"{label} e{epoch + 1:02d}", batch_idx, total_batches)

        train_loss = running / max(count, 1)
        val_loss = _evaluate_loader(
            model=model,
            loader=val_loader,
            obs_stats=obs_stats_dev,
            obs_dim=obs_dim,
            quantile_alpha=quantile_alpha,
            peak_weight=1.0,
            peak_threshold_t=peak_threshold_t,
            device=device,
        )
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[{label}] epoch={epoch + 1:02d}/{cfg.epochs} "
            f"train={train_loss:.5f} val={val_loss:.5f} lr={lr_now:.2e} ss={ss_prob:.2f}"
        )

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stopping_patience:
                print(f"[{label}] early stop after {epoch + 1} epochs")
                break

    model.load_state_dict(best_state)
    print(f"[{label}] best_val={best_val:.5f}")
    return model


@torch.no_grad()
def _autoregressive_forecast_transformed(
    *,
    model: nn.Module,
    history_ts: list[pd.Timestamp],
    history_y_t: list[float],
    history_ctx: np.ndarray,
    target_ts: list[pd.Timestamp],
    seq_len: int,
    obs_stats: ObsNormStats,
    obs_dim: int,
    device: torch.device,
    label: str,
    rollout_scale_steps: int,
) -> np.ndarray:
    """Forecast transformed target recursively with predicted values."""
    model = model.to(device)
    model.eval()

    obs_stats_dev = ObsNormStats(
        mean=obs_stats.mean.to(device),
        std=obs_stats.std.to(device),
    )

    preds_t: list[float] = []
    hist_t = list(history_ts)
    hist_y = [float(v) for v in history_y_t]
    hist_ctx = np.asarray(history_ctx, dtype=np.float32)
    if hist_ctx.ndim != 2 or hist_ctx.shape[0] != len(hist_y):
        raise ValueError("history_ctx must align with history_y_t")

    total_steps = len(target_ts)
    progress_step = max(1, total_steps // 25)
    _print_progress(label, 0, total_steps)

    for step_idx, ts in enumerate(target_ts, start=1):
        if len(hist_y) < seq_len:
            raise ValueError(
                f"History shorter than seq_len at inference: {len(hist_y)} < {seq_len}"
            )

        win_ts = pd.Series(hist_t[-seq_len:])
        win_t = _normalized_time_features(win_ts)

        # Warmup for lag24/rolling24 features.
        tail_needed = seq_len + 24
        tail_y = np.asarray(hist_y[-tail_needed:], dtype=np.float32)
        tail_ctx = hist_ctx[-tail_needed:]
        obs_tail = _build_observed_features(tail_y, tail_ctx)
        win_obs = obs_tail[-seq_len:]

        src = np.concatenate([win_obs, win_t], axis=1).astype(np.float32, copy=False)
        tfy = _normalized_time_features(
            pd.Series([ts]),
            rollout_steps=np.asarray([step_idx - 1], dtype=np.float32),
            rollout_scale_steps=rollout_scale_steps,
        )[0]

        src_t = torch.from_numpy(src[None, ...]).to(device)
        src_t = _normalize_src(src_t, obs_stats=obs_stats_dev, obs_dim=obs_dim)
        tfy_t = torch.from_numpy(tfy[None, ...]).to(device)

        pred_t = float(model(src_t, tfy_t).item())
        preds_t.append(pred_t)

        hist_t.append(ts)
        hist_y.append(pred_t)
        # Future context is unknown; persist last known context row.
        next_ctx = hist_ctx[-1:, :]
        hist_ctx = np.vstack([hist_ctx, next_ctx])

        if step_idx % progress_step == 0 or step_idx == total_steps:
            _print_progress(label, step_idx, total_steps)

    return np.asarray(preds_t, dtype=np.float64)


def _fit_p90_offset(
    *,
    point_model: nn.Module,
    train_ts: list[pd.Timestamp],
    y_raw_full: np.ndarray,
    y_t_full: np.ndarray,
    ctx_full: np.ndarray,
    val_start_index: int,
    seq_len: int,
    obs_stats: ObsNormStats,
    obs_dim: int,
    device: torch.device,
    cfg: TrainConfig,
    use_log_target: bool,
    calibration_quantile: float,
    peak_threshold_kw: float,
) -> float:
    history_ts = train_ts[:val_start_index]
    history_y_t = y_t_full[:val_start_index].astype(np.float64).tolist()
    history_y_raw = y_raw_full[:val_start_index].astype(np.float64).tolist()
    history_ctx = ctx_full[:val_start_index].astype(np.float32, copy=False)
    val_ts = train_ts[val_start_index:]

    if len(history_ts) <= seq_len or len(val_ts) == 0:
        return 0.0

    val_pred_t = _autoregressive_forecast_transformed(
        model=point_model,
        history_ts=history_ts,
        history_y_t=history_y_t,
        history_ctx=history_ctx,
        target_ts=val_ts,
        seq_len=seq_len,
        obs_stats=obs_stats,
        obs_dim=obs_dim,
        device=device,
        label="val-cal",
        rollout_scale_steps=cfg.rollout_scale_steps,
    )

    val_pred_raw_model = _target_inverse(val_pred_t, use_log_target=use_log_target)
    val_pred_raw = _blend_point_with_persistence(
        model_pred_raw=val_pred_raw_model,
        history_raw=history_y_raw,
        cfg=cfg,
        peak_threshold_kw=peak_threshold_kw,
    )

    val_true_raw = y_raw_full[
        val_start_index : val_start_index + len(val_pred_raw)
    ].astype(np.float64)

    if len(val_true_raw) == 0:
        return 0.0

    residual = val_true_raw - val_pred_raw
    offset = float(np.quantile(residual, calibration_quantile))
    return max(0.0, offset)


def _fit_point_offset(
    *,
    point_model: nn.Module,
    train_ts: list[pd.Timestamp],
    y_raw_full: np.ndarray,
    y_t_full: np.ndarray,
    ctx_full: np.ndarray,
    val_start_index: int,
    seq_len: int,
    obs_stats: ObsNormStats,
    obs_dim: int,
    device: torch.device,
    cfg: TrainConfig,
    use_log_target: bool,
    calibration_quantile: float,
    peak_threshold_kw: float,
) -> float:
    """Calibrate point forecast bias using recursive validation residuals."""
    history_ts = train_ts[:val_start_index]
    history_y_t = y_t_full[:val_start_index].astype(np.float64).tolist()
    history_y_raw = y_raw_full[:val_start_index].astype(np.float64).tolist()
    history_ctx = ctx_full[:val_start_index].astype(np.float32, copy=False)
    val_ts = train_ts[val_start_index:]

    if len(history_ts) <= seq_len or len(val_ts) == 0:
        return 0.0

    val_pred_t = _autoregressive_forecast_transformed(
        model=point_model,
        history_ts=history_ts,
        history_y_t=history_y_t,
        history_ctx=history_ctx,
        target_ts=val_ts,
        seq_len=seq_len,
        obs_stats=obs_stats,
        obs_dim=obs_dim,
        device=device,
        label="val-point-cal",
        rollout_scale_steps=cfg.rollout_scale_steps,
    )

    val_pred_raw_model = _target_inverse(val_pred_t, use_log_target=use_log_target)
    val_pred_raw = _blend_point_with_persistence(
        model_pred_raw=val_pred_raw_model,
        history_raw=history_y_raw,
        cfg=cfg,
        peak_threshold_kw=peak_threshold_kw,
    )

    val_true_raw = y_raw_full[
        val_start_index : val_start_index + len(val_pred_raw)
    ].astype(np.float64)

    if len(val_true_raw) == 0:
        return 0.0

    residual = val_true_raw - val_pred_raw
    return float(np.quantile(residual, calibration_quantile))


def run_transformer_baseline(cfg: TrainConfig) -> pd.DataFrame:
    device = _resolve_device(cfg.device)
    print(f"[transformer] device={device}")

    hourly = load_hourly_total(REEFER_CSV)
    hourly_ctx = _build_hourly_context_features(REEFER_CSV)
    hourly = hourly.merge(hourly_ctx, on="ts", how="left")
    ctx_cols = [c for c in hourly.columns if c not in {"ts", TARGET_COL}]
    for c in ctx_cols:
        hourly[c] = hourly[c].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    targets = pd.read_csv(TARGET_CSV)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)
    targets["orig_idx"] = np.arange(len(targets), dtype=np.int64)

    target_start = targets["ts"].min()
    train_df = hourly.loc[hourly["ts"] < target_start].copy().reset_index(drop=True)

    if len(train_df) < 300:
        raise ValueError(f"Training history too short: {len(train_df)} rows")

    seq_len = min(cfg.seq_len, len(train_df) - 1)
    # if seq_len < 48:
    #     raise ValueError(f"Derived seq_len too small: {seq_len}")

    src, tfy, y_t, sample_target_indices, y_raw_full = _build_training_tensors(
        train_df,
        seq_len=seq_len,
        use_log_target=cfg.use_log_target,
        rollout_scale_steps=cfg.rollout_scale_steps,
        context_cols=ctx_cols,
    )

    total_samples = len(y_t)
    val_count = max(cfg.min_val_samples, int(total_samples * cfg.val_ratio))
    val_count = min(val_count, max(1, total_samples // 3))
    train_count = total_samples - val_count
    if train_count < 200:
        raise ValueError(
            f"Too few training samples after split: train={train_count}, total={total_samples}"
        )

    src_train, tfy_train, y_train = (
        src[:train_count],
        tfy[:train_count],
        y_t[:train_count],
    )
    src_val, tfy_val, y_val = src[train_count:], tfy[train_count:], y_t[train_count:]

    obs_dim = src_train.shape[-1] - 4
    obs_stats = _compute_obs_stats(src_train, obs_dim=obs_dim)

    train_ds = TensorDataset(src_train, tfy_train, y_train)
    val_ds = TensorDataset(src_val, tfy_val, y_val)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(cfg.batch_size, 256),
        shuffle=False,
        drop_last=False,
    )

    peak_threshold_t = float(np.quantile(y_train.numpy(), cfg.peak_quantile))
    peak_threshold_kw = float(np.quantile(y_raw_full, cfg.point_peak_trigger_quantile))

    print(
        f"[transformer] rows={len(train_df)} samples={total_samples} train={train_count} "
        f"val={val_count} seq_len={seq_len} obs_dim={obs_dim}"
    )

    point_model = HTTransformerRegressor(
        t2v_dim=cfg.t2v_dim,
        model_dim=cfg.point_model_dim,
        num_heads=cfg.point_num_heads,
        num_layers=cfg.point_num_layers,
        dropout=cfg.point_dropout,
        dim_feedforward=cfg.point_dim_feedforward,
        observed_dim=obs_dim,
        tfx_dim=4,
        tfy_dim=5,
        use_rollout_depth_for_horizon=True,
    )
    p90_model: nn.Module | None = None
    if cfg.train_p90_model:
        p90_model = HTTransformerRegressor(
            t2v_dim=cfg.t2v_dim,
            model_dim=cfg.p90_model_dim,
            num_heads=cfg.p90_num_heads,
            num_layers=cfg.p90_num_layers,
            dropout=cfg.p90_dropout,
            dim_feedforward=cfg.p90_dim_feedforward,
            observed_dim=obs_dim,
            tfx_dim=4,
            tfy_dim=5,
            use_rollout_depth_for_horizon=True,
        )

    print("[transformer] training point model...")
    point_model = _train_one_model(
        model=point_model,
        train_loader=train_loader,
        val_loader=val_loader,
        obs_stats=obs_stats,
        obs_dim=obs_dim,
        cfg=cfg,
        quantile_alpha=None,
        peak_threshold_t=peak_threshold_t,
        device=device,
        label="point",
    )

    if p90_model is not None:
        print("[transformer] training p90 model...")
        p90_model = _train_one_model(
            model=p90_model,
            train_loader=train_loader,
            val_loader=val_loader,
            obs_stats=obs_stats,
            obs_dim=obs_dim,
            cfg=cfg,
            quantile_alpha=cfg.quantile_alpha,
            peak_threshold_t=peak_threshold_t,
            device=device,
            label="p90",
        )
    else:
        print("[transformer] p90 model training skipped (fast mode)")

    train_ts = [pd.Timestamp(v) for v in train_df["ts"]]
    y_t_full = _target_transform(y_raw_full, use_log_target=cfg.use_log_target)
    ctx_full = train_df[ctx_cols].to_numpy(dtype=np.float32, copy=False)
    val_start_idx = int(sample_target_indices[train_count])

    print("[transformer] calibrating p90 offset on validation tail...")
    point_offset = _fit_point_offset(
        point_model=point_model,
        train_ts=train_ts,
        y_raw_full=y_raw_full,
        y_t_full=y_t_full,
        ctx_full=ctx_full,
        val_start_index=val_start_idx,
        seq_len=seq_len,
        obs_stats=obs_stats,
        obs_dim=obs_dim,
        device=device,
        cfg=cfg,
        use_log_target=cfg.use_log_target,
        calibration_quantile=cfg.point_bias_calibration_quantile,
        peak_threshold_kw=peak_threshold_kw,
    )
    print(f"[transformer] calibrated point offset={point_offset:.3f} kW")

    p90_offset = _fit_p90_offset(
        point_model=point_model,
        train_ts=train_ts,
        y_raw_full=y_raw_full,
        y_t_full=y_t_full,
        ctx_full=ctx_full,
        val_start_index=val_start_idx,
        seq_len=seq_len,
        obs_stats=obs_stats,
        obs_dim=obs_dim,
        device=device,
        cfg=cfg,
        use_log_target=cfg.use_log_target,
        calibration_quantile=cfg.p90_calibration_quantile,
        peak_threshold_kw=peak_threshold_kw,
    )
    print(f"[transformer] calibrated p90 offset={p90_offset:.3f} kW")

    targets_sorted = targets.sort_values("ts").reset_index(drop=True)
    target_ts = [pd.Timestamp(v) for v in targets_sorted["ts"]]

    history_ts = train_ts
    history_y_t = y_t_full.astype(np.float64).tolist()
    history_y_raw = y_raw_full.astype(np.float64).tolist()
    history_ctx = ctx_full

    print("[transformer] autoregressive inference for point forecast...")
    pred_point_t = _autoregressive_forecast_transformed(
        model=point_model,
        history_ts=history_ts,
        history_y_t=history_y_t,
        history_ctx=history_ctx,
        target_ts=target_ts,
        seq_len=seq_len,
        obs_stats=obs_stats,
        obs_dim=obs_dim,
        device=device,
        label="forecast-point",
        rollout_scale_steps=cfg.rollout_scale_steps,
    )

    pred_p90_t_model: np.ndarray | None = None
    if p90_model is not None:
        print("[transformer] autoregressive inference for p90 forecast...")
        pred_p90_t_model = _autoregressive_forecast_transformed(
            model=p90_model,
            history_ts=history_ts,
            history_y_t=history_y_t,
            history_ctx=history_ctx,
            target_ts=target_ts,
            seq_len=seq_len,
            obs_stats=obs_stats,
            obs_dim=obs_dim,
            device=device,
            label="forecast-p90",
            rollout_scale_steps=cfg.rollout_scale_steps,
        )

    pred_point_raw_model = _target_inverse(
        pred_point_t, use_log_target=cfg.use_log_target
    )
    pred_point_raw = _blend_point_with_persistence(
        model_pred_raw=pred_point_raw_model,
        history_raw=history_y_raw,
        cfg=cfg,
        peak_threshold_kw=peak_threshold_kw,
    )
    pred_point_raw = pred_point_raw + point_offset

    pred_p90_raw_cal = _build_p90_from_point(
        point_raw=pred_point_raw,
        history_raw=history_y_raw,
        base_offset_kw=p90_offset,
        cfg=cfg,
    )
    if pred_p90_t_model is None:
        pred_p90_raw = pred_p90_raw_cal
    else:
        pred_p90_raw_model = _target_inverse(
            pred_p90_t_model, use_log_target=cfg.use_log_target
        )
        pred_p90_raw = (
            cfg.p90_blend * pred_p90_raw_model
            + (1.0 - cfg.p90_blend) * pred_p90_raw_cal
        )

    pred_point_raw = np.clip(pred_point_raw, a_min=0.0, a_max=None)
    pred_p90_raw = np.maximum(pred_p90_raw, pred_point_raw)
    pred_p90_raw = np.clip(pred_p90_raw, a_min=0.0, a_max=None)

    out_sorted = targets_sorted.copy()
    out_sorted["pred_power_kw"] = np.round(pred_point_raw, 2)
    out_sorted["pred_p90_kw"] = np.round(pred_p90_raw, 2)

    out = (
        targets[["orig_idx", "timestamp_utc"]]
        .merge(
            out_sorted[["orig_idx", "pred_power_kw", "pred_p90_kw"]],
            on="orig_idx",
            how="left",
        )
        .sort_values("orig_idx")
    )

    submission = out[["timestamp_utc", "pred_power_kw", "pred_p90_kw"]]
    TRANSFORMER_OUT.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(TRANSFORMER_OUT, index=False)

    print(
        f"[transformer] submission written: {TRANSFORMER_OUT.relative_to(PROJECT_ROOT)} "
        f"({len(submission)} rows)"
    )
    return submission


def main() -> None:
    run_transformer_baseline(TrainConfig())


if __name__ == "__main__":
    main()
