from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset

from data_builder import SequenceForecastDataset
from eval import combined_score, mae, pinball_loss
from utils import select_numeric_feature_cols

from model import ReeferTFT, TFTConfig
from losses import competition_aligned_loss, point_only_peak_weighted_loss

KNOWN_FUTURE_CANDIDATES = [
    "hour",
    "weekday",
    "month",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "doy_sin",
    "doy_cos",
]


@dataclass
class NormalizationStats:
    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    known_mean: torch.Tensor
    known_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


@dataclass
class FitLoopConfig:
    device: torch.device
    cfg: TFTConfig
    norm: NormalizationStats
    max_steps_per_epoch: int
    show_loader_progress: bool
    target_col: str
    accuracy_rel_tol: float
    print_epoch_metrics: bool
    label_prefix: str = ""
    loss_mode: str = "joint"  # "point_only" or "joint"
    train_peak_threshold: float = 0.0


@dataclass
class PreparedTrainingState:
    device: torch.device
    show_loader_progress: bool
    parquet_df: pl.DataFrame
    usable_feature_cols: list[str]
    known_future_cols: list[str]
    seq_ds: SequenceForecastDataset
    sample_size: int
    train_size: int
    valid_size: int
    split_gap: int
    train_loader: DataLoader
    valid_loader: DataLoader
    norm_stats: NormalizationStats
    first_sample: dict[str, torch.Tensor]
    cfg: TFTConfig
    model: ReeferTFT
    optimizer: torch.optim.Optimizer
    baseline_metrics: dict[str, float]


def _parse_int_csv(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return sorted(set(values))


def _parse_float_csv(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return sorted(set(values))


def _target_display_scale(target_col: str) -> tuple[float, str]:
    lower = target_col.lower()
    if "power_kw" in lower:
        return 1.0, "kW"
    if "power_w" in lower:
        return 1.0 / 1000.0, "kW"
    if "energy_wh" in lower:
        # For hourly aggregates, Wh and average W over the same hour are numerically equivalent.
        return 1.0 / 1000.0, "kW"
    return 1.0, "raw"


def _resolve_path(root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return root / p


def _build_minimal_sequence_parquet(
    raw_csv_path: Path, out_parquet_path: Path, target_col: str
) -> Path:
    if not raw_csv_path.exists():
        raise FileNotFoundError(
            f"Cannot build fallback parquet, raw CSV not found: {raw_csv_path}"
        )

    df = pl.read_csv(
        raw_csv_path,
        separator=";",
        infer_schema=False,
        ignore_errors=True,
        truncate_ragged_lines=True,
    )
    if "EventTime" not in df.columns:
        raise ValueError(
            "Fallback CSV missing 'EventTime' column required for timestamps."
        )

    target_src_col = None
    for c in ["TtlEnergyConsHour", "Energy", "AvPowerCons"]:
        if c in df.columns:
            target_src_col = c
            break
    if target_src_col is None:
        raise ValueError("Fallback CSV has no usable target source column.")

    keep_candidates = [
        "EventTime",
        target_src_col,
        "AvPowerCons",
        "TemperatureSetPoint",
        "TemperatureAmbient",
        "TemperatureReturn",
        "RemperatureSupply",
        "ContainerSize",
        "HardwareType",
    ]
    keep_cols = [c for c in keep_candidates if c in df.columns]
    df = df.select(keep_cols).rename({"EventTime": "timestamp_utc"})
    df = df.with_columns(
        pl.col("timestamp_utc").str.to_datetime(strict=False).alias("timestamp_utc")
    )

    df = df.filter(pl.col("timestamp_utc").is_not_null())

    numeric_candidates = [
        target_src_col,
        "AvPowerCons",
        "TemperatureSetPoint",
        "TemperatureAmbient",
        "TemperatureReturn",
        "RemperatureSupply",
    ]
    existing_numeric = [c for c in numeric_candidates if c in df.columns]
    df = df.with_columns(
        [
            pl.col(c).str.replace_all(",", ".").cast(pl.Float32, strict=False).alias(c)
            for c in existing_numeric
        ]
    )
    df = df.with_columns(
        pl.col("timestamp_utc").dt.truncate("1h").alias("timestamp_utc")
    )

    agg_exprs = [pl.col(target_src_col).sum().alias("target_energy_wh")]
    if "AvPowerCons" in df.columns:
        agg_exprs.append(pl.col("AvPowerCons").mean().alias("av_power_mean"))
    if "TemperatureSetPoint" in df.columns:
        agg_exprs.append(
            pl.col("TemperatureSetPoint").mean().alias("temp_setpoint_mean")
        )
    if "TemperatureAmbient" in df.columns:
        agg_exprs.append(pl.col("TemperatureAmbient").mean().alias("temp_ambient_mean"))
    if "TemperatureReturn" in df.columns:
        agg_exprs.append(pl.col("TemperatureReturn").mean().alias("temp_return_mean"))
    if "RemperatureSupply" in df.columns:
        agg_exprs.append(pl.col("RemperatureSupply").mean().alias("temp_supply_mean"))
    if "ContainerSize" in df.columns:
        agg_exprs.append(
            pl.col("ContainerSize").n_unique().alias("container_size_unique")
        )
    if "HardwareType" in df.columns:
        agg_exprs.append(
            pl.col("HardwareType").n_unique().alias("hardware_type_unique")
        )

    df = df.group_by("timestamp_utc").agg(agg_exprs).sort("timestamp_utc")

    df = df.with_columns((pl.col("target_energy_wh") / 1000.0).alias("target_power_kw"))
    if target_col not in df.columns:
        df = df.with_columns(pl.col("target_energy_wh").alias(target_col))

    df = df.with_columns(
        [
            pl.col("timestamp_utc").dt.hour().alias("hour"),
            pl.col("timestamp_utc").dt.weekday().alias("weekday"),
            pl.col("timestamp_utc").dt.month().alias("month"),
            (2 * pl.lit(3.141592653589793) * pl.col("timestamp_utc").dt.hour() / 24)
            .sin()
            .alias("hour_sin"),
            (2 * pl.lit(3.141592653589793) * pl.col("timestamp_utc").dt.hour() / 24)
            .cos()
            .alias("hour_cos"),
            (2 * pl.lit(3.141592653589793) * pl.col("timestamp_utc").dt.weekday() / 7)
            .sin()
            .alias("weekday_sin"),
            (2 * pl.lit(3.141592653589793) * pl.col("timestamp_utc").dt.weekday() / 7)
            .cos()
            .alias("weekday_cos"),
            (
                2
                * pl.lit(3.141592653589793)
                * pl.col("timestamp_utc").dt.ordinal_day()
                / 366
            )
            .sin()
            .alias("doy_sin"),
            (
                2
                * pl.lit(3.141592653589793)
                * pl.col("timestamp_utc").dt.ordinal_day()
                / 366
            )
            .cos()
            .alias("doy_cos"),
        ]
    )

    # Leakage-safe autoregressive target features built from historical values only.
    df = df.with_columns(
        [
            pl.col("target_power_kw").shift(1).alias("target_power_kw_lag_1"),
            pl.col("target_power_kw").shift(24).alias("target_power_kw_lag_24"),
            pl.col("target_power_kw").shift(48).alias("target_power_kw_lag_48"),
            pl.col("target_power_kw").shift(72).alias("target_power_kw_lag_72"),
            pl.col("target_power_kw").shift(168).alias("target_power_kw_lag_168"),
            pl.col("target_power_kw")
            .shift(1)
            .rolling_mean(24)
            .alias("target_power_kw_rollmean_24_tminus1"),
            pl.col("target_power_kw")
            .shift(1)
            .rolling_std(24)
            .alias("target_power_kw_rollstd_24_tminus1"),
            pl.col("target_power_kw")
            .shift(1)
            .rolling_mean(168)
            .alias("target_power_kw_rollmean_168_tminus1"),
            pl.col("target_power_kw")
            .shift(1)
            .rolling_std(168)
            .alias("target_power_kw_rollstd_168_tminus1"),
            (
                pl.col("target_power_kw").shift(24)
                - pl.col("target_power_kw").shift(48)
            ).alias("target_power_kw_delta_day_vs_prevday"),
        ]
    )

    out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_parquet_path, compression="zstd")
    return out_parquet_path


def _to_device_stats(
    values: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=0)
    std = values.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean.to(device), std.to(device)


def _normalize_batch(
    observed_x: torch.Tensor,
    known_future_x: torch.Tensor,
    y: torch.Tensor,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    known_mean: torch.Tensor,
    known_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observed_x = (observed_x - obs_mean.view(1, 1, -1)) / obs_std.view(1, 1, -1)
    known_future_x = (known_future_x - known_mean.view(1, 1, -1)) / known_std.view(
        1, 1, -1
    )
    y = (y - y_mean.view(1, 1)) / y_std.view(1, 1)
    return observed_x, known_future_x, y


def _relative_tolerance_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, rel_tol: float
) -> float:
    if y_true.size == 0:
        return float("nan")
    denom = np.maximum(np.abs(y_true), 1e-6)
    hits = np.abs(y_true - y_pred) <= (rel_tol * denom)
    return float(np.mean(hits))


def _iter_with_progress(
    loader: DataLoader,
    desc: str,
    enabled: bool,
    max_steps: int = 0,
):
    total_steps = len(loader)
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)

    if not enabled:
        for step_idx, batch in enumerate(loader):
            if max_steps > 0 and step_idx >= max_steps:
                break
            yield step_idx, batch
        return

    for step_idx, batch in enumerate(loader):
        if max_steps > 0 and step_idx >= max_steps:
            break

        done = step_idx + 1
        if total_steps > 0:
            pct = 100.0 * done / total_steps
            print(f"\r{desc}: {done}/{total_steps} ({pct:5.1f}%)", end="", flush=True)
        else:
            print(f"\r{desc}: {done}", end="", flush=True)

        yield step_idx, batch

    print()


def _evaluate(
    model: ReeferTFT,
    valid_loader: DataLoader,
    device: torch.device,
    cfg: TFTConfig,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    known_mean: torch.Tensor,
    known_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    accuracy_rel_tol: float,
    show_loader_progress: bool,
    epoch_idx: int,
    train_peak_threshold: float,
) -> dict[str, float]:
    model.eval()

    running_val_loss = 0.0
    val_steps = 0
    y_true_all = []
    point_all = []
    p90_all = []

    with torch.no_grad():
        for _, batch in _iter_with_progress(
            loader=valid_loader,
            desc=f"epoch {epoch_idx + 1} valid",
            enabled=show_loader_progress,
        ):
            observed_x = batch["encoder_x"].to(device)
            known_future_x = batch["known_future_x"].to(device)
            timestamp_cont = batch["timestamp_cont"].to(device)
            timestamp_index = batch["timestamp_index"].to(device).long()
            y_raw = batch["y"].to(device)

            observed_x, known_future_x, y_norm = _normalize_batch(
                observed_x=observed_x,
                known_future_x=known_future_x,
                y=y_raw,
                obs_mean=obs_mean,
                obs_std=obs_std,
                known_mean=known_mean,
                known_std=known_std,
                y_mean=y_mean,
                y_std=y_std,
            )

            static_x = None
            if cfg.static_dim > 0:
                static_x = torch.zeros(
                    observed_x.size(0), cfg.static_dim, device=device
                )

            out = model(
                observed_x=observed_x,
                known_future_x=known_future_x,
                static_x=static_x,
                timestamp_cont=timestamp_cont,
                timestamp_index=timestamp_index,
            )

            val_loss = competition_aligned_loss(
                y_true=y_norm,
                point_pred=out["point"],
                p90_pred=out["p90"],
                peak_threshold=train_peak_threshold,  # or a validation threshold proxy
                point_loss_type="huber",
                point_huber_delta=1.0,
                weight_mae_all=0.5,
                weight_mae_peak=0.3,
                weight_p90=0.2,
                monotonicity_weight=0.05,
            )
            running_val_loss += float(val_loss.item())
            val_steps += 1

            point_raw = out["point"] * y_std.view(1, 1) + y_mean.view(1, 1)
            p90_raw = out["p90"] * y_std.view(1, 1) + y_mean.view(1, 1)

            y_true_all.append(y_raw.detach().cpu().numpy().reshape(-1))
            point_all.append(point_raw.detach().cpu().numpy().reshape(-1))
            p90_all.append(p90_raw.detach().cpu().numpy().reshape(-1))

    if val_steps == 0:
        model.train()
        return {
            "val_loss": float("nan"),
            "val_mae": float("nan"),
            "val_pinball_p90": float("nan"),
            "val_score": float("nan"),
            "val_accuracy": float("nan"),
        }

    y_true_np = np.concatenate(y_true_all)
    point_np = np.concatenate(point_all)
    p90_np = np.concatenate(p90_all)

    peak_threshold = np.quantile(y_true_np, 0.9)
    peak_mask = y_true_np >= peak_threshold
    if not np.any(peak_mask):
        peak_mask = np.ones_like(y_true_np, dtype=bool)

    metrics = {
        "val_loss": running_val_loss / max(1, val_steps),
        "val_mae": float(mae(y_true_np, point_np)),
        "val_pinball_p90": float(pinball_loss(y_true_np, p90_np, q=0.9)),
        "val_score": float(combined_score(y_true_np, point_np, p90_np, peak_mask)),
        "val_accuracy": _relative_tolerance_accuracy(
            y_true_np, point_np, rel_tol=accuracy_rel_tol
        ),
    }
    model.train()
    return metrics


def _compute_lag24_baseline_metrics(
    seq_ds: SequenceForecastDataset,
    valid_start_idx: int,
    sample_size: int,
    accuracy_rel_tol: float,
) -> dict[str, float]:
    y_true_all = []
    point_all = []
    p90_all = []

    for idx in range(valid_start_idx, sample_size):
        i = idx + seq_ds.valid_start
        y_true = seq_ds.y[i : i + seq_ds.horizon]
        y_point = seq_ds.y[i - 24 : i]
        if len(y_true) != seq_ds.horizon or len(y_point) != seq_ds.horizon:
            continue
        y_p90 = np.maximum(1.10 * y_point, y_point)

        y_true_all.append(y_true)
        point_all.append(y_point)
        p90_all.append(y_p90)

    if not y_true_all:
        return {
            "val_mae": float("nan"),
            "val_pinball_p90": float("nan"),
            "val_score": float("nan"),
            "val_accuracy": float("nan"),
        }

    y_true_np = np.concatenate(y_true_all)
    point_np = np.concatenate(point_all)
    p90_np = np.concatenate(p90_all)

    peak_threshold = np.quantile(y_true_np, 0.9)
    peak_mask = y_true_np >= peak_threshold
    if not np.any(peak_mask):
        peak_mask = np.ones_like(y_true_np, dtype=bool)

    return {
        "val_mae": float(mae(y_true_np, point_np)),
        "val_pinball_p90": float(pinball_loss(y_true_np, p90_np, q=0.9)),
        "val_score": float(combined_score(y_true_np, point_np, p90_np, peak_mask)),
        "val_accuracy": _relative_tolerance_accuracy(
            y_true_np, point_np, rel_tol=accuracy_rel_tol
        ),
    }


def _run_train_epoch(
    model: ReeferTFT,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    fit_cfg: FitLoopConfig,
    epoch: int,
) -> tuple[float, int, float]:
    running_loss = 0.0
    steps = 0
    train_acc_hits = 0
    train_acc_total = 0

    for _, batch in _iter_with_progress(
        loader=train_loader,
        desc=f"{fit_cfg.label_prefix}epoch {epoch + 1} train".strip(),
        enabled=fit_cfg.show_loader_progress,
        max_steps=fit_cfg.max_steps_per_epoch,
    ):
        optimizer.zero_grad()

        observed_x = batch["encoder_x"].to(fit_cfg.device)
        known_future_x = batch["known_future_x"].to(fit_cfg.device)
        timestamp_cont = batch["timestamp_cont"].to(fit_cfg.device)
        timestamp_index = batch["timestamp_index"].to(fit_cfg.device).long()
        y_raw = batch["y"].to(fit_cfg.device)

        observed_x, known_future_x, y = _normalize_batch(
            observed_x=observed_x,
            known_future_x=known_future_x,
            y=y_raw,
            obs_mean=fit_cfg.norm.obs_mean,
            obs_std=fit_cfg.norm.obs_std,
            known_mean=fit_cfg.norm.known_mean,
            known_std=fit_cfg.norm.known_std,
            y_mean=fit_cfg.norm.y_mean,
            y_std=fit_cfg.norm.y_std,
        )

        static_x = None
        if fit_cfg.cfg.static_dim > 0:
            static_x = torch.zeros(
                observed_x.size(0), fit_cfg.cfg.static_dim, device=fit_cfg.device
            )

        out = model(
            observed_x=observed_x,
            known_future_x=known_future_x,
            static_x=static_x,
            timestamp_cont=timestamp_cont,
            timestamp_index=timestamp_index,
        )

        if fit_cfg.loss_mode == "point_only":
            loss = point_only_peak_weighted_loss(
                y_true=y,
                point_pred=out["point"],
                peak_threshold=fit_cfg.train_peak_threshold,
                base_loss="huber",
                huber_delta=1.0,
                weight_all=0.7,
                weight_peak=0.3,
            )
        elif fit_cfg.loss_mode == "joint":
            loss = competition_aligned_loss(
                y_true=y,
                point_pred=out["point"],
                p90_pred=out["p90"],
                peak_threshold=fit_cfg.train_peak_threshold,
                point_loss_type="huber",
                point_huber_delta=1.0,
                weight_mae_all=0.5,
                weight_mae_peak=0.3,
                weight_p90=0.2,
                monotonicity_weight=0.0,
            )
        else:
            raise ValueError(f"Unknown loss_mode: {fit_cfg.loss_mode}")

        point_raw = out["point"] * fit_cfg.norm.y_std.view(
            1, 1
        ) + fit_cfg.norm.y_mean.view(1, 1)
        err = torch.abs(point_raw - y_raw)
        tol = fit_cfg.accuracy_rel_tol * torch.clamp(torch.abs(y_raw), min=1e-6)
        train_acc_hits += int((err <= tol).sum().item())
        train_acc_total += int(y_raw.numel())

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        steps += 1

    avg_loss = running_loss / max(1, steps)
    train_accuracy = (
        float(train_acc_hits / train_acc_total) if train_acc_total > 0 else float("nan")
    )
    return avg_loss, steps, train_accuracy


def _print_epoch_metrics(
    fit_cfg: FitLoopConfig,
    target_unit: str,
    target_scale: float,
    epoch: int,
    steps: int,
    avg_loss: float,
    train_accuracy: float,
    val_metrics: dict[str, float],
) -> None:
    val_mae_disp = val_metrics["val_mae"] * target_scale
    val_p90_disp = val_metrics["val_pinball_p90"] * target_scale
    val_score_disp = val_metrics["val_score"] * target_scale
    train_acc_pct = 100.0 * train_accuracy
    val_acc_pct = 100.0 * val_metrics["val_accuracy"]
    prefix = f"{fit_cfg.label_prefix} " if fit_cfg.label_prefix else ""
    print(
        f"{prefix}epoch={epoch + 1} train_steps={steps} train_loss={avg_loss:.6f} "
        f"train_acc_rel{int(100 * fit_cfg.accuracy_rel_tol)}pct={train_acc_pct:.2f}% "
        f"val_loss={val_metrics['val_loss']:.6f} "
        f"val_acc_rel{int(100 * fit_cfg.accuracy_rel_tol)}pct={val_acc_pct:.2f}% "
        f"val_mae_{target_unit}={val_mae_disp:.6f} "
        f"val_pinball_p90_{target_unit}={val_p90_disp:.6f} "
        f"val_score_{target_unit}={val_score_disp:.6f}"
    )


def _fit_model(
    model: ReeferTFT,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    fit_cfg: FitLoopConfig,
    epochs: int,
) -> tuple[float, dict[str, float], int]:
    model.train()
    last_avg_loss = float("nan")
    last_val_metrics = {
        "val_loss": float("nan"),
        "val_mae": float("nan"),
        "val_pinball_p90": float("nan"),
        "val_score": float("nan"),
        "val_accuracy": float("nan"),
    }
    last_steps = 0

    target_scale, target_unit = _target_display_scale(fit_cfg.target_col)

    for epoch in range(epochs):
        avg_loss, steps, train_accuracy = _run_train_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            fit_cfg=fit_cfg,
            epoch=epoch,
        )
        val_metrics = _evaluate(
            model=model,
            valid_loader=valid_loader,
            device=fit_cfg.device,
            cfg=fit_cfg.cfg,
            obs_mean=fit_cfg.norm.obs_mean,
            obs_std=fit_cfg.norm.obs_std,
            known_mean=fit_cfg.norm.known_mean,
            known_std=fit_cfg.norm.known_std,
            y_mean=fit_cfg.norm.y_mean,
            y_std=fit_cfg.norm.y_std,
            accuracy_rel_tol=fit_cfg.accuracy_rel_tol,
            show_loader_progress=fit_cfg.show_loader_progress,
            epoch_idx=epoch,
            train_peak_threshold=fit_cfg.train_peak_threshold,
        )

        if fit_cfg.print_epoch_metrics:
            _print_epoch_metrics(
                fit_cfg=fit_cfg,
                target_unit=target_unit,
                target_scale=target_scale,
                epoch=epoch,
                steps=steps,
                avg_loss=avg_loss,
                train_accuracy=train_accuracy,
                val_metrics=val_metrics,
            )

        last_avg_loss = avg_loss
        last_val_metrics = val_metrics
        last_steps = steps

    return last_avg_loss, last_val_metrics, last_steps


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ReeferTFT on the sequence dataset.",
    )
    parser.add_argument(
        "--raw-csv-path", default="participant_package/daten/reefer_release.csv"
    )
    parser.add_argument("--parquet-path", default="outputs/reefer_sequence.parquet")
    parser.add_argument("--target-col", default="target_power_kw")
    parser.add_argument("--encoder-len", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--accuracy-rel-tol", type=float, default=0.10)
    parser.add_argument("--auto-tune-capacity-lr", action="store_true")
    parser.add_argument("--hidden-dim-candidates", default="64,96,128")
    parser.add_argument("--learning-rate-candidates", default="1e-3,7e-4,5e-4")
    parser.add_argument("--tune-epochs", type=int, default=6)
    parser.add_argument("--tune-max-trials", type=int, default=8)
    parser.add_argument("--static-dim", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-loader-progress", action="store_true")
    return parser


def _load_parquet_and_columns(
    args: argparse.Namespace,
    parquet_path: Path,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    feature_df = pl.read_parquet(parquet_path)
    parquet_df = pl.read_parquet(parquet_path)

    feature_cols = select_numeric_feature_cols(
        feature_df,
        exclude={
            "timestamp_utc",
            args.target_col,
            "target_energy_wh",
            "target_power_kw",
        },
    )
    if not feature_cols:
        raise ValueError("No numeric feature columns found for sequence dataset.")

    available_cols = set(parquet_df.columns)
    known_future_cols = [c for c in KNOWN_FUTURE_CANDIDATES if c in available_cols]
    if not known_future_cols:
        raise ValueError(
            "No known-future columns found in parquet. "
            "Expected some of: " + ", ".join(KNOWN_FUTURE_CANDIDATES)
        )

    if args.target_col not in available_cols:
        raise ValueError(f"Target column '{args.target_col}' is missing in parquet.")

    usable_feature_cols = [c for c in feature_cols if c in available_cols]
    if not usable_feature_cols:
        raise ValueError("No selected feature columns exist in the training parquet.")

    return parquet_df, usable_feature_cols, known_future_cols


def _build_dataset_and_loaders(
    args: argparse.Namespace,
    parquet_path: Path,
    usable_feature_cols: list[str],
    known_future_cols: list[str],
) -> tuple[SequenceForecastDataset, int, int, int, int, DataLoader, DataLoader]:
    seq_ds = SequenceForecastDataset(
        parquet_path=str(parquet_path),
        feature_cols=usable_feature_cols,
        known_future_cols=known_future_cols,
        target_col=args.target_col,
        encoder_len=args.encoder_len,
        horizon=args.horizon,
    )

    if len(seq_ds) == 0:
        raise ValueError(
            "Sequence dataset is empty. "
            "Try smaller --encoder-len / --horizon or provide a longer time series."
        )

    sample_size = (
        len(seq_ds) if args.sample_size <= 0 else min(args.sample_size, len(seq_ds))
    )
    if sample_size < 2:
        raise ValueError("Need at least 2 samples to split into train and validation.")

    valid_size = max(1, int(sample_size * args.valid_ratio))
    if valid_size >= sample_size:
        valid_size = sample_size - 1
    train_size = sample_size - valid_size

    split_gap = max(0, args.horizon - 1)
    if train_size <= split_gap:
        raise ValueError(
            "Not enough samples for a leakage-free split with current horizon. "
            "Use a larger dataset, lower --valid-ratio, or smaller --horizon."
        )
    train_size = train_size - split_gap

    train_ds = Subset(seq_ds, range(train_size))
    valid_start_idx = train_size + split_gap
    valid_ds = Subset(seq_ds, range(valid_start_idx, sample_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return (
        seq_ds,
        sample_size,
        train_size,
        len(valid_ds),
        split_gap,
        train_loader,
        valid_loader,
    )


def _build_norm_stats(
    seq_ds: SequenceForecastDataset,
    train_size: int,
    horizon: int,
    device: torch.device,
) -> NormalizationStats:
    train_cutoff = seq_ds.valid_start + train_size
    obs_train_end = max(1, train_cutoff - 1)
    known_train_end = min(train_cutoff + horizon - 1, len(seq_ds.x_known))
    y_train_start = seq_ds.valid_start
    y_train_end = min(train_cutoff + horizon - 1, len(seq_ds.y))

    obs_values = torch.from_numpy(
        np.array(seq_ds.x_all[:obs_train_end], copy=True)
    ).float()
    known_values = torch.from_numpy(
        np.array(seq_ds.x_known[:known_train_end], copy=True)
    ).float()
    y_values = (
        torch.from_numpy(np.array(seq_ds.y[y_train_start:y_train_end], copy=True))
        .float()
        .unsqueeze(-1)
    )

    obs_mean, obs_std = _to_device_stats(obs_values, device)
    known_mean, known_std = _to_device_stats(known_values, device)
    y_mean_vec, y_std_vec = _to_device_stats(y_values, device)
    return NormalizationStats(
        obs_mean=obs_mean,
        obs_std=obs_std,
        known_mean=known_mean,
        known_std=known_std,
        y_mean=y_mean_vec.squeeze(-1),
        y_std=y_std_vec.squeeze(-1),
    )


def _build_model_and_optimizer(
    args: argparse.Namespace,
    first_sample: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[TFTConfig, ReeferTFT, torch.optim.Optimizer]:
    cfg = TFTConfig(
        static_dim=args.static_dim,
        observed_dim=first_sample["encoder_x"].shape[-1],
        known_future_dim=first_sample["known_future_x"].shape[-1],
        hidden_dim=args.hidden_dim,
        output_dim=2,
        max_encoder_length=args.encoder_len,
        max_decoder_length=args.horizon,
        timestamp_num_embeddings=24,
        timestamp_continuous_dim=first_sample["timestamp_cont"].shape[-1],
    )
    model = ReeferTFT(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    return cfg, model, optimizer


def _prepare_training_state(args: argparse.Namespace) -> PreparedTrainingState:
    show_loader_progress = not args.no_loader_progress

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parents[1]
    parquet_path = _resolve_path(project_root, args.parquet_path)
    raw_csv_path = _resolve_path(project_root, args.raw_csv_path)

    parquet_path = _build_minimal_sequence_parquet(
        raw_csv_path=raw_csv_path,
        out_parquet_path=parquet_path,
        target_col=args.target_col,
    )
    print(f"Built parquet from raw CSV: {parquet_path}")

    parquet_df, usable_feature_cols, known_future_cols = _load_parquet_and_columns(
        args, parquet_path
    )
    (
        seq_ds,
        sample_size,
        train_size,
        valid_size,
        split_gap,
        train_loader,
        valid_loader,
    ) = _build_dataset_and_loaders(
        args=args,
        parquet_path=parquet_path,
        usable_feature_cols=usable_feature_cols,
        known_future_cols=known_future_cols,
    )
    norm_stats = _build_norm_stats(
        seq_ds=seq_ds, train_size=train_size, horizon=args.horizon, device=device
    )
    first_sample = seq_ds[0]
    cfg, model, optimizer = _build_model_and_optimizer(
        args=args, first_sample=first_sample, device=device
    )

    valid_start_idx = train_size + split_gap

    baseline_metrics = _compute_lag24_baseline_metrics(
        seq_ds=seq_ds,
        valid_start_idx=valid_start_idx,
        sample_size=sample_size,
        accuracy_rel_tol=args.accuracy_rel_tol,
    )
    baseline_scale, baseline_unit = _target_display_scale(args.target_col)
    print(
        f"naive_lag24 baseline val_mae_{baseline_unit}={baseline_metrics['val_mae'] * baseline_scale:.6f} "
        f"val_pinball_p90_{baseline_unit}={baseline_metrics['val_pinball_p90'] * baseline_scale:.6f} "
        f"val_score_{baseline_unit}={baseline_metrics['val_score'] * baseline_scale:.6f} "
        f"val_acc_rel{int(100 * args.accuracy_rel_tol)}pct={100.0 * baseline_metrics['val_accuracy']:.2f}%"
    )

    return PreparedTrainingState(
        device=device,
        show_loader_progress=show_loader_progress,
        parquet_df=parquet_df,
        usable_feature_cols=usable_feature_cols,
        known_future_cols=known_future_cols,
        seq_ds=seq_ds,
        sample_size=sample_size,
        train_size=train_size,
        valid_size=valid_size,
        split_gap=split_gap,
        train_loader=train_loader,
        valid_loader=valid_loader,
        norm_stats=norm_stats,
        first_sample=first_sample,
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        baseline_metrics=baseline_metrics,
    )


def _run_capacity_lr_tuning(
    args: argparse.Namespace,
    state: PreparedTrainingState,
    val_metrics: dict[str, float],
) -> None:
    if not args.auto_tune_capacity_lr:
        return

    baseline_mae = state.baseline_metrics["val_mae"]
    if not (np.isfinite(baseline_mae) and val_metrics["val_mae"] <= baseline_mae):
        print("Skipping tuning sweep because baseline parity is not reached yet.")
        return

    print("Baseline parity reached; starting capacity/lr tuning sweep.")
    hidden_dim_candidates = _parse_int_csv(args.hidden_dim_candidates)
    lr_candidates = _parse_float_csv(args.learning_rate_candidates)
    max_trials = max(1, args.tune_max_trials)

    best_trial = {
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "val_score": val_metrics["val_score"],
        "val_mae": val_metrics["val_mae"],
    }

    trials_run = 0
    for hidden_dim in hidden_dim_candidates:
        for learning_rate in lr_candidates:
            if trials_run >= max_trials:
                break
            if (
                hidden_dim == args.hidden_dim
                and abs(learning_rate - args.learning_rate) < 1e-12
            ):
                continue

            tune_cfg = TFTConfig(
                static_dim=args.static_dim,
                observed_dim=state.first_sample["encoder_x"].shape[-1],
                known_future_dim=state.first_sample["known_future_x"].shape[-1],
                hidden_dim=hidden_dim,
                output_dim=2,
                max_encoder_length=args.encoder_len,
                max_decoder_length=args.horizon,
                timestamp_num_embeddings=24,
                timestamp_continuous_dim=state.first_sample["timestamp_cont"].shape[-1],
            )
            tune_model = ReeferTFT(tune_cfg).to(state.device)
            tune_optimizer = torch.optim.Adam(
                tune_model.parameters(),
                lr=learning_rate,
                weight_decay=args.weight_decay,
            )

            _, tune_metrics, _ = _fit_model(
                model=tune_model,
                optimizer=tune_optimizer,
                train_loader=state.train_loader,
                valid_loader=state.valid_loader,
                fit_cfg=FitLoopConfig(
                    device=state.device,
                    cfg=tune_cfg,
                    norm=state.norm_stats,
                    max_steps_per_epoch=args.max_steps_per_epoch,
                    show_loader_progress=False,
                    target_col=args.target_col,
                    accuracy_rel_tol=args.accuracy_rel_tol,
                    print_epoch_metrics=False,
                    label_prefix="tune",
                ),
                epochs=max(1, args.tune_epochs),
            )

            target_scale, target_unit = _target_display_scale(args.target_col)
            print(
                f"tune hidden_dim={hidden_dim} lr={learning_rate:.6g} "
                f"val_mae_{target_unit}={tune_metrics['val_mae'] * target_scale:.6f} "
                f"val_score_{target_unit}={tune_metrics['val_score'] * target_scale:.6f}"
            )

            if tune_metrics["val_score"] < best_trial["val_score"]:
                best_trial = {
                    "hidden_dim": hidden_dim,
                    "learning_rate": learning_rate,
                    "val_score": tune_metrics["val_score"],
                    "val_mae": tune_metrics["val_mae"],
                }
            trials_run += 1
        if trials_run >= max_trials:
            break

    target_scale, target_unit = _target_display_scale(args.target_col)
    print(
        f"best_tuned_config hidden_dim={best_trial['hidden_dim']} "
        f"lr={best_trial['learning_rate']:.6g} "
        f"val_mae_{target_unit}={best_trial['val_mae'] * target_scale:.6f} "
        f"val_score_{target_unit}={best_trial['val_score'] * target_scale:.6f}"
    )


def _print_training_summary(state: PreparedTrainingState) -> None:
    print("Sequence dataset loaded successfully.")
    print(f"Rows in parquet: {state.parquet_df.height}")
    print(f"Feature columns: {len(state.usable_feature_cols)}")
    print(f"Known-future columns: {len(state.known_future_cols)}")
    print(f"Samples in sequence dataset: {len(state.seq_ds)}")
    print(f"Training sample size: {state.sample_size}")
    print(f"Train samples: {state.train_size}")
    print(f"Validation samples: {state.valid_size}")
    print(f"Train/valid split gap (samples): {state.split_gap}")
    print(
        f"Observed feature norm mean(abs): {float(state.norm_stats.obs_mean.abs().mean()):.6f}"
    )
    print(
        f"Observed feature norm std(mean): {float(state.norm_stats.obs_std.mean()):.6f}"
    )
    print(
        f"Known-future norm mean(abs): {float(state.norm_stats.known_mean.abs().mean()):.6f}"
    )
    print(
        f"Known-future norm std(mean): {float(state.norm_stats.known_std.mean()):.6f}"
    )
    print(f"Target mean (raw): {float(state.norm_stats.y_mean.item()):.6f}")
    print(f"Target std (raw): {float(state.norm_stats.y_std.item()):.6f}")
    print(f"encoder_x sample shape: {tuple(state.first_sample['encoder_x'].shape)}")
    print(
        f"known_future_x sample shape: {tuple(state.first_sample['known_future_x'].shape)}"
    )
    print(
        f"decoder_known_x sample shape: {tuple(state.first_sample['decoder_known_x'].shape)}"
    )
    print(
        f"timestamp_cont sample shape: {tuple(state.first_sample['timestamp_cont'].shape)}"
    )
    print(
        f"timestamp_index sample shape: {tuple(state.first_sample['timestamp_index'].shape)}"
    )
    print(f"y sample shape: {tuple(state.first_sample['y'].shape)}")


def _fit_model_two_stage(
    model: ReeferTFT,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    fit_cfg: FitLoopConfig,
    stage1_epochs: int,
    stage2_epochs: int,
) -> tuple[float, dict[str, float], int]:
    print("Starting stage 1: point-only peak-weighted training")
    _fit_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        fit_cfg=FitLoopConfig(
            device=fit_cfg.device,
            cfg=fit_cfg.cfg,
            norm=fit_cfg.norm,
            max_steps_per_epoch=fit_cfg.max_steps_per_epoch,
            show_loader_progress=fit_cfg.show_loader_progress,
            target_col=fit_cfg.target_col,
            accuracy_rel_tol=fit_cfg.accuracy_rel_tol,
            print_epoch_metrics=fit_cfg.print_epoch_metrics,
            label_prefix="stage1",
            loss_mode="point_only",
            train_peak_threshold=fit_cfg.train_peak_threshold,
        ),
        epochs=stage1_epochs,
    )

    print("Starting stage 2: joint competition-aligned training")
    return _fit_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        fit_cfg=FitLoopConfig(
            device=fit_cfg.device,
            cfg=fit_cfg.cfg,
            norm=fit_cfg.norm,
            max_steps_per_epoch=fit_cfg.max_steps_per_epoch,
            show_loader_progress=fit_cfg.show_loader_progress,
            target_col=fit_cfg.target_col,
            accuracy_rel_tol=fit_cfg.accuracy_rel_tol,
            print_epoch_metrics=fit_cfg.print_epoch_metrics,
            label_prefix="stage2",
            loss_mode="joint",
            train_peak_threshold=fit_cfg.train_peak_threshold,
        ),
        epochs=stage2_epochs,
    )


def _run_training_pipeline(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    state = _prepare_training_state(args)
    y_train_raw = (
        torch.from_numpy(
            np.array(
                state.seq_ds.y[
                    state.seq_ds.valid_start : state.seq_ds.valid_start
                    + state.train_size
                    + args.horizon
                    - 1
                ],
                copy=True,
            )
        )
        .float()
        .to(state.device)
    )

    y_train_norm = (y_train_raw - state.norm_stats.y_mean) / state.norm_stats.y_std
    train_peak_threshold = float(torch.quantile(y_train_norm.reshape(-1), 0.9).item())

    _, val_metrics, _ = _fit_model_two_stage(
        model=state.model,
        optimizer=state.optimizer,
        train_loader=state.train_loader,
        valid_loader=state.valid_loader,
        fit_cfg=FitLoopConfig(
            device=state.device,
            cfg=state.cfg,
            norm=state.norm_stats,
            max_steps_per_epoch=args.max_steps_per_epoch,
            show_loader_progress=state.show_loader_progress,
            target_col=args.target_col,
            accuracy_rel_tol=args.accuracy_rel_tol,
            print_epoch_metrics=True,
            train_peak_threshold=train_peak_threshold,
        ),
        stage1_epochs=10,
        stage2_epochs=4,
    )

    _run_capacity_lr_tuning(args=args, state=state, val_metrics=val_metrics)
    _print_training_summary(state)


def main() -> None:
    args = build_arg_parser().parse_args()
    _run_training_pipeline(args)


if __name__ == "__main__":
    main()
