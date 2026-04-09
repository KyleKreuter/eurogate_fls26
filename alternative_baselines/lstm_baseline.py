"""LSTM baseline for Reefer Peak Load Challenge.

What this baseline does:
- Builds one-step-ahead autoregressive LSTM forecasts from hourly power history.
- Trains separate models for point and p90 forecasts.
- Excludes January 2026 rows from the training set.
- Writes a submission CSV to lightgbm/submissions/lstm_baseline.csv.

Run:
    uv run python alternative_baselines/lstm_baseline.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from baseline import (  # noqa: E402
    PROJECT_ROOT,
    REEFER_CSV,
    SUBMISSIONS_DIR,
    TARGET_COL,
    TARGET_CSV,
    load_hourly_total,
)

LSTM_OUT = SUBMISSIONS_DIR / "lstm_baseline.csv"


@dataclass
class TrainConfig:
    seq_len: int = 24
    batch_size: int = 128
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    hidden_size: int = 96
    num_layers: int = 36
    dropout: float = 0.2
    val_hours: int = 200 * 24
    early_stopping_patience: int = 5
    q_alpha: float = 0.9
    device: str = "auto"


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


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


def _exclude_january_2026(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    jan_mask = (df["ts"].dt.year == 2026) & (df["ts"].dt.month == 1)
    removed = int(jan_mask.sum())
    return df.loc[~jan_mask].copy(), removed


def _sin_cos(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * values.astype(np.float32) / np.float32(period)
    return np.sin(angle), np.cos(angle)


def _build_step_features(df: pd.DataFrame) -> np.ndarray:
    hour_sin, hour_cos = _sin_cos(df["ts"].dt.hour.to_numpy(), 24.0)
    dow_sin, dow_cos = _sin_cos(df["ts"].dt.dayofweek.to_numpy(), 7.0)
    month_sin, month_cos = _sin_cos(df["ts"].dt.month.to_numpy() - 1, 12.0)

    power = df[TARGET_COL].to_numpy(dtype=np.float32)
    x = np.column_stack(
        [
            power,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            month_sin,
            month_cos,
        ]
    )
    return x.astype(np.float32, copy=False)


def _build_sequences(
    df: pd.DataFrame,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(df) <= seq_len:
        return np.empty((0, seq_len, 7), dtype=np.float32), np.empty(
            (0,), dtype=np.float32
        )

    df = df.sort_values("ts").reset_index(drop=True)
    step_x = _build_step_features(df)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)
    ts = df["ts"].to_numpy()

    one_hour = np.timedelta64(1, "h")
    seqs: list[np.ndarray] = []
    labels: list[float] = []

    for idx in range(seq_len, len(df)):
        window = ts[idx - seq_len : idx + 1]
        if np.any(np.diff(window) != one_hour):
            continue
        seqs.append(step_x[idx - seq_len : idx, :])
        labels.append(float(y[idx]))

    if not seqs:
        return np.empty((0, seq_len, 7), dtype=np.float32), np.empty(
            (0,), dtype=np.float32
        )

    return np.stack(seqs).astype(np.float32), np.asarray(labels, dtype=np.float32)


def _pinball_loss(
    pred: torch.Tensor, target: torch.Tensor, alpha: float
) -> torch.Tensor:
    diff = target - pred
    return torch.mean(torch.maximum(alpha * diff, (alpha - 1.0) * diff))


def _fit_scalers(
    x_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x_mean = x_train.mean(axis=(0, 1), keepdims=True)
    x_std = x_train.std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-6:
        y_std = 1.0

    return x_mean.astype(np.float32), x_std.astype(np.float32), y_mean, y_std


def _train_model(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    loss_name: str,
    q_alpha: float | None = None,
) -> LSTMRegressor:
    device = _resolve_device(cfg.device)
    model = LSTMRegressor(
        input_size=x_train.shape[-1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    if len(x_val) > 0:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
            batch_size=cfg.batch_size,
            shuffle=False,
        )
    else:
        val_loader = None

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val = np.inf
    stale_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses: list[float] = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            if loss_name == "mae":
                loss = torch.mean(torch.abs(pred - yb))
            elif loss_name == "pinball":
                assert q_alpha is not None
                loss = _pinball_loss(pred, yb, q_alpha)
            else:
                raise ValueError(f"Unsupported loss: {loss_name}")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_losses.append(float(loss.detach().cpu()))

        val_loss = float(np.mean(train_losses))
        if val_loader is not None:
            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    if loss_name == "mae":
                        batch_loss = torch.mean(torch.abs(pred - yb))
                    else:
                        assert q_alpha is not None
                        batch_loss = _pinball_loss(pred, yb, q_alpha)
                    val_losses.append(float(batch_loss.detach().cpu()))
            val_loss = float(np.mean(val_losses))

        print(
            f"[lstm:{loss_name}] epoch {epoch:02d}/{cfg.epochs} "
            f"train={np.mean(train_losses):.4f} val={val_loss:.4f}"
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.early_stopping_patience:
                print(f"[lstm:{loss_name}] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def _make_inference_sequence(
    history: pd.DataFrame,
    ts: pd.Timestamp,
    seq_len: int,
) -> np.ndarray:
    hist = history.loc[history["ts"] < ts].sort_values("ts").tail(seq_len).copy()
    if len(hist) < seq_len:
        raise ValueError(f"Not enough history before {ts} to build sequence")

    one_hour = np.timedelta64(1, "h")
    ts_np = hist["ts"].to_numpy()
    if np.any(np.diff(ts_np) != one_hour):
        raise ValueError(f"History before {ts} is not continuous")

    return _build_step_features(hist)


def _predict_one(
    model: LSTMRegressor,
    seq: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: float,
    y_std: float,
    device: torch.device,
) -> float:
    seq_n = (seq[None, :, :] - x_mean) / x_std
    x = torch.from_numpy(seq_n.astype(np.float32)).to(device)
    with torch.no_grad():
        pred_n = float(model(x).cpu().numpy()[0])
    return float(pred_n * y_std + y_mean)


def _time_holdout_split(
    df: pd.DataFrame, valid_hours: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= valid_hours:
        return df.copy(), pd.DataFrame(columns=df.columns)
    idx = len(df) - valid_hours
    return df.iloc[:idx].copy(), df.iloc[idx:].copy()


def main() -> None:
    cfg = TrainConfig()
    device = _resolve_device(cfg.device)

    hourly = load_hourly_total(REEFER_CSV).sort_values("ts").reset_index(drop=True)
    targets = pd.read_csv(TARGET_CSV)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)
    targets = targets.sort_values("ts").reset_index(drop=True)

    target_start = targets["ts"].min()
    raw_train_df = hourly.loc[hourly["ts"] < target_start].copy()
    train_df, jan_removed = _exclude_january_2026(raw_train_df)

    print(
        f"[lstm] Trainingsbereich (vor Filter): {raw_train_df['ts'].min()} -> {raw_train_df['ts'].max()} "
        f"({len(raw_train_df)} Zeilen)"
    )
    print(f"[lstm] Januar-2026 aus Training entfernt: {jan_removed} Zeilen")
    print(f"[lstm] Trainingsbereich (nach Filter): {len(train_df)} Zeilen")

    tr_df, val_df = _time_holdout_split(train_df.sort_values("ts"), cfg.val_hours)

    x_tr, y_tr = _build_sequences(tr_df, cfg.seq_len)
    x_val, y_val = _build_sequences(val_df, cfg.seq_len)

    if len(x_tr) == 0:
        raise RuntimeError("No training sequences available after filtering.")

    print(f"[lstm] Sequenzen train/val: {len(x_tr)}/{len(x_val)}")

    x_mean, x_std, y_mean, y_std = _fit_scalers(x_tr, y_tr)
    x_tr_n = ((x_tr - x_mean) / x_std).astype(np.float32)
    y_tr_n = ((y_tr - y_mean) / y_std).astype(np.float32)

    if len(x_val) > 0:
        x_val_n = ((x_val - x_mean) / x_std).astype(np.float32)
        y_val_n = ((y_val - y_mean) / y_std).astype(np.float32)
    else:
        x_val_n = np.empty((0, cfg.seq_len, x_tr.shape[-1]), dtype=np.float32)
        y_val_n = np.empty((0,), dtype=np.float32)

    point_model = _train_model(
        x_train=x_tr_n,
        y_train=y_tr_n,
        x_val=x_val_n,
        y_val=y_val_n,
        cfg=cfg,
        loss_name="mae",
    )

    p90_model = _train_model(
        x_train=x_tr_n,
        y_train=y_tr_n,
        x_val=x_val_n,
        y_val=y_val_n,
        cfg=cfg,
        loss_name="pinball",
        q_alpha=cfg.q_alpha,
    )

    history = hourly.loc[hourly["ts"] < target_start, ["ts", TARGET_COL]].copy()

    pred_point: list[float] = []
    pred_p90: list[float] = []

    for ts in targets["ts"]:
        seq = _make_inference_sequence(history, ts, cfg.seq_len)

        point = _predict_one(
            point_model,
            seq,
            x_mean,
            x_std,
            y_mean,
            y_std,
            device,
        )
        p90 = _predict_one(
            p90_model,
            seq,
            x_mean,
            x_std,
            y_mean,
            y_std,
            device,
        )

        point = max(point, 0.0)
        p90 = max(p90, point)

        pred_point.append(point)
        pred_p90.append(p90)

        history = pd.concat(
            [history, pd.DataFrame({"ts": [ts], TARGET_COL: [point]})],
            ignore_index=True,
        )

    submission = pd.DataFrame(
        {
            "timestamp_utc": targets["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pred_power_kw": np.round(np.asarray(pred_point), 2),
            "pred_p90_kw": np.round(np.asarray(pred_p90), 2),
        }
    )

    LSTM_OUT.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(LSTM_OUT, index=False)
    print(
        f"[lstm] Submission geschrieben: {LSTM_OUT.relative_to(PROJECT_ROOT)} "
        f"({len(submission)} Zeilen)"
    )


if __name__ == "__main__":
    main()
