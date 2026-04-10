"""Bottom-up decomposition forecast for the Reefer Peak Load Challenge.

Core idea:
    total_power_kw(t) = num_containers(t) × mean_power_per_container_kw(t)

Two separate LightGBM models are trained and multiplied at inference time:

    1. Count model   – predicts num_active_containers
       Features: count_lag24h, count_lag168h, hour, dow, month,
                 is_weekend, is_holiday, temperature_2m

    2. Per-unit model – predicts power_kw / num_containers
       Features: pu_lag24h, pu_lag168h, temperature_2m,
                 mean_setpoint_lag24h, share_deep_frozen_lag24h,
                 hour, dow, month

Why this should handle the January distribution shift better:
    - Container count is schedule-driven and highly predictable via lag_24h.
      The count model sees the actual January counts through its lag features.
    - Per-unit power is thermodynamics-driven: ambient temperature in January
      is ~3-5°C vs ~18°C in summer. Lower ambient → less heat ingress →
      less compressor work → lower per-unit power. The per-unit model can
      learn this physical relationship directly from temperature, instead of
      having to infer it from a biased all-season aggregate.

P90:
    rf_richfeat.csv (pinball = 9.38, best in pool). Falls back to a
    calibrated 90th-percentile offset from training residuals if missing.

Training:
    Winter months only: {1, 2, 3, 11, 12} – same seasonal filter as
    rf_richfeat.py. This keeps both sub-models in the winter distribution.

Run:
    uv run python lightgbm/decomposition_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# weather_external MUST be imported before baseline – baseline.py removes
# the local directory from sys.path to avoid the lightgbm package collision.
from weather_external import load_cth_weather  # noqa: E402

from baseline import (  # noqa: E402
    PROJECT_ROOT,
    REEFER_CSV,
    SUBMISSIONS_DIR,
    TARGET_COL,
    TARGET_CSV,
    train_lgbm,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DECOMP_OUT = SUBMISSIONS_DIR / "decomposition.csv"
P90_SOURCE = SUBMISSIONS_DIR / "rf_richfeat.csv"

# Tight winter filter: only Nov/Dec/Jan.
# The broader {1,2,3,11,12} filter includes months with ~480 containers
# (Nov/Dec peak), biasing the count model upward relative to January's
# actual ~380-430 container range.
SEASONAL_MONTHS: frozenset[int] = frozenset({11, 12, 1})

HOLIDAY_MONTH_DAYS: frozenset[tuple[int, int]] = frozenset(
    {(1, 1), (1, 6), (12, 24), (12, 25), (12, 26), (12, 31)}
)

COUNT_FEATURES: list[str] = [
    "count_lag24h",
    "count_lag168h",
    "lag_24h",        # total power 24h earlier: strong proxy for count level
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "temperature_2m",
]

PER_UNIT_FEATURES: list[str] = [
    "pu_lag24h",
    "pu_lag168h",
    "temperature_2m",
    "mean_setpoint_lag24h",
    "share_deep_frozen_lag24h",
    "hour",
    "dow",
    "month",
]


# ---------------------------------------------------------------------------
# Data loading with decomposition-relevant aggregates
# ---------------------------------------------------------------------------
def load_hourly_decomp(csv_path: Path) -> pd.DataFrame:
    """Read reefer CSV and produce hourly aggregates for the decomposition.

    Returns a DataFrame with columns:
        ts, power_kw, num_containers, mean_setpoint, share_deep_frozen
    The series is gap-filled to a continuous 1-hour grid.
    """
    print(f"[load] Lese {csv_path.name} fuer Dekompositions-Features ...")
    df = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        usecols=["EventTime", "AvPowerCons", "TemperatureSetPoint"],
    )
    df["EventTime"] = pd.to_datetime(df["EventTime"], utc=True)

    sp = df["TemperatureSetPoint"]
    df["is_deep_frozen"] = (sp < -15).astype(np.int8)

    hourly = (
        df.groupby("EventTime", sort=True)
        .agg(
            power_w_sum=("AvPowerCons", "sum"),
            num_containers=("AvPowerCons", "count"),
            mean_setpoint=("TemperatureSetPoint", "mean"),
            share_deep_frozen=("is_deep_frozen", "mean"),
        )
        .reset_index()
        .rename(columns={"EventTime": "ts"})
    )

    hourly[TARGET_COL] = hourly["power_w_sum"] / 1000.0
    hourly = hourly.drop(columns=["power_w_sum"])

    # Gap-fill to a continuous hourly grid
    full_range = pd.date_range(
        start=hourly["ts"].min(), end=hourly["ts"].max(), freq="1h", tz="UTC"
    )
    hourly = (
        hourly.set_index("ts")
        .reindex(full_range)
        .rename_axis("ts")
        .reset_index()
    )
    hourly[TARGET_COL] = hourly[TARGET_COL].fillna(0.0)
    hourly["num_containers"] = hourly["num_containers"].fillna(0).astype(np.float32)
    hourly["mean_setpoint"] = hourly["mean_setpoint"].ffill().fillna(-10.0)
    hourly["share_deep_frozen"] = hourly["share_deep_frozen"].ffill().fillna(0.0)

    print(
        f"[load] {len(hourly)} Stunden, "
        f"power_kw mean={hourly[TARGET_COL].mean():.1f}, "
        f"containers mean={hourly['num_containers'].mean():.1f}"
    )
    return hourly


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add lag features, time features, and weather; return enriched DataFrame."""
    out = hourly.sort_values("ts").reset_index(drop=True).copy()

    # Per-unit power (undefined / NaN when no containers are plugged in)
    out["mean_power_per_unit"] = np.where(
        out["num_containers"] > 0,
        out[TARGET_COL] / out["num_containers"],
        np.nan,
    )

    # Lag of total power (24h): strong predictor for both sub-models
    out["lag_24h"] = out[TARGET_COL].shift(24)

    # Lags on count and per-unit (both >= 24h → fully legal)
    out["count_lag24h"] = out["num_containers"].shift(24)
    out["count_lag168h"] = out["num_containers"].shift(168)
    out["pu_lag24h"] = out["mean_power_per_unit"].shift(24)
    out["pu_lag168h"] = out["mean_power_per_unit"].shift(168)
    out["mean_setpoint_lag24h"] = out["mean_setpoint"].shift(24)
    out["share_deep_frozen_lag24h"] = out["share_deep_frozen"].shift(24)

    # Time features
    ts = out["ts"]
    out["hour"] = ts.dt.hour.astype(np.int16)
    out["dow"] = ts.dt.dayofweek.astype(np.int16)
    out["month"] = ts.dt.month.astype(np.int16)
    out["is_weekend"] = (out["dow"] >= 5).astype(np.int8)

    md = list(zip(ts.dt.month, ts.dt.day))
    out["is_holiday"] = np.fromiter(
        (1 if t in HOLIDAY_MONTH_DAYS else 0 for t in md),
        dtype=np.int8,
        count=len(out),
    )

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load and engineer features
    # ------------------------------------------------------------------
    hourly = load_hourly_decomp(REEFER_CSV)
    hourly = add_features(hourly)

    # Weather (no lag needed – at target time, a 24h weather forecast
    # is available; same convention as rf_richfeat.py)
    weather = load_cth_weather()
    hourly = hourly.merge(weather[["ts", "temperature_2m"]], on="ts", how="left")
    hourly["temperature_2m"] = hourly["temperature_2m"].ffill().bfill().fillna(5.0)

    # ------------------------------------------------------------------
    # 2. Target timestamps
    # ------------------------------------------------------------------
    targets = pd.read_csv(TARGET_CSV)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)
    target_start = targets["ts"].min()
    target_end = targets["ts"].max()
    print(
        f"[decomp] Target: {target_start} -> {target_end} "
        f"({len(targets)} Stunden)"
    )

    # ------------------------------------------------------------------
    # 3. Seasonal training split
    # ------------------------------------------------------------------
    train_mask = (hourly["ts"] < target_start) & hourly["ts"].dt.month.isin(
        SEASONAL_MONTHS
    )
    train = hourly[train_mask].copy()
    print(
        f"[decomp] Saisonales Training (Monate {sorted(SEASONAL_MONTHS)}): "
        f"{len(train)} Stunden, mean_power={train[TARGET_COL].mean():.1f} kW"
    )

    # Count model training set: rows with valid lag features
    # lag_24h NaNs only affect the first 24 rows – negligible
    train_count = train.dropna(subset=COUNT_FEATURES + ["num_containers"]).copy()
    print(f"[decomp] Count-Modell Training: {len(train_count)} Zeilen")

    # Per-unit model training set: exclude zero-container hours
    train_pu = train.dropna(subset=PER_UNIT_FEATURES + ["mean_power_per_unit"]).copy()
    train_pu = train_pu[train_pu["num_containers"] > 0].copy()
    print(f"[decomp] Per-Unit-Modell Training: {len(train_pu)} Zeilen")

    # ------------------------------------------------------------------
    # 4. Train sub-models
    # ------------------------------------------------------------------
    print("[decomp] Trainiere Count-Modell (regression_l1) ...")
    m_count = train_lgbm(
        train_count[COUNT_FEATURES],
        train_count["num_containers"].astype(float),
        objective="regression_l1",
        num_boost_round=600,
    )

    print("[decomp] Trainiere Per-Unit-Modell (regression_l1) ...")
    m_pu = train_lgbm(
        train_pu[PER_UNIT_FEATURES],
        train_pu["mean_power_per_unit"].astype(float),
        objective="regression_l1",
        num_boost_round=600,
    )

    # ------------------------------------------------------------------
    # 5. Predict for target timestamps
    # ------------------------------------------------------------------
    all_features = list(dict.fromkeys(COUNT_FEATURES + PER_UNIT_FEATURES))
    target_feat = targets[["ts"]].merge(hourly[["ts"] + all_features], on="ts", how="left")

    missing = int(target_feat[all_features].isna().any(axis=1).sum())
    if missing:
        print(f"[decomp] WARN: {missing} Target-Zeilen mit NaN-Features, fuellen mit 0")
    target_feat[COUNT_FEATURES] = target_feat[COUNT_FEATURES].fillna(0.0)
    target_feat[PER_UNIT_FEATURES] = target_feat[PER_UNIT_FEATURES].fillna(0.0)

    pred_count = m_count.predict(target_feat[COUNT_FEATURES])
    pred_per_unit = m_pu.predict(target_feat[PER_UNIT_FEATURES])

    # Point forecast: product of sub-models (clip negatives)
    pred_point = np.clip(pred_count, 0.0, None) * np.clip(pred_per_unit, 0.0, None)

    print(
        f"[decomp] pred_count: mean={pred_count.mean():.1f}, "
        f"range=[{pred_count.min():.1f}, {pred_count.max():.1f}]"
    )
    print(
        f"[decomp] pred_per_unit: mean={pred_per_unit.mean():.3f}, "
        f"range=[{pred_per_unit.min():.3f}, {pred_per_unit.max():.3f}]"
    )
    print(
        f"[decomp] pred_power_kw: mean={pred_point.mean():.1f}, "
        f"range=[{pred_point.min():.1f}, {pred_point.max():.1f}]"
    )

    # ------------------------------------------------------------------
    # 6. P90
    # ------------------------------------------------------------------
    if P90_SOURCE.exists():
        p90_df = pd.read_csv(P90_SOURCE)
        p90_df["ts"] = pd.to_datetime(p90_df["timestamp_utc"], utc=True)
        merged_p90 = targets[["ts"]].merge(
            p90_df[["ts", "pred_p90_kw"]], on="ts", how="left"
        )
        pred_p90 = np.maximum(
            merged_p90["pred_p90_kw"].fillna(0.0).values, pred_point
        )
        print(f"[decomp] P90 aus {P90_SOURCE.name} (pinball=9.38)")
    else:
        # Calibrated fallback: 90th percentile of in-sample residuals on
        # the full training set (not just seasonal)
        train_all = hourly[hourly["ts"] < target_start].dropna(
            subset=COUNT_FEATURES + PER_UNIT_FEATURES + [TARGET_COL]
        )
        train_all = train_all[train_all["num_containers"] > 0]
        in_sample_count = m_count.predict(train_all[COUNT_FEATURES])
        in_sample_pu = m_pu.predict(train_all[PER_UNIT_FEATURES])
        in_sample_pred = (
            np.clip(in_sample_count, 0, None) * np.clip(in_sample_pu, 0, None)
        )
        residuals = train_all[TARGET_COL].values - in_sample_pred
        spread = float(np.quantile(residuals, 0.9))
        pred_p90 = np.maximum(pred_point + max(spread, 0.0), pred_point)
        print(
            f"[decomp] WARN: {P90_SOURCE.name} fehlt, "
            f"nutze Residual-Offset P90 (spread={spread:.1f} kW)"
        )

    # ------------------------------------------------------------------
    # 7. Write submission
    # ------------------------------------------------------------------
    submission = pd.DataFrame(
        {
            "timestamp_utc": targets["timestamp_utc"].values,
            "pred_power_kw": np.round(pred_point, 2),
            "pred_p90_kw": np.round(pred_p90, 2),
        }
    )
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(DECOMP_OUT, index=False, float_format="%.2f")
    print(
        f"[decomp] -> {DECOMP_OUT.relative_to(PROJECT_ROOT)} "
        f"({len(submission)} Zeilen)"
    )


if __name__ == "__main__":
    main()
