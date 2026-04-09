"""Random Forest baseline for Reefer Peak Load Challenge.

This baseline reuses the existing data contracts from `lightgbm/baseline.py`
and derives a richer feature set directly from `reefer_release.csv`.

Run:
    uv run python lightgbm/random_forest_baseline.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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

RF_OUT = SUBMISSIONS_DIR / "random_forest_baseline.csv"

BASE_FEATURES: list[str] = [
    "hour",
    "dow",
    "month",
    "is_weekend",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_24h",
    "lag_48h",
    "lag_72h",
    "lag_168h",
    "roll_mean_6h",
    "roll_mean_24h",
    "roll_std_24h",
    "roll_mean_168h",
]

CONTEXT_FEATURES: list[str] = [
    "container_count",
    "visit_count",
    "hardware_type_count",
    "temp_setpoint_mean",
    "temp_ambient_mean",
    "temp_return_mean",
    "temp_supply_mean",
    "temp_gap_ambient_setpoint_mean",
    "temp_gap_return_supply_mean",
    "container_size_40_share",
    "stack_tier_mean",
]

FEATURES: list[str] = BASE_FEATURES + CONTEXT_FEATURES


def _load_hourly_context_features(csv_path: Path) -> pd.DataFrame:
    """Build hourly context features from reefer_release container-level rows."""
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
    print(f"[rf] Lese Kontext-Features aus {csv_path.name} ...")
    raw = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        usecols=lambda c: c in set(usecols),
        low_memory=False,
    )

    raw["EventTime"] = pd.to_datetime(raw["EventTime"], utc=True)
    raw["ts"] = raw["EventTime"].dt.floor("1h")

    for col in [
        "TemperatureSetPoint",
        "TemperatureAmbient",
        "TemperatureReturn",
        "RemperatureSupply",
        "stack_tier",
    ]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    size_num = pd.to_numeric(raw["ContainerSize"], errors="coerce")
    raw["is_size_40"] = (size_num >= 40).astype(np.float32)

    raw["temp_gap_ambient_setpoint"] = (
        raw["TemperatureAmbient"] - raw["TemperatureSetPoint"]
    )
    raw["temp_gap_return_supply"] = raw["TemperatureReturn"] - raw["RemperatureSupply"]

    out = (
        raw.groupby("ts", sort=True)
        .agg(
            container_count=("container_uuid", "nunique"),
            visit_count=("container_visit_uuid", "nunique"),
            hardware_type_count=("HardwareType", "nunique"),
            temp_setpoint_mean=("TemperatureSetPoint", "mean"),
            temp_ambient_mean=("TemperatureAmbient", "mean"),
            temp_return_mean=("TemperatureReturn", "mean"),
            temp_supply_mean=("RemperatureSupply", "mean"),
            temp_gap_ambient_setpoint_mean=("temp_gap_ambient_setpoint", "mean"),
            temp_gap_return_supply_mean=("temp_gap_return_supply", "mean"),
            container_size_40_share=("is_size_40", "mean"),
            stack_tier_mean=("stack_tier", "mean"),
        )
        .reset_index()
    )

    for col in out.columns:
        if col != "ts":
            out[col] = (
                out[col]
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .fillna(0.0)
                .astype(np.float32)
            )

    return out


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build time, lag and rolling features from hourly power series."""
    out = df.copy()
    out["hour"] = out["ts"].dt.hour.astype("int16")
    out["dow"] = out["ts"].dt.dayofweek.astype("int16")
    out["month"] = out["ts"].dt.month.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5).astype("int16")

    out["lag_1h"] = out[TARGET_COL].shift(1)
    out["lag_2h"] = out[TARGET_COL].shift(2)
    out["lag_3h"] = out[TARGET_COL].shift(3)
    out["lag_24h"] = out[TARGET_COL].shift(24)
    out["lag_48h"] = out[TARGET_COL].shift(48)
    out["lag_72h"] = out[TARGET_COL].shift(72)
    out["lag_168h"] = out[TARGET_COL].shift(168)

    shifted = out[TARGET_COL].shift(1)
    out["roll_mean_6h"] = shifted.rolling(6, min_periods=1).mean()
    out["roll_mean_24h"] = shifted.rolling(24, min_periods=1).mean()
    out["roll_std_24h"] = shifted.rolling(24, min_periods=2).std()
    out["roll_mean_168h"] = shifted.rolling(168, min_periods=1).mean()

    return out


def _fit_forests(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[RandomForestRegressor, RandomForestRegressor]:
    """Train two forests: one for point estimate and one for spread-aware p90."""
    point = RandomForestRegressor(
        n_estimators=600,
        criterion="absolute_error",
        max_depth=None,
        min_samples_leaf=3,
        max_features=0.7,
        n_jobs=-1,
        random_state=42,
    )

    p90 = RandomForestRegressor(
        n_estimators=900,
        criterion="absolute_error",
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=43,
    )

    print("[rf] Trainiere Point-Forest ...")
    point.fit(X, y)
    print("[rf] Trainiere P90-Forest ...")
    p90.fit(X, y)
    return point, p90


def _predict_point_and_p90(
    point_model: RandomForestRegressor,
    p90_model: RandomForestRegressor,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict point forecast and derive p90 from tree-wise distribution."""
    point_pred = point_model.predict(X)

    tree_preds = np.column_stack([est.predict(X) for est in p90_model.estimators_])
    p90_pred = np.quantile(tree_preds, 0.90, axis=1)

    p90_pred = np.maximum(p90_pred, point_pred)
    point_pred = np.clip(point_pred, a_min=0.0, a_max=None)
    p90_pred = np.clip(p90_pred, a_min=0.0, a_max=None)
    return point_pred, p90_pred


def main() -> None:
    hourly = load_hourly_total(REEFER_CSV)
    print(
        f"[rf] Zeitbereich: {hourly['ts'].min()} -> {hourly['ts'].max()}, "
        f"{len(hourly)} Stunden"
    )

    context = _load_hourly_context_features(REEFER_CSV)

    feat = _add_features(hourly)
    feat = feat.merge(context, on="ts", how="left")

    for col in CONTEXT_FEATURES:
        feat[col] = feat[col].ffill().fillna(0.0)

    feat = feat.dropna(subset=FEATURES).reset_index(drop=True)

    targets = pd.read_csv(TARGET_CSV)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)

    target_start = targets["ts"].min()
    train_df = feat.loc[feat["ts"] < target_start].copy()
    print(f"[rf] Trainings-Zeilen: {len(train_df)}, Feature-Anzahl: {len(FEATURES)}")

    point_model, p90_model = _fit_forests(train_df[FEATURES], train_df[TARGET_COL])

    target_feat = targets[["ts"]].merge(feat, on="ts", how="left")
    missing = int(target_feat[FEATURES].isna().any(axis=1).sum())
    if missing:
        print(f"[rf] WARN: {missing} Target-Stunden ohne Feature-Match")

    pred_point, pred_p90 = _predict_point_and_p90(
        point_model, p90_model, target_feat[FEATURES]
    )

    submission = pd.DataFrame(
        {
            "timestamp_utc": target_feat["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pred_power_kw": np.round(pred_point, 2),
            "pred_p90_kw": np.round(pred_p90, 2),
        }
    )
    RF_OUT.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(RF_OUT, index=False)
    print(
        f"[rf] Submission geschrieben: {RF_OUT.relative_to(PROJECT_ROOT)} "
        f"({len(submission)} Zeilen)"
    )


if __name__ == "__main__":
    main()
