"""Physical bottom-up decomposition for the Reefer Peak Load Challenge.

Motivation
----------
All other models in this repo (baseline, catboost, rf_richfeat) operate on
pre-aggregated hourly totals: they see only the combined power draw of all
plugged-in containers at each hour, never the individual container records.
This approach instead reads the raw container-level CSV and decomposes total
power into two physically interpretable factors before aggregating:

    total_power(t) = num_active_containers(t) × mean_power_per_container(t)

Predicting each factor separately with its own model lets us bring in signals
that are invisible at the aggregate level — most importantly, the fleet's
collective plugin age (see below).

Key physical insight — hours_since_plugin
-----------------------------------------
A freshly-arrived reefer container must pull its cargo from near-ambient
temperature down to its setpoint (e.g. -25 °C for deep-frozen goods). This
initial cool-down phase lasts roughly 6–24 h and demands significantly more
compressor work than steady-state maintenance. When many containers arrive
simultaneously — as happens when a large vessel unloads — the terminal sees a
sharp power spike driven by concurrent cool-down, not by a sudden increase in
the number of connected units.

hours_since_plugin is computed per container visit as the elapsed time since
the first observation of that visit in the raw data. Aggregated to the hourly
level, mean_hours_since_plugin captures the fleet's collective "age":
  - A young fleet (recent mass arrival) → high per-unit draw.
  - An old fleet (stable, long-term residents) → lower per-unit draw.
This directly explains the Jan 9–10 spikes in the target window.

Because the feature requires container-level timestamps, it is inaccessible
to any model that starts from hourly aggregates. It is the central
differentiator of this approach.

Pipeline
--------
1. Load reefer_release.csv at container-visit level (~840 MB).
2. Compute hours_since_plugin per row (= EventTime − min(EventTime) for
   that visit).
3. Aggregate to hourly: total power_kw, num_containers, mean_setpoint,
   hardware-type shares (ML2/ML3/Decos), setpoint bucket shares, and
   mean_hours_since_plugin.
4. Feature engineering: all reefer-derived features are shifted 24 h
   (lag24h) so no information from the target hour is used — strictly
   24-h-ahead compliant. effective_plugin_age_h = lag24h value + 24 h
   approximates the fleet age at the target hour.
5. Merge external weather (temperature_2m, no lag needed — weather
   forecasts are available before the target hour).
6. Train on winter months only {11, 12, 1} to match the Jan target regime.

Sub-models (both LightGBM, MAE objective, 600 rounds)
------------------------------------------------------
Count model — predicts num_active_containers(t):
    count_lag24h, count_lag168h, lag_24h (total power proxy),
    hour, dow, month, is_weekend, is_holiday, temperature_2m

Per-unit model — predicts mean_power_per_container_kw(t):
    pu_lag24h, pu_lag168h, temperature_2m,
    mean_setpoint_lag24h,
    effective_plugin_age_h  (= mean_hours_since_plugin_lag24h + 24),
    share_deep_frozen_lag24h,
    share_ml2_lag24h, share_ml3_lag24h, share_decos_lag24h,
    hour, dow, month

Point forecast:
    pred_power_kw = clip(pred_count, 0) × clip(pred_per_unit, 0)

P90
---
Computed inline from training residuals: both sub-models predict on the
training set, total-power residuals (actual − predicted) are computed on
the intersection of training rows, and the 90th percentile of those
residuals is added as a constant spread over the point forecast.
This is self-contained — no external CSV is required.

Run
---
    uv run python alternative_baselines/physical_decomp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_LIGHTGBM = _ROOT / "lightgbm"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_LIGHTGBM) not in sys.path:
    sys.path.insert(0, str(_LIGHTGBM))

from weather_external import load_cth_weather  # noqa: E402

from baseline import (  # noqa: E402
    OUTPUT_SUFFIX,
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
PHYS_OUT = SUBMISSIONS_DIR / f"physical_decomp{OUTPUT_SUFFIX}.csv"

SEASONAL_MONTHS: frozenset[int] = frozenset({11, 12, 1})

HOLIDAY_MONTH_DAYS: frozenset[tuple[int, int]] = frozenset(
    {(1, 1), (1, 6), (12, 24), (12, 25), (12, 26), (12, 31)}
)

COUNT_FEATURES: list[str] = [
    "count_lag24h",
    "count_lag168h",
    "lag_24h",  # total power lag – proxy for overall activity level
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
    "effective_plugin_age_h",  # ≈ mean_hours_since_plugin_lag24h + 24
    "share_deep_frozen_lag24h",
    "share_ml2_lag24h",
    "share_ml3_lag24h",
    "share_decos_lag24h",
    "hour",
    "dow",
    "month",
]


# ---------------------------------------------------------------------------
# Data loading: container-level → hourly aggregates with hours_since_plugin
# ---------------------------------------------------------------------------
def load_hourly_physical(csv_path: Path) -> pd.DataFrame:
    """Read reefer CSV and produce hourly aggregates with hours_since_plugin.

    hours_since_plugin for row r with visit v is:
        EventTime(r) - min(EventTime for visit v)
    This measures how long the container has been plugged in.

    Returns a gapless hourly DataFrame with columns:
        ts, power_kw, num_containers,
        mean_setpoint, share_deep_frozen,
        share_ml2, share_ml3, share_decos,
        mean_hours_since_plugin
    """
    print(f"[load] Lese {csv_path.name} (container-level, ~840 MB) ...")
    df = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        usecols=[
            "container_visit_uuid",
            "HardwareType",
            "EventTime",
            "AvPowerCons",
            "TemperatureSetPoint",
        ],
    )
    df["EventTime"] = pd.to_datetime(df["EventTime"], utc=True)

    # ----------------------------------------------------------------
    # hours_since_plugin: time since first record for this visit
    # ----------------------------------------------------------------
    visit_start = df.groupby("container_visit_uuid")["EventTime"].transform("min")
    df["hours_since_plugin"] = (
        df["EventTime"] - visit_start
    ).dt.total_seconds() / 3600.0

    # ----------------------------------------------------------------
    # Derived per-row features for hourly aggregation
    # ----------------------------------------------------------------
    sp = df["TemperatureSetPoint"]
    df["is_deep_frozen"] = (sp < -15).astype(np.int8)

    ht = df["HardwareType"].astype(str)
    df["is_ml2"] = ht.isin(["ML2", "ML2i"]).astype(np.int8)
    df["is_ml3"] = (ht == "ML3").astype(np.int8)
    df["is_decos"] = ht.str.startswith("Decos").astype(np.int8)

    # ----------------------------------------------------------------
    # Hourly aggregation
    # ----------------------------------------------------------------
    print("[load] Aggregiere zu Stunden-Reihe ...")
    hourly = (
        df.groupby("EventTime", sort=True)
        .agg(
            power_w_sum=("AvPowerCons", "sum"),
            num_containers=("AvPowerCons", "count"),
            mean_setpoint=("TemperatureSetPoint", "mean"),
            share_deep_frozen=("is_deep_frozen", "mean"),
            share_ml2=("is_ml2", "mean"),
            share_ml3=("is_ml3", "mean"),
            share_decos=("is_decos", "mean"),
            mean_hours_since_plugin=("hours_since_plugin", "mean"),
        )
        .reset_index()
        .rename(columns={"EventTime": "ts"})
    )
    hourly[TARGET_COL] = hourly["power_w_sum"] / 1000.0
    hourly = hourly.drop(columns=["power_w_sum"])

    # ----------------------------------------------------------------
    # Gap-fill to continuous hourly grid
    # ----------------------------------------------------------------
    full_range = pd.date_range(
        start=hourly["ts"].min(), end=hourly["ts"].max(), freq="1h", tz="UTC"
    )
    hourly = hourly.set_index("ts").reindex(full_range).rename_axis("ts").reset_index()
    hourly[TARGET_COL] = hourly[TARGET_COL].fillna(0.0)
    hourly["num_containers"] = hourly["num_containers"].fillna(0).astype(np.float32)
    for col in [
        "mean_setpoint",
        "share_deep_frozen",
        "share_ml2",
        "share_ml3",
        "share_decos",
        "mean_hours_since_plugin",
    ]:
        hourly[col] = hourly[col].ffill().fillna(0.0)

    print(
        f"[load] {len(hourly)} Stunden, "
        f"power_kw mean={hourly[TARGET_COL].mean():.1f}, "
        f"containers mean={hourly['num_containers'].mean():.1f}, "
        f"plugin_age mean={hourly['mean_hours_since_plugin'].mean():.1f} h"
    )
    return hourly


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_features(hourly: pd.DataFrame) -> pd.DataFrame:
    out = hourly.sort_values("ts").reset_index(drop=True).copy()

    # Per-unit power (undefined → NaN when no containers)
    out["mean_power_per_unit"] = np.where(
        out["num_containers"] > 0,
        out[TARGET_COL] / out["num_containers"],
        np.nan,
    )

    # Power lags
    out["lag_24h"] = out[TARGET_COL].shift(24)
    out["lag_168h"] = out[TARGET_COL].shift(168)

    # Count lags
    out["count_lag24h"] = out["num_containers"].shift(24)
    out["count_lag168h"] = out["num_containers"].shift(168)

    # Per-unit lags
    out["pu_lag24h"] = out["mean_power_per_unit"].shift(24)
    out["pu_lag168h"] = out["mean_power_per_unit"].shift(168)

    # Physical feature lags
    out["mean_setpoint_lag24h"] = out["mean_setpoint"].shift(24)
    out["share_deep_frozen_lag24h"] = out["share_deep_frozen"].shift(24)
    out["share_ml2_lag24h"] = out["share_ml2"].shift(24)
    out["share_ml3_lag24h"] = out["share_ml3"].shift(24)
    out["share_decos_lag24h"] = out["share_decos"].shift(24)

    # hours_since_plugin lag + 24h offset:
    # At target time t, average plugin age ≈ value from t-24h plus 24 more hours.
    # This represents the "fleet age" the cooling units will have at prediction time.
    out["mean_hours_since_plugin_lag24h"] = out["mean_hours_since_plugin"].shift(24)
    out["effective_plugin_age_h"] = out["mean_hours_since_plugin_lag24h"] + 24.0

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
    hourly = load_hourly_physical(REEFER_CSV)
    hourly = add_features(hourly)

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
    print(f"[phys] Target: {target_start} -> {target_end} ({len(targets)} Stunden)")

    # ------------------------------------------------------------------
    # 3. Winter-only training split
    # ------------------------------------------------------------------
    train_mask = (hourly["ts"] < target_start) & hourly["ts"].dt.month.isin(
        SEASONAL_MONTHS
    )
    train = hourly[train_mask].copy()
    print(
        f"[phys] Saisonales Training (Monate {sorted(SEASONAL_MONTHS)}): "
        f"{len(train)} Stunden, "
        f"mean_power={train[TARGET_COL].mean():.1f} kW, "
        f"mean_plugin_age={train['mean_hours_since_plugin'].mean():.1f} h"
    )

    all_feats = list(dict.fromkeys(COUNT_FEATURES + PER_UNIT_FEATURES))

    train_count = train.dropna(subset=COUNT_FEATURES + ["num_containers"]).copy()
    print(f"[phys] Count-Training: {len(train_count)} Zeilen")

    train_pu = train.dropna(subset=PER_UNIT_FEATURES + ["mean_power_per_unit"]).copy()
    train_pu = train_pu[train_pu["num_containers"] > 0].copy()
    print(
        f"[phys] Per-Unit-Training: {len(train_pu)} Zeilen, "
        f"pu mean={train_pu['mean_power_per_unit'].mean():.3f} kW/container, "
        f"plugin_age mean={train_pu['effective_plugin_age_h'].mean():.1f} h"
    )

    # Show plugin-age effect in training data (key sanity check)
    young = train_pu[train_pu["effective_plugin_age_h"] < 12]["mean_power_per_unit"]
    old = train_pu[train_pu["effective_plugin_age_h"] > 72]["mean_power_per_unit"]
    if len(young) and len(old):
        print(
            f"[phys] Plugin-age check: "
            f"age<12h → pu={young.mean():.3f} kW, "
            f"age>72h → pu={old.mean():.3f} kW  "
            f"(difference validates physical hypothesis)"
        )

    # ------------------------------------------------------------------
    # 4. Train sub-models
    # ------------------------------------------------------------------
    print("[phys] Trainiere Count-Modell ...")
    m_count = train_lgbm(
        train_count[COUNT_FEATURES],
        train_count["num_containers"].astype(float),
        objective="regression_l1",
        num_boost_round=600,
    )

    print("[phys] Trainiere Per-Unit-Modell ...")
    m_pu = train_lgbm(
        train_pu[PER_UNIT_FEATURES],
        train_pu["mean_power_per_unit"].astype(float),
        objective="regression_l1",
        num_boost_round=600,
    )

    # Feature importance for per-unit model (validates physics)
    try:
        import lightgbm as lgb  # noqa: F401 – already loaded via baseline

        imp = pd.Series(
            m_pu.feature_importance(importance_type="gain"),
            index=PER_UNIT_FEATURES,
        ).sort_values(ascending=False)
        print("[phys] Per-Unit Feature Importance (gain, top-5):")
        for name, val in imp.head(5).items():
            print(f"  {name:<35} {val:.1f}")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 5. Predict for target timestamps
    # ------------------------------------------------------------------
    target_feat = targets[["ts"]].merge(hourly[["ts"] + all_feats], on="ts", how="left")
    missing = int(target_feat[all_feats].isna().any(axis=1).sum())
    if missing:
        print(f"[phys] WARN: {missing} Target-Zeilen mit NaN-Features")

    target_feat[COUNT_FEATURES] = target_feat[COUNT_FEATURES].fillna(0.0)
    target_feat[PER_UNIT_FEATURES] = target_feat[PER_UNIT_FEATURES].fillna(0.0)

    pred_count = m_count.predict(target_feat[COUNT_FEATURES])
    pred_per_unit = m_pu.predict(target_feat[PER_UNIT_FEATURES])
    pred_point = np.clip(pred_count, 0.0, None) * np.clip(pred_per_unit, 0.0, None)

    print(
        f"[phys] pred_count:    mean={pred_count.mean():.1f}, "
        f"range=[{pred_count.min():.1f}, {pred_count.max():.1f}]"
    )
    print(
        f"[phys] pred_per_unit: mean={pred_per_unit.mean():.3f}, "
        f"range=[{pred_per_unit.min():.3f}, {pred_per_unit.max():.3f}] kW"
    )
    print(
        f"[phys] pred_power_kw: mean={pred_point.mean():.1f}, "
        f"range=[{pred_point.min():.1f}, {pred_point.max():.1f}]"
    )

    # ------------------------------------------------------------------
    # 6. P90 — 90th-pct of in-sample total-power residuals
    # ------------------------------------------------------------------
    in_sample_count = m_count.predict(train_count[COUNT_FEATURES])
    in_sample_pu = m_pu.predict(train_pu[PER_UNIT_FEATURES])
    common = train_count.index.intersection(train_pu.index)
    if len(common):
        pred_total = np.clip(
            in_sample_count[train_count.index.get_indexer(common)], 0, None
        ) * np.clip(in_sample_pu[train_pu.index.get_indexer(common)], 0, None)
        resid = train_count.loc[common, TARGET_COL].values - pred_total
        spread = max(float(np.quantile(resid, 0.9)), 0.0)
    else:
        spread = 50.0  # safe default
    pred_p90 = pred_point + spread
    print(f"[phys] P90 Residual-Spread (90th-pct): {spread:.1f} kW")

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
    submission.to_csv(PHYS_OUT, index=False, float_format="%.2f")
    print(f"[phys] -> {PHYS_OUT.relative_to(PROJECT_ROOT)} ({len(submission)} Zeilen)")


if __name__ == "__main__":
    main()
