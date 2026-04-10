"""k-NN Analog Forecasting for the Reefer Peak Load Challenge.

Core idea:
    For each target hour, find the K most physically similar historical hours
    and take their inverse-distance-weighted average actual power.

    "Physically similar" is defined by normalised Euclidean distance in the
    space of:
        hour_sin / hour_cos    – time-of-day (daily cycle)
        dow_sin / dow_cos      – day-of-week (weekly cycle)
        count_lag24h           – container load signal (schedule-driven)
        temperature_2m         – ambient temperature (cooling-demand signal)
        lag_24h                – actual power 24 h earlier (level anchor)

Why this is immune to the summer distribution-shift problem:
    Tree models bias toward the all-year mean (~1000 kW) because most
    training samples are from warm months.  k-NN makes no such global fit.
    Instead, for a January target hour with temp ≈ 3°C and ~400 containers,
    the nearest historical neighbors will be other cold-month hours with
    similar counts — not the summer hours with temp ≈ 20°C.

    First attempt used the full training set and included lag_24h in the
    distance features.  The problem: summer hours with accidentally low
    lag_24h (after a quiet day) appeared as "close" neighbors and pulled the
    prediction toward the 1001 kW summer mean (result: mae_all=83).

    Fix: restrict the pool to winter months {10,11,12,1,2,3} and drop
    lag_24h from the distance metric (temperature + container count already
    encode the physical context more purely).  lag_168h (same weekday last
    week) is added instead as a level anchor that stays within the season.

Winter-only training pool:
    Months {10, 11, 12, 1, 2, 3}.  January 2025 + recent autumn/winter
    months give ≈ 4300 candidate hours — all cold-weather, all at or below
    the January power level.

Weighting:
    Inverse squared distance: w_i = 1 / (d_i^2 + eps).
    Very close neighbors dominate; faraway outliers have negligible weight.

P90:
    rf_richfeat.csv (pinball = 9.38, best in pool).

Run:
    uv run python lightgbm/knn_analog.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

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
    load_hourly_with_container_mix,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KNN_OUT = SUBMISSIONS_DIR / "knn_analog.csv"
P90_SOURCE = SUBMISSIONS_DIR / "rf_richfeat.csv"

# Number of nearest neighbors (experiment: 5, 10, 20, 50)
K: int = 20

# Months included in the training pool.  Winter-only: avoids contamination
# from summer hours that have similar lag_24h values but much higher power.
POOL_MONTHS: frozenset[int] = frozenset({10, 11, 12, 1, 2, 3})

# Features used in the distance computation.  All must be present in the
# hourly DataFrame after feature engineering below.
# lag_24h is intentionally EXCLUDED: it anchors to the absolute power level,
# which varies with season and thus lets summer outliers into the pool.
# temperature_2m + count_lag24h already encode the physical context.
# lag_168h (same weekday, same hour, one week ago) is a within-season anchor.
KNN_FEATURES: list[str] = [
    "hour_sin",       # daily cycle (sin/cos avoids the 23→0 discontinuity)
    "hour_cos",
    "dow_sin",        # weekly cycle
    "dow_cos",
    "count_lag24h",   # container count 24h earlier (volume signal)
    "temperature_2m", # ambient temperature (per-unit power signal)
    "lag_168h",       # same hour, same weekday, one week ago (level anchor)
]

# Inverse-distance weight exponent: w ∝ 1 / distance^WEIGHT_EXP
WEIGHT_EXP: float = 2.0


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add all features needed for the k-NN distance computation."""
    out = hourly.sort_values("ts").reset_index(drop=True).copy()

    # Lag features (>= 24h, legal for 24h-ahead forecasting)
    out["lag_24h"] = out[TARGET_COL].shift(24)
    out["lag_168h"] = out[TARGET_COL].shift(168)
    out["count_lag24h"] = out["num_active_containers"].shift(24)

    # Cyclic time encodings (avoids 23→0 and 6→0 discontinuities)
    hour = out["ts"].dt.hour
    dow = out["ts"].dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load + engineer features
    # ------------------------------------------------------------------
    hourly = load_hourly_with_container_mix(REEFER_CSV)
    hourly = build_features(hourly)

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
        f"[knn] Target: {target_start} -> {target_end} ({len(targets)} Stunden)"
    )

    # ------------------------------------------------------------------
    # 3. Train / query split
    #    Training = ALL hours before target_start with no NaN features.
    #    Full dataset: the temperature + count features already encode the
    #    season, so we let k-NN naturally ignore summer outliers.
    # ------------------------------------------------------------------
    train = hourly[
        (hourly["ts"] < target_start)
        & hourly["ts"].dt.month.isin(POOL_MONTHS)
    ].dropna(subset=KNN_FEATURES + [TARGET_COL]).copy()
    print(
        f"[knn] Training-Pool (Monate {sorted(POOL_MONTHS)}): {len(train)} Stunden, "
        f"mean_power={train[TARGET_COL].mean():.1f} kW, "
        f"temp_mean={train['temperature_2m'].mean():.1f}°C"
    )

    target_feat = targets[["ts"]].merge(
        hourly[["ts"] + KNN_FEATURES], on="ts", how="left"
    )
    missing = int(target_feat[KNN_FEATURES].isna().any(axis=1).sum())
    if missing:
        print(f"[knn] WARN: {missing} Target-Zeilen mit NaN-Features, fuellen mit 0")
    target_feat[KNN_FEATURES] = target_feat[KNN_FEATURES].fillna(0.0)

    # ------------------------------------------------------------------
    # 4. Normalise (z-score using training statistics)
    #    Critical: all features live on very different scales
    #    (hour_sin ∈ [-1,1] vs count_lag24h ∈ [0, 600]).
    # ------------------------------------------------------------------
    X_train = train[KNN_FEATURES].values.astype(np.float64)
    X_target = target_feat[KNN_FEATURES].values.astype(np.float64)

    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0  # avoid division by zero for constant features

    X_train_norm = (X_train - feat_mean) / feat_std
    X_target_norm = (X_target - feat_mean) / feat_std

    print(f"[knn] Features ({len(KNN_FEATURES)}): {KNN_FEATURES}")
    for i, name in enumerate(KNN_FEATURES):
        print(
            f"  {name:<22} train_mean={feat_mean[i]:8.3f}  "
            f"train_std={feat_std[i]:8.3f}"
        )

    # ------------------------------------------------------------------
    # 5. k-NN query via KDTree
    # ------------------------------------------------------------------
    print(f"[knn] Baue KDTree aus {len(X_train_norm)} Trainings-Stunden ...")
    tree = KDTree(X_train_norm)

    print(f"[knn] Suche {K} naechste Nachbarn fuer {len(X_target_norm)} Ziel-Stunden ...")
    distances, indices = tree.query(X_target_norm, k=K, workers=-1)
    # distances: (n_target, K), indices: (n_target, K)

    # ------------------------------------------------------------------
    # 6. Inverse-distance weighted average
    # ------------------------------------------------------------------
    eps = 1e-8
    weights = 1.0 / (distances ** WEIGHT_EXP + eps)
    weights = weights / weights.sum(axis=1, keepdims=True)

    y_train = train[TARGET_COL].values
    pred_point = (weights * y_train[indices]).sum(axis=1)
    pred_point = np.maximum(pred_point, 0.0)

    print(
        f"[knn] pred_power_kw: mean={pred_point.mean():.1f}, "
        f"range=[{pred_point.min():.1f}, {pred_point.max():.1f}]"
    )

    # Diagnostics: for each target hour, log nearest neighbor distance
    min_dist = distances[:, 0]
    print(
        f"[knn] NN-Distanz (normalisiert): "
        f"median={np.median(min_dist):.3f}, "
        f"p90={np.quantile(min_dist, 0.9):.3f}, "
        f"max={min_dist.max():.3f}"
    )

    # ------------------------------------------------------------------
    # 7. P90 from rf_richfeat (best pinball in pool)
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
        print(f"[knn] P90 aus {P90_SOURCE.name} (pinball=9.38)")
    else:
        # Fallback: 90th percentile of neighbor power values
        neighbor_vals = y_train[indices]           # (n_target, K)
        pred_p90 = np.quantile(neighbor_vals, 0.9, axis=1)
        pred_p90 = np.maximum(pred_p90, pred_point)
        print(f"[knn] WARN: {P90_SOURCE.name} fehlt, nutze Nachbar-Quantile als P90")

    # ------------------------------------------------------------------
    # 8. Write submission
    # ------------------------------------------------------------------
    submission = pd.DataFrame(
        {
            "timestamp_utc": targets["timestamp_utc"].values,
            "pred_power_kw": np.round(pred_point, 2),
            "pred_p90_kw": np.round(pred_p90, 2),
        }
    )
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(KNN_OUT, index=False, float_format="%.2f")
    print(
        f"[knn] -> {KNN_OUT.relative_to(PROJECT_ROOT)} ({len(submission)} Zeilen)"
    )


if __name__ == "__main__":
    main()
