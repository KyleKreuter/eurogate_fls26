"""Linear Regression and SVM baselines for the Reefer Peak Load Challenge.

These models serve as comparison points against the tree-based approaches
(LightGBM, RandomForest) and the physically-motivated model.

24h-ahead compliance
--------------------
All features use only data available at t-24h or earlier:
  - Lag features start at lag_24h (no lag_1h/2h/3h).
  - Rolling statistics are anchored at shift(24), so roll_mean_24h at time t
    is the mean of t-24…t-47, all safely in the past at forecast time.

Models
------
Ridge:
    L2-regularised linear regression. Fast baseline; establishes the floor.

SVR — three kernels, each written to its own submission CSV:
  linear  : LinearSVR (primal solver, scales well, ~equivalent to Ridge under
             the hood but with epsilon-insensitive loss instead of squared loss)
  rbf     : SVR with RBF kernel — captures non-linear interactions,
             typical workhorse for medium-sized datasets (~8k rows)
  poly    : SVR with degree-2 polynomial kernel — explicit quadratic
             interactions without the full complexity of RBF

P90 strategy
------------
Residual-spread: compute in-sample residuals (actual − predicted), take the
90th percentile as a constant additive offset. Consistent with physical_decomp.

Run
---
    uv run python alternative_baselines/linear_svm_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_LIGHTGBM = _ROOT / "lightgbm"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_LIGHTGBM) not in sys.path:
    sys.path.insert(0, str(_LIGHTGBM))

from baseline import (  # noqa: E402
    PROJECT_ROOT,
    REEFER_CSV,
    SUBMISSIONS_DIR,
    TARGET_COL,
    TARGET_CSV,
    load_hourly_total,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
LINEAR_OUT = SUBMISSIONS_DIR / "linear_regression_baseline.csv"
SVM_LINEAR_OUT = SUBMISSIONS_DIR / "svm_linear_baseline.csv"
SVM_RBF_OUT = SUBMISSIONS_DIR / "svm_rbf_baseline.csv"
SVM_POLY_OUT = SUBMISSIONS_DIR / "svm_poly_baseline.csv"

# ---------------------------------------------------------------------------
# Features — strictly 24h-ahead compliant
# Lags start at 24h; rolling stats anchored at shift(24).
# ---------------------------------------------------------------------------
FEATURES: list[str] = [
    "hour",
    "dow",
    "month",
    "is_weekend",
    "lag_24h",
    "lag_48h",
    "lag_72h",
    "lag_168h",
    "roll_mean_24h",   # mean of t-24 … t-47
    "roll_std_24h",    # std  of t-24 … t-47
    "roll_mean_168h",  # mean of t-24 … t-191
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["ts"].dt.hour.astype("int16")
    out["dow"] = out["ts"].dt.dayofweek.astype("int16")
    out["month"] = out["ts"].dt.month.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5).astype("int16")

    out["lag_24h"] = out[TARGET_COL].shift(24)
    out["lag_48h"] = out[TARGET_COL].shift(48)
    out["lag_72h"] = out[TARGET_COL].shift(72)
    out["lag_168h"] = out[TARGET_COL].shift(168)

    # Anchor rolling window at t-24 so every value in the window is ≥ 24h old.
    shifted24 = out[TARGET_COL].shift(24)
    out["roll_mean_24h"] = shifted24.rolling(24, min_periods=1).mean()
    out["roll_std_24h"] = shifted24.rolling(24, min_periods=2).std()
    out["roll_mean_168h"] = shifted24.rolling(168, min_periods=1).mean()

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _residual_p90_spread(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    resid = y.values - model.predict(X)
    return max(float(np.quantile(resid, 0.9)), 0.0)


def _write_submission(
    timestamps: pd.Series,
    pred_point: np.ndarray,
    spread: float,
    out_path: Path,
    label: str,
) -> None:
    pred_point = np.clip(pred_point, 0.0, None)
    pred_p90 = np.maximum(pred_point + spread, pred_point)
    sub = pd.DataFrame(
        {
            "timestamp_utc": timestamps.values,
            "pred_power_kw": np.round(pred_point, 2),
            "pred_p90_kw": np.round(pred_p90, 2),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False, float_format="%.2f")
    print(
        f"[{label}] -> {out_path.relative_to(PROJECT_ROOT)} "
        f"({len(sub)} Zeilen, mean={pred_point.mean():.1f} kW, "
        f"p90_spread={spread:.1f} kW)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load and engineer features
    # ------------------------------------------------------------------
    hourly = load_hourly_total(REEFER_CSV)
    print(
        f"[lin/svm] Zeitbereich: {hourly['ts'].min()} -> {hourly['ts'].max()}, "
        f"{len(hourly)} Stunden"
    )

    feat = _add_features(hourly)
    feat = feat.dropna(subset=FEATURES).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Target timestamps and training split
    # ------------------------------------------------------------------
    targets = pd.read_csv(TARGET_CSV)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)
    target_start = targets["ts"].min()
    target_end = targets["ts"].max()
    print(f"[lin/svm] Target: {target_start} -> {target_end} ({len(targets)} Stunden)")

    train_df = feat.loc[feat["ts"] < target_start].copy()
    print(f"[lin/svm] Trainings-Zeilen: {len(train_df)}, Features: {len(FEATURES)}")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET_COL]

    # ------------------------------------------------------------------
    # 3. Build target feature rows
    # ------------------------------------------------------------------
    target_feat = targets[["ts"]].merge(feat, on="ts", how="left")
    missing = int(target_feat[FEATURES].isna().any(axis=1).sum())
    if missing:
        print(f"[lin/svm] WARN: {missing} Target-Stunden ohne Feature-Match")
    X_target = target_feat[FEATURES].fillna(0.0)

    # ------------------------------------------------------------------
    # 4. Ridge regression
    # ------------------------------------------------------------------
    print("[lin/svm] Trainiere Ridge ...")
    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    ridge.fit(X_train, y_train)
    _write_submission(
        targets["timestamp_utc"],
        ridge.predict(X_target),
        _residual_p90_spread(ridge, X_train, y_train),
        LINEAR_OUT,
        "Ridge",
    )

    # ------------------------------------------------------------------
    # 5. SVR — linear kernel (LinearSVR, primal, fast)
    # ------------------------------------------------------------------
    print("[lin/svm] Trainiere SVR linear ...")
    svr_linear = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearSVR(epsilon=0.1, C=1.0, max_iter=5000, random_state=42)),
    ])
    svr_linear.fit(X_train, y_train)
    _write_submission(
        targets["timestamp_utc"],
        svr_linear.predict(X_target),
        _residual_p90_spread(svr_linear, X_train, y_train),
        SVM_LINEAR_OUT,
        "SVR-linear",
    )

    # ------------------------------------------------------------------
    # 6. SVR — RBF kernel
    # ------------------------------------------------------------------
    print("[lin/svm] Trainiere SVR RBF ...")
    svr_rbf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=100.0, epsilon=10.0, gamma="scale")),
    ])
    svr_rbf.fit(X_train, y_train)
    _write_submission(
        targets["timestamp_utc"],
        svr_rbf.predict(X_target),
        _residual_p90_spread(svr_rbf, X_train, y_train),
        SVM_RBF_OUT,
        "SVR-rbf",
    )

    # ------------------------------------------------------------------
    # 7. SVR — polynomial kernel (degree 2)
    # ------------------------------------------------------------------
    print("[lin/svm] Trainiere SVR poly (degree=2) ...")
    svr_poly = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="poly", degree=2, C=100.0, epsilon=10.0, gamma="scale",
                      coef0=1.0)),
    ])
    svr_poly.fit(X_train, y_train)
    _write_submission(
        targets["timestamp_utc"],
        svr_poly.predict(X_target),
        _residual_p90_spread(svr_poly, X_train, y_train),
        SVM_POLY_OUT,
        "SVR-poly",
    )


if __name__ == "__main__":
    main()
