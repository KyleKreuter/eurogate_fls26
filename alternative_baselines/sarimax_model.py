"""SARIMAX point forecast for the Reefer Peak Load Challenge.

Model:
    Two-stage: lag_24h baseline + SARIMAX(2, 0, 2)(0, 1, 1)[24] on residuals

Why the previous two versions failed:
    v1  SARIMAX(1,1,1)(0,1,1,24) on raw power:
        Forecast converged to seasonal mean ~840 kW and capped at 938 kW,
        missing all actual peaks up to 1028 kW.  mae_peak = 183.
    v2  SARIMAX(1,0,1)(0,0,0) on raw power with lag_24h as exogenous:
        Without differencing, the AR process mean-reverted to the training
        unconditional mean over 223 steps → mae_all = 145.

Why the residual approach works:
    Define  resid_t = power_kw_t – lag_24h_t.
    This residual is already roughly stationary (mean ≈ 0, much lower
    variance than raw power) so the ARMA process converges properly even
    over 223 steps.  The exogenous variables (count, temperature, time)
    explain *why* a given hour deviates from "same hour yesterday".
    Final prediction: lag_24h_target + SARIMAX_residual_forecast.

    On peak days (Jan 9-10): lag_24h is already high (Jan 8-9 were also
    high-load days), so the baseline is correct.  The SARIMAX residual
    adjusts for weather and container-count changes.

Exogenous regressors on the residual (all legal for 24h-ahead forecasting):
    - count_lag24h   : container count 24h earlier (volume change signal)
    - temperature_2m : ambient temperature (per-unit power change)
    - hour_sin/cos   : daily cycle in the residual pattern
    - dow_sin/cos    : weekly cycle

P90 source:
    rf_richfeat.csv (pinball = 9.38, best in pool). Falls back to the
    SARIMAX 90th-percentile prediction interval if the file is missing.

Run:
    uv run python lightgbm/sarimax_model.py
"""

from __future__ import annotations

import sys
import warnings
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
    load_hourly_with_container_mix,
)

from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SARIMAX_OUT = SUBMISSIONS_DIR / "sarimax.csv"
P90_SOURCE = SUBMISSIONS_DIR / "rf_richfeat.csv"

# How many days of training history before the target window to use.
# 60 days = ~Nov 1 – Dec 31: full winter context without summer bias.
TRAIN_DAYS: int = 60

# SARIMA orders for the RESIDUAL series (power - lag_24h):
# AR(2)/MA(2): captures up to 2-hour autocorrelation in the residuals.
# d=0: residuals are already near-stationary (mean ≈ 0).
# Seasonal MA(1) at period 24: residuals still have a weak daily shape.
ORDER = (2, 0, 2)
SEASONAL_ORDER = (0, 1, 1, 24)

EXOG_COLS: list[str] = [
    "count_lag24h",   # container count delta signal
    "temperature_2m", # ambient temperature
    "hour_sin",       # residual daily cycle
    "hour_cos",
    "dow_sin",        # weekly pattern
    "dow_cos",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load hourly series with container count
    # ------------------------------------------------------------------
    hourly = load_hourly_with_container_mix(REEFER_CSV)
    hourly = hourly.sort_values("ts").reset_index(drop=True)

    # Lag features (fully legal: >= 24h)
    hourly["lag_24h"] = hourly[TARGET_COL].shift(24)
    hourly["count_lag24h"] = hourly["num_active_containers"].shift(24).fillna(0.0)

    # Residual from 24h lag – this is the SARIMAX endogenous target
    hourly["resid_24h"] = hourly[TARGET_COL] - hourly["lag_24h"]

    # ------------------------------------------------------------------
    # 2. Weather + time features
    # ------------------------------------------------------------------
    weather = load_cth_weather()
    hourly = hourly.merge(weather[["ts", "temperature_2m"]], on="ts", how="left")
    hourly["temperature_2m"] = hourly["temperature_2m"].ffill().bfill().fillna(5.0)

    hour = hourly["ts"].dt.hour
    dow = hourly["ts"].dt.dayofweek
    hourly["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    hourly["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    hourly["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    hourly["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # ------------------------------------------------------------------
    # 3. Target timestamps
    # ------------------------------------------------------------------
    targets = pd.read_csv(TARGET_CSV)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)
    target_start = targets["ts"].min()
    target_end = targets["ts"].max()
    print(
        f"[sarimax] Target: {target_start} -> {target_end} "
        f"({len(targets)} Stunden)"
    )

    # ------------------------------------------------------------------
    # 4. Training window: last TRAIN_DAYS days before target
    # ------------------------------------------------------------------
    train_start = target_start - pd.Timedelta(days=TRAIN_DAYS)
    train = hourly[
        (hourly["ts"] >= train_start) & (hourly["ts"] < target_start)
    ].dropna(subset=["resid_24h"] + EXOG_COLS).copy()
    print(
        f"[sarimax] Training: {train['ts'].min()} -> {train['ts'].max()}, "
        f"{len(train)} Stunden"
    )
    print(
        f"[sarimax] Residual stats: mean={train['resid_24h'].mean():.1f}, "
        f"std={train['resid_24h'].std():.1f} kW"
    )

    endog = train["resid_24h"].values.astype(float)
    exog_train = train[EXOG_COLS].values.astype(float)

    # ------------------------------------------------------------------
    # 5. Fit SARIMAX on residuals
    # ------------------------------------------------------------------
    print(f"[sarimax] Fitting SARIMAX{ORDER}x{SEASONAL_ORDER} auf Residuen ...")
    model = SARIMAX(
        endog,
        exog=exog_train,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp=False, maxiter=200)

    print(f"[sarimax] AIC={result.aic:.1f}  BIC={result.bic:.1f}")
    print(f"[sarimax] Konvergiert: {result.mle_retvals.get('converged', '?')}")

    # ------------------------------------------------------------------
    # 6. Exogenous features + lag_24h baseline for target period
    # ------------------------------------------------------------------
    feat_cols = ["ts", "lag_24h"] + EXOG_COLS
    exog_full = hourly[feat_cols]
    target_feat = targets[["ts"]].merge(exog_full, on="ts", how="left")
    target_feat[EXOG_COLS] = target_feat[EXOG_COLS].fillna(0.0)
    # lag_24h for target: Jan 1 uses Dec 31, Jan 2 uses Jan 1 (in release file), etc.
    lag_24h_target = target_feat["lag_24h"].fillna(
        train[TARGET_COL].iloc[-24:].mean()  # fallback for any missing
    ).values
    exog_test = target_feat[EXOG_COLS].values.astype(float)

    # ------------------------------------------------------------------
    # 7. Forecast residuals, then add lag_24h baseline
    # ------------------------------------------------------------------
    print(f"[sarimax] Forecasting {len(targets)} Residual-Schritte ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = result.get_forecast(steps=len(targets), exog=exog_test)

    pred_resid = fc.predicted_mean
    pred_mean = np.maximum(lag_24h_target + pred_resid, 0.0)
    print(
        f"[sarimax] resid forecast: mean={pred_resid.mean():.1f}, "
        f"range=[{pred_resid.min():.1f}, {pred_resid.max():.1f}]"
    )
    print(
        f"[sarimax] pred_power_kw: mean={pred_mean.mean():.1f}, "
        f"range=[{pred_mean.min():.1f}, {pred_mean.max():.1f}]"
    )

    # ------------------------------------------------------------------
    # 8. P90
    # ------------------------------------------------------------------
    if P90_SOURCE.exists():
        p90_df = pd.read_csv(P90_SOURCE)
        p90_df["ts"] = pd.to_datetime(p90_df["timestamp_utc"], utc=True)
        merged_p90 = targets[["ts"]].merge(
            p90_df[["ts", "pred_p90_kw"]], on="ts", how="left"
        )
        pred_p90 = np.maximum(
            merged_p90["pred_p90_kw"].fillna(0.0).values, pred_mean
        )
        print(f"[sarimax] P90 aus {P90_SOURCE.name} (pinball=9.38)")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            conf = fc.conf_int(alpha=0.2)
        # Add lag_24h to get CI on raw power
        pred_p90 = np.maximum(
            lag_24h_target + conf.iloc[:, 1].values, pred_mean
        )
        print(
            f"[sarimax] WARN: {P90_SOURCE.name} fehlt, "
            f"nutze SARIMAX-Konfidenzintervall als P90"
        )

    # ------------------------------------------------------------------
    # 9. Write submission
    # ------------------------------------------------------------------
    submission = pd.DataFrame(
        {
            "timestamp_utc": targets["timestamp_utc"].values,
            "pred_power_kw": np.round(pred_mean, 2),
            "pred_p90_kw": np.round(pred_p90, 2),
        }
    )
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SARIMAX_OUT, index=False, float_format="%.2f")
    print(
        f"[sarimax] -> {SARIMAX_OUT.relative_to(PROJECT_ROOT)} "
        f"({len(submission)} Zeilen)"
    )


if __name__ == "__main__":
    main()
