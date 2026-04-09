"""Produktive RandomForest-Modelle als Base-Modelle fuer das honest_blend.

Dieses Skript erzeugt die beiden RandomForest-Submissions, die im
honest_blend.py als Inputs kombiniert werden:

    legal_rf_big_s1.csv
        RF mit lag_48h und lag_72h zusaetzlich zu den Baseline-Lags.
        2000 Baeume, min_samples_leaf=6, max_features=0.5, seed=1.
        Beste mae_peak (23.93) aller regelkonformen Einzelmodelle.

    legal_rf_s1.csv
        RF nur mit Baseline-Lags (lag_24h, lag_168h) + Wetter +
        Container-Mix. 1000 Baeume, min_samples_leaf=6, max_features=0.5,
        seed=1. Bestes mae_all (42.40) der kompakteren RFs.

Beide nutzen Mirror-Year-Synthese fuer die ersten Trainingszeilen
(Jan 1-7 2025), damit der Post-Feiertags-Januar-Ramp-up ueberhaupt
im Training vorkommt.

Regeln der Challenge: 24h-ahead Forecast, keine Lags < 24h.

Ausfuehren:
    uv run python lightgbm/productive.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from baseline import (
    CONTAINER_MIX_BASE_COLS,
    HARD_CUTOFF_TS,
    OUTPUT_SUFFIX,
    PROJECT_ROOT,
    REEFER_CSV,
    SUBMISSIONS_DIR,
    TARGET_COL,
    TARGET_CSV,
    add_features,
    extend_post_cutoff_with_mirror,
    load_hourly_with_container_mix,
)


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
WEATHER_CACHE = (
    PROJECT_ROOT
    / "weather_data_lean"
    / "final"
    / "open_meteo_complete"
    / "openmeteo_cth_hamburg.csv"
)

# Deutsche Feiertage im relevanten Zeitraum. Besonders wichtig: 1. und
# 6. Januar fallen direkt ins Target-Fenster.
HOLIDAY_DATES: set = {
    pd.Timestamp("2025-01-01").date(),
    pd.Timestamp("2025-01-06").date(),
    pd.Timestamp("2025-04-18").date(),
    pd.Timestamp("2025-04-21").date(),
    pd.Timestamp("2025-05-01").date(),
    pd.Timestamp("2025-05-29").date(),
    pd.Timestamp("2025-06-09").date(),
    pd.Timestamp("2025-10-03").date(),
    pd.Timestamp("2025-12-24").date(),
    pd.Timestamp("2025-12-25").date(),
    pd.Timestamp("2025-12-26").date(),
    pd.Timestamp("2025-12-31").date(),
    pd.Timestamp("2026-01-01").date(),
    pd.Timestamp("2026-01-06").date(),
}

BIG_OUT = SUBMISSIONS_DIR / f"legal_rf_big_s1{OUTPUT_SUFFIX}.csv"
SMALL_OUT = SUBMISSIONS_DIR / f"legal_rf_s1{OUTPUT_SUFFIX}.csv"


# ---------------------------------------------------------------------------
# Wetter-Loader
# ---------------------------------------------------------------------------
def load_weather() -> pd.DataFrame:
    df = pd.read_csv(WEATHER_CACHE)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.rename(
        columns={
            "temperature_2m": "temp_c",
            "shortwave_radiation": "shortwave",
            "wind_speed_10m": "wind_kmh",
        }
    )
    return df[["ts", "temp_c", "shortwave", "wind_kmh"]]


# ---------------------------------------------------------------------------
# Mirror-Year-Synthese
# ---------------------------------------------------------------------------
def synthesize_mirror_lags(feat: pd.DataFrame) -> pd.DataFrame:
    """Fuellt NaN-Lag-Werte in Jan 1-7 2025 aus den Mirror-Werten ein Jahr
    spaeter (Dec 25-31 2025 etc.). Der Wochentag bleibt durch Offset-Tage
    erhalten: 24h->364d, 48h->363d, 72h->362d, 168h->358d.
    """
    out = feat.sort_values("ts").reset_index(drop=True).copy()
    lookup = pd.Series(
        out[TARGET_COL].to_numpy(),
        index=pd.DatetimeIndex(out["ts"]),
    )
    offsets = {
        "lag_24h": 364,
        "lag_48h": 363,
        "lag_72h": 362,
        "lag_168h": 358,
    }
    for col, off in offsets.items():
        if col not in out.columns:
            continue
        missing = out[col].isna()
        if not missing.any():
            continue
        mirror_ts = out.loc[missing, "ts"] + pd.Timedelta(days=off)
        mirror_vals = mirror_ts.map(lookup)
        has = mirror_vals.notna()
        idx = missing[missing].index[has.to_numpy()]
        out.loc[idx, col] = mirror_vals[has].to_numpy()
        print(f"[mirror] {col}: {int(has.sum())} Werte synthetisiert")
    return out


# ---------------------------------------------------------------------------
# Feature-Builder
# ---------------------------------------------------------------------------
def build_features(extra_lags: list[int] | None = None):
    """Baut hourly + Features + optional zusaetzliche Lags."""
    print("[feat] Lade Basis mit Container-Mix ...")
    hourly_mix = load_hourly_with_container_mix(REEFER_CSV)

    # Leakage-Schutz: hourly_mix enthaelt jetzt nur noch Zeilen bis
    # HARD_CUTOFF_TS. Fuer die Target-Stunden im Januar 2026 muessen wir die
    # Reihe per Mirror-Year synthetisch fortschreiben, damit lag_24h/lag_168h
    # und die Container-Mix-Lags wohldefiniert sind.
    targets_probe = pd.read_csv(TARGET_CSV)
    targets_probe["ts"] = pd.to_datetime(targets_probe["timestamp_utc"], utc=True)
    hourly_mix = extend_post_cutoff_with_mirror(
        hourly_mix,
        post_range_end=targets_probe["ts"].max(),
        cutoff=HARD_CUTOFF_TS,
    )

    weather_df = load_weather()

    feat = add_features(hourly_mix)
    if extra_lags:
        for lag in extra_lags:
            feat[f"lag_{lag}h"] = feat[TARGET_COL].shift(lag)

    feat = synthesize_mirror_lags(feat)

    feat["is_weekend"] = (feat["ts"].dt.dayofweek >= 5).astype("int8")
    feat["is_holiday"] = feat["ts"].dt.date.isin(HOLIDAY_DATES).astype("int8")
    for col in CONTAINER_MIX_BASE_COLS:
        feat[f"{col}_lag_24h"] = feat[col].shift(24)
    feat = feat.merge(weather_df, on="ts", how="left")

    base_features = [
        "hour", "dow", "is_weekend", "is_holiday",
        "lag_24h", "lag_168h",
        "temp_c", "shortwave", "wind_kmh",
        "num_active_containers_lag_24h",
        "mean_heat_gap_lag_24h",
        "anteil_tiefkuehl_lag_24h",
    ]
    if extra_lags:
        base_features += [f"lag_{lag}h" for lag in extra_lags]
    feat = feat.dropna(subset=base_features).reset_index(drop=True)

    targets = pd.read_csv(TARGET_CSV)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)
    target_start = targets["ts"].min()

    train_df = feat.loc[feat["ts"] < target_start].copy()
    target_feat = targets[["ts"]].merge(feat, on="ts", how="left")
    print(f"[feat] {len(train_df)} Trainings-Zeilen, {len(base_features)} Features")
    return train_df, target_feat, base_features


# ---------------------------------------------------------------------------
# Helfer: Submission schreiben
# ---------------------------------------------------------------------------
def write_submission(ts_col, point, p90, out_path: Path) -> None:
    p90 = np.maximum(p90, point)
    point = np.clip(point, 0, None)
    p90 = np.clip(p90, 0, None)
    df = pd.DataFrame(
        {
            "timestamp_utc": ts_col,
            "pred_power_kw": np.round(point, 2),
            "pred_p90_kw": np.round(p90, 2),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.2f")
    print(
        f"[write] {out_path.relative_to(PROJECT_ROOT)} "
        f"(point mean={point.mean():.1f})"
    )


def train_rf(
    X_train, y_train, X_target,
    *, n_estimators, min_samples_leaf, max_features, seed, label,
):
    print(
        f"[{label}] Trainiere RF: n_est={n_estimators}, "
        f"min_leaf={min_samples_leaf}, max_feat={max_features}, seed={seed}"
    )
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion="absolute_error",
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_train, y_train)
    point = rf.predict(X_target)
    tree_preds = np.column_stack(
        [est.predict(X_target) for est in rf.estimators_]
    )
    p90 = np.quantile(tree_preds, 0.90, axis=1)
    return point, p90


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ----- Modell 1: legal_rf_big_s1 (mit lag_48h und lag_72h) -----
    print()
    print("=" * 72)
    print("Modell 1: legal_rf_big_s1 (RF mit lag_48h + lag_72h)")
    print("=" * 72)
    train_df, target_feat, features = build_features(extra_lags=[48, 72])
    X_train = train_df[features].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    X_target = target_feat[features].to_numpy()
    ts_col = target_feat["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    point, p90 = train_rf(
        X_train, y_train, X_target,
        n_estimators=2000,
        min_samples_leaf=6,
        max_features=0.5,
        seed=1,
        label="big_s1",
    )
    write_submission(ts_col, point, p90, BIG_OUT)

    # ----- Modell 2: legal_rf_s1 (nur Baseline-Lags) -----
    print()
    print("=" * 72)
    print("Modell 2: legal_rf_s1 (RF mit lag_24h + lag_168h)")
    print("=" * 72)
    train_df, target_feat, features = build_features(extra_lags=None)
    X_train = train_df[features].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    X_target = target_feat[features].to_numpy()
    ts_col = target_feat["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    point, p90 = train_rf(
        X_train, y_train, X_target,
        n_estimators=1000,
        min_samples_leaf=6,
        max_features=0.5,
        seed=1,
        label="s1",
    )
    write_submission(ts_col, point, p90, SMALL_OUT)


if __name__ == "__main__":
    main()
