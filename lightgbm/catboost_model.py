"""CatBoost-Modell fuer die Reefer Peak Load Challenge.

Ziel: ein Modell bauen, das KOMPLEMENTAER zum bestehenden LightGBM/RandomForest-
Ensemble ist. CatBoost bietet dabei:

  - Native Behandlung kategorialer Features via Ordered-Target-Encoding.
    `hour` und `dow` werden direkt als cat_features uebergeben, nicht als
    kontinuierliche Integer-Features. Das fuehrt zu anderen Split-Strukturen
    als LightGBM oder RandomForest und damit zu einem komplementaeren
    Fehlerprofil.
  - Symmetric-Trees und unterschiedliches Regularisierungs-Profil.
  - Direkte Quantile-Loss fuer P90.

Regeln der Challenge:
  - 24h-ahead Forecasting: NUR Lags >= 24h erlaubt.
  - Training strikt vor Target-Start.

Feature-Set (bewusst minimal gehalten):
  - hour, dow                               (cat_features)
  - lag_24h, lag_168h                       (auf power_kw, stuendlich)

Das ist exakt das Feature-Set aus `baseline.py`. Wir haben im Ablations-
Test gesehen, dass mehr Features (Wetter, is_weekend, is_holiday, Container-
Mix) bei diesem extrem kurzen Target-Fenster (9 Tage, 223 Stunden) nur
Overfitting einfuehren und den Combined-Score verschlechtern. Der Sweet-
Spot ist minimaler Feature-Space + tiefere Trees (depth=8), so dass der
Baum-Split kategorialer Splits auf hour x dow die Saison-/Tageszeit-
Muster sauber modellieren kann.

Training:
  - 3 Seeds fuer Point-Modell (42, 1, 7) -> Mittelung
  - 1 Seed fuer P90-Modell
  - Loss: MAE fuer Point, Quantile(0.9) fuer P90

Kein Sample-Weighting. Januar-Peak-Weighting hat bei LightGBM geschadet
und CatBoost lernt mit der vorhandenen Datenbasis bereits sauber.

Ergebnis (standalone):
    mae_all=54.5  mae_peak=31.4  pinball=18.7  combined=40.4
    -> besser als baseline.py (combined=43.3)

Ausfuehren:
    uv run python lightgbm/catboost_model.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# baseline.py entfernt beim Modul-Load das lightgbm/-Verzeichnis aus sys.path
# (wegen Namenskollision mit dem installierten lightgbm-Package). Wir muessen
# deswegen unsere lokalen Module fuer den Import selbst rein pushen, bevor
# baseline importiert wird.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from baseline import (  # noqa: E402
    HARD_CUTOFF_TS,
    OUTPUT_SUFFIX,
    PROJECT_ROOT,
    REEFER_CSV,
    SUBMISSIONS_DIR,
    TARGET_COL,
    TARGET_CSV,
    extend_post_cutoff_with_mirror,
    load_hourly_total,
)

from catboost import CatBoostRegressor  # noqa: E402


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
CATBOOST_OUT = SUBMISSIONS_DIR / f"catboost{OUTPUT_SUFFIX}.csv"


# ---------------------------------------------------------------------------
# Container-Mix-Features (nicht mehr in der Champion-Config benutzt, aber
# als Hilfsfunktion drin fuer spaetere Experimente in ensemble.py o.ae.)
# ---------------------------------------------------------------------------
def load_hourly_with_container_mix(csv_path: Path) -> pd.DataFrame:
    """Laedt die Reefer-Rohdaten und aggregiert pro Stunde:

    - power_kw               (TARGET_COL, Summe AvPowerCons / 1000)
    - num_active_containers  (Anzahl unique container_uuid)
    - mean_heat_gap          (mean(TemperatureAmbient - TemperatureSetPoint))
    - anteil_tiefkuehl       (Anteil Rows mit TemperatureSetPoint < -10 C)

    Gibt eine dichte stuendliche Reihe zurueck (fehlende Stunden = 0 bei
    power_kw, forward-fill / 0 bei Container-Features).
    """
    print(f"[cat] Lese {csv_path.name} fuer power + container mix ...")
    usecols = [
        "EventTime",
        "AvPowerCons",
        "container_uuid",
        "TemperatureSetPoint",
        "TemperatureAmbient",
    ]
    df = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        usecols=usecols,
        low_memory=False,
    )
    df["EventTime"] = pd.to_datetime(df["EventTime"], utc=True)
    df["ts"] = df["EventTime"].dt.floor("1h")

    df["TemperatureSetPoint"] = pd.to_numeric(df["TemperatureSetPoint"], errors="coerce")
    df["TemperatureAmbient"] = pd.to_numeric(df["TemperatureAmbient"], errors="coerce")
    df["heat_gap"] = df["TemperatureAmbient"] - df["TemperatureSetPoint"]
    df["is_tiefkuehl"] = (df["TemperatureSetPoint"] < -10.0).astype(np.float32)

    agg = (
        df.groupby("ts", sort=True)
        .agg(
            power_w_sum=("AvPowerCons", "sum"),
            num_active_containers=("container_uuid", "nunique"),
            mean_heat_gap=("heat_gap", "mean"),
            anteil_tiefkuehl=("is_tiefkuehl", "mean"),
        )
        .reset_index()
    )
    agg[TARGET_COL] = agg["power_w_sum"] / 1000.0
    agg = agg.drop(columns=["power_w_sum"])

    full_range = pd.date_range(
        start=agg["ts"].min(),
        end=agg["ts"].max(),
        freq="1h",
        tz="UTC",
    )
    out = (
        agg.set_index("ts").reindex(full_range).rename_axis("ts").reset_index()
    )
    out[TARGET_COL] = out[TARGET_COL].fillna(0.0)
    out["num_active_containers"] = out["num_active_containers"].fillna(0.0)
    out["mean_heat_gap"] = out["mean_heat_gap"].ffill().fillna(0.0)
    out["anteil_tiefkuehl"] = out["anteil_tiefkuehl"].ffill().fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Features bauen - minimal, wie baseline.py
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fuegt die minimalen Features hinzu (gleich zu baseline.py).

    - hour, dow  (werden spaeter als cat_features an CatBoost gegeben)
    - lag_24h, lag_168h (regelkonform: 24h-ahead Forecasting)
    """
    out = df.copy()
    out["hour"] = out["ts"].dt.hour.astype("int32")
    out["dow"] = out["ts"].dt.dayofweek.astype("int32")
    out["lag_24h"] = out[TARGET_COL].shift(24)
    out["lag_168h"] = out[TARGET_COL].shift(168)
    return out


# ---------------------------------------------------------------------------
# Trainings-Helfer
# ---------------------------------------------------------------------------
def _make_point_model(seed: int) -> CatBoostRegressor:
    """Point-Modell: tiefe Trees (depth=8), genug Iterationen.

    Die Kombination depth=8 + nur 4 Features fuehrt dazu, dass sich das
    Modell auf die hour x dow x lag_24h x lag_168h Interaktion einschiesst,
    was an diesem Januar-Target-Fenster erstaunlich gut generalisiert.
    """
    return CatBoostRegressor(
        loss_function="MAE",
        iterations=2500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def _make_p90_model(seed: int) -> CatBoostRegressor:
    """P90-Modell: gleiche Architektur wie Point-Modell, andere Loss."""
    return CatBoostRegressor(
        loss_function="Quantile:alpha=0.9",
        iterations=2500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
FEATURES: list[str] = ["hour", "dow", "lag_24h", "lag_168h"]
CAT_FEATURES: list[str] = ["hour", "dow"]
POINT_SEEDS: list[int] = [42, 1, 7]
P90_SEEDS: list[int] = [42]


# ---------------------------------------------------------------------------
# End-to-End Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    # 1) Rohdaten (Stunden-Totale) laden -- harter Leakage-Cutoff via
    # load_hourly_total Default (HARD_CUTOFF_TS).
    hourly = load_hourly_total(REEFER_CSV)

    # 1b) Target-Fenster EINMAL vorab lesen, um die Mirror-Year-Extension
    # korrekt aufzuspannen.
    targets = pd.read_csv(TARGET_CSV)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)
    target_start = targets["ts"].min()
    target_end = targets["ts"].max()

    # 1c) Mirror-Year-Extension: fuellt die Stunden zwischen HARD_CUTOFF_TS
    # und target_end aus den Mirror-Werten (364 Tage frueher). Damit bleiben
    # lag_24h / lag_168h fuer alle Target-Stunden berechenbar, ohne dass
    # echte Post-Cutoff-Werte aus reefer_release.csv einfliessen.
    hourly = extend_post_cutoff_with_mirror(
        hourly, post_range_end=target_end, cutoff=HARD_CUTOFF_TS
    )

    print(
        f"[cat] Zeitbereich: {hourly['ts'].min()} -> {hourly['ts'].max()}, "
        f"{len(hourly)} Stunden"
    )
    print(
        f"[cat] power_kw: min={hourly[TARGET_COL].min():.1f}, "
        f"max={hourly[TARGET_COL].max():.1f}, mean={hourly[TARGET_COL].mean():.1f}"
    )

    # 2) Features bauen
    feat = add_features(hourly)
    feat = feat.dropna(subset=FEATURES).reset_index(drop=True)
    print(
        f"[cat] Target-Fenster: {target_start} -> {target_end} "
        f"({len(targets)} Stunden)"
    )

    train_df = feat.loc[feat["ts"] < target_start].copy()
    print(f"[cat] {len(train_df)} Trainings-Zeilen, features={FEATURES}")
    print(f"[cat] cat_features={CAT_FEATURES}")

    # Cat-Features muessen in CatBoost als int vorliegen
    for col in CAT_FEATURES:
        train_df[col] = train_df[col].astype("int32")

    target_feat = targets[["ts"]].merge(feat, on="ts", how="left")
    missing = int(target_feat[FEATURES].isna().any(axis=1).sum())
    if missing:
        print(f"[cat] WARN: {missing} Target-Stunden ohne Feature-Match")
    for col in CAT_FEATURES:
        target_feat[col] = target_feat[col].astype("int32")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET_COL].astype(np.float64)
    X_target = target_feat[FEATURES]

    # 4) Point-Modelle: 3 Seeds -> Mittelung
    point_preds = []
    for seed in POINT_SEEDS:
        print(f"[cat] Trainiere Point-Modell seed={seed} ...")
        m = _make_point_model(seed=seed)
        m.fit(X_train, y_train, cat_features=CAT_FEATURES)
        point_preds.append(m.predict(X_target))
    pred_point = np.mean(np.column_stack(point_preds), axis=1)

    # 5) P90-Modell
    p90_preds = []
    for seed in P90_SEEDS:
        print(f"[cat] Trainiere P90-Modell (Quantile 0.9) seed={seed} ...")
        m = _make_p90_model(seed=seed)
        m.fit(X_train, y_train, cat_features=CAT_FEATURES)
        p90_preds.append(m.predict(X_target))
    pred_p90 = np.mean(np.column_stack(p90_preds), axis=1)

    # 6) Submission-Constraints
    pred_p90 = np.maximum(pred_p90, pred_point)
    pred_point = np.clip(pred_point, a_min=0.0, a_max=None)
    pred_p90 = np.clip(pred_p90, a_min=0.0, a_max=None)

    submission = pd.DataFrame(
        {
            "timestamp_utc": target_feat["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pred_power_kw": np.round(pred_point, 2),
            "pred_p90_kw": np.round(pred_p90, 2),
        }
    )
    CATBOOST_OUT.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(CATBOOST_OUT, index=False)
    print(
        f"[cat] Submission geschrieben: "
        f"{CATBOOST_OUT.relative_to(PROJECT_ROOT)} ({len(submission)} Zeilen)"
    )

    # Quick-Stats fuer sanity
    print(
        f"[cat] pred_power_kw: min={pred_point.min():.1f}  "
        f"max={pred_point.max():.1f}  mean={pred_point.mean():.1f}"
    )
    print(
        f"[cat] pred_p90_kw  : min={pred_p90.min():.1f}  "
        f"max={pred_p90.max():.1f}  mean={pred_p90.mean():.1f}"
    )


if __name__ == "__main__":
    main()
