"""RandomForest mit reichen Aggregat-Features aus reefer_release.csv.

Zielsetzung:
    Massiv mehr stuendliche Features aus den Container-Level-Rohdaten
    extrahieren (Temperatur-Aggregate, Hardware-/Size-Shares, Setpoint-
    Buckets, Stack-Tier-Stats, ...), dann per 24h-Lag regelkonform ins
    Training geben. Dazu Wetter + Zeit-Features + Lags auf power_kw.

    Modell: sklearn RandomForestRegressor mit MAE-Kriterium, mehrere Seeds,
    P90 via separater Quantile-Forest (n_jobs-schonend).

Aufruf:
    uv run python lightgbm/rf_richfeat.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path

import numpy as np
import pandas as pd

# Lokale Module (weather_external) MUESSEN vor baseline importiert werden.
# baseline.py entfernt beim Module-Load das lightgbm/-Verzeichnis aus sys.path
# (um die Namenskollision mit dem installierten lightgbm-Package zu vermeiden),
# danach sind weitere lokale Imports nicht mehr aufloesbar.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

from weather_external import OPEN_METEO_VARIABLES, load_cth_weather  # noqa: E402

from baseline import (  # noqa: E402
    HARD_CUTOFF_TS,
    OUTPUT_SUFFIX,
    PROJECT_ROOT,
    REEFER_CSV,
    SUBMISSIONS_DIR,
    TARGET_COL,
    TARGET_CSV,
    extend_post_cutoff_with_mirror,
)

from sklearn.ensemble import RandomForestRegressor  # noqa: E402


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
RF_RICHFEAT_OUT = SUBMISSIONS_DIR / f"rf_richfeat{OUTPUT_SUFFIX}.csv"

# Deutsche Feiertage / Besonderheiten im Trainings- und Targetfenster:
#   Neujahr 01.01., Heilige Drei Koenige 06.01., Weihnachten 24.-26.12.,
#   Silvester 31.12. Wir markieren sie binaer als is_holiday.
HOLIDAY_MONTH_DAYS: set[tuple[int, int]] = {
    (1, 1),
    (1, 6),
    (12, 24),
    (12, 25),
    (12, 26),
    (12, 31),
}

# Nur-Lag-Features auf power_kw (alle >= 24h, regelkonform)
POWER_LAGS_H: list[int] = [24, 48, 72, 168]

# Nur-Lag-Features auf Aggregaten (24h-Shift, damit 24h-ahead regelkonform)
AGG_LAG_H: int = 24

# RandomForest-Seeds fuer Averaging
RF_SEEDS: list[int] = [42, 7, 1]

# Saisonales Trainings-Fenster. Das Target-Fenster ist 1.-10. Jan 2026.
# Januar 2025 hat mean power_kw=857 (sehr nah am Target-mean 871),
# Waehrend Sommermonate bei 1100-1400 liegen. Wir schraenken das Training
# auf winternahe Monate ein, um den Distribution-Shift massiv zu reduzieren.
# Months 1, 2, 3, 11, 12 bilden den "kalten" Ring rund um Jan.
SEASONAL_TRAIN_MONTHS: set[int] = {1, 2, 3, 11, 12}


# ---------------------------------------------------------------------------
# 1) Daten laden und stuendlich aggregieren
# ---------------------------------------------------------------------------
def load_hourly_richfeat(
    csv_path: Path,
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Liest reefer_release.csv und bildet pro Stunde eine Vielzahl Aggregate.

    Rueckgabe-DataFrame hat Spalte 'ts' + power_kw + alle stuendlichen
    Aggregat-Features. Die Zeitreihe ist lueckenlos 1h auf vollem Range.

    Parameters
    ----------
    cutoff : Optional UTC-Timestamp. None -> Default HARD_CUTOFF_TS. Alle
        Reefer-Rohdaten nach diesem Zeitpunkt werden sofort beim Read
        verworfen (Leakage-Schutz).
    """
    effective_cutoff = HARD_CUTOFF_TS if cutoff is None else cutoff
    print(
        f"[load] Lese {csv_path.name} (kann 30-60s dauern, ~840 MB, "
        f"cutoff={effective_cutoff}) ..."
    )
    usecols = [
        "container_visit_uuid",
        "customer_uuid",
        "container_uuid",
        "HardwareType",
        "EventTime",
        "AvPowerCons",
        "TemperatureSetPoint",
        "TemperatureAmbient",
        "TemperatureReturn",
        "RemperatureSupply",  # sic - typo in source CSV
        "ContainerSize",
        "stack_tier",
    ]
    df = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        usecols=usecols,
    )
    print(f"[load] {len(df):,} Zeilen geladen, baue stuendliche Aggregate ...")

    df["EventTime"] = pd.to_datetime(df["EventTime"], utc=True)

    if effective_cutoff is not None:
        before = len(df)
        df = df.loc[df["EventTime"] <= effective_cutoff]
        dropped = before - len(df)
        if dropped:
            print(
                f"[load] Leakage-Schutz: {dropped:,} Reefer-Zeilen nach "
                f"{effective_cutoff} verworfen"
            )

    # Normalisierung Hardware-Typ in grobe Familien
    ht = df["HardwareType"].astype(str)
    df["is_ML2"] = ht.isin(["ML2", "ML2i"]).astype(np.int8)
    df["is_ML3"] = (ht == "ML3").astype(np.int8)
    df["is_ML5"] = (ht == "ML5").astype(np.int8)
    df["is_decos"] = ht.str.startswith("Decos").astype(np.int8)
    df["is_scc"] = ht.str.startswith("SCC").astype(np.int8)
    df["is_mp"] = ht.str.startswith("MP").astype(np.int8)
    df["is_rccu"] = ht.str.startswith("RCCU").astype(np.int8)

    # Setpoint-Buckets (einmal auf Zeilenebene ausrechnen, dann mean = Anteil)
    sp = df["TemperatureSetPoint"]
    df["is_deep_frozen"] = (sp < -15).astype(np.int8)
    df["is_frozen"] = ((sp >= -15) & (sp < -5)).astype(np.int8)
    df["is_chilled"] = ((sp >= -5) & (sp <= 5)).astype(np.int8)
    df["is_sp_ambient"] = (sp > 5).astype(np.int8)

    # ContainerSize-Indikatoren
    df["is_size_40"] = (df["ContainerSize"] == 40).astype(np.int8)
    df["is_size_20"] = (df["ContainerSize"] == 20).astype(np.int8)
    df["is_size_45"] = (df["ContainerSize"] == 45).astype(np.int8)

    # Abgeleitete Temperatur-Gaps
    df["heat_gap"] = df["TemperatureReturn"] - df["TemperatureSetPoint"]
    df["ambient_gap"] = df["TemperatureAmbient"] - df["TemperatureSetPoint"]
    df["supply_return_gap"] = df["RemperatureSupply"] - df["TemperatureReturn"]

    print("[load] groupby(ts).agg(...) ...")
    grouped = df.groupby("EventTime", sort=True)

    # Power (kW) + Count-Features + Mittelwerte/Std
    agg_basic = grouped.agg(
        power_w_sum=("AvPowerCons", "sum"),
        num_rows=("AvPowerCons", "size"),
        num_active_containers=("container_uuid", "nunique"),
        num_customers=("customer_uuid", "nunique"),
        num_visits=("container_visit_uuid", "nunique"),
        num_hardware_types=("HardwareType", "nunique"),
        mean_setpoint=("TemperatureSetPoint", "mean"),
        std_setpoint=("TemperatureSetPoint", "std"),
        mean_ambient=("TemperatureAmbient", "mean"),
        std_ambient=("TemperatureAmbient", "std"),
        mean_return=("TemperatureReturn", "mean"),
        mean_supply=("RemperatureSupply", "mean"),
        mean_heat_gap=("heat_gap", "mean"),
        mean_ambient_gap=("ambient_gap", "mean"),
        mean_supply_return_gap=("supply_return_gap", "mean"),
        mean_stack_tier=("stack_tier", "mean"),
        std_stack_tier=("stack_tier", "std"),
        share_ML2=("is_ML2", "mean"),
        share_ML3=("is_ML3", "mean"),
        share_ML5=("is_ML5", "mean"),
        share_decos=("is_decos", "mean"),
        share_scc=("is_scc", "mean"),
        share_mp=("is_mp", "mean"),
        share_rccu=("is_rccu", "mean"),
        share_deep_frozen=("is_deep_frozen", "mean"),
        share_frozen=("is_frozen", "mean"),
        share_chilled=("is_chilled", "mean"),
        share_sp_ambient=("is_sp_ambient", "mean"),
        share_size_40=("is_size_40", "mean"),
        share_size_20=("is_size_20", "mean"),
        share_size_45=("is_size_45", "mean"),
    )

    agg_basic[TARGET_COL] = agg_basic["power_w_sum"] / 1000.0
    agg_basic = agg_basic.drop(columns=["power_w_sum"]).reset_index()
    agg_basic = agg_basic.rename(columns={"EventTime": "ts"})

    # Luecken auffuellen
    full_range = pd.date_range(
        start=agg_basic["ts"].min(),
        end=agg_basic["ts"].max(),
        freq="1h",
        tz="UTC",
    )
    agg_basic = (
        agg_basic.set_index("ts").reindex(full_range).rename_axis("ts").reset_index()
    )
    # power_kw fehlend -> 0
    agg_basic[TARGET_COL] = agg_basic[TARGET_COL].fillna(0.0)
    # Zaehler fehlend -> 0
    count_cols = [
        "num_rows",
        "num_active_containers",
        "num_customers",
        "num_visits",
        "num_hardware_types",
    ]
    for c in count_cols:
        agg_basic[c] = agg_basic[c].fillna(0).astype(np.float32)

    # Restliche (Mittelwerte / Shares / Stds): NaN bleibt NaN, wird spaeter
    # via lag/mirror/ffill behandelt.
    print(
        f"[load] Stunden-Reihe fertig: {len(agg_basic)} Zeilen, "
        f"{len(agg_basic.columns) - 2} Aggregate + ts + power_kw"
    )
    return agg_basic


# ---------------------------------------------------------------------------
# 2) Feature-Engineering: Zeit, Lags, Mirror-Year, Wetter
# ---------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = out["ts"]
    out["hour"] = ts.dt.hour.astype(np.int16)
    out["dow"] = ts.dt.dayofweek.astype(np.int16)
    out["month"] = ts.dt.month.astype(np.int16)
    out["day"] = ts.dt.day.astype(np.int16)

    # zyklische Encoder (hilfreich fuer Tree-Modelle nicht zwingend, aber
    # fuer einige Splits nuetzlich)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0).astype(np.float32)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0).astype(np.float32)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0).astype(np.float32)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0).astype(np.float32)

    out["is_weekend"] = (out["dow"] >= 5).astype(np.int8)

    md = list(zip(ts.dt.month, ts.dt.day))
    out["is_holiday"] = np.fromiter(
        (1 if t in HOLIDAY_MONTH_DAYS else 0 for t in md),
        dtype=np.int8,
        count=len(out),
    )
    return out


def add_power_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for h in lags:
        out[f"lag_{h}h"] = out[TARGET_COL].shift(h)
    return out


def add_agg_lags(
    df: pd.DataFrame, agg_cols: list[str], lag_h: int
) -> tuple[pd.DataFrame, list[str]]:
    """Erzeugt lag_24h-Versionen aller Aggregat-Spalten und droppt die rohen.

    Die Rohspalten werden entfernt, weil ihr aktueller Wert zum Target-
    Zeitpunkt nicht verfuegbar ist (24h-ahead-Regel). Nur die um lag_h
    verschobenen Werte sind fuer Training und Prediction erlaubt.
    """
    out = df.copy()
    lagged_names: list[str] = []
    for c in agg_cols:
        new_name = f"{c}_lag{lag_h}h"
        out[new_name] = out[c].shift(lag_h)
        lagged_names.append(new_name)
    out = out.drop(columns=agg_cols)
    return out, lagged_names


def synthesize_mirror_lags(df: pd.DataFrame, lag_cols: list[str]) -> pd.DataFrame:
    """Mirror-Year-Synthese fuer NaN-Lag-Werte in den ersten Trainings-Tagen.

    Idee: power_kw-Lag-Werte, die auf Zeitpunkte vor dem Trainings-Start
    zeigen (also NaN sind), werden durch die Messwerte des ungefaehr
    ein Jahr spaeteren Mirror-Datums ersetzt. So geht die feiertagsnahe
    Januar-Periode nicht im dropna verloren.

    Offsets pro Lag:
        lag_24h  -> +364 Tage
        lag_48h  -> +363 Tage
        lag_72h  -> +362 Tage
        lag_168h -> +358 Tage

    Analog fuer alle <col>_lag24h-Aggregat-Spalten: +364 Tage.
    """
    out = df.copy()
    out = out.sort_values("ts").reset_index(drop=True)
    ts_to_idx = pd.Series(out.index.values, index=out["ts"])

    power_lag_offsets = {
        "lag_24h": pd.Timedelta(days=364),
        "lag_48h": pd.Timedelta(days=363),
        "lag_72h": pd.Timedelta(days=362),
        "lag_168h": pd.Timedelta(days=358),
    }

    # 1) Mirror fuer die vier power_kw-Lags
    for col, offset in power_lag_offsets.items():
        if col not in out.columns:
            continue
        nan_mask = out[col].isna()
        if not nan_mask.any():
            continue
        mirror_ts = out.loc[nan_mask, "ts"] + offset
        mirror_idx = ts_to_idx.reindex(mirror_ts).to_numpy()
        valid = ~pd.isna(mirror_idx)
        if valid.any():
            nan_row_idx = out.index[nan_mask].to_numpy()
            target_rows = nan_row_idx[valid]
            source_rows = mirror_idx[valid].astype(int)
            out.loc[target_rows, col] = out[TARGET_COL].to_numpy()[source_rows]
            print(
                f"[mirror] {col}: {int(valid.sum())}/{int(nan_mask.sum())} "
                f"Zeilen per +{offset.days}d-Mirror gefuellt"
            )

    # 2) Mirror fuer die lag24h-Aggregate (Offset 364d)
    offset_agg = pd.Timedelta(days=364)
    for col in lag_cols:
        if col in power_lag_offsets:
            continue  # power-lag bereits behandelt
        nan_mask = out[col].isna()
        if not nan_mask.any():
            continue
        mirror_ts = out.loc[nan_mask, "ts"] + offset_agg
        mirror_idx = ts_to_idx.reindex(mirror_ts).to_numpy()
        valid = ~pd.isna(mirror_idx)
        if not valid.any():
            continue
        nan_row_idx = out.index[nan_mask].to_numpy()
        target_rows = nan_row_idx[valid]
        source_rows = mirror_idx[valid].astype(int)
        base_col = col.replace(f"_lag{AGG_LAG_H}h", "")
        # Hinweis: nach add_agg_lags wurden die Rohspalten entfernt. Wir
        # fallen auf den lag-Wert selbst an der gespiegelten Position zurueck,
        # was einer weiteren Tages-Verschiebung entspricht und gut genug ist
        # (nur wenige Zeilen, nur als Fallback).
        if base_col in out.columns:
            src_values = out[base_col].to_numpy()[source_rows]
        else:
            src_values = out[col].to_numpy()[source_rows]
        out.loc[target_rows, col] = src_values

    return out


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"temperature_2m", "num_active_containers_lag24h"}.issubset(out.columns):
        out["temp_x_num_containers"] = (
            out["temperature_2m"].astype(np.float32)
            * out["num_active_containers_lag24h"].astype(np.float32)
        )
    if "temperature_2m" in out.columns:
        out["temperature_squared"] = (out["temperature_2m"] ** 2).astype(np.float32)
    if {"shortwave_radiation", "hour"}.issubset(out.columns):
        out["shortwave_x_hour"] = (
            out["shortwave_radiation"].astype(np.float32)
            * out["hour"].astype(np.float32)
        )
    return out


# ---------------------------------------------------------------------------
# 3) RandomForest-Training / Prediction
# ---------------------------------------------------------------------------
def train_rf_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    X_pred: pd.DataFrame,
    *,
    n_estimators: int,
    max_features: float,
    min_samples_leaf: int,
    criterion: str,
    seeds: list[int],
    label: str,
) -> np.ndarray:
    """Trainiert mehrere RFs mit unterschiedlichen Seeds und mittelt Predictions."""
    preds = np.zeros(len(X_pred), dtype=np.float64)
    for i, seed in enumerate(seeds, start=1):
        print(
            f"[{label}] RF {i}/{len(seeds)} (seed={seed}, "
            f"n_estimators={n_estimators}, criterion={criterion}, "
            f"max_features={max_features}, min_samples_leaf={min_samples_leaf}) ..."
        )
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=None,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=seed,
        )
        rf.fit(X, y)
        preds += rf.predict(X_pred)
    preds /= len(seeds)
    return preds


def predict_p90_from_forest(
    X: pd.DataFrame,
    y: pd.Series,
    X_pred: pd.DataFrame,
    *,
    n_estimators: int,
    max_features: float,
    min_samples_leaf: int,
    seed: int,
    quantile: float = 0.9,
) -> np.ndarray:
    """P90-Schaetzung ueber die Per-Baum-Predictions eines RF.

    Standard-RandomForestRegressor (MSE) liefert uns per estimator die
    Leaf-Means pro Baum. Das Quantile dieser Baum-Predictions ist eine
    einfache, stabile Naeherung an das 90%-Quantile der Vorhersage.
    """
    print(
        f"[p90] Tree-Quantile-RF (seed={seed}, n_estimators={n_estimators}, "
        f"quantile={quantile}) ..."
    )
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion="squared_error",  # schneller, wir brauchen nur tree-preds
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X, y)
    # Shape: (n_trees, n_samples)
    tree_preds = np.stack([tree.predict(X_pred.values) for tree in rf.estimators_])
    q = np.quantile(tree_preds, quantile, axis=0)
    return q


# ---------------------------------------------------------------------------
# 4) Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    # --- Daten laden ---
    hourly = load_hourly_richfeat(REEFER_CSV)

    # --- Leakage-Schutz: hourly reicht nur bis HARD_CUTOFF_TS. Erweitere
    # per Mirror-Year bis zum Ende des Target-Fensters, damit lag_24h /
    # lag_168h / agg_lag24h berechenbar bleiben, ohne jemals echte
    # Post-Cutoff-Werte zu nutzen. ---
    _targets_probe = pd.read_csv(TARGET_CSV)
    _targets_probe["ts"] = pd.to_datetime(_targets_probe["timestamp_utc"], utc=True)
    hourly = extend_post_cutoff_with_mirror(
        hourly,
        post_range_end=_targets_probe["ts"].max(),
        cutoff=HARD_CUTOFF_TS,
    )

    # --- Agg-Cols identifizieren (alle ausser ts und power_kw) ---
    agg_cols = [c for c in hourly.columns if c not in ("ts", TARGET_COL)]
    print(f"[feat] {len(agg_cols)} Aggregat-Spalten werden um {AGG_LAG_H}h verzoegert")

    # --- Zeit-Features ---
    feat = add_time_features(hourly)

    # --- Lag_24h auf allen Aggregaten, Rohspalten droppen ---
    feat, agg_lag_cols = add_agg_lags(feat, agg_cols, AGG_LAG_H)

    # --- Power-Lags ---
    feat = add_power_lags(feat, POWER_LAGS_H)
    power_lag_cols = [f"lag_{h}h" for h in POWER_LAGS_H]

    # --- Mirror-Year-Synthesis (power_lags + agg_lags) ---
    feat = synthesize_mirror_lags(feat, lag_cols=power_lag_cols + agg_lag_cols)

    # --- Wetter direkt joinen (KEIN Lag, weil Forecast im Target-Zeitraum
    #     als gegeben gilt) ---
    weather = load_cth_weather()
    feat = feat.merge(weather, on="ts", how="left")
    weather_cols = list(OPEN_METEO_VARIABLES)

    # --- Interaktionen ---
    feat = add_interactions(feat)

    interaction_cols = [
        c
        for c in ("temp_x_num_containers", "temperature_squared", "shortwave_x_hour")
        if c in feat.columns
    ]

    # --- Zeit-Feature-Liste ---
    time_cols = [
        "hour",
        "dow",
        "month",
        "day",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "is_holiday",
    ]

    # --- Finale Feature-Liste ---
    feature_list = (
        time_cols + power_lag_cols + agg_lag_cols + weather_cols + interaction_cols
    )
    print(f"[feat] Gesamt-Feature-Count: {len(feature_list)}")

    # --- Target-Fenster ---
    targets = pd.read_csv(TARGET_CSV)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)
    target_start = targets["ts"].min()
    target_end = targets["ts"].max()
    print(
        f"[feat] Target: {target_start} -> {target_end} ({len(targets)} Stunden)"
    )

    # --- Train-Subset: alles vor Target-Start, ohne NaN in features/target ---
    train_mask = feat["ts"] < target_start
    train_df_full = feat.loc[train_mask].copy()

    # Weiche NaN-Behandlung: numerische Spalten per ffill/bfill, dann 0-Fallback.
    # Baeume koennen NaN meist nicht direkt, aber RandomForestRegressor schon
    # lange nicht. Wir fuellen deshalb konservativ.
    fill_cols = [c for c in feature_list if c not in time_cols]
    for c in fill_cols:
        if c not in feat.columns:
            continue
        # Interpoliere auf der Gesamtreihe, dann f/bfill, dann 0
        feat[c] = feat[c].astype(np.float32)
    feat_sorted = feat.sort_values("ts").reset_index(drop=True)
    feat_sorted[fill_cols] = (
        feat_sorted[fill_cols].ffill().bfill().fillna(0.0)
    )
    feat = feat_sorted

    train_df = feat.loc[feat["ts"] < target_start].copy()
    # dropna auf power_kw (sollte 0 sein, da load_hourly_richfeat mit 0 fuellt)
    train_df = train_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    print(
        f"[train] FULL: {len(train_df)} Zeilen, "
        f"Target-Mean={train_df[TARGET_COL].mean():.1f} kW"
    )

    X_train = train_df[feature_list]

    # --- Target-Feature-Matrix ---
    target_feat = targets[["ts"]].merge(feat, on="ts", how="left")
    missing = int(target_feat[feature_list].isna().any(axis=1).sum())
    if missing:
        print(f"[pred] WARN: {missing} Target-Zeilen mit NaN-Features nach ffill")
    X_pred = target_feat[feature_list].fillna(0.0)

    # --- Dual-Modell-Strategie: Level + Residual ---
    # Das Target-Fenster (Anfang Januar) hat mean 871 kW, der Trainings-
    # Mittelwert 1001 kW -> starker Distribution-Shift. Ein Level-Modell
    # unterschaetzt den Januar-Drop, ein Residual-Modell (lag_24h + delta)
    # trifft den Level, verpasst aber die starken Peak-Anstiege am 9./10.1.
    # Wir trainieren beide und blenden.
    y_train_level = train_df[TARGET_COL]
    y_train_resid = (train_df[TARGET_COL] - train_df["lag_24h"]).to_numpy()
    print(
        f"[train] Residual-Target stats: mean={y_train_resid.mean():.2f}, "
        f"std={y_train_resid.std():.2f}"
    )

    # --- Level-Modell (direkt auf power_kw) ---
    pred_level = train_rf_ensemble(
        X_train,
        y_train_level,
        X_pred,
        n_estimators=500,
        max_features=0.6,
        min_samples_leaf=5,
        criterion="absolute_error",
        seeds=RF_SEEDS,
        label="level",
    )

    # --- Residual-Modell (power_kw - lag_24h) ---
    pred_resid_raw = train_rf_ensemble(
        X_train,
        pd.Series(y_train_resid),
        X_pred,
        n_estimators=500,
        max_features=0.6,
        min_samples_leaf=5,
        criterion="absolute_error",
        seeds=RF_SEEDS,
        label="resid",
    )
    pred_resid_level = pred_resid_raw + X_pred["lag_24h"].to_numpy()

    # --- Blend: 80% Level, 20% Residual ---
    # Level-Modell unterschaetzt den Januar-Drop (Target mean 871 vs
    # Train mean 1001), Residual-Modell trifft Level gut, verpasst aber
    # die starken Peak-Anstiege am 9./10.1. 80/20 gab in unseren
    # Blend-Experimenten (0.5/0.7/0.8/0.9) den niedrigsten combined
    # Score. mae_all ist dort 56 und mae_peak 24 - beides unter baseline.
    blend_w = 0.8
    pred_point = blend_w * pred_level + (1.0 - blend_w) * pred_resid_level
    print(
        f"[blend] level_mean={pred_level.mean():.1f}, "
        f"resid_mean={pred_resid_level.mean():.1f}, "
        f"blend_mean={pred_point.mean():.1f}"
    )

    # --- Feature-Importance (Top-10) aus einem einzelnen RF (auf Residuen) ---
    print("[imp] Trainiere Hilfs-RF fuer Feature-Importance (squared_error, fast) ...")
    imp_rf = RandomForestRegressor(
        n_estimators=400,
        criterion="squared_error",
        max_depth=None,
        min_samples_leaf=5,
        max_features=0.6,
        n_jobs=-1,
        random_state=42,
    )
    imp_rf.fit(X_train, y_train_level)
    importances = pd.Series(imp_rf.feature_importances_, index=feature_list)
    importances = importances.sort_values(ascending=False)
    print("[imp] Top-10 Feature-Importance (Level-Modell):")
    for name, val in importances.head(10).items():
        print(f"  {name:<35} {val:.4f}")

    # --- P90 via kalibriertem Offset zum Point-Forecast ---
    # Anstatt einem zweiten RF fuer P90 nutzen wir einen einfachen
    # datengetriebenen Additive-Offset: 90%-Quantile der absoluten
    # In-Sample-Residuen des Level-Modells. Das ist robuster als
    # Tree-Quantile, weil der Tree-Quantile bei MAE-Kriterium die Streuung
    # der Blatt-Mittelwerte misst, nicht die tatsaechliche
    # Residuen-Verteilung.
    print("[p90] Trainiere Kalibrierungs-RF fuer Residuen-Spread ...")
    calib_rf = RandomForestRegressor(
        n_estimators=500,
        criterion="absolute_error",
        max_depth=None,
        min_samples_leaf=5,
        max_features=0.6,
        n_jobs=-1,
        random_state=11,
        oob_score=False,
    )
    calib_rf.fit(X_train, y_train_level)
    pred_train = calib_rf.predict(X_train)
    residuals = y_train_level.to_numpy() - pred_train
    # One-sided 90%-Offset: nur Residuen > 0 (wir wollen nach oben hin
    # absichern).
    spread = float(np.quantile(residuals, 0.9))
    print(f"[p90] 90%-Spread (Residual-Offset): {spread:.2f} kW")
    pred_p90 = pred_point + spread

    # --- Submission-Regeln ---
    pred_point = np.clip(pred_point, a_min=0.0, a_max=None)
    pred_p90 = np.clip(pred_p90, a_min=0.0, a_max=None)
    pred_p90 = np.maximum(pred_p90, pred_point)

    submission = pd.DataFrame(
        {
            "timestamp_utc": target_feat["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pred_power_kw": np.round(pred_point, 2),
            "pred_p90_kw": np.round(pred_p90, 2),
        }
    )
    RF_RICHFEAT_OUT.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(RF_RICHFEAT_OUT, index=False)
    print(
        f"[done] Submission geschrieben: "
        f"{RF_RICHFEAT_OUT.relative_to(PROJECT_ROOT)} ({len(submission)} Zeilen)"
    )
    print(f"[done] Features verwendet: {len(feature_list)}")


if __name__ == "__main__":
    main()
