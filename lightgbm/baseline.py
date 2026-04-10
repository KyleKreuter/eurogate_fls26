"""Fester Referenz-Baseline fuer die Reefer Peak Load Challenge (Eurogate CTH Hamburg).

Diese Datei ist bewusst minimal und eingefroren: simpelste sinnvolle Features,
kein Peak-Weighting, keine Hyperparameter-Tricks. Sie dient als "ehrlicher
schlechtester Score", gegen den wir jede Verbesserung in productive.py messen.

Pipeline:
    Rohdaten -> Stunden-Totale -> minimale Features -> 2x LightGBM
    -> submissions/baseline.csv

Zwei Modelle:
    - Point-Forecast  -> objective='regression_l1'  (MAE -> mae_all/mae_peak)
    - P90-Quantile    -> objective='quantile', alpha=0.9  (-> pinball_p90)

Zusaetzlich exportiert diese Datei die gemeinsamen Bausteine
(load_hourly_total, add_features, train_lgbm, mae, pinball, Konstanten),
die von productive.py und eval.py per Import wiederverwendet werden.

Ausfuehren:
    uv run python lightgbm/baseline.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Der Ordner dieses Skripts heisst "lightgbm/" und kollidiert beim Import mit
# dem installierten Package. Wir entfernen das Skript-Verzeichnis aus sys.path,
# bevor wir lightgbm importieren, damit der Import eindeutig im site-packages
# landet.
import sys as _sys

_HERE = Path(__file__).resolve().parent
_sys.path = [p for p in _sys.path if Path(p).resolve() != _HERE]

import lightgbm as lgb  # noqa: E402


# ---------------------------------------------------------------------------
# Projekt-Konstanten (werden von productive.py und eval.py importiert)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATEN_DIR = PROJECT_ROOT / "participant_package" / "daten"
REEFER_CSV = DATEN_DIR / "reefer_release.csv"

# Per default das offizielle Target. Ueber die Environment-Variable
# EUROGATE_TARGET_CSV kann ein alternatives Target-Fenster injiziert werden
# (z.B. ein Pre-Target Holdout fuer ehrliches Stacking). Alle Base-Scripts,
# die diese Konstante importieren, nutzen dann automatisch das Alternativ-
# Target. Trainings-Cutoff = min(target_ts) -> legal per Konstruktion.
_TARGET_CSV_ENV = os.environ.get("EUROGATE_TARGET_CSV")
TARGET_CSV = Path(_TARGET_CSV_ENV) if _TARGET_CSV_ENV else DATEN_DIR / "target_timestamps.csv"

# ---------------------------------------------------------------------------
# OPTIONALER HARD-CUTOFF (Defense in Depth, default deaktiviert)
# ---------------------------------------------------------------------------
# Hintergrund: Die offizielle Challenge-Spec (EVALUATION_AND_WINNER_SELECTION.md,
# Zeile 14 und 50) sagt explizit, dass der Organizer-Rerun mit der
# "complete reefer release data" laeuft und empfiehlt "yesterday's same hour"
# als Baseline-Feature. Das heisst, lag_24h aus echten 2026er Werten ist im
# 24h-ahead-Setting regelkonform - das Modell wird einmal trainiert (cutoff
# bei target_start), und die Prediction darf alles kennen, was zum Zeitpunkt
# t-24h bekannt war.
#
# Deswegen steht HARD_CUTOFF_TS default in der fernen Zukunft -> alle
# Loader lesen die volle Reihe, extend_post_cutoff_with_mirror wird zum
# No-Op (post_range_end <= cutoff).
#
# Wenn man den strikten "Single-Shot-Forecast"-Modus wieder aktivieren
# moechte (z.B. fuer einen internen Leakage-Audit), setzt man die Env-Var
# EUROGATE_HARD_CUTOFF=2025-12-31T23:00:00. Die Mechanik bleibt als
# Defense-in-Depth im Code erhalten.
_HARD_CUTOFF_ENV = os.environ.get("EUROGATE_HARD_CUTOFF")
HARD_CUTOFF_TS: pd.Timestamp = pd.Timestamp(
    _HARD_CUTOFF_ENV if _HARD_CUTOFF_ENV else "2099-12-31 23:00:00",
    tz="UTC",
)
# Mirror-Year Offset: 364 Tage = exakt 52 Wochen, erhaelt den Wochentag.
MIRROR_YEAR_OFFSET = pd.Timedelta(days=364)

OUTPUT_SUFFIX = os.environ.get("EUROGATE_OUTPUT_SUFFIX", "")

# Result-Dateien leben unterhalb des lightgbm/-Ordners, nicht im Projekt-Root.
LIGHTGBM_DIR = Path(__file__).resolve().parent
SUBMISSIONS_DIR = LIGHTGBM_DIR / "submissions"
BASELINE_OUT = SUBMISSIONS_DIR / f"baseline{OUTPUT_SUFFIX}.csv"

FEATURES: list[str] = ["hour", "dow", "lag_24h", "lag_168h"]
TARGET_COL: str = "power_kw"

# Container-Mix-Features, die pro Stunde aus reefer_release.csv aggregiert
# werden. productive.py baut spaeter lag_24h-Versionen davon.
CONTAINER_MIX_BASE_COLS: list[str] = [
    "num_active_containers",
    "mean_heat_gap",
    "anteil_tiefkuehl",
]

# Definition fuer "Spitzenlast-Stunden" - wird in eval.py und productive.py
# konsistent verwendet. Oberste 15% der wahren Werte.
PEAK_QUANTILE: float = 0.85


# ---------------------------------------------------------------------------
# 1) Daten laden und aggregieren
# ---------------------------------------------------------------------------
def load_hourly_total(
    csv_path: Path,
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Liest die Reefer-Rohdaten und gibt Stunden-Totale in kW zurueck.

    - Deutsches CSV-Format: Trennzeichen ';' und Dezimalkomma ','
    - Nur die zwei noetigen Spalten einlesen -> deutlich weniger Speicher
    - EventTime wird als UTC interpretiert (laut Doku bereits UTC)
    - Leere Stunden werden zu 0.0 kW aufgefuellt, damit `shift(24)` spaeter
      wirklich "24 Stunden zurueck" bedeutet und nicht "24 Zeilen zurueck".

    Parameters
    ----------
    cutoff : Optional UTC-Timestamp. Wenn None -> Default HARD_CUTOFF_TS
        (KEINE Reefer-Rohdaten nach 2025-12-31 23:00 UTC). eval.py ruft
        explizit mit cutoff=None auf, um die volle Reihe fuer den
        Ground-Truth zu lesen. Alle Trainings-/Feature-Scripts MUESSEN
        den Default verwenden.
    """
    effective_cutoff = HARD_CUTOFF_TS if cutoff is None else cutoff
    # cutoff=False (via Sentinel) ermoeglicht eval.py, den Schutz gezielt
    # zu deaktivieren. Wir mappen False -> None (= voller Read).
    print(f"[load] Lese {csv_path.name} (cutoff={effective_cutoff}) ...")
    df = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        usecols=["EventTime", "AvPowerCons"],
    )
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

    hourly = (
        df.groupby("EventTime", sort=True)["AvPowerCons"]
        .sum()
        .div(1000.0)  # Watt -> Kilowatt
        .rename(TARGET_COL)
        .reset_index()
        .rename(columns={"EventTime": "ts"})
    )

    full_range = pd.date_range(
        start=hourly["ts"].min(),
        end=hourly["ts"].max(),
        freq="1h",
        tz="UTC",
    )
    hourly = (
        hourly.set_index("ts")
        .reindex(full_range)
        .rename_axis("ts")
        .reset_index()
    )
    hourly[TARGET_COL] = hourly[TARGET_COL].fillna(0.0)
    return hourly


# ---------------------------------------------------------------------------
# 1b) Container-Mix-Loader (fuer productive.py)
# ---------------------------------------------------------------------------
def load_hourly_with_container_mix(
    csv_path: Path,
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Wie load_hourly_total, aber mit zusaetzlichen Container-Mix-Spalten.

    Liest den CSV EINMAL mit mehreren Spalten und aggregiert pro Stunde:
        - power_kw                (Summe AvPowerCons in kW)
        - num_active_containers   (Anzahl Container-Zeilen pro Stunde)
        - mean_heat_gap           (mean(TemperatureReturn - TemperatureSetPoint))
        - anteil_tiefkuehl        (Anteil Container mit SetPoint < -15 deg C)

    Die neuen Features sind physikalisch naeher am Stromverbrauch als externe
    Wetter- oder Kalender-Daten und sind unempfindlich gegen Distribution Shift,
    weil sie direkt aus den gleichen Reefer-Rohdaten kommen.

    Parameters
    ----------
    cutoff : Optional UTC-Timestamp. None -> Default HARD_CUTOFF_TS. Rohdaten
        nach dem Cutoff werden sofort beim Read verworfen (Leakage-Schutz).
    """
    effective_cutoff = HARD_CUTOFF_TS if cutoff is None else cutoff
    print(
        f"[load] Lese {csv_path.name} (mit Container-Mix, "
        f"cutoff={effective_cutoff}) ..."
    )
    df = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        usecols=[
            "EventTime",
            "AvPowerCons",
            "TemperatureSetPoint",
            "TemperatureReturn",
        ],
    )
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

    # Per-Zeilen-Features fuer die Aggregation
    df["heat_gap"] = df["TemperatureReturn"] - df["TemperatureSetPoint"]
    df["is_tiefkuehl"] = (df["TemperatureSetPoint"] < -15).astype("int8")

    hourly = (
        df.groupby("EventTime", sort=True)
        .agg(
            power_sum_w=("AvPowerCons", "sum"),
            num_active_containers=("AvPowerCons", "count"),
            mean_heat_gap=("heat_gap", "mean"),
            anteil_tiefkuehl=("is_tiefkuehl", "mean"),
        )
        .reset_index()
        .rename(columns={"EventTime": "ts"})
    )
    hourly[TARGET_COL] = hourly["power_sum_w"].div(1000.0)
    hourly = hourly.drop(columns=["power_sum_w"])

    # Stuendlich reindexen, damit shift(24) spaeter wirklich "24 Stunden zurueck"
    # bedeutet und nicht "24 Zeilen zurueck".
    full_range = pd.date_range(
        start=hourly["ts"].min(),
        end=hourly["ts"].max(),
        freq="1h",
        tz="UTC",
    )
    hourly = (
        hourly.set_index("ts")
        .reindex(full_range)
        .rename_axis("ts")
        .reset_index()
    )
    hourly[TARGET_COL] = hourly[TARGET_COL].fillna(0.0)
    hourly["num_active_containers"] = (
        hourly["num_active_containers"].fillna(0).astype("int32")
    )
    # Bei mean-Spalten: kurzer Forward-Fill, Rest mit 0.
    hourly["mean_heat_gap"] = hourly["mean_heat_gap"].ffill().fillna(0.0)
    hourly["anteil_tiefkuehl"] = hourly["anteil_tiefkuehl"].ffill().fillna(0.0)
    return hourly


# ---------------------------------------------------------------------------
# 1c) Mirror-Year-Extension fuer die Zeit nach dem HARD_CUTOFF
# ---------------------------------------------------------------------------
def extend_post_cutoff_with_mirror(
    hourly: pd.DataFrame,
    post_range_end: pd.Timestamp,
    mirror_cols: list[str] | None = None,
    cutoff: pd.Timestamp | None = None,
    mirror_offset: pd.Timedelta = MIRROR_YEAR_OFFSET,
) -> pd.DataFrame:
    """Erweitert eine pre-cutoff hourly-Reihe um synthetische Post-Cutoff-Stunden.

    Fuer jede Stunde im Bereich (cutoff, post_range_end] wird eine neue Zeile
    angelegt. Die Werte fuer alle Spalten in `mirror_cols` kommen aus dem
    Mirror-Year-Lookup (Zeitpunkt minus `mirror_offset`, default 364 Tage =
    52 Wochen, Wochentags-treu). Damit bleiben lag_24h / lag_168h fuer die
    Target-Prediction berechenbar, OHNE dass jemals echte Post-Cutoff-Werte
    aus reefer_release.csv eingelesen werden.

    Rationale: productive.py hat dieses Muster bereits zum Auffuellen der
    ersten Januar-Trainings-Zeilen (Jan 2025 nutzt Dec 2025 als Mirror).
    Hier wenden wir es spiegelbildlich auf das Target-Fenster an.

    Parameters
    ----------
    hourly : DataFrame mit Spalte 'ts' (UTC), monoton sortiert, ohne Luecken.
        Wird NICHT in place veraendert.
    post_range_end : Bis einschliesslich dieser UTC-Stunde wird erweitert.
        Typischerweise max(target_timestamps['ts']).
    mirror_cols : Welche Spalten per Mirror-Year gefuellt werden sollen.
        None -> alle numerischen Spalten ausser 'ts'.
    cutoff : Startpunkt der Erweiterung. None -> HARD_CUTOFF_TS.
    mirror_offset : Timedelta zurueck fuer das Mirror-Lookup.

    Returns
    -------
    Erweiterter DataFrame mit luecken-loser stuendlicher Reihe von
    hourly['ts'].min() bis post_range_end.
    """
    effective_cutoff = HARD_CUTOFF_TS if cutoff is None else cutoff
    if post_range_end <= effective_cutoff:
        return hourly.copy()

    if mirror_cols is None:
        mirror_cols = [c for c in hourly.columns if c != "ts"]

    # Lookup: ts -> row
    hourly_indexed = hourly.set_index("ts")

    post_start = effective_cutoff + pd.Timedelta(hours=1)
    post_range = pd.date_range(post_start, post_range_end, freq="1h", tz="UTC")

    mirror_ts = post_range - mirror_offset
    post_df = pd.DataFrame({"ts": post_range})
    filled_count = 0
    for col in mirror_cols:
        if col not in hourly_indexed.columns:
            post_df[col] = 0.0
            continue
        src = hourly_indexed[col]
        # map ueber Index (ts) - fehlende Mirror-Werte werden NaN
        vals = src.reindex(mirror_ts).to_numpy()
        post_df[col] = vals
        filled_count += int(pd.notna(vals).sum())

    # NaN-Fallback: Vorwaerts-/Rueckwaerts-Fill aus den pre-cutoff Zeilen,
    # dann 0. Wichtig fuer Mix-Spalten wie mean_heat_gap, wo "0" physikalisch
    # Unsinn waere.
    for col in mirror_cols:
        if post_df[col].isna().any():
            if col == TARGET_COL:
                post_df[col] = post_df[col].fillna(0.0)
            else:
                # Letzter bekannter Pre-Cutoff-Wert als konservativer Fallback
                last_known = hourly_indexed[col].dropna().iloc[-1] if col in hourly_indexed.columns and hourly_indexed[col].dropna().size else 0.0
                post_df[col] = post_df[col].fillna(last_known)

    print(
        f"[mirror-ext] {len(post_range)} Stunden nach Cutoff per Mirror-Year "
        f"({mirror_offset.days}d) synthetisiert "
        f"(Mirror-Treffer: {filled_count}/{len(post_range) * len(mirror_cols)})"
    )

    result = pd.concat([hourly, post_df], ignore_index=True, sort=False)
    result = result.sort_values("ts").reset_index(drop=True)
    return result


def add_container_mix_lags(
    df: pd.DataFrame,
    mix_cols: list[str] | None = None,
    lag: int = 24,
) -> pd.DataFrame:
    """Fuegt lag-<lag>h Versionen der Mix-Spalten hinzu.

    Fuer jede Spalte `col` in `mix_cols` wird eine neue Spalte
    `{col}_lag_{lag}h` angelegt, die den Wert `lag` Stunden frueher enthaelt.
    Das ist wichtig fuer die 24h-ahead-Forecasting-Semantik: Zum
    Vorhersagezeitpunkt fuer Stunde t kennen wir den Container-Mix zu t-24h,
    aber nicht zum Zeitpunkt t selbst.
    """
    if mix_cols is None:
        mix_cols = CONTAINER_MIX_BASE_COLS
    out = df.sort_values("ts").reset_index(drop=True).copy()
    for col in mix_cols:
        out[f"{col}_lag_{lag}h"] = out[col].shift(lag)
    return out


# ---------------------------------------------------------------------------
# 2) Features bauen
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fuegt die minimalen Baseline-Features hinzu.

    Erwartet eine sortierte Stundenreihe ohne Luecken (wie aus load_hourly_total).
    """
    out = df.copy()
    out["hour"] = out["ts"].dt.hour.astype("int16")
    out["dow"] = out["ts"].dt.dayofweek.astype("int16")
    out["lag_24h"] = out[TARGET_COL].shift(24)
    out["lag_168h"] = out[TARGET_COL].shift(168)
    return out


# ---------------------------------------------------------------------------
# 3) LightGBM-Training
# ---------------------------------------------------------------------------
# Default-Hyperparameter fuer LightGBM. baseline.py nutzt diese unveraendert.
# productive.py kann einzelne Keys ueberschreiben via params_override.
DEFAULT_LGBM_PARAMS: dict = {
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}


def train_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    objective: str,
    alpha: float | None = None,
    num_boost_round: int = 500,
    weight: np.ndarray | pd.Series | None = None,
    params_override: dict | None = None,
) -> lgb.Booster:
    """Trainiert einen LightGBM-Booster.

    Parameters
    ----------
    X, y : Feature-Matrix und Target-Vektor.
    objective : LightGBM-Objective, z.B. 'regression_l1' oder 'quantile'.
    alpha : Quantile bei 'quantile' (z.B. 0.9 fuer P90).
    num_boost_round : Anzahl Boosting-Runden.
    weight : Optional Sample-Weights, gleiche Laenge wie y.
    params_override : Optional dict zum Ueberschreiben einzelner LightGBM-
        Hyperparameter. Wird ueber DEFAULT_LGBM_PARAMS gemerged. baseline.py
        uebergibt None -> reine Defaults, productive.py kann damit eine
        staerker regularisierte Konfiguration anfordern.
    """
    params = {**DEFAULT_LGBM_PARAMS, "objective": objective}
    if alpha is not None:
        params["alpha"] = alpha
    if params_override:
        params.update(params_override)
    dataset = lgb.Dataset(X, label=y, weight=weight)
    return lgb.train(params, dataset, num_boost_round=num_boost_round)


# ---------------------------------------------------------------------------
# 4) Gemeinsame Metriken (werden von eval.py importiert)
# ---------------------------------------------------------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in kW."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def pinball(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.9) -> float:
    """Pinball-Loss (asymmetrische Quantile-Loss) fuer Quantile-Predictions."""
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


# ---------------------------------------------------------------------------
# 5) End-to-End-Pipeline (Baseline-Version: ohne Peak-Weighting)
# ---------------------------------------------------------------------------
def run_training_and_submission(
    *,
    weight_fn=None,
    out_path: Path = BASELINE_OUT,
    label: str = "baseline",
    extra_features_df: pd.DataFrame | None = None,
    features: list[str] | None = None,
    hourly_df: pd.DataFrame | None = None,
    lgbm_params_override: dict | None = None,
    num_boost_round: int = 500,
) -> pd.DataFrame:
    """Trainiert Point- und P90-Modell und schreibt eine Submission.

    Shared-Pipeline-Funktion fuer baseline.py (ohne Extras) und productive.py
    (mit Peak-Weighting, Zusatz-Features wie Wetter, Container-Mix, ...).

    Parameters
    ----------
    weight_fn : Optional callable (y: pd.Series) -> np.ndarray.
        Bekommt die Trainings-Targets und gibt das Sample-Weight-Array zurueck.
        None -> kein Weighting (baseline).
    out_path : Wo die Submission-CSV landet.
    label : Nur fuer Log-Ausgaben.
    extra_features_df : Optional DataFrame mit Spalte 'ts' plus beliebig vielen
        weiteren Spalten (z.B. Feiertags-Features). Wird per left-join auf
        `ts` in den Feature-DataFrame gemerged, bevor die Lag-NaNs gedroppt
        werden. None -> keine Zusatz-Features (baseline).
    features : Optional Feature-Liste, die fuer Training/Prediction verwendet
        wird. None -> die minimalen Baseline-FEATURES. Wenn extra_features_df
        uebergeben wird, muss `features` auch die neuen Spaltennamen enthalten.
    hourly_df : Optional vorbereitetes Stunden-Aggregat mit 'ts' und
        mindestens 'power_kw'. Wenn angegeben, wird NICHT noch einmal der
        grosse reefer_release.csv gelesen - spart ~30s pro Lauf. productive.py
        nutzt das, um Container-Mix-Features in einem einzigen CSV-Read zu
        extrahieren.
    """
    feat_list = features if features is not None else FEATURES

    # Target-Fenster VORZIEHEN: wir brauchen target_end fuer die Mirror-Year-
    # Extension, bevor wir Lag-Features bauen. Das Target wird spaeter unten
    # nochmal gelesen (Kosten: ein paar Millisekunden, vernachlaessigbar).
    _targets_probe = pd.read_csv(TARGET_CSV)
    _targets_probe["ts"] = pd.to_datetime(_targets_probe["timestamp_utc"], utc=True)
    _target_end_probe = _targets_probe["ts"].max()

    if hourly_df is None:
        hourly = load_hourly_total(REEFER_CSV)
        # Reefer-Rohdaten sind jetzt hart bei HARD_CUTOFF_TS abgeschnitten.
        # Erweitere die Reihe per Mirror-Year um die Stunden im Target-
        # Fenster, damit add_features spaeter lag_24h/lag_168h leakage-frei
        # berechnen kann.
        hourly = extend_post_cutoff_with_mirror(
            hourly, post_range_end=_target_end_probe
        )
    else:
        hourly = hourly_df.copy()
        print(
            f"[{label}] Verwende vorgegebenes hourly_df "
            f"({len(hourly)} Zeilen, Spalten: {list(hourly.columns)})"
        )

    # add_features nur aufrufen, wenn die Baseline-Features noch nicht im
    # hourly-DataFrame sind. productive.py kann `add_features` vorher selbst
    # aufrufen, synthetisierte Lag-Werte einfuegen und dann den fertigen
    # DataFrame uebergeben. In dem Fall wuerden wir hier sonst die Synthese
    # ueberschreiben.
    baseline_feat_cols = set(FEATURES)
    if baseline_feat_cols.issubset(hourly.columns):
        print(
            f"[{label}] hourly_df ist bereits featurized "
            f"({sorted(baseline_feat_cols & set(hourly.columns))}), "
            f"skip add_features"
        )
        feat = hourly
    else:
        feat = add_features(hourly)
    print(
        f"[{label}] Zeitbereich: {hourly['ts'].min()} -> {hourly['ts'].max()}, "
        f"{len(hourly)} Stunden"
    )
    print(
        f"[{label}] power_kw: min={hourly[TARGET_COL].min():.1f}, "
        f"max={hourly[TARGET_COL].max():.1f}, "
        f"mean={hourly[TARGET_COL].mean():.1f}"
    )

    if extra_features_df is not None:
        before = len(feat)
        feat = feat.merge(extra_features_df, on="ts", how="left")
        added = [c for c in extra_features_df.columns if c != "ts"]
        print(
            f"[{label}] Extra-Features gejoint: {added} "
            f"(Zeilen vorher/nachher: {before}/{len(feat)})"
        )

    feat = feat.dropna(subset=feat_list).reset_index(drop=True)

    targets = pd.read_csv(TARGET_CSV)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)
    target_start = targets["ts"].min()
    target_end = targets["ts"].max()
    print(
        f"[{label}] Target-Fenster: {target_start} -> {target_end} "
        f"({len(targets)} Stunden)"
    )

    # Training strikt VOR dem Target-Fenster, damit das Modell nie das
    # Target-Ground-Truth zu Gesicht bekommt.
    train_df = feat.loc[feat["ts"] < target_start].copy()
    print(f"[{label}] {len(train_df)} Trainings-Zeilen, features={feat_list}")

    # weight_fn bekommt den kompletten train_df (incl. 'ts' und feat_list).
    # Damit kann productive.py zeitbezogen gewichten (z.B. Monats-Gewichtung,
    # um Januar-Zeilen im Training staerker zu betonen, weil wir nur auf Jan
    # getestet werden).
    weight = weight_fn(train_df) if weight_fn is not None else None

    if lgbm_params_override:
        print(
            f"[{label}] LightGBM Params Override: {lgbm_params_override} "
            f"(num_boost_round={num_boost_round})"
        )

    print(f"[{label}] Trainiere Point-Modell (regression_l1) ...")
    m_point = train_lgbm(
        train_df[feat_list],
        train_df[TARGET_COL],
        objective="regression_l1",
        weight=weight,
        num_boost_round=num_boost_round,
        params_override=lgbm_params_override,
    )
    print(f"[{label}] Trainiere P90-Modell (quantile alpha=0.9) ...")
    m_p90 = train_lgbm(
        train_df[feat_list],
        train_df[TARGET_COL],
        objective="quantile",
        alpha=0.9,
        num_boost_round=num_boost_round,
        params_override=lgbm_params_override,
        weight=weight,
    )

    # Feature-Zeilen fuer die Target-Timestamps aus dem vollstaendigen feat-Set
    # holen. Lag-Werte beziehen sich auf tatsaechliche historische power_kw.
    target_feat = targets[["ts"]].merge(feat, on="ts", how="left")
    missing = int(target_feat[feat_list].isna().any(axis=1).sum())
    if missing:
        print(f"[{label}] WARN: {missing} Target-Stunden ohne Feature-Match, fuelle auf")
        target_feat[feat_list] = (
            target_feat[feat_list].ffill().bfill().fillna(0.0)
        )

    pred_point = m_point.predict(target_feat[feat_list])
    pred_p90 = m_p90.predict(target_feat[feat_list])
    # Harte Submission-Regel: pred_p90_kw >= pred_power_kw
    pred_p90 = np.maximum(pred_p90, pred_point)
    # Keine negativen Werte
    pred_point = np.clip(pred_point, a_min=0.0, a_max=None)
    pred_p90 = np.clip(pred_p90, a_min=0.0, a_max=None)

    submission = pd.DataFrame(
        {
            "timestamp_utc": target_feat["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pred_power_kw": np.round(pred_point, 2),
            "pred_p90_kw": np.round(pred_p90, 2),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # float_format erzwingt exakt 2 Nachkommastellen fuer alle numerischen
    # Spalten. Das matcht den Template-Stil (1234.56, 1360.00) und verhindert,
    # dass pandas Trailing-Zeros droppt (989.1 statt 989.10). Wenn der Scorer
    # strikt String-basiert vergleicht, kann das einen Unterschied machen.
    submission.to_csv(out_path, index=False, float_format="%.2f")
    print(
        f"[{label}] Submission geschrieben: "
        f"{out_path.relative_to(PROJECT_ROOT)} ({len(submission)} Zeilen)"
    )
    return submission


def main() -> None:
    """Baseline-Lauf ohne Peak-Weighting."""
    run_training_and_submission(weight_fn=None, out_path=BASELINE_OUT, label="baseline")


if __name__ == "__main__":
    main()
