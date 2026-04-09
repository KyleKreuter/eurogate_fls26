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
TARGET_CSV = DATEN_DIR / "target_timestamps.csv"
# Result-Dateien leben unterhalb des lightgbm/-Ordners, nicht im Projekt-Root.
LIGHTGBM_DIR = Path(__file__).resolve().parent
SUBMISSIONS_DIR = LIGHTGBM_DIR / "submissions"
BASELINE_OUT = SUBMISSIONS_DIR / "baseline.csv"

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
def load_hourly_total(csv_path: Path) -> pd.DataFrame:
    """Liest die Reefer-Rohdaten und gibt Stunden-Totale in kW zurueck.

    - Deutsches CSV-Format: Trennzeichen ';' und Dezimalkomma ','
    - Nur die zwei noetigen Spalten einlesen -> deutlich weniger Speicher
    - EventTime wird als UTC interpretiert (laut Doku bereits UTC)
    - Leere Stunden werden zu 0.0 kW aufgefuellt, damit `shift(24)` spaeter
      wirklich "24 Stunden zurueck" bedeutet und nicht "24 Zeilen zurueck".
    """
    print(f"[load] Lese {csv_path.name} ...")
    df = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        usecols=["EventTime", "AvPowerCons"],
    )
    df["EventTime"] = pd.to_datetime(df["EventTime"], utc=True)

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
def load_hourly_with_container_mix(csv_path: Path) -> pd.DataFrame:
    """Wie load_hourly_total, aber mit zusaetzlichen Container-Mix-Spalten.

    Liest den CSV EINMAL mit mehreren Spalten und aggregiert pro Stunde:
        - power_kw                (Summe AvPowerCons in kW)
        - num_active_containers   (Anzahl Container-Zeilen pro Stunde)
        - mean_heat_gap           (mean(TemperatureReturn - TemperatureSetPoint))
        - anteil_tiefkuehl        (Anteil Container mit SetPoint < -15 deg C)

    Die neuen Features sind physikalisch naeher am Stromverbrauch als externe
    Wetter- oder Kalender-Daten und sind unempfindlich gegen Distribution Shift,
    weil sie direkt aus den gleichen Reefer-Rohdaten kommen.
    """
    print(f"[load] Lese {csv_path.name} (mit Container-Mix) ...")
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

    if hourly_df is None:
        hourly = load_hourly_total(REEFER_CSV)
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
        print(f"[{label}] WARN: {missing} Target-Stunden ohne Feature-Match")

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
