"""Blend-Gewichte via Walk-Forward-Cross-Validation tunen.

Ziel: Statt fixer Uniform-Gewichte fuer den Blend der 3 RF-Modelle
(legal_rf_big_s1, legal_rf_s1, rf_richfeat) lernen wir **legitime**
datengetriebene Gewichte auf einem Pre-Target-Holdout.

Warum das KEIN GT-Sniffing ist:
    Der Holdout liegt VOR dem offiziellen Target-Fenster (2026-01-01+).
    Konkret: 2025-12-15 bis 2025-12-31 23:00 UTC. Diese Stunden sind
    Teil des legalen Trainings-Bereichs und ihre echten power_kw-Werte
    sind "fair game". Wir nutzen sie, um die Kombinationsgewichte zu
    fitten - nicht, um Modell-Parameter oder Features zu lernen.
    Die resultierenden Gewichte werden auf das echte Target-Fenster
    uebertragen, OHNE dass jemals Januar-2026-Ground-Truth gesehen wurde.

Ablauf:
    1. Schreibe target_timestamps_dec_holdout.csv (Dec 15-31 2025).
    2. Setze Env-Vars EUROGATE_TARGET_CSV und EUROGATE_OUTPUT_SUFFIX="_dec"
       und rufe productive.py + rf_richfeat.py via subprocess auf. Diese
       Scripts cutten Training automatisch bei target_start=2025-12-15
       -> ehrliches out-of-sample, keine Dezember-Zeilen im Training.
    3. Lade die 3 erzeugten *_dec.csv Files und die echten Dec-15-31 GT.
    4. Finde via scipy.optimize.minimize(method='SLSQP') die optimalen
       Gewichte (w_big, w_s1, w_rich) mit sum=1 und w >= 0, die
       combined_score minimieren.
    5. Schreibe lightgbm/blend_weights.json. honest_blend.py liest das
       optional und nutzt eine "custom_weighted" Strategie, falls der
       File existiert (Fallback: uniform_3_rf).

Ausfuehren (vom Projekt-Root):
    uv run python lightgbm/tune_blend.py

Laufzeit: ~7-10 min (2x productive.py + 1x rf_richfeat.py im Dec-Modus).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from baseline import (  # noqa: E402
    DATEN_DIR,
    PEAK_QUANTILE,
    PROJECT_ROOT,
    REEFER_CSV,
    SUBMISSIONS_DIR,
    TARGET_COL,
    load_hourly_total,
)


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
DEC_HOLDOUT_START = pd.Timestamp("2025-12-15 00:00:00", tz="UTC")
DEC_HOLDOUT_END = pd.Timestamp("2025-12-31 23:00:00", tz="UTC")

DEC_TARGET_CSV = DATEN_DIR / "target_timestamps_dec_holdout.csv"
DEC_OUTPUT_SUFFIX = "_dec"

# Die 3 RF-Modelle, deren Gewichte wir lernen. Reihenfolge ist wichtig -
# sie wird 1:1 in blend_weights.json und honest_blend.py gespiegelt.
TUNE_MODELS: list[str] = [
    "legal_rf_big_s1",
    "legal_rf_s1",
    "rf_richfeat",
]

# Quelle fuer p90: wie in honest_blend.py a priori fest (rf_richfeat).
P90_MODEL: str = "rf_richfeat"

BLEND_WEIGHTS_JSON = _HERE / "blend_weights.json"


# ---------------------------------------------------------------------------
# 1) Dec-Holdout target_timestamps schreiben
# ---------------------------------------------------------------------------
def ensure_dec_target_csv() -> Path:
    """Schreibt DEC_TARGET_CSV (nur wenn noch nicht vorhanden)."""
    if DEC_TARGET_CSV.exists():
        print(f"[dec] Existiert: {DEC_TARGET_CSV.relative_to(PROJECT_ROOT)}")
        return DEC_TARGET_CSV

    hours = pd.date_range(DEC_HOLDOUT_START, DEC_HOLDOUT_END, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"timestamp_utc": hours.strftime("%Y-%m-%dT%H:%M:%SZ")}
    )
    DEC_TARGET_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEC_TARGET_CSV, index=False)
    print(
        f"[dec] Geschrieben: {DEC_TARGET_CSV.relative_to(PROJECT_ROOT)} "
        f"({len(df)} Stunden, {DEC_HOLDOUT_START} -> {DEC_HOLDOUT_END})"
    )
    return DEC_TARGET_CSV


# ---------------------------------------------------------------------------
# 2) Base-Scripts im Dec-Modus aufrufen
# ---------------------------------------------------------------------------
def run_base_scripts_in_dec_mode(target_csv: Path) -> None:
    """Ruft productive.py und rf_richfeat.py mit alternativem Target auf."""
    env = os.environ.copy()
    env["EUROGATE_TARGET_CSV"] = str(target_csv)
    env["EUROGATE_OUTPUT_SUFFIX"] = DEC_OUTPUT_SUFFIX

    scripts = [
        ("productive", _HERE / "productive.py"),
        ("rf_richfeat", _HERE / "rf_richfeat.py"),
    ]
    for label, script in scripts:
        # Check ob alle erwarteten Outputs bereits existieren -> skip
        expected = _expected_outputs_for(label)
        if all((SUBMISSIONS_DIR / name).exists() for name in expected):
            print(
                f"[run] Skip {label}: {expected} existieren bereits "
                f"(loesche *_dec.csv, um neu zu tunen)"
            )
            continue

        print()
        print("=" * 72)
        print(f"[run] Starte {label} im Dec-Modus: {script.name}")
        print("=" * 72, flush=True)
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{label} ist mit Exit-Code {result.returncode} abgebrochen."
            )


def _expected_outputs_for(label: str) -> list[str]:
    if label == "productive":
        return [
            f"legal_rf_big_s1{DEC_OUTPUT_SUFFIX}.csv",
            f"legal_rf_s1{DEC_OUTPUT_SUFFIX}.csv",
        ]
    if label == "rf_richfeat":
        return [f"rf_richfeat{DEC_OUTPUT_SUFFIX}.csv"]
    raise ValueError(label)


# ---------------------------------------------------------------------------
# 3) GT + Dec-Submissions laden
# ---------------------------------------------------------------------------
def load_dec_gt(target_csv: Path) -> pd.DataFrame:
    """Baut (ts, y_true) fuer das Dec-Holdout aus reefer_release.csv."""
    targets = pd.read_csv(target_csv)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)

    # HARD_CUTOFF_TS ist default=2099 (deaktiviert). Selbst wenn es gesetzt
    # waere, liegt das Dec-Holdout vor dem Cut - dann klappt der Default.
    hourly = load_hourly_total(REEFER_CSV)
    gt = targets[["ts"]].merge(
        hourly.rename(columns={TARGET_COL: "y_true"}), on="ts", how="left"
    )
    missing = int(gt["y_true"].isna().sum())
    if missing:
        raise RuntimeError(
            f"{missing} Dec-Holdout-Stunden ohne Ground-Truth - "
            f"reefer_release.csv deckt den Zeitraum nicht ab."
        )
    return gt


def load_dec_submission(model_name: str, gt_ts: pd.Series) -> dict[str, np.ndarray]:
    """Laedt ein *_dec.csv und richtet es an gt_ts aus."""
    path = SUBMISSIONS_DIR / f"{model_name}{DEC_OUTPUT_SUFFIX}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Erwartetes Tuning-File fehlt: {path}")
    sub = pd.read_csv(path)
    sub["ts"] = pd.to_datetime(sub["timestamp_utc"], utc=True)
    merged = pd.DataFrame({"ts": gt_ts}).merge(sub, on="ts", how="left")
    if merged[["pred_power_kw", "pred_p90_kw"]].isna().any().any():
        raise RuntimeError(f"{path.name}: Zeilen fehlen gegenueber GT")
    return {
        "point": merged["pred_power_kw"].to_numpy(),
        "p90": merged["pred_p90_kw"].to_numpy(),
    }


# ---------------------------------------------------------------------------
# 4) Metriken (Kopie, um Zyklus mit eval.py zu vermeiden)
# ---------------------------------------------------------------------------
def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mae_peak(y_true, y_pred) -> float:
    threshold = np.quantile(y_true, PEAK_QUANTILE)
    mask = y_true >= threshold
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def pinball(y_true, y_pred, alpha: float = 0.9) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def combined(y_true, y_point, y_p90) -> float:
    return (
        0.5 * mae(y_true, y_point)
        + 0.3 * mae_peak(y_true, y_point)
        + 0.2 * pinball(y_true, y_p90)
    )


# ---------------------------------------------------------------------------
# 5) Optimierung
# ---------------------------------------------------------------------------
def optimize_weights(
    y_true: np.ndarray,
    points: list[np.ndarray],
    p90_fixed: np.ndarray,
) -> tuple[list[float], float]:
    """Findet w >= 0, sum(w) = 1, die combined(y_true, w@points, p90) min."""

    def objective(w: np.ndarray) -> float:
        y_point = np.sum([w[i] * points[i] for i in range(len(points))], axis=0)
        y_p90_aligned = np.maximum(p90_fixed, y_point)
        return combined(y_true, y_point, y_p90_aligned)

    n = len(points)
    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )
    if not result.success:
        print(f"[opt] WARN: SLSQP hat nicht konvergiert: {result.message}")
    weights = result.x.tolist()
    return weights, float(result.fun)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[tune] Dec-Holdout Walk-Forward-CV fuer Blend-Gewichte")
    print(f"[tune] Projekt-Root: {PROJECT_ROOT}")

    # 1) Dec-Target
    dec_csv = ensure_dec_target_csv()

    # 2) Base-Scripts mit EUROGATE_TARGET_CSV aufrufen
    run_base_scripts_in_dec_mode(dec_csv)

    # 3) GT + Submissions laden
    print("\n[tune] Lade Dec-Holdout Ground Truth aus reefer_release.csv ...")
    gt = load_dec_gt(dec_csv)
    y_true = gt["y_true"].to_numpy()
    print(
        f"[tune] GT: {len(gt)} Stunden, mean={y_true.mean():.1f} kW, "
        f"max={y_true.max():.1f} kW"
    )

    preds = {name: load_dec_submission(name, gt["ts"]) for name in TUNE_MODELS}
    points = [preds[name]["point"] for name in TUNE_MODELS]
    p90_fixed = preds[P90_MODEL]["p90"]

    # 4) Einzel-Scores zur Orientierung
    print("\n[tune] Einzel-Scores auf Dec-Holdout (mit eigenem p90):")
    for name in TUNE_MODELS:
        p = preds[name]["point"]
        own_p90 = preds[name]["p90"]
        cb = combined(y_true, p, np.maximum(own_p90, p))
        print(
            f"  {name:<22} combined={cb:7.2f}  "
            f"mae_all={mae(y_true, p):6.2f}  "
            f"mae_peak={mae_peak(y_true, p):6.2f}  "
            f"pinball={pinball(y_true, own_p90):6.2f}"
        )

    # Uniform-Referenz
    uniform_point = np.mean(points, axis=0)
    uniform_p90 = np.maximum(p90_fixed, uniform_point)
    uniform_score = combined(y_true, uniform_point, uniform_p90)
    print(
        f"\n[tune] Uniform-3 Referenz:  combined={uniform_score:7.2f}  "
        f"mae_all={mae(y_true, uniform_point):6.2f}  "
        f"mae_peak={mae_peak(y_true, uniform_point):6.2f}  "
        f"pinball={pinball(y_true, uniform_p90):6.2f}"
    )

    # 5) SLSQP-Optimierung
    print("\n[tune] Optimiere Gewichte (SLSQP, sum=1, w>=0) ...")
    weights, opt_score = optimize_weights(y_true, points, p90_fixed)

    print("\n[tune] Optimale Gewichte:")
    for name, w in zip(TUNE_MODELS, weights):
        print(f"  {name:<22}  w = {w:.4f}")
    print(f"  {'TOTAL':<22}  w = {sum(weights):.4f}")

    # Optimiertes Ergebnis auf Dec
    opt_point = np.sum([weights[i] * points[i] for i in range(len(points))], axis=0)
    opt_p90 = np.maximum(p90_fixed, opt_point)
    print(
        f"\n[tune] Optimierte Blend-Score:  combined={opt_score:7.2f}  "
        f"(Uniform: {uniform_score:.2f}, Delta: {opt_score - uniform_score:+.2f})"
    )
    print(
        f"  mae_all={mae(y_true, opt_point):6.2f}  "
        f"mae_peak={mae_peak(y_true, opt_point):6.2f}  "
        f"pinball={pinball(y_true, opt_p90):6.2f}"
    )

    # 6) Schreibe blend_weights.json
    payload = {
        "tuned_on": "dec_2025_12_15_to_2025_12_31",
        "holdout_hours": int(len(gt)),
        "models": TUNE_MODELS,
        "p90_source": P90_MODEL,
        "weights": weights,
        "dec_uniform_score": uniform_score,
        "dec_tuned_score": opt_score,
    }
    BLEND_WEIGHTS_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\n[tune] -> {BLEND_WEIGHTS_JSON.relative_to(PROJECT_ROOT)}")
    print(
        "[tune] honest_blend.py liest diese Datei automatisch und nutzt sie "
        "als 'custom_weighted' Strategie."
    )


if __name__ == "__main__":
    main()
