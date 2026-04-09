"""Submission-Scorer fuer die Reefer Peak Load Challenge.

Dieses Skript ist kein Trainings-Tool, sondern ein reiner BEWERTER. Es:

1. Baut die Ground Truth aus `reefer_release.csv` fuer das oeffentliche
   Target-Fenster (`target_timestamps.csv`), indem es die Container-Level-
   Daten pro Stunde aufsummiert und in kW umrechnet.

2. Scannt den Ordner `submissions/` nach allen *.csv Dateien.

3. Bewertet jede Submission gegen die Ground Truth mit den drei offiziellen
   Metriken der Challenge plus dem Combined Score:

       Combined = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90

4. Zeigt eine Vergleichstabelle, sortiert nach Combined Score (bester zuerst).

"Peak-Stunden" sind als oberste 15% der wahren Werte im Target-Fenster
definiert (konsistent mit dem Wert, mit dem productive.py traininert).

Ausfuehren:
    uv run python lightgbm/eval.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baseline import (
    PEAK_QUANTILE,
    PROJECT_ROOT,
    REEFER_CSV,
    SUBMISSIONS_DIR,
    TARGET_COL,
    TARGET_CSV,
    load_hourly_total,
    mae,
    pinball,
)


# ---------------------------------------------------------------------------
# Zusaetzliche Metriken (spezifisch fuer das Scoring)
# ---------------------------------------------------------------------------
def mae_peak(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    peak_quantile: float = PEAK_QUANTILE,
) -> float:
    """MAE nur auf den Stunden mit den hoechsten tatsaechlichen Werten.

    "Peak" ist hier: oberste (1 - peak_quantile) der wahren Verbrauchswerte.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    threshold = np.quantile(y_true, peak_quantile)
    mask = y_true >= threshold
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def combined_score(mae_all_v: float, mae_peak_v: float, pinball_v: float) -> float:
    """Offizielle Scoring-Formel: 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90."""
    return 0.5 * mae_all_v + 0.3 * mae_peak_v + 0.2 * pinball_v


# ---------------------------------------------------------------------------
# Ground Truth aus reefer_release.csv + target_timestamps.csv bauen
# ---------------------------------------------------------------------------
def load_ground_truth() -> pd.DataFrame:
    """Baut den (ts, power_kw)-Ground-Truth fuer das oeffentliche Target-Fenster.

    Rueckgabe: DataFrame mit Spalten ['ts', 'y_true'], eine Zeile pro
    Target-Timestamp, in der exakten Reihenfolge der target_timestamps.csv.

    Sonderrolle gegenueber allen anderen Scripts: eval.py ist der EINZIGE
    Ort, an dem reefer_release.csv ohne HARD_CUTOFF_TS gelesen werden darf.
    Das ist der Scoring-Schritt und braucht naturgemaess die wahren
    power_kw-Werte auch nach 2025-12-31. Alle Trainings-/Feature-Scripts
    ruefen den Default-Cutoff an und sind damit leakage-frei.
    """
    # cutoff=pd.NaT wuerde ignoriert, also explizit einen Datumswert in der
    # Zukunft uebergeben, der garantiert alle echten Daten einschliesst.
    hourly = load_hourly_total(
        REEFER_CSV, cutoff=pd.Timestamp("2099-12-31 23:00:00", tz="UTC")
    )

    targets = pd.read_csv(TARGET_CSV)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)

    gt = targets[["ts"]].merge(
        hourly.rename(columns={TARGET_COL: "y_true"}), on="ts", how="left"
    )
    missing = int(gt["y_true"].isna().sum())
    if missing:
        print(
            f"[gt] WARN: {missing} Target-Stunden ohne Ground-Truth. "
            f"Werden aus der Bewertung ausgeschlossen."
        )
        gt = gt.dropna(subset=["y_true"]).reset_index(drop=True)
    return gt


# ---------------------------------------------------------------------------
# Eine einzelne Submission bewerten
# ---------------------------------------------------------------------------
def score_submission(
    submission_path: Path, gt: pd.DataFrame
) -> dict[str, float | int | str]:
    """Laedt eine Submission-CSV und bewertet sie gegen die Ground Truth."""
    sub = pd.read_csv(submission_path)
    sub["ts"] = pd.to_datetime(sub["timestamp_utc"], utc=True)

    # Format-Sanity-Checks (die Challenge verlangt genau diese)
    if not set(["timestamp_utc", "pred_power_kw", "pred_p90_kw"]).issubset(sub.columns):
        return {"file": submission_path.name, "error": "fehlende Spalten"}
    if sub[["pred_power_kw", "pred_p90_kw"]].isna().any().any():
        return {"file": submission_path.name, "error": "NaN in Predictions"}
    if (sub["pred_power_kw"] < 0).any() or (sub["pred_p90_kw"] < 0).any():
        return {"file": submission_path.name, "error": "negative Werte"}
    if (sub["pred_p90_kw"] < sub["pred_power_kw"]).any():
        return {
            "file": submission_path.name,
            "error": "p90 < point in mind. einer Zeile",
        }

    merged = gt.merge(sub, on="ts", how="inner")
    if len(merged) != len(gt):
        return {
            "file": submission_path.name,
            "error": (
                f"{len(gt) - len(merged)} Target-Stunden fehlen in Submission"
            ),
        }

    y_true = merged["y_true"].to_numpy()
    y_point = merged["pred_power_kw"].to_numpy()
    y_p90 = merged["pred_p90_kw"].to_numpy()

    m_all = mae(y_true, y_point)
    m_peak = mae_peak(y_true, y_point)
    pb = pinball(y_true, y_p90, alpha=0.9)
    comb = combined_score(m_all, m_peak, pb)
    rel_err = m_all / float(np.mean(y_true))

    return {
        "file": submission_path.name,
        "rows": len(merged),
        "mae_all": m_all,
        "mae_peak": m_peak,
        "pinball_p90": pb,
        "combined": comb,
        "rel_err_pct": rel_err * 100.0,
    }


# ---------------------------------------------------------------------------
# Alle Submissions im submissions/-Ordner scannen und vergleichen
# ---------------------------------------------------------------------------
def main() -> None:
    if not SUBMISSIONS_DIR.exists():
        print(
            f"[eval] Ordner {SUBMISSIONS_DIR.relative_to(PROJECT_ROOT)} existiert "
            f"nicht. Zuerst baseline.py oder productive.py laufen lassen."
        )
        return

    csv_files = sorted(SUBMISSIONS_DIR.glob("*.csv"))
    if not csv_files:
        print(
            f"[eval] Keine CSVs in {SUBMISSIONS_DIR.relative_to(PROJECT_ROOT)} "
            f"gefunden. Zuerst baseline.py und/oder productive.py laufen lassen."
        )
        return

    print(f"[eval] Gefundene Submissions: {[p.name for p in csv_files]}")
    print("[eval] Baue Ground Truth aus reefer_release.csv ...")
    gt = load_ground_truth()
    print(
        f"[eval] Ground Truth: {len(gt)} Stunden, "
        f"mean={gt['y_true'].mean():.1f} kW, "
        f"max={gt['y_true'].max():.1f} kW"
    )

    peak_threshold = float(np.quantile(gt["y_true"], PEAK_QUANTILE))
    n_peak = int((gt["y_true"] >= peak_threshold).sum())
    print(
        f"[eval] Peak-Schwelle (q={PEAK_QUANTILE}): {peak_threshold:.1f} kW "
        f"({n_peak} Peak-Stunden im Target-Fenster)"
    )

    results = [score_submission(p, gt) for p in csv_files]

    # Fehlerhafte Submissions zuerst melden
    errors = [r for r in results if "error" in r]
    valid = [r for r in results if "error" not in r]
    for r in errors:
        print(f"[eval] FEHLER {r['file']}: {r['error']}")

    if not valid:
        print("[eval] Keine bewertbare Submission uebrig.")
        return

    valid.sort(key=lambda r: r["combined"])

    print()
    print("=" * 92)
    print(" Submission-Vergleich (sortiert nach Combined Score, niedriger = besser)")
    print("=" * 92)
    print(
        f" {'rank':>4}  {'file':<28}  {'mae_all':>9}  {'mae_peak':>9}  "
        f"{'pinball':>8}  {'combined':>9}  {'rel.err':>8}"
    )
    print("-" * 92)
    best = valid[0]["combined"]
    for rank, r in enumerate(valid, start=1):
        delta = r["combined"] - best
        delta_str = "   -----" if rank == 1 else f"+{delta:7.2f}"
        print(
            f" {rank:>4}  {r['file']:<28}  "
            f"{r['mae_all']:9.2f}  "
            f"{r['mae_peak']:9.2f}  "
            f"{r['pinball_p90']:8.2f}  "
            f"{r['combined']:9.2f}  "
            f"{r['rel_err_pct']:7.2f}%   {delta_str}"
        )
    print("-" * 92)
    print()
    print(" Score-Formel: 0.5*mae_all + 0.3*mae_peak + 0.2*pinball_p90")
    print(f" Peak-Definition: oberste {int((1 - PEAK_QUANTILE) * 100)}% der wahren Werte")


if __name__ == "__main__":
    main()
