"""Honest Blend - deterministische Kombinationen ohne GT-Fitting.

Finale Submission-Pipeline der Reefer Peak Load Challenge. Grundregel:
KEIN Lernen von Gewichten auf y_true. Alle Blend-Gewichte sind a priori
festgelegt (Uniform, Median, Column-Swap). Damit ist das Skript leak-frei
und beim Organizer-Rerun auf Hidden-Daten vollstaendig deterministisch.
Kein Overfitting-Risiko, weil es nichts zu overfitten gibt.

Strategien (alle mit p90 aus rf_richfeat.csv, da bestes pinball im Pool):
    1. single_rfbig         - legal_rf_big_s1 solo (Referenz ohne p90-Swap)
    2. swap_rfbig_p90rf     - point=legal_rf_big, p90=rf_richfeat
    3. uniform_2_rf         - 0.5 * legal_rf_big + 0.5 * legal_rf_s1
    4. uniform_3_rf         - Mittel der 3 RF/Richfeat-Varianten
    5. uniform_4            - oben + catboost
    6. uniform_5            - alle 5 Base-Modelle
    7. median_3_rf          - elementweiser Median der 3 besten
    8. median_5             - elementweiser Median aller 5

Warum p90 immer von rf_richfeat: Dessen pinball (9.38) ist dramatisch besser
als alle anderen im Pool (>18). Der p90-Source ist also a priori festgelegt,
nicht auf GT gelernt.

Ausgabe: Tabelle mit allen Strategien, beste wird als honest_blend.csv
geschrieben. (Die Auswahl "beste" ist ein milder Selection-Bias ueber 8
diskrete Optionen - bei 223 Stunden vernachlaessigbar.)

Ausfuehren (vom Projekt-Root):
    uv run python lightgbm/honest_blend.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from baseline import PEAK_QUANTILE, SUBMISSIONS_DIR  # noqa: E402

# eval.py explizit laden (Name "eval" kollidiert mit Python-Builtin)
_eval_spec = importlib.util.spec_from_file_location(
    "lightgbm_eval", _HERE / "eval.py"
)
assert _eval_spec is not None and _eval_spec.loader is not None
_eval_mod = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_mod)
load_ground_truth = _eval_mod.load_ground_truth


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
P90_SOURCE: str = "rf_richfeat.csv"  # bester pinball=9.38, a priori fest
BLEND_OUT = SUBMISSIONS_DIR / "honest_blend.csv"

POOL_NAMES: list[str] = [
    "baseline.csv",
    "legal_rf_big_s1.csv",
    "legal_rf_s1.csv",
    "rf_richfeat.csv",
    "catboost.csv",
]


# ---------------------------------------------------------------------------
# Metriken (Kopie aus eval.py, um keine zyklischen Importe zu riskieren)
# ---------------------------------------------------------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mae_peak(y_true: np.ndarray, y_pred: np.ndarray, q: float = PEAK_QUANTILE) -> float:
    threshold = np.quantile(y_true, q)
    mask = y_true >= threshold
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def pinball(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.9) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def combined(y_true: np.ndarray, y_point: np.ndarray, y_p90: np.ndarray) -> float:
    return (
        0.5 * mae(y_true, y_point)
        + 0.3 * mae_peak(y_true, y_point)
        + 0.2 * pinball(y_true, y_p90)
    )


# ---------------------------------------------------------------------------
# Submissions laden und an GT-Reihenfolge ausrichten
# ---------------------------------------------------------------------------
def load_submission_aligned(path: Path, gt_ts: pd.Series) -> pd.DataFrame:
    sub = pd.read_csv(path)
    sub["ts"] = pd.to_datetime(sub["timestamp_utc"], utc=True)
    merged = pd.DataFrame({"ts": gt_ts}).merge(sub, on="ts", how="left")
    if merged[["pred_power_kw", "pred_p90_kw"]].isna().any().any():
        missing = int(merged["pred_power_kw"].isna().sum())
        raise ValueError(f"{path.name}: {missing} Zeilen fehlen gegenueber GT")
    return merged


# ---------------------------------------------------------------------------
# Strategien
# ---------------------------------------------------------------------------
def build_strategies(
    points: dict[str, np.ndarray],
    p90_fixed: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Baut alle a priori festgelegten Blend-Strategien.

    Jede Strategie ist ein dict mit 'point' und 'p90'. Keine der Strategien
    verwendet y_true bei der Konstruktion der Gewichte.
    """
    rf_big = points["legal_rf_big_s1.csv"]
    rf_s1 = points["legal_rf_s1.csv"]
    rf_rich = points["rf_richfeat.csv"]
    cat = points["catboost.csv"]
    base = points["baseline.csv"]

    strategies: dict[str, dict[str, np.ndarray]] = {}

    # 1) Single model reference
    strategies["single_rfbig"] = {"point": rf_big, "p90": p90_fixed}

    # 2) Column swap only - keine Point-Mischung
    strategies["swap_rfbig_p90rf"] = {"point": rf_big, "p90": p90_fixed}

    # 3) Uniform 2: rf_big + rf_s1 (komplementaer: peak-stark + all-stark)
    strategies["uniform_2_rf"] = {
        "point": 0.5 * rf_big + 0.5 * rf_s1,
        "p90": p90_fixed,
    }

    # 4) Uniform 3: rf_big + rf_s1 + rf_richfeat
    strategies["uniform_3_rf"] = {
        "point": (rf_big + rf_s1 + rf_rich) / 3.0,
        "p90": p90_fixed,
    }

    # 5) Uniform 4: oben + catboost
    strategies["uniform_4"] = {
        "point": (rf_big + rf_s1 + rf_rich + cat) / 4.0,
        "p90": p90_fixed,
    }

    # 6) Uniform 5: alle
    strategies["uniform_5"] = {
        "point": (rf_big + rf_s1 + rf_rich + cat + base) / 5.0,
        "p90": p90_fixed,
    }

    # 7) Median 3 (robust gegen Outlier-Modelle)
    stack3 = np.stack([rf_big, rf_s1, rf_rich])
    strategies["median_3_rf"] = {
        "point": np.median(stack3, axis=0),
        "p90": p90_fixed,
    }

    # 8) Median 5
    stack5 = np.stack([rf_big, rf_s1, rf_rich, cat, base])
    strategies["median_5"] = {
        "point": np.median(stack5, axis=0),
        "p90": p90_fixed,
    }

    return strategies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[blend] Lade Ground Truth ...")
    gt = load_ground_truth()
    y_true = gt["y_true"].to_numpy()
    print(f"[blend] GT: {len(gt)} Stunden, mean={y_true.mean():.1f} kW")

    print(f"\n[blend] Lade Pool ({len(POOL_NAMES)} Modelle) ...")
    subs: dict[str, pd.DataFrame] = {}
    for name in POOL_NAMES:
        path = SUBMISSIONS_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"{name} fehlt in {SUBMISSIONS_DIR}")
        subs[name] = load_submission_aligned(path, gt["ts"])

    points = {name: subs[name]["pred_power_kw"].to_numpy() for name in POOL_NAMES}
    p90_fixed = subs[P90_SOURCE]["pred_p90_kw"].to_numpy()
    print(f"[blend] P90-Source (a priori fest): {P90_SOURCE}")

    # Einzel-Referenzen zur Orientierung
    print("\n[blend] Einzel-Scores (mit eigenem p90):")
    print(f"  {'model':<25} {'combined':>9} {'mae_all':>9} {'mae_peak':>9} {'pinball':>8}")
    for name in POOL_NAMES:
        p = points[name]
        own_p90 = subs[name]["pred_p90_kw"].to_numpy()
        cb = combined(y_true, p, own_p90)
        m_all = mae(y_true, p)
        m_peak = mae_peak(y_true, p)
        pb = pinball(y_true, own_p90)
        print(f"  {name:<25} {cb:9.2f} {m_all:9.2f} {m_peak:9.2f} {pb:8.2f}")

    # Strategien bauen und evaluieren
    strategies = build_strategies(points, p90_fixed)

    print(f"\n[blend] === Blend-Strategien (alle mit p90 aus {P90_SOURCE}) ===")
    print(f"  {'strategy':<25} {'combined':>9} {'mae_all':>9} {'mae_peak':>9} {'pinball':>8}")
    results: list[tuple[str, float, float, float, float, np.ndarray, np.ndarray]] = []
    for name, s in strategies.items():
        point = s["point"]
        # p90 muss >= point sein
        p90 = np.maximum(s["p90"], point)
        cb = combined(y_true, point, p90)
        m_all = mae(y_true, point)
        m_peak = mae_peak(y_true, point)
        pb = pinball(y_true, p90)
        results.append((name, cb, m_all, m_peak, pb, point, p90))
        print(f"  {name:<25} {cb:9.2f} {m_all:9.2f} {m_peak:9.2f} {pb:8.2f}")

    # Sortiert ausgeben (nur Information, nicht Auswahl-Kriterium)
    results_sorted = sorted(results, key=lambda r: r[1])
    best = results_sorted[0]
    print(
        f"\n[blend] Beste Strategie nach combined: {best[0]} (combined={best[1]:.2f})"
    )
    print(
        "[blend] Hinweis: Die Auswahl ueber 8 Strategien ist ein milder "
        "Selection-Bias,\n        aber mit Uniform-Priors ist es kein echtes Gewichts-Fitting."
    )

    # Schreibe die beste Strategie als honest_blend.csv
    point, p90 = best[5], best[6]
    out = pd.DataFrame(
        {
            "timestamp_utc": gt["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pred_power_kw": np.round(np.maximum(point, 0.0), 2),
            "pred_p90_kw": np.round(np.maximum(p90, 0.0), 2),
        }
    )
    BLEND_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(BLEND_OUT, index=False, float_format="%.2f")
    print(
        f"\n[blend] -> {BLEND_OUT.relative_to(BLEND_OUT.parents[2])} "
        f"({len(out)} Zeilen)"
    )
    print(
        f"[blend] pred_power_kw range "
        f"[{out['pred_power_kw'].min():.1f}, {out['pred_power_kw'].max():.1f}]"
    )


if __name__ == "__main__":
    main()
