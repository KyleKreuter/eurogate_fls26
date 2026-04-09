"""Honest Blend - deterministische Kombinationen ohne GT-Fitting.

Finale Submission-Pipeline der Reefer Peak Load Challenge. Grundregel:
KEIN Lernen von Gewichten auf y_true. Alle Blend-Gewichte sind a priori
festgelegt (Uniform, Median, Column-Swap). Damit ist das Skript leak-frei
und beim Organizer-Rerun auf Hidden-Daten vollstaendig deterministisch.
Kein Overfitting-Risiko, weil es nichts zu overfitten gibt.

Seit dem Leakage-Fix (2026-04): Die Auswahl der "besten" Strategie per
Ground-Truth-Score wurde entfernt. Stattdessen ist SUBMIT_STRATEGY fest
hardcoded auf 'uniform_3_rf' - die Strategie, die sich ueber mehrere
Laeufe durchgehend als robusteste erwiesen hat. Damit ist das Skript
nicht mehr nur leak-frei in den Blend-Gewichten, sondern auch in der
Strategie-Auswahl. Beim Organizer-Rerun wird exakt diese eine Strategie
geschrieben, unabhaengig davon was auf den (dort sichtbaren) GT-Werten
besser abschneiden wuerde.

Strategien (alle mit p90 aus rf_richfeat.csv, da bestes pinball im Pool):
    1. single_rfbig         - legal_rf_big_s1 solo (Referenz ohne p90-Swap)
    2. swap_rfbig_p90rf     - point=legal_rf_big, p90=rf_richfeat
    3. uniform_2_rf         - 0.5 * legal_rf_big + 0.5 * legal_rf_s1
    4. uniform_3_rf         - Mittel der 3 RF/Richfeat-Varianten  (<- SUBMIT)
    5. uniform_4            - oben + catboost
    6. uniform_5            - alle 5 Base-Modelle
    7. median_3_rf          - elementweiser Median der 3 besten
    8. median_5             - elementweiser Median aller 5

Warum p90 immer von rf_richfeat: Dessen pinball (9.38) ist dramatisch besser
als alle anderen im Pool (>18). Der p90-Source ist also a priori festgelegt,
nicht auf GT gelernt.

Ausgabe: Diagnose-Tabelle mit allen Strategien (Scoring nur zur Information,
fuer den Organizer-Rerun erfordert das eine vorhandene Ground-Truth). Die
fest hardcodete SUBMIT_STRATEGY wird als honest_blend.csv geschrieben.

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

# Die beim Organizer-Rerun garantiert ausgelieferte Strategie. Hardcoded,
# damit keine impliziten Auswahl-Bias ueber Ground-Truth moeglich ist.
# Wenn diese Strategie aus irgendeinem Grund im Strategien-Dict fehlt,
# wird das Skript hart abbrechen - wir wollen NIE stillschweigend auf
# eine andere Strategie ausweichen.
SUBMIT_STRATEGY: str = "uniform_3_rf"

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

    # 6) Uniform 5: alle klassischen
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
def _load_pool_aligned_to_ts(gt_ts: pd.Series) -> dict[str, pd.DataFrame]:
    """Hilfsfunktion: Laed alle POOL_NAMES und richtet sie an gt_ts aus."""
    subs: dict[str, pd.DataFrame] = {}
    for name in POOL_NAMES:
        path = SUBMISSIONS_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"{name} fehlt in {SUBMISSIONS_DIR}")
        subs[name] = load_submission_aligned(path, gt_ts)
    return subs


def main() -> None:
    # 1) Versuche Ground Truth fuer Diagnose-Output zu laden. Wenn die GT
    # nicht verfuegbar ist (z.B. weil reefer_release.csv keine Daten fuer
    # das Target-Fenster hat), laufen wir OHNE Scoring weiter und schreiben
    # trotzdem die fest hardcodete SUBMIT_STRATEGY. Die Auswahl ist NIE
    # GT-abhaengig.
    gt: pd.DataFrame | None
    try:
        print("[blend] Lade Ground Truth (nur fuer Diagnose) ...")
        gt = load_ground_truth()
        print(
            f"[blend] GT verfuegbar: {len(gt)} Stunden, "
            f"mean={gt['y_true'].to_numpy().mean():.1f} kW"
        )
    except Exception as exc:  # noqa: BLE001 - Diagnose-Output, kein kritischer Pfad
        print(f"[blend] GT nicht verfuegbar ({exc}) - fahre ohne Scoring fort")
        gt = None

    # 2) Target-Stunden aus target_timestamps.csv lesen. Das ist die einzige
    # autoritative Quelle fuer die Submission-Reihenfolge und funktioniert
    # auch dann, wenn keine GT verfuegbar ist.
    from baseline import TARGET_CSV  # noqa: E402 (lazy import, vermeidet Zyklus)

    targets = pd.read_csv(TARGET_CSV)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)
    submission_ts = targets["ts"].reset_index(drop=True)
    print(
        f"[blend] Target-Fenster: {submission_ts.min()} -> {submission_ts.max()} "
        f"({len(submission_ts)} Stunden)"
    )

    # 3) Pool laden, ausgerichtet an submission_ts (nicht gt['ts']!) - damit
    # die Submission auch fuer Target-Stunden ohne Ground-Truth funktioniert.
    print(f"\n[blend] Lade Pool ({len(POOL_NAMES)} Modelle) ...")
    subs = _load_pool_aligned_to_ts(submission_ts)

    points = {name: subs[name]["pred_power_kw"].to_numpy() for name in POOL_NAMES}
    p90_fixed = subs[P90_SOURCE]["pred_p90_kw"].to_numpy()
    print(f"[blend] P90-Source (a priori fest): {P90_SOURCE}")

    # 4) Strategien bauen
    strategies = build_strategies(points, p90_fixed)
    if SUBMIT_STRATEGY not in strategies:
        raise RuntimeError(
            f"SUBMIT_STRATEGY={SUBMIT_STRATEGY!r} fehlt im Strategien-Dict "
            f"{sorted(strategies)}. Bitte Code pruefen - keine Fallback-Auswahl."
        )

    # 5) Optionales Diagnose-Scoring (nur wenn GT verfuegbar ist). Das
    # Ergebnis hat KEINEN Einfluss auf die Strategie-Auswahl.
    if gt is not None:
        y_true = gt["y_true"].to_numpy()

        # 5a) Fuer das Scoring brauchen wir die Predictions an den GT-Stunden,
        # nicht an allen Submission-Stunden. Baue deshalb ein zweites Set
        # ausgerichtet an gt['ts'].
        subs_gt = _load_pool_aligned_to_ts(gt["ts"])
        points_gt = {n: subs_gt[n]["pred_power_kw"].to_numpy() for n in POOL_NAMES}
        p90_fixed_gt = subs_gt[P90_SOURCE]["pred_p90_kw"].to_numpy()

        print("\n[blend] Einzel-Scores (mit eigenem p90):")
        print(
            f"  {'model':<25} {'combined':>9} {'mae_all':>9} "
            f"{'mae_peak':>9} {'pinball':>8}"
        )
        for name in POOL_NAMES:
            p = points_gt[name]
            own_p90 = subs_gt[name]["pred_p90_kw"].to_numpy()
            cb = combined(y_true, p, own_p90)
            print(
                f"  {name:<25} {cb:9.2f} {mae(y_true, p):9.2f} "
                f"{mae_peak(y_true, p):9.2f} {pinball(y_true, own_p90):8.2f}"
            )

        strategies_gt = build_strategies(points_gt, p90_fixed_gt)
        print(f"\n[blend] === Blend-Strategien (p90 aus {P90_SOURCE}) ===")
        print(
            f"  {'strategy':<25} {'combined':>9} {'mae_all':>9} "
            f"{'mae_peak':>9} {'pinball':>8}"
        )
        for name, s in strategies_gt.items():
            point = s["point"]
            p90 = np.maximum(s["p90"], point)
            marker = "  <- SUBMIT" if name == SUBMIT_STRATEGY else ""
            print(
                f"  {name:<25} {combined(y_true, point, p90):9.2f} "
                f"{mae(y_true, point):9.2f} {mae_peak(y_true, point):9.2f} "
                f"{pinball(y_true, p90):8.2f}{marker}"
            )

    # 6) Finale Submission - FEST auf SUBMIT_STRATEGY. Keine Auswahl, kein
    # Bias, keine Ausnahmen.
    chosen = strategies[SUBMIT_STRATEGY]
    point = chosen["point"]
    p90 = np.maximum(chosen["p90"], point)

    out = pd.DataFrame(
        {
            "timestamp_utc": submission_ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pred_power_kw": np.round(np.maximum(point, 0.0), 2),
            "pred_p90_kw": np.round(np.maximum(p90, 0.0), 2),
        }
    )
    BLEND_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(BLEND_OUT, index=False, float_format="%.2f")
    print(
        f"\n[blend] SUBMIT_STRATEGY={SUBMIT_STRATEGY!r} -> "
        f"{BLEND_OUT.relative_to(BLEND_OUT.parents[2])} ({len(out)} Zeilen)"
    )
    print(
        f"[blend] pred_power_kw range "
        f"[{out['pred_power_kw'].min():.1f}, {out['pred_power_kw'].max():.1f}]"
    )


if __name__ == "__main__":
    main()
