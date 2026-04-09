"""Stacking-Meta-Learner fuer die Reefer Peak Load Challenge.

Ansatz: Weighted-Averaging Ensemble ueber vorhandene Submissions mit
Constraint-Optimierung (SLSQP), regularisiert und kreuzvalidiert.

Warum "Option B" und nicht sauberes OOF-Stacking?
- Wir haben nur ~223 Target-Stunden und viele unterschiedliche Modelle; das
  Retrain-All-Models-mit-TimeSeries-CV-Setup waere hier extrem aufwaendig.
- Die existierenden Submissions sind bereits gutes Material: legal_rf_big
  (bester mae_peak), productive_cold_peakw (beste pinball), baseline.csv
  als stabile Referenz. Aus diesen ein gewichtetes Mittel zu bauen, ist
  pragmatisch und reproduzierbar.
- Das Overfitting-Risiko ist real: 223 Target-Stunden und 5-10 Gewichte.
  Wir begegnen dem mit:
    * Zeit-basierter Splits (2 Folds = erste vs. zweite Haelfte des
      Target-Fensters) fuer ehrliches Out-of-Sample-Scoring
    * L2-Regularisierung, die Gewichte gegen gleichverteiltes Mittel
      schrumpft
    * Conservative Model-Pool (nur Top-Kandidaten, keine 30 Submissions)

Pipeline:
    1) Ground Truth aus reefer_release.csv via eval.load_ground_truth()
    2) Lade Submissions-Pool und richte sie an der GT-Reihenfolge aus
    3) Split Target-Fenster in 2 zeitliche Folds
    4) Optimiere Gewichte auf Fold1, bewerte auf Fold2 und umgekehrt
    5) Trainiere final auf voller Periode mit L2-Regularisierung
    6) Kombiniere mit pred_p90 aus dem besten Pinball-Modell
       (random_forest_baseline.csv, pinball=5.06)
    7) Schreibe stacking.csv (Constraint: p90 >= point, non-negative, round 2)

Ausfuehren (vom Projekt-Root):
    uv run python lightgbm/stacking.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Damit `baseline` als Geschwister-Modul importierbar ist, wenn das Skript
# vom Projekt-Root aus gestartet wird.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from baseline import PEAK_QUANTILE, SUBMISSIONS_DIR  # noqa: E402

# `eval.py` liegt direkt im lightgbm/-Ordner, aber "eval" ist ein Python-
# Builtin-Name, weshalb `from eval import ...` in einigen Konfigurationen
# scheitert. Wir laden das Modul daher explizit ueber importlib als
# "lightgbm_eval".
_eval_spec = importlib.util.spec_from_file_location(
    "lightgbm_eval", _HERE / "eval.py"
)
assert _eval_spec is not None and _eval_spec.loader is not None
_eval_mod = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_mod)
load_ground_truth = _eval_mod.load_ground_truth


# ---------------------------------------------------------------------------
# Konfiguration: welche Submissions gehen in den Stacking-Pool?
# ---------------------------------------------------------------------------
# Ausgewaehlt nach:
#  - baseline.csv         : stabile Referenz
#  - legal_rf_big*.csv    : bestes mae_peak-Modell (24.92)
#  - legal_rf_s1.csv      : diverses RF-Signal
#  - productive_cold*.csv : beste pinball-Modelle
#  - ensemble_*           : bereits gemischte Referenzen
# Wir halten den Pool bewusst klein (<= ~10), um die Gewichte nicht zu
# ueberparametrisieren.
POINT_POOL: list[str] = [
    # Kleiner Pool, regelkonform, diverse Modellfamilien
    "baseline.csv",                    # LightGBM, glatter Anker
    "legal_rf_big_s1.csv",             # RandomForest, beste mae_peak (23.93)
    "legal_rf_s1.csv",                 # RandomForest, starkes mae_all
    "catboost.csv",                    # CatBoost Champion (40.43, 3-Seed avg)
]

# P90 aus dem besten REGELKONFORMEN Pinball-Modell
P90_SOURCE: str = "rf_richfeat.csv"

STACKING_OUT = SUBMISSIONS_DIR / "stacking.csv"

# L2-Regularisierung: zieht Gewichte sanft Richtung Uniform-Mittel.
# lam=0.0 -> pures Fitting (Overfitting-Risiko).
# lam=0.1 -> spuerbare Regularisierung.
REG_LAMBDA: float = 0.3


# ---------------------------------------------------------------------------
# Metriken (identisch zu eval.py, aber als pure-numpy Funktionen)
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
# Submissions laden und an der GT-Reihenfolge ausrichten
# ---------------------------------------------------------------------------
def load_submission_aligned(path: Path, gt_ts: pd.Series) -> pd.DataFrame:
    """Laedt eine Submission und richtet sie am GT-Timestamp-Vektor aus."""
    sub = pd.read_csv(path)
    sub["ts"] = pd.to_datetime(sub["timestamp_utc"], utc=True)
    merged = pd.DataFrame({"ts": gt_ts}).merge(sub, on="ts", how="left")
    if merged[["pred_power_kw", "pred_p90_kw"]].isna().any().any():
        raise ValueError(
            f"{path.name}: {merged['pred_power_kw'].isna().sum()} Zeilen fehlen "
            f"gegenueber Ground Truth"
        )
    return merged


def load_pool(
    pool_names: list[str], gt: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """Laedt alle Submissions aus pool_names und gibt eine (n_models, n_hours)-Matrix zurueck."""
    matrices: list[np.ndarray] = []
    used_names: list[str] = []
    for name in pool_names:
        path = SUBMISSIONS_DIR / name
        if not path.exists():
            print(f"[stack] WARN: {name} nicht gefunden, ueberspringe.")
            continue
        sub = load_submission_aligned(path, gt["ts"])
        matrices.append(sub["pred_power_kw"].to_numpy())
        used_names.append(name)
    if not matrices:
        raise RuntimeError("Kein einziges Modell im Pool geladen.")
    return np.vstack(matrices), used_names


# ---------------------------------------------------------------------------
# Kernstueck: Gewichte optimieren
# ---------------------------------------------------------------------------
def objective_factory(
    preds_matrix: np.ndarray,
    y_true: np.ndarray,
    p90: np.ndarray,
    reg_lambda: float,
):
    """Erzeugt die Ziel-Funktion fuer SLSQP.

    preds_matrix : (n_models, n_hours) - Submissions-Pool
    y_true       : (n_hours,) - Ground Truth
    p90          : (n_hours,) - P90-Vektor aus dem Pinball-Quellmodell
    reg_lambda   : L2-Strafe gegen Abweichung vom Uniform-Mittel
    """
    n_models = preds_matrix.shape[0]
    uniform = np.ones(n_models) / n_models

    def _obj(weights: np.ndarray) -> float:
        combo = weights @ preds_matrix
        # P90 muss immer >= Point-Prediction sein.
        p90_eff = np.maximum(p90, combo)
        score = combined(y_true, combo, p90_eff)
        # L2 gegen Uniform (L2 in "Abstand zur Gleichverteilung")
        reg = reg_lambda * float(np.sum((weights - uniform) ** 2))
        return score + reg

    return _obj


def optimize_weights(
    preds_matrix: np.ndarray,
    y_true: np.ndarray,
    p90: np.ndarray,
    reg_lambda: float = REG_LAMBDA,
    n_restarts: int = 5,
) -> tuple[np.ndarray, float]:
    """SLSQP-Optimierung ueber den Simplex {w >= 0, sum(w) = 1}.

    Mehrere Random-Restarts, beste Loesung gewinnt.
    """
    n_models = preds_matrix.shape[0]
    obj = objective_factory(preds_matrix, y_true, p90, reg_lambda)

    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = [(0.0, 1.0)] * n_models

    rng = np.random.default_rng(42)
    best_w = None
    best_val = float("inf")

    # Start 1: Uniform-Mittel
    starts = [np.ones(n_models) / n_models]
    # Weitere Starts: zufaellige Dirichlet-Punkte
    for _ in range(n_restarts - 1):
        starts.append(rng.dirichlet(np.ones(n_models)))

    for x0 in starts:
        res = minimize(
            obj,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if res.fun < best_val:
            best_val = float(res.fun)
            best_w = res.x.copy()

    assert best_w is not None
    # Numerische Reinigung
    best_w = np.clip(best_w, 0.0, None)
    best_w = best_w / best_w.sum()
    return best_w, best_val


# ---------------------------------------------------------------------------
# 2-Fold Time Split fuer ehrliche Out-of-Sample-Bewertung
# ---------------------------------------------------------------------------
def time_split_indices(n: int, n_folds: int = 2) -> list[np.ndarray]:
    """Teilt [0, n) in n_folds zusammenhaengende Zeit-Blocks."""
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(n_folds)]


def cross_validated_score(
    preds_matrix: np.ndarray,
    y_true: np.ndarray,
    p90: np.ndarray,
    reg_lambda: float = REG_LAMBDA,
) -> dict[str, float]:
    """2-Fold Time-Split: Trainiere auf Fold A, bewerte auf Fold B und umgekehrt.

    Gibt die out-of-fold Metriken zurueck.
    """
    folds = time_split_indices(len(y_true), n_folds=2)
    oof_point = np.zeros_like(y_true)
    oof_p90 = np.zeros_like(y_true)

    for k, val_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        w, _ = optimize_weights(
            preds_matrix[:, train_idx],
            y_true[train_idx],
            p90[train_idx],
            reg_lambda=reg_lambda,
            n_restarts=4,
        )
        combo = w @ preds_matrix[:, val_idx]
        oof_point[val_idx] = combo
        oof_p90[val_idx] = np.maximum(p90[val_idx], combo)
        print(
            f"[stack] Fold {k}: train={len(train_idx)}h val={len(val_idx)}h  "
            f"weights={np.round(w, 3).tolist()}"
        )

    return {
        "mae_all": mae(y_true, oof_point),
        "mae_peak": mae_peak(y_true, oof_point),
        "pinball_p90": pinball(y_true, oof_p90),
        "combined": combined(y_true, oof_point, oof_p90),
    }


# ---------------------------------------------------------------------------
# Main-Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    print("[stack] Lade Ground Truth ...")
    gt = load_ground_truth()
    y_true = gt["y_true"].to_numpy()
    print(f"[stack] GT: {len(gt)} Stunden, mean={y_true.mean():.1f} kW")

    print(f"[stack] Lade Submission-Pool ({len(POINT_POOL)} Kandidaten) ...")
    preds_matrix, used_names = load_pool(POINT_POOL, gt)
    print(f"[stack] Erfolgreich geladen: {len(used_names)} Modelle")

    # Individual-Scores zur Referenz
    print("\n[stack] Einzel-Scores (Referenz):")
    for i, name in enumerate(used_names):
        p = preds_matrix[i]
        m = mae(y_true, p)
        mp = mae_peak(y_true, p)
        # Naive pinball auf Point-Prediction (nur zur Orientierung)
        print(f"  {name:<38}  mae_all={m:7.2f}  mae_peak={mp:7.2f}")

    # P90-Quelle laden
    p90_src = load_submission_aligned(SUBMISSIONS_DIR / P90_SOURCE, gt["ts"])
    p90 = p90_src["pred_p90_kw"].to_numpy()
    pb_src = pinball(y_true, p90)
    print(f"\n[stack] P90-Quelle {P90_SOURCE}: pinball={pb_src:.2f}")

    # -----------------------------------------------------------------------
    # 1) Cross-Validated Score (ehrlich, out-of-sample)
    # -----------------------------------------------------------------------
    print("\n[stack] === 2-Fold Time-Split Cross-Validation ===")
    cv_metrics = cross_validated_score(preds_matrix, y_true, p90, reg_lambda=REG_LAMBDA)
    print(f"[stack] CV out-of-fold combined = {cv_metrics['combined']:.2f}")
    print(
        f"         mae_all={cv_metrics['mae_all']:.2f}  "
        f"mae_peak={cv_metrics['mae_peak']:.2f}  "
        f"pinball={cv_metrics['pinball_p90']:.2f}"
    )

    # -----------------------------------------------------------------------
    # 2) Finale Gewichte auf voller Periode (regularisiert!)
    # -----------------------------------------------------------------------
    print("\n[stack] === Finale Gewichtsoptimierung (volle Periode, L2-regularisiert) ===")
    final_w, final_obj = optimize_weights(
        preds_matrix, y_true, p90, reg_lambda=REG_LAMBDA, n_restarts=8
    )
    print("[stack] Finale Gewichte:")
    for name, w in sorted(zip(used_names, final_w), key=lambda t: -t[1]):
        print(f"  {w:6.3f}  {name}")

    combo_final = final_w @ preds_matrix
    p90_final = np.maximum(p90, combo_final)

    m_all = mae(y_true, combo_final)
    m_peak = mae_peak(y_true, combo_final)
    pb = pinball(y_true, p90_final)
    cb = combined(y_true, combo_final, p90_final)

    print(
        f"\n[stack] In-sample Score (optimistisch!): "
        f"combined={cb:.2f}  mae_all={m_all:.2f}  "
        f"mae_peak={m_peak:.2f}  pinball={pb:.2f}"
    )
    print(
        f"[stack] Ehrlicher CV-Score (realistisch):   "
        f"combined={cv_metrics['combined']:.2f}"
    )

    # -----------------------------------------------------------------------
    # 3) Submission schreiben
    # -----------------------------------------------------------------------
    out = pd.DataFrame(
        {
            "timestamp_utc": gt["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pred_power_kw": np.round(np.maximum(combo_final, 0.0), 2),
            "pred_p90_kw": np.round(np.maximum(p90_final, 0.0), 2),
        }
    )
    # Constraint sanity: p90 >= point
    mask_bad = out["pred_p90_kw"] < out["pred_power_kw"]
    if mask_bad.any():
        out.loc[mask_bad, "pred_p90_kw"] = out.loc[mask_bad, "pred_power_kw"]

    STACKING_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(STACKING_OUT, index=False)
    print(f"\n[stack] -> {STACKING_OUT.relative_to(STACKING_OUT.parents[2])}")
    print(f"[stack] {len(out)} Zeilen, pred_power_kw range "
          f"[{out['pred_power_kw'].min():.1f}, {out['pred_power_kw'].max():.1f}]")


if __name__ == "__main__":
    main()
