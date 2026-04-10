"""DFT (Discrete Fourier Transform) decomposition baseline.

Reefer load is dominated by two periodic drivers: a daily cooling cycle
(24 h period) and a weekly arrival/departure rhythm (168 h period). This
model decomposes the training series into sinusoidal components at those
physically meaningful frequencies and simply evaluates the same sinusoids
at the target timestamps — no lag features, no ML model, purely periodic
extrapolation.

Algorithm
---------
1. Load the full training series (ts < target_start, no leakage).
2. Remove a linear trend so the FFT operates on a stationary signal.
3. Compute the FFT of the detrended series.
4. Zero out every bin except:
     - DC (mean level)
     - Harmonics of the 24 h daily  cycle up to order N_DAILY_HARMONICS
     - Harmonics of the 168 h weekly cycle up to order N_WEEKLY_HARMONICS
5. Evaluate the retained sinusoids at each target time index:
       ŷ(t) = (1/N) · Σ_k  Y_filtered[k] · exp(2πi · k · t / N)
   where t is hours elapsed since the start of the training series.
6. Re-add the linear trend at the target times.
7. P90 = point + 90th-pct of in-sample residuals (same strategy as
   physical_decomp.py).

Why this extrapolates correctly
--------------------------------
Each retained sinusoid exp(2πi · k · t / N) is periodic with period N/k
hours.  The daily bin is k ≈ N/24, giving period ≈ 24 h; the weekly bin is
k ≈ N/168, giving period ≈ 168 h.  Evaluating at t > N simply continues
the same cycle — no information from the target window is needed at all.

Frequency selection rationale
------------------------------
Keeping ALL bins would perfectly reproduce the training signal but would
extrapolate pure noise. Keeping only physically motivated harmonics acts as
a band-pass filter that retains the genuine periodic structure and discards
idiosyncratic day-to-day variation.

Run
---
    uv run python alternative_baselines/dft_decomp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_LIGHTGBM = _ROOT / "lightgbm"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_LIGHTGBM) not in sys.path:
    sys.path.insert(0, str(_LIGHTGBM))

from baseline import (  # noqa: E402
    PROJECT_ROOT,
    REEFER_CSV,
    SUBMISSIONS_DIR,
    TARGET_COL,
    TARGET_CSV,
    load_hourly_total,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DFT_OUT = SUBMISSIONS_DIR / "dft_decomp.csv"

# Daily harmonics: periods 24, 12, 8, 6, ... hours (captures intraday shape)
N_DAILY_HARMONICS: int = 12
# Weekly harmonics: periods 168, 84, 56, 42 hours (captures weekly rhythm)
N_WEEKLY_HARMONICS: int = 4


# ---------------------------------------------------------------------------
# Core DFT logic
# ---------------------------------------------------------------------------
def _select_bins(n: int) -> np.ndarray:
    """FFT bin indices to retain (DC + daily/weekly harmonics + conjugates)."""
    keep: set[int] = {0}
    for h in range(1, N_DAILY_HARMONICS + 1):
        k = round(n * h / 24)
        if 0 < k < n:
            keep.add(k)
            keep.add(n - k)  # conjugate — required for real-valued output
    for h in range(1, N_WEEKLY_HARMONICS + 1):
        k = round(n * h / 168)
        if 0 < k < n:
            keep.add(k)
            keep.add(n - k)
    return np.array(sorted(keep))


def _fit_predict(
    y_train: np.ndarray,
    t_train: np.ndarray,
    t_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the DFT model and return (in-sample reconstruction, target predictions).

    Parameters
    ----------
    y_train : power_kw values for training hours
    t_train : integer hour indices for training (0, 1, 2, …, N-1)
    t_pred  : float hour indices for target timestamps (hours since t_train[0])
    """
    n = len(y_train)

    # 1. Linear detrend (handles slow drift over the year)
    poly = np.polyfit(t_train, y_train, 1)
    trend_train = np.polyval(poly, t_train)
    y_det = y_train - trend_train

    # 2. FFT on detrended signal
    Y = np.fft.fft(y_det)

    # 3. Band-pass: zero all bins except physical harmonics
    bins = _select_bins(n)
    Y_filt = np.zeros(n, dtype=complex)
    Y_filt[bins] = Y[bins]

    print(
        f"[dft] Retained {len(bins)} / {n} FFT bins "
        f"(daily harmonics ×{N_DAILY_HARMONICS}, weekly ×{N_WEEKLY_HARMONICS})"
    )

    # 4. In-sample reconstruction for residual P90 computation
    y_insample = np.fft.ifft(Y_filt).real + trend_train

    # 5. Extrapolate: evaluate retained sinusoids at each target time index
    #    ŷ(t) = (1/N) · Σ_k  Y_filt[k] · exp(2πi · k · t / N)
    phases = np.exp(2j * np.pi * bins[np.newaxis, :] * t_pred[:, np.newaxis] / n)
    y_pred = (phases @ Y_filt[bins]).real / n

    # 6. Re-add trend at target times
    y_pred += np.polyval(poly, t_pred)

    return y_insample, y_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load series
    # ------------------------------------------------------------------
    hourly = load_hourly_total(REEFER_CSV)
    print(
        f"[dft] Zeitbereich: {hourly['ts'].min()} -> {hourly['ts'].max()}, "
        f"{len(hourly)} Stunden"
    )

    # ------------------------------------------------------------------
    # 2. Target timestamps
    # ------------------------------------------------------------------
    targets = pd.read_csv(TARGET_CSV)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)
    target_start = targets["ts"].min()
    target_end = targets["ts"].max()
    print(f"[dft] Target: {target_start} -> {target_end} ({len(targets)} Stunden)")

    # ------------------------------------------------------------------
    # 3. Training split — December only (closest seasonal analogue)
    # ------------------------------------------------------------------
    train = hourly[
        (hourly["ts"] < target_start) & (hourly["ts"].dt.month == 12)
    ].copy()
    print(f"[dft] Trainings-Zeilen (Dezember): {len(train)}")

    # Hour-index: 0, 1, 2, …, N-1 for training; float offsets for targets
    t0 = train["ts"].iloc[0]
    t_train = np.arange(len(train), dtype=float)
    t_pred = (targets["ts"] - t0).dt.total_seconds().values / 3600.0

    y_train = train[TARGET_COL].values

    # ------------------------------------------------------------------
    # 4. Fit and predict
    # ------------------------------------------------------------------
    y_insample, y_pred_raw = _fit_predict(y_train, t_train, t_pred)

    # In-sample diagnostics
    resid = y_train - y_insample
    print(
        f"[dft] In-sample MAE:  {np.mean(np.abs(resid)):.1f} kW  "
        f"(train mean={y_train.mean():.1f} kW)"
    )

    # ------------------------------------------------------------------
    # 5. P90 spread from in-sample residuals
    # ------------------------------------------------------------------
    spread = max(float(np.quantile(resid, 0.9)), 0.0)
    print(f"[dft] P90 Residual-Spread (90th-pct): {spread:.1f} kW")

    pred_point = np.clip(y_pred_raw, 0.0, None)
    pred_p90 = np.maximum(pred_point + spread, pred_point)

    print(
        f"[dft] pred_power_kw: mean={pred_point.mean():.1f}, "
        f"range=[{pred_point.min():.1f}, {pred_point.max():.1f}]"
    )

    # ------------------------------------------------------------------
    # 6. Write submission
    # ------------------------------------------------------------------
    submission = pd.DataFrame(
        {
            "timestamp_utc": targets["timestamp_utc"].values,
            "pred_power_kw": np.round(pred_point, 2),
            "pred_p90_kw": np.round(pred_p90, 2),
        }
    )
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(DFT_OUT, index=False, float_format="%.2f")
    print(f"[dft] -> {DFT_OUT.relative_to(PROJECT_ROOT)} ({len(submission)} Zeilen)")


if __name__ == "__main__":
    main()
