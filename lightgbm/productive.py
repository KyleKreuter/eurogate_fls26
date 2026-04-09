"""Produktiv-Modell fuer die Reefer Peak Load Challenge.

Das ist das Modell, das wir aktiv iterieren und verbessern. Start-Konfiguration:

    baseline.py (minimale Features)
    + Peak-Weighting (mult=10)                    <- aktuell aktiv
    + Wetter-Features                              <- spaeter
    + Container-Mix-Features (stack_tier, ...)    <- spaeter
    + P90-Calibration / Conformal Prediction      <- spaeter

Jede Aenderung hier wird ueber eval.py gegen baseline.csv gemessen.

Ausfuehren:
    uv run python lightgbm/productive.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from baseline import (
    PEAK_QUANTILE,
    PROJECT_ROOT,
    SUBMISSIONS_DIR,
    run_training_and_submission,
)


# Peak-Weight-Multiplier fuer das Training. 1.0 = keine Gewichtung (identisch
# zu baseline.py). Wir haben das aktuell auf 1.0 zurueckgesetzt, weil das
# Tuning auf Okt/Nov/Dez (mult=10 gewinnt) auf dem Target-Fenster Anfang
# Januar den Score VERSCHLECHTERT hat. Das ist ein Distribution-Shift-Fall:
# Target-Fenster ist ein ruhiger Zeitraum ohne echte Peaks, da schadet das
# Weighting. Wird neu getunt, sobald wir echte Features (Wetter, Container-Mix)
# einbauen und/oder ein januar-aehnliches Val-Fenster haben.
PEAK_MULTIPLIER: float = 1.0

PRODUCTIVE_OUT = SUBMISSIONS_DIR / "productive.csv"


def peak_weights_factory(
    multiplier: float, peak_quantile: float = PEAK_QUANTILE
):
    """Gibt eine Funktion zurueck, die aus einem Target-Vektor Sample-Weights macht.

    Stunden im oberen `peak_quantile`-Quantil bekommen `multiplier`, alle
    anderen 1.0. Der umschliessende Factory-Aufbau ist noetig, weil
    `run_training_and_submission` eine parameterlose Weight-Funktion erwartet
    (sie bekommt nur den Target-Vektor).
    """
    def _weight_fn(y: pd.Series) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        threshold = np.quantile(y_arr, peak_quantile)
        return np.where(y_arr >= threshold, float(multiplier), 1.0)

    return _weight_fn


def main() -> None:
    # mult=1.0 -> komplett ohne Weighting, exakt derselbe Pfad wie in
    # baseline.py. Damit laesst sich per `diff submissions/baseline.csv
    # submissions/productive.csv` bitgenau pruefen, dass beide Pipelines
    # identisch sind.
    if PEAK_MULTIPLIER == 1.0:
        print("[productive] Peak-Weighting AUS (mult=1.0) -> identisch zu baseline")
        weight_fn = None
    else:
        print(
            f"[productive] Peak-Weighting aktiv: mult={PEAK_MULTIPLIER}, "
            f"peak_quantile={PEAK_QUANTILE}"
        )
        weight_fn = peak_weights_factory(
            multiplier=PEAK_MULTIPLIER, peak_quantile=PEAK_QUANTILE
        )
    run_training_and_submission(
        weight_fn=weight_fn,
        out_path=PRODUCTIVE_OUT,
        label="productive",
    )


if __name__ == "__main__":
    main()
