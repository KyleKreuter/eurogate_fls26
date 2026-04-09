"""Produktiv-Modell fuer die Reefer Peak Load Challenge.

Das ist das Modell, das wir aktiv iterieren und verbessern. Start-Konfiguration:

    baseline.py (minimale Features: hour, dow, lag_24h, lag_168h)
    + shortwave_radiation (Open-Meteo)            <- aktiv
    + temperature_2m      (Open-Meteo)            <- aktiv (NEU)
    + wind_speed_10m      (Open-Meteo)            <- aktiv (NEU)
    + Peak-Weighting                              <- aus (mult=1.0)
    + Container-Mix-Features (stack_tier, ...)    <- spaeter
    + P90-Calibration / Conformal Prediction      <- spaeter

Jede Aenderung hier wird ueber eval.py gegen baseline.csv gemessen.

Ausfuehren:
    uv run python lightgbm/productive.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# WICHTIG: lokale Module (wie weather_external) MUESSEN vor `baseline`
# importiert werden. baseline.py entfernt beim Module-Load das lightgbm/-
# Verzeichnis aus sys.path (um die Namenskollision mit dem installierten
# lightgbm-Package zu vermeiden), danach sind weitere lokale Imports nicht
# mehr aufloesbar.
from weather_external import OPEN_METEO_VARIABLES, load_cth_weather

from baseline import (
    FEATURES,
    PEAK_QUANTILE,
    SUBMISSIONS_DIR,
    run_training_and_submission,
)


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
# Peak-Weight-Multiplier. 1.0 = keine Gewichtung. Wir haben ihn bewusst auf
# 1.0 gesetzt, weil das frueher auf Okt-Dez getunte mult=10 auf dem Januar-
# Target-Fenster Distribution-Shift-artig schadet. Wird neu getunt, sobald
# wir ein januar-aehnliches Val-Fenster haben.
PEAK_MULTIPLIER: float = 1.0

# Feature-Set fuer productive.py. Baseline-Features plus alle drei Open-Meteo-
# Variablen. Weitere Features werden hier angehaengt.
PRODUCTIVE_FEATURES: list[str] = FEATURES + OPEN_METEO_VARIABLES

PRODUCTIVE_OUT = SUBMISSIONS_DIR / "productive.csv"


# ---------------------------------------------------------------------------
# Peak-Weights
# ---------------------------------------------------------------------------
def peak_weights_factory(
    multiplier: float, peak_quantile: float = PEAK_QUANTILE
):
    """Gibt eine Funktion zurueck, die aus einem Target-Vektor Sample-Weights macht.

    Stunden im oberen `peak_quantile`-Quantil bekommen `multiplier`, alle
    anderen 1.0.
    """
    def _weight_fn(y: pd.Series) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        threshold = np.quantile(y_arr, peak_quantile)
        return np.where(y_arr >= threshold, float(multiplier), 1.0)

    return _weight_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"[productive] Feature-Set: {PRODUCTIVE_FEATURES}")

    if PEAK_MULTIPLIER == 1.0:
        print("[productive] Peak-Weighting AUS (mult=1.0)")
        weight_fn = None
    else:
        print(
            f"[productive] Peak-Weighting aktiv: mult={PEAK_MULTIPLIER}, "
            f"peak_quantile={PEAK_QUANTILE}"
        )
        weight_fn = peak_weights_factory(
            multiplier=PEAK_MULTIPLIER, peak_quantile=PEAK_QUANTILE
        )

    # Wetter-Daten laden (aus Open-Meteo-Cache, ggf. Erstdownload)
    weather = load_cth_weather()
    print(f"[productive] weather: {len(weather)} Zeilen, Variablen={OPEN_METEO_VARIABLES}")
    for var in OPEN_METEO_VARIABLES:
        s = weather[var]
        print(
            f"[productive]   {var:<22} "
            f"min={s.min():7.1f}  max={s.max():7.1f}  mean={s.mean():7.1f}"
        )

    run_training_and_submission(
        weight_fn=weight_fn,
        out_path=PRODUCTIVE_OUT,
        label="productive",
        extra_features_df=weather,
        features=PRODUCTIVE_FEATURES,
    )


if __name__ == "__main__":
    main()
