"""Open-Meteo Historical Weather Archive -> Cache -> DataFrame.

Die Open-Meteo-Archive-API (https://archive-api.open-meteo.com) liefert
kostenlose stuendliche Wetter-Historie ohne API-Key. Wir laden einmalig
die drei relevanten Variablen fuer den Standort CTH Hamburg-Waltershof
und cachen das Ergebnis als CSV unter:
    `weather_data_lean/final/open_meteo_complete/openmeteo_cth_hamburg.csv`

Die drei Variablen:
    - shortwave_radiation  (W/m^2)  - Sonneneinstrahlung
    - temperature_2m       (deg C)  - Lufttemperatur 2 m
    - wind_speed_10m       (km/h)   - Windgeschwindigkeit 10 m

wind_direction_10m ist bewusst weggelassen: schwache Korrelation zwischen
CTH-Station und OpenMeteo-Reanalyse (MAE ~44 deg), zirkulaere Variable,
und physikalisch kein starker Prediktor fuer Reefer-Stromverbrauch.

Beim zweiten Aufruf wird die CSV gelesen statt neu geladen. Die Cache-CSV
committen wir mit ins Repo - dadurch braucht der Organizer-Rerun keine
Internetverbindung.

Benutzung:
    from weather_external import load_cth_weather
    weather = load_cth_weather()
    # DataFrame: ts (UTC), shortwave_radiation, temperature_2m, wind_speed_10m
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


# Koordinaten Eurogate Container Terminal Hamburg (CTH, Predoehlkai/Waltershof)
CTH_LATITUDE: float = 53.532
CTH_LONGITUDE: float = 9.924

# Default-Zeitraum matcht unsere reefer_release.csv
DEFAULT_START: str = "2025-01-01"
DEFAULT_END: str = "2026-01-10"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Cache-Location fuer den Eigen-Download aus der Open-Meteo-API. Liegt bewusst
# unter weather_data_lean/final/open_meteo_complete/, damit alle Wetter-Quellen
# nebeneinander an einem Ort liegen (lean-Files und vollstaendiger Cache).
OPEN_METEO_CACHE_DIR = PROJECT_ROOT / "weather_data_lean" / "final" / "open_meteo_complete"
OPENMETEO_CACHE = OPEN_METEO_CACHE_DIR / "openmeteo_cth_hamburg.csv"

# Die drei Variablen, die wir von Open-Meteo wollen. wind_direction_10m ist
# bewusst weggelassen (s. Modul-Docstring).
OPEN_METEO_VARIABLES: list[str] = [
    "shortwave_radiation",
    "temperature_2m",
    "wind_speed_10m",
]

# Pfade zu den bereits aufbereiteten "lean" Wetter-Dateien (von Kyle vorbereitet).
# CTH_lean.csv ist die echte Messstation am Container Terminal Hamburg.
# open-meteo_lean.csv ist eine Open-Meteo-Reanalyse fuer denselben Zeitraum.
# Beide haben identische Spalten, damit sie vergleichbar sind.
LEAN_WEATHER_DIR = PROJECT_ROOT / "weather_data_lean" / "final"
CTH_LEAN_CSV = LEAN_WEATHER_DIR / "CTH_lean.csv"
OPENMETEO_LEAN_CSV = LEAN_WEATHER_DIR / "open-meteo_lean.csv"


def download_openmeteo(
    *,
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    variables: list[str],
    cache_path: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Laedt stuendliche Wetter-Historie von Open-Meteo und cached sie.

    Parameters
    ----------
    latitude, longitude : Standort in Dezimalgrad.
    start, end : Inklusive Datumsgrenzen im Format 'YYYY-MM-DD'.
    variables : Liste der gewuenschten Open-Meteo hourly-Variablen,
        z.B. ['shortwave_radiation', 'temperature_2m'].
    cache_path : Wohin die CSV gecached wird.
    force : Wenn True, re-downloaden auch wenn Cache existiert.

    Returns
    -------
    DataFrame mit Spalten ['ts', *variables]. `ts` ist UTC-aware.
    """
    if not force and cache_path.exists():
        print(f"[weather] Cache gefunden: {cache_path.relative_to(PROJECT_ROOT)}")
        df = pd.read_csv(cache_path)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }
    print(f"[weather] Lade Open-Meteo: {start} -> {end}, vars={variables}")
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()

    hourly = payload["hourly"]
    df = pd.DataFrame(
        {
            "ts": hourly["time"],
            **{v: hourly[v] for v in variables},
        }
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"[weather] {len(df)} Zeilen nach {cache_path.relative_to(PROJECT_ROOT)}")
    return df


def load_cth_weather(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    force: bool = False,
) -> pd.DataFrame:
    """Laedt shortwave_radiation + temperature_2m + wind_speed_10m fuer CTH Hamburg.

    Rueckgabe: DataFrame mit Spalten ['ts', 'shortwave_radiation',
    'temperature_2m', 'wind_speed_10m']. `ts` ist UTC-aware.
    """
    return download_openmeteo(
        latitude=CTH_LATITUDE,
        longitude=CTH_LONGITUDE,
        start=start,
        end=end,
        variables=OPEN_METEO_VARIABLES,
        cache_path=OPENMETEO_CACHE,
        force=force,
    )


if __name__ == "__main__":
    # Direkter Aufruf: forciert Download, zeigt Sanity-Stats fuer alle Variablen.
    df = load_cth_weather(force=True)
    print()
    print(f"Zeilen       : {len(df)}")
    print(f"Zeitbereich  : {df['ts'].min()} -> {df['ts'].max()}")
    for var in OPEN_METEO_VARIABLES:
        s = df[var]
        unit = {
            "shortwave_radiation": "W/m^2",
            "temperature_2m": "deg C",
            "wind_speed_10m": "km/h",
        }.get(var, "")
        print(
            f"  {var:<22} min={s.min():7.2f}  max={s.max():7.2f}  "
            f"mean={s.mean():7.2f}  NaN={s.isna().sum():3d}  {unit}"
        )
