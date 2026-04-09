"""Open-Meteo Historical Weather Archive -> Cache -> DataFrame.

Die Open-Meteo-Archive-API (https://archive-api.open-meteo.com) liefert
kostenlose stuendliche Wetter-Historie ohne API-Key. Wir laden einmalig
die gewuenschten Variablen fuer den Standort CTH Hamburg-Waltershof und
cachen das Ergebnis als CSV unter
`participant_package/daten/external/openmeteo_cth_hamburg.csv`.

Beim zweiten Aufruf wird die CSV gelesen statt neu geladen. Die Cache-CSV
committen wir mit ins Repo - dadurch braucht der Organizer-Rerun keine
Internetverbindung.

Benutzung:
    from weather_external import load_cth_shortwave
    weather = load_cth_shortwave()   # DataFrame: ts (UTC), shortwave_radiation
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

# Cache-Location
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "participant_package" / "daten" / "external"
OPENMETEO_CACHE = EXTERNAL_DIR / "openmeteo_cth_hamburg.csv"


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


def load_cth_shortwave(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    force: bool = False,
) -> pd.DataFrame:
    """Convenience: laedt shortwave_radiation fuer CTH Hamburg-Waltershof."""
    return download_openmeteo(
        latitude=CTH_LATITUDE,
        longitude=CTH_LONGITUDE,
        start=start,
        end=end,
        variables=["shortwave_radiation"],
        cache_path=OPENMETEO_CACHE,
        force=force,
    )


if __name__ == "__main__":
    # Direkter Aufruf: forciert Download, zeigt Sanity-Stats.
    df = load_cth_shortwave(force=True)
    print()
    print(f"Zeilen       : {len(df)}")
    print(f"Zeitbereich  : {df['ts'].min()} -> {df['ts'].max()}")
    print(
        f"shortwave    : min={df['shortwave_radiation'].min():.1f}, "
        f"max={df['shortwave_radiation'].max():.1f}, "
        f"mean={df['shortwave_radiation'].mean():.1f} W/m^2"
    )
    print(f"NaN-Anteil   : {df['shortwave_radiation'].isna().mean():.4%}")
