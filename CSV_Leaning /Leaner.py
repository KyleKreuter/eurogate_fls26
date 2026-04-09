import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
WETTER_DIR = PROJECT_DIR / "participant_package" / "daten" / "wetterdaten"

TIMEZONE = "Europe/Berlin"
TIME_FORMAT = "%Y-%m-%dT%H:%M%z"
COL_TEMP = "temperature_2m (°C)"
COL_WIND = "wind_speed_10m (km/h)"
COL_WDIR = "wind_direction_10m (°)"
COLUMNS = ["time", COL_TEMP, COL_WIND, COL_WDIR]

LOCATIONS = {
    "Zentralgate": {
        "cth_temp": BASE_DIR / "CTH_Temperatur_Zentralgate  Okt 25 - 23 Feb 26.csv",
        "cth_wind": BASE_DIR / "CTH_Wind_Zentralgate  Okt 25 - 23 Feb 26.csv",
        "cth_wdir": BASE_DIR / "CTH_Windrichtung_Zentralgate  Okt 25 - 23 Feb 26.csv",
        "meteo": BASE_DIR / "open-meteo-Zentralgate.csv",
        "cth_out": BASE_DIR / "CTH_Zentralgate_lean.csv",
        "meteo_out": BASE_DIR / "open-meteo-Zentralgate_lean.csv",
    },
    "Halle3": {
        "cth_temp": WETTER_DIR / "CTH_Temperatur_VC_Halle3 Okt 25 - 23 Feb 26.csv",
        "cth_wind": WETTER_DIR / "CTH_Wind_VC_Halle3  Okt 25 - 23 Feb 26.csv",
        "cth_wdir": BASE_DIR / "CTH_Windrichtung_VC_Halle3  Okt 25 - 23 Feb 26.csv",
        "meteo": BASE_DIR / "open-meteo.csv",
        "cth_out": BASE_DIR / "CTH_Halle3_lean.csv",
        "meteo_out": BASE_DIR / "open-meteo_lean.csv",
    },
}


def circular_mean(angles):
    angles = angles.dropna()
    if angles.empty:
        return np.nan
    rads = np.deg2rad(angles)
    mean_sin = np.mean(np.sin(rads))
    mean_cos = np.mean(np.cos(rads))
    result = np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360
    return round(result, 1)


PLAUSIBLE_RANGES = {
    "temp": (-50, 60),
    "wind": (0, 42),     # m/s (vor ×3.6 Konvertierung)
    "wdir": (0, 360),
}


def read_cth(path, sensor_type):
    df = pd.read_csv(path, sep=";", decimal=",", usecols=["UtcTimestamp", "Value"])
    df["UtcTimestamp"] = pd.to_datetime(df["UtcTimestamp"]).dt.tz_localize("UTC")
    df["hour_utc"] = df["UtcTimestamp"].dt.floor("h")
    lo, hi = PLAUSIBLE_RANGES[sensor_type]
    n_before = len(df)
    df.loc[~df["Value"].between(lo, hi), "Value"] = np.nan
    n_filtered = df["Value"].isna().sum()
    if n_filtered > 0:
        print(f"    Plausibilitätsfilter: {n_filtered}/{n_before} Werte außerhalb [{lo}, {hi}] → NaN")
    return df


def aggregate_hourly(df, agg_func="mean"):
    if agg_func == "circular_mean":
        hourly = df.groupby("hour_utc")["Value"].apply(circular_mean).reset_index()
    else:
        hourly = df.groupby("hour_utc")["Value"].mean().round(1).reset_index()
    hourly.columns = ["time", "value"]

    # Stunden mit >50% gefilterten Werten → NaN (Sensordefekt für ganze Stunde)
    nan_ratio = df.groupby("hour_utc")["Value"].apply(lambda x: x.isna().mean())
    bad_hours = nan_ratio[nan_ratio > 0.5].index
    if len(bad_hours) > 0:
        hourly.loc[hourly["time"].isin(bad_hours), "value"] = np.nan
        print(f"    Stunden mit >50% Sensordefekt → NaN: {len(bad_hours)}")

    return hourly


def process_location(name, cfg):
    print(f"=== {name} ===")

    # CTH Temperatur
    print(f"  Lese CTH Temperatur...")
    temp = aggregate_hourly(read_cth(cfg["cth_temp"], "temp"))
    temp = temp.rename(columns={"value": COL_TEMP})

    # CTH Wind (m/s → km/h)
    print(f"  Lese CTH Wind...")
    wind = aggregate_hourly(read_cth(cfg["cth_wind"], "wind"))
    wind["value"] = (wind["value"] * 3.6).round(1)
    wind = wind.rename(columns={"value": COL_WIND})

    # CTH Windrichtung (zirkulärer Mittelwert)
    print(f"  Lese CTH Windrichtung...")
    wdir = aggregate_hourly(read_cth(cfg["cth_wdir"], "wdir"), agg_func="circular_mean")
    wdir = wdir.rename(columns={"value": COL_WDIR})

    # Zusammenführen (outer merge → NaN wo CTH-Daten fehlen)
    cth = temp.merge(wind, on="time", how="outer").merge(wdir, on="time", how="outer")
    cth = cth.dropna(subset=[COL_TEMP, COL_WIND, COL_WDIR], how="all")
    cth = cth.sort_values("time").reset_index(drop=True)
    cth["time"] = cth["time"].dt.tz_convert(TIMEZONE)

    # Open-Meteo einlesen
    print(f"  Lese Open-Meteo...")
    meteo = pd.read_csv(cfg["meteo"], skiprows=3)
    meteo["time"] = pd.to_datetime(meteo["time"]).dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
    meteo = meteo[["time", COL_TEMP, COL_WIND, COL_WDIR]]

    # Überlappungszeitraum
    overlap_start = max(cth["time"].min(), meteo["time"].min())
    overlap_end = min(cth["time"].max(), meteo["time"].max())

    cth_lean = cth[(cth["time"] >= overlap_start) & (cth["time"] <= overlap_end)].copy()
    meteo_lean = meteo[(meteo["time"] >= overlap_start) & (meteo["time"] <= overlap_end)].copy()

    # Ausgabe schreiben
    cth_lean["time"] = cth_lean["time"].dt.strftime(TIME_FORMAT)
    cth_lean[COLUMNS].to_csv(cfg["cth_out"], index=False)

    meteo_lean["time"] = meteo_lean["time"].dt.strftime(TIME_FORMAT)
    meteo_lean[COLUMNS].to_csv(cfg["meteo_out"], index=False)

    print(f"  CTH:   {len(cth_lean)} Stunden -> {cfg['cth_out'].name}")
    print(f"  Meteo: {len(meteo_lean)} Stunden -> {cfg['meteo_out'].name}")
    print(f"  Zeitraum: {overlap_start} bis {overlap_end}")
    print()


for name, cfg in LOCATIONS.items():
    process_location(name, cfg)
