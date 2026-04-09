import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

CTH_ZENTRAL = BASE_DIR / "CTH_Zentralgate_lean.csv"
CTH_HALLE3 = BASE_DIR / "CTH_Halle3_lean.csv"
CTH_OUT = BASE_DIR / "CTH_lean.csv"

COL_TEMP = "temperature_2m (°C)"
COL_WIND = "wind_speed_10m (km/h)"
COL_WDIR = "wind_direction_10m (°)"


def circular_mean_pair(a, b):
    angles = [x for x in [a, b] if not np.isnan(x)]
    if not angles:
        return np.nan
    rads = np.deg2rad(angles)
    result = np.rad2deg(np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads)))) % 360
    return round(result, 1)


z = pd.read_csv(CTH_ZENTRAL)
h = pd.read_csv(CTH_HALLE3)

merged = z.merge(h, on="time", how="outer", suffixes=("_z", "_h"))
merged = merged.sort_values("time").reset_index(drop=True)

# Temperatur + Wind: arithmetischer Mittelwert (NaN ignorieren → Einzelwert wenn nur einer)
merged[COL_TEMP] = merged[[f"{COL_TEMP}_z", f"{COL_TEMP}_h"]].mean(axis=1).round(1)
merged[COL_WIND] = merged[[f"{COL_WIND}_z", f"{COL_WIND}_h"]].mean(axis=1).round(1)

# Windrichtung: zirkulärer Mittelwert
merged[COL_WDIR] = merged.apply(
    lambda r: circular_mean_pair(r[f"{COL_WDIR}_z"], r[f"{COL_WDIR}_h"]), axis=1
)

result = merged[["time", COL_TEMP, COL_WIND, COL_WDIR]]
result.to_csv(CTH_OUT, index=False)

print(f"Zentralgate: {len(z)} Zeilen")
print(f"Halle3:      {len(h)} Zeilen")
print(f"Combined:    {len(result)} Zeilen -> {CTH_OUT.name}")
print(f"Zeitraum:    {result['time'].iloc[0]} bis {result['time'].iloc[-1]}")
