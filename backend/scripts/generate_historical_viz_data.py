import pandas as pd
import zipfile
import json
import numpy as np
from pathlib import Path

PUBLIC_DIR = Path('/Users/rkohlbach/Downloads/participant_package_lokal')
REEFER_CSV_PATH = PUBLIC_DIR / 'reefer_release.csv'
REEFER_ZIP_PATH = PUBLIC_DIR / 'reefer_release.zip'
OUTPUT_JSON = PUBLIC_DIR / 'dashboard' / 'historical_visualizations.json'

def load_data() -> pd.DataFrame:
    print("Loading 3.7 million historical rows...")
    if REEFER_CSV_PATH.exists():
        df = pd.read_csv(REEFER_CSV_PATH, sep=';', decimal=',', 
                         usecols=['container_uuid', 'EventTime', 'TemperatureSetPoint', 'TemperatureAmbient', 'AvPowerCons'])
    else:
        with zipfile.ZipFile(REEFER_ZIP_PATH) as zf:
            with zf.open("reefer_release.csv") as raw:
                df = pd.read_csv(raw, sep=';', decimal=',', 
                                 usecols=['container_uuid', 'EventTime', 'TemperatureSetPoint', 'TemperatureAmbient', 'AvPowerCons'])
                
    df['EventTime'] = pd.to_datetime(df['EventTime'])
    return df

def generate_insights(df: pd.DataFrame):
    payload = {}

    print("Crunching: Container Connection Duration distribution...")
    # Calculate connection time per container
    durations = df.groupby('container_uuid')['EventTime'].agg(['min', 'max'])
    durations['hours'] = (durations['max'] - durations['min']).dt.total_seconds() / 3600.0
    
    # Binning the durations
    bins = [-1, 24, 72, 168, 336, float('inf')]
    labels = ['< 24 Hours', '1-3 Days', '3-7 Days', '1-2 Weeks', '> 2 Weeks']
    durations['category'] = pd.cut(durations['hours'], bins=bins, labels=labels)
    dwell_counts = durations['category'].value_counts().reindex(labels)
    
    payload['dwell_time'] = {
        'labels': labels,
        'counts': dwell_counts.fillna(0).tolist()
    }

    print("Crunching: Ambient Temperature vs Power consumption...")
    # Clean anomalies out of temperatures first
    valid_ambient = df[(df['TemperatureAmbient'] > -30) & (df['TemperatureAmbient'] < 50) & (df['AvPowerCons'] >= 0)].copy()
    # Floor to nearest 2 degrees for smooth bins
    valid_ambient['Ambient_Bin'] = (valid_ambient['TemperatureAmbient'] / 2).round() * 2
    ambient_power = valid_ambient.groupby('Ambient_Bin')['AvPowerCons'].mean().reset_index()
    ambient_power = ambient_power.sort_values('Ambient_Bin')
    
    payload['ambient_power'] = {
        'temps': ambient_power['Ambient_Bin'].tolist(),
        'power_kw': (ambient_power['AvPowerCons'] / 1000.0).round(2).tolist()
    }
    
    print("Crunching: SetPoint vs Power consumption...")
    valid_setpoint = df[(df['TemperatureSetPoint'] > -40) & (df['TemperatureSetPoint'] < 30) & (df['AvPowerCons'] >= 0)].copy()
    valid_setpoint['SetPoint_Bin'] = (valid_setpoint['TemperatureSetPoint'] / 5).round() * 5
    sp_power = valid_setpoint.groupby('SetPoint_Bin')['AvPowerCons'].mean().reset_index()
    sp_power = sp_power.sort_values('SetPoint_Bin')
    
    payload['setpoint_power'] = {
        'temps': sp_power['SetPoint_Bin'].tolist(),
        'power_kw': (sp_power['AvPowerCons'] / 1000.0).round(2).tolist()
    }

    print("Saving to JSON payload for dashboard...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(payload, f)
    print("Done!")

if __name__ == '__main__':
    df = load_data()
    generate_insights(df)
