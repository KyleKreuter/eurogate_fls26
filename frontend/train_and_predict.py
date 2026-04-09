from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:
    import subprocess
    import sys
    print("Installing strict dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "pandas", "numpy"])
    from sklearn.ensemble import HistGradientBoostingRegressor


def find_public_dir() -> Path:
    candidates = [
        Path.cwd(),
        Path.cwd() / "challenge" / "release" / "v1" / "public",
        Path.cwd() / "challenge" / "bundle" / "v1" / "participant_package",
        Path.cwd().parent,
        Path.cwd().parent / "public",
    ]
    for candidate in candidates:
        if (candidate / "target_timestamps.csv").exists():
            return candidate
    return Path('/Users/rkohlbach/Downloads/participant_package_lokal')

PUBLIC_DIR = find_public_dir()
TARGETS_CSV = PUBLIC_DIR / "target_timestamps.csv"
OUTPUT_CSV = PUBLIC_DIR / "predictions.csv"

# Unzipped reefer release file if available, else zip
REEFER_CSV = PUBLIC_DIR / "reefer_release.csv"
REEFER_ZIP = PUBLIC_DIR / "reefer_release.zip"

def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1]
    return datetime.fromisoformat(text.replace(" ", "T"))

def load_target_hours(path: Path) -> list[datetime]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [parse_timestamp(row["timestamp_utc"]) for row in reader if row.get("timestamp_utc")]

def aggregate_hourly_load() -> pd.DataFrame:
    print("Loading reefer data...")
    if REEFER_CSV.exists():
        df = pd.read_csv(REEFER_CSV, sep=';', decimal=',', usecols=['EventTime', 'AvPowerCons'])
    elif REEFER_ZIP.exists():
        with zipfile.ZipFile(REEFER_ZIP) as zf:
            with zf.open("reefer_release.csv") as raw:
                df = pd.read_csv(raw, sep=';', decimal=',', usecols=['EventTime', 'AvPowerCons'])
    else:
        raise FileNotFoundError("Could not find reefer_release.csv or .zip")

    # Drop NAs
    df = df.dropna(subset=['EventTime', 'AvPowerCons'])
    df = df[(df['AvPowerCons'] >= 0)]

    print("Aggregating to hourly load (kW)...")
    # Convert EventTime
    df['EventTime'] = pd.to_datetime(df['EventTime'])
    df['EventTime'] = df['EventTime'].dt.tz_localize(None) # Ensure timezone-naive
    
    # Floor to hour
    df['hour'] = df['EventTime'].dt.floor('H')
    
    # Group by hour and sum power, convert to kW
    hourly_kw = df.groupby('hour')['AvPowerCons'].sum() / 1000.0
    
    hourly_df = hourly_kw.reset_index()
    hourly_df = hourly_df.rename(columns={'hour': 'timestamp', 'AvPowerCons': 'target_kw'})
    hourly_df = hourly_df.sort_values('timestamp').reset_index(drop=True)
    return hourly_df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Building historical features...")
    min_ts = df['timestamp'].min()
    max_ts = df['timestamp'].max()
    full_range = pd.date_range(min_ts, max_ts, freq='H')
    df_full = pd.DataFrame({'timestamp': full_range})
    df_full = df_full.merge(df, on='timestamp', how='left')
    
    df_full['target_kw'] = df_full['target_kw'].ffill()

    # Time features
    df_full['hour'] = df_full['timestamp'].dt.hour
    df_full['dayofweek'] = df_full['timestamp'].dt.dayofweek
    df_full['month'] = df_full['timestamp'].dt.month
    df_full['is_weekend'] = df_full['dayofweek'].isin([5, 6]).astype(int)
    
    # Lags: t-24, t-168
    df_full['load_t24'] = df_full['target_kw'].shift(24)
    df_full['load_t168'] = df_full['target_kw'].shift(168)
    
    df_full['rolling_mean_t168_24h'] = df_full['load_t168'].rolling(window=24, min_periods=1).mean()
    
    return df_full

def train_and_predict():
    target_hours_list = load_target_hours(TARGETS_CSV)
    if not target_hours_list:
        print("No target hours found.")
        return

    load_df = aggregate_hourly_load()
    features_df = build_features(load_df)
    
    train_df = features_df.dropna().copy()
    feature_cols = ['hour', 'dayofweek', 'month', 'is_weekend', 'load_t24', 'load_t168', 'rolling_mean_t168_24h']
    
    X_train = train_df[feature_cols]
    y_train = train_df['target_kw']
    
    print(f"Training models on {len(X_train)} historical hours...")
    # Model 1: Point Forecast (MAE optimized usually works well for absolute error)
    point_model = HistGradientBoostingRegressor(loss='absolute_error', random_state=42, max_iter=200)
    point_model.fit(X_train, y_train)
    
    # Model 2: P90 Forecast -> Alpha = 0.90
    p90_model = HistGradientBoostingRegressor(loss='quantile', quantile=0.90, random_state=42, max_iter=200)
    p90_model.fit(X_train, y_train)
    
    print("Forecasting target hours...")
    
    target_df = pd.DataFrame({'timestamp': target_hours_list})
    target_df['timestamp'] = pd.to_datetime(target_df['timestamp']).dt.tz_localize(None)
    
    all_df = pd.concat([features_df[['timestamp', 'target_kw']], target_df], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    all_df['hour'] = all_df['timestamp'].dt.hour
    all_df['dayofweek'] = all_df['timestamp'].dt.dayofweek
    all_df['month'] = all_df['timestamp'].dt.month
    all_df['is_weekend'] = all_df['dayofweek'].isin([5, 6]).astype(int)
    
    all_df['load_t24'] = all_df['target_kw'].shift(24)
    all_df['load_t168'] = all_df['target_kw'].shift(168)
    all_df['rolling_mean_t168_24h'] = all_df['load_t168'].rolling(window=24, min_periods=1).mean()
    
    target_features = all_df[all_df['timestamp'].isin(target_df['timestamp'])].copy()
    
    # Fill gaps if the target hours are disconnected by more than t-168 from history
    overall_mean = load_df['target_kw'].mean()
    target_features['load_t24'] = target_features['load_t24'].fillna(overall_mean)
    target_features['load_t168'] = target_features['load_t168'].fillna(overall_mean)
    target_features['rolling_mean_t168_24h'] = target_features['rolling_mean_t168_24h'].fillna(overall_mean)
    
    X_test = target_features[feature_cols]
    
    point_preds = point_model.predict(X_test)
    p90_preds = p90_model.predict(X_test)
    
    predictions = []
    # Using simple indexing loop 
    for i, row in enumerate(target_features.itertuples()):
        pred_kw = float(point_preds[i])
        pred_p90 = float(p90_preds[i])
        
        # Enforce Constraints
        pred_p90 = max(pred_p90, pred_kw)
        pred_kw = max(0.0, pred_kw)
        pred_p90 = max(0.0, pred_p90)
        
        ts_str = row.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        predictions.append({
            "timestamp_utc": ts_str,
            "pred_power_kw": round(pred_kw, 6),
            "pred_p90_kw": round(pred_p90, 6),
        })
        
    print(f"Writing {len(predictions)} predictions to {OUTPUT_CSV}")
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "pred_power_kw", "pred_p90_kw"])
        writer.writeheader()
        writer.writerows(predictions)
        
    print("Done! Submission file generated successfully.")

if __name__ == '__main__':
    train_and_predict()
