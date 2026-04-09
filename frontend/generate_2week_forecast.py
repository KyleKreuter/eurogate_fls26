from __future__ import annotations
import csv
import zipfile
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

PUBLIC_DIR = Path('/Users/rkohlbach/Downloads/participant_package_lokal')
REEFER_CSV = PUBLIC_DIR / "reefer_release.csv"
REEFER_ZIP = PUBLIC_DIR / "reefer_release.zip"
OUTPUT_CSV = PUBLIC_DIR / "dashboard" / "dashboard_data.csv"

def aggregate_hourly_load() -> pd.DataFrame:
    print("Loading historical data...")
    if REEFER_CSV.exists():
        df = pd.read_csv(REEFER_CSV, sep=';', decimal=',', usecols=['EventTime', 'AvPowerCons'])
    else:
        with zipfile.ZipFile(REEFER_ZIP) as zf:
            with zf.open("reefer_release.csv") as raw:
                df = pd.read_csv(raw, sep=';', decimal=',', usecols=['EventTime', 'AvPowerCons'])

    df = df.dropna(subset=['EventTime', 'AvPowerCons'])
    df = df[(df['AvPowerCons'] >= 0)]
    df['EventTime'] = pd.to_datetime(df['EventTime']).dt.tz_localize(None)
    df['hour'] = df['EventTime'].dt.floor('h')
    
    hourly_kw = df.groupby('hour')['AvPowerCons'].sum() / 1000.0
    hourly_df = hourly_kw.reset_index().rename(columns={'hour': 'timestamp', 'AvPowerCons': 'target_kw'})
    return hourly_df.sort_values('timestamp').reset_index(drop=True)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    min_ts = df['timestamp'].min()
    max_ts = df['timestamp'].max()
    full_range = pd.date_range(min_ts, max_ts, freq='h')
    df_full = pd.DataFrame({'timestamp': full_range})
    df_full = df_full.merge(df, on='timestamp', how='left')
    
    df_full['target_kw'] = df_full['target_kw'].ffill()
    df_full['hour'] = df_full['timestamp'].dt.hour
    df_full['dayofweek'] = df_full['timestamp'].dt.dayofweek
    df_full['month'] = df_full['timestamp'].dt.month
    df_full['is_weekend'] = df_full['dayofweek'].isin([5, 6]).astype(int)
    
    df_full['load_t24'] = df_full['target_kw'].shift(24)
    df_full['load_t168'] = df_full['target_kw'].shift(168)
    df_full['rolling_mean_t168_24h'] = df_full['load_t168'].rolling(window=24, min_periods=1).mean()
    
    return df_full

def generate():
    load_df = aggregate_hourly_load()
    features_df = build_features(load_df)
    
    train_df = features_df.dropna().copy()
    feature_cols = ['hour', 'dayofweek', 'month', 'is_weekend', 'load_t24', 'load_t168', 'rolling_mean_t168_24h']
    
    X_train = train_df[feature_cols]
    y_train = train_df['target_kw']
    
    print("Training ML Models...")
    point_model = HistGradientBoostingRegressor(loss='absolute_error', random_state=42, max_iter=200)
    point_model.fit(X_train, y_train)
    
    p90_model = HistGradientBoostingRegressor(loss='quantile', quantile=0.90, random_state=42, max_iter=200)
    p90_model.fit(X_train, y_train)
    
    # Generate exactly 336 hours (14 days constraints)
    print("Synthesizing 2-Week Forecast Horizon...")
    target_hours = pd.date_range(start='2026-01-01', periods=336, freq='h')
    
    target_df = pd.DataFrame({'timestamp': target_hours})
    
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
    
    overall_mean = load_df['target_kw'].mean()
    target_features['load_t24'] = target_features['load_t24'].fillna(overall_mean)
    target_features['load_t168'] = target_features['load_t168'].fillna(overall_mean)
    target_features['rolling_mean_t168_24h'] = target_features['rolling_mean_t168_24h'].fillna(overall_mean)
    
    X_test = target_features[feature_cols]
    point_preds = point_model.predict(X_test)
    p90_preds = p90_model.predict(X_test)
    
    target_features['pred_power_kw'] = np.maximum(0, point_preds)
    target_features['pred_p90_kw'] = np.maximum(target_features['pred_power_kw'], np.maximum(0, p90_preds))
    
    # Calculate exactly 364 days ago
    target_features['timestamp_lastyear'] = target_features['timestamp'] - pd.Timedelta(days=364)
    
    merged = target_features.merge(load_df, left_on='timestamp_lastyear', right_on='timestamp', how='left', suffixes=('', '_y'))
    merged['history_lastyear_kw'] = merged['target_kw_y']
    merged['history_lastyear_kw'] = merged['history_lastyear_kw'].fillna(overall_mean)
    
    # Format and save
    predictions = []
    for row in merged.itertuples():
        ts_str = row.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        predictions.append({
            "timestamp_utc": ts_str,
            "pred_power_kw": round(row.pred_power_kw, 6),
            "pred_p90_kw": round(row.pred_p90_kw, 6),
            "history_lastyear_kw": round(row.history_lastyear_kw, 6)
        })
        
    print(f"Pushing {len(predictions)} extended targets to the dashboard...")
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "pred_power_kw", "pred_p90_kw", "history_lastyear_kw"])
        writer.writeheader()
        writer.writerows(predictions)

if __name__ == '__main__':
    generate()
