import pandas as pd

def generate():
    print("Loading predictions...")
    preds = pd.read_csv('predictions.csv')
    preds['timestamp'] = pd.to_datetime(preds['timestamp_utc']).dt.tz_localize(None)

    print("Loading historical data from reefer_release.csv...")
    df = pd.read_csv('reefer_release.csv', sep=';', decimal=',', usecols=['EventTime', 'AvPowerCons'])
    df = df.dropna(subset=['EventTime', 'AvPowerCons'])
    df = df[(df['AvPowerCons'] >= 0)]
    df['EventTime'] = pd.to_datetime(df['EventTime']).dt.tz_localize(None)
    df['hour'] = df['EventTime'].dt.floor('h')
    
    print("Aggregating...")
    hourly_kw = df.groupby('hour')['AvPowerCons'].sum() / 1000.0
    hourly_df = hourly_kw.reset_index().rename(columns={'hour': 'timestamp', 'AvPowerCons': 'history_kw'})

    # 364 days ago ensures we align perfectly to the same day of the week as last year!
    preds['timestamp_lastyear'] = preds['timestamp'] - pd.Timedelta(days=364) 

    print("Merging...")
    merged = preds.merge(hourly_df, left_on='timestamp_lastyear', right_on='timestamp', how='left', suffixes=('', '_y'))
    merged['history_lastyear_kw'] = merged['history_kw']

    if merged['history_lastyear_kw'].isnull().any():
        print("Some fallback filling executed for missing historical slots...")
        fallback_mean = hourly_df['history_kw'].mean()
        merged['history_lastyear_kw'] = merged['history_lastyear_kw'].fillna(fallback_mean)

    out_cols = ['timestamp_utc', 'pred_power_kw', 'pred_p90_kw', 'history_lastyear_kw']
    
    # Save to the dashboard folder
    merged[out_cols].to_csv('dashboard/dashboard_data.csv', index=False)
    print("Successfully generated dashboard/dashboard_data.csv")

if __name__ == '__main__':
    generate()
