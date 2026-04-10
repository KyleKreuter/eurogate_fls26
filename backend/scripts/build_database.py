import pandas as pd
import sqlite3
import zipfile
from pathlib import Path
import os

PUBLIC_DIR = Path('/Users/rkohlbach/Downloads/participant_package_lokal')
REEFER_CSV_PATH = PUBLIC_DIR / 'reefer_release.csv'
REEFER_ZIP_PATH = PUBLIC_DIR / 'reefer_release.zip'
DB_PATH = PUBLIC_DIR / 'dashboard' / 'reefer.db'

# All columns from the source CSV
ALL_COLS = [
    'container_visit_uuid',
    'customer_uuid',
    'container_uuid',
    'HardwareType',
    'EventTime',
    'AvPowerCons',
    'TtlEnergyConsHour',
    'TtlEnergyCons',
    'TemperatureSetPoint',
    'TemperatureAmbient',
    'TemperatureReturn',
    'RemperatureSupply',   # note: typo in source data, keep as-is
    'ContainerSize',
    'stack_tier',
]

def setup_database():
    print(f"Creating SQL database at {DB_PATH}")

    if DB_PATH.exists():
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)

    print("Loading all columns from 3.7M row CSV...")
    if REEFER_CSV_PATH.exists():
        df = pd.read_csv(REEFER_CSV_PATH, sep=';', decimal=',', usecols=ALL_COLS)
    else:
        with zipfile.ZipFile(REEFER_ZIP_PATH) as zf:
            with zf.open("reefer_release.csv") as raw:
                df = pd.read_csv(raw, sep=';', decimal=',', usecols=ALL_COLS)

    df['EventTime'] = pd.to_datetime(df['EventTime'])
    df = df.sort_values(['container_uuid', 'EventTime'])

    print("Computing per-visit statistics using native container_visit_uuid...")
    # The data already has a visit UUID — use it directly
    visit_stats = (
        df.groupby(['container_uuid', 'container_visit_uuid'])
        .agg(
            visit_start=('EventTime', 'min'),
            visit_end=('EventTime', 'max'),
            reading_count=('EventTime', 'count'),
            hardware_type=('HardwareType', 'first'),
            container_size=('ContainerSize', 'first'),
            avg_power_kw=('AvPowerCons', lambda x: round(x.mean() / 1000.0, 2)),
        )
        .reset_index()
    )
    visit_stats['duration_hours'] = (
        (visit_stats['visit_end'] - visit_stats['visit_start'])
        .dt.total_seconds() / 3600
    ).round(1)
    visit_stats['visit_start'] = visit_stats['visit_start'].dt.strftime('%Y-%m-%d %H:%M:%S')
    visit_stats['visit_end']   = visit_stats['visit_end'].dt.strftime('%Y-%m-%d %H:%M:%S')

    print("Computing per-container aggregate stats...")
    container_stats = (
        visit_stats.groupby('container_uuid')
        .agg(
            num_visits=('container_visit_uuid', 'count'),
            total_connected_hours=('duration_hours', 'sum'),
            avg_visit_hours=('duration_hours', 'mean'),
            last_visit_start=('visit_start', 'max'),
            last_visit_end=('visit_end', 'max'),
        )
        .reset_index()
    )
    container_stats['total_connected_hours'] = container_stats['total_connected_hours'].round(1)
    container_stats['avg_visit_hours']       = container_stats['avg_visit_hours'].round(1)

    print("Writing events table to SQLite (all columns)...")
    df['EventTime'] = df['EventTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.to_sql('events', conn, if_exists='replace', index=False, chunksize=200000)

    print("Writing visit_stats table...")
    visit_stats.to_sql('visit_stats', conn, if_exists='replace', index=False)

    print("Writing container_stats table...")
    container_stats.to_sql('container_stats', conn, if_exists='replace', index=False)

    print("Building indexes...")
    cursor = conn.cursor()
    cursor.execute('CREATE INDEX idx_container ON events(container_uuid)')
    cursor.execute('CREATE INDEX idx_visit_uuid ON events(container_visit_uuid)')
    cursor.execute('CREATE INDEX idx_vs_container ON visit_stats(container_uuid)')
    conn.commit()
    conn.close()

    print("Done! Full-column database built with native visit UUIDs.")

if __name__ == '__main__':
    setup_database()
