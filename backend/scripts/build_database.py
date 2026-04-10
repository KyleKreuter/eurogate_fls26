import pandas as pd
import sqlite3
import zipfile
from pathlib import Path
import os
import argparse
import sys

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

def setup_database(source_csv: Path, output_db: Path):
    print(f"Creating SQL database at {output_db}")

    output_db.parent.mkdir(parents=True, exist_ok=True)

    if output_db.exists():
        os.remove(output_db)

    conn = sqlite3.connect(output_db)

    print("Loading all columns from 3.7M row CSV...")
    if source_csv.exists():
        df = pd.read_csv(source_csv, sep=';', decimal=',', usecols=ALL_COLS)
    else:
        zip_path = source_csv.with_suffix('.zip')
        with zipfile.ZipFile(zip_path) as zf:
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
    default_source_csv = (
        Path(__file__).resolve().parents[2]
        / "participant_package"
        / "daten"
        / "reefer_release.csv"
    )
    default_output_db = Path(__file__).resolve().parents[1] / "reefer.db"

    parser = argparse.ArgumentParser(
        description="Build the reefer SQLite database from raw telemetry CSV."
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path(os.environ.get("REEFER_SOURCE_CSV", default_source_csv)),
        help=(
            "Path to the raw reefer_release.csv source file. "
            "Overrides the REEFER_SOURCE_CSV environment variable. "
            f"Default: {default_source_csv}"
        ),
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path(os.environ.get("REEFER_DB_PATH", default_output_db)),
        help=(
            "Path where the SQLite reefer.db will be written. "
            "Overrides the REEFER_DB_PATH environment variable. "
            f"Default: {default_output_db}"
        ),
    )
    args = parser.parse_args()

    source_csv: Path = args.source_csv
    output_db: Path = args.output_db

    # Accept either the CSV directly or a sibling .zip as a fallback.
    if not source_csv.exists() and not source_csv.with_suffix('.zip').exists():
        print(
            f"Source CSV not found at {source_csv}. "
            "If using Git LFS, run 'git lfs pull' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    setup_database(source_csv, output_db)
