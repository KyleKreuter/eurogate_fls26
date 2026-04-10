from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class FeatureConfig:
    reefer_csv: str
    output_dir: str
    target_col: str = "Energy"   # prefer Energy for hourly consumption framing
    time_col: str = "EventTime"


CFG = FeatureConfig(
    reefer_csv="participant_package/daten/reefer_release.csv",
    output_dir="outputs",
)


# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def add_calendar_features(df: pl.LazyFrame, ts_col: str) -> pl.LazyFrame:
    return df.with_columns([
        pl.col(ts_col).dt.hour().alias("hour"),
        pl.col(ts_col).dt.weekday().alias("weekday"),  # Mon=1..Sun=7 in Polars
        (pl.col(ts_col).dt.weekday().is_in([6, 7])).cast(pl.Int8).alias("is_weekend"),
        pl.col(ts_col).dt.month().alias("month"),
        pl.col(ts_col).dt.ordinal_day().alias("dayofyear"),
        (2 * np.pi * pl.col(ts_col).dt.hour() / 24).sin().alias("hour_sin"),
        (2 * np.pi * pl.col(ts_col).dt.hour() / 24).cos().alias("hour_cos"),
        (2 * np.pi * pl.col(ts_col).dt.weekday() / 7).sin().alias("weekday_sin"),
        (2 * np.pi * pl.col(ts_col).dt.weekday() / 7).cos().alias("weekday_cos"),
        (2 * np.pi * pl.col(ts_col).dt.ordinal_day() / 366).sin().alias("doy_sin"),
        (2 * np.pi * pl.col(ts_col).dt.ordinal_day() / 366).cos().alias("doy_cos"),
    ])


def add_setpoint_bins(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns([
        pl.when(pl.col("TemperatureSetPoint") <= -18)
        .then(pl.lit("deep_frozen"))
        .when(pl.col("TemperatureSetPoint") <= -5)
        .then(pl.lit("frozen"))
        .when(pl.col("TemperatureSetPoint") <= 5)
        .then(pl.lit("chilled"))
        .otherwise(pl.lit("other"))
        .alias("SetPointBin")
    ])


def safe_float_cols(df: pl.LazyFrame, cols: list[str]) -> pl.LazyFrame:
    exprs = []
    for c in cols:
        exprs.append(
            pl.col(c)
            .cast(pl.Utf8)
            .str.replace_all(",", ".")
            .cast(pl.Float32, strict=False)
            .alias(c)
        )
    return df.with_columns(exprs)


# ----------------------------
# Step 1: Build hourly reefer aggregates
# ----------------------------

def build_hourly_reefer_features(cfg: FeatureConfig) -> pl.DataFrame:
    lf = pl.scan_csv(
        cfg.reefer_csv,
        separator=";",
        infer_schema=False,
        ignore_errors=True,
    )

    schema_names = lf.collect_schema().names()
    rename_map = {}
    if cfg.time_col in schema_names:
        rename_map[cfg.time_col] = "timestamp_utc"
    if "container_visit_uuid" in schema_names:
        rename_map["container_visit_uuid"] = "ContainerVisitID"
    if "TtlEnergyConsHour" in schema_names:
        rename_map["TtlEnergyConsHour"] = "Energy"
    if "AvPowerCons" in schema_names:
        rename_map["AvPowerCons"] = "Power"
    if "RemperatureSupply" in schema_names:
        rename_map["RemperatureSupply"] = "TemperatureSupply"
    if rename_map:
        lf = lf.rename(rename_map)

    # keep only needed columns early
    keep_cols = [
        "timestamp_utc",
        "ContainerVisitID",
        "HardwareType",
        "Power",
        "Energy",
        "EnergyTotal",
        "TemperatureSetPoint",
        "TemperatureAmbient",
        "TemperatureReturn",
        "TemperatureSupply",
        "ContainerSize",
    ]
    existing = [c for c in keep_cols if c in lf.collect_schema().names()]
    lf = lf.select(existing)

    lf = lf.with_columns([
        pl.col("timestamp_utc").str.to_datetime(strict=False).alias("timestamp_utc")
    ])
    lf = lf.filter(pl.col("timestamp_utc").is_not_null())

    # cast numeric columns
    numeric_cols = [
        c for c in [
            "Power",
            "Energy",
            "EnergyTotal",
            "TemperatureSetPoint",
            "TemperatureAmbient",
            "TemperatureReturn",
            "TemperatureSupply",
        ] if c in existing
    ]
    lf = safe_float_cols(lf, numeric_cols)

    # truncate to hour just in case
    lf = lf.with_columns([
        pl.col("timestamp_utc").dt.truncate("1h").alias("timestamp_utc")
    ])

    # lightweight derived fields
    if "TemperatureReturn" in existing and "TemperatureSupply" in existing:
        lf = lf.with_columns([
            (pl.col("TemperatureReturn") - pl.col("TemperatureSupply")).alias("temp_delta_return_supply")
        ])

    if "TemperatureAmbient" in existing and "TemperatureSetPoint" in existing:
        lf = lf.with_columns([
            (pl.col("TemperatureAmbient") - pl.col("TemperatureSetPoint")).alias("temp_gap_ambient_setpoint")
        ])

    if "TemperatureReturn" in existing and "TemperatureSetPoint" in existing:
        lf = lf.with_columns([
            (pl.col("TemperatureReturn") - pl.col("TemperatureSetPoint")).alias("temp_gap_return_setpoint")
        ])

    if "TemperatureSupply" in existing and "TemperatureSetPoint" in existing:
        lf = lf.with_columns([
            (pl.col("TemperatureSupply") - pl.col("TemperatureSetPoint")).alias("temp_gap_supply_setpoint")
        ])

    lf = add_setpoint_bins(lf)

    # container-level visit age proxy: first timestamp per visit
    if "ContainerVisitID" in existing:
        visit_start = (
            lf.group_by("ContainerVisitID")
            .agg(pl.col("timestamp_utc").min().alias("visit_start_ts"))
        )
        lf = (
            lf.join(visit_start, on="ContainerVisitID", how="left")
            .with_columns([
                ((pl.col("timestamp_utc").cast(pl.Int64) - pl.col("visit_start_ts").cast(pl.Int64))
                 / 3_600_000_000).alias("visit_age_hours")
            ])
        )

    # hourly aggregate features
    agg_exprs = [
        pl.len().alias("row_count"),
    ]

    if "ContainerVisitID" in existing:
        agg_exprs += [
            pl.col("ContainerVisitID").n_unique().alias("active_container_count"),
            pl.col("visit_age_hours").mean().alias("visit_age_hours_mean"),
            pl.col("visit_age_hours").median().alias("visit_age_hours_median"),
        ]

    if "Energy" in existing:
        agg_exprs += [
            pl.col("Energy").sum().alias("target_energy_wh"),
            pl.col("Energy").mean().alias("energy_mean_wh"),
            pl.col("Energy").std().alias("energy_std_wh"),
            pl.col("Energy").max().alias("energy_max_wh"),
        ]

    if "Power" in existing:
        agg_exprs += [
            pl.col("Power").sum().alias("total_power_w"),
            pl.col("Power").mean().alias("power_mean_w"),
            pl.col("Power").std().alias("power_std_w"),
            pl.col("Power").max().alias("power_max_w"),
        ]

    for c in [
        "TemperatureSetPoint",
        "TemperatureAmbient",
        "TemperatureReturn",
        "TemperatureSupply",
        "temp_delta_return_supply",
        "temp_gap_ambient_setpoint",
        "temp_gap_return_setpoint",
        "temp_gap_supply_setpoint",
    ]:
        if c in lf.collect_schema().names():
            agg_exprs += [
                pl.col(c).mean().alias(f"{c}_mean"),
                pl.col(c).std().alias(f"{c}_std"),
                pl.col(c).max().alias(f"{c}_max"),
            ]

    hourly = lf.group_by("timestamp_utc").agg(agg_exprs).sort("timestamp_utc").collect()

    # HardwareType counts
    if "HardwareType" in existing:
        hw_counts = (
            lf.group_by(["timestamp_utc", "HardwareType"])
            .agg(pl.len().alias("cnt"))
            .with_columns([
                pl.concat_str([pl.lit("hw_"), pl.col("HardwareType").cast(pl.Utf8)]).alias("feature_name")
            ])
            .collect()
        )
        hw_counts = hw_counts.pivot(
            values="cnt",
            index="timestamp_utc",
            on="feature_name",
            aggregate_function="sum",
        ).sort("timestamp_utc")
        hourly = hourly.join(hw_counts, on="timestamp_utc", how="left")

    # ContainerSize counts
    if "ContainerSize" in existing:
        size_counts = (
            lf.group_by(["timestamp_utc", "ContainerSize"])
            .agg(pl.len().alias("cnt"))
            .with_columns([
                pl.concat_str([pl.lit("size_"), pl.col("ContainerSize").cast(pl.Utf8)]).alias("feature_name")
            ])
            .collect()
        )
        size_counts = size_counts.pivot(
            values="cnt",
            index="timestamp_utc",
            on="feature_name",
            aggregate_function="sum",
        ).sort("timestamp_utc")
        hourly = hourly.join(size_counts, on="timestamp_utc", how="left")

    # Setpoint bin counts
    setpoint_counts = (
        lf.group_by(["timestamp_utc", "SetPointBin"])
        .agg(pl.len().alias("cnt"))
        .with_columns([
            pl.concat_str([pl.lit("spbin_"), pl.col("SetPointBin")]).alias("feature_name")
        ])
        .collect()
    )
    setpoint_counts = setpoint_counts.pivot(
        values="cnt",
        index="timestamp_utc",
        on="feature_name",
        aggregate_function="sum",
    ).sort("timestamp_utc")
    hourly = hourly.join(setpoint_counts, on="timestamp_utc", how="left")

    # fill null count pivots with zero
    count_like_cols = [c for c in hourly.columns if c.startswith(("hw_", "size_", "spbin_"))]
    if count_like_cols:
        hourly = hourly.with_columns([pl.col(c).fill_null(0) for c in count_like_cols])

    return hourly


# ----------------------------
# Step 2: Create lagged features
# ----------------------------

def add_lags_and_rolls(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort("timestamp_utc")

    # calendar features
    lf = add_calendar_features(df.lazy(), "timestamp_utc")

    # lag target features based on aggregate energy
    target = "target_energy_wh"
    lag_hours = [24, 48, 72, 168]
    for lag in lag_hours:
        lf = lf.with_columns([
            pl.col(target).shift(lag).alias(f"{target}_lag_{lag}")
        ])

    # rolling features must end before prediction information boundary
    # these are still okay for supervised training when built from historical rows
    for win in [24, 48, 168]:
        lf = lf.with_columns([
            pl.col(target).shift(24).rolling_mean(win).alias(f"{target}_rollmean_{win}_ending_tminus24"),
            pl.col(target).shift(24).rolling_std(win).alias(f"{target}_rollstd_{win}_ending_tminus24"),
            pl.col(target).shift(24).rolling_max(win).alias(f"{target}_rollmax_{win}_ending_tminus24"),
        ])

    # lag weather
    weather_cols = [
        c for c in lf.collect_schema().names()
        if c not in {"timestamp_utc", target}
        and any(k in c.lower() for k in ["temp", "humidity", "wind", "precip", "pressure", "solar", "cloud"])
    ]
    for c in weather_cols:
        for lag in [24, 48, 168]:
            lf = lf.with_columns([
                pl.col(c).shift(lag).alias(f"{c}_lag_{lag}")
            ])

    # simple interactions
    cols = lf.collect_schema().names()
    if "active_container_count" in cols and "TemperatureAmbient_mean" in cols:
        lf = lf.with_columns([
            (pl.col("active_container_count") * pl.col("TemperatureAmbient_mean"))
            .alias("active_count_x_ambient_mean")
        ])

    if "active_container_count" in cols and "TemperatureSetPoint_mean" in cols:
        lf = lf.with_columns([
            (pl.col("active_container_count") * pl.col("TemperatureSetPoint_mean"))
            .alias("active_count_x_setpoint_mean")
        ])

    if "TemperatureAmbient_mean" in cols and "TemperatureSetPoint_mean" in cols:
        lf = lf.with_columns([
            (pl.col("TemperatureAmbient_mean") - pl.col("TemperatureSetPoint_mean"))
            .alias("ambient_minus_setpoint_mean")
        ])

    return lf.collect()


# ----------------------------
# Step 3: Save final feature table
# ----------------------------

def main() -> None:
    ensure_dir(CFG.output_dir)

    df = build_hourly_reefer_features(CFG)
    df = add_lags_and_rolls(df)

    # drop earliest rows where lag history is unavailable
    df = df.filter(pl.col("target_energy_wh_lag_168").is_not_null())

    out_path = Path(CFG.output_dir) / "features_hourly.parquet"
    df.write_parquet(out_path, compression="zstd")

    print(f"Saved feature table: {out_path}")
    print(df.shape)
    print(df.head(3))


if __name__ == "__main__":
    main()