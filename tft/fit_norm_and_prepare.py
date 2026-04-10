from __future__ import annotations

from pathlib import Path

import polars as pl

from utils import (
    time_split,
    fit_standard_scaler,
    apply_standard_scaler_inplace,
    select_numeric_feature_cols,
)


PARQUET_PATH = "outputs/features_hourly.parquet"
OUT_DIR = Path("outputs")
TRAIN_END = "2024-10-31T23:00:00"
VALID_END = "2024-11-30T23:00:00"

TARGET_COL = "target_energy_wh"
EXCLUDE = {"timestamp_utc", TARGET_COL}


def main():
    df = pl.read_parquet(PARQUET_PATH)

    feature_cols = select_numeric_feature_cols(df, exclude=EXCLUDE)

    train_df, valid_df = time_split(df, train_end_ts=TRAIN_END, valid_end_ts=VALID_END)

    stats = fit_standard_scaler(
        train_df=train_df,
        feature_cols=feature_cols,
        out_json_path=OUT_DIR / "norm_stats.json",
    )

    train_scaled = apply_standard_scaler_inplace(train_df, feature_cols, stats)
    valid_scaled = apply_standard_scaler_inplace(valid_df, feature_cols, stats)

    train_scaled.write_parquet(OUT_DIR / "train_scaled.parquet", compression="zstd")
    valid_scaled.write_parquet(OUT_DIR / "valid_scaled.parquet", compression="zstd")

    print(f"Train shape: {train_scaled.shape}")
    print(f"Valid shape: {valid_scaled.shape}")
    print(f"Saved scaler to: {OUT_DIR / 'norm_stats.json'}")


if __name__ == "__main__":
    main()