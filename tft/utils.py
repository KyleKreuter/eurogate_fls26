from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


def time_split(
    df: pl.DataFrame,
    train_end_ts: str,
    valid_end_ts: str | None = None,
):
    train = df.filter(pl.col("timestamp_utc") <= pl.lit(train_end_ts).str.to_datetime())
    if valid_end_ts is None:
        valid = df.filter(pl.col("timestamp_utc") > pl.lit(train_end_ts).str.to_datetime())
    else:
        valid = df.filter(
            (pl.col("timestamp_utc") > pl.lit(train_end_ts).str.to_datetime()) &
            (pl.col("timestamp_utc") <= pl.lit(valid_end_ts).str.to_datetime())
        )
    return train, valid


def fit_standard_scaler(
    train_df: pl.DataFrame,
    feature_cols: list[str],
    out_json_path: str | Path,
) -> dict:
    stats = {}
    for c in feature_cols:
        arr = train_df[c].to_numpy()
        arr = arr.astype(np.float32, copy=False)

        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0

        stats[c] = {"mean": mean, "std": std}

    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f)

    return stats


def load_scaler(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_standard_scaler_inplace(
    df: pl.DataFrame,
    feature_cols: list[str],
    stats: dict,
) -> pl.DataFrame:
    exprs = []
    for c in feature_cols:
        mean = stats[c]["mean"]
        std = stats[c]["std"]
        exprs.append(((pl.col(c) - mean) / std).alias(c))
    return df.with_columns(exprs)


def select_numeric_feature_cols(df: pl.DataFrame, exclude: set[str]) -> list[str]:
    out = []
    for c, dtype in zip(df.columns, df.dtypes):
        if c in exclude:
            continue
        if dtype in (
            pl.Float32, pl.Float64,
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        ):
            out.append(c)
    return out
