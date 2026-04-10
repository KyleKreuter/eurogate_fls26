from __future__ import annotations

import math
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class SequenceForecastDataset(Dataset):
    """
    Returns:
      encoder_x:       [encoder_len, n_features]
      known_future_x:  [encoder_len + horizon, n_known_future_features]
      decoder_known_x: [horizon, n_known_future_features]
      timestamp_cont:  [encoder_len + horizon, 6]
      timestamp_index: [encoder_len + horizon]
      y:               [horizon]
    """

    def __init__(
        self,
        parquet_path: str,
        feature_cols: list[str],
        known_future_cols: list[str],
        target_col: str = "target_power_kw",
        encoder_len: int = 168,
        horizon: int = 24,
        start_ts: str | None = None,
        end_ts: str | None = None,
    ) -> None:
        df = pl.read_parquet(parquet_path).sort("timestamp_utc")

        if start_ts is not None:
            df = df.filter(pl.col("timestamp_utc") >= pl.lit(start_ts).str.to_datetime())
        if end_ts is not None:
            df = df.filter(pl.col("timestamp_utc") <= pl.lit(end_ts).str.to_datetime())

        # FIX: deduplicate cols that appear in both lists to avoid redundant model inputs
        known_future_set = set(known_future_cols)
        feature_cols = [c for c in feature_cols if c not in known_future_set]

        all_needed = set(feature_cols) | set(known_future_cols)
        fill_exprs = [pl.col(c).fill_null(0).alias(c) for c in all_needed]
        df = df.with_columns(fill_exprs)

        self.encoder_len = encoder_len
        self.horizon = horizon

        # FIX: precompute timestamp arrays as numpy for vectorised __getitem__
        ts_series = df["timestamp_utc"]
        hours = ts_series.dt.hour().to_numpy().astype(np.float32)
        weekdays = ts_series.dt.weekday().to_numpy().astype(np.float32)
        doys = ts_series.dt.ordinal_day().to_numpy().astype(np.float32)

        TWO_PI = 2.0 * math.pi
        # [N, 6]
        self._ts_cont = np.stack([
            np.sin(TWO_PI * hours / 24.0),
            np.cos(TWO_PI * hours / 24.0),
            np.sin(TWO_PI * weekdays / 7.0),
            np.cos(TWO_PI * weekdays / 7.0),
            np.sin(TWO_PI * doys / 366.0),
            np.cos(TWO_PI * doys / 366.0),
        ], axis=1).astype(np.float32)

        self._ts_index = hours.astype(np.int64)  # hour-of-day as discrete index [0, 23]

        self.x_all = df.select(feature_cols).to_numpy().astype(np.float32, copy=False)
        self.x_known = df.select(known_future_cols).to_numpy().astype(np.float32, copy=False)
        self.y = df[target_col].to_numpy().astype(np.float32, copy=False)

        self.valid_start = encoder_len
        self.valid_end = len(df) - horizon + 1

    def __len__(self) -> int:
        return max(0, self.valid_end - self.valid_start)

    def __getitem__(self, idx: int):
        i = idx + self.valid_start
        start = i - self.encoder_len
        end = i + self.horizon

        encoder_x = np.array(self.x_all[start:i], copy=True)           # [encoder_len, F]
        known_future_x = np.array(self.x_known[start:end], copy=True)  # [encoder_len + horizon, K]
        decoder_known_x = np.array(self.x_known[i:end], copy=True)     # [horizon, K]
        y = np.array(self.y[i:end], copy=True)                         # [horizon]

        # FIX: vectorised slice — no Python loop over timestamps
        timestamp_cont = np.array(self._ts_cont[start:end], copy=True)   # [encoder_len + horizon, 6]
        timestamp_index = np.array(self._ts_index[start:end], copy=True) # [encoder_len + horizon]

        return {
            "encoder_x": torch.from_numpy(encoder_x),
            "known_future_x": torch.from_numpy(known_future_x),
            "decoder_known_x": torch.from_numpy(decoder_known_x),
            "timestamp_cont": torch.from_numpy(timestamp_cont),
            "timestamp_index": torch.from_numpy(timestamp_index),
            "y": torch.from_numpy(y),
        }