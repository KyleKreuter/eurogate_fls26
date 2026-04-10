"""Full-feature hybrid stacker for the Reefer Peak Load Challenge.

This model combines multiple LightGBM experts and learns a feature-conditioned
meta-model on rolling out-of-fold predictions:

- Point experts:
        1) baseline-style LightGBM (minimal lag features)
        2) full-feature LightGBM
        3) peak-weighted full-feature LightGBM
- Quantile experts (P90):
        4) baseline-style quantile LightGBM (alpha=0.9)
        5) full-feature quantile LightGBM (alpha=0.9)

The meta-level uses full feature context plus expert outputs to predict:
- point forecast (`pred_power_kw`)
- p90 forecast (`pred_p90_kw`)

Finally, p90 is conformal-calibrated on a recent calibration window and enforced
to satisfy submission constraints.

Run:
        uv run python lightgbm/hybrid_full_feature.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Important import order: weather_external must be imported before baseline.
from weather_external import OPEN_METEO_VARIABLES, load_cth_weather

from baseline import (
    PEAK_QUANTILE,
    PROJECT_ROOT,
    REEFER_CSV,
    SUBMISSIONS_DIR,
    TARGET_COL,
    TARGET_CSV,
    load_hourly_total,
)

import lightgbm as lgb


OUT_PATH = SUBMISSIONS_DIR / "hybrid_full_feature.csv"

N_FOLDS = 5
VAL_HOURS = 24 * 14
MIN_TRAIN_HOURS = 24 * 90
CALIB_HOURS = 24 * 10

PEAK_WEIGHT_MULT = 2.2
META_PEAK_WEIGHT_MULT = 1.8
EXPERT_NUM_BOOST_ROUND = 500
META_NUM_BOOST_ROUND = 350
EXOG_NUM_BOOST_ROUND = 180
SEED = 42
HORIZON_HOURS = 24


def _lgb_params(
    objective: str,
    alpha: float | None = None,
    params_override: dict[str, float | int | str] | None = None,
) -> dict[str, float | int | str]:
    params: dict[str, float | int | str] = {
        "objective": objective,
        "metric": "mae" if objective != "quantile" else "quantile",
        "learning_rate": 0.04,
        "num_leaves": 63,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": SEED,
    }
    if alpha is not None:
        params["alpha"] = float(alpha)
    if params_override is not None:
        params.update(params_override)
    return params


def _train_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    objective: str,
    alpha: float | None = None,
    weight: np.ndarray | None = None,
    params_override: dict[str, float | int | str] | None = None,
    num_boost_round: int = EXPERT_NUM_BOOST_ROUND,
) -> lgb.Booster:
    dataset = lgb.Dataset(X, label=y, weight=weight)
    return lgb.train(
        _lgb_params(objective=objective, alpha=alpha, params_override=params_override),
        dataset,
        num_boost_round=num_boost_round,
    )


def _peak_weights(y: np.ndarray) -> np.ndarray:
    threshold = float(np.quantile(y, PEAK_QUANTILE))
    return np.where(y >= threshold, PEAK_WEIGHT_MULT, 1.0).astype(np.float32)


def _load_hourly_context_features(
    csv_path: str | pd.io.common.FilePath,
) -> pd.DataFrame:
    """Container-level hourly context aggregates from reefer_release.csv."""
    usecols = [
        "EventTime",
        "container_uuid",
        "container_visit_uuid",
        "HardwareType",
        "TemperatureSetPoint",
        "TemperatureAmbient",
        "TemperatureReturn",
        "RemperatureSupply",
        "ContainerSize",
        "stack_tier",
    ]

    raw = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        usecols=lambda c: c in set(usecols),
        low_memory=False,
    )
    raw["EventTime"] = pd.to_datetime(raw["EventTime"], utc=True)
    raw["ts"] = raw["EventTime"].dt.floor("1h")

    if "container_uuid" in raw.columns:
        container_key = raw["container_uuid"].astype("string")
    elif "container_visit_uuid" in raw.columns:
        container_key = raw["container_visit_uuid"].astype("string")
    else:
        container_key = pd.Series(pd.NA, index=raw.index, dtype="string")

    raw["container_key"] = container_key

    for col in [
        "TemperatureSetPoint",
        "TemperatureAmbient",
        "TemperatureReturn",
        "RemperatureSupply",
        "stack_tier",
    ]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    if "ContainerSize" in raw.columns:
        size_num = pd.to_numeric(raw["ContainerSize"], errors="coerce")
        raw["is_size_40"] = (size_num >= 40).astype(np.float32)
    else:
        raw["is_size_40"] = 0.0

    raw["temp_gap_ambient_setpoint"] = raw.get("TemperatureAmbient") - raw.get(
        "TemperatureSetPoint"
    )
    raw["temp_gap_return_supply"] = raw.get("TemperatureReturn") - raw.get(
        "RemperatureSupply"
    )

    out = (
        raw.groupby("ts", sort=True)
        .agg(
            container_count=("container_key", "nunique"),
            visit_count=("container_visit_uuid", "nunique"),
            hardware_type_count=("HardwareType", "nunique"),
            temp_setpoint_mean=("TemperatureSetPoint", "mean"),
            temp_ambient_mean=("TemperatureAmbient", "mean"),
            temp_return_mean=("TemperatureReturn", "mean"),
            temp_supply_mean=("RemperatureSupply", "mean"),
            temp_gap_ambient_setpoint_mean=("temp_gap_ambient_setpoint", "mean"),
            temp_gap_return_supply_mean=("temp_gap_return_supply", "mean"),
            stack_tier_mean=("stack_tier", "mean"),
            container_size_40_share=("is_size_40", "mean"),
        )
        .reset_index()
    )

    out["visit_per_container"] = out["visit_count"] / np.maximum(
        out["container_count"], 1.0
    )
    out["hw_per_container"] = out["hardware_type_count"] / np.maximum(
        out["container_count"], 1.0
    )
    out["temp_ambient_minus_supply_mean"] = (
        out["temp_ambient_mean"] - out["temp_supply_mean"]
    )
    out["temp_setpoint_minus_supply_mean"] = (
        out["temp_setpoint_mean"] - out["temp_supply_mean"]
    )

    for col in out.columns:
        if col == "ts":
            continue
        out[col] = (
            out[col]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
            .astype(np.float32)
        )

    return out


def _add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("ts").reset_index(drop=True)

    out["hour"] = out["ts"].dt.hour.astype("int16")
    out["dow"] = out["ts"].dt.dayofweek.astype("int16")
    out["month"] = out["ts"].dt.month.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")

    hour_angle = 2.0 * np.pi * out["hour"].astype(np.float32) / 24.0
    dow_angle = 2.0 * np.pi * out["dow"].astype(np.float32) / 7.0
    month_angle = 2.0 * np.pi * (out["month"].astype(np.float32) - 1.0) / 12.0
    out["hour_sin"] = np.sin(hour_angle)
    out["hour_cos"] = np.cos(hour_angle)
    out["dow_sin"] = np.sin(dow_angle)
    out["dow_cos"] = np.cos(dow_angle)
    out["month_sin"] = np.sin(month_angle)
    out["month_cos"] = np.cos(month_angle)

    lag_hours = [24, 48, 72, 168, 336]
    for lag in lag_hours:
        out[f"lag_{lag}h"] = out[TARGET_COL].shift(lag)

    shifted = out[TARGET_COL].shift(HORIZON_HOURS)
    for win in [24, 72, 168]:
        out[f"roll_mean_{win}h"] = shifted.rolling(
            win, min_periods=max(2, win // 4)
        ).mean()
    for win in [24, 72, 168]:
        out[f"roll_std_{win}h"] = shifted.rolling(
            win, min_periods=max(3, win // 4)
        ).std()
    out["roll_max_24h"] = shifted.rolling(24, min_periods=6).max()
    out["roll_min_24h"] = shifted.rolling(24, min_periods=6).min()

    out["delta_lag_24_168"] = out["lag_24h"] - out["lag_168h"]
    out["trend_24h"] = out["lag_24h"] - out["roll_mean_24h"]
    out["trend_168h"] = out["roll_mean_24h"] - out["roll_mean_168h"]
    out["lag_ratio_24_168"] = out["lag_24h"] / out["lag_168h"].replace(0.0, np.nan)

    return out


def _build_feature_table() -> pd.DataFrame:
    hourly = load_hourly_total(REEFER_CSV)
    feat = _add_core_features(hourly)

    weather = load_cth_weather()
    weather_cols = [c for c in OPEN_METEO_VARIABLES if c in weather.columns]
    feat = feat.merge(weather, on="ts", how="left")

    context = _load_hourly_context_features(REEFER_CSV)
    context_cols = [c for c in context.columns if c != "ts"]
    feat = feat.merge(context, on="ts", how="left")

    for col in weather_cols + context_cols:
        if col not in feat.columns:
            continue
        clean_col = feat[col].replace([np.inf, -np.inf], np.nan).ffill()
        feat[col] = clean_col
        feat[f"{col}_lag24h"] = clean_col.shift(24)
        feat[f"{col}_lag48h"] = clean_col.shift(48)
        feat[f"{col}_lag168h"] = clean_col.shift(168)

    if "container_count" in feat.columns:
        cnt_lag24 = feat["container_count"].shift(24)
        feat["power_per_container_lag24"] = feat["lag_24h"] / np.maximum(cnt_lag24, 1.0)
        feat["container_count_lag24"] = feat["container_count"].shift(24)
        feat["container_count_lag168"] = feat["container_count"].shift(168)
        feat["container_count_delta_24_168"] = (
            feat["container_count_lag24"] - feat["container_count_lag168"]
        )
    if "temp_ambient_mean" in feat.columns:
        feat["ambient_lag24h"] = feat["temp_ambient_mean"].shift(24)
    elif "temperature_2m" in feat.columns:
        feat["ambient_lag24h"] = feat["temperature_2m"].shift(24)

    if "temperature_2m" in feat.columns and "lag_24h" in feat.columns:
        feat["temp_x_lag24"] = feat["temperature_2m"] * feat["lag_24h"]

    return feat


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in {"ts", TARGET_COL}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _build_time_folds(n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(N_FOLDS):
        val_end = n_rows - (N_FOLDS - 1 - fold_idx) * VAL_HOURS
        val_start = val_end - VAL_HOURS
        if val_end > n_rows:
            continue
        if val_start < MIN_TRAIN_HOURS:
            continue
        train_idx = np.arange(0, val_start)
        val_idx = np.arange(val_start, val_end)
        folds.append((train_idx, val_idx))
    return folds


def _existing_features(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _add_predicted_exogenous_features(
    feat: pd.DataFrame,
    target_start: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    """Forecast selected exogenous signals and expose them as causal proxy features.

    For rows before target_start we create OOF predictions to avoid in-sample leakage.
    For target rows we train on full pre-target history and predict forward.
    """
    exog_targets = _existing_features(
        feat,
        OPEN_METEO_VARIABLES
        + [
            "container_count",
            "visit_count",
            "hardware_type_count",
            "temp_setpoint_mean",
            "temp_ambient_mean",
            "temp_return_mean",
            "temp_supply_mean",
            "temp_gap_ambient_setpoint_mean",
            "temp_gap_return_supply_mean",
            "visit_per_container",
            "hw_per_container",
        ],
    )

    time_cols = _existing_features(
        feat,
        [
            "hour",
            "dow",
            "month",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
        ],
    )
    power_hist_cols = _existing_features(
        feat,
        [
            "lag_24h",
            "lag_48h",
            "lag_72h",
            "lag_168h",
            "roll_mean_24h",
            "roll_mean_168h",
        ],
    )

    train_mask = feat["ts"] < target_start
    target_mask = feat["ts"] >= target_start

    pred_cols: list[str] = []
    for col in exog_targets:
        own_lag_cols = _existing_features(
            feat,
            [f"{col}_lag24h", f"{col}_lag48h", f"{col}_lag168h"],
        )
        model_cols = _dedupe_keep_order(time_cols + power_hist_cols + own_lag_cols)
        if len(model_cols) < 3:
            continue

        pre = feat.loc[train_mask, ["ts", col] + model_cols].copy()
        pre["row_idx"] = pre.index
        pre = pre.dropna(subset=[col] + model_cols).reset_index(drop=True)
        if len(pre) < (MIN_TRAIN_HOURS + VAL_HOURS):
            continue

        folds = _build_time_folds(len(pre))
        if not folds:
            continue

        oof_pred = np.full(len(pre), np.nan, dtype=np.float32)
        for tr_idx, val_idx in folds:
            m = _train_lgbm(
                pre.iloc[tr_idx][model_cols],
                pre.iloc[tr_idx][col],
                objective="regression_l1",
                num_boost_round=EXOG_NUM_BOOST_ROUND,
                params_override={
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "min_data_in_leaf": 24,
                },
            )
            oof_pred[val_idx] = m.predict(pre.iloc[val_idx][model_cols]).astype(
                np.float32
            )

        final_model = _train_lgbm(
            pre[model_cols],
            pre[col],
            objective="regression_l1",
            num_boost_round=EXOG_NUM_BOOST_ROUND,
            params_override={
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 24,
            },
        )

        pred_col = f"{col}_pred24h"
        feat[pred_col] = np.nan

        oof_mask = np.isfinite(oof_pred)
        if oof_mask.any():
            feat.loc[pre.loc[oof_mask, "row_idx"].to_numpy(), pred_col] = oof_pred[
                oof_mask
            ]

        tgt = feat.loc[target_mask, model_cols].copy()
        valid_tgt = tgt.notna().all(axis=1)
        if valid_tgt.any():
            feat.loc[tgt.index[valid_tgt], pred_col] = final_model.predict(
                tgt.loc[valid_tgt, model_cols]
            ).astype(np.float32)

        lag24_col = f"{col}_lag24h"
        lag168_col = f"{col}_lag168h"
        if lag24_col in feat.columns:
            feat[pred_col] = feat[pred_col].fillna(feat[lag24_col])
        if lag168_col in feat.columns:
            feat[pred_col] = feat[pred_col].fillna(feat[lag168_col])
        feat[pred_col] = feat[pred_col].ffill().fillna(0.0).astype(np.float32)

        pred_cols.append(pred_col)
        print(
            f"[hybrid_full] Exog-Forecast {col} -> {pred_col} "
            f"(train_rows={len(pre)}, folds={len(folds)})"
        )

    return feat, pred_cols


def _fit_experts(
    train_df: pd.DataFrame,
    base_features: list[str],
    full_features: list[str],
    productive_features: list[str],
    optimal_features: list[str],
) -> dict[str, lgb.Booster]:
    y = train_df[TARGET_COL]
    w_peak = _peak_weights(y.to_numpy())

    models = {
        "point_base": _train_lgbm(
            train_df[base_features],
            y,
            objective="regression_l1",
            num_boost_round=EXPERT_NUM_BOOST_ROUND,
            params_override={
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 20,
            },
        ),
        "point_full": _train_lgbm(
            train_df[full_features],
            y,
            objective="regression_l1",
            num_boost_round=EXPERT_NUM_BOOST_ROUND,
        ),
        "point_peak": _train_lgbm(
            train_df[full_features],
            y,
            objective="regression_l1",
            weight=w_peak,
            num_boost_round=EXPERT_NUM_BOOST_ROUND,
            params_override={
                "learning_rate": 0.05,
                "num_leaves": 47,
                "min_data_in_leaf": 24,
            },
        ),
        "point_productive": _train_lgbm(
            train_df[productive_features],
            y,
            objective="regression_l1",
            num_boost_round=EXPERT_NUM_BOOST_ROUND,
            params_override={
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 20,
            },
        ),
        "point_optimal": _train_lgbm(
            train_df[optimal_features],
            y,
            objective="regression_l1",
            num_boost_round=EXPERT_NUM_BOOST_ROUND,
            params_override={
                "learning_rate": 0.05,
                "num_leaves": 47,
                "min_data_in_leaf": 24,
            },
        ),
        "q_base": _train_lgbm(
            train_df[base_features],
            y,
            objective="quantile",
            alpha=0.9,
            num_boost_round=EXPERT_NUM_BOOST_ROUND,
            params_override={
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 20,
            },
        ),
        "q_full": _train_lgbm(
            train_df[full_features],
            y,
            objective="quantile",
            alpha=0.9,
            num_boost_round=EXPERT_NUM_BOOST_ROUND,
        ),
        "q_productive": _train_lgbm(
            train_df[productive_features],
            y,
            objective="quantile",
            alpha=0.9,
            num_boost_round=EXPERT_NUM_BOOST_ROUND,
            params_override={
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 20,
            },
        ),
        "q_optimal": _train_lgbm(
            train_df[optimal_features],
            y,
            objective="quantile",
            alpha=0.9,
            num_boost_round=EXPERT_NUM_BOOST_ROUND,
            params_override={
                "learning_rate": 0.05,
                "num_leaves": 47,
                "min_data_in_leaf": 24,
            },
        ),
    }
    return models


def _predict_experts(
    models: dict[str, lgb.Booster],
    df: pd.DataFrame,
    base_features: list[str],
    full_features: list[str],
    productive_features: list[str],
    optimal_features: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["pred_point_base"] = models["point_base"].predict(df[base_features])
    out["pred_point_full"] = models["point_full"].predict(df[full_features])
    out["pred_point_peak"] = models["point_peak"].predict(df[full_features])
    out["pred_point_productive"] = models["point_productive"].predict(
        df[productive_features]
    )
    out["pred_point_optimal"] = models["point_optimal"].predict(df[optimal_features])
    out["pred_q_base"] = models["q_base"].predict(df[base_features])
    out["pred_q_full"] = models["q_full"].predict(df[full_features])
    out["pred_q_productive"] = models["q_productive"].predict(df[productive_features])
    out["pred_q_optimal"] = models["q_optimal"].predict(df[optimal_features])
    return out


def _score_public(
    y_true: np.ndarray, y_point: np.ndarray, y_p90: np.ndarray
) -> dict[str, float]:
    mae_all = float(np.mean(np.abs(y_true - y_point)))
    threshold = float(np.quantile(y_true, PEAK_QUANTILE))
    mask = y_true >= threshold
    mae_peak = float(np.mean(np.abs(y_true[mask] - y_point[mask])))
    pinball = float(
        np.mean(np.maximum(0.9 * (y_true - y_p90), -0.1 * (y_true - y_p90)))
    )
    combined = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball
    return {
        "mae_all": mae_all,
        "mae_peak": mae_peak,
        "pinball_p90": pinball,
        "combined": combined,
    }


def main() -> None:
    print("[hybrid_full] Baue Features ...")
    feat = _build_feature_table()

    targets = pd.read_csv(TARGET_CSV)
    targets["ts"] = pd.to_datetime(targets["timestamp_utc"], utc=True)
    target_start = targets["ts"].min()

    feat, predicted_exogenous_features = _add_predicted_exogenous_features(
        feat,
        target_start,
    )

    all_features = _numeric_feature_columns(feat)
    time_features = _existing_features(
        feat,
        [
            "hour",
            "dow",
            "month",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
        ],
    )
    target_history_features = _existing_features(
        feat,
        [
            "lag_24h",
            "lag_48h",
            "lag_72h",
            "lag_168h",
            "lag_336h",
            "roll_mean_24h",
            "roll_mean_72h",
            "roll_mean_168h",
            "roll_std_24h",
            "roll_std_72h",
            "roll_std_168h",
            "roll_max_24h",
            "roll_min_24h",
            "delta_lag_24_168",
            "trend_24h",
            "trend_168h",
            "lag_ratio_24_168",
        ],
    )
    derived_causal_features = _existing_features(
        feat,
        [
            "power_per_container_lag24",
            "ambient_lag24h",
            "container_count_lag24",
            "container_count_lag168",
            "container_count_delta_24_168",
            "temp_x_lag24",
        ],
    )
    lagged_exogenous_features = sorted(
        c for c in all_features if c.endswith("_lag24h") or c.endswith("_lag168h")
    )

    base_features = list(time_features)
    productive_features = _dedupe_keep_order(
        time_features
        + _existing_features(
            feat,
            ["lag_24h", "lag_168h"]
            + [f"{c}_lag24h" for c in OPEN_METEO_VARIABLES]
            + [f"{c}_pred24h" for c in OPEN_METEO_VARIABLES],
        )
    )
    optimal_features = _dedupe_keep_order(
        time_features
        + _existing_features(
            feat,
            [
                "lag_24h",
                "lag_168h",
                "roll_mean_24h",
                "roll_mean_168h",
                "power_per_container_lag24",
                "ambient_lag24h",
                "container_count_lag24",
                "container_count_lag168",
                "container_count_delta_24_168",
            ],
        )
        + _existing_features(
            feat,
            [
                "container_count_pred24h",
                "temp_ambient_mean_pred24h",
                "temp_setpoint_mean_pred24h",
                "temperature_2m_pred24h",
            ],
        )
    )
    full_features = _dedupe_keep_order(
        time_features
        + target_history_features
        + derived_causal_features
        + lagged_exogenous_features
        + predicted_exogenous_features
    )

    if not base_features:
        raise RuntimeError("Keine Zeitmerkmale gefunden.")

    train_df = feat.loc[feat["ts"] < target_start].copy().reset_index(drop=True)
    train_df = train_df.dropna(subset=base_features).reset_index(drop=True)

    target_df = targets[["ts"]].merge(feat, on="ts", how="left")
    missing_target = int(target_df[full_features].isna().all(axis=1).sum())
    if missing_target:
        print(f"[hybrid_full] WARN: {missing_target} Target-Zeilen ohne Feature-Match")

    print(
        f"[hybrid_full] Training rows={len(train_df)}, target rows={len(target_df)}, "
        f"base_features={len(base_features)}, full_features={len(full_features)}, "
        f"productive_features={len(productive_features)}, "
        f"optimal_features={len(optimal_features)}"
    )

    folds = _build_time_folds(len(train_df))
    if not folds:
        raise RuntimeError(
            "Keine gueltigen Rolling-Folds gebaut. Passe VAL_HOURS/MIN_TRAIN_HOURS an."
        )
    print(f"[hybrid_full] Rolling OOF-Folds: {len(folds)}")

    oof = pd.DataFrame(
        {
            "pred_point_base": np.nan,
            "pred_point_full": np.nan,
            "pred_point_peak": np.nan,
            "pred_point_productive": np.nan,
            "pred_point_optimal": np.nan,
            "pred_q_base": np.nan,
            "pred_q_full": np.nan,
            "pred_q_productive": np.nan,
            "pred_q_optimal": np.nan,
        },
        index=train_df.index,
    )

    for fold_i, (tr_idx, val_idx) in enumerate(folds, start=1):
        fold_train = train_df.iloc[tr_idx]
        fold_val = train_df.iloc[val_idx]

        fold_models = _fit_experts(
            fold_train,
            base_features,
            full_features,
            productive_features,
            optimal_features,
        )
        fold_pred = _predict_experts(
            fold_models,
            fold_val,
            base_features,
            full_features,
            productive_features,
            optimal_features,
        )
        oof.iloc[val_idx] = fold_pred.to_numpy()

        print(
            f"[hybrid_full] Fold {fold_i}/{len(folds)}: "
            f"train={len(tr_idx)} val={len(val_idx)}"
        )

    expert_cols = [
        "pred_point_base",
        "pred_point_full",
        "pred_point_peak",
        "pred_point_productive",
        "pred_point_optimal",
        "pred_q_base",
        "pred_q_full",
        "pred_q_productive",
        "pred_q_optimal",
    ]
    stack_df = pd.concat(
        [train_df.reset_index(drop=True), oof.reset_index(drop=True)], axis=1
    )
    stack_df = stack_df.dropna(subset=expert_cols).reset_index(drop=True)
    stack_df = stack_df.sort_values("ts").reset_index(drop=True)

    if len(stack_df) < 500:
        raise RuntimeError("Zu wenige OOF-Zeilen fuer robustes Stacking.")

    calib_size = min(CALIB_HOURS, max(24, len(stack_df) // 5))
    if len(stack_df) <= calib_size + 200:
        calib_size = max(24, len(stack_df) // 4)

    meta_fit_df = stack_df.iloc[:-calib_size].copy()
    meta_calib_df = stack_df.iloc[-calib_size:].copy()

    point_meta_features = full_features + [
        "pred_point_base",
        "pred_point_full",
        "pred_point_peak",
        "pred_point_productive",
        "pred_point_optimal",
    ]
    q_meta_features = full_features + [
        "pred_q_base",
        "pred_q_full",
        "pred_q_productive",
        "pred_q_optimal",
        "pred_point_base",
        "pred_point_full",
        "pred_point_peak",
        "pred_point_productive",
        "pred_point_optimal",
    ]

    print(
        f"[hybrid_full] Meta-Training: fit={len(meta_fit_df)} rows, "
        f"calib={len(meta_calib_df)} rows"
    )

    y_meta = meta_fit_df[TARGET_COL].to_numpy()
    meta_peak_thr = float(np.quantile(y_meta, PEAK_QUANTILE))
    meta_weight = np.where(y_meta >= meta_peak_thr, META_PEAK_WEIGHT_MULT, 1.0).astype(
        np.float32
    )

    point_meta_model = _train_lgbm(
        meta_fit_df[point_meta_features],
        meta_fit_df[TARGET_COL],
        objective="regression_l1",
        weight=meta_weight,
        num_boost_round=META_NUM_BOOST_ROUND,
        params_override={
            "learning_rate": 0.04,
            "num_leaves": 31,
            "min_data_in_leaf": 24,
        },
    )
    q_meta_model = _train_lgbm(
        meta_fit_df[q_meta_features],
        meta_fit_df[TARGET_COL],
        objective="quantile",
        alpha=0.9,
        weight=meta_weight,
        num_boost_round=META_NUM_BOOST_ROUND,
        params_override={
            "learning_rate": 0.04,
            "num_leaves": 31,
            "min_data_in_leaf": 24,
        },
    )

    # Tune a post-meta point blend against anchor experts on a calibration split.
    tune_size = max(24, len(meta_calib_df) // 2)
    calib_tune_df = meta_calib_df.iloc[:tune_size].copy()
    calib_conf_df = meta_calib_df.iloc[tune_size:].copy()
    if len(calib_conf_df) < 24:
        calib_conf_df = meta_calib_df.copy()

    y_tune = calib_tune_df[TARGET_COL].to_numpy()
    point_tune_meta = point_meta_model.predict(calib_tune_df[point_meta_features])
    q_tune_meta = q_meta_model.predict(calib_tune_df[q_meta_features])

    anchor_cols = {
        "none": None,
        "base": "pred_point_base",
        "full": "pred_point_full",
        "peak": "pred_point_peak",
        "productive": "pred_point_productive",
        "optimal": "pred_point_optimal",
    }

    best = None
    for anchor_name, anchor_col in anchor_cols.items():
        if anchor_col is None:
            anchor_tune = point_tune_meta
        else:
            anchor_tune = calib_tune_df[anchor_col].to_numpy()

        for gamma in np.linspace(0.0, 0.70, 15):
            point_tune = (1.0 - gamma) * point_tune_meta + gamma * anchor_tune
            resid_tune = y_tune - point_tune
            for q_conf in [0.88, 0.90, 0.92]:
                offset_tune = float(np.quantile(resid_tune, q_conf))
                p90_tune = np.maximum.reduce(
                    [q_tune_meta, point_tune + offset_tune, point_tune]
                )
                s = _score_public(y_tune, point_tune, p90_tune)
                cand = (
                    s["combined"],
                    anchor_name,
                    float(gamma),
                    float(q_conf),
                    s,
                )
                if best is None or cand[0] < best[0]:
                    best = cand

    assert best is not None
    _, best_anchor, best_gamma, best_q_conf, best_metrics = best

    y_conf = calib_conf_df[TARGET_COL].to_numpy()
    point_conf_meta = point_meta_model.predict(calib_conf_df[point_meta_features])
    if best_anchor == "none":
        anchor_conf = point_conf_meta
    else:
        anchor_conf = calib_conf_df[anchor_cols[best_anchor]].to_numpy()
    point_conf = (1.0 - best_gamma) * point_conf_meta + best_gamma * anchor_conf
    conformal_offset = float(np.quantile(y_conf - point_conf, best_q_conf))
    calib_coverage = float(np.mean((point_conf + conformal_offset) >= y_conf))

    print(
        f"[hybrid_full] best blend: anchor={best_anchor}, gamma={best_gamma:.2f}, "
        f"q_conf={best_q_conf:.2f}, tune_combined={best_metrics['combined']:.2f}"
    )
    print(
        f"[hybrid_full] Conformal offset={conformal_offset:.2f} kW, "
        f"calib coverage={calib_coverage:.3f}"
    )

    print("[hybrid_full] Refit experts auf vollem Training ...")
    final_expert_models = _fit_experts(
        train_df,
        base_features,
        full_features,
        productive_features,
        optimal_features,
    )
    target_expert_pred = _predict_experts(
        final_expert_models,
        target_df,
        base_features,
        full_features,
        productive_features,
        optimal_features,
    )
    target_stack = pd.concat(
        [target_df.reset_index(drop=True), target_expert_pred.reset_index(drop=True)],
        axis=1,
    )

    pred_point_meta = point_meta_model.predict(target_stack[point_meta_features])
    if best_anchor == "none":
        anchor_target = pred_point_meta
    else:
        anchor_target = target_stack[anchor_cols[best_anchor]].to_numpy()
    pred_point = (1.0 - best_gamma) * pred_point_meta + best_gamma * anchor_target
    pred_p90_meta = q_meta_model.predict(target_stack[q_meta_features])
    pred_p90_floor = pred_point + conformal_offset

    pred_point = np.clip(pred_point, a_min=0.0, a_max=None)
    pred_p90 = np.maximum.reduce([pred_p90_meta, pred_p90_floor, pred_point])
    pred_p90 = np.clip(pred_p90, a_min=0.0, a_max=None)

    submission = pd.DataFrame(
        {
            "timestamp_utc": targets["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pred_power_kw": np.round(pred_point, 2),
            "pred_p90_kw": np.round(pred_p90, 2),
        }
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False)

    print(
        f"[hybrid_full] Submission geschrieben: "
        f"{OUT_PATH.relative_to(PROJECT_ROOT)} ({len(submission)} Zeilen)"
    )

    if target_df[TARGET_COL].notna().all():
        y_true = target_df[TARGET_COL].to_numpy()
        metrics = _score_public(y_true=y_true, y_point=pred_point, y_p90=pred_p90)
        print(
            "[hybrid_full] Public-window metrics: "
            f"mae_all={metrics['mae_all']:.2f}, "
            f"mae_peak={metrics['mae_peak']:.2f}, "
            f"pinball_p90={metrics['pinball_p90']:.2f}, "
            f"combined={metrics['combined']:.2f}"
        )


if __name__ == "__main__":
    main()
