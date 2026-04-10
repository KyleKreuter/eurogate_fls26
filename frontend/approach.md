# Reefer Peak Load Challenge - Model Approach

## Summary
The goal of this solution is to predict the aggregate hourly power demand of plugged-in reefer containers (`pred_power_kw`) and provide a robust data-driven 90th percentile upper-bound estimate (`pred_p90_kw`). To move beyond the naive moving average baseline, this solution implements a lightweight Machine Learning pipeline using `scikit-learn`'s `HistGradientBoostingRegressor`.

## Data Preparation & Anomaly Filtering
Upon analyzing the historical `reefer_release.csv`, several critical anomalies were discovered, including:
- Broken sensors registering ambient, supply, and return temperatures below `-50°C`.
- Aggregation errors where Single-Hour Energy Consumption (`TtlEnergyConsHour`) impossibly exceeded `1.5` Million Watt-hours.

Despite these broken metrics, the target metric itself (`AvPowerCons`) remained stable and clean. The pipeline extracts `EventTime` and `AvPowerCons`, sums the total power across all reefers logged per hour, converts it to Kilowatts (kW), and uses this robust metric as the modeling target.

## Feature Engineering
The pipeline constructs features entirely free of forward-looking data leakage to respect the 24-hour ahead forecasting rules:
- **Time/Calendar Features**: `hour`, `dayofweek`, `month`, and `is_weekend`. 
- **Lag Features**: Since reefer load is highly cyclical, we extract the power consumption from exactly 24 hours ago (`load_t24`), exactly 1 week ago (`load_t168`), and calculate the rolling 24-hour mean centered around the `t-168` lag (`rolling_mean_t168_24h`). 

## Modeling
Two separate models are trained on the extracted features using `HistGradientBoostingRegressor` (an implementation natively capable of handling any missing data inputs automatically):
1. **Point Forecast (`pred_power_kw`)**: Optimized against `loss='absolute_error'` to robustly minimize Mean Absolute Error (MAE) and naturally ignore outsized statistical noise.
2. **Quantile Forecast (`pred_p90_kw`)**: Optimized against `loss='quantile'` with parameter `quantile=0.90`. This completely removes the reliance on static multipliers (like `1.10x`) and instead trains the tree to systematically output the true 90th percentile bounds based on specific calendar interactions.

## Post-Processing
Predictions are post-processed to explicitly enforce the rules of the competition constraints:
- Non-negative clamping to 0.
- Floor bounding where `pred_p90_kw` is forced to `max(pred_p90_kw, pred_power_kw)`.

## Reproducibility
The approach is entirely consolidated into the `train_and_predict.py` script. The script dynamically searches for the challenge data environment folder to ensure it runs out-of-the-box on the organizer's computers to extract the private scoring set effortlessly.
