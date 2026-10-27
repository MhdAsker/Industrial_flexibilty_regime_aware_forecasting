# Leakage-safe day-ahead deterministic baselines

## Forecast protocol

Forecasts are issued at **12:00 Europe/Berlin on D-1** for every physical 15-minute interval in local calendar day D. UTC is the unique internal key. A historical quarter-hour load is considered available only when its interval end is no later than issuance. Evaluation uses only `target_quality_primary` observations (coverage >=75%, 135/180), without imputing ground truth.

| split      |   forecast_days |   grid_targets |   eligible_targets |
|:-----------|----------------:|---------------:|-------------------:|
| test       |             366 |          35136 |              31339 |
| train      |             730 |          70080 |              69832 |
| validation |             365 |          35040 |              32676 |

Local-day lengths are preserved: 92 intervals: 4 days, 96 intervals: 1453 days, 100 intervals: 4 days. The prediction artifact retains local/UTC issue and target times and lead minutes.

## Leakage controls

All historical-load baselines call one centralized interval-end availability gate. B1 performs an adaptive timestamp lookup rather than a 96-row shift; after-noon target quarters cannot use the preceding day's after-noon load. B2 uses exact local calendar matching seven days earlier and leaves missing/DST-unavailable values absent. B3 uses only eligible prior weeks. Observed weather, measured PV, and prices are excluded. No train-fitted scaler or test-derived transformation is used.

## Data availability

The full target-day grid is forecast where a baseline input exists. Metrics report generated predictions, eligible evaluated targets, and evaluation fraction separately. Long outages remain represented by `segment_id_primary`.

## Distribution shift

| split      |    mean |    std |   median |    Q10 |    Q25 |     Q75 |     Q90 |     Q95 |     n |
|:-----------|--------:|-------:|---------:|-------:|-------:|--------:|--------:|--------:|------:|
| test       | 15.3644 | 8.9400 |  12.9243 | 5.0319 | 6.3058 | 21.9053 | 27.7089 | 32.4272 | 31339 |
| train      | 12.6475 | 8.3079 |  11.4149 | 4.4552 | 5.0391 | 19.1407 | 25.3449 | 28.1399 | 69832 |
| validation | 12.9539 | 8.4831 |  11.3417 | 3.9295 | 4.4679 | 19.0299 | 25.3459 | 29.2800 | 32676 |

Test mean load is 2.717 kW above TRAIN. The quarter-hour mean shift is positive throughout the local day and ranges from 1.883 to 3.473 kW; the largest local-time increase occurs near quarter 68. Every weekday/weekend mean is also higher, by 1.982 to 3.266 kW. The monthly shift is less uniform (-3.186 to 5.330 kW), with the strongest increase in month 8; December is not directly representative because measured 2024 load ends early in that month. Thus the increase is broad across hours and weekdays but seasonally concentrated. This is observed temporal/distribution shift, not causal concept drift.

## Baselines

- B0 persistence: latest primary-quality completed interval at issuance, held constant.
- B1 safe recent day: latest earlier local date with the same local quarter and completed by issuance.
- B2 weekly: exact same local quarter seven calendar days earlier; unavailable if absent.
- B3 recent-week profile: 4 prior weeks using the median.
- B4 calendar profile: TRAIN-only median grouped by month,day_of_week,quarter_hour_of_day.
- B5 baseline ensemble: median of B1, B2, and B3, requiring all three.

## Validation results

| model                  | split      |   n_predictions |   n_forecasts_generated |   n_targets_evaluated |   fraction_evaluated |    MAE |    RMSE |    bias |   WAPE |   normalized_MAE |   MAE_CI_low |   MAE_CI_high |   RMSE_CI_low |   RMSE_CI_high |   WAPE_CI_low |   WAPE_CI_high |
|:-----------------------|:-----------|----------------:|------------------------:|----------------------:|---------------------:|-------:|--------:|--------:|-------:|-----------------:|-------------:|--------------:|--------------:|---------------:|--------------:|---------------:|
| B5_baseline_ensemble   | validation |           33344 |                   33344 |                 31656 |               0.9688 | 2.3777 |  4.0249 | -0.1282 | 0.1853 |           0.1880 |       2.1885 |        2.5645 |        3.7355 |         4.3141 |        0.1705 |         0.1996 |
| B3_recent_week_profile | validation |           35040 |                   35040 |                 32676 |               1.0000 | 2.6438 |  4.2655 | -0.0474 | 0.2041 |           0.2090 |       2.4497 |        2.8331 |        4.0028 |         4.5460 |        0.1894 |         0.2203 |
| B2_weekly              | validation |           33344 |                   33344 |                 31656 |               0.9688 | 2.8544 |  4.8125 | -0.1021 | 0.2225 |           0.2257 |       2.6130 |        3.0828 |        4.4386 |         5.1536 |        0.2046 |         0.2406 |
| B4_calendar_profile    | validation |           35040 |                   35040 |                 32676 |               1.0000 | 3.0389 |  4.4924 | -0.3788 | 0.2346 |           0.2403 |       2.8283 |        3.2533 |        4.2066 |         4.7878 |        0.2189 |         0.2516 |
| B1_safe_recent_day     | validation |           35040 |                   35040 |                 32676 |               1.0000 | 4.7490 |  7.8119 | -0.0025 | 0.3666 |           0.3755 |       4.4248 |        5.0672 |        7.3777 |         8.2290 |        0.3383 |         0.3965 |
| B0_persistence         | validation |           35040 |                   35040 |                 32676 |               1.0000 | 9.5534 | 12.4873 |  6.5045 | 0.7375 |           0.7554 |       8.9719 |       10.1182 |       11.8733 |        13.1372 |        0.6918 |         0.7884 |

## Frozen baseline selection

B3, B4, and optional B5 were selected using 2023 only. The frozen configuration was written before constructing final TEST predictions. Selection rule details and candidate scores are preserved in `config/baseline_experiment.json`. No 2024 result changed a definition.

## Test results

| model                  | split   |   n_predictions |   n_forecasts_generated |   n_targets_evaluated |   fraction_evaluated |     MAE |    RMSE |    bias |   WAPE |   normalized_MAE |   MAE_CI_low |   MAE_CI_high |   RMSE_CI_low |   RMSE_CI_high |   WAPE_CI_low |   WAPE_CI_high |
|:-----------------------|:--------|----------------:|------------------------:|----------------------:|---------------------:|--------:|--------:|--------:|-------:|-----------------:|-------------:|--------------:|--------------:|---------------:|--------------:|---------------:|
| B5_baseline_ensemble   | test    |           31339 |                   31339 |                 30145 |               0.9619 |  2.5713 |  4.1518 | -0.1082 | 0.1648 |           0.2033 |       2.3786 |        2.7681 |        3.7976 |         4.4976 |        0.1508 |         0.1793 |
| B3_recent_week_profile | test    |           34973 |                   34973 |                 31339 |               1.0000 |  2.7096 |  4.2319 |  0.0032 | 0.1764 |           0.2142 |       2.5101 |        2.9119 |        3.9174 |         4.5614 |        0.1628 |         0.1909 |
| B2_weekly              | test    |           31339 |                   31339 |                 30145 |               0.9619 |  3.1387 |  5.0160 | -0.0803 | 0.2012 |           0.2482 |       2.8893 |        3.3903 |        4.5756 |         5.4309 |        0.1841 |         0.2180 |
| B4_calendar_profile    | test    |           35136 |                   35136 |                 31339 |               1.0000 |  3.3893 |  4.8741 | -2.3234 | 0.2206 |           0.2680 |       3.1736 |        3.6008 |        4.6008 |         5.1451 |        0.2090 |         0.2321 |
| B1_safe_recent_day     | test    |           35136 |                   35136 |                 31339 |               1.0000 |  5.1376 |  8.2195 |  0.0027 | 0.3344 |           0.4062 |       4.8205 |        5.4923 |        7.8078 |         8.6594 |        0.3088 |         0.3618 |
| B0_persistence         | test    |           35136 |                   35136 |                 31339 |               1.0000 | 10.0206 | 12.9067 |  6.9690 | 0.6522 |           0.7923 |       9.4636 |       10.5758 |       12.3794 |        13.4341 |        0.6131 |         0.6909 |

Negative bias means underprediction. The 2024 mean increase is preserved rather than rescaled away.

## Lead-time analysis

| model                  | split   | grouping       | group   |   n_predictions |   n_forecasts_generated |   n_targets_evaluated |   fraction_evaluated |     MAE |    RMSE |    bias |   WAPE |   normalized_MAE |
|:-----------------------|:--------|:---------------|:--------|----------------:|------------------------:|----------------------:|---------------------:|--------:|--------:|--------:|-------:|-----------------:|
| B0_persistence         | test    | lead_time_band | 12-18h  |            8784 |                    8784 |                  7810 |               1.0000 | 12.6621 | 15.1867 | 12.0082 | 1.2249 |           1.0012 |
| B0_persistence         | test    | lead_time_band | 18-24h  |            8784 |                    8784 |                  7849 |               1.0000 |  7.5536 | 10.2134 |  2.3953 | 0.3784 |           0.5972 |
| B0_persistence         | test    | lead_time_band | 24-30h  |            8784 |                    8784 |                  7837 |               1.0000 |  7.3383 |  9.8884 |  1.5000 | 0.3523 |           0.5802 |
| B0_persistence         | test    | lead_time_band | 30-37h  |            8784 |                    8784 |                  7843 |               1.0000 | 12.5392 | 15.2928 | 11.9929 | 1.2164 |           0.9914 |
| B1_safe_recent_day     | test    | lead_time_band | 12-18h  |            8784 |                    8784 |                  7810 |               1.0000 |  2.1058 |  3.9362 | -0.0063 | 0.2037 |           0.1665 |
| B1_safe_recent_day     | test    | lead_time_band | 18-24h  |            8784 |                    8784 |                  7849 |               1.0000 |  6.4862 |  9.4746 | -0.0104 | 0.3249 |           0.5128 |
| B1_safe_recent_day     | test    | lead_time_band | 24-30h  |            8784 |                    8784 |                  7837 |               1.0000 |  9.7885 | 12.2924 |  0.0127 | 0.4700 |           0.7739 |
| B1_safe_recent_day     | test    | lead_time_band | 30-37h  |            8784 |                    8784 |                  7843 |               1.0000 |  2.1597 |  3.7019 |  0.0150 | 0.2095 |           0.1708 |
| B2_weekly              | test    | lead_time_band | 12-18h  |            7810 |                    7810 |                  7486 |               0.9585 |  2.2487 |  3.7370 | -0.0170 | 0.2139 |           0.1778 |
| B2_weekly              | test    | lead_time_band | 18-24h  |            7849 |                    7849 |                  7567 |               0.9641 |  4.1935 |  6.3016 | -0.0915 | 0.2067 |           0.3316 |
| B2_weekly              | test    | lead_time_band | 24-30h  |            7837 |                    7837 |                  7541 |               0.9622 |  4.0357 |  6.0190 | -0.1440 | 0.1912 |           0.3191 |
| B2_weekly              | test    | lead_time_band | 30-37h  |            7843 |                    7843 |                  7551 |               0.9628 |  2.0682 |  3.2594 | -0.0683 | 0.1978 |           0.1635 |
| B3_recent_week_profile | test    | lead_time_band | 12-18h  |            8760 |                    8760 |                  7810 |               1.0000 |  1.8830 |  3.1406 | -0.0268 | 0.1822 |           0.1489 |
| B3_recent_week_profile | test    | lead_time_band | 18-24h  |            8742 |                    8742 |                  7849 |               1.0000 |  3.5086 |  5.1430 |  0.0892 | 0.1758 |           0.2774 |
| B3_recent_week_profile | test    | lead_time_band | 24-30h  |            8736 |                    8736 |                  7837 |               1.0000 |  3.5459 |  5.1398 |  0.0649 | 0.1702 |           0.2804 |
| B3_recent_week_profile | test    | lead_time_band | 30-37h  |            8735 |                    8735 |                  7843 |               1.0000 |  1.8974 |  2.9785 | -0.1145 | 0.1841 |           0.1500 |
| B4_calendar_profile    | test    | lead_time_band | 12-18h  |            8784 |                    8784 |                  7810 |               1.0000 |  2.8690 |  4.1251 | -2.4264 | 0.2775 |           0.2268 |
| B4_calendar_profile    | test    | lead_time_band | 18-24h  |            8784 |                    8784 |                  7849 |               1.0000 |  3.5705 |  5.1437 | -1.4593 | 0.1789 |           0.2823 |
| B4_calendar_profile    | test    | lead_time_band | 24-30h  |            8784 |                    8784 |                  7837 |               1.0000 |  4.2706 |  5.9919 | -2.8893 | 0.2050 |           0.3377 |
| B4_calendar_profile    | test    | lead_time_band | 30-37h  |            8784 |                    8784 |                  7843 |               1.0000 |  2.8453 |  3.9532 | -2.5201 | 0.2760 |           0.2250 |
| B5_baseline_ensemble   | test    | lead_time_band | 12-18h  |            7810 |                    7810 |                  7486 |               0.9585 |  1.8072 |  3.1518 | -0.0369 | 0.1719 |           0.1429 |
| B5_baseline_ensemble   | test    | lead_time_band | 18-24h  |            7849 |                    7849 |                  7567 |               0.9641 |  3.2976 |  5.0079 | -0.0819 | 0.1626 |           0.2607 |
| B5_baseline_ensemble   | test    | lead_time_band | 24-30h  |            7837 |                    7837 |                  7541 |               0.9622 |  3.4239 |  5.0722 | -0.1922 | 0.1622 |           0.2707 |
| B5_baseline_ensemble   | test    | lead_time_band | 30-37h  |            7843 |                    7843 |                  7551 |               0.9628 |  1.7494 |  2.8530 | -0.1213 | 0.1673 |           0.1383 |

## Calendar/operating-context analysis

Compact weekday/weekend, holiday, month, quarter-hour, TRAIN-load-magnitude, and long-versus-short contiguous-segment tables are stored separately. A long segment is defined as at least 96 eligible intervals (24 hours). These are diagnostic slices, not tuning inputs.

## Measurement-quality sensitivity

| model                  | split   | grouping     | group   |   n_predictions |   n_forecasts_generated |   n_targets_evaluated |   fraction_evaluated |     MAE |    RMSE |    bias |   WAPE |   normalized_MAE |
|:-----------------------|:--------|:-------------|:--------|----------------:|------------------------:|----------------------:|---------------------:|--------:|--------:|--------:|-------:|-----------------:|
| B0_persistence         | test    | quality_band | 75-80%  |           31339 |                   31339 |                 31339 |               1.0000 | 10.0206 | 12.9067 |  6.9690 | 0.6522 |           0.7923 |
| B1_safe_recent_day     | test    | quality_band | 75-80%  |           31339 |                   31339 |                 31339 |               1.0000 |  5.1376 |  8.2195 |  0.0027 | 0.3344 |           0.4062 |
| B2_weekly              | test    | quality_band | 75-80%  |           30145 |                   30145 |                 30145 |               0.9619 |  3.1387 |  5.0160 | -0.0803 | 0.2012 |           0.2482 |
| B3_recent_week_profile | test    | quality_band | 75-80%  |           31339 |                   31339 |                 31339 |               1.0000 |  2.7096 |  4.2319 |  0.0032 | 0.1764 |           0.2142 |
| B4_calendar_profile    | test    | quality_band | 75-80%  |           31339 |                   31339 |                 31339 |               1.0000 |  3.3893 |  4.8741 | -2.3234 | 0.2206 |           0.2680 |
| B5_baseline_ensemble   | test    | quality_band | 75-80%  |           30145 |                   30145 |                 30145 |               0.9619 |  2.5713 |  4.1518 | -0.1082 | 0.1648 |           0.2033 |

## Daily-block uncertainty

Confidence intervals above use 2,000 deterministic bootstrap resamples of whole local target days (seed 20241027), not independent quarter-hours.

| model_name             | split      |   median_daily_MAE |   P90_daily_MAE |
|:-----------------------|:-----------|-------------------:|----------------:|
| B0_persistence         | test       |            10.0099 |         16.7713 |
| B0_persistence         | validation |             9.0763 |         16.6352 |
| B1_safe_recent_day     | test       |             4.3736 |         10.0449 |
| B1_safe_recent_day     | validation |             4.0156 |          9.1937 |
| B2_weekly              | test       |             2.6281 |          5.2595 |
| B2_weekly              | validation |             2.5711 |          5.6731 |
| B3_recent_week_profile | test       |             2.4458 |          4.7826 |
| B3_recent_week_profile | validation |             2.4332 |          4.9017 |
| B4_calendar_profile    | test       |             3.0739 |          6.1143 |
| B4_calendar_profile    | validation |             2.5021 |          5.8837 |
| B5_baseline_ensemble   | test       |             2.2158 |          4.2920 |
| B5_baseline_ensemble   | validation |             2.1009 |          4.3600 |

## Pairwise baseline comparison

| model_a                | model_b                |   n_paired_days |   mean_MAE_a_minus_b_kw |   median_difference_kw |   fraction_days_a_better | magnitude   | consistency              |
|:-----------------------|:-----------------------|----------------:|------------------------:|-----------------------:|-------------------------:|:------------|:-------------------------|
| B0_persistence         | B1_safe_recent_day     |             330 |                  4.8744 |                 5.4698 |                   0.2212 | large       | directionally consistent |
| B0_persistence         | B2_weekly              |             320 |                  6.9794 |                 6.8605 |                   0.1156 | large       | directionally consistent |
| B0_persistence         | B3_recent_week_profile |             330 |                  7.3016 |                 7.0053 |                   0.0818 | large       | directionally consistent |
| B0_persistence         | B4_calendar_profile    |             330 |                  6.6236 |                 6.3147 |                   0.1242 | large       | directionally consistent |
| B0_persistence         | B5_baseline_ensemble   |             320 |                  7.5409 |                 7.2001 |                   0.0906 | large       | directionally consistent |
| B1_safe_recent_day     | B2_weekly              |             320 |                  2.0527 |                 1.4511 |                   0.3469 | large       | directionally consistent |
| B1_safe_recent_day     | B3_recent_week_profile |             330 |                  2.4271 |                 1.7821 |                   0.3061 | large       | directionally consistent |
| B1_safe_recent_day     | B4_calendar_profile    |             330 |                  1.7491 |                 1.2851 |                   0.3848 | large       | directionally consistent |
| B1_safe_recent_day     | B5_baseline_ensemble   |             320 |                  2.6142 |                 1.9972 |                   0.2625 | large       | directionally consistent |
| B2_weekly              | B3_recent_week_profile |             320 |                  0.4038 |                 0.2481 |                   0.3750 | moderate    | directionally consistent |
| B2_weekly              | B4_calendar_profile    |             320 |                 -0.3067 |                -0.1900 |                   0.5656 | moderate    | inconsistent             |
| B2_weekly              | B5_baseline_ensemble   |             320 |                  0.5614 |                 0.3078 |                   0.1906 | moderate    | directionally consistent |
| B3_recent_week_profile | B4_calendar_profile    |             330 |                 -0.6780 |                -0.4098 |                   0.6394 | moderate    | directionally consistent |
| B3_recent_week_profile | B5_baseline_ensemble   |             320 |                  0.1576 |                 0.0649 |                   0.3563 | small       | directionally consistent |
| B4_calendar_profile    | B5_baseline_ensemble   |             320 |                  0.8681 |                 0.5224 |                   0.3375 | moderate    | directionally consistent |

## Failure cases

Worst test days are recorded in `worst_test_days.csv`. Baselines can be unavailable when exact historical inputs fail quality/DST/calendar requirements; those values remain missing. B0 can react poorly to schedule changes, B2 to atypical prior weeks, and static profiles to the higher 2024 load regime.

## Implications for learned models

A later deterministic model must beat these frozen baselines without using target-time observed weather, future PV, or unverified price availability. Forecast issuance semantics, interval-end availability, chronological splits, day-block uncertainty, and the 2024 distribution shift must remain explicit. No learned or probabilistic model was trained here.
