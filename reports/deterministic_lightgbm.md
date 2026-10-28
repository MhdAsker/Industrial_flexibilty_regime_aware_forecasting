# Deterministic leakage-safe LightGBM

## Research question

Can a leakage-safe learned model improve day-ahead industrial load forecasts beyond strong adaptive seasonal baselines?

## Operational protocol

Issuance is 12:00 Europe/Berlin on D-1 for all physical 15-minute intervals of local day D. UTC is the unique key. Metrics use only primary-quality targets. The authoritative availability gate is reused from `day_ahead_baselines.py`, and a historical interval is usable only after its interval end.

## Point-in-time feature design

Predictors comprise deterministic local calendar/lead time, canonical holiday and bridge-day fields, safe B0-B3 baseline components, same-quarter prior-week analogs, issuance-time rolling load state, and recent operating-level summaries. NaNs are preserved for LightGBM native handling. Target coverage/quality, observed target-time weather, measured PV, and price are excluded.

## Baselines

B3 is the full-coverage adaptive four-week median. B5 is the frozen median ensemble with lower coverage. Both are evaluated on exact common targets against the learned model.

## Development procedure

TRAIN 2021-2022 was used for fitting and 2023 only for ablation, compact hyperparameter/formulation selection, and early stopping. TEST was not supplied to development APIs. Selection used mean daily validation MAE with aggregate, holiday, and stability diagnostics.

## Hyperparameter selection

The machine-readable log contains four predefined ablations and eight compact search candidates—four parameter sets for each of direct and residual-to-B3 formulations. No broad sweep was run.

## Ablation study

| candidate           |    MAE |   RMSE |    bias |   WAPE |   validation_holiday_MAE |   best_iteration |
|:--------------------|-------:|-------:|--------:|-------:|-------------------------:|-----------------:|
| A_calendar          | 2.9908 | 4.2972 | -0.4690 | 0.2309 |                   6.8962 |               62 |
| B_calendar_analogs  | 2.3528 | 3.6841 | -0.3624 | 0.1816 |                   5.2803 |              145 |
| C_plus_recent_state | 2.3746 | 3.7238 | -0.1378 | 0.1833 |                   5.8301 |              249 |
| D_plus_holidays     | 2.2575 | 3.4824 | -0.3045 | 0.1743 |                   1.8059 |              172 |

## Validation results

| model                  | split      |   n_predictions |   n_forecasts_generated |   n_targets_evaluated |   fraction_evaluated |    MAE |   RMSE |    bias |   WAPE |   normalized_MAE |   MAE_CI_low |   MAE_CI_high |   RMSE_CI_low |   RMSE_CI_high |   WAPE_CI_low |   WAPE_CI_high |
|:-----------------------|:-----------|----------------:|------------------------:|----------------------:|---------------------:|-------:|-------:|--------:|-------:|-----------------:|-------------:|--------------:|--------------:|---------------:|--------------:|---------------:|
| LGBM_DIRECT            | validation |           32676 |                   32676 |                 32676 |               1.0000 | 2.2372 | 3.4546 | -0.3270 | 0.1727 |           0.1769 |       2.0852 |        2.3874 |        3.2310 |         3.6858 |        0.1620 |         0.1841 |
| B5_baseline_ensemble   | validation |           33344 |                   33344 |                 31656 |               0.9688 | 2.3777 | 4.0249 | -0.1282 | 0.1853 |           0.1880 |       2.1885 |        2.5645 |        3.7355 |         4.3141 |        0.1705 |         0.1996 |
| B3_recent_week_profile | validation |           35040 |                   35040 |                 32676 |               1.0000 | 2.6438 | 4.2655 | -0.0474 | 0.2041 |           0.2090 |       2.4497 |        2.8331 |        4.0028 |         4.5460 |        0.1894 |         0.2203 |

## Frozen model

Selected `LGBM_DIRECT` with `direct` formulation, `regression_l1` objective, and 135 boosting iterations. The frozen configuration was written before TEST scoring.

## Final 2024 TEST results

| model                  | split   |   n_predictions |   n_forecasts_generated |   n_targets_evaluated |   fraction_evaluated |    MAE |   RMSE |    bias |   WAPE |   normalized_MAE |   MAE_CI_low |   MAE_CI_high |   RMSE_CI_low |   RMSE_CI_high |   WAPE_CI_low |   WAPE_CI_high |
|:-----------------------|:--------|----------------:|------------------------:|----------------------:|---------------------:|-------:|-------:|--------:|-------:|-----------------:|-------------:|--------------:|--------------:|---------------:|--------------:|---------------:|
| LGBM_DIRECT            | test    |           35136 |                   35136 |                 31339 |               1.0000 | 2.3520 | 3.4659 | -0.4371 | 0.1531 |           0.1860 |       2.2241 |        2.4808 |        3.2954 |         3.6483 |        0.1449 |         0.1612 |
| B5_baseline_ensemble   | test    |           31339 |                   31339 |                 30145 |               0.9619 | 2.5713 | 4.1518 | -0.1082 | 0.1648 |           0.2033 |       2.3786 |        2.7681 |        3.7976 |         4.4976 |        0.1508 |         0.1793 |
| B3_recent_week_profile | test    |           34973 |                   34973 |                 31339 |               1.0000 | 2.7096 | 4.2319 |  0.0032 | 0.1764 |           0.2142 |       2.5101 |        2.9119 |        3.9174 |         4.5614 |        0.1628 |         0.1909 |

The final model was refit once on eligible 2021-2023 examples using the frozen feature list, parameters, objective, and boosting iteration count.

## B3/B5 comparison

| split      | learned_model   | baseline               |   n_common_targets |   n_paired_days |   learned_MAE_common |   baseline_MAE_common |   mean_daily_MAE_difference_kw |   median_daily_MAE_difference_kw |   difference_CI_low |   difference_CI_high |   fraction_days_learned_wins |
|:-----------|:----------------|:-----------------------|-------------------:|----------------:|---------------------:|----------------------:|-------------------------------:|---------------------------------:|--------------------:|---------------------:|-----------------------------:|
| validation | LGBM_DIRECT     | B3_recent_week_profile |              32676 |             345 |               2.2372 |                2.6438 |                        -0.3980 |                          -0.1918 |             -0.5399 |              -0.2617 |                       0.5971 |
| validation | LGBM_DIRECT     | B5_baseline_ensemble   |              31656 |             338 |               2.1986 |                2.3777 |                        -0.1833 |                          -0.0583 |             -0.3155 |              -0.0501 |                       0.5444 |
| test       | LGBM_DIRECT     | B3_recent_week_profile |              31339 |             330 |               2.3520 |                2.7096 |                        -0.3537 |                          -0.1753 |             -0.5089 |              -0.1958 |                       0.5970 |
| test       | LGBM_DIRECT     | B5_baseline_ensemble   |              30145 |             320 |               2.3509 |                2.5713 |                        -0.2171 |                          -0.0356 |             -0.3773 |              -0.0657 |                       0.5312 |

Negative paired differences favor LightGBM. Confidence intervals resample whole target days.

## Lead-time analysis

| model                  | split   | grouping       | group   |   n_predictions |   n_forecasts_generated |   n_targets_evaluated |   fraction_evaluated |    MAE |   RMSE |    bias |   WAPE |   normalized_MAE |
|:-----------------------|:--------|:---------------|:--------|----------------:|------------------------:|----------------------:|---------------------:|-------:|-------:|--------:|-------:|-----------------:|
| B3_recent_week_profile | test    | lead_time_band | 12-18h  |            8760 |                    8760 |                  7810 |               1.0000 | 1.8830 | 3.1406 | -0.0268 | 0.1822 |           0.1489 |
| B3_recent_week_profile | test    | lead_time_band | 18-24h  |            8742 |                    8742 |                  7849 |               1.0000 | 3.5086 | 5.1430 |  0.0892 | 0.1758 |           0.2774 |
| B3_recent_week_profile | test    | lead_time_band | 24-30h  |            8736 |                    8736 |                  7837 |               1.0000 | 3.5459 | 5.1398 |  0.0649 | 0.1702 |           0.2804 |
| B3_recent_week_profile | test    | lead_time_band | 30-37h  |            8735 |                    8735 |                  7843 |               1.0000 | 1.8974 | 2.9785 | -0.1145 | 0.1841 |           0.1500 |
| B5_baseline_ensemble   | test    | lead_time_band | 12-18h  |            7810 |                    7810 |                  7486 |               0.9585 | 1.8072 | 3.1518 | -0.0369 | 0.1719 |           0.1429 |
| B5_baseline_ensemble   | test    | lead_time_band | 18-24h  |            7849 |                    7849 |                  7567 |               0.9641 | 3.2976 | 5.0079 | -0.0819 | 0.1626 |           0.2607 |
| B5_baseline_ensemble   | test    | lead_time_band | 24-30h  |            7837 |                    7837 |                  7541 |               0.9622 | 3.4239 | 5.0722 | -0.1922 | 0.1622 |           0.2707 |
| B5_baseline_ensemble   | test    | lead_time_band | 30-37h  |            7843 |                    7843 |                  7551 |               0.9628 | 1.7494 | 2.8530 | -0.1213 | 0.1673 |           0.1383 |
| LGBM_DIRECT            | test    | lead_time_band | 12-18h  |            8784 |                    8784 |                  7810 |               1.0000 | 1.7223 | 2.5750 | -0.0471 | 0.1666 |           0.1362 |
| LGBM_DIRECT            | test    | lead_time_band | 18-24h  |            8784 |                    8784 |                  7849 |               1.0000 | 2.8285 | 3.9993 | -0.5605 | 0.1417 |           0.2236 |
| LGBM_DIRECT            | test    | lead_time_band | 24-30h  |            8784 |                    8784 |                  7837 |               1.0000 | 3.0707 | 4.2885 | -0.9433 | 0.1474 |           0.2428 |
| LGBM_DIRECT            | test    | lead_time_band | 30-37h  |            8784 |                    8784 |                  7843 |               1.0000 | 1.7839 | 2.6483 | -0.1960 | 0.1730 |           0.1410 |

## Holiday analysis

| model                  | split   | grouping   | group              |   n_predictions |   n_forecasts_generated |   n_targets_evaluated |   fraction_evaluated |    MAE |    RMSE |    bias |   WAPE |   normalized_MAE |
|:-----------------------|:--------|:-----------|:-------------------|----------------:|------------------------:|----------------------:|---------------------:|-------:|--------:|--------:|-------:|-----------------:|
| B3_recent_week_profile | test    | day_type   | bridge_day         |             384 |                     384 |                   288 |               1.0000 | 3.6023 |  4.4683 |  2.1500 | 0.2369 |           0.2848 |
| B3_recent_week_profile | test    | day_type   | day_after_holiday  |             672 |                     672 |                   575 |               1.0000 | 2.1387 |  3.3764 | -0.3031 | 0.1787 |           0.1691 |
| B3_recent_week_profile | test    | day_type   | day_before_holiday |             956 |                     956 |                   823 |               1.0000 | 3.0925 |  4.3249 | -1.8222 | 0.1925 |           0.2445 |
| B3_recent_week_profile | test    | day_type   | normal_weekday     |           22398 |                   22398 |                 20387 |               1.0000 | 2.9325 |  4.2753 | -0.2393 | 0.1608 |           0.2319 |
| B3_recent_week_profile | test    | day_type   | public_holiday     |            1152 |                    1152 |                   864 |               1.0000 | 8.2790 | 10.8944 |  7.4625 | 1.0450 |           0.6546 |
| B3_recent_week_profile | test    | day_type   | weekend            |            9411 |                    9411 |                  8402 |               1.0000 | 1.5669 |  2.6358 | -0.0492 | 0.1678 |           0.1239 |
| B5_baseline_ensemble   | test    | day_type   | bridge_day         |             287 |                     287 |                   287 |               0.9965 | 3.4327 |  4.3691 |  2.3189 | 0.2260 |           0.2714 |
| B5_baseline_ensemble   | test    | day_type   | day_after_holiday  |             480 |                     480 |                   479 |               0.8330 | 2.4118 |  3.6993 | -0.5198 | 0.1783 |           0.1907 |
| B5_baseline_ensemble   | test    | day_type   | day_before_holiday |             764 |                     764 |                   762 |               0.9259 | 2.9440 |  4.2122 | -1.5886 | 0.1794 |           0.2328 |
| B5_baseline_ensemble   | test    | day_type   | normal_weekday     |           20450 |                   20450 |                 19690 |               0.9658 | 2.6884 |  4.0814 | -0.5816 | 0.1455 |           0.2126 |
| B5_baseline_ensemble   | test    | day_type   | public_holiday     |             764 |                     764 |                   764 |               0.8843 | 9.6727 | 11.8827 |  8.9538 | 1.1534 |           0.7648 |
| B5_baseline_ensemble   | test    | day_type   | weekend            |            8594 |                    8594 |                  8163 |               0.9716 | 1.5685 |  2.6701 |  0.2626 | 0.1668 |           0.1240 |
| LGBM_DIRECT            | test    | day_type   | bridge_day         |             384 |                     384 |                   288 |               1.0000 | 2.3631 |  2.9033 |  1.4093 | 0.1554 |           0.1868 |
| LGBM_DIRECT            | test    | day_type   | day_after_holiday  |             672 |                     672 |                   575 |               1.0000 | 2.0395 |  2.9599 | -0.4322 | 0.1704 |           0.1613 |
| LGBM_DIRECT            | test    | day_type   | day_before_holiday |             956 |                     956 |                   823 |               1.0000 | 2.7520 |  3.8610 | -1.1295 | 0.1713 |           0.2176 |
| LGBM_DIRECT            | test    | day_type   | normal_weekday     |           22560 |                   22560 |                 20387 |               1.0000 | 2.6582 |  3.8210 | -0.8216 | 0.1458 |           0.2102 |
| LGBM_DIRECT            | test    | day_type   | public_holiday     |            1152 |                    1152 |                   864 |               1.0000 | 3.4636 |  4.3235 |  2.7958 | 0.4372 |           0.2739 |
| LGBM_DIRECT            | test    | day_type   | weekend            |            9412 |                    9412 |                  8402 |               1.0000 | 1.4763 |  2.2603 |  0.1678 | 0.1581 |           0.1167 |

Holiday categories are deterministic. Bridge day means a weekday between a public holiday and an adjacent weekend day. Small categories should not be overinterpreted.

## Distribution-shift analysis

| model                  | split   | grouping   |   group |   n_predictions |   n_forecasts_generated |   n_targets_evaluated |   fraction_evaluated |    MAE |   RMSE |    bias |   WAPE |   normalized_MAE |
|:-----------------------|:--------|:-----------|--------:|----------------:|------------------------:|----------------------:|---------------------:|-------:|-------:|--------:|-------:|-----------------:|
| B3_recent_week_profile | test    | month      |       1 |            2976 |                    2976 |                  2552 |               1.0000 | 2.2303 | 3.7264 | -0.6392 | 0.2194 |           0.1763 |
| B3_recent_week_profile | test    | month      |       2 |            2784 |                    2784 |                  2668 |               1.0000 | 2.0948 | 3.3831 | -1.2001 | 0.1690 |           0.1656 |
| B3_recent_week_profile | test    | month      |       3 |            2972 |                    2972 |                  2649 |               1.0000 | 2.2499 | 3.9705 | -0.1847 | 0.1779 |           0.1779 |
| B3_recent_week_profile | test    | month      |       4 |            2880 |                    2880 |                  2855 |               1.0000 | 3.6984 | 5.2298 | -0.3905 | 0.2680 |           0.2924 |
| B3_recent_week_profile | test    | month      |       5 |            2976 |                    2976 |                  2973 |               1.0000 | 3.5197 | 4.9924 | -1.3042 | 0.2146 |           0.2783 |
| B3_recent_week_profile | test    | month      |       6 |            2880 |                    2880 |                  2876 |               1.0000 | 2.7173 | 4.2741 | -1.3376 | 0.1453 |           0.2148 |
| B3_recent_week_profile | test    | month      |       7 |            2976 |                    2976 |                  2943 |               1.0000 | 1.8462 | 3.0380 | -0.1324 | 0.0910 |           0.1460 |
| B3_recent_week_profile | test    | month      |       8 |            2976 |                    2976 |                  2973 |               1.0000 | 2.0139 | 3.1841 | -0.3719 | 0.0964 |           0.1592 |
| B3_recent_week_profile | test    | month      |       9 |            2880 |                    2880 |                  2878 |               1.0000 | 3.3906 | 5.0721 |  2.7380 | 0.2036 |           0.2681 |
| B3_recent_week_profile | test    | month      |      10 |            2980 |                    2980 |                  2970 |               1.0000 | 2.8569 | 4.3951 |  0.9499 | 0.1967 |           0.2259 |
| B3_recent_week_profile | test    | month      |      11 |            2880 |                    2880 |                  2876 |               1.0000 | 3.1494 | 4.5590 |  1.7278 | 0.2707 |           0.2490 |
| B3_recent_week_profile | test    | month      |      12 |            2813 |                    2813 |                   126 |               1.0000 | 0.9192 | 1.5625 |  0.6982 | 0.1590 |           0.0727 |
| LGBM_DIRECT            | test    | month      |       1 |            2976 |                    2976 |                  2552 |               1.0000 | 1.7529 | 2.5507 | -0.1124 | 0.1725 |           0.1386 |
| LGBM_DIRECT            | test    | month      |       2 |            2784 |                    2784 |                  2668 |               1.0000 | 1.7150 | 2.7568 | -0.6097 | 0.1384 |           0.1356 |
| LGBM_DIRECT            | test    | month      |       3 |            2972 |                    2972 |                  2649 |               1.0000 | 2.2350 | 3.2162 |  0.2464 | 0.1768 |           0.1767 |
| LGBM_DIRECT            | test    | month      |       4 |            2880 |                    2880 |                  2855 |               1.0000 | 3.0471 | 4.1360 | -0.0703 | 0.2208 |           0.2409 |
| LGBM_DIRECT            | test    | month      |       5 |            2976 |                    2976 |                  2973 |               1.0000 | 1.9918 | 2.8099 | -0.5643 | 0.1215 |           0.1575 |
| LGBM_DIRECT            | test    | month      |       6 |            2880 |                    2880 |                  2876 |               1.0000 | 2.4393 | 3.6952 | -1.3159 | 0.1304 |           0.1929 |
| LGBM_DIRECT            | test    | month      |       7 |            2976 |                    2976 |                  2943 |               1.0000 | 2.1719 | 3.4792 | -1.5880 | 0.1070 |           0.1717 |
| LGBM_DIRECT            | test    | month      |       8 |            2976 |                    2976 |                  2973 |               1.0000 | 2.8612 | 4.5836 | -2.3030 | 0.1370 |           0.2262 |
| LGBM_DIRECT            | test    | month      |       9 |            2880 |                    2880 |                  2878 |               1.0000 | 2.2991 | 3.0974 | -0.0772 | 0.1381 |           0.1818 |
| LGBM_DIRECT            | test    | month      |      10 |            2980 |                    2980 |                  2970 |               1.0000 | 2.6272 | 3.4421 |  0.8572 | 0.1808 |           0.2077 |
| LGBM_DIRECT            | test    | month      |      11 |            2880 |                    2880 |                  2876 |               1.0000 | 2.6625 | 3.7005 |  0.8054 | 0.2289 |           0.2105 |
| LGBM_DIRECT            | test    | month      |      12 |            2976 |                    2976 |                   126 |               1.0000 | 1.0068 | 1.6188 |  0.8489 | 0.1741 |           0.0796 |

The 2024 regime increase is retained. Recent-level features allow adaptation without target-time information; monthly bias is reported rather than normalized away.

## High-load performance

| model                  | split   | grouping       | group   |   n_predictions |   n_forecasts_generated |   n_targets_evaluated |   fraction_evaluated |    MAE |   RMSE |    bias |   WAPE |   normalized_MAE |
|:-----------------------|:--------|:---------------|:--------|----------------:|------------------------:|----------------------:|---------------------:|-------:|-------:|--------:|-------:|-----------------:|
| B3_recent_week_profile | test    | load_magnitude | Q4_high |           10200 |                   10200 |                 10177 |               1.0000 | 3.7955 | 5.0731 | -1.2222 | 0.1438 |           0.3001 |
| B5_baseline_ensemble   | test    | load_magnitude | Q4_high |           10110 |                   10110 |                 10087 |               0.9912 | 3.5839 | 4.9401 | -1.5212 | 0.1356 |           0.2834 |
| LGBM_DIRECT            | test    | load_magnitude | Q4_high |           10200 |                   10200 |                 10177 |               1.0000 | 3.4064 | 4.5807 | -2.1910 | 0.1290 |           0.2693 |

Magnitude bins use TRAIN-only quartiles. Weekday/weekend results:

| model       | split   | grouping   | group   |   n_predictions |   n_forecasts_generated |   n_targets_evaluated |   fraction_evaluated |    MAE |   RMSE |    bias |   WAPE |   normalized_MAE |
|:------------|:--------|:-----------|:--------|----------------:|------------------------:|----------------------:|---------------------:|-------:|-------:|--------:|-------:|-----------------:|
| LGBM_DIRECT | test    | is_weekend | False   |           25152 |                   25152 |                 22365 |               1.0000 | 2.6882 | 3.8349 | -0.6790 | 0.1507 |           0.2125 |
| LGBM_DIRECT | test    | is_weekend | True    |            9984 |                    9984 |                  8974 |               1.0000 | 1.5141 | 2.3020 |  0.1658 | 0.1646 |           0.1197 |

## Feature importance

| feature                        |   gain_importance |   split_importance |   permutation_MAE_increase_validation |
|:-------------------------------|------------------:|-------------------:|--------------------------------------:|
| same_quarter_prev4w_mean       |       719043.3288 |                 56 |                                0.6668 |
| baseline_B3_kw                 |       424697.8612 |                 95 |                                0.9237 |
| baseline_B2_kw                 |       415174.1238 |                 43 |                                0.7808 |
| same_quarter_prev8w_median     |        96344.8247 |                 43 |                                0.3789 |
| baseline_B1_kw                 |        93811.8081 |                123 |                                0.1931 |
| cos_year                       |        53196.4819 |                125 |                                0.0871 |
| min_load_last_24h              |        43369.3407 |                135 |                                0.0677 |
| is_public_holiday              |        36134.3359 |                 77 |                                0.1818 |
| sin_year                       |        28761.5357 |                 80 |                                0.0056 |
| recent_level_ratio_7d_28d      |        27368.2119 |                 56 |                                0.0006 |
| same_quarter_prev4w_std        |        23487.4974 |                 23 |                                0.0082 |
| recent_level_difference_7d_28d |        22905.2511 |                 53 |                                0.0005 |
| cos_hour                       |        21754.4866 |                 51 |                                0.1291 |
| sin_week                       |        21709.1053 |                 83 |                                0.0245 |
| recent_7d_median               |        20768.6635 |                 47 |                                0.1015 |
| recent_7d_mean                 |        18936.9743 |                 39 |                                0.0409 |
| day_of_week                    |        15722.1983 |                 46 |                                0.0335 |
| mean_load_last_12h             |        15461.2080 |                 55 |                                0.0080 |
| recent_28d_std                 |        14913.7891 |                 59 |                               -0.0026 |
| same_quarter_prev8w_mean       |        13466.1647 |                  8 |                                0.0778 |

Gain/split importance comes from the final refit. Permutation MAE increases use the untouched development validation predictions and do not drive selection.

## Failure cases

Deterministic representative learned-model dates are strong: 2024-05-19, median: 2024-05-28, poor: 2024-11-27. The difficult holiday is 2024-10-03, and the highest-mean-load day is 2024-08-01. Errors are described by calendar and load context only; no undocumented operational cause is inferred.

## Limitations

No point-in-time weather forecast or verified-at-noon price is available. Ground-truth outages reduce evaluable targets. Holiday counts are small, and this is one industrial facility. Feature importance is associative, not causal.

## Implications for probabilistic forecasting

Decision category: **A. Clear meaningful improvement over adaptive baselines**. This classification uses aggregate/common-target metrics, paired day bootstrap intervals, holiday and horizon behavior, and prediction coverage. No probabilistic model or uncertainty forecast was trained.
