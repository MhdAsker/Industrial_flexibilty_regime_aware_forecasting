# SIMULATED INDUSTRIAL FLEXIBILITY

## Motivation

This study tests whether finalized load forecasts improve price-aware decisions under transparent simulated flexibility assumptions.

## Decision problem

We optimize a **simulated shiftable-load envelope derived from measured industrial demand**. We do not optimize the actual SPARK production schedule.

## Why simulated flexibility is required

SPARK does not provide a complete production schedule, job precedence graph, deadlines, setup constraints, production quantities, or verified flexibility contracts. Flexible fractions are scenarios, not observed plant capabilities.

## Price availability and operational timing

The canonical SPARK metadata describes a German day-ahead wholesale price in UTC and EUR/MWh, but does not identify its exchange or publication timestamp. In 2024 all four quarter-hours within each UTC hour carry one repeated hourly value. [EPEX SPOT trading documentation](https://www.epexspot.com/en/tradingproducts) states 12:00 D-1 order-book closure and result publication from 12:55 onward; its [19 December 2024 SDAC notice](https://www.epexspot.com/en/news/15-minute-products-set-go-live-june-2025-european-single-day-ahead-coupling-sdac) scheduled 15-minute day-ahead products for 2025. Thus the complete profile is not known at 12:00 and source provenance is insufficient to assign interval-specific availability times.

The study is therefore an **ex-post decision simulation**. The 12:00 D-1 load-forecast protocol is unchanged, but realized delivery prices are oracle information for the decision layer.

## Forecast inputs

F0 uses eligible actual load only for perfect foresight. F1 uses the finalized deterministic LightGBM. F2 uses frozen R0 q10/q25/q50/q75/q90/q95. B3 is retained as a simple comparator. R4 is not used.

## Flexibility assumptions

The primary scenario uses 20% flexible load, a ±2 h physical-time window, and a cap equal to each planning scenario's original daily peak. Sensitivities use 5/10/20/30%, ±1/±2/±4 h, and cap multipliers 1.0/1.1/1.25. Energy is conserved; no load is deleted.

## Optimization formulation

Indices are origin interval $i$, destination interval $t$, and probabilistic scenario $s$. $L_{t,s}$ is load (kW), $B_{t,s}=(1-f)L_{t,s}$ is inflexible load, $F_{i,s}=fL_{i,s}$ is original flexible load, $a_{i,t}$ is the routing fraction, $P_t$ is EUR/kWh, and $C_s$ is daily EUR cost. The linear objective minimizes probability-weighted daily cost. Constraints require each origin's routing fractions to sum to one, prohibit routing outside the shift window, enforce non-negativity, and cap scenario grid load. The load-independent column bound `sum_i(a_i,t) <= (c - 1 + f) / f` guarantees cap multiplier `c` for any nonnegative realized profile; at `c = 1` routing is doubly stochastic and cannot increase the realized daily peak.

## Perfect foresight benchmark

| scenario   | forecast_method      |   flexibility_fraction |   shift_window_hours |   power_cap_multiplier |   days |   infeasible_days |   mean_no_flex_cost_eur |   mean_realized_cost_eur |   mean_savings_eur |   median_savings_eur |   mean_savings_pct |   mean_regret_eur |   p90_regret_eur |   max_regret_eur |   mean_forecast_efficiency |   mean_peak_reduction_kw |   mean_energy_shifted_kwh |   realized_cap_violation_days |   negative_price_energy_shifted_kwh |   negative_price_savings_contribution_eur |
|:-----------|:---------------------|-----------------------:|---------------------:|-----------------------:|-------:|------------------:|------------------------:|-------------------------:|-------------------:|---------------------:|-------------------:|------------------:|-----------------:|-----------------:|---------------------------:|-------------------------:|--------------------------:|------------------------------:|------------------------------------:|------------------------------------------:|
| PRIMARY    | D1_PERFECT_FORESIGHT |                0.20000 |                    2 |                1.00000 |    274 |                 0 |                28.38160 |                 28.25709 |            0.12451 |              0.09520 |            0.42720 |           0.00000 |          0.00000 |          0.00000 |                    1.00000 |                  0.85089 |                   4.85424 |                             0 |                            50.06125 |                                   0.19578 |

## Deterministic forecast optimization

| scenario   | forecast_method   |   flexibility_fraction |   shift_window_hours |   power_cap_multiplier |   days |   infeasible_days |   mean_no_flex_cost_eur |   mean_realized_cost_eur |   mean_savings_eur |   median_savings_eur |   mean_savings_pct |   mean_regret_eur |   p90_regret_eur |   max_regret_eur |   mean_forecast_efficiency |   mean_peak_reduction_kw |   mean_energy_shifted_kwh |   realized_cap_violation_days |   negative_price_energy_shifted_kwh |   negative_price_savings_contribution_eur |
|:-----------|:------------------|-----------------------:|---------------------:|-----------------------:|-------:|------------------:|------------------------:|-------------------------:|-------------------:|---------------------:|-------------------:|------------------:|-----------------:|-----------------:|---------------------------:|-------------------------:|--------------------------:|------------------------------:|------------------------------------:|------------------------------------------:|
| PRIMARY    | B3_SCHEDULING     |                0.20000 |                    2 |                1.00000 |    274 |                 0 |                28.38160 |                 28.29794 |            0.08366 |              0.05596 |            0.24306 |           0.04084 |          0.07994 |          0.31664 |                    0.42481 |                  0.79314 |                   4.50082 |                             0 |                            42.78648 |                                  -0.00457 |
| PRIMARY    | D2_DETERMINISTIC  |                0.20000 |                    2 |                1.00000 |    274 |                 0 |                28.38160 |                 28.29188 |            0.08972 |              0.05908 |            0.27956 |           0.03478 |          0.07340 |          0.23319 |                    0.48075 |                  0.74966 |                   4.38230 |                             0 |                            37.71969 |                                  -0.00056 |

## Probabilistic forecast optimization

R0 scenarios are q10/q50/q90 with weights 0.2/0.6/0.2. These are transparent decision-analysis support points, not claimed point masses of an exact distribution.

| scenario   | forecast_method   |   flexibility_fraction |   shift_window_hours |   power_cap_multiplier |   days |   infeasible_days |   mean_no_flex_cost_eur |   mean_realized_cost_eur |   mean_savings_eur |   median_savings_eur |   mean_savings_pct |   mean_regret_eur |   p90_regret_eur |   max_regret_eur |   mean_forecast_efficiency |   mean_peak_reduction_kw |   mean_energy_shifted_kwh |   realized_cap_violation_days |   negative_price_energy_shifted_kwh |   negative_price_savings_contribution_eur |
|:-----------|:------------------|-----------------------:|---------------------:|-----------------------:|-------:|------------------:|------------------------:|-------------------------:|-------------------:|---------------------:|-------------------:|------------------:|-----------------:|-----------------:|---------------------------:|-------------------------:|--------------------------:|------------------------------:|------------------------------------:|------------------------------------------:|
| PRIMARY    | D3_PROBABILISTIC  |                0.20000 |                    2 |                1.00000 |    274 |                 0 |                28.38160 |                 28.29053 |            0.09107 |              0.05914 |            0.28003 |           0.03343 |          0.06882 |          0.18286 |                    0.51295 |                  0.74853 |                   4.41658 |                             0 |                            40.98853 |                                   0.01573 |

## Risk-aware formulation

A predefined q95 upper-tail sensitivity replaces q90 with q95. CVaR is not used because three quantile support points are not a sufficiently resolved probability distribution for a strong CVaR interpretation.

## Primary scenario

20% flexible, ±2 h, 1.0× peak cap. Only complete physical local days with 92/96/100 eligible intervals and complete prices are evaluated.

## Sensitivity scenarios

| scenario           | forecast_method   |   flexibility_fraction |   shift_window_hours |   power_cap_multiplier |   days |   infeasible_days |   mean_no_flex_cost_eur |   mean_realized_cost_eur |   mean_savings_eur |   median_savings_eur |   mean_savings_pct |   mean_regret_eur |   p90_regret_eur |   max_regret_eur |   mean_forecast_efficiency |   mean_peak_reduction_kw |   mean_energy_shifted_kwh |   realized_cap_violation_days |   negative_price_energy_shifted_kwh |   negative_price_savings_contribution_eur |
|:-------------------|:------------------|-----------------------:|---------------------:|-----------------------:|-------:|------------------:|------------------------:|-------------------------:|-------------------:|---------------------:|-------------------:|------------------:|-----------------:|-----------------:|---------------------------:|-------------------------:|--------------------------:|------------------------------:|------------------------------------:|------------------------------------------:|
| SENS_F05_W1_CAP100 | D3_PROBABILISTIC  |                0.05000 |                    1 |                1.00000 |    274 |                 0 |                28.38160 |                 28.37633 |            0.00528 |              0.00326 |            0.01708 |               nan |              nan |              nan |                        nan |                  0.15704 |                   0.71091 |                             0 |                             6.70451 |                                   0.00210 |
| SENS_F05_W2_CAP100 | D3_PROBABILISTIC  |                0.05000 |                    2 |                1.00000 |    274 |                 0 |                28.38160 |                 28.35882 |            0.02278 |              0.01488 |            0.07008 |               nan |              nan |              nan |                        nan |                  0.20240 |                   1.10092 |                             0 |                             9.67926 |                                   0.00376 |
| SENS_F05_W4_CAP100 | D3_PROBABILISTIC  |                0.05000 |                    4 |                1.00000 |    274 |                 0 |                28.38160 |                 28.29982 |            0.08178 |              0.06678 |            0.24903 |               nan |              nan |              nan |                        nan |                  0.32781 |                   2.49272 |                             0 |                            20.28917 |                                  -0.00601 |
| SENS_F10_W1_CAP100 | D3_PROBABILISTIC  |                0.10000 |                    1 |                1.00000 |    274 |                 0 |                28.38160 |                 28.37104 |            0.01056 |              0.00651 |            0.03407 |               nan |              nan |              nan |                        nan |                  0.32692 |                   1.43662 |                             0 |                            13.37179 |                                   0.00406 |
| SENS_F10_W2_CAP100 | D3_PROBABILISTIC  |                0.10000 |                    2 |                1.00000 |    274 |                 0 |                28.38160 |                 28.33607 |            0.04553 |              0.02957 |            0.13933 |               nan |              nan |              nan |                        nan |                  0.40469 |                   2.20715 |                             0 |                            20.53972 |                                   0.00766 |
| SENS_F10_W4_CAP100 | D3_PROBABILISTIC  |                0.10000 |                    4 |                1.00000 |    274 |                 0 |                28.38160 |                 28.21808 |            0.16352 |              0.13356 |            0.49838 |               nan |              nan |              nan |                        nan |                  0.62468 |                   4.98577 |                             0 |                            41.29738 |                                  -0.01191 |
| SENS_F20_W1_CAP100 | D3_PROBABILISTIC  |                0.20000 |                    1 |                1.00000 |    274 |                 0 |                28.38160 |                 28.36047 |            0.02113 |              0.01303 |            0.06867 |               nan |              nan |              nan |                        nan |                  0.63604 |                   2.86492 |                             0 |                            26.81288 |                                   0.00829 |
| SENS_F20_W2_CAP110 | D3_PROBABILISTIC  |                0.20000 |                    2 |                1.10000 |    274 |                 0 |                28.38160 |                 27.20767 |            1.17393 |              0.98210 |            4.60144 |               nan |              nan |              nan |                        nan |                 -0.86016 |                  24.28129 |                             0 |                           576.70892 |                                   7.25245 |
| SENS_F20_W2_CAP125 | D3_PROBABILISTIC  |                0.20000 |                    2 |                1.25000 |    274 |                 0 |                28.38160 |                 26.98525 |            1.39635 |              1.16154 |            5.37530 |               nan |              nan |              nan |                        nan |                 -3.49951 |                  35.37809 |                             0 |                          1015.71346 |                                  13.26259 |
| SENS_F20_W4_CAP100 | D3_PROBABILISTIC  |                0.20000 |                    4 |                1.00000 |    274 |                 0 |                28.38160 |                 28.05453 |            0.32707 |              0.26713 |            0.99554 |               nan |              nan |              nan |                        nan |                  1.14289 |                   9.96358 |                             0 |                            82.48120 |                                  -0.02414 |
| SENS_F30_W1_CAP100 | D3_PROBABILISTIC  |                0.30000 |                    1 |                1.00000 |    274 |                 0 |                28.38160 |                 28.34997 |            0.03163 |              0.01934 |            0.10069 |               nan |              nan |              nan |                        nan |                  0.85590 |                   4.27888 |                             0 |                            37.85543 |                                   0.01169 |
| SENS_F30_W2_CAP100 | D3_PROBABILISTIC  |                0.30000 |                    2 |                1.00000 |    274 |                 0 |                28.38160 |                 28.24501 |            0.13659 |              0.08866 |            0.41995 |               nan |              nan |              nan |                        nan |                  1.08684 |                   6.63966 |                             0 |                            62.17052 |                                   0.02271 |
| SENS_F30_W4_CAP100 | D3_PROBABILISTIC  |                0.30000 |                    4 |                1.00000 |    274 |                 0 |                28.38160 |                 27.89099 |            0.49061 |              0.40069 |            1.49271 |               nan |              nan |              nan |                        nan |                  1.56093 |                  14.93870 |                             0 |                           122.82438 |                                  -0.03606 |

## Economic results

| scenario   | forecast_method      |   flexibility_fraction |   shift_window_hours |   power_cap_multiplier |   days |   infeasible_days |   mean_no_flex_cost_eur |   mean_realized_cost_eur |   mean_savings_eur |   median_savings_eur |   mean_savings_pct |   mean_regret_eur |   p90_regret_eur |   max_regret_eur |   mean_forecast_efficiency |   mean_peak_reduction_kw |   mean_energy_shifted_kwh |   realized_cap_violation_days |   negative_price_energy_shifted_kwh |   negative_price_savings_contribution_eur |
|:-----------|:---------------------|-----------------------:|---------------------:|-----------------------:|-------:|------------------:|------------------------:|-------------------------:|-------------------:|---------------------:|-------------------:|------------------:|-----------------:|-----------------:|---------------------------:|-------------------------:|--------------------------:|------------------------------:|------------------------------------:|------------------------------------------:|
| PRIMARY    | B3_SCHEDULING        |                0.20000 |                    2 |                1.00000 |    274 |                 0 |                28.38160 |                 28.29794 |            0.08366 |              0.05596 |            0.24306 |           0.04084 |          0.07994 |          0.31664 |                    0.42481 |                  0.79314 |                   4.50082 |                             0 |                            42.78648 |                                  -0.00457 |
| PRIMARY    | D0_NO_FLEX           |                0.20000 |                    2 |                1.00000 |    274 |                 0 |                28.38160 |                 28.38160 |            0.00000 |              0.00000 |            0.00000 |           0.12451 |          0.20560 |          2.95516 |                    0.00000 |                  0.00000 |                   0.00000 |                             0 |                             0.00000 |                                   0.00000 |
| PRIMARY    | D1_PERFECT_FORESIGHT |                0.20000 |                    2 |                1.00000 |    274 |                 0 |                28.38160 |                 28.25709 |            0.12451 |              0.09520 |            0.42720 |           0.00000 |          0.00000 |          0.00000 |                    1.00000 |                  0.85089 |                   4.85424 |                             0 |                            50.06125 |                                   0.19578 |
| PRIMARY    | D2_DETERMINISTIC     |                0.20000 |                    2 |                1.00000 |    274 |                 0 |                28.38160 |                 28.29188 |            0.08972 |              0.05908 |            0.27956 |           0.03478 |          0.07340 |          0.23319 |                    0.48075 |                  0.74966 |                   4.38230 |                             0 |                            37.71969 |                                  -0.00056 |
| PRIMARY    | D3_PROBABILISTIC     |                0.20000 |                    2 |                1.00000 |    274 |                 0 |                28.38160 |                 28.29053 |            0.09107 |              0.05914 |            0.28003 |           0.03343 |          0.06882 |          0.18286 |                    0.51295 |                  0.74853 |                   4.41658 |                             0 |                            40.98853 |                                   0.01573 |

## Peak-demand results

Peak change is an outcome, not the objective. The robust routing-column constraint guarantees the configured realized peak cap for every nonnegative load profile without using future labels.

## Forecast-value analysis

Forecast efficiency is realized forecast-based savings divided by same-day perfect-foresight savings when positive.

## Regret analysis

| comparison           | metric                             |   estimate |   ci_low |   ci_high |
|:---------------------|:-----------------------------------|-----------:|---------:|----------:|
| B3_SCHEDULING        | mean_daily_savings_eur             |    0.08366 |  0.06482 |   0.10932 |
| B3_SCHEDULING        | mean_daily_savings_pct             |    0.24306 |  0.20145 |   0.28908 |
| B3_SCHEDULING        | mean_daily_regret_eur              |    0.04084 |  0.03646 |   0.04578 |
| D0_NO_FLEX           | mean_daily_savings_eur             |    0.00000 |  0.00000 |   0.00000 |
| D0_NO_FLEX           | mean_daily_savings_pct             |    0.00000 |  0.00000 |   0.00000 |
| D0_NO_FLEX           | mean_daily_regret_eur              |    0.12451 |  0.10296 |   0.15225 |
| D1_PERFECT_FORESIGHT | mean_daily_savings_eur             |    0.12451 |  0.10351 |   0.15287 |
| D1_PERFECT_FORESIGHT | mean_daily_savings_pct             |    0.42720 |  0.33513 |   0.55210 |
| D1_PERFECT_FORESIGHT | mean_daily_regret_eur              |    0.00000 |  0.00000 |   0.00000 |
| D2_DETERMINISTIC     | mean_daily_savings_eur             |    0.08972 |  0.06902 |   0.11782 |
| D2_DETERMINISTIC     | mean_daily_savings_pct             |    0.27956 |  0.21929 |   0.35648 |
| D2_DETERMINISTIC     | mean_daily_regret_eur              |    0.03478 |  0.03122 |   0.03865 |
| D3_PROBABILISTIC     | mean_daily_savings_eur             |    0.09107 |  0.07113 |   0.11892 |
| D3_PROBABILISTIC     | mean_daily_savings_pct             |    0.28003 |  0.22382 |   0.34841 |
| D3_PROBABILISTIC     | mean_daily_regret_eur              |    0.03343 |  0.03004 |   0.03691 |
| D3_MINUS_D2          | paired_mean_savings_difference_eur |    0.00135 | -0.00007 |   0.00273 |

| comparison                              |   days |   d3_win_fraction |   tie_fraction |   mean_savings_difference_eur |   median_savings_difference_eur |   d2_p90_regret_eur |   d3_p90_regret_eur |   d2_worst_regret_eur |   d3_worst_regret_eur |
|:----------------------------------------|-------:|------------------:|---------------:|------------------------------:|--------------------------------:|--------------------:|--------------------:|----------------------:|----------------------:|
| D3_PROBABILISTIC minus D2_DETERMINISTIC |    274 |           0.56934 |        0.00365 |                       0.00135 |                         0.00068 |             0.07340 |             0.06882 |               0.23319 |               0.18286 |

## Holiday safeguards

The policy was frozen as no flexibility on public holidays because R0 holiday calibration is unreliable. Normal-policy holiday results remain available for comparison.

| scenario                 | forecast_method      | dimension         | group   |   days |   mean_savings_eur |   mean_savings_pct |   mean_regret_eur |   p90_regret_eur |   realized_cap_violation_days |
|:-------------------------|:---------------------|:------------------|:--------|-------:|-------------------:|-------------------:|------------------:|-----------------:|------------------------------:|
| PRIMARY                  | B3_SCHEDULING        | is_public_holiday | False   |    265 |            0.08632 |            0.25367 |           0.04176 |          0.08027 |                             0 |
| PRIMARY                  | B3_SCHEDULING        | is_public_holiday | True    |      9 |            0.00551 |           -0.06957 |           0.01398 |          0.02683 |                             0 |
| PRIMARY                  | D0_NO_FLEX           | is_public_holiday | False   |    265 |            0.00000 |            0.00000 |           0.12808 |          0.20652 |                             0 |
| PRIMARY                  | D0_NO_FLEX           | is_public_holiday | True    |      9 |            0.00000 |            0.00000 |           0.01948 |          0.04666 |                             0 |
| PRIMARY                  | D1_PERFECT_FORESIGHT | is_public_holiday | False   |    265 |            0.12808 |            0.45274 |           0.00000 |          0.00000 |                             0 |
| PRIMARY                  | D1_PERFECT_FORESIGHT | is_public_holiday | True    |      9 |            0.01948 |           -0.32471 |           0.00000 |          0.00000 |                             0 |
| PRIMARY                  | D2_DETERMINISTIC     | is_public_holiday | False   |    265 |            0.09269 |            0.28988 |           0.03539 |          0.07396 |                             0 |
| PRIMARY                  | D2_DETERMINISTIC     | is_public_holiday | True    |      9 |            0.00251 |           -0.02451 |           0.01697 |          0.03345 |                             0 |
| PRIMARY                  | D3_PROBABILISTIC     | is_public_holiday | False   |    265 |            0.09398 |            0.29426 |           0.03410 |          0.06993 |                             0 |
| PRIMARY                  | D3_PROBABILISTIC     | is_public_holiday | True    |      9 |            0.00567 |           -0.13890 |           0.01381 |          0.02674 |                             0 |
| PRIMARY_HOLIDAY_FALLBACK | D3_HOLIDAY_FALLBACK  | is_public_holiday | False   |    265 |            0.09398 |            0.29426 |           0.03410 |          0.06993 |                             0 |
| PRIMARY_HOLIDAY_FALLBACK | D3_HOLIDAY_FALLBACK  | is_public_holiday | True    |      9 |            0.00000 |            0.00000 |           0.01948 |          0.04666 |                             0 |
| PRIMARY_Q95_SENSITIVITY  | D3_Q95_SENSITIVITY   | is_public_holiday | False   |    265 |            0.09316 |            0.28718 |           0.03491 |          0.07373 |                             0 |
| PRIMARY_Q95_SENSITIVITY  | D3_Q95_SENSITIVITY   | is_public_holiday | True    |      9 |            0.00625 |           -0.21220 |           0.01324 |          0.02407 |                             0 |

## High-load safeguards

The q95 sensitivity is reported separately and was not selected from TEST performance.

| scenario                 | forecast_method      | dimension    | group   |   days |   mean_savings_eur |   mean_savings_pct |   mean_regret_eur |   p90_regret_eur |   realized_cap_violation_days |
|:-------------------------|:---------------------|:-------------|:--------|-------:|-------------------:|-------------------:|------------------:|-----------------:|------------------------------:|
| PRIMARY                  | B3_SCHEDULING        | regime_level | HIGH    |    181 |            0.06714 |            0.18836 |           0.04665 |          0.08588 |                             0 |
| PRIMARY                  | B3_SCHEDULING        | regime_level | LOW     |     32 |            0.08602 |            0.33247 |           0.01908 |          0.03310 |                             0 |
| PRIMARY                  | B3_SCHEDULING        | regime_level | NORMAL  |     61 |            0.13145 |            0.35845 |           0.03504 |          0.06843 |                             0 |
| PRIMARY                  | D0_NO_FLEX           | regime_level | HIGH    |    181 |            0.00000 |            0.00000 |           0.11379 |          0.19790 |                             0 |
| PRIMARY                  | D0_NO_FLEX           | regime_level | LOW     |     32 |            0.00000 |            0.00000 |           0.10510 |          0.18548 |                             0 |
| PRIMARY                  | D0_NO_FLEX           | regime_level | NORMAL  |     61 |            0.00000 |            0.00000 |           0.16649 |          0.22440 |                             0 |
| PRIMARY                  | D1_PERFECT_FORESIGHT | regime_level | HIGH    |    181 |            0.11379 |            0.38520 |           0.00000 |          0.00000 |                             0 |
| PRIMARY                  | D1_PERFECT_FORESIGHT | regime_level | LOW     |     32 |            0.10510 |            0.45694 |           0.00000 |          0.00000 |                             0 |
| PRIMARY                  | D1_PERFECT_FORESIGHT | regime_level | NORMAL  |     61 |            0.16649 |            0.53624 |           0.00000 |          0.00000 |                             0 |
| PRIMARY                  | D2_DETERMINISTIC     | regime_level | HIGH    |    181 |            0.07408 |            0.23031 |           0.03971 |          0.07698 |                             0 |
| PRIMARY                  | D2_DETERMINISTIC     | regime_level | LOW     |     32 |            0.08773 |            0.34424 |           0.01737 |          0.03681 |                             0 |
| PRIMARY                  | D2_DETERMINISTIC     | regime_level | NORMAL  |     61 |            0.13720 |            0.39173 |           0.02930 |          0.06435 |                             0 |
| PRIMARY                  | D3_PROBABILISTIC     | regime_level | HIGH    |    181 |            0.07579 |            0.22832 |           0.03800 |          0.07946 |                             0 |
| PRIMARY                  | D3_PROBABILISTIC     | regime_level | LOW     |     32 |            0.09013 |            0.38009 |           0.01498 |          0.03121 |                             0 |
| PRIMARY                  | D3_PROBABILISTIC     | regime_level | NORMAL  |     61 |            0.13693 |            0.38097 |           0.02956 |          0.06539 |                             0 |
| PRIMARY_HOLIDAY_FALLBACK | D3_HOLIDAY_FALLBACK  | regime_level | HIGH    |    181 |            0.07552 |            0.23548 |           0.03827 |          0.07946 |                             0 |
| PRIMARY_HOLIDAY_FALLBACK | D3_HOLIDAY_FALLBACK  | regime_level | LOW     |     32 |            0.09013 |            0.38009 |           0.01498 |          0.03121 |                             0 |
| PRIMARY_HOLIDAY_FALLBACK | D3_HOLIDAY_FALLBACK  | regime_level | NORMAL  |     61 |            0.13690 |            0.38021 |           0.02959 |          0.06539 |                             0 |
| PRIMARY_Q95_SENSITIVITY  | D3_Q95_SENSITIVITY   | regime_level | HIGH    |    181 |            0.07458 |            0.21459 |           0.03921 |          0.08203 |                             0 |
| PRIMARY_Q95_SENSITIVITY  | D3_Q95_SENSITIVITY   | regime_level | LOW     |     32 |            0.09017 |            0.38017 |           0.01493 |          0.02691 |                             0 |
| PRIMARY_Q95_SENSITIVITY  | D3_Q95_SENSITIVITY   | regime_level | NORMAL  |     61 |            0.13704 |            0.38009 |           0.02945 |          0.06423 |                             0 |

## Price-volatility dependence

| forecast_method      | predictor                                 | outcome     |   pearson |   spearman |
|:---------------------|:------------------------------------------|:------------|----------:|-----------:|
| B3_SCHEDULING        | forecast_mae_deterministic_kw             | regret_eur  |   0.33639 |    0.43794 |
| B3_SCHEDULING        | forecast_mae_deterministic_kw             | savings_eur |   0.12966 |    0.26849 |
| B3_SCHEDULING        | probabilistic_wis                         | regret_eur  |   0.34114 |    0.46661 |
| B3_SCHEDULING        | probabilistic_wis                         | savings_eur |   0.11497 |    0.24713 |
| B3_SCHEDULING        | cost_weighted_abs_error_deterministic_eur | regret_eur  |   0.45997 |    0.43365 |
| B3_SCHEDULING        | cost_weighted_abs_error_deterministic_eur | savings_eur |   0.47326 |    0.43299 |
| B3_SCHEDULING        | daily_price_p90_p10_eur_per_mwh           | regret_eur  |   0.44101 |    0.30081 |
| B3_SCHEDULING        | daily_price_p90_p10_eur_per_mwh           | savings_eur |   0.46816 |   -0.11953 |
| B3_SCHEDULING        | daily_price_std_eur_per_mwh               | regret_eur  |   0.50429 |    0.33958 |
| B3_SCHEDULING        | daily_price_std_eur_per_mwh               | savings_eur |   0.56378 |   -0.05802 |
| D0_NO_FLEX           | forecast_mae_deterministic_kw             | regret_eur  |   0.17735 |    0.37359 |
| D0_NO_FLEX           | forecast_mae_deterministic_kw             | savings_eur | nan       |  nan       |
| D0_NO_FLEX           | probabilistic_wis                         | regret_eur  |   0.16508 |    0.36703 |
| D0_NO_FLEX           | probabilistic_wis                         | savings_eur | nan       |  nan       |
| D0_NO_FLEX           | cost_weighted_abs_error_deterministic_eur | regret_eur  |   0.50722 |    0.51008 |
| D0_NO_FLEX           | cost_weighted_abs_error_deterministic_eur | savings_eur | nan       |  nan       |
| D0_NO_FLEX           | daily_price_p90_p10_eur_per_mwh           | regret_eur  |   0.49920 |    0.03581 |
| D0_NO_FLEX           | daily_price_p90_p10_eur_per_mwh           | savings_eur | nan       |  nan       |
| D0_NO_FLEX           | daily_price_std_eur_per_mwh               | regret_eur  |   0.59628 |    0.10326 |
| D0_NO_FLEX           | daily_price_std_eur_per_mwh               | savings_eur | nan       |  nan       |
| D1_PERFECT_FORESIGHT | forecast_mae_deterministic_kw             | regret_eur  | nan       |  nan       |
| D1_PERFECT_FORESIGHT | forecast_mae_deterministic_kw             | savings_eur |   0.17735 |    0.37359 |
| D1_PERFECT_FORESIGHT | probabilistic_wis                         | regret_eur  | nan       |  nan       |
| D1_PERFECT_FORESIGHT | probabilistic_wis                         | savings_eur |   0.16508 |    0.36703 |
| D1_PERFECT_FORESIGHT | cost_weighted_abs_error_deterministic_eur | regret_eur  | nan       |  nan       |
| D1_PERFECT_FORESIGHT | cost_weighted_abs_error_deterministic_eur | savings_eur |   0.50722 |    0.51008 |
| D1_PERFECT_FORESIGHT | daily_price_p90_p10_eur_per_mwh           | regret_eur  | nan       |  nan       |
| D1_PERFECT_FORESIGHT | daily_price_p90_p10_eur_per_mwh           | savings_eur |   0.49920 |    0.03581 |
| D1_PERFECT_FORESIGHT | daily_price_std_eur_per_mwh               | regret_eur  | nan       |  nan       |
| D1_PERFECT_FORESIGHT | daily_price_std_eur_per_mwh               | savings_eur |   0.59628 |    0.10326 |
| D2_DETERMINISTIC     | forecast_mae_deterministic_kw             | regret_eur  |   0.39914 |    0.47445 |
| D2_DETERMINISTIC     | forecast_mae_deterministic_kw             | savings_eur |   0.12401 |    0.25703 |
| D2_DETERMINISTIC     | probabilistic_wis                         | regret_eur  |   0.41682 |    0.50220 |
| D2_DETERMINISTIC     | probabilistic_wis                         | savings_eur |   0.10881 |    0.24084 |
| D2_DETERMINISTIC     | cost_weighted_abs_error_deterministic_eur | regret_eur  |   0.39000 |    0.46347 |
| D2_DETERMINISTIC     | cost_weighted_abs_error_deterministic_eur | savings_eur |   0.46457 |    0.42746 |
| D2_DETERMINISTIC     | daily_price_p90_p10_eur_per_mwh           | regret_eur  |   0.23481 |    0.30793 |
| D2_DETERMINISTIC     | daily_price_p90_p10_eur_per_mwh           | savings_eur |   0.47901 |   -0.11537 |
| D2_DETERMINISTIC     | daily_price_std_eur_per_mwh               | regret_eur  |   0.27796 |    0.34760 |
| D2_DETERMINISTIC     | daily_price_std_eur_per_mwh               | savings_eur |   0.57253 |   -0.06218 |
| D3_PROBABILISTIC     | forecast_mae_deterministic_kw             | regret_eur  |   0.38611 |    0.44087 |
| D3_PROBABILISTIC     | forecast_mae_deterministic_kw             | savings_eur |   0.13045 |    0.28684 |
| D3_PROBABILISTIC     | probabilistic_wis                         | regret_eur  |   0.40363 |    0.48207 |
| D3_PROBABILISTIC     | probabilistic_wis                         | savings_eur |   0.11517 |    0.26939 |
| D3_PROBABILISTIC     | cost_weighted_abs_error_deterministic_eur | regret_eur  |   0.40999 |    0.44153 |
| D3_PROBABILISTIC     | cost_weighted_abs_error_deterministic_eur | savings_eur |   0.47112 |    0.45226 |
| D3_PROBABILISTIC     | daily_price_p90_p10_eur_per_mwh           | regret_eur  |   0.28814 |    0.31058 |
| D3_PROBABILISTIC     | daily_price_p90_p10_eur_per_mwh           | savings_eur |   0.47996 |   -0.09952 |
| D3_PROBABILISTIC     | daily_price_std_eur_per_mwh               | regret_eur  |   0.33263 |    0.34862 |
| D3_PROBABILISTIC     | daily_price_std_eur_per_mwh               | savings_eur |   0.57493 |   -0.03798 |

## Negative-price behavior

Negative prices are retained. Energy moved into negative-price intervals is reported rather than treated as an error.

## Optional PV case study

Not implemented. Limited PV history and the lack of a frozen operational PV forecast would make it an oracle extension and would delay the primary load-flexibility study.

## Representative days

| selection_rule            | date       | scenario   | forecast_method   |   flexibility_fraction |   effective_flexibility_fraction |   shift_window_hours |   power_cap_multiplier |   physical_intervals |   no_flex_cost |   optimized_planned_cost |   optimized_realized_cost |   savings_eur |   savings_pct |   energy_shifted_kwh |   mean_temporal_shift_hours |   maximum_temporal_shift_hours |   intervals_changed |   fraction_total_energy_shifted |   peak_before_kw |   peak_after_kw |   peak_reduction_kw |   realized_cap_violation_kw |   planned_cap_violation_kw |   energy_shifted_into_negative_price_kwh |   negative_price_savings_contribution_eur | solver_status   |   solver_status_code | is_public_holiday   | regime_level   | regime_calendar   |   forecast_mae_deterministic_kw |   forecast_bias_deterministic_kw |   forecast_mae_b3_kw |   forecast_mae_q50_kw |   probabilistic_wis |   cost_weighted_abs_error_deterministic_eur |   daily_price_range_eur_per_mwh |   daily_price_p90_p10_eur_per_mwh |   daily_price_std_eur_per_mwh |   negative_price_intervals | has_negative_price   |   perfect_foresight_cost |   perfect_foresight_savings_eur |   regret_eur |   forecast_efficiency |
|:--------------------------|:-----------|:-----------|:------------------|-----------------------:|---------------------------------:|---------------------:|-----------------------:|---------------------:|---------------:|-------------------------:|--------------------------:|--------------:|--------------:|---------------------:|----------------------------:|-------------------------------:|--------------------:|--------------------------------:|-----------------:|----------------:|--------------------:|----------------------------:|---------------------------:|-----------------------------------------:|------------------------------------------:|:----------------|---------------------:|:--------------------|:---------------|:------------------|--------------------------------:|---------------------------------:|---------------------:|----------------------:|--------------------:|--------------------------------------------:|--------------------------------:|----------------------------------:|------------------------------:|---------------------------:|:---------------------|-------------------------:|--------------------------------:|-------------:|----------------------:|
| median_savings            | 2024-03-05 | PRIMARY    | D3_PROBABILISTIC  |                0.20000 |                          0.20000 |                    2 |                1.00000 |                   96 |       25.24973 |                 32.11177 |                  25.19065 |       0.05908 |       0.23398 |              5.70588 |                     1.11446 |                        2.00000 |                  92 |                         0.01677 |         26.12044 |        25.99622 |             0.12422 |                     0.00000 |                    0.00000 |                                  0.00000 |                                  -0.00000 | optimal         |                    0 | False               | HIGH           | normal_weekday    |                         2.59949 |                          2.19471 |              1.64088 |               3.33997 |             2.18314 |                                     4.95682 |                        53.13000 |                          31.79000 |                      13.32118 |                          0 | False                |                 25.18267 |                         0.06706 |      0.00798 |               0.88104 |
| high_savings              | 2024-11-06 | PRIMARY    | D3_PROBABILISTIC  |                0.20000 |                          0.20000 |                    2 |                1.00000 |                   96 |      104.23318 |                 95.42868 |                 101.33488 |       2.89830 |       2.78059 |              9.78165 |                     1.23227 |                        2.00000 |                  85 |                         0.02429 |         32.64825 |        31.64328 |             1.00496 |                     0.00000 |                    0.00000 |                                  0.00000 |                                  -0.00000 | optimal         |                    0 | False               | NORMAL         | normal_weekday    |                         2.92880 |                         -0.26444 |              3.68231 |               3.31253 |             1.80147 |                                    17.67267 |                       724.64000 |                         425.20000 |                     205.99248 |                          0 | False                |                101.27801 |                         2.95516 |      0.05687 |               0.98076 |
| low_savings               | 2024-10-03 | PRIMARY    | D3_PROBABILISTIC  |                0.20000 |                          0.20000 |                    2 |                1.00000 |                   96 |       10.12690 |                 20.72833 |                  10.14594 |      -0.01904 |      -0.18805 |              1.55665 |                     1.30207 |                        2.00000 |                  89 |                         0.01059 |         11.77536 |        11.65493 |             0.12043 |                     0.00000 |                    0.00000 |                                  0.00000 |                                  -0.00000 | optimal         |                    0 | True                | HIGH           | public_holiday    |                         6.24749 |                          6.24749 |             13.90379 |               5.33576 |             3.24543 |                                    11.06936 |                        81.34000 |                          61.19000 |                      20.57780 |                          0 | False                |                 10.11811 |                         0.00879 |      0.02783 |              -2.16618 |
| high_regret_probabilistic | 2024-11-12 | PRIMARY    | D3_PROBABILISTIC  |                0.20000 |                          0.20000 |                    2 |                1.00000 |                   96 |       71.01163 |                 68.25920 |                  70.41895 |       0.59268 |       0.83462 |              7.68481 |                     1.32195 |                        2.00000 |                  93 |                         0.02023 |         29.22109 |        28.87218 |             0.34892 |                     0.00000 |                    0.00000 |                                  0.00000 |                                  -0.00000 | optimal         |                    0 | False               | NORMAL         | normal_weekday    |                         2.72394 |                         -0.31809 |              2.39854 |               2.90412 |             1.63672 |                                    11.74064 |                       215.20000 |                         120.69000 |                      55.13673 |                          0 | False                |                 70.23609 |                         0.77553 |      0.18286 |               0.76422 |
| high_regret_deterministic | 2024-10-29 | PRIMARY    | D2_DETERMINISTIC  |                0.20000 |                          0.20000 |                    2 |                1.00000 |                   96 |       63.83835 |                 62.29245 |                  63.46612 |       0.37222 |       0.58307 |              9.34671 |                     1.25038 |                        2.00000 |                  93 |                         0.02198 |         35.93693 |        33.60878 |             2.32815 |                     0.00000 |                    0.00000 |                                  0.00000 |                                  -0.00000 | optimal         |                    0 | False               | HIGH           | normal_weekday    |                         3.06946 |                          0.54028 |              3.84574 |               3.00587 |             1.92578 |                                    10.44835 |                       187.63000 |                         112.70000 |                      49.95474 |                          0 | False                |                 63.23293 |                         0.60541 |      0.23319 |               0.61483 |
| negative_price_day        | 2024-04-02 | PRIMARY    | D3_PROBABILISTIC  |                0.20000 |                          0.20000 |                    2 |                1.00000 |                   96 |       12.27947 |                  9.90310 |                  11.95909 |       0.32037 |       2.60899 |              7.92462 |                     1.31761 |                        2.00000 |                  86 |                         0.01980 |         38.66978 |        34.20158 |             4.46820 |                     0.00000 |                    0.00000 |                                  1.03826 |                                   0.00051 | optimal         |                    0 | False               | NORMAL         | day_after_holiday |                         3.29428 |                         -2.22486 |              2.86450 |               3.41765 |             2.05798 |                                     3.21026 |                        96.48000 |                          78.20000 |                      35.96874 |                         16 | True                 |                 11.89269 |                         0.38677 |      0.06640 |               0.82832 |

## Limitations

Prices are ex-post oracle information at the decision layer; flexible load is simulated; routing is divisible and omits jobs, minimum run times, setups, rebound, production deadlines, network tariffs, and contractual constraints. Results are conditional on one facility and the chosen sensitivity grid.

## Implications for industrial energy management

Decision category: **B. Forecast-aware flexibility creates value, but probabilistic forecasting adds little beyond the deterministic model**. Results quantify the value of a simulated load-shifting envelope, not actual factory-production control.
