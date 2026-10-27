# SILVER data-layer validation summary

Generated: 2026-10-27T12:21:42.485157+00:00

Canonical data are direct aggregations of the official 5-second source. No interpolation, clipping, smoothing, or coverage masking is applied.

## TotalLoad

`valid` means at least one observation in an interval. `complete` means exactly 180/180 observations at 15 minutes. The large difference is mostly caused by frequent isolated missing readings: common coverage modes are around 75–77%, so 179/180 is not being rejected or relabelled as missing.

| Year | Raw availability | >0 | 100% | ≥99% | ≥95% | ≥90% | ≥80% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 78.77% | 35,021 (99.95%) | 2,342 (6.68%) | 2,344 (6.69%) | 2,488 (7.10%) | 2,493 (7.11%) | 5,454 (15.57%) |
| 2022 | 79.64% | 34,920 (99.66%) | 78 (0.22%) | 78 (0.22%) | 1,653 (4.72%) | 4,686 (13.37%) | 10,663 (30.43%) |
| 2023 | 71.67% | 33,019 (94.23%) | 0 (0.00%) | 0 (0.00%) | 14 (0.04%) | 70 (0.20%) | 84 (0.24%) |
| 2024 | 68.31% | 31,407 (89.39%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |

### Best contiguous high-quality windows

| Year | Threshold | Duration | Start UTC | End UTC (inclusive) |
|---:|---:|---:|---|---|
| 2021 | ≥99% | 167.75 h | 2021-11-27T23:15:00+00:00 | 2021-12-04T22:45:00+00:00 |
| 2021 | ≥95% | 373.75 h | 2021-11-22T12:15:00+00:00 | 2021-12-08T01:45:00+00:00 |
| 2021 | ≥90% | 373.75 h | 2021-11-22T12:15:00+00:00 | 2021-12-08T01:45:00+00:00 |
| 2022 | ≥99% | 18.50 h | 2022-07-20T10:00:00+00:00 | 2022-07-21T04:15:00+00:00 |
| 2022 | ≥95% | 180.50 h | 2022-09-07T11:15:00+00:00 | 2022-09-14T23:30:00+00:00 |
| 2022 | ≥90% | 284.75 h | 2022-09-07T11:15:00+00:00 | 2022-09-19T07:45:00+00:00 |
| 2023 | ≥99% | 0 h | — | — |
| 2023 | ≥95% | 0.50 h | 2023-03-08T02:45:00+00:00 | 2023-03-08T03:00:00+00:00 |
| 2023 | ≥90% | 5.50 h | 2023-03-08T00:15:00+00:00 | 2023-03-08T05:30:00+00:00 |
| 2024 | ≥99% | 0 h | — | — |
| 2024 | ≥95% | 0 h | — | — |
| 2024 | ≥90% | 0 h | — | — |

### Complete periods and zero-observation gaps

| Year | Days ≥99% | Days ≥95% | Weeks ≥95% | Months ≥95% | Longest zero-observation gap | Gaps >24 h | Gaps >7 d |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 19 | 22 | 1 | 0 | 1.25 h | 0 | 0 |
| 2022 | 0 | 13 | 0 | 0 | 24.75 h | 1 | 0 |
| 2023 | 0 | 0 | 0 | 0 | 328.50 h | 2 | 2 |
| 2024 | 0 | 0 | 0 | 0 | 712.25 h | 3 | 1 |

These are zero-observation gaps and are distinct from partially observed intervals; all threshold and gap counts are in `missingness_report.json`.

## PV

PV has 1,486,088 valid source samples (23.50% of 2024). Exact first/last valid source timestamps are 2024-01-01 00:00:00 and 2024-04-09 00:00:00 Europe/Berlin wall time; the first/last usable 15-minute UTC intervals are 2023-12-31T23:00:00+00:00 and 2024-04-08T22:00:00+00:00.

At 15 minutes: 8,252 intervals are complete, 86 local days are complete at ≥95%, and 10 full local weeks are complete at ≥95%.

Using observed radiation: measured nighttime zero=3,915, nighttime missing=13,588, daylight missing=13,288, valid daylight generation=2,674. Missing PV is never replaced by zero.

## Joint Load + PV

| Threshold for both | Joint intervals | Complete days | Complete weeks | Longest contiguous |
|---:|---:|---:|---:|---:|
| ≥80% | 0 | 0 | 0 | 0.00 h |
| ≥90% | 0 | 0 | 0 | 0.00 h |
| ≥95% | 0 | 0 | 0 | 0.00 h |
| ≥99% | 0 | 0 | 0 | 0.00 h |

## DST

Source timestamps are naive Europe/Berlin wall times. Each spring contains 720 labelled `02:xx` rows that are nonexistent in civil time. Numeric values in those rows are excluded from UTC aggregation and counted explicitly. Each autumn contains one unlabeled 720-row `02:xx` hour with no duplicate source timestamps, so the physical fold cannot be reconstructed from sequence. It is assigned to the first/CEST fold; the second fold remains missing. No rows are silently shifted and no second fold is fabricated.

| Signal/year | Spring numeric rows excluded | Autumn numeric ambiguous rows |
|---|---:|---:|
| total_load 2021 | 551 | 553 |
| total_load 2022 | 646 | 599 |
| total_load 2023 | 549 | 552 |
| total_load 2024 | 551 | 555 |
| pv 2024 | 720 | 0 |

## Exogenous data

SPARK weather is observed historical weather, not historical NWP/forecast data and not point-in-time-safe Day-Ahead input.

Observed weather is available hourly in UTC and is labelled `observed_weather`. Prices and holidays are present only in a legacy aligned artifact; emissions are not present locally. No exogenous dataset is promoted into canonical SILVER or merged into a modelling table by this task.

## Storage

| Category | Bytes | GiB |
|---|---:|---:|
| raw | 9,713,475,218 | 9.046 |
| silver | 156,691,081 | 0.146 |
| reports | 3,460,067 | 0.003 |
| legacy_processed_and_final | 6,240,522 | 0.006 |
| total_project | 9,885,004,669 | 9.206 |

## Canonical status

- **RAW:** `data/raw/*.zip` (immutable official archives).
- **SILVER CANONICAL:** `data/silver/{total_load,pv}/{1min,10min,15min}/*.parquet` with sibling manifests.
- **LEGACY / NONCANONICAL:** `data/processed/` and `data/final/`; do not use these Prompt-1 outputs for thesis modelling.

## Issues to resolve before GOLD modelling

- No 2024 TotalLoad interval reaches 80% coverage, so there is no joint Load/PV interval at ≥80%. A lower reversible threshold or a different experiment design requires explicit scientific justification.
- PV ends on 9 April 2024, which severely limits seasonal joint forecasting.
- Emissions are absent locally; prices and holidays exist only in a legacy combined file.
- Weather units have source metadata/value-scale inconsistencies (humidity, cloud cover, and sunshine duration). Do not silently rescale them.
