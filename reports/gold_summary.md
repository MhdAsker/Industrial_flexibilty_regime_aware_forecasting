# GOLD quality summary

## Primary load dataset

The full UTC-indexed 15-minute grid is retained. No target values are interpolated, filled, smoothed, or clipped.

|   year |   rows |   primary |   strict |   empty |   mean_coverage |   segments |   median_duration_hours |   max_duration_hours |
|-------:|-------:|----------:|---------:|--------:|----------------:|-----------:|------------------------:|---------------------:|
|   2021 |  35040 |     34948 |     2488 |      19 |        0.787659 |         51 |                  165    |               883.25 |
|   2022 |  35040 |     34884 |     1653 |     120 |        0.796325 |         34 |                  183.75 |              1234    |
|   2023 |  35040 |     32676 |       14 |    2021 |        0.716636 |         87 |                   48.75 |               409    |
|   2024 |  35136 |     31339 |        0 |    3729 |        0.683045 |         53 |                  148.5  |               319    |

### Chronological splits and target distribution

| split      |   rows |   primary_targets |   strict_targets |   primary_mean_load_kw |   primary_std_load_kw |   primary_min_load_kw |   primary_q25_load_kw |   primary_median_load_kw |   primary_q75_load_kw |   primary_max_load_kw |
|:-----------|-------:|------------------:|-----------------:|-----------------------:|----------------------:|----------------------:|----------------------:|-------------------------:|----------------------:|----------------------:|
| test       |  35136 |             31339 |                0 |                15.3644 |               8.94005 |               3.87409 |               6.30583 |                  12.9243 |               21.9053 |               59.6638 |
| train      |  70080 |             69832 |             4141 |                12.6475 |               8.30785 |               2.78362 |               5.03907 |                  11.4149 |               19.1407 |               52.234  |
| validation |  35040 |             32676 |               14 |                12.9539 |               8.48308 |               2.11239 |               4.46791 |                  11.3417 |               19.0299 |               50.5666 |

The primary rule is coverage >=75% (135/180), selected empirically from the finalized masking study; it is not a heuristic. Missingness is mixed, with a dominant approximately 20-second periodic pattern plus clustered outages.

### Coverage distribution by year

|   year |   min |      p05 |      p25 |   median |      p75 |      p95 |      max |
|-------:|------:|---------:|---------:|---------:|---------:|---------:|---------:|
|   2021 |     0 | 0.761111 | 0.766667 | 0.766667 | 0.772222 | 1        | 1        |
|   2022 |     0 | 0.761111 | 0.766667 | 0.766667 | 0.827778 | 0.911111 | 1        |
|   2023 |     0 | 0        | 0.761111 | 0.761111 | 0.766667 | 0.766667 | 0.988889 |
|   2024 |     0 | 0        | 0.761111 | 0.766667 | 0.766667 | 0.766667 | 0.788889 |

The strict >=95% subset is a sensitivity resource, not a parallel 2021–2024 experiment: it has only 14 intervals in 2023 and none in 2024.

## Feature availability

Calendar and Baden-Württemberg/Karlsruhe holiday features are future-known and safe. Target coverage and segment fields are outcome-quality metadata, not assumed future-known predictors. Contemporaneous observed DWD weather is retained for retrospective/oracle analysis and is explicitly not day-ahead-safe. Day-ahead price availability is conditional on the eventual forecast issuance time because publication timing is not encoded in the source.

No load lags or rolling statistics are included in base GOLD; consequently no hidden window boundary or row-shift assumptions are present. Emissions status: `optional_not_in_primary_gold`.

## PV dataset

The secondary full-grid 2024 PV panel has 35,136 rows and 8,252 intervals at >=95% coverage. Its measured usable range is 2023-12-31T23:00:00+00:00 through 2024-04-08T21:45:00+00:00. The source represents only January through early April measurements and cannot support a full-year PV claim.

## Net-load dataset

The joint dataset has 7,392 intervals after requiring load >=75% and PV >=95%. Net load equals load minus PV after validation that PV is positive generation. Its usable range is 2024-01-05T07:45:00+00:00 through 2024-04-08T21:45:00+00:00.

It contains 62 complete local-calendar days, 3 complete ISO weeks, and a longest contiguous block of 1,084 intervals (271 hours), from 2024-03-28T15:00:00+00:00 through 2024-04-08T21:45:00+00:00.

It is limited exploratory data, not evidence for annual seasonal net-load forecasting. Future measured PV is not permitted as a load-only predictor.

## Leakage audit

- Future-known: deterministic calendar and published holiday variables.
- Past-only: measured load can only become a predictor through explicitly historical timestamps in a later experiment pipeline.
- Retrospective/oracle: contemporaneous observed weather and future measured PV.
- Conditional: day-ahead price, pending a formal forecast issuance time.
- Prohibited for primary load forecasting: contemporaneous/future load, target-quality outcomes, future measured PV, and contemporaneous observed future weather.
