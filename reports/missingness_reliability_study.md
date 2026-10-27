# Missingness and 15-minute aggregation reliability

## Executive decision

Use **P_total coverage >= 75% (at least 135 of 180 samples)** as the primary
15-minute TotalLoad rule. Use **coverage >= 95% (at least 171 samples)** as the
strict sensitivity rule. Coverage alone is sufficient for the primary rule:
adding `longest missing run <= 60 s` and requiring both halves to be observed
changes empirical-mask P95 error only from 0.01935 to 0.01933 kW in the pooled
candidate analysis, while discarding additional intervals.

The 75% choice is an empirical breakpoint, not a sample-retention compromise.
At the dominant ~76% real mask, aggregate MAE is 0.0060 kW and P95 absolute
error is 0.0212 kW. At ~70%, real masks become non-periodic/edge-clustered and
P95 rises to 0.1364 kW (0.2286 kW in high-variability intervals). The typical
full-reference load is 7.4435 kW.

## 1. Data and design

Only the audited official ZIP members
`EPI_TotalLoad/P_total/{2021..2024}_P_total.csv.xz` were streamed. The source
audits establish a complete, monotonic 5-second local timestamp grid, so the
study read the single numeric column and retained every NaN position. No source
member was extracted, no value was imputed, and canonical SILVER was not
modified.

There are 2,420 complete reference intervals: 2,342 in 2021 and 78 in 2022.
The separately labelled >=99% and >=95% tiers are retained in
`reference_intervals_5s.npz`, but only 100% intervals serve as masking ground
truth. Random masks use five fixed seeds. Empirical masks preserve the exact
180-position geometry of real 2022–2024 intervals.

## 2. Missingness mechanism

Missingness is **mixed, but dominated by a systematic 20-second pattern**.
The validity autocorrelation at 20 seconds is 0.574, 0.502, 0.732, and 0.750
for 2021–2024, whereas lags of 5–15 seconds are negative. Median valid-run
length is three samples and median/P95/P99 missing-run length is one sample in
every year. This is consistent with roughly three recorded 5-second positions
followed by one missing position.

Long outages coexist with that periodic mechanism. The longest missing runs
are 5,680 s (2021), 89,610 s (2022), 1,183,795 s (2023), and 2,564,195 s
(2024). Isolated one-sample gaps account for 99.47%, 98.26%, 78.16%, and
66.40% of all missing samples respectively; the falling share in 2023–2024 is
caused by clustered outages.

## 3. Position within the minute

All twelve second positions (00, 05, ..., 55) have nearly identical validity.
Within-year max-minus-min differences are only 0.22, 0.31, 0.23, and 0.18
percentage points. Thus the 70–80% regime is not caused by a uniquely bad
second-of-minute slot; it is a rolling/repeated 20-second acquisition pattern.
See `validity_by_second_of_minute.csv` and the corresponding figure.

## 4. Geometry within 15 minutes

The median interval contains 138/180 valid values (76.67%) in every year. The
median first-half, second-half, and each five-minute-third coverage are also
76.67%, showing that the dominant mask is distributed across the interval
(Case A), not simply a missing final block (Case B). Median first-valid position
is zero, median last-valid position is 179, and the median interval has 42
one-sample missing runs.

Outage intervals form the exception: P95 longest within-interval missing run is
180 samples in 2023–2024 because fully missing intervals are common. Full
per-year quantiles are in `within_interval_summary.csv`.

## 5. Dependence on time and observed load

Time-of-day dependence is small: the range of hourly mean coverage is 0.58,
0.58, 0.46, and 0.47 percentage points for 2021–2024. Weekday/weekend mean
coverage differs by at most 0.83 percentage points.

Coverage has negative Spearman association with observed interval mean load
(-0.253, -0.246, -0.301, -0.325 by year) and max load (-0.248, -0.236,
-0.289, -0.324). This does **not** establish causation: observed power is itself
computed from partial samples, operating regimes are temporally structured,
and fully missing intervals have no observed load. Among nonempty intervals,
load-quintile mean coverage is much flatter, especially in 2023–2024
(76.21–76.56%). The safe conclusion is a detectable association and possible
bias risk, not a load-triggered failure mechanism.

## 6. Controlled masking results

Absolute errors below are in kW. Empirical estimates use real masks within two
samples of the requested coverage; `n` is reported because real SPARK masks are
strongly concentrated near 76%, leaving sparse support at some nominal levels.

| Coverage | Random MAE | Random P95 | Empirical n | Empirical actual coverage | Empirical MAE | Empirical P95 | Empirical P99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 95% | 0.00879 | 0.03121 | 37 | 94.865% | 0.00715 | 0.02645 | 0.04457 |
| 90% | 0.01269 | 0.04596 | 1,214 | 90.198% | 0.00509 | 0.01818 | 0.04519 |
| 80% | 0.01873 | 0.06653 | 7 | 79.365% | 0.01123 | 0.03632 | 0.04113 |
| 75% | 0.02164 | 0.07734 | 3,600 | 76.037% | 0.00600 | 0.02119 | 0.04953 |
| 70% | 0.02465 | 0.08715 | 69 | 70.491% | 0.04158 | 0.13644 | 0.26875 |
| 60% | 0.03078 | 0.10992 | 5 | 59.667% | 0.06224 | 0.13027 | 0.14239 |

Random error rises smoothly as coverage falls. Real masks behave much better
than random masks at the dominant 75% level because samples span the interval;
the sharp deterioration at 70% reflects different, clustered mask geometry.
Empirical results at 95%, 80%, and 60% are exploratory because `n` is small.

## 7. 2023/2024 mask transfer

At the dominant 75% band, 1,200 masks from each source year were transferred
separately to each reference year:

| Mask source -> reference | Actual coverage | MAE kW | P95 kW | P99 kW |
|---|---:|---:|---:|---:|
| 2023 -> 2021 | 75.999% | 0.00594 | 0.02032 | 0.04674 |
| 2023 -> 2022 | 75.999% | 0.00660 | 0.02461 | 0.03744 |
| 2024 -> 2021 | 76.057% | 0.00622 | 0.02385 | 0.05897 |
| 2024 -> 2022 | 76.057% | 0.00614 | 0.02344 | 0.03812 |

This direct transfer supports use of the dominant 2023/2024 partial intervals.
At ~70%, the 2024 masks are riskier: P95 is 0.1611 kW on 2021 references and
0.2138 kW on 2022 references (31 masks each).

## 8. High-variability intervals

Reference intervals were split into tertiles by full-data 5-second standard
deviation. Under empirical masks, high-variability P95 errors are 0.0444 kW at
95%, 0.0387 kW at 90%, 0.0388 kW at the dominant 75%, and 0.2286 kW at 70%.
The 75% rule therefore remains reliable during volatile intervals; 70% does
not show comparable robustness. The 80% high-variability result (0.0413 kW)
has only two observations and is not independently persuasive.

## 9. Short-gap interpolation sensitivity

For empirical 75% masks, available-sample means have MAE/P95 of
0.00654/0.02278 kW in the sensitivity subsample. Filling isolated <=5-second
internal gaps changes this only to 0.00600/0.02009 kW; allowing 10–30 seconds
adds no further benefit because the empirical gaps are mostly isolated.
Interpolation can help artificial random masks, but those masks do not reflect
the dominant SPARK mechanism. The recommended target therefore remains
non-imputed.

## 10. Threshold decision table

| Threshold | 2021 | 2022 | 2023 | 2024 | Random MAE kW | Empirical MAE kW | Empirical P95 kW | High-var empirical P95 kW |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | 34,998 | 34,916 | 32,850 | 31,395 | 0.03768 | 0.25352 | 0.73746 | 0.84994 |
| 60% | 34,992 | 34,914 | 32,843 | 31,389 | 0.03078 | 0.06224 | 0.13027 | 0.14542 |
| 65% | 34,986 | 34,913 | 32,839 | 31,381 | 0.02738 | 0.03031 | 0.08099 | 0.08991 |
| 70% | 34,980 | 34,912 | 32,831 | 31,372 | 0.02465 | 0.04158 | 0.13644 | 0.22862 |
| **75%** | **34,948** | **34,884** | **32,676** | **31,339** | **0.02164** | **0.00600** | **0.02119** | **0.03881** |
| 80% | 5,454 | 10,663 | 84 | 0 | 0.01873 | 0.01123 | 0.03632 | 0.04133 |
| 90% | 2,493 | 4,686 | 70 | 0 | 0.01269 | 0.00509 | 0.01818 | 0.03867 |
| 95% | 2,488 | 1,653 | 14 | 0 | 0.00879 | 0.00715 | 0.02645 | 0.04440 |
| 99% | 2,344 | 78 | 0 | 0 | 0.00401 | 0.00374 | 0.01322 | 0.02862 |

The apparent non-monotonic empirical values arise because masks at different
coverages are different real mechanisms and sample sizes, not nested random
subsamples. This is exactly why the empirical-mask experiment matters.

## 11. Recommended rules and forecasting windows

- Primary: coverage >=75% (valid_count >=135), no interpolation.
- Strict sensitivity: coverage >=95% (valid_count >=171).
- Exploratory lower rule: none recommended. The ~70% masks show a clear error
  and geometry change, and the 60–65% empirical support is too sparse.

Best primary-rule contiguous windows (UTC) are:

| Year | Start | End | Intervals | Hours |
|---:|---|---|---:|---:|
| 2021 | 2021-05-05 04:45 | 2021-06-10 23:45 | 3,533 | 883.25 |
| 2022 | 2022-02-18 07:30 | 2022-04-10 17:15 | 4,936 | 1,234.00 |
| 2023 | 2023-01-17 10:15 | 2023-02-03 11:00 | 1,636 | 409.00 |
| 2024 | 2024-08-28 18:00 | 2024-09-11 00:45 | 1,276 | 319.00 |

All top-five windows are in `forecasting_windows.json`. The strict rule is not
usable as a common four-year modelling filter because it retains only 14
intervals in 2023 and none in 2024.

## 12. Load/PV implication

With TotalLoad >=75% and PV >=95%, 2024 has 7,392 jointly usable intervals,
62 complete local-calendar days, three complete ISO weeks, and a longest
contiguous block of 1,084 intervals (271 hours), from 2024-03-28 15:00 UTC to
2024-04-08 21:45 UTC. This supports a limited early-2024 net-load experiment,
not a full-year or multi-season PV/net-load claim.

## 13. Exogenous resolution

The official 8.32 MB `01exogenous_data.zip` was checksum-validated. Original
price, holiday, weather, and station files were copied into
`data/silver/exogenous/source`; 2021–2024 canonical Parquets were built directly
from them with no filling or value conversion. Prices are UTC EUR/MWh at 15
minutes; holidays are Europe/Berlin local dates; weather is hourly UTC. The
emissions file is present in the archive and catalogued but, as optional, was
not extracted. Humidity, cloud-cover, and sunshine units are resolved in
`reports/weather_unit_resolution.md`.

## 14. Suitability and limitations

- Industrial load forecasting: suitable under the >=75% target rule, with the
  quality mask retained for auditing and sample weighting/sensitivity checks.
- Regime-aware forecasting: suitable, including volatile regimes, but report
  the observed coverage/load association and test conclusions at >=95% where
  the year supports it.
- PV forecasting: suitable only for the measured 2024-01-01 to 2024-04-09
  period; not for a full annual/seasonal claim.
- Joint net-load forecasting: suitable only as a limited early-2024 experiment.

There is no remaining missingness-reliability reason to block a load-only GOLD
table using the primary rule. GOLD must still preserve target-quality flags,
must not treat observed weather as point-in-time-safe day-ahead forecasts, and
must not imply full-year PV coverage. The sparse empirical support away from
the dominant ~76% mask is a remaining limitation, and is why no lower
exploratory threshold is recommended.

## Reproducibility products

The script `src/missingness_reliability.py` generates all CSV/JSON/NPZ products
and six figures under `reports/missingness/`; fixed seeds are recorded in
`study_results.json`. Tests cover deterministic masks, mask geometry,
short-gap interpolation, longest runs, and composite rules. Canonical SILVER
files are read-only inputs.
