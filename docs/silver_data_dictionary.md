# SILVER data dictionary

## Scope and provenance

Canonical files are the Parquet files under `data/silver/`. `total_load` comes
from `EPI_TotalLoad/P_total/YYYY_P_total.csv.xz`; `pv` comes from
`EPI_PV/AC_ActivePower/2024_AC_ActivePower.csv.xz`. Both are official members of
the cleaned KIT SPARK dataset (DOI `10.35097/bjdg3m3rg5jv3skk`). Each 1-, 10-,
and 15-minute output is aggregated directly from the 5-second member—not from a
coarser SILVER file.

Power values are float64. No interpolation, clipping, smoothing, rounding, or
coverage masking is applied. A missing statistic means no valid observation
was available (except sample standard deviation, which is also undefined for a
single observation). Low-coverage statistics remain present and filtering is
left to downstream sensitivity analysis.

## Timestamp semantics

- `timestamp_utc` is the unique, monotonic canonical interval start.
- `timestamp_source_local` preserves the naive Europe/Berlin wall-time label
  corresponding to the source grid. `timestamp_local` is retained as a
  compatibility alias with identical values.
- Spring `02:xx` values are labelled with nonexistent civil times and cannot be
  mapped to UTC; they are excluded and counted in `reports/dst/`.
- The single autumn `02:xx` hour is unlabeled by fold. It is assigned to the
  first/CEST fold. The second fold is not fabricated. Consequently local labels
  repeat on the canonical UTC grid during fall-back; UTC remains unique.

## Fields

| Field | Type/unit | Definition and missing-value meaning |
|---|---|---|
| `timestamp_utc` | timezone-aware timestamp, UTC | Canonical left edge of the interval. |
| `timestamp_local` | naive timestamp, Europe/Berlin | Compatibility local-wall-time interval label. |
| `timestamp_source_local` | naive timestamp, Europe/Berlin | Explicit source-wall-time label; identical to `timestamp_local` for the canonical bin. |
| `p_mean_w` | float64, W | Arithmetic mean of available original 5-second values. |
| `p_std_w` | float64, W | Sample standard deviation (`ddof=1`); missing for fewer than two values. |
| `p_min_w`, `p_max_w` | float64, W | Minimum/maximum of available values; never clipped. |
| `p_median_w` | float64, W | Median of available values. |
| `p_q05_w`, `p_q25_w`, `p_q75_w`, `p_q95_w` | float64, W | Linear empirical quantiles of available values. |
| `p_mean_kw`, `p_max_kw` | float64, kW | Watt fields divided by 1000 without rounding. |
| `energy_kwh_observed` | float64, kWh | `p_mean_w / 1000 × valid_count × 5 / 3600`; energy represented only by observed samples, not gap-scaled. |
| `valid_count` | int32 | Finite source measurements contributing to the interval. |
| `expected_count` | int32 | 12 (1 min), 120 (10 min), or 180 (15 min). |
| `missing_count` | int32 | `expected_count - valid_count`. |
| `coverage` | float64, ratio | `valid_count / expected_count`; never used to erase statistics. |
| `quality_100` | bool | `coverage >= 1.00`. |
| `quality_99` | bool | `coverage >= 0.99`. |
| `quality_95` | bool | `coverage >= 0.95`. |
| `quality_90` | bool | `coverage >= 0.90`. |
| `quality_80` | bool | `coverage >= 0.80`. |
| `duplicate_observations_present` | bool | Flags an interval containing duplicate source wall-time labels if detected. |
| `extreme_high` | bool | Audit flag: interval maximum exceeds yearly `Q3 + 3×IQR`; values are retained unchanged. |

## Signal semantics and limitations

`total_load` is total active power in W; positive means demand. `pv` is inverter
AC active-power output in W; positive means generation. PV exists only in 2024
and is measured only through approximately 9 April. Missing PV is not zero.

The weather artifact under `data/processed/` contains **observed historical
weather**, not historical weather forecasts/NWP. It is legacy/noncanonical and
must not be treated as point-in-time-safe Day-Ahead input. Everything under
`data/processed/` and `data/final/` is a legacy Prompt-1 artifact and is not a
canonical thesis-modelling input. Its delivered values also conflict with some
metadata labels: humidity and cloud cover are roughly scaled to 0–1, while
sunshine duration reaches 3600 despite a `min/h` label (consistent with
seconds/hour). These values are preserved and must not be silently rescaled.
