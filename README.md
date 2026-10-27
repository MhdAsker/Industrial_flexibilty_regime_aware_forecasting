# KIT SPARK auditable silver-data pipeline

This repository converts the official 5-second KIT SPARK measurements into
UTC-indexed 1-, 10-, and 15-minute silver datasets. It does not impute, clip,
smooth, or train forecasting models.

## Immutable inputs

Place the official archives at:

- `data/raw/EPI_TotalLoad.zip` (9,701,876,967 bytes; MD5
  `cbbffe29f9d8e224b9221cb5904a00bb`)
- `data/raw/EPI_PV.zip` (3,272,618 bytes; MD5
  `f34a321b3aee4beca9864c09c68b2e36`)

The pipeline validates the official size and MD5 and never modifies either
archive. It streams only the selected `.csv.xz` member through ZIP, XZ and
chunked CSV readers; it never permanently extracts an archive. A
size/mtime-bound `.verified.json` sidecar prevents repeated hashing of an
unchanged multi-gigabyte archive.

If an official archive is genuinely absent, it can be acquired safely with:

```powershell
python src/download_raw.py total_load
python src/download_raw.py pv
```

An existing archive is validated and never overwritten. A transfer is written
to `.part`, checked for size and MD5, and atomically renamed only after it
passes.

## Processing

```powershell
python src/resample_total_load.py
python src/resample_pv.py
python src/audit_dst.py
python src/analyze_missingness.py
python src/validate_silver.py
python src/missingness_reliability.py
python src/canonicalize_exogenous.py
python src/build_gold.py
python -m pytest -q
```

Each year is processed separately in float64. Every resolution is generated
directly from the original 5-second values. Output Parquet files are Zstandard
compressed and organized under `data/silver/{signal}/{resolution}/`. A sibling
manifest binds each output to the archive MD5, ZIP-member CRC and sizes, full
configuration, processing-code SHA256, output SHA256, and row count. A rerun
skips only outputs whose identities and hashes still match.

## Silver schema and energy

Every interval retains `timestamp_utc` and the corresponding naive
`timestamp_local`, plus mean, sample standard deviation, min, max, median,
q05/q25/q75/q95, valid/expected/missing counts, coverage, quality flags at
100/99/95/90/80%, duplicate and extreme flags, and `p_mean_kw`/`p_max_kw`.
Low-coverage statistics are retained unchanged.

`energy_kwh_observed` is the rectangular-rule energy represented only by valid
5-second samples:

`p_mean_w / 1000 * valid_count * 5 / 3600`.

It is not scaled to a full interval, so missing observations are neither
imputed nor treated as zero.

## DST policy

The source timestamps are naive Europe/Berlin wall times. Spring timestamps in
the nonexistent 02:00 hour cannot map to UTC and are flagged and excluded from
canonical UTC aggregation. The source contains only one unlabelled autumn
02:00 hour, so its physical fold is unrecoverable. Configuration explicitly
assigns it to the first (CEST) fold; the second fold remains missing. Both the
decision and affected row counts are recorded under `reports/dst/`. No
observation is fabricated.

## GOLD research datasets

`python src/build_gold.py` creates the full-grid load panel and its primary and
strict filtered views, plus the limited-period 2024 PV and joint net-load
datasets under `data/gold/`. The primary load rule is the empirically validated
75% coverage threshold (135/180 samples); the strict 95% view is only a
sensitivity resource. GOLD never imputes targets or exogenous data, preserves
UTC gaps and contiguous-segment metadata, and uses chronological splits:
2021-2022 train, 2023 validation, and 2024 test.

Observed weather is retained only as a clearly marked retrospective/oracle
field and is not day-ahead-safe. Day-ahead price eligibility remains
conditional on the experiment's forecast issuance time. See
`docs/gold_feature_availability.md`, `docs/gold_data_dictionary.md`,
`reports/gold_summary.md`, and `reports/gold_manifest.json` for the full
leakage and provenance audit.

## Reports

- `reports/raw_quality/`: row-level source audits by signal and year
- `reports/dst/`: DST observations and explicit resolution policy
- `reports/missingness_report.json`: availability, complete periods, gap
  thresholds, contiguous windows, PV daylight analysis and load/PV overlap
- `reports/figures/`: 15-minute availability heatmaps

Configuration, paths, selected measurements, years, resolutions and thresholds
are centralized in `config/pipeline_config.json`.

## Canonical status

- `data/raw/`: **RAW**, immutable official archives.
- `data/silver/`: **SILVER CANONICAL**, thesis-ready audited aggregates.
- `data/processed/` and `data/final/`: **LEGACY / NONCANONICAL** Prompt-1
  artifacts. They are preserved but must not be used for thesis modelling.

The final validation report is `reports/silver_summary.md`, with machine-readable
details in `reports/silver_validation.json`, `reports/missingness_report.json`,
and `reports/exogenous_audit.json`. Field-level provenance is documented in
`docs/silver_data_dictionary.md` and `.csv`.

SPARK weather is **observed historical weather**, not historical forecasts/NWP.
It must not be treated as point-in-time-safe Day-Ahead input.

## Missingness reliability decision

The empirical masking study in `reports/missingness_reliability_study.md`
supports a primary TotalLoad rule of at least 75% within-interval coverage
(135/180 samples), with 95% as a strict sensitivity threshold. The dominant
missingness mechanism is a distributed 20-second acquisition pattern; error
rises sharply for the different, clustered masks near 70% coverage. Canonical
SILVER remains unchanged.

Official price, holiday, and weather sources are now canonicalized separately
under `data/silver/exogenous/`. Weather-unit conflicts and their evidence are
documented in `reports/weather_unit_resolution.md`; delivered source values are
preserved unchanged.
