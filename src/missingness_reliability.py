"""Empirical reliability study for 15-minute means from SPARK 5-second P_total.

The script reads only the audited P_total members directly from the official ZIP,
never extracts them, never changes SILVER, and writes compact analysis products.
"""

from __future__ import annotations

import argparse
import io
import json
import lzma
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from timeseries_core import ROOT, load_config


YEARS = (2021, 2022, 2023, 2024)
TARGET_COVERAGES = (0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50)
DECISION_THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99)
SEEDS = (1729, 2718, 3141, 5772, 8119)
REPORT_DIR = ROOT / "reports" / "missingness"
FIGURE_DIR = REPORT_DIR / "figures"


def longest_true_run(rows: np.ndarray) -> np.ndarray:
    """Longest consecutive True run for each row of a 2-D boolean array."""
    rows = np.asarray(rows, dtype=bool)
    if rows.ndim != 2:
        raise ValueError("rows must be two-dimensional")
    current = np.zeros(rows.shape[0], dtype=np.int16)
    longest = np.zeros(rows.shape[0], dtype=np.int16)
    for column in rows.T:
        current = np.where(column, current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def mask_features(valid: np.ndarray) -> pd.DataFrame:
    """Describe spatial distribution of observations within 180-position masks."""
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim == 1:
        valid = valid[None, :]
    if valid.ndim != 2 or valid.shape[1] != 180:
        raise ValueError("valid masks must have shape (n, 180)")
    count = valid.sum(axis=1).astype(np.int16)
    missing = ~valid
    starts = missing & np.column_stack((np.ones(len(valid), dtype=bool), ~missing[:, :-1]))
    number_missing_runs = starts.sum(axis=1).astype(np.int16)
    any_valid = count > 0
    first = np.where(any_valid, valid.argmax(axis=1), -1).astype(np.int16)
    last = np.where(any_valid, 179 - valid[:, ::-1].argmax(axis=1), -1).astype(np.int16)
    return pd.DataFrame({
        "valid_count": count,
        "coverage": count / 180.0,
        "longest_missing_run_samples": longest_true_run(missing),
        "number_missing_runs": number_missing_runs,
        "first_valid_position": first,
        "last_valid_position": last,
        "fraction_first_half": valid[:, :90].mean(axis=1),
        "fraction_second_half": valid[:, 90:].mean(axis=1),
        "fraction_third_1": valid[:, :60].mean(axis=1),
        "fraction_third_2": valid[:, 60:120].mean(axis=1),
        "fraction_third_3": valid[:, 120:].mean(axis=1),
    })


def random_masks(n_rows: int, coverage: float, seed: int) -> np.ndarray:
    """Return deterministic masks with exactly round(180*coverage) valid positions."""
    keep = int(round(180 * coverage))
    if not 0 < keep <= 180:
        raise ValueError("coverage must retain between 1 and 180 samples")
    random = np.random.default_rng(seed).random((n_rows, 180))
    chosen = np.argpartition(random, keep - 1, axis=1)[:, :keep]
    masks = np.zeros((n_rows, 180), dtype=bool)
    masks[np.arange(n_rows)[:, None], chosen] = True
    return masks


def apply_quality_rule(features: pd.DataFrame, coverage_min: float,
                       longest_missing_run_max: int | None = None,
                       require_both_halves: bool = False) -> pd.Series:
    keep = features["coverage"].ge(coverage_min)
    if longest_missing_run_max is not None:
        keep &= features["longest_missing_run_samples"].le(longest_missing_run_max)
    if require_both_halves:
        keep &= features["fraction_first_half"].gt(0) & features["fraction_second_half"].gt(0)
    return keep


def interpolate_short_internal_gaps(values: np.ndarray, max_gap_samples: int) -> np.ndarray:
    """Linearly fill only internal NaN runs no longer than max_gap_samples."""
    result = np.asarray(values, dtype=np.float64).copy()
    for row in result:
        missing = np.isnan(row)
        if not missing.any():
            continue
        padded = np.r_[False, missing, False].astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends, strict=True):
            if start > 0 and end < len(row) and end - start <= max_gap_samples:
                row[start:end] = np.linspace(row[start - 1], row[end], end - start + 2)[1:-1]
    return result


def error_summary(frame: pd.DataFrame, typical_load_kw: float) -> dict:
    error = frame["error_kw"].to_numpy(dtype=float)
    absolute = np.abs(error)
    truth = frame["true_mean_kw"].to_numpy(dtype=float)
    stable = np.abs(truth) >= max(0.1 * typical_load_kw, 0.1)
    relative = absolute[stable] / np.abs(truth[stable]) if stable.any() else np.array([])
    return {
        "n": int(len(frame)),
        "mean_bias_kw": float(np.mean(error)),
        "median_bias_kw": float(np.median(error)),
        "mae_kw": float(np.mean(absolute)),
        "rmse_kw": float(np.sqrt(np.mean(error ** 2))),
        "p50_absolute_error_kw": float(np.quantile(absolute, 0.50)),
        "p90_absolute_error_kw": float(np.quantile(absolute, 0.90)),
        "p95_absolute_error_kw": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error_kw": float(np.quantile(absolute, 0.99)),
        "max_absolute_error_kw": float(np.max(absolute)),
        "mae_percent_typical_load": float(100 * np.mean(absolute) / typical_load_kw),
        "p95_percent_typical_load": float(100 * np.quantile(absolute, 0.95) / typical_load_kw),
        "median_relative_error_percent_stable": float(100 * np.median(relative)) if len(relative) else math.nan,
        "p95_relative_error_percent_stable": float(100 * np.quantile(relative, 0.95)) if len(relative) else math.nan,
        "relative_error_denominator_rule": "abs(true mean) >= max(10% of typical load, 0.1 kW)",
    }


def summarize_groups(frame: pd.DataFrame, groups: list[str], typical_load_kw: float) -> pd.DataFrame:
    records = []
    for keys, group in frame.groupby(groups, observed=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(groups, keys, strict=True))
        record.update(error_summary(group, typical_load_kw))
        record["actual_coverage_mean"] = float(group["actual_coverage"].mean())
        records.append(record)
    return pd.DataFrame(records)


def run_lengths_1d(binary: np.ndarray, target: bool) -> np.ndarray:
    selected = np.asarray(binary, dtype=bool) == target
    padded = np.r_[False, selected, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return ends - starts


def observed_statistics(values: np.ndarray, valid: np.ndarray) -> pd.DataFrame:
    count = valid.sum(axis=1)
    total = np.where(valid, values, 0).sum(axis=1, dtype=np.float64)
    mean = np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)
    maximum = np.where(valid, values, -np.inf).max(axis=1)
    minimum = np.where(valid, values, np.inf).min(axis=1)
    maximum[count == 0] = np.nan
    minimum[count == 0] = np.nan
    centered = np.where(valid, values - mean[:, None], 0.0)
    variance = np.divide((centered ** 2).sum(axis=1), count - 1,
                         out=np.full(len(values), np.nan), where=count > 1)
    return pd.DataFrame({
        "observed_mean_kw": mean / 1000.0,
        "observed_max_kw": maximum / 1000.0,
        "observed_std_kw": np.sqrt(variance) / 1000.0,
        "observed_range_kw": (maximum - minimum) / 1000.0,
    })


def read_year_values(year: int) -> np.ndarray:
    cfg = load_config()
    signal = cfg["signals"]["total_load"]
    archive = ROOT / signal["archive"]
    member = signal["member_template"].format(year=year)
    pieces: list[np.ndarray] = []
    with zipfile.ZipFile(archive) as zf, zf.open(member) as zipped, lzma.LZMAFile(zipped) as decompressed:
        for chunk in pd.read_csv(decompressed, usecols=[signal["measurement"]],
                                 dtype={signal["measurement"]: "float32"},
                                 chunksize=cfg["chunk_rows"]):
            pieces.append(chunk[signal["measurement"]].to_numpy(dtype=np.float32, copy=True))
    values = np.concatenate(pieces)
    expected = (pd.Timestamp(f"{year + 1}-01-01") - pd.Timestamp(f"{year}-01-01")).days * 17280
    if len(values) != expected:
        raise RuntimeError(f"{year}: {len(values)} source rows != {expected} expected 5-second positions")
    return values.reshape(-1, 180)


def canonicalize_features(year: int, local_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = ROOT / "data" / "silver" / "total_load" / "15min" / f"total_load_{year}_15min.parquet"
    silver = pd.read_parquet(path, columns=["timestamp_utc", "timestamp_local", "valid_count", "coverage",
                                            "p_mean_kw", "p_max_kw", "p_std_w"])
    start = pd.Timestamp(f"{year}-01-01")
    positions = ((silver["timestamp_local"] - start) / pd.Timedelta(minutes=15)).astype(int)
    canonical = local_features.iloc[positions.to_numpy()].reset_index(drop=True).copy()
    duplicate_local = silver["timestamp_local"].duplicated(keep="first")
    if duplicate_local.any():
        canonical.loc[duplicate_local, ["valid_count", "coverage", "fraction_first_half",
                                        "fraction_second_half", "fraction_third_1", "fraction_third_2",
                                        "fraction_third_3"]] = 0
        canonical.loc[duplicate_local, ["first_valid_position", "last_valid_position"]] = -1
        canonical.loc[duplicate_local, "longest_missing_run_samples"] = 180
        canonical.loc[duplicate_local, "number_missing_runs"] = 1
    if not np.array_equal(canonical["valid_count"].to_numpy(), silver["valid_count"].to_numpy()):
        mismatch = int((canonical["valid_count"].to_numpy() != silver["valid_count"].to_numpy()).sum())
        raise RuntimeError(f"{year}: {mismatch} canonical feature counts disagree with validated SILVER")
    canonical.insert(0, "timestamp_local", silver["timestamp_local"].to_numpy())
    canonical.insert(0, "timestamp_utc", silver["timestamp_utc"].to_numpy())
    return canonical, silver


def create_error_rows(reference: np.ndarray, masks: np.ndarray, reference_meta: pd.DataFrame,
                      method: str, target_coverage: float, seed: int, mask_source_year: int | str,
                      rng: np.random.Generator | None = None) -> pd.DataFrame:
    if len(masks) != len(reference):
        if rng is None:
            rng = np.random.default_rng(seed)
        choice = rng.integers(0, len(reference), size=len(masks))
        reference = reference[choice]
        reference_meta = reference_meta.iloc[choice].reset_index(drop=True)
    true = reference.mean(axis=1, dtype=np.float64) / 1000.0
    count = masks.sum(axis=1)
    partial = np.divide(np.where(masks, reference, 0).sum(axis=1, dtype=np.float64), count,
                        out=np.full(len(masks), np.nan), where=count > 0) / 1000.0
    features = mask_features(masks)
    out = pd.DataFrame({
        "method": method,
        "target_coverage": target_coverage,
        "actual_coverage": count / 180.0,
        "seed": seed,
        "mask_source_year": mask_source_year,
        "reference_year": reference_meta["year"].to_numpy(),
        "variability_group": reference_meta["variability_group"].to_numpy(),
        "true_mean_kw": true,
        "partial_mean_kw": partial,
        "error_kw": partial - true,
    })
    return pd.concat([out, features.drop(columns=["valid_count", "coverage"])], axis=1)


def sample_empirical_masks(masks: np.ndarray, coverage: float, maximum: int, seed: int,
                           tolerance_samples: int = 2) -> tuple[np.ndarray, int]:
    target = int(round(coverage * 180))
    counts = masks.sum(axis=1)
    distance = np.abs(counts - target)
    nearest = int(distance.min())
    permitted = min(tolerance_samples, nearest) if nearest > tolerance_samples else tolerance_samples
    candidates = np.flatnonzero(distance <= permitted)
    rng = np.random.default_rng(seed)
    if len(candidates) > maximum:
        candidates = rng.choice(candidates, maximum, replace=False)
    return masks[candidates], nearest


def interval_run_summary(features: pd.DataFrame, year: int) -> list[dict]:
    numeric = [c for c in features.columns if c not in {"timestamp_local", "timestamp_utc"}]
    records = []
    for column in numeric:
        values = features[column].to_numpy(dtype=float)
        records.append({
            "year": year, "metric": column, "mean": float(np.nanmean(values)),
            "p50": float(np.nanquantile(values, .50)), "p90": float(np.nanquantile(values, .90)),
            "p95": float(np.nanquantile(values, .95)), "p99": float(np.nanquantile(values, .99)),
            "max": float(np.nanmax(values)),
        })
    return records


def longest_windows(timestamps: pd.Series, usable: np.ndarray, top_n: int = 5) -> list[dict]:
    starts = np.flatnonzero(usable & np.r_[True, ~usable[:-1]])
    ends = np.flatnonzero(usable & np.r_[~usable[1:], True])
    rows = []
    for start, end in zip(starts, ends, strict=True):
        rows.append({"start_utc": timestamps.iloc[start].isoformat(),
                     "end_utc": timestamps.iloc[end].isoformat(),
                     "intervals": int(end - start + 1),
                     "hours": float((end - start + 1) / 4)})
    return sorted(rows, key=lambda x: x["intervals"], reverse=True)[:top_n]


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@dataclass
class StudyData:
    masks: dict[int, np.ndarray]
    local_features: dict[int, pd.DataFrame]
    canonical_features: dict[int, pd.DataFrame]
    silver: dict[int, pd.DataFrame]
    reference_values: np.ndarray
    reference_meta: pd.DataFrame
    tier_values: np.ndarray
    tier_meta: pd.DataFrame


def load_and_characterize() -> tuple[StudyData, dict]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    masks: dict[int, np.ndarray] = {}
    local_features: dict[int, pd.DataFrame] = {}
    canonical_features: dict[int, pd.DataFrame] = {}
    silver_frames: dict[int, pd.DataFrame] = {}
    second_records, periodicity_records, within_records = [], [], []
    time_records, load_records, run_records = [], [], []
    complete_values, complete_meta, tier_values, tier_meta = [], [], [], []

    for year in YEARS:
        print(f"Reading audited P_total {year}", flush=True)
        values = read_year_values(year)
        valid = np.isfinite(values)
        masks[year] = valid.copy()
        features = pd.concat([mask_features(valid), observed_statistics(values, valid)], axis=1)
        local_index = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="15min", inclusive="left")
        features.insert(0, "timestamp_local", local_index)
        local_features[year] = features
        canonical, silver = canonicalize_features(year, features.drop(columns="timestamp_local"))
        canonical_features[year] = canonical
        silver_frames[year] = silver
        within_records.extend(interval_run_summary(features, year))

        positions = valid.reshape(-1, 12)
        for pos, rate in enumerate(positions.mean(axis=0)):
            second_records.append({"year": year, "second": pos * 5, "valid_probability": float(rate),
                                   "valid_count": int(positions[:, pos].sum()), "total_count": len(positions)})
        flat = valid.ravel()
        centered = flat.astype(np.float64) - flat.mean()
        denom = float(np.dot(centered, centered))
        for lag in (1, 2, 3, 4, 6, 12):
            ac = float(np.dot(centered[:-lag], centered[lag:]) / denom)
            periodicity_records.append({"year": year, "lag_samples": lag, "lag_seconds": lag * 5,
                                        "autocorrelation": ac})
        missing_runs = run_lengths_1d(flat, False)
        valid_runs = run_lengths_1d(flat, True)
        run_records.append({
            "year": year, "valid_probability": float(flat.mean()),
            "missing_runs": int(len(missing_runs)),
            "isolated_missing_runs": int((missing_runs == 1).sum()),
            "isolated_missing_fraction_of_missing_samples": float((missing_runs == 1).sum() / missing_runs.sum()),
            "missing_run_p50_samples": float(np.quantile(missing_runs, .5)),
            "missing_run_p95_samples": float(np.quantile(missing_runs, .95)),
            "missing_run_p99_samples": float(np.quantile(missing_runs, .99)),
            "longest_missing_run_samples": int(missing_runs.max()),
            "longest_missing_run_seconds": int(missing_runs.max() * 5),
            "valid_run_p50_samples": float(np.quantile(valid_runs, .5)),
            "valid_run_p95_samples": float(np.quantile(valid_runs, .95)),
        })

        by_hour = features.assign(hour=local_index.hour).groupby("hour", observed=True)
        for hour, group in by_hour:
            time_records.append({"year": year, "dimension": "hour", "value": int(hour),
                                 "intervals": len(group), "mean_coverage": float(group.coverage.mean()),
                                 "median_coverage": float(group.coverage.median())})
        weekend = local_index.dayofweek >= 5
        for label, selector in (("weekday", ~weekend), ("weekend", weekend)):
            group = features.loc[selector]
            time_records.append({"year": year, "dimension": "day_type", "value": label,
                                 "intervals": len(group), "mean_coverage": float(group.coverage.mean()),
                                 "median_coverage": float(group.coverage.median())})
        correlations = features[["coverage", "observed_mean_kw", "observed_max_kw", "observed_std_kw",
                                 "observed_range_kw"]].corr(method="spearman")["coverage"]
        for variable, correlation in correlations.items():
            if variable != "coverage":
                load_records.append({"year": year, "analysis": "spearman", "group": variable,
                                     "intervals": int(features[["coverage", variable]].dropna().shape[0]),
                                     "mean_coverage": math.nan, "value": float(correlation)})
        valid_means = features["observed_mean_kw"].notna()
        if valid_means.sum() >= 5:
            bins = pd.qcut(features.loc[valid_means, "observed_mean_kw"], 5, labels=False, duplicates="drop")
            for band, indices in bins.groupby(bins).groups.items():
                group = features.loc[indices]
                load_records.append({"year": year, "analysis": "observed_load_quintile",
                                     "group": int(band) + 1, "intervals": len(group),
                                     "mean_coverage": float(group.coverage.mean()),
                                     "value": float(group.observed_mean_kw.mean())})

        if year in (2021, 2022):
            for threshold in (1.00, .99, .95):
                selected = features.valid_count.ge(math.ceil(threshold * 180))
                if threshold == 1.00:
                    arr = values[selected.to_numpy()].copy()
                    complete_values.append(arr)
                    complete_meta.append(pd.DataFrame({"year": year,
                                                       "timestamp_local": local_index[selected],
                                                       "coverage_tier": "100"}))
                selected_tier = values[selected.to_numpy()].copy()
                tier_values.append(selected_tier)
                tier_meta.append(pd.DataFrame({"year": year, "timestamp_local": local_index[selected],
                                               "coverage_tier": f">={int(threshold * 100)}"}))
        del values

    reference_values = np.concatenate(complete_values).astype(np.float32)
    reference_meta = pd.concat(complete_meta, ignore_index=True)
    std = reference_values.std(axis=1, ddof=1) / 1000.0
    ranges = (reference_values.max(axis=1) - reference_values.min(axis=1)) / 1000.0
    slope = np.polyfit(np.arange(180), reference_values.T, 1)[0] * 12 / 1000.0
    reference_meta["std_kw"] = std
    reference_meta["range_kw"] = ranges
    reference_meta["slope_kw_per_min"] = slope
    reference_meta["variability_group"] = pd.qcut(std, 3, labels=["low", "medium", "high"])
    tiers_array = np.concatenate(tier_values).astype(np.float32)
    tiers_meta = pd.concat(tier_meta, ignore_index=True)
    np.savez_compressed(REPORT_DIR / "reference_intervals_5s.npz", values_w=tiers_array,
                        year=tiers_meta.year.to_numpy(),
                        timestamp_local_ns=tiers_meta.timestamp_local.astype("int64").to_numpy(),
                        coverage_tier=tiers_meta.coverage_tier.to_numpy(dtype="U5"))

    pd.DataFrame(second_records).to_csv(REPORT_DIR / "validity_by_second_of_minute.csv", index=False)
    pd.DataFrame(periodicity_records).to_csv(REPORT_DIR / "missingness_periodicity.csv", index=False)
    pd.DataFrame(within_records).to_csv(REPORT_DIR / "within_interval_summary.csv", index=False)
    pd.DataFrame(time_records).to_csv(REPORT_DIR / "time_dependence.csv", index=False)
    pd.DataFrame(load_records).to_csv(REPORT_DIR / "load_dependence.csv", index=False)
    pd.DataFrame(run_records).to_csv(REPORT_DIR / "missing_run_summary.csv", index=False)

    second = pd.DataFrame(second_records)
    for year, group in second.groupby("year"):
        plt.plot(group.second, group.valid_probability, marker="o", label=str(year))
    plt.xlabel("Second within minute")
    plt.ylabel("P(valid)")
    plt.ylim(0, 1)
    plt.legend()
    save_plot(FIGURE_DIR / "validity_by_second_of_minute.png")
    data = StudyData(masks, local_features, canonical_features, silver_frames,
                     reference_values, reference_meta, tiers_array, tiers_meta)
    tier_counts = [{"year": int(year), "coverage_tier": str(tier), "intervals": int(count)}
                   for (year, tier), count in tiers_meta.groupby(["year", "coverage_tier"]).size().items()]
    meta = {"reference_100_intervals": int(len(reference_values)),
            "reference_tiers": tier_counts,
            "run_summary": run_records}
    return data, meta


def masking_experiments(data: StudyData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    ref = data.reference_values
    meta = data.reference_meta
    typical = float(np.median(ref.mean(axis=1)) / 1000.0)
    random_rows, empirical_rows, transfer_rows = [], [], []
    for coverage in TARGET_COVERAGES:
        for seed in SEEDS:
            masks = random_masks(len(ref), coverage, seed)
            random_rows.append(create_error_rows(ref, masks, meta, "random", coverage, seed, "random"))
        for source_year in (2022, 2023, 2024):
            masks, nearest = sample_empirical_masks(data.masks[source_year], coverage, 1200,
                                                     seed=source_year * 1000 + int(coverage * 100))
            if nearest <= 2:
                empirical_rows.append(create_error_rows(ref, masks, meta, "empirical", coverage,
                                                         source_year, source_year))
            for ref_year in (2021, 2022):
                selection = meta.year.eq(ref_year).to_numpy()
                if not selection.any():
                    continue
                transfer_rows.append(create_error_rows(ref[selection], masks, meta.loc[selection].reset_index(drop=True),
                                                       "empirical_transfer", coverage, source_year + ref_year,
                                                       source_year))
    random_detail = pd.concat(random_rows, ignore_index=True)
    empirical_detail = pd.concat(empirical_rows, ignore_index=True)
    transfer_detail = pd.concat(transfer_rows, ignore_index=True)
    detail = pd.concat([random_detail, empirical_detail], ignore_index=True)
    summary = summarize_groups(detail, ["method", "target_coverage", "variability_group"], typical)
    overall = summarize_groups(detail, ["method", "target_coverage"], typical)
    combined = pd.concat([overall.assign(variability_group="all"), summary], ignore_index=True)
    combined.to_csv(REPORT_DIR / "masking_error_summary.csv", index=False)
    transfer_by_variability = summarize_groups(
        transfer_detail, ["mask_source_year", "reference_year", "target_coverage", "variability_group"], typical)
    transfer_overall = summarize_groups(
        transfer_detail, ["mask_source_year", "reference_year", "target_coverage"], typical).assign(
            variability_group="all")
    pd.concat([transfer_overall, transfer_by_variability], ignore_index=True).to_csv(
        REPORT_DIR / "empirical_2023_2024_transfer_summary.csv", index=False)
    return detail, transfer_detail, combined, typical


def interpolation_experiment(data: StudyData, typical: float) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(9127)
    for method in ("random", "empirical"):
        for coverage in TARGET_COVERAGES:
            if method == "random":
                masks = random_masks(min(300, len(data.reference_values)), coverage, 9127 + int(coverage * 100))
            else:
                pooled = np.concatenate([data.masks[y] for y in (2022, 2023, 2024)])
                masks, nearest = sample_empirical_masks(pooled, coverage, 300, 8127 + int(coverage * 100))
                if nearest > 2:
                    continue
            choice = rng.integers(0, len(data.reference_values), size=len(masks))
            truth_values = data.reference_values[choice].astype(np.float64)
            truth = truth_values.mean(axis=1) / 1000.0
            partial = np.where(masks, truth_values, np.nan)
            for gap in (0, 1, 2, 3, 6):
                candidate = partial if gap == 0 else interpolate_short_internal_gaps(partial, gap)
                estimate = np.nanmean(candidate, axis=1) / 1000.0
                frame = pd.DataFrame({"error_kw": estimate - truth, "true_mean_kw": truth,
                                      "actual_coverage": masks.mean(axis=1)})
                record = {"method": method, "target_coverage": coverage,
                          "max_interpolated_gap_seconds": gap * 5}
                record.update(error_summary(frame, typical))
                rows.append(record)
    result = pd.DataFrame(rows)
    result.to_csv(REPORT_DIR / "interpolation_sensitivity.csv", index=False)
    return result


def quality_candidates(data: StudyData, empirical_detail: pd.DataFrame, typical: float) -> pd.DataFrame:
    records = []
    for coverage in DECISION_THRESHOLDS:
        for max_run, halves in ((None, False), (12, True), (24, True), (36, True), (60, True)):
            simulated_keep = empirical_detail.actual_coverage.ge(coverage)
            if max_run is not None:
                simulated_keep &= empirical_detail.longest_missing_run_samples.le(max_run)
            if halves:
                simulated_keep &= (empirical_detail.fraction_first_half.gt(0)
                                   & empirical_detail.fraction_second_half.gt(0))
            chosen = empirical_detail.loc[simulated_keep]
            if len(chosen) < 100:
                continue
            record = {"coverage_min": coverage, "max_missing_run_samples": max_run,
                      "max_missing_run_seconds": None if max_run is None else max_run * 5,
                      "require_both_halves": halves}
            record.update(error_summary(chosen, typical))
            for year in YEARS:
                feature = data.canonical_features[year]
                record[f"intervals_retained_{year}"] = int(apply_quality_rule(feature, coverage, max_run, halves).sum())
            records.append(record)
    result = pd.DataFrame(records)
    result.to_csv(REPORT_DIR / "quality_rule_candidates.csv", index=False)
    return result


def decision_table(data: StudyData, summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    all_rows = summary[summary.variability_group.eq("all")]
    high_rows = summary[summary.variability_group.eq("high")]
    for threshold in DECISION_THRESHOLDS:
        row = {"coverage_threshold": threshold}
        for year in YEARS:
            row[f"intervals_retained_{year}"] = int(data.silver[year].coverage.ge(threshold).sum())
        for method in ("random", "empirical"):
            source = all_rows[(all_rows.method == method) & (all_rows.target_coverage == threshold)]
            if len(source):
                row[f"estimated_aggregate_mae_kw_{method}"] = float(source.iloc[0].mae_kw)
                row[f"p95_error_kw_{method}"] = float(source.iloc[0].p95_absolute_error_kw)
                row[f"p99_error_kw_{method}"] = float(source.iloc[0].p99_absolute_error_kw)
        high = high_rows[(high_rows.method == "empirical") & (high_rows.target_coverage == threshold)]
        row["high_variability_p95_error_kw"] = float(high.iloc[0].p95_absolute_error_kw) if len(high) else math.nan
        records.append(row)
    result = pd.DataFrame(records)
    result.to_csv(REPORT_DIR / "coverage_threshold_decision.csv", index=False)
    return result


def availability_outputs(data: StudyData, primary: dict, strict: dict) -> tuple[dict, dict]:
    retained = {"primary": {}, "strict": {}}
    windows = {"primary": {}, "strict": {}}
    for name, rule in (("primary", primary), ("strict", strict)):
        for year in YEARS:
            keep = apply_quality_rule(data.canonical_features[year], **rule).to_numpy()
            retained[name][str(year)] = int(keep.sum())
            windows[name][str(year)] = longest_windows(data.canonical_features[year].timestamp_utc, keep)
    pv_path = ROOT / "data" / "silver" / "pv" / "15min" / "pv_2024_15min.parquet"
    pv = pd.read_parquet(pv_path, columns=["timestamp_utc", "coverage"])
    load = data.canonical_features[2024]
    usable_load = apply_quality_rule(load, **primary)
    merged = load[["timestamp_utc"]].assign(load_usable=usable_load.to_numpy()).merge(
        pv.assign(pv_usable=pv.coverage.ge(.95)), on="timestamp_utc", how="inner")
    joint = (merged.load_usable & merged.pv_usable).to_numpy()
    local = merged.timestamp_utc.dt.tz_convert("Europe/Berlin")
    daily = pd.DataFrame({"date": local.dt.date, "joint": joint}).groupby("date").joint.agg(["sum", "size"])
    iso = local.dt.isocalendar()
    weekly = pd.DataFrame({"year": iso.year.to_numpy(), "week": iso.week.to_numpy(), "joint": joint}).groupby(
        ["year", "week"]).joint.agg(["sum", "size"])
    joint_result = {
        "load_rule": primary, "pv_rule": {"coverage_min": .95},
        "joint_usable_intervals": int(joint.sum()),
        "complete_joint_days": int((daily["sum"] == daily["size"]).sum()),
        "complete_joint_weeks": int((weekly["sum"] == weekly["size"]).sum()),
        "longest_contiguous_joint_block": longest_windows(merged.timestamp_utc, joint, 1)[0] if joint.any() else None,
    }
    (REPORT_DIR / "forecasting_windows.json").write_text(json.dumps(windows, indent=2), encoding="utf-8")
    (REPORT_DIR / "joint_pv_overlap.json").write_text(json.dumps(joint_result, indent=2), encoding="utf-8")
    return {"retained": retained, "windows": windows}, joint_result


def make_figures(data: StudyData, summary: pd.DataFrame, decision: pd.DataFrame,
                 joint: dict, primary: dict) -> None:
    overall = summary[summary.variability_group.eq("all")]
    for metric, filename, ylabel in (("mae_kw", "aggregation_error_vs_coverage.png", "MAE (kW)"),
                                     ("p95_absolute_error_kw", "p95_error_vs_coverage.png", "P95 absolute error (kW)")):
        for method, group in overall.groupby("method"):
            group = group.sort_values("target_coverage")
            plt.plot(group.target_coverage * 100, group[metric], marker="o", label=method)
        plt.xlabel("Retained 5-second coverage (%)")
        plt.ylabel(ylabel)
        plt.legend()
        save_plot(FIGURE_DIR / filename)
    for year in YEARS:
        plt.plot(decision.coverage_threshold * 100, decision[f"intervals_retained_{year}"], marker="o", label=str(year))
    plt.xlabel("Coverage threshold (%)")
    plt.ylabel("Retained 15-minute intervals")
    plt.legend()
    save_plot(FIGURE_DIR / "retained_intervals_vs_coverage.png")
    empirical = summary[(summary.method == "empirical") & (summary.variability_group != "all")]
    for group_name, group in empirical.groupby("variability_group", observed=True):
        group = group.sort_values("target_coverage")
        plt.plot(group.target_coverage * 100, group.p95_absolute_error_kw, marker="o", label=str(group_name))
    plt.xlabel("Coverage (%)")
    plt.ylabel("P95 absolute error (kW)")
    plt.legend()
    save_plot(FIGURE_DIR / "error_by_interval_variability.png")
    load = data.canonical_features[2024]
    pv = pd.read_parquet(ROOT / "data" / "silver" / "pv" / "15min" / "pv_2024_15min.parquet",
                         columns=["timestamp_utc", "coverage"])
    merged = load[["timestamp_utc"]].assign(load_ok=apply_quality_rule(load, **primary).to_numpy()).merge(
        pv.assign(pv_ok=pv.coverage.ge(.95)), on="timestamp_utc")
    merged["joint"] = merged.load_ok & merged.pv_ok
    monthly = merged.set_index("timestamp_utc")[["load_ok", "pv_ok", "joint"]].resample("MS").sum()
    monthly.index = monthly.index.strftime("%Y-%m")
    monthly.plot(kind="bar")
    plt.ylabel("Usable 15-minute intervals")
    plt.xlabel("Month")
    save_plot(FIGURE_DIR / "joint_load_pv_availability_2024.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-coverage", type=float, default=.75)
    parser.add_argument("--primary-max-gap-seconds", type=int, default=0,
                        help="Zero disables the optional longest-gap constraint")
    parser.add_argument("--primary-require-both-halves", action="store_true")
    parser.add_argument("--strict-coverage", type=float, default=.95)
    args = parser.parse_args()
    data, characterization = load_and_characterize()
    detail, transfer, summary, typical = masking_experiments(data)
    interpolation = interpolation_experiment(data, typical)
    empirical_detail = detail[detail.method.eq("empirical")].copy()
    candidates = quality_candidates(data, empirical_detail, typical)
    decision = decision_table(data, summary)
    primary = {"coverage_min": args.primary_coverage,
               "longest_missing_run_max": (args.primary_max_gap_seconds // 5
                                               if args.primary_max_gap_seconds else None),
               "require_both_halves": args.primary_require_both_halves}
    strict = {"coverage_min": args.strict_coverage,
              "longest_missing_run_max": None, "require_both_halves": False}
    availability, joint = availability_outputs(data, primary, strict)
    make_figures(data, summary, decision, joint, primary)
    result = {
        "design": {"years": YEARS, "target_coverages": TARGET_COVERAGES, "random_seeds": SEEDS,
                   "empirical_mask_sources": [2022, 2023, 2024],
                   "ground_truth": "coverage=100% intervals from 2021-2022",
                   "typical_facility_load_kw": typical},
        "characterization": characterization,
        "rules_evaluated_for_outputs": {"primary": primary, "strict": strict},
        "availability": availability, "joint_pv": joint,
        "output_rows": {"masking_detail_in_memory": len(detail), "transfer_detail_in_memory": len(transfer),
                        "masking_summary": len(summary), "interpolation_summary": len(interpolation),
                        "quality_candidates": len(candidates)},
    }
    (REPORT_DIR / "study_results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result["output_rows"], indent=2))


if __name__ == "__main__":
    main()
