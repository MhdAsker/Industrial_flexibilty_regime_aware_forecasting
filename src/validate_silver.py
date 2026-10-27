"""Independent final validation and reporting for the canonical SILVER layer.

This module never downloads data and never recomputes valid aggregations.  It
checks every canonical Parquet, performs small direct-from-source spot checks,
adds only missing audit metadata, finalizes manifests, and writes reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from timeseries_core import (
    ROOT,
    file_digest,
    iter_member_csv,
    load_config,
    localize_explicit,
    member_info,
    utc_year_index,
)

VALIDATOR_VERSION = 1
THRESHOLDS = (1.00, 0.99, 0.95, 0.90, 0.80)
THRESHOLD_NAMES = {1.00: "100", 0.99: "99", 0.95: "95", 0.90: "90", 0.80: "80"}
STAT_COLUMNS = [
    "p_mean_w", "p_std_w", "p_min_w", "p_max_w", "p_median_w",
    "p_q05_w", "p_q25_w", "p_q75_w", "p_q95_w",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def raw_manifest(config: dict) -> dict:
    result = {
        "dataset": "KIT SPARK cleaned dataset",
        "doi": "10.35097/bjdg3m3rg5jv3skk",
        "source": "https://radar.kit.edu/radar/en/dataset/bjdg3m3rg5jv3skk",
        "generated_utc": utc_now(),
        "archives": {},
    }
    for signal, cfg in config["signals"].items():
        archive = ROOT / cfg["archive"]
        if archive.stat().st_size != cfg["archive_size"]:
            raise AssertionError(f"Unexpected archive size: {archive}")
        sidecar = archive.with_suffix(archive.suffix + ".verified.json")
        saved = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        md5 = saved.get("md5") or file_digest(archive, "md5")
        if md5.lower() != cfg["archive_md5"].lower():
            raise AssertionError(f"Official MD5 mismatch: {archive}")
        sha256 = saved.get("sha256") or file_digest(archive, "sha256")
        saved.update({
            "path": str(archive.resolve()), "size": archive.stat().st_size,
            "mtime_ns": archive.stat().st_mtime_ns, "md5": md5.lower(), "sha256": sha256.lower(),
            "immutable": True,
        })
        atomic_json(sidecar, saved)
        members = {}
        for year in cfg["years"]:
            member = cfg["member_template"].format(year=year)
            members[str(year)] = member_info(archive, member)
        result["archives"][signal] = {
            **saved, "official_md5": cfg["archive_md5"], "members_used": members,
        }
    atomic_json(ROOT / "reports" / "raw_archive_manifest.json", result)
    return result


def _schema_migration(frame: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, bool]:
    """Add the explicit source-wall-time alias without recomputing statistics."""
    if "timestamp_source_local" in frame.columns:
        return frame, False
    if "timestamp_local" not in frame.columns:
        raise AssertionError(f"{path}: timestamp_local missing")
    position = frame.columns.get_loc("timestamp_local") + 1
    frame.insert(position, "timestamp_source_local", frame["timestamp_local"].copy())
    return frame, True


def _ordered_quantiles_valid(frame: pd.DataFrame) -> bool:
    cols = ["p_min_w", "p_q05_w", "p_q25_w", "p_median_w", "p_q75_w", "p_q95_w", "p_max_w"]
    values = frame.loc[frame["valid_count"].gt(0), cols]
    return bool((values.diff(axis=1).iloc[:, 1:] >= -1e-10).all().all())


def validate_parquet(path: Path, manifest_path: Path, signal: str, year: int, resolution: str,
                     expected_count: int, raw_audit: dict, dst_audit: dict,
                     archive_sha256: str) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    frame = pd.read_parquet(path)
    frame, migrated = _schema_migration(frame, path)
    ts = pd.to_datetime(frame["timestamp_utc"], utc=True)
    local_expected = ts.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    expected_index = utc_year_index(year, load_config()["resolutions"][resolution]["frequency"], "Europe/Berlin")
    mapped_numeric = raw_audit["numeric_count"] - dst_audit[
        "spring_nonexistent_rows_with_numeric_values_excluded_from_utc"
    ]
    required = {
        "timestamp_utc", "timestamp_local", "timestamp_source_local", *STAT_COLUMNS,
        "p_mean_kw", "p_max_kw", "valid_count", "expected_count", "missing_count", "coverage",
        "quality_100", "quality_99", "quality_95", "quality_90", "quality_80",
    }
    checks = {
        "required_columns": required.issubset(frame.columns),
        "row_count": len(frame) == len(expected_index),
        "timestamp_range": len(frame) == len(expected_index) and ts.iloc[0] == expected_index[0] and ts.iloc[-1] == expected_index[-1],
        "timestamp_unique": not ts.duplicated().any(),
        "utc_monotonic": ts.is_monotonic_increasing,
        "utc_grid_exact": pd.DatetimeIndex(ts).equals(expected_index),
        "local_conversion_exact": frame["timestamp_local"].equals(local_expected),
        "source_local_preserved": frame["timestamp_source_local"].equals(frame["timestamp_local"]),
        "expected_count": bool(frame["expected_count"].eq(expected_count).all()),
        "count_bounds": bool(frame["valid_count"].between(0, expected_count).all()),
        "missing_count": bool(frame["missing_count"].eq(expected_count - frame["valid_count"]).all()),
        "coverage": bool(np.allclose(frame["coverage"], frame["valid_count"] / expected_count, rtol=0, atol=1e-15)),
        "quality_flags": all(bool(frame[f"quality_{name}"].eq(frame["coverage"].ge(t)).all()) for t, name in THRESHOLD_NAMES.items()),
        "mapped_raw_count": int(frame["valid_count"].sum()) == mapped_numeric,
        "float64_statistics": all(str(frame[c].dtype) == "float64" for c in [*STAT_COLUMNS, "p_mean_kw", "p_max_kw", "coverage"]),
        "watts_to_kw": bool(np.allclose(frame["p_mean_kw"], frame["p_mean_w"] / 1000, equal_nan=True) and np.allclose(frame["p_max_kw"], frame["p_max_w"] / 1000, equal_nan=True)),
        "statistics_retained_below_80": bool(frame.loc[frame["valid_count"].gt(0) & frame["coverage"].lt(.8), "p_mean_w"].notna().all()),
        "empty_intervals_are_missing": bool(frame.loc[frame["valid_count"].eq(0), STAT_COLUMNS].isna().all().all()),
        "nonempty_intervals_have_core_stats": bool(frame.loc[frame["valid_count"].gt(0), ["p_mean_w", "p_min_w", "p_max_w", "p_median_w"]].notna().all().all()),
        "ordered_quantiles": _ordered_quantiles_valid(frame),
        "no_clipping_global_min": math.isclose(float(frame["p_min_w"].min()), float(raw_audit["min_w"]), rel_tol=0, abs_tol=1e-12),
        "no_clipping_global_max": math.isclose(float(frame["p_max_w"].max()), float(raw_audit["max_w"]), rel_tol=0, abs_tol=1e-12),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"{path}: failed checks {failed}")
    if migrated:
        temp = path.with_suffix(".parquet.part")
        frame.to_parquet(temp, index=False, compression="zstd")
        if pq.ParquetFile(temp).metadata.num_rows != len(frame):
            temp.unlink(missing_ok=True)
            raise AssertionError(f"{path}: schema migration row verification failed")
        temp.replace(path)
    digest = file_digest(path)
    manifest.update({
        "input_archive_sha256": archive_sha256,
        "input_archive_md5": manifest["identity"]["archive_md5"],
        "input_member": manifest["identity"]["member"]["member"],
        "processing_code_version": {
            "pipeline_version": manifest["identity"]["pipeline_version"],
            "core_sha256": manifest["identity"]["core_sha256"],
        },
        "processing_timestamp_utc": manifest.get("processing_timestamp_utc", original_mtime),
        "validation_timestamp_utc": utc_now(),
        "validator_version": VALIDATOR_VERSION,
        "resolution": resolution,
        "row_count": len(frame),
        "output_rows": len(frame),
        "output_size": path.stat().st_size,
        "output_sha256": digest,
        "status": "SILVER_CANONICAL_VALIDATED",
        "schema_migration": "added timestamp_source_local alias only" if migrated else "none",
        "validation_checks": checks,
    })
    atomic_json(manifest_path, manifest)
    return frame, {"path": str(path.resolve()), "status": "pass", "schema_migrated": migrated, **checks}


def run_windows(mask: pd.Series, timestamps: pd.Series, minutes: int) -> list[dict]:
    values = mask.fillna(False).to_numpy(dtype=bool)
    starts = np.flatnonzero(values & np.r_[True, ~values[:-1]])
    ends = np.flatnonzero(values & np.r_[~values[1:], True])
    windows = []
    for start, end in zip(starts, ends, strict=True):
        windows.append({
            "start_utc": pd.Timestamp(timestamps.iloc[start]).isoformat(),
            "end_utc_inclusive": pd.Timestamp(timestamps.iloc[end]).isoformat(),
            "intervals": int(end - start + 1),
            "duration_minutes": int((end - start + 1) * minutes),
        })
    return windows


def complete_periods(frame: pd.DataFrame, mask: pd.Series) -> dict:
    local = pd.to_datetime(frame["timestamp_local"])
    dates = local.dt.date
    base = pd.DataFrame({"date": dates, "ok": mask.to_numpy(dtype=bool)})
    daily = base.groupby("date").agg(ok=("ok", "all"), intervals=("ok", "size"))
    base["week"] = pd.to_datetime(base["date"]) - pd.to_timedelta(pd.to_datetime(base["date"]).dt.weekday, unit="D")
    weekly = base.groupby("week").agg(ok=("ok", "all"), days=("date", "nunique"))
    month_key = pd.to_datetime(base["date"]).dt.to_period("M")
    monthly = base.assign(month=month_key).groupby("month").agg(ok=("ok", "all"), days=("date", "nunique"))
    month_expected = monthly.index.days_in_month
    return {
        "complete_days": int(daily["ok"].sum()),
        "complete_weeks": int((weekly["ok"] & weekly["days"].eq(7)).sum()),
        "complete_months": int((monthly["ok"] & monthly["days"].eq(month_expected)).sum()),
    }


def coverage_metrics(frame: pd.DataFrame, interval_minutes: int) -> dict:
    result: dict[str, object] = {
        "calendar_intervals": len(frame),
        "valid_definition": "valid_count > 0",
        "complete_definition": "coverage == 1.0 (all expected 5-second observations)",
        "thresholds": {},
    }
    for threshold in THRESHOLDS:
        name = THRESHOLD_NAMES[threshold]
        mask = frame["coverage"].ge(threshold)
        windows = run_windows(mask, frame["timestamp_utc"], interval_minutes)
        periods = complete_periods(frame, mask)
        result["thresholds"][name] = {
            "minimum_coverage": threshold,
            "intervals": int(mask.sum()),
            "percent_of_calendar": float(100 * mask.mean()),
            **periods,
            "longest_contiguous": max(windows, key=lambda x: x["intervals"], default=None),
        }
    any_valid = frame["valid_count"].gt(0)
    result["thresholds"][">0"] = {
        "minimum_coverage": ">0",
        "intervals": int(any_valid.sum()),
        "percent_of_calendar": float(100 * any_valid.mean()),
    }
    zero_gaps = run_windows(~any_valid, frame["timestamp_utc"], interval_minutes)
    result["zero_observation_gaps"] = {
        "longest": max(zero_gaps, key=lambda x: x["intervals"], default=None),
        "counts": {
            ">15_minutes": sum(x["duration_minutes"] > 15 for x in zero_gaps),
            ">1_hour": sum(x["duration_minutes"] > 60 for x in zero_gaps),
            ">6_hours": sum(x["duration_minutes"] > 360 for x in zero_gaps),
            ">24_hours": sum(x["duration_minutes"] > 1440 for x in zero_gaps),
            ">7_days": sum(x["duration_minutes"] > 10080 for x in zero_gaps),
        },
    }
    return result


def direct_source_checks(signal: str, year: int, frames: dict[str, pd.DataFrame], config: dict) -> dict:
    """Compare deterministic full/partial intervals with original 5-second values."""
    cfg = config["signals"][signal]
    targets: dict[str, list[pd.Timestamp]] = {}
    collected: dict[tuple[str, pd.Timestamp], list[float]] = {}
    for resolution, frame in frames.items():
        candidates = []
        full = frame.loc[frame["valid_count"].eq(frame["valid_count"].max()), "timestamp_utc"]
        partial = frame.loc[frame["valid_count"].gt(0) & frame["coverage"].lt(0.8), "timestamp_utc"]
        if not full.empty:
            candidates.append(pd.Timestamp(full.iloc[0]))
        if not partial.empty:
            candidates.append(pd.Timestamp(partial.iloc[0]))
        targets[resolution] = candidates
        for target in candidates:
            collected[(resolution, target)] = []

    member = cfg["member_template"].format(year=year)
    mapped_numeric = 0
    first_valid_source_local = last_valid_source_local = None
    first_valid_mapped_utc = last_valid_mapped_utc = None
    for raw in iter_member_csv(ROOT / cfg["archive"], member, config["timestamp_column"], cfg["measurement"], config["chunk_rows"]):
        local = pd.to_datetime(raw[config["timestamp_column"]], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        values = pd.to_numeric(raw[cfg["measurement"]], errors="coerce").astype("float64")
        utc, _, _ = localize_explicit(local, config["timezone"], config["dst"]["single_ambiguous_fold"])
        finite = values.notna() & np.isfinite(values) & utc.notna()
        source_finite = values.notna() & np.isfinite(values) & local.notna()
        if source_finite.any():
            valid_local = local[source_finite]
            if first_valid_source_local is None:
                first_valid_source_local = valid_local.iloc[0].isoformat(sep=" ")
            last_valid_source_local = valid_local.iloc[-1].isoformat(sep=" ")
        if finite.any():
            valid_utc = utc[finite]
            if first_valid_mapped_utc is None:
                first_valid_mapped_utc = valid_utc.iloc[0].isoformat()
            last_valid_mapped_utc = valid_utc.iloc[-1].isoformat()
        mapped_numeric += int(finite.sum())
        for resolution, stamps in targets.items():
            if not stamps:
                continue
            bins = utc.dt.floor(config["resolutions"][resolution]["frequency"])
            for stamp in stamps:
                selected = values[finite & bins.eq(stamp)]
                if not selected.empty:
                    collected[(resolution, stamp)].extend(selected.tolist())

    checks = {
        "mapped_numeric_rows": mapped_numeric,
        "first_valid_source_local": first_valid_source_local,
        "last_valid_source_local": last_valid_source_local,
        "first_valid_mapped_utc": first_valid_mapped_utc,
        "last_valid_mapped_utc": last_valid_mapped_utc,
        "resolutions": {},
    }
    for resolution, stamps in targets.items():
        output = frames[resolution].set_index("timestamp_utc")
        checks["resolutions"][resolution] = []
        for stamp in stamps:
            values = pd.Series(collected[(resolution, stamp)], dtype="float64")
            row = output.loc[stamp]
            direct = {
                "valid_count": int(values.count()), "p_mean_w": float(values.mean()),
                "p_std_w": float(values.std()), "p_min_w": float(values.min()),
                "p_max_w": float(values.max()), "p_median_w": float(values.median()),
                "p_q05_w": float(values.quantile(.05)), "p_q25_w": float(values.quantile(.25)),
                "p_q75_w": float(values.quantile(.75)), "p_q95_w": float(values.quantile(.95)),
            }
            passed = int(row["valid_count"]) == direct["valid_count"] and all(
                np.isclose(row[key], value, equal_nan=True, rtol=1e-12, atol=1e-12)
                for key, value in direct.items() if key != "valid_count"
            )
            checks["resolutions"][resolution].append({
                "timestamp_utc": stamp.isoformat(), "pass": bool(passed), "direct": direct,
            })
            if not passed:
                raise AssertionError(f"Direct-source check failed: {signal} {year} {resolution} {stamp}")
    return checks


def exogenous_audit() -> dict:
    report: dict[str, object] = {
        "layer_status": "LEGACY_NONCANONICAL_SOURCE_FOR_AUDIT_ONLY",
        "warning": "SPARK weather is observed historical weather, not historical NWP/forecast data and not point-in-time-safe Day-Ahead input.",
        "datasets": {},
    }
    weather_path = ROOT / "data" / "processed" / "weather_all_original_2021_2024.parquet"
    if weather_path.exists():
        weather = pd.read_parquet(weather_path)
        ts = pd.to_datetime(weather["utc_datetime"], utc=True)
        expected = pd.date_range("2021-01-01", "2025-01-01", freq="h", inclusive="left", tz="UTC")
        columns = {}
        for col in weather.columns:
            if col == "utc_datetime":
                continue
            columns[col] = {
                "nonmissing": int(weather[col].notna().sum()),
                "missing": int(weather[col].isna().sum()),
                "missing_percent": float(100 * weather[col].isna().mean()),
                "min": float(weather[col].min()) if pd.api.types.is_numeric_dtype(weather[col]) and weather[col].notna().any() else None,
                "max": float(weather[col].max()) if pd.api.types.is_numeric_dtype(weather[col]) and weather[col].notna().any() else None,
            }
        report["datasets"]["observed_weather"] = {
            "path": str(weather_path.resolve()), "timezone": "UTC", "resolution": "1 hour",
            "rows": len(weather), "start": ts.min().isoformat(), "end": ts.max().isoformat(),
            "unique": not ts.duplicated().any(), "monotonic": ts.is_monotonic_increasing,
            "complete_hourly_grid": pd.DatetimeIndex(ts).equals(expected), "columns": columns,
            "unit_warning": "Delivered humidity and cloud-cover values appear normalized to roughly 0..1 despite metadata labels of percent and oktas. Sunshine duration reaches 3600 although metadata says min/h, indicating seconds/hour. Values are preserved without silent rescaling.",
        }
    combined_path = ROOT / "data" / "processed" / "exogenous_15min_2021_2024.parquet"
    if combined_path.exists():
        combined = pd.read_parquet(combined_path)
        ts = pd.to_datetime(combined["timestamp_utc"], utc=True)
        report["datasets"]["legacy_aligned_exogenous"] = {
            "path": str(combined_path.resolve()), "timezone": "UTC", "resolution": "15 minutes",
            "rows": len(combined), "start": ts.min().isoformat(), "end": ts.max().isoformat(),
            "unique": not ts.duplicated().any(), "monotonic": ts.is_monotonic_increasing,
            "price_unit": "EUR/MWh", "price_nonmissing": int(combined["price_eur_per_mwh"].notna().sum()),
            "holiday_timezone": "Europe/Berlin local calendar date",
            "holiday_rows_available": int(combined["holiday_data_available"].fillna(False).sum()),
            "note": "Combined Prompt-1 artifact; not canonical SILVER and must not be used as a modelling table.",
        }
    report["datasets"]["emissions"] = {
        "status": "not_present_locally",
        "expected_metadata": "electricity_emissions_15min_transnetbw_2018_2024.csv; UTC; 15 minutes; MW and g/kWh",
    }
    report["datasets"]["holidays"] = {
        "status": "present only inside legacy aligned exogenous Parquet",
        "timezone": "Europe/Berlin local date", "resolution": "daily source, repeated by local-date join in legacy file",
    }
    report["datasets"]["prices"] = {
        "status": "present only inside legacy aligned exogenous Parquet",
        "timezone": "UTC", "resolution": "15 minutes", "unit": "EUR/MWh",
    }
    atomic_json(ROOT / "reports" / "exogenous_audit.json", report)
    return report


def pv_analysis(frame: pd.DataFrame, raw_audit: dict, source_check: dict | None = None) -> dict:
    valid = frame["valid_count"].gt(0)
    weather_path = ROOT / "data" / "processed" / "weather_all_original_2021_2024.parquet"
    daylight = pd.Series(False, index=frame.index)
    method = "unavailable"
    if weather_path.exists():
        weather = pd.read_parquet(weather_path)
        rad_col = next(c for c in weather.columns if "radiation_global" in c)
        weather["utc_datetime"] = pd.to_datetime(weather["utc_datetime"], utc=True)
        radiation = weather.set_index("utc_datetime")[rad_col]
        daylight = pd.Series(
            radiation.reindex(pd.DatetimeIndex(frame["timestamp_utc"]).floor("h")).to_numpy() > 0,
            index=frame.index,
        )
        method = "observed hourly global radiation > 0, applied only within its labelled UTC hour"
    mean_zero = frame["p_mean_w"].eq(0)
    mean_positive = frame["p_mean_w"].gt(0)
    return {
        "first_valid_source_timestamp_local": source_check.get("first_valid_source_local") if source_check else None,
        "last_valid_source_timestamp_local": source_check.get("last_valid_source_local") if source_check else None,
        "first_valid_source_timestamp_utc": source_check.get("first_valid_mapped_utc") if source_check else None,
        "last_valid_source_timestamp_utc": source_check.get("last_valid_mapped_utc") if source_check else None,
        "first_valid_interval_utc": frame.loc[valid, "timestamp_utc"].min().isoformat(),
        "last_valid_interval_utc": frame.loc[valid, "timestamp_utc"].max().isoformat(),
        "total_valid_raw_samples": raw_audit["numeric_count"],
        "mapped_valid_raw_samples_after_dst_exclusion": int(frame["valid_count"].sum()),
        "raw_availability_percent": raw_audit["availability_percent"],
        "daylight_method": method,
        "measured_nighttime_zero_intervals": int((~daylight & valid & mean_zero).sum()),
        "nighttime_missing_intervals": int((~daylight & ~valid).sum()),
        "daylight_missing_intervals": int((daylight & ~valid).sum()),
        "valid_daylight_generation_intervals": int((daylight & valid & mean_positive).sum()),
        "valid_daylight_zero_intervals": int((daylight & valid & mean_zero).sum()),
        "warning": "Missing PV is never replaced with zero. Night/day labels use observed radiation and are retrospective, not forecast-safe.",
    }


def overlap_metrics(load: pd.DataFrame, pv: pd.DataFrame) -> dict:
    merged = load[["timestamp_utc", "timestamp_local", "coverage"]].merge(
        pv[["timestamp_utc", "coverage"]], on="timestamp_utc", suffixes=("_load", "_pv"), validate="one_to_one"
    )
    result = {}
    for threshold in (0.80, 0.90, 0.95, 0.99):
        mask = merged["coverage_load"].ge(threshold) & merged["coverage_pv"].ge(threshold)
        periods = complete_periods(merged.rename(columns={"timestamp_local_load": "timestamp_local"}), mask)
        windows = run_windows(mask, merged["timestamp_utc"], 15)
        result[THRESHOLD_NAMES[threshold]] = {
            "minimum_coverage_both": threshold, "joint_intervals": int(mask.sum()),
            "percent_of_calendar": float(100 * mask.mean()),
            "joint_complete_days": periods["complete_days"],
            "joint_complete_weeks": periods["complete_weeks"],
            "longest_contiguous": max(windows, key=lambda x: x["intervals"], default=None),
        }
    return result


def make_figures(frames_15: dict[tuple[str, int], pd.DataFrame]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        return []
    outdir = ROOT / "reports" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = []

    fig, axes = plt.subplots(5, 1, figsize=(16, 11), constrained_layout=True)
    for ax, ((signal, year), frame) in zip(axes, sorted(frames_15.items()), strict=True):
        local = pd.to_datetime(frame["timestamp_local"])
        matrix = pd.DataFrame({"date": local.dt.date, "slot": local.dt.hour * 4 + local.dt.minute // 15,
                               "coverage": frame["coverage"]}).pivot_table(index="slot", columns="date", values="coverage", aggfunc="mean")
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"{signal} {year}"); ax.set_ylabel("quarter-hour")
    axes[-1].set_xlabel("local calendar day")
    fig.colorbar(image, ax=axes, label="coverage", shrink=.8)
    path = outdir / "annual_availability_heatmaps.png"; fig.savefig(path, dpi=150); plt.close(fig); outputs.append(str(path))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for (signal, year), frame in sorted(frames_15.items()):
        axes[0 if signal == "total_load" else 1].hist(frame["coverage"], bins=np.linspace(0, 1, 101), histtype="step", label=str(year))
    axes[0].set_title("TotalLoad 15-minute coverage"); axes[1].set_title("PV 15-minute coverage")
    for ax in axes: ax.set_xlabel("coverage"); ax.set_ylabel("intervals"); ax.legend()
    path = outdir / "coverage_histograms.png"; fig.savefig(path, dpi=150); plt.close(fig); outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(16, 4), constrained_layout=True)
    for row, year in enumerate(range(2021, 2025)):
        frame = frames_15[("total_load", year)]
        ax.scatter(frame["timestamp_utc"], np.full(len(frame), row), c=frame["coverage"], s=1, vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(4), ["2021", "2022", "2023", "2024"]); ax.set_title("TotalLoad contiguous-data timeline"); ax.set_xlabel("UTC")
    path = outdir / "contiguous_data_timeline.png"; fig.savefig(path, dpi=150); plt.close(fig); outputs.append(str(path))

    load, pv = frames_15[("total_load", 2024)], frames_15[("pv", 2024)]
    joint = load[["timestamp_utc", "coverage"]].merge(pv[["timestamp_utc", "coverage"]], on="timestamp_utc", suffixes=("_load", "_pv"))
    state = np.select([joint.coverage_load.ge(.8) & joint.coverage_pv.ge(.8), joint.coverage_load.gt(0) & joint.coverage_pv.gt(0)], [2, 1], 0)
    fig, ax = plt.subplots(figsize=(16, 2.8), constrained_layout=True)
    ax.scatter(joint.timestamp_utc, np.zeros(len(joint)), c=state, s=3, cmap=ListedColormap(["#d9d9d9", "#fdae61", "#1a9850"]), vmin=0, vmax=2)
    ax.set_yticks([]); ax.set_title("2024 Load/PV overlap: grey none, orange >0, green both ≥80%")
    path = outdir / "pv_load_overlap_2024.png"; fig.savefig(path, dpi=150); plt.close(fig); outputs.append(str(path))
    return outputs


def markdown_report(report: dict, raw: dict, exogenous: dict, storage: dict) -> str:
    lines = [
        "# SILVER data-layer validation summary", "",
        f"Generated: {report['generated_utc']}", "",
        "Canonical data are direct aggregations of the official 5-second source. No interpolation, clipping, smoothing, or coverage masking is applied.", "",
        "## TotalLoad", "",
        "`valid` means at least one observation in an interval. `complete` means exactly 180/180 observations at 15 minutes. The large difference is mostly caused by frequent isolated missing readings: common coverage modes are around 75–77%, so 179/180 is not being rejected or relabelled as missing.", "",
        "| Year | Raw availability | >0 | 100% | ≥99% | ≥95% | ≥90% | ≥80% |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for year in range(2021, 2025):
        metrics = report["coverage"]["total_load"][str(year)]
        audit = report["raw_audits"]["total_load"][str(year)]
        t = metrics["thresholds"]
        def cell(key: str) -> str:
            return f"{t[key]['intervals']:,} ({t[key]['percent_of_calendar']:.2f}%)"
        lines.append(f"| {year} | {audit['availability_percent']:.2f}% | {cell('>0')} | {cell('100')} | {cell('99')} | {cell('95')} | {cell('90')} | {cell('80')} |")
    lines.extend(["", "### Best contiguous high-quality windows", "",
                  "| Year | Threshold | Duration | Start UTC | End UTC (inclusive) |",
                  "|---:|---:|---:|---|---|"])
    for year in range(2021, 2025):
        for threshold in ("99", "95", "90"):
            window = report["coverage"]["total_load"][str(year)]["thresholds"][threshold]["longest_contiguous"]
            if window:
                lines.append(f"| {year} | ≥{threshold}% | {window['duration_minutes'] / 60:.2f} h | {window['start_utc']} | {window['end_utc_inclusive']} |")
            else:
                lines.append(f"| {year} | ≥{threshold}% | 0 h | — | — |")
    lines.extend(["", "### Complete periods and zero-observation gaps", "",
                  "| Year | Days ≥99% | Days ≥95% | Weeks ≥95% | Months ≥95% | Longest zero-observation gap | Gaps >24 h | Gaps >7 d |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for year in range(2021, 2025):
        metrics = report["coverage"]["total_load"][str(year)]
        gap = metrics["zero_observation_gaps"]
        longest = gap["longest"]["duration_minutes"] if gap["longest"] else 0
        lines.append(f"| {year} | {metrics['thresholds']['99']['complete_days']} | {metrics['thresholds']['95']['complete_days']} | {metrics['thresholds']['95']['complete_weeks']} | {metrics['thresholds']['95']['complete_months']} | {longest / 60:.2f} h | {gap['counts']['>24_hours']} | {gap['counts']['>7_days']} |")
    lines.extend(["", "These are zero-observation gaps and are distinct from partially observed intervals; all threshold and gap counts are in `missingness_report.json`.", "",
                  "## PV", ""])
    pv = report["pv_analysis"]
    pvt = report["coverage"]["pv"]["2024"]["thresholds"]
    lines.extend([
        f"PV has {pv['total_valid_raw_samples']:,} valid source samples ({pv['raw_availability_percent']:.2f}% of 2024). Exact first/last valid source timestamps are {pv['first_valid_source_timestamp_local']} and {pv['last_valid_source_timestamp_local']} Europe/Berlin wall time; the first/last usable 15-minute UTC intervals are {pv['first_valid_interval_utc']} and {pv['last_valid_interval_utc']}.", "",
        f"At 15 minutes: {pvt['100']['intervals']:,} intervals are complete, {pvt['95']['complete_days']:,} local days are complete at ≥95%, and {pvt['95']['complete_weeks']:,} full local weeks are complete at ≥95%.", "",
        f"Using observed radiation: measured nighttime zero={pv['measured_nighttime_zero_intervals']:,}, nighttime missing={pv['nighttime_missing_intervals']:,}, daylight missing={pv['daylight_missing_intervals']:,}, valid daylight generation={pv['valid_daylight_generation_intervals']:,}. Missing PV is never replaced by zero.", "",
        "## Joint Load + PV", "",
        "| Threshold for both | Joint intervals | Complete days | Complete weeks | Longest contiguous |",
        "|---:|---:|---:|---:|---:|",
    ])
    for threshold, item in report["overlap_2024"].items():
        longest = item["longest_contiguous"]["duration_minutes"] if item["longest_contiguous"] else 0
        lines.append(f"| ≥{threshold}% | {item['joint_intervals']:,} | {item['joint_complete_days']:,} | {item['joint_complete_weeks']:,} | {longest / 60:.2f} h |")
    lines.extend(["", "## DST", "",
                  "Source timestamps are naive Europe/Berlin wall times. Each spring contains 720 labelled `02:xx` rows that are nonexistent in civil time. Numeric values in those rows are excluded from UTC aggregation and counted explicitly. Each autumn contains one unlabeled 720-row `02:xx` hour with no duplicate source timestamps, so the physical fold cannot be reconstructed from sequence. It is assigned to the first/CEST fold; the second fold remains missing. No rows are silently shifted and no second fold is fabricated.", "",
                  "| Signal/year | Spring numeric rows excluded | Autumn numeric ambiguous rows |",
                  "|---|---:|---:|"])
    for item in report["dst_audits"]:
        lines.append(f"| {item['signal']} {item['year']} | {item['spring_nonexistent_rows_with_numeric_values_excluded_from_utc']:,} | {item['autumn_ambiguous_rows_with_numeric_values']:,} |")
    lines.extend(["", "## Exogenous data", "",
                  exogenous["warning"], "",
                  "Observed weather is available hourly in UTC and is labelled `observed_weather`. Prices and holidays are present only in a legacy aligned artifact; emissions are not present locally. No exogenous dataset is promoted into canonical SILVER or merged into a modelling table by this task.", "",
                  "## Storage", "",
                  "| Category | Bytes | GiB |", "|---|---:|---:|"])
    for name, size in storage.items():
        lines.append(f"| {name} | {size:,} | {size / 2**30:.3f} |")
    lines.extend(["", "## Canonical status", "",
                  "- **RAW:** `data/raw/*.zip` (immutable official archives).",
                  "- **SILVER CANONICAL:** `data/silver/{total_load,pv}/{1min,10min,15min}/*.parquet` with sibling manifests.",
                  "- **LEGACY / NONCANONICAL:** `data/processed/` and `data/final/`; do not use these Prompt-1 outputs for thesis modelling.", "",
                  "## Issues to resolve before GOLD modelling", "",
                  "- No 2024 TotalLoad interval reaches 80% coverage, so there is no joint Load/PV interval at ≥80%. A lower reversible threshold or a different experiment design requires explicit scientific justification.",
                  "- PV ends on 9 April 2024, which severely limits seasonal joint forecasting.",
                  "- Emissions are absent locally; prices and holidays exist only in a legacy combined file.",
                  "- Weather units have source metadata/value-scale inconsistencies (humidity, cloud cover, and sunshine duration). Do not silently rescale them.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-direct-source", action="store_true", help="Skip the slower raw-member spot checks")
    args = parser.parse_args()
    config = load_config()
    raw = raw_manifest(config)
    previous_validation_path = ROOT / "reports" / "silver_validation.json"
    previous_direct = {}
    if args.skip_direct_source and previous_validation_path.exists():
        previous_direct = json.loads(previous_validation_path.read_text(encoding="utf-8")).get("direct_source_checks", {})
    validation: dict[str, object] = {"generated_utc": utc_now(), "outputs": {}, "direct_source_checks": previous_direct}
    coverage: dict[str, dict] = {"total_load": {}, "pv": {}}
    frames_15: dict[tuple[str, int], pd.DataFrame] = {}
    raw_audits: dict[str, dict] = {"total_load": {}, "pv": {}}
    dst_audits = []

    for signal, signal_cfg in config["signals"].items():
        validation["outputs"][signal] = {}
        for year in signal_cfg["years"]:
            raw_audit = json.loads((ROOT / "reports" / "raw_quality" / f"{signal}_{year}.json").read_text(encoding="utf-8"))
            dst_audit = json.loads((ROOT / "reports" / "dst" / f"{signal}_{year}.json").read_text(encoding="utf-8"))
            raw_audits[signal][str(year)] = raw_audit
            dst_audits.append(dst_audit)
            year_frames = {}
            validation["outputs"][signal][str(year)] = {}
            for resolution, resolution_cfg in config["resolutions"].items():
                path = ROOT / "data" / "silver" / signal / resolution / f"{signal}_{year}_{resolution}.parquet"
                manifest = path.with_suffix(".manifest.json")
                frame, checks = validate_parquet(
                    path, manifest, signal, year, resolution, resolution_cfg["expected_count"],
                    raw_audit, dst_audit, raw["archives"][signal]["sha256"],
                )
                year_frames[resolution] = frame
                validation["outputs"][signal][str(year)][resolution] = checks
                if resolution == "15min":
                    frames_15[(signal, year)] = frame
                    coverage[signal][str(year)] = coverage_metrics(frame, 15)
            if not args.skip_direct_source:
                validation["direct_source_checks"][f"{signal}_{year}"] = direct_source_checks(signal, year, year_frames, config)

    atomic_json(ROOT / "reports" / "silver_validation.json", validation)
    exogenous = exogenous_audit()
    pv_source_check = validation["direct_source_checks"].get("pv_2024")
    pv = pv_analysis(frames_15[("pv", 2024)], raw_audits["pv"]["2024"], pv_source_check)
    overlap = overlap_metrics(frames_15[("total_load", 2024)], frames_15[("pv", 2024)])
    figures = make_figures(frames_15)
    storage = {
        "raw": directory_size(ROOT / "data" / "raw"),
        "silver": directory_size(ROOT / "data" / "silver"),
        "reports": directory_size(ROOT / "reports"),
        "legacy_processed_and_final": directory_size(ROOT / "data" / "processed") + directory_size(ROOT / "data" / "final"),
        "total_project": directory_size(ROOT),
    }
    missingness = {
        "generated_utc": utc_now(), "coverage": coverage, "pv_analysis": pv,
        "overlap_2024": overlap, "dst_audits": dst_audits, "raw_audits": raw_audits,
        "figures": figures, "storage": storage,
    }
    atomic_json(ROOT / "reports" / "missingness_report.json", missingness)
    summary = markdown_report(missingness, raw, exogenous, storage)
    (ROOT / "reports" / "silver_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": "PASS", "outputs_validated": 15,
                      "direct_source_checks": bool(validation["direct_source_checks"]),
                      "summary": str(ROOT / 'reports' / 'silver_summary.md'), "storage": storage}, indent=2))


if __name__ == "__main__":
    main()
