"""Availability, contiguous-window, PV daylight, overlap, and heatmap reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from timeseries_core import ROOT, load_config


def run_lengths(mask: pd.Series, interval_minutes: int = 15) -> list[int]:
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    return [int(n * interval_minutes) for _, n in mask[mask].groupby(groups[mask]).size().items()]


def period_completeness(frame: pd.DataFrame, frequency: str) -> int:
    expected = {"D": 96, "W-MON": 7 * 96, "MS": None}[frequency]
    local = frame.set_index("timestamp_local")
    grouped = local["quality_100"].resample(frequency)
    if expected is None:
        complete = grouped.agg(["all", "size"])
        calendar_expected = complete.index.days_in_month * 96
        return int((complete["all"] & complete["size"].eq(calendar_expected)).sum())
    complete = grouped.agg(["all", "size"])
    return int((complete["all"] & complete["size"].eq(expected)).sum())


def metrics(frame: pd.DataFrame) -> dict:
    valid = frame["valid_count"].gt(0)
    complete = frame["quality_100"]
    gaps = run_lengths(~valid)
    sequences = run_lengths(valid)
    return {
        "calendar_intervals": len(frame),
        "valid_intervals_any_observation": int(valid.sum()),
        "complete_intervals_100pct": int(complete.sum()),
        "complete_days": period_completeness(frame, "D"),
        "complete_weeks": period_completeness(frame, "W-MON"),
        "complete_months": period_completeness(frame, "MS"),
        "longest_contiguous_valid_minutes": max(sequences, default=0),
        "longest_zero_availability_gap_minutes": max(gaps, default=0),
        "gaps_gt_15min": sum(x > 15 for x in gaps),
        "gaps_gt_1hour": sum(x > 60 for x in gaps),
        "gaps_gt_6hours": sum(x > 360 for x in gaps),
        "gaps_gt_24hours": sum(x > 1440 for x in gaps),
        "gaps_gt_7days": sum(x > 10080 for x in gaps),
    }


def plot_availability(frame: pd.DataFrame, title: str, output: Path) -> None:
    import matplotlib.pyplot as plt
    local = frame.copy()
    local["date"] = local["timestamp_local"].dt.date
    local["quarter"] = local["timestamp_local"].dt.hour * 4 + local["timestamp_local"].dt.minute // 15
    matrix = local.pivot_table(index="date", columns="quarter", values="coverage", aggfunc="mean").T
    daily = local.groupby("date")["coverage"].mean()
    fig, (timeline, heatmap) = plt.subplots(2, 1, figsize=(16, 6), height_ratios=[1, 3])
    timeline.plot(pd.to_datetime(daily.index), daily.to_numpy(), linewidth=0.8)
    timeline.set_ylim(-0.02, 1.02); timeline.set_ylabel("daily mean"); timeline.set_title(title)
    image = heatmap.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
    heatmap.set_ylabel("15-minute slot of local day"); heatmap.set_xlabel("local date")
    fig.colorbar(image, ax=heatmap, label="coverage")
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150); plt.close(fig)


def radiation_daylight(index: pd.Series) -> pd.Series | None:
    weather = ROOT / "data" / "processed" / "weather_all_original_2021_2024.parquet"
    if not weather.exists():
        return None
    source = pd.read_parquet(weather)
    time_col = next((c for c in source if c.lower() == "utc_datetime"), None)
    rad_col = next((c for c in source if "radiation_global" in c), None)
    if not time_col or not rad_col:
        return None
    source[time_col] = pd.to_datetime(source[time_col], utc=True)
    hourly = source.set_index(time_col)[rad_col].sort_index()
    lookup = pd.DatetimeIndex(index).floor("h")
    return pd.Series(hourly.reindex(lookup).to_numpy() > 0, index=index.index)


def main() -> None:
    cfg = load_config()
    report: dict[str, object] = {"valid_definition": "valid_count > 0", "complete_definition": "coverage >= 1.0",
                                 "signals": {}, "pv_analysis": {}, "overlap": {}}
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for signal, signal_cfg in cfg["signals"].items():
        report["signals"][signal] = {}
        for year in signal_cfg["years"]:
            path = ROOT / "data" / "silver" / signal / "15min" / f"{signal}_{year}_15min.parquet"
            if not path.exists():
                report["signals"][signal][str(year)] = {"status": "source_not_processed"}
                continue
            frame = pd.read_parquet(path)
            frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
            frame["timestamp_local"] = pd.to_datetime(frame["timestamp_local"])
            frames[(signal, year)] = frame
            report["signals"][signal][str(year)] = metrics(frame)
            try:
                plot_availability(frame, f"{signal} availability {year}",
                                  ROOT / "reports" / "figures" / f"{signal}_{year}_availability.png")
            except ImportError:
                report.setdefault("warnings", []).append("matplotlib unavailable; PNG heatmaps not created")

    pv = frames.get(("pv", 2024))
    if pv is not None:
        valid = pv["valid_count"].gt(0)
        daylight = radiation_daylight(pv["timestamp_utc"])
        if daylight is None:
            local_hour = pv["timestamp_local"].dt.hour + pv["timestamp_local"].dt.minute / 60
            daylight = local_hour.between(6, 18, inclusive="left")
            daylight_method = "local clock fallback 06:00-18:00 (radiation unavailable)"
        else:
            daylight_method = "hourly global radiation > 0 J/cm2, mapped to its four quarters"
        zero = pv["p_mean_w"].eq(0) & valid
        valid_runs = run_lengths(valid)
        report["pv_analysis"] = {
            "published_date_range": [pv.loc[valid, "timestamp_utc"].min().isoformat() if valid.any() else None,
                                     pv.loc[valid, "timestamp_utc"].max().isoformat() if valid.any() else None],
            "full_calendar_coverage": float(pv["valid_count"].sum() / pv["expected_count"].sum()),
            "daylight_method": daylight_method,
            "daylight_sample_coverage": float(pv.loc[daylight, "valid_count"].sum() / pv.loc[daylight, "expected_count"].sum()),
            "nighttime_measured_zero_intervals": int((~daylight & zero).sum()),
            "nighttime_missing_intervals": int((~daylight & ~valid).sum()),
            "daytime_missing_intervals": int((daylight & ~valid).sum()),
            "longest_continuous_valid_minutes": max(valid_runs, default=0),
            "complete_valid_days": period_completeness(pv, "D"),
            "complete_valid_weeks": period_completeness(pv, "W-MON"),
        }
        load = frames.get(("total_load", 2024))
        if load is not None:
            merged = load[["timestamp_utc", "valid_count"]].merge(
                pv[["timestamp_utc", "valid_count"]], on="timestamp_utc", suffixes=("_load", "_pv"))
            overlap = merged["valid_count_load"].gt(0) & merged["valid_count_pv"].gt(0)
            report["overlap"]["2024"] = {"valid_15min_intervals": int(overlap.sum()),
                                                    "fraction_of_calendar": float(overlap.mean())}
        else:
            report["overlap"]["2024"] = {"status": "TotalLoad 2024 not processed"}

    output = ROOT / "reports" / "missingness_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
