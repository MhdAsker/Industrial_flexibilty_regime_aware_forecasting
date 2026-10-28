"""Point-in-time feature construction for the fixed 12:00 D-1 protocol."""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from day_ahead_baselines import (INTERVAL, ROOT, TZ, b0_persistence, b1_safe_recent_day,
                                 b2_weekly, b3_week_profile, base_prediction_rows,
                                 load_gold, make_history_index, observation_available)

SPEC_PATH = ROOT / "config" / "lightgbm_features.json"
FEATURE_PATH = ROOT / "data" / "features" / "lightgbm_features.parquet"
DICTIONARY_PATH = ROOT / "docs" / "lightgbm_feature_dictionary.md"
MANIFEST_PATH = ROOT / "results" / "lightgbm" / "feature_matrix_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exact_week_value(history, target_date: date, quarter: int, issue_utc: pd.Timestamp, weeks: int) -> float:
    items = history.by_date_quarter.get((target_date - timedelta(days=7 * weeks), int(quarter)), [])
    values = [value for timestamp, value in items if observation_available(timestamp, issue_utc)]
    return float(values[-1]) if values else np.nan


def week_summary(history, target_date: date, quarter: int, issue_utc: pd.Timestamp,
                 weeks: int) -> tuple[float, float, float, int]:
    values = [exact_week_value(history, target_date, quarter, issue_utc, lag)
              for lag in range(1, weeks + 1)]
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(values):
        return np.nan, np.nan, np.nan, 0
    return float(np.median(values)), float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else np.nan, len(values)


def safe_slice(timestamp_ns: np.ndarray, values: np.ndarray, issue_utc: pd.Timestamp,
               hours: int) -> np.ndarray:
    cutoff = (issue_utc - INTERVAL).value
    start = (issue_utc - pd.Timedelta(hours=hours)).value
    left = np.searchsorted(timestamp_ns, start, side="left")
    right = np.searchsorted(timestamp_ns, cutoff, side="right")
    return values[left:right]


def summarize(values: np.ndarray) -> tuple[float, float, float, float, int]:
    if not len(values):
        return np.nan, np.nan, np.nan, np.nan, 0
    return (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            float(np.min(values)), float(np.max(values)), int(len(values)))


def issuance_features(load: pd.DataFrame, issues: pd.Series) -> pd.DataFrame:
    eligible = load[load.target_quality_primary & load.load_mean_kw.notna()].sort_values("timestamp_utc")
    timestamp_ns = eligible.timestamp_utc.astype("int64").to_numpy()
    values = eligible.load_mean_kw.to_numpy(float)
    dates = eligible.date_local.dt.date.to_numpy()
    records = []
    for issue in sorted(pd.unique(issues)):
        issue = pd.Timestamp(issue)
        cutoff = issue - INTERVAL
        position = np.searchsorted(timestamp_ns, cutoff.value, side="right") - 1
        row = {"forecast_issue_time_utc": issue}
        if position >= 0:
            source = pd.Timestamp(timestamp_ns[position], tz="UTC")
            row["latest_load_kw"] = float(values[position])
            row["latest_load_age_minutes"] = float((issue - (source + INTERVAL)) / pd.Timedelta(minutes=1))
        else:
            row["latest_load_kw"] = np.nan; row["latest_load_age_minutes"] = np.nan
        for hours in (1, 3, 6, 12, 24, 48):
            selected = safe_slice(timestamp_ns, values, issue, hours)
            mean, std, minimum, maximum, count = summarize(selected)
            row[f"mean_load_last_{hours}h"] = mean
            if hours != 48:
                row[f"std_load_last_{hours}h"] = std
            if hours == 24:
                row["min_load_last_24h"] = minimum; row["max_load_last_24h"] = maximum
            row[f"count_load_last_{hours}h"] = count
            row[f"fraction_load_last_{hours}h"] = count / (hours * 4)
        for days in (1, 3, 7, 14, 28):
            selected = safe_slice(timestamp_ns, values, issue, days * 24)
            row[f"recent_{days}d_mean"] = float(np.mean(selected)) if len(selected) else np.nan
            if days in (7, 28):
                row[f"recent_{days}d_median"] = float(np.median(selected)) if len(selected) else np.nan
                row[f"recent_{days}d_std"] = float(np.std(selected, ddof=1)) if len(selected) > 1 else np.nan
        denominator = row["recent_28d_mean"]
        row["recent_level_ratio_7d_28d"] = (row["recent_7d_mean"] / denominator
                                             if np.isfinite(denominator) and abs(denominator) > 1e-6 else np.nan)
        row["recent_level_difference_7d_28d"] = row["recent_7d_mean"] - row["recent_28d_mean"]
        issue_local = issue.tz_convert(TZ)
        midnight_local = pd.Timestamp(issue_local.date(), tz=TZ)
        midnight_utc = midnight_local.tz_convert("UTC")
        left = np.searchsorted(timestamp_ns, midnight_utc.value, side="left")
        right = np.searchsorted(timestamp_ns, cutoff.value, side="right")
        partial = values[left:right]
        mean, std, _, maximum, count = summarize(partial)
        row |= {"previous_day_midnight_to_issue_mean": mean, "previous_day_midnight_to_issue_std": std,
                "previous_day_midnight_to_issue_max": maximum, "previous_day_midnight_to_issue_count": count}
        trend_values = safe_slice(timestamp_ns, values, issue, 14 * 24)
        trend_dates = dates[np.searchsorted(timestamp_ns, (issue - pd.Timedelta(days=14)).value, side="left"):
                            np.searchsorted(timestamp_ns, cutoff.value, side="right")]
        if len(trend_values):
            daily = pd.DataFrame({"date": trend_dates, "value": trend_values}).groupby("date").value.mean()
            row["recent_daily_trend_14d_kw_per_day"] = (float(np.polyfit(np.arange(len(daily)), daily.values, 1)[0])
                                                         if len(daily) >= 3 else np.nan)
        else:
            row["recent_daily_trend_14d_kw_per_day"] = np.nan
        records.append(row)
    return pd.DataFrame(records)


def holiday_name_mapping(load: pd.DataFrame) -> dict[str, int]:
    names = sorted(str(name) for name in load.loc[load.split.eq("train"), "holiday_name"].dropna().unique()
                   if str(name).strip())
    return {name: index for index, name in enumerate(names)}


def add_bridge_day(frame: pd.DataFrame) -> pd.Series:
    calendar = frame.drop_duplicates("forecast_date_local").set_index("forecast_date_local")
    holidays = calendar.is_public_holiday.to_dict()
    result = []
    for value, weekday in zip(frame.forecast_date_local, frame.day_of_week, strict=True):
        day = pd.Timestamp(value)
        previous_day, next_day = day - pd.Timedelta(days=1), day + pd.Timedelta(days=1)
        previous_weekend = previous_day.dayofweek >= 5
        next_weekend = next_day.dayofweek >= 5
        result.append(bool(weekday < 5 and ((holidays.get(previous_day, False) and next_weekend) or
                                            (holidays.get(next_day, False) and previous_weekend))))
    return pd.Series(result, index=frame.index, dtype=bool)


def build_features() -> pd.DataFrame:
    load = load_gold()
    # Add fields required by feature generation but omitted by the baseline loader.
    extra = pd.read_parquet(ROOT / "data" / "gold" / "load_15min_base.parquet",
                            columns=["timestamp_utc", "year", "day", "hour_of_day", "minute_of_hour",
                                     "sin_hour", "cos_hour", "sin_week", "cos_week", "sin_year", "cos_year",
                                     "holiday_name", "is_day_before_holiday", "is_day_after_holiday"])
    load = load.merge(extra, on="timestamp_utc", how="left", validate="one_to_one")
    history = make_history_index(load)
    frames = [base_prediction_rows(load, split) for split in ("train", "validation", "test")]
    frame = pd.concat(frames, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
    issue_state = issuance_features(load, frame.forecast_issue_time_utc)
    frame = frame.merge(issue_state, on="forecast_issue_time_utc", how="left", validate="many_to_one")
    mapping = holiday_name_mapping(load)
    frame["holiday_name_code"] = frame.holiday_name.map(mapping).fillna(-1).astype("int16")
    frame["is_bridge_day"] = add_bridge_day(frame)
    frame["lead_time_hours"] = (frame.lead_time_minutes / 60).astype("float32")

    analog_records = []
    for row in frame.itertuples(index=False):
        target_date = pd.Timestamp(row.forecast_date_local).date()
        issue = row.forecast_issue_time_utc
        quarter = int(row.quarter_hour_of_day)
        b0, b0_source = b0_persistence(history, issue)
        b1, b1_source = b1_safe_recent_day(history, target_date, quarter, issue)
        b2, b2_source = b2_weekly(history, target_date, quarter, issue)
        b3, b3_count = b3_week_profile(history, target_date, quarter, issue, 4, "median")
        four = week_summary(history, target_date, quarter, issue, 4)
        eight = week_summary(history, target_date, quarter, issue, 8)
        twelve = week_summary(history, target_date, quarter, issue, 12)
        record = {"baseline_B0_kw": b0, "baseline_B1_kw": b1, "baseline_B2_kw": b2,
                  "baseline_B3_kw": b3, "B0_available": np.isfinite(b0), "B1_available": np.isfinite(b1),
                  "B2_available": np.isfinite(b2), "B3_available": np.isfinite(b3),
                  "most_recent_same_quarter_kw": b1,
                  "same_quarter_7d_kw": exact_week_value(history, target_date, quarter, issue, 1),
                  "same_quarter_14d_kw": exact_week_value(history, target_date, quarter, issue, 2),
                  "same_quarter_21d_kw": exact_week_value(history, target_date, quarter, issue, 3),
                  "same_quarter_28d_kw": exact_week_value(history, target_date, quarter, issue, 4),
                  "same_quarter_prev4w_median": four[0], "same_quarter_prev4w_mean": four[1],
                  "same_quarter_prev4w_std": four[2], "same_quarter_prev8w_median": eight[0],
                  "same_quarter_prev8w_mean": eight[1], "same_quarter_prev8w_std": eight[2],
                  "same_weekday_quarter_prev12w_median": twelve[0],
                  "same_weekday_quarter_prev12w_mean": twelve[1],
                  "same_weekday_quarter_prev12w_std": twelve[2],
                  "latest_analog_source_time_utc": max([x for x in (b0_source, b1_source, b2_source)
                                                                if pd.notna(x)], default=pd.NaT),
                  "B3_source_count": b3_count}
        analog_records.append(record)
    frame = pd.concat([frame, pd.DataFrame(analog_records)], axis=1)
    return frame


def feature_dictionary(spec: dict) -> None:
    descriptions = {
        "is_bridge_day": "Weekday between a public holiday and an adjacent weekend day.",
        "holiday_name_code": "TRAIN-derived stable code; unseen names map to -1.",
        "latest_load_age_minutes": "Minutes from the latest source interval end to issuance.",
        "recent_daily_trend_14d_kw_per_day": "OLS slope of safe pre-issuance daily/partial-day means.",
    }
    rows = []
    for group, features in spec["groups"].items():
        for feature in features:
            historical = group not in {"calendar", "holiday"}
            rows.append({"feature": feature, "group": group,
                "meaning": descriptions.get(feature, feature.replace("_", " ")),
                "source_time": "target calendar" if not historical else "historical measurements/baselines",
                "availability_rule": "known before issuance" if not historical else "source interval end <= issue UTC",
                "unit": "kW" if any(token in feature for token in ("load", "_kw", "mean", "median", "std", "min", "max")) else
                        ("minutes" if "minutes" in feature else ("hours" if "hours" in feature else "unitless")),
                "missingness_handling": "preserve NaN; LightGBM native routing" if historical else "deterministic",
                "predictor_eligible": True})
    prohibited = [{"feature": feature, "group": "prohibited", "meaning": "Excluded leakage/availability risk",
        "source_time": "target-time or unverified", "availability_rule": "not available/allowed at issuance",
        "unit": "various", "missingness_handling": "not included", "predictor_eligible": False}
        for feature in spec["classification"]["prohibited"]]
    table = pd.DataFrame(rows + prohibited)
    DICTIONARY_PATH.write_text("# LightGBM point-in-time feature dictionary\n\n" + table.to_markdown(index=False) +
        "\n\nThe static finalized B4 calendar prediction is not used as a TRAIN predictor because doing so would "
        "encode full-TRAIN target aggregates into their own rows. Calendar structure is represented directly.\n",
        encoding="utf-8")


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    frame = build_features()
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FEATURE_PATH, index=False, compression="zstd")
    check = pd.read_parquet(FEATURE_PATH, columns=["timestamp_utc", "split"])
    if len(check) != len(frame) or not check.timestamp_utc.is_unique:
        raise RuntimeError("Feature matrix post-write validation failed")
    feature_dictionary(spec)
    manifest = {"rows": len(frame), "split_rows": frame.split.value_counts().to_dict(),
        "eligible_targets": frame.groupby("split").target_quality_primary.sum().astype(int).to_dict(),
        "feature_spec_sha256": sha256(SPEC_PATH), "source_code_sha256": sha256(Path(__file__)),
        "gold_sha256": sha256(ROOT / "data" / "gold" / "load_15min_base.parquet"),
        "output_sha256": sha256(FEATURE_PATH), "output_bytes": FEATURE_PATH.stat().st_size,
        "authoritative_availability": spec["authoritative_availability_function"],
        "unsafe_predictors_present": False}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
