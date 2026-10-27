"""Build auditable GOLD research panels from validated canonical SILVER inputs.

No target or exogenous value is imputed. Calendar transformations are
deterministic. Observed weather and measured PV are explicitly classified as
retrospective/oracle information rather than day-ahead-safe predictors.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from timeseries_core import ROOT, file_digest


CONFIG_PATH = ROOT / "config" / "gold_config.json"
GOLD_DIR = ROOT / "data" / "gold"
MANIFEST_PATH = ROOT / "reports" / "gold_manifest.json"
SUMMARY_PATH = ROOT / "reports" / "gold_summary.md"
AVAILABILITY_CSV = ROOT / "docs" / "gold_feature_availability.csv"
AVAILABILITY_MD = ROOT / "docs" / "gold_feature_availability.md"
DICTIONARY_CSV = ROOT / "docs" / "gold_data_dictionary.csv"
DICTIONARY_MD = ROOT / "docs" / "gold_data_dictionary.md"

LOAD_INPUTS = [ROOT / "data" / "silver" / "total_load" / "15min" / f"total_load_{year}_15min.parquet"
               for year in range(2021, 2025)]
PV_INPUT = ROOT / "data" / "silver" / "pv" / "15min" / "pv_2024_15min.parquet"
WEATHER_INPUT = ROOT / "data" / "silver" / "exogenous" / "canonical" / "weather_karlsruhe_hourly_original_2021_2024.parquet"
PRICE_INPUT = ROOT / "data" / "silver" / "exogenous" / "canonical" / "electricity_prices_15min_de_2021_2024.parquet"
HOLIDAY_INPUT = ROOT / "data" / "silver" / "exogenous" / "canonical" / "holidays_karlsruhe_bw_2021_2024.parquet"

OUTPUTS = {
    "load_15min_base": GOLD_DIR / "load_15min_base.parquet",
    "load_15min_primary": GOLD_DIR / "load_15min_primary.parquet",
    "load_15min_strict": GOLD_DIR / "load_15min_strict.parquet",
    "pv_2024_15min": GOLD_DIR / "pv_2024_15min.parquet",
    "net_load_2024_15min": GOLD_DIR / "net_load_2024_15min.parquet",
}


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def calendar_frame(timestamp_utc: pd.Series, timezone_name: str) -> pd.DataFrame:
    """Deterministic future-known calendar values from UTC event times."""
    local_aware = timestamp_utc.dt.tz_convert(timezone_name)
    local = local_aware.dt.tz_localize(None)
    hour = local.dt.hour.astype("int8")
    minute = local.dt.minute.astype("int8")
    day_of_week = local.dt.dayofweek.astype("int8")
    day_of_year = local.dt.dayofyear.astype("int16")
    week = local.dt.isocalendar().week.astype("int16")
    quarter = (hour.astype("int16") * 4 + minute.astype("int16") // 15).astype("int16")
    hour_phase = 2 * np.pi * (hour.astype(float) + minute.astype(float) / 60) / 24
    week_phase = 2 * np.pi * (day_of_week.astype(float) + hour.astype(float) / 24) / 7
    days_in_year = np.where(local.dt.is_leap_year, 366.0, 365.0)
    year_phase = 2 * np.pi * (day_of_year.astype(float) - 1
                              + hour.astype(float) / 24) / days_in_year
    return pd.DataFrame({
        "timestamp_utc": timestamp_utc,
        "timestamp_local": local,
        "date_local": local.dt.normalize(),
        "year": local.dt.year.astype("int16"),
        "month": local.dt.month.astype("int8"),
        "day": local.dt.day.astype("int8"),
        "hour": hour,
        "minute": minute,
        "weekday": local.dt.day_name(),
        "hour_of_day": hour,
        "minute_of_hour": minute,
        "quarter_hour_of_day": quarter,
        "day_of_week": day_of_week,
        "is_weekend": day_of_week.ge(5),
        "day_of_year": day_of_year,
        "week_of_year": week,
        "sin_hour": np.sin(hour_phase).astype("float32"),
        "cos_hour": np.cos(hour_phase).astype("float32"),
        "sin_week": np.sin(week_phase).astype("float32"),
        "cos_week": np.cos(week_phase).astype("float32"),
        "sin_year": np.sin(year_phase).astype("float32"),
        "cos_year": np.cos(year_phase).astype("float32"),
    })


def add_holidays(frame: pd.DataFrame) -> pd.DataFrame:
    holidays = pd.read_parquet(HOLIDAY_INPUT).rename(columns={
        "date": "date_local", "is_holiday": "is_public_holiday"})
    holidays["date_local"] = pd.to_datetime(holidays["date_local"])
    holiday_dates = set(holidays.loc[holidays.is_public_holiday, "date_local"])
    holidays["is_day_before_holiday"] = (holidays.date_local + pd.Timedelta(days=1)).isin(holiday_dates)
    holidays["is_day_after_holiday"] = (holidays.date_local - pd.Timedelta(days=1)).isin(holiday_dates)
    result = frame.merge(holidays[["date_local", "is_public_holiday", "holiday_name",
                                   "is_day_before_holiday", "is_day_after_holiday"]],
                         on="date_local", how="left", validate="many_to_one")
    for column in ("is_public_holiday", "is_day_before_holiday", "is_day_after_holiday"):
        result[column] = result[column].astype("bool")
    return result


def add_observed_weather(frame: pd.DataFrame) -> pd.DataFrame:
    weather = pd.read_parquet(WEATHER_INPUT).rename(columns={
        "utc_datetime": "weather_observation_time_utc",
        "temperature_air_mean_2m_°C": "observed_temperature_c",
        "humidity_%": "observed_humidity_fraction",
        "cloud_cover_total_oktas_(0–8)": "observed_cloud_cover_fraction",
        "sunshine_duration_minutes_per_hour": "observed_sunshine_duration_seconds",
        "radiation_global_J/cm²_per_hour": "observed_global_radiation_j_cm2_per_hour",
    })
    keep = ["weather_observation_time_utc", "observed_temperature_c", "observed_humidity_fraction",
            "observed_cloud_cover_fraction", "observed_sunshine_duration_seconds",
            "observed_global_radiation_j_cm2_per_hour"]
    weather = weather[keep]
    result = frame.copy()
    result["weather_observation_time_utc"] = result.timestamp_utc.dt.floor("h")
    return result.merge(weather, on="weather_observation_time_utc", how="left", validate="many_to_one")


def add_prices(frame: pd.DataFrame) -> pd.DataFrame:
    prices = pd.read_parquet(PRICE_INPUT).rename(columns={
        "utc_datetime": "price_delivery_time_utc", "price_eur_per_mwh": "day_ahead_price_eur_per_mwh"})
    result = frame.copy()
    result["price_delivery_time_utc"] = result.timestamp_utc
    return result.merge(prices, on="price_delivery_time_utc", how="left", validate="one_to_one")


def add_split(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    mapping = {year: split for split, years in config["splits"].items() for year in years}
    result = frame.copy()
    result["split"] = result.year.map(mapping).astype("category")
    if result.split.isna().any():
        raise RuntimeError("At least one GOLD timestamp is outside the configured chronological splits")
    return result


def segment_metadata(timestamp: pd.Series, usable: pd.Series) -> pd.DataFrame:
    usable = usable.astype(bool).reset_index(drop=True)
    timestamp = timestamp.reset_index(drop=True)
    previous_row_usable = usable.shift(1, fill_value=False)
    consecutive_time = timestamp.diff().eq(pd.Timedelta(minutes=15)).fillna(False)
    starts = usable & (~previous_row_usable | ~consecutive_time)
    ids = starts.cumsum().where(usable).astype("Int32")
    previous_usable_time = timestamp.where(usable).ffill().shift(1)
    minutes = (timestamp - previous_usable_time).dt.total_seconds().div(60).astype("float64")
    position = usable.groupby(ids, dropna=True).cumcount().add(1).where(usable).astype("Int32")
    return pd.DataFrame({"segment_id_primary": ids,
                         "position_in_primary_segment": position,
                         "minutes_since_previous_usable_target": minutes})


def build_load(config: dict) -> pd.DataFrame:
    silver = pd.concat([pd.read_parquet(path) for path in LOAD_INPUTS], ignore_index=True)
    silver = silver.sort_values("timestamp_utc").reset_index(drop=True)
    if not silver.timestamp_utc.is_unique or not silver.timestamp_utc.is_monotonic_increasing:
        raise RuntimeError("Canonical load timestamps are not unique and monotonic")
    base = calendar_frame(silver.timestamp_utc, config["timezone"])
    rename = {
        "p_mean_kw": "load_mean_kw", "p_std_w": "load_std_w", "p_min_w": "load_min_w",
        "p_max_w": "load_max_w", "p_median_w": "load_median_w", "p_q05_w": "load_q05_w",
        "p_q25_w": "load_q25_w", "p_q75_w": "load_q75_w", "p_q95_w": "load_q95_w",
        "valid_count": "load_valid_count", "expected_count": "load_expected_count",
        "missing_count": "load_missing_count", "coverage": "load_coverage",
        "quality_100": "load_quality_100", "quality_99": "load_quality_99",
        "quality_95": "load_quality_95", "quality_90": "load_quality_90",
        "quality_80": "load_quality_80", "duplicate_observations_present": "load_duplicate_observations_present",
        "extreme_high": "load_extreme_high",
    }
    columns = list(rename)
    metrics = silver[columns].rename(columns=rename)
    for old in ("std", "min", "max", "median", "q05", "q25", "q75", "q95"):
        metrics[f"load_{old}_kw"] = metrics.pop(f"load_{old}_w") / 1000.0
    base = pd.concat([base, metrics], axis=1)
    base["load_energy_kwh"] = base.load_mean_kw * 0.25
    base["load_quality_75"] = base.load_coverage.ge(config["primary_quality"]["coverage_min"])
    base["target_quality_primary"] = base.load_quality_75
    base["target_quality_strict"] = base.load_coverage.ge(config["strict_quality"]["coverage_min"])
    base = add_holidays(base)
    base = add_observed_weather(base)
    base = add_prices(base)
    base = add_split(base, config)
    base = pd.concat([base, segment_metadata(base.timestamp_utc, base.target_quality_primary)], axis=1)
    return base


def build_pv(config: dict) -> pd.DataFrame:
    silver = pd.read_parquet(PV_INPUT).sort_values("timestamp_utc").reset_index(drop=True)
    if not silver.timestamp_utc.is_unique:
        raise RuntimeError("Canonical PV timestamps are not unique")
    pv = calendar_frame(silver.timestamp_utc, config["timezone"])
    rename = {
        "p_mean_kw": "pv_mean_kw", "p_std_w": "pv_std_w", "p_min_w": "pv_min_w", "p_max_w": "pv_max_w",
        "valid_count": "pv_valid_count", "expected_count": "pv_expected_count", "missing_count": "pv_missing_count",
        "coverage": "pv_coverage", "quality_100": "pv_quality_100", "quality_99": "pv_quality_99",
        "quality_95": "pv_quality_95", "quality_90": "pv_quality_90", "quality_80": "pv_quality_80",
    }
    metrics = silver[list(rename)].rename(columns=rename)
    for old in ("std", "min", "max"):
        metrics[f"pv_{old}_kw"] = metrics.pop(f"pv_{old}_w") / 1000.0
    pv = pd.concat([pv, metrics], axis=1)
    pv["pv_target_quality"] = pv.pv_coverage.ge(config["pv_quality"]["coverage_min"])
    pv = add_holidays(pv)
    pv = add_observed_weather(pv)
    pv["dataset_role"] = "secondary_limited_period"
    return pv


def build_net_load(load: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    load_columns = ["timestamp_utc", "timestamp_local", "date_local", "year", "month", "day", "hour", "minute",
                    "weekday", "hour_of_day", "minute_of_hour", "quarter_hour_of_day", "day_of_week", "is_weekend",
                    "day_of_year", "week_of_year", "sin_hour", "cos_hour", "sin_week", "cos_week", "sin_year",
                    "cos_year", "is_public_holiday", "holiday_name", "is_day_before_holiday", "is_day_after_holiday",
                    "observed_temperature_c", "observed_humidity_fraction", "observed_cloud_cover_fraction",
                    "observed_sunshine_duration_seconds", "observed_global_radiation_j_cm2_per_hour",
                    "weather_observation_time_utc", "load_mean_kw", "load_coverage", "load_valid_count",
                    "target_quality_primary", "split", "segment_id_primary"]
    pv_columns = ["timestamp_utc", "pv_mean_kw", "pv_coverage", "pv_valid_count", "pv_quality_95", "pv_target_quality"]
    joint = load.loc[load.year.eq(2024), load_columns].merge(pv[pv_columns], on="timestamp_utc", validate="one_to_one")
    joint = joint[joint.target_quality_primary & joint.pv_target_quality].copy()
    # Canonical metadata defines positive AC_ActivePower as generation.
    joint["net_load_kw"] = joint.load_mean_kw - joint.pv_mean_kw
    joint["pv_sign_convention"] = "positive_generation"
    joint["dataset_role"] = "limited_exploratory_net_load"
    return joint.reset_index(drop=True)


def feature_availability_rows() -> list[dict]:
    rows: list[dict] = []
    def add(names, source, unit, native, event, availability, future, safe, requires, primary, retro, notes):
        for name in names:
            rows.append({"feature_name": name, "source": source, "unit": unit, "native_resolution": native,
                         "event_time": event, "availability_time": availability, "future_known": future,
                         "day_ahead_safe": safe, "requires_historical_forecast": requires,
                         "allowed_primary_model": primary, "allowed_retrospective_model": retro, "notes": notes})
    add(["timestamp_utc", "timestamp_local", "date_local", "weather_observation_time_utc",
         "price_delivery_time_utc"], "canonical event-time identifiers", "datetime", "15min/1h",
        "as named", "deterministic identifier; observation availability follows associated variable", "identifier",
        "identifier", "no", "index_only", "index_only", "UTC is the unique chronological key")
    add(["split"], "deterministic GOLD time split", "category", "15min", "target interval",
        "known from study design", "yes", "yes", "no", "partition_only", "partition_only",
        "train=2021-2022, validation=2023, test=2024")
    add(["year", "month", "day", "hour", "minute", "weekday", "hour_of_day", "minute_of_hour",
         "quarter_hour_of_day", "day_of_week", "is_weekend", "day_of_year", "week_of_year",
         "sin_hour", "cos_hour", "sin_week", "cos_week", "sin_year", "cos_year"],
        "deterministic Europe/Berlin calendar", "various", "15min", "target delivery interval",
        "known indefinitely in advance", "yes", "yes", "no", "yes", "yes", "DST derived from UTC")
    add(["is_public_holiday", "holiday_name", "is_day_before_holiday", "is_day_after_holiday"],
        "SPARK Karlsruhe/Baden-Württemberg holiday calendar", "boolean/string", "1D local", "local calendar date",
        "published calendar; known in advance", "yes", "yes", "no", "yes", "yes", "joined by local date")
    add(["load_mean_kw", "load_std_kw", "load_min_kw", "load_max_kw", "load_median_kw", "load_q05_kw",
         "load_q25_kw", "load_q75_kw", "load_q95_kw"],
        "canonical TotalLoad SILVER P_total", "kW", "15min from 5s", "target interval",
        "after interval measurement/aggregation", "no", "no", "no", "target_only", "target_only",
        "load_mean_kw is the primary target")
    add(["load_energy_kwh"], "GOLD deterministic derivation from load_mean_kw", "kWh", "15min", "target interval",
        "after interval measurement/aggregation", "no", "no", "no", "target_only", "target_only",
        "derived exactly as load_mean_kw * 0.25 h; does not replace the power target")
    add(["load_valid_count", "load_expected_count", "load_missing_count", "load_coverage", "load_quality_100",
         "load_quality_99", "load_quality_95", "load_quality_90", "load_quality_80", "load_quality_75",
         "target_quality_primary", "target_quality_strict", "load_duplicate_observations_present", "load_extreme_high",
         "segment_id_primary", "position_in_primary_segment", "minutes_since_previous_usable_target"],
        "canonical TotalLoad SILVER / GOLD quality derivation", "count/fraction/boolean", "15min", "target interval",
        "after target observation", "no", "no", "no", "metadata_only", "metadata_only",
        "retain for filtering/audit; do not assume future target quality is known")
    for name, unit in (("observed_temperature_c", "degree Celsius"),
                       ("observed_humidity_fraction", "decimal fraction 0-1"),
                       ("observed_cloud_cover_fraction", "decimal fraction of eighths"),
                       ("observed_sunshine_duration_seconds", "seconds per hour"),
                       ("observed_global_radiation_j_cm2_per_hour", "J/cm2 per hour")):
        add([name], "canonical SPARK/DWD observed weather", unit, "1h", "weather observation hour",
            "after observation; publication latency not encoded", "no", "no", "yes", "no", "yes",
            "contemporaneous observed weather is oracle/retrospective, not historical NWP")
    add(["day_ahead_price_eur_per_mwh"], "canonical German day-ahead price", "EUR/MWh", "15min",
        "delivery interval", "after market publication; exact auction publication time absent", "conditional",
        "conditional", "no", "conditional", "yes", "depends on forecast issuance time")
    add(["pv_mean_kw", "pv_std_kw", "pv_min_kw", "pv_max_kw"], "canonical PV SILVER AC_ActivePower", "kW",
        "15min from 5s", "target interval", "after PV measurement", "no", "no", "yes", "no", "yes",
        "future measured PV prohibited for primary load forecasting; positive means generation")
    add(["pv_valid_count", "pv_expected_count", "pv_missing_count", "pv_coverage", "pv_quality_100",
         "pv_quality_99", "pv_quality_95", "pv_quality_90", "pv_quality_80", "pv_target_quality"],
        "canonical PV SILVER / GOLD quality derivation", "count/fraction/boolean", "15min", "target interval",
        "after PV observation", "no", "no", "no", "metadata_only", "metadata_only", "PV quality metadata")
    add(["net_load_kw"], "GOLD load minus positive PV generation", "kW", "15min", "target interval",
        "after both measurements unless both are forecast", "no", "no", "yes", "target_only", "target_only",
        "limited net-load target; measured future PV is not a predictor")
    add(["dataset_role", "pv_sign_convention"], "GOLD scope/provenance metadata", "string", "dataset/15min",
        "not an event-valued predictor", "known from dataset construction", "yes", "yes", "no",
        "metadata_only", "metadata_only", "scope and sign safeguards; not predictive features")
    return rows


def descriptions() -> dict[str, str]:
    result = {
        "timestamp_utc": "Unique canonical UTC interval start.", "timestamp_local": "Europe/Berlin civil interval start without timezone offset.",
        "date_local": "Europe/Berlin local calendar date.", "split": "Deterministic chronological research split.",
        "load_mean_kw": "Primary quarter-hour mean industrial demand.", "load_energy_kwh": "Quarter-hour energy derived as load_mean_kw multiplied by 0.25 h.",
        "target_quality_primary": "True when load coverage is at least 75% (135/180).",
        "target_quality_strict": "True when load coverage is at least 95% (171/180).",
        "segment_id_primary": "Identifier for a contiguous UTC run of primary-quality targets; null otherwise.",
        "position_in_primary_segment": "One-based position within a primary-quality segment.",
        "minutes_since_previous_usable_target": "Elapsed minutes since the preceding primary-quality target.",
        "weather_observation_time_utc": "Native hourly observed-weather timestamp used for alignment.",
        "price_delivery_time_utc": "Day-ahead market delivery-interval timestamp.",
        "dataset_role": "Dataset scope/role warning.", "net_load_kw": "Industrial load minus positive on-site PV generation.",
        "pv_sign_convention": "Explicit sign interpretation used for net-load derivation.",
    }
    return result


def aggregation_description(name: str) -> str:
    """Describe how each GOLD field is formed without implying imputation."""
    if name in {"load_mean_kw", "pv_mean_kw"}:
        return "arithmetic mean of valid 5-second samples"
    if name in {"load_std_kw", "pv_std_kw"}:
        return "sample standard deviation of valid 5-second samples"
    if name.endswith("_min_kw"):
        return "minimum of valid 5-second samples"
    if name.endswith("_max_kw"):
        return "maximum of valid 5-second samples"
    if name.endswith("_median_kw"):
        return "median of valid 5-second samples"
    for quantile in ("q05", "q25", "q75", "q95"):
        if f"_{quantile}_kw" in name:
            return f"{quantile[1:]}th percentile of valid 5-second samples"
    if name.endswith("_valid_count"):
        return "count of valid 5-second samples"
    if name.endswith("_expected_count"):
        return "fixed expected count for a complete 15-minute interval (180)"
    if name.endswith("_missing_count"):
        return "expected_count minus valid_count"
    if name.endswith("_coverage"):
        return "valid_count divided by expected_count"
    if "quality" in name:
        return "deterministic threshold comparison; no imputation"
    if name == "load_energy_kwh":
        return "load_mean_kw multiplied by 0.25 hours"
    if name == "net_load_kw":
        return "load_mean_kw minus positive pv_mean_kw"
    if name.startswith("observed_"):
        return "exact join from the containing native UTC weather hour; no interpolation"
    if name == "day_ahead_price_eur_per_mwh":
        return "exact join on the 15-minute delivery timestamp; no filling"
    if name in {"is_public_holiday", "holiday_name", "is_day_before_holiday", "is_day_after_holiday"}:
        return "deterministic join/derivation by Europe/Berlin local date"
    if name in {"sin_hour", "cos_hour", "sin_week", "cos_week", "sin_year", "cos_year"}:
        return "deterministic cyclical transform of local civil time"
    if name in {"year", "month", "day", "hour", "minute", "weekday", "hour_of_day", "minute_of_hour",
                "quarter_hour_of_day", "day_of_week", "is_weekend", "day_of_year", "week_of_year"}:
        return "deterministic derivation from Europe/Berlin local civil time"
    return "none; identifier, source value, or deterministic metadata"


def write_docs(frames: dict[str, pd.DataFrame]) -> None:
    availability = pd.DataFrame(feature_availability_rows())
    documented = set(availability.feature_name)
    actual = set().union(*(set(frame.columns) for frame in frames.values()))
    missing = sorted(actual - documented)
    if missing:
        raise RuntimeError(f"Feature-availability metadata missing GOLD columns: {missing}")
    availability.to_csv(AVAILABILITY_CSV, index=False)
    AVAILABILITY_MD.write_text("# GOLD feature availability and leakage classes\n\n"
        "`yes` means safe for the primary day-ahead design; `conditional` requires a formally defined issuance time; "
        "`metadata_only` and `target_only` are not predictors. No lag or rolling features are included in base GOLD.\n\n"
        + availability.to_markdown(index=False) + "\n", encoding="utf-8")

    available = availability.set_index("feature_name").to_dict("index")
    desc = descriptions()
    rows = []
    for artifact, frame in frames.items():
        for name, dtype in frame.dtypes.items():
            meta = available.get(name, {})
            rows.append({"dataset": artifact, "name": name, "description": desc.get(name, name.replace("_", " ")),
                         "source": meta.get("source", "GOLD deterministic derivation"), "unit": meta.get("unit", ""),
                         "resolution": "15min", "aggregation": aggregation_description(name),
                         "quality_meaning": (desc.get(name, meta.get("notes", ""))
                                             if any(token in name for token in ("quality", "coverage", "valid_count",
                                                                                "missing_count", "expected_count",
                                                                                "segment")) else ""),
                         "timezone": "UTC" if name.endswith("_utc") else ("Europe/Berlin" if name in {"timestamp_local", "date_local"} else ""),
                         "point_in_time_availability": meta.get("availability_time", "deterministic/output metadata"),
                         "primary_model_eligibility": meta.get("allowed_primary_model", "not_a_predictor"),
                         "known_limitations": meta.get("notes", ""), "dtype": str(dtype)})
    dictionary = pd.DataFrame(rows)
    dictionary.to_csv(DICTIONARY_CSV, index=False)
    DICTIONARY_MD.write_text("# GOLD data dictionary\n\n" + dictionary.to_markdown(index=False) + "\n", encoding="utf-8")


def contiguous_summary(frame: pd.DataFrame) -> pd.DataFrame:
    usable = frame[frame.target_quality_primary & frame.segment_id_primary.notna()]
    return usable.groupby(["year", "segment_id_primary"], observed=True).agg(
        start_utc=("timestamp_utc", "min"), end_utc=("timestamp_utc", "max"), intervals=("timestamp_utc", "size"),
        duration_hours=("timestamp_utc", lambda x: len(x) * .25)).reset_index()


def artifact_record(path: Path, frame: pd.DataFrame, input_files: list[Path]) -> dict:
    return {"path": str(path.relative_to(ROOT)), "rows": len(frame),
            "timestamp_start": frame.timestamp_utc.min().isoformat(), "timestamp_end": frame.timestamp_utc.max().isoformat(),
            "timestamp_unique": bool(frame.timestamp_utc.is_unique),
            "schema": [{"name": name, "dtype": str(dtype)} for name, dtype in frame.dtypes.items()],
            "output_bytes": path.stat().st_size, "output_sha256": file_digest(path),
            "input_silver_files": [{"path": str(source.relative_to(ROOT)), "sha256": file_digest(source)}
                                   for source in input_files]}


def joint_statistics(load: pd.DataFrame, net: pd.DataFrame) -> dict:
    membership = load.loc[load.year.eq(2024), ["timestamp_utc", "date_local"]].copy()
    membership["joint"] = membership.timestamp_utc.isin(set(net.timestamp_utc))
    daily = membership.groupby("date_local").joint.agg(["sum", "size"])
    local = membership.timestamp_utc.dt.tz_convert("Europe/Berlin")
    iso = local.dt.isocalendar()
    weekly = pd.DataFrame({"iso_year": iso.year.to_numpy(), "iso_week": iso.week.to_numpy(),
                           "joint": membership.joint.to_numpy()}).groupby(["iso_year", "iso_week"]).joint.agg(["sum", "size"])
    usable = membership.joint.to_numpy()
    starts = np.flatnonzero(usable & np.r_[True, ~usable[:-1]])
    ends = np.flatnonzero(usable & np.r_[~usable[1:], True])
    longest = max((int(end - start + 1), int(start), int(end)) for start, end in zip(starts, ends, strict=True))
    return {"joint_usable_intervals": int(len(net)),
            "complete_local_days": int((daily["sum"] == daily["size"]).sum()),
            "complete_iso_weeks": int((weekly["sum"] == weekly["size"]).sum()),
            "longest_contiguous_intervals": longest[0], "longest_contiguous_hours": longest[0] * .25,
            "longest_start_utc": membership.timestamp_utc.iloc[longest[1]].isoformat(),
            "longest_end_utc": membership.timestamp_utc.iloc[longest[2]].isoformat()}


def write_summary(load: pd.DataFrame, pv: pd.DataFrame, net: pd.DataFrame, config: dict) -> None:
    segments = contiguous_summary(load)
    split_counts = load.groupby("split", observed=True).agg(rows=("timestamp_utc", "size"),
        primary_targets=("target_quality_primary", "sum"), strict_targets=("target_quality_strict", "sum"))
    primary_load = load[load.target_quality_primary]
    split_distribution = primary_load.groupby("split", observed=True).agg(
        primary_mean_load_kw=("load_mean_kw", "mean"), primary_std_load_kw=("load_mean_kw", "std"),
        primary_min_load_kw=("load_mean_kw", "min"), primary_q25_load_kw=("load_mean_kw", lambda x: x.quantile(.25)),
        primary_median_load_kw=("load_mean_kw", "median"), primary_q75_load_kw=("load_mean_kw", lambda x: x.quantile(.75)),
        primary_max_load_kw=("load_mean_kw", "max"))
    split_summary = split_counts.join(split_distribution)
    annual = load.groupby("year").agg(rows=("timestamp_utc", "size"), primary=("target_quality_primary", "sum"),
        strict=("target_quality_strict", "sum"), empty=("load_valid_count", lambda x: int((x == 0).sum())),
        mean_coverage=("load_coverage", "mean"))
    coverage_distribution = load.groupby("year").load_coverage.quantile([0, .05, .25, .5, .75, .95, 1]).unstack()
    coverage_distribution.columns = ["min", "p05", "p25", "median", "p75", "p95", "max"]
    seg_annual = segments.groupby("year").agg(segments=("segment_id_primary", "size"),
        median_duration_hours=("duration_hours", "median"), max_duration_hours=("duration_hours", "max"))
    pv_good = pv[pv.pv_target_quality]
    joint = joint_statistics(load, net)
    lines = ["# GOLD quality summary", "", "## Primary load dataset", "",
             "The full UTC-indexed 15-minute grid is retained. No target values are interpolated, filled, smoothed, or clipped.", "",
             annual.join(seg_annual).to_markdown(), "", "### Chronological splits and target distribution", "",
             split_summary.to_markdown(), "", "The primary rule is coverage >=75% (135/180), selected empirically from the finalized masking study; it is not a heuristic. Missingness is mixed, with a dominant approximately 20-second periodic pattern plus clustered outages.", "",
             "### Coverage distribution by year", "", coverage_distribution.to_markdown(), "",
             "The strict >=95% subset is a sensitivity resource, not a parallel 2021–2024 experiment: it has only 14 intervals in 2023 and none in 2024.", "",
             "## Feature availability", "",
             "Calendar and Baden-Württemberg/Karlsruhe holiday features are future-known and safe. Target coverage and segment fields are outcome-quality metadata, not assumed future-known predictors. Contemporaneous observed DWD weather is retained for retrospective/oracle analysis and is explicitly not day-ahead-safe. Day-ahead price availability is conditional on the eventual forecast issuance time because publication timing is not encoded in the source.", "",
             "No load lags or rolling statistics are included in base GOLD; consequently no hidden window boundary or row-shift assumptions are present. Emissions status: `optional_not_in_primary_gold`.", "",
             "## PV dataset", "",
             f"The secondary full-grid 2024 PV panel has {len(pv):,} rows and {len(pv_good):,} intervals at >=95% coverage. Its measured usable range is {pv_good.timestamp_utc.min().isoformat()} through {pv_good.timestamp_utc.max().isoformat()}. The source represents only January through early April measurements and cannot support a full-year PV claim.", "",
             "## Net-load dataset", "",
             f"The joint dataset has {len(net):,} intervals after requiring load >=75% and PV >=95%. Net load equals load minus PV after validation that PV is positive generation. Its usable range is {net.timestamp_utc.min().isoformat()} through {net.timestamp_utc.max().isoformat()}.", "",
             f"It contains {joint['complete_local_days']} complete local-calendar days, {joint['complete_iso_weeks']} complete ISO weeks, and a longest contiguous block of {joint['longest_contiguous_intervals']:,} intervals ({joint['longest_contiguous_hours']:.0f} hours), from {joint['longest_start_utc']} through {joint['longest_end_utc']}.", "",
             "It is limited exploratory data, not evidence for annual seasonal net-load forecasting. Future measured PV is not permitted as a load-only predictor.", "",
             "## Leakage audit", "",
             "- Future-known: deterministic calendar and published holiday variables.",
             "- Past-only: measured load can only become a predictor through explicitly historical timestamps in a later experiment pipeline.",
             "- Retrospective/oracle: contemporaneous observed weather and future measured PV.",
             "- Conditional: day-ahead price, pending a formal forecast issuance time.",
             "- Prohibited for primary load forecasting: contemporaneous/future load, target-quality outcomes, future measured PV, and contemporaneous observed future weather.", ""]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def identity(config: dict) -> dict:
    inputs = [*LOAD_INPUTS, PV_INPUT, WEATHER_INPUT, PRICE_INPUT, HOLIDAY_INPUT,
              ROOT / config["missingness_study"], ROOT / config["weather_unit_resolution"]]
    return {"gold_version": config["gold_version"], "config": config,
            "code_sha256": file_digest(Path(__file__)),
            "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": file_digest(path), "bytes": path.stat().st_size}
                       for path in inputs]}


def valid_existing(expected_identity: dict) -> bool:
    if not MANIFEST_PATH.exists() or not all(path.exists() for path in OUTPUTS.values()):
        return False
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("identity") != expected_identity:
        return False
    by_name = manifest.get("artifacts", {})
    return all(name in by_name and by_name[name]["output_sha256"] == file_digest(path)
               for name, path in OUTPUTS.items())


def main() -> None:
    config = load_config()
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    expected_identity = identity(config)
    if valid_existing(expected_identity):
        print("GOLD outputs and manifest identity are unchanged and valid; skipping rebuild")
        return
    load = build_load(config)
    pv = build_pv(config)
    net = build_net_load(load, pv)
    frames = {
        "load_15min_base": load,
        "load_15min_primary": load[load.target_quality_primary].copy(),
        "load_15min_strict": load[load.target_quality_strict].copy(),
        "pv_2024_15min": pv,
        "net_load_2024_15min": net,
    }
    for name, frame in frames.items():
        frame.to_parquet(OUTPUTS[name], index=False, compression="zstd")
        check = pd.read_parquet(OUTPUTS[name], columns=["timestamp_utc"])
        if len(check) != len(frame) or not check.timestamp_utc.is_unique:
            raise RuntimeError(f"Post-write validation failed for {name}")
    write_docs(frames)
    write_summary(load, pv, net, config)
    input_map = {
        "load_15min_base": [*LOAD_INPUTS, WEATHER_INPUT, PRICE_INPUT, HOLIDAY_INPUT],
        "load_15min_primary": [*LOAD_INPUTS, WEATHER_INPUT, PRICE_INPUT, HOLIDAY_INPUT],
        "load_15min_strict": [*LOAD_INPUTS, WEATHER_INPUT, PRICE_INPUT, HOLIDAY_INPUT],
        "pv_2024_15min": [PV_INPUT, WEATHER_INPUT, HOLIDAY_INPUT],
        "net_load_2024_15min": [LOAD_INPUTS[-1], PV_INPUT, WEATHER_INPUT, HOLIDAY_INPUT],
    }
    joint = joint_statistics(load, net)
    segments = contiguous_summary(load)
    manifest = {
        "status": "GOLD_RESEARCH_DATASETS_VALIDATED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": expected_identity,
        "git_commit": git_commit(),
        "git_commit_note": "null means repository initialized but no commit exists",
        "artifacts": {name: artifact_record(OUTPUTS[name], frame, input_map[name]) for name, frame in frames.items()},
        "quality_rules": config["primary_quality"] | {"strict": config["strict_quality"], "pv": config["pv_quality"]},
        "weather": {"alignment": config["weather_alignment"], "interpolation": "none",
                    "field_class": "observed_historical_weather_retrospective_oracle"},
        "price": {"unit": "EUR/MWh", "availability_class": config["price_availability_class"],
                  "availability_time": "not encoded; evaluate against formal forecast issuance time"},
        "pv_sign_convention": "positive means generation",
        "net_load_definition": "load_mean_kw - pv_mean_kw",
        "scientific_summary": {
            "primary_quality_by_year": {str(year): int(value) for year, value in
                                        load.groupby("year").target_quality_primary.sum().items()},
            "primary_segments_by_year": {str(year): int(value) for year, value in
                                          segments.groupby("year").size().items()},
            "joint_load_pv": joint,
        },
        "emissions_status": config["emissions_status"],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({name: {"rows": len(frame), "path": str(OUTPUTS[name])} for name, frame in frames.items()}, indent=2))


if __name__ == "__main__":
    main()
