import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"


@pytest.fixture(scope="module")
def load():
    return pd.read_parquet(GOLD / "load_15min_base.parquet")


@pytest.fixture(scope="module")
def availability():
    return pd.read_csv(ROOT / "docs" / "gold_feature_availability.csv").set_index("feature_name")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def test_full_timeline_is_unique_monotonic_and_uncompressed(load):
    assert len(load) == 140_256
    assert load.timestamp_utc.is_unique
    assert load.timestamp_utc.is_monotonic_increasing
    assert load.timestamp_utc.diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    assert load.timestamp_local.duplicated().any()  # expected autumn DST folds


def test_primary_and_strict_quality_rules_are_exact(load):
    assert (load.load_expected_count == 180).all()
    assert load.load_quality_75.equals(load.load_valid_count.ge(135))
    assert load.target_quality_primary.equals(load.load_coverage.ge(.75))
    assert load.target_quality_strict.equals(load.load_valid_count.ge(171))
    expected = {2021: 34_948, 2022: 34_884, 2023: 32_676, 2024: 31_339}
    assert load.groupby("year").target_quality_primary.sum().astype(int).to_dict() == expected


def test_low_quality_targets_are_not_imputed(load):
    empty = load.load_valid_count.eq(0)
    target_columns = ["load_mean_kw", "load_std_kw", "load_min_kw", "load_max_kw", "load_median_kw",
                      "load_q05_kw", "load_q25_kw", "load_q75_kw", "load_q95_kw", "load_energy_kwh"]
    assert load.loc[empty, target_columns].isna().all().all()
    assert load.loc[~empty, "load_mean_kw"].notna().all()
    assert np.allclose(load.load_energy_kwh, load.load_mean_kw * .25, equal_nan=True)


def test_chronological_splits_are_disjoint_and_exact(load):
    expected = {2021: "train", 2022: "train", 2023: "validation", 2024: "test"}
    assert all(load.loc[load.year.eq(year), "split"].eq(split).all() for year, split in expected.items())
    ranges = load.groupby("split", observed=True).timestamp_utc.agg(["min", "max"])
    assert ranges.loc["train", "max"] < ranges.loc["validation", "min"] < ranges.loc["test", "min"]


def test_primary_segments_do_not_bridge_gaps(load):
    assert load.loc[~load.target_quality_primary, "segment_id_primary"].isna().all()
    usable = load[load.target_quality_primary]
    for _, group in usable.groupby("segment_id_primary", observed=True):
        assert group.timestamp_utc.diff().dropna().eq(pd.Timedelta(minutes=15)).all()
        assert group.position_in_primary_segment.tolist() == list(range(1, len(group) + 1))
    starts = usable.groupby("segment_id_primary", observed=True).head(1)
    assert starts.position_in_primary_segment.eq(1).all()


def test_no_lag_or_rolling_feature_can_hide_future_information(load):
    lag_columns = [c for c in load if c.startswith("load_lag_")]
    rolling_columns = [c for c in load if c.startswith("rolling_")]
    assert lag_columns == []
    assert rolling_columns == []
    prohibited_learned = [c for c in load if re.search(r"scaled|normalized|zscore|train_mean", c, re.I)]
    assert prohibited_learned == []


def test_observed_weather_is_not_day_ahead_safe(availability):
    fields = [x for x in availability.index if x.startswith("observed_")]
    assert fields
    assert availability.loc[fields, "day_ahead_safe"].eq("no").all()
    assert availability.loc[fields, "allowed_primary_model"].eq("no").all()
    assert availability.loc[fields, "requires_historical_forecast"].eq("yes").all()


def test_future_measured_pv_is_prohibited_for_primary_load_model(availability):
    fields = ["pv_mean_kw", "pv_std_kw", "pv_min_kw", "pv_max_kw"]
    assert availability.loc[fields, "day_ahead_safe"].eq("no").all()
    assert availability.loc[fields, "allowed_primary_model"].eq("no").all()


def test_price_is_conditional_on_issuance_time(availability):
    price = availability.loc["day_ahead_price_eur_per_mwh"]
    assert price.day_ahead_safe == "conditional"
    assert price.allowed_primary_model == "conditional"
    assert "issuance" in price.notes


def test_weather_alignment_is_exact_hour_and_unfilled(load):
    assert load.weather_observation_time_utc.equals(load.timestamp_utc.dt.floor("h"))
    weather = pd.read_parquet(
        ROOT / "data" / "silver" / "exogenous" / "canonical" / "weather_karlsruhe_hourly_original_2021_2024.parquet")
    missing_hours = set(weather.loc[weather["temperature_air_mean_2m_°C"].isna(), "utc_datetime"])
    mapped = load.weather_observation_time_utc.isin(missing_hours)
    assert mapped.any()
    assert load.loc[mapped, "observed_temperature_c"].isna().all()


def test_pv_and_net_load_scope_and_sign():
    pv = pd.read_parquet(GOLD / "pv_2024_15min.parquet")
    net = pd.read_parquet(GOLD / "net_load_2024_15min.parquet")
    assert int(pv.pv_target_quality.sum()) == 8_252
    assert pv.dataset_role.eq("secondary_limited_period").all()
    assert len(net) == 7_392
    assert net.timestamp_utc.is_unique
    assert net.target_quality_primary.all() and net.pv_target_quality.all()
    assert np.allclose(net.net_load_kw, net.load_mean_kw - net.pv_mean_kw)
    assert net.pv_sign_convention.eq("positive_generation").all()


def test_filtered_variants_preserve_timestamps_and_gaps(load):
    primary = pd.read_parquet(GOLD / "load_15min_primary.parquet")
    strict = pd.read_parquet(GOLD / "load_15min_strict.parquet")
    assert primary.timestamp_utc.equals(load.loc[load.target_quality_primary, "timestamp_utc"].reset_index(drop=True))
    assert strict.timestamp_utc.equals(load.loc[load.target_quality_strict, "timestamp_utc"].reset_index(drop=True))
    assert primary.timestamp_utc.diff().gt(pd.Timedelta(minutes=15)).any()


def test_manifest_hashes_and_silver_inputs_are_valid():
    manifest = json.loads((ROOT / "reports" / "gold_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "GOLD_RESEARCH_DATASETS_VALIDATED"
    for artifact in manifest["artifacts"].values():
        assert digest(ROOT / artifact["path"]) == artifact["output_sha256"]
    for silver_manifest in (ROOT / "data" / "silver").glob("**/*.manifest.json"):
        saved = json.loads(silver_manifest.read_text(encoding="utf-8"))
        output = Path(saved["output"])
        assert output.exists()
        assert digest(output) == saved["output_sha256"]


def test_raw_archive_verification_cache_still_binds_hash_to_file_state():
    for name in ("EPI_TotalLoad.zip", "EPI_PV.zip"):
        archive = ROOT / "data" / "raw" / name
        saved = json.loads((archive.with_suffix(".zip.verified.json")).read_text(encoding="utf-8"))
        stat = archive.stat()
        assert stat.st_size == saved["size"]
        assert stat.st_mtime_ns == saved["mtime_ns"]
        assert saved["md5"] in {"cbbffe29f9d8e224b9221cb5904a00bb", "f34a321b3aee4beca9864c09c68b2e36"}
