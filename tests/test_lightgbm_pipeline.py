import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from day_ahead_features import issuance_features  # noqa: E402
from train_lightgbm import assert_predictors_safe, fit_development, refit_pretest  # noqa: E402


@pytest.fixture(scope="module")
def feature_frame():
    return pd.read_parquet(ROOT / "data" / "features" / "lightgbm_features.parquet")


@pytest.fixture(scope="module")
def frozen():
    return json.loads((ROOT / "config" / "lightgbm_frozen.json").read_text(encoding="utf-8"))


def test_feature_spec_reuses_authoritative_availability_engine():
    spec = json.loads((ROOT / "config" / "lightgbm_features.json").read_text(encoding="utf-8"))
    assert spec["authoritative_availability_function"] == \
        "src/day_ahead_baselines.py::observation_available"


def test_no_target_or_quality_value_is_a_predictor(frozen):
    forbidden = {"load_mean_kw", "load_coverage", "load_valid_count", "target_quality_primary"}
    assert forbidden.isdisjoint(frozen["features"])


def test_no_weather_pv_or_price_predictors(frozen):
    tokens = ("observed_", "temperature", "radiation", "humidity", "cloud", "sunshine", "pv_", "price")
    assert not [feature for feature in frozen["features"] if any(token in feature.lower() for token in tokens)]


def test_synthetic_post_noon_extreme_cannot_change_issuance_features():
    issue = pd.Timestamp("2024-01-08 12:00", tz="Europe/Berlin").tz_convert("UTC")
    base = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2024-01-08 09:00Z", "2024-01-08 10:45Z"]),
        "date_local": pd.to_datetime(["2024-01-08", "2024-01-08"]),
        "target_quality_primary": [True, True], "load_mean_kw": [5.0, 10.0]})
    attacked = pd.concat([base, pd.DataFrame({"timestamp_utc": [pd.Timestamp("2024-01-08 11:00Z")],
        "date_local": [pd.Timestamp("2024-01-08")], "target_quality_primary": [True],
        "load_mean_kw": [1_000_000.0]})], ignore_index=True)
    left = issuance_features(base, pd.Series([issue])).iloc[0]
    right = issuance_features(attacked, pd.Series([issue])).iloc[0]
    for column in left.index.drop("forecast_issue_time_utc"):
        assert left[column] == pytest.approx(right[column], nan_ok=True)
    assert left.latest_load_kw == 10.0


def test_all_historical_analog_sources_end_by_issuance(feature_frame):
    sourced = feature_frame[feature_frame.latest_analog_source_time_utc.notna()]
    assert (sourced.latest_analog_source_time_utc + pd.Timedelta(minutes=15) <=
            sourced.forecast_issue_time_utc).all()
    assert feature_frame.latest_load_age_minutes.dropna().ge(0).all()


def test_previous_day_partial_window_never_uses_afternoon_extreme():
    issue = pd.Timestamp("2024-01-08 11:00Z")
    rows = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2024-01-08 00:00Z", "2024-01-08 10:45Z",
                                                           "2024-01-08 16:00Z"]),
        "date_local": pd.to_datetime(["2024-01-08"] * 3), "target_quality_primary": True,
        "load_mean_kw": [4.0, 8.0, 999_999.0]})
    result = issuance_features(rows, pd.Series([issue])).iloc[0]
    assert result.previous_day_midnight_to_issue_max == 8.0


def test_chronological_split_separation(feature_frame):
    bounds = feature_frame.groupby(feature_frame.split.astype(str)).timestamp_utc.agg(["min", "max"])
    assert bounds.loc["train", "max"] < bounds.loc["validation", "min"]
    assert bounds.loc["validation", "max"] < bounds.loc["test", "min"]


def test_development_api_rejects_test_rows():
    with pytest.raises(RuntimeError, match="TEST rows"):
        fit_development(pd.DataFrame({"split": ["test"]}), [], "direct", {})


def test_pretest_refit_api_rejects_test_rows(frozen):
    with pytest.raises(RuntimeError, match="2021-2023"):
        refit_pretest(pd.DataFrame({"split": ["test"]}), frozen["features"], frozen)


def test_frozen_configuration_precedes_test_prediction_artifact(frozen):
    config = ROOT / "config" / "lightgbm_frozen.json"
    predictions = ROOT / "results" / "lightgbm" / "lightgbm_predictions.parquet"
    assert frozen["written_before_test_scoring"] is True
    assert frozen["test_data_used_for_selection"] is False
    assert config.stat().st_mtime_ns <= predictions.stat().st_mtime_ns


def test_final_refit_count_is_exactly_eligible_2021_2023(feature_frame):
    manifest = json.loads((ROOT / "results" / "lightgbm" / "lightgbm_run_manifest.json").read_text())
    expected = int(feature_frame[feature_frame.split.isin(["train", "validation"])].target_quality_primary.sum())
    assert manifest["pretest_training_examples"] == expected == 102_508


def test_missing_features_do_not_prevent_complete_predictions():
    predictions = pd.read_parquet(ROOT / "results" / "lightgbm" / "lightgbm_predictions.parquet")
    learned = predictions[predictions.model_name.eq("LGBM_DIRECT") & predictions.split.eq("test")]
    assert len(learned) == 35_136
    assert learned.y_pred_kw.notna().all()
    assert learned.target_time_utc.is_unique


def test_feature_matrix_and_frozen_predictors_are_deterministic_and_safe(feature_frame, frozen):
    spec = json.loads((ROOT / "config" / "lightgbm_features.json").read_text())
    assert_predictors_safe(frozen["features"], spec)
    assert set(frozen["features"]).issubset(feature_frame.columns)
    assert frozen["feature_matrix_sha256"] == hashlib.sha256(
        (ROOT / "data" / "features" / "lightgbm_features.parquet").read_bytes()).hexdigest()


def test_run_manifest_hashes_match_outputs():
    manifest_path = ROOT / "results" / "lightgbm" / "lightgbm_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, metadata in manifest["outputs"].items():
        path = manifest_path.parent / relative
        assert path.stat().st_size == metadata["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
