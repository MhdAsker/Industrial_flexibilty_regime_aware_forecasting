import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from day_ahead_baselines import (  # noqa: E402
    b0_persistence,
    b1_safe_recent_day,
    b2_weekly,
    calendar_profiles,
    day_block_bootstrap,
    issue_time_for_target_date,
    local_day_grid,
    make_history_index,
    metric_values,
    observation_available,
)


def synthetic_load(rows):
    frame = pd.DataFrame(rows, columns=["timestamp_utc", "date_local", "quarter_hour_of_day", "load_mean_kw"])
    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    frame["date_local"] = pd.to_datetime(frame.date_local)
    frame["target_quality_primary"] = True
    return frame


@pytest.fixture(scope="module")
def predictions():
    return pd.read_parquet(ROOT / "results" / "baselines" / "day_ahead_baseline_predictions.parquet")


def test_interval_end_issuance_cutoff():
    issue = pd.Timestamp("2024-01-08 12:00", tz="Europe/Berlin").tz_convert("UTC")
    assert observation_available(pd.Timestamp("2024-01-08 10:45", tz="UTC"), issue)
    assert not observation_available(pd.Timestamp("2024-01-08 11:00", tz="UTC"), issue)
    with pytest.raises(ValueError):
        observation_available(pd.Timestamp("2024-01-08 10:45"), issue)


def test_future_extreme_cannot_affect_earlier_forecast():
    load = synthetic_load([
        ("2024-01-07 10:00Z", "2024-01-07", 44, 5.0),
        ("2024-01-08 10:45Z", "2024-01-08", 44, 7.0),
        ("2024-01-08 11:00Z", "2024-01-08", 44, 1_000_000.0),
    ])
    history = make_history_index(load)
    issue = pd.Timestamp("2024-01-08 11:00Z")
    value, source = b0_persistence(history, issue)
    assert value == 7.0
    assert source == pd.Timestamp("2024-01-08 10:45Z")


def test_b1_adapts_lookback_across_noon_cutoff():
    # Tuesday target issued Monday 12:00 CET == 11:00 UTC.
    load = synthetic_load([
        ("2024-01-07 07:00Z", "2024-01-07", 32, 8.0),
        ("2024-01-07 17:00Z", "2024-01-07", 72, 18.0),
        ("2024-01-08 07:00Z", "2024-01-08", 32, 9.0),
        ("2024-01-08 17:00Z", "2024-01-08", 72, 999.0),
    ])
    history = make_history_index(load)
    issue = issue_time_for_target_date(date(2024, 1, 9)).tz_convert("UTC")
    early, _ = b1_safe_recent_day(history, date(2024, 1, 9), 32, issue)
    late, _ = b1_safe_recent_day(history, date(2024, 1, 9), 72, issue)
    assert early == 9.0
    assert late == 18.0


def test_weekly_uses_exact_local_calendar_match_and_missing_stays_missing():
    load = synthetic_load([("2024-01-02 17:00Z", "2024-01-02", 72, 13.0)])
    history = make_history_index(load)
    issue = issue_time_for_target_date(date(2024, 1, 9)).tz_convert("UTC")
    assert b2_weekly(history, date(2024, 1, 9), 72, issue)[0] == 13.0
    assert np.isnan(b2_weekly(history, date(2024, 1, 9), 73, issue)[0])


def test_dst_local_target_days_have_92_96_100_unique_utc_intervals():
    expected = {"2024-02-01": 96, "2024-03-31": 92, "2024-10-27": 100}
    for day, count in expected.items():
        grid = local_day_grid(day)
        assert len(grid) == count
        assert grid.target_time_utc.is_unique


def test_prediction_splits_and_timestamp_uniqueness(predictions):
    assert set(predictions.split) == {"validation", "test"}
    for _, group in predictions.groupby(["split", "model_name"]):
        assert group.target_time_utc.is_unique
    assert predictions.loc[predictions.split.eq("validation"), "target_time_utc"].max() < \
        predictions.loc[predictions.split.eq("test"), "target_time_utc"].min()


def test_artifact_sources_never_extend_past_issue(predictions):
    sourced = predictions[predictions.latest_source_time_utc.notna()]
    assert (sourced.latest_source_time_utc + pd.Timedelta(minutes=15) <= sourced.forecast_issue_time_utc).all()


def test_metric_mask_excludes_low_quality_truth():
    frame = pd.DataFrame({"target_quality_primary": [True, False], "y_true_kw": [10.0, 1_000.0],
                          "y_pred_kw": [8.0, 0.0]})
    result = metric_values(frame, train_scale=10.0)
    assert result["n_targets_evaluated"] == 1
    assert result["MAE"] == 2.0


def test_day_block_bootstrap_is_reproducible():
    frame = pd.DataFrame({"forecast_date_local": pd.to_datetime(["2024-01-01"] * 2 + ["2024-01-02"] * 2),
        "target_quality_primary": True, "y_true_kw": [1.0, 2.0, 3.0, 4.0], "y_pred_kw": [1.5, 1.5, 2.5, 5.0]})
    assert day_block_bootstrap(frame, reps=100, seed=42) == day_block_bootstrap(frame, reps=100, seed=42)


def test_missing_history_returns_unavailable():
    history = make_history_index(synthetic_load([]))
    issue = pd.Timestamp("2024-01-01", tz="UTC")
    assert np.isnan(b0_persistence(history, issue)[0])
    assert np.isnan(b1_safe_recent_day(history, date(2024, 1, 2), 0, issue)[0])
    assert np.isnan(b2_weekly(history, date(2024, 1, 8), 0, issue)[0])


def test_frozen_configuration_uses_validation_only_and_excludes_unsafe_sources():
    config = json.loads((ROOT / "config" / "baseline_experiment.json").read_text(encoding="utf-8"))
    assert config["selection"]["selection_split"] == "validation_2023_only"
    assert set(config["excluded"]) == {"observed weather", "measured PV", "day-ahead price"}
    assert all("test" not in key.lower() for key in config["selection"])


def test_calendar_profile_is_fit_from_train_only():
    frame = pd.DataFrame({"split": ["train", "train", "validation", "test"],
        "target_quality_primary": [True] * 4, "load_mean_kw": [10.0, 14.0, 1_000.0, 2_000.0],
        "day_of_week": [1] * 4, "quarter_hour_of_day": [20] * 4, "month": [1] * 4})
    profile = calendar_profiles(frame, with_month=False)
    assert profile.loc[(1, 20)] == 12.0


def test_primary_quality_equivalence_is_preserved():
    gold = pd.read_parquet(ROOT / "data" / "gold" / "load_15min_base.parquet",
                           columns=["target_quality_primary", "load_valid_count"])
    assert gold.target_quality_primary.equals(gold.load_valid_count.ge(135))
