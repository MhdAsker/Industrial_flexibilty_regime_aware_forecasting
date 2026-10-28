import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from probabilistic_lightgbm import QCOLS, block_bootstrap  # noqa: E402
from regime_probabilistic import (  # noqa: E402
    apply_group_calibration,
    apply_hierarchical,
    assign_regimes,
    calendar_regime,
    fit_conditional_calibration,
    learn_regime_definitions,
)


@pytest.fixture(scope="module")
def frozen():
    return json.loads((ROOT / "config" / "regime_probabilistic_frozen.json").read_text())


@pytest.fixture(scope="module")
def predictions():
    return pd.read_parquet(
        ROOT / "results" / "regime_probabilistic" / "regime_probabilistic_predictions.parquet"
    )


def _calibration_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "split": ["validation"] * 5,
            "forecast_date_local": pd.to_datetime(["2023-01-01"] * 5),
            "target_quality_primary": [True] * 5,
            "y_true_kw": [10.0] * 5,
            "regime": ["large", "large", "large", "small", "small"],
        }
    )
    for index, column in enumerate(QCOLS):
        frame[column] = float(index)
    return frame


def test_calendar_precedence_is_deterministic():
    frame = pd.DataFrame(
        {
            "is_public_holiday": [True, False, False, False, False, False],
            "is_bridge_day": [True, True, False, False, False, False],
            "is_day_before_holiday": [True, True, True, False, False, False],
            "is_day_after_holiday": [True, True, True, True, False, False],
            "is_weekend": [True, True, True, True, True, False],
        }
    )
    assert calendar_regime(frame).tolist() == [
        "public_holiday", "bridge_day", "day_before_holiday",
        "day_after_holiday", "weekend", "normal_weekday",
    ]


def test_regime_definition_api_accepts_train_only_and_rejects_2024():
    train = pd.DataFrame(
        {
            "split": ["train"] * 3,
            "forecast_issue_time_utc": pd.date_range("2022-01-01", periods=3, tz="UTC"),
            "recent_7d_mean": [1.0, 2.0, 3.0],
            "recent_7d_median": [1.0, 2.0, 3.0],
            "recent_level_ratio_7d_28d": [.8, 1.0, 1.2],
            "recent_7d_std": [1.0, 2.0, 3.0],
        }
    )
    assert "recent_7d_mean_q33_q67" in learn_regime_definitions(train)
    with pytest.raises(RuntimeError, match="TRAIN only"):
        learn_regime_definitions(train.assign(split="test"))


def test_calibration_api_rejects_test_and_non_2023():
    frame = _calibration_frame()
    with pytest.raises(RuntimeError, match="validation"):
        fit_conditional_calibration(frame.assign(split="test"), "regime", minimum=3)
    with pytest.raises(RuntimeError, match="non-2023"):
        fit_conditional_calibration(
            frame.assign(forecast_date_local=pd.Timestamp("2024-01-01")), "regime", minimum=3
        )


def test_minimum_group_size_and_calibration_values_are_correct():
    fit = fit_conditional_calibration(_calibration_frame(), "regime", minimum=3)
    assert fit["counts"] == {"large": 3, "small": 2}
    assert set(fit["groups"]) == {"large"}
    assert fit["groups"]["large"] == {
        column: 10.0 - float(index) for index, column in enumerate(QCOLS)
    }


def test_missing_conditional_group_falls_back_to_global():
    frame = _calibration_frame()
    fit = fit_conditional_calibration(frame, "regime", minimum=3)
    result, _ = apply_group_calibration(frame.iloc[[3]], fit, "synthetic")
    assert result.fallback_level.tolist() == ["global"]


def test_hierarchy_precedence_and_fallback():
    frame = pd.DataFrame(
        {
            "regime_calendar": ["public_holiday", "normal_weekday", "normal_weekday", "bridge_day"],
            "regime_level": ["HIGH", "HIGH", "NORMAL", "NORMAL"],
            "regime_horizon": ["H1", "H1", "H1", "missing"],
        }
    )
    for index, column in enumerate(QCOLS):
        frame[column] = float(index)
    adjustment = lambda value: {column: value for column in QCOLS}
    calibration = {
        "public_holiday": adjustment(4.0),
        "level_high": adjustment(3.0),
        "horizon": {"H1": adjustment(2.0)},
        "global": adjustment(1.0),
    }
    result, _ = apply_hierarchical(frame, calibration)
    assert result.fallback_level.tolist() == [
        "public_holiday", "level_HIGH", "horizon:H1", "global"
    ]
    assert result.q50.tolist() == [7.0, 6.0, 5.0, 4.0]


def test_assignment_rejects_target_based_level_and_is_reproducible():
    frame = pd.DataFrame(
        {
            "load_mean_kw": [10.0], "lead_time_minutes": [720],
            "is_public_holiday": [False], "is_bridge_day": [False],
            "is_day_before_holiday": [False], "is_day_after_holiday": [False],
            "is_weekend": [False], "recent_7d_std": [1.0], "recent_7d_mean": [2.0],
        }
    )
    safe = {"feature": "recent_7d_mean", "lower": 1.0, "upper": 3.0}
    assert assign_regimes(frame, safe).regime_level.tolist() == assign_regimes(
        frame, safe
    ).regime_level.tolist()
    with pytest.raises(RuntimeError, match="target information"):
        assign_regimes(frame, {"feature": "load_mean_kw", "lower": 1.0, "upper": 3.0})


def test_frozen_regimes_use_safe_features_and_train_thresholds(frozen):
    selected = frozen["regime_definitions"]["selected"]
    forbidden = ("target", "weather", "pv", "price", "coverage", "residual")
    assert not any(token in selected["feature"].lower() for token in forbidden)
    features = pd.read_parquet(
        ROOT / "data" / "features" / "lightgbm_features.parquet",
        columns=["split", "forecast_issue_time_utc", selected["feature"]],
    )
    issue = features[features.split.astype(str).eq("train")].drop_duplicates(
        "forecast_issue_time_utc"
    )
    values = issue[selected["feature"]].dropna()
    probabilities = selected["probabilities"]
    assert selected["lower"] == pytest.approx(values.quantile(probabilities[0]))
    assert selected["upper"] == pytest.approx(values.quantile(probabilities[1]))


def test_frozen_configuration_precedes_test_artifact_and_excludes_test_labels(frozen):
    config = ROOT / "config" / "regime_probabilistic_frozen.json"
    artifact = ROOT / "results" / "regime_probabilistic" / "regime_probabilistic_predictions.parquet"
    assert frozen["written_before_test_scoring"] is True
    assert frozen["test_labels_used"] is False
    assert config.stat().st_mtime_ns <= artifact.stat().st_mtime_ns


def test_prediction_quantiles_are_monotonic(predictions):
    values = predictions[QCOLS].to_numpy()
    finite = np.isfinite(values).all(axis=1)
    assert np.all(np.diff(values[finite], axis=1) >= -1e-12)


def test_r0_exactly_reproduces_finalized_global_calibration(predictions):
    regime = predictions[
        predictions.model_name.eq("R0_GLOBAL_CALIBRATION") & predictions.split.eq("test")
    ].sort_values("target_time_utc")
    previous = pd.read_parquet(ROOT / "results" / "probabilistic" / "probabilistic_predictions.parquet")
    previous = previous[
        previous.model_name.eq("Q_LGBM")
        & previous.calibration_state.eq("calibrated")
        & previous.split.eq("test")
    ].sort_values("target_time_utc")
    assert regime.target_time_utc.tolist() == previous.target_time_utc.tolist()
    assert np.array_equal(regime[QCOLS].to_numpy(), previous[QCOLS].to_numpy())


def test_empirical_comparators_and_all_required_methods_are_present(predictions):
    test_models = set(predictions.loc[predictions.split.eq("test"), "model_name"])
    assert {
        "PB0_B3_empirical", "PB1_LGBM_empirical", "R0_GLOBAL_CALIBRATION",
        "R1_CALENDAR_CONDITIONAL", "R2_HORIZON_CONDITIONAL",
        "R3_LEVEL_CONDITIONAL", "R4_HIERARCHICAL",
    } <= test_models


def test_day_block_bootstrap_remains_reproducible():
    daily = pd.DataFrame(
        {"mean_pinball_loss": [1.0, 2.0], "coverage_90": [.8, 1.0],
         "width_90": [5.0, 7.0], "WIS": [2.0, 3.0]}
    )
    assert block_bootstrap(daily, reps=100, seed=31) == block_bootstrap(daily, reps=100, seed=31)


def test_regime_manifest_hashes_match_outputs():
    path = ROOT / "results" / "regime_probabilistic" / "regime_probabilistic_run_manifest.json"
    manifest = json.loads(path.read_text())
    for relative, metadata in manifest["outputs"].items():
        output = path.parent / relative
        assert output.stat().st_size == metadata["bytes"]
        assert hashlib.sha256(output.read_bytes()).hexdigest() == metadata["sha256"]

