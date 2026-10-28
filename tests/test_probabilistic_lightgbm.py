import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from day_ahead_baselines import observation_available  # noqa: E402
from probabilistic_lightgbm import (  # noqa: E402
    QCOLS,
    QUANTILES,
    block_bootstrap,
    calibration_adjustments,
    crossing_diagnostics,
    interval_coverage,
    interval_score_values,
    monotonic_correct,
    pinball_values,
    validate_quantiles,
    wis_values,
)


@pytest.fixture(scope="module")
def predictions():
    return pd.read_parquet(ROOT / "results" / "probabilistic" / "probabilistic_predictions.parquet")


@pytest.fixture(scope="module")
def frozen():
    return json.loads((ROOT / "config" / "probabilistic_calibration_frozen.json").read_text())


def test_allowed_quantile_levels_are_exact():
    validate_quantiles()
    with pytest.raises(ValueError):
        validate_quantiles(np.array([.1, .5, .9]))


def test_crossing_detection_and_monotonic_correction():
    raw = np.array([[1.0, 3.0, 2.0], [1.0, 2.0, 3.0]])
    diagnostics = crossing_diagnostics(raw)
    assert diagnostics["crossing_rate"] == .5
    corrected = monotonic_correct(raw)
    assert np.all(np.diff(corrected, axis=1) >= 0)
    assert np.array_equal(corrected[0], [1.0, 2.0, 3.0])


def test_pinball_loss_known_values():
    y = np.array([2.0, 0.0])
    prediction = np.array([1.0, 1.0])
    assert np.allclose(pinball_values(y, prediction, .5), [.5, .5])
    assert np.allclose(pinball_values(y, prediction, .9), [.9, .1])


def test_interval_coverage_and_score_known_values():
    y = np.array([2.0, 4.0, 0.0])
    lower = np.ones(3); upper = np.full(3, 3.0)
    assert interval_coverage(y, lower, upper).tolist() == [True, False, False]
    assert np.allclose(interval_score_values(y, lower, upper, .2), [2.0, 12.0, 12.0])


def test_wis_known_symmetric_interval_value():
    frame = pd.DataFrame({"q05": [1.0], "q10": [1.0], "q25": [1.0], "q50": [2.0],
                          "q75": [3.0], "q90": [3.0], "q95": [3.0]})
    # 0.5*0 + (0.25+0.10+0.05)*width 2, divided by 3.5.
    assert wis_values(np.array([2.0]), frame)[0] == pytest.approx(.8 / 3.5)


def test_calibration_api_accepts_validation_only_and_rejects_test():
    base = pd.DataFrame({"split": ["validation"], "target_quality_primary": [True],
                         "y_true_kw": [10.0]} | {column: [9.0] for column in QCOLS})
    adjustments = calibration_adjustments(base)
    assert set(adjustments) == set(QCOLS)
    with pytest.raises(RuntimeError, match="validation"):
        calibration_adjustments(base.assign(split="test"))


def test_predictions_are_monotonic_after_correction(predictions):
    values = predictions[QCOLS].to_numpy()
    finite = np.isfinite(values).all(axis=1)
    assert np.all(np.diff(values[finite], axis=1) >= -1e-12)


def test_raw_crossing_is_measured_not_hidden():
    crossing = json.loads((ROOT / "results" / "probabilistic" / "quantile_crossing.json").read_text())
    assert crossing["validation"]["crossing_rate"] > 0
    assert crossing["test"]["crossing_rate"] > 0
    assert crossing["test"]["max_crossing_magnitude_kw"] > 0


def test_frozen_calibration_uses_2023_only_and_precedes_test(frozen):
    config = ROOT / "config" / "probabilistic_calibration_frozen.json"
    artifact = ROOT / "results" / "probabilistic" / "probabilistic_predictions.parquet"
    assert frozen["calibration_data"] == "2023 validation only"
    assert frozen["test_labels_used"] is False
    assert frozen["written_before_test_scoring"] is True
    assert config.stat().st_mtime_ns <= artifact.stat().st_mtime_ns


def test_probabilistic_models_reuse_safe_deterministic_features():
    deterministic = json.loads((ROOT / "config" / "lightgbm_frozen.json").read_text())
    forbidden = ("load_mean_kw", "load_coverage", "target_quality", "observed_", "pv_", "price")
    assert not [name for name in deterministic["features"] if any(token in name.lower() for token in forbidden)]


def test_authoritative_issuance_engine_is_unchanged():
    issue = pd.Timestamp("2024-01-08 11:00Z")
    assert observation_available(pd.Timestamp("2024-01-08 10:45Z"), issue)
    assert not observation_available(pd.Timestamp("2024-01-08 11:00Z"), issue)


def test_day_block_bootstrap_is_reproducible():
    daily = pd.DataFrame({"mean_pinball_loss": [1.0, 2.0], "coverage_90": [.8, 1.0],
                          "width_90": [5.0, 7.0], "WIS": [2.0, 3.0]})
    assert block_bootstrap(daily, reps=100, seed=9) == block_bootstrap(daily, reps=100, seed=9)


def test_empirical_baselines_have_frozen_pretest_residual_distributions(frozen):
    assert set(frozen["PB0_B3_train_residual_quantiles"]) == set(QCOLS)
    assert set(frozen["PB1_LGBM_validation_residual_quantiles"]) == set(QCOLS)
    assert frozen["test_labels_used"] is False


def test_full_test_prediction_coverage(predictions):
    test = predictions[(predictions.model_name.eq("Q_LGBM")) &
                       predictions.calibration_state.eq("calibrated") & predictions.split.eq("test")]
    assert len(test) == 35_136
    assert test.q50.notna().all()
    assert test.loc[test.target_quality_primary, "q50"].notna().sum() == 31_339
    assert test.target_time_utc.is_unique


def test_probabilistic_manifest_hashes_match_outputs():
    path = ROOT / "results" / "probabilistic" / "probabilistic_run_manifest.json"
    manifest = json.loads(path.read_text())
    for relative, metadata in manifest["outputs"].items():
        output = path.parent / relative
        assert output.stat().st_size == metadata["bytes"]
        assert hashlib.sha256(output.read_bytes()).hexdigest() == metadata["sha256"]
