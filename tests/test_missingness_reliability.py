import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from missingness_reliability import (  # noqa: E402
    apply_quality_rule,
    create_error_rows,
    interpolate_short_internal_gaps,
    longest_true_run,
    mask_features,
    random_masks,
)


def test_random_masks_are_deterministic_and_exact():
    first = random_masks(20, 0.75, 1729)
    second = random_masks(20, 0.75, 1729)
    other = random_masks(20, 0.75, 1730)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.all(first.sum(axis=1) == 135)


def test_masking_experiment_error_uses_full_reference_mean():
    reference = np.arange(180, dtype=np.float32)[None, :]
    mask = np.zeros((1, 180), dtype=bool)
    mask[:, :90] = True
    metadata = pd.DataFrame({"year": [2021], "variability_group": ["high"]})
    result = create_error_rows(reference, mask, metadata, "test", .5, 1, "test")
    assert np.isclose(result.loc[0, "true_mean_kw"], 0.0895)
    assert np.isclose(result.loc[0, "partial_mean_kw"], 0.0445)
    assert np.isclose(result.loc[0, "error_kw"], -0.045)


def test_mask_features_capture_location_and_runs():
    mask = np.ones((1, 180), dtype=bool)
    mask[0, 0:3] = False
    mask[0, 80:85] = False
    mask[0, 179] = False
    result = mask_features(mask).iloc[0]
    assert result.valid_count == 171
    assert result.longest_missing_run_samples == 5
    assert result.number_missing_runs == 3
    assert result.first_valid_position == 3
    assert result.last_valid_position == 178
    assert result.fraction_first_half < result.fraction_second_half


def test_quality_rule_combines_coverage_gap_and_halves():
    features = pd.DataFrame({
        "coverage": [.75, .75, .95],
        "longest_missing_run_samples": [10, 20, 3],
        "fraction_first_half": [.7, .8, 0.0],
        "fraction_second_half": [.8, .7, .95],
    })
    selected = apply_quality_rule(features, .70, longest_missing_run_max=12, require_both_halves=True)
    assert selected.tolist() == [True, False, False]


def test_short_gap_interpolation_is_internal_and_bounded():
    values = np.arange(10, dtype=float)[None, :]
    values[0, 0] = np.nan
    values[0, 3:5] = np.nan
    values[0, 7:10] = np.nan
    filled = interpolate_short_internal_gaps(values, 2)
    assert np.isnan(filled[0, 0])
    assert np.allclose(filled[0, 3:5], [3.0, 4.0])
    assert np.isnan(filled[0, 7:10]).all()


def test_longest_true_run_multiple_rows():
    rows = np.array([[False, True, True, False, True], [True, True, True, False, False]])
    assert longest_true_run(rows).tolist() == [2, 3]
