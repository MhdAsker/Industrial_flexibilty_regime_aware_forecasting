from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timeseries_core import RawAudit, _finish_resolution, _output_valid, aggregate_frame, file_digest, localize_explicit


def five_second_frame(periods: int, start: str = "2024-01-01T00:00:00Z", value: float = 1000.0):
    return pd.DataFrame({
        "timestamp_utc": pd.date_range(start, periods=periods, freq="5s"),
        "value": np.full(periods, value, dtype="float64"),
    })


def finish(frame: pd.DataFrame, frequency: str, count: int):
    agg = aggregate_frame(frame, frequency)
    return _finish_resolution([agg], 2024, frequency, count, "Europe/Berlin", "test", set())


def first_populated(result: pd.DataFrame) -> pd.Series:
    return result.loc[result["valid_count"].gt(0)].iloc[0]


def test_five_seconds_to_one_minute():
    agg = aggregate_frame(five_second_frame(24), "1min")
    assert list(agg["valid_count"]) == [12, 12]
    assert list(agg["mean"]) == [1000.0, 1000.0]


def test_five_seconds_to_fifteen_minutes():
    agg = aggregate_frame(five_second_frame(360), "15min")
    assert list(agg["valid_count"]) == [180, 180]
    assert list(agg["median"]) == [1000.0, 1000.0]


def test_five_seconds_to_ten_minutes_is_direct():
    frame = five_second_frame(240)
    frame.loc[:119, "value"] = np.arange(120, dtype="float64")
    agg = aggregate_frame(frame, "10min")
    assert list(agg["valid_count"]) == [120, 120]
    assert agg.iloc[0]["mean"] == pytest.approx(np.arange(120).mean())
    assert agg.iloc[0]["q05"] == pytest.approx(pd.Series(np.arange(120)).quantile(.05))


def test_coverage_and_missing_samples_are_preserved():
    frame = five_second_frame(180).drop(index=range(90, 180))
    result = first_populated(finish(frame, "15min", 180))
    assert result["valid_count"] == 90
    assert result["missing_count"] == 90
    assert result["coverage"] == pytest.approx(0.5)
    assert result["p_mean_w"] == 1000.0
    assert not result["quality_80"]


def test_no_interpolation_and_all_threshold_flags_are_reversible():
    frame = five_second_frame(180).drop(index=[0, 1])
    row = first_populated(finish(frame, "15min", 180))
    assert row["valid_count"] == 178
    assert row["coverage"] == pytest.approx(178 / 180)
    assert bool(row["quality_95"])
    assert not bool(row["quality_99"])
    assert row["p_mean_w"] == 1000.0


def test_quantiles_use_original_five_second_values():
    frame = five_second_frame(180)
    frame["value"] = np.arange(180, dtype="float64")
    row = first_populated(finish(frame, "15min", 180))
    values = pd.Series(np.arange(180, dtype="float64"))
    for column, quantile in [("p_q05_w", .05), ("p_q25_w", .25), ("p_median_w", .5),
                             ("p_q75_w", .75), ("p_q95_w", .95)]:
        assert row[column] == pytest.approx(values.quantile(quantile))


def test_negative_and_zero_values_are_preserved():
    frame = five_second_frame(12)
    frame["value"] = [-5.0, 0.0, *([2.0] * 10)]
    row = first_populated(finish(frame, "1min", 12))
    assert row["p_min_w"] == -5.0
    assert row["p_max_w"] == 2.0
    assert row["valid_count"] == 12


def test_watts_to_kw_and_observed_energy():
    row = first_populated(finish(five_second_frame(180, value=2500.0), "15min", 180))
    assert row["p_mean_kw"] == pytest.approx(2.5)
    assert row["p_max_kw"] == pytest.approx(2.5)
    assert row["energy_kwh_observed"] == pytest.approx(0.625)


def test_dst_nonexistent_and_ambiguous_are_explicit():
    naive = pd.Series(pd.to_datetime(["2024-03-31 02:30:00", "2024-10-27 02:30:00"]))
    utc, ambiguous, bad = localize_explicit(naive, "Europe/Berlin", "first")
    assert pd.isna(utc.iloc[0])
    assert bad.tolist() == [True, False]
    assert ambiguous.tolist() == [False, True]
    assert utc.iloc[1] == pd.Timestamp("2024-10-27 00:30:00Z")
    second, _, _ = localize_explicit(naive, "Europe/Berlin", "second")
    assert second.iloc[1] == pd.Timestamp("2024-10-27 01:30:00Z")


def test_duplicate_timestamps_are_counted():
    local = pd.Series(pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:00"]))
    raw = pd.Series(["1", "2"], dtype="string")
    values = pd.Series([1.0, 2.0])
    audit = RawAudit(2024)
    audit.update(local, raw, values, np.array([False, False]), np.array([False, False]), 5)
    assert audit.duplicate_timestamp_count == 1


def test_missing_grid_samples_and_gap_are_counted():
    local = pd.Series(pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:15"]))
    raw = pd.Series(["1", "1"], dtype="string")
    values = pd.Series([1.0, 1.0])
    audit = RawAudit(2024)
    audit.update(local, raw, values, np.zeros(2, dtype=bool), np.zeros(2, dtype=bool), 5)
    result = audit.finish(4)
    assert result["missing_timestamp_count"] == 2
    assert result["longest_missing_gap_samples"] == 2


def test_extreme_maximum_is_not_clipped_and_is_flagged():
    frame = five_second_frame(24 * 12)
    frame.loc[12 * 20, "value"] = 1_000_000.0
    result = finish(frame, "1min", 12)
    extreme_row = result.loc[result["p_max_w"].idxmax()]
    assert extreme_row["p_max_w"] == 1_000_000.0
    assert bool(extreme_row["extreme_high"])


def test_manifest_validation_is_repeatable(tmp_path):
    output = tmp_path / "sample.parquet"
    pd.DataFrame({"x": [1.0]}).to_parquet(output, index=False)
    identity = {"source": "immutable"}
    manifest = tmp_path / "sample.manifest.json"
    manifest.write_text(__import__("json").dumps({
        "identity": identity, "output_size": output.stat().st_size,
        "output_sha256": file_digest(output),
    }), encoding="utf-8")
    assert _output_valid(output, manifest, identity)
    assert _output_valid(output, manifest, identity)
