import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flexibility_optimization import (  # noqa: E402
    INTERVAL_HOURS,
    apply_routing,
    cost,
    evaluate_decision,
    optimize_routing,
    prepare_prices,
)
from probabilistic_lightgbm import QCOLS  # noqa: E402


def _two_interval_day(actual=(10.0, 0.0), holiday=False):
    utc = pd.date_range("2024-01-02 00:00Z", periods=2, freq="15min")
    frame = pd.DataFrame({
        "forecast_date_local": ["2024-01-02"] * 2,
        "target_time_utc": utc,
        "target_time_local": utc.tz_convert("Europe/Berlin").tz_localize(None),
        "y_true_kw": actual,
        "deterministic_kw": [10.0, 0.0],
        "baseline_B3_kw": [10.0, 0.0],
        "price_eur_per_kwh": [1.0, .1],
        "price_eur_per_mwh": [1000.0, 100.0],
        "is_public_holiday": [holiday] * 2,
        "regime_level": ["NORMAL"] * 2,
        "regime_calendar": ["public_holiday" if holiday else "normal_weekday"] * 2,
    })
    for index, column in enumerate(QCOLS):
        frame[column] = np.asarray([10.0, 0.0]) + (index - 3) * .1
    return frame


def test_price_conversion_preserves_negative_prices():
    source = pd.DataFrame({"utc_datetime": pd.to_datetime(["2024-01-01 00:00Z"]),
                           "price_eur_per_mwh": [-50.0]})
    result = prepare_prices(source)
    assert result.price_eur_per_kwh.iloc[0] == -.05
    assert result.available_at_12_d_minus_1.iloc[0] == False


def test_dst_local_day_lengths_are_physical():
    utc = pd.date_range("2024-03-30 22:00Z", "2024-04-01 22:00Z", freq="15min")
    fall = pd.date_range("2024-10-26 22:00Z", "2024-10-28 23:00Z", freq="15min")
    source = pd.DataFrame({"utc_datetime": utc.append(fall), "price_eur_per_mwh": 1.0})
    result = prepare_prices(source)
    counts = result.groupby("market_date").size()
    assert counts["2024-03-31"] == 92
    assert counts["2024-10-27"] == 100


def test_analytically_obvious_shift_moves_energy_to_cheaper_interval():
    solved = optimize_routing(np.array([[10.0, 0.0]]), np.array([1.0]), np.array([1.0, .1]),
                              flexibility_fraction=.5, shift_window_hours=1,
                              power_cap_multiplier=1.0)
    scheduled = solved["planned_schedules"][0]
    assert np.allclose(scheduled, [5.0, 5.0])
    assert cost(scheduled, np.array([1.0, .1])) < cost(np.array([10.0, 0.0]), np.array([1.0, .1]))


def test_energy_conservation_and_nonnegative_schedule():
    loads = np.array([[5.0, 10.0, 2.0], [6.0, 9.0, 3.0]])
    solved = optimize_routing(loads, np.array([.5, .5]), np.array([.2, .1, .3]), .2, 1, 1.0)
    assert solved["conservation_error_kwh"] < 1e-8
    assert solved["planned_schedules"].min() >= 0
    assert np.allclose(solved["planned_schedules"].sum(axis=1), loads.sum(axis=1))


def test_shift_window_is_enforced():
    solved = optimize_routing(np.array([[10.0, 0.0, 0.0]]), np.array([1.0]),
                              np.array([1.0, .9, 0.0]), .5, .25, 2.0)
    allocation = solved["allocation"].tocoo()
    assert np.all(np.abs(allocation.row-allocation.col) <= 1)
    assert solved["planned_schedules"][0, 2] == pytest.approx(0.0)


def test_planned_peak_cap_is_enforced():
    loads = np.array([[10.0, 2.0, 2.0]])
    solved = optimize_routing(loads, np.array([1.0]), np.array([1.0, 0.0, .1]), .5, 1, 1.0)
    assert solved["planned_schedules"].max() <= 10.0 + 1e-7
    assert solved["planned_cap_violation_kw"] == 0.0


def test_objective_and_realized_cost_calculation():
    loads = np.array([[10.0, 0.0]])
    prices = np.array([1.0, .1])
    solved = optimize_routing(loads, np.array([1.0]), prices, .5, 1, 1.0)
    assert solved["planned_cost"] == pytest.approx(cost(solved["planned_schedules"][0], prices))
    assert cost(np.array([4.0, 2.0]), prices) == pytest.approx(1.05)


def test_perfect_foresight_is_no_worse_than_forecast_routing():
    actual = np.array([10.0, 0.0, 5.0])
    prices = np.array([1.0, .1, .5])
    perfect = optimize_routing(actual[None, :], np.array([1.0]), prices, .3, 1, 1.0)
    forecast = optimize_routing(np.array([[0.0, 10.0, 5.0]]), np.array([1.0]), prices, .3, 1, 1.0)
    realized_forecast = apply_routing(actual, forecast["allocation"], .3)
    assert cost(perfect["planned_schedules"][0], prices) <= cost(realized_forecast, prices) + 1e-9


def test_forecast_driven_allocation_does_not_use_actual_target():
    scenario = np.array([[10.0, 0.0]])
    first = evaluate_decision(_two_interval_day((10.0, 0.0)), "D2", "test", scenario,
                              np.array([1.0]), .5, 1, 1.0)[2]
    second = evaluate_decision(_two_interval_day((1.0, 20.0)), "D2", "test", scenario,
                               np.array([1.0]), .5, 1, 1.0)[2]
    assert np.array_equal(first.toarray(), second.toarray())
    realized = apply_routing(np.array([1.0, 20.0]), first, .5)
    assert realized.max() <= 20.0 + 1e-9


def test_holiday_fallback_disables_flexibility():
    day = _two_interval_day((10.0, 0.0), holiday=True)
    row, schedule, _ = evaluate_decision(day, "D3_HOLIDAY_FALLBACK", "test",
        day[["q10", "q50", "q90"]].to_numpy().T, np.array([.2, .6, .2]), .2, 2, 1.0,
        holiday_fallback=True)
    assert row["solver_status"] == "holiday_fallback_no_flex"
    assert row["savings_eur"] == pytest.approx(0.0)
    assert np.allclose(schedule.original_grid_load, schedule.optimized_grid_load)


def test_probabilistic_policy_uses_frozen_r0_not_r4():
    config = json.loads((ROOT / "config" / "regime_probabilistic_frozen.json").read_text())
    simulation = json.loads((ROOT / "config" / "flexibility_simulation.json").read_text())
    assert config["selected_method"] == "R0_GLOBAL_CALIBRATION"
    assert set(simulation["probabilistic_scenarios"]["standard"]) == {"q10", "q50", "q90"}


def test_solver_is_deterministically_reproducible():
    args = (np.array([[4.0, 8.0, 1.0]]), np.array([1.0]), np.array([.4, .1, .2]), .2, 1, 1.0)
    first = optimize_routing(*args); second = optimize_routing(*args)
    assert np.array_equal(first["allocation"].toarray(), second["allocation"].toarray())
    assert first["planned_cost"] == second["planned_cost"]


def test_regret_definition_known_value():
    forecast_realized_cost = 12.5
    perfect_cost = 10.0
    assert forecast_realized_cost - perfect_cost == pytest.approx(2.5)


def test_finalized_forecasting_manifests_still_match():
    manifests = [
        ROOT / "results" / "baselines" / "baseline_run_manifest.json",
        ROOT / "results" / "lightgbm" / "lightgbm_run_manifest.json",
        ROOT / "results" / "probabilistic" / "probabilistic_run_manifest.json",
        ROOT / "results" / "regime_probabilistic" / "regime_probabilistic_run_manifest.json",
    ]
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        for relative, metadata in manifest["outputs"].items():
            output = manifest_path.parent / relative
            assert output.stat().st_size == metadata["bytes"]
            assert hashlib.sha256(output.read_bytes()).hexdigest() == metadata["sha256"]


def test_flexibility_manifest_hashes_match_outputs():
    manifest_path = ROOT / "results" / "flexibility" / "flexibility_run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, metadata in manifest["outputs"].items():
        output = manifest_path.parent / relative
        assert output.stat().st_size == metadata["bytes"]
        assert hashlib.sha256(output.read_bytes()).hexdigest() == metadata["sha256"]

