"""Ex-post price-aware optimization of a simulated shiftable-load envelope.

This is not actual production scheduling. It applies transparent load-shifting
assumptions to finalized 2024 load forecasts and evaluates schedules on actual
eligible load and canonical delivery prices.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from day_ahead_baselines import ROOT
from probabilistic_lightgbm import QCOLS, wis_values

CONFIG_PATH = ROOT / "config" / "flexibility_simulation.json"
PRICE_PATH = ROOT / "data" / "silver" / "exogenous" / "canonical" / "electricity_prices_15min_de_2021_2024.parquet"
PRICE_SOURCE = ROOT / "data" / "silver" / "exogenous" / "source" / "electricity_prices_15min_de_2018_2024.csv"
LGBM_PATH = ROOT / "results" / "lightgbm" / "lightgbm_predictions.parquet"
PROB_PATH = ROOT / "results" / "probabilistic" / "probabilistic_predictions.parquet"
BASELINE_PATH = ROOT / "results" / "baselines" / "day_ahead_baseline_predictions.parquet"
REGIME_MANIFEST = ROOT / "results" / "regime_probabilistic" / "regime_probabilistic_run_manifest.json"
REGIME_PRED_PATH = ROOT / "results" / "regime_probabilistic" / "regime_probabilistic_predictions.parquet"
RESULTS = ROOT / "results" / "flexibility"
FIGURES = RESULTS / "figures"
REPORT = ROOT / "reports" / "flexibility_optimization.md"
INTERVAL_HOURS = 0.25


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    result = prices.rename(columns={"utc_datetime": "delivery_time_utc"}).copy()
    result["delivery_time_utc"] = pd.to_datetime(result.delivery_time_utc, utc=True)
    result["delivery_time_local"] = result.delivery_time_utc.dt.tz_convert("Europe/Berlin")
    result = result[result.delivery_time_local.dt.year.eq(2024)].sort_values("delivery_time_utc")
    result["price_eur_per_kwh"] = result.price_eur_per_mwh / 1000.0
    result["market_date"] = result.delivery_time_local.dt.date.astype(str)
    result["delivery_interval"] = result.groupby("market_date").cumcount()
    result["availability_time_local"] = pd.NaT
    result["available_at_12_d_minus_1"] = False
    result["decision_timing_class"] = "ex_post_publication_time_not_encoded"
    result["market_representation"] = "German day-ahead wholesale; exchange not identified in SPARK metadata"
    return result


def build_decision_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    deterministic = pd.read_parquet(LGBM_PATH)
    deterministic = deterministic[(deterministic.model_name.eq("LGBM_DIRECT")) &
        deterministic.model_stage.eq("final_pretest_refit") & deterministic.split.eq("test")].copy()
    columns = ["forecast_date_local", "forecast_issue_time_local", "forecast_issue_time_utc",
        "target_time_local", "target_time_utc", "y_true_kw", "y_pred_kw", "baseline_B3_kw",
        "target_quality_primary", "is_public_holiday", "recent_level_difference_7d_28d"]
    frame = deterministic[columns].rename(columns={"y_pred_kw": "deterministic_kw"})
    probabilistic = pd.read_parquet(PROB_PATH)
    probabilistic = probabilistic[(probabilistic.model_name.eq("Q_LGBM")) &
        probabilistic.calibration_state.eq("calibrated") & probabilistic.split.eq("test")]
    frame = frame.merge(probabilistic[["target_time_utc", *QCOLS]], on="target_time_utc",
                        how="left", validate="one_to_one")
    regime = pd.read_parquet(REGIME_PRED_PATH,columns=["model_name","split","target_time_utc",
                                                       "regime_level","regime_calendar"])
    regime = regime[(regime.model_name.eq("R0_GLOBAL_CALIBRATION"))&regime.split.eq("test")]
    frame = frame.merge(regime[["target_time_utc","regime_level","regime_calendar"]],
                        on="target_time_utc",how="left",validate="one_to_one")
    prices = prepare_prices(pd.read_parquet(PRICE_PATH))
    frame = frame.merge(prices, left_on="target_time_utc", right_on="delivery_time_utc",
                        how="left", validate="one_to_one")
    frame["forecast_date_local"] = frame.forecast_date_local.astype(str)
    return frame, prices


def complete_days(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = frame.groupby("forecast_date_local", sort=True).agg(
        physical_intervals=("target_time_utc", "size"),
        eligible_intervals=("target_quality_primary", "sum"),
        actual_nonmissing=("y_true_kw", "count"),
        price_nonmissing=("price_eur_per_kwh", "count"),
    ).reset_index()
    summary["is_complete"] = ((summary.physical_intervals == summary.eligible_intervals) &
        (summary.physical_intervals == summary.actual_nonmissing) &
        (summary.physical_intervals == summary.price_nonmissing))
    keep = set(summary.loc[summary.is_complete, "forecast_date_local"])
    return frame[frame.forecast_date_local.isin(keep)].copy(), summary


def allowed_edges(n_intervals: int, window_intervals: int) -> tuple[np.ndarray, np.ndarray]:
    origins, destinations = [], []
    for origin in range(n_intervals):
        destinations_for_origin = range(max(0, origin-window_intervals),
                                        min(n_intervals, origin+window_intervals+1))
        for destination in destinations_for_origin:
            origins.append(origin); destinations.append(destination)
    return np.asarray(origins, dtype=np.int32), np.asarray(destinations, dtype=np.int32)


def optimize_routing(load_scenarios: np.ndarray, probabilities: np.ndarray, prices: np.ndarray,
                     flexibility_fraction: float, shift_window_hours: int,
                     power_cap_multiplier: float) -> dict:
    scenarios = np.maximum(np.asarray(load_scenarios, dtype=float), 0.0)
    if scenarios.ndim == 1:
        scenarios = scenarios[None, :]
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    scenario_count, n = scenarios.shape
    window_intervals = int(round(shift_window_hours / INTERVAL_HOURS))
    origin, destination = allowed_edges(n, window_intervals)
    edges = len(origin)

    expected_origin_load = probabilities @ scenarios
    objective = flexibility_fraction * expected_origin_load[origin] * prices[destination] * INTERVAL_HOURS
    equality = coo_matrix((np.ones(edges), (origin, np.arange(edges))), shape=(n, edges)).tocsr()
    equality_rhs = np.ones(n)

    inequality_rows, inequality_cols, inequality_data = [], [], []
    inequality_rhs = np.empty(scenario_count*n+n)
    for scenario_index, load in enumerate(scenarios):
        offset = scenario_index*n
        inequality_rows.extend((offset+destination).tolist())
        inequality_cols.extend(np.arange(edges).tolist())
        inequality_data.extend((flexibility_fraction*load[origin]).tolist())
        cap = power_cap_multiplier * float(load.max())
        inequality_rhs[offset:offset+n] = cap - (1-flexibility_fraction)*load
    # A load-independent column-sum bound guarantees the same cap after the
    # frozen routing policy is applied to any nonnegative realized profile.
    # For cap=1 this is a doubly-stochastic routing matrix.
    robust_column_bound = (power_cap_multiplier-(1-flexibility_fraction))/flexibility_fraction
    robust_offset = scenario_count*n
    inequality_rows.extend((robust_offset+destination).tolist())
    inequality_cols.extend(np.arange(edges).tolist())
    inequality_data.extend(np.ones(edges).tolist())
    inequality_rhs[robust_offset:robust_offset+n] = robust_column_bound
    inequality = coo_matrix((inequality_data, (inequality_rows, inequality_cols)),
                            shape=(scenario_count*n+n, edges)).tocsr()
    solution = linprog(objective, A_ub=inequality, b_ub=inequality_rhs,
                       A_eq=equality, b_eq=equality_rhs, bounds=(0.0, 1.0), method="highs")
    if not solution.success:
        return {"success": False, "status": solution.message, "solver_status_code": solution.status}
    allocation = coo_matrix((solution.x, (origin, destination)), shape=(n, n)).tocsr()
    scheduled = np.column_stack([(1-flexibility_fraction)*load +
        flexibility_fraction*(allocation.T @ load) for load in scenarios])
    scheduled = scheduled.T
    conservation_error = np.max(np.abs(scheduled.sum(axis=1)-scenarios.sum(axis=1))) * INTERVAL_HOURS
    cap_violation = max(float(np.max(scheduled[index] - power_cap_multiplier*scenarios[index].max()))
                        for index in range(scenario_count))
    if conservation_error > 1e-7 or cap_violation > 1e-6 or solution.x.min() < -1e-9:
        raise AssertionError("Solver result violates planned feasibility")
    return {"success": True, "status": "optimal", "solver_status_code": solution.status,
            "allocation": allocation, "planned_schedules": scheduled,
            "planned_cost": float(np.sum(probabilities[:,None]*scheduled*prices[None,:])*INTERVAL_HOURS),
            "conservation_error_kwh": float(conservation_error),
            "planned_cap_violation_kw": max(0.0, cap_violation)}


def identity_routing(n: int, scenarios: np.ndarray, probabilities: np.ndarray,
                     prices: np.ndarray) -> dict:
    scenarios = np.atleast_2d(np.maximum(scenarios, 0.0))
    probabilities = np.asarray(probabilities, float); probabilities /= probabilities.sum()
    return {"success": True, "status": "holiday_fallback_no_flex", "solver_status_code": 0,
            "allocation": scipy.sparse.eye(n, format="csr"), "planned_schedules": scenarios,
            "planned_cost": float(np.sum(probabilities[:,None]*scenarios*prices[None,:])*INTERVAL_HOURS),
            "conservation_error_kwh": 0.0, "planned_cap_violation_kw": 0.0}


def apply_routing(actual_load: np.ndarray, allocation, flexibility_fraction: float) -> np.ndarray:
    actual = np.maximum(np.asarray(actual_load, float), 0.0)
    return (1-flexibility_fraction)*actual + flexibility_fraction*(allocation.T @ actual)


def cost(load: np.ndarray, prices_eur_per_kwh: np.ndarray) -> float:
    return float(np.sum(np.asarray(load)*np.asarray(prices_eur_per_kwh))*INTERVAL_HOURS)


def routing_operations(actual: np.ndarray, scheduled: np.ndarray, allocation,
                       flexibility_fraction: float) -> dict:
    original_flexible = flexibility_fraction*actual
    scheduled_flexible = flexibility_fraction*(allocation.T @ actual)
    energy_shifted = .5*float(np.abs(scheduled_flexible-original_flexible).sum())*INTERVAL_HOURS
    coo = allocation.tocoo(); moved = coo.row != coo.col
    moved_energy = flexibility_fraction*actual[coo.row[moved]]*coo.data[moved]*INTERVAL_HOURS
    shifts = np.abs(coo.col[moved]-coo.row[moved])*INTERVAL_HOURS
    mean_shift = float(np.average(shifts, weights=moved_energy)) if moved_energy.sum() > 1e-12 else 0.0
    max_shift = float(shifts[moved_energy > 1e-10].max()) if np.any(moved_energy > 1e-10) else 0.0
    return {"energy_shifted_kwh": energy_shifted, "mean_temporal_shift_hours": mean_shift,
            "maximum_temporal_shift_hours": max_shift,
            "intervals_changed": int(np.sum(np.abs(scheduled-actual) > 1e-7)),
            "fraction_total_energy_shifted": energy_shifted/(actual.sum()*INTERVAL_HOURS),
            "original_flexible": original_flexible, "scheduled_flexible": scheduled_flexible}


def day_diagnostics(day: pd.DataFrame) -> dict:
    actual = day.y_true_kw.to_numpy(float); deterministic = day.deterministic_kw.to_numpy(float)
    b3 = day.baseline_B3_kw.to_numpy(float); q50 = day.q50.to_numpy(float)
    price = day.price_eur_per_kwh.to_numpy(float); price_mwh = day.price_eur_per_mwh.to_numpy(float)
    quantiles = day[QCOLS]
    return {"forecast_mae_deterministic_kw": float(np.mean(np.abs(deterministic-actual))),
        "forecast_bias_deterministic_kw": float(np.mean(deterministic-actual)),
        "forecast_mae_b3_kw": float(np.mean(np.abs(b3-actual))),
        "forecast_mae_q50_kw": float(np.mean(np.abs(q50-actual))),
        "probabilistic_wis": float(wis_values(actual,quantiles).mean()),
        "cost_weighted_abs_error_deterministic_eur": float(np.sum(np.abs(deterministic-actual)*np.abs(price))*INTERVAL_HOURS),
        "daily_price_range_eur_per_mwh": float(price_mwh.max()-price_mwh.min()),
        "daily_price_p90_p10_eur_per_mwh": float(np.quantile(price_mwh,.9)-np.quantile(price_mwh,.1)),
        "daily_price_std_eur_per_mwh": float(price_mwh.std()),
        "negative_price_intervals": int((price_mwh<0).sum()),
        "has_negative_price": bool((price_mwh<0).any())}


def evaluate_decision(day: pd.DataFrame, decision_method: str, scenario_id: str,
                      load_scenarios: np.ndarray, probabilities: np.ndarray,
                      flexibility_fraction: float, shift_window_hours: int,
                      power_cap_multiplier: float, holiday_fallback: bool=False,
                      force_no_flex: bool=False) -> tuple[dict,pd.DataFrame,object]:
    actual = day.y_true_kw.to_numpy(float); prices = day.price_eur_per_kwh.to_numpy(float)
    if force_no_flex:
        solved = identity_routing(len(day), load_scenarios, probabilities, prices)
        solved["status"] = "no_flex_benchmark"
        effective_fraction = 0.0
    elif holiday_fallback and bool(day.is_public_holiday.iloc[0]):
        solved = identity_routing(len(day), load_scenarios, probabilities, prices)
        effective_fraction = 0.0
    else:
        solved = optimize_routing(load_scenarios, probabilities, prices, flexibility_fraction,
                                  shift_window_hours, power_cap_multiplier)
        effective_fraction = flexibility_fraction
    if not solved["success"]:
        row = {"date":day.forecast_date_local.iloc[0],"scenario":scenario_id,
               "forecast_method":decision_method,"solver_status":solved["status"]}
        return row,pd.DataFrame(),None
    realized = apply_routing(actual, solved["allocation"], effective_fraction)
    operations = routing_operations(actual,realized,solved["allocation"],effective_fraction)
    no_flex_cost = cost(actual,prices); realized_cost = cost(realized,prices)
    actual_cap = power_cap_multiplier*actual.max()
    negative = prices < 0
    energy_into_negative = float(np.maximum(operations["scheduled_flexible"][negative]-
                                             operations["original_flexible"][negative],0).sum()*INTERVAL_HOURS)
    negative_price_contribution = -float(np.sum((realized[negative]-actual[negative])*
                                                  prices[negative])*INTERVAL_HOURS)
    row = {"date":day.forecast_date_local.iloc[0],"scenario":scenario_id,
        "forecast_method":decision_method,"flexibility_fraction":flexibility_fraction,
        "effective_flexibility_fraction":effective_fraction,"shift_window_hours":shift_window_hours,
        "power_cap_multiplier":power_cap_multiplier,"physical_intervals":len(day),
        "no_flex_cost":no_flex_cost,"optimized_planned_cost":solved["planned_cost"],
        "optimized_realized_cost":realized_cost,"savings_eur":no_flex_cost-realized_cost,
        "savings_pct":100*(no_flex_cost-realized_cost)/no_flex_cost if abs(no_flex_cost)>1e-12 else np.nan,
        "energy_shifted_kwh":operations["energy_shifted_kwh"],
        "mean_temporal_shift_hours":operations["mean_temporal_shift_hours"],
        "maximum_temporal_shift_hours":operations["maximum_temporal_shift_hours"],
        "intervals_changed":operations["intervals_changed"],
        "fraction_total_energy_shifted":operations["fraction_total_energy_shifted"],
        "peak_before_kw":float(actual.max()),"peak_after_kw":float(realized.max()),
        "peak_reduction_kw":float(actual.max()-realized.max()),
        "realized_cap_violation_kw":max(0.0,float(realized.max()-actual_cap)),
        "planned_cap_violation_kw":solved["planned_cap_violation_kw"],
        "energy_shifted_into_negative_price_kwh":energy_into_negative,
        "negative_price_savings_contribution_eur":negative_price_contribution,
        "solver_status":solved["status"],"solver_status_code":solved["solver_status_code"],
        "is_public_holiday":bool(day.is_public_holiday.iloc[0]),
        "regime_level":str(day.regime_level.iloc[0]),
        "regime_calendar":str(day.regime_calendar.iloc[0])} | day_diagnostics(day)
    schedule = pd.DataFrame({"date":row["date"],"timestamp":day.target_time_utc.to_numpy(),
        "target_time_local":day.target_time_local.to_numpy(),"price_eur_per_mwh":day.price_eur_per_mwh.to_numpy(),
        "price_eur_per_kwh":prices,"actual_load":actual,
        "forecast_load":np.maximum(np.average(np.atleast_2d(load_scenarios),axis=0,weights=probabilities),0),
        "q10":day.q10.to_numpy(),"q25":day.q25.to_numpy(),"q50":day.q50.to_numpy(),
        "q75":day.q75.to_numpy(),"q90":day.q90.to_numpy(),"q95":day.q95.to_numpy(),
        "original_flexible_load":operations["original_flexible"],
        "optimized_flexible_load":operations["scheduled_flexible"],"original_grid_load":actual,
        "optimized_grid_load":realized,"decision_method":decision_method,"scenario":scenario_id})
    return row,schedule,solved["allocation"]


def bootstrap_summary(primary: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed); rows=[]
    for method,group in primary.groupby("forecast_method"):
        values=group[["savings_eur","savings_pct","regret_eur"]].to_numpy(float)
        draws=rng.integers(0,len(values),size=(repetitions,len(values)))
        samples=values[draws].mean(axis=1)
        rows.append({"comparison":method,"metric":"mean_daily_savings_eur","estimate":values[:,0].mean(),
                     "ci_low":np.quantile(samples[:,0],.025),"ci_high":np.quantile(samples[:,0],.975)})
        rows.append({"comparison":method,"metric":"mean_daily_savings_pct","estimate":np.nanmean(values[:,1]),
                     "ci_low":np.nanquantile(samples[:,1],.025),"ci_high":np.nanquantile(samples[:,1],.975)})
        rows.append({"comparison":method,"metric":"mean_daily_regret_eur","estimate":values[:,2].mean(),
                     "ci_low":np.quantile(samples[:,2],.025),"ci_high":np.quantile(samples[:,2],.975)})
    pivot=primary.pivot(index="date",columns="forecast_method",values="savings_eur")
    if {"D2_DETERMINISTIC","D3_PROBABILISTIC"} <= set(pivot):
        difference=(pivot.D3_PROBABILISTIC-pivot.D2_DETERMINISTIC).dropna().to_numpy()
        draws=rng.integers(0,len(difference),size=(repetitions,len(difference)))
        samples=difference[draws].mean(axis=1)
        rows.append({"comparison":"D3_MINUS_D2","metric":"paired_mean_savings_difference_eur",
                     "estimate":difference.mean(),"ci_low":np.quantile(samples,.025),
                     "ci_high":np.quantile(samples,.975)})
    return pd.DataFrame(rows)


def aggregate_results(daily: pd.DataFrame) -> pd.DataFrame:
    columns=["scenario","forecast_method","flexibility_fraction","shift_window_hours","power_cap_multiplier"]
    return daily.groupby(columns,dropna=False).agg(days=("date","nunique"),
        infeasible_days=("solver_status",lambda x:int((~x.isin(
            ["optimal","holiday_fallback_no_flex","no_flex_benchmark"])).sum())),
        mean_no_flex_cost_eur=("no_flex_cost","mean"),mean_realized_cost_eur=("optimized_realized_cost","mean"),
        mean_savings_eur=("savings_eur","mean"),median_savings_eur=("savings_eur","median"),
        mean_savings_pct=("savings_pct","mean"),mean_regret_eur=("regret_eur","mean"),
        p90_regret_eur=("regret_eur",lambda x:np.quantile(x,.9)),max_regret_eur=("regret_eur","max"),
        mean_forecast_efficiency=("forecast_efficiency","mean"),mean_peak_reduction_kw=("peak_reduction_kw","mean"),
        mean_energy_shifted_kwh=("energy_shifted_kwh","mean"),
        realized_cap_violation_days=("realized_cap_violation_kw",lambda x:int((x>1e-6).sum())),
        negative_price_energy_shifted_kwh=("energy_shifted_into_negative_price_kwh","sum"),
        negative_price_savings_contribution_eur=("negative_price_savings_contribution_eur","sum")).reset_index()


def correlations(primary: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for method,group in primary.groupby("forecast_method"):
        for predictor in ["forecast_mae_deterministic_kw","probabilistic_wis",
                          "cost_weighted_abs_error_deterministic_eur",
                          "daily_price_p90_p10_eur_per_mwh","daily_price_std_eur_per_mwh"]:
            for outcome in ["regret_eur","savings_eur"]:
                if group[predictor].nunique(dropna=True)<2 or group[outcome].nunique(dropna=True)<2:
                    pearson=spearman=np.nan
                else:
                    pearson=group[predictor].corr(group[outcome],method="pearson")
                    spearman=group[predictor].corr(group[outcome],method="spearman")
                rows.append({"forecast_method":method,"predictor":predictor,"outcome":outcome,
                    "pearson":pearson,"spearman":spearman})
    return pd.DataFrame(rows)


def pairwise_summary(primary: pd.DataFrame) -> pd.DataFrame:
    d2=primary[primary.forecast_method.eq("D2_DETERMINISTIC")][["date","savings_eur","regret_eur"]]
    d3=primary[primary.forecast_method.eq("D3_PROBABILISTIC")][["date","savings_eur","regret_eur"]]
    paired=d2.merge(d3,on="date",suffixes=("_d2","_d3"),validate="one_to_one")
    difference=paired.savings_eur_d3-paired.savings_eur_d2
    return pd.DataFrame([{"comparison":"D3_PROBABILISTIC minus D2_DETERMINISTIC",
        "days":len(paired),"d3_win_fraction":float((difference>1e-12).mean()),
        "tie_fraction":float((difference.abs()<=1e-12).mean()),
        "mean_savings_difference_eur":float(difference.mean()),
        "median_savings_difference_eur":float(difference.median()),
        "d2_p90_regret_eur":float(paired.regret_eur_d2.quantile(.9)),
        "d3_p90_regret_eur":float(paired.regret_eur_d3.quantile(.9)),
        "d2_worst_regret_eur":float(paired.regret_eur_d2.max()),
        "d3_worst_regret_eur":float(paired.regret_eur_d3.max())}])


def conditional_summary(daily: pd.DataFrame) -> pd.DataFrame:
    selected=daily[daily.scenario.isin(["PRIMARY","PRIMARY_HOLIDAY_FALLBACK","PRIMARY_Q95_SENSITIVITY"])]
    rows=[]
    for dimension in ["is_public_holiday","regime_level"]:
        for (scenario,method,value),group in selected.groupby(["scenario","forecast_method",dimension]):
            rows.append({"scenario":scenario,"forecast_method":method,"dimension":dimension,"group":str(value),
                "days":group.date.nunique(),"mean_savings_eur":group.savings_eur.mean(),
                "mean_savings_pct":group.savings_pct.mean(),"mean_regret_eur":group.regret_eur.mean(),
                "p90_regret_eur":group.regret_eur.quantile(.9),
                "realized_cap_violation_days":int((group.realized_cap_violation_kw>1e-6).sum())})
    return pd.DataFrame(rows)


def representative_days(primary: pd.DataFrame) -> pd.DataFrame:
    d3=primary[primary.forecast_method.eq("D3_PROBABILISTIC")].sort_values("date").copy()
    median=d3.iloc[(d3.savings_eur-d3.savings_eur.median()).abs().argmin()]
    rows=[("median_savings",median),("high_savings",d3.loc[d3.savings_eur.idxmax()]),
          ("low_savings",d3.loc[d3.savings_eur.idxmin()]),
          ("high_regret_probabilistic",d3.loc[d3.regret_eur.idxmax()])]
    d2=primary[primary.forecast_method.eq("D2_DETERMINISTIC")]
    rows.append(("high_regret_deterministic",d2.loc[d2.regret_eur.idxmax()]))
    negative=d3[d3.has_negative_price]
    if len(negative):rows.append(("negative_price_day",negative.loc[negative.savings_eur.idxmax()]))
    return pd.DataFrame([{"selection_rule":name}|row.to_dict() for name,row in rows])


def make_figures(daily,aggregate,schedules,primary,representatives):
    FIGURES.mkdir(parents=True,exist_ok=True)
    example=representatives.loc[representatives.selection_rule.eq("median_savings"),"date"].iloc[0]
    selected=schedules[(schedules.date.eq(example))&schedules.decision_method.isin(
        ["D1_PERFECT_FORESIGHT","D2_DETERMINISTIC","D3_PROBABILISTIC"])]
    base=selected[selected.decision_method.eq("D3_PROBABILISTIC")]
    fig,ax=plt.subplots(figsize=(11,5)); ax2=ax.twinx()
    ax.plot(pd.to_datetime(base.timestamp),base.actual_load,label="actual original",color="black")
    for method,g in selected.groupby("decision_method"):
        ax.plot(pd.to_datetime(g.timestamp),g.optimized_grid_load,label=method,alpha=.8)
    ax2.step(pd.to_datetime(base.timestamp),base.price_eur_per_mwh,label="price",color="tab:red",alpha=.35)
    ax.set_ylabel("kW");ax2.set_ylabel("EUR/MWh");ax.set_title(f"Median-savings day: {example}")
    ax.legend(fontsize=7,loc="upper left");fig.tight_layout();fig.savefig(FIGURES/"example_daily_schedule.png",dpi=180);plt.close(fig)

    a=aggregate[aggregate.scenario.eq("PRIMARY")]
    fig,ax=plt.subplots(figsize=(9,5));ax.bar(a.forecast_method,a.mean_realized_cost_eur)
    ax.set_ylabel("Mean EUR/day");ax.tick_params(axis="x",rotation=25);ax.set_title("Realized daily cost")
    fig.tight_layout();fig.savefig(FIGURES/"cost_comparison.png",dpi=180);plt.close(fig)
    sens=aggregate[aggregate.scenario.str.startswith("SENS_F")]
    fig,ax=plt.subplots(figsize=(8,5))
    for window,g in sens.groupby("shift_window_hours"):
        ax.plot(g.flexibility_fraction*100,g.mean_savings_eur,marker="o",label=f"±{window:g} h")
    ax.set(xlabel="Flexible fraction (%)",ylabel="Mean savings EUR/day",title="Flexibility sensitivity")
    ax.legend();fig.tight_layout();fig.savefig(FIGURES/"savings_flexibility_fraction.png",dpi=180);plt.close(fig)
    chosen=sens[sens.flexibility_fraction.eq(.2)]
    fig,ax=plt.subplots(figsize=(7,4.5));ax.plot(chosen.shift_window_hours,chosen.mean_savings_eur,marker="o")
    ax.set(xlabel="Shift window (hours)",ylabel="Mean savings EUR/day",title="Shift-window sensitivity")
    fig.tight_layout();fig.savefig(FIGURES/"savings_shift_window.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for method,g in primary[primary.forecast_method.isin(["D2_DETERMINISTIC","D3_PROBABILISTIC"])].groupby("forecast_method"):
        ax.hist(g.regret_eur,bins=30,alpha=.5,label=method)
    ax.set(xlabel="Regret EUR/day",title="Economic regret");ax.legend();fig.tight_layout();fig.savefig(FIGURES/"regret_distribution.png",dpi=180);plt.close(fig)
    d2=primary[primary.forecast_method.eq("D2_DETERMINISTIC")]
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(d2.forecast_mae_deterministic_kw,d2.regret_eur,s=12,alpha=.6)
    ax.set(xlabel="Deterministic MAE (kW)",ylabel="Regret EUR/day",title="Forecast error and regret")
    fig.tight_layout();fig.savefig(FIGURES/"forecast_mae_regret.png",dpi=180);plt.close(fig)
    d3=primary[primary.forecast_method.eq("D3_PROBABILISTIC")]
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(d3.daily_price_p90_p10_eur_per_mwh,d3.savings_eur,s=12,alpha=.6)
    ax.set(xlabel="Daily P90-P10 price spread (EUR/MWh)",ylabel="Savings EUR/day",title="Price spread and savings")
    fig.tight_layout();fig.savefig(FIGURES/"price_spread_savings.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(d3.peak_reduction_kw,d3.savings_eur,s=12,alpha=.6)
    ax.set(xlabel="Peak reduction (kW)",ylabel="Savings EUR/day",title="Peak effect and savings")
    fig.tight_layout();fig.savefig(FIGURES/"peak_reduction_savings.png",dpi=180);plt.close(fig)


def write_report(config,price_audit,day_summary,aggregate,bootstrap,correlation,pairwise,
                 conditional,representatives,category):
    primary=aggregate[aggregate.scenario.eq("PRIMARY")]
    sensitivity=aggregate[aggregate.scenario.str.startswith("SENS_F")]
    text=f"""# SIMULATED INDUSTRIAL FLEXIBILITY

## Motivation

This study tests whether finalized load forecasts improve price-aware decisions under transparent simulated flexibility assumptions.

## Decision problem

We optimize a **simulated shiftable-load envelope derived from measured industrial demand**. We do not optimize the actual SPARK production schedule.

## Why simulated flexibility is required

SPARK does not provide a complete production schedule, job precedence graph, deadlines, setup constraints, production quantities, or verified flexibility contracts. Flexible fractions are scenarios, not observed plant capabilities.

## Price availability and operational timing

{price_audit}

The study is therefore an **ex-post decision simulation**. The 12:00 D-1 load-forecast protocol is unchanged, but realized delivery prices are oracle information for the decision layer.

## Forecast inputs

F0 uses eligible actual load only for perfect foresight. F1 uses the finalized deterministic LightGBM. F2 uses frozen R0 q10/q25/q50/q75/q90/q95. B3 is retained as a simple comparator. R4 is not used.

## Flexibility assumptions

The primary scenario uses 20% flexible load, a ±2 h physical-time window, and a cap equal to each planning scenario's original daily peak. Sensitivities use 5/10/20/30%, ±1/±2/±4 h, and cap multipliers 1.0/1.1/1.25. Energy is conserved; no load is deleted.

## Optimization formulation

Indices are origin interval $i$, destination interval $t$, and probabilistic scenario $s$. $L_{{t,s}}$ is load (kW), $B_{{t,s}}=(1-f)L_{{t,s}}$ is inflexible load, $F_{{i,s}}=fL_{{i,s}}$ is original flexible load, $a_{{i,t}}$ is the routing fraction, $P_t$ is EUR/kWh, and $C_s$ is daily EUR cost. The linear objective minimizes probability-weighted daily cost. Constraints require each origin's routing fractions to sum to one, prohibit routing outside the shift window, enforce non-negativity, and cap scenario grid load. The load-independent column bound `sum_i(a_i,t) <= (c - 1 + f) / f` guarantees cap multiplier `c` for any nonnegative realized profile; at `c = 1` routing is doubly stochastic and cannot increase the realized daily peak.

## Perfect foresight benchmark

{primary[primary.forecast_method.eq('D1_PERFECT_FORESIGHT')].to_markdown(index=False,floatfmt='.5f')}

## Deterministic forecast optimization

{primary[primary.forecast_method.isin(['B3_SCHEDULING','D2_DETERMINISTIC'])].to_markdown(index=False,floatfmt='.5f')}

## Probabilistic forecast optimization

R0 scenarios are q10/q50/q90 with weights 0.2/0.6/0.2. These are transparent decision-analysis support points, not claimed point masses of an exact distribution.

{primary[primary.forecast_method.eq('D3_PROBABILISTIC')].to_markdown(index=False,floatfmt='.5f')}

## Risk-aware formulation

A predefined q95 upper-tail sensitivity replaces q90 with q95. CVaR is not used because three quantile support points are not a sufficiently resolved probability distribution for a strong CVaR interpretation.

## Primary scenario

20% flexible, ±2 h, 1.0× peak cap. Only complete physical local days with 92/96/100 eligible intervals and complete prices are evaluated.

## Sensitivity scenarios

{sensitivity.to_markdown(index=False,floatfmt='.5f')}

## Economic results

{primary.to_markdown(index=False,floatfmt='.5f')}

## Peak-demand results

Peak change is an outcome, not the objective. The robust routing-column constraint guarantees the configured realized peak cap for every nonnegative load profile without using future labels.

## Forecast-value analysis

Forecast efficiency is realized forecast-based savings divided by same-day perfect-foresight savings when positive.

## Regret analysis

{bootstrap.to_markdown(index=False,floatfmt='.5f')}

{pairwise.to_markdown(index=False,floatfmt='.5f')}

## Holiday safeguards

The policy was frozen as no flexibility on public holidays because R0 holiday calibration is unreliable. Normal-policy holiday results remain available for comparison.

{conditional[conditional.dimension.eq('is_public_holiday')].to_markdown(index=False,floatfmt='.5f')}

## High-load safeguards

The q95 sensitivity is reported separately and was not selected from TEST performance.

{conditional[conditional.dimension.eq('regime_level')].to_markdown(index=False,floatfmt='.5f')}

## Price-volatility dependence

{correlation.to_markdown(index=False,floatfmt='.5f')}

## Negative-price behavior

Negative prices are retained. Energy moved into negative-price intervals is reported rather than treated as an error.

## Optional PV case study

Not implemented. Limited PV history and the lack of a frozen operational PV forecast would make it an oracle extension and would delay the primary load-flexibility study.

## Representative days

{representatives.to_markdown(index=False,floatfmt='.5f')}

## Limitations

Prices are ex-post oracle information at the decision layer; flexible load is simulated; routing is divisible and omits jobs, minimum run times, setups, rebound, production deadlines, network tariffs, and contractual constraints. Results are conditional on one facility and the chosen sensitivity grid.

## Implications for industrial energy management

Decision category: **{category}**. Results quantify the value of a simulated load-shifting envelope, not actual factory-production control.
"""
    REPORT.write_text(text,encoding="utf-8")


def main():
    config=json.loads(CONFIG_PATH.read_text());RESULTS.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    frame,prices=build_decision_frame();complete,day_summary=complete_days(frame)
    prices.to_parquet(RESULTS/"decision_ready_prices_2024.parquet",index=False,compression="zstd")
    day_summary.to_csv(RESULTS/"flexibility_day_eligibility.csv",index=False)
    rows=[];schedules=[]
    primary=config["primary_scenario"]; f=primary["flexibility_fraction"];w=primary["shift_window_hours"];cap=primary["power_cap_multiplier"]
    for date,day in complete.groupby("forecast_date_local",sort=True):
        day=day.sort_values("target_time_utc").reset_index(drop=True)
        cases=[
            ("D0_NO_FLEX","PRIMARY",day.y_true_kw.to_numpy()[None,:],np.array([1.0]),False,True),
            ("D1_PERFECT_FORESIGHT","PRIMARY",day.y_true_kw.to_numpy()[None,:],np.array([1.0]),False,False),
            ("B3_SCHEDULING","PRIMARY",day.baseline_B3_kw.to_numpy()[None,:],np.array([1.0]),False,False),
            ("D2_DETERMINISTIC","PRIMARY",day.deterministic_kw.to_numpy()[None,:],np.array([1.0]),False,False),
            ("D3_PROBABILISTIC","PRIMARY",day[["q10","q50","q90"]].to_numpy().T,np.array([.2,.6,.2]),False,False),
            ("D3_HOLIDAY_FALLBACK","PRIMARY_HOLIDAY_FALLBACK",day[["q10","q50","q90"]].to_numpy().T,np.array([.2,.6,.2]),True,False),
            ("D3_Q95_SENSITIVITY","PRIMARY_Q95_SENSITIVITY",day[["q10","q50","q95"]].to_numpy().T,np.array([.2,.6,.2]),False,False),
        ]
        for method,scenario,loads,probabilities,fallback,no_flex in cases:
            row,schedule,_=evaluate_decision(day,method,scenario,loads,probabilities,f,w,cap,fallback,no_flex)
            rows.append(row)
            if len(schedule):schedules.append(schedule)
        for sf in config["sensitivity"]["flexibility_fractions"]:
            for sw in config["sensitivity"]["shift_window_hours"]:
                if sf==f and sw==w:continue
                scenario=f"SENS_F{int(sf*100):02d}_W{sw}_CAP100"
                row,_,_=evaluate_decision(day,"D3_PROBABILISTIC",scenario,
                    day[["q10","q50","q90"]].to_numpy().T,np.array([.2,.6,.2]),sf,sw,1.0)
                rows.append(row)
        for sc in config["sensitivity"]["power_cap_multipliers"]:
            if sc==cap:continue
            scenario=f"SENS_F20_W2_CAP{int(sc*100):03d}"
            row,_,_=evaluate_decision(day,"D3_PROBABILISTIC",scenario,
                day[["q10","q50","q90"]].to_numpy().T,np.array([.2,.6,.2]),f,w,sc)
            rows.append(row)
    daily=pd.DataFrame(rows);schedule=pd.concat(schedules,ignore_index=True)
    infeasible=daily[daily.solver_status.isin(["optimal","holiday_fallback_no_flex","no_flex_benchmark"]).eq(False)]
    if len(infeasible):raise RuntimeError(f"Infeasible days encountered: {len(infeasible)}")
    perfect=daily[(daily.forecast_method.eq("D1_PERFECT_FORESIGHT"))&daily.scenario.eq("PRIMARY")][
        ["date","optimized_realized_cost","savings_eur"]].rename(columns={"optimized_realized_cost":"perfect_foresight_cost",
        "savings_eur":"perfect_foresight_savings_eur"})
    daily=daily.merge(perfect,on="date",how="left",validate="many_to_one")
    daily["regret_eur"]=daily.optimized_realized_cost-daily.perfect_foresight_cost
    daily["forecast_efficiency"]=np.where(daily.perfect_foresight_savings_eur>1e-12,
        daily.savings_eur/daily.perfect_foresight_savings_eur,np.nan)
    # The sensitivity grid changes the feasible set; its primary-scenario
    # perfect benchmark is not comparable, so do not publish false regret.
    sensitivity_mask=daily.scenario.str.startswith("SENS_F")
    daily.loc[sensitivity_mask,["perfect_foresight_cost","perfect_foresight_savings_eur",
                                "regret_eur","forecast_efficiency"]]=np.nan
    daily.to_csv(RESULTS/"flexibility_daily_results.csv",index=False)
    schedule.to_parquet(RESULTS/"flexibility_schedules.parquet",index=False,compression="zstd")
    aggregate=aggregate_results(daily);aggregate.to_csv(RESULTS/"flexibility_summary.csv",index=False)
    primary_daily=daily[daily.scenario.eq("PRIMARY")]
    bootstrap=bootstrap_summary(primary_daily,config["bootstrap_repetitions"],config["random_seed"])
    bootstrap.to_csv(RESULTS/"flexibility_bootstrap.csv",index=False)
    correlation=correlations(primary_daily);correlation.to_csv(RESULTS/"flexibility_correlations.csv",index=False)
    pairwise=pairwise_summary(primary_daily);pairwise.to_csv(RESULTS/"flexibility_pairwise.csv",index=False)
    conditional=conditional_summary(daily);conditional.to_csv(RESULTS/"flexibility_conditional_summary.csv",index=False)
    representatives=representative_days(primary_daily);representatives.to_csv(RESULTS/"representative_days.csv",index=False)
    price_audit=("The canonical SPARK metadata describes a German day-ahead wholesale price in UTC and EUR/MWh, "
        "but does not identify its exchange or publication timestamp. In 2024 all four quarter-hours within each UTC hour "
        "carry one repeated hourly value. [EPEX SPOT trading documentation](https://www.epexspot.com/en/tradingproducts) "
        "states 12:00 D-1 order-book closure and result publication from 12:55 onward; its "
        "[19 December 2024 SDAC notice](https://www.epexspot.com/en/news/15-minute-products-set-go-live-june-2025-european-single-day-ahead-coupling-sdac) "
        "scheduled 15-minute day-ahead products for 2025. Thus the complete "
        "profile is not known at 12:00 and source provenance is insufficient to assign interval-specific availability times.")
    d2=primary_daily[primary_daily.forecast_method.eq("D2_DETERMINISTIC")]
    d3=primary_daily[primary_daily.forecast_method.eq("D3_PROBABILISTIC")]
    paired=d2[["date","savings_eur"]].merge(d3[["date","savings_eur"]],on="date",suffixes=("_d2","_d3"))
    difference=(paired.savings_eur_d3-paired.savings_eur_d2).mean()
    perfect_mean=primary_daily[primary_daily.forecast_method.eq("D1_PERFECT_FORESIGHT")].savings_eur.mean()
    if perfect_mean<=0:category="C. Forecast errors or limited flexibility eliminate most decision value"
    elif abs(difference)<.02*perfect_mean:category="B. Forecast-aware flexibility creates value, but probabilistic forecasting adds little beyond the deterministic model"
    elif difference>0:category="A. Probabilistic forecasts provide meaningful economic/operational advantage over deterministic forecasting"
    else:category="D. Results are too assumption-dependent for strong conclusions"
    make_figures(daily,aggregate,schedule,primary_daily,representatives)
    write_report(config,price_audit,day_summary,aggregate,bootstrap,correlation,pairwise,
                 conditional,representatives,category)
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"git_commit":git_commit(),
        "python_version":platform.python_version(),"scipy_version":scipy.__version__,"solver":"scipy.optimize.linprog HiGHS",
        "source_hash":digest(Path(__file__)),
        "input_hashes":{"canonical_price":digest(PRICE_PATH),"original_price_source":digest(PRICE_SOURCE),
            "deterministic_predictions":digest(LGBM_PATH),"probabilistic_predictions":digest(PROB_PATH),
            "baseline_predictions":digest(BASELINE_PATH),"regime_predictions":digest(REGIME_PRED_PATH),
            "regime_manifest":digest(REGIME_MANIFEST),"configuration":digest(CONFIG_PATH)},
        "configuration":config,"random_seeds":{"bootstrap":config["random_seed"]},
        "scenario_definitions":{"D1":"actual oracle","B3":"recent-week profile","D2":"deterministic LightGBM",
            "D3":"R0 q10/q50/q90 weights 0.2/0.6/0.2","D3_Q95":"R0 q10/q50/q95 weights 0.2/0.6/0.2"},
        "decision_category":category,"outputs":{}}
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file() and path.name!="flexibility_run_manifest.json":
            manifest["outputs"][str(path.relative_to(RESULTS))]={"bytes":path.stat().st_size,"sha256":digest(path)}
    (RESULTS/"flexibility_run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"regime_commit":git_commit(),"complete_days":complete.forecast_date_local.nunique(),
        "category":category,"primary":aggregate[aggregate.scenario.eq("PRIMARY")].to_dict("records")},indent=2,default=str))


if __name__=="__main__":main()
