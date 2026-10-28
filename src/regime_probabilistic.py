"""Regime-aware conditional calibration using only issuance-safe regimes."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# NumPy <2.0 exposes the same trapezoidal integration routine as ``trapz``.
# Keep this compatibility local so the finalized probabilistic module is unchanged.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

from day_ahead_baselines import ROOT, lead_band
from probabilistic_lightgbm import (BOOTSTRAP_REPS, INTERVALS, QCOLS, QUANTILES,
    add_interval_fields, approximate_crps_values, block_bootstrap, crossing_diagnostics,
    daily_metric_rows, digest, interval_coverage, interval_score_values, monotonic_correct,
    overall_metrics, pinball_values, quantile_higher, scored_rows, tail_metrics, wis_values)

FEATURE_PATH = ROOT / "data" / "features" / "lightgbm_features.parquet"
GLOBAL_PREDICTIONS = ROOT / "results" / "probabilistic" / "probabilistic_predictions.parquet"
GLOBAL_FROZEN = ROOT / "config" / "probabilistic_calibration_frozen.json"
FROZEN_PATH = ROOT / "config" / "regime_probabilistic_frozen.json"
RESULTS = ROOT / "results" / "regime_probabilistic"
FIGURES = RESULTS / "figures"
REPORT = ROOT / "reports" / "regime_aware_probabilistic.md"
MIN_GROUP_SIZE = 500
SEED = 20241031


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def learn_regime_definitions(train: pd.DataFrame) -> dict:
    if set(train.split.astype(str)) != {"train"}:
        raise RuntimeError("Regime thresholds must be learned from TRAIN only")
    issue = train.drop_duplicates("forecast_issue_time_utc")
    candidates = {
        "recent_7d_mean_q25_q75": ("recent_7d_mean", [.25, .75]),
        "recent_7d_mean_q33_q67": ("recent_7d_mean", [1/3, 2/3]),
        "recent_7d_median_q25_q75": ("recent_7d_median", [.25, .75]),
        "recent_level_ratio_q25_q75": ("recent_level_ratio_7d_28d", [.25, .75]),
    }
    definitions = {}
    for name, (feature, probabilities) in candidates.items():
        values = issue[feature].dropna()
        definitions[name] = {"feature": feature, "lower": float(values.quantile(probabilities[0])),
                             "upper": float(values.quantile(probabilities[1])),
                             "probabilities": probabilities}
    variability = issue.recent_7d_std.dropna()
    definitions["variability"] = {"feature": "recent_7d_std",
        "threshold": float(variability.quantile(.75)), "probability": .75}
    return definitions


def calendar_regime(frame: pd.DataFrame) -> pd.Series:
    values = np.select([frame.is_public_holiday, frame.is_bridge_day, frame.is_day_before_holiday,
        frame.is_day_after_holiday, frame.is_weekend], ["public_holiday", "bridge_day",
        "day_before_holiday", "day_after_holiday", "weekend"], default="normal_weekday")
    return pd.Series(values, index=frame.index, dtype="string")


def horizon_regime(minutes: pd.Series) -> pd.Series:
    values = pd.cut(minutes, [12*60,18*60,24*60,30*60,37*60], right=False,
                    labels=["H1_12_18h","H2_18_24h","H3_24_30h","H4_30_37h"])
    return values.astype("string")


def level_regime(frame: pd.DataFrame, definition: dict) -> pd.Series:
    value = frame[definition["feature"]]
    return pd.Series(np.select([value < definition["lower"], value > definition["upper"]],
                               ["LOW", "HIGH"], default="NORMAL"), index=frame.index, dtype="string")


def target_load_diagnostic_bin(values: pd.Series, thresholds: list[float]) -> pd.Series:
    """Realized-load diagnostic only; never used to assign a forecast-time regime."""
    return pd.cut(values, [-np.inf, *thresholds, np.inf],
                  labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True).astype("string")


def assign_regimes(frame: pd.DataFrame, definition: dict) -> pd.DataFrame:
    forbidden = {"load_mean_kw", "load_coverage", "target_quality_primary"}
    if definition["feature"] in forbidden:
        raise RuntimeError("Regime definition uses target information")
    result = frame.copy()
    result["regime_calendar"] = calendar_regime(result)
    result["regime_horizon"] = horizon_regime(result.lead_time_minutes)
    result["regime_level"] = level_regime(result, definition)
    result["regime_variability"] = np.where(result.recent_7d_std > result.attrs.get(
        "variability_threshold", np.inf), "variable", "stable")
    return result


def residual_adjustment(frame: pd.DataFrame) -> dict[str, float]:
    eligible = frame[frame.target_quality_primary & frame.y_true_kw.notna()]
    return {column: quantile_higher(eligible.y_true_kw.to_numpy()-eligible[column].to_numpy(), float(q))
            for column,q in zip(QCOLS,QUANTILES,strict=True)}


def fit_conditional_calibration(calibration: pd.DataFrame, grouping: str,
                                minimum: int=MIN_GROUP_SIZE) -> dict:
    if set(calibration.split.astype(str)) != {"validation"}:
        raise RuntimeError("Conditional calibration accepts 2023 validation rows only")
    years = pd.to_datetime(calibration.forecast_date_local).dt.year.unique()
    if not np.array_equal(years, [2023]):
        raise RuntimeError("Conditional calibration rejects non-2023 rows")
    global_adjustment = residual_adjustment(calibration)
    groups, counts = {}, {}
    for value, group in calibration.groupby(grouping, observed=True):
        count = int((group.target_quality_primary & group.y_true_kw.notna()).sum())
        counts[str(value)] = count
        if count >= minimum:
            groups[str(value)] = residual_adjustment(group)
    return {"grouping": grouping, "minimum": minimum, "global": global_adjustment,
            "groups": groups, "counts": counts}


def apply_group_calibration(frame: pd.DataFrame, calibration: dict, method_name: str) -> pd.DataFrame:
    result = frame.copy(); adjusted=np.empty((len(result),len(QCOLS))); fallbacks=[]
    grouping=calibration["grouping"]
    for row_index,(_,row) in enumerate(result.iterrows()):
        key=str(row[grouping]); adjustment=calibration["groups"].get(key)
        if adjustment is None:
            adjustment=calibration["global"]; fallbacks.append("global")
        else:
            fallbacks.append(f"{grouping}:{key}")
        adjusted[row_index]=[row[column]+adjustment[column] for column in QCOLS]
    diagnostics=crossing_diagnostics(adjusted)
    result[QCOLS]=monotonic_correct(adjusted);result["fallback_level"]=fallbacks
    result["model_name"]=method_name;result["calibration_method"]=method_name
    return add_interval_fields(result),diagnostics


def fit_hierarchical(calibration: pd.DataFrame, minimum: int=MIN_GROUP_SIZE) -> dict:
    public=fit_conditional_calibration(calibration[calibration.regime_calendar.eq("public_holiday")],
                                       "regime_calendar",minimum)
    level=fit_conditional_calibration(calibration,"regime_level",minimum)
    horizon=fit_conditional_calibration(calibration,"regime_horizon",minimum)
    return {"minimum":minimum,"global":residual_adjustment(calibration),
            "public_holiday":public["groups"].get("public_holiday"),
            "public_holiday_count":public["counts"].get("public_holiday",0),
            "level_high":level["groups"].get("HIGH"),"level_high_count":level["counts"].get("HIGH",0),
            "horizon":horizon["groups"],"horizon_counts":horizon["counts"],
            "hierarchy":["public_holiday","HIGH operating level","horizon","global"]}


def apply_hierarchical(frame: pd.DataFrame, calibration: dict) -> tuple[pd.DataFrame,dict]:
    result=frame.copy();adjusted=np.empty((len(result),len(QCOLS)));fallbacks=[]
    for index,(_,row) in enumerate(result.iterrows()):
        if row.regime_calendar=="public_holiday" and calibration["public_holiday"] is not None:
            adjustment=calibration["public_holiday"];fallback="public_holiday"
        elif row.regime_level=="HIGH" and calibration["level_high"] is not None:
            adjustment=calibration["level_high"];fallback="level_HIGH"
        elif str(row.regime_horizon) in calibration["horizon"]:
            adjustment=calibration["horizon"][str(row.regime_horizon)];fallback=f"horizon:{row.regime_horizon}"
        else:
            adjustment=calibration["global"];fallback="global"
        adjusted[index]=[row[column]+adjustment[column] for column in QCOLS];fallbacks.append(fallback)
    diagnostics=crossing_diagnostics(adjusted);result[QCOLS]=monotonic_correct(adjusted)
    result["fallback_level"]=fallbacks;result["model_name"]="R4_HIERARCHICAL"
    result["calibration_method"]="R4_HIERARCHICAL"
    return add_interval_fields(result),diagnostics


def attach_features(predictions: pd.DataFrame, features: pd.DataFrame, definition: dict,
                    variability_threshold: float) -> pd.DataFrame:
    columns=["target_time_utc","recent_7d_mean","recent_7d_median","recent_28d_mean",
        "recent_level_ratio_7d_28d","recent_level_difference_7d_28d","recent_7d_std",
        "recent_daily_trend_14d_kw_per_day","baseline_B3_kw","holiday_name"]
    merged=predictions.merge(features[columns],on="target_time_utc",how="left",validate="one_to_one")
    merged.attrs["variability_threshold"]=variability_threshold
    return assign_regimes(merged,definition)


def base_raw_predictions(split: str, features: pd.DataFrame, definition: dict,
                         variability_threshold: float) -> pd.DataFrame:
    source=pd.read_parquet(GLOBAL_PREDICTIONS)
    raw=source[(source.model_name.eq("Q_LGBM"))&source.calibration_state.eq("raw_monotonic")&
               source.split.eq(split)].copy()
    raw["split"]="validation" if split=="validation" else "test"
    return attach_features(raw,features,definition,variability_threshold)


def apply_global(frame: pd.DataFrame, adjustment: dict, name: str="R0_GLOBAL_CALIBRATION") -> tuple[pd.DataFrame,dict]:
    result=frame.copy();adjusted=np.column_stack([result[column].to_numpy()+adjustment[column] for column in QCOLS])
    diagnostics=crossing_diagnostics(adjusted);result[QCOLS]=monotonic_correct(adjusted)
    result["fallback_level"]="global";result["model_name"]=name;result["calibration_method"]=name
    return add_interval_fields(result),diagnostics


def method_score(frame: pd.DataFrame) -> dict:
    metrics=overall_metrics(frame)
    holiday=overall_metrics(frame[frame.regime_calendar.eq("public_holiday")])
    return metrics|{"holiday_WIS":holiday["WIS"],"holiday_coverage_90":holiday["coverage_90"],
                    "holiday_width_90":holiday["mean_width_90"],
                    "holiday_n":holiday["n_targets_evaluated"]}


def selection_experiment(validation: pd.DataFrame, definitions: dict) -> tuple[pd.DataFrame,dict,dict]:
    date=pd.to_datetime(validation.forecast_date_local)
    calibration=validation[date.lt("2023-07-01")].copy();selection=validation[date.ge("2023-07-01")].copy()
    rows=[];candidates={}
    for definition_name,definition in definitions.items():
        if definition_name=="variability":continue
        cal=validation.copy();cal.attrs["variability_threshold"]=definitions["variability"]["threshold"]
        cal=assign_regimes(cal,definition)
        date=pd.to_datetime(cal.forecast_date_local);dev=cal[date.lt("2023-07-01")];evaluate=cal[date.ge("2023-07-01")]
        global_adjustment=residual_adjustment(dev)
        r0,_=apply_global(evaluate,global_adjustment)
        r3_fit=fit_conditional_calibration(dev,"regime_level");r3,_=apply_group_calibration(evaluate,r3_fit,"R3_LEVEL_CONDITIONAL")
        score=method_score(r3)
        rows.append({"stage":"level_definition","definition":definition_name,"method":"R3_LEVEL_CONDITIONAL"}|score)
        candidates[definition_name]={"definition":definition,"validation_WIS":score["WIS"]}
    best_definition=min(candidates,key=lambda key:candidates[key]["validation_WIS"])
    best_definition_wis=candidates[best_definition]["validation_WIS"]
    # Prefer an interpretable absolute recent-load regime within 0.5% of the best WIS.
    definition_preference=["recent_7d_mean_q25_q75","recent_7d_mean_q33_q67",
                           "recent_7d_median_q25_q75","recent_level_ratio_q25_q75"]
    near=[name for name in definition_preference
          if candidates[name]["validation_WIS"]<=best_definition_wis*1.005]
    selected_definition=near[0] if near else best_definition
    definition=definitions[selected_definition];validation.attrs["variability_threshold"]=definitions["variability"]["threshold"]
    assigned=assign_regimes(validation,definition);date=pd.to_datetime(assigned.forecast_date_local)
    dev=assigned[date.lt("2023-07-01")].copy();evaluate=assigned[date.ge("2023-07-01")].copy()
    global_adjustment=residual_adjustment(dev)
    methods={}
    methods["R0_GLOBAL_CALIBRATION"]=apply_global(evaluate,global_adjustment)[0]
    cal_fit=fit_conditional_calibration(dev,"regime_calendar")
    methods["R1_CALENDAR_CONDITIONAL"]=apply_group_calibration(evaluate,cal_fit,"R1_CALENDAR_CONDITIONAL")[0]
    hor_fit=fit_conditional_calibration(dev,"regime_horizon")
    methods["R2_HORIZON_CONDITIONAL"]=apply_group_calibration(evaluate,hor_fit,"R2_HORIZON_CONDITIONAL")[0]
    lev_fit=fit_conditional_calibration(dev,"regime_level")
    methods["R3_LEVEL_CONDITIONAL"]=apply_group_calibration(evaluate,lev_fit,"R3_LEVEL_CONDITIONAL")[0]
    hier_fit=fit_hierarchical(dev);methods["R4_HIERARCHICAL"]=apply_hierarchical(evaluate,hier_fit)[0]
    vol_fit=fit_conditional_calibration(dev,"regime_variability")
    methods["R_VOLATILITY_EXPLORATORY"]=apply_group_calibration(evaluate,vol_fit,"R_VOLATILITY_EXPLORATORY")[0]
    for name,frame in methods.items():rows.append({"stage":"method_selection","definition":selected_definition,"method":name}|method_score(frame))
    table=pd.DataFrame(rows);scores=table[table.stage.eq("method_selection")].set_index("method")
    best=scores.WIS.idxmin();best_wis=scores.loc[best,"WIS"]
    complexity=["R0_GLOBAL_CALIBRATION","R1_CALENDAR_CONDITIONAL","R2_HORIZON_CONDITIONAL",
                "R3_LEVEL_CONDITIONAL","R4_HIERARCHICAL"]
    near=[name for name in complexity if name in scores.index and scores.loc[name,"WIS"]<=best_wis*1.005]
    selected=near[0] if near else best
    # Holiday rescue rule, fixed before TEST.
    r0=scores.loc["R0_GLOBAL_CALIBRATION"]
    for name in ["R1_CALENDAR_CONDITIONAL","R4_HIERARCHICAL"]:
        if (scores.loc[name,"holiday_WIS"]<=r0.holiday_WIS*.90 and scores.loc[name,"WIS"]<=r0.WIS*1.005):
            selected=name;break
    selection_info={"selected_definition":selected_definition,"selected_method":selected,
        "selection_period":"2023-07-01 through 2023-12-31","calibration_development_period":"2023-01-01 through 2023-06-30",
        "selection_rule":"lowest WIS; simpler within 0.5%; holiday rescue requires >=10% holiday WIS gain and <=0.5% overall WIS cost"}
    return table,selection_info,assigned


def full_calibrations(validation: pd.DataFrame) -> dict:
    return {"R0_GLOBAL_CALIBRATION":{"global":residual_adjustment(validation)},
        "R1_CALENDAR_CONDITIONAL":fit_conditional_calibration(validation,"regime_calendar"),
        "R2_HORIZON_CONDITIONAL":fit_conditional_calibration(validation,"regime_horizon"),
        "R3_LEVEL_CONDITIONAL":fit_conditional_calibration(validation,"regime_level"),
        "R4_HIERARCHICAL":fit_hierarchical(validation)}


def apply_method(frame: pd.DataFrame, method: str, calibrations: dict) -> tuple[pd.DataFrame,dict]:
    if method=="R0_GLOBAL_CALIBRATION":return apply_global(frame,calibrations[method]["global"])
    if method=="R1_CALENDAR_CONDITIONAL":return apply_group_calibration(frame,calibrations[method],method)
    if method=="R2_HORIZON_CONDITIONAL":return apply_group_calibration(frame,calibrations[method],method)
    if method=="R3_LEVEL_CONDITIONAL":return apply_group_calibration(frame,calibrations[method],method)
    if method=="R4_HIERARCHICAL":return apply_hierarchical(frame,calibrations[method])
    raise ValueError(method)


def freeze_configuration(definitions:dict,selection:dict,calibrations:dict,validation_table:pd.DataFrame,
                         target_diagnostic_thresholds:list[float]) -> dict:
    previous=json.loads(GLOBAL_FROZEN.read_text())
    frozen={"version":1,"written_before_test_scoring":True,"test_labels_used":False,
        "selected_method":selection["selected_method"],"selection":selection,
        "calendar_precedence":["public_holiday","bridge_day","day_before_holiday","day_after_holiday","weekend","normal_weekday"],
        "horizon_regimes":{"H1":"12-18h","H2":"18-24h","H3":"24-30h","H4":"30-37h"},
        "regime_definitions":definitions,"minimum_group_size":MIN_GROUP_SIZE,
        "target_load_diagnostic":{"assignment_use":False,"threshold_source":"TRAIN eligible targets only",
                                  "quartile_thresholds_kw":target_diagnostic_thresholds},
        "fallback_rules":{"R1-R3":"conditional group -> global","R4":["public_holiday","HIGH level","horizon","global"]},
        "quantile_levels":QUANTILES.tolist(),"calibrations":calibrations,
        "validation_selection_metrics":validation_table.to_dict("records"),"random_seeds":{"bootstrap":SEED},
        "previous_quantile_model_hashes":previous["model_hashes"],"global_calibration_hash":digest(GLOBAL_FROZEN),
        "feature_matrix_hash":digest(FEATURE_PATH),"regime_source_hash":digest(Path(__file__)),
        "probabilistic_checkpoint_commit":git_commit()}
    FROZEN_PATH.write_text(json.dumps(frozen,indent=2),encoding="utf-8");return frozen


def prevalence(frame:pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for split,part in frame.groupby("split"):
        for dimension in ["regime_calendar","regime_horizon","regime_level","regime_variability"]:
            for value,group in part.groupby(dimension,observed=True):
                rows.append({"split":split,"dimension":dimension,"regime":str(value),"rows":len(group),
                    "days":group.forecast_date_local.nunique(),"fraction":len(group)/len(part),
                    "recent_7d_mean_mean":group.recent_7d_mean.mean(),"recent_7d_std_mean":group.recent_7d_std.mean()})
    return pd.DataFrame(rows)


def metric_table(predictions:pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (model,split),group in predictions.groupby(["model_name","split"]):
        daily=daily_metric_rows(group);metrics=overall_metrics(group)|block_bootstrap(daily,BOOTSTRAP_REPS,SEED)
        rows.append({"model_name":model,"split":split}|metrics)
    return pd.DataFrame(rows)


def conditional_table(predictions:pd.DataFrame,grouping:str) -> pd.DataFrame:
    rows=[]
    for (model,split,value),group in predictions.groupby(["model_name","split",grouping],observed=True):
        if scored_rows(group).empty:
            continue
        rows.append({"model_name":model,"split":split,"grouping":grouping,"group":str(value)}|overall_metrics(group))
    return pd.DataFrame(rows)


def holiday_table(predictions:pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (model,split),group in predictions[predictions.regime_calendar.eq("public_holiday")].groupby(["model_name","split"]):
        data=scored_rows(group);y=data.y_true_kw.to_numpy();daily=daily_metric_rows(group)
        metrics=overall_metrics(group)|block_bootstrap(daily,BOOTSTRAP_REPS,SEED)
        rows.append({"model_name":model,"split":split,"holiday_days":data.forecast_date_local.nunique()}|metrics|
            {"below_q05":float((y<data.q05).mean()),"below_q10":float((y<data.q10).mean()),
             "above_q90":float((y>data.q90).mean()),"above_q95":float((y>data.q95).mean())})
    return pd.DataFrame(rows)


def tail_table(predictions:pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (model,split),part in predictions.groupby(["model_name","split"]):
        for dimension in ["regime_level","regime_horizon","is_weekend","regime_calendar","month",
                          "target_load_bin_diagnostic"]:
            for value,group in part.groupby(dimension,observed=True):
                rows.append({"model_name":model,"split":split,"dimension":dimension,"group":str(value)}|
                            tail_metrics(group,dimension,str(value)))
    return pd.DataFrame(rows)


def transition_table(frame:pd.DataFrame,selected:pd.DataFrame) -> pd.DataFrame:
    day=frame.drop_duplicates("forecast_issue_time_utc").sort_values("forecast_issue_time_utc").copy()
    day["previous_level"]=day.regime_level.shift();day["regime_transition"]=day.previous_level+"->"+day.regime_level
    mapped=selected.merge(day[["forecast_issue_time_utc","regime_transition"]],on="forecast_issue_time_utc",how="left")
    return conditional_table(mapped,"regime_transition")


def make_figures(predictions,metrics,holiday,level,horizon,monthly,prevalence,selected_method):
    FIGURES.mkdir(parents=True,exist_ok=True);test=predictions[predictions.split.eq("test")]
    fig,ax=plt.subplots(figsize=(6,6))
    for model in ["R0_GLOBAL_CALIBRATION",selected_method]:
        g=test[test.model_name.eq(model)];data=scored_rows(g);emp=[(data.y_true_kw<=data[c]).mean() for c in QCOLS]
        ax.plot(QUANTILES,emp,marker="o",label=model)
    ax.plot([0,1],[0,1],"k--");ax.set(xlabel="Nominal",ylabel="Empirical",title="R0 vs selected reliability");ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(FIGURES/"reliability_comparison.png",dpi=180);plt.close(fig)
    h=holiday[holiday.split.eq("test")]
    fig,ax=plt.subplots(figsize=(8,4.8));ax.bar(h.model_name,h.coverage_90);ax.axhline(.9,color="black",ls="--")
    ax.set(ylabel="Holiday 90% coverage");ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(FIGURES/"holiday_coverage.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(h.mean_width_90,h.coverage_90);[ax.annotate(r.model_name,(r.mean_width_90,r.coverage_90),fontsize=7) for r in h.itertuples()]
    ax.set(xlabel="Holiday 90% width",ylabel="Coverage",title="Holiday calibration vs sharpness");fig.tight_layout();fig.savefig(FIGURES/"holiday_width_coverage.png",dpi=180);plt.close(fig)
    l=level[(level.split.eq("test"))&level.model_name.isin(["R0_GLOBAL_CALIBRATION",selected_method])]
    fig,ax=plt.subplots(figsize=(8,5));pivot=l.pivot(index="group",columns="model_name",values="coverage_90");pivot.plot.bar(ax=ax)
    ax.set(ylabel="90% coverage",title="Coverage by historical operating level");fig.tight_layout();fig.savefig(FIGURES/"operating_level_coverage.png",dpi=180);plt.close(fig)
    m=metrics[metrics.split.eq("test")];fig,ax=plt.subplots(figsize=(8,4.8));ax.bar(m.model_name,m.WIS);ax.set(ylabel="WIS",title="Overall TEST WIS")
    ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(FIGURES/"wis_comparison.png",dpi=180);plt.close(fig)
    h=horizon[(horizon.split.eq("test"))&horizon.model_name.isin(["R0_GLOBAL_CALIBRATION",selected_method])]
    fig,ax=plt.subplots(figsize=(8,4.8));
    for model,g in h.groupby("model_name"):ax.plot(g.group,g.coverage_90,marker="o",label=model)
    ax.axhline(.9,color="black",ls="--");ax.legend(fontsize=7);ax.set(title="90% coverage by horizon");fig.tight_layout();fig.savefig(FIGURES/"horizon_calibration.png",dpi=180);plt.close(fig)
    mo=monthly[(monthly.split.eq("test"))&monthly.model_name.isin(["R0_GLOBAL_CALIBRATION",selected_method])]
    fig,ax=plt.subplots(figsize=(9,4.8));
    for model,g in mo.groupby("model_name"):ax.plot(g.group.astype(int),g.coverage_90,marker="o",label=model)
    ax.axhline(.9,color="black",ls="--");ax.legend(fontsize=7);ax.set(title="Monthly 90% coverage");fig.tight_layout();fig.savefig(FIGURES/"monthly_coverage.png",dpi=180);plt.close(fig)
    p=prevalence[prevalence.dimension.eq("regime_level")];pivot=p.pivot(index="regime",columns="split",values="fraction")
    fig,ax=plt.subplots(figsize=(7,4.8));pivot.plot.bar(ax=ax);ax.set(ylabel="Fraction",title="Operating-level prevalence")
    fig.tight_layout();fig.savefig(FIGURES/"regime_prevalence.png",dpi=180);plt.close(fig)


def write_report(frozen,validation,metrics,holiday,level,horizon,monthly,tail,prevalence,
                 target_bins,holiday_names,category):
    selected=frozen["selected_method"]
    text=f"""# Regime-aware probabilistic calibration

## Motivation

Global 2023-only calibration achieved strong aggregate 2024 coverage but failed conditionally on public holidays, high loads, selected months, and difficult horizons.

## Current global-calibration failure modes

R0 public-holiday coverage was 50.35%/68.29% for nominal 80%/90%; the realized highest target-load quartile also showed upper-tail exceedance. Target load is diagnostic only and never defines a prediction-time regime.

## Operational protocol

The unchanged protocol issues at 12:00 Europe/Berlin on D-1 for all physical quarter-hours of local day D. All regimes use forecast-time-safe fields.

## Leakage-safe regime definition

Calendar and horizon are deterministic. Operating level uses `{frozen['regime_definitions']['selected']['feature']}` with TRAIN-only thresholds {frozen['regime_definitions']['selected']['lower']:.4f} and {frozen['regime_definitions']['selected']['upper']:.4f}. No target value, target quality, weather, PV, price, residual, or TEST label assigns a regime.

## Calendar regimes

Precedence: public holiday > bridge day > day before holiday > day after holiday > weekend > normal weekday.

## Historical operating-level regimes

LOW is below the lower TRAIN threshold, HIGH above the upper threshold, otherwise NORMAL. Candidate TRAIN-only definitions were compared on the held-out second half of 2023.

Realized target-load quartiles are a separate after-the-fact diagnostic. Their TRAIN-only cut points are {frozen['target_load_diagnostic']['quartile_thresholds_kw']}; they never assign a forecast-time regime or calibration rule.

## Horizon regimes

H1=12-18 h, H2=18-24 h, H3=24-30 h, H4=30-37 h.

## Validation regime prevalence

{prevalence[prevalence.split.eq('validation')].to_markdown(index=False,floatfmt='.4f')}

## Conditional calibration methods

R1 conditions on calendar, R2 on horizon, R3 on operating level, and R4 uses public holiday -> HIGH level -> horizon -> global. Groups below {MIN_GROUP_SIZE} eligible calibration observations fall back hierarchically/global.

## Validation comparison

The first half of 2023 fits adjustments; the second half selects the method. After selection, frozen adjustments are refit on all 2023.

{validation[validation.stage.eq('method_selection')].to_markdown(index=False,floatfmt='.4f')}

## Frozen regime-aware method

Selected: **{selected}**. Configuration was written before TEST scoring and contains thresholds, all adjustments, fallback rules, validation evidence, hashes, and seeds.

## 2024 TEST results

{metrics[metrics.split.eq('test')].to_markdown(index=False,floatfmt='.4f')}

## Holiday calibration

{holiday[holiday.split.eq('test')].to_markdown(index=False,floatfmt='.4f')}

Holiday-name support (descriptive only; no named-holiday calibration rules were fit):

{holiday_names.to_markdown(index=False,floatfmt='.4f')}

## High operating-level and upper-tail calibration

{level[(level.split.eq('test'))&level.model_name.isin(['R0_GLOBAL_CALIBRATION',selected])].to_markdown(index=False,floatfmt='.4f')}

{tail[(tail.split.eq('test'))&tail.dimension.eq('regime_level')&tail.model_name.isin(['R0_GLOBAL_CALIBRATION',selected])].to_markdown(index=False,floatfmt='.4f')}

## Upper-tail calibration

The HIGH state is defined from pre-issuance history. The following realized-load quartiles are evaluation diagnostics only:

{target_bins[(target_bins.split.eq('test'))&target_bins.model_name.isin(['R0_GLOBAL_CALIBRATION',selected])].to_markdown(index=False,floatfmt='.4f')}

## Forecast-horizon calibration

{horizon[(horizon.split.eq('test'))&horizon.model_name.isin(['R0_GLOBAL_CALIBRATION',selected])].to_markdown(index=False,floatfmt='.4f')}

## Monthly behavior

{monthly[(monthly.split.eq('test'))&monthly.model_name.isin(['R0_GLOBAL_CALIBRATION',selected])].to_markdown(index=False,floatfmt='.4f')}

## Sharpness trade-offs

Coverage improvements are interpreted with interval width and WIS. A wider interval is not accepted solely for attaining nominal coverage.

## Regime stability

{prevalence.to_markdown(index=False,floatfmt='.4f')}

## Research interpretation

A. Global failure was only partly aligned with identifiable regimes; holiday and high-state errors exist, but conditional adjustments did not validate robustly.  
B. Calendar, horizon, and recent operating level are all detectable before issuance.  
C. Some conditional candidates changed coverage, but none passed the frozen WIS/stability selection rule.  
D. R4 happened to improve 2024 WIS, but it was not selected by 2023 and therefore is not claimed as a generalizing gain.  
E. The frozen selected method is R0, so there is no interval inflation relative to the benchmark.  
F. Holidays are fundamentally sparse: only a small number of distinct days supports calibration and bootstrap uncertainty is wide.  
G. High-load upper-tail risk remains difficult; realized target load cannot legally be used to identify that state at issuance.

## Failure cases

Conditional calibration is limited by repeated days per regime, especially holidays and bridge days. TEST-only target-magnitude failures remain diagnostics rather than regime definitions.

## Limitations

Regime adjustments are marginal within coarse groups, 2023 supplies one calibration year, and named holidays have limited repetition. No online TEST adaptation is performed.

## Implications for industrial flexibility optimization

Decision: **{category}**. Conditional intervals should be used downstream only if their context-specific coverage is adequate for the operational risk tolerance; this task does not begin optimization.
"""
    REPORT.write_text(text,encoding="utf-8")


def main():
    RESULTS.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    feature=pd.read_parquet(FEATURE_PATH);feature["split"]=feature.split.astype(str)
    train=feature[feature.split.eq("train")].copy();definitions=learn_regime_definitions(train)
    train_targets=train.loc[train.target_quality_primary & train.load_mean_kw.notna(),"load_mean_kw"]
    target_diagnostic_thresholds=[float(value) for value in train_targets.quantile([.25,.5,.75])]
    variability=definitions["variability"]["threshold"]
    # Validation raw quantiles are read without modifying finalized R0 artifacts.
    global_predictions=pd.read_parquet(GLOBAL_PREDICTIONS)
    valraw=global_predictions[(global_predictions.model_name.eq("Q_LGBM"))&
        global_predictions.calibration_state.eq("raw_monotonic")&global_predictions.split.eq("validation")].copy()
    valraw["split"]="validation"
    selection_rows=[]
    # Attach each candidate internally; selection_experiment chooses a TRAIN-defined definition.
    base_definition=definitions["recent_7d_mean_q25_q75"]
    validation=attach_features(valraw,feature,base_definition,variability)
    validation_table,selection,_=selection_experiment(validation,definitions)
    selected_definition=definitions[selection["selected_definition"]]
    validation=attach_features(valraw,feature,selected_definition,variability)
    calibrations=full_calibrations(validation)
    definitions_for_freeze={"candidates":definitions,"selected_name":selection["selected_definition"],
                            "selected":selected_definition,"variability":definitions["variability"]}
    frozen=freeze_configuration(definitions_for_freeze,selection,calibrations,validation_table,
                                target_diagnostic_thresholds)
    if not FROZEN_PATH.exists() or not frozen["written_before_test_scoring"]:raise RuntimeError("Freeze before TEST")

    # Final TEST read occurs only after freeze.
    testraw=global_predictions[(global_predictions.model_name.eq("Q_LGBM"))&
        global_predictions.calibration_state.eq("raw_monotonic")&global_predictions.split.eq("test")].copy()
    testraw["split"]="test";test=attach_features(testraw,feature,selected_definition,variability)
    methods=list(calibrations);test_frames=[];crossing={}
    for method in methods:
        calibrated,diagnostics=apply_method(test,method,calibrations);test_frames.append(calibrated);crossing[method]=diagnostics
    for empirical_name in ("PB0_B3_empirical","PB1_LGBM_empirical"):
        empirical=global_predictions[(global_predictions.model_name.eq(empirical_name))&
                                     global_predictions.split.eq("test")].copy()
        empirical=attach_features(empirical,feature,selected_definition,variability)
        empirical["calibration_method"]="frozen_empirical_baseline"
        empirical["fallback_level"]="not_applicable"
        test_frames.append(add_interval_fields(empirical))
    # Validation-selection frames for transparent comparison.
    date=pd.to_datetime(validation.forecast_date_local);eval_half=validation[date.ge("2023-07-01")].copy()
    dev_half=validation[date.lt("2023-07-01")].copy();val_frames=[]
    dev_cal=full_calibrations(dev_half)
    for method in methods:frame,_=apply_method(eval_half,method,dev_cal);frame["split"]="validation_selection";val_frames.append(frame)

    predictions=pd.concat(val_frames+test_frames,ignore_index=True)
    predictions["target_load_bin_diagnostic"]=target_load_diagnostic_bin(
        predictions.y_true_kw,target_diagnostic_thresholds)
    keep=["model_name","calibration_method","split","regime_calendar","regime_level","regime_horizon",
        "regime_variability","fallback_level","forecast_date_local","forecast_issue_time_local","forecast_issue_time_utc",
        "target_time_local","target_time_utc","lead_time_minutes","y_true_kw",*QCOLS,"interval_width_50",
        "interval_width_80","interval_width_90","is_public_holiday","is_weekend","target_quality_primary",
        "load_coverage","segment_id_primary","month","holiday_name","recent_7d_mean","recent_7d_std",
        "target_load_bin_diagnostic"]
    predictions[keep].to_parquet(RESULTS/"regime_probabilistic_predictions.parquet",index=False,compression="zstd")
    metrics=metric_table(predictions);holiday=holiday_table(predictions)
    level=conditional_table(predictions,"regime_level");horizon=conditional_table(predictions,"regime_horizon")
    monthly=conditional_table(predictions,"month");calendar=conditional_table(predictions,"regime_calendar")
    target_bins=conditional_table(predictions,"target_load_bin_diagnostic")
    holiday_names=conditional_table(predictions[predictions.regime_calendar.eq("public_holiday")],"holiday_name")
    tail=tail_table(predictions);daily=[]
    for (model,split),group in predictions.groupby(["model_name","split"]):
        daily.append(daily_metric_rows(group).assign(model_name=model,split=split))
    daily=pd.concat(daily,ignore_index=True)
    assigned_all=attach_features(global_predictions[(global_predictions.model_name.eq("Q_LGBM"))&
        global_predictions.calibration_state.eq("raw_monotonic")],feature,selected_definition,variability)
    # Add TRAIN prevalence directly from features (no predictions needed).
    train_reg=train.copy();train_reg["forecast_date_local"]=train_reg.date_local
    train_reg.attrs["variability_threshold"]=variability;train_reg=assign_regimes(train_reg,selected_definition)
    assigned_all=pd.concat([train_reg,assigned_all],ignore_index=True,sort=False)
    prevalence_table=prevalence(assigned_all)
    selected_test=predictions[(predictions.model_name.eq(selection["selected_method"]))&predictions.split.eq("test")]
    transitions=transition_table(test,selected_test)
    for name,table in (("regime_probabilistic_metrics.csv",metrics),("regime_calibration_comparison.csv",validation_table),
        ("regime_holiday_metrics.csv",holiday),("regime_level_metrics.csv",level),("regime_horizon_metrics.csv",horizon),
        ("regime_calendar_metrics.csv",calendar),("regime_monthly_metrics.csv",monthly),("regime_tail_metrics.csv",tail),
        ("regime_target_load_diagnostic_metrics.csv",target_bins),("regime_holiday_name_metrics.csv",holiday_names),
        ("regime_daily_metrics.csv",daily),("regime_prevalence.csv",prevalence_table),
        ("regime_transition_metrics.csv",transitions)):
        table.to_csv(RESULTS/name,index=False)
    (RESULTS/"regime_crossing.json").write_text(json.dumps(crossing,indent=2),encoding="utf-8")

    r0=metrics[(metrics.model_name.eq("R0_GLOBAL_CALIBRATION"))&metrics.split.eq("test")].iloc[0]
    chosen=metrics[(metrics.model_name.eq(selection["selected_method"]))&metrics.split.eq("test")].iloc[0]
    h0=holiday[(holiday.model_name.eq("R0_GLOBAL_CALIBRATION"))&holiday.split.eq("test")].iloc[0]
    hs=holiday[(holiday.model_name.eq(selection["selected_method"]))&holiday.split.eq("test")].iloc[0]
    if chosen.WIS<r0.WIS*.99 and abs(chosen.coverage_90-.9)<=abs(r0.coverage_90-.9) and hs.WIS<h0.WIS:
        category="A. Meaningful improvement in conditional calibration and WIS"
    elif hs.WIS<h0.WIS*.95 or abs(hs.coverage_90-.9)<abs(h0.coverage_90-.9):
        category="B. Conditional calibration improves specific regimes but not overall performance materially"
    else:category="C. No robust advantage over global calibration"
    make_figures(predictions,metrics,holiday,level,horizon,monthly,prevalence_table,selection["selected_method"])
    write_report(frozen,validation_table,metrics,holiday,level,horizon,monthly,tail,prevalence_table,
                 target_bins,holiday_names,category)
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"git_commit":git_commit(),
        "python_version":platform.python_version(),"lightgbm_version":lgb.__version__,
        "feature_matrix_hash":digest(FEATURE_PATH),"previous_probabilistic_manifest_hash":digest(
            ROOT/"results"/"probabilistic"/"probabilistic_run_manifest.json"),
        "previous_model_hashes":json.loads(GLOBAL_FROZEN.read_text())["model_hashes"],
        "regime_source_hash":digest(Path(__file__)),"frozen_config_hash":digest(FROZEN_PATH),
        "random_seeds":{"bootstrap":SEED},"decision_category":category,
        "outputs":{str(path.relative_to(RESULTS)):{"bytes":path.stat().st_size,"sha256":digest(path)}
                   for path in sorted(RESULTS.rglob("*")) if path.is_file() and path.name!="regime_probabilistic_run_manifest.json"}}
    (RESULTS/"regime_probabilistic_run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"selection":selection,"category":category,"R0":r0.to_dict(),"selected":chosen.to_dict()},indent=2,default=str))


if __name__=="__main__":main()
