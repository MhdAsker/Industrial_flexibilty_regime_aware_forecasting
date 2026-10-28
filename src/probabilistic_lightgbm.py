"""Leakage-safe probabilistic LightGBM and empirical interval benchmarks."""
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

from day_ahead_baselines import ROOT, SEED, lead_band

FEATURE_PATH = ROOT / "data" / "features" / "lightgbm_features.parquet"
FEATURE_SPEC = ROOT / "config" / "lightgbm_features.json"
DETERMINISTIC_FROZEN = ROOT / "config" / "lightgbm_frozen.json"
DETERMINISTIC_PREDICTIONS = ROOT / "results" / "lightgbm" / "lightgbm_predictions.parquet"
FROZEN_PATH = ROOT / "config" / "probabilistic_calibration_frozen.json"
RESULTS = ROOT / "results" / "probabilistic"
FIGURES = RESULTS / "figures"
REPORT = ROOT / "reports" / "probabilistic_lightgbm.md"

QUANTILES = np.array([.05, .10, .25, .50, .75, .90, .95])
QCOLS = [f"q{int(q*100):02d}" for q in QUANTILES]
INTERVALS = {"50": ("q25", "q75", .50), "80": ("q10", "q90", .20),
             "90": ("q05", "q95", .10)}
MODEL_SEED = 20241029
BOOTSTRAP_SEED = 20241030
BOOTSTRAP_REPS = 2000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def validate_quantiles(levels: np.ndarray = QUANTILES) -> None:
    if not np.array_equal(levels, QUANTILES) or np.any(np.diff(levels) <= 0):
        raise ValueError("Quantile levels must be exactly 0.05,0.10,0.25,0.50,0.75,0.90,0.95")


def pinball_values(y: np.ndarray, prediction: np.ndarray, quantile: float) -> np.ndarray:
    error = y - prediction
    return np.maximum(quantile * error, (quantile - 1) * error)


def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (y >= lower) & (y <= upper)


def interval_score_values(y: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                          alpha: float) -> np.ndarray:
    width = upper - lower
    return width + (2 / alpha) * (lower - y) * (y < lower) + (2 / alpha) * (y - upper) * (y > upper)


def wis_values(y: np.ndarray, quantile_frame: pd.DataFrame) -> np.ndarray:
    score = .5 * np.abs(y - quantile_frame.q50.to_numpy())
    for lower, upper, alpha in INTERVALS.values():
        score += (alpha / 2) * interval_score_values(y, quantile_frame[lower].to_numpy(),
                                                      quantile_frame[upper].to_numpy(), alpha)
    return score / (len(INTERVALS) + .5)


def approximate_crps_values(y: np.ndarray, quantile_frame: pd.DataFrame) -> np.ndarray:
    losses = np.column_stack([pinball_values(y, quantile_frame[column].to_numpy(), q)
                              for column, q in zip(QCOLS, QUANTILES, strict=True)])
    return 2 * np.trapezoid(losses, QUANTILES, axis=1)


def crossing_diagnostics(raw: np.ndarray) -> dict:
    differences = np.diff(raw, axis=1)
    crossings = differences < 0
    magnitude = np.maximum(-differences, 0)
    return {"timestamps": len(raw), "timestamps_with_any_crossing": int(crossings.any(axis=1).sum()),
        "crossing_rate": float(crossings.any(axis=1).mean()),
        "mean_crossing_magnitude_kw_when_present": float(magnitude[crossings].mean()) if crossings.any() else 0.0,
        "max_crossing_magnitude_kw": float(magnitude.max(initial=0)),
        "adjacent_pair_rates": {f"{QCOLS[i]}>{QCOLS[i+1]}": float(crossings[:, i].mean())
                                for i in range(crossings.shape[1])}}


def monotonic_correct(raw: np.ndarray) -> np.ndarray:
    return np.sort(raw, axis=1)


def sorting_impact(frame: pd.DataFrame) -> dict:
    data = frame[frame.target_quality_primary & frame.y_true_kw.notna()].copy()
    y = data.y_true_kw.to_numpy()
    raw = data[[f"raw_{column}" for column in QCOLS]].to_numpy()
    corrected = data[QCOLS].to_numpy()
    def summarize(values: np.ndarray) -> tuple[float, float]:
        pinball = np.mean([pinball_values(y, values[:, index], float(q)).mean()
                           for index, q in enumerate(QUANTILES)])
        calibration = np.mean([abs((y <= values[:, index]).mean() - q)
                               for index, q in enumerate(QUANTILES)])
        return float(pinball), float(calibration)
    raw_pinball, raw_calibration = summarize(raw)
    sorted_pinball, sorted_calibration = summarize(corrected)
    return {"raw_unsorted_mean_pinball": raw_pinball,
            "monotonic_sorted_mean_pinball": sorted_pinball,
            "pinball_change_sorted_minus_raw": sorted_pinball - raw_pinball,
            "raw_unsorted_calibration_error": raw_calibration,
            "monotonic_sorted_calibration_error": sorted_calibration,
            "calibration_error_change_sorted_minus_raw": sorted_calibration - raw_calibration}


def quantile_higher(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="higher"))


def calibration_adjustments(validation: pd.DataFrame) -> dict[str, float]:
    if set(validation.split.astype(str)) != {"validation"}:
        raise RuntimeError("Calibration API accepts 2023 validation rows only")
    eligible = validation[validation.target_quality_primary]
    return {column: quantile_higher(eligible.y_true_kw.to_numpy() - eligible[column].to_numpy(), q)
            for column, q in zip(QCOLS, QUANTILES, strict=True)}


def apply_calibration(frame: pd.DataFrame, adjustments: dict[str, float]) -> pd.DataFrame:
    result = frame.copy()
    adjusted = np.column_stack([result[column].to_numpy() + adjustments[column] for column in QCOLS])
    result[QCOLS] = monotonic_correct(adjusted)
    return result


def safe_features() -> tuple[list[str], dict]:
    frozen = json.loads(DETERMINISTIC_FROZEN.read_text(encoding="utf-8"))
    features = frozen["features"]
    prohibited = ("load_mean_kw", "load_coverage", "target_quality", "observed_", "pv_", "price")
    unsafe = [feature for feature in features if any(token in feature.lower() for token in prohibited)]
    if unsafe:
        raise RuntimeError(f"Unsafe finalized feature list: {unsafe}")
    return features, frozen


def quantile_parameters(alpha: float) -> dict:
    return {"objective": "quantile", "alpha": alpha, "num_leaves": 15, "learning_rate": .05,
        "min_child_samples": 100, "feature_fraction": 1.0, "bagging_fraction": 1.0,
        "bagging_freq": 0, "reg_alpha": 0.0, "reg_lambda": .1, "n_estimators": 500,
        "random_state": MODEL_SEED, "deterministic": True, "force_col_wise": True,
        "n_jobs": -1, "verbosity": -1, "max_bin": 255}


def train_quantile_models(development: pd.DataFrame, features: list[str]) -> tuple[dict[str, lgb.LGBMRegressor], pd.DataFrame, dict]:
    if set(development.split.astype(str)) - {"train", "validation"}:
        raise RuntimeError("TEST rows supplied to quantile development")
    train = development[development.split.eq("train") & development.target_quality_primary]
    validation = development[development.split.eq("validation") & development.target_quality_primary]
    models, predictions, iterations = {}, {}, {}
    for column, quantile in zip(QCOLS, QUANTILES, strict=True):
        model = lgb.LGBMRegressor(**quantile_parameters(float(quantile)))
        model.fit(train[features], train.load_mean_kw, eval_set=[(validation[features], validation.load_mean_kw)],
                  eval_metric="quantile", callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        models[column] = model
        iterations[column] = int(model.best_iteration_)
        predictions[column] = model.predict(validation[features], num_iteration=model.best_iteration_)
    result = validation.copy()
    raw = np.column_stack([predictions[column] for column in QCOLS])
    for index, column in enumerate(QCOLS):
        result[f"raw_{column}"] = raw[:, index]
    result[QCOLS] = monotonic_correct(raw)
    result["y_true_kw"] = result.load_mean_kw
    return models, result, iterations


def predict_quantiles(models: dict[str, lgb.LGBMRegressor], rows: pd.DataFrame,
                      features: list[str]) -> pd.DataFrame:
    raw = np.column_stack([models[column].predict(rows[features], num_iteration=models[column].best_iteration_)
                           for column in QCOLS])
    result = rows.copy()
    for index, column in enumerate(QCOLS):
        result[f"raw_{column}"] = raw[:, index]
    result[QCOLS] = monotonic_correct(raw)
    result["y_true_kw"] = result.load_mean_kw
    return result


def residual_quantiles(residuals: np.ndarray) -> dict[str, float]:
    return {column: quantile_higher(residuals, float(q)) for column, q in zip(QCOLS, QUANTILES, strict=True)}


def empirical_prediction(base: pd.DataFrame, center: pd.Series, residuals: dict[str, float],
                         model_name: str, split: str) -> pd.DataFrame:
    result = base.copy()
    for column in QCOLS:
        result[column] = center.to_numpy() + residuals[column]
        result[f"raw_{column}"] = result[column]
    result["y_true_kw"] = result.load_mean_kw
    result["model_name"] = model_name; result["calibration_state"] = "empirical"
    result["split"] = split
    return result


def add_interval_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for label, (lower, upper, _) in INTERVALS.items():
        result[f"interval_width_{label}"] = result[upper] - result[lower]
    return result


def scored_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.target_quality_primary & frame.y_true_kw.notna() & frame.q50.notna()].copy()


def overall_metrics(frame: pd.DataFrame) -> dict:
    data = scored_rows(frame); y = data.y_true_kw.to_numpy()
    quantile_losses = [pinball_values(y, data[column].to_numpy(), q).mean()
                       for column, q in zip(QCOLS, QUANTILES, strict=True)]
    empirical = [float((y <= data[column].to_numpy()).mean()) for column in QCOLS]
    row = {"n_predictions": int(frame.q50.notna().sum()), "n_targets_evaluated": len(data),
        "fraction_evaluated": float(len(data) / max(int(frame.target_quality_primary.sum()), 1)),
        "q50_MAE": float(np.mean(np.abs(data.q50.to_numpy() - y))),
        "q50_RMSE": float(np.sqrt(np.mean(np.square(data.q50.to_numpy() - y)))),
        "q50_bias": float(np.mean(data.q50.to_numpy() - y)),
        "mean_pinball_loss": float(np.mean(quantile_losses)),
        "mean_absolute_quantile_calibration_error": float(np.mean(np.abs(np.asarray(empirical) - QUANTILES))),
        "WIS": float(wis_values(y, data).mean()),
        "approximate_CRPS_truncated_q05_q95": float(approximate_crps_values(y, data).mean())}
    for label, (lower, upper, alpha) in INTERVALS.items():
        coverage = interval_coverage(y, data[lower].to_numpy(), data[upper].to_numpy())
        width = data[upper].to_numpy() - data[lower].to_numpy()
        row |= {f"coverage_{label}": float(coverage.mean()), f"coverage_error_{label}": float(coverage.mean()-(1-alpha)),
                f"mean_width_{label}": float(width.mean()), f"median_width_{label}": float(np.median(width)),
                f"p90_width_{label}": float(np.quantile(width, .9)),
                f"interval_score_{label}": float(interval_score_values(y, data[lower].to_numpy(),
                                                                         data[upper].to_numpy(), alpha).mean())}
    return row


def quantile_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    data = scored_rows(frame); y = data.y_true_kw.to_numpy(); rows = []
    for column, q in zip(QCOLS, QUANTILES, strict=True):
        empirical = float((y <= data[column].to_numpy()).mean())
        rows.append({"quantile": column, "nominal_probability": q,
                     "pinball_loss": float(pinball_values(y, data[column].to_numpy(), float(q)).mean()),
                     "empirical_probability": empirical, "calibration_error": empirical-q})
    return pd.DataFrame(rows)


def interval_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    data = scored_rows(frame); y = data.y_true_kw.to_numpy(); rows=[]
    for label, (lower, upper, alpha) in INTERVALS.items():
        width = data[upper].to_numpy()-data[lower].to_numpy()
        coverage = interval_coverage(y,data[lower].to_numpy(),data[upper].to_numpy()).mean()
        rows.append({"interval": label, "nominal_coverage": 1-alpha, "empirical_coverage": coverage,
            "coverage_error": coverage-(1-alpha), "mean_width": width.mean(), "median_width": np.median(width),
            "p90_width": np.quantile(width,.9), "interval_score": interval_score_values(
                y,data[lower].to_numpy(),data[upper].to_numpy(),alpha).mean()})
    return pd.DataFrame(rows)


def daily_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    data = scored_rows(frame); y=data.y_true_kw.to_numpy()
    data["q50_abs_error"] = np.abs(data.q50.to_numpy()-y)
    data["mean_pinball"] = np.column_stack([pinball_values(y,data[c].to_numpy(),q)
        for c,q in zip(QCOLS,QUANTILES,strict=True)]).mean(axis=1)
    data["WIS_row"] = wis_values(y,data)
    data["approx_CRPS_row"] = approximate_crps_values(y,data)
    for label,(lower,upper,_) in INTERVALS.items():
        data[f"covered_{label}"] = interval_coverage(y,data[lower].to_numpy(),data[upper].to_numpy()).astype(float)
        data[f"width_{label}"] = data[upper]-data[lower]
    columns={"n_targets":("y_true_kw","size"),"q50_MAE":("q50_abs_error","mean"),
             "mean_pinball_loss":("mean_pinball","mean"),"WIS":("WIS_row","mean"),
             "approximate_CRPS":("approx_CRPS_row","mean")}
    for label in INTERVALS:
        columns[f"coverage_{label}"]=(f"covered_{label}","mean")
        columns[f"width_{label}"]=(f"width_{label}","mean")
    return data.groupby("forecast_date_local").agg(**columns).reset_index()


def block_bootstrap(daily: pd.DataFrame, reps: int=BOOTSTRAP_REPS, seed: int=BOOTSTRAP_SEED) -> dict:
    values=daily[["mean_pinball_loss","coverage_90","width_90","WIS"]].to_numpy(float)
    rng=np.random.default_rng(seed); draws=rng.integers(0,len(values),size=(reps,len(values)))
    samples=values[draws].mean(axis=1); result={}
    for index,name in enumerate(["mean_pinball","coverage_90","width_90","WIS"]):
        result[f"{name}_CI_low"],result[f"{name}_CI_high"]=[float(x) for x in np.quantile(samples[:,index],[.025,.975])]
    return result


def conditional_metrics(frame: pd.DataFrame, grouping: str) -> pd.DataFrame:
    rows=[]
    for value,group in frame[frame[grouping].notna()].groupby(grouping,observed=True):
        metric=overall_metrics(group)
        rows.append({"grouping":grouping,"group":str(value)}|metric)
    return pd.DataFrame(rows)


def tail_metrics(frame: pd.DataFrame, context: str, value: str) -> dict:
    data=scored_rows(frame); y=data.y_true_kw.to_numpy()
    if not len(data):
        return {"context":context,"group":value,"n":0,"q05_lower_breach_rate":np.nan,
                "q10_lower_breach_rate":np.nan,"q90_exceedance_rate":np.nan,
                "q95_exceedance_rate":np.nan}
    return {"context":context,"group":value,"n":len(data),
        "q05_lower_breach_rate":float((y<data.q05.to_numpy()).mean()),
        "q10_lower_breach_rate":float((y<data.q10.to_numpy()).mean()),
        "q90_exceedance_rate":float((y>data.q90.to_numpy()).mean()),
        "q95_exceedance_rate":float((y>data.q95.to_numpy()).mean())}


def freeze_calibration(models: dict, iterations: dict, adjustments: dict,
                       pb0_residuals: dict, pb1_residuals: dict,
                       validation_raw: pd.DataFrame) -> dict:
    RESULTS.mkdir(parents=True,exist_ok=True)
    model_hashes={}
    for column,model in models.items():
        path=RESULTS/f"quantile_model_{column}.txt"; model.booster_.save_model(str(path)); model_hashes[column]=digest(path)
    raw_metrics=overall_metrics(validation_raw)
    calibrated_metrics=overall_metrics(apply_calibration(validation_raw,adjustments))
    frozen={"version":1,"written_before_test_scoring":True,"calibration_data":"2023 validation only",
        "test_labels_used":False,"method":"additive marginal quantile recalibration",
        "quantile_levels":QUANTILES.tolist(),"intervals":INTERVALS,"adjustments_kw":adjustments,
        "best_iterations":iterations,"model_hashes":model_hashes,"PB0_B3_train_residual_quantiles":pb0_residuals,
        "PB1_LGBM_validation_residual_quantiles":pb1_residuals,
        "validation_raw_metrics":raw_metrics,"validation_calibrated_metrics":calibrated_metrics,
        "seeds":{"model":MODEL_SEED,"bootstrap":BOOTSTRAP_SEED},"feature_matrix_sha256":digest(FEATURE_PATH),
        "deterministic_feature_spec_sha256":digest(FEATURE_SPEC),"deterministic_commit":git_commit()}
    FROZEN_PATH.write_text(json.dumps(frozen,indent=2),encoding="utf-8")
    return frozen


def build_evaluation_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame,...]:
    overall=[]; quantile=[]; interval=[]; daily=[]; lead=[]; context=[]; monthly=[]; tails=[]
    for (model,state,split),frame in predictions.groupby(["model_name","calibration_state","split"],observed=True):
        identity={"model_name":model,"calibration_state":state,"split":split}
        day=daily_metric_rows(frame); daily.append(day.assign(**identity))
        overall.append(identity|overall_metrics(frame)|block_bootstrap(day))
        quantile.append(quantile_metrics(frame).assign(**identity))
        interval.append(interval_metrics(frame).assign(**identity))
        work=frame.copy(); work["lead_time_band"]=lead_band(work.lead_time_minutes)
        work["day_type"]=np.select([work.is_public_holiday,work.is_bridge_day,work.is_day_before_holiday,
            work.is_day_after_holiday,work.is_weekend],["public_holiday","bridge_day","day_before_holiday",
            "day_after_holiday","weekend"],default="normal_weekday")
        for grouping,target in (("lead_time_band",lead),("day_type",context),("is_weekend",context),
                                ("load_magnitude",context)):
            table=conditional_metrics(work,grouping); target.append(table.assign(**identity))
        monthly.append(conditional_metrics(work,"month").assign(**identity))
        tails.append(pd.DataFrame([tail_metrics(work,"overall","all")]).assign(**identity))
        for label,mask in (("public_holiday",work.is_public_holiday),("high_load",work.load_magnitude.eq("Q4_highest")),
                           ("summer",work.month.isin([6,7,8])),("lead_18_30h",work.lead_time_minutes.between(18*60,30*60-1)),
                           ("weekend",work.is_weekend),("low_load",work.load_magnitude.eq("Q1_low"))):
            tails.append(pd.DataFrame([tail_metrics(work[mask],"subset",label)]).assign(**identity))
    return (pd.DataFrame(overall), *(pd.concat(x,ignore_index=True)
             for x in (quantile,interval,daily,lead,context,monthly,tails)))


def make_figures(predictions: pd.DataFrame, overall: pd.DataFrame, quantile: pd.DataFrame,
                 interval: pd.DataFrame, lead: pd.DataFrame, context: pd.DataFrame,
                 monthly: pd.DataFrame, importance: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True,exist_ok=True)
    q=quantile[quantile.model_name.eq("Q_LGBM")]
    fig,ax=plt.subplots(figsize=(6,6))
    for (state,split),g in q.groupby(["calibration_state","split"]):
        ax.plot(g.nominal_probability,g.empirical_probability,marker="o",label=f"{state} {split}")
    ax.plot([0,1],[0,1],"k--");ax.set(xlabel="Nominal probability",ylabel="Empirical probability",title="Quantile reliability")
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(FIGURES/"quantile_reliability.png",dpi=180);plt.close(fig)

    test=interval[interval.split.eq("test")]
    fig,ax=plt.subplots(figsize=(8,4.8))
    for (model,state),g in test.groupby(["model_name","calibration_state"]):
        ax.plot(g.nominal_coverage,g.empirical_coverage,marker="o",label=f"{model} {state}")
    ax.plot([.45,.95],[.45,.95],"k--");ax.set(xlabel="Nominal coverage",ylabel="Empirical coverage",title="TEST interval calibration")
    ax.legend(fontsize=6);fig.tight_layout();fig.savefig(FIGURES/"nominal_vs_empirical_coverage.png",dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5))
    for (model,state),g in test.groupby(["model_name","calibration_state"]):
        ax.scatter(g.mean_width,g.empirical_coverage,label=f"{model} {state}")
    ax.set(xlabel="Mean interval width (kW)",ylabel="Empirical coverage",title="Coverage versus sharpness")
    ax.legend(fontsize=6);fig.tight_layout();fig.savefig(FIGURES/"coverage_vs_width.png",dpi=180);plt.close(fig)

    chosen=predictions[(predictions.model_name.eq("Q_LGBM"))&(predictions.calibration_state.eq("calibrated"))&predictions.split.eq("test")]
    day=daily_metric_rows(chosen); normal=chosen.groupby("forecast_date_local").is_public_holiday.first()
    normal_days=day[day.forecast_date_local.isin(normal[~normal].index)]
    typical=normal_days.loc[(normal_days.WIS-normal_days.WIS.median()).abs().idxmin(),"forecast_date_local"]
    high=chosen.groupby("forecast_date_local").y_true_kw.mean().idxmax()
    holiday_days=day[day.forecast_date_local.isin(normal[normal].index)]; holiday=holiday_days.sort_values("WIS",ascending=False).iloc[0].forecast_date_local
    worst=day.sort_values("WIS",ascending=False).iloc[0].forecast_date_local
    days=[typical,high,holiday,worst];labels=["typical","high-load","holiday","worst WIS"]
    fig,axes=plt.subplots(4,1,figsize=(10,12))
    for ax,target,label in zip(axes,days,labels,strict=True):
        g=chosen[chosen.forecast_date_local.eq(target)];x=g.target_time_local.dt.tz_localize(None)
        ax.fill_between(x,g.q05,g.q95,alpha=.15,label="90%")
        ax.fill_between(x,g.q10,g.q90,alpha=.2,label="80%")
        ax.fill_between(x,g.q25,g.q75,alpha=.3,label="50%")
        ax.plot(x,g.q50,label="q50");ax.plot(x,g.y_true_kw,color="black",label="actual")
        ax.set(title=f"{label}: {pd.Timestamp(target).date()}",ylabel="kW")
    axes[0].legend(ncol=5,fontsize=7);fig.tight_layout();fig.savefig(FIGURES/"representative_forecast_fans.png",dpi=180);plt.close(fig)

    l=lead[(lead.model_name.eq("Q_LGBM"))&(lead.split.eq("test"))]
    fig,ax=plt.subplots(figsize=(8,4.8))
    for state,g in l.groupby("calibration_state"):
        ax.plot(g.group,g.mean_width_90,marker="o",label=state)
    ax.set(xlabel="Lead time",ylabel="90% mean width (kW)",title="Interval width by horizon");ax.legend()
    fig.tight_layout();fig.savefig(FIGURES/"interval_width_by_lead.png",dpi=180);plt.close(fig)

    m=monthly[(monthly.model_name.eq("Q_LGBM"))&(monthly.calibration_state.eq("calibrated"))&(monthly.split.eq("test"))]
    fig,ax=plt.subplots(figsize=(9,4.8));ax.plot(m.group.astype(int),m.coverage_90,marker="o",label="90% coverage")
    ax.plot(m.group.astype(int),m.q50_bias,marker="s",label="q50 bias (kW)");ax.axhline(.9,color="gray",ls="--")
    ax.set(xlabel="Month",title="Monthly 2024 calibration and bias");ax.legend();fig.tight_layout()
    fig.savefig(FIGURES/"monthly_2024_calibration.png",dpi=180);plt.close(fig)

    h=context[(context.model_name.eq("Q_LGBM"))&(context.calibration_state.eq("calibrated"))&
              context.split.eq("test")&context.grouping.eq("load_magnitude")]
    fig,ax=plt.subplots(figsize=(8,4.8));ax.bar(h.group,h.coverage_90);ax.axhline(.9,color="black",ls="--")
    ax.set(ylabel="90% coverage",title="High-load upper-tail calibration");fig.tight_layout()
    fig.savefig(FIGURES/"high_load_calibration.png",dpi=180);plt.close(fig)

    h=context[(context.model_name.eq("Q_LGBM"))&(context.calibration_state.eq("calibrated"))&
              context.split.eq("test")&context.grouping.eq("day_type")]
    fig,ax=plt.subplots(figsize=(9,4.8));ax.bar(h.group,h.mean_width_90);ax.set(ylabel="90% mean width (kW)",title="Holiday-context uncertainty")
    ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(FIGURES/"holiday_interval_width.png",dpi=180);plt.close(fig)

    top=importance.groupby(["quantile","feature"]).gain_importance.sum().reset_index()
    chosen_features=top.groupby("feature").gain_importance.sum().nlargest(12).index
    pivot=top[top.feature.isin(chosen_features)].pivot(index="feature",columns="quantile",values="gain_importance").fillna(0)
    pivot.plot.barh(figsize=(10,7));plt.title("Quantile feature importance");plt.tight_layout()
    plt.savefig(FIGURES/"quantile_feature_importance.png",dpi=180);plt.close()


def write_report(frozen: dict, crossing: dict, overall: pd.DataFrame, quantile: pd.DataFrame,
                 interval: pd.DataFrame, lead: pd.DataFrame, context: pd.DataFrame,
                 monthly: pd.DataFrame, tails: pd.DataFrame, importance: pd.DataFrame,
                 predictions: pd.DataFrame, category: str) -> None:
    test=overall[overall.split.eq("test")]
    qtest=quantile[quantile.split.eq("test")]
    itest=interval[interval.split.eq("test")]
    qlead=lead[(lead.model_name.eq("Q_LGBM"))&lead.split.eq("test")]
    holiday=context[(context.model_name.eq("Q_LGBM"))&context.split.eq("test")&context.grouping.eq("day_type")]
    weekend=context[(context.model_name.eq("Q_LGBM"))&context.split.eq("test")&context.grouping.eq("is_weekend")]
    magnitude=context[(context.model_name.eq("Q_LGBM"))&context.split.eq("test")&context.grouping.eq("load_magnitude")]
    months=monthly[(monthly.model_name.eq("Q_LGBM"))&monthly.split.eq("test")]
    tail=tails[(tails.model_name.eq("Q_LGBM"))&tails.split.eq("test")]
    text=f"""# Probabilistic leakage-safe LightGBM

## Research question

Can leakage-safe probabilistic LightGBM produce calibrated and sharp day-ahead prediction intervals under temporal regime change?

## Operational forecasting protocol

The fixed protocol issues at 12:00 Europe/Berlin on D-1 for every physical quarter-hour of local day D. Only eligible target measurements are scored. The deterministic point-in-time feature list is reused unchanged.

## Probabilistic model

Seven independent LightGBM quantile regressors use the deterministic structure with quantile objectives. They are trained only on 2021-2022. Unlike deterministic LightGBM, they are not refit on 2023 because that year remains the calibration set.

## Quantile levels

`{', '.join(f'{q:.2f}' for q in QUANTILES)}` define 50%, 80%, and 90% central intervals.

## Training and calibration design

2023 supplies early stopping, raw validation evaluation, and one frozen additive marginal recalibration: for each q, the q-quantile of validation `y - predicted_q` is added to future predictions. Calibration parameters are written before TEST. PB0 uses TRAIN B3 residual quantiles; PB1 uses out-of-sample 2023 deterministic-LightGBM residual quantiles.

## Leakage controls

No target value/quality, target-time observed weather, future PV, price, post-issuance load, or TEST label enters predictors or calibration. TEST rows are rejected by calibration/development APIs.

## Raw quantile validation

Best iterations by quantile: `{json.dumps(frozen['best_iterations'])}`.

## Quantile crossing

Raw validation crossing rate is {crossing['validation']['crossing_rate']:.4%}; raw TEST crossing rate is {crossing['test']['crossing_rate']:.4%}. Raw predictions are retained. Per-row sorting supplies monotonic quantiles for scoring before and after calibration. On TEST, sorting changes mean pinball by {crossing['test']['sorting_impact']['pinball_change_sorted_minus_raw']:.6f} and mean absolute calibration error by {crossing['test']['sorting_impact']['calibration_error_change_sorted_minus_raw']:.6f}; therefore the impact is reported rather than hidden.

## Calibration method

The frozen adjustments are `{json.dumps(frozen['adjustments_kw'])}` kW. This is marginal calibration, not conditional or online TEST recalibration.

## Validation calibration

Raw and calibrated validation metrics are stored in the frozen configuration and result tables.

## Frozen probabilistic configuration

`config/probabilistic_calibration_frozen.json` binds model hashes, adjustments, empirical residual quantiles, seeds, feature hash, and validation metrics before TEST access.

## 2024 TEST results

{test.to_markdown(index=False,floatfmt='.4f')}

## Probabilistic baseline comparison

{itest.to_markdown(index=False,floatfmt='.4f')}

## Reliability

{qtest.to_markdown(index=False,floatfmt='.4f')}

Mean absolute quantile calibration error is the mean over the seven levels of `|P(y <= q_hat) - q|`.

## Sharpness and proper scoring rules

Interval width is interpreted jointly with coverage. WIS uses median weight 0.5 and interval weights alpha/2, divided by K+0.5. Approximate CRPS is twice the trapezoidal integral of pinball loss over the available q05-q95 grid and is explicitly truncated/approximate.

## Forecast-horizon behavior

{qlead.to_markdown(index=False,floatfmt='.4f')}

## Holiday uncertainty

{holiday.to_markdown(index=False,floatfmt='.4f')}

## Weekday/weekend uncertainty

{weekend.to_markdown(index=False,floatfmt='.4f')}

## High-load and upper-tail behavior

{magnitude.to_markdown(index=False,floatfmt='.4f')}

{tail.to_markdown(index=False,floatfmt='.4f')}

## Summer 2024 regime shift and monthly calibration

{months.to_markdown(index=False,floatfmt='.4f')}

Any June-August undercoverage is evidence of frozen temporal calibration drift, not a trigger for TEST recalibration.

## Quantile feature importance

{importance[importance['quantile'].isin(['q10','q50','q90'])].groupby('quantile').head(10).to_markdown(index=False,floatfmt='.3f')}

## Representative forecast distributions

Fans use deterministic choices: median-WIS normal day, highest-mean-load day, worst holiday WIS day, and worst overall WIS day. No operational cause is inferred.

## Limitations

Calibration is marginal and based on one year. Regime shift can invalidate frozen coverage, context groups are uneven, and the discrete quantile grid does not define a full continuous distribution or exact CRPS/PIT.

## Implications for regime-aware forecasting

Decision: **{category}**. No neural, foundation, or probabilistic model beyond the stated LightGBM/empirical methods was trained.
"""
    REPORT.write_text(text,encoding="utf-8")


def main() -> None:
    validate_quantiles();RESULTS.mkdir(parents=True,exist_ok=True)
    all_features=pd.read_parquet(FEATURE_PATH);all_features["split"]=all_features.split.astype(str)
    features,det_frozen=safe_features()
    development=all_features[all_features.split.isin(["train","validation"])].copy()
    models,validation_raw,iterations=train_quantile_models(development,features)
    raw_validation_array=validation_raw[[f"raw_{c}" for c in QCOLS]].to_numpy()
    validation_crossing=crossing_diagnostics(raw_validation_array)
    adjustments=calibration_adjustments(validation_raw)

    train=development[development.split.eq("train")&development.target_quality_primary]
    pb0_mask=train.baseline_B3_kw.notna();pb0_residuals=residual_quantiles(
        (train.loc[pb0_mask,"load_mean_kw"]-train.loc[pb0_mask,"baseline_B3_kw"]).to_numpy())
    deterministic=pd.read_parquet(DETERMINISTIC_PREDICTIONS)
    det_validation=deterministic[(deterministic.model_name.eq("LGBM_DIRECT"))&deterministic.split.eq("validation")]
    pb1_residuals=residual_quantiles((det_validation.y_true_kw-det_validation.y_pred_kw).to_numpy())
    frozen=freeze_calibration(models,iterations,adjustments,pb0_residuals,pb1_residuals,validation_raw)
    if not FROZEN_PATH.exists() or not frozen["written_before_test_scoring"]:
        raise RuntimeError("Calibration must be frozen before TEST")

    # TEST access begins only after frozen calibration exists.
    test=all_features[all_features.split.eq("test")].copy()
    test_raw=predict_quantiles(models,test,features)
    test_crossing=crossing_diagnostics(test_raw[[f"raw_{c}" for c in QCOLS]].to_numpy())
    validation_crossing["sorting_impact"]=sorting_impact(validation_raw)
    test_crossing["sorting_impact"]=sorting_impact(test_raw)
    validation_raw["model_name"]="Q_LGBM";validation_raw["calibration_state"]="raw_monotonic"
    test_raw["model_name"]="Q_LGBM";test_raw["calibration_state"]="raw_monotonic"
    validation_calibrated=apply_calibration(validation_raw,adjustments)
    validation_calibrated["calibration_state"]="calibrated"
    test_calibrated=apply_calibration(test_raw,adjustments);test_calibrated["calibration_state"]="calibrated"

    pb0_validation=empirical_prediction(development[development.split.eq("validation")],
        development.loc[development.split.eq("validation"),"baseline_B3_kw"],pb0_residuals,"PB0_B3_empirical","validation")
    pb0_test=empirical_prediction(test,test.baseline_B3_kw,pb0_residuals,"PB0_B3_empirical","test")
    det_test=deterministic[(deterministic.model_name.eq("LGBM_DIRECT"))&deterministic.split.eq("test")][
        ["target_time_utc","y_pred_kw"]].rename(columns={"y_pred_kw":"deterministic_center"})
    test_pb1_base=test.merge(det_test,on="target_time_utc",how="left",validate="one_to_one")
    pb1_test=empirical_prediction(test_pb1_base,test_pb1_base.deterministic_center,pb1_residuals,
                                  "PB1_LGBM_empirical","test")

    frames=[validation_raw,validation_calibrated,test_raw,test_calibrated,pb0_validation,pb0_test,pb1_test]
    predictions=pd.concat([add_interval_fields(frame) for frame in frames],ignore_index=True)
    train_edges=np.unique(np.quantile(train.load_mean_kw,[0,.25,.5,.75,1]))
    predictions["load_magnitude"]=pd.cut(predictions.y_true_kw,train_edges,include_lowest=True,
        labels=["Q1_low","Q2_medium","Q3_high","Q4_highest"]).astype(object).fillna("outside_train_range")
    keep=["model_name","calibration_state","split","forecast_date_local","forecast_issue_time_local",
        "forecast_issue_time_utc","target_time_local","target_time_utc","lead_time_minutes","y_true_kw",
        *[f"raw_{c}" for c in QCOLS],*QCOLS,"interval_width_50","interval_width_80","interval_width_90",
        "is_public_holiday","is_weekend","is_day_before_holiday","is_day_after_holiday","is_bridge_day",
        "segment_id_primary","load_coverage","target_quality_primary","month","quarter_hour_of_day","load_magnitude"]
    predictions[keep].to_parquet(RESULTS/"probabilistic_predictions.parquet",index=False,compression="zstd")

    overall,quantile,interval,daily,lead,context,monthly,tails=build_evaluation_tables(predictions)
    overall.to_csv(RESULTS/"probabilistic_metrics.csv",index=False)
    quantile.to_csv(RESULTS/"probabilistic_quantile_metrics.csv",index=False)
    interval.to_csv(RESULTS/"probabilistic_interval_metrics.csv",index=False)
    daily.to_csv(RESULTS/"probabilistic_daily_metrics.csv",index=False)
    lead.to_csv(RESULTS/"probabilistic_lead_time_metrics.csv",index=False)
    context.to_csv(RESULTS/"probabilistic_context_metrics.csv",index=False)
    monthly.to_csv(RESULTS/"probabilistic_monthly_metrics.csv",index=False)
    tails.to_csv(RESULTS/"probabilistic_tail_metrics.csv",index=False)
    crossing={"validation":validation_crossing,"test":test_crossing}
    (RESULTS/"quantile_crossing.json").write_text(json.dumps(crossing,indent=2),encoding="utf-8")

    importance=[]
    for column in ("q10","q50","q90"):
        model=models[column]
        importance.append(pd.DataFrame({"quantile":column,"feature":features,
            "gain_importance":model.booster_.feature_importance(importance_type="gain"),
            "split_importance":model.booster_.feature_importance(importance_type="split")}))
    importance=pd.concat(importance).sort_values(["quantile","gain_importance"],ascending=[True,False])
    importance.to_csv(RESULTS/"probabilistic_feature_importance.csv",index=False)

    qtest=overall[(overall.model_name.eq("Q_LGBM"))&overall.split.eq("test")]
    raw=qtest[qtest.calibration_state.eq("raw_monotonic")].iloc[0]
    calibrated=qtest[qtest.calibration_state.eq("calibrated")].iloc[0]
    pb0=overall[(overall.model_name.eq("PB0_B3_empirical"))&overall.split.eq("test")].iloc[0]
    pb1=overall[(overall.model_name.eq("PB1_LGBM_empirical"))&overall.split.eq("test")].iloc[0]
    raw_cal_error=raw.mean_absolute_quantile_calibration_error
    cal_cal_error=calibrated.mean_absolute_quantile_calibration_error
    holiday_row=context[(context.model_name.eq("Q_LGBM"))&context.calibration_state.eq("calibrated")&
                        context.split.eq("test")&context.grouping.eq("day_type")&
                        context.group.eq("public_holiday")].iloc[0]
    month_rows=monthly[(monthly.model_name.eq("Q_LGBM"))&monthly.calibration_state.eq("calibrated")&
                       monthly.split.eq("test")&monthly.n_targets_evaluated.ge(500)]
    context_miscalibration=(holiday_row.coverage_90 < .85 or
                            (month_rows.coverage_90-.90).abs().max() > .08)
    if (cal_cal_error<.03 and calibrated.WIS<min(pb0.WIS,pb1.WIS)*.98 and
            abs(calibrated.coverage_90-.9)<.03 and not context_miscalibration):
        category="A. Well calibrated and meaningfully sharper/better than empirical probabilistic baselines"
    elif calibrated.WIS<max(pb0.WIS,pb1.WIS) or cal_cal_error<raw_cal_error or context_miscalibration:
        category="B. Useful but requiring calibration or showing context-dependent miscalibration"
    else:
        category="C. Poorly calibrated / no meaningful advantage over simple empirical uncertainty baselines"
    make_figures(predictions,overall,quantile,interval,lead,context,monthly,importance)
    write_report(frozen,crossing,overall,quantile,interval,lead,context,monthly,tails,importance,predictions,category)

    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"python_version":platform.python_version(),
        "lightgbm_version":lgb.__version__,"numpy_version":np.__version__,"git_commit":git_commit(),
        "gold_sha256":json.loads((ROOT/"reports"/"gold_manifest.json").read_text())["artifacts"]["load_15min_base"]["output_sha256"],
        "feature_matrix_sha256":digest(FEATURE_PATH),"deterministic_feature_spec_sha256":digest(FEATURE_SPEC),
        "quantile_source_sha256":digest(Path(__file__)),"calibration_sha256":digest(FROZEN_PATH),
        "model_hashes":frozen["model_hashes"],"seeds":frozen["seeds"],"decision_category":category,
        "outputs":{str(path.relative_to(RESULTS)):{"bytes":path.stat().st_size,"sha256":digest(path)}
                   for path in sorted(RESULTS.rglob("*")) if path.is_file() and path.name!="probabilistic_run_manifest.json"}}
    (RESULTS/"probabilistic_run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"category":category,"crossing":crossing,"test":qtest.to_dict("records")},indent=2,default=str))


if __name__=="__main__":
    main()
