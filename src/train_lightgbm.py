"""Deterministic LightGBM development, freeze, pre-test refit, and evaluation."""
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

from day_ahead_baselines import (ROOT, SEED, day_block_bootstrap, daily_metrics,
                                 lead_band, metric_values)

FEATURE_PATH = ROOT / "data" / "features" / "lightgbm_features.parquet"
FEATURE_SPEC = ROOT / "config" / "lightgbm_features.json"
FROZEN_PATH = ROOT / "config" / "lightgbm_frozen.json"
BASELINE_PREDICTIONS = ROOT / "results" / "baselines" / "day_ahead_baseline_predictions.parquet"
RESULTS = ROOT / "results" / "lightgbm"
FIGURES = RESULTS / "figures"
REPORT = ROOT / "reports" / "deterministic_lightgbm.md"
BOOTSTRAP_REPS = 2000
MODEL_SEED = 20241028


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


def load_inputs() -> tuple[pd.DataFrame, dict, float]:
    frame = pd.read_parquet(FEATURE_PATH)
    frame["split"] = frame.split.astype(str)
    spec = json.loads(FEATURE_SPEC.read_text(encoding="utf-8"))
    train_scale = float(frame.loc[frame.split.eq("train") & frame.target_quality_primary,
                                  "load_mean_kw"].abs().mean())
    return frame, spec, train_scale


def group_features(spec: dict, groups: list[str]) -> list[str]:
    return list(dict.fromkeys(feature for group in groups for feature in spec["groups"][group]))


def assert_predictors_safe(features: list[str], spec: dict) -> None:
    prohibited = set(spec["classification"]["prohibited"])
    forbidden_tokens = ("load_coverage", "target_quality", "observed_", "pv_", "price_")
    overlap = prohibited.intersection(features)
    named = [name for name in features if any(token in name for token in forbidden_tokens)]
    if overlap or named or "load_mean_kw" in features:
        raise RuntimeError(f"Unsafe model predictors: {sorted(overlap.union(named))}")


def base_parameters() -> dict:
    return {"objective": "regression_l1", "verbosity": -1, "n_estimators": 1800,
            "random_state": MODEL_SEED, "deterministic": True, "force_col_wise": True,
            "n_jobs": -1, "max_bin": 255}


def fit_development(development: pd.DataFrame, features: list[str], formulation: str,
                    parameters: dict) -> tuple[lgb.LGBMRegressor, pd.DataFrame, dict]:
    if set(development.split) - {"train", "validation"}:
        raise RuntimeError("TEST rows supplied to development fitting")
    train = development[development.split.eq("train") & development.target_quality_primary].copy()
    validation = development[development.split.eq("validation") & development.target_quality_primary].copy()
    if formulation == "residual_B3":
        train = train[train.baseline_B3_kw.notna()]
        validation = validation[validation.baseline_B3_kw.notna()]
        y_train = train.load_mean_kw - train.baseline_B3_kw
        y_validation = validation.load_mean_kw - validation.baseline_B3_kw
    elif formulation == "direct":
        y_train, y_validation = train.load_mean_kw, validation.load_mean_kw
    else:
        raise ValueError(formulation)
    model = lgb.LGBMRegressor(**base_parameters(), **parameters)
    model.fit(train[features], y_train, eval_set=[(train[features], y_train),
        (validation[features], y_validation)], eval_names=["train", "validation"], eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    train_raw = model.predict(train[features], num_iteration=model.best_iteration_)
    validation_raw = model.predict(validation[features], num_iteration=model.best_iteration_)
    if formulation == "residual_B3":
        train_prediction = train.baseline_B3_kw.to_numpy() + train_raw
        validation_prediction = validation.baseline_B3_kw.to_numpy() + validation_raw
    else:
        train_prediction, validation_prediction = train_raw, validation_raw
    predicted = validation.copy()
    predicted["y_true_kw"] = predicted.load_mean_kw
    predicted["y_pred_kw"] = validation_prediction
    predicted["model_name"] = "development_candidate"
    error = validation_prediction - validation.load_mean_kw.to_numpy()
    validation_daily = pd.DataFrame({"date": validation.forecast_date_local,
                                     "absolute_error": np.abs(error)}).groupby("date").absolute_error.mean()
    metrics = metric_values(predicted, float(development.loc[development.split.eq("train") &
        development.target_quality_primary, "load_mean_kw"].abs().mean()))
    metrics |= {"training_examples": len(train), "validation_examples": len(validation),
                "training_MAE": float(np.mean(np.abs(train_prediction - train.load_mean_kw.to_numpy()))),
                "validation_mean_daily_MAE": float(validation_daily.mean()),
                "validation_median_daily_MAE": float(validation_daily.median()),
                "validation_holiday_MAE": float(np.mean(np.abs(error[validation.is_public_holiday.to_numpy()])))
                    if validation.is_public_holiday.any() else np.nan,
                "best_iteration": int(model.best_iteration_)}
    return model, predicted, metrics


def ablation_and_search(development: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict, lgb.LGBMRegressor, pd.DataFrame]:
    logs = []
    default = {"num_leaves": 31, "learning_rate": .05, "min_child_samples": 100,
               "feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 0,
               "reg_alpha": .1, "reg_lambda": .1}
    for name, groups in spec["ablations"].items():
        features = group_features(spec, groups); assert_predictors_safe(features, spec)
        _, _, metrics = fit_development(development, features, "direct", default)
        logs.append({"stage": "ablation", "candidate": name, "formulation": "direct",
                     "feature_groups": "+".join(groups), "parameters": json.dumps(default, sort_keys=True)} | metrics)

    features = group_features(spec, spec["ablations"]["D_plus_holidays"])
    candidates = [
        {"num_leaves": 15, "learning_rate": .05, "min_child_samples": 100, "feature_fraction": 1.0,
         "bagging_fraction": 1.0, "bagging_freq": 0, "reg_alpha": .0, "reg_lambda": .1},
        {"num_leaves": 31, "learning_rate": .05, "min_child_samples": 100, "feature_fraction": .8,
         "bagging_fraction": 1.0, "bagging_freq": 0, "reg_alpha": .1, "reg_lambda": .1},
        {"num_leaves": 31, "learning_rate": .02, "min_child_samples": 50, "feature_fraction": 1.0,
         "bagging_fraction": .8, "bagging_freq": 1, "reg_alpha": .1, "reg_lambda": 1.0},
        {"num_leaves": 63, "learning_rate": .02, "min_child_samples": 250, "feature_fraction": .8,
         "bagging_fraction": .8, "bagging_freq": 1, "reg_alpha": 1.0, "reg_lambda": 1.0},
    ]
    fitted = {}
    for formulation in ("direct", "residual_B3"):
        for index, parameters in enumerate(candidates, 1):
            model, prediction, metrics = fit_development(development, features, formulation, parameters)
            name = f"{formulation}_search_{index}"
            logs.append({"stage": "search", "candidate": name, "formulation": formulation,
                         "feature_groups": "D_plus_holidays", "parameters": json.dumps(parameters, sort_keys=True)} | metrics)
            fitted[name] = (model, prediction, parameters, metrics)
    log = pd.DataFrame(logs)
    # Primary validation mean daily MAE; require no material aggregate/holiday instability.
    search = log[log.stage.eq("search")].sort_values(
        ["validation_mean_daily_MAE", "MAE", "validation_holiday_MAE", "candidate"])
    selected_name = search.iloc[0].candidate
    model, prediction, parameters, metrics = fitted[selected_name]
    selection = {"candidate": selected_name, "formulation": search.iloc[0].formulation,
                 "parameters": parameters, "best_iteration": int(metrics["best_iteration"]),
                 "validation_metrics": {key: float(metrics[key]) for key in
                     ("MAE", "RMSE", "bias", "WAPE", "normalized_MAE", "validation_mean_daily_MAE",
                      "validation_median_daily_MAE", "validation_holiday_MAE", "training_MAE")},
                 "training_examples": int(metrics["training_examples"]),
                 "validation_examples": int(metrics["validation_examples"])}
    return log, selection, model, prediction


def freeze_configuration(spec: dict, features: list[str], selection: dict) -> dict:
    frozen = {"version": 1, "written_before_test_scoring": True,
        "operational_protocol": "12:00 Europe/Berlin D-1 for complete local day D",
        "selection_data": "TRAIN 2021-2022 and VALIDATION 2023 only", "test_data_used_for_selection": False,
        "model_name": "LGBM_RESIDUAL_B3" if selection["formulation"] == "residual_B3" else "LGBM_DIRECT",
        "formulation": selection["formulation"], "objective": "regression_l1", "features": features,
        "feature_groups": spec["ablations"]["D_plus_holidays"], "hyperparameters": selection["parameters"],
        "best_iteration": selection["best_iteration"], "random_seeds": {"model": MODEL_SEED, "numpy": MODEL_SEED,
            "bootstrap": SEED, "permutation": MODEL_SEED + 1},
        "validation_selection": selection, "refit_data": "eligible 2021-2023 only",
        "feature_matrix_sha256": digest(FEATURE_PATH), "feature_spec_sha256": digest(FEATURE_SPEC),
        "baseline_checkpoint_commit": git_commit(), "unsafe_features_excluded": True}
    FROZEN_PATH.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return frozen


def refit_pretest(pretest: pd.DataFrame, features: list[str], frozen: dict) -> lgb.LGBMRegressor:
    if set(pretest.split) - {"train", "validation"}:
        raise RuntimeError("Final pre-test refit may contain only 2021-2023")
    rows = pretest[pretest.target_quality_primary].copy()
    if frozen["formulation"] == "residual_B3":
        rows = rows[rows.baseline_B3_kw.notna()]
        target = rows.load_mean_kw - rows.baseline_B3_kw
    else:
        target = rows.load_mean_kw
    parameters = dict(frozen["hyperparameters"])
    final_parameters = base_parameters() | parameters
    final_parameters["n_estimators"] = int(frozen["best_iteration"])
    model = lgb.LGBMRegressor(**final_parameters)
    model.fit(rows[features], target, callbacks=[lgb.log_evaluation(0)])
    model.pretest_training_examples_ = len(rows)
    return model


def predict_model(model: lgb.LGBMRegressor, rows: pd.DataFrame, features: list[str], frozen: dict,
                  split: str, stage: str) -> pd.DataFrame:
    data = rows[rows.split.eq(split)].copy()
    raw = model.predict(data[features], num_iteration=int(frozen["best_iteration"]))
    prediction = raw + data.baseline_B3_kw.to_numpy() if frozen["formulation"] == "residual_B3" else raw
    data["y_true_kw"] = data.load_mean_kw
    data["y_pred_kw"] = prediction
    data["model_name"] = frozen["model_name"]
    data["model_stage"] = stage
    return data


def comparator_frames(frame: pd.DataFrame, split: str) -> list[pd.DataFrame]:
    base = frame[frame.split.eq(split)].copy()
    result = []
    for model, column in (("B3_recent_week_profile", "baseline_B3_kw"),):
        item = base.copy(); item["model_name"] = model; item["model_stage"] = "frozen_baseline"
        item["y_true_kw"] = item.load_mean_kw; item["y_pred_kw"] = item[column]; result.append(item)
    baseline = pd.read_parquet(BASELINE_PREDICTIONS)
    b5 = baseline[(baseline.split.eq(split)) & baseline.model_name.eq("B5_baseline_ensemble")][
        ["target_time_utc", "y_pred_kw"]].rename(columns={"y_pred_kw": "b5_prediction"})
    item = base.merge(b5, on="target_time_utc", how="left", validate="one_to_one")
    item["model_name"] = "B5_baseline_ensemble"; item["model_stage"] = "frozen_baseline"
    item["y_true_kw"] = item.load_mean_kw; item["y_pred_kw"] = item.b5_prediction; result.append(item)
    return result


def metrics_table(predictions: pd.DataFrame, train_scale: float) -> pd.DataFrame:
    rows = []
    for (model, split), group in predictions.groupby(["model_name", "split"], observed=True):
        rows.append({"model": model, "split": split} | metric_values(group, train_scale) |
                    day_block_bootstrap(group, BOOTSTRAP_REPS, SEED))
    return pd.DataFrame(rows).sort_values(["split", "MAE"])


def paired_comparison(predictions: pd.DataFrame, learned_name: str) -> pd.DataFrame:
    rows = []
    for split in ("validation", "test"):
        subset = predictions[predictions.split.eq(split)]
        wide = subset.pivot(index="target_time_utc", columns="model_name", values="y_pred_kw")
        meta = subset[subset.model_name.eq(learned_name)].set_index("target_time_utc")
        for comparator in ("B3_recent_week_profile", "B5_baseline_ensemble"):
            common = meta.target_quality_primary & wide[learned_name].notna() & wide[comparator].notna()
            table = pd.DataFrame({"date": meta.loc[common, "forecast_date_local"],
                "learned_abs": (wide.loc[common, learned_name] - meta.loc[common, "y_true_kw"]).abs(),
                "baseline_abs": (wide.loc[common, comparator] - meta.loc[common, "y_true_kw"]).abs()})
            daily = table.groupby("date").mean(); daily["difference"] = daily.learned_abs - daily.baseline_abs
            rng = np.random.default_rng(SEED)
            values = daily.difference.to_numpy()
            draws = rng.choice(values, size=(BOOTSTRAP_REPS, len(values)), replace=True).mean(axis=1)
            rows.append({"split": split, "learned_model": learned_name, "baseline": comparator,
                "n_common_targets": int(common.sum()), "n_paired_days": len(daily),
                "learned_MAE_common": float(table.learned_abs.mean()),
                "baseline_MAE_common": float(table.baseline_abs.mean()),
                "mean_daily_MAE_difference_kw": float(values.mean()),
                "median_daily_MAE_difference_kw": float(np.median(values)),
                "difference_CI_low": float(np.quantile(draws, .025)),
                "difference_CI_high": float(np.quantile(draws, .975)),
                "fraction_days_learned_wins": float((values < 0).mean())})
    return pd.DataFrame(rows)


def grouped_metrics(predictions: pd.DataFrame, train_scale: float) -> pd.DataFrame:
    data = predictions.copy()
    data["lead_time_band"] = lead_band(data.lead_time_minutes)
    data["day_type"] = np.select([data.is_public_holiday, data.is_bridge_day,
        data.is_day_before_holiday, data.is_day_after_holiday, data.is_weekend],
        ["public_holiday", "bridge_day", "day_before_holiday", "day_after_holiday", "weekend"], default="normal_weekday")
    train_target = data[(data.split.eq("validation")) & data.target_quality_primary].y_true_kw
    # Fixed TRAIN quantiles are passed via attrs by the caller.
    edges = data.attrs["train_load_edges"]
    data["load_magnitude"] = pd.cut(data.y_true_kw, edges, include_lowest=True,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"]).astype(object)
    data["load_magnitude"] = data.load_magnitude.fillna("outside_train_range")
    rows = []
    for grouping in ("lead_time_band", "day_type", "is_weekend", "month", "quarter_hour_of_day", "load_magnitude"):
        for (model, split, value), group in data[data[grouping].notna()].groupby(
                ["model_name", "split", grouping], observed=True):
            rows.append({"model": model, "split": split, "grouping": grouping, "group": str(value)} |
                        metric_values(group, train_scale))
    return pd.DataFrame(rows)


def feature_importance(model: lgb.LGBMRegressor, development_model: lgb.LGBMRegressor,
                       validation: pd.DataFrame, features: list[str], frozen: dict) -> pd.DataFrame:
    importance = pd.DataFrame({"feature": features,
        "gain_importance": model.booster_.feature_importance(importance_type="gain"),
        "split_importance": model.booster_.feature_importance(importance_type="split")})
    eligible = validation[validation.target_quality_primary].copy()
    if frozen["formulation"] == "residual_B3":
        eligible = eligible[eligible.baseline_B3_kw.notna()]
    raw = development_model.predict(eligible[features], num_iteration=frozen["best_iteration"])
    reference = raw + eligible.baseline_B3_kw.to_numpy() if frozen["formulation"] == "residual_B3" else raw
    reference_mae = float(np.mean(np.abs(reference - eligible.load_mean_kw.to_numpy())))
    top = importance.nlargest(25, "gain_importance").feature
    rng = np.random.default_rng(MODEL_SEED + 1)
    permutation = {}
    for feature in top:
        shuffled = eligible[features].copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        raw_prediction = development_model.predict(shuffled, num_iteration=frozen["best_iteration"])
        prediction = raw_prediction + eligible.baseline_B3_kw.to_numpy() if frozen["formulation"] == "residual_B3" else raw_prediction
        permutation[feature] = float(np.mean(np.abs(prediction - eligible.load_mean_kw.to_numpy())) - reference_mae)
    importance["permutation_MAE_increase_validation"] = importance.feature.map(permutation)
    return importance.sort_values("gain_importance", ascending=False)


def make_figures(predictions: pd.DataFrame, metrics: pd.DataFrame, context: pd.DataFrame,
                 importance: pd.DataFrame, learned_name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    comparison = metrics[metrics.model.isin([learned_name, "B3_recent_week_profile", "B5_baseline_ensemble"])]
    pivot = comparison.pivot(index="model", columns="split", values="MAE")
    fig, ax = plt.subplots(figsize=(8, 4.8)); pivot.plot.bar(ax=ax, color=["#ff7f00", "#377eb8"])
    ax.set(ylabel="MAE (kW)", title="Learned model versus adaptive baselines"); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(FIGURES / "model_mae_comparison.png", dpi=180); plt.close(fig)

    lead = context[(context.grouping.eq("lead_time_band")) & context.split.eq("test") &
                   context.model.isin([learned_name, "B3_recent_week_profile", "B5_baseline_ensemble"])]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for model, group in lead.groupby("model"):
        ax.plot(group.group, group.MAE, marker="o", label=model)
    ax.set(ylabel="MAE (kW)", xlabel="Lead-time band", title="Test error by lead time"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "error_by_lead_time.png", dpi=180); plt.close(fig)

    month = context[(context.grouping.eq("month")) & context.split.eq("test") &
                    context.model.isin([learned_name, "B3_recent_week_profile", "B4_calendar_profile"])]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for model, group in month.groupby("model"):
        ax.plot(group.group.astype(int), group.bias, marker="o", label=model)
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="Month", ylabel="Bias (kW)", title="Monthly TEST bias")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES / "monthly_test_bias.png", dpi=180); plt.close(fig)

    holiday = context[(context.grouping.eq("day_type")) & context.split.eq("test") &
                      context.model.isin([learned_name, "B3_recent_week_profile", "B5_baseline_ensemble"])]
    hp = holiday.pivot(index="group", columns="model", values="MAE")
    fig, ax = plt.subplots(figsize=(10, 5)); hp.plot.bar(ax=ax)
    ax.set(ylabel="MAE (kW)", title="TEST performance by holiday context"); ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(FIGURES / "holiday_context_mae.png", dpi=180); plt.close(fig)

    top = importance.head(20).sort_values("gain_importance")
    fig, ax = plt.subplots(figsize=(8, 7)); ax.barh(top.feature, top.gain_importance, color="#377eb8")
    ax.set(xlabel="Gain importance", title="Final LightGBM feature importance")
    fig.tight_layout(); fig.savefig(FIGURES / "feature_importance.png", dpi=180); plt.close(fig)

    selected = predictions[(predictions.model_name.eq(learned_name)) & predictions.split.eq("test") &
                           predictions.target_quality_primary].copy()
    selected["abs_error"] = (selected.y_pred_kw - selected.y_true_kw).abs()
    daily = selected.groupby("forecast_date_local").abs_error.mean()
    days = [daily.index[(daily - daily.quantile(q)).abs().argmin()] for q in (.1, .5, .9)]
    wide = predictions[predictions.split.eq("test") & predictions.forecast_date_local.isin(days) &
                       predictions.model_name.isin([learned_name, "B3_recent_week_profile", "B5_baseline_ensemble"])]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    for ax, day, label in zip(axes, days, ["strong", "median", "poor"], strict=True):
        sample = wide[wide.forecast_date_local.eq(day)]
        truth = sample[sample.model_name.eq(learned_name)]
        ax.plot(truth.target_time_local.dt.tz_localize(None), truth.y_true_kw, color="black", label="actual")
        for model, group in sample.groupby("model_name"):
            ax.plot(group.target_time_local.dt.tz_localize(None), group.y_pred_kw, label=model, alpha=.8)
        ax.set(title=f"{label} learned-model day: {pd.Timestamp(day).date()}", ylabel="kW")
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGURES / "representative_test_days.png", dpi=180); plt.close(fig)

    regime = selected.dropna(subset=["recent_level_difference_7d_28d"]).copy()
    regime["correction"] = regime.y_pred_kw - regime.baseline_B3_kw
    sample = regime.sample(min(5000, len(regime)), random_state=MODEL_SEED)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(sample.recent_level_difference_7d_28d, sample.correction,
        s=5, alpha=.2); ax.set(xlabel="Recent 7d minus 28d mean (kW)", ylabel="Model correction to B3 (kW)",
        title="Recent operating level and learned correction")
    fig.tight_layout(); fig.savefig(FIGURES / "recent_regime_vs_correction.png", dpi=180); plt.close(fig)


def write_report(frozen: dict, log: pd.DataFrame, metrics: pd.DataFrame, pairwise: pd.DataFrame,
                 context: pd.DataFrame, importance: pd.DataFrame, daily: pd.DataFrame,
                 predictions: pd.DataFrame, category: str) -> None:
    learned = frozen["model_name"]
    validation = metrics[metrics.split.eq("validation")]
    test = metrics[metrics.split.eq("test")]
    ablation = log[log.stage.eq("ablation")][["candidate", "MAE", "RMSE", "bias", "WAPE",
                                                "validation_holiday_MAE", "best_iteration"]]
    lead = context[(context.split.eq("test")) & context.grouping.eq("lead_time_band") &
                   context.model.isin([learned, "B3_recent_week_profile", "B5_baseline_ensemble"])]
    holiday = context[(context.split.eq("test")) & context.grouping.eq("day_type") &
                      context.model.isin([learned, "B3_recent_week_profile", "B5_baseline_ensemble"])]
    weekday = context[(context.split.eq("test")) & context.grouping.eq("is_weekend") & context.model.eq(learned)]
    high = context[(context.split.eq("test")) & context.grouping.eq("load_magnitude") & context.group.eq("Q4_high")]
    month = context[(context.split.eq("test")) & context.grouping.eq("month") &
                    context.model.isin([learned, "B3_recent_week_profile"])]
    learned_daily = daily[(daily.split.eq("test")) & daily.model_name.eq(learned)].sort_values("MAE")
    representative = {"strong": learned_daily.iloc[int(.1 * (len(learned_daily)-1))].forecast_date_local,
                      "median": learned_daily.iloc[int(.5 * (len(learned_daily)-1))].forecast_date_local,
                      "poor": learned_daily.iloc[int(.9 * (len(learned_daily)-1))].forecast_date_local}
    holiday_days = learned_daily.merge(predictions[predictions.model_name.eq(learned)][
        ["forecast_date_local", "is_public_holiday"]].drop_duplicates(), on="forecast_date_local")
    difficult_holiday = holiday_days[holiday_days.is_public_holiday].sort_values("MAE", ascending=False).head(1)
    high_days = predictions[(predictions.model_name.eq(learned)) & predictions.split.eq("test")].groupby(
        "forecast_date_local").y_true_kw.mean().sort_values(ascending=False).head(1)
    text = f"""# Deterministic leakage-safe LightGBM

## Research question

Can a leakage-safe learned model improve day-ahead industrial load forecasts beyond strong adaptive seasonal baselines?

## Operational protocol

Issuance is 12:00 Europe/Berlin on D-1 for all physical 15-minute intervals of local day D. UTC is the unique key. Metrics use only primary-quality targets. The authoritative availability gate is reused from `day_ahead_baselines.py`, and a historical interval is usable only after its interval end.

## Point-in-time feature design

Predictors comprise deterministic local calendar/lead time, canonical holiday and bridge-day fields, safe B0-B3 baseline components, same-quarter prior-week analogs, issuance-time rolling load state, and recent operating-level summaries. NaNs are preserved for LightGBM native handling. Target coverage/quality, observed target-time weather, measured PV, and price are excluded.

## Baselines

B3 is the full-coverage adaptive four-week median. B5 is the frozen median ensemble with lower coverage. Both are evaluated on exact common targets against the learned model.

## Development procedure

TRAIN 2021-2022 was used for fitting and 2023 only for ablation, compact hyperparameter/formulation selection, and early stopping. TEST was not supplied to development APIs. Selection used mean daily validation MAE with aggregate, holiday, and stability diagnostics.

## Hyperparameter selection

The machine-readable log contains four predefined ablations and eight compact search candidates—four parameter sets for each of direct and residual-to-B3 formulations. No broad sweep was run.

## Ablation study

{ablation.to_markdown(index=False, floatfmt='.4f')}

## Validation results

{validation.to_markdown(index=False, floatfmt='.4f')}

## Frozen model

Selected `{learned}` with `{frozen['formulation']}` formulation, `{frozen['objective']}` objective, and {frozen['best_iteration']} boosting iterations. The frozen configuration was written before TEST scoring.

## Final 2024 TEST results

{test.to_markdown(index=False, floatfmt='.4f')}

The final model was refit once on eligible 2021-2023 examples using the frozen feature list, parameters, objective, and boosting iteration count.

## B3/B5 comparison

{pairwise.to_markdown(index=False, floatfmt='.4f')}

Negative paired differences favor LightGBM. Confidence intervals resample whole target days.

## Lead-time analysis

{lead.to_markdown(index=False, floatfmt='.4f')}

## Holiday analysis

{holiday.to_markdown(index=False, floatfmt='.4f')}

Holiday categories are deterministic. Bridge day means a weekday between a public holiday and an adjacent weekend day. Small categories should not be overinterpreted.

## Distribution-shift analysis

{month.to_markdown(index=False, floatfmt='.4f')}

The 2024 regime increase is retained. Recent-level features allow adaptation without target-time information; monthly bias is reported rather than normalized away.

## High-load performance

{high.to_markdown(index=False, floatfmt='.4f')}

Magnitude bins use TRAIN-only quartiles. Weekday/weekend results:

{weekday.to_markdown(index=False, floatfmt='.4f')}

## Feature importance

{importance.head(20).to_markdown(index=False, floatfmt='.4f')}

Gain/split importance comes from the final refit. Permutation MAE increases use the untouched development validation predictions and do not drive selection.

## Failure cases

Deterministic representative learned-model dates are {', '.join(f'{key}: {pd.Timestamp(value).date()}' for key, value in representative.items())}. The difficult holiday is {pd.Timestamp(difficult_holiday.iloc[0].forecast_date_local).date() if len(difficult_holiday) else 'unavailable'}, and the highest-mean-load day is {pd.Timestamp(high_days.index[0]).date()}. Errors are described by calendar and load context only; no undocumented operational cause is inferred.

## Limitations

No point-in-time weather forecast or verified-at-noon price is available. Ground-truth outages reduce evaluable targets. Holiday counts are small, and this is one industrial facility. Feature importance is associative, not causal.

## Implications for probabilistic forecasting

Decision category: **{category}**. This classification uses aggregate/common-target metrics, paired day bootstrap intervals, holiday and horizon behavior, and prediction coverage. No probabilistic model or uncertainty forecast was trained.
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    frame, spec, train_scale = load_inputs()
    development = frame[frame.split.isin(["train", "validation"])].copy()
    log, selection, development_model, validation_prediction = ablation_and_search(development, spec)
    log.to_csv(RESULTS / "development_experiment_log.csv", index=False)
    features = group_features(spec, spec["ablations"]["D_plus_holidays"])
    assert_predictors_safe(features, spec)
    frozen = freeze_configuration(spec, features, selection)

    # TEST is accessed only after the frozen file exists on disk.
    if not FROZEN_PATH.exists() or not frozen["written_before_test_scoring"]:
        raise RuntimeError("Frozen configuration must exist before TEST scoring")
    final_model = refit_pretest(development, features, frozen)
    test_prediction = predict_model(final_model, frame, features, frozen, "test", "final_pretest_refit")
    validation_prediction["model_name"] = frozen["model_name"]
    validation_prediction["model_stage"] = "development_train_2021_2022"
    selected_columns = ["model_name", "model_stage", "split", "forecast_date_local",
        "forecast_issue_time_local", "forecast_issue_time_utc", "target_time_local", "target_time_utc",
        "lead_time_minutes", "y_true_kw", "y_pred_kw", "target_quality_primary", "load_coverage",
        "segment_id_primary", "is_weekend", "is_public_holiday", "is_day_before_holiday",
        "is_day_after_holiday", "is_bridge_day", "month", "quarter_hour_of_day", "baseline_B3_kw",
        "recent_level_difference_7d_28d"]
    predictions = pd.concat([validation_prediction[selected_columns], test_prediction[selected_columns],
        *[item[selected_columns] for split in ("validation", "test") for item in comparator_frames(frame, split)]],
        ignore_index=True).sort_values(["split", "model_name", "target_time_utc"])
    predictions["error_kw"] = predictions.y_pred_kw - predictions.y_true_kw
    predictions["absolute_error_kw"] = predictions.error_kw.abs()
    predictions.to_parquet(RESULTS / "lightgbm_predictions.parquet", index=False, compression="zstd")

    metrics = metrics_table(predictions, train_scale); metrics.to_csv(RESULTS / "lightgbm_metrics.csv", index=False)
    daily = daily_metrics(predictions); daily.to_csv(RESULTS / "lightgbm_daily_metrics.csv", index=False)
    pairwise = paired_comparison(predictions, frozen["model_name"])
    pairwise.to_csv(RESULTS / "lightgbm_pairwise_baseline.csv", index=False)
    predictions.attrs["train_load_edges"] = np.unique(np.quantile(
        frame.loc[frame.split.eq("train") & frame.target_quality_primary, "load_mean_kw"], [0, .25, .5, .75, 1]))
    context = grouped_metrics(predictions, train_scale); context.to_csv(RESULTS / "lightgbm_context_metrics.csv", index=False)
    context[context.grouping.eq("lead_time_band")].to_csv(RESULTS / "lightgbm_lead_time_metrics.csv", index=False)
    log[log.stage.eq("ablation")].to_csv(RESULTS / "lightgbm_ablation_validation.csv", index=False)
    importance = feature_importance(final_model, development_model,
        frame[frame.split.eq("validation")], features, frozen)
    importance.to_csv(RESULTS / "lightgbm_feature_importance.csv", index=False)
    final_model.booster_.save_model(str(RESULTS / "lightgbm_model.txt"))
    development_model.booster_.save_model(str(RESULTS / "lightgbm_development_model.txt"))

    test_pair = pairwise[pairwise.split.eq("test")]
    b3 = test_pair[test_pair.baseline.eq("B3_recent_week_profile")].iloc[0]
    b5 = test_pair[test_pair.baseline.eq("B5_baseline_ensemble")].iloc[0]
    learned_test = metrics[(metrics.split.eq("test")) & metrics.model.eq(frozen["model_name"])].iloc[0]
    if b3.mean_daily_MAE_difference_kw < -.25 and b3.difference_CI_high < 0 and learned_test.fraction_evaluated >= .99:
        category = "A. Clear meaningful improvement over adaptive baselines"
    elif b3.mean_daily_MAE_difference_kw < 0 or b5.mean_daily_MAE_difference_kw < 0:
        category = "B. Small/inconsistent improvement"
    else:
        category = "C. No improvement over adaptive baselines"
    make_figures(predictions, metrics, context, importance, frozen["model_name"])
    write_report(frozen, log, metrics, pairwise, context, importance, daily, predictions, category)

    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "python_version": platform.python_version(), "lightgbm_version": lgb.__version__,
        "feature_matrix_sha256": digest(FEATURE_PATH), "feature_spec_sha256": digest(FEATURE_SPEC),
        "frozen_config_sha256": digest(FROZEN_PATH), "source_code_sha256": digest(Path(__file__)),
        "pretest_training_examples": int(final_model.pretest_training_examples_), "decision_category": category,
        "outputs": {str(path.relative_to(RESULTS)): {"bytes": path.stat().st_size, "sha256": digest(path)}
                    for path in sorted(RESULTS.rglob("*")) if path.is_file() and
                    path.name != "lightgbm_run_manifest.json"}}
    (RESULTS / "lightgbm_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"selection": selection, "category": category,
                      "pretest_training_examples": final_model.pretest_training_examples_}, indent=2))


if __name__ == "__main__":
    main()
