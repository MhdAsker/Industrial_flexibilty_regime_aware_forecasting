"""Leakage-safe deterministic day-ahead baselines for the SPARK GOLD panel.

Protocol: issue at 12:00 Europe/Berlin on D-1 and forecast every physical
quarter-hour in local calendar day D. Canonical comparisons use UTC. A load
interval is available only after its 15-minute interval has ended.
"""
from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold" / "load_15min_base.parquet"
CONFIG_PATH = ROOT / "config" / "baseline_experiment.json"
RESULTS = ROOT / "results" / "baselines"
FIGURES = RESULTS / "figures"
REPORT = ROOT / "reports" / "day_ahead_baselines.md"
TZ = "Europe/Berlin"
INTERVAL = pd.Timedelta(minutes=15)
SEED = 20241027
BOOTSTRAP_REPS = 2000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def issue_time_for_target_date(target_date: date | pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(target_date).date()
    naive = datetime.combine(day - timedelta(days=1), time(12, 0))
    return pd.Timestamp(naive, tz=TZ)


def observation_available(source_timestamp_utc, issue_timestamp_utc, quality=True) -> bool:
    """Central point-in-time gate; quarter-hour values exist at interval end."""
    if not bool(quality) or pd.isna(source_timestamp_utc) or pd.isna(issue_timestamp_utc):
        return False
    source = pd.Timestamp(source_timestamp_utc)
    issue = pd.Timestamp(issue_timestamp_utc)
    if source.tzinfo is None or issue.tzinfo is None:
        raise ValueError("Availability comparisons require timezone-aware timestamps")
    return source + INTERVAL <= issue


def local_day_grid(target_date: date | str | pd.Timestamp) -> pd.DataFrame:
    day = pd.Timestamp(target_date).date()
    start = pd.Timestamp(datetime.combine(day, time.min), tz=TZ)
    end = pd.Timestamp(datetime.combine(day + timedelta(days=1), time.min), tz=TZ)
    local = pd.date_range(start, end, freq="15min", inclusive="left")
    utc = local.tz_convert("UTC")
    issue_local = issue_time_for_target_date(day)
    frame = pd.DataFrame({"target_time_local": local, "target_time_utc": utc})
    frame["forecast_date_local"] = pd.Timestamp(day)
    frame["forecast_issue_time_local"] = issue_local
    frame["forecast_issue_time_utc"] = issue_local.tz_convert("UTC")
    frame["lead_time_minutes"] = ((frame.target_time_utc - issue_local.tz_convert("UTC")) /
                                  pd.Timedelta(minutes=1)).astype("int32")
    return frame


@dataclass
class HistoryIndex:
    by_quarter: dict[int, list[tuple[pd.Timestamp, date, float]]]
    quarter_timestamps: dict[int, list[pd.Timestamp]]
    by_date_quarter: dict[tuple[date, int], list[tuple[pd.Timestamp, float]]]
    chronological: list[tuple[pd.Timestamp, float]]
    chronological_timestamps: list[pd.Timestamp]


def make_history_index(load: pd.DataFrame) -> HistoryIndex:
    eligible = load.loc[load.target_quality_primary & load.load_mean_kw.notna(),
                        ["timestamp_utc", "date_local", "quarter_hour_of_day", "load_mean_kw"]].copy()
    by_quarter: dict[int, list[tuple[pd.Timestamp, date, float]]] = {}
    by_date_quarter: dict[tuple[date, int], list[tuple[pd.Timestamp, float]]] = {}
    for row in eligible.itertuples(index=False):
        day = pd.Timestamp(row.date_local).date()
        item = (row.timestamp_utc, day, float(row.load_mean_kw))
        by_quarter.setdefault(int(row.quarter_hour_of_day), []).append(item)
        by_date_quarter.setdefault((day, int(row.quarter_hour_of_day)), []).append(
            (row.timestamp_utc, float(row.load_mean_kw)))
    for items in by_quarter.values():
        items.sort(key=lambda x: x[0])
    for items in by_date_quarter.values():
        items.sort(key=lambda x: x[0])
    chronological = sorted([(row.timestamp_utc, float(row.load_mean_kw))
                            for row in eligible.itertuples(index=False)], key=lambda x: x[0])
    quarter_timestamps = {quarter: [item[0] for item in items] for quarter, items in by_quarter.items()}
    return HistoryIndex(by_quarter, quarter_timestamps, by_date_quarter, chronological,
                        [item[0] for item in chronological])


def latest_available_value(items: Iterable[tuple[pd.Timestamp, float]], issue_utc: pd.Timestamp) -> float:
    valid = [value for timestamp, value in items if observation_available(timestamp, issue_utc)]
    return float(valid[-1]) if valid else np.nan


def b0_persistence(history: HistoryIndex, issue_utc: pd.Timestamp) -> tuple[float, pd.Timestamp | pd.NaT]:
    position = bisect_right(history.chronological_timestamps, issue_utc - INTERVAL) - 1
    if position < 0:
        return np.nan, pd.NaT
    timestamp, value = history.chronological[position]
    return float(value), timestamp


def b1_safe_recent_day(history: HistoryIndex, target_date: date, quarter: int,
                       issue_utc: pd.Timestamp) -> tuple[float, pd.Timestamp | pd.NaT]:
    items = history.by_quarter.get(int(quarter), [])
    timestamps = history.quarter_timestamps.get(int(quarter), [])
    cutoff = issue_utc - INTERVAL
    position = bisect_right(timestamps, cutoff) - 1
    while position >= 0:
        timestamp, source_date, value = items[position]
        if source_date < target_date and observation_available(timestamp, issue_utc):
            return float(value), timestamp
        position -= 1
    return np.nan, pd.NaT


def b2_weekly(history: HistoryIndex, target_date: date, quarter: int,
              issue_utc: pd.Timestamp) -> tuple[float, pd.Timestamp | pd.NaT]:
    source_date = target_date - timedelta(days=7)
    items = history.by_date_quarter.get((source_date, int(quarter)), [])
    valid = [(timestamp, value) for timestamp, value in items
             if observation_available(timestamp, issue_utc)]
    if not valid:
        return np.nan, pd.NaT
    timestamp, value = valid[-1]
    return float(value), timestamp


def b3_week_profile(history: HistoryIndex, target_date: date, quarter: int,
                    issue_utc: pd.Timestamp, weeks: int, statistic: str) -> tuple[float, int]:
    values: list[float] = []
    for lag in range(1, weeks + 1):
        source_date = target_date - timedelta(days=7 * lag)
        items = history.by_date_quarter.get((source_date, int(quarter)), [])
        valid = [value for timestamp, value in items if observation_available(timestamp, issue_utc)]
        if valid:
            values.append(float(valid[-1]))
    if not values:
        return np.nan, 0
    value = np.median(values) if statistic == "median" else np.mean(values)
    return float(value), len(values)


def load_gold() -> pd.DataFrame:
    columns = ["timestamp_utc", "timestamp_local", "date_local", "split", "load_mean_kw",
               "load_coverage", "target_quality_primary", "segment_id_primary", "quarter_hour_of_day",
               "day_of_week", "month", "is_weekend", "is_public_holiday"]
    frame = pd.read_parquet(GOLD, columns=columns).sort_values("timestamp_utc").reset_index(drop=True)
    frame["split"] = frame.split.astype(str)
    return frame


def base_prediction_rows(load: pd.DataFrame, split: str) -> pd.DataFrame:
    part = load[load.split.eq(split)].copy()
    segment_sizes = part.loc[part.target_quality_primary & part.segment_id_primary.notna()].groupby(
        "segment_id_primary", observed=True).size()
    part["primary_segment_intervals"] = part.segment_id_primary.map(segment_sizes).astype("Int32")
    local = part.timestamp_utc.dt.tz_convert(TZ)
    part["target_time_local"] = local
    part["target_time_utc"] = part.timestamp_utc
    part["forecast_date_local"] = part.date_local
    issue = {pd.Timestamp(day).date(): issue_time_for_target_date(day) for day in part.date_local.unique()}
    part["forecast_issue_time_local"] = [issue[pd.Timestamp(day).date()] for day in part.date_local]
    part["forecast_issue_time_utc"] = [value.tz_convert("UTC") for value in part.forecast_issue_time_local]
    part["lead_time_minutes"] = ((part.target_time_utc - part.forecast_issue_time_utc) /
                                  pd.Timedelta(minutes=1)).astype("int32")
    return part


def generate_history_baselines(load: pd.DataFrame, split: str, history: HistoryIndex,
                               b3_candidates: list[dict]) -> pd.DataFrame:
    base = base_prediction_rows(load, split)
    records: list[dict] = []
    issue_cache: dict[pd.Timestamp, tuple[float, pd.Timestamp | pd.NaT]] = {}
    for row in base.itertuples(index=False):
        target_date = pd.Timestamp(row.forecast_date_local).date()
        issue = row.forecast_issue_time_utc
        if issue not in issue_cache:
            issue_cache[issue] = b0_persistence(history, issue)
        b0, b0_source = issue_cache[issue]
        b1, b1_source = b1_safe_recent_day(history, target_date, row.quarter_hour_of_day, issue)
        b2, b2_source = b2_weekly(history, target_date, row.quarter_hour_of_day, issue)
        common = {"split": split, "forecast_date_local": row.forecast_date_local,
                  "forecast_issue_time_local": row.forecast_issue_time_local,
                  "forecast_issue_time_utc": issue, "target_time_local": row.target_time_local,
                  "target_time_utc": row.target_time_utc, "lead_time_minutes": row.lead_time_minutes,
                  "y_true_kw": row.load_mean_kw, "target_quality_primary": row.target_quality_primary,
                  "load_coverage": row.load_coverage, "segment_id_primary": row.segment_id_primary,
                  "primary_segment_intervals": row.primary_segment_intervals,
                  "quarter_hour_of_day": row.quarter_hour_of_day, "day_of_week": row.day_of_week,
                  "month": row.month, "is_weekend": row.is_weekend,
                  "is_public_holiday": row.is_public_holiday}
        for model, prediction, source in (("B0_persistence", b0, b0_source),
                                          ("B1_safe_recent_day", b1, b1_source),
                                          ("B2_weekly", b2, b2_source)):
            records.append(common | {"model_name": model, "y_pred_kw": prediction,
                                     "latest_source_time_utc": source, "source_count": 1 if pd.notna(prediction) else 0})
        for candidate in b3_candidates:
            prediction, count = b3_week_profile(history, target_date, row.quarter_hour_of_day, issue,
                                                 candidate["weeks"], candidate["statistic"])
            records.append(common | {"model_name": candidate["name"], "y_pred_kw": prediction,
                                     "latest_source_time_utc": pd.NaT, "source_count": count})
    return pd.DataFrame.from_records(records)


def calendar_profiles(load: pd.DataFrame, with_month: bool) -> pd.Series:
    train = load[load.split.eq("train") & load.target_quality_primary & load.load_mean_kw.notna()]
    keys = ["month", "day_of_week", "quarter_hour_of_day"] if with_month else ["day_of_week", "quarter_hour_of_day"]
    return train.groupby(keys, observed=True).load_mean_kw.median()


def add_calendar_candidate(base: pd.DataFrame, profile: pd.Series, name: str, with_month: bool) -> pd.DataFrame:
    keys = ["month", "day_of_week", "quarter_hour_of_day"] if with_month else ["day_of_week", "quarter_hour_of_day"]
    lookup = profile.rename("y_pred_kw").reset_index()
    result = base.drop_duplicates("target_time_utc").drop(
        columns=["y_pred_kw", "model_name", "latest_source_time_utc", "source_count"], errors="ignore"
    ).merge(lookup, on=keys, how="left", validate="many_to_one")
    result["model_name"] = name
    result["latest_source_time_utc"] = pd.NaT
    result["source_count"] = 0
    return result


def metric_values(frame: pd.DataFrame, train_scale: float) -> dict:
    eligible = frame.target_quality_primary & frame.y_true_kw.notna() & frame.y_pred_kw.notna()
    scored = frame.loc[eligible]
    if scored.empty:
        return {key: np.nan for key in ("MAE", "RMSE", "bias", "WAPE", "normalized_MAE")} | {
            "n_predictions": int(frame.y_pred_kw.notna().sum()),
            "n_forecasts_generated": int(frame.y_pred_kw.notna().sum()), "n_targets_evaluated": 0,
            "fraction_evaluated": 0.0}
    error = scored.y_pred_kw - scored.y_true_kw
    mae = float(error.abs().mean())
    generated = int(frame.y_pred_kw.notna().sum())
    return {"n_predictions": generated, "n_forecasts_generated": generated, "n_targets_evaluated": len(scored),
            "fraction_evaluated": float(len(scored) / max(int(frame.target_quality_primary.sum()), 1)),
            "MAE": mae, "RMSE": float(np.sqrt(np.mean(np.square(error)))), "bias": float(error.mean()),
            "WAPE": float(error.abs().sum() / scored.y_true_kw.abs().sum()),
            "normalized_MAE": float(mae / train_scale)}


def day_block_bootstrap(frame: pd.DataFrame, reps: int = BOOTSTRAP_REPS, seed: int = SEED) -> dict:
    scored = frame[frame.target_quality_primary & frame.y_true_kw.notna() & frame.y_pred_kw.notna()].copy()
    scored["abs_error"] = (scored.y_pred_kw - scored.y_true_kw).abs()
    scored["sq_error"] = np.square(scored.y_pred_kw - scored.y_true_kw)
    scored["abs_y"] = scored.y_true_kw.abs()
    daily = scored.groupby("forecast_date_local").agg(n=("y_true_kw", "size"), ae=("abs_error", "sum"),
        se=("sq_error", "sum"), ay=("abs_y", "sum"))
    if daily.empty:
        return {"MAE_CI_low": np.nan, "MAE_CI_high": np.nan, "RMSE_CI_low": np.nan,
                "RMSE_CI_high": np.nan, "WAPE_CI_low": np.nan, "WAPE_CI_high": np.nan}
    values = daily[["n", "ae", "se", "ay"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(reps, len(values)))
    sampled = values[draws].sum(axis=1)
    distributions = {"MAE": sampled[:, 1] / sampled[:, 0],
                     "RMSE": np.sqrt(sampled[:, 2] / sampled[:, 0]),
                     "WAPE": sampled[:, 1] / sampled[:, 3]}
    result = {}
    for metric, distribution in distributions.items():
        result[f"{metric}_CI_low"], result[f"{metric}_CI_high"] = np.quantile(distribution, [.025, .975])
    return {key: float(value) for key, value in result.items()}


def metric_table(predictions: pd.DataFrame, train_scale: float, bootstrap=True) -> pd.DataFrame:
    rows = []
    for (model, split), group in predictions.groupby(["model_name", "split"], observed=True):
        row = {"model": model, "split": split} | metric_values(group, train_scale)
        if bootstrap and split in {"validation", "test"}:
            row |= day_block_bootstrap(group)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "MAE", "model"], na_position="last")


def daily_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = predictions[predictions.target_quality_primary & predictions.y_true_kw.notna() &
                         predictions.y_pred_kw.notna()].copy()
    scored["error"] = scored.y_pred_kw - scored.y_true_kw
    scored["abs_error"] = scored.error.abs()
    scored["sq_error"] = scored.error.pow(2)
    return scored.groupby(["model_name", "split", "forecast_date_local"], observed=True).agg(
        n_evaluable=("error", "size"), MAE=("abs_error", "mean"),
        RMSE=("sq_error", lambda x: float(np.sqrt(x.mean()))), bias=("error", "mean")).reset_index()


def subgroup_metrics(predictions: pd.DataFrame, grouping: str, train_scale: float) -> pd.DataFrame:
    rows = []
    available = predictions[predictions[grouping].notna()]
    for (model, split, group_value), group in available.groupby(["model_name", "split", grouping], observed=True):
        rows.append({"model": model, "split": split, "grouping": grouping, "group": str(group_value)} |
                    metric_values(group, train_scale))
    return pd.DataFrame(rows)


def lead_band(minutes: pd.Series) -> pd.Categorical:
    return pd.cut(minutes, bins=[0, 18*60, 24*60, 30*60, 37*60], right=False,
                  labels=["12-18h", "18-24h", "24-30h", "30-37h"])


def add_ensembles(predictions: pd.DataFrame, b3_name: str) -> pd.DataFrame:
    components = predictions[predictions.model_name.isin(["B1_safe_recent_day", "B2_weekly", b3_name])]
    wide = components.pivot(index="target_time_utc", columns="model_name", values="y_pred_kw")
    valid = wide.notna().all(axis=1)
    metadata = predictions[predictions.model_name.eq("B1_safe_recent_day")].set_index("target_time_utc")
    rows = []
    for statistic, name in (("mean", "B5_candidate_mean"), ("median", "B5_candidate_median")):
        values = wide.mean(axis=1) if statistic == "mean" else wide.median(axis=1)
        values = values.where(valid)
        result = metadata.copy()
        result["y_pred_kw"] = values
        result["model_name"] = name
        result["latest_source_time_utc"] = pd.NaT
        result["source_count"] = np.where(valid, 3, 0)
        rows.append(result.reset_index())
    return pd.concat([predictions, *rows], ignore_index=True)


def distribution_tables(load: pd.DataFrame) -> dict[str, pd.DataFrame]:
    eligible = load[load.target_quality_primary & load.load_mean_kw.notna()].copy()
    summary = eligible.groupby("split").load_mean_kw.agg(mean="mean", std="std", median="median",
        Q10=lambda x: x.quantile(.10), Q25=lambda x: x.quantile(.25), Q75=lambda x: x.quantile(.75),
        Q90=lambda x: x.quantile(.90), Q95=lambda x: x.quantile(.95), n="size").reset_index()
    profiles = {}
    for key in ("quarter_hour_of_day", "day_of_week", "month"):
        profiles[key] = eligible.pivot_table(index=key, columns="split", values="load_mean_kw", aggfunc="mean").reset_index()
    return {"summary": summary, **profiles}


def create_figures(load: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame,
                   daily: pd.DataFrame, frozen_models: list[str]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    eligible = load[load.target_quality_primary & load.load_mean_kw.notna()]
    colors = {"train": "#377eb8", "validation": "#ff7f00", "test": "#4daf4a"}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for split in ("train", "validation", "test"):
        ax.hist(eligible.loc[eligible.split.eq(split), "load_mean_kw"], bins=60, density=True,
                histtype="step", linewidth=1.8, label=split, color=colors[split])
    ax.set(xlabel="Industrial load (kW)", ylabel="Density", title="Eligible load distribution by chronological split")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "load_distribution_by_split.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for split in ("train", "validation", "test"):
        profile = eligible[eligible.split.eq(split)].groupby("quarter_hour_of_day").load_mean_kw.mean()
        ax.plot(profile.index / 4, profile.values, label=split, color=colors[split])
    ax.set(xlabel="Local hour", ylabel="Mean load (kW)", title="Average local-day load profile")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "average_daily_profiles.png", dpi=180); plt.close(fig)

    final = metrics[metrics.model.isin(frozen_models) & metrics.split.isin(["validation", "test"])]
    pivot = final.pivot(index="model", columns="split", values="MAE").sort_values("test")
    fig, ax = plt.subplots(figsize=(9, 5)); pivot.plot.bar(ax=ax, color=[colors["test"], colors["validation"]])
    ax.set(ylabel="MAE (kW)", title="Deterministic baseline MAE"); ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(FIGURES / "baseline_mae_comparison.png", dpi=180); plt.close(fig)

    scored = predictions[predictions.model_name.isin(frozen_models) & predictions.split.eq("test") &
                         predictions.target_quality_primary & predictions.y_pred_kw.notna()].copy()
    scored["abs_error"] = (scored.y_pred_kw - scored.y_true_kw).abs()
    curve = scored.groupby(["model_name", "lead_time_minutes"]).abs_error.mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, group in curve.groupby("model_name"):
        ax.plot(group.lead_time_minutes / 60, group.abs_error, label=model, alpha=.85)
    ax.set(xlabel="Lead time (hours)", ylabel="MAE (kW)", title="Test error by forecast horizon")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGURES / "error_by_lead_time.png", dpi=180); plt.close(fig)

    best = final[final.split.eq("validation")].sort_values("MAE").iloc[0].model
    day = daily[(daily.model_name.eq(best)) & daily.split.eq("test")]
    fig, ax = plt.subplots(figsize=(8, 4.8)); ax.hist(day.MAE, bins=35, color="#377eb8", edgecolor="white")
    ax.axvline(day.MAE.median(), color="black", linestyle="--", label="median")
    ax.axvline(day.MAE.quantile(.9), color="#e41a1c", linestyle=":", label="P90")
    ax.set(xlabel="Daily MAE (kW)", ylabel="Forecast days", title=f"Test daily MAE: {best}")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "daily_test_mae_distribution.png", dpi=180); plt.close(fig)

    quantiles = day.MAE.quantile([.1, .5, .9])
    selected = [day.iloc[(day.MAE - value).abs().argsort().iloc[0]].forecast_date_local for value in quantiles]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    for ax, selected_day, label in zip(axes, selected, ["strong (P10)", "typical (P50)", "poor (P90)"], strict=True):
        sample = predictions[(predictions.model_name.eq(best)) & predictions.split.eq("test") &
                             predictions.forecast_date_local.eq(selected_day)]
        x = sample.target_time_local.dt.tz_localize(None)
        ax.plot(x, sample.y_true_kw, color="black", label="observed")
        ax.plot(x, sample.y_pred_kw, color="#377eb8", label="forecast")
        ax.set(ylabel="kW", title=f"{label}: {pd.Timestamp(selected_day).date()}")
    axes[0].legend(); axes[-1].set_xlabel("Local target time")
    fig.tight_layout(); fig.savefig(FIGURES / "representative_forecast_days.png", dpi=180); plt.close(fig)


def write_report(config: dict, load: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame,
                 daily: pd.DataFrame, lead: pd.DataFrame, quality: pd.DataFrame,
                 distribution: dict[str, pd.DataFrame], pairwise: pd.DataFrame) -> None:
    frozen = config["frozen_baselines"]
    selected_metrics = metrics[metrics.model.isin(frozen) & metrics.split.isin(["validation", "test"])]
    counts = load.groupby("split").agg(forecast_days=("date_local", "nunique"),
        grid_targets=("timestamp_utc", "size"), eligible_targets=("target_quality_primary", "sum")).reset_index()
    dst_counts = load.groupby("date_local").size().value_counts().sort_index()
    dist = distribution["summary"]
    shift = dist.set_index("split").loc["test", "mean"] - dist.set_index("split").loc["train", "mean"]
    qprofile = distribution["quarter_hour_of_day"].copy(); qprofile["delta"] = qprofile.test - qprofile.train
    wprofile = distribution["day_of_week"].copy(); wprofile["delta"] = wprofile.test - wprofile.train
    mprofile = distribution["month"].copy(); mprofile["delta"] = mprofile.test - mprofile.train
    daily_summary = daily[daily.model_name.isin(frozen) & daily.split.isin(["validation", "test"])].groupby(
        ["model_name", "split"]).MAE.agg(median_daily_MAE="median", P90_daily_MAE=lambda x: x.quantile(.9)).reset_index()
    worst = daily[daily.model_name.isin(frozen) & daily.split.eq("test")].sort_values("MAE", ascending=False).groupby(
        "model_name", observed=True).head(10)
    text = f"""# Leakage-safe day-ahead deterministic baselines

## Forecast protocol

Forecasts are issued at **12:00 Europe/Berlin on D-1** for every physical 15-minute interval in local calendar day D. UTC is the unique internal key. A historical quarter-hour load is considered available only when its interval end is no later than issuance. Evaluation uses only `target_quality_primary` observations (coverage >=75%, 135/180), without imputing ground truth.

{counts.to_markdown(index=False)}

Local-day lengths are preserved: {', '.join(f'{int(k)} intervals: {int(v)} days' for k, v in dst_counts.items())}. The prediction artifact retains local/UTC issue and target times and lead minutes.

## Leakage controls

All historical-load baselines call one centralized interval-end availability gate. B1 performs an adaptive timestamp lookup rather than a 96-row shift; after-noon target quarters cannot use the preceding day's after-noon load. B2 uses exact local calendar matching seven days earlier and leaves missing/DST-unavailable values absent. B3 uses only eligible prior weeks. Observed weather, measured PV, and prices are excluded. No train-fitted scaler or test-derived transformation is used.

## Data availability

The full target-day grid is forecast where a baseline input exists. Metrics report generated predictions, eligible evaluated targets, and evaluation fraction separately. Long outages remain represented by `segment_id_primary`.

## Distribution shift

{dist.to_markdown(index=False, floatfmt='.4f')}

Test mean load is {shift:.3f} kW above TRAIN. The quarter-hour mean shift is positive throughout the local day and ranges from {qprofile.delta.min():.3f} to {qprofile.delta.max():.3f} kW; the largest local-time increase occurs near quarter {int(qprofile.loc[qprofile.delta.idxmax(), 'quarter_hour_of_day'])}. Every weekday/weekend mean is also higher, by {wprofile.delta.min():.3f} to {wprofile.delta.max():.3f} kW. The monthly shift is less uniform ({mprofile.delta.min():.3f} to {mprofile.delta.max():.3f} kW), with the strongest increase in month {int(mprofile.loc[mprofile.delta.idxmax(), 'month'])}; December is not directly representative because measured 2024 load ends early in that month. Thus the increase is broad across hours and weekdays but seasonally concentrated. This is observed temporal/distribution shift, not causal concept drift.

## Baselines

- B0 persistence: latest primary-quality completed interval at issuance, held constant.
- B1 safe recent day: latest earlier local date with the same local quarter and completed by issuance.
- B2 weekly: exact same local quarter seven calendar days earlier; unavailable if absent.
- B3 recent-week profile: {config['selection']['B3']['weeks']} prior weeks using the {config['selection']['B3']['statistic']}.
- B4 calendar profile: TRAIN-only median grouped by {config['selection']['B4']['grouping']}.
{('- B5 baseline ensemble: ' + config['selection']['B5']['statistic'] + ' of B1, B2, and B3, requiring all three.' if config['selection']['B5']['retained'] else '- B5 was omitted because it did not satisfy the validation-only retention rule.')}

## Validation results

{selected_metrics[selected_metrics.split.eq('validation')].to_markdown(index=False, floatfmt='.4f')}

## Frozen baseline selection

B3, B4, and optional B5 were selected using 2023 only. The frozen configuration was written before constructing final TEST predictions. Selection rule details and candidate scores are preserved in `config/baseline_experiment.json`. No 2024 result changed a definition.

## Test results

{selected_metrics[selected_metrics.split.eq('test')].to_markdown(index=False, floatfmt='.4f')}

Negative bias means underprediction. The 2024 mean increase is preserved rather than rescaled away.

## Lead-time analysis

{lead[lead.split.eq('test')].to_markdown(index=False, floatfmt='.4f')}

## Calendar/operating-context analysis

Compact weekday/weekend, holiday, month, quarter-hour, TRAIN-load-magnitude, and long-versus-short contiguous-segment tables are stored separately. A long segment is defined as at least 96 eligible intervals (24 hours). These are diagnostic slices, not tuning inputs.

## Measurement-quality sensitivity

{quality[quality.split.eq('test')].to_markdown(index=False, floatfmt='.4f')}

## Daily-block uncertainty

Confidence intervals above use {BOOTSTRAP_REPS:,} deterministic bootstrap resamples of whole local target days (seed {SEED}), not independent quarter-hours.

{daily_summary.to_markdown(index=False, floatfmt='.4f')}

## Pairwise baseline comparison

{pairwise.to_markdown(index=False, floatfmt='.4f')}

## Failure cases

Worst test days are recorded in `worst_test_days.csv`. Baselines can be unavailable when exact historical inputs fail quality/DST/calendar requirements; those values remain missing. B0 can react poorly to schedule changes, B2 to atypical prior weeks, and static profiles to the higher 2024 load regime.

## Implications for learned models

A later deterministic model must beat these frozen baselines without using target-time observed weather, future PV, or unverified price availability. Forecast issuance semantics, interval-end availability, chronological splits, day-block uncertainty, and the 2024 distribution shift must remain explicit. No learned or probabilistic model was trained here.
"""
    REPORT.write_text(text, encoding="utf-8")
    worst.to_csv(RESULTS / "worst_test_days.csv", index=False)


def pairwise_daily(daily: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    test = daily[daily.split.eq("test") & daily.model_name.isin(models)].pivot(
        index="forecast_date_local", columns="model_name", values="MAE")
    rows = []
    for i, first in enumerate(models):
        for second in models[i + 1:]:
            common = test[[first, second]].dropna()
            difference = common[first] - common[second]
            magnitude = "large" if abs(difference.mean()) >= 1 else ("small" if abs(difference.mean()) < .25 else "moderate")
            consistency = "inconsistent" if .4 <= (difference < 0).mean() <= .6 else "directionally consistent"
            rows.append({"model_a": first, "model_b": second, "n_paired_days": len(common),
                         "mean_MAE_a_minus_b_kw": difference.mean(), "median_difference_kw": difference.median(),
                         "fraction_days_a_better": (difference < 0).mean(), "magnitude": magnitude,
                         "consistency": consistency})
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    load = load_gold()
    history = make_history_index(load)
    train_scale = float(load.loc[load.split.eq("train") & load.target_quality_primary, "load_mean_kw"].abs().mean())
    b3_candidates = [{"name": f"B3_candidate_{weeks}w_{statistic}", "weeks": weeks, "statistic": statistic}
                     for weeks in (4, 8) for statistic in ("median", "mean")]

    # Selection stage: no TEST predictions or metrics exist at this point.
    validation = generate_history_baselines(load, "validation", history, b3_candidates)
    val_base = validation[validation.model_name.eq("B1_safe_recent_day")]
    validation = pd.concat([validation,
        add_calendar_candidate(val_base, calendar_profiles(load, False), "B4_candidate_dow_qod", False),
        add_calendar_candidate(val_base, calendar_profiles(load, True), "B4_candidate_month_dow_qod", True)], ignore_index=True)
    val_metrics = metric_table(validation, train_scale, bootstrap=False)
    b3_scores = val_metrics[val_metrics.model.str.startswith("B3_candidate")].sort_values("MAE")
    selected_b3 = b3_scores.iloc[0].model
    selected_b3_def = next(item for item in b3_candidates if item["name"] == selected_b3)
    b4_scores = val_metrics[val_metrics.model.str.startswith("B4_candidate")].sort_values("MAE")
    selected_b4 = b4_scores.iloc[0].model
    validation = add_ensembles(validation, selected_b3)
    val_metrics = metric_table(validation, train_scale, bootstrap=False)
    ensemble_scores = val_metrics[val_metrics.model.str.startswith("B5_candidate")].sort_values("MAE")
    best_ensemble = ensemble_scores.iloc[0]
    # Compare ensemble and components on exactly the same eligible timestamps.
    selection_wide = validation.pivot(index="target_time_utc", columns="model_name", values="y_pred_kw")
    selection_meta = validation[validation.model_name.eq("B1_safe_recent_day")].set_index("target_time_utc")
    common = selection_meta.target_quality_primary & selection_wide[best_ensemble.model].notna()
    components = ["B1_safe_recent_day", "B2_weekly", selected_b3]
    common_mae = {model: float((selection_wide.loc[common, model] -
                                selection_meta.loc[common, "y_true_kw"]).abs().mean()) for model in components}
    best_component_name = min(common_mae, key=common_mae.get)
    component_best = common_mae[best_component_name]
    paired_rows = pd.DataFrame({"forecast_date_local": selection_meta.loc[common, "forecast_date_local"],
        "ensemble": (selection_wide.loc[common, best_ensemble.model] - selection_meta.loc[common, "y_true_kw"]).abs(),
        "component": (selection_wide.loc[common, best_component_name] - selection_meta.loc[common, "y_true_kw"]).abs()})
    paired = paired_rows.groupby("forecast_date_local")[["ensemble", "component"]].mean()
    retain_b5 = bool(best_ensemble.MAE <= component_best * .995 and (paired.ensemble < paired.component).mean() > .5)
    b5_statistic = "mean" if best_ensemble.model.endswith("mean") else "median"

    frozen = ["B0_persistence", "B1_safe_recent_day", "B2_weekly", "B3_recent_week_profile", "B4_calendar_profile"]
    if retain_b5:
        frozen.append("B5_baseline_ensemble")
    config = {"experiment_version": 1, "protocol": {"timezone": TZ, "issuance_local_time": "12:00 D-1",
        "target": "complete local calendar day D", "resolution_minutes": 15,
        "observation_available_at": "source interval end <= forecast issue UTC",
        "target_quality": "load_coverage >= 0.75 and load_valid_count >= 135",
        "splits": {"train": [2021, 2022], "validation": [2023], "test": [2024]}},
        "selection": {"selection_split": "validation_2023_only",
            "B3": selected_b3_def | {"candidate_scores_MAE": dict(zip(b3_scores.model, b3_scores.MAE))},
            "B4": {"grouping": "month,day_of_week,quarter_hour_of_day" if "month" in selected_b4 else "day_of_week,quarter_hour_of_day",
                   "statistic": "TRAIN-only median", "candidate_scores_MAE": dict(zip(b4_scores.model, b4_scores.MAE))},
            "B5": {"retained": retain_b5, "statistic": b5_statistic, "validation_MAE": float(best_ensemble.MAE),
                   "best_component_MAE": float(component_best), "paired_day_win_fraction": float((paired.ensemble < paired.component).mean()),
                   "retention_rule": ">=0.5% validation MAE improvement and >50% paired-day wins"}},
        "normalization": {"metric": "MAE divided by TRAIN eligible mean absolute load", "train_scale_kw": train_scale},
        "bootstrap": {"unit": "local target day", "repetitions": BOOTSTRAP_REPS, "seed": SEED},
        "excluded": ["observed weather", "measured PV", "day-ahead price"], "frozen_baselines": frozen,
        "gold_input_sha256": digest(GOLD), "checkpoint_commit": "39c1faa244752f5d5a7cd7082110d3a18f3310ea"}
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Frozen stage: construct TEST only after the frozen configuration is persisted.
    test = generate_history_baselines(load, "test", history, [selected_b3_def])
    test_base = test[test.model_name.eq("B1_safe_recent_day")]
    with_month = "month" in config["selection"]["B4"]["grouping"]
    test = pd.concat([test, add_calendar_candidate(test_base, calendar_profiles(load, with_month),
                                                   selected_b4, with_month)], ignore_index=True)
    test = add_ensembles(test, selected_b3)
    # Keep only frozen definitions and give selected candidates stable scientific names.
    validation = validation[validation.model_name.isin(["B0_persistence", "B1_safe_recent_day", "B2_weekly",
        selected_b3, selected_b4, best_ensemble.model] if retain_b5 else
        ["B0_persistence", "B1_safe_recent_day", "B2_weekly", selected_b3, selected_b4])].copy()
    test = test[test.model_name.isin(["B0_persistence", "B1_safe_recent_day", "B2_weekly", selected_b3,
        selected_b4, best_ensemble.model] if retain_b5 else
        ["B0_persistence", "B1_safe_recent_day", "B2_weekly", selected_b3, selected_b4])].copy()
    rename = {selected_b3: "B3_recent_week_profile", selected_b4: "B4_calendar_profile",
              best_ensemble.model: "B5_baseline_ensemble"}
    validation.model_name = validation.model_name.replace(rename); test.model_name = test.model_name.replace(rename)
    predictions = pd.concat([validation, test], ignore_index=True).sort_values(["split", "model_name", "target_time_utc"])
    required = ["model_name", "split", "forecast_date_local", "forecast_issue_time_local", "forecast_issue_time_utc",
        "target_time_local", "target_time_utc", "lead_time_minutes", "y_true_kw", "y_pred_kw",
        "target_quality_primary", "load_coverage", "segment_id_primary", "is_weekend", "is_public_holiday",
        "quarter_hour_of_day", "day_of_week", "month", "latest_source_time_utc", "source_count",
        "primary_segment_intervals"]
    predictions[required].to_parquet(RESULTS / "day_ahead_baseline_predictions.parquet", index=False, compression="zstd")

    metrics = metric_table(predictions, train_scale)
    metrics.to_csv(RESULTS / "baseline_metrics.csv", index=False)
    daily = daily_metrics(predictions); daily.to_csv(RESULTS / "baseline_daily_metrics.csv", index=False)
    analysis = predictions.copy(); analysis["lead_time_band"] = lead_band(analysis.lead_time_minutes)
    lead = subgroup_metrics(analysis, "lead_time_band", train_scale); lead.to_csv(RESULTS / "baseline_lead_time_metrics.csv", index=False)
    analysis["quality_band"] = pd.cut(analysis.load_coverage, [.75, .80, .90, .95, 1.000001], right=False,
                                      labels=["75-80%", "80-90%", "90-95%", ">=95%"])
    quality = subgroup_metrics(analysis, "quality_band", train_scale); quality.to_csv(RESULTS / "baseline_quality_band_metrics.csv", index=False)
    train_eligible = load[load.split.eq("train") & load.target_quality_primary].load_mean_kw
    edges = np.unique(np.quantile(train_eligible, [0, .25, .5, .75, 1]))
    analysis["train_load_quantile_bin"] = pd.cut(analysis.y_true_kw, edges, include_lowest=True, duplicates="drop")
    analysis["train_load_quantile_bin"] = analysis.train_load_quantile_bin.astype(object).fillna("outside_train_range")
    analysis["segment_class"] = np.where(analysis.primary_segment_intervals.ge(96), "long_>=24h", "short_<24h")
    context = pd.concat([subgroup_metrics(analysis, key, train_scale) for key in
                         ("is_weekend", "is_public_holiday", "month", "quarter_hour_of_day",
                          "train_load_quantile_bin", "segment_class")],
                        ignore_index=True)
    context.to_csv(RESULTS / "baseline_context_metrics.csv", index=False)
    distribution = distribution_tables(load)
    for name, table in distribution.items():
        table.to_csv(RESULTS / f"distribution_shift_{name}.csv", index=False)
    pairwise = pairwise_daily(daily, frozen); pairwise.to_csv(RESULTS / "baseline_pairwise_daily.csv", index=False)
    create_figures(load, predictions, metrics, daily, frozen)
    write_report(config, load, predictions, metrics, daily, lead, quality, distribution, pairwise)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "code_sha256": digest(Path(__file__)),
        "config_sha256": digest(CONFIG_PATH), "gold_input_sha256": digest(GOLD), "checkpoint_commit": config["checkpoint_commit"],
        "outputs": {str(path.relative_to(RESULTS)): {"bytes": path.stat().st_size, "sha256": digest(path)} for path in
                    sorted(RESULTS.rglob("*")) if path.is_file() and path.name != "baseline_run_manifest.json"}}
    (RESULTS / "baseline_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
