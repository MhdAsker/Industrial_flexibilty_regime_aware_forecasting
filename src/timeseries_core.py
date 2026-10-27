"""Auditable, memory-bounded processing of SPARK 5-second measurements."""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import math
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "pipeline_config.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def file_digest(path: Path, algorithm: str = "sha256", block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        while block := source.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: Path, expected_size: int, expected_md5: str, cache: bool = True) -> dict:
    """Verify immutable input once and reuse a size/mtime-bound sidecar."""
    if not path.exists():
        raise FileNotFoundError(path)
    stat = path.stat()
    if stat.st_size != expected_size:
        raise ValueError(f"{path}: size {stat.st_size} != {expected_size}")
    sidecar = path.with_suffix(path.suffix + ".verified.json")
    if cache and sidecar.exists():
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        if (saved.get("size") == stat.st_size and saved.get("mtime_ns") == stat.st_mtime_ns
                and saved.get("md5") == expected_md5):
            return saved
    actual = file_digest(path, "md5")
    if actual != expected_md5:
        raise ValueError(f"{path}: MD5 {actual} != official {expected_md5}")
    result = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "md5": actual}
    sidecar.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def member_info(archive: Path, member: str) -> dict:
    with zipfile.ZipFile(archive) as zf:
        info = zf.getinfo(member)
        return {"member": member, "file_size": info.file_size, "compress_size": info.compress_size,
                "crc32": f"{info.CRC:08x}"}


def iter_member_csv(archive: Path, member: str, timestamp_col: str, value_col: str,
                    chunksize: int) -> Iterator[pd.DataFrame]:
    """Stream ZIP -> selected XZ member -> CSV without permanent extraction."""
    with zipfile.ZipFile(archive) as zf, zf.open(member) as zipped, lzma.LZMAFile(zipped) as decompressed:
        for chunk in pd.read_csv(
            decompressed,
            usecols=[timestamp_col, value_col],
            dtype={timestamp_col: "string", value_col: "string"},
            chunksize=chunksize,
            keep_default_na=True,
        ):
            yield chunk


def ambiguous_mask(naive: pd.Series, timezone: str) -> np.ndarray:
    """True where both DST folds are possible."""
    valid = naive.notna()
    first = naive.dt.tz_localize(timezone, ambiguous=True, nonexistent="NaT")
    second = naive.dt.tz_localize(timezone, ambiguous=False, nonexistent="NaT")
    return (valid & first.notna() & second.notna() & (first.astype("int64") != second.astype("int64"))).to_numpy()


def localize_explicit(naive: pd.Series, timezone: str, fold_policy: str = "first") -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Localize naive source times using a documented, testable fold policy."""
    amb = ambiguous_mask(naive, timezone)
    if fold_policy not in {"first", "second", "nat"}:
        raise ValueError(f"Unsupported ambiguous fold policy: {fold_policy}")
    ambiguous_arg: bool | str = {"first": True, "second": False, "nat": "NaT"}[fold_policy]
    localized = naive.dt.tz_localize(timezone, ambiguous=ambiguous_arg, nonexistent="NaT")
    nonexistent_or_bad = (naive.isna() | localized.isna()).to_numpy() & ~amb
    return localized.dt.tz_convert("UTC"), amb, nonexistent_or_bad


def aggregate_frame(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate one sorted frame; caller prevents interval splits across frames."""
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.set_index("timestamp_utc")["value"].groupby(pd.Grouper(freq=frequency, origin="epoch"))
    result = grouped.agg(["mean", "std", "min", "max", "median", "count"])
    quantiles = grouped.quantile([0.05, 0.25, 0.75, 0.95]).unstack()
    quantiles.columns = ["q05", "q25", "q75", "q95"]
    result = result.join(quantiles).rename(columns={"count": "valid_count"})
    result.index.name = "timestamp_utc"
    return result


@dataclass
class RawAudit:
    year: int
    row_count: int = 0
    parsed_timestamp_count: int = 0
    invalid_timestamp_count: int = 0
    duplicate_timestamp_count: int = 0
    non_monotonic_count: int = 0
    numeric_count: int = 0
    nan_count: int = 0
    nonnumeric_count: int = 0
    negative_count: int = 0
    zero_count: int = 0
    positive_infinity_count: int = 0
    negative_infinity_count: int = 0
    missing_timestamp_count: int = 0
    ambiguous_timestamp_count: int = 0
    nonexistent_or_bad_timestamp_count: int = 0
    first_timestamp_local: str | None = None
    last_timestamp_local: str | None = None
    min_w: float | None = None
    max_w: float | None = None
    gap_lengths_samples: Counter = field(default_factory=Counter)
    longest_missing_gap_samples: int = 0
    _previous_timestamp: pd.Timestamp | None = None
    _open_invalid_run: int = 0

    def _close_gap(self) -> None:
        if self._open_invalid_run:
            self.gap_lengths_samples[self._open_invalid_run] += 1
            self.longest_missing_gap_samples = max(self.longest_missing_gap_samples, self._open_invalid_run)
            self._open_invalid_run = 0

    def update(self, local: pd.Series, raw_values: pd.Series, values: pd.Series, amb: np.ndarray,
               bad_localization: np.ndarray, sample_seconds: int) -> None:
        self.row_count += len(local)
        parsed = local.notna()
        self.parsed_timestamp_count += int(parsed.sum())
        self.invalid_timestamp_count += int((~parsed).sum())
        self.ambiguous_timestamp_count += int(amb.sum())
        self.nonexistent_or_bad_timestamp_count += int(bad_localization.sum())
        original_missing = raw_values.isna() | raw_values.str.strip().eq("")
        numeric = values.notna() & np.isfinite(values)
        self.numeric_count += int(numeric.sum())
        self.nan_count += int(original_missing.sum())
        self.nonnumeric_count += int((~original_missing & values.isna()).sum())
        self.positive_infinity_count += int(np.isposinf(values).sum())
        self.negative_infinity_count += int(np.isneginf(values).sum())
        finite_values = values[numeric]
        self.negative_count += int((finite_values < 0).sum())
        self.zero_count += int((finite_values == 0).sum())
        if not finite_values.empty:
            low, high = float(finite_values.min()), float(finite_values.max())
            self.min_w = low if self.min_w is None else min(self.min_w, low)
            self.max_w = high if self.max_w is None else max(self.max_w, high)
        valid_local = local[parsed]
        if not valid_local.empty:
            if self.first_timestamp_local is None:
                self.first_timestamp_local = valid_local.iloc[0].isoformat(sep=" ")
            self.last_timestamp_local = valid_local.iloc[-1].isoformat(sep=" ")
            shifted = valid_local.shift(1)
            if self._previous_timestamp is not None:
                shifted.iloc[0] = self._previous_timestamp
            deltas = (valid_local - shifted).dt.total_seconds()
            self.duplicate_timestamp_count += int((deltas == 0).sum())
            self.non_monotonic_count += int((deltas < 0).sum())
            positive_steps = deltas[deltas > sample_seconds]
            self.missing_timestamp_count += int(((positive_steps / sample_seconds).round() - 1).clip(lower=0).sum())
            self._previous_timestamp = valid_local.iloc[-1]

        # Availability gaps include explicit invalid readings and absent grid slots.
        for ts, is_valid in zip(local, numeric, strict=True):
            if pd.isna(ts):
                continue
            missing_between = 0
            if getattr(self, "_gap_previous_timestamp", None) is not None:
                delta = (ts - self._gap_previous_timestamp).total_seconds()
                if delta > sample_seconds:
                    missing_between = max(0, int(round(delta / sample_seconds)) - 1)
            if missing_between:
                self._open_invalid_run += missing_between
            if bool(is_valid):
                self._close_gap()
            else:
                self._open_invalid_run += 1
            self._gap_previous_timestamp = ts

    def finish(self, expected_samples: int) -> dict:
        self._close_gap()
        result = {k: v for k, v in vars(self).items() if not k.startswith("_") and k != "gap_lengths_samples"}
        result["availability_percent"] = 100.0 * self.numeric_count / expected_samples
        result["expected_samples"] = expected_samples
        result["invalid_or_unavailable_samples"] = max(0, expected_samples - self.numeric_count)
        result["longest_missing_gap_seconds"] = self.longest_missing_gap_samples * 5
        result["gap_length_distribution_samples"] = {
            str(k): v for k, v in sorted(self.gap_lengths_samples.items())
        }
        return result


def utc_year_index(year: int, frequency: str, timezone: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{year}-01-01", tz=timezone).tz_convert("UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz=timezone).tz_convert("UTC")
    return pd.date_range(start, end, freq=frequency, inclusive="left")


def expected_raw_samples(year: int, sample_seconds: int) -> int:
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year + 1}-01-01")
    return int((end - start).total_seconds() / sample_seconds)


def _output_valid(output: Path, manifest: Path, identity: dict) -> bool:
    if not output.exists() or not manifest.exists():
        return False
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    if saved.get("identity") != identity:
        return False
    if saved.get("output_size") != output.stat().st_size:
        return False
    return saved.get("output_sha256") == file_digest(output)


def _finish_resolution(parts: list[pd.DataFrame], year: int, frequency: str, expected_count: int,
                       timezone: str, prefix: str, duplicate_bins: set[pd.Timestamp]) -> pd.DataFrame:
    if parts:
        out = pd.concat(parts).sort_index()
        if out.index.has_duplicates:
            raise RuntimeError("An aggregation bin crossed chunks; this is a pipeline bug")
    else:
        out = pd.DataFrame()
    out = out.reindex(utc_year_index(year, frequency, timezone))
    out.index.name = "timestamp_utc"
    out["valid_count"] = out.get("valid_count", 0).fillna(0).astype("int32")
    out["expected_count"] = np.int32(expected_count)
    out["missing_count"] = (expected_count - out["valid_count"]).clip(lower=0).astype("int32")
    out["coverage"] = out["valid_count"].div(expected_count).astype("float64")
    for threshold in (100, 99, 95, 90, 80):
        out[f"quality_{threshold}"] = out["coverage"] >= threshold / 100
    out["duplicate_observations_present"] = out.index.isin(duplicate_bins)
    out["p_mean_kw"] = out["mean"] / 1000.0
    out["p_max_kw"] = out["max"] / 1000.0
    # Observed rectangular-rule energy only; no scaling or imputation for missing samples.
    out["energy_kwh_observed"] = out["mean"] / 1000.0 * out["valid_count"] * 5.0 / 3600.0
    finite_max = out["max"].dropna()
    if len(finite_max):
        q1, q3 = finite_max.quantile([0.25, 0.75])
        upper = float(q3 + 3.0 * (q3 - q1))
    else:
        upper = math.nan
    out["extreme_high"] = out["max"].ge(upper) if math.isfinite(upper) else False
    out["timestamp_local"] = out.index.tz_convert(timezone).tz_localize(None)
    ordered = ["timestamp_local", "mean", "std", "min", "max", "median", "q05", "q25", "q75", "q95",
               "p_mean_kw", "p_max_kw", "energy_kwh_observed", "valid_count", "expected_count", "missing_count",
               "coverage", "quality_100", "quality_99", "quality_95", "quality_90", "quality_80",
               "duplicate_observations_present", "extreme_high"]
    out = out.reset_index()[["timestamp_utc", *ordered]]
    return out.rename(columns={c: f"p_{c}_w" for c in ["mean", "std", "min", "max", "median", "q05", "q25", "q75", "q95"]})


def process_signal_year(signal_name: str, year: int, force: bool = False) -> dict:
    config = load_config()
    signal = config["signals"][signal_name]
    if year not in signal["years"]:
        raise ValueError(f"{signal_name} is not published for {year}")
    archive = ROOT / signal["archive"]
    verification = verify_archive(archive, signal["archive_size"], signal["archive_md5"])
    member = signal["member_template"].format(year=year)
    member_meta = member_info(archive, member)
    identity = {
        "archive_md5": verification["md5"], "member": member_meta,
        "year": year, "signal": signal_name, "config": config,
        "pipeline_version": 3,
        "core_sha256": file_digest(Path(__file__)),
    }
    outputs = {name: ROOT / "data" / "silver" / signal_name / name / f"{signal_name}_{year}_{name}.parquet"
               for name in config["resolutions"]}
    manifests = {name: path.with_suffix(".manifest.json") for name, path in outputs.items()}
    audit_path = ROOT / "reports" / "raw_quality" / f"{signal_name}_{year}.json"
    dst_path = ROOT / "reports" / "dst" / f"{signal_name}_{year}.json"
    if not force and audit_path.exists() and dst_path.exists() and all(
        _output_valid(outputs[name], manifests[name], identity) for name in outputs
    ):
        return {"status": "skipped_valid", "signal": signal_name, "year": year,
                "outputs": {k: str(v) for k, v in outputs.items()}}

    parts = {name: [] for name in config["resolutions"]}
    tails = {name: pd.DataFrame(columns=["timestamp_utc", "value"]) for name in config["resolutions"]}
    duplicate_bins = {name: set() for name in config["resolutions"]}
    audit = RawAudit(year)
    previous_local: pd.Timestamp | None = None
    spring_rows = autumn_rows = 0
    spring_valid_rows = autumn_valid_rows = 0

    for raw in iter_member_csv(archive, member, config["timestamp_column"], signal["measurement"], config["chunk_rows"]):
        local = pd.to_datetime(raw[config["timestamp_column"]], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        raw_values = raw[signal["measurement"]]
        values = pd.to_numeric(raw_values, errors="coerce").astype("float64")
        utc, amb, bad = localize_explicit(local, config["timezone"], config["dst"]["single_ambiguous_fold"])
        audit.update(local, raw_values, values, amb, bad, config["sample_seconds"])
        spring_rows += int(bad.sum())
        autumn_rows += int(amb.sum())
        finite = values.notna() & np.isfinite(values)
        spring_valid_rows += int((bad & finite.to_numpy()).sum())
        autumn_valid_rows += int((amb & finite.to_numpy()).sum())
        if audit.non_monotonic_count:
            raise RuntimeError(f"{member} is non-monotonic; external sorting is required before aggregation")
        frame = pd.DataFrame({"timestamp_utc": utc, "timestamp_local": local, "value": values})
        frame = frame.dropna(subset=["timestamp_utc"])
        if previous_local is not None and not frame.empty and frame["timestamp_local"].iloc[0] == previous_local:
            pass
        duplicated = frame["timestamp_local"].duplicated(keep=False)
        previous_local = frame["timestamp_local"].iloc[-1] if not frame.empty else previous_local
        for name, resolution in config["resolutions"].items():
            current = pd.concat([tails[name], frame[["timestamp_utc", "value"]]], ignore_index=True)
            if current.empty:
                continue
            bins = current["timestamp_utc"].dt.floor(resolution["frequency"])
            if duplicated.any():
                duplicate_bins[name].update(frame.loc[duplicated, "timestamp_utc"].dt.floor(resolution["frequency"]).tolist())
            last_bin = bins.iloc[-1]
            complete = current.loc[bins != last_bin]
            tails[name] = current.loc[bins == last_bin].copy()
            aggregated = aggregate_frame(complete, resolution["frequency"])
            if not aggregated.empty:
                parts[name].append(aggregated)

    for name, resolution in config["resolutions"].items():
        tail_agg = aggregate_frame(tails[name], resolution["frequency"])
        if not tail_agg.empty:
            parts[name].append(tail_agg)
        result = _finish_resolution(parts[name], year, resolution["frequency"], resolution["expected_count"],
                                    config["timezone"], signal_name, duplicate_bins[name])
        output = outputs[name]
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".parquet.part")
        result.to_parquet(temp, index=False, compression="zstd")
        expected_rows = len(utc_year_index(year, resolution["frequency"], config["timezone"]))
        actual_rows = pq.ParquetFile(temp).metadata.num_rows
        if actual_rows != expected_rows:
            temp.unlink(missing_ok=True)
            raise RuntimeError(f"{output}: {actual_rows} rows != {expected_rows}")
        temp.replace(output)
        manifest_data = {"identity": identity, "output": str(output), "output_rows": actual_rows,
                         "output_size": output.stat().st_size, "output_sha256": file_digest(output)}
        manifests[name].write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    expected = expected_raw_samples(year, config["sample_seconds"])
    audit_result = audit.finish(expected)
    pmax = result["p_max_w"].dropna()
    if len(pmax):
        q1, q3 = pmax.quantile([0.25, 0.75])
        extreme_threshold = float(q3 + 3.0 * (q3 - q1))
    else:
        extreme_threshold = None
    audit_result.update({"signal": signal_name, "measurement": signal["measurement"], "unit": signal["unit"],
                         "sign_convention": signal["sign_convention"], "source_archive": str(archive),
                         "source_member": member,
                         "extreme_definition": "15-minute interval maximum above Q3 + 3*IQR of yearly interval maxima",
                         "extreme_high_threshold_w": extreme_threshold,
                         "extreme_high_15min_intervals": int(result["extreme_high"].sum())})
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_result, indent=2), encoding="utf-8")
    dst = {
        "signal": signal_name, "year": year, "timezone": config["timezone"],
        "source_timestamps": "naive local wall time",
        "spring_nonexistent_or_bad_rows": spring_rows,
        "spring_nonexistent_rows_with_numeric_values_excluded_from_utc": spring_valid_rows,
        "autumn_ambiguous_rows": autumn_rows,
        "autumn_ambiguous_rows_with_numeric_values": autumn_valid_rows,
        "ambiguous_policy": config["dst"]["single_ambiguous_fold"],
        "resolution": "The publication contains one unlabelled autumn 02:00 hour. It is explicitly assigned to the first (CEST) fold; the second fold remains unavailable and no observations are fabricated.",
        "canonical_time": "UTC", "local_time_preserved": True,
    }
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(dst, indent=2), encoding="utf-8")
    return {"status": "processed", "signal": signal_name, "year": year,
            "audit": audit_result, "outputs": {k: str(v) for k, v in outputs.items()}}
