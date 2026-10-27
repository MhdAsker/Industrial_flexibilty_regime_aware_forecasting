"""Memory-bounded preparation of the small SPARK forecasting subset.

The RADAR publication exposes machine-level ZIPs.  EPI_TotalLoad.zip is 9.7 GB
and its endpoint does not honor HTTP Range, so this program deliberately does
not download it.  Put only the four P_total .csv.xz members listed in
config/required_files.csv into data/raw_temp and run ``aggregate-electrical``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_temp"
PROCESSED = ROOT / "data" / "processed"
FINAL = ROOT / "data" / "final"
META = ROOT / "data" / "metadata" / "extracted" / "00meta_data"
YEARS = range(2021, 2025)
EXPECTED_PER_BIN = 180
MIN_COVERAGE = 0.80
DATASET_ID = "bjdg3m3rg5jv3skk"
RADAR_PAGE = f"https://radar.kit.edu/radar/en/dataset/{DATASET_ID}"
RADAR_FRONTEND = "https://radar.kit.edu/radar"
MAX_AUTOMATIC_DOWNLOAD = 25 * 1024 * 1024


@dataclass(frozen=True)
class Signal:
    machine: str
    measurement: str
    prefix: str


SIGNALS = (
    Signal("EPI_TotalLoad", "P_total", "industrial_load"),
    Signal("EPI_PV", "AC_ActivePower", "pv_generation"),
)


def discover_public_files(session: requests.Session) -> dict[str, dict]:
    """Discover official top-level files; do not construct archive URLs."""
    response = session.get(
        f"{RADAR_FRONTEND}/ajax/getFolderContent",
        params={
            "folderId": DATASET_ID,
            "draw": 1,
            "start": 0,
            "length": 100,
            "order[0][column]": 0,
            "order[0][dir]": "asc",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    children = []
    for row in payload.get("data", []):
        children.extend(row.get("children", []))
    return {
        item["fileName"]: {
            "id": item["id"],
            "size": int(item["binary"]["size"]),
            "md5": item["binary"]["checksum"].lower(),
        }
        for item in children
    }


def _accept_download_terms(session: requests.Session) -> None:
    page = session.get(RADAR_PAGE, timeout=60)
    page.raise_for_status()
    match = re.search(r'<meta name="_csrf" content="([^"]+)"', page.text)
    if not match:
        raise RuntimeError("RADAR CSRF token was not found")
    response = session.post(
        f"{RADAR_FRONTEND}/ajax/acceptDownloadTc",
        headers={
            "X-CSRF-TOKEN": match.group(1),
            "X-Requested-With": "XMLHttpRequest",
            "Referer": RADAR_PAGE,
        },
        timeout=60,
    )
    response.raise_for_status()


def _download_discovered(session: requests.Session, info: dict, destination: Path) -> None:
    if info["size"] > MAX_AUTOMATIC_DOWNLOAD:
        raise ValueError(
            f"Refusing {info['size']:,}-byte download (limit {MAX_AUTOMATIC_DOWNLOAD:,})"
        )
    link = session.get(
        f"{RADAR_FRONTEND}/file/getDownloadLink/{info['id']}",
        headers={"X-Requested-With": "XMLHttpRequest", "Referer": RADAR_PAGE},
        timeout=60,
    )
    link.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.md5()  # RADAR publishes MD5 for transport verification.
    try:
        with session.get(link.json()["downloadUrl"], stream=True, timeout=(30, 240)) as response:
            response.raise_for_status()
            with partial.open("wb") as target:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        target.write(block)
                        digest.update(block)
        if partial.stat().st_size != info["size"] or digest.hexdigest() != info["md5"]:
            raise RuntimeError(f"Size/checksum verification failed for {destination.name}")
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def fetch_small_archives() -> None:
    requested = {
        "00meta_data.zip": ROOT / "data" / "metadata" / "00meta_data.zip",
        "EPI_PV.zip": RAW / "EPI_PV.zip",
        "01exogenous_data.zip": RAW / "01exogenous_data.zip",
    }
    with requests.Session() as session:
        files = discover_public_files(session)
        _accept_download_terms(session)
        for name, destination in requested.items():
            if name not in files:
                raise FileNotFoundError(f"RADAR did not advertise {name}")
            if destination.exists():
                current = hashlib.md5(destination.read_bytes()).hexdigest()
                if destination.stat().st_size == files[name]["size"] and current == files[name]["md5"]:
                    print(f"VERIFIED {destination}")
                    continue
            _download_discovered(session, files[name], destination)
            print(f"DOWNLOADED {destination}")
    metadata_zip = requested["00meta_data.zip"]
    with zipfile.ZipFile(metadata_zip) as zf:
        zf.extractall(ROOT / "data" / "metadata" / "extracted")


def _metadata_path(*candidates: str) -> Path:
    for name in candidates:
        path = META / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the metadata files exist: {candidates}")


def metadata_summary() -> dict:
    measurements = json.loads(
        _metadata_path("00measurement_details.json", "00measurement_deatils.json").read_text(encoding="utf-8")
    )
    machines = json.loads(_metadata_path("00machine_details.json").read_text(encoding="utf-8"))
    availability = pd.read_csv(_metadata_path("02data_availability.csv"))
    result: dict[str, object] = {
        "timestamp_column": "WsDateTime",
        "timestamp_format": "YYYY-MM-DD HH:MM:SS (naive text)",
        "measurement_timezone": "Europe/Berlin (CET/CEST)",
        "target_timezone": "UTC",
        "sampling_seconds": 5,
        "dst_policy": {
            "nonexistent_spring_hour": "NaT/drop (the cleaned local grid contains placeholders)",
            "ambiguous_fall_hour": "first fold / CEST; source has no offset or fold marker",
        },
        "signals": {},
    }
    by_name = {x["measurement"]: x for x in measurements}
    for signal in SIGNALS:
        rows = availability[
            (availability["machine"] == signal.machine)
            & (availability["measurement"] == signal.measurement)
            & (availability["year"].isin(YEARS))
        ]
        result["signals"][signal.prefix] = {
            "machine": signal.machine,
            "measurement": signal.measurement,
            **by_name[signal.measurement],
            "machine_description": machines[signal.machine]["long_desc"],
            "published_years": machines[signal.machine]["years"],
            "availability_2021_2024": {
                str(int(r.year)): float(r.availability) for r in rows.itertuples()
            },
        }
    return result


def print_summary() -> None:
    print(json.dumps(metadata_summary(), indent=2, ensure_ascii=False))


def _localize_to_utc(values: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    naive = pd.to_datetime(values, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    # True selects the daylight-saving occurrence for the source's single
    # fall-back 02:xx hour. The source has already collapsed the repeated local
    # hour, so both physical occurrences cannot be reconstructed.
    localized = naive.dt.tz_localize("Europe/Berlin", ambiguous=True, nonexistent="NaT")
    nonexistent_or_bad = localized.isna().to_numpy()
    ambiguous = (
        (naive.dt.month == 10)
        & (naive.dt.weekday == 6)
        & (naive.dt.day >= 25)
        & (naive.dt.hour == 2)
    ).to_numpy()
    return pd.DatetimeIndex(localized.dt.tz_convert("UTC")), nonexistent_or_bad, ambiguous


def _year_index_utc(year: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{year}-01-01", tz="Europe/Berlin").tz_convert("UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Berlin").tz_convert("UTC")
    return pd.date_range(start, end, freq="15min", inclusive="left")


def aggregate_xz(source: Path, signal: Signal, year: int, output: Path) -> dict:
    partials: list[pd.DataFrame] = []
    bad_timestamps = 0
    ambiguous_timestamps = 0
    for chunk in pd.read_csv(
        source,
        compression="xz",
        usecols=["WsDateTime", signal.measurement],
        dtype={signal.measurement: "float32"},
        chunksize=500_000,
    ):
        ts, bad, ambiguous = _localize_to_utc(chunk["WsDateTime"])
        bad_timestamps += int(bad.sum())
        ambiguous_timestamps += int(ambiguous.sum())
        value = chunk[signal.measurement].to_numpy(dtype=np.float32, copy=False)
        valid_ts = ~ts.isna()
        frame = pd.DataFrame({"timestamp_utc": ts[valid_ts], "value": value[valid_ts]})
        frame = frame.set_index("timestamp_utc")
        grouped = frame["value"].groupby(pd.Grouper(freq="15min", origin="epoch"))
        p = grouped.agg(["count", "sum", "min", "max"])
        p["sumsq"] = frame["value"].astype("float64").pow(2).groupby(
            pd.Grouper(freq="15min", origin="epoch")
        ).sum()
        partials.append(p.reset_index())

    if not partials:
        raise ValueError(f"No rows read from {source}")
    all_parts = pd.concat(partials, ignore_index=True)
    grouped = all_parts.groupby("timestamp_utc", sort=True)
    combined = grouped.agg(
        valid_count=("count", "sum"),
        total=("sum", "sum"),
        sumsq=("sumsq", "sum"),
        minimum=("min", "min"),
        maximum=("max", "max"),
    )
    count = combined["valid_count"].astype("int32")
    mean = combined["total"] / count.replace(0, np.nan)
    variance = (combined["sumsq"] - combined["total"].pow(2) / count.replace(0, np.nan)) / (
        count - 1
    ).replace(0, np.nan)
    variance = variance.clip(lower=0)
    combined["mean"] = mean
    combined["std"] = np.sqrt(variance)
    combined["coverage"] = count / EXPECTED_PER_BIN
    combined = combined.reindex(_year_index_utc(year))
    combined.index.name = "timestamp_utc"
    combined["valid_count"] = combined["valid_count"].fillna(0).astype("int16")
    combined["coverage"] = (combined["valid_count"] / EXPECTED_PER_BIN).astype("float32")
    insufficient = combined["coverage"] < MIN_COVERAGE
    for col in ["mean", "std", "minimum", "maximum"]:
        combined.loc[insufficient, col] = np.nan
        combined[col] = combined[col].astype("float32")
    combined = combined.rename(
        columns={
            "mean": f"{signal.prefix}_mean_w",
            "std": f"{signal.prefix}_std_w",
            "minimum": f"{signal.prefix}_min_w",
            "maximum": f"{signal.prefix}_max_w",
            "valid_count": f"{signal.prefix}_valid_count",
            "coverage": f"{signal.prefix}_coverage",
        }
    )
    combined = combined.drop(columns=["total", "sumsq"])
    combined["local_datetime"] = combined.index.tz_convert("Europe/Berlin").tz_localize(None)
    combined["local_utc_offset_minutes"] = [
        int(x.utcoffset().total_seconds() // 60) for x in combined.index.tz_convert("Europe/Berlin")
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.reset_index().to_parquet(output, index=False, compression="zstd")
    check = pq.ParquetFile(output)
    expected_rows = len(_year_index_utc(year))
    if check.metadata.num_rows != expected_rows:
        raise RuntimeError(f"Parquet verification failed: {check.metadata.num_rows} != {expected_rows}")
    return {
        "source": str(source),
        "output": str(output),
        "rows": expected_rows,
        "bad_or_nonexistent_source_timestamps": bad_timestamps,
        "ambiguous_fall_source_timestamps_assigned_cest": ambiguous_timestamps,
        "parquet_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def _find_input(signal: Signal, year: int) -> Path | None:
    candidates = [
        RAW / f"{year}_{signal.measurement}.csv.xz",
        RAW / signal.machine / signal.measurement / f"{year}_{signal.measurement}.csv.xz",
    ]
    return next((p for p in candidates if p.exists()), None)


def extract_member(archive: Path, member: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        with zf.open(member) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    return destination


def aggregate_all(delete_source: bool) -> None:
    reports = []
    pv_archive = RAW / "EPI_PV.zip"
    if pv_archive.exists() and not _find_input(SIGNALS[1], 2024):
        extract_member(
            pv_archive,
            "EPI_PV/AC_ActivePower/2024_AC_ActivePower.csv.xz",
            RAW / "2024_AC_ActivePower.csv.xz",
        )
    for signal in SIGNALS:
        for year in YEARS:
            source = _find_input(signal, year)
            if source is None:
                print(f"MISSING {signal.machine}/{signal.measurement}/{year}")
                continue
            output = PROCESSED / f"{signal.prefix}_{year}_15min.parquet"
            report = aggregate_xz(source, signal, year, output)
            reports.append(report)
            print(json.dumps(report))
            if delete_source:
                source.unlink()
    if delete_source and pv_archive.exists() and (PROCESSED / "pv_generation_2024_15min.parquet").exists():
        pv_archive.unlink()
    (PROCESSED / "aggregation_report.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )


def process_exogenous(archive: Path, delete_source: bool) -> None:
    wanted = {
        "weather_data_karlsruhe_hourly_2018_2024.csv",
        "holidays_kalsruhe_bw_2018_2024.csv",
        "holidays_karlsruhe_bw_2018_2024.csv",  # tolerate corrected spelling
        "electricity_prices_15min_de_2018_2024.csv",
    }
    with tempfile.TemporaryDirectory(dir=RAW) as td:
        temp = Path(td)
        with zipfile.ZipFile(archive) as zf:
            members = {Path(n).name: n for n in zf.namelist() if Path(n).name in wanted}
            for base, member in members.items():
                with zf.open(member) as src, (temp / base).open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
        weather_path = temp / "weather_data_karlsruhe_hourly_2018_2024.csv"
        price_path = temp / "electricity_prices_15min_de_2018_2024.csv"
        holiday_paths = list(temp.glob("holidays*k*ruhe_bw_2018_2024.csv"))
        if not (weather_path.exists() and price_path.exists() and holiday_paths):
            raise FileNotFoundError(f"Required exogenous members not found; got {sorted(members)}")

        weather = pd.read_csv(weather_path)
        weather["utc_datetime"] = pd.to_datetime(weather["utc_datetime"], utc=True)
        weather = weather[weather["utc_datetime"].dt.year.isin(YEARS)]
        weather.to_parquet(PROCESSED / "weather_all_original_2021_2024.parquet", index=False, compression="zstd")

        selected = [
            "utc_datetime",
            "temperature_air_mean_2m_°C",
            "radiation_global_J/cm²_per_hour",
            "cloud_cover_total_oktas_(0–8)",
            "humidity_%",
        ]
        weather = weather[selected].set_index("utc_datetime").sort_index()
        # An hourly observation is attached only to the four quarters in its
        # own labelled UTC hour. A missing hour remains four missing quarters.
        offsets = [pd.Timedelta(minutes=x) for x in (0, 15, 30, 45)]
        expanded = pd.concat([weather.set_axis(weather.index + off) for off in offsets]).sort_index()
        expanded["weather_source_age_minutes"] = [x.minute % 60 for x in expanded.index]
        expanded["weather_available"] = expanded[selected[1:]].notna().any(axis=1)

        prices = pd.read_csv(price_path)
        prices["utc_datetime"] = pd.to_datetime(prices["utc_datetime"], utc=True)
        prices = prices[prices["utc_datetime"].dt.year.isin(YEARS)].set_index("utc_datetime")
        prices["price_available"] = prices["price_eur_per_mwh"].notna()

        holidays = pd.read_csv(holiday_paths[0])
        holidays["date"] = pd.to_datetime(holidays["date"]).dt.date
        holiday_lookup = holidays.set_index("date")
        full_index = pd.date_range(
            pd.Timestamp("2021-01-01", tz="Europe/Berlin").tz_convert("UTC"),
            pd.Timestamp("2025-01-01", tz="Europe/Berlin").tz_convert("UTC"),
            freq="15min",
            inclusive="left",
        )
        out = pd.DataFrame(index=full_index)
        out.index.name = "timestamp_utc"
        out = out.join(expanded, how="left").join(prices[["price_eur_per_mwh", "price_available"]], how="left")
        local_dates = pd.Series(out.index.tz_convert("Europe/Berlin").date, index=out.index)
        out["local_date"] = local_dates
        out["weekday"] = local_dates.map(holiday_lookup["weekday"])
        out["is_holiday"] = local_dates.map(holiday_lookup["is_holiday"])
        out["holiday_name"] = local_dates.map(holiday_lookup["holiday_name"])
        out["holiday_data_available"] = out["weekday"].notna()
        out.reset_index().to_parquet(PROCESSED / "exogenous_15min_2021_2024.parquet", index=False, compression="zstd")
    if delete_source:
        archive.unlink()


def build_final() -> None:
    index = pd.date_range(
        pd.Timestamp("2021-01-01", tz="Europe/Berlin").tz_convert("UTC"),
        pd.Timestamp("2025-01-01", tz="Europe/Berlin").tz_convert("UTC"),
        freq="15min",
        inclusive="left",
    )
    out = pd.DataFrame(index=index)
    out.index.name = "timestamp_utc"
    for signal in SIGNALS:
        files = sorted(PROCESSED.glob(f"{signal.prefix}_*_15min.parquet"))
        if files:
            part = pd.concat([pd.read_parquet(p).set_index("timestamp_utc") for p in files])
            part = part.drop(columns=["local_datetime", "local_utc_offset_minutes"], errors="ignore")
            out = out.join(part, how="left")
    exogenous = PROCESSED / "exogenous_15min_2021_2024.parquet"
    if exogenous.exists():
        out = out.join(pd.read_parquet(exogenous).set_index("timestamp_utc"), how="left")
    # Keep a stable schema while waiting for manually supplied TotalLoad files.
    # Missing source years are represented as missing data, never as zeros.
    for col in ("industrial_load_mean_w", "pv_generation_mean_w"):
        if col not in out:
            out[col] = np.nan
    out["net_load_mean_w"] = out["industrial_load_mean_w"] - out["pv_generation_mean_w"]
    out["net_load_available"] = out["net_load_mean_w"].notna()
    out["local_datetime"] = out.index.tz_convert("Europe/Berlin").tz_localize(None)
    FINAL.mkdir(parents=True, exist_ok=True)
    target = FINAL / "spark_forecasting_15min_2021_2024.parquet"
    out.reset_index().to_parquet(target, index=False, compression="zstd")
    pf = pq.ParquetFile(target)
    print(json.dumps({"output": str(target), "rows": pf.metadata.num_rows, "columns": pf.schema.names}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("summary")
    sub.add_parser("fetch-small")
    agg = sub.add_parser("aggregate-electrical")
    agg.add_argument("--delete-source", action="store_true")
    exo = sub.add_parser("process-exogenous")
    exo.add_argument("archive", type=Path)
    exo.add_argument("--delete-source", action="store_true")
    sub.add_parser("build-final")
    args = parser.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if args.command == "fetch-small":
        fetch_small_archives()
    elif args.command == "summary":
        print_summary()
    elif args.command == "aggregate-electrical":
        aggregate_all(args.delete_source)
    elif args.command == "process-exogenous":
        process_exogenous(args.archive, args.delete_source)
    elif args.command == "build-final":
        build_final()


if __name__ == "__main__":
    main()
