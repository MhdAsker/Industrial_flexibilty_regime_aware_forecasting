"""Acquire and canonicalize only the small official SPARK exogenous archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests

from download_raw import FRONTEND, PAGE, accept_terms, discovery
from timeseries_core import ROOT, file_digest


ARCHIVE_NAME = "01exogenous_data.zip"
EXPECTED_SIZE = 8_324_985
EXPECTED_MD5 = "6d8f51351e734cf63cec90f3cc7a46d4"
FILE_ID = "hq6kj9fyw6h2xccp"
SELECTED = {
    "weather": "01exogenous_data/weather_data_karlsruhe_hourly_2018_2024.csv",
    "stations": "01exogenous_data/weather_data_stations_used.csv",
    "holidays": "01exogenous_data/holidays_kalsruhe_bw_2018_2024.csv",
    "prices": "01exogenous_data/electricity_prices_15min_de_2018_2024.csv",
}
EMISSIONS = "01exogenous_data/electricity_emissions_15min_transnetbw_2018_2024.csv"


def acquire() -> Path:
    target = ROOT / "data" / "raw" / ARCHIVE_NAME
    if target.exists():
        if target.stat().st_size != EXPECTED_SIZE or file_digest(target, "md5") != EXPECTED_MD5:
            raise RuntimeError(f"Existing {target} does not match official size/MD5")
        return target
    with requests.Session() as session:
        remote = discovery(session).get(ARCHIVE_NAME)
        expected = {"id": FILE_ID, "size": EXPECTED_SIZE, "md5": EXPECTED_MD5}
        if remote != expected:
            raise RuntimeError(f"Official discovery metadata changed: {remote!r} != {expected!r}")
        accept_terms(session)
        link = session.get(f"{FRONTEND}/file/getDownloadLink/{FILE_ID}", headers={
            "X-Requested-With": "XMLHttpRequest", "Referer": PAGE}, timeout=60)
        link.raise_for_status()
        partial = target.with_suffix(".zip.part")
        digest = hashlib.md5()
        written = 0
        with session.get(link.json()["downloadUrl"], stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with partial.open("wb") as destination:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        destination.write(block)
                        digest.update(block)
                        written += len(block)
        if written != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_MD5:
            raise RuntimeError("Exogenous archive failed official size/MD5 validation")
        partial.replace(target)
    return target


def extract_sources(archive: Path) -> tuple[dict, bool]:
    source_dir = ROOT / "data" / "silver" / "exogenous" / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for label, member in SELECTED.items():
            if member not in names:
                raise RuntimeError(f"Official archive is missing required member {member}")
            target = source_dir / Path(member).name
            with zf.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            manifest[label] = {"member": member, "path": str(target.relative_to(ROOT)),
                               "bytes": target.stat().st_size, "sha256": file_digest(target)}
        emissions_present = EMISSIONS in names
        if emissions_present:
            info = zf.getinfo(EMISSIONS)
            manifest["emissions_catalogue_only"] = {
                "member": EMISSIONS, "uncompressed_bytes": info.file_size,
                "compressed_bytes": info.compress_size, "crc32": f"{info.CRC:08x}",
                "status": "present in official archive; optional and not extracted"}
    return manifest, emissions_present


def canonicalize(manifest: dict) -> dict:
    source_dir = ROOT / "data" / "silver" / "exogenous" / "source"
    output = ROOT / "data" / "silver" / "exogenous" / "canonical"
    output.mkdir(parents=True, exist_ok=True)
    results = {}

    price = pd.read_csv(source_dir / Path(SELECTED["prices"]).name)
    price["utc_datetime"] = pd.to_datetime(price["utc_datetime"], utc=True, errors="raise")
    price = price[price.utc_datetime.dt.year.between(2021, 2024)].sort_values("utc_datetime")
    price_path = output / "electricity_prices_15min_de_2021_2024.parquet"
    price.to_parquet(price_path, index=False, compression="zstd")
    results["prices"] = validate_time_table(price, "utc_datetime", "15min", "EUR/MWh", price_path)

    holiday = pd.read_csv(source_dir / Path(SELECTED["holidays"]).name)
    holiday["date"] = pd.to_datetime(holiday["date"], errors="raise")
    holiday = holiday[holiday.date.dt.year.between(2021, 2024)].sort_values("date")
    holiday_path = output / "holidays_karlsruhe_bw_2021_2024.parquet"
    holiday.to_parquet(holiday_path, index=False, compression="zstd")
    results["holidays"] = validate_time_table(holiday, "date", "1D", "local calendar", holiday_path)

    weather = pd.read_csv(source_dir / Path(SELECTED["weather"]).name)
    weather["utc_datetime"] = pd.to_datetime(weather["utc_datetime"], utc=True, errors="raise")
    weather = weather[weather.utc_datetime.dt.year.between(2021, 2024)].sort_values("utc_datetime")
    weather_path = output / "weather_karlsruhe_hourly_original_2021_2024.parquet"
    weather.to_parquet(weather_path, index=False, compression="zstd")
    results["weather"] = validate_time_table(weather, "utc_datetime", "1h", "published original units", weather_path)

    for label, result in results.items():
        result["source"] = manifest[label]
        result["output_sha256"] = file_digest(ROOT / result["path"])
        result["values_transformed"] = False
    return results


def validate_time_table(frame: pd.DataFrame, column: str, frequency: str, unit: str, path: Path) -> dict:
    timestamp = frame[column]
    expected = pd.date_range(timestamp.min(), timestamp.max(), freq=frequency)
    return {"path": str(path.relative_to(ROOT)), "rows": len(frame), "time_column": column,
            "timezone": "UTC" if column == "utc_datetime" else "Europe/Berlin local calendar date",
            "frequency": frequency, "unit": unit, "start": timestamp.min().isoformat(),
            "end": timestamp.max().isoformat(), "unique": bool(timestamp.is_unique),
            "monotonic": bool(timestamp.is_monotonic_increasing),
            "complete_grid": bool(len(expected) == len(timestamp) and timestamp.reset_index(drop=True).equals(
                pd.Series(expected, name=column))) }


def main() -> None:
    archive = acquire()
    sources, emissions = extract_sources(archive)
    canonical = canonicalize(sources)
    report = {
        "status": "CANONICAL_EXOGENOUS_SILVER_VALIDATED",
        "official_archive": {"path": str(archive.relative_to(ROOT)), "bytes": archive.stat().st_size,
                             "md5": file_digest(archive, "md5"), "sha256": file_digest(archive)},
        "sources": sources, "canonical": canonical,
        "emissions": sources.get("emissions_catalogue_only", {"status": "not present"}),
        "note": "Canonical files preserve source values; no unit conversion, filling, or interpolation was applied.",
    }
    target = ROOT / "reports" / "exogenous_canonicalization.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
