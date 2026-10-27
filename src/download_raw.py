"""Checksum-safe downloader for absent official SPARK top-level archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import requests

from timeseries_core import ROOT, load_config, verify_archive

DATASET_ID = "bjdg3m3rg5jv3skk"
PAGE = f"https://radar.kit.edu/radar/en/dataset/{DATASET_ID}"
FRONTEND = "https://radar.kit.edu/radar"


def discovery(session: requests.Session) -> dict[str, dict]:
    response = session.get(f"{FRONTEND}/ajax/getFolderContent", params={
        "folderId": DATASET_ID, "draw": 1, "start": 0, "length": 100,
        "order[0][column]": 0, "order[0][dir]": "asc"}, timeout=60)
    response.raise_for_status()
    items = [item for row in response.json().get("data", []) for item in row.get("children", [])]
    return {x["fileName"]: {"id": x["id"], "size": int(x["binary"]["size"]),
                                  "md5": x["binary"]["checksum"].lower()} for x in items}


def accept_terms(session: requests.Session) -> None:
    response = session.get(PAGE, timeout=60)
    response.raise_for_status()
    match = re.search(r'<meta name="_csrf" content="([^"]+)"', response.text)
    if not match:
        raise RuntimeError("RADAR CSRF token not found")
    accepted = session.post(f"{FRONTEND}/ajax/acceptDownloadTc", headers={
        "X-CSRF-TOKEN": match.group(1), "X-Requested-With": "XMLHttpRequest", "Referer": PAGE}, timeout=60)
    accepted.raise_for_status()


def download(signal_name: str) -> None:
    cfg = load_config()["signals"][signal_name]
    target = ROOT / cfg["archive"]
    if target.exists():
        verify_archive(target, cfg["archive_size"], cfg["archive_md5"])
        print(f"VERIFIED {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    archive_name = target.name
    with requests.Session() as session:
        remote = discovery(session)[archive_name]
        if remote["size"] != cfg["archive_size"] or remote["md5"] != cfg["archive_md5"]:
            raise RuntimeError(f"Official metadata changed for {archive_name}: {remote}")
        accept_terms(session)
        link = session.get(f"{FRONTEND}/file/getDownloadLink/{remote['id']}", headers={
            "X-Requested-With": "XMLHttpRequest", "Referer": PAGE}, timeout=60)
        link.raise_for_status()
        partial = target.with_suffix(target.suffix + ".part")
        digest = hashlib.md5()
        written = 0
        with session.get(link.json()["downloadUrl"], stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with partial.open("wb") as destination:
                for block in response.iter_content(8 * 1024 * 1024):
                    if block:
                        destination.write(block)
                        digest.update(block)
                        written += len(block)
                        if written // (512 * 1024 * 1024) != (written - len(block)) // (512 * 1024 * 1024):
                            print(json.dumps({"downloaded_bytes": written, "expected_bytes": remote["size"]}), flush=True)
        if written != remote["size"] or digest.hexdigest() != remote["md5"]:
            raise RuntimeError(f"Download validation failed; partial retained at {partial}")
        partial.replace(target)
    verify_archive(target, cfg["archive_size"], cfg["archive_md5"])
    print(f"DOWNLOADED_AND_VERIFIED {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("signal", choices=["total_load", "pv"])
    args = parser.parse_args()
    download(args.signal)


if __name__ == "__main__":
    main()
