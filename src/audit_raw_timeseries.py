"""Run or display raw-quality audits for the selected SPARK signals."""

from __future__ import annotations

import argparse
import json

from timeseries_core import ROOT, load_config, process_signal_year


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("signal", choices=["total_load", "pv"])
    parser.add_argument("--year", type=int, action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--show-only", action="store_true")
    args = parser.parse_args()
    years = args.year or load_config()["signals"][args.signal]["years"]
    for year in years:
        report = ROOT / "reports" / "raw_quality" / f"{args.signal}_{year}.json"
        if args.show_only:
            print(report.read_text(encoding="utf-8"))
        else:
            result = process_signal_year(args.signal, year, args.force)
            print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
