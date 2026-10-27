from __future__ import annotations

import argparse
import json

from timeseries_core import load_config, process_signal_year


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TotalLoad silver data directly from 5-second P_total")
    parser.add_argument("--year", type=int, action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    years = args.year or load_config()["signals"]["total_load"]["years"]
    for year in years:
        print(json.dumps(process_signal_year("total_load", year, args.force), default=str))


if __name__ == "__main__":
    main()
