"""Consolidate explicit DST decisions and observations into one report."""

from __future__ import annotations

import json

from timeseries_core import ROOT


def main() -> None:
    source = ROOT / "reports" / "dst"
    output = source / "dst_audit_all.json"
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(source.glob("*.json"))
               if path != output]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"reports": reports}, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
