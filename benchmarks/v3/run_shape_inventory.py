#!/usr/bin/env python3
"""Join one full FX capture to its selected matched-baseline trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shape_inventory import build_inventory, write_inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_report", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    capture = json.loads(args.capture_report.read_text())
    baseline = json.loads(args.baseline_report.read_text())
    inventory = build_inventory(capture, baseline)
    write_inventory(args.output, inventory)
    print({
        "cell_id": inventory["cell_id"],
        "output": str(args.output),
        "hot_shape_count": len(inventory["hot_shapes"]),
        "baseline": inventory["selected_baseline_id"],
        "coverage": inventory["coverage"],
    })


if __name__ == "__main__":
    main()
