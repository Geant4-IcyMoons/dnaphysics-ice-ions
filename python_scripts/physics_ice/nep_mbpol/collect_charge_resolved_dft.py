#!/usr/bin/env python3
"""Collect counterpoise-corrected fixed-charge ion--H2O DFT potentials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soft_dft import collect_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless every task and CDFT constraint is complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path, manifest_path = collect_workflow(
        args.manifest, output_directory=args.output_directory
    )
    result = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Table: {csv_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Numerical status: {result['numerical_status']}")
    print(f"Physics status: {result['physics_status']}")
    if args.require_complete and result["numerical_status"] != "calculations_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
