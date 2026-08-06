#!/usr/bin/env python3
"""Collect mixed-CDFT gaps and couplings for low-energy capture channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from low_energy_charge_exchange import collect_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_path, collection_path = collect_workflow(
        args.manifest, output_directory=args.output_directory
    )
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    print(f"Coupling table: {table_path}")
    print(f"Manifest: {collection_path}")
    print(f"Numerical status: {collection['numerical_status']}")
    print(f"Physics status: {collection['physics_status']}")
    print(f"Cross-section status: {collection['cross_section_status']}")
    if args.require_complete and collection["numerical_status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
