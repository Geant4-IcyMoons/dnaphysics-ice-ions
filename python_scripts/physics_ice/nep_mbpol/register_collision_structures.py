#!/usr/bin/env python3
"""Validate equilibrated ice snapshots and write a collision-input registry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bca.structure import load_ice_structure  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structures", nargs="+", type=Path)
    parser.add_argument(
        "--frame",
        default="last",
        help="Frame index to register, or 'last' (default).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "collision_structures.json",
    )
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="Register non-accepted snapshots as diagnostic-only inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_index = -1 if args.frame.lower() == "last" else int(args.frame)
    structures = [
        load_ice_structure(
            path,
            frame_index=frame_index,
            allow_unvalidated=args.allow_unvalidated,
        )
        for path in args.structures
    ]
    output = args.output.expanduser().resolve()
    records = []
    for structure in structures:
        record = structure.manifest_record()
        record["path"] = os.path.relpath(structure.source_path, output.parent)
        records.append(record)
    registry = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "all_collision_ready": all(item.collision_ready for item in structures),
        "path_base": "directory containing this registry",
        "structures": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Wrote {output}")
    for structure in structures:
        print(
            f"{structure.phase}: frame {structure.frame_index}, "
            f"{structure.atom_count:,} atoms, {structure.density_g_cm3:.6f} g/cm^3, "
            f"{structure.use_class}"
        )


if __name__ == "__main__":
    main()
