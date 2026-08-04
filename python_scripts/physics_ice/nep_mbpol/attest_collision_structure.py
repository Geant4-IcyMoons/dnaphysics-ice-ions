#!/usr/bin/env python3
"""Attest an ice trajectory after independent structural acceptance tests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bca.structure import file_sha256, load_ice_structure  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure", type=Path)
    parser.add_argument("--phase", required=True)
    parser.add_argument(
        "--validation-report",
        action="append",
        required=True,
        type=Path,
        help="Accepted RDF/order/thermodynamic report; repeat for multiple files.",
    )
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--confirm-accepted",
        action="store_true",
        help="Confirm that the documented phase-specific acceptance tests passed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_accepted:
        raise ValueError(
            "Refusing to mark the structure collision-ready without "
            "--confirm-accepted."
        )
    structure_path = args.structure.expanduser().resolve()
    reports = [path.expanduser().resolve() for path in args.validation_report]
    missing = [path for path in reports if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    # Parse and validate H2O/PBC before creating the attestation.  This does not
    # itself establish phase quality; that evidence belongs to the reports.
    structure = load_ice_structure(structure_path, allow_unvalidated=True)
    output = (
        args.metadata.expanduser().resolve()
        if args.metadata
        else structure_path.with_suffix(".json")
    )
    existing: dict[str, object] = {}
    if output.is_file():
        value = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Existing metadata must be a JSON object.")
        existing = value
    existing.update(
        {
            "phase": args.phase,
            "state": "equilibrated and structurally validated for collision sampling",
            "collision_ready": True,
            "collision_attested_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": structure.source_sha256,
            "atoms": structure.atom_count,
            "water_molecules": structure.water_molecule_count,
            "density_g_cm3": structure.density_g_cm3,
            "validation_reports": [
                {
                    "path": os.path.relpath(path, output.parent),
                    "sha256": file_sha256(path),
                }
                for path in reports
            ],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Wrote {output}")
    print(
        f"Attested {args.phase}: {structure.atom_count:,} atoms, "
        f"{structure.density_g_cm3:.6f} g/cm^3"
    )


if __name__ == "__main__":
    main()
