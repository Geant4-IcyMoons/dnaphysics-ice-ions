#!/usr/bin/env python3
"""Run one restart-safe shard of a low-energy mixed-CDFT capture workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from low_energy_charge_exchange import run_workflow_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--cp2k-command",
        help="Command prefix, for example 'mpiexec -n 8 cp2k.psmp'.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completed, skipped = run_workflow_tasks(
        args.manifest,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        cp2k_command=args.cp2k_command,
        progress=not args.no_progress,
    )
    print(f"CP2K calculations completed this invocation: {completed}")
    print(f"Checksum-compatible calculations skipped: {skipped}")


if __name__ == "__main__":
    main()
