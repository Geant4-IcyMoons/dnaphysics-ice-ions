#!/usr/bin/env python3
"""Generate restartable NLH H/O binary-collision kernels for ion transport."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bca.config import (  # noqa: E402
    DEFAULT_ENERGY_MAX_EV,
    DEFAULT_ENERGY_MIN_EV,
    DEFAULT_ENERGY_POINTS,
    DEFAULT_IMPACT_POINTS,
    DEFAULT_MINIMUM_TURNING_POTENTIAL_EV,
    DEFAULT_PROJECTILES,
    DEFAULT_QUADRATURE_ORDER,
    DEFAULT_WORKERS,
)
from bca.tables import KernelTableConfig, generate_kernel_tables  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projectiles", nargs="+", default=DEFAULT_PROJECTILES)
    parser.add_argument("--energy-min-ev", type=float, default=DEFAULT_ENERGY_MIN_EV)
    parser.add_argument("--energy-max-ev", type=float, default=DEFAULT_ENERGY_MAX_EV)
    parser.add_argument("--energy-points", type=int, default=DEFAULT_ENERGY_POINTS)
    parser.add_argument("--impact-points", type=int, default=DEFAULT_IMPACT_POINTS)
    parser.add_argument(
        "--minimum-turning-potential-ev",
        type=float,
        default=DEFAULT_MINIMUM_TURNING_POTENTIAL_EV,
        help="NLH hard-collision boundary (default: 30 eV; published minimum: 10 eV).",
    )
    parser.add_argument(
        "--quadrature-order", type=int, default=DEFAULT_QUADRATURE_ORDER
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=HERE / "collision_kernels",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute all blocks even when matching checkpoints exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = KernelTableConfig(
        projectiles=tuple(args.projectiles),
        energy_min_ev=args.energy_min_ev,
        energy_max_ev=args.energy_max_ev,
        energy_points=args.energy_points,
        impact_points=args.impact_points,
        minimum_turning_potential_ev=args.minimum_turning_potential_ev,
        quadrature_order=args.quadrature_order,
        workers=args.workers,
    )
    print(
        f"Grid: {len(config.projectiles)} projectiles x 2 targets x "
        f"{config.energy_points} energies x {config.impact_points} impact points"
    )
    print(f"Workers: {config.workers}")
    print(
        "Scope: independent-atom hard-collision kernels; these are not yet "
        "phase-resolved ice cross sections."
    )
    csv_path, manifest_path = generate_kernel_tables(
        config,
        args.output_directory,
        resume=not args.no_resume,
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
