#!/usr/bin/env python3
"""Generate carbon charge-changing cross sections with the shared CTMC engine.

Carbon-specific physics data
----------------------------
The C0--C6+ charge states, projectile binding energies, shell occupancies,
water orbitals, screened-core potential, and many-electron construction are
from:

    T. Liamsuwan and H. Nikjoo, Phys. Med. Biol. 58 (2013) 641--672.
    https://doi.org/10.1088/0031-9155/58/3/641

The numerical engine is implemented in :mod:`charge_exchange_ctmc`. Carbon
and oxygen deliberately have separate entry points, PBS scripts, output
directories, checkpoints, and metadata, while sharing the same equations.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import charge_exchange_ctmc


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shared engine with the immutable carbon projectile definition."""
    return charge_exchange_ctmc.main("carbon", argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrupted; the carbon CTMC checkpoint was preserved.",
            file=sys.stderr,
        )
        raise SystemExit(130)
