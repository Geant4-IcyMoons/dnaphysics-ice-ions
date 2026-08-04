#!/usr/bin/env python3
"""Generate oxygen charge-changing cross sections with the shared CTMC engine.

Oxygen-specific physics data
----------------------------
The O0--O7+ ground configurations, outer-shell occupancies, and successive
ionization energies are from NIST Atomic Spectra Database 5.12:

    https://physics.nist.gov/cgi-bin/ASD/ie.pl?units=1&spectra=O&shells_out=on
    https://doi.org/10.18434/T4W30F

The bare O-16 nuclear mass is derived from the NIST neutral-atom relative mass
15.99491461957(17) u by subtracting eight electron masses and restoring the
summed O0--O7+ electronic binding energy.

The N=7 and N=8 spectator-screening rows are taken directly from Table I of:

    R. H. Garvey, C. H. Jackman, and A. E. S. Green,
    Phys. Rev. A 12 (1975) 1144--1152.
    https://doi.org/10.1103/PhysRevA.12.1144

The three-body trajectory, water target, event classification, and IEVM/IPM
construction follow:

    T. Liamsuwan and H. Nikjoo, Phys. Med. Biol. 58 (2013) 641--672.
    https://doi.org/10.1088/0031-9155/58/3/641

Published CTMC calculations for bare O8+ impact on water independently
establish applicability of CTMC to this projectile-target pair:

    A. Jorge et al., Phys. Rev. A 99 (2019) 062701.
    https://doi.org/10.1103/PhysRevA.99.062701

The numerical engine is implemented in :mod:`charge_exchange_ctmc`. Carbon
and oxygen deliberately have separate entry points, PBS scripts, output
directories, checkpoints, and metadata, while sharing the same equations.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import charge_exchange_ctmc


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shared engine with the immutable oxygen projectile definition."""
    return charge_exchange_ctmc.main("oxygen", argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrupted; the oxygen CTMC checkpoint was preserved.",
            file=sys.stderr,
        )
        raise SystemExit(130)
