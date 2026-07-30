#!/usr/bin/env python3
"""Generate sulfur charge-changing cross sections with the shared CTMC engine.

Sulfur-specific physics data
----------------------------
The S0--S15+ ground configurations, outer-shell occupancies, and successive
ionization energies are from NIST Atomic Spectra Database 5.12:

    https://physics.nist.gov/cgi-bin/ASD/ie.pl?units=1&spectra=S&shells_out=on
    https://doi.org/10.18434/T4W30F

The bare S-32 nuclear mass is derived from the NIST neutral-atom relative mass
31.9720711744(14) u by subtracting sixteen electron masses and restoring the
summed S0--S15+ electronic binding energy.

The N=11--N=16 spectator-screening rows are taken directly from Table I of:

    R. H. Garvey, C. H. Jackman, and A. E. S. Green,
    Phys. Rev. A 12 (1975) 1144--1152.
    https://doi.org/10.1103/PhysRevA.12.1144

The three-body trajectory, water target, event classification, and IEVM/IPM
construction follow:

    T. Liamsuwan and H. Nikjoo, Phys. Med. Biol. 58 (2013) 641--672.
    https://doi.org/10.1088/0031-9155/58/3/641

CTMC calculations covering all sulfur charge states against molecular
hydrogen provide independent element-level applicability evidence:

    D. R. Schultz et al., At. Data Nucl. Data Tables 142 (2021) 101443.
    https://doi.org/10.1016/j.adt.2021.101443

That sulfur-H2 work is not used as a source of substituted parameters for the
Liamsuwan--Nikjoo sulfur-H2O extension. No direct sulfur-H2O benchmark is
claimed.

The numerical engine is implemented in :mod:`charge_exchange_ctmc`. Carbon,
oxygen, and sulfur deliberately have separate entry points, PBS scripts,
output directories, checkpoints, and metadata, while sharing the same
equations.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import charge_exchange_ctmc


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shared engine with the immutable sulfur projectile definition."""
    return charge_exchange_ctmc.main("sulfur", argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrupted; the sulfur CTMC checkpoint was preserved.",
            file=sys.stderr,
        )
        raise SystemExit(130)
