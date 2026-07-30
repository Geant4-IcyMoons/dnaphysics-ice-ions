#!/usr/bin/env python3
"""Generate Li-7 charge-changing cross sections with the shared CTMC engine.

Lithium-specific physics data
-----------------------------
The process definitions, charge states, and energy grid are from:

    N. D-Kondo et al., Phys. Med. Biol. 69 (2024) 145016.
    https://doi.org/10.1088/1361-6560/ad5f72

That work calculates the six single-electron transitions Li1+->Li0,
Li2+->Li1+, Li3+->Li2+, Li0->Li1+, Li1+->Li2+, and Li2+->Li3+ in H2O at
ten logarithmically spaced energies from 1 keV/u to 10 MeV/u. This entry point
preserves all ten points and appends a directly calculated endpoint at
100 MeV total Li-7 energy (100000/7 keV/u). The appended point is an explicit
model extension, not a value attributed to the paper.

The paper refers to the Liamsuwan--Nikjoo and Tran et al. CTMC work for the
initial-condition construction used by the shared engine:

    https://doi.org/10.1088/0031-9155/58/3/641
    https://doi.org/10.1016/j.nimb.2015.10.017

The Li0--Li2+ ground configurations, outer-shell occupancies, and successive
ionization energies are from NIST Atomic Spectra Database 5.12:

    https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Li
    https://doi.org/10.18434/T4W30F

The bare Li-7 nuclear mass is derived from the NIST neutral-atom relative
mass 7.0160034366(45) u by subtracting three electron masses and restoring
the summed Li0--Li2+ electronic binding energy.

No numerical cross section is digitized from a plot. Unpublished
impact-parameter cutoffs and start separations remain mandatory explicit
convergence inputs. The generator does not invent a direct Li3+->Li1+
double-capture channel.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import charge_exchange_ctmc


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shared engine with the immutable Li-7 projectile definition."""
    return charge_exchange_ctmc.main("lithium", argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrupted; the lithium CTMC checkpoint was preserved.",
            file=sys.stderr,
        )
        raise SystemExit(130)
