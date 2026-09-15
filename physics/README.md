# Physics

[Dielectric inelastic processes](inelastic_dielectric/README.md) and
[nuclear elastic collisions](elastic/README.md) have separate workflows.
The combined [handoff study](elastic/handoff/README.md) uses shared
[ice structures](../models/ice/README.md), outside the process directories.

CTMC and the Geant4 application remain separate from this elastic migration.
Executable diagnostic code does not establish physical qualification.

`constants.py` holds the common material and projectile constants for the
dielectric workflow; the retained elastic isotope conventions are documented
in `elastic/bca/config.py`.

[RT-TDDFT](low_energy/RT_TDDFT/README.md) prepares finite ice targets and
provides a runnable isolated H+--H2O collision and capture workflow. Local and
MPI execution checks pass; numerical convergence and comparison with published
capture results remain outstanding.
