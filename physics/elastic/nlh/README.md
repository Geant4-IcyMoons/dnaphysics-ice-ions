# NLH projectile--water pair potentials

This directory contains the published Nordlund--Lehtola--Hobler (NLH)
short-range repulsive potentials needed for projectiles H, He, C, O, and S
against the H and O atoms of water ice. There are ten directional
projectile--target combinations and nine unique nuclear pairs because H--O
and O--H are identical.

For internuclear separation `r` in angstrom,

```text
phi(r) = sum_i a_i exp(-b_i r)
V(r) = (e^2 / 4 pi epsilon_0) Z1 Z2 phi(r) / r .
```

`coefficients.csv` is the required subset of the corrected coefficient table
published by the authors. `potential.py` evaluates the screening function,
potential, derivative, and outward radial force for scalar or NumPy-array
distances. For example:

```python
from nlh import potential_ev, radial_force_ev_per_angstrom

energy = potential_ev(0.1, "S", "O")
force = radial_force_ev_per_angstrom(0.1, "S", "O")
```

The functions reject values below 10 eV by default. The paper identifies
`V_rep >= 10 eV` as the region where a purely repulsive pair description can
be assumed, and reports its strongest general agreement above 30 eV. The
pair-specific RMS errors at both thresholds are retained in the CSV and the
`NLHCoefficients` object. Setting `enforce_fit_domain=False` is provided only
for diagnostics; it does not make the low-energy extrapolation physical.

These potentials are nuclear and charge-state independent: H, He, C, O, and S
refer to atomic number, not the projectile's instantaneous ionic charge.

## Coupling boundary

This implementation supplies the authoritative short-range pair energy and
force. It does not yet add projectile atoms to the two-species NEP-MB-pol
model. NEP-MB-pol begins with `nep4 2 O H` and treats only atoms belonging to
the water target; a projectile, including an H or O projectile, must retain a
separate dynamical role.

Do not splice NLH directly to the existing pretrained NEP-MB-pol potential
with an arbitrary distance blend. The systematic literature route fixes the
repulsive pair term, subtracts it from projectile-containing DFT training
energies, and trains the many-body residual. Until such a residual model or a
validated species-aware trajectory coupling is available, use this module to
verify/generate short-range pair values rather than claiming a complete
NLH--NEP ice force field.

## Sources

- K. Nordlund, S. Lehtola, and G. Hobler, *Physical Review A* **111**, 032818
  (2025), <https://doi.org/10.1103/PhysRevA.111.032818>.
- Corrected open dataset, Zenodo record 17302337,
  <https://doi.org/10.5281/zenodo.17302337>.
