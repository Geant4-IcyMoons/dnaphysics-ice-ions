# Projectile densities and potentials

```bash
python -m physics.inelastic_dielectric.projectile_potentials.benchmarking.benchmark_projectile_form_factors --workers 10
python -m pytest -q tests/test_projectile_form_factors.py
```

The benchmark covers all 38 H/He/C/O/S states: electron counts, radial
Fourier transforms, analytic hydrogenic/bare limits, basis convergence,
helium reference checks, and PWBA/RPWBA integrands. Figures and reports use
`plots/form_factors/` beneath this directory. They are Git-ignored.

[REFERENCE.md](REFERENCE.md) retains the original evidence and its physical
limitations. Atomic input regeneration is a separate, optional PySCF step
documented in [the component README](../README.md), not a generation-time
dependency or a fit to stopping-power curves.
