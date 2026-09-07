# PWBA benchmark index

The shared all-state projectile benchmark evaluates PWBA and RPWBA against
the same atomic data, analytic limits, and refined momentum grids:

```bash
python -m physics.inelastic_dielectric.projectile_potentials.benchmarking.benchmark_projectile_form_factors --workers 10
python -m pytest -q tests/test_ion_dcs_export_grid.py tests/test_modular_layout.py
```

Its report and PDF are in `../../projectile_potentials/benchmarking/plots/form_factors/`.
The target sum-rule/stopping audit is in `../../k_shell/benchmarking/`.
These routines are shared rather than copied into two kernel directories.
The migration fixture checks 456 pre-move kernel values; it is numerical
regression evidence, not a reference stopping curve.
