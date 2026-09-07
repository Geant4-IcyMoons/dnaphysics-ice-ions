# RPWBA benchmark index

```bash
python -m pytest -q tests/test_projectile_relativistic_dcs.py tests/test_modular_layout.py
python -m physics.inelastic_dielectric.projectile_potentials.benchmarking.benchmark_projectile_form_factors --workers 10
```

The checks cover the vacuum/medium transverse ratio, dilute limit,
relativistic momentum bounds, mass/energy conventions, and all fixed charge
states. The shared report and PDF are under
`../../projectile_potentials/benchmarking/plots/form_factors/`.
No duplicate atomic data or second copy of the benchmark is maintained here.
