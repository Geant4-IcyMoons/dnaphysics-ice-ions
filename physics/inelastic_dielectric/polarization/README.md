# Polarization correction

Production generation uses two implementations:

- `barkas_dcs.py`: the Salvat/SBETHE point-projectile Barkas kernel, optical
  OOS normalization, charge scaling, final-DCS assembly, and diagnostics.
- `screened_barkas.py`: the frozen-projectile nonlinear-oscillator
  approximation for electron-bearing states. See [equations and limits](SCREENED_BARKAS.md).

The umbrella name is **polarization**, not a claim that the screened formula
is an established Barkas correction for every ion or neutral atom. Bare
states dispatch to Salvat; screened states retain the experimental model
identity and its convergence/perturbative rejection checks.

`--include-barkas-dcs=true` remains the generation switch. The corrected
table is Born plus correction, with the same interaction charge in the
quadratic and cubic terms. No scalar rescaling is substituted for the
additive kernel, and no Bloch DCS is included.

## Benchmarks

The [close-collision comparison](CLOSE_COLLISIONS.md) adds a nonrelativistic
Section-4-inspired alternative to the impact cutoff. It is benchmark-only;
the production prescription is not replaced. Its shift is evaluated from
the existing atomic potentials, with no fitted screening radius.

All routines and their figure/report folders are under `benchmarking/`:

```bash
python -m physics.inelastic_dielectric.polarization.benchmarking.check_screened_oscillator_nonlinear --workers 10
python -m physics.inelastic_dielectric.polarization.benchmarking.check_screened_oscillator_nonlinear --plot-only
python -m physics.inelastic_dielectric.polarization.benchmarking.compare_screened_barkas_point_projectiles --projectile proton --workers 10
python -m physics.inelastic_dielectric.polarization.benchmarking.compare_screened_barkas_point_projectiles --projectile alpha --workers 10
python -m physics.inelastic_dielectric.polarization.benchmarking.compare_close_collisions --workers 10
python -m physics.inelastic_dielectric.polarization.benchmarking.benchmark_barkas_sbethe
```

The all-state figure and its original numerical report are in
`benchmarking/plots/nonlinear_oscillator/`. Point-proton and alpha comparisons
use `benchmarking/plots/point_proton/` and `point_alpha/`. SBETHE output uses
`benchmarking/plots/sbethe_v2/`; its downloaded Fortran/build work is separate
under `benchmarking/runs/`. The experimental matching comparison uses
`benchmarking/plots/close_collisions/`. Plots and generated reports are Git-ignored.

`--plot-only` requires an existing report and does not run a numerical
benchmark or rewrite its original provenance. A fresh clone must first run
the benchmark or receive the saved report. The oscillator benchmark checks
the numerical cubic coefficient; it does not validate ice DCS or screened
relativity. See [the benchmark record](benchmarking/README.md).
