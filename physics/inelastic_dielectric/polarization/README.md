# Nonlinear polarization

The only correction backend is the full nonlinear frozen-projectile oscillator.
It is used for bare, partially stripped and neutral H, He, C, O and S. Cubic
truncation and analytic point-projectile Barkas are no longer generator options.

## Code

- `nonlinear_oscillator.py`: frozen radial fields, full/leading encounter
  equations, rotating-amplitude integration and impact quadrature.
- `nonlinear_polarization.py`: optical spectral assignment, kinematics,
  loss-level checkpoints and energy-parallel execution.
- `correction.py`: OOS normalization, scalar charge conventions, assembly
  with PWBA/RPWBA Born, TCS and stopping diagnostics.
- `diagnostic_tables.py`: flagged raw estimates, separate from transport products.
- `plot_correction.py`: Courier/Plasma PDF diagnostics from saved components.
- [benchmarking](benchmarking/README.md): retained full-strength comparison.

## Generation

```bash
ICE_TYPE=amorphous python -m physics.inelastic_dielectric.generate_cross_sections \
  --projectile He --charge-state 1 --include-barkas-dcs=true \
  --relativistic-projectile-dcs=true --energy-unit total \
  --energy-min-MeV 0.1 --energy-max-MeV 100 --dE 1000 --dq 1000
```

The existing `--include-barkas-dcs` switch and `barkas_*` data names remain
for active job/table consumers. They now select/store nonlinear polarization,
including higher even and odd terms, not a pure cubic Barkas term. Correction-off
runs retain Born. The model metadata is `frozen-full-nonlinear-oscillator-v1`.
Older corrected caches and energy patches are rejected, not relabelled.
Regenerate polarization-on tables; Born-only tables are unchanged.
Use new output/cache directories, or explicitly replace old table ranges
with `--no-merge-energy-patches`. Do not merge old cubic and nonlinear patches.

The correction is full minus leading within the same oscillator model, added
once to the independent Born DCS. Channel allocation is bookkeeping. Negative
final DCS and nonconverged integrals cannot be exported for transport.
Correction/Born magnitude is reported, not used as a cubic-validity cutoff.

This remains an experimental ice spectral approximation. Removing truncation
improves the oscillator solution; it does not establish exact ice DCS or fully
relativistic nonlinear response. See [equations and limits](NONLINEAR_POLARIZATION.md).
