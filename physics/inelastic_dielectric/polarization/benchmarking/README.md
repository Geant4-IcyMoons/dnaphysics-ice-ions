# Electronic excitation and ionisation benchmarks

The screened optical-oscillator calculation and its explicit physical
rejections are documented in [SCREENED_BARKAS.md](../SCREENED_BARKAS.md).
It is not an electron-gas calculation or a validated all-state ice DCS.
`tests/test_screened_barkas.py` checks the point-charge limit, radial fields,
cubic scaling, numerical integration, and the rejection of uncontrolled
additive corrections. Passing those rejection tests is not physical approval.

`check_screened_oscillator_nonlinear.py` independently integrates the full
nonrelativistic oscillator equation for all 38 frozen states of H, He, C, O,
and S, including neutrals and bare nuclei. Its 342 cases span three impact
parameters and three dimensionless frequencies per state. Electron-tail
quadrature supplies the field independently of the production radial-charge
and gradient routines; only the atomic density input is shared. Opposite
interaction signs and decreasing strengths isolate the cubic coefficient.
ODE tolerance, integration-window, radial-grid, strength-extrapolation, and
kernel-grid checks accompany independent linear Fourier-force integrals.
Bare kernels are also compared directly with Salvat at two velocities.

The all-state run passes the preassigned 0.1% relative checks: maximum cubic
coefficient discrepancy 0.00240%, largest refinement change 0.00824%, and
maximum bare Salvat discrepancy 0.06442%. These are numerical errors, not
physical uncertainties. Full-strength oscillator ratios are reported
separately; 139 of the 342 sampled encounters at the illustrative speed
v=5 atomic units have |cubic/leading| >= 1. Successful derivative extraction
does not establish that a perturbative truncation is accurate there.

This checks the screened oscillator calculation, not its ice-DCS
interpretation, screened relativistic extension, or unavailable
Schinner-Sigmund reference curves. No ICRU or Matias comparison is performed.
Outputs, settings and source/data checksums are under
`physics/inelastic_dielectric/polarization/benchmarking/plots/nonlinear_oscillator/`.

The PDF figure uses the shared AAS full width and paper font size, a
Courier family, panel letters, and a panel-height Plasma colorbar. Colors
identify b/r0; each curve takes the maximum over sampled oscillator
frequencies at that impact parameter. Solid curves show the cubic-coefficient
comparison; dashed curves show the largest refinement or linear check.
Re-render the saved numerical results without recomputing or changing their
original provenance:

```bash
python physics/inelastic_dielectric/polarization/benchmarking/check_screened_oscillator_nonlinear.py --plot-only
```

```bash
python physics/inelastic_dielectric/polarization/benchmarking/check_screened_oscillator_nonlinear.py --workers 10
python -m pytest -q tests/test_screened_oscillator_nonlinear.py tests/test_screened_barkas.py tests/test_barkas_dcs.py tests/test_projectile_form_factors.py
```

Use `--element He --charge 1` to restrict the benchmark. The default covers
every state. `tests/test_screened_barkas.py` separately exercises correction
routing and recorded physical rejections against both PWBA and RPWBA Born
kernels, with real small-grid table exports for each element's one-electron
state. Tests use temporary directories, not production cross-section files.
An additional 76-case synthetic-cache regression checks both kernels and
all charge states for charge provenance, single application of the correction,
and piecewise-linear TCS/CDF area consistency. These synthetic values test
the file interface only; they are not physical cross sections or simulation
results.
The combined software checks passed: 491 tests, including the explicit
rejection cases. No production physics formula changes were needed for this
extension of the benchmark. See the generation controls and limitations in
[SCREENED_BARKAS.md](../SCREENED_BARKAS.md#reproduction).

`benchmark_projectile_form_factors.py` checks the 38 fixed projectile states
of H, He, C, O, and S against analytic limits, independent radial Fourier
quadrature, atomic-basis convergence, and the production PWBA/RPWBA kernels.
Its report separates numerical acceptance from experimental validation.
See [the model definition](../../projectile_potentials/README.md) for equations,
provenance, commands, and the limitations for neutral and partially stripped
projectiles. Outputs go to `physics/inelastic_dielectric/projectile_potentials/benchmarking/plots/form_factors/`.

`benchmark_barkas_sbethe.py` compares the Barkas differential-correction
implementation with the independently distributed SBETHE v2 calculation. It
checks the downloaded archive checksum before compiling or executing it and
keeps formula parity separate from differences in the target optical data.

Primary reference: Salvat and Quesada, SBETHE v2 dataset,
<https://data.mendeley.com/datasets/7zw25f428t/2>.

Run from the repository root:

```bash
python \
  physics/inelastic_dielectric/polarization/benchmarking/benchmark_barkas_sbethe.py
```
