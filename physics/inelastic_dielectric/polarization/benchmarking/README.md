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

The original stored all-state run passed the 0.1% relative checks: maximum cubic
coefficient discrepancy 0.00240%, largest refinement change 0.00824%, and
maximum bare Salvat discrepancy 0.06442%. These are numerical errors, not
physical uncertainties. Full-strength oscillator ratios are reported
separately; 139 of the 342 sampled encounters at the illustrative speed
v=5 atomic units have |cubic/leading| >= 1. Successful derivative extraction
does not establish that a perturbative truncation is accurate there.

This checks the screened oscillator calculation, not its ice-DCS
interpretation, screened relativistic extension, or the
Schinner-Sigmund reference curves. No ICRU or Matias comparison is performed.
Outputs, settings and source/data checksums are under
`physics/inelastic_dielectric/polarization/benchmarking/plots/nonlinear_oscillator/`.

## Adaptive-integration audit (2026-09-08)

`check_integration_failures.py` separates numerical convergence from physical
acceptance using the production phase-resolved/adaptive integrator. Its
report is `benchmarking/runs/integration/integration_report.json`.
Reproduce from the repository root:

```bash
python -m physics.inelastic_dielectric.polarization.benchmarking.check_integration_failures --workers 10 --loss-points 24 --dq 1000
```

The sampled states are H0, He0, He+, and He2+, with total kinetic energies
0.1, 0.3, 1, 3, 10, 30, 40, and 100 MeV. Twenty-four logarithmic loss nodes
are supplemented by 15, 50, and 1000 eV and nodes immediately around the
O K edge when kinematically allowed. Both phases and PWBA/RPWBA are tested.
The benchmark reuses exactly identical full target-response calls between
Born channels; a bitwise regression checks that this changes no cross section.

- 678 screened kernel nodes: no numerical failures.
- Before removal, a comparison with the former integrator found a maximum
  difference of 0.004300%, with no numerical failures from either method.
  This historical comparison is not a retained second numerical backend.
- New relative tolerance: 0.01%, versus the former 1%, with the unchanged
  absolute dimensionless floor 1e-10*Z^3. Impact, time, and tail errors are
  checked separately; none is compared to the Born magnitude for acceptance.
- All 128 sampled phase/kernel/energy baselines are finite and nonnegative;
  no negative or nonfinite corrected totals were found.
- 60 of 128 correction-on cases still fail the screened correction/Born
  guard. These are physical-acceptance failures, not numerical failures.

| State | Sampled total energies rejected in both phases and both kernels |
| --- | --- |
| H0 | 0.1, 0.3, 1, 3 MeV |
| He0 | All eight sampled energies |
| He+ | 0.1, 0.3, 1 MeV |
| He2+ | None; the unchanged bare Salvat dispatch is used |

For He0 at 40 MeV total and W=15 eV, amorphous PWBA gives Born
7.9918712449e-25 m2/eV and correction 2.2684164884e-24 m2/eV, a ratio of
2.8384046976. The kernel's estimated relative numerical error is 2.61e-10.
Ratios are 2.79344 for amorphous RPWBA, 2.65872 for hexagonal PWBA, and
2.61660 for hexagonal RPWBA. Thus the previously recorded failure survives
an independently discretized integral and dq=1000 Born evaluation.

After legacy removal and benchmark migration, all 736 pipeline tests passed
in ten deterministic shards, including generation/export, cache provenance,
independent ODE and Salvat comparisons, and close-collision matching. One
pre-existing invalid-escape SyntaxWarning was reported by the layout scan.
The all-state tests retain the previously
recorded C/O/S physical rejections at their sampled energies. No complete
production-grid or experimental validation is claimed, and no production
tables, potentials, target response, or acceptance guards were changed.

All current oscillator benchmarks use `oscillator_quadrature.py`. The
nonlinear ODE and bare Salvat formulas supply independent references; no
fixed-grid oscillator fallback is retained. Regenerate numerical reports
when benchmarking the current implementation. Existing figure reports
retain their original provenance and must not be relabelled as fresh runs.

## Figure and reference checks

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

`compare_close_collisions.py` compares the existing cutoff with an explicitly
adapted Section 4 close/distant matching prescription. It retains the same
screened forces and ice OOS in both, at common nonrelativistic kinematics.
See [derivation and limitations](../CLOSE_COLLISIONS.md). The generator is
unchanged; numerical PASS is not acceptance of the matched approximation.

Run from the repository root:

```bash
python \
  physics/inelastic_dielectric/polarization/benchmarking/benchmark_barkas_sbethe.py
```
