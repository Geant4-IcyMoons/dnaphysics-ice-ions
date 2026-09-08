# Nonlinear polarization checks

## Retained comparison

```bash
python -m physics.inelastic_dielectric.polarization.benchmarking.compare_full_strength_oscillator --workers 10
```

Outputs are `plots/full_strength_oscillator/comparison.json` and
`plots/full_strength_oscillator/full_strength_oscillator.pdf`.
The PDF uses the common Courier/Plasma style. `--plot-only` redraws the
saved report without recalculating or changing its provenance.

The curves compare the same classical oscillator at leading order, leading
plus its cubic derivative, and full physical interaction strength. The
derivative is extracted with sign reversal and Richardson extrapolation,
only in this benchmark. It is not a production option or an independent
ice-DCS/experimental validation.

The default matrix has 216 encounters: H+, H0, He+, He0; v=3,5,10 atomic
units; oscillator energies 15 and 50 eV; nine b/b_min values from 1 to 16.
This is nonrelativistic, with b_min=0.5616/v bohr. Reversing the entire
potential isolates odd/even terms, not another ionic state. Checks cover ODE
tolerance, radial-field refinement, time windows, leading Fourier integrals
and derivative extrapolation. Signed differences are retained.

## Regression tests

`tests/test_nonlinear_oscillator.py` compares full solutions with separately
integrated displacement equations and the point-field leading Bessel response.
It also checks normalized fields for all 38 states, the impact Jacobian,
velocity reconstruction, units, cutoff and signed assembly.
`tests/test_full_strength_oscillator.py` tests the retained comparison.

`test_nonlinear_polarization.py`, `test_polarization_assembly.py` and
`test_polarization_stacking.py` check metadata, cache reuse, final TCS/CDF
densities and PWBA/RPWBA stacking. Expensive interface matrices use an
explicitly named synthetic kernel fixture, not physics validation. The Born
kernels in the stacking checks are real.
`test_screened_polarization_checkpoints.py` includes real nonlinear loss
integrals and serial/parallel checkpoint reuse.

```bash
python -m pytest -q tests/test_nonlinear_oscillator.py tests/test_full_strength_oscillator.py \
  tests/test_nonlinear_polarization.py tests/test_polarization_assembly.py \
  tests/test_polarization_stacking.py tests/test_screened_polarization_checkpoints.py
```

Saved spectra can be inspected with `polarization/plot_correction.py`.
No production-energy campaign or ICRU/Matias comparison is implied.
Old cubic, close-collision crossover and SBETHE-only benchmark code is removed.

## Replacement checks (2026-09-08)

- All 729 inelastic-pipeline tests passed across focused and broader regression
  runs. Materials-library tests were outside this change. Three interface
  shards were interrupted when their old tests triggered large real integrals;
  they were updated to the explicit synthetic fixture and all 66 cases rerun
  successfully. This is separate from the real-kernel checks below.
- The regenerated 216-case comparison passed every convergence check, with
  no unresolved odd residuals. Maximum relative numerical check: 9.75e-5.
  Leading plus cubic differs from full response by over 10% in 36 cases.
  The largest relative truncation error is 2.15686, not an error in the
  nonlinear solution. These are encounter energies, not ice stopping powers.
- Real nonlinear loss integrals and restart checks passed. Sixteen small
  exports used H+, H0, He2+ and He0 at 10 MeV total energy, both phases and
  PWBA/RPWBA. The two loss nodes (15 and 50 eV) test assembly and final
  TCS/CDF consistency, not production-grid convergence or core sampling.
- For H0 at 10 MeV total, M/m_e=1837.152673 and W=15 eV, K_full=156.3217441.
  Increasing x_max from 50 to 100 and tightening impact rtol from 1e-3 to
  1e-4 left the reported value unchanged. This selected-state check is not
  an all-state or all-energy impact-tail bound.

The nonlinear spectral assignment and electric-field relativistic extension
remain model assumptions. These results do not establish agreement with
ICRU/Matias or transport validity over an entire requested energy range.
