# Frozen-projectile numerical benchmark

Date: 2026-09-07. Model: `frozen-spherical-hf-v1`.

**PASS for the checks below. Experimental stopping validation remains open.**
The model and its limitations are defined in
[PROJECTILE_FORM_FACTORS.md](../README.md).

## Data and protocol

The library covers 38 states: H0..H+, He0..He2+, C0..C6+, O0..O8+, and
S0..S16+. Five bare states and five hydrogenic states are analytic; the other
28 are computed with PySCF 2.14.0 using spherical monopoles of RHF/ROHF
ground-configuration determinants. Atomic calculations use ten independent
workers and one numerical-library thread per worker.

Atomic-data SHA256:
`62041208d54f95ad176b5e09f805eb3d34a75e255933a73e5acf6c4eda3616f7`.

The reproducible benchmark is
`../benchmarking/benchmark_projectile_form_factors.py`. Its JSON report records
software versions and source checksums. Default outputs are
`physics/inelastic_dielectric/projectile_potentials/benchmarking/plots/form_factors/benchmark.json` and
`projectile_form_factors.pdf` in that same directory. These generated
plots and diagnostics are ignored by Git.

The production-kernel check evaluates 1368 DCS samples: all states, both ice
phases, PWBA and RPWBA, 1/10/100 MeV/u, and three loss/channel choices:
15 eV excitation, 50 eV outer-shell ionisation, and 1000 eV K-continuum
ionisation. Each sample is evaluated at Nq=1000 and 2000. These are selected
kernel samples, not a full convergence campaign for every energy-loss bin
or integrated stopping curve.

## Maximum discrepancies

| Check | Observed maximum | Acceptance bound | Result |
| --- | ---: | ---: | --- |
| Analytic electron count | 7.99e-15 electrons | 1e-8 | PASS |
| Independent radial count | 7.11e-15 electrons | 3e-8 | PASS |
| Analytic transform versus radial Fourier quadrature | 4.22e-11 electrons | 3e-8 | PASS |
| Hydrogenic reference transform | 2.22e-16 electrons | 1e-14 | PASS |
| Interpolated charge amplitude, relative | 9.12e-10 | 2e-7 | PASS |
| Radial export versus direct PySCF AO density, scaled | 2.75e-15 | 1e-10 | PASS |
| QZ-to-5Z squared screening factor, relative | 0.00197013 | 0.005 | PASS |
| Nq=1000 versus 2000 DCS, relative | 3.86e-9 | 0.01 | PASS |

All sampled DCS values are finite and positive. QZ-to-5Z convergence is a
radial-basis check, not a bound on HF, spherical-monopole, or first-Born
physical error. Bare-state selections reproduce the existing default
kernel exactly; neutral amplitudes obey the expected low-k squared-charge
k^4 limit and remain nonzero at finite k.

## Independent atomic reference

The neutral-He comparison uses [Koga, Phys. Rev. A 41, 1274 (1990)](https://doi.org/10.1103/PhysRevA.41.1274),
Eq. (7) and Tables I--II. The published, unnormalized three-exponential
density is normalized to two electrons only within the benchmark. Its
rounded coefficients do not enter our atomic calculation or generator.

| Quantity | Maximum discrepancy | Acceptance bound | Result |
| --- | ---: | ---: | --- |
| Energy versus near-HF limit | 5.31e-5 hartree | 1e-4 | PASS |
| Form factor versus published density approximation | 5.22e-5 electrons | 1e-3 | PASS |
| First through fourth radial moments, relative | 0.00222917 | 0.005 | PASS |

This is an independent atomic HF reference for He. It is not an external
all-state validation and does not establish agreement with measured stopping.
No ICRU data, empirical effective charge, or fitted target multiplier was used.

## Integration scope

Regression checks cover target-channel integration, explicit-bare parity,
charge-state filename and cache separation, incompatible-cache rejection,
TCS integration of exported DCS, and exact serial/spawn-worker agreement.
The C++ transport charge ladder and projectile-changing processes are not
connected by this implementation. RPWBA for electron-bearing projectiles
uses the documented Breit/static electric-form-factor approximation; its
composite-projectile validity is not established by the numerical checks.

Final regression run: **394 passed** (179.63 s), using:

```bash
python -m pytest -q tests/test_projectile_form_factors.py \
  tests/test_projectile_relativistic_dcs.py tests/test_ion_dcs_export_grid.py \
  tests/test_ion_optical_normalization.py \
  tests/test_hydrogenic_kshell.py tests/test_barkas_dcs.py
```
