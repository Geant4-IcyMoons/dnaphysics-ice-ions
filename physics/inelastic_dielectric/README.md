# Dielectric inelastic cross sections

One entry point assembles the phase-specific ice response, projectile density,
PWBA/RPWBA kernel, oxygen K continuum, and optional polarization correction:

```bash
python -m physics.inelastic_dielectric.generate_cross_sections --help
```

Run commands from the repository root. Direct execution of
`physics/inelastic_dielectric/generate_cross_sections.py` is also supported.
The former `python_scripts/` commands are deliberately absent on this branch.

## Layout

| Location | Responsibility |
| --- | --- |
| `generate_cross_sections.py` | CLI, assembly, workers, cache/provenance checks, DAT/NPZ export |
| [pwba](pwba/README.md) | Nonrelativistic projectile integrands and shared kernel state |
| [rpwba](rpwba/README.md) | Relativistic longitudinal/transverse integrands |
| [finite_q](finite_q/README.md) | Optical-limit and finite-q ice response |
| [k_shell](k_shell/README.md) | Published hydrogenic oxygen continuum and sum-rule audit |
| [projectile_potentials](projectile_potentials/README.md) | Frozen densities, fields, form factors, and atomic-data generation |
| [polarization](polarization/README.md) | Salvat Barkas and screened-oscillator corrections |
| `numerics.py` | Shared quadrature, unchanged by the relocation |
| [validation](validation/README.md) | Normalization, applicability, and numerical evidence |
| `jobs/generate_cross_sections.pbs` | Optional PBS launcher |

Each benchmark lives with its component. Figures and their numerical reports
go under that component's `benchmarking/plots/`; external build/download work
goes under `benchmarking/runs/`. These directories are Git-ignored.
Regression tests are collected in the repository's `tests/` directory.

## Generation

PWBA, amorphous ice, protons, no polarization correction:

```bash
ICE_TYPE=amorphous python -u -m physics.inelastic_dielectric.generate_cross_sections \
  --projectile proton --include-kshell --kshell-model hydrogenic-gos \
  --energy-min-MeV 0.1 --energy-max-MeV 100 --energy-unit total \
  --energy-points 1000 --dE 1000 --dq 1000 --include-barkas-dcs=false
```

For RPWBA add `--relativistic-projectile-dcs=true`. For the polarization
correction use `--include-barkas-dcs=true`. The historical CLI name and
`_barkas_dcs` table tag are retained to identify compatible tables; that tag
means **Born plus correction**, not correction alone. Use
`ICE_TYPE=hexagonal` for the other phase. Default generation is PWBA,
hydrogenic K shell included, no polarization correction.

Fixed charge-state example (C3+, RPWBA, no polarization):

```bash
ICE_TYPE=hexagonal python -u -m physics.inelastic_dielectric.generate_cross_sections \
  --projectile C --charge-state 3 --relativistic-projectile-dcs=true \
  --include-barkas-dcs=false --energy-unit per_u \
  --energy-min-MeV 1 --energy-max-MeV 100 --dE 1000 --dq 1000
```

H, He, C, O, and S support integer states from neutral to bare. A missing
`--charge-state` selects the historical bare-projectile convention. `per_u`
inputs are converted to total kinetic energy before kernel evaluation.
Fixed-state screening must not be combined with a scalar Zeff correction.

Workers default to 10 or the scheduler allocation; override with
`ICE_MAX_WORKERS`. Atomic-data generation also defaults to 10 workers.

## Products and integration

- DAT DCS/TCS: `output/tables/`.
- NPZ caches and separate Born/polarization diagnostics: `output/caches/`.
- DCS columns retain the existing scale: multiply by `1e-22/3.343` to obtain
  m2/eV. TCS uses the same area scale and the integral of the final DCS.
- Tables represent a density linear in loss energy between nodes. The
  Geant4 consumer on `ion` builds its sampling CDF from this density. This
  Python-only branch does not build or run Geant4, nor add charge exchange.
- Compatible energy patches merge by default. `--no-merge-energy-patches`
  replaces the selected table range; incompatible metadata fails explicitly.
- No neighbouring Geant4 installation is modified automatically. To copy
  completed tables, explicitly pass `--geant4-data-dir /path/to/dna`.
- Generation no longer emits the old, unrelated multi-plot campaigns. Use
  the component benchmark/diagnostic scripts for figures.

Generated outputs are ignored by Git. Existing calculations are not silently
relabelled or numerically updated by moving the code.

## Scientific limits

The relocation changes organization, not the adopted formulas or cutoffs.
The ion K continuum remains unscaled; the molecular allocation and the
polarization OOS normalization remain separate. The latter uses 8 valence
plus 2 core electrons. Missing core bound excitations are not invented.

The screened-oscillator correction is an approximation, not validated ice
DCS for every charge state. Numerical agreement with the bare Salvat limit
does not validate screened relativity or the bound-target spectral treatment.
Nonconvergence, negative totals, and an uncontrolled correction are still
rejected. Bloch corrections cannot be enabled in the DCS pipeline.

```bash
python -m pytest -q
```

The tests check implementation and export consistency. They do not establish
agreement with ICRU/Matias or convergence of every requested production grid.
