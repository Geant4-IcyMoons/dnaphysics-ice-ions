# Ion dielectric cross sections

`ion_modular` contains the standalone ice dielectric DCS/TCS workflow.
There is no legacy `python_scripts/` tree or Geant4 application. CTMC and
elastic folders are empty placeholders; their implementations remain on `ion`.

```text
physics/
  constants.py
  inelastic_dielectric/
    generate_cross_sections.py
    pwba/
    rpwba/
    finite_q/
    k_shell/
    projectile_potentials/
    polarization/
    validation/
    jobs/
  ctmc/                       (empty)
  elastic/
    hard_collisions/          (empty)
    soft_collisions/          (empty)
tests/
```

Start with the [generation guide](physics/inelastic_dielectric/README.md).
The [cluster transition notes](physics/inelastic_dielectric/validation/CLUSTER_SETUP.md)
record the preserved legacy work, local environment, CLI repair, and
restart-safe production scheduling.
Each component documents its code, references, benchmarks, and validity
limits. Benchmark figures are PDF, stored under `benchmarking/plots/` with
their routines. Generated plots, reports, caches, and production tables are
ignored by `git add .`.

## Install and run

Python 3.11 or newer:

```bash
python -m pip install -r requirements.txt
python -m physics.inelastic_dielectric.generate_cross_sections --help
python -m pytest -q
```

Example: hexagonal ice, proton RPWBA plus polarization:

```bash
ICE_TYPE=hexagonal python -u -m physics.inelastic_dielectric.generate_cross_sections \
  --projectile proton --relativistic-projectile-dcs=true \
  --include-barkas-dcs=true --include-kshell --kshell-model hydrogenic-gos \
  --energy-min-MeV 0.1 --energy-max-MeV 100 --energy-unit total \
  --energy-points 1000 --dE 1000 --dq 1000
```

Defaults write `physics/inelastic_dielectric/output/{tables,caches}/`.
Use `--output-dir` and `--cache-dir` for other destinations. Copying to
Geant4 requires explicit `--geant4-data-dir`; no nearby install is overwritten.

Frozen densities for all H/He/C/O/S states are included. Atomic-data
regeneration additionally requires PySCF. The external SBETHE benchmark
requires `gfortran` and network access. Neither is needed for ordinary DCS
generation. Install Courier, Courier New, or Nimbus Mono PS for figures.

## Scope

Operational generation is not physical validation for every state and
energy. The screened polarization correction remains an approximation with
explicit rejection checks. See [validation](physics/inelastic_dielectric/validation/README.md)
and [polarization benchmarks](physics/inelastic_dielectric/polarization/README.md).
