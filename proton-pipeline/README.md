# Proton Pipeline

Standalone copy of the proton simulation pipeline, separated from the main `dnaphysics-ice` tree so it can be versioned and uploaded independently.

## Included

- `dnaphysics_proton.cc`
- proton physics list for water+ice excitation/ionisation tiling
- proton ice excitation and ionisation models
- shared detector, action, ROOT logging, and diagnostics code needed to run and analyze the proton simulation
- minimal plotting scripts used in this project

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

From the package root:

```bash
export DNA_PHYSICS=ice_hex
./build/dnaphysics_proton e1_proton.mac
```

`DNA_PHYSICS` supported values:

- `ice_hex`
- `ice_am`
- `water`

## Required Geant4-DNA data

This package expects the custom proton ice cross-section `.dat` files to be present under:

`$G4LEDATA/dna/`

The expected filenames are listed in [cross_sections/README.md](./cross_sections/README.md).

## Python diagnostics

Included in `python_scripts/`:

- `plot_diagnostics_all_processes.py`
- `plot_depth_diagnostics.py`
- `constants.py`
- `root_utils.py`

These scripts expect the usual ROOT output file produced by the run, typically `dna.root`.
