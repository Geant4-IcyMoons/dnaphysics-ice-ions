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
./build/dnaphysics_proton e1_proton.mac 12 proton 10 10 100000 1
```

Arguments after the macro are:

```text
threads particle Emin_MeV Emax_MeV events particles_per_event
```

If `Emin_MeV == Emax_MeV`, the source is monoenergetic. If the two
energies differ, the source energy is drawn uniformly over that interval.
The default macro uses a straight beam direction, `/gps/direction 0 0 1`.

Select table mode with:

```bash
export DNA_PROTON_BARKAS_DCS=0   # bare Born tables
export DNA_PROTON_BARKAS_DCS=1   # Born+Barkas corrected tables
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

- `plotting/plot_diagnostics_all_processes_protons.py`
- `plotting/plot_ion_root_observables.py`
- `plot_depth_diagnostics.py`
- `constants.py`
- `root_utils.py`

The proton macro writes full ROOT step diagnostics by default.  The output
contains process/model strings, channel index, per-process macroscopic cross
section, per-channel microscopic cross section, energy loss, step length, and
event energy-budget trees.

Examples:

```bash
python ../python_scripts/plotting/plot_diagnostics_all_processes_protons.py \
  --root dna.root --processes all --no-show

python ../python_scripts/plotting/plot_ion_root_observables.py \
  --root dna.root --particle proton --emin-eV 1e5 --emax-eV 1e8
```
