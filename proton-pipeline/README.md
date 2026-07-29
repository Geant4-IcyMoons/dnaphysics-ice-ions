# Proton/Alpha Ion Pipeline

Standalone copy of the proton simulation pipeline, separated from the main `dnaphysics-ice` tree so it can be versioned and uploaded independently.

## Included

- `dnaphysics_proton.cc`
- proton/alpha physics list for generated ice excitation/ionisation tables
- optional Dingfelder charge exchange for proton and helium charge states
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

Select table and charge-exchange modes independently with:

```bash
export DNA_ION_BARKAS_DCS=0       # bare Born tables
export DNA_ION_BARKAS_DCS=1       # Born+Barkas corrected tables
export DNA_ION_CHARGE_EXCHANGE=0  # fixed proton or alpha charge
export DNA_ION_CHARGE_EXCHANGE=1  # Dingfelder charge-state transitions
```

The older `DNA_PROTON_BARKAS_DCS` and
`DNA_PROTON_ENABLE_CHARGE_EXCHANGE` names remain supported as aliases.

Examples:

```bash
# Proton, bare DCS, fixed charge
DNA_PHYSICS=ice_am DNA_ION_BARKAS_DCS=0 DNA_ION_CHARGE_EXCHANGE=0 \
  ./build/dnaphysics_proton e1_proton.mac 12 proton 10 10 100000 1

# Alpha, Born+Barkas DCS with Dingfelder charge exchange
DNA_PHYSICS=ice_am DNA_ION_BARKAS_DCS=1 DNA_ION_CHARGE_EXCHANGE=1 \
  ./build/dnaphysics_proton e1_proton.mac 12 alpha 10 10 100000 1
```

For alpha transport with charge exchange, the generated alpha tables apply
to the bare `alpha` state. Geant4-DNA's Dingfelder/Miller-Green/Rudd models
transport the `alpha+` and neutral `helium` states. The Dingfelder and captured-
state models used here are liquid-water parameterizations evaluated at the
selected H2O material density; their use in ice is therefore an explicit model
approximation for both proton and alpha simulations.

`DNA_PHYSICS` supported values:

- `ice_hex`: 0.917 g/cm3
- `ice_am`: 0.940 g/cm3
- `water`: 1.000 g/cm3

The detector constructs the corresponding H2O material automatically and
prints its material name, mass density, and molecular number density during
initialization. A macro-level `/dna/test/setMatDens` command remains available
only for explicit user overrides.

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
