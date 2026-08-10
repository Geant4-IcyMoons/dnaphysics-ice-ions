# H/He/C/O/S Ion Pipeline

Focused ion-transport executable within the main `dnaphysics-ice` tree. It
keeps pipeline-specific actions locally and shares the repository's common
hard-collision process and data product as a single source of truth.

## Included

- `dnaphysics_proton.cc`
- proton/alpha physics list for generated ice excitation/ionisation tables
- carbon-12 threshold-defined NLH hard-elastic process and recoil generation
- validation-pending full universal-ZBL elastic baseline for H, He, C, O and S
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

Select exactly one custom nuclear-elastic mode with:

```bash
export DNA_ION_ELASTIC_MODEL=off
export DNA_ION_ELASTIC_MODEL=zbl_full
export DNA_ION_ELASTIC_MODEL=nlh_hard
```

The legacy `DNA_ION_HARD_ELASTIC=1` maps to `nlh_hard` when the new selector
is absent. `zbl_full` and `nlh_hard` cannot run together.

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

# Oxygen-16, full ZBL diagnostic baseline at 1 MeV total kinetic energy
DNA_PHYSICS=ice_am DNA_ION_ELASTIC_MODEL=zbl_full \
DNA_ZBL_ALLOW_VALIDATION_PENDING=1 \
DNA_ION_ENABLE_EXCITATION=0 DNA_ION_ENABLE_IONISATION=0 \
  ./build/dnaphysics_proton e1_proton.mac 12 oxygen 1 1 100000 1
```

For alpha transport with charge exchange, the generated alpha tables apply
to the bare `alpha` state. Geant4-DNA's Dingfelder/Miller-Green/Rudd models
transport the `alpha+` and neutral `helium` states. The Dingfelder and captured-
state models used here are liquid-water parameterizations evaluated at the
selected H2O material density; their use in ice is therefore an explicit model
approximation for both proton and alpha simulations.

## Universal-ZBL diagnostic elastic model

`G4DNAZBLFullElastic` evaluates universal-ZBL screened binary collisions on
the H and O atoms of the selected H2O material. It uses nuclear atomic number
and isotope mass, not ionic charge state. All incident energy limits are total
projectile kinetic energies:

```bash
export DNA_ZBL_MIN_ENERGY_EV=1000
export DNA_ZBL_MAX_ENERGY_EV=100000000
export DNA_ZBL_MIN_TRANSFER_EV=10
export DNA_ZBL_RECOIL_THRESHOLD_EV=10
```

Each accepted collision changes the projectile direction and subtracts the
exact recoil energy. Recoils above the transport threshold are emitted as H-1
or O-16 secondaries; smaller recoils are local non-ionising deposition.
Electronic excitation, ionisation, electronic stopping and charge exchange
remain separate processes.

The minimum-transfer cutoff defines which ZBL collisions are represented. It
is unrelated to the NLH 10/30 eV turning-potential domains and requires a
convergence study. The model is a complete diagnostic baseline, not a
soft-only correction. It therefore cannot be enabled with `nlh_hard`.

The current Dingfelder charge-state builder also registers complete HTran
elastic scattering for captured H/He states. To prevent overlap,
`zbl_full` plus `DNA_ION_CHARGE_EXCHANGE=1` currently fails loudly. That
builder must be separated before the two processes can coexist.

## Carbon NLH hard elastic validation

The carbon process is a retained-domain hard-collision model, not a complete
elastic cross section. For target atom `t=H,O`, it evaluates the microscopic
cross section analytically on every call:

```text
sigma_Ct^hard(T) = pi r_th,Ct^2 (1 - V_min/E_cm,Ct), E_cm,Ct > V_min,
                   0,                                  otherwise.
```

This equals `pi b_max,Ct^2`; the cross section is never interpolated. Only the
CM angle is interpolated, linearly in collision-area quantile and log--log in
total projectile energy. The table's 0.5% bound is a numerical interpolation
tolerance, not physical accuracy. At the retained 30 eV boundary, the
published NLH pair-potential RMS errors are 3.56% for C--H and 9.51% for C--O;
these are reported separately and do not directly bound a transport
observable.

For pure water ice the process obtains H and O atom densities from the actual
Geant4 material and uses exactly

```text
Sigma_C^hard = n_H2O (2 sigma_CH^hard + sigma_CO^hard).
```

Thus changing `DNA_PHYSICS` changes the macroscopic rate through the selected
material density; it does not create a second microscopic table. The carbon
table covers 1 keV--100 MeV total C-12 energy. It is charge-state independent
and emits H-1 or O-16 nuclear-recoil secondaries. Electronic excitation,
ionisation, carbon charge exchange, recoil charge-state evolution, and recoil
cascades are not supplied by this process.

Atomistic phase/orientation validation is a release gate. The committed table
is marked `atomistic_validation_pending`, so ordinary production is refused.
Engineering validation must acknowledge that state explicitly:

```bash
export DNA_PHYSICS=ice_hex
export DNA_ION_ELASTIC_MODEL=nlh_hard
export DNA_NLH_ALLOW_VALIDATION_PENDING=1
export DNA_ION_ENABLE_EXCITATION=0
export DNA_ION_ENABLE_IONISATION=0
export DNA_ION_CHARGE_EXCHANGE=0
export DNA_RANDOM_SEED=12345

./build/dnaphysics_proton carbon_hard_validation.mac \
  12 carbon 0.01 0.01 1000 1
```

If accepted amorphous/hexagonal structures or crystallographic directions
give appreciably different hard-recoil/angular observables, density scaling
alone fails and a validated phase/orientation correction must be implemented
before changing the release status to `accepted`.

HTran is already a complete proton/helium elastic treatment. The code rejects
`DNA_ION_HARD_ELASTIC=1` for H or He; a future NLH H/He implementation must use
either a validated energy handoff or an explicit non-overlapping angular or
impact-parameter partition.

Rebuild the compact carbon product and run the deterministic tests with:

```bash
../python_scripts/physics_ice/nep_mbpol/.venv/bin/python \
  ../python_scripts/physics_ice/nep_mbpol/export_nlh_geant4_table.py
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The carbon-only hexagonal atomistic matrix is three attested structures by
three orientations by six base energies, or 54 restart-safe cases:

```bash
cd ..
qsub -v RUN_MODE=carbon_variance_pilot -J 0-17 \
  pbs/run_nlh_hard_collision_trajectories.pbs

# Submit only after the pilot has fixed the full-matrix sampling budget.
qsub -v RUN_MODE=carbon_base_grid -J 0-53 \
  pbs/run_nlh_hard_collision_trajectories.pbs
```

The first 18 fixed-sample jobs measure variance and throughput; they are not a
convergence claim. The 54 jobs are a structure-sensitive decision-gate
calculation and do not generate the analytic hard cross section. The complete
staged protocol is in
[`CARBON_GATE.md`](../python_scripts/physics_ice/process_evidence/hard_nuclear_collisions/validation/CARBON_GATE.md).
A matched accepted amorphous matrix and a documented statistical comparison
remain required before continuum production release.

`DNA_PHYSICS` supported values:

- `ice_hex`: 0.9335 g/cm3 (ice Ih at 100 K)
- `ice_am`: 0.9343471678603292 g/cm3 (validated 80 K EPSR LDA cell)
- `water`: 1.000 g/cm3

The detector constructs the corresponding H2O material automatically and
prints its material name, mass density, and molecular number density during
initialization. Geant4's Dingfelder charge-increase and charge-decrease models
then multiply their microscopic H2O cross sections by that material's
`G4DNAMolecularMaterial` number density. A macro-level
`/dna/test/setMatDens` command remains available only for explicit user
overrides.

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
