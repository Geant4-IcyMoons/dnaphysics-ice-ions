# Periodic ice: first electronic-stopping scaffold

## Scope and deliverable

`bulk.py` accepts a complete periodic H2O structure, independent of phase name,
atom ordering or molecule count. It generates Octopus ground-state and
constant-velocity propagation inputs, copies the H/O PBE pseudopotentials,
and records source, cell, trajectory, settings and file hashes in a manifest.
It neither crops a large structure nor creates or validates a smaller ice phase.
Right-handed orthogonal and triclinic cells are supported.

The first physical observable is the energy gained by the electronic system
and its frozen host along a driven proton trajectory. The supplied analyzer
reports a **finite-path total-energy slope in eV/angstrom** over an explicitly
chosen distance window. Conservative variations along the path and the initial
transient contribute to this diagnostic. Only after numerical convergence,
trajectory averaging and transient checks can it be interpreted as bulk
electronic stopping. A regression residual is not ensemble uncertainty.

This is an input and analysis scaffold, not a validated phase-resolved result.
It does not calculate exclusive excitation, ionization, or H+ to H probabilities.
Projectiles C, O and S require separate electronic-state and pseudopotential
validation and are not implemented by substituting their symbols for H.

## Initial state and boundaries

The initial ground state includes the extra H nucleus and a total cell charge
of +1: the electron count equals that of the neutral water host. Electrons
relax in the full charged cell, so the projectile is **not constrained to remain
bare H+**. This prepares a relaxed charged-cell stopping calculation, not a
beam-entry capture experiment. The periodic electrostatic convention is supplied
by Octopus; charge/background and finite-size effects require convergence.
There is no vacuum, absorber, projectile-frame extraction, or additional
classical host force. Host nuclei remain fixed and only the projectile moves,
at a prescribed constant velocity.

This follows the type of periodic, frozen-host, constant-velocity calculation
used for liquid water by [Gu et al. (2020)](https://arxiv.org/html/2006.12410).
Their sampling and cell-size results are not convergence evidence for ice.
The present implementation uses Octopus with PBE pseudopotentials rather than
their CP2K all-electron representation. It is not a reproduction of that study.

## Supply an ice cell

Use extended XYZ (or `.xyz.gz`) with H/O species, positions in angstrom,
`Lattice="..."` containing nine row-major components, and `pbc="T T T"`.
The shared structure loader requires the existing validation sidecar and
verified report hashes by default; `--metadata` selects an explicit sidecar.
`--frame` selects a frame (default last). A phase label alone does not establish
structural acceptance. `--diagnostic` permits unaccepted structures and retains
that status in the manifest.

The molecule count is exactly the oxygen count in the supplied cell. The
current accepted large snapshots remain valid sources but are expensive to
propagate directly. A proposed first study would prepare independently validated
cells around 128 waters, then compare 256 and larger cells as needed. These
counts are starting proposals, not established minima. Do not wrap an arbitrary
cutout of amorphous ice into a new periodic cell.

From the repository root:

```bash
python -m physics.low_energy.RT_TDDFT.bulk prepare \
  --structure /absolute/path/to/accepted-ice.xyz \
  --settings physics/low_energy/RT_TDDFT/examples/bulk/settings.json \
  --pseudo-directory /path/to/octopus/share/octopus/pseudopotentials/pseudo-dojo.org/nc-sr-05_pbe_standard \
  --output /scratch/ice-phase/path-001
```

Edit the settings for the actual cell. Energy is total projectile kinetic energy
in eV, direction is Cartesian, initial position is fractional, lengths are in
angstrom and timestep is in atomic units. The example's 20 keV, grid and timestep
are provisional. The builder normalizes the direction, rounds the duration down
to a whole output interval, and rejects trajectories leaving the selected cell.
This guard does not prove absence of periodic-image or wake interactions.
Choose several initial positions, directions and independent configurations;
there is no single global impact parameter for disordered bulk ice.

## Execute and inspect

Run each bundle in its own directory using the intended cluster's Octopus 16.4
installation and scheduler allocation. The MPI launcher depends on that cluster.
Use a fresh bundle for each run; do not reuse a directory containing an old TD
restart. The ground-state restart and TD calculation must share the directory:

```bash
cd /scratch/ice-phase/path-001
cp gs.inp inp
/path/to/octopus > gs.log 2>&1
# Check exit status and SCF convergence before continuing.
cp td.inp inp
/path/to/octopus > td.log 2>&1
```

The inputs request energy, positions, velocities and forces in atomic units.
Density/orbital movies are not requested by default. Preserve the solver version,
binary hash, scheduler resource settings and logs with the completed run.
The preparer does not submit jobs or launch a production calculation.

From the repository root, after successful completion:

```bash
python -m physics.low_energy.RT_TDDFT.bulk analyze \
  --case /scratch/ice-phase/path-001 --fit-start 2 --fit-end 7 \
  > /scratch/ice-phase/path-001/energy-slope.json
```

The example fit window is illustrative; select it after inspecting the transient
and checking window sensitivity. The analyzer checks input hashes, step/time
consistency and completion, then fits Octopus's total energy versus prescribed
travel distance. It does not yet independently check force-work consistency or
SCF quality. Inspect the trajectory and ground-state log before accepting a run.

## Validation and next deliverable

Tests cover phase-independent ingestion, skew cells, preservation of the full
host, trajectory containment, invalid settings, provenance, energy units and
rejection of incomplete runs. Optional Octopus checks exercise tiny diagnostic
periodic cells; they establish input compatibility, not accuracy in real ice.

```bash
python -m pytest tests/test_rt_tddft_bulk.py -q
OCTOPUS_TEST_EXECUTABLE=/path/to/octopus python -m pytest tests/test_rt_tddft_bulk.py -q
```

The eventual phase-resolved deliverable is a reproducible ensemble of completed
trajectories, energy/force diagnostics and a stopping-versus-energy dataset for
each accepted phase, with sampling and numerical uncertainties. Before that:
validate smaller periodic structures; check ground-state quality, grid, timestep,
k points, cell size, initial preparation, fit interval and path sampling; compare
force work against energy gain. Projectile charge diagnostics require a separately
validated partition in the overlapping bulk density. No bulk capture cross section
should be inferred from this scaffold's energy slope.
