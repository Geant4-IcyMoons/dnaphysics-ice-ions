# Low-energy single-electron capture framework

This package prepares the smallest new low-energy ion process: single-electron
capture from H2O,

```text
X(q+) + H2O -> X((q-1)+) + H2O+.
```

It is a general workflow, not a carbon-specific formula. The canonical species
definition supplies the complete neutral-to-bare charge ladder. For carbon the
default therefore prepares C6+->C5+, C5+->C4+, ..., C+->C0. Neutral carbon is
represented as the final state; C- is outside the registered non-negative
charge ladder.

## What is implemented

- `channels.py` enumerates every selected q->q-1 transition, conserves charge,
  and verifies that the ground projectile product plus ground H2O+ has a
  spin-coupled component with the conserved entrance multiplicity.
- `workflow.py` prepares two all-electron CP2K CDFT states at each molecular
  geometry. Both states have identical nuclei, total charge, and total
  multiplicity. Their constrained projectile populations differ by exactly
  one electron.
- The runner converges the states independently, then follows the CP2K
  recommendation to restart them in `MIXED_CDFT`. The collector records the
  diabatic energy gap, state overlap, and Lowdin electronic coupling.
- `landau_zener.py` provides the unit-explicit one-crossing probability and
  impact-parameter integral. It requires the radial speed from an independently
  selected nuclear trajectory and does not silently assume a straight path.

CP2K documents CDFT charge-localized states and mixed-CDFT couplings at
<https://manual.cp2k.org/cp2k-2025_2-branch/methods/dft/constrained.html>.

## Prepare carbon

From `python_scripts/physics_ice/nep_mbpol`:

```bash
python prepare_low_energy_charge_exchange.py --projectile C --runtime-smoke
```

The smoke workflow contains one 6-A geometry for each of the six carbon
capture transitions. Remove `--runtime-smoke` to prepare the registered radial
grid and five orientations. Select a subset only for diagnostics:

```bash
python prepare_low_energy_charge_exchange.py \
  --projectile C --incident-charges 1 2
```

The same command accepts H, He, O, or S, or a complete external species JSON.
No code changes are needed when its validated q=0..Z ladder is present.

## Run and collect

```bash
python run_low_energy_charge_exchange.py WORKFLOW_MANIFEST \
  --cp2k-command "mpiexec -n 8 cp2k.psmp"

python collect_low_energy_charge_exchange.py WORKFLOW_MANIFEST \
  --require-complete
```

Work is sharded by complete channel/geometry units, so both CDFT states and the
dependent mixed calculation remain together:

```bash
python run_low_energy_charge_exchange.py WORKFLOW_MANIFEST \
  --shard-count 10 --shard-index 0
```

Every input and completed result is checksum-linked. Existing compatible
states are skipped on restart. Mixed inputs are generated only after both
state wavefunctions and their converged constraint strengths exist.

## Scientific status and next gate

The output is a **validation-pending molecular coupling dataset**, not a cross
section. Production use still requires:

1. basis, grid, cell, functional, charge-localization, spin-state and
   long-range convergence for representative channels;
2. converged crossings and slopes for every retained orientation and q;
3. curved classical trajectories and radial crossing speeds;
4. Landau--Zener impact-parameter and orientation integration from 100 eV
   total energy through the CTMC overlap region;
5. comparison with the same CTMC observable (`SC`, not `SC+TI`) without an
   empirical scale factor; and
6. only then, a Geant4 table and handover rule.

The first target state is the ground H2O+ `X 2B1` channel formed from the H2O
`1b1` donor orbital. Excited target-ion channels, multiple capture,
projectile loss, transfer ionisation and spin-changing capture are deliberately
excluded. They require additional electronic states and validation rather
than copied parameters.
