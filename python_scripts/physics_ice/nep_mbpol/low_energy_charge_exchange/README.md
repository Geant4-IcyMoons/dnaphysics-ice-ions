# Low-energy single-electron capture framework

This package defines the channel bookkeeping and numerical primitives for a
future low-energy single-electron-capture calculation from H2O,

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
- `workflow.py` writes an immutable planning manifest for the two required
  all-electron CDFT branches at each molecular geometry. Both state identities
  have identical nuclei, total charge, and total multiplicity; their required
  projectile populations differ by exactly one electron. It deliberately
  writes no state input or wavefunction-only restart.
- `cp2k.py` contains a lower-level `MIXED_CDFT` renderer that requires explicit
  converged multipliers and two preconverged same-geometry wavefunctions. It
  keeps both CDFT Hamiltonians active at fixed multiplier (`MAX_SCF 0`) and
  emits no optimizer step. This renderer and its parser are unit tested, but
  accepted branch-state handoff is not implemented, so they are not connected
  to an executable workflow.
- `landau_zener.py` provides the unit-explicit one-crossing probability and
  impact-parameter integral. It requires the radial speed from an independently
  selected nuclear trajectory and does not silently assume a straight path.
- The workflow runner and collector fail before starting CP2K or writing a
  coupling table. This prevents a finite strength plus an unpaired WFN from
  being promoted as a diabatic state.

CP2K documents CDFT charge-localized states and mixed-CDFT couplings at
<https://manual.cp2k.org/cp2k-2025_2-branch/methods/dft/constrained.html>.

## Prepare a carbon planning manifest

From `python_scripts/physics_ice/nep_mbpol`:

```bash
python prepare_low_energy_charge_exchange.py --projectile C --runtime-smoke
```

The smoke plan contains one 6-A geometry for each of the six carbon capture
transitions. It produces no CDFT input and runs no electronic calculation.
Remove `--runtime-smoke` to record the registered radial grid and five
orientations. Select a subset only for planning diagnostics:

```bash
python prepare_low_energy_charge_exchange.py \
  --projectile C --incident-charges 1 2
```

The same command accepts H, He, O, or S, or a complete external species JSON.
No species-specific code changes are needed when its complete q=0..Z ladder
is present. Registry completeness does not validate either diabatic branch.

## Execution is deliberately blocked

The obsolete execution and collection PBS launchers have been removed. The
Python runner and collector reject the planning manifest with
`branch_handoff_pending` before a subprocess or output table is created. An
executable path requires two
immutable, reciprocal validated branch records whose geometry, total charge,
multiplicity and spin mode, projectile population, CP2K settings and image,
multiplier, WFN checksum, density, and ordered trace all match the planned
state identity. That loader and its `MIXED_CDFT` handoff remain to be written.

The separate carbon q=1, 12-A soft-CDFT pilot supplies at most evidence for the
entrance population at one geometry. It does not validate the capture-product
diabat: for `C+ + H2O -> C + H2O+`, both diabats have total charge +1, whereas
their projectile populations are five and six electrons. The existing branch
runner's `--charge` option sets both total charge and the asymptotic projectile
target, so invoking it with q=0 would describe neutral C plus neutral H2O, not
the required product state. The future handoff must represent these quantities
independently and must not infer the second state from the q=1 pilot.

## Scientific status and next gate

The current output is a **validation-pending planning manifest**, not a
molecular coupling dataset or cross section. Execution first requires the
accepted-state loader described above. Production use would then still
require:

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
