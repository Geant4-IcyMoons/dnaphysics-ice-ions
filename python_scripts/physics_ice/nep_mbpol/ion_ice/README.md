# Unified ion--ice collision workflow

This directory is the canonical control layer for the atomistic nuclear-
elastic model. It joins the ice structures, NLH pair potentials, adaptive
binary kernels, phase-resolved hard transport, charge-resolved soft-DFT work,
and Geant4 table/runtime gates without treating an incomplete component as a
validated model.

CTMC charge exchange remains a separate physical process. Its only future
interface here is the Geant4 charge state: a CTMC transition `q -> q'` changes
which independently validated charge-resolved soft table is selected. CTMC
does not generate, fit, or validate a nuclear-elastic table.

The validation-pending low-energy complement is implemented separately in
`../low_energy_charge_exchange/`. It reads this registry's complete q=0..Z
ladder and prepares every spin-conserving single-capture step q->q-1. It does
not change the hard/soft nuclear-elastic dependency graph and has no accepted
Geant4 or CTMC handover until molecular couplings, trajectories, cross
sections, and their overlap have been validated.

## Sources of truth

- `species/<symbol>.json` holds one isotope mass and provenance, aliases, and
  the declared state of every ion-specific component.
- `phases/<phase_id>.json` holds the preparation, accepted structure registry,
  density definition, and structural-validation protocol for one ice phase.
- `framework.json` holds shared H2O composition, units, validity boundaries,
  and numerical tolerances.
- `resources.json` holds bounded PBS allocations, calibration status, and the
  exact accounting evidence behind each exception to the 64-core/128-GB
  uncalibrated default.
- `../../process_evidence/` separates independent benchmarks from scientific
  validation protocols and generated acceptance records for every process.
- `compatibility.py` verifies that the established NLH/BCA backend still
  agrees exactly with the shared registry. This adapter allows current
  production jobs to finish without changing their numerical code.

No registry declaration can create missing physics. A component is marked
`implemented` only when its species-specific inputs and backend exist.
Unknown coefficients, electronic states, basis sets, transition rules, and
phase corrections remain explicitly `missing` or `blocked`.

## Dependency graph

```text
accepted ice structure ----------------------+
                                              |
sourced NLH H/O pairs -> adaptive kernels ----+-> hard transport
                         -> dense benchmark        -> phase/orientation gate
                         -> hard Geant4 table -----+-> hard runtime ---+
                                                                       |
sourced q=0..Z definition -> molecular CDFT -> physical DFT gates       |
                                             -> ice environments        |
                                             -> residual soft model     |
                                             -> soft Geant4 table       |
                                             -> soft runtime -----------+-> bundle
```

The final bundle is emitted only when all hard and soft products, the selected
phase decision, and both Geant4 consumers are accepted. Successful PBS exit is
not a scientific acceptance decision. In particular, a phase-preparation job
cannot automatically unlock transport: its output must first pass the phase
validator and appear in an accepted collision registry.

The 0.5% kernel, transport, and radial tolerances are numerical interpolation
or sampling requirements. They are not physical uncertainties. NLH fit error,
DFT-method uncertainty, finite-structure effects, and experimental comparison
are recorded and assessed separately.

## Inspect the present model

Run from the repository root:

```bash
PY=python_scripts/physics_ice/nep_mbpol/.venv/bin/python
MODEL=python_scripts/physics_ice/nep_mbpol/ion_ice_model.py

$PY $MODEL validate
$PY $MODEL status --projectiles all --phases all
$PY $MODEL status --projectiles C O --phases hexagonal_ih_100k \
  --json-output ion_ice_runs/status.json
```

`status` reports these states:

- `complete`: the declared product and its required checks are present;
- `running`: a matching restartable workflow is incomplete;
- `runnable`: accepted dependencies exist and a task can be prepared;
- `validation_pending`: numerical output exists but a scientific gate has not
  passed;
- `missing_input`: species- or phase-specific evidence is absent;
- `blocked`: a dependency or implementation prevents the stage.

The command exits successfully for an incomplete model because incompleteness
is a reported scientific state, not a software failure.

## Plan and submit independent work

Planning is read-only and does not submit or execute anything:

```bash
$PY $MODEL plan \
  --projectiles H He C O S \
  --phases hexagonal_ih_100k amorphous_lda_80k \
  --output ion_ice_runs/plan.json
```

The signed JSON plan contains deduplicated tasks, exact command arguments,
dependencies, blockers, and completion gates. Independent species are separate
PBS jobs and can run concurrently. The current task granularity is:

- one GPU array per missing ice preparation;
- one 16-core, 4-GB CPU job per missing projectile kernel and benchmark;
- one 64-core, 16-GB CPU job per projectile and accepted phase for adaptive
  hard transport;
- one 8-core, 128-GB CPU job per projectile for the adaptive molecular CDFT
  pilot, running one eight-rank CP2K calculation at a time;
- optional fixed-batch multi-node CDFT uses one 8-core, 128-GB job per shard,
  again with one eight-rank CP2K calculation per shard;
- deterministic Geant4 table exporters as local reducer tasks.

Every PBS task in the signed plan records total CPUs, total scheduler memory,
requested GB/CPU, GPU count, calibration status, evidence, and the exact
`select` expression. The universal ceilings are 256 CPUs and 512 GB. A task
without measured evidence receives 64 CPUs and 128 GB (2 GB/CPU), except a
single-GPU structure job whose established topology is four host CPUs and 8
GB. Kernel and hard-transport profiles use PBS observations; hard-transport
memory remains provisional until the currently running jobs finish. All long
stages are restart-safe and their underlying task queues report `tqdm`
progress.

The soft allocation is a measured exception to the per-core default. The
30-A, 800-Ry all-electron GAPW grid exceeded 32 GB in single-rank smokes and
approached 60 GB; eight MPI ranks distribute that grid and remove the severe
single-rank CPU bottleneck. Exact-input scaling job 108842 completed in 43:04
at 797% average CPU and 76050652 kB peak PBS RSS; this is the recorded
calibration for the 8-core/128-GB profile.

Evaluate a completed task from scheduler accounting with:

```bash
$PY $MODEL calibrate-resources \
  --stage nlh_kernel_and_benchmark \
  --job-ids 106763 106765 \
  --output ion_ice_runs/kernel_resources.json
```

The report calculates peak resident GB per allocated CPU and average used CPU
cores from `resources_used.mem`, CPU time, and wall time. Its proposed memory
uses twofold headroom on both total and per-core observed RSS plus a 4-GB
floor; its CPU proposal targets
at most 85% average utilization. These are resource-sizing rules, not physics
parameters. Running-job memory is only a lower bound and cannot automatically
replace a profile. Review task concurrency and scaling, then update
`resources.json` manually with the job IDs and result. A custom reviewed file
can be supplied to `plan` with `--resource-profiles`; the same hard ceilings
are enforced.

No CP2K module is installed on the cluster at present. Install and validate
the official CP2K 2025.2 MPICH container on a compute node with
`pbs/install_cp2k_container.pbs`, then use the executable wrapper
`pbs/cp2k_container_wrapper.sh`. A soft task remains blocked unless the plan
records this executable or a module:

```bash
$PY $MODEL plan --projectiles H He C O S --phases hexagonal_ih_100k \
  --cp2k-exe pbs/cp2k_container_wrapper.sh \
  --output ion_ice_runs/hexagonal.json
```

Inspect the plan before either explicit mutation:

```bash
$PY $MODEL submit ion_ice_runs/plan.json --confirm
$PY $MODEL run-local ion_ice_runs/plan.json --confirm
```

`submit` uses `afterok` only for computational gates whose executable returns
nonzero on failure, such as the independent kernel benchmark. A receipt is
written immediately after every submission. Reusing a plan with an existing
receipt is refused to prevent accidental duplicate jobs. Local reducers use a
separate receipt and likewise refuse duplicate execution.

The status and submission paths also query PBS by the exact registered job
name. Matching active phase, kernel, hard-transport, or soft-DFT work is
reported as `running`; submission rechecks this immediately before `qsub` and
refuses a stale or duplicate task. An active amorphous array is therefore not
offered again merely because its accepted structure registry is still absent.

After jobs finish, rerun `status` and create a fresh plan. Manual scientific
gates remain visible until an accepted, checksum-linked validation record is
provided.

## Geant4 products

The hard exporter writes a projectile-specific differential angular table and
manifest. The threshold-defined microscopic cross section is evaluated
analytically, not interpolated:

```text
sigma_Pt^hard = pi r_th,Pt^2 (1 - V_min/E_cm,Pt),  E_cm,Pt > V_min,
                0,                                  otherwise.
```

For pure water ice,

```text
Sigma_P^hard = n_H2O (2 sigma_PH^hard + sigma_PO^hard).
```

Thus density enters once through the selected phase's molecular number
density. It does not alter the microscopic pair table.

H and He require special handling: HTran is already a complete elastic model,
not a soft-only contribution. Their NLH products cannot be installed alongside
HTran until a validated, non-overlapping energy or impact-parameter partition
exists. The registry reports this as a blocked hard runtime.

Assembly is deliberately strict:

```bash
$PY $MODEL assemble --projectile C --phase hexagonal_ih_100k \
  --output-directory ion_ice_runs/geant4
```

The command refuses to write a bundle while any phase validation, hard/soft
table, or runtime gate is incomplete. A bundle manifest contains checksums for
the species definition, phase definition, hard table, soft table, and hard
phase validation; it explicitly records that CTMC is not embedded.

## Add a new atomic projectile

Create a strict skeleton using an authoritative isotope mass source:

```bash
$PY $MODEL init-species \
  --symbol Li --name lithium --atomic-number 3 \
  --mass-number 7 --neutral-atomic-mass-u 7.0160034366 \
  --mass-source "NIST Atomic Weights and Isotopic Compositions" \
  --mass-url "https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl"
```

The new file is intentionally unusable: all five components begin as
`missing`. Then supply only real, independently traceable definitions:

1. Add projectile--H and projectile--O NLH coefficient rows only if the
   underlying dataset supports them. Extend the BCA mass/alias inventory and
   its regression tests. `validate` must pass before a kernel job is planned.
2. Mark hard transport implemented only after the common solver accepts the
   species and its two-body kinematics have unit and reference tests.
3. For soft DFT, define every state `q=0..Z` with exactly `Z-q` projectile
   electrons, sourced ground-state configuration/term and multiplicity, a
   verified all-electron CP2K basis, and either sourced explicit H/O distances
   or real NLH overlap radii. The schema rejects incomplete ladders.
4. Generate and independently benchmark the two adaptive hard kernels.
5. Run hard transport separately for every accepted phase, then perform the
   replica/orientation and amorphous--hexagonal decision gate. Store accepted
   records as `hard_collision_validation/<phase_id>/<symbol>.json`.
6. Run the adaptive molecular CDFT pilot. Passing the radial tolerance is not
   enough: basis, grid, cell, functional, state, charge-localization,
   asymptotic, counterpoise, and NLH-overlap checks are mandatory.
7. Calculate representative phase environments, fit the declared residual
   soft model without double counting NLH, and independently validate the
   stopping and angular moments.
8. Implement and test the projectile-generic Geant4 hard and soft consumers.
   Mark tables and runtimes accepted only after their physical gates pass.

At every step, `status` identifies the next missing evidence and `plan`
creates only tasks whose prerequisites permit meaningful computation. It will
never synthesize another element's coefficient, basis, charge ladder, phase
correction, or handoff rule.

## Add an ice phase

Add `phases/<phase_id>.json` with a reproducible preparation job, explicit
temperature/ensemble and density definition, validation protocol, replica
seeds, and initially `null` structure registry. After validation, register
only checksummed, collision-ready snapshots and point `structure_registry` to
that accepted record. The same registered phase then feeds every projectile;
it is never regenerated per ion.

## Method and data provenance

The registries store item-level provenance. The principal sources used by the
connected workflows are:

- Xu et al., NEP-MB-pol, *npj Computational Materials* **11**, 279 (2025),
  <https://doi.org/10.1038/s41524-025-01777-1>; released runtime and pretrained
  model, <https://doi.org/10.5281/zenodo.15033656>. These supply water-target
  energies and forces, not projectile interactions.
- Eltareb, Lopez, and Giovambattista, *Communications Chemistry* **7**, 36
  (2024), <https://doi.org/10.1038/s42004-024-01117-2>. Only the published
  240--80 K isobaric cooling segment motivates the explicitly labelled adapted
  amorphous protocol.
- Rottger et al., *Acta Crystallographica B* **68**, 91--100 (2012),
  <https://doi.org/10.1107/S0108768111046908>. Its corrected ice-Ih cell-volume
  and lattice polynomials define the 100 K fixed cell and density.
- Nguyen and Molinero, *Journal of Physical Chemistry B* **119**, 9369--9376
  (2015), <https://doi.org/10.1021/jp510289t>; Bernal and Fowler, *Journal of
  Chemical Physics* **1**, 515--548 (1933),
  <https://doi.org/10.1063/1.1749327>. These define CHILL+ phase recognition
  and proton-topology checks, respectively.
- The 80 K LDA target is the checksum-pinned EPSR model and three neutron
  datasets in the STFC ISIS `AmorIce` archive,
  <https://doi.org/10.5286/edata/729>. Its experimental and structural basis is
  Finney et al., *Physical Review Letters* **88**, 225503 (2002),
  <https://doi.org/10.1103/PhysRevLett.88.225503>, and Bowron et al.,
  *Journal of Chemical Physics* **125**, 194502 (2006),
  <https://doi.org/10.1063/1.2378921>.
- Nordlund, Lehtola, and Hobler, *Physical Review A* **111**, 032818 (2025),
  <https://doi.org/10.1103/PhysRevA.111.032818>; corrected coefficient archive,
  <https://doi.org/10.5281/zenodo.17302337>. These are the only source of the
  registered H/He/C/O/S--H/O short-range pair fits and their reported fit
  errors.
- Mendenhall and Weller, *Nuclear Instruments and Methods B* **227**, 420--430
  (2005), <https://doi.org/10.1016/j.nimb.2004.08.014>. This is the classical
  screened-potential scattering integration underlying the binary kernel.
- Tran et al., *Nuclear Instruments and Methods B* **343**, 132--137 (2015),
  <https://doi.org/10.1016/j.nimb.2014.10.016>. This is the proton/alpha water
  elastic model represented by HTran and is why H/He overlap is forbidden.
- Holmberg and Laasonen, *Journal of Chemical Theory and Computation* **13**,
  587--601 (2017), <https://doi.org/10.1021/acs.jctc.6b01085>; CP2K constrained-
  DFT and GAPW documentation, <https://manual.cp2k.org/>. These define the
  molecular fixed-charge pilot machinery. The CP2K 2025.2 UZH basis library
  supplies the registered all-electron H/He/C/O/S basis entries. NIST ASD SRD
  78, <https://doi.org/10.18434/T4W30F>, supplies every registered H/He/C/O/S
  ground configuration and term from q=0 through q=Z.
- Dvoretzky, Kiefer, and Wolfowitz, *Annals of Mathematical Statistics* **27**,
  642--669 (1956), <https://doi.org/10.1214/aoms/1177728174>, provides the
  empirical-distribution confidence band used by hard-transport convergence.
  Newey and West, *Econometrica* **55**, 703--708 (1987),
  <https://doi.org/10.2307/1913610>, and Holm, *Scandinavian Journal of
  Statistics* **6**, 65--70 (1979), <https://www.jstor.org/stable/4615733>,
  define the thermodynamic trend uncertainty and familywise correction used
  in hexagonal-ice validation.

`../PROVENANCE.json`, each species/phase JSON, and the component-specific
READMEs retain the exact local files, versions, checksums, definitions, and
scope limitations. Citations document a definition or input; they do not turn
an adapted protocol or a validation-pending calculation into a published
result.
