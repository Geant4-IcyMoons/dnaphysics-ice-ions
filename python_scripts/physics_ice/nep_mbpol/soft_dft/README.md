# Charge-resolved soft-interaction DFT workflow

This package prepares fixed-charge ion--water interaction surfaces for a later
soft nuclear-stopping model. It keeps the three physical components separate:

```text
current Geant4 charge q -> DFT-derived soft table V_q
hard encounter         -> charge-independent retained-domain NLH kernel
charge-changing event  -> CTMC q -> q', then select V_q'
```

CTMC is not executed during DFT generation. DFT does not predict the
charge-changing probability. Geant4 will eventually select the table for the
current charge, sample charge exchange independently with CTMC cross sections,
and select the new table only after a transition.

## Scientific scope

The implementation is a **validation-pending molecular pilot**, not a finished
ice potential. It scans one projectile against a fixed H2O molecule over five
defined orientations. All calculations use fixed nuclei, all-electron GAPW,
`POTENTIAL ALL`, a CP2K Becke population constraint, and a fixed-geometry
Boys--Bernardi counterpoise subtraction:

```text
E_int(q,R,Omega) = E_ion+H2O(full basis)
                 - E_ion(with H2O ghost basis)
                 - E_H2O(with ion ghost basis).
```

The CDFT target is `Z-q` electrons on the projectile. The total calculation
charge is `q`, and its multiplicity comes from the declared ion state. The
isolated reference for the fully stripped state `q=Z` is exactly zero and is
recorded analytically. No long-range offset, NLH blend, fitted scaling, or
density correction is applied.

The built-in H, He, C, O, and S definitions contain every state from q=0 to
q=Z, using the NIST ASD ground configuration and term, and a verified CP2K
2025.2 `TZVPP-MOLOPT-GGA-ae` basis entry. Their scan nodes include the exact
projectile--H/O NLH radii for 100, 60, 30, 20, and 10 eV plus numerical probes
through 6 A. The 10--30 eV region tests hard/soft overlap; it is not a
switching function. A complete sourced definition is an executable input,
not evidence that its PBE/CDFT surface is physically valid.

The numerical engine is element-independent. A different ion is enabled by a
validated JSON definition, not by extrapolating carbon. The definition must
give the complete non-negative atomic charge ladder `q=0..Z`, the exact `Z-q`
electron count, multiplicity, configuration, term, state provenance, an
all-electron projectile basis, and a sourced scan grid. Definitions missing
any charge are rejected. This makes the calculation executable for arbitrary
ions while keeping physical evidence ion-specific.

For heavy ions, successful execution does not establish validity. Relativistic
treatment, basis availability, electronic-state convergence, charge
localization, and the applicability of the chosen functional require separate
validation. Configuration and term labels document the intended state; a
population constraint plus total multiplicity alone does not prove that CP2K
converged to that state.

## Projectile-definition format

The canonical H/He/C/O/S definitions live in `../ion_ice/species/`. The shared
schema records isotope provenance and hard/soft/Geant4 availability as well
as the soft-DFT charge ladder. Create a strict missing-component skeleton with:

```bash
../.venv/bin/python ../ion_ice_model.py init-species \
  --symbol <symbol> --name <name> --atomic-number <Z> \
  --mass-number <A> --neutral-atomic-mass-u <mass> \
  --mass-source <authoritative-source> --mass-url <source-url>
```

Then change `components.soft_dft.status` to `implemented` only after supplying
all `q=0..Z` states, their state provenance, the verified all-electron basis,
and a sourced scan grid. Use `kind: nlh_overlap` only when the same species
definition and backend contain real projectile--H/O NLH pairs. Otherwise use
explicit, sourced H- and O-anchored distances. The schema rejects missing
charges and the workflow never invents ground states, basis sets, or scan
boundaries. The full species schema and onboarding procedure are documented
in `../ion_ice/README.md`.

Negative ions and molecular projectiles are outside this schema and require a
separate physical extension; “complete charge ladder” here means the neutral
atom through its fully stripped positive ion.

## Adaptive contract

`adaptive_charge_resolved_dft.py` is the recommended controller. Every base
separation remains mandatory. It calculates direct quarter-, midpoint-, and
three-quarter probes for every interval, orientation, and requested charge.
An interval is bisected if either:

- its largest absolute interpolation residual, divided by the largest
  absolute endpoint or direct-probe energy for that charge and interval,
  exceeds 0.5%; or
- direct and interpolated energies have opposite signs.

One common radial mesh is retained for all requested charges, controlled by
the worst charge state. There is no arbitrary small-energy floor. The 0.5%
tolerance concerns numerical table interpolation only; it is not DFT, NLH, or
physical accuracy.

Every generation is an immutable, configuration-hashed workflow. Completed
tasks are checksum-reused, so interruption loses at most the CP2K calculations
currently executing. For carbon's seven charges, the 60-point base mesh needs
840 CP2K calculations and 60 analytic bare-ion references. Direct validation
of every initial interval adds 2,310 CP2K calculations. Further work is
determined by measured interpolation residuals.

## Prepare and run

From the repository root, prepare any built-in workflow:

```bash
python_scripts/physics_ice/nep_mbpol/.venv/bin/python \
  python_scripts/physics_ice/nep_mbpol/prepare_charge_resolved_dft.py \
  --projectile C
```

For an external ion definition and, optionally, a diagnostic charge subset:

```bash
python_scripts/physics_ice/nep_mbpol/.venv/bin/python \
  python_scripts/physics_ice/nep_mbpol/prepare_charge_resolved_dft.py \
  --projectile-definition /absolute/path/to/ION.json --charges 0 1
```

Omitting `--charges` selects the complete declared ladder. Output defaults to
`soft_collision_dft_runs/<symbol>_molecular_pilot/`. Repeating preparation is
idempotent. The default PBE/TZVPP settings remain a pilot until basis, GAPW
grid, cell, SCF/CDFT, and functional sensitivities are measured.

Run the adaptive controller locally where CP2K is installed:

```bash
python_scripts/physics_ice/nep_mbpol/.venv/bin/python \
  python_scripts/physics_ice/nep_mbpol/adaptive_charge_resolved_dft.py \
  --projectile-definition /absolute/path/to/ION.json \
  --cp2k-command /path/to/cp2k.psmp \
  --cpus-per-calculation 1 --parallel-calculations 1
```

Completed tasks are skipped on rerun. CP2K is not vendored, and no CP2K module
is currently visible on Chemfarm. Install and validate the official CP2K
2025.2 MPICH container on a compute node with:

```bash
qsub pbs/install_cp2k_container.pbs
```

Before a large adaptive scan, run every charge state at one 6-A geometry.
This checks the executable, basis, and SCF/CDFT plumbing and is never accepted
as a physical potential table:

```bash
for element in H He C O S; do
  qsub -v "PROJECTILE=${element}" pbs/run_charge_resolved_dft_smoke.pbs
done
```

## PBS and multi-node execution

After CP2K is installed, submit one automatic, eight-core adaptive run for a
built-in ion or an external definition:

```bash
bash pbs/launch_adaptive_charge_resolved_dft.sh C
bash pbs/launch_adaptive_charge_resolved_dft.sh /absolute/path/to/ION.json
```

The pilot request is 8 CPUs, 128 GB, and 500 hours through the `idle` router
to `idlex`. It starts one eight-rank, single-thread-per-rank CP2K calculation.
The 30-A, 800-Ry all-electron GAPW grid is memory-bound: single-rank smoke
attempts exceeded 32 GB and approached 60 GB, whereas MPI distributes the
grid and uses the allocated CPUs. Exact-input job 108842 completed in 43:04
at 797% average CPU and 76050652 kB peak PBS RSS. The hard limits are 256 CPUs
and 512 GB.
Use `ion_ice_model.py calibrate-resources --stage soft_molecular_dft` with
completed PBS job IDs to measure peak GB/CPU and average CPU utilization.
`tqdm` reports completed tasks for every adaptive batch.

For multi-node operation, first prepare the next immutable batch:

```bash
python_scripts/physics_ice/nep_mbpol/.venv/bin/python \
  python_scripts/physics_ice/nep_mbpol/adaptive_charge_resolved_dft.py \
  --projectile-definition /absolute/path/to/ION.json --prepare-only
```

Then submit the printed manifest as any number of deterministic array shards:

```bash
bash pbs/launch_charge_resolved_dft.sh \
  /absolute/path/to/workflow.manifest.json 16
```

After the array finishes, repeat `--prepare-only` to collect it and create only
the next necessary refinement batch. Never run two shard layouts concurrently
against the same workflow directories.

Each fixed-batch array shard likewise requests 8 CPUs and 128 GB and executes
one calculation. The PBS scripts accept `CP2K_EXE` or `CP2K_MODULE`. The
adaptive script also
accepts `PROJECTILE`, `PROJECTILE_DEFINITION`, `OUTPUT_ROOT`, a comma-separated
`CHARGES`, `MPI_RANKS_PER_CALCULATION`, `OMP_THREADS_PER_RANK`, and
`PARALLEL_CALCULATIONS` through the job environment.

## Collection and acceptance gates

Collect a fixed batch with:

```bash
python_scripts/physics_ice/nep_mbpol/.venv/bin/python \
  python_scripts/physics_ice/nep_mbpol/collect_charge_resolved_dft.py \
  /path/to/workflow.manifest.json --require-complete
```

Output names use the lower-case element symbol and include the full projectile
definition in their manifest. Even a numerically converged adaptive table
remains `validation_pending`. Before Geant4 use, one must:

1. establish basis, grid, cell, SCF, CDFT, and functional convergence;
2. verify charge localization and the intended electronic state for every q;
3. demonstrate that the unshifted interaction approaches zero;
4. validate the 10--30 eV NLH overlap where NLH data exist, without a gap or
   double counting;
5. repeat representative amorphous- and ice-Ih environment calculations; and
6. derive and independently benchmark soft stopping and angular moments.

The CTMC tables are subsequently needed to sample `q -> q'`; they do not
validate these DFT surfaces.

## Provenance

- CP2K constrained-DFT documentation:
  <https://manual.cp2k.org/cp2k-2025_2-branch/methods/dft/constrained.html>
- N. Holmberg and K. Laasonen, *J. Chem. Theory Comput.* **13**, 587--601
  (2017), <https://doi.org/10.1021/acs.jctc.6b01085>.
- CP2K all-electron GAPW documentation:
  <https://manual.cp2k.org/trunk/methods/dft/gapw.html>
- NIST Atomic Spectra Database, SRD 78:
  <https://doi.org/10.18434/T4W30F>.
- NIST fixed H2O pilot geometry:
  <https://www.nist.gov/mml/csd/chemical-informatics-group/benchmark-results-tip4p2005-water>.
- Boys--Bernardi counterpoise support in CP2K:
  <https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/BSSE.html>.
