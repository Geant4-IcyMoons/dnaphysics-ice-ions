# Charge-resolved soft-interaction DFT workflow

This package defines the fail-closed inputs and numerical gates for future
fixed-charge ion--water interaction surfaces. It keeps the three physical
components separate:

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

The target is a **validation-pending molecular pilot**, not a finished ice
potential. The current executable path solves only one fixed-nucleus
projectile--H2O complex at a time with all-electron GAPW, `POTENTIAL ALL`, and
a CP2K density-based Hirshfeld population constraint. The five-orientation
radial mesh, its counterpoise components, and their branch-state handoff remain
pending. Density-based Hirshfeld weights avoid empirical atomic radii and are
documented by CP2K as generally more robust than Becke or Gaussian Hirshfeld
populations. The planned fixed-geometry Boys--Bernardi subtraction is

```text
E_int(q,R,Omega) = E_ion+H2O(full basis)
                 - E_ion(with H2O ghost basis)
                 - E_H2O(with ion ghost basis).
```

The CDFT target is `Z-q` electrons on the projectile. The total calculation
charge is `q`, and its multiplicity comes from the declared ion state. The
mesh design records the isolated fully stripped reference `q=Z` analytically
as zero. No long-range offset, NLH blend, fitted scaling, or density correction
is part of the branch gate.

The built-in H, He, C, O, and S definitions contain every state from q=0 to
q=Z, using the NIST ASD ground configuration and term, and a verified CP2K
2025.2 `TZVPP-MOLOPT-GGA-ae` basis entry. Their scan nodes include the exact
projectile--H/O NLH radii for 100, 60, 30, 20, and 10 eV plus numerical probes
through 6 A. The 10--30 eV region tests hard/soft overlap; it is not a
switching function. A complete sourced definition is an executable input,
not evidence that its PBE/CDFT surface is physically valid.

The registry schema is element-independent. A different ion is described by a
reviewed JSON definition, not by extrapolating carbon. The definition must
give the complete non-negative atomic charge ladder `q=0..Z`, the exact `Z-q`
electron count, multiplicity, configuration, term, state provenance, an
all-electron projectile basis, and a sourced scan grid. Definitions missing
any charge are rejected. Schema completeness does not make the pending mesh
executable or supply ion-specific physical evidence.

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

## Full-mesh integration status

The charge-resolved molecular mesh is deliberately fail-closed. Manifest
construction records every ion--water complex as
`execution: validated_cdft_branch`, but the legacy shard runner cannot execute
or accept that task type and raises before starting a shard that contains a
complex. In particular, a converged wavefunction by itself is no longer a
valid complex restart: it lacks the reciprocal multiplier, density, trace,
state-identity, and branch evidence required by the gate below. Handoff of an
accepted branch record into the full radial/orientation mesh is not yet
implemented.

Consequently, `prepare_charge_resolved_dft.py`,
`adaptive_charge_resolved_dft.py`, the legacy smoke job, and the adaptive PBS
launchers are not current production entry points for complex CDFT. They must
not be used to claim a generated or collected soft-potential table. The only
executable path documented here is the one-geometry numerical branch gate.

The intended later adaptive contract remains a design requirement: retain one
common radial mesh for all requested charges, evaluate direct quarter-, mid-,
and three-quarter interval probes, and refine on a relative interpolation
residual above 0.5% or an energy-sign mismatch. That interpolation tolerance
would control only table numerics; it would not establish DFT, NLH, or physical
accuracy. Implementation must first consume immutable accepted branch states
without weakening any identity or reciprocal-root gate.

CP2K is not vendored. Install the official CP2K 2025.2 MPICH image on a compute
node before the one-geometry pilot:

```bash
qsub pbs/install_cp2k_container.pbs
```

## One-geometry CDFT branch gate

`run_cdft_branch_gate.py` calibrates and tests one charge-localized electronic
branch before that state is used in a molecular scan. It is a restart-safe
numerical gate, not a potential generator. The full task, CP2K settings,
tolerances, launcher and image checksums, geometry, and optional parent state
form an immutable configuration signature. Completed CP2K calculations and
the atomic `branch.checkpoint.json` are checksum-verified and reused after an
interruption; the calculation interrupted at the wall-time boundary may be
repeated. Do not run two writers against the same output root.

At a fresh anchor, omit `--parent-state`. Every calibration multiplier then
starts independently from `SCF_GUESS ATOMIC` and rebuilds the registered
charge-specific `KIND/BS` occupation; no calibration probe inherits another
probe's wavefunction. Only after a bracket is selected do the two optimized
roots restart from their respective, checksum-matched endpoint wavefunctions.
This independence is part of the initial-state test, not an avoidable cost.
The PBS pilot defaults to `FRESH_ANCHOR=1` and refuses a `PARENT_STATE` in that
mode.

### Fixed-multiplier probes and in-process BISECT bracket

For a multiplier `lambda`, define the parsed residual as

```text
r(lambda) = N_C(lambda) - N_target.
```

Each calibration probe keeps `&CDFT` and `&OUTER_SCF ON`, supplies an explicit
`STRENGTH lambda`, sets `TYPE CDFT_CONSTRAINT`, and sets the CDFT outer
`MAX_SCF 0`. It also names the reviewed `BISECT` optimizer because CP2K
2025.2's `qs_scf_output.F` aborts while printing CDFT setup information if the
optimizer is left at its unrecognized `NONE` default; `MAX_SCF 0` still exits
before any multiplier update. This source-defined path converges the inner
electronic SCF and evaluates the CDFT Hamiltonian at the supplied multiplier.
It is therefore a fixed-multiplier CDFT
calculation, not the earlier invalid calculation in which `&CDFT` was omitted
and the nominal strengths had no effect. The distinction follows the CP2K
2025.2 [`qs_outer_scf.F`](https://github.com/cp2k/cp2k/blob/v2025.2/src/qs_outer_scf.F)
implementation; `MAX_SCF 0` is used only for these probes, not for either
optimized root.

Two independent probe processes do not populate CP2K's live outer-SCF
history. The gate first finds two inner-converged, same-fingerprint endpoints
`lambda_L < lambda_U` with opposite residuals and negative response
`dN_C/dlambda < 0`. It then performs two independent, uninterrupted optimized
CP2K runs. CP2K's pre-bracket fallback update is

```text
lambda_next = lambda_start - STEP_SIZE * r(lambda_start).
```

The injected steps are therefore exactly

```text
s_L = (lambda_L - lambda_U) / r(lambda_L),
s_U = (lambda_U - lambda_L) / r(lambda_U).
```

The first lower-start update visits `lambda_U`, and the first upper-start
update visits `lambda_L`. Thus each production process, rather than the Python
driver alone, accumulates both residual signs before CP2K uses `BISECT`. The
parser rejects a run unless its ordered trace contains the requested start
endpoint, the exact opposite endpoint within the endpoint tolerance, opposite
residual signs, and a converged root. CP2K documents `BISECT` as the difficult
one-dimensional option and `STEP_SIZE` as the initial steepest-descent step in
the [2025.2 CDFT outer-SCF input reference](https://manual.cp2k.org/cp2k-2025_2-branch/CP2K_INPUT/FORCE_EVAL/DFT/QS/CDFT/OUTER_SCF.html).

The reciprocal roots must agree within explicit, signature-bound tolerances
in multiplier, constrained Lagrangian, population, `S**2`, and total-density
cube RMS. Fixed-multiplier probes at the mean root plus and minus `0.05` Ha
must retain the same branch, show the expected negative population response,
and give negative discrete curvature of the constrained Lagrangian. A
checkpoint status of `validated` consequently means only that this numerical
branch gate passed for this exact geometry and configuration.

### Charge, spin, and physical interpretation

`--charge q` sets the charge of the complete ion--water calculation, while the
projectile density-Hirshfeld target is `Z-q`. For the first carbon q=1 pilot
this is a total charge of +1, a target population of five electrons on carbon,
an unrestricted doublet, and an atomic `1s2 2s2 2p1 2P1/2` starting guess. A
later q=4 calculation would instead use total charge +4, target population two,
a restricted singlet, and the atomic `1s2 1S0` guess. These inputs encode the
intended asymptotic ion plus neutral-water state, but a Hirshfeld population is
a density partition and is not by itself proof of a formal fragment charge or
a unique diabatic state.

CP2K's `KIND/BS` `NEL` is not a literal spin-resolved electron removal. In
v2025.2 `init_atom_electronic_state` applies one half of each `NEL` value to
the corresponding spin occupation. The registry therefore uses even shifts:
q=1 has no alpha 2p shift and `BETA NEL -2`, which converts neutral spherical
2p occupations `(1,1)` to `(1,0)` and gives exactly five carbon electrons.
Across the registered carbon ladder, the two spin sums satisfy
`sum(NEL_alpha)+sum(NEL_beta)=-2q` and their difference constructs the declared
alpha-majority multiplicity. This convention is taken directly from CP2K
v2025.2 [`qs_kind_types.F`](https://github.com/cp2k/cp2k/blob/v2025.2/src/qs_kind_types.F),
not fitted to the molecular calculation.

The declared multiplicity, RKS/UKS choice, and `KIND/BS` atomic guess likewise
do not uniquely select the final term or orbital occupation. `KIND/BS` affects
only `SCF_GUESS ATOMIC`; subsequent continuation is controlled by the saved
wavefunction. Audit alpha/beta electron counts, `S**2` for unrestricted
states, orbital and spin character, charge localization, and branch
switching/hysteresis. In particular, the q=1 carbon `2p` population does not
select a unique magnetic orbital. A q=1 result does not validate q=4, and a
q=4 result cannot establish q=1, 2, or 3. No state is physically accepted
merely because its SCF or branch gate terminates.

Physical validation is separate from numerical branch reproducibility. It
requires the separated-fragment limit; sensitivity to charge definition,
basis, grid, cell, and functional; orbital and spin identity; continuation in
separation without hysteresis; counterpoise consistency; amorphous-ice and
ice-Ih environments; hard/soft NLH overlap; and independent stopping and
angular benchmarks. `--publish-state` publishes only a checksum-bound
numerical wavefunction--multiplier pair. Omit it for the first pilot.

Force validation is also outside this branch runner. Before interpreting any
density-Hirshfeld CDFT force from the exact CP2K 2025.2 image, first reproduce
the official `QS/regtest-cdft-hirshfeld-3` HeH regression with that image:
`HeH-noconstraint.inp` must run before `HeH-cdft-1.inp` because the latter
restarts its wavefunction. The CDFT test uses `RUN_TYPE ENERGY_FORCE`, target
`1.0`, strength `0.186894372937` Ha, and an active CDFT outer `MAX_SCF 50`.
Although its top-level input declares `@SET MAX_SCF 0`, none of the included
files consumes that variable; it is not evidence for the fixed-multiplier
probe path above. CP2K's
[`TEST_FILES.toml`](https://github.com/cp2k/cp2k/blob/v2025.2/tests/QS/regtest-cdft-hirshfeld-3/TEST_FILES.toml)
registers `FORCES| Total atomic force` references `0.1450972684448` for the
unconstrained precursor and `0.1552195046628` for the constrained calculation,
each at tolerance `1e-7`. Preserved input copies are under
[`benchmarking/cp2k_2025_2_hirshfeld_force/`](benchmarking/cp2k_2025_2_hirshfeld_force/).
This executable-level prerequisite is necessary because the 2025.2 input
reference still labels Hirshfeld CDFT as partial with no forces even though
the tagged regression exercises them. Passing HeH would establish only that
the installed implementation reproduces its official force regression; it
would not validate a carbon--water force.

Create or refresh the exact executable receipt with:

```bash
qsub pbs/run_cp2k_hirshfeld_force_regression.pbs
```

`run_cdft_branch_gate.pbs` fails closed unless the resulting
`process_evidence/soft_nuclear_collisions/validation/benchmarks/cp2k_2025_2_hirshfeld_force/validation.json`
has `status: passed`. Before launching carbon it reruns the validator against
the checksum-pinned inputs, output files, and paired wavefunction, and requires
CP2K 2025.2 revision `c3a8adfec5` plus image SHA-256
`06cda9afc9d0f8a6ab30318352804276adfde467f385a54e15fc0141a921a59c`.

For a carbon--water force check, fully reoptimize both the inner electronic SCF
and the multiplier on the same state branch at every `+delta R` and `-delta R`
geometry. Compare analytic forces with central energy differences at several
decreasing displacements. Never reuse a fixed multiplier across displaced
geometries or infer force consistency from a single displacement.

### Exact carbon q=1 pilot

From the repository root, submit the fresh, non-publishing, 12-A oxygen-back
pilot exactly as follows:

```bash
qsub -v "REPO_ROOT=/gpfs01/work/yoffegid/dnaphysics-ice-ions,PROJECTILE=C,CHARGE=1,ORIENTATION=oxygen_back,SEPARATION_ANGSTROM=12.0" \
  pbs/run_cdft_branch_gate.pbs
```

This command invokes only the soft-CDFT branch gate. It neither runs nor
validates the separate Landau--Zener charge-transfer workflow.

The PBS file requests one 16-rank, single-thread CP2K process and 160 GB, the
measured envelope used for the carbon diagonalization gates. It deliberately
omits `--publish-state`. Repeating the same command resumes the same signed
configuration and reuses verified completed work. The direct equivalent is:

```bash
export CP2K_MPI_RANKS=16 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_scripts/physics_ice/nep_mbpol/.venv/bin/python -u \
  python_scripts/physics_ice/nep_mbpol/run_cdft_branch_gate.py \
  --projectile C --charge 1 --orientation oxygen_back \
  --separation-angstrom 12.0 \
  --output-root /gpfs01/work/yoffegid/dnaphysics-ice-ions/python_scripts/physics_ice/process_evidence/soft_nuclear_collisions/validation/runs/c_q1_cdft_branch_gate \
  --cp2k-command /gpfs01/work/yoffegid/dnaphysics-ice-ions/pbs/cp2k_container_wrapper.sh \
  --cp2k-image /gpfs01/work/yoffegid/dnaphysics-ice-ions/software/cp2k/cp2k-2025.2-mpich-x86_64-psmp.sif
```

Review the complete q=1 trace, densities, spin, and reciprocal-root evidence
before starting another charge. The exact fresh q=4 follow-up command, if that
review supports proceeding, is:

```bash
qsub -N c_q4_cdft_branch \
  -v "REPO_ROOT=/gpfs01/work/yoffegid/dnaphysics-ice-ions,PROJECTILE=C,CHARGE=4,ORIENTATION=oxygen_back,SEPARATION_ANGSTROM=12.0" \
  pbs/run_cdft_branch_gate.pbs
```

It creates a distinct q=4 configuration tree and is not a continuation or
validation consequence of q=1.

## Branch-runner resources and rejected paths

The branch-gate PBS job runs one 16-rank, single-thread CP2K process with
160 GB for up to 500 hours through the `idle` router to `idlex`. The
reviewed branch runner explicitly forces conventional
`DIAGONALIZATION` for the complex and `BISECT` for the multiplier. Earlier
OT/`3PNT` and unbracketed optimizer trials are rejected-path evidence, not
current operational alternatives.

The 30-A, 800-Ry all-electron GAPW grid is memory-bound. Single-rank smoke
attempts exceeded 32 GB and approached 60 GB. Exact-input job 108842 used
76050652 kB peak PBS RSS, and the 16-rank counterpoise comparison used
79087808 kB. The first q=4/q=6 diagonalization attempts were cgroup-killed at
128275776 and 130554232 kB under a 128-GB request; the corrected all-charge
gate later reached 127463144 kB. The 160-GB request is therefore retained for
this pilot from measured carbon evidence, not presented as a universal
resource model.

The rejected zero-strength q=4 state and its replacement logic are preserved
in the
[carbon state-initialization record](benchmarking/carbon_state_initialization/README.md)
and its
[checksum-bound evidence](benchmarking/carbon_state_initialization/zero_strength_q4_rejection.json).
Those records establish why an omitted CDFT loop or an unpaired wavefunction
cannot seed a branch; they do not establish that the replacement branch has
passed.

## Future mesh acceptance gates

There is currently no accepted complex mesh to collect: the branch-to-mesh
handoff is pending and the legacy collector cannot promote an old complex
result. After that integration is implemented, even a numerically converged
adaptive table must remain `validation_pending` until it passes all of the
following:

1. basis, grid, cell, inner-SCF, CDFT, and functional convergence;
2. charge localization and intended orbital/spin identity for every q;
3. the unshifted separated-fragment limit;
4. the 10--30 eV NLH overlap, where source data exist, without a gap or double
   counting;
5. representative amorphous-ice and ice-Ih environments;
6. the force checks specified above; and
7. independent soft-stopping and angular-moment benchmarks.

CTMC tables will later sample `q -> q'`; they neither create nor validate the
fixed-charge CDFT surfaces.

## Independent GPAW numerical pilot

The separate GPAW path tests whether the CP2K failures are specific to its
nested SCF/CDFT solver. It uses GPAW 25.7.0's documented Gaussian-Hirshfeld
charge and spin constraints with L-BFGS-B. It is not ALMO and does not remove
the population-definition ambiguity. A converged result is therefore labelled
`physical_state_status=validation_pending` and cannot enter a soft-potential or
Geant4 table.

Prepare and submit the first C+ and C2+ calculations at 12 A in all five
molecular orientations:

```bash
qsub pbs/install_gpaw_runtime.pbs
# After the installation job succeeds:
bash pbs/launch_gpaw_cdft_pilot.sh plan
bash pbs/launch_gpaw_cdft_pilot.sh submit
# After every array task exits successfully:
bash pbs/launch_gpaw_cdft_pilot.sh collect
```

The array has ten independent tasks. Each task uses eight MPI ranks, 8 GB and
up to 96 hours; hence all ten can use 80 cores concurrently. An eight-rank
dry run decomposes the real-space grid 2 x 1 x 4 and estimates about 16 MiB of
wavefunction/eigensolver storage per rank, so the memory request retains a
large safety margin without reserving 32 GB for a four-atom calculation. A completed
multiplier evaluation atomically checkpoints the GPAW wavefunction together
with both multipliers and residuals. An exact-signature restart may reuse that
pair, but not a lone wavefunction. Every completed task is rejected unless the
underlying SCF converges, both charge and spin residuals are at most 0.01
electron, the solution remains away from the numerical multiplier bounds, and
energies and analytical forces are finite.

The 0.01-electron gate is the documented GPAW cDFT optimizer scale, not a
physical uncertainty. The +/-100 eV bounds are numerical safety rails chosen
to exceed the carbon q<=4 charge-transfer energy scale; a bound-contact result
is rejected, and bound sensitivity remains a later gate. The standard carbon
PAW dataset freezes 1s2, so this pilot permits C0--C4 only. C5+ and C6+ require
a separately converged all-electron or custom-setup method.

The first q=1 smoke copied the GPAW He2 example's explicit Davidson/Pulay
settings and produced an unconstrained-SCF limit cycle before cDFT began. Its
rejected trace and the source-backed replacement are recorded in
[`benchmarking/gpaw_q1_initial_scf_2026-08-09/`](benchmarking/gpaw_q1_initial_scf_2026-08-09/README.md).
The default solver and a five-step Davidson control repeated the failure:
improving orbital updates alone did not make the density iteration contract.
The final controlled pair used GPAW's documented difficult-SCF mixer values
(`beta=0.04`, `method=difference`, `nmaxold=8`, `weight=100`) with either the
default two or five Davidson updates. These are independent fresh atomic-start
runs; neither may reuse a rejected wavefunction. The installed 25.7.0 release
does not implement the newer `msr1` backend mentioned by the current online
guide, so the available Pulay backend is retained explicitly rather than
silently requesting an unsupported method.

Both controls reached the 333-iteration limit without converging the initial
unconstrained SCF. Consequently, no GPAW cDFT multiplier iteration has yet run
and the larger array remains blocked. This independent failure is evidence
that replacing CP2K's outer optimizer alone does not solve the entrance-state
initialization problem; it is not evidence that the physical soft interaction
is absent.

The subsequent constrained-first DO-MOM diagnostic removes that unconstrained
complex solve. It prepares the C+ and H2O fragments separately, constructs the
three molecular-frame carbon 2p components plus a fractional-occupation
diagnostic, installs cDFT, restores the prepared full-system density matrix,
and then begins direct optimization. All four 12-A components passed the
numerical gate in PBS array 113111; exact results and limitations are recorded
in
[`benchmarking/gpaw_prepared_q1_2026-08-09/`](benchmarking/gpaw_prepared_q1_2026-08-09/README.md).
The three pure components remain separate validation-pending states. No radial
or geometry array is authorized by this numerical result alone.

The launcher uses a smoke-gated PBS array. Task 0 must terminate successfully
before tasks 1--9 become eligible; a numerically rejected smoke therefore
cannot release the production pilot. Override `SCF_PROFILE` and `OUTPUT_ROOT`
to compare a reviewed profile without mixing result trees. Independent
charge/orientation tasks are the scalable dimension; assigning hundreds of
ranks to a single four-atom calculation would waste CPUs.

Before any physical use, independently repeated solutions must agree in
energy, multipliers, charge/spin populations and densities; the separated
fragment limit, grid, cell, PAW setup, population definition and central-force
differences must also converge. Only then should radial continuation be added.

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
- GPAW constrained-DFT documentation:
  <https://gpaw.readthedocs.io/documentation/cdft/cdft.html>.
- B. Melander, E. Ö. Jónsson, and J. J. Mortensen, *J. Chem. Theory Comput.*
  **12**, 5367--5378 (2016), <https://doi.org/10.1021/acs.jctc.6b00815>.
