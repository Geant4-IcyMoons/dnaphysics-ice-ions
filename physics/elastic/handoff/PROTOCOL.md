# NLH–ZBL handoff protocol for hexagonal and amorphous ice

Status: executable diagnostic study, 2026-09-14. Combined trajectories and
binary refinement calculations are implemented. A compute-node smoke test
passed for both phases; this is software/integration evidence, not physical
qualification. The full study has not been run.

## Scientific question

Determine whether carbon nuclear stopping and angular transport in the retained
static-ice, sequential-binary model are insensitive to the NLH–ZBL partition.
Agreement between partitions is numerical/model-partition evidence, not an
experimental validation of either interaction potential.

The partition variable is NLH potential energy at closest approach, not recoil
energy. Candidate boundaries are 30, 60 and 100 eV. These are study choices;
the literature does not prescribe them as NLH–ZBL switching thresholds.
The previously suggested 15 eV case is excluded from the primary study: the
current soft backend explicitly requires a boundary of at least 30 eV.

Nordlund, Lehtola & Hobler (2025), PRA 111, 032818, report agreement between
quantum-chemical repulsive potentials above approximately 30 eV. This supports
testing the short-range NLH domain, not the accuracy of universal ZBL below it.
https://doi.org/10.1103/PhysRevA.111.032818
https://www.mv.helsinki.fi/home/knordlun/pub/Nor25.pdf

Mendenhall & Weller (2005) describe screened-potential scattering with explicit
projectile/recoil tracking. This supports the scattering framework, not an
ice-specific switching threshold.
https://doi.org/10.1016/j.nimb.2004.08.014
https://arxiv.org/abs/physics/0406066

These references support binary scattering and studying the sensitivity of
transport observables to the interaction potential. They do not establish the
proposed NLH-inside/ZBL-outside partition for carbon in ice. The three candidate
boundaries, the cutoff sequence, the numerical target and the statistical target
below are project choices. Physical validation additionally requires suitable
experimental or independently validated reference observables; full ZBL alone
is not such a reference.

## Shared inputs and phase coverage

Read the accepted structures and their metadata from
[models/ice](../../../models/ice/README.md), verifying the registry checksums.
Do not regenerate or rescale the structures for this comparison.

| Phase | Accepted inputs | Incidence | Pilot conditions | Full conditions |
| --- | --- | --- | --- | --- |
| Hexagonal Ih, 100 K | Three snapshots, seed1000/2000/3000 | Basal axis, c axis, isotropic | 18: seed1000 × three directions × six energies | 54 |
| Amorphous LDA, 80 K | One EPSR snapshot | Isotropic | 6: one snapshot × six energies | 6 |

Use identical structures, entrance-energy conventions and path length across
the boundary variants. The pair potentials are shared by both phases; their
binary maps are calculated once. Structure and incidence determine the
phase-dependent trajectory response. Nuclear charge, rather than ionic charge
state, defines these retained potentials; do not duplicate cases for C0–C6.

## 1. Binary calculations before ice histories

Use C–H and C–O independently, initially at the six production entrance energies
1, 10, 100 keV and 1, 10, 100 MeV (total carbon energy). Extend the energy grid
where interpolation checks or later slowing-down histories require it.

For each energy and boundary, compute the common impact parameter from the NLH
turning condition. Integrate NLH on the inner disk and ZBL on the complementary
outer annulus. Record recoil, scattering angle and potential/force differences
on both sides of the boundary. The two branches may assign different recoil
energies at the same impact parameter; do not assert a unique recoil switch.

Integrate stopping and angular-transport moments deterministically with explicit
subintervals at each boundary and adequate resolution of the small-impact
region. Compare each observable separately when refining quadrature; a vector
norm dominated by energy transfer must not mask angular-transport error.
Check the existing fixed-order ZBL orbit approximation against independently
refined deflection integration, as well as refining the impact integration.
Require changes below 0.5% in each moment for this numerical study, a project
choice that resolves a 5% boundary comparison without Monte Carlo noise.

Test outer recoil cutoffs 1e-4, 1e-5 and 1e-6 eV independently of the inner
boundary. Extend this sequence only if the retained stopping/transport changes
remain material. Collision counts need not converge as the cutoff decreases.
Do not silently truncate the NLH disk if a candidate ZBL outer radius lies
inside it; flag the geometry and resolve the cutoff before transport.

Output per pair/energy: the three hybrid moments, full-ZBL diagnostic moments,
boundary recoil predictions from both models, convergence evidence, and costs.
Combine H/O moments with the declared water stoichiometry only after retaining
the separate pair evidence. Full ZBL is a comparator, not ground truth.

## 2. Combined transport implementation and verification

One history must select NLH or ZBL at each encounter using the same current
energy and the common impact boundary. Update direction and energy before
finding the next encounter. Do not add separately propagated hard-only and
soft-only ice responses. Ensure exact partition coverage, collision kinematics,
sampling weights and moment-control normalization across the boundary.

Use precomputed, independently checked pair maps to avoid performing direct
orbit integration for every ice encounter. Include the boundary in interpolation
and sampling breakpoints. Existing 30 eV NLH maps can be reused only on their
compatible domain with explicit provenance; other domains require checked maps.

Verify branch selection, conservation, empty domains, interpolation, unbiased
weights and deterministic restart before submitting histories. Sources, maps,
structures, energy convention, cutoffs, path length and seed scheme belong in
the signed campaign identity. Do not merge incompatible checkpoints.

## 3. Dedicated ice simulations

Start with 24 physical conditions: the six entrance energies above, one accepted
Ih snapshot with basal, c-axis and isotropic incidence, and the accepted amorphous
snapshot with isotropic incidence. Use the production 100-angstrom path length.
Run each with all three boundaries: 72 combined-trajectory cohorts.
Use the outer cutoff selected by the binary checks, not a separately tuned
cutoff for each boundary.

Add 24 full-ZBL comparator cohorts using the same conditions and cutoff.
These diagnose the effect of replacing the inner potential, not a pass/fail
requirement that the hybrid reproduce ZBL.

At the conditions with the largest boundary response, low-energy sensitivity or
channeling response, repeat the combined histories with the next smaller outer
cutoff. This tests whether changed trajectories amplify the omitted soft tail.
If stable, extend the three-boundary comparison to the other two accepted Ih
snapshots: 108 further cohorts, giving 180 combined cohorts over all 60 declared
production conditions. These stages avoid replicating an already failing model.
One amorphous snapshot does not establish amorphous structural-ensemble error.

The prepared full matrix also includes 36 further full-ZBL controls for the
two additional Ih snapshots, for 60 controls overall. Thus the initial study
has 96 cohorts (72 combined and 24 controls); the full planned matrix has 240
(180 combined and 60 controls). A cohort comprises many independent histories,
not one trajectory. Targeted cutoff and additional-energy checks are extra,
selected from the initial results rather than expanded into a full factorial
campaign. The replica stage does not start automatically if the pilot exposes
a material unresolved handoff problem.

Reuse the same entrance conditions across boundary comparisons when practical,
but measure the resulting whole-history covariance; shared seeds alone do not
justify claiming a variance reduction. Otherwise use independent-cohort errors.
Use independent production histories after any proposal-training stage.

The current preparation routine assigns independent seeds to cohorts. Paired
entrance sampling is an optional implementation improvement; it is not currently
implemented and must not be assumed when calculating comparison errors.

## 4. Statistical decisions and interpretation

Retain the production target of 5% estimated relative standard error for scalar
outputs. Assess handoff sensitivity using the relative difference of stopping
and angular transport, normalized by the smaller positive estimate. Report the
comparison SE, including measured covariance when paired. An observed difference
below 5% with comparison SE below 5% is an operational screening criterion,
not proof of equivalence within 5% or a simultaneous confidence guarantee.
Compare every pair of candidate boundaries (30/60, 60/100 and 30/100), for
both observables, separately in each phase/structure/direction/energy condition.
Passing adjacent comparisons alone can conceal a larger endpoint difference.
Use the same test for adjacent outer cutoffs. Retain individual comparison
values in the report rather than only a phase-averaged pass flag.
Near-threshold/noisy comparisons receive targeted additional sampling; report
unresolved comparisons explicitly. Zero estimates or zero empirical variance
do not automatically pass. Do not tighten every cohort simply to resolve one
difficult comparison, or relabel a resolved model difference as sampling noise.

If boundary sensitivity is resolved above 5%, inspect the affected pair/energy
and scattering range before spending more on ice replicas. A mismatch is not
fixed by additional histories. No fitted splice or smoothing is authorized by
agreement tests alone. A plateau across candidate boundaries supports retaining
30 eV if it participates in that plateau; it does not prove physical accuracy.
If a continuous plateau is uncertain, add an intermediate boundary only in the
affected conditions. Preserve any restrictions on the final rule's validity.

A common handoff is supported only over the conditions in which both phases
pass the operational comparisons and the binary and cutoff checks are resolved.
Do not average a failure in one phase against a pass in the other. If only a
restricted energy range is supported, report that range; do not extrapolate or
introduce a phase-dependent fitting rule merely to obtain agreement.

## 5. Execution and deliverables

Perform independent pair/condition work on PBS with bounded process pools and
single-threaded numerical libraries. Benchmark a small cohort before choosing
CPU allocations; do not request 10,000 CPUs for this diagnostic by default.
Persist completed pair products and bounded history blocks atomically. Use
deterministic task seeds independent of worker count and tqdm completed/total
counts for both maps and histories. A terminated job must resume completed
work and reject changed physical settings.

Deliver boundary/cutoff comparisons, per-observable errors, branch-resolved
diagnostics, CPU-hours and trajectories/second, source/input hashes, and a
decision of supported, sensitive, or unresolved per condition. Preserve the
small evidence products; remove only superseded scratch after verifying their
replacement. Current CTMC jobs and existing production evidence are unaffected.

## Preparation and implementation status

The executable preparation routine uses [carbon_ice.json](carbon_ice.json):

```bash
python -m physics.elastic.handoff.study --output physics/elastic/handoff/runs/carbon/study.json
```

Preparation verifies inputs and records 12 initial pair/energy tasks and 240
planned ice cohorts. See [README.md](README.md) for binary, trajectory, reduction
and PBS commands. Runs require an explicit outer cutoff and retain independent
source/setting identities. Results do not automatically certify a handoff.

The implementation uses direct orbit solves and a defensive uniform/collision-
tube proposal with exact weights and bounded parallel history blocks. It does
not yet use precomputed maps or optional paired proposals discussed above. Measure runtime
before scaling; these remain possible optimizations after reference validation.
The reducer writes independent-cohort boundary comparisons; final assessment
also requires inspecting binary refinement and cutoff comparisons. It does not
automatically select a boundary or start the replica stage.

### Terminal tracks

Both phases retain histories that fall below the declared terminal energy,
including traveled length and residual energy. The observable ends there;
no below-threshold recoil or angular continuation is invented. Compare 1 and
2 eV terminal energies where these histories are frequent, separately from the
outer recoil cutoff. A small residual energy does not establish convergence of
angular transport. Maximum-collision and numerical failures remain errors.
