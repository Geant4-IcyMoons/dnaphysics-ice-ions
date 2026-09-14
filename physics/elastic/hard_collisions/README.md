# Hard-collision production protocol

[certify.py](certify.py) is the single public runner; [_differential.py](_differential.py)
is its private two-body numerical backend. One signed configuration connects
pair differential cross sections, explicit-ice transport, adaptive sampling,
qualification and export. Compatible completed products are production evidence;
qualification does not require a second, duplicate trajectory campaign.

The current carbon specification selects **scalar production** (schema 7):
5% estimated relative standard error for all five scalar outputs. It retains
calibration followed by independent production, adds samples in 20% increments
rounded to a history block, and checks scalar energy interpolation by a measured
midpoint change of at most 5% with comparison SE at most 5%. Clear failures
refine the energy grid; noisy comparisons receive further samples. This is
CTMC-style observed convergence, not a simultaneous confidence certificate.

Scalar production does not train against histogram TV, refine response bins,
or wait for distribution interpolation. Its coarse H/O histogram is diagnostic.
Pair-map numerical accuracy, geometry checks, independent baseline comparison,
importance normalization, whole-history covariance, tail evidence, zero-variance
review and deterministic replay remain. The budgets remain explicit computational
limits; reaching one does not certify a case. Unlike current CTMC, this mode has
no automatic 10% fallback. No measured speed-up or new ice qualification is claimed.

`output_scope=joint_distribution` retains the stricter joint-distribution
workflow described below. Its statistical TV, grid and distribution-interpolation
requirements apply only to that scope. The default Python Policy retains this
explicit alternative; `carbon_ice.json` selects the simpler scalar workflow.
Scalar export writes a scalar-only index and tables, with
`scalar_response_ready=true`, `response_tables_ready=false`, and no claim of a
qualified joint distribution. Raw events remain reusable for a later distribution
analysis. These are finite-path ice response observables, not new microscopic DCS.

Historical schema-3 binary products remain unchanged. A new signed campaign uses
`prepare --reuse-pairs` to import their unchanged numerical payloads and repeat
consumer checks. Old phase histories and acceptance are not silently relabelled.
The historical feasibility reports describe older simultaneous-bound policies,
not measured runtime or qualification of this scalar workflow.

Carbon's binary stage is complete: **962 C–H/C–O maps and 240 energy
intervals**, spanning 1 keV–100 MeV total energy, passed the numerical tests
and independent consumer verification. Job 233312 used eight CPUs for
3 minutes 12 seconds. The production object is
[differential/index.json](runs/carbon_ice/differential/index.json). No new ice
trajectories were launched: the phase preflight remains blocked in 18 cases.

That completed binary campaign retains its original schema-3 signature and
unchanged products. It is not silently re-signed as schema-7 evidence. The
current `run` refuses to resume it under changed source or policy. A new
`prepare --reuse-pairs SOURCE` imports unchanged numerical payloads with source
provenance and repeats their consumer checks, without new pair solves or
importing old phase acceptance. Read-only
planning comparison uses `compare-scalar-intervals`; the single current
[comparison](runs/carbon_ice/scalar_estimator_comparison.json) records legacy
estimators and corrected physical-support planning without altering historical
checkpoints, tables or acceptance flags. Efficient phase certification remains
unresolved; no new production campaign has been submitted.

## Physical and output contracts

- **Microscopic pair DCS:** stationary-target NLH scattering against H and O,
  represented by a correlated angle–recoil area map and integrated joint-bin
  cross sections. This retained nuclear potential is charge-state independent.
- **Structured-ice response:** complete trajectories through declared frozen
  snapshots, retaining atomic positions and successive deflections. Products
  contain finite-path, entrance-conditioned collision rates, stopping/transport
  moments and joint H/O–polar-angle–recoil bin masses.

Dividing finite-path counts by density does not establish an instantaneous,
homogeneous phase-specific collision law. Phase differences enter through the
structures, not a fitted density-only correction. These outputs do not claim a
drop-in Geant4 process, a qualified azimuth closure or structural-ensemble
uncertainty. Overlapping multi-centre forces and recoil cascades remain outside
the retained sequential-binary model. Recoil energy is transferred to a target
recoil; it is not automatically local energy deposition.

## One particle/material configuration

[carbon_ice.json](carbon_ice.json) declares a 100-Å path and six entrance
**total kinetic energies**, 1 keV–100 MeV: 54 cases across three accepted 100-K
hexagonal Ih snapshots and basal/c-axis/isotropic incidence, plus six isotropic
cases for one accepted 80-K EPSR amorphous snapshot.

Preparation accepts `--projectile` and `--material`. Retained kernels support
H, He, C, O and S against H/O; other elements require supported potentials.
Structures, phase labels, energy domains and input hashes are checked.
Certificates cover only declared particles, snapshots and directions/energies.
One amorphous snapshot is not independent structural replicas, and discrete
directions do not certify a continuous orientation domain.

## Unified adaptive sequence

1. **Freeze inputs and policy.** Record runtime, source/environment hashes,
   structures, potential and hard boundary. Cover pair energies down to the
   retained kernel floor: later collisions can fall below entrance energies.
2. **Produce pair DCS deterministically.** For `q=(b/b_max)^2`, the area measure
   is `dσ=π b_max² dq`. Refine direct quadrature, the area grid and joint-bin
   preimages where recoil, transport or bin-mass errors require it. Explicit
   logarithmic support resolves small-impact-parameter contributions. No Monte
   Carlo histories are needed to integrate the two-body law.
3. **Refine energy interpolation.** Independently test maps at the quarter,
   middle and three-quarter positions of each log-energy interval; split failed
   intervals. Interpolate CM angle, then derive lab angle and recoil together
   from exact two-body kinematics. Evaluate total hard area analytically at the
   requested energy. Keep every lab-angle branch, never independent marginals.
4. **Reuse pair maps in ice transport.** The same qualified products feed
   `PairMapTable`; phase trajectories do not revert to the coarse original
   angle table. Uniform azimuth applies only to an isolated central-potential
   pair; explicit target geometry sets scattering planes in structured ice.
5. **Check feasibility before broad ice sampling.** Compatible retained means
   provide an optimistic range-penalty diagnostic, not qualification. A failure
   preserves data and reusable pair DCS while reporting the estimator obstacle.
6. **Train and test targeted sampling.** Historical replay seeds and fresh
   pilots guide defensive overlapping periodic proposals. Minimize predicted
   CPU cost times the worst tolerance-scaled scalar influence variance, with a
   trajectory-level L1 influence surrogate for distribution TV. This surrogate
   is not a confidence bound or proof of globally optimal sampling. Frozen
   atoms are never moved. Exact generating-density weights are retained.
   Independent baseline/targeted cohorts test usefulness; inefficient targeting
   falls back to the baseline without relabelling observations.
7. **Refine missing evidence and export.** Add sampling at declared looks,
   rebin saved events, and test energy/grid resolution separately. Recompute
   gates, deterministic replay, coverage, checksums, units and kinematics.
   Pair success exports `differential/index.json`; separately qualified ice
   cases export `tables/index.json` and `production_handoff.json`. Completed
   products are reused. A finished process or passed pair stage does not imply
   passed ice qualification.

The hard boundary remains the signed minimum-turning-potential definition;
it is not adjusted to make a run pass. Pair checks are deterministic numerical
refinement evidence, not rigorous continuous-domain error enclosures or bounds
on physical-model accuracy.

The reference evaluates the same scattering integral in a positive form,
using the turning-point identity and `expm1` for potential differences. This
avoids cancellation in `π - 2I` at high energy; it is not a fitted correction
or a change of potential. Turning-root residuals are checked explicitly.
Analytic Coulomb tests cover weak and strong scattering. Cache identities
distinguish this reference from the earlier cancellation-prone evaluation.

## Acceptance requirements

New phase campaigns require **5% estimated relative standard error** for
collision rate, nuclear stopping, angular transport, mean recoil and mean
`1-cos(theta)` per collision. The SE is computed from centred, likelihood-weighted
whole-history ratio influences, including numerator/denominator covariance.
Zero means or zero empirical variance require review rather than automatically
passing. This is not 5% physical-model accuracy, nor a finite-sample,
simultaneous or optional-stopping-valid 95% guarantee. Monitoring a sample SE
does not establish those stronger properties. The definition follows the
relative-SE convention in the [MCNP manual, Table 1.3](https://mcnp.lanl.gov/pdf_files/TechReport_2024_LANL_LA-UR-24-24602Rev.1_KuleszaAdamsEtAl.pdf);
the tolerance is our project choice and adequate tail sampling is still needed.

`scalar_precision_mode=simultaneous_bound` explicitly selects the older,
stricter scalar rule. Stored `interval` fields remain conservative bounded
intervals, not normal intervals inferred from the SE. The report states which
rule controlled scalar acceptance. In `joint_distribution` mode, joint H/O–polar-angle–recoil **bin masses**
require TV error at
most **0.005**. The bound uses simultaneous cell intervals and probability
conservation, including empty cells; per-bin/CDF precision is not substituted
for TV. It does not certify an unbinned density or azimuth distribution.

The scalar finite-sample intervals use two one-sided grid-betting mixtures
(Waudby-Smith & Ramdas, Appendix B.6). Twenty fixed fractions of each admissible
betting interval are averaged, and either side rejects only above `2/alpha`.
The ordinary weighted point estimates are unchanged. Ratios use simultaneous
numerator/denominator enclosures. Complete likelihood-weighted trajectories are
the independent units, not individual collisions; the declared maximum weight
and physical bounds determine their support. No observed-maximum truncation,
Gaussian approximation or self-normalized-weight assumption is introduced.

One quarter of the global error probability covers four scalar means, the
importance-normalization mean and both complete tail partitions across all
allowed cases, attempts and both
cohorts. These bounds are time-uniform and grid-independent, so no spending over
looks or grid levels is needed for scalars. Histogram assessment retains
bounded-iid empirical Bernstein intervals with one quarter of the budget;
interpolation histograms retain one half, including both selectable cohorts.
Training and selected replays never certify themselves. Certification requires
a contiguous deterministic checkpoint prefix; exact repeated observations can
be compressed by integer multiplicity without changing the betting capital.

The method is software-tested and connected to assessment, export and future
signed runs. Its speed-up on actual NLH production is not established. Retained
historical means alone cannot reconstruct a betting interval. The comparison
therefore uses constant sequences at retained ratios, full-path normalization
and ideal unit weights: an optimistic planning scenario, not measured sampling
uncertainty. In that scenario all 18 collision-rate checks pass with betting
(none with the old bound), but all 18 cases still fail the other scalar checks.

The unrestricted rare-tail support remains too costly for the retained
high-energy preflight even with betting. The geometry-preserving tail
estimator below is implemented, but its production feasibility is unresolved.
Kernel refinement, energy interpolation,
TV interpolation, CDF resolution and moment compression have separate numerical
allowances, currently 0.005. They are not a 0.5% bound on total error. All limits
and defaults live in `Policy`; exhausted limits remain explicit. The 5% and
0.005 requirements are project choices, not literature constants.
`distribution_tv_tolerance` controls statistical joint-bin acceptance and
proposal training; `pair_distribution_tv_tolerance` independently controls
deterministic binary-map bin integration and root allowance. Both remain 0.005.

## Tail estimator: literature assessment and implementation contract

Implementation, 2026-09-09. This is part of `certify.py`, not another executable.
Software verification does not certify the physical campaign or its runtime.

[Khodyrev et al. (2011), Section II](https://arxiv.org/pdf/0904.2151) is directly
relevant to rare ion scattering in crystals: weighted showers target otherwise
infrequent encounters, and adaptive entrance-coordinate sampling reduces
between-primary variation. Their thermal-displacement resampling requires its
own physical probability law, including correlations. Our retained snapshots
do not supply that conditional law. We can adapt entrance sampling while keeping
the atoms fixed; moving encountered atoms or independently choosing their
scattering angles would change the model. Cloning a fixed trajectory without
changing a legitimately sampled input would not supply independent evidence.

[Etore & Jourdain (2010)](https://arxiv.org/abs/0711.4514) supports adaptive
allocation between strata. Its asymptotic optimality does not supply our
finite-sample confidence guarantee. For known input-stratum probabilities
`p_h`, variance `v_h` and cost `c_h`, minimizing stratified variance at fixed
cost gives `n_h proportional to p_h * sqrt(v_h/c_h)`. This cost extension follows
from constrained minimization; multiple observables require a worst-tolerance
objective, not that single-observable rule verbatim. Unknown outcome-tail
probabilities cannot be inserted as though they were known input volumes.
[He & Owen (2014)](https://arxiv.org/html/1411.3954) supports the existing
defensive-mixture alternative when convenient input strata are unavailable.

The implementation has four parts:

1. **Resolve tail contributions, without discarding them.** Partition each
   complete history's recoil and angular score into six fixed bands, expressed
   as fractions of its proven support. Every observation contributes to exactly
   one band per score. Estimate each unconditional weighted contribution;
   sum their simultaneous intervals and intersect with the separately budgeted
   whole-score interval. Empty bands retain positive upper limits. These are
   not conditional strata with assumed known probabilities. The existing
   event products separately retain H/O and joint angle--recoil information.
   Proposal training adds band-specific influences and entrance seeds, scaled
   by the total observable's tolerance, not a relative requirement for each
   vanishing tail. The pair solver already resolves contributing impact
   intervals; pair-area probabilities are not whole-history tail probabilities.
2. **Bound scores from physics and the actual sampling law.** In units where
   `m` and `M` are projectile and target rest energies, stationary-target elastic
   kinematics gives
   `T_max(E) = 2*M*E*(E+2*m) / ((m+M)^2 + 2*M*E)`.
   The current two-body backend uses the equivalent `2*p_cm^2/M`. Along a
   non-energizing primary history, total recoil cannot exceed entrance energy
   `E0`; events each transferring at least `T_lo > 0` number at most
   `floor(E0/T_lo)`. Each angular score `1-cos(theta)` is at most two.
   These are physical support bounds, not observed maxima. Bounds on weighted
   band scores must also include the generating-density ratio over its whole
   support. A bound for one encounter, or a straight incoming ray, does not
   bound later encounters after deflection. The exact kinematic support follows
   from energy--momentum conservation; the history bounds follow by summing
   nonnegative transfers. Neither establishes that rare tails are negligible.
3. **Demonstrate usefulness before broad allocation.** Freeze the pilot-trained
   design and compare independent baseline/targeted histories on limiting
   scalar and joint-bin uncertainties per CPU-second. Account for training
   cost, all tail bands, zero-observation bands and simultaneous coverage.
   Smaller empirical variance alone is insufficient if the proven support
   still makes the confidence requirement infeasible. Preserve fallback
   coverage and report an unresolved bound rather than asserting a speed-up.
4. **Use the same production loop and products.** Allocate the next blocks to
   the limiting cases/bands, refine numerical maps only where their independent
   checks fail, checkpoint every completed block, and export only accepted
   tables. Store proposal, bounds, confidence spending and compatibility
   identities with the evidence. Reuse unchanged qualified pair maps; changed
   acceptance must not relabel old signed trajectories or require duplicated
   production sampling. `prepare --reuse-pairs` checks model, backend,
   environment, numerical policy and payload hashes, records their source,
   and defers transport until independent consumer checks pass again. It does
   not import incompatible phase statistics.

The angular history bound is derived in `physical_scalar_bounds`. With
`K=max((M_max/m+E0/(2m))/sqrt(2), 2/log(2))`, it is
`sum(1-cos(theta)) <= K*log(E0/E_floor)+2`. The final two allows the single
collision crossing the floor. This is about 35.22 at 100 MeV for carbon with
a 1-keV floor, independent of the 10,000-collision safety cap.

The safety cap was previously also used as a count-confidence support. That
is not a physical justification. Production assessment, histogram interpolation
and export now use an energy-based count bound from the qualified pair maps.
Their minimum CM angle and stationary-target masses give a positive lower
bound `f_min` on fractional recoil. Logarithmic energy decay then bounds the
number of encounters by `ceil(log(E0/E_floor)/(-log(1-f_min)))+1`. Area-linear
and log-energy interpolation preserve angle minima. The implementation guards
floating-point subtraction and refuses an unresolved recoil floor. This is a
bound on the implemented interpolant; numerical-map uncertainty remains a
separate test. It can be much larger than the safety cap. Legacy-cap comparison
columns are historical diagnostics, not valid physical confidence certificates.

Bounds remain broad for rare multiple encounters. Score partitioning does not
prove a smaller maximum likelihood-weighted tail score, and cannot manufacture
a small probability for an unseen tail. New signed runs allow bounded independent
tail pilots despite aggregate-only historical warnings; the existing pilot-first
scheduler prevents broad expansion until pilot feasibility passes. The diagnostic
uses observed band means only when complete fresh event cohorts provide them.
No new physical pilot or measured NLH speed-up is claimed by software tests.

[Mendenhall & Weller (2012)](https://arxiv.org/pdf/1109.0910) demonstrates why
cross-section biasing must compensate propagation/beam depletion as well as
interaction weights. That transport law is not a replacement for our explicit
geometry. Likewise, the Geant4 manual's optional cross-section hardening is
not evidence that unweighted impact-parameter scaling is valid for our scores.
Conditional collision/track-length estimators would require a proven conditional
encounter law for these snapshots before replacing any recorded collision.

The implemented choice is therefore tail-aware, geometry-preserving entrance
sampling with physical score bounds, not independent atom resampling, a fitted
tail cutoff or another confidence-formula change. This is a literature-informed
NLH adaptation requiring production validation; no cited paper establishes its runtime or
guarantees all-case convergence. Binary DCS remain complete; structured-ice
tail certification and its speed-up remain unresolved.

## Execution and restart

The two stages share the same model, signed policy and atomic checkpoints:

1. `calibrate ROOT --workers N` qualifies the binary numerical maps, trains
   per-case sampling and a finite output-grid family, checks doubled geometric
   search windows on calibration histories, and stops at the frozen-design
   boundary. It writes `calibration/index.json`; it does not collect the
   independent baseline/targeted production cohorts. Calibration-ready means
   ready for independent sampling, not that final phase distributions or their
   energy interpolation are already qualified.
2. `produce ROOT --workers N` requires the intact calibration handoff and uses
   separately seeded baseline/targeted histories, never pooled training rows.
   It writes `production/report.json` and exports accepted products to `tables/`.
   The predetermined output-grid family may be refined without changing the
   physics or proposal. If energy interpolation requires a new case, that case
   trains and freezes its own design before independent sampling; existing
   frozen cases are not retrained. Missing evidence or exhausted budgets remain
   explicit, not automatic certification.

These commands do not change the physical hard/soft boundary. Their output
separation does not itself make rare-tail uncertainty cheap. In `joint_distribution` mode, joint-TV and numerical-distribution requirements may still prevent export. Scalar mode does not claim these distribution products.

From the repository root, using one Python environment:

```bash
python -B physics/elastic/hard_collisions/certify.py prepare \
  physics/elastic/hard_collisions/runs/carbon_ice_scalar \
  physics/elastic/hard_collisions/carbon_ice.json \
  --reuse-pairs physics/elastic/hard_collisions/runs/carbon_ice
python -B physics/elastic/hard_collisions/certify.py status \
  physics/elastic/hard_collisions/runs/carbon_ice_scalar
```

Preparation submits nothing. Another matrix needs a separately signed root,
optionally with `--material amorphous --projectile O`. Schema 7 cannot silently
resume earlier acceptance or change signed inputs. Historical evidence is
read-only and is not pooled into a new certificate.

To avoid repeating unchanged pair quadrature, add `--reuse-pairs OLD_ROOT` to
`prepare NEW_ROOT SPEC`. Only the new root is written; the old signed campaign
is preserved. Incomplete preparation remains explicitly non-runnable.

On a compute allocation use `calibrate ROOT --workers N`, followed by
`produce ROOT --workers N` after calibration is ready. `run` remains the combined
diagnostic/controller-test route; it is not the recommended staged entry point.
[run_certifier.pbs](run_certifier.pbs) provides a 256-CPU/512-GB launcher via
the `idle` routing queue to `idlex` (direct `idlex` submission is disabled);
set absolute `CERTIFIER_ROOT`, `CERTIFIER_CODE` and `CERTIFIER_PYTHON` paths.
Set `NLH_STAGE=calibrate` (the default) or `NLH_STAGE=produce`; bounded PBS
continuations preserve the selected stage and worker count.
When overriding PBS resources, set `NLH_WORKERS` to that allocation's CPU count.
The queue replenishes as workers finish, without a slowest-worker wave barrier.
Requested workers must fit the allocation and available task graph.

One controller owns the campaign lock. Pair evaluations and trajectory blocks
are committed atomically with provenance. Restart adopts checked products and
fills missing ranges; a gap does not discard later completed blocks. Per-history
seeds and sorted reductions do not depend on worker completion order.
Cost-informed adaptation uses saved pilot timings and frozen designs. Only
unfinished work may repeat. `NLH_CONTINUATIONS` permits explicitly bounded
wall-time continuations, not restart after user cancellation or failed gates.
Outside PBS, only bounded smoke checks of at most 64 histories and two workers
are allowed; shared login-node memory stays below 8 GB.

`feasibility ROOT` refreshes planning; `verify-differential ROOT` checks the
binary DCS index, maps and independent interpolation evidence without new
solves. `verify-tables ROOT` checks a completed phase handoff without
trajectories. The canonical campaign keeps one current
readiness/feasibility report, immutable designs, reusable quadrature caches,
event blocks with receipts and accepted products. New blocks also retain
free-flight segments, incoming energies/directions and terminal exposure, so
later encounter-law analysis need not repeat complete histories.
[verification.json](verification.json)
records software/runtime evidence separately. The scalar-production revision is recorded separately from historical physical checks.

To repeat the bounded estimator comparison without new collisions:

```bash
python -B physics/elastic/hard_collisions/certify.py compare-scalar-intervals \
  physics/elastic/hard_collisions/runs/carbon_ice
```

This verifies the source manifest signature and replaces only
`scalar_estimator_comparison.json`. It does not claim to validate all historical
raw data or change the campaign's signed policy. Broad sampling stays paused.

## Primary references and applicability

- [Mendenhall & Weller (2005)](https://arxiv.org/abs/physics/0406066) and the
  [Geant4 ion-scattering reference](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/elastic_scattering/nuclearrec.html)
  support direct classical screened-potential scattering and consistent total
  area/event kinematics. They do not validate our NLH potential, precise hard
  boundary or density-only transport through correlated ice.
- [He & Owen (2014)](https://arxiv.org/html/1411.3954) supports defensive mixture
  optimization. The documented CPU-cost reparameterization and TV surrogate
  are implementation choices requiring independent performance tests.
- [Maurer & Pontil (2009), Theorem 4](https://arxiv.org/pdf/0907.3740) supplies the
  bounded-iid mean inequality, applied to both signs with a finite-family union
  bound. No unweighted multinomial formula is applied to weighted trajectories.
- [Waudby-Smith & Ramdas (2023), Appendix B.6](https://arxiv.org/html/2010.09686v7#A2.SS6)
  supplies diversified constant-grid betting. Our two one-sided mixtures use
  equal error spending and monotone outward-bracket inversion. The fixed grid
  is not claimed optimal, nor does this literature validate the NLH physics.
- [Vasques & Larsen (2014)](https://ricvasques.github.io/assets/papers/2014ANE1.pdf)
  explains direction/path-dependent transport in correlated media. It motivates
  retaining geometry, not direct validation or a supplied closure for
  energy-changing NLH transport through ice.

Run `python -B -m pytest -q physics/elastic/hard_collisions/tests` for software
tests; `NLH_REAL_RUNTIME_TESTS=1` includes bounded real-runtime checks. Software
tests, numerical refinement and physical validation are distinct categories.

## Retained read-only diagnostics

Older diagnostics remain useful for provenance and proposal seeds:

```bash
python -B physics/elastic/hard_collisions/audit_variance.py \
  CAMPAIGN_ROOT CAMPAIGN_ROOT/variance_diagnostic \
  --energies-ev 100000 10000000 100000000 --workers 4 --tail-batches 3
python -B physics/elastic/hard_collisions/replay_saved_tails.py \
  CAMPAIGN_ROOT/variance_diagnostic \
  --legacy-commit 5f4adfe2585c4707c6d53dba10d580e876e1a688 \
  --case seed1000/basal_a_axis/100000000eV
```

The audit freezes a committed prefix and identifies variance-bearing batches.
Saved distribution arrays contain only uniform-component histories; a selected
extreme replay is not an unbiased population sample. Replays verify original
runtime/geometry/kernel provenance and retain full events. These tools neither
alter checkpoints nor submit jobs. Their delta-method uncertainties are
diagnostic, not the current acceptance rule.
