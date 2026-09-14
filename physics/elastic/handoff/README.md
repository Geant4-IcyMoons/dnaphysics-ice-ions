# Combined NLH–ZBL handoff study

Both [ice phases](../../../models/ice/README.md) use the same microscopic
kernel. `kernel.py` selects NLH inside the turning-potential boundary and
ZBL outside at each encounter; `run.py` propagates the resulting trajectory.
The potentials use the same isotope kinematics in this runner. No potential
blending or empirical correction is introduced.

## Run from the repository root

Install `requirements.txt` in a Python environment, then prepare the study:

```bash
python -m physics.elastic.handoff.study --output physics/elastic/handoff/runs/carbon/study.json
```

On a compute allocation, run a binary task (indices 0–11):

```bash
python -m physics.elastic.handoff.binary physics/elastic/handoff/runs/carbon/study.json --task-index 0 --output physics/elastic/handoff/runs/carbon/binary
```

This compares orbit orders 64/128 and impact quadrature resolutions separately
for all three boundaries, all three cutoffs, and full ZBL. Each product is saved
immediately; repeating the command resumes completed products. Refine further
with `--points` if the impact check fails. Inspect both moments separately.

Run a trajectory cohort (indices 0–239), explicitly choosing a cutoff after
reviewing the binary results. The following cutoff is an example, not qualified:

```bash
python -m physics.elastic.handoff.run run physics/elastic/handoff/runs/carbon/study.json --case-index 0 --cutoff-ev 1e-5 --workers 8 --output physics/elastic/handoff/runs/carbon/ice
python -m physics.elastic.handoff.run reduce --output physics/elastic/handoff/runs/carbon/ice
```

PBS submission, from the repository root:

```bash
qsub -v PYTHON_BIN=/absolute/path/to/python,STUDY=physics/elastic/handoff/runs/carbon/study.json,OUTPUT=physics/elastic/handoff/runs/carbon/ice,CUTOFF_EV=1e-5,CASE_INDEX=0 physics/elastic/handoff/run.pbs
```

Use `qsub -J 0-239 ...` only when all planned cases are justified. Pilot cases
are identified by `stage=pilot` in the manifest; their indices need not be
contiguous. The launcher requests eight CPUs via `idle` (routing to `idlex`).
Set `ORBIT_ORDER=128` for an independent integration-order check and change
`CUTOFF_EV` for cutoff studies. Compare those results as well as boundaries;
the reducer currently screens boundary changes within each order/cutoff group.

Each bounded history block has worker-independent seeds and an atomic checkpoint.
Repeat the command to resume; raising `--max-histories` preserves completed
blocks. One writer owns each cohort/settings directory. Source changes require
fresh preparation. The runner targets 5% SE on stopping and angular transport,
not on cutoff-dependent collision counts or every rare process. Reaching the
history budget is reported as `sampling_limit`, not success. A track that reaches the terminal energy is retained with its traveled length
and residual energy. Collision limits and solver failures still fail explicitly;
successful blocks in the same wave are preserved and checkpoint gaps are filled
on resume before assessing precision. No stopped histories are discarded.

## Scope and evidence

The [protocol](PROTOCOL.md) defines 180 combined and 60 full-ZBL control cohorts,
with later replicas conditional on the pilot evidence. `study.py` verifies
structure hashes and assigns deterministic identifiers. `binary.py` records
numerical refinement; `run.py` exports stopping in eV/angstrom and angular
transport in inverse angstrom, standard errors, branch counts and ambiguous
encounters. Raw per-history moments permit reanalysis. These are finite-path
ice responses, not microscopic cross sections; the latter come from binary.py.

The compute-node smoke test (`python -m physics.elastic.handoff.smoke`) checks
both phases, both branches, conservation, orbit refinement and checkpoint reuse.
It does not establish convergence of the full parameter range. All results
retain `handoff_qualified=false` until a scientific assessment establishes the
supported domain. Full ZBL is a comparator, not experimental truth.

The direct orbit implementation is a runnable reference for the dedicated
study. It does not yet use interpolated pair maps. The default sampler uses uniform entrance positions. The collision-tube
proposal remains an explicit option; its benefit is condition dependent.
Measure variance per CPU-second before scaling. Simultaneous
many-atom forces, recoil cascades and electronic interactions are not included.

## Sampling and terminal energy

`--tube-fraction 0.8` draws 80% of entrance positions from the existing periodic
collision-tube proposal and 20% uniformly. Logarithmic impact-area intervals
allocate draws to rare close encounters. The proposal density sums all tube
preimages, including overlaps; every whole-history numerator and track-length
denominator receives the exact target/proposal weight. The uniform component
bounds weights by five. Set `--tube-fraction 0` for an independent uniform
baseline. The 80% fraction is a numerical tuning choice, not a physical factor.
This follows defensive mixture importance sampling [Hesterberg (1995)](https://doi.org/10.1080/00401706.1995.10484303);
no universal speedup is implied by that reference.

The terminal energy defaults to 1 eV and can be raised with
`--terminal-energy-ev` (PBS: `TERMINAL_ENERGY_EV`). A track that crosses it ends
normally for the declared above-threshold observable. Its residual energy is
recorded, not assigned to local deposition or converted into fictitious recoils.
Outputs are ratios of weighted recoil/angle sums to weighted traveled length,
including the variable lengths of terminated tracks. They are not the response
of a particle propagated through the full requested path below the floor.
The residual-energy-per-tracked-length diagnostic does not bound missing angular
scattering; compare terminal settings (e.g. 1 versus 2 eV) where termination is
frequent. All handoff and terminal-sensitivity qualification flags remain false.

`--tube-fraction` and terminal energy enter the checkpoint identity. Old uniform
schema-1 blocks are preserved but not silently mixed with new schema-2 histories.
Streams are independent between different cutoff, orbit, floor and proposal
settings, and deterministic across worker counts at fixed settings. Progress
reports include mean weights, effective sample size, terminated fraction and
worker/CPU seconds. CPU-scaled estimator variance, rather than trajectories per
second alone, measures whether the proposal helps. The standard errors use
centered whole-history residuals and retain denominator covariance.

### Selecting the sampler

Uniform sampling (`--tube-fraction 0`) is the default. The first 1,024-history
tube tests did not establish a general efficiency gain: in crystalline ice,
a single history contributed 82–92% of the estimated angular variance. Do not
clip these histories or treat their absence in a small control as convergence.
Compare independent controls at the same physical settings and history budget.
`relative_variance_cpu_seconds` measures estimated cost to relative precision
(lower is better); `largest_history_variance_fraction` exposes unstable variance
estimates. These diagnostics are not additional qualification tolerances.
Retain a nonuniform proposal only after replicated controls support its benefit
for both observables. Use fresh production histories after sampler selection;
the pilot estimates and sampler choices are not independent.
