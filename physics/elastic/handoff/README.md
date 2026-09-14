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
history budget is reported as `sampling_limit`, not success. A trajectory that
hits its energy floor or collision limit fails explicitly rather than silently
biasing the sample. Completed blocks remain available after such failures.

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
study. It does not yet use interpolated pair maps or optimized importance
sampling; measure throughput before launching a large campaign. Simultaneous
many-atom forces, recoil cascades and electronic interactions are not included.
