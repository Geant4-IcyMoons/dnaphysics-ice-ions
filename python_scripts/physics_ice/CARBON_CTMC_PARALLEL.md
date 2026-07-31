# Carbon CTMC parallel execution

`generate_carbon_charge_exchange_ctmc.py` uses process-level parallelism
because every trajectory is independent and the Numba DOP853 kernel is
CPU-bound. It explicitly limits BLAS/OpenMP libraries to one thread per
process, so `--workers 64` maps to 64 CPU cores without nested
oversubscription.

## One node

The default `--workers 0` reads the process CPU affinity, including PBS/Slurm
cpusets. An explicit production invocation is:

```bash
python -u python_scripts/physics_ice/generate_carbon_charge_exchange_ctmc.py \
  --workers 64 \
  --trajectory-chunk-size 8 \
  --start-separation-au '...' \
  --target-bmax-au '...' \
  --loss-bmax-au '...'
```

The three boundary arguments are deliberately mandatory. The paper gives
only endpoint/range facts for the per-energy initial separation and the
orbital-/charge-specific `bmax` values; it does not publish the intermediate
values. They must come from the paper's stated convergence procedure and are
never synthesized by this code.

## Numerical tolerance

Production uses DOP853 `rtol=1e-11` and `atol=1e-13`. These values were
selected with a paired 10,000-trajectory convergence calculation for the
tolerance-calibration channel (1 keV/u, C0, 3a1, b=0.25 a.u.). Relative to a
further refinement to `3e-12/3e-14`, all event-probability changes were at
most 1.21 binomial standard errors. The maximum energy drift was
`5.39e-4`, versus `1.91e-2` at the former `1e-10/1e-12` setting and
`5.28e-3` at the further refinement. No trajectory failed or ended in the
paper's non-physical both-bound state.

The relative-coordinate origin follows Appendix A of the paper: H2O is the
initial reference core for target ionisation, while the projectile is the
initial reference core for projectile electron loss (the paper's prescribed
interchange of `p` and `t`). During either scheme, the solver applies that
same exact `p`/`t` coordinate interchange whenever the electron becomes closer
to the other core. This keeps the electron local after charge transfer and
avoids subtracting two positions displaced by up to 20,000 a.u.; it changes no
equation or physical event definition. If an adaptive integration still fails
during a singular close encounter, the identical initial state is retried at
`1e-12/1e-14`, then `1e-13/1e-15`. This changes no random draw, equation,
boundary, or event definition. A finite endpoint with relative total-energy
drift above `1e-3` is likewise retried at all three physical-time tolerances
and then, if necessary, with the identical equations in the positive Sundman
time parameter already used for singular close encounters. It is never
committed unless an independent endpoint check meets the same numerical
quality limit. This is a solver-accuracy check, not a new physical event
definition.

The exact previously worst production channel (1 keV/u, C0 projectile loss,
`b=5` a.u.) was rerun for all 10,000 deterministic trajectories after this
coordinate correction. All trajectories completed, and the maximum relative
total-energy drift was `4.00e-5`.

The paper prescribes increasing the initial separation and integration
interval when the non-physical `E_et<=0, E_ep<=0` probability is not
negligible. At 1 keV/u, increasing the supplied separation from 10,000 to
20,000 a.u. resolved all five saved both-bound trajectories into the paper's
physical channels. Production therefore starts at 20,000 a.u. at 1 keV/u. If
an individual endpoint is still both-bound, its identical sampled phase is
rerun at geometrically doubled boundaries until it reaches one of the paper's
physical classifications. A both-bound endpoint is never assigned to a
channel.

Each work item contains at most eight trajectories from one physical channel
by PBS default. Use `--dry-run` to see the exact
trajectory and work-chunk counts before submitting. The two `tqdm` bars are
owned by the parent process and report completed trajectories and finalized
probability points.

The checkpoint includes raw event counts for every completed contiguous chunk.
Changing `--workers`, `--pending-factor`, or `--trajectory-chunk-size` when
resuming is safe and does not change any trajectory's random stream. By
default, an atomic checkpoint is written after 100 finalized probability
points or 300 seconds, whichever comes first. Each write is recorded in the
PBS log with a timestamp and completed point/trajectory counts.

Rerunning the identical command loads the checkpoint automatically. PBS
`SIGTERM` and `SIGUSR1` signals trigger prompt worker shutdown and a final
checkpoint before exit; at worst, only unfinished in-flight chunks are
discarded.

## Multiple nodes

Nodes use stable, non-overlapping point shards and write separate atomic
checkpoints to a shared output directory. Charge is the fastest-varying
partition index:

```text
point_id = q_index + n_charge * (impact_index + n_impact * energy_index)
owner = point_id modulo N
```

Thus `N=7` assigns one complete carbon charge state to each 64-core job.
`N=14`, `21`, and so on assign two, three, and so on jobs per charge state.
Any other positive `N` still produces disjoint, near-equal point partitions.
More generally, for `N=7*k`, charge `q` is owned by jobs
`q, q+7, ..., q+7*(k-1)`. Within that charge, job `q+7*r` owns precisely the
energy/impact points satisfying
`(impact_index + n_impact*energy_index) modulo k = r`.

For the default 41-energy, 101-impact, 10,000-trajectory grid:

| Jobs | Cores | Partition |
| ---: | ---: | --- |
| 7 | 448 | one 64-core job per charge |
| 14 | 896 | two 64-core jobs per charge |
| 84 | 5,376 | twelve 64-core jobs per charge |

With seven jobs, each charge owns 4,141 energy/impact points. Charges C0
through C5 each evaluate 200,900,000 trajectories; fully stripped C6
evaluates 167,690,000 because it has no projectile-loss channel.

This PBS server limits an array to 50 elements, so the 84 production shards
are submitted as two arrays with non-overlapping offsets:

```bash
qsub -J 0-41 \
  -v START_SEPARATION_AU,TARGET_BMAX_AU,LOSS_BMAX_AU,SHARD_COUNT=84,SHARD_OFFSET=0 \
  pbs/generate_carbon_charge_exchange_ctmc.pbs
qsub -J 0-41 \
  -v START_SEPARATION_AU,TARGET_BMAX_AU,LOSS_BMAX_AU,SHARD_COUNT=84,SHARD_OFFSET=42 \
  pbs/generate_carbon_charge_exchange_ctmc.pbs

# After every array element finishes successfully:
qsub -l select=1:ncpus=1:mem=4gb \
  -v START_SEPARATION_AU,TARGET_BMAX_AU,LOSS_BMAX_AU,SHARD_COUNT=84,MERGE_ONLY=1 \
  pbs/generate_carbon_charge_exchange_ctmc.pbs
```

This supplies up to 5,376 concurrent workers. Submit the merge after every
array element finishes successfully; this PBS installation does not support
an array-wide `afterokarray` dependency. The merge verifies that every shard
is complete before writing the final `.dat`, `.csv`, probability archive,
metadata, and canonical merged checkpoint.

Live 64-worker production shards used approximately 3.6--3.8 GiB of resident
memory during the July 2026 audit. The PBS request therefore retains 4 GB
total per 64-core job; this is a job-wide request, not 4 GB per core.
The script submits through the user-accessible `idle` routing queue; PBS then
places the job in the `idlex` execution queue.

The same workflow can be launched without PBS. Start one command per node,
using a shared output directory and distinct indices:

```bash
python -u python_scripts/physics_ice/generate_carbon_charge_exchange_ctmc.py \
  --workers 64 --shard-count 84 --shard-index 0
# Repeat concurrently with --shard-index 1 through 83.
```

After all nodes finish:

```bash
python -u python_scripts/physics_ice/generate_carbon_charge_exchange_ctmc.py \
  --shard-count 84 --merge-shards
```

All physics and grid arguments, including the seed, must match across shards
and the merge command. Worker count and chunk size may differ.

## Tuning

- Start with one worker per allocated physical core. If simultaneous
  multithreading is enabled, benchmark physical cores versus all logical CPUs;
  the adaptive integrator is compute-heavy and may not benefit from SMT.
- Keep `--pending-factor 2` unless the workers starve. Raising it increases
  only the bounded parent queue, not Monte Carlo statistics.
- Reduce `--trajectory-chunk-size` if the final workers have visibly long
  tasks; increase it if trajectories are exceptionally short and scheduling
  overhead becomes measurable.
- Checkpoints are triggered after 100 finalized points or 300 seconds by
  default. Both controls are configurable.
