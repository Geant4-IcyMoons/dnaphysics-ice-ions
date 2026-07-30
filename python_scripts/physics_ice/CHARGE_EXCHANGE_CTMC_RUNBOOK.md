# Running the C, Li, O, and S charge-exchange CTMC generators

This is the operational guide for generating microscopic charge-changing
cross sections for carbon, lithium, oxygen, and sulfur projectiles colliding
with an isolated H2O molecule. The scientific definitions and their sources
are documented separately in
[`CHARGE_EXCHANGE_CTMC_PROVENANCE.md`](CHARGE_EXCHANGE_CTMC_PROVENANCE.md).

The generators write cross sections in cm2 per H2O molecule. They do not
apply an ice density. A later Geant4 table model must form the macroscopic
cross section as `Sigma(E,q) = n_H2O * sigma(E,q)`. At present, this repository
generates the C/Li/O/S tables but does not yet register the Geant4 process
that consumes them and changes the transported heavy ion's charge state.

## Entry points and default grids

Run the element-specific entry point; do not invoke
`charge_exchange_ctmc.py` directly.

| Element | Entry point | Charge states | Default energy grid | Required projectile-loss cutoffs |
|:---|:---|:---|:---|---:|
| C-12 | `generate_carbon_charge_exchange_ctmc.py` | C0--C6+ | 41 log points, 1--10000 keV/u | 6, ordered C0--C5+ |
| Li-7 | `generate_lithium_charge_exchange_ctmc.py` | Li0--Li3+ | 10 published points, 1--10000 keV/u, plus 100 MeV total Li-7 | 3, ordered Li0--Li2+ |
| O-16 | `generate_oxygen_charge_exchange_ctmc.py` | O0--O8+ | 41 log points, 1--10000 keV/u | 8, ordered O0--O7+ |
| S-32 | `generate_sulfur_charge_exchange_ctmc.py` | S0--S16+ | 41 log points, 1--10000 keV/u | 16, ordered S0--S15+ |

All four default to 101 impact-parameter points and 10,000 trajectories per
orbital and impact point. These are production statistics. A run with smaller
values is useful for software testing but is not a production cross-section
calculation.

The runtime dependencies are Python, NumPy, SciPy, Numba, and tqdm. On the
cluster, the PBS scripts default to the `yoffe_venv` Conda environment. Verify
the environment before a large submission:

```bash
source /apps01/apps/anaconda3-2022.10/etc/profile.d/conda.sh
conda activate yoffe_venv
python -c 'import numba, numpy, scipy, tqdm; print("CTMC dependencies OK")'
```

## Mandatory convergence inputs

Every run requires three quoted lists:

- `START_SEPARATION_AU`: one initial core separation for every selected
  energy: 41 values for the default C/O/S grids and 11 for the default Li
  grid.
- `TARGET_BMAX_AU`: five H2O-orbital impact cutoffs, ordered
  `1b1,3a1,1b2,2a1,1a1`.
- `LOSS_BMAX_AU`: one projectile-electron-loss cutoff for every non-bare
  projectile charge state; the required counts and orders are in the table
  above.

These are numerical-domain convergence controls, not adjustable physical
coefficients. Several intermediate values are not published, so the code
intentionally has no fallback values. Obtain them from paired boundary
convergence runs and retain the resulting boundary-probability evidence.
Never fill a list by interpolation or copy another element's values unless a
documented convergence calculation establishes those values for the requested
projectile and grid.

Comma- and semicolon-separated lists are accepted. Semicolons are convenient
for PBS environment variables:

```bash
export START_SEPARATION_AU='value_1;value_2;...;value_n'
export TARGET_BMAX_AU='value_1;value_2;value_3;value_4;value_5'
export LOSS_BMAX_AU='value_1;value_2;...;value_m'
```

The placeholders above are deliberately not runnable physics values. All
compute shards and the final merge must receive byte-for-byte equivalent
physics and grid arguments.

## Check a run before integrating

From the repository root, select the element and pass the converged lists:

```bash
ATOM=carbon
python -u "python_scripts/physics_ice/generate_${ATOM}_charge_exchange_ctmc.py" \
  --workers 64 \
  --trajectory-chunk-size 8 \
  --start-separation-au "${START_SEPARATION_AU}" \
  --target-bmax-au "${TARGET_BMAX_AU}" \
  --loss-bmax-au "${LOSS_BMAX_AU}" \
  --dry-run
```

`--dry-run` validates the arguments and prints the energy/charge/impact grid,
the exact remaining trajectory count, and the work-chunk count. Remove only
`--dry-run` to start the calculation.

Use `--help` on an element entry point for every optional control. In
particular, `--energies`, `--charges`, `--impact-points`, and
`--trajectories` change the scientific grid or Monte Carlo statistics and
therefore also change the checkpoint signature.

## CPU parallelism

The numerical work has two levels of parallelism:

1. Within one node, one Python worker process runs per allocated CPU core.
   The Numba DOP853 trajectory kernel is CPU-bound. BLAS/OpenMP thread pools
   are fixed at one thread to avoid nested oversubscription.
2. Across nodes, independent *shards* own disjoint energy/charge/impact
   points and write separate checkpoints into the shared output directory.

The code is not MPI-based. One Python process cannot use CPUs on several
nodes. Consequently, do not request `select=4:ncpus=64` for one process and
expect it to use 256 cores. Submit four 64-core shard jobs instead.

The supplied PBS scripts request:

```text
select=1:ncpus=64:mem=4gb
```

The 4 GB is total memory for the 64-core job, not 4 GB per core. Live carbon
production used approximately 3.6--3.8 GiB per 64-worker shard. Keep the 4 GB
request unless a measured run for the chosen element establishes a different
requirement.

| Desired concurrent CPUs | Recommended layout | Notes |
|---:|:---|:---|
| 1--64 | one shard on one node | Set workers to the allocated cores. |
| 256 | four 64-core shards | Usually schedules sooner than one very large node. |
| 1,280 | twenty 64-core shards | Requires twenty array elements to run concurrently. |
| 5,376 | eighty-four 64-core shards | The existing carbon production layout. |

If the cluster has a genuine 256-core single node, a copy of an element PBS
script may instead request `select=1:ncpus=256:mem=16gb`. The script reads
`PBS_NCPUS` and will start 256 local workers without another code change.
Measure resident memory and throughput on one such job before scaling it out.

Any positive shard count is correct: ownership is deterministic and
non-overlapping. Multiples of the number of charge states often make the
point partition easier to inspect:

| Element | Charge states | Example shard counts |
|:---|---:|:---|
| carbon | 7 | 7, 14, 28, 84 |
| lithium | 4 | 4, 8, 20, 84 |
| oxygen | 9 | 9, 18, 45, 90 |
| sulfur | 17 | 17, 34, 85 |

Different charge states and energies have different trajectory costs, so
equal point counts do not imply equal wall time. Extra shards improve
load distribution, but total speed remains limited by the number of array
elements that PBS actually runs concurrently.

## PBS launcher example

[`pbs/launch_charge_exchange_ctmc_example.sh`](../../pbs/launch_charge_exchange_ctmc_example.sh)
splits an arbitrary shard count into PBS arrays, respecting this cluster's
default 50-element array limit. It forwards the mandatory lists by environment
variable and never supplies scientific values itself.

For example, submit 84 carbon shards, each using the 64-core resource request
in the carbon PBS file:

```bash
export START_SEPARATION_AU='converged values for all selected energies'
export TARGET_BMAX_AU='five converged values'
export LOSS_BMAX_AU='six converged carbon values'

bash pbs/launch_charge_exchange_ctmc_example.sh submit carbon 84
```

The launcher prints every returned PBS job ID. The requested allocation is
84 nodes and 5,376 cores if all array elements run simultaneously; queued
elements consume no CPU. To plan a submission without calling `qsub`, use:

```bash
bash pbs/launch_charge_exchange_ctmc_example.sh plan carbon 84
```

After every compute array element finishes successfully, submit the one-core
merge:

```bash
bash pbs/launch_charge_exchange_ctmc_example.sh merge carbon 84
```

This PBS installation does not provide a reliable array-wide dependency, so
the launcher intentionally does not submit the merge automatically. Check all
array elements first. Use the corresponding element name (`lithium`,
`oxygen`, or `sulfur`) and its correctly sized cutoff lists for another atom.

The launcher forwards these optional environment variables when set:
`MINIMUM_INTEGRATION_TIME_AU`, `CONDA_SH`, `CONDA_ENV`, `PYTHON_BIN`,
`TRAJECTORY_CHUNK_SIZE`, `PENDING_FACTOR`, `CHECKPOINT_EVERY`,
`CHECKPOINT_SECONDS`, `RTOL`, `ATOL`, `RETRY_RTOL`, `RETRY_ATOL`,
`MAXIMUM_RELATIVE_ENERGY_DRIFT`, `BOUNDARY_EXTENSION_FACTOR`, `EXTRA_ARGS`,
and `REPO_ROOT`. `EXTRA_ARGS` is for trusted, whitespace-separated command
line options. For example, a deliberately non-production software test could
set:

```bash
export EXTRA_ARGS='--energies 100 --charges 0 --impact-points 3 --trajectories 2'
```

Do not attach that setting to a production run.

## Checkpoints, stopping, and resuming

Each shard writes an atomic checkpoint at least every 100 completed
probability points or 300 seconds by default. PBS `SIGTERM` and `SIGUSR1`
trigger worker shutdown and a final checkpoint. The parent owns both tqdm
progress bars, so PBS output shows attempted trajectories, completed grid
points, and timestamped checkpoint messages.

To resume, submit the same element with the same shard count, shard indices,
physics/grid controls, seed, and output directory. Matching checkpoints load
automatically. It is safe to change worker count, pending factor, or trajectory
chunk size when resuming because deterministic trajectory identities do not
depend on scheduling. Do not use `--no-resume` for production recovery.

If a shard reports a deterministic trajectory failure, it writes a JSON
diagnostic beside the checkpoint and exits without counting that trajectory.
Investigate the saved identity; do not hide it by increasing
`--max-failure-fraction`.

## Output and merge

Unmerged checkpoints are written under:

```text
cross_sections/<element>_charge_exchange/
```

For a multi-shard run they are named:

```text
<element>_charge_exchange_ctmc_checkpoint.shard-NNNNN-of-MMMMM.npz
```

The merge refuses to proceed if any shard checkpoint is missing, incomplete,
or has a different configuration signature. A successful merge writes:

- `<element>_charge_exchange_h2o.dat`
- `<element>_charge_exchange_h2o.csv`
- `<element>_charge_exchange_probabilities.npz`
- `<element>_charge_exchange_metadata.json`
- the canonical merged checkpoint

The `.dat` and `.csv` columns contain energy, charge state, single capture,
target ionization, single loss, loss ionization, charge decrease, and charge
increase. Cross sections remain microscopic values in cm2 per H2O molecule.

Carbon-specific validation history and the exact shard ownership equation are
in [`CARBON_CTMC_PARALLEL.md`](CARBON_CTMC_PARALLEL.md).
