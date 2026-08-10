# Ion--ice workflow commands

Run these commands from the repository root. They are the maintained entry
points; do not invoke shared implementation modules directly. Numerical
completion does not by itself establish physical validity. Each section links
the corresponding provenance and acceptance documentation.

## Environment check

```bash
cd /gpfs01/work/yoffegid/dnaphysics-ice-ions
source /apps01/apps/anaconda3-2022.10/etc/profile.d/conda.sh
conda activate yoffe_venv
python -c 'import numba, numpy, scipy, tqdm; print("CTMC dependencies OK")'
```

The NEP-MB-pol workflows use their pinned virtual environment directly:

```bash
python_scripts/physics_ice/nep_mbpol/.venv/bin/python --version
```

## Carbon CTMC paper-reproduction grid

The following is the exact fixed-grid configuration launched in August 2026
to reproduce the isolated-H2O carbon calculation before adaptive refinement.
It uses 168 independently restartable 64-core shards, split into PBS arrays of
at most 42 elements. The initial separations and impact cutoffs are retained
numerical-domain choices from the convergence campaign; Liamsuwan and Nikjoo
(2013) did not publish these intermediate numerical values. They are not
physical coefficients and the resulting table remains validation-pending.

```bash
export REPO_ROOT=/gpfs01/work/yoffegid/dnaphysics-ice-ions
export START_SEPARATION_AU='20000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;1000'
export TARGET_BMAX_AU='25;25;25;25;1'
export LOSS_BMAX_AU='20;20;20;20;20;20'
export TRAJECTORY_CHUNK_SIZE=8
export PENDING_FACTOR=2
export CHECKPOINT_EVERY=50
export CHECKPOINT_SECONDS=120
export RTOL=1e-11
export ATOL=1e-13
export RETRY_RTOL=1e-12
export RETRY_ATOL=1e-14
export MAXIMUM_INTEGRATION_STEPS=100000000
export MAXIMUM_RELATIVE_ENERGY_DRIFT=1e-3
export BOUNDARY_EXTENSION_FACTOR=2
export PBS_ARRAY_MAX=42
export EXTRA_ARGS='--output-dir /gpfs01/work/yoffegid/dnaphysics-ice-ions/python_scripts/physics_ice/process_evidence/charge_exchange_ctmc/benchmarking/runs/carbon_paper_reproduction_production/input --energy-min-kev-u 1 --energy-max-kev-u 10000 --energy-points 41 --energy-spacing log --charges 0:6 --impact-points 101 --trajectories 10000'

bash pbs/launch_charge_exchange_ctmc_example.sh plan carbon 168
bash pbs/launch_charge_exchange_ctmc_example.sh submit carbon 168
```

Submitting the same command again resumes signature-compatible checkpoints;
it does not restart completed trajectories. After every base shard finishes:

```bash
bash pbs/launch_charge_exchange_ctmc_example.sh merge carbon 168
bash pbs/launch_charge_exchange_ctmc_example.sh refine-base carbon 168
# Wait for every refine-base shard to finish successfully.
bash pbs/launch_charge_exchange_ctmc_example.sh refine-intervals carbon 168
# Wait for every refine-interval shard to finish successfully.
bash pbs/launch_charge_exchange_ctmc_example.sh merge-refined carbon 168
```

Then run the Liamsuwan-paper comparison and strict release diagnostics:

```bash
qsub -q idlex \
  -v INPUT_DIR=/gpfs01/work/yoffegid/dnaphysics-ice-ions/python_scripts/physics_ice/process_evidence/charge_exchange_ctmc/benchmarking/runs/carbon_paper_reproduction_production/input \
  pbs/benchmark_carbon_charge_exchange_ctmc.pbs
```

The strict release gate additionally requires an independently computed
expanded-impact-cutoff reference; no unpublished boundary value is invented.
Scientific definitions, outputs, restart/rebalance commands, and citations are
in [`process_evidence/charge_exchange_ctmc/validation/RUNBOOK.md`](process_evidence/charge_exchange_ctmc/validation/RUNBOOK.md)
and [`PROVENANCE.md`](process_evidence/charge_exchange_ctmc/validation/PROVENANCE.md).

## Other CTMC projectiles

Use the same five-stage launcher with `lithium`, `oxygen`, or `sulfur`, but
supply that projectile's independently converged separation and cutoff lists.
Do not copy the carbon values:

```bash
export START_SEPARATION_AU='projectile-specific converged values'
export TARGET_BMAX_AU='five projectile-specific converged values'
export LOSS_BMAX_AU='one value for every non-bare projectile charge state'
bash pbs/launch_charge_exchange_ctmc_example.sh plan oxygen 90
```

The runbook lists the required charge ladders and list lengths. CTMC produces
microscopic cross sections per H2O molecule; amorphous and hexagonal rates are
formed later from the selected phase's molecular density.

## Hard NLH collision campaign

Launch the restart-safe adaptive carbon campaign. The supervisor prepares
immutable case waves, splits them across PBS arrays, checkpoints every batch,
and submits the next wave only after the previous wave succeeds:

```bash
MAX_GLOBAL_SHARDS=256 \
  bash pbs/launch_adaptive_nlh_particle_shards.sh C
```

The same entry point supports every registered projectile:

```bash
for projectile in H He O S; do
  MAX_GLOBAL_SHARDS=256 \
    bash pbs/launch_adaptive_nlh_particle_shards.sh "${projectile}"
done
```

The enforced default is a 0.5% numerical tolerance at 95% confidence, not a
0.5% physical-potential uncertainty. H and He remain gated on a non-overlapping
handover with HTran. See
[`process_evidence/hard_nuclear_collisions/validation/README.md`](process_evidence/hard_nuclear_collisions/validation/README.md)
and [`nep_mbpol/bca/README.md`](nep_mbpol/bca/README.md).

## Full-ZBL diagnostic elastic baseline

Generate full-domain atomistic C/O/S backend manifests for both accepted ice
phases from 10 keV through 100 MeV total projectile energy:

```bash
python -m \
  python_scripts.physics_ice.process_evidence.soft_nuclear_collisions.zbl.generate_backend \
  --phases hexagonal_ih_100k amorphous_lda_80k \
  --projectiles C O S \
  --energy-min-ev 1e4 --energy-max-ev 1e8 \
  --transfer-cutoffs-ev 1 10 30 \
  --output-directory \
  python_scripts/physics_ice/process_evidence/soft_nuclear_collisions/validation/runs/zbl_full_atomistic
```

The manifests bind the analytic full-ZBL interaction to the accepted periodic
hexagonal and amorphous coordinates. Carbon q=0..6, oxygen q=0..8, and sulfur
q=0..16 alias the corresponding elemental kernel. H and He remain assigned to
the complete HTran elastic model.

Build and test the validation-pending runtime locally:

```bash
cmake -S proton-pipeline -B proton-pipeline/build
cmake --build proton-pipeline/build -j10
ctest --test-dir proton-pipeline/build --output-on-failure
python -m pytest \
  tests/test_zbl_soft_collision.py \
  tests/test_zbl_atomistic_backend.py -q
```

Run one C/O/S projectile with total kinetic-energy limits in MeV:

```bash
DNA_PHYSICS=ice_am \
DNA_ION_ELASTIC_MODEL=zbl_full \
DNA_ZBL_ALLOW_VALIDATION_PENDING=1 \
DNA_ION_ENABLE_EXCITATION=0 \
DNA_ION_ENABLE_IONISATION=0 \
DNA_ION_CHARGE_EXCHANGE=0 \
  proton-pipeline/build/dnaphysics_proton \
  proton-pipeline/e1_proton.mac 10 oxygen 0.1 100 100000 1
```

`zbl_full` is a complete screened binary-collision baseline, not an additive
soft term. Do not enable NLH or `G4NuclearStopping` in the same run. The
validation protocol is in
[`process_evidence/soft_nuclear_collisions/validation/README.md`](process_evidence/soft_nuclear_collisions/validation/README.md).

## Charge-resolved soft DFT

Run the fresh, non-publishing carbon q=1 numerical branch pilot first:

```bash
qsub -v "REPO_ROOT=/gpfs01/work/yoffegid/dnaphysics-ice-ions,PROJECTILE=C,CHARGE=1,ORIENTATION=oxygen_back,SEPARATION_ANGSTROM=12.0" \
  pbs/run_cdft_branch_gate.pbs
```

This tests reciprocal numerical CDFT-root reproducibility for one exact
geometry; it does not generate or physically validate a soft-potential table.
The full radial/orientation mesh has no accepted branch-state handoff yet and
therefore fails closed. Do not use the legacy smoke, adaptive, supervisor, or
collector launchers for complex CDFT production. CTMC remains separate: a
future Geant4 model will sample a charge transition and then select an
independently validated fixed-charge surface. Exact scope, acceptance gates,
the q=4 follow-up command, and primary citations are recorded in
[`nep_mbpol/soft_dft/README.md`](nep_mbpol/soft_dft/README.md).

## Accepted ice structures

Re-run the published 80 K EPSR amorphous-ice import and validation:

```bash
qsub -q idlex pbs/run_epsr_lda80k_validation.pbs
```

Regenerate the three 100 K experimental-cell ice-Ih replicas on GPUs after
the documented 80 K source preparation exists:

```bash
qsub -q gpuq_private -J 0-2 \
  -N nep_ih_100K \
  -v HEXAGONAL_PROTOCOL=100K_EXPERIMENTAL \
  pbs/run_nep_mbpol_hexagonal.pbs
```

Ordinary collision production should reuse the accepted, checksum-registered
snapshots rather than repeat structure generation. Models, validation status,
and primary citations are in [`ice_structures/README.md`](ice_structures/README.md).

## Low-energy Landau--Zener charge exchange

Prepare an isolated-H2O planning manifest with:

```bash
LOW_ENERGY_ROOT=/gpfs01/work/yoffegid/dnaphysics-ice-ions/python_scripts/physics_ice/process_evidence/low_energy_charge_exchange/validation/runs/c_q1_hirshfeld_density_smoke
python_scripts/physics_ice/nep_mbpol/.venv/bin/python \
  python_scripts/physics_ice/nep_mbpol/prepare_low_energy_charge_exchange.py \
  --projectile C --incident-charges 1 --runtime-smoke \
  --output-root "${LOW_ENERGY_ROOT}"
```

This command records the two required branch identities per state but creates
no CP2K input. There is currently no execution or collection PBS launcher:
the Python runner and collector fail with `branch_handoff_pending` before
starting CP2K. Execution requires an accepted-state loader that verifies the
exact geometry, charge, spin, projectile population, multiplier, WFN, density,
trace, settings, and executable provenance for both diabats. This remains
infrastructure for future mixed-CDFT/Landau--Zener work, not an accepted
low-energy cross-section model or Geant4 process. See
[`nep_mbpol/low_energy_charge_exchange/README.md`](nep_mbpol/low_energy_charge_exchange/README.md).

## Geant4 runtime example

For processes already connected to the executable, select the ice phase and
enable the available ion processes explicitly:

```bash
DNA_PHYSICS=ice_am \
DNA_ION_BARKAS_DCS=1 \
DNA_ION_CHARGE_EXCHANGE=1 \
./proton-pipeline/build/dnaphysics_proton \
  proton-pipeline/e1_proton.mac 8 alpha 100 100 300 1
```

The runtime applies the selected phase's molecular density when converting a
microscopic per-H2O cross section to a macroscopic rate. A generated table is
not available to Geant4 merely because its generator exists: consult the
component registry and acceptance state in
[`nep_mbpol/ion_ice/README.md`](nep_mbpol/ion_ice/README.md).
