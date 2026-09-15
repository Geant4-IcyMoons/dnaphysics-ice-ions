# Cubic ice Ic at 100 K

This protocol prepares three 8,000-water periodic cells with GenIce2 2.2.13.3
and the retained NEP-MB-pol/GPUMD-v3.9.3 runtime. Seeds 1000, 2000, and
3000 generate distinct proton arrangements. The initial structures pass
the Bernal–Fowler rules and have 100% cubic classification under CHILL+.
Initial topology checks do not establish thermal equilibration.

## Density and physical scope

The user accepted equal molecular volumes for H2O Ic and H2O Ih at 100 K
on 2026-09-15. The four-molecule Ih volume, 128.188109 Å³, comes from the
corrected polynomial of [Röttger et al. (2012)](https://doi.org/10.1107/S0108768111046908).
Thus the conventional eight-water Ic cell has `a = 6.352713148 Å`, and the
10 × 10 × 10 supercell has side 63.527131482 Å. With the project molar mass
18.01528 g mol⁻¹, the imposed density is 0.9334742974 g cm⁻³. These digits
specify the construction; they are not experimental precision for Ic.
See the [density evidence](../README.md) for the H2O/D2O distinction.

The structures describe classical nuclei at a constrained volume. They do
not include nuclear quantum fluctuations, and their validation does not
establish the accuracy of NEP-MB-pol forces for Ic. Stress is reported as a
diagnostic, without rescaling the cell to force zero stress. Cubic ion
transport is outside this preparation: snapshots remain `collision_ready:
false` until a consuming cubic-phase workflow is separately configured.

The final report also tests geometry at ±0.1% relative density, motivated by
the sub-0.1% D2O polytype differences in the 2026 study cited in the parent
README. This is an illustrative sensitivity interval, not a confidence
interval for H2O Ic. The diagnostic reports changes in neighbor distances,
local order, ice rules, and the number of molecules in the plotted crop.
It does not evaluate force or thermodynamic sensitivity.

## Preparation and restart

`config.json` is the common protocol definition. GenIce constructs the
diamond oxygen network with strict depolarization; the conversion changes
molecular origins to set the cell while retaining internal water geometry.
A fixed-cell FIRE minimization precedes 1 ns of NVT equilibration and
1 ns of NVT sampling, at a 0.2 fs time step. The thermostat is Nosé–Hoover
chain with the same coupling parameter as the retained Ih preparation.

The run consists of twenty 100 ps blocks. Each new block reads the preceding
full-precision `dump.xyz` positions and velocities. GPUMD reinitializes its
thermostat chain at each boundary; this segmentation is part of the protocol,
including uninterrupted runs. Checkpoints preserve eight-decimal XYZ output,
not an exact binary integrator state. No bitwise equivalence with a single
unsegmented trajectory is claimed.

Each completed block is committed atomically with output checksums. A rerun
verifies all earlier blocks and rejects changed settings, source files, or
corrupted products. It replaces only the current uncommitted block under a
per-seed lock. A cancellation therefore loses at most 100 ps. The test in
`test_run.py` exercises interruption, reuse, and corruption rejection with a
synthetic executable; it does not test physical dynamics.

The GPU request is one GPU, four compilation CPUs, and 8 GB per independent
replica. The CUDA engine cannot use a 256-CPU allocation. Compilation occurs
on the allocated node. The one-process final plot uses one CPU and 2 GB;
the `idle` routing queue is necessary because this cluster rejects direct
submissions to `idlex`.

From the repository root:

```bash
qsub -J 0-2 -o "$PWD/models/ice/cubic_ic_100K/logs" models/ice/cubic_ic_100K/run.pbs
# Use the actual returned array job ID in the dependency:
qsub -W 'depend=afterok:ARRAY_ID[].pbs02' \
  -o "$PWD/models/ice/cubic_ic_100K/logs" models/ice/cubic_ic_100K/collect.pbs
```

The local Python environment needs `/apps01/apps/anaconda3-2022.10/lib`
in `LD_LIBRARY_PATH` for its bzip2 extension; the PBS scripts set it.
The four Nimbus Mono PS font faces are staged under `runs/fonts` for the
compute-node renderer. The retained runtime's `PROVENANCE.json` identifies
the published [NEP-MB-pol artifact](https://doi.org/10.5281/zenodo.15033656).
Preparation receipts pin the actual GPUMD source and shared code hashes.

## Validation and figure

Ten sampling frames per replica must pass fourfold coordination, both
Bernal–Fowler rules, at least 99% bulk cubic CHILL+ classification with zero
hexagonal classifications, and seven allowed Ic reflections being local
reciprocal-space maxima. Thermodynamic tests use the same Newey–West
estimators as the retained Ih audit, with Holm correction within each seed
and Bonferroni allocation across three seeds. Temperature must include
100 K in its corresponding confidence interval. No resolved drift is not
proof of complete equilibrium.

Each successful validation writes a checksummed compressed snapshot and a
numerical report. `validate.py --collect` requires all three reports and
distinct initial proton configurations before writing `manifest.json` and
the nine-panel `../ice_structure_comparison.png`, in Ih/Ic/LDA row order.
The density assumption is retained in the manifest. Until these steps
complete, the previous two-row figure remains the available comparison.

Local-order definitions: [Nguyen & Molinero (2015)](https://doi.org/10.1021/jp510289t).
Ice rules: [Bernal & Fowler (1933)](https://doi.org/10.1063/1.1749327).
Generator: [Matsumoto et al. (2018)](https://doi.org/10.1002/jcc.25077).

## Authorized publication

The user authorized adding the validated material and regenerated figure to
`ion_modular` and pushing on 2026-09-15. `publish.pbs` runs only after the
collection job succeeds. `publish.py` verifies the reviewed source hashes,
three scientific reports, material checksums, and figure provenance before
creating a commit from an isolated Git index. Its explicit file list excludes
runtime directories, logs, and unrelated changes. Remote changes to those
same files cause rejection; the push never forces history. The shared local
HEAD and index remain unchanged. Publication status is recorded in
`runs/publish_result.json`.

