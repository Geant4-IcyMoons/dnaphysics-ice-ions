# Cluster transition to ion_modular

Date: 2026-09-07. Upstream snapshot:
`dc0cae9af97962863437cc06c6da18c9b6f6c5b3`.

The active checkout at `/gpfs01/work/yoffegid/dnaphysics-ice-ions` is now
`ion_modular`, tracking `origin/ion_modular`. No remote commit was changed.
CTMC, hard/soft collisions, and the Geant4 application have not been migrated
into `physics/`; their legacy paths and generated products remain in place.
Existing PBS jobs were neither cancelled nor resubmitted.

The tracked ion working tree is preserved in stash commit
`03f81151e29807d66462d4a00536b7494626c18d`. Original root configuration and
legacy tests, including untracked tests, are retained in
`.ion-migration-20260907.MtMx55/` at the repository root. That directory has
recovery notes and is locally Git-ignored. The old source paths used by jobs
were not unlinked during the switch; source checksums matched before and
afterward. Do not apply the legacy stash indiscriminately onto the new layout.

## Environment and entry point

Use `.venv/bin/python` from the repository root. This local Python 3.13
environment inherits the installed scientific packages from `yoffe_venv` and
adds `openpyxl` 3.1.5 plus `et_xmlfile` 2.0.0 locally. The shared environment
was not modified. `pip check` passes.

```bash
.venv/bin/python -m physics.inelastic_dielectric.generate_cross_sections --help
```

For example, a small neutral-hydrogen PWBA export:

```bash
ICE_TYPE=amorphous ICE_MAX_WORKERS=2 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m physics.inelastic_dielectric.generate_cross_sections \
  --projectile H --charge-state 0 --include-barkas-dcs=false \
  --energy-min-MeV 10 --energy-max-MeV 20 --energy-unit total \
  --energy-points 2 --dE 12 --dq 24 \
  --output-dir physics/inelastic_dielectric/validation/runs/example/tables \
  --cache-dir physics/inelastic_dielectric/validation/runs/example/caches
```

This grid is for checking execution only, not numerical convergence. Default
production paths are `physics/inelastic_dielectric/output/{tables,caches}`;
legacy cross sections are not overwritten. For the PBS launcher, explicitly
set `PYTHON_BIN` to the absolute repository `.venv/bin/python` path.

## CLI repair and operational checks

The upstream legacy scalar parser also consumed `--charge-state`, rejecting
Q=0 and unnecessarily setting `explicit_charge` for nonzero fixed states.
The alias has been removed from that parser; `--explicit-charge` and
`--q-charge` remain positive-only. No physical kernel was changed.

`tests/test_modular_cli_smoke.py` exercises real two-energy subprocess exports
for H0/H+ in both phases and both Born modes, plus bare-proton RPWBA with
polarization in both phases. It checks four DAT products per case, incident
nodes, finite nonnegative DCS, integrated TCS, one complete NPZ cache, and
byte-identical DAT exports on a second invocation using that cache.
The all-state CLI parsing tests separately cover both option spellings and
all 38 states. Numerical/physical validation remains separate from execution.

## Restart-safe production execution

The local modular generator now restores the legacy incident-energy/channel
task decomposition, without changing its physics kernels. Completed integrated
energies and Born DCS tasks are written immediately by atomic replacement;
in-flight work is bounded to one energy per integration worker or one
energy/channel per DCS worker. Progress bars count completed energies, tasks,
and exported rows. Production grids expose enough tasks for 256 workers.

Checkpoint manifests identify source and input-data hashes, CLI arguments,
material settings, grid and dispersion coefficients. Incompatible resumes
fail explicitly. Worker counts can change. Final DAT/NPZ products are atomic
per file; an interrupted multi-file export is regenerated from saved tasks.
Bare-proton Barkas assembly is vectorized and can be repeated from the saved
Born DCS. Screened polarization now also saves each completed loss within
each incident-energy row, including its quadrature error and completion mask.
An interrupted worker repeats at most its current loss integral. Source,
model, mass, velocity and loss-grid checks reject incompatible resumes.

`tests/test_modular_checkpoints.py` exercises cancellation after saved DCS tasks,
atomic-write failure, and incompatible-resume rejection. The real CLI smoke
tests remove a completed cache and one task from each stage, then require
byte-identical tables after resuming with a different worker count. These are
execution/regression tests, not production-grid convergence evidence.

The PBS launcher supports explicit output/cache destinations, source-hash
verification, a cache-directory lock and live logs. Production submissions use
a fixed source snapshot; rerun that snapshot with the same paths and arguments
to resume. No automatic resubmission daemon is installed.

The eight-case proton launcher is:

```bash
.venv/bin/python -m physics.inelastic_dielectric.jobs.submit_campaign --projectile proton
```

Use `--projectile alpha` for He2+ instead. Each campaign contains two phases,
PWBA/RPWBA and Barkas off/on. Each job requests
256 CPUs, 512 GB and 72 hours. The campaign submitter now defaults to `p72`
(publicx), as requested on 2026-09-08; `--queue idle` explicitly selects idlex.
The earlier proton campaign remains on idlex. The production grid is 1000
logarithmic total kinetic energies from 0.1 to 100 MeV, dE=1000 and dq=1000,
with hydrogenic K continuum, no Bloch correction and no energy-patch merging.
For alpha particles the same 0.1–100 MeV range is total kinetic energy, not
MeV/u. Bare He2+ dispatches to the point-projectile Salvat polarization term,
not the experimental screened-oscillator correction. He0 and He+ are not
included by default. Select them explicitly with `--charge-state 0` or
`--charge-state 1` and `--polarization off`. Neutral hydrogen uses
`--projectile proton --charge-state 0 --polarization off`. The submission
entry point now accepts `--polarization on` for the requested experimental
screened-polarization attempts after adding loss-level checkpoints. Physical
and quadrature rejection checks remain enabled; accepted submission does
not imply that a complete table can be exported for the requested range.
RPWBA includes the transverse and density-effect terms. It creates a separate
campaign under `output/runs/`, `output/tables/` and `output/caches/`; the run
directory records all qsub commands, returned job IDs, source snapshot/hash,
base commit, working-tree patch and Python package versions. Existing jobs
and legacy products are not changed. Invoking it again creates a new campaign,
not a resume; use the recorded qsub command to resume a particular case.

During restoration, two existing tests still supplied the former two-field
channel task tuple. They were updated to the four-field bounded-task format;
both serial/ten-worker comparison reruns passed. Two CLI resume checks also
rejected source changes made while the tests were running, as designed; both
passed on rerun against the finalized source (8 passed including the six
checkpoint unit cases). In total, all 36 checkpoint/export/CLI cases and all
160 focused physics cases passed across the initial runs and targeted reruns.
The physics run emitted ten Python fork-from-multithreaded-process deprecation
warnings; no deadlock occurred. The launcher smoke test
at 0.1 and 100 MeV (dE=12, dq=24, two workers) exported all four RPWBA+Barkas
amorphous products with exit code 0. Its coarse grid tests execution only.

The first submission preflight rejected the snapshot before any qsub call:
the source-hash exclusion used absolute rather than physics-relative path
components, accidentally excluding snapshots stored under `output/runs`.
That path handling was corrected and covered by a relocation regression.
All seven checkpoint tests plus the real RPWBA+Barkas amorphous CLI resume
test then passed (8 passed). No physics formula changed in this repair.

Screened polarization is experimental and has documented physical acceptance
failures for several low charge states. Keep its runtime rejection checks;
successful exports or baseline regressions do not validate that correction.
See [SCREENED_BARKAS.md](../polarization/SCREENED_BARKAS.md).
