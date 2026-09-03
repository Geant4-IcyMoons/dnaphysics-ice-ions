# Carbon NLH hard-collision validation protocol

## Scope and immutable definitions

This protocol validates the retained-domain C--H/O hard-elastic model before
it is released for Geant4 production. It does not validate electronic stopping,
charge exchange, chemistry, or a complete recoil cascade.

For target atom \(t\in\{\mathrm{H},\mathrm{O}\}\), the microscopic cross section is
threshold-defined:

\[
\sigma_{\mathrm{C}t}^{\mathrm{hard}}(T)=
\begin{cases}
\pi r_{\mathrm{th,C}t}^{2}\left(1-
V_{\min}/E_{\mathrm{cm,C}t}\right), & E_{\mathrm{cm,C}t}>V_{\min},\\
0, & E_{\mathrm{cm,C}t}\leq V_{\min}.
\end{cases}
\]

This is \(\pi b_{\max,\mathrm{C}t}^{2}\). Geant4 evaluates it analytically at
the current total C-12 kinetic energy; it is not read from or interpolated
between tabulated cross sections. In pure water ice,

\[
\Sigma_{\mathrm C}^{\mathrm{hard}}=
n_{\mathrm{H_2O}}\left(2\sigma_{\mathrm{CH}}^{\mathrm{hard}}+
\sigma_{\mathrm{CO}}^{\mathrm{hard}}\right).
\]

The 0.5% kernel criterion is a numerical interpolation tolerance on the
scattering-angle representation. It is not physical accuracy. The potential
fit uncertainties are reported independently; at the retained 30 eV boundary,
the source reports RMS errors of 3.56% for C--H and 9.51% for C--O.

## Decision gates

1. **Provenance and deterministic numerics.** Verify source and compact-table
   checksums, exact threshold cross sections at off-grid energies, interpolation
   probes against the Python reference, two-body energy conservation, and the
   pure-water density law. Any failure blocks release.
2. **Variance and throughput pilot.** Run 100,000 fixed trajectories for each
   combination of three attested 100 K ice-Ih snapshots, three directions, and
   the 1 keV and 100 MeV endpoints: 18 jobs. These fixed samples estimate
   observable variances and runtime only; they make no convergence claim.
3. **Hexagonal base matrix.** Use the pilot variances to set explicit sample
   counts and freeze each case's absolute confidence-width tolerances from the
   independent raw calibration estimate using JCGM 101:2008 section 7.9.2 and
   the project's predeclared requirement of two meaningful significant digits.
   JCGM defines the decimal tolerance; it does not prescribe the number of
   digits. Then evaluate three snapshots, three directions, and six base
   energies from 1 keV to 100 MeV. Report the Monte Carlo confidence intervals
   separately from the 0.5% kernel-interpolation budget. The two-digit rule is
   frozen for acceptance before further sampling; looser sensitivity results
   cannot replace it.
4. **Energy refinement.** Add midpoint energies wherever direct atomistic
   observables are not reproduced within their declared numerical/statistical
   envelope. Refine until the rate, nuclear stopping, transport moment, recoil
   energy, and angular distributions satisfy their individually documented
   criteria. Distribution tails require bin-wise tests; convergence of first
   moments alone is insufficient.
5. **Phase gate.** Repeat the accepted sampling plan for independently
   validated amorphous structures. Compare amorphous with orientationally
   averaged and direction-resolved hexagonal results at matched energy and path
   length. A statistically resolved effect must be reported. If its size
   exceeds the predeclared tolerance of the downstream Geant4 observable,
   density scaling alone is rejected and a phase/orientation correction must be
   constructed and validated. No universal correction factor is assumed.
6. **Release.** Change `atomistic_validation_pending` to `accepted` only after
   all preceding gates pass and the comparison products, settings, checksums,
   and uncertainty separation are archived. Until then, Geant4 production is
   refused unless the explicit engineering override is set.

HTran is a complete H/He elastic treatment, not a soft-only kernel. It must not
overlap NLH. Any later H/He integration requires either a validated energy
handoff or an explicit non-overlapping angular/impact-parameter partition.

## Commands

Rebuild and test the compact Geant4 carbon product:

```bash
python_scripts/physics_ice/nep_mbpol/.venv/bin/python \
  python_scripts/physics_ice/nep_mbpol/export_nlh_geant4_table.py
cmake --build proton-pipeline/build -j
ctest --test-dir proton-pipeline/build --output-on-failure
```

Submit the restart-safe carbon variance pilot:

```bash
qsub -v RUN_MODE=carbon_variance_pilot -J 0-17 \
  pbs/run_nlh_hard_collision_trajectories.pbs
```

Production uses the particle controller, which performs the independent
calibration and passes the frozen per-observable widths to the simulator:

```bash
bash pbs/launch_adaptive_nlh_particle_shards.sh C
```

Every case writes checksum-bound batches immediately and resumes only from a
configuration-compatible checkpoint.

The active carbon wave was launched before the hard/soft kernel consolidation
and is signed as trajectory implementation v3. Reassess its preserved
sufficient statistics without resampling:

```bash
python_scripts/physics_ice/nep_mbpol/.venv/bin/python \
  python_scripts/physics_ice/nep_mbpol/audit_nlh_absolute_fixed_width.py \
  <campaign-root>
```

Only cases that fail this audit may receive additional trajectories. Their
v3 trajectory stream must use the pinned node-local compatibility runner,
never the v4 default runner:

```bash
qsub -J 0-<failed-case-count-minus-one> \
  -v WAVE_MANIFEST=<wave>,AUDIT=<absolute-fixed-width-audit>,\
WORKERS=64,REPO_ROOT="$PWD",\
LEGACY_COMMIT=5f4adfe2585c4707c6d53dba10d580e876e1a688 \
  pbs/run_adaptive_nlh_particle_shard_v3_resume.pbs
```

Submit this launcher as an array over the audit's failed-case indices. Each
element advances one case only to its next predeclared scheduled look; rerun
the audit before requesting another look.

The pinned revision is the repository state that launched the v3 production
jobs. It is extracted only to compute-node scratch. The canonical v3 batch
directories remain the sole source of progress; v4 namespaces cannot be
combined with them.

The particle-generic adaptive implementation superseding a manually fixed
carbon sample count is `adaptive_nlh_particle_transport.py`. It uses the same
gates for every supported projectile and is launched as one PBS allocation per
particle by `pbs/launch_adaptive_nlh_particles.sh`. The original 18-case carbon
pilot remains estimator-calibration evidence; it is not a production table.
