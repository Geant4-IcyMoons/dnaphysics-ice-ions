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
   counts, then evaluate three snapshots, three directions, and six base
   energies from 1 keV to 100 MeV. Report the Monte Carlo confidence intervals
   separately from the 0.5% kernel-interpolation budget.
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

After the pilot-derived sampling plan is recorded, submit the 54-case base
matrix with the chosen adaptive bounds:

```bash
qsub -v RUN_MODE=carbon_base_grid,MINIMUM_TRAJECTORIES=1000,\
MAXIMUM_TRAJECTORIES=<pilot-derived-limit>,\
STATISTICAL_RELATIVE_TOLERANCE=<declared-MC-tolerance> -J 0-53 \
  pbs/run_nlh_hard_collision_trajectories.pbs
```

Every case writes checksum-bound batches immediately and resumes only from a
configuration-compatible checkpoint.
