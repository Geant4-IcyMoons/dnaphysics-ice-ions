# GPAW q=1 fixed-multiplier decision gate

This benchmark closes the GPAW route for constructing the carbon q=1
soft-collision entrance state at 12 A in the oxygen-back orientation.  It does
not imply that the physical diabatic state is absent.

## Evidence

The five outer-optimizer diagnostics are preserved under
`process_evidence/soft_nuclear_collisions/validation/runs/` in
`gpaw_q1_multiplier_calibration_gmf_20260810_v1`.  Four calculations converged
their first fixed-multiplier inner solve while retaining the selected carbon
`p_z` orbital.  Their exact first charge-multiplier changes were

```text
-5.9468531300e-5 Ha  charge-only, start 0.05 Ha
-5.9467886037e-5 Ha  charge-only, start 0.15 Ha
-5.9455828875e-5 Ha  charge+spin, start (0.05, 0.05) Ha
-5.9446624141e-5 Ha  charge+spin, start (0.05, 0.15) Ha
```

Each tiny update was followed immediately by an approximately four-electron
change in the carbon population and loss of the selected orbital branch.  The
remaining `(0.15, 0.15)` calculation reached its bounded 333-iteration inner
limit without a first multiplier update.  No state from these calculations is
accepted or restartable.  `preupdate_summary.json` and the per-task
`preupdate_trace.jsonl` files preserve only the pre-update evidence; complete
raw traces are retained solely as a rejection audit.

PBS array `113512[0-9]` then evaluated fixed charge multipliers
`0.05, 0.075, 0.10, 0.125, 0.15 Ha` without an outer optimizer, using two
independent Davidson preparations at every multiplier.  The fragment-defined
target, calculated from independently prepared C+ and H2O densities with the
same cell, grid, PAW augmentation and GPAW CDFT weight definition, was

```text
N_C,target = 4.999999720606075 electrons.
```

The stable-complex carbon populations decreased only from
`4.999940531477153` to `4.999940531431119` electron over the entire `0.10 Ha`
interval.  Thus the response magnitude was only `4.60e-11` electron, while the
target residual remained approximately `-5.919e-5` electron.  The independent
repetitions agreed to at most `1.78e-15` electron in population,
`2.90e-13 eV` in corrected KS energy, and `9.56e-10` electron in integrated
pseudo-density difference.  The selected `p_z` identity was retained.

The fragment target differs from the integer five-electron target by only
`2.79e-7` electron.  It therefore cannot explain the stable `5.919e-5`
residual.  The residual is also many orders of magnitude larger than measured
repeatability, so relaxing the numerical tolerance would hide a systematic
failure rather than accommodate numerical noise.

The low Hessian spectrum contains near-zero modes and the counted inertia
varies by one between independent Davidson preparations.  The state, density,
energy and population are nevertheless reproducible; exact Hessian inertia is
not regarded as resolved by this scan.

## Decision

The fixed-multiplier state is reproducible but has essentially zero population
response to the multiplier in the tested interval.  GPAW's outer update is
small, not an aggressive overshoot, yet it enters a different electronic
branch.  This representation therefore provides no continuous usable root for
the requested q=1 population at this geometry.  It is not eligible for a
soft-potential table, geometry continuation, or force validation.

No tolerance is redefined, no post-collapse wavefunction is retained, and no
geometry array is launched.  The next independent pilot is state-specific
OpenMolcas RASSCF/CASSCF followed by RASSI state interaction.  OpenMolcas is
not described as a block-localized-wavefunction implementation: the pilot
tests whether separately optimized entrance and capture states and their
coupling can be represented in a common multiconfigurational active space.

Method documentation:

- GPAW CDFT: <https://gpaw.readthedocs.io/documentation/cdft/cdft.html>
- GPAW direct orbital optimization:
  <https://gpaw.readthedocs.io/documentation/do/do.html>
- OpenMolcas RASSCF:
  <https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/rasscf.html>
- OpenMolcas RASSI:
  <https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/rassi.html>
