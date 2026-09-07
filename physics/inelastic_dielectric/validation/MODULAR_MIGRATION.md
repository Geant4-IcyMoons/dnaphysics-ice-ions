# Modular migration validation

The `ion_modular` layout was checked against the pre-migration source snapshot
based on `ion` commit `32848806d12e2aaac370631ef56af503783503a0`, including
the existing dielectric working-tree changes. The reference fixture records
the original source hashes in `tests/data/modular_kernel_reference.json`.

- All 456 sampled kernel values were bitwise identical before and after the
  module extraction: both phases, PWBA/RPWBA, and all 38 charge states of
  H, He, C, O, and S. This checks migration, not physical accuracy.
- The ten retained test files passed, totaling 663 tests. Tests were run
  per file; moved-constant imports and a figure-spacing failure were repaired
  and the affected tests rerun successfully.
- Small end-to-end generator runs passed for amorphous proton PWBA and
  hexagonal proton RPWBA with polarization. These two-energy, coarse-grid
  runs check table/cache export, not production numerical convergence.
- Proton and alpha comparison PDFs were regenerated from their existing
  benchmark reports. The nonlinear benchmark PDF uses the same figure style.
  No external SBETHE, ICRU, or Matias comparison was recomputed in this migration.

No physical formulas were intentionally changed by the extraction. Generated
products are ignored; Geant4 data export now requires an explicit destination.
The executable and other collision implementations remain on `ion`, not in
this dielectric-only branch. Screened polarization and its relativistic
extension retain the validity limitations documented in their component.
