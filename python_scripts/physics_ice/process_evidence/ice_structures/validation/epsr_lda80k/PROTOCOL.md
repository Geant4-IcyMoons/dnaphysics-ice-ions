# Validation protocol: published EPSR LDA model at 80 K

## Decision being made

This gate decides whether the exact 80 K LDA coordinate model retained in the
STFC EPSR archive is faithfully converted and suitable as a static periodic
geometry for ion--ice collision sampling. It does not claim a new EPSR
refinement, independent thermodynamic validation, or validation of a collision
potential.

## Hard gates

Acceptance requires all of the following:

1. The `AmorIce.zip` archive and every consumed LDA file match the pinned
   SHA-256 values in `source_manifest.json`.
2. `LDA80K.ato` parses as exactly 3,000 O-H-H molecules, 9,000 atoms, a
   45.796719 A cubic box, and 80 K.
3. The box-derived atomic number density matches the higher-precision `rho`
   recorded in the archived EPSR input to the precision allowed by the printed
   box length. Mass density is derived independently from the molecular count,
   volume, molar mass, and exact Avogadro constant.
4. Every molecule retains the archived harmonic-restraint topology and its
   0.976 A O-H and 1.55 A H-H targets. The instantaneous distances are reported
   as distributions; they are not incorrectly required to be rigid.
5. Extended-XYZ conversion preserves every coordinate modulo the periodic cell.
6. The EPSR input and output consistently retain three experimental datasets
   and 1,721 accumulated configurations, with three finite reported R factors.

The coordinate-derived partial RDFs, CHILL+ populations, and independently
aligned experimental/fit residuals are diagnostics. No numerical threshold is
added to them: the accepted physical provenance is the exact published,
diffraction-refined artifact and its archived fit record. This avoids turning
an arbitrary implementation tolerance into a claim of physical accuracy.

## Outputs and limitations

The validator writes an atomic JSON decision, partial-RDF CSV, three-panel PNG,
and deterministic gzip-compressed extended XYZ. Only a successful JSON gate is
followed by collision attestation and registry generation.

The result is one statistically refined coordinate realization. The 1,721
configurations reported by EPSR support the archived averaged observables but
are not 1,721 retained coordinate files. Phase/orientation uncertainty in
collision observables must therefore be quantified through projectile
direction sampling and, when genuine independent EPSR realizations become
available, cross-realization comparison.

Primary sources and the exact commands are recorded in
`../../../../ice_structures/epsr_lda80k/README.md`.
