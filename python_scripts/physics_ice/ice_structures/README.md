# Validated ice structures

This directory is the sole structure library for ion--ice calculations.  A
production workflow must read a phase's `collision_structures.json`; it must
not select an XYZ file from a preparation or run directory.  The two accepted
models describe different phases and temperatures and are not interchangeable.

| Phase ID | Physical model | Accepted configurations | Density (g cm^-3) | Production registry |
|---|---|---:|---:|---|
| `amorphous_lda_80k` | published neutron-constrained EPSR LDA at 80 K | 1 | 0.9343471679 | `epsr_lda80k/artifacts/collision_structures.json` |
| `hexagonal_ih_100k` | NEP-MB-pol-equilibrated proton-disordered ice Ih in the experimental 100 K cell | 3 | 0.9334742974 imposed; 0.9334748313 after XYZ rounding | `hexagonal_ih_100K_experimental/collision_structures.json` |

The registries contain relative paths, hashes, phase labels, periodic cells,
and acceptance flags.  The ion--ice phase definitions in
`../nep_mbpol/ion_ice/phases/` point to these registries, and the hard-collision
drivers use this directory by default.

## Amorphous LDA at 80 K

`epsr_lda80k/` imports the exact final coordinate realization from the STFC
ISIS `AmorIce.zip` archive, DOI
[10.5286/edata/729](https://doi.org/10.5286/edata/729).  It contains 3,000 H2O
molecules in a 45.796719 A cubic periodic cell.  The reported density is
calculated from that composition and volume; this repository neither rescales
nor relaxes the published structure.  Source hashes, conversion rules,
diffraction inputs, RDF diagnostics, CHILL+ results, and the acceptance
decision are recorded in `epsr_lda80k/source_manifest.json` and
`epsr_lda80k/artifacts/epsr_lda80k_validation.json`.

The scientific basis is Finney et al., *Phys. Rev. Lett.* **88**, 225503
(2002), [10.1103/PhysRevLett.88.225503](https://doi.org/10.1103/PhysRevLett.88.225503);
Bowron et al., *J. Chem. Phys.* **125**, 194502 (2006),
[10.1063/1.2378921](https://doi.org/10.1063/1.2378921); and Soper,
*J. Chem. Phys.* **150**, 234503 (2019),
[10.1063/1.5096460](https://doi.org/10.1063/1.5096460).  Coordinate parsing
follows the authoritative EPSR v26 manual linked in `epsr_lda80k/README.md`.
This is one published realization, not three statistically independent
replicas.

## Hexagonal ice Ih at 100 K

`hexagonal_ih_100K_experimental/` contains three independently
proton-disordered, fully periodic 8,192-water configurations.  GenIce2
generated seeds 1000, 2000, and 3000 under the Bernal--Fowler ice rules and
strict depolarization.  Each was first relaxed at 80 K, mapped by fractional
coordinates to the 100 K experimental H2O cell of Rottger et al., minimized,
equilibrated for 1 ns in NVT, and sampled for 1 ns in NVT with NEP-MB-pol and
the matching velocity seed.  Fixing the cell is the physical experimental-cell
definition; the resulting density is an input and is not counted as validation.

The three replicas passed the retained thermodynamic, Bernal--Fowler, CHILL+,
Bragg-peak, and lineage gates.  Exact settings and results are in the model
README, `manifest.json`, and `validation/hexagonal_ih_100K_validation.json`.
The initial GenIce2 cells are isolated under
`preparation/hexagonal_ih_genice2/` and are explicitly not collision-ready.

Definitions and methods follow Rottger et al., *Acta Cryst. B* **68**, 91
(2012), [10.1107/S0108768111046908](https://doi.org/10.1107/S0108768111046908);
Matsumoto et al., *J. Comput. Chem.* **39**, 61 (2018),
[10.1002/jcc.25077](https://doi.org/10.1002/jcc.25077); Nguyen and Molinero,
*J. Phys. Chem. B* **119**, 9369 (2015),
[10.1021/jp510289t](https://doi.org/10.1021/jp510289t); Bernal and Fowler,
*J. Chem. Phys.* **1**, 515 (1933),
[10.1063/1.1749327](https://doi.org/10.1063/1.1749327); and Kuhs and Lehmann,
*J. Phys. Chem.* **87**, 4312 (1983),
[10.1021/j100244a063](https://doi.org/10.1021/j100244a063).  The water force
model is Xu et al., *npj Comput. Mater.* **11**, 279 (2025),
[10.1038/s41524-025-01777-1](https://doi.org/10.1038/s41524-025-01777-1),
with the pinned model artifact DOI 10.5281/zenodo.15033656.

## Rejected candidates and retained evidence

`rejected_candidates/amorphous_lda_80K_candidate/` retains only the compact
decision record for the superseded NEP-MB-pol melt--quench campaign.  Its
1.013--1.022 g cm^-3 cells and the subsequent fixed-density stress scan did not
establish experimental LDA.  They must not be used as production targets.  Raw
trajectories and scheduler logs were deleted after the numerical reports and
plots were retained; they are reproducible from the versioned inputs and PBS
workflows.  This negative result is kept to prevent the failed route from being
mistaken for an accepted model.

## Reproduction and use

For the amorphous import and audit, follow `epsr_lda80k/README.md`.  For a new
hexagonal preparation, first regenerate the three 80 K relaxed sources and
then run the documented `100K_EXPERIMENTAL` PBS protocol; the compact accepted
snapshots are sufficient for ordinary collision production.  Never infer
acceptance from a filename: require `collision_ready: true` and verify the
registered SHA-256 digest before sampling.

These atomistic models provide target geometry and phase-dependent collision
sequences.  They do not themselves define charge exchange, electronic
stopping, or short-range projectile--atom potentials.

`ice_structure_comparison.png` is the common six-panel overview: panels
(a)--(c) show the accepted seed-1000 ice-Ih replica and panels (d)--(f) show
the accepted EPSR LDA configuration. Every panel samples a centered 14 A
physical cube. Display limits are fitted tightly to each view, and the 5 A
reference in panel (f) provides the absolute spatial scale. It is
regenerated with:

```bash
PHYSICS_ICE_PLOT_FONT_DIR=/path/to/staged/nimbus-mono \
python_scripts/physics_ice/nep_mbpol/.venv/bin/python \
  python_scripts/physics_ice/nep_mbpol/plot_hexagonal_ice_structure.py \
  --structure python_scripts/physics_ice/ice_structures/hexagonal_ih_100K_experimental/seed1000_final.xyz.gz \
  --amorphous-structure python_scripts/physics_ice/ice_structures/epsr_lda80k/artifacts/lda80k_epsr.xyz.gz \
  --output-stem python_scripts/physics_ice/ice_structures/ice_structure_comparison
```

Only panel (f) carries the 5 A scale bar, and horizontal coordinate labels
appear only on the bottom row. See `CLEANUP.md` for the removed run products.
