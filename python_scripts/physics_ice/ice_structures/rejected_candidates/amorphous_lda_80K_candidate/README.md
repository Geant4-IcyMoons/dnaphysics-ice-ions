# Rejected NEP-MB-pol 80 K amorphous-ice candidates

This directory is retained as negative validation evidence only. The accepted
amorphous production model is `../../epsr_lda80k/`; no file here is registered
as collision-ready.

The three NPT melt--quench trajectories completed, but these cells are **not
collision-ready LDA structures**.  Their final 80 K densities are
1.013--1.022 g/cm3, above both experimental LDA (approximately 0.94 g/cm3) and
the 0.98 g/cm3 result reported for the q-TIP4P/F cooling protocol adapted here.
CHILL+ finds only about 0.5% local Ih plus Ic environments, so the cells are
amorphous rather than failed melts.  The oxygen-only diffraction diagnostic
places the first peak at 1.88--1.90 inverse angstrom, above the 1.7--1.8
inverse-angstrom LDA range quoted by Eltareb et al.  The defensible assignment
is therefore a reproducible dense/intermediate amorphous candidate, not
validated experimental-density LDA.

Reproduce the analysis from the repository root with:

```bash
python_scripts/physics_ice/nep_mbpol/.venv/bin/python \
  python_scripts/physics_ice/nep_mbpol/validate_amorphous_ice.py
```

The `validation/` directory contains two PNG figures, a JSON decision record,
and the plotted structural data as CSV.  No PDF is generated.  The JSON report
records the limitations of the oxygen-only structure factor and the single
saved final-hold frame per replica.

`preview_seed1000.png` is the matching three-view rendering of the final seed
1000 configuration.  It is a visualization of a rejected candidate, not an
attested collision structure.

Before another 20 ns preparation, run a short 80 K density--stress scan for
all three structures over 0.93--1.02 g/cm3.  This tests whether NEP-MB-pol can
mechanically sustain an amorphous network at the experimental density.  Do not
merely rescale a final snapshot and call it LDA: a constrained-density cell
must be equilibrated, its residual stress reported, and its RDF and local order
revalidated.

The implemented preliminary gate evaluates 0.94, 0.98, and 1.02 g/cm3 for all
three seeds. Each case performs 0.5 ns of NVT equilibration and 0.5 ns of NVT
sampling at 80 K. The nine molecular-dynamics tasks require CUDA and therefore
run through the public GPU route; `idlex` has no GPUs. The dependent collector
is a CPU job routed through `idle` to `idlex`:

```bash
scan_job=$(qsub -J 0-8 pbs/run_nep_mbpol_amorphous_density_scan.pbs)
qsub -W "depend=afterok:${scan_job}" \
  pbs/collect_nep_mbpol_amorphous_density_scan.pbs
```

Each case preserves a completed equilibration block independently of the
sampling block. An interrupted block loses at most 0.5 ns and is retained as
an incomplete attempt rather than overwritten. The collector tests whether
the autocorrelation-aware 95% confidence interval of mean pressure contains
0.1 MPa, and reports RDF, structure-factor, and CHILL+ diagnostics. It does not
declare O--O agreement until an uncertainty-bearing experimental reference
table has been ingested.

Protocol and comparison sources:

- Eltareb, Lopez, and Giovambattista, *Communications Chemistry* **7**, 36
  (2024), <https://doi.org/10.1038/s42004-024-01117-2>.
- Finney et al., *Physical Review Letters* **88**, 225503 (2002),
  <https://doi.org/10.1103/PhysRevLett.88.225503>.
- Loerting et al., *Physical Chemistry Chemical Physics* **13**, 8783 (2011),
  <https://doi.org/10.1039/C0CP02600J>.
- Nguyen and Molinero, *Journal of Physical Chemistry B* **119**, 9369 (2015),
  <https://doi.org/10.1021/jp510289t>.
