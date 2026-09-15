# Atomistic ice models

`registry.json` identifies the accepted inputs:

- `hexagonal_ih_100K_experimental/`: three periodic ice-Ih snapshots at 100 K.
- `epsr_lda80k/`: one EPSR amorphous low-density ice snapshot at 80 K.

Each model retains its structure checksums, metadata, references and validation
products. These are atomistic inputs, distinct from the dielectric response
parameterizations under `physics/inelastic_dielectric/finite_q/`.

The structures were relocated without changing their contents. Legacy local
paths remain symlinks because historical campaign manifests reference them.
No historical manifest has been re-signed. Structure acceptance does not
validate a projectile interaction or the NLH–ZBL handoff.

## Structure comparison

`plot_structure.py` retains the existing molecular renderer and accepts an
optional `--cubic-structure` input. The comparison order is Ih, Ic (when
supplied), and LDA. Each row contains three views of a centered 14 Å cube;
the bottom-right panel contains the 5 Å scale bar. A JSON sidecar records
the input hashes and displayed molecule counts. Rendering does not establish
structure acceptance. The [cubic preparation](cubic_ic_100K/README.md) adds
its model to `registry.json` only after all three replicas pass validation.

```bash
python -m models.ice.plot_structure \
  --structure models/ice/hexagonal_ih_100K_experimental/seed1000_final.xyz.gz \
  --amorphous-structure models/ice/epsr_lda80k/artifacts/lda80k_epsr.xyz.gz \
  --output-stem models/ice/ice_structure_comparison
```

Add `--cubic-structure models/ice/cubic_ic_100K/seed1000_final.xyz.gz` once
the cubic preparation and validation are complete. The retained legacy
renderer path is a symlink to this implementation. The collection job also
updates the original comparison-image path after successful validation.

### Cubic density at 100 K: evidence and accepted approximation

The checked sources do not establish a directly measured density for pure
H2O ice Ic at 100 K. The 2026 preprint by
[del Rosso et al.](https://arxiv.org/html/2602.13053v1) measures pure **D2O**
Ic over 10–205 K. It finds small density differences between the polytypes;
these measurements cannot be treated as H2O densities. Its arXiv source
archive does not contain the supplementary numerical tables mentioned in
the manuscript. The reported raw data are available from the authors upon
request; no request has been sent.

Our H2O Ih cell follows [Röttger et al. (2012)](https://doi.org/10.1107/S0108768111046908).
Evaluating the existing implementation at 100 K gives a four-molecule volume
of 128.188109 Å³ and density 0.9334742974 g cm⁻³. Assuming equal molecular
volumes for Ic and Ih would give an eight-molecule cubic cell with
`a = (2 × 128.188109 Å³)^(1/3) = 6.352713148 Å` and the same density.
These digits reproduce the proposed construction, not experimental accuracy
for Ic. The user accepted the equal-volume approximation on 2026-09-15.
The [cubic preparation](cubic_ic_100K/README.md) records the protocol and
validation requirements. A numerical uncertainty for H2O Ic has not been
established.
