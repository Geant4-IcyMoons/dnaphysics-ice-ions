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
