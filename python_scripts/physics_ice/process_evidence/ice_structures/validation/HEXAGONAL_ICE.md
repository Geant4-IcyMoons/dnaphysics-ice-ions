# Validation protocol for the 100 K ice-Ih collision structures

## Scope and decision

This protocol decides whether the three completed, periodic 100 K
NEP-MB-pol replicas are suitable target geometries for subsequent collision
sampling. It does not validate NEP-MB-pol for radiation damage, the NLH
projectile coupling, or a full recoil cascade. The imposed density is checked
for integrity but is not counted as independent evidence of ice-Ih order.

The implementation is `validate_hexagonal_ice.py`. It analyzes all ten saved
configurations and all 500 thermodynamic records in the documented 1 ns
sampling block of each seed (1000, 2000, and 3000). It writes one JSON decision
record, partial-RDF and Bragg CSV tables, and one PNG summary figure. The raw
trajectories remain outside version control.

## Published definitions and fixed inputs

1. The fixed 100 K volume and `c/a` ratio are evaluated from the corrected H2O
   ice-Ih polynomials of Rottger et al., *Acta Cryst. B* **68**, 91 (2012),
   <https://doi.org/10.1107/S0108768111046908>. The validation recomputes this
   cell rather than copying the values from the run manifest.
2. Bulk ice is classified with CHILL+ exactly as defined by Nguyen and
   Molinero, *J. Phys. Chem. B* **119**, 9369--9376 (2015),
   <https://doi.org/10.1021/jp510289t>: four nearest oxygen neighbors, the
   normalized `l=3` spherical-harmonic correlation, staggered bonds at
   `c(i,j) <= -0.8`, eclipsed bonds at `-0.35 <= c(i,j) <= 0.25`, and bulk Ih
   assigned by three staggered plus one eclipsed bond with exactly four
   neighbors inside 3.5 A. The paper reports at least 99% recognition of
   equilibrated bulk Ih at 250 and 270 K; that published benchmark is the
   minimum accepted fraction in every saved 100 K frame.
3. Proton topology must obey the Bernal--Fowler rules exactly: every oxygen is
   the nearest oxygen of two hydrogens, and every edge of the four-connected
   nearest-neighbor oxygen network contains exactly one proton. See Bernal and
   Fowler, *J. Chem. Phys.* **1**, 515--548 (1933),
   <https://doi.org/10.1063/1.1749327>.
4. The expected low-order ice-Ih reciprocal lengths are computed from the
   Rottger `a` and `c` values,

   `q_hkl = 2*pi*sqrt[(4/3)(h^2 + hk + k^2)/a^2 + l^2/c^2]`.

   The (100), (002), (101), (102), (110), (103), and (112) families must each
   be a local intensity maximum relative to the immediately adjacent
   supercell reciprocal bins in every sampling frame. Low-temperature neutron
   structure provenance is Kuhs and Lehmann, *J. Phys. Chem.* **87**,
   4312--4313 (1983), <https://doi.org/10.1021/j100244a063>.

No fitted scale factor, phase-dependent correction, or visual pass/fail
judgment enters these definitions.

## Thermodynamic stationarity

For each replica, fit a linear trend over the 500-point sampling block to
temperature, potential energy, and all six stress components. Estimate slope
uncertainty using a Bartlett-kernel Newey--West heteroscedasticity- and
autocorrelation-consistent covariance. Correct the 24 two-sided slope tests
together by the Holm method at familywise `alpha=0.05`. Acceptance requires no
resolved drift and requires 100 K to lie inside each replica's 95% HAC
mean-temperature interval. Pressure has no acceptance target because the cell
is deliberately fixed at the experimental density; its mean is reported as a
model/cell mismatch diagnostic.

These tests establish that no drift is resolved on the sampled one-nanosecond
window. They do not prove equilibration on every longer time scale.

## Structural and replica checks

For every saved sampling frame:

- verify H2O stoichiometry, full periodicity, and the paper-derived cell;
- apply CHILL+ and require its published bulk-Ih benchmark;
- require the two Bernal--Fowler rules exactly;
- compute O--O, O--H, and H--H partial RDFs through 8 A;
- require each expected Ih reflection to be a local reciprocal-space maximum.

The proton-bond orientation hash must remain unchanged within each replica.
The three hashes must be different, and their lineage must reach three
different GenIce2 seeds generated with `--depol strict`. RDFs, energies, Bragg
ratios, and pairwise proton-orientation differences are reported across the
replicas without inventing an additional numerical similarity cutoff; each
replica must independently pass the phase and topology definitions.

## Run and reproduce

From `python_scripts/physics_ice/nep_mbpol`:

```bash
.venv/bin/python validate_hexagonal_ice.py
```

The expected outputs are under
`../ice_structures/hexagonal_ih_100K_experimental/validation/`:

- `hexagonal_ih_100K_validation.json` -- complete decision and per-frame data;
- `hexagonal_ih_100K_partial_rdf.csv` -- all three partial RDFs by replica;
- `hexagonal_ih_100K_bragg.csv` -- reflection and adjacent-bin intensities;
- `hexagonal_ih_100K_validation.png` -- compact diagnostic summary.

The analysis processes one configuration at a time and stays far below the
7.5 GB login-node memory ceiling. A `tqdm` bar reports the 30 analyzed frames.

Only after the JSON status is `accepted` may the compact snapshots be
attested and registered:

```bash
for seed in 1000 2000 3000; do
  .venv/bin/python attest_collision_structure.py \
    "../ice_structures/hexagonal_ih_100K_experimental/seed${seed}_final.xyz.gz" \
    --phase "hexagonal ice Ih, 100 K, experimental cell" \
    --validation-report \
      ../ice_structures/hexagonal_ih_100K_experimental/validation/hexagonal_ih_100K_validation.json \
    --confirm-accepted
done

.venv/bin/python register_collision_structures.py \
  ../ice_structures/hexagonal_ih_100K_experimental/seed{1000,2000,3000}_final.xyz.gz \
  --output \
    ../ice_structures/hexagonal_ih_100K_experimental/collision_structures.json
```

If any criterion fails, retain `VALIDATION_PENDING`, do not attest the
snapshot, inspect the recorded failing frame/observable, and extend or repair
the physical preparation rather than loosening the definition.
