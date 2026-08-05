# Completed 100 K ice-Ih candidate snapshots

This directory preserves the compact final snapshots from the three completed
NEP-MB-pol/GPUMD preparation runs. Each system contains 8,192 water molecules
(24,576 atoms) in a fully periodic orthorhombic cell at the experimental 100 K
ice-Ih density, 0.9334742974 g/cm3. There is no vacuum gap.

The preparation maps the fractional coordinates of an independently relaxed
80 K ice-Ih replica into the 100 K experimental cell derived from the volume
and `c/a` fits of Rottger et al. (2012), performs a FIRE minimization, and then
runs 1 ns of NVT equilibration followed by 1 ns of NVT sampling. Structure
and velocity seeds are paired as 1000, 2000, and 3000. The exact protocol is
implemented by `../../prepare_hexagonal_ice_experimental_cell.py`,
`../../inputs/hexagonal_ih_100K_experimental_density.in`, and
`../../../../../pbs/run_nep_mbpol_hexagonal.pbs`.

The gzip-compressed XYZ files contain positions, masses, velocities, and
periodic-cell data. Deterministic gzip archives preserve the GPUMD output
byte-for-byte while avoiding source-diff whitespace noise. `manifest.json`
records both compressed and uncompressed checksums, lineage, run completion
times, and thermodynamic summaries. `preview_seed1000.png` is a visualization
only and is not simulation input.

These are completed **candidate** structures, not yet accepted collision
targets. Their numerical dynamics completed successfully, but the required
ice-Ih RDF, Bragg/order, hydrogen-bond-network, and cross-replica validation is
still pending. Do not change that status merely because the target density is
exact by construction.

Reference: Rottger et al. (2012), corrected H2O and D2O ice-Ih lattice
polynomials, <https://doi.org/10.1107/S0108768111046908>.
