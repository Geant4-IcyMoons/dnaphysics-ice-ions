# Validated 100 K ice-Ih collision structures

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

The three structures passed the documented acceptance protocol in
`../../HEXAGONAL_ICE_VALIDATION.md`. Across all 30 sampling frames, CHILL+
classified 100% of molecules as bulk ice Ih, both Bernal--Fowler rules held
exactly, and all seven tested low-order Ih reflections were local
reciprocal-space maxima. No thermodynamic drift was resolved after a Holm
familywise correction, the three proton configurations are distinct, and all
lineage checks passed. The complete numerical record is under `validation/`;
`collision_structures.json` registers all three snapshots as production
collision inputs.

The exact 0.9334742974 g/cm3 density is an imposed experimental-cell input,
not independent validation evidence. The archived GPUMD restart headers have
limited decimal precision and therefore report 0.9334748 g/cm3 when reparsed;
the full-precision sampling trajectories and preparation record retain the
paper-derived cell. This sub-part-per-million serialization difference does
not alter the accepted phase or topology.

Primary definitions and provenance are Rottger et al. (2012),
<https://doi.org/10.1107/S0108768111046908>; Nguyen and Molinero (2015),
<https://doi.org/10.1021/jp510289t>; Bernal and Fowler (1933),
<https://doi.org/10.1063/1.1749327>; and Kuhs and Lehmann (1983),
<https://doi.org/10.1021/j100244a063>.
