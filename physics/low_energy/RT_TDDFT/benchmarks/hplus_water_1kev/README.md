# H+–H2O at 1 keV: trajectory and capture stationarity

[Molecular collision visualization](collision.pdf) · [Trajectory and stationarity diagnostics](benchmark.pdf)

The molecular PDF shows three measured times from one collision: approach,
closest approach to oxygen, and departure. Positions use an equal-scale xy
projection. Atom radii and bond strokes are illustrative. The projectile
label denotes its nucleus, not an asserted outgoing charge state.

**This is one collision, not an ensemble.** The time samples and continuation
stages are parts of the same trajectory. No statistical uncertainty band is
available. Numerical uncertainty requires grid, timestep, boundary and other
convergence studies; the stationarity range is not an error bar. Additional
impact parameters and orientations would characterize variation between cases,
not automatically the uncertainty of this particular trajectory.

The independent Octopus 16.4 calculation uses impact parameter b = 2 bohr
and the water orientation `xy_bisector_positive_x`. The collision stage ends
at 299.9 atomic time units. Clearing time starts at this handoff into the
outgoing projectile's inertial frame; it is not the collision clock.

- **(a)** Measured nuclear trajectories in the laboratory xy plane. Dots mark
  the initial positions and crosses the handoff positions. The two axes use
  different scales; apparent slopes do not give scattering angles.
- **(b)** Determinant probabilities for the electron count within the entire
  25-bohr projectile-frame sphere, including its absorbing boundary.
- **(c)** P(1), with an expanded vertical scale. The final value is 0.335077333.
- **(d)** Maximum range over any P(n) within the last three samples, spanning
  100 au. The first passing sampled window ends at 250 au. At 600 au the
  range is 0.000302942, below the unchanged 0.001 tolerance.

Points are measured samples; connecting lines guide the eye. These are
regional Kohn–Sham determinant counts, not a direct measurement of asymptotic
charge-state yields. Stationarity passes; grid, timestep, boundary and
separation convergence and comparison with published curves remain outstanding.
This figure is not a reproduction of a Hong et al. figure.

## Reproduce

From the repository root:

```bash
.venv/bin/python -m physics.low_energy.RT_TDDFT.benchmarks.hplus_water_1kev.plot
```

The CSVs retain the full nuclear trajectory and all 13 probability samples.
`provenance.json` records source paths, SHA-256 hashes, and physical settings.
To export fresh tables from this completed case, add `--case /path/to/case`;
the exporter reads `clearing_0024000/capture.json`. Raw orbitals are unnecessary
for regenerating this figure. Only vector PDFs are produced and overwritten on regeneration; plots have no grids.

## Retained execution records

`collision_result.json` records the completed initial calculation (PBS
246063.pbs02), including its original failed stationarity check at 200 au.
`capture.json` records the complete probability series through 600 au after
continuation job 247653.pbs02; its final stationarity check passes. These
records deliberately preserve their original scopes. They include input,
solver and orbital hashes; large orbital/checkpoint files remain on the cluster.

Only this one molecular collision has been executed here. The separate
[periodic-ice scaffold](../../docs/BULK.md) is the starting point for a future
phase-resolved stopping study, not an ensemble underlying this figure.
