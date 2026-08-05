# Phase-resolved elastic-collision infrastructure

This package is the interface between equilibrated NEP-MB-pol ice snapshots
and the future ion-trajectory calculation. It currently provides two
validated prerequisites:

1. strict ingestion and provenance checking for periodic extended-XYZ ice
   snapshots; and
2. independent-atom NLH collision kernels for projectiles H, He, C, O, and S
   against the H and O nuclei in ice.

It does **not** yet produce phase-resolved amorphous- or hexagonal-ice cross
sections. That result requires the next structure-aware stage to propagate a
projectile through the registered atomic coordinates, sequence binary
collisions without double counting, sample orientations and independent
snapshots, and validate the result against the independent-atom limit. The
manifests label this limitation explicitly.

## Physical definition of one kernel

For a projectile with total kinetic energy `T`, `scattering.py` obtains the
relativistically exact center-of-mass kinetic energy for a stationary target.
For each impact parameter it solves the distance of closest approach in the
published NLH pair potential and evaluates the classical central-potential
deflection integral with Gauss-Legendre quadrature. The center-of-mass angle is
then transformed using exact two-body kinematics to give the projectile lab
angle and target recoil energy.

The default hard-collision boundary is the impact parameter whose turning
potential is 30 eV, the range in which NLH reports its strongest general
agreement. The published 10 eV lower domain can be selected explicitly. This
quantity is a **pair-potential validity boundary**, not a 10 or 30 eV recoil
cut and not an energy-deposition threshold. Weak, distant scattering below
the boundary is intentionally absent until a validated long-range
projectile--ice interaction is supplied.

The default energy range is 1 keV--100 MeV in **total projectile kinetic
energy**. The 1 keV floor keeps even the least favorable supported pair,
S--H, just inside the 30 eV center-of-mass domain. Point counts are not fixed.
The generator starts from 121 logarithmic energies and adaptively inserts
pair-specific energies after direct quarter/midpoint/three-quarter checks. At
every energy it independently refines the collision-area coordinate
`q=(b/b_max)^2` wherever linear interpolation of the center-of-mass angle,
followed by exact two-body kinematics, fails recoil, transport, or angular
tests. The runtime reader must use this same interpolation contract; it must
not interpolate angle and recoil as unrelated quantities.

The per-axis tolerance is 0.25%, giving a nominal two-axis budget of 0.5%.
Every output manifest records the actual point counts and maximum estimated
errors. The hard-collision cross section is evaluated from the exact
threshold formula and must not be interpolated through its onset. This avoids
the S--H threshold error found in the original fixed-grid diagnostic.

## 1. Attest equilibrated structures

Run the commands below from the parent `nep_mbpol` directory.

Do not attest the committed initial ice-Ih cell. After the cluster trajectory
passes the phase-specific RDF, structure-factor/order, density, and
thermodynamic-stationarity checks described in the parent README, preserve the
reports and run:

```bash
python3 attest_collision_structure.py final_hexagonal_trajectory.xyz \
  --phase "hexagonal ice Ih, 80 K" \
  --validation-report hexagonal_validation.json \
  --confirm-accepted
```

Repeat for every accepted independent trajectory. The sidecar records the
exact SHA-256 of both the trajectory and validation evidence. Register the
accepted inputs with:

```bash
python3 register_collision_structures.py \
  final_hexagonal_seed1000.xyz \
  final_hexagonal_seed2000.xyz \
  final_hexagonal_seed3000.xyz \
  --output collision_structures_hexagonal.json
```

The reader selects the last frame by default. Use `--frame N` to register a
specific frame. `--allow-unvalidated` exists only for checking the plumbing;
it marks every such entry `diagnostic-only`.

## 2. Generate reusable pair kernels

The production default uses ten worker processes and restartable per-energy
checkpoints:

```bash
python3 generate_nlh_collision_kernels.py
```

The output directory contains:

- `nlh_collision_kernels.csv`: H/O collision geometry, angles, and recoil
  kernels on pair- and energy-specific adaptive meshes. The CSV intentionally
  stores only projectile, target, energy, area quantile, and CM angle. Impact
  parameter, lab angle, recoil, and outgoing energy are exact derived
  quantities and are not duplicated;
- `nlh_collision_kernels.manifest.json`: numerical configuration, physical
  scope, and excluded physics; and
- `.checkpoints/<configuration hash>/`: restart blocks.

For a quick infrastructure check:

```bash
python3 generate_nlh_collision_kernels.py \
  --projectiles C \
  --energy-min-ev 10000 \
  --energy-max-ev 100000 \
  --base-energy-points 3 \
  --axis-relative-tolerance 0.05 \
  --max-energy-points 32 \
  --max-impact-points 256 \
  --quadrature-order 32 \
  --workers 2 \
  --output-directory collision_kernels_test
```

Do not install the CSV in Geant4 yet. It is the collision kernel consumed by
the forthcoming structure-aware trajectory generator, not the final
phase-specific macroscopic cross-section table.

## 3. Reproduce the independent dense-reference benchmark

The internal adaptive estimator is checked against direct solutions on a
separate grid extending down to `q=10^-24`:

```bash
python3 benchmark_nlh_adaptive_kernels.py
```

The command tests H, He, C, O, and S against H and O at six energies spanning
1 keV--100 MeV, uses ten workers, and fails with a nonzero status if any
recoil, transport, angular, or quadrature metric exceeds 0.5%. The production
96-point scattering quadrature is compared independently with 192 points. The
command writes the complete case tables and summary under
`collision_benchmarks/`.

## Tests

```bash
python3 -m pytest -q \
  ../../../tests/test_nlh_potential.py \
  ../../../tests/test_nlh_bca_scattering.py \
  ../../../tests/test_nlh_bca_structure.py \
  ../../../tests/test_nlh_bca_adaptivity.py \
  ../../../tests/test_nlh_bca_tables.py
```

The tests cover the published NLH evaluator, turning-potential boundary,
monotonic deflection/recoil behavior, energy conservation, quadrature
convergence, adaptive discovery of the narrow high-energy head-on region,
threshold energy refinement, structure attestation, deterministic
multiprocessing, and resume.

The underlying NLH references and corrected dataset are documented in
`../nlh/README.md`.
