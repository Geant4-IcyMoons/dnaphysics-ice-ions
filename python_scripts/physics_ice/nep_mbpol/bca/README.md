# Phase-resolved elastic-collision infrastructure

This package connects equilibrated NEP-MB-pol ice snapshots to retained-domain
NLH hard-collision trajectories. It provides:

1. strict ingestion and provenance checking for periodic extended-XYZ ice
   snapshots; and
2. independent-atom NLH collision kernels for projectiles H, He, C, O, and S
   against the H and O nuclei in ice;
3. a checksum-validated adaptive-kernel runtime reader; and
4. periodic structure-aware sequencing of primary H, He, C, O, and S hard
   collisions,
   including exact projectile deflection and emitted-recoil kinematics.

The trajectory stage produces phase- and orientation-resolved **hard-event
samples**, not a complete elastic cross section. Soft distant scattering,
simultaneous many-atom forces, recoil-cascade reinsertion, lattice relaxation,
electronic stopping, and charge exchange remain separate physics. Every run
manifest states these exclusions and reports collisions for which another hard
candidate overlaps the binary encounter.

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

The accepted 100 K ice-Ih replicas are already attested under
`../structures/hexagonal_ih_100K_experimental/`. Their portable
`collision_structures.json` registry points to the three gzip-compressed final
snapshots, and the checksummed evidence is under its `validation/` directory.
The XYZ reader supports both plain and `.gz` inputs.

## 2. Generate reusable pair kernels

The production default uses ten worker processes and restartable per-energy
checkpoints:

```bash
python3 generate_nlh_collision_kernels.py
```

On PBS, the repository launcher uses one 256-CPU, 512-GB `idlex` allocation,
pins threaded numerical libraries to one thread per worker, and generates all
five supported projectile families:

```bash
qsub pbs/generate_nlh_collision_kernels.pbs
```

Submit from the repository root. A repeated submission resumes the same
configuration-hashed checkpoints and does not recompute completed kernels.

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

Do not install the CSV directly in Geant4. It is the differential collision
kernel consumed by the structure-aware trajectory generator, not a final total
elastic cross-section table.

## 3. Run hard trajectories through an accepted ice cell

The runtime requires the exact attested snapshot and its validation evidence.
For a c-axis calculation through ice Ih:

```bash
python3 simulate_nlh_hard_collisions.py final_hexagonal_seed1000.xyz \
  --projectile C \
  --energy-ev 100000 \
  --direction 0 0 1 \
  --path-length-angstrom 100
```

Repeat with `--projectile H`, `He`, `O`, and `S`. `H` and `He` are the
proton/helium-projectile entries; the dominant-isotope masses are used, and the
short-range nuclear potential is charge-state independent. Use
`--isotropic-directions` only for an explicitly orientation-averaged target;
it must not be described as an oriented single-crystal result. The default is
ten worker processes. Trajectory seeds depend only on the master seed and
trajectory index, so outputs are independent of worker count and completion
order.

If `--trajectories` is omitted, production sampling is adaptive. Independent
histories are added in restartable batches of 1,000 and assessed after
1,000, 2,000, 4,000, ... histories, up to 1,024,000 by default. Sampling stops
only when the asymptotic 95% simultaneous relative confidence half-width is at
most 0.5% for the hard-event rate, hard nuclear stopping, hard transport rate,
mean recoil energy per collision, and mean `1-cos(theta_lab)` per collision. The
trajectory is the independent statistical unit, so collisions correlated
along one history are not falsely counted as independent. Student-t ratio
intervals use a Bonferroni correction over all five observables and every
scheduled look, preventing interim checks from weakening the stated
confidence level.

Every completed batch is checksum-protected below
`.trajectory_checkpoints/<configuration hash>/`; rerunning the same command
resumes without repeating finished trajectories. Reaching the maximum without
passing all gates writes the diagnostic outputs and exits nonzero. Use an
explicit fixed count only for plumbing or an externally controlled study:

```bash
python3 simulate_nlh_hard_collisions.py final_hexagonal_seed1000.xyz \
  --projectile C \
  --energy-ev 100000 \
  --isotropic-directions \
  --trajectories 1000
```

The 0.5% Monte Carlo gate is separate from the adaptive collision-kernel
interpolation budget. It certifies normalization and first energy/angular
moments, not rare tails or a binned angular/recoil CDF. Those distributions
require a separate confidence-band test when the final Geant4 reducer is
added.

The output directory contains:

- `hard_collision_trajectories.csv`: initial/final projectile state and total
  recoil energy for each history;
- `hard_collision_events.csv`: target atom and periodic image, impact
  parameter, incoming/outgoing directions, recoil direction and energy for
  every retained hard event; and
- `hard_collision_run.manifest.json`: exact structure/kernel provenance,
  configuration, sampled hard rate, uncorrelated independent-atom reference,
  ambiguity count, statistical intervals, and excluded physics.

`--allow-unvalidated` is restricted to plumbing tests with the committed
initial Ih cell. Its manifest remains `diagnostic-only`; such a run is not a
phase-resolved scientific result.

## 4. Reproduce the independent dense-reference benchmark

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
  ../../../tests/test_nlh_bca_convergence.py \
  ../../../tests/test_nlh_bca_tables.py \
  ../../../tests/test_nlh_bca_runtime.py
```

The tests cover the published NLH evaluator, turning-potential boundary,
monotonic deflection/recoil behavior, energy conservation, quadrature
convergence, adaptive discovery of the narrow high-energy head-on region,
threshold energy refinement, structure attestation, deterministic
multiprocessing, resume, checksum-validated runtime interpolation, exact
periodic images, search-window invariance, and structure-to-kernel collision
sequencing. The trajectory sampler additionally tests its simultaneous
confidence correction, clustered ratio uncertainty, adaptive schedule, and
restart statistics.

The underlying NLH references and corrected dataset are documented in
`../nlh/README.md`.
