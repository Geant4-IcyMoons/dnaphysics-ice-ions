# Low-energy RT-TDDFT

## Current status

The first target is a 1 keV H+--H2O single-electron capture calculation.
An **independent collision workflow is runnable**, including neutral-water
initialization, Ehrenfest propagation, a projectile-frame orbital handoff,
and determinant electron counting. The full-resolution molecular calculation and continuation have completed;
late-time stationarity passes at 600 au. See the
[results and PDF visualizations](benchmarks/hplus_water_1kev/README.md). Numerical convergence and comparison with Hong et al.
remain outstanding; this is not a validated reproduction of their results.
See the [runnable calculation](docs/BENCHMARK.md) and the
[reference protocol](benchmarks/hong_2016/README.md).

| Component | Status |
|---|---|
| Accepted amorphous/Ih structure loading and finite-cluster extraction | Implemented and tested |
| Octopus target ground state and unperturbed propagation | Implemented; local execution checks recorded |
| Frozen-target, prescribed-proton feasibility runner | Implemented; not the published benchmark |
| Complex-orbital determinant counting and capture-curve assembly | Analytical, synthetic and solver execution checks pass |
| Independent molecular collision and projectile-frame handoff | Implemented; full-resolution execution and late-time stationarity pass |
| Numerical convergence and published reference comparison | Outstanding |
| Periodic frozen-host H input scaffold and finite-path energy analysis | Implemented; tiny periodic solver checks only |
| Validated bulk stopping, embedding, bulk channel rates, C/O/S | Outstanding |

No transport cross sections or validated 100 eV results are supplied here.
The eight-electron capture analyzer currently targets one water molecule.
P1 counts one electron in the separated projectile region and is inclusive
over residual-target states; it does not establish an intact H2O+ product.

## Layout

```text
bulk.py                    periodic H stopping inputs and finite-path energy diagnostic
prepare.py                 accepted ice -> finite-target input bundles
run.py                     target ground state + unperturbed control
hplus.py                   separate prescribed-proton execution prototype
benchmark.py               runnable isolated H+--H2O collision and capture workflow
frame.py                   complex orbital translation and Galilean boost
capture.py                 native complex orbitals -> electron-count probabilities
capture_curve.py           stationary analyses -> 2*pi*b*P1 curve
benchmarks/hong_2016/       reference protocol, missing assets, analysis instructions
checks/single_water/       earlier target execution records
checks/hplus_capture/      local/MPI verification and startup-failure record
jobs/hplus_water.pbs       ChemFarm PBS launcher
examples/                  prepared five-water targets from both ice phases
docs/                      installation, prototype, and HPC design
```

Solver source is the pinned sibling [Octopus submodule](../Octopus).
Build details and solver-test limitations are in
[INSTALLATION.md](docs/INSTALLATION.md). Binaries and build products are not
tracked. The Mac installation has OpenMP but no MPI; the
[Chemfarm Linux installation](docs/LINUX_INSTALLATION.md) has MPI and OpenMP
and passed the recorded installation checks.

## Periodic bulk ice

The [bulk scaffold](docs/BULK.md) accepts a complete periodic H2O cell of any
phase, with structural validation required by default. It preserves every atom,
prepares a relaxed charged cell and drives one H nucleus through the frozen host.
It does not prepare a bare incoming H+ state or produce capture probabilities.
See the guide for inputs, execution, analysis and outstanding convergence work.

## Target preparation and controls

Use Python with the root requirements installed. Commands run from the
repository root. The [shared ice registry](../../../models/ice/README.md)
provides the accepted structures; source hashes and cell data are checked.
Extraction preserves whole molecules and their recorded geometries.

```bash
python -m physics.low_energy.RT_TDDFT.prepare \
  --phase amorphous_lda_80k --replica 0 --center 0 --molecules 5 \
  --spacing 0.3 --padding 6 --dt-au 0.02 --steps 100 \
  --energy-ev 1000 --impact 1 --separation 15 \
  --output physics/low_energy/RT_TDDFT/cases/amorphous_5water

python -m physics.low_energy.RT_TDDFT.run \
  --case physics/low_energy/RT_TDDFT/cases/amorphous_5water \
  --output physics/low_energy/RT_TDDFT/runs/amorphous_5water \
  --executable octopus
```

Use `--phase hexagonal_ih_100k` for Ih. Replica and molecule indices are
zero-based. These commands prepare finite targets and run controls; they do
not execute the `collision.fragment` supplied in each bundle. Grid, box, and
time-step settings are starting values, not converged recommendations.

The [prescribed-proton prototype](docs/HPLUS_PILOT.md) is a separate diagnostic
path. Its sphere-integrated densities must not be used as capture probabilities.
Desktop five-water attempts were stopped and yielded no completed collision
results; their intermediate output is not included.

## Capture analysis and HPC direction

See the [benchmark instructions](benchmarks/hong_2016/README.md) for native
orbital exports, the analysis manifest, command-line usage, and limitations.
The [HPC design](docs/HPC_DESIGN.md) specifies the literature-first validation
sequence and distinguishes molecular channels from bulk electronic stopping.
A smaller periodic counterpart of each accepted ice structure still requires
preparation and structural validation; an arbitrary periodically wrapped cutout
is not sufficient.

Independent molecular encounters can run as separate HPC jobs. The benchmark
runner accepts an MPI launcher for a cluster-built Octopus; local testing used
OpenMP. Scheduler allocation and MPI execution are documented in the benchmark guide.

## Tests

```bash
python -m pytest tests/test_rt_tddft.py tests/test_rt_tddft_hplus.py \
  tests/test_rt_tddft_capture.py tests/test_rt_tddft_benchmark.py -q
```

The capture tests exercise analytical and synthetic data. The optional solver
integration test runs a coarse collision; see the benchmark guide for its
explicit opt-in command. It does not establish physical accuracy. Generated `cases/`, `runs/`, and temporary output should
remain untracked; retained checks must include provenance and explicit limits.
