# Low-energy RT-TDDFT

## Current status

The first target is the published 1 keV H+--H2O single-electron capture
benchmark: H+ captures one electron and becomes neutral H. The probability
analysis is implemented; **the published collision is not yet reproduced or
ready to run as a complete benchmark**. Its missing reference inputs and
propagation stages are recorded in the
[benchmark protocol](benchmarks/hong_2016/README.md).

| Component | Status |
|---|---|
| Accepted amorphous/Ih structure loading and finite-cluster extraction | Implemented and tested |
| Octopus target ground state and unperturbed propagation | Implemented; local execution checks recorded |
| Frozen-target, prescribed-proton feasibility runner | Implemented; not the published benchmark |
| Complex-orbital determinant counting and capture-curve assembly | Analytical and synthetic tests pass |
| Published collision, projectile-frame translation, numerical comparison | Incomplete |
| Periodic ice collisions, embedding, bulk channel rates, C/O/S | Not implemented |

No transport cross sections or validated 100 eV results are supplied here.
The eight-electron capture analyzer currently targets one water molecule.
P1 counts one electron in the separated projectile region and is inclusive
over residual-target states; it does not establish an intact H2O+ product.

## Layout

```text
prepare.py                 accepted ice -> finite-target input bundles
run.py                     target ground state + unperturbed control
hplus.py                   separate prescribed-proton execution prototype
capture.py                 native complex orbitals -> electron-count probabilities
capture_curve.py           stationary analyses -> 2*pi*b*P1 curve
benchmarks/hong_2016/       reference protocol, missing assets, analysis instructions
checks/single_water/       compact completed execution records, not physical validation
examples/                  prepared five-water targets from both ice phases
docs/                      installation, prototype, and HPC design
```

Solver source is the pinned sibling [Octopus submodule](../Octopus).
Build details and solver-test limitations are in
[INSTALLATION.md](docs/INSTALLATION.md). Binaries and build products are not
tracked. The local installation has OpenMP but no MPI.

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

Independent encounters can run as separate HPC jobs once their collision
workflow is complete. No scheduler submission or 12-worker benchmark launcher
is implemented by the current analysis tools.

## Tests

```bash
python -m pytest tests/test_rt_tddft.py tests/test_rt_tddft_hplus.py \
  tests/test_rt_tddft_capture.py -q
```

The capture tests exercise analytical and synthetic data, not benchmark
collision results. Generated `cases/`, `runs/`, and temporary output should
remain untracked; retained checks must include provenance and explicit limits.
