# example_3

`example_3` is the multi-particle, multi-threaded Geant4-IcyMoons example.

It is meant to mirror the way the Europa energy-library runs are configured:

- multiple primaries in one run
- multi-threaded execution
- cosine angular distribution
- geometry-limited `maxtheta`

It runs a single macro, [`example_3.mac`](example_3.mac), which injects a `0.5 MeV` electron source.

The macro runs:

- `/run/beamOn 100`

So the output ROOT file contains `100` events total by default.

## Files

- macro: [`example_3.mac`](example_3.mac)
- runner: [`run_example_3.sh`](run_example_3.sh)
- plotter: [`plot_example_3_depth_diagnostics.py`](plot_example_3_depth_diagnostics.py)
- output directory: [`outputs/`](outputs/)

## Physics, Threads, And Angular Distribution

Default physics mode:

- `DNA_PHYSICS=ice_am`

Threading in the macro:

- `/run/numberOfThreads 10`

The runner also launches `dnaphysics` with `10` threads, matching the macro.

Logging mode:

- `/dna/test/setLogMode minimal`

Angular distribution:

- `/gps/ang/type cos`
- `/gps/ang/mintheta 0 deg`
- `/gps/ang/minphi 0 deg`
- `/gps/ang/maxphi 360 deg`
- `/gps/ang/rot1 1 0 0`
- `/gps/ang/rot2 0 -1 0`
- `/dna/test/setMaxTheta true`

This is the cosine-theta setup used in the Europa-style workflow.

`/dna/test/setMaxTheta` controls how wide the incoming angular cone is allowed to be.

Physically, with a cosine source you are sampling a distribution of incident directions around the surface normal. If that cone is too wide, many sampled electrons would be aimed so obliquely that they never hit the top face of the slab at all; they would fly past the slab sideways instead of entering the ice.

- `true`: computes the largest polar angle for which a particle launched from the current source position would still intersect the slab footprint. In other words, it trims the cosine distribution to directions that can actually enter the slab.
- `false`: sets `/gps/ang/maxtheta 90 deg`, so the cosine distribution is not geometry-limited and can include directions that miss the slab laterally.

For the standard setup used here, the slab starts at `z = 0` and the source is placed at `z < 0`. The code uses the smaller lateral half-width of the slab and the source-to-surface distance to find the grazing angle to the slab edge, then uses that as `maxtheta`.

## Geometry And Source

The macro currently uses:

- slab size: `100 x 100 x 100 mm`
- material: `G4_WATER_ICE_USER`
- density: `0.94 g/cm^3`
- source position: `0 0 -0.001 mm`
- particle: `e-`
- beam energy: `0.5 MeV`
- particles: `100`

## Required Paths

You need:

- a `dnaphysics` executable
- `G4LEDATA` pointing to a valid Geant4 `G4EMLOW...` directory

Typical executable path:

- `<repo>/dnaphysics-ice/build/dnaphysics`

Typical `G4LEDATA`:

- `<repo>/geant4_projects/g4_custom_ice/install/share/Geant4/data/G4EMLOW8.6.1`

If needed:

```bash
export G4LEDATA="/absolute/path/to/G4EMLOW8.6.1"
```

## Run example_3

From the `dnaphysics-ice` repo directory:

run:

```bash
./examples/example_3/run_example_3.sh \
  --binary "./build/dnaphysics"
```

Optional arguments:

- `--output-dir PATH`
- `--physics ice_am|ice_hex|water`

What the runner does:

- creates the output directory if needed
- changes into that output directory
- sets `DNA_ROOT_BASENAME=example_3`
- sets `DNA_NTUPLE_FILES=0`
- launches `dnaphysics` on [`example_3.mac`](example_3.mac)

Expected ROOT output:

- `outputs/example_3.root`

## Plot example_3

The plotting script reads the single `0.5 MeV` batch in this example and produces depth-diagnostic summary panels.

Run:

```bash
python3 examples/example_3/plot_example_3_depth_diagnostics.py
```

or explicitly:

```bash
python3 examples/example_3/plot_example_3_depth_diagnostics.py \
  --root "examples/example_3/outputs/example_3.root" \
  --out "examples/example_3/outputs/example_3_depth_diagnostics.png"
```

The plot shows:

- electron deposited energy per primary vs depth
- all-particle deposited energy per primary vs depth
- primary and secondary electron kinetic energy vs depth
- cumulative deposited-energy fraction vs depth

## What You Can Change

Inside the macro:

- thread count: `/run/numberOfThreads ...`
- slab size: `/dna/test/setIceSize X Y Z mm`
- material density: `/dna/test/setMatDens ...`
- material choice: `/dna/test/setMat ...`
- source position: `/gps/pos/centre ...`
- angular distribution and geometry cap:
  - `/gps/ang/type ...`
  - `/gps/ang/rot1 ...`
  - `/gps/ang/rot2 ...`
  - `/dna/test/setMaxTheta ...`
- the beam energy: `/gps/ene/mono ...`
- particle count: `/run/beamOn ...`

Outside the macro:

- physics mode through `DNA_PHYSICS`, or the runner option `--physics`

If you change the thread count, keep the macro `/run/numberOfThreads` and the runner’s thread count aligned.

## Runtime Note

This is the heaviest of the three examples. It combines a `100 x 100 x 100 mm` slab, cosine angular sampling, multi-threading, and `100` particles at `0.5 MeV`.
