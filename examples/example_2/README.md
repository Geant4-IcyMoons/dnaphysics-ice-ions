# example_2

`example_2` is the full-logging Geant4-IcyMoons example.

It runs a single macro, [`example_2.mac`](example_2.mac), which injects one electron at each of these energies, sequentially, into amorphous ice:

- `0.1 MeV`
- `1 MeV`
- `10 MeV`

So the output ROOT file contains three events in one file, not three separate files.

## Files

- macro: [`example_2.mac`](example_2.mac)
- runner: [`run_example_2.sh`](run_example_2.sh)
- plotter: [`plot_example_2_process_diagnostics.py`](plot_example_2_process_diagnostics.py)
- output directory: [`outputs/`](outputs/)

## Physics And Logging

Default physics mode:

- `DNA_PHYSICS=ice_am`

Logging mode inside the macro:

- `/dna/test/setLogMode full`

This example is intended for full process diagnostics. The full ROOT file includes the richer `step` schema plus the `config` and `track` trees needed for process-level inspection.

## Geometry And Source

The macro currently uses:

- slab size: `10 x 10 x 100 mm`
- material: `G4_WATER_ICE_USER`
- density: `0.94 g/cm^3`
- source position: `0 0 -0.01 mm`
- direction: `0 0 1`
- particle: `e-`
- one primary per energy: `/run/beamOn 1`

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

## Run example_2

From the `dnaphysics-ice` repo directory:

run:

```bash
./examples/example_2/run_example_2.sh \
  --binary "./build/dnaphysics"
```

Optional arguments:

- `--output-dir PATH`
- `--threads N`
- `--physics ice_am|ice_hex|water`

What the runner does:

- creates the output directory if needed
- changes into that output directory
- sets `DNA_ROOT_BASENAME=example_2`
- sets `DNA_NTUPLE_FILES=0`
- launches `dnaphysics` on [`example_2.mac`](example_2.mac)

Expected ROOT output:

- `outputs/example_2.root`

## Plot example_2

The plotting script is designed for this single ROOT file and separates the three event energies internally using `eventID` and the event tree.

Run:

```bash
python3 examples/example_2/plot_example_2_process_diagnostics.py
```

or explicitly:

```bash
python3 examples/example_2/plot_example_2_process_diagnostics.py \
  --root "examples/example_2/outputs/example_2.root" \
  --out "examples/example_2/outputs/example_2_process_diagnostics.png"
```

The plot shows, by event energy:

- electron step counts by process
- electron deposited energy by process
- electron kinetic-energy loss by process
- event energy-budget fractions

## What You Can Change

Inside the macro:

- logging level: `/dna/test/setLogMode ...`
- slab size: `/dna/test/setIceSize X Y Z mm`
- material density: `/dna/test/setMatDens ...`
- material choice: `/dna/test/setMat ...`
- source position: `/gps/pos/centre ...`
- source direction: `/gps/direction ...`
- the three energies: `/gps/ene/mono ...`

Outside the macro:

- physics mode through `DNA_PHYSICS`, or the runner option `--physics`
- thread count through `--threads`

If you switch to `water`, also update the material lines in the macro accordingly.

## Runtime Note

This is a real Geant4-DNA track-structure example with full logging. The `1 MeV` and especially `10 MeV` events can take noticeable time, and a `10 cm` slab plus full logging makes this example heavier than `example_1`.
