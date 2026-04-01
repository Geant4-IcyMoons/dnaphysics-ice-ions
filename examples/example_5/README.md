# example_5

`example_5` is the hexagonal-ice W-value Geant4-IcyMoons example.

It mirrors the current production W-value workflow used by `dnaphysics-ice`: the same energy scan, the same event-count aliases, the same `minimal` logging mode, and the same text-output format for the W-value rows.

## Files

- macro: [`example_5.mac`](example_5.mac)
- runner: [`run_example_5.sh`](run_example_5.sh)
- plotter: [`plot_example_5_wvalue.py`](plot_example_5_wvalue.py)
- output directory: [`outputs/`](outputs/)

## Physics And Logging

Default physics mode:

- `DNA_PHYSICS=ice_hex`

Logging mode inside the macro:

- `/dna/test/setLogMode minimal`

This keeps the ROOT file compact while still producing the W-value text output through `DNA_WVALUE_FILE`.

## Geometry, Material, And Source

The macro currently uses:

- material: `G4_WATER`
- slab size: `20000 x 20000 x 20000 mm`
- particle: `e-`
- source position: `0 0 100 mm`
- source direction: `0 0 1`
- energy scan: `12 eV` through `1 MeV`

The event counts match the current workflow:

- `100` events per point from `12 eV` through `90 keV`
- `10` events per point from `100 keV` through `900 keV`
- `5` events at `1 MeV`

For `DNA_PHYSICS=ice_hex`, the current ice workflow applies the hexagonal-ice nominal density automatically while keeping the Geant4 material name as `G4_WATER`.

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

## Run example_5

From the `dnaphysics-ice` repo directory:

run:

```bash
./examples/example_5/run_example_5.sh \
  --binary "./build/dnaphysics"
```

Optional arguments:

- `--output-dir PATH`
- `--wvalue-file NAME`

What the runner does:

- creates the output directory if needed
- changes into that output directory
- sets `DNA_PHYSICS=ice_hex`
- sets `DNA_ROOT_BASENAME=example_5`
- sets `DNA_WVALUE_FILE` to the requested text filename
- sets `DNA_NTUPLE_FILES=0`
- launches `dnaphysics` on [`example_5.mac`](example_5.mac)

Expected outputs:

- `outputs/example_5.root`
- `outputs/example_5_wvalue.txt`

The W-value text file is appended to if it already exists. Remove it first if you want a clean restart.

## Plot example_5

Run:

```bash
python3 examples/example_5/plot_example_5_wvalue.py
```

or explicitly:

```bash
python3 examples/example_5/plot_example_5_wvalue.py \
  --txt "examples/example_5/outputs/example_5_wvalue.txt" \
  --out "examples/example_5/outputs/example_5_wvalue.png"
```

The plot shows:

- W value vs primary energy
- mean ionisation count vs primary energy

## W-value Text Format

Rows appended to the text file have the format:

- `E0[eV]  <Nion>  rms(Nion)  w[eV]  w_err[eV]`

## What You Can Change

Inside the macro:

- event-count aliases: `/control/alias NEVT ...`
- logging mode: `/dna/test/setLogMode ...`
- slab size: `/dna/test/setIceSize X Y Z mm`
- material line: `/dna/test/setMat ...`
- source position: `/gps/pos/centre ...`
- source direction: `/gps/direction ...`
- energy grid: `/gps/ene/mono ...`

Outside the macro:

- W-value text filename through `--wvalue-file`

If you change the thread count, keep the macro `/run/numberOfThreads 12` and the runner’s final `dnaphysics ... 12` argument aligned.

## Runtime Note

This example runs a full low-energy energy scan rather than a single event. It is much heavier than `example_1`, especially because the low-energy points use `100` events each.
