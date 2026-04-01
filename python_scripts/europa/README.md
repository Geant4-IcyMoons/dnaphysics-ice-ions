# Europa Energy Library Generator

This README documents the macro-and-table generator:

- [`europa/generate_europa_energy_library.py`](generate_europa_energy_library.py)

It is the README for the generator itself, not the full end-to-end production pipeline.

## What The Generator Physically Does

The generator takes the Europa bombardment maps and turns them into a simulation library that can be reused offline.

The idea is:

1. Divide Europa into latitude-longitude cells.
2. For each cell, infer which incident electron energies are physically relevant there.
3. Build one shared global energy grid that is fine enough to represent the electron spectrum across all valid cells.
4. Write one Geant4 macro per global energy.
5. Write tables that tell you, for each Europa cell, which global energies apply there and how strongly each one should be weighted.

Thus the generator does not do the transport simulation itself. It prepares:

- the source-energy library
- the per-energy Geant4 macros
- the offline reweighting tables

The reason for this design is that many Europa cells reuse the same incident energies. Instead of running a separate Geant4 job for every cell, the code runs each global energy once and reweights offline.

## Quick Command

From repository root:

```bash
python3 python_scripts/europa/generate_europa_energy_library.py \
  --out-dir dnaphysics-ice/europa_energy_library \
  --macro-name europa_energy_library.mac \
  --n-lat 30 --n-lon 30 \
  --tol 1e-3 --initial-bins 100 \
  --global-e-min 0.01 --global-e-max 100 \
  --leading-cap 100 --trailing-min 0.01 \
  --n-per-energy-thresholds 0.1 1 10 100 \
  --n-per-energy-values 10000 1000 50 5 \
  --dna-physics ice_am \
  --threads 10 \
  --x-half-mm 1000 --y-half-mm 1000 --z-thickness-mm 10000 \
  --source-x-mm 0 --source-y-mm 0 --source-z-mm -0.001 \
  --source-dir-x 0 --source-dir-y 0 --source-dir-z 1 \
  --angular-dist cos \
  --manual-density-gcm3 0.5 \
  --log-mode minimal
```

Then run the generated macros:

```bash
cd europa_energy_library
DNA_ROOT_MAX_MB=2048 ./run_per_energy.sh ../build/dnaphysics 10 ice_am
```

## What Gets Written

The generator writes:

- `europa_energy_library.mac`
  A single combined macro containing all global energies in one file.
- `macros/*.mac`
  One macro per global energy. This is what `run_per_energy.sh` uses.
- `global_energy_bins.csv`
  The common global energy grid and its bin metadata.
- `latlon_cell_ranges.csv`
  The physically allowed local energy range for each Europa cell.
- `latlon_energy_scaling.csv.gz`
  The offline weights that map local cell spectra onto the global energy bins.
- `per_energy_runs.csv`
  The run table that links each global energy to its macro and expected ROOT basename.
- `run_per_energy.sh`
  The helper script that loops over all generated macros and produces one ROOT output per global energy.

## Physical Picture Behind The Main Parameters

The parameters fall into five groups:

- Europa discretization: how finely you sample the moon surface.
- Energy library design: which incident energies exist in the shared library.
- Monte Carlo statistics: how many particles you simulate at each library energy.
- Geant4 transport setup: what slab, source, material, logging, and physics configuration each generated macro uses.
- Output bookkeeping: where files are written and how they are named.

Below, each parameter is explained in that physical context.

## 1) Europa Surface Discretization

### `--n-lat`

Number of latitude bins used to partition Europa.

Physical meaning:

- Larger values mean you resolve north-south variations in the bombardment maps more finely.
- Smaller values mean you average over larger latitude bands.

Tradeoff:

- Larger `n-lat` gives more spatial fidelity.
- But it also creates more distinct local energy ranges that the global energy library must satisfy.

### `--n-lon`

Number of longitude bins used to partition Europa.

Physical meaning:

- Larger values resolve leading-trailing and dawn-dusk structure more finely.
- Smaller values blend neighboring longitudes together.

Tradeoff:

- Higher `n-lon` captures sharper longitudinal gradients.
- But it usually forces a denser global energy grid and larger scaling tables.

## 2) Energy-Library Design

### `--tol`

Tolerance for the automatic search that decides how many global logarithmic energy bins are needed.

Physical meaning:

- The code wants the discrete sum `sum[J(E_i) DeltaE_i]` to match the continuous spectral integral `integral J(E) dE`.
- It enforces that not just globally, but on every valid cell-specific energy range.

So `--tol` controls how accurately the global energy library represents the underlying continuous incident spectrum.

Interpretation:

- Smaller `tol` means the discrete library more faithfully reproduces the continuous spectrum in every cell.
- Larger `tol` means a coarser, cheaper library, but with more spectral approximation error.

### `--initial-bins`

Starting guess for the automatic log-bin search.

Physical meaning:

- This is not the final number of bins unless the first guess already satisfies `--tol`.
- It only tells the search where to begin before doubling until the tolerance is met.

Practical meaning:

- A better initial guess can save generator time.
- It does not directly set the final physics resolution unless the tolerance is already satisfied there.

### `--global-e-min`

Forced lower edge of the shared global energy library, in MeV.

Physical meaning:

- No simulated library energy will go below this value, even if some local cell would allow lower energies.
- It defines the lowest incident energy you are willing to represent anywhere on Europa.

Use it when:

- You want to exclude very low energies because they are unimportant for your observable.
- Or you want the whole library to start at a known common floor.

### `--global-e-max`

Forced upper edge of the shared global energy library, in MeV.

Physical meaning:

- No simulated library energy will go above this ceiling.
- It is the global maximum energy you are willing to transport anywhere on Europa.

Use it when:

- You want to truncate the high-energy tail for cost reasons.
- Or you know your science case only needs energies below a fixed ceiling.

### `--leading-cap`

Upper energy cap applied to the leading hemisphere, in MeV.

Physical meaning:

- In the current legacy logic, the leading-side map supplies a lower cutoff `E_min`.
- The generator then assumes that all higher energies up to `leading_cap` are allowed in that cell.

So `leading_cap` is not a resolution parameter. It is a physical assumption about how far upward in energy the leading-side allowed range extends.

### `--trailing-min`

Fallback minimum energy for trailing-side cells when the trailing map has no positive entries.

Physical meaning:

- On the trailing side, the map logic returns an allowed band bounded by the positive entries in that map-derived vector.
- If a trailing cell has no positive values, `trailing_min` acts as the fallback lower edge.

So this parameter mainly matters in low-flux or poorly constrained trailing-side cells.

## 3) Monte Carlo Statistics Per Energy

### `--n-per-energy`

Single default number of primary electrons to simulate at every global energy.

Physical meaning:

- If you use one fixed value, every global energy gets the same Monte Carlo effort.

This is simple, but often inefficient:

- low energies usually need many more primaries because they matter strongly for dose/deposition and can have higher statistical variability
- high energies are expensive to transport, so you may intentionally run fewer

### `--n-per-energy-thresholds`

Piecewise upper-energy thresholds, in MeV, that define simulation-statistics bands.

Physical meaning:

- These thresholds split the energy axis into intervals.
- Each interval gets its own particle count from `--n-per-energy-values`.

The code uses the convention:

- energies in `(threshold[i-1], threshold[i]]` get `value[i]`

Example:

- thresholds `0.1 1 10 100`
- values `10000 1000 50 5`

means:

- simulate very heavily below `0.1 MeV`
- still heavily from `0.1` to `1 MeV`
- much more lightly from `1` to `10 MeV`
- very lightly above `10 MeV`

### `--n-per-energy-values`

Particle counts paired with `--n-per-energy-thresholds`.

Physical meaning:

- This is where you decide how much Monte Carlo noise you can tolerate at each energy scale.

Typical physical reasoning:

- low-energy electrons dominate shallow energy deposition and often need better statistics
- high-energy electrons are expensive and may contribute more smoothly in integrated observables, so fewer primaries can be acceptable

## 4) Transport Physics And Material Choice

### `--dna-physics`

Default physics mode written into the generated runner, one of:

- `water`
- `ice_hex`
- `ice_am`

Physical meaning:

- This chooses which transport model family is used when the generated macros are later executed.

Interpretation:

- `water` uses the liquid-water setup
- `ice_hex` uses the hexagonal-ice custom physics setup
- `ice_am` uses the amorphous-ice custom physics setup

This parameter does not change Europa’s incident spectrum. It changes how those incident electrons are transported through the slab.

### `--manual-density-gcm3`

Optional manual density override, in `g/cm^3`.

Physical meaning:

- If this value is used, the generated macros force a custom material density using `/dna/test/setMatDens`.
- In the current script version, the command-line default is `0.5`, so generated macros use `0.5 g/cm^3` unless you explicitly change it.
- The phase-default densities (`ice_hex -> 0.917`, `ice_am -> 0.94`, `water -> 1.0`) are only used when the generator is configured to leave `manual_density_gcm3` unset.

Use it when:

- You want to study porous or compacted ice at a non-nominal density.
- You want geometry and transport phase to stay fixed while varying density as a separate physical control.

Important:

- A manual density override changes stopping, penetration, and energy deposition per unit length.
- So it is not just a bookkeeping parameter. It changes the slab’s physical column density.

### `--log-mode`

Logging mode for the generated macros:

- `minimal`
- `full`

Physical meaning:

- This does not change the transport itself.
- It changes how much diagnostic information is written per step.

Interpretation:

- `minimal` is the cheaper production mode
- `full` is the richer debug/diagnostic mode

## 5) Slab Geometry

### `--x-half-mm`

Half-width of the slab in `x`, in mm.

Physical meaning:

- The full slab width is `2 * x-half-mm`.
- This controls how much lateral area is available for oblique primaries to enter and for secondaries to remain inside.

If you increase it:

- fewer oblique primaries miss the slab laterally
- fewer secondaries escape from the sides

### `--y-half-mm`

Half-width of the slab in `y`, in mm.

Physical meaning is the same as `x-half-mm`, but in the orthogonal horizontal direction.

### `--z-thickness-mm`

Slab thickness in `z`, in mm.

Physical meaning:

- The slab extends from `z = 0` to `z = z-thickness`.
- This controls the available column depth for penetration and energy deposition.

If you increase it:

- more high-energy electrons can stop inside the slab instead of escaping out the back
- forward escape generally decreases
- deep deposition becomes better represented

## 6) Source Position

### `--source-x-mm`

Source `x` coordinate, in mm.

Physical meaning:

- Shifts the source left or right relative to the slab center.
- Off-center sources make lateral escape asymmetric and can shrink the allowed angular cone when `--angular-dist cos` is used with geometry-limited `maxtheta`.

### `--source-y-mm`

Source `y` coordinate, in mm.

Same physical interpretation as `source-x-mm`, but in the orthogonal lateral direction.

### `--source-z-mm`

Source `z` coordinate, in mm.

Physical meaning:

- Controls how far above or below the slab entrance surface the source is placed.
- For the standard incidence setup, negative values mean the source is just above the slab, since the slab starts at `z = 0`.

If the source is farther away above the slab:

- the same angular spread covers a larger lateral footprint before reaching the surface
- more oblique directions can miss the slab

The usual production choice is a small negative offset just above the surface.

## 7) Source Direction And Angular Distribution

### `--source-dir-x`, `--source-dir-y`, `--source-dir-z`

Components of the nominal source-direction axis.

Physical meaning:

- These define the central incidence direction.
- If `--angular-dist none`, this is the exact beam direction.
- If `--angular-dist cos`, this is the axis around which the cosine distribution is centered.

So these parameters set the mean arrival direction of the incident electrons.

### `--angular-dist`

Angular model for the primary beam:

- `none`
- `cos`

Physical meaning:

- `none` means a perfectly collimated beam. Every primary has exactly the same direction.
- `cos` means a cosine-law angular distribution around the source-direction axis.

When to use each:

- `none` is useful for clean beam tests and controlled debugging
- `cos` is more realistic for diffuse surface incidence and is the Europa-style choice

In the current generator, `cos` also enables geometry-limited `maxtheta` in the generated macros. That means the code trims the angular cone so sampled directions still intersect the slab footprint instead of flying past it sideways.

## 8) Execution And Output Bookkeeping

### `--threads`

Thread count written into the generated macros and used as the default in the generated runner.

Physical meaning:

- This does not change the transported physics.
- It only changes how much parallelism Geant4 uses when each per-energy job is executed.

### `--out-dir`

Directory where the whole generated library is written.

Physical meaning:

- None directly. This is a bookkeeping parameter.
- But it determines where your macros, tables, runner, and later ROOT outputs live.

### `--macro-name`

Filename for the combined all-energies macro.

Physical meaning:

- None directly. This is a convenience/bookkeeping parameter.
- It is useful when you want multiple generated libraries in different folders or with different naming schemes.

## How To Choose Reasonable Values

If your goal is better physical fidelity:

- increase `n-lat` and `n-lon`
- decrease `tol`
- use a wider slab if you want to suppress lateral-miss artifacts
- choose `cos` angular distribution if you want diffuse incidence rather than a pencil beam

If your goal is lower runtime:

- increase `tol`
- reduce `n-lat` and `n-lon`
- reduce low-energy `n-per-energy-values`
- reduce `global-e-max` if your science case does not need the highest energies

If your goal is to study transport sensitivity:

- vary `dna-physics`
- vary `manual-density-gcm3`
- vary slab thickness and the source angular distribution separately

## Notes On Leading And Trailing Hemisphere Logic

The current script inherits the legacy range logic:

- leading hemisphere: map value acts like a local lower cutoff `E_min`, and allowed energies extend upward to `leading-cap`
- trailing hemisphere: the map contributes a local allowed band whose positive entries define the usable `E_min` and `E_max`

So the generator is not assuming the same spectral support everywhere on Europa. That is exactly why the offline scaling tables are needed.

## Running The Generated Macros

After generation:

```bash
cd europa_energy_library
DNA_ROOT_MAX_MB=2048 ./run_per_energy.sh ../build/dnaphysics 10 ice_am
```

That script:

- loops over `per_energy_runs.csv`
- runs one Geant4 job per global energy
- writes one ROOT file per energy under `root/`
- skips energies that already have output

## Related Guide

For the full Europa production workflow, including how to run and consume the generated library, see:

- [`../README_europa.md`](../../README_europa.md)
