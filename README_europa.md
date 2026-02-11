# Europa -> dnaphysics-ice Run Guide

This guide describes the full pipeline:

1. Build the Europa global energy library (90x90 lat/lon by default).
2. Generate one monoenergetic macro per global energy bin.
3. Run `dnaphysics` once per energy so each energy gets its own ROOT file.

## 1) Generate the Europa energy library

From `geant4_projects/dnaphysics-ice`:

```bash
python python_scripts/generate_europa_energy_library.py \
  --n-lat 90 \
  --n-lon 90 \
  --tol 1e-4 \
  --global-e-max 100 \
  --n-per-energy 100 \
  --dna-physics ice_am \
  --manual-density-gcm3 0.94 \
  --threads 12 \
  --x-half-mm 500 \
  --y-half-mm 500 \
  --z-thickness-mm 1000 \
  --source-x-mm 0 \
  --source-y-mm 0 \
  --source-z-mm -0.01 \
  --out-dir europa_energy_library
```

Requested high-stat custom-density example:

```bash
python python_scripts/generate_europa_energy_library.py \
  --n-lat 90 \
  --n-lon 90 \
  --tol 1e-4 \
  --global-e-max 100 \
  --n-per-energy 1000 \
  --threads 12 \
  --x-half-mm 5000 \
  --y-half-mm 5000 \
  --z-thickness-mm 1000 \
  --source-x-mm 0 \
  --source-y-mm 0 \
  --source-z-mm -0.01 \
  --out-dir europa_energy_library \
  --manual-density-gcm3 0.5 \
  --angular-dist cos \
  --dna-physics ice_am
```

Safe single-line equivalent (avoids shell line-continuation mistakes):

```bash
python python_scripts/generate_europa_energy_library.py --n-lat 90 --n-lon 90 --tol 1e-4 --global-e-max 100 --n-per-energy 1000 --threads 12 --x-half-mm 5000 --y-half-mm 5000 --z-thickness-mm 1000 --source-x-mm 0 --source-y-mm 0 --source-z-mm -0.01 --out-dir europa_energy_library --manual-density-gcm3 0.5 --angular-dist cos --dna-physics ice_am
```

From repository root (safe copy-paste):

```bash
cd geant4_projects/dnaphysics-ice && python python_scripts/generate_europa_energy_library.py --n-lat 90 --n-lon 90 --tol 1e-4 --global-e-max 100 --n-per-energy 1000 --threads 12 --x-half-mm 5000 --y-half-mm 5000 --z-thickness-mm 1000 --source-x-mm 0 --source-y-mm 0 --source-z-mm -0.01 --out-dir europa_energy_library --manual-density-gcm3 0.5 --dna-physics ice_am --angular-dist cos
```

Important:

- Use `ice_am` (not `ice_amt`).
- In multiline commands, each trailing `\` must be the very last character on the line.
- Use `--angular-dist cos` for cosine-law incidence (`theta` cosine-distributed in `[0, 90] deg`, `phi` uniform in `[0, 360] deg`).
- If you see `zsh: command not found: naphysics-ice`, run from repo root with the `cd geant4_projects/dnaphysics-ice && ...` command above.

Key geometry defaults in this generator:

- Ice slab spans `z = 0` to `z = 1 m` (`z-thickness-mm = 1000`).
- Slab is centered at `x=y=0`.
- Default source is at `0 0 -0.01 mm` (just above slab surface), directed along `+z`.

Density behavior:

- Set default phase for execution with `--dna-physics {water|ice_hex|ice_am}`.
- Default density (if `--manual-density-gcm3` is omitted):
  - `ice_hex` -> `0.917 g/cm^3`
  - `ice_am` -> `0.94 g/cm^3`
  - `water` -> `1.0 g/cm^3`
- Optional explicit override: `--manual-density-gcm3 <value>`.

Energy binning:

- Global energy bins are logarithmically spaced.
- `--tol` controls the maximum allowed mismatch in `sum(J(E_i)\Delta E_i)/\int J(E)\,dE` for every valid lat/lon cell range (`[E_min, E_max]`), not only a single global range.
- Because the criterion is enforced on all valid cells, stricter `--tol` can significantly increase the final number of global bins.

## 2) What files are generated

Under `europa_energy_library/`:

- `global_energy_bins.csv`: global unique energy bins and metadata.
- `latlon_cell_ranges.csv`: allowed local ranges per lat/lon cell.
- `latlon_energy_scaling.csv.gz`: per-cell scaling factors.
- `europa_energy_library.mac`: combined macro (all energies in one macro).
- `macros/*.mac`: one macro per energy bin.
- `per_energy_runs.csv`: mapping table for per-energy runs.
- `run_per_energy.sh`: batch runner (one dnaphysics process per energy).

## 3) Run dnaphysics per energy (separate ROOT per energy)

Run the generated script:

```bash
cd europa_energy_library
./run_per_energy.sh ../build/dnaphysics 12 ice_am
```

Arguments:

1. `dnaphysics` binary path (default: `../build/dnaphysics`)
2. thread count (default: value used during generation)
3. physics mode (`water`, `ice_hex`, `ice_am`; default: what you set in `--dna-physics`)

The runner enforces:

- `DNA_ROOT_BASENAME` per energy (unique name includes phase and density),
- `DNA_NTUPLE_FILES=0` (single ROOT file per energy run).
- Resume behavior: if the target ROOT file already exists for an energy, that energy is skipped.

Output ROOT files are written to:

- `europa_energy_library/root/`

File naming scheme:

- `dna_<phase>_rho<density>_E<index>_<energy-tag>.root`

Example:

- `dna_ice_am_rho0p94_E00042_1p23456789MeV.root`

## 4) Per-Macro Energy-Budget Check

You can audit each macro/root pair with:

```bash
python python_scripts/check_europa_energy_balance.py \
  --run-table europa_energy_library/per_energy_runs.csv \
  --out europa_energy_library/energy_balance_summary.csv
```

This writes one row per macro with:

- `E_in_expected_MeV = E_center_MeV * sim_particles`
- `E_deposited_MeV = sum(step.totalEnergyDeposit)`
- `E_not_deposited_MeV = E_in_expected_MeV - E_deposited_MeV`
- `deposited_fraction`
- `status` (`ok`, `missing_root`, `missing_step_tree`, `over_budget`, `read_error`)
- `budget_mode`:
  - `exact_event_tree` (full event-level budget available)
  - `deposition_only` (older files without event budget tree)

Important interpretation:

- For new runs, the ROOT `event` tree includes:
  - `primaryEnergy`, `depositedEnergy`, `escapedEnergy`,
  - `escapedBackEnergy` (top-surface exit), `escapedForwardEnergy` (bottom exit),
  - `escapedLateralEnergy` (side exit), and `closureEnergy`.
- This lets you check per-macro closure directly:
  `E_in ≈ E_dep + E_escape (+ closure residual)`.
- Older ROOT files fall back to deposition-only mode.
- To get `exact_event_tree`, rebuild `dnaphysics` and rerun the relevant macros
  with the updated executable.

## 5) Notes

- Use `ice_hex` / `ice_am` for ice runs, `water` for water runs.
- If you rerun with the same setup, existing files with the same basename are replaced by Geant4 run startup cleanup logic.
- For custom manual density runs, regenerate macros with `--manual-density-gcm3` so geometry/material setup in macros is explicit.
