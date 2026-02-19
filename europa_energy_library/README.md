# Europa Energy Library Commands

## 1) Generate Europa macros and tables

From repository root:

```bash
python3 geant4_projects/dnaphysics-ice/python_scripts/generate_europa_energy_library.py \
  --out-dir geant4_projects/dnaphysics-ice/europa_energy_library \
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

## 2) Loop over all generated macros and produce ROOT files

```bash
cd geant4_projects/dnaphysics-ice/europa_energy_library
DNA_ROOT_MAX_MB=2048 ./run_per_energy.sh ../build/dnaphysics 10 ice_am
```

