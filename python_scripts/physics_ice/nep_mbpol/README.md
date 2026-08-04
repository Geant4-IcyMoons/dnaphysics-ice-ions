# NEP-MB-pol ice-structure runtime

This directory contains the published NEP-MB-pol runtime needed to prepare and
evolve atomistic water-ice structures. It is separate from the Geant4 runtime:
the molecular-dynamics calculations will eventually generate projectile
scattering/recoil tables that Geant4 can read.

## Included locally

- `model/nep-mbpol.nep.txt`: published pretrained NEP4 potential for O and H.
- `upstream/GPUMD-v3.9.3`: GPUMD source version supplied by the authors.
- `upstream/03-Demo-MD`: published classical- and path-integral-MD examples.
- `downloads`: checksum-verified original Zenodo archives.
- `downloads/genice2-*.whl`: pinned ice-Ih structure generator package.
- `requirements.txt`: pinned GenIce2 version for constructing an initial ice-Ih
  hydrogen-bond network.

The 1.5 GB training/test archive is not needed to run the pretrained model and
is therefore not downloaded. It can be obtained from the provenance URL if we
later decide to retrain or independently repeat the authors' validation.

## Reproduce the download

```bash
./download_upstream.sh
```

The downloader is idempotent and verifies the MD5 values published by Zenodo
before extracting anything.

## Build GPUMD on a GPU node

GPUMD-v3.9.3 requires an NVIDIA GPU, CUDA, `make`, and a compatible C++
compiler. It cannot be compiled into its intended CUDA configuration on a Mac
without an NVIDIA GPU.

```bash
cd upstream/GPUMD-v3.9.3/src
make
```

Then run the unmodified upstream density demonstration:

```bash
cd upstream/03-Demo-MD/CMD/Density
../../../GPUMD-v3.9.3/src/gpumd
```

## Ice starting structures

NEP-MB-pol supplies energies and forces; it does not construct an ice lattice
from nothing. Install GenIce2 in a Python environment to generate a
hydrogen-disordered ice-Ih starting network:

```bash
python3 -m pip install -r requirements.txt
genice2 --rep 8 8 8 1h --format xyz > ice_ih_initial.xyz
```

The pinned GenIce2 wheel is also cached under `downloads/`; installing that
wheel still allows `pip` to obtain its Python dependencies when necessary.

The repository generator performs the GPUMD conversion and uses the shared
hexagonal-ice density constant by default:

```bash
.venv/bin/python generate_hexagonal_ice.py
```

This produces a periodic 8x8x8 ice-Ih cell containing 8,192 water molecules
(24,576 atoms), with the hexagonal c-axis along z, together with a JSON
provenance/validation record. The
structure is an initial hydrogen-disordered crystal and must still be relaxed
and equilibrated with `model/nep-mbpol.nep.txt` on a GPU node.

## Cluster workflow: low-density amorphous ice

The project constant
`ICE_AMORPHOUS_DENSITY_G_CM3 = 0.94` represents low-density amorphous ice
(LDA). Density fixes the macroscopic state but does not make a molecular
configuration amorphous. Do not obtain the amorphous target by randomly moving
atoms or by simply rescaling ice Ih. The crystalline memory must be erased in
a liquid state, followed by a controlled quench and structural validation.

The reference cooling path below follows the literature rationale of
equilibrating liquid water at 240 K and cooling it to 80 K at 10 K/ns near
ambient pressure (Giovambattista et al., 2024,
<https://doi.org/10.1038/s42004-024-01117-2>). That paper used q-TIP4P/F, not
NEP-MB-pol. Consequently, the cooling path is a literature-supported starting
protocol whose output must be validated with NEP-MB-pol; it must not be
presented as a previously published NEP-MB-pol LDA recipe.

### 1. Check out and build on the NVIDIA node

The repository branch containing this runtime can be checked out under the
cluster work directory as follows:

```bash
cd /work/yoffegid/dnaphysics-ice-ions
git fetch origin
git switch ion
git pull --ff-only

cd python_scripts/physics_ice/nep_mbpol
./download_upstream.sh
cd upstream/GPUMD-v3.9.3/src
make
./gpumd --help >/dev/null 2>&1 || true
cd ../../..
```

Load the cluster's CUDA/compiler modules before `make`. GPUMD-v3.9.3 is a
CUDA program; an Apple GPU cannot replace the NVIDIA node for this production
trajectory.

### 2. Prepare independent starting cells

Install the pinned structure generator and make at least three independent
hydrogen-disordered Ih cells. These are only starting points and will be fully
melted before the quench.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

for seed in 1000 2000 3000; do
    .venv/bin/python generate_hexagonal_ice.py \
        --rep 8 8 8 \
        --seed "${seed}" \
        --output "structures/ice_ih_8x8x8_seed${seed}_melt_start.xyz"
done
```

Do not set the starting crystal to 0.94 g/cm3. Let the NPT trajectory determine
the liquid and glass volumes. The final low-temperature density is an output
to validate against 0.94 g/cm3, rather than a correction applied afterward.

### 3. Run the melt, quench, and final hold

For each seed, create an otherwise identical run directory:

```bash
seed=1000
run_dir="runs/amorphous_seed${seed}"
mkdir -p "${run_dir}"
cp "structures/ice_ih_8x8x8_seed${seed}_melt_start.xyz" \
   "${run_dir}/model.xyz"
cp model/nep-mbpol.nep.txt "${run_dir}/nep.txt"
cd "${run_dir}"
```

Place the following in `run.in`. GPUMD uses femtoseconds for time and GPa for
pressure. At a 0.2 fs step, the 240--80 K quench contains 80 million steps,
which is 16 ns and therefore 10 K/ns. The pressure `0.0001` GPa is 0.1 MPa.

```text
potential       nep.txt
time_step       0.2
velocity        350
dump_thermo     10000
dump_exyz       5000000 0 0
dump_restart    500000

# Melt ice Ih for 1 ns at 350 K and 0.1 MPa.
ensemble        npt_ber 350 350 200 0.0001 10.0 2000
run             5000000

# Bring the liquid to 240 K over 1 ns, then hold for 1 ns.
ensemble        npt_ber 350 240 200 0.0001 10.0 2000
run             5000000
ensemble        npt_ber 240 240 200 0.0001 10.0 2000
run             5000000

# Literature-reference LDA quench: 240 K to 80 K at 10 K/ns.
ensemble        npt_ber 240 80 200 0.0001 10.0 2000
run             80000000

# Hold the resulting glass at 80 K and 0.1 MPa for 1 ns.
ensemble        npt_ber 80 80 200 0.0001 10.0 2000
run             5000000
```

Run it with the scheduler's normal single-GPU wrapper:

```bash
/absolute/path/to/GPUMD-v3.9.3/src/gpumd > gpumd.log 2>&1
```

Submit the three seeds as independent jobs. The trajectory is sequential and
cannot be divided among GPUs, but the independent seeds can run concurrently.
`restart.xyz` is overwritten every 100 ps and is the recovery point after a
wall-time interruption. Preserve the temperature reached at interruption when
constructing a continuation input; do not restart the entire cooling ramp from
240 K.

### 4. Acceptance tests

A configuration is accepted as LDA only after all of the following checks:

1. The 350 K trajectory has lost the sharp ice-Ih peaks and crystalline-order
   signature before cooling begins. If not, extend the melt stage.
2. The final O--O radial distribution function is consistent with published
   or experimental LDA data.
3. The oxygen structure factor contains diffuse amorphous features and no
   Bragg peaks.
4. A local-order metric such as tetrahedrality plus Steinhardt/bond-order or
   ice-classification analysis finds no system-spanning crystal.
5. Temperature, energy, pressure, cell volume, and density are stationary in
   the final 80 K hold, with density consistent with the project value
   0.94 g/cm3.
6. The three seeds give statistically consistent RDFs, structure factors, and
   densities. A single visually disordered snapshot is not sufficient.

If the final density is inconsistent with 0.94 g/cm3, reject or revise the
thermodynamic path. Do not force agreement by rescaling the final coordinates
without a subsequent equilibrated, validated trajectory.

### 5. Files to return from the cluster

For every seed, retain `run.in`, `gpumd.log`, `thermo.out`, the final
`restart.xyz`, and enough snapshots to reproduce the validation. Record the
GPUMD commit/version, GPU model, potential checksum, seed, time step,
thermostat/barostat settings, cooling rate, and all run lengths. Large raw
trajectories should remain in cluster storage; commit only the validated final
cells, compact validation tables/plots, and their provenance records.

Once a final cell has been accepted, it is frozen and reused for every H, He,
C, O, and S projectile calculation. It is not regenerated for each Geant4
run. Geant4 continues to use the shared 0.94 g/cm3 material density; the
atomistic cell supplies structural information for separately validated
projectile--H and projectile--O interactions.

Generate the full-width AAS paper figure (PDF plus 300-dpi PNG) with:

```bash
.venv/bin/python plot_hexagonal_ice_structure.py
```

## Current physics boundary

The pretrained file begins with `nep4 2 O H`; it models only interactions
within the water target. It does **not** yet include H, He, C, O, or S
projectiles. The next implementation stage must keep the projectile outside
the water NEP descriptors and add separately validated projectile-H and
projectile-O pair forces. Even an oxygen projectile needs a distinct role/type
from an oxygen atom belonging to the ice.

## Provenance

- Xu et al., *NEP-MB-pol: A unified machine-learned framework for fast and
  accurate prediction of water's thermodynamic and transport properties*.
- Zenodo record: <https://doi.org/10.5281/zenodo.15033656>
- GPUMD installation documentation: <https://gpumd.org/installation.html>
- GenIce2: <https://pypi.org/project/genice2/>

See `PROVENANCE.json` for exact archive names, sizes, URLs, and checksums.
