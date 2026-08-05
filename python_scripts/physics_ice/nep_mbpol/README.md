# NEP-MB-pol ice-structure runtime

This directory contains the published NEP-MB-pol runtime needed to prepare and
evolve atomistic water-ice structures. It is separate from the Geant4 runtime:
accepted snapshots now feed the retained-domain NLH hard-trajectory driver,
whose validated outputs will subsequently supply Geant4 scattering/recoil
tables.

The collision infrastructure is under `bca/`. It validates and registers
accepted equilibrated snapshots, generates restartable NLH H/O binary-
collision kernels, reads them with checksum and interpolation-contract checks,
and sequences hard primary-projectile encounters through the periodic atomic
coordinates. See `bca/README.md`: these are phase-resolved hard-event samples,
not yet complete elastic cross sections or Geant4 runtime tables.

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

## Cluster workflow: candidate low-density amorphous ice

The project constant
`ICE_AMORPHOUS_DENSITY_G_CM3 = 0.94` is the experimental reference density for
low-density amorphous ice (LDA), not a volume constraint for this preparation.
Every stage is NPT at 0.1 MPa, so the cell volume and density evolve. The final
density is a prediction to validate, together with the structure; it must not
be imposed by rescaling. Do not obtain the amorphous target by randomly moving
atoms or by simply rescaling ice Ih. The crystalline memory must be erased in
a liquid state, followed by a controlled quench and structural validation.

The 240--80 K cooling segment follows Eltareb, Lopez, and Giovambattista
(Commun. Chem. 7, 36, 2024,
<https://doi.org/10.1038/s42004-024-01117-2>): three independent equilibrated
q-TIP4P/F liquids at 240 K were cooled isobarically at 10 K/ns using OpenMM, a
PILE thermostat, and a Monte Carlo barostat. This project instead uses
NEP-MB-pol and GPUMD's `npt_ber` integrator. Its added 350 K melt, 350--240 K
ramp, 240 K hold, and final 80 K hold make the complete 20 ns trajectory an
adapted protocol. Its outputs are candidate LDA configurations until all
acceptance tests below pass; this must not be presented as a published
NEP-MB-pol LDA recipe.

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
The three dump commands are repeated because GPUMD defines them as
non-propagating: each applies only to the next `run` block.

```text
potential       nep.txt
time_step       0.2
# Use the replica's recorded structure seed here (1000, 2000, or 3000).
velocity        350 seed 1000

# Melt ice Ih for 1 ns at 350 K and 0.1 MPa.
ensemble        npt_ber 350 350 200 0.0001 10.0 2000
dump_thermo     10000
dump_exyz       5000000 0 0
dump_restart    500000
run             5000000

# Bring the liquid to 240 K over 1 ns, then hold for 1 ns.
ensemble        npt_ber 350 240 200 0.0001 10.0 2000
dump_thermo     10000
dump_exyz       5000000 0 0
dump_restart    500000
run             5000000
ensemble        npt_ber 240 240 200 0.0001 10.0 2000
dump_thermo     10000
dump_exyz       5000000 0 0
dump_restart    500000
run             5000000

# Literature-derived cooling segment: 240 K to 80 K at 10 K/ns.
ensemble        npt_ber 240 80 200 0.0001 10.0 2000
dump_thermo     10000
dump_exyz       5000000 0 0
dump_restart    500000
run             80000000

# Hold the resulting glass at 80 K and 0.1 MPa for 1 ns.
ensemble        npt_ber 80 80 200 0.0001 10.0 2000
dump_thermo     10000
dump_exyz       5000000 0 0
dump_restart    500000
run             5000000
```

Run it with the scheduler's normal single-GPU wrapper:

```bash
/absolute/path/to/GPUMD-v3.9.3/src/gpumd > gpumd.log 2>&1
```

Submit the three seeds as independent jobs. GPUMD supports distributing one
trajectory over several GPUs, but its documentation recommends more than about
100,000 atoms per GPU for good efficiency. This 24,576-atom cell should
therefore use one GPU per seed, with independent seeds running concurrently.
`restart.xyz` is overwritten every 100 ps and is the recovery point after a
wall-time interruption. Preserve the temperature reached at interruption when
constructing a continuation input; do not restart the entire cooling ramp from
240 K.

### PBS launcher on Chemfarm

The repository PBS script requests one 48 GB GPU, four CPU cores, and 8 GB of
host RAM per independent seed. It compiles an architecture-matched GPUMD binary
in compute-node scratch, so compilation does not run on the login node. Submit
all three documented seeds as an array:

```bash
cd /work/yoffegid/dnaphysics-ice-ions
qsub -J 0-2 pbs/run_nep_mbpol_amorphous.pbs
```

The array indices map to structure and velocity seeds 1000, 2000, and 3000;
the launcher renders and records a distinct `velocity 350 seed N` command in
each replica's `run.in`. The `ps` private GPU node
has eight GPUs, so all three can run concurrently on one node when it is
available. Use one trajectory per GPU: the 24,576-atom cell is below GPUMD's
roughly 100,000-atoms-per-GPU threshold for efficient multi-GPU domain
decomposition.

The private launcher requests 500 hours. For the public GPU queue, whose
current maximum is 120 hours, override the queue and wall time at submission:

```bash
qsub -q gpuq -l walltime=120:00:00 -J 0-2 \
    pbs/run_nep_mbpol_amorphous.pbs
```

Eight GB is a conservative initial host-memory request for this cell and also
covers compilation. After the first completed or deliberately short benchmark,
inspect `resources_used.mem` with `qstat -fx JOB_ID`; reduce it to 4 GB only if
the measured peak leaves a comfortable margin. PBS `mem` is host RAM, not the
48 GB memory attached to the requested GPU.

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

## NLH short-range projectile interactions

The published pair-specific Nordlund--Lehtola--Hobler potentials required for
H, He, C, O, and S projectiles against target H and O are included under
`nlh/`. The evaluator supplies the screening function, pair energy, energy
derivative, and radial force, and is regression-tested against the authors'
reference implementation. See `nlh/README.md` for the formula, validity range,
provenance, and the boundary between this completed pair-potential layer and
the remaining species-aware trajectory coupling.

Generate the full-width, 300-dpi PNG paper figure with:

```bash
.venv/bin/python plot_hexagonal_ice_structure.py
```

To render an equilibrated GPUMD restart without replacing the initial-cell
figure, select the structure and a distinct output stem:

```bash
.venv/bin/python plot_hexagonal_ice_structure.py \
    --structure runs/hexagonal_ih_80K_seed1000/restart.xyz \
    --output-stem ../output/hexagonal_ice_equilibrated_80K_seed1000
```

## Classical 80 K ice-Ih equilibration

Hexagonal ice uses a separate preparation from the amorphous melt--quench.
The same three GenIce2 cells are relaxed at fixed cell with GPUMD's FIRE
minimizer and then evolved for 1 ns of equilibration plus 1 ns of sampling at
80 K and 0.1 MPa. Orthorhombic NPT control lets the two basal dimensions and
the c-axis dimension respond independently. This is a project preparation
protocol, not a published NEP-MB-pol ice-Ih recipe.

Structure and velocity seeds are explicitly paired as 1000, 2000, and 3000.
The trajectory snapshots include velocities and are written every 100 ps;
thermodynamic data are written every 2 ps and the restart is refreshed every
100 ps. Run the three independent replicas concurrently with one GPU each:

```bash
cd /work/yoffegid/dnaphysics-ice-ions
qsub -J 0-2 pbs/run_nep_mbpol_hexagonal.pbs
```

On the public GPU queue:

```bash
qsub -q gpuq -l walltime=48:00:00 -J 0-2 \
    pbs/run_nep_mbpol_hexagonal.pbs
```

The job creates `DYNAMICS_COMPLETED` and `VALIDATION_PENDING`; completion of
the numerical trajectory does not itself accept the model. Before recoil use,
check stationarity of temperature, energy, pressure, volume, density, and cell
dimensions over the sampling block; verify the expected ice-Ih O--O RDF and
Bragg pattern; confirm that proton disorder and the hydrogen-bond network are
valid; and compare the three replicas. Classical NEP-MB-pol dynamics at 80 K
does not explicitly represent nuclear quantum effects, which must remain a
stated limitation of this preparation.

## Bulk ice Ih at the experimental 100 K density

For the 100 K bulk-space-ice target, use a fully periodic cell with no vacuum
gap.  The experimental cell is defined from the corrected H2O ice-Ih
polynomial coefficients in Table 1 of Rottger et al. (2012),
<https://doi.org/10.1107/S0108768111046908>.  At 100 K their unit-cell-volume
fit gives 128.188109 A3 and hence 0.933474297 g/cm3 for four H2O molecules.
Their independent lattice fits give `a = 4.49648151 A` and
`c = 7.32062320 A`.  Because independently fitted polynomials are not exactly
geometrically consistent, the preparation enforces the volume fit and retains
the fitted `c/a` ratio.  The resulting 24,576-atom orthorhombic box is
71.94499804 x 62.30619598 x 58.56603887 A.

Each completed 80 K replica supplies a relaxed ice-Ih configuration.  The
preparation script preserves its fractional coordinates while mapping it into
the experimental 100 K cell, discards the old velocities, and records the
source checksum and all paper coefficients in JSON.  GPUMD then performs a
fixed-cell FIRE relaxation, 1 ns of NVT equilibration, and 1 ns of NVT
sampling at 100 K with an explicit velocity seed.  NVT preserves the
experimental density; the resulting pressure is a diagnostic of the
NEP-MB-pol/experimental-cell mismatch, not a reason to resize the cell.

Submit the three replicas on one GPU each without altering the completed 80 K
runs:

```bash
cd /work/yoffegid/dnaphysics-ice-ions
qsub -q gpuq -N nep_ih_100K -l walltime=48:00:00 \
    -v HEXAGONAL_PROTOCOL=100K_EXPERIMENTAL -J 0-2 \
    pbs/run_nep_mbpol_hexagonal.pbs
```

Outputs are written to
`runs/hexagonal_ih_100K_experimental_seed{1000,2000,3000}`.  Each directory
contains `experimental_cell.json`, which makes the density transformation
reproducible.  As for the 80 K preparation, `DYNAMICS_COMPLETED` means only
that GPUMD finished.  Accept the structures for recoil work only after the
sampling block has stable temperature, energy, and stress; ice-Ih RDF and
Bragg order are retained; and the three replicas agree.  The fixed periodic
cell makes the density exact by construction.

The three compact completed final snapshots and their checksummed run summary
are versioned under
`structures/hexagonal_ih_100K_experimental/`. Full trajectories and scheduler
logs remain unversioned. The manifest retains `validation_pending` until the
structural acceptance tests above have been completed.

## Current physics boundary

The pretrained file begins with `nep4 2 O H`; it models only interactions
within the water target. Projectiles therefore remain separate from the water
NEP descriptors. `simulate_nlh_hard_collisions.py` now links C, O, and S
projectiles to target H/O atoms through the adaptive NLH hard kernels while
retaining that distinction, including for an oxygen projectile.

This driver intentionally holds the lattice fixed and returns target recoils
as recorded secondaries. It does not reinsert them, evolve radiation damage,
or supply the missing soft distant interaction. Those additions require
separate validation and must not be inferred from the existence of the
structure-aware hard driver.

Production trajectory sampling is convergence-controlled independently of the
adaptive collision-kernel mesh. With no fixed `--trajectories` argument, the
driver adds deterministic, restartable batches until asymptotic 95%
simultaneous confidence half-widths for the hard rate, nuclear stopping, and
first energy/angular moments are all at most 0.5%, or exits nonzero at the
configured limit. See `bca/README.md` for the statistical contract and its
explicit exclusion of distribution-tail convergence.

## Provenance

- Xu et al., *NEP-MB-pol: A unified machine-learned framework for fast and
  accurate prediction of water's thermodynamic and transport properties*.
- Zenodo record: <https://doi.org/10.5281/zenodo.15033656>
- GPUMD installation documentation: <https://gpumd.org/installation.html>
- GenIce2: <https://pypi.org/project/genice2/>
- Rottger et al., corrected H2O and D2O ice-Ih lattice polynomials:
  <https://doi.org/10.1107/S0108768111046908>
- Nordlund, Lehtola, and Hobler, *Phys. Rev. A* **111**, 032818 (2025):
  <https://doi.org/10.1103/PhysRevA.111.032818>
- Corrected NLH open data: <https://doi.org/10.5281/zenodo.17302337>

See `PROVENANCE.json` for exact archive names, sizes, URLs, and checksums.
