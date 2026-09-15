# H+ collision feasibility pilot

For the new isolated-water collision and capture workflow, use
[BENCHMARK.md](BENCHMARK.md). The older runner below reports density diagnostics.

The `hplus` runner propagates one prescribed 1 keV proton past a frozen water
molecule or cluster extracted, without relaxation, from the accepted ice registry. It
prepares neutral water first, then reads its orbitals on the identical mesh
with the proton added only for time propagation. This addresses initial
fragment charge preparation; finite-separation polarization remains unchecked.
The extracted target retains ice geometry but has no embedding.

From the repository root, with NumPy and SciPy available:

```bash
python -m physics.low_energy.RT_TDDFT.hplus \
  --phase amorphous_lda_80k --center 0 --energy-ev 1000 \
  --dt 0.04 0.02 --workers 2 --threads 2 \
  --output physics/low_energy/RT_TDDFT/runs/hplus_first
```

Each independent time-step case uses two OpenMP threads and one OpenBLAS
thread. Within a case, ground-state preparation precedes propagation. The
local Octopus installation has OpenMP but no MPI. Two concurrent cases use
four OpenMP threads; this two-case pilot has no work for ten concurrent workers.
Future impact parameters and source molecules can be distributed independently.

The runner refuses an existing output directory, retains input and solver
hashes, checks ground-state convergence and process exit codes, and records
real-space density integrals. Post-processing checks that the initial TD
density equals the neutral-water density and that the target remains fixed
while the projectile follows its prescribed path. Cube text precision limits
integral accuracy. The cube parser expects scalar densities on a Bohr grid.

Outputs include electron populations in 1, 1.5, and 2 angstrom spheres moving
with the projectile, and total norm removed from the grid. These are capture
and emission diagnostics. Neither is an exclusive event probability. Target
excitation is not yet separated; total energy alone cannot supply it when
capture, emission, and externally prescribed nuclear motion coexist.

The initial 0.4 angstrom grid, LDA standard pseudopotentials, absorbing boundary,
finite flight distance, and fixed straight trajectory are feasibility settings.
Time-step agreement cannot establish spatial convergence or physical accuracy.
The result records any geometric overlap of the largest analysis sphere with
the absorber at the final step. Projectile-bound-state projections, target
channel analysis, longer flight and target controls, grid/box/absorber studies,
and literature comparisons remain required before reporting channel yields.
No C/O/S collision or transport integration is implemented by this pilot.

## Ice phases and system size

Use `--molecules 5 --transverse-half 8 --dt 0.04` for the initial five-water
cluster calculation. Select `--phase amorphous_lda_80k` or
`--phase hexagonal_ih_100k`, with separate output directories. These two phase
cases can run concurrently. They share numerical settings and cluster size;
a single encounter per phase cannot establish a phase-averaged difference.

The full 8,192-water hexagonal cell has 65,536 valence electrons, or 32,768
doubly occupied orbitals. Its approximately 71.945 by 62.3062 by 58.566 angstrom
cell contains roughly 4.1 million grid points at 0.4 angstrom spacing. One
complex128 orbital array would require about 2.15 TB, before solver buffers.
This is a storage estimate, not a full-system performance measurement. Keeping
all molecules in the physical model requires a smaller quantum region coupled
to the surrounding ice. That embedding and mechanical coupling are future
work, not a capability of the present finite-cluster runner.
