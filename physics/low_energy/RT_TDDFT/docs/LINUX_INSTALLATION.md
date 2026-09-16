# Octopus 16.4 on Chemfarm Linux

## Source and configuration

The source submodule is pinned to
`af05332d6a12d9f2e1f0c280a28a42234c6c1f4b` (Octopus 16.4).
The local branch was advanced to `46768a8c5` with existing working edits
preserved. The RT-TDDFT Python tests passed: 13 tests.

The Linux configuration uses GCC/GFortran 13.2.0, OpenMPI 4.1.5,
OpenMP, FFTW 3.3.10, GSL 2.8, LibXC 6.2.2, and sequential MKL 2023.2
BLAS/LAPACK through the GNU Fortran interface. Spglib is built from commit
`199d970517f5bf34cfbffe310597c889c03c0e6d` (v2.3.0).
GPU backends and native-architecture optimization are disabled. Optional
CGAL discovery is disabled because the cluster CGAL installation has an
unresolved MPFR dependency; the finite-cluster controls do not require it.
The internally packaged METIS implementation is used.

Build directory: `/work/yoffegid/software/octopus-16.4-build`.
Install prefix: `$HOME/.local/opt/octopus-16.4`.
Compilation used the short PBS queue on a compute node with 4 CPUs and 8 GB; Ninja preserves
completed objects if compilation is interrupted. This allocation limits
compiler memory and accommodates the Fortran dependency graph.

The compute nodes do not provide the login node's CMake. Shared build tools
are installed in `/work/yoffegid/software/octopus-build-tools` (CMake 3.31.10,
Ninja 1.13.2, tqdm 4.70.1). Use the native CMake binaries so that invoking
the build tools does not require the Python runtime library path:

```bash
export PATH="/work/yoffegid/software/octopus-build-tools/lib/python3.11/site-packages/cmake/data/bin:/work/yoffegid/software/octopus-build-tools/bin:$PATH"
module load mkl/2023
```
Configuration uses the native binary and the system library environment:
the Anaconda library path interferes with the login node's RPM-based
`pkg-config` wrapper. Configuration therefore selects the platform-specific
`pkg-config` executable explicitly.

From the repository root, configure with:

```bash
/work/yoffegid/software/octopus-build-tools/lib/python3.11/site-packages/cmake/data/bin/cmake -S physics/low_energy/Octopus \
  -B /work/yoffegid/software/octopus-16.4-build -G Ninja \
  -DCMAKE_MAKE_PROGRAM=/work/yoffegid/software/octopus-build-tools/bin/ninja \
  -DPKG_CONFIG_EXECUTABLE=/usr/bin/x86_64-redhat-linux-gnu-pkg-config \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/opt/octopus-16.4" \
  -DCMAKE_C_COMPILER=/apps01/apps/gcc/gcc-13.2.0/bin/gcc \
  -DCMAKE_CXX_COMPILER=/apps01/apps/gcc/gcc-13.2.0/bin/g++ \
  -DCMAKE_Fortran_COMPILER=/apps01/apps/gcc/gcc-13.2.0/bin/gfortran \
  -DCPP_EXECUTABLE=/apps01/apps/gcc/gcc-13.2.0/bin/cpp \
  -DMPI_C_COMPILER=/apps01/apps/openmpi-4.1.5-gcc13/bin/mpicc \
  -DMPI_Fortran_COMPILER=/apps01/apps/openmpi-4.1.5-gcc13/bin/mpifort \
  -DOCTOPUS_MPI=ON -DOCTOPUS_OpenMP=ON -DOCTOPUS_NATIVE=OFF \
  -DBLA_VENDOR=Intel10_64lp_seq \
  -DCMAKE_PREFIX_PATH='/apps01/apps/gsl-2.8;/apps01/apps/libxc-6.2.2' \
  -DFFTW_ROOT=/apps01/apps/fftw-3.3.10 \
  -DCMAKE_DISABLE_FIND_PACKAGE_CGAL=ON
```

On the allocated compute node:

```bash
cmake --build /work/yoffegid/software/octopus-16.4-build --parallel 4
cmake --install /work/yoffegid/software/octopus-16.4-build
```

## Execution checks (2026-09-15)

Installation and the selected execution checks passed on compute node
`cfm048`. The executable reports `octopus 16.4 (git commit af05332d6a)`;
its SHA-256 is
`596531bfe3fe9cc86355698843d1effd86dbbc2d6434045338f7e4ce16e19488`.
Activation was also checked in a shell with `LD_LIBRARY_PATH` initially
unset; the version command succeeds and `ldd` reports no missing libraries.

The cluster FFTW libraries require Intel compiler runtimes as well as the
GNU runtime used by Octopus. Both GNU and Intel OpenMP libraries are linked.
The one- and two-thread controls below check this configuration, but do not
qualify arbitrary thread counts or mixed-node execution. The upstream linker
reported an executable-stack requirement from `mixing_preconditioner.F90.o`;
no source patch or linker workaround was applied.

| Check | Result |
| --- | --- |
| RT-TDDFT Python tests | 13 passed |
| `finite_systems_3d/01-carbon_atom` | Passed with two MPI ranks |
| `real_time/03-td_self_consistent` | Passed with two MPI ranks |
| `real_time/19-td_move_ions` | Passed with two MPI ranks |
| Amorphous-derived one-water GS and ten TD steps | Completed with one and two OpenMP threads |
| Hexagonal-derived one-water GS and ten TD steps | Completed with one and two OpenMP threads |

The ice controls use a 0.4 Å grid, 4 Å padding, and a 0.02 atomic-unit time
step. SCF convergence was reported after 16 iterations (amorphous-derived)
and 15 iterations (hexagonal-derived), accompanied by a warning that some
states were not fully converged. Initial total energies differed between
one and two threads by 1.08e-12 and 1.59e-12 eV, respectively. All four TD
outputs contain steps 0 through 10 and finite numbers. The default energy
output does not independently recompute every component at every step;
repeated printed totals are not an energy-conservation validation. The
missing TD restart warning is expected for the first propagation, which
reads the GS restart. These are installation checks, not a grid, box,
time-step, or collision convergence study.

The first MPI test attempt could not launch because PBS supplied one MPI
slot. The successful rerun requested `select=1:ncpus=4:mpiprocs=4:mem=8gb`
and set `OCT_TEST_MPI_NPROCS=2`, `OMP_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. The ice controls separately used
`OMP_NUM_THREADS=1` and `2`. Upstream tests can be repeated with:

```bash
ctest --test-dir /work/yoffegid/software/octopus-16.4-build \
  --output-on-failure \
  -R 'finite_systems_3d/01-carbon_atom|real_time/03-td_self_consistent|real_time/19-td_move_ions'
```

The successful build job was `245293.pbs02` (8 min 7 s); the final check job
was `245372.pbs02` (44 s). Both exited with status zero. Intermediate objects,
build logs, scheduler scripts, and disposable restart/density outputs were
removed after review. The installed binaries, pinned source, shared build
tools, and input bundles remain. Selected numerical outputs and execution
logs are retained locally in `runs/linux_installation_checks.tar.gz`.
Archive SHA-256: `e66a084e942da8930d2e0cc630b12d8cfd58c1619e465e33692a71a5de772f65`.
Reconfigure and rebuild before repeating the upstream tests.

Activate the installed solver with:

```bash
source physics/low_energy/RT_TDDFT/activate_octopus.sh
octopus --version
```

The activation script selects this cluster's matching MPI and runtime libraries.
Run scientific calculations on compute nodes with explicit MPI and OpenMP
counts that fit the allocation; keep BLAS threads at one when distributing
work through MPI. The collision fragment remains incomplete.
