# Local Octopus installation

Source: official [Octopus repository](https://gitlab.com/octopus-code/octopus),
release tag `16.4`, commit `af05332d6a12d9f2e1f0c280a28a42234c6c1f4b`.
The version-16 download page still linked 16.3 when checked; the official
repository supplied the newer 16.4 release tag.

The source is tracked as a Git submodule at `physics/low_energy/Octopus`.
Its `install` symlink points to the local binary prefix and is excluded from
the submodule status through its local Git exclusion file. GitHub records the
source commit; compiled binaries and machine-specific links are not versioned.
After cloning this repository, run `git submodule update --init` to obtain the
pinned source, then build/install it locally.

Install prefix: `/Users/yoffegid/.local/opt/octopus-16.4` (Apple Silicon macOS).
Activate from the repository root:

```bash
source physics/low_energy/RT_TDDFT/activate_octopus.sh
octopus --version
```

The build uses GCC 16.1 (C/C++) and GFortran 13.4, CMake 4.1, Ninja, OpenMP,
OpenBLAS, FFTW, GSL, and METIS. Ninja, METIS, and GCC 13 were added with Homebrew;
other system dependencies were already present. MPI and GPU backends are disabled. METIS is not used by this non-MPI build.

## Reproduction

Follow the [official macOS CMake guide](https://octopus-code.org/documentation/16/manual/installation/macos_cmake_installation/).
GCC 16 failed internally compiling `ions.F90`, both with and without OpenMP.
GFortran 13 successfully compiled that section. The installed OpenMPI Fortran
module was incompatible with GFortran 13, so MPI is disabled. The C/C++
compiler remains GCC 16; no upstream source patches were required. Build outside the source repository and install to a new prefix:

```bash
git clone --depth 1 --branch 16.4 https://gitlab.com/octopus-code/octopus.git octopus-source
cmake -S octopus-source -B octopus-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/opt/octopus-16.4" \
  -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-16 \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-16 \
  -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran-13 \
  -DCPP_EXECUTABLE=/opt/homebrew/bin/cpp-16 \
  -DCMAKE_PREFIX_PATH='/opt/homebrew/opt/openblas;/opt/homebrew' \
  -DBLA_VENDOR=OpenBLAS -DOCTOPUS_MPI=OFF -DOCTOPUS_OpenMP=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_SPARSKIT=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_Libxc=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build octopus-build --parallel 10
cmake --install octopus-build
```

The CMake policy setting resolves a configure failure in the fetched LibXC
project under CMake 4. SPARSKIT discovery is disabled to avoid the macOS SDK
library-name conflict described upstream. No Octopus source patches are used.

CMake fetched LibXC commit `c5bcf344b55d8939053b998d1b73cedb7c2b1bfb`
and Spglib commit `199d970517f5bf34cfbffe310597c889c03c0e6d`. The LibXC
upstream reference is a branch; for exact reproduction check out these commits
separately and pass their paths as `FETCHCONTENT_SOURCE_DIR_LIBXC` and
`FETCHCONTENT_SOURCE_DIR_SPGLIB` at configuration. The build enables LibXC
first-, second-, and third-order derivatives.

## Installation verification (2026-09-15)

`octopus --version` reports `octopus 16.4 (git commit af05332)`.
The executable SHA-256 is
`91cbb9f94447df8c3cc6190483968efb7601c1e124a3c939d3cd471652c5a691`.
The binary links the Homebrew GCC runtime libraries; keep those dependencies
installed. It was built and tested on this Mac, not tested for portability to
older macOS versions. The linker emitted deployment-target warnings (16 versus
26) from the mixed Homebrew toolchain; the installed binary executed normally.

Upstream checks were run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`:

```bash
ctest --test-dir octopus-build --output-on-failure --parallel 3 \
  -R 'finite_systems_3d/01-carbon_atom|real_time/03-td_self_consistent|real_time/19-td_move_ions'
```

- `finite_systems_3d/01-carbon_atom`: passed.
- `real_time/03-td_self_consistent`: passed.
- `real_time/19-td_move_ions`: execution completed, but one restart-energy
  assertion failed. Calculated -29.644745389617995 Hartree versus reference
  -29.64474538961837: absolute difference 3.76587649952853e-13 Hartree,
  tolerance 2.96e-13. All other assertions, including constant-velocity motion,
  passed. The difference is consistent with floating-point sensitivity, but
  this remains a failed upstream test; its tolerance was not changed.

The installed executable also completed GS and ten TD steps for one molecule
extracted from `amorphous_lda_80k`, replica 0, molecule 0. Input preparation used
`--molecules 1 --spacing 0.4 --padding 4 --dt-au 0.02 --steps 10
--energy-ev 1000 --impact 1 --separation 15`, with two OpenMP threads and one
OpenBLAS thread. The GS SCF converged in 16 iterations but warned that some
states were not fully converged. TD read the GS restart and completed; its
missing-TD-restart warning is expected for this first propagation. This was
an installation/input compatibility check, not a convergence study or collision.

Two generator errors discovered by the real executable were corrected:
`UnitsInput` is obsolete (input now uses default atomic units and explicit
`*angstrom` factors), and `Output` requires a block. Both retained example
bundles were regenerated. Intermediate builds and test outputs were removed;
the pinned source, build recipe, installed software, and this check record remain.
