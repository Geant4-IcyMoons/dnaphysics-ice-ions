# CP2K constrained-DFT reference benchmarks

These benchmarks test the CP2K executable and the documented CDFT execution
sequence before any ion--water result is accepted. They are software and
algorithm benchmarks, not validation of a C--H2O interaction surface.

The independent GPAW q=1 diagnostics are documented in
[`gpaw_prepared_q1_2026-08-09/`](gpaw_prepared_q1_2026-08-09/) and the final
fixed-multiplier decision gate in
[`gpaw_q1_fixed_lambda_2026-08-10/`](gpaw_q1_fixed_lambda_2026-08-10/).
The latter rejects the tested GPAW construction for production and identifies
the OpenMolcas RASSCF/RASSI multistate pilot as the next method test.

The inputs are copied without scientific modification from the official
[`cp2k/cp2k-examples`](https://github.com/cp2k/cp2k-examples) repository at
commit `d71583a411a0a7d8557e94a24b87a3ceb87ec34e` (2026-04-26):

- `cdft/hirshfeld`: Zn2+ at 5 A, ordinary DFT followed by two restarted
  density-Hirshfeld CDFT states and mixed-CDFT coupling;
- `cdft/water`: the water-dimer charge-transfer-energy example, including
  ordinary DFT, restarted CDFT, and fragment-density variants.

The PBS wrapper changes only the paths to the CP2K executable and MPI launcher.
The examples retain their upstream geometries, Hamiltonian, basis,
pseudopotentials, grids, solvers, convergence thresholds, and analysis.
It uses eight MPI ranks because the upstream shell driver tests its serial case
with the regular expression `[[ $ncores =~ 1 ]]`; values such as 16 therefore
incorrectly select the unavailable serial executable `cp2k.sopt`. This launcher
workaround does not alter a CP2K input or numerical setting.

CP2K removed the `MAP_CONSISTENT` keyword after making consistent mapping
unconditional. CP2K developer Jurg Hutter states that it had long been the
default and was removed because there was no longer a reason to allow it to be
disabled. The CP2K 2025.2 compatibility copy therefore removes only this
rejected, behaviorally redundant keyword; the unmodified upstream copy and
its checksums remain preserved beside it. See the official CP2K discussion:
<https://groups.google.com/g/cp2k/c/sawps-ymH6s/m/nakOpoLLBQAJ>.

The sequence and numerical settings are documented in the official
[CP2K CDFT tutorial](https://manual.cp2k.org/cp2k-2025_2-branch/methods/dft/constrained.html).
The underlying implementation is described by Holmberg and Laasonen,
J. Chem. Theory Comput. 13, 587--601 (2017),
[doi:10.1021/acs.jctc.6b01085](https://doi.org/10.1021/acs.jctc.6b01085).
Density-Hirshfeld CDFT forces and reliability diagnostics are analyzed by
Ahart, Rosso, and Blumberger, J. Chem. Theory Comput. 18, 4438--4446 (2022),
[doi:10.1021/acs.jctc.2c00284](https://doi.org/10.1021/acs.jctc.2c00284).

## CP2K 2025.2 HeH force prerequisite

The directory
[`cp2k_2025_2_hirshfeld_force/`](cp2k_2025_2_hirshfeld_force/) preserves the
six input/include files from the exact installed image path
`/opt/cp2k/tests/QS/regtest-cdft-hirshfeld-3/`. Their SHA-256 checksums were
compared byte-for-byte with that image on 2026-08-09. The official
[`TEST_FILES.toml`](https://github.com/cp2k/cp2k/blob/v2025.2/tests/QS/regtest-cdft-hirshfeld-3/TEST_FILES.toml)
requires this order:

1. `HeH-noconstraint.inp`, which writes `HeH-noconstraint-1_0.wfn`;
2. `HeH-cdft-1.inp`, which restarts from that wavefunction.

Both use `RUN_TYPE ENERGY_FORCE`. The constrained input uses density-Hirshfeld
CDFT with target `1.0`, strength `0.186894372937` Ha, and active CDFT outer
`MAX_SCF 50`. Its otherwise unused top-level `@SET MAX_SCF 0` does not alter
the hard-coded inner or outer limits in the included files. Matcher `M072`
extracts column 5 of
`FORCES| Total atomic force`; the registered values are
`0.1450972684448` for the unconstrained precursor and `0.1552195046628` for
the CDFT calculation, each at tolerance `1e-7`.

Reproducing those two official results with the exact CP2K image is a software
prerequisite before any density-Hirshfeld CDFT force is interpreted. This gate
is necessary because the CP2K 2025.2
[`CDFT` input page](https://manual.cp2k.org/cp2k-2025_2-branch/CP2K_INPUT/FORCE_EVAL/DFT/QS/CDFT.html)
still labels Hirshfeld CDFT as partial with no forces, whereas the tagged test
suite contains this energy-and-force regression. A pass establishes only that
the installed image reproduces the upstream HeH regression; it is not a
validation of carbon--water forces or of a soft-collision potential.

Run the exact executable regression and create its checksum-bound
`validation.json` with:

```bash
qsub pbs/run_cp2k_hirshfeld_force_regression.pbs
```

Passing these examples establishes that the installed CP2K CDFT and restart
machinery reproduce the official workflow. It does not establish that PBE,
the chosen diabatic constraint, or CDFT itself is quantitatively accurate for
highly charged carbon approaching water. That target requires separate basis,
grid, functional, state-localization, asymptotic, and reference-data tests.

Run a staged official example with:

```bash
qsub -v "REFERENCE_CASE=hirshfeld,RUN_DIR=/absolute/staged/hirshfeld,REPO_ROOT=/gpfs01/work/yoffegid/dnaphysics-ice-ions" \
  pbs/run_cp2k_cdft_reference_benchmark.pbs
```
