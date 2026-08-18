# CP2K 2025.2 density-Hirshfeld force regression

This directory is an exact copy of the seven text files required by CP2K's
`tests/QS/regtest-cdft-hirshfeld-3/HeH-cdft-1.inp` regression at tag
[`v2025.2`](https://github.com/cp2k/cp2k/tree/v2025.2/tests/QS/regtest-cdft-hirshfeld-3).
The validator checks their upstream SHA-256 values before accepting a result.

CP2K matcher `M072` reads the last `FORCES| Total atomic force` value. The
upstream `TEST_FILES.toml` requires `0.1552195046628` with relative tolerance
`1e-7`. The local validator reproduces that definition exactly. The
unconstrained calculation must run first because the constrained input consumes
its wavefunction.

Run the restart-safe cluster gate from the repository root:

```bash
qsub pbs/run_cp2k_hirshfeld_force_regression.pbs
```

Passing this gate tests the exact CP2K executable's implementation of
density-Hirshfeld CDFT forces. It does not validate the chosen carbon--water
diabatic state, exchange-correlation functional, basis, or soft-collision
surface. Those are separate gates.

Method provenance: Ahart, Rosso, and Blumberger, *J. Chem. Theory Comput.* 18,
4438--4446 (2022),
[doi:10.1021/acs.jctc.2c00284](https://doi.org/10.1021/acs.jctc.2c00284).
