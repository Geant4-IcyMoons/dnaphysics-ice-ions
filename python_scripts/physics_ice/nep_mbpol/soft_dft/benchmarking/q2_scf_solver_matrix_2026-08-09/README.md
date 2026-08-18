# Carbon q=2 unconstrained SCF solver diagnostic

These files preserve the rejected 2026-08-09 numerical-control matrix for a
restricted carbon q=2 plus H2O complex at 12 A in the oxygen-back orientation.
CDFT was disabled, every calculation used `SCF_GUESS ATOMIC`, and no restart
wavefunction was supplied. The runs were deliberately cancelled after their
diagnostic behavior was established; PBS exit status 143 records that
cancellation and is not an intrinsic CP2K crash.

No calculation converged to the required residual below `1e-7`. None is an
accepted electronic state, potential, force, or restart source. Restart
wavefunctions were intentionally discarded. Each subdirectory retains the
exact CP2K input, complete text trace, and PBS standard output needed to
reproduce or compare the solver trajectory.

| Label | PBS job | SCF method | Accepted rows | Best residual | Energy at best residual (Ha) | Final residual | Final energy (Ha) | Interpretation |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `3pnt_full_all` | 112909 | OT/CG, 3PNT, FULL_ALL, gap 0.001 Ha | 54 | 3.443e-5 | -112.8155247143 | 1.604324e-2 | -113.1527839648 | Repeated rebound and movement between energy basins |
| `gold_full_all` | 113069 | OT/CG, GOLD, FULL_ALL, gap 0.001 Ha | 6 | 1.15970e-3 | -112.8153779749 | 1.15970e-3 | -112.8153779749 | Monotonic early contraction, incomplete and unvalidated |
| `adapt_full_all` | 113070 | OT/CG, ADAPT, FULL_ALL, gap 0.001 Ha | 23 | 3.589e-5 | -112.8155246236 | 6.20225e-3 | -113.1516502339 | ADAPT overshoot after near-convergence; rejected trajectory |
| `gold_full_kinetic` | 113071 | OT/CG, GOLD, FULL_KINETIC | 6 | 8.801048e-2 | -112.5568511643 | 8.801048e-2 | -112.5568511643 | Contracted too slowly to be useful |
| `direct_p_mixing` | 113072 | diagonalization, DIRECT_P_MIXING, alpha 0.2 | 63 | 2.1865634e-1 | -113.1405217928 | 2.3051280e-1 | -113.0901818671 | Stable large-amplitude limit cycle |

All jobs used 16 CPUs and approximately 63.4 GB peak resident memory. Job
112909 ran for 1:16:09; jobs 113069--113072 ran for approximately 45 minutes.
The q=2 unconstrained restricted problem is retained only as a diagnostic. It
must not select a production solver or stand in for a charge-localized CDFT
diabatic state.
