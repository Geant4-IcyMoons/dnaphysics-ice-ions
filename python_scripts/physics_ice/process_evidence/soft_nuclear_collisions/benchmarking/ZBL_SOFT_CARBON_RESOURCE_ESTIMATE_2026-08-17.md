# Carbon ZBL-soft resource estimate, 2026-08-17

## Scope

The campaign contains 1,230 independently restartable cases: 41 total carbon
energies, three numerical recoil cutoffs, and ten registered
structure/orientation combinations. Each case samples a 100 Angstrom path and
must pass the preregistered 0.5% scalar and trajectory-CDF gates.

## Measured throughput

Single-core tests used the accepted 3,000-molecule amorphous structure, the
production collision-tube/control-variate estimator, and ten trajectories per
point. Measured CPU time per 100 Angstrom trajectory was:

| recoil cutoff (eV) | 10 keV | 1 MeV | 100 MeV |
|---:|---:|---:|---:|
| 1e-4 | 0.53 s | 1.06 s | 0.56 s |
| 1e-5 | 0.83 s | 2.20 s | 1.04 s |
| 1e-6 | 1.32 s | 4.86 s | 2.00 s |

Peak resident memory was 100 MB for one worker. The production request of 256
workers and 64 GB per case therefore includes approximately a factor-of-two
memory margin. Actual multi-core efficiency must be read from PBS accounting;
the estimate assumes 70% useful parallel efficiency.

## One-week estimate

The simultaneous DKW requirement is expected to force approximately 400,000
trajectories per case even when scalar moments converge earlier. Interpolating
the measured timings over the energy grid gives approximately 210,000
core-hours for the full campaign. Ideal completion in seven days requires
about 1,250 continuously occupied cores; allowing for 70% efficiency requires
about 1,800 cores, or seven to eight 256-core nodes on average. Queue wait is
not included.

This is a computational estimate, not evidence that the ZBL soft annulus is
physically accurate. The campaign must still assess cutoff convergence,
handoff behavior, many-centre ambiguity, phase/orientation dependence, and
independent stopping/angular moments.
