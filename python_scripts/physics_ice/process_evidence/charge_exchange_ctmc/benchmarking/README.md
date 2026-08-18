# CTMC charge-exchange benchmarks

`benchmark_carbon_charge_exchange_ctmc.py` constructs the observables defined
in figures 12--14 and equation 23 of Liamsuwan and Nikjoo, *Phys. Med. Biol.*
**58**, 641--672 (2013), <https://doi.org/10.1088/0031-9155/58/3/641>.

It compares like with like: pure single capture (`SC`), pure single loss
(`SL`), and equilibrium fractions from the adjacent total decrease/increase
rates. It never digitizes or fits unavailable paper data silently. Boundary,
energy-drift, probability-conservation, channel-identity and archive-integrity
checks are emitted with the presentation plot.

The presentation plot converts the calculated energy per nucleon to total
C-12 kinetic energy and displays the common Geant4 model interval from 10 keV
to 100 MeV. Paper/data residuals are calculated only where the computed curve
has support; sparse diagnostic runs never extend their endpoint values into
unsampled energies. A production CTMC grid spans 1--10,000 keV/u
(12 keV--120 MeV total). The figure-12 separation factors are written
compactly as $10^{q-5}\sigma_{\rm SC}$ on the capture ordinate.

The final benchmark also requires `C3_100keVpu.zip`, supplied by the user as
CTMC81 output received from Liamsuwan. Its six tables provide direct primitive
probabilities versus impact parameter for C3+ + H2O at 100 keV/u: target
ionization and capture for bound-electron orbitals L1--L5, and projectile
electron loss. The exact archive and member SHA-256 hashes, header settings,
grid integrity, and printed probability normalization are recorded. The
calculated primitive curves are compared only over the reference domains and
written to `carbon_ctmc_formal_100keV_u_benchmark.png`; the digitized
figures-12--14 comparison remains a separate output.

The archive contains no embedded citation, transfer record, random seed or
license metadata. Its target and projectile-loss calculations use different
initial separations, integration intervals and trajectory counts.
Consequently, the benchmark is mandatory and its residuals are reported
without fitting. Archive integrity and complete evaluation of all primitive
curves are release gates.

## Parameter-matched CTMC81 reproduction

`run_ctmc81_c3_reproduction.py` is a separate validation calculation. It does
not overwrite the completed production grid. It reads every disclosed value
from the reviewed archive and reproduces the following six calculations:

| channel | impact grid | trajectories per point | initial z (a.u.) | minimum tEND (a.u.) |
|---|---:|---:|---:|---:|
| L1 | 100 points, 0--10 a.u. | 20,000 | 1,000 | 1,000 |
| L2 | 100 points, 0--10 a.u. | 20,000 | 1,000 | 1,000 |
| L3 | 50 points, 0--5 a.u. | 20,000 | 1,000 | 1,000 |
| L4 | 50 points, 0--5 a.u. | 20,000 | 1,000 | 1,000 |
| L5 | 10 points, 0--1 a.u. | 20,000 | 1,000 | 1,000 |
| projectile loss | 100 points, 0--10 a.u. | 10,000 | 10,000 | 6,000 |

The energy is 100 keV/u, charge is C3+, mass is 12 u, and the recorded CTMC81
speed is 2.00055347542277 a.u. `null` endpoints are retained rather than
silently reassigned. The initial electron follows the Olson--Salop angular
construction explicitly cited by section 2.1 of Liamsuwan and Nikjoo (2013):
its position direction is isotropic and its momentum is uniform in the plane
tangent to that position. This historical prescription is distinct from an
independent-isotropic interpretation of equation (4), which remains only as a
separately signed sensitivity mode. The paper describes converged time
intervals of at least 1000 a.u. at intermediate energies; the reference files
disclose the values above and a 1000-a.u. terminal target--projectile
distance. A trajectory ends only after both the listed `tEND` and distance
gate are met while the nuclei recede.

For projectile loss, this reproduction assigns the disclosed impact parameter
and velocity to the screened projectile nucleus, as prescribed in section 3.2,
and adds the bound electron relative to that nucleus after the Appendix-B
projectile--target switch. The production engine retains
`projectile_ion_com_balanced` as the separately signed stationary-
microcanonical candidate; its projectile-ion centre of mass follows the
nominal beam path and it must not be mixed with the CTMC81 reproduction mode.

The default run contains two independent replicas. Each replica has exactly
the reference trajectory count, while their combined estimator has smaller
Monte Carlo uncertainty. DOP853 uses `rtol=1e-11`, `atol=1e-13`, tighter retry
levels, and an independent relative-energy-drift gate of `1e-4`. CTMC81's
recorded scalar `TOL=1e-6` is implementation-specific and is not falsely
identified with DOP853's error norm. Instead, selected identical phases are
rerun at tenfold tighter tolerances and must retain the same event class.

Pointwise comparison uses the combined variance of the independent CTMC81 and
new binomial samples, including the reference table's four-decimal rounding.
A Bonferroni simultaneous 95% band is declared before inspecting the result.
This is a reproducibility test, not a 0.5% physical-accuracy claim and not a
fitted correction.

Submit the restart-safe 256-core calculation with:

```bash
qsub pbs/run_ctmc81_c3_reproduction.pbs
```

The default calculation comprises 14.4 million trajectories: two exact
7.2-million-trajectory replicas. Atomic checkpoints are written under
`benchmarking/runs/ctmc81_c3_100keV_u_olson_salop_reproduction/`. The final
outputs are a pointwise CSV, a six-panel probability plot and a
machine-readable summary. The earlier independent-isotropic result is retained
in its separate directory as failed sensitivity evidence and cannot satisfy
the production-release gate.
