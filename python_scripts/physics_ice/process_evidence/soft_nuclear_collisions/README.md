# Carbon soft-collision electronic-structure investigation

## Final status

The carbon fixed-charge soft-collision project is closed as **physical
validation failed**. None of the CP2K, GPAW or OpenMolcas carbon--water results
described here is approved for a Geant4 table, stopping model or production
potential.

This is a failure of the tested computational routes, not evidence that soft
elastic collisions are absent. The independent production work may retain:

- NLH discrete hard nuclear collisions within the validated NLH domain; and
- CTMC charge exchange within its separately validated energy domain.

Those processes do not supply the unresolved long-range/soft elastic
contribution. Production documentation must identify that missing component;
it must not silently set it to zero.

## Diagnostic universal-ZBL replacement

The repository now contains a validation-pending universal-ZBL screened
binary-collision baseline in `zbl/` and the corresponding
`G4DNAZBLFullElastic` Geant4 process in the repository-level `include/` and
`src/` build directories. It supports H, He, C, O and S projectiles against
the H and O nuclei of water ice from 1 keV to 100 MeV **total projectile
kinetic energy**.

This model is a practical replacement for the complete nuclear-elastic
interaction during diagnostic runs. It is not the missing charge-resolved
soft potential sought below, because it has no molecular orientation,
polarization, many-centre force, or ionic-charge dependence. Its role is to
quantify transport with a stable standard screened potential while the
ab-initio route remains closed.

The runtime modes are deliberately exclusive:

```text
off       : no custom nuclear-elastic process
zbl_full  : complete universal-ZBL nuclear-elastic baseline
nlh_hard  : retained-domain carbon NLH hard collisions only
```

`zbl_full` must not be added to `nlh_hard`. A future ZBL-soft/NLH-hard model
requires a validated potential- or impact-parameter partition; the NLH 30 eV
turning-potential boundary is not a 30 eV recoil-transfer cut. The shared ion
registry therefore records the ZBL runtime as an available diagnostic backend
but keeps final soft-runtime assembly blocked.

## Target quantity and validation logic

We sought charge-resolved diabatic interaction surfaces

```text
V_q(R, Omega): C^(q+) + H2O before electron capture,
```

where `R` is projectile--molecule separation and `Omega` is molecular
orientation. A later transport implementation would propagate on `V_q`, use
CTMC to sample a charge-changing event, and then select `V_q'`. The electronic
structure calculation was not intended to replace CTMC.

The first channel was

```text
C+ + H2O -> C + H2O+.
```

The experimental separated-fragment threshold is approximately 1.361 eV,
obtained from the difference between the water and carbon first ionization
energies in the [NIST Chemistry WebBook](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=20)
and [NIST Atomic Spectra Database](https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html).
This threshold is an external physical gate, not a fitted parameter.

Before calculating short separations, crossings or forces, a method had to
demonstrate:

1. the intended fragment charge, spin and orbital identity;
2. reproducibility from independent electronic preparations;
3. a stable separated-fragment threshold near 1.361 eV;
4. agreement between the corrected 20 Angstrom supermolecule and a same-level
   fragment calculation with counterpoise controls;
5. controlled basis, active-space, functional and correlation dependence;
6. continuous state following from large to smaller separation; and
7. finite-difference force consistency before table generation.

The investigation remained molecular and validation-only. No ice structure,
short-distance grid, crossing position, force curve or Geant4 table was
accepted.

## Summary of executed routes

| Route | Object represented | Useful result | Decisive failure |
|---|---|---|---|
| CP2K density-Hirshfeld CDFT | A DFT density constrained to a target carbon population | Official CDFT and force regressions reproduced; failure-safe branch machinery developed | Carbon--water SCF/state branch and multiplier response were not reproducibly validated |
| GPAW Gaussian-Hirshfeld CDFT | A PAW/PBE density with carbon charge and optional spin constraints | DO-MOM retained distinct carbon `p_x`, `p_y`, `p_z` states | The retained branch had essentially zero population response, then switched discontinuously after a tiny multiplier update |
| OpenMolcas GASSCF/CASSCF | Explicit multiconfigurational entrance and transfer manifolds | Reproducible common six-state numerical subspace and asymptotically vanishing coupling | Supermolecule and fragment thresholds disagreed; results depended strongly on active space/common orbitals |
| OpenMolcas XMS-CASPT2 | Dynamic-correlation correction to the common CASSCF manifold | All bounded calculations terminated normally | Threshold remained wrong and strongly basis/IPEA dependent |

## Route 1: CP2K constrained DFT

### What the model does

Constrained density-functional theory adds a Lagrange multiplier to the
Kohn--Sham energy so that a weighted fragment population reaches a specified
target. Our intended carbon target was `N_C = Z - q`; for q=1 this is five
electrons. The implementation used CP2K's density-Hirshfeld population,
all-electron GAPW, PBE and the `TZVPP-MOLOPT-GGA-ae` projectile basis in a
30 Angstrom cell with an 800 Ry grid. Counterpoise subtraction was planned as

```text
E_int = E(C^q+ + H2O; full basis)
      - E(C^q+; H2O ghost basis)
      - E(H2O; C ghost basis).
```

This defines a population-constrained DFT state. It does not guarantee a
unique formal-charge or orbital diabatic state.

### Provenance

- CP2K 2025.2, exact tested container revision `c3a8adfec5`.
- CP2K's [CDFT documentation](https://manual.cp2k.org/cp2k-2025_2-branch/methods/dft/constrained.html)
  and tagged [`qs_outer_scf.F`](https://github.com/cp2k/cp2k/blob/v2025.2/src/qs_outer_scf.F)
  defined the multiplier optimization and restart semantics.
- Holmberg and Laasonen, *J. Chem. Theory Comput.* **13**, 587--601 (2017),
  [doi:10.1021/acs.jctc.6b01085](https://doi.org/10.1021/acs.jctc.6b01085),
  describes CP2K CDFT.
- Ahart, Rosso and Blumberger, *J. Chem. Theory Comput.* **18**, 4438--4446
  (2022), [doi:10.1021/acs.jctc.2c00284](https://doi.org/10.1021/acs.jctc.2c00284),
  treats density-Hirshfeld CDFT forces and reliability.
- Carbon electronic configurations and terms were taken from the
  [NIST Atomic Spectra Database](https://physics.nist.gov/asd).

### Software-reference tests that passed

The upstream CP2K Zn and water CDFT examples were retained with only the
obsolete `MAP_CONSISTENT` keyword removed for 2025.2 compatibility. The exact
CP2K density-Hirshfeld HeH `ENERGY_FORCE` regression also passed: the total
force was `0.15521950466331`, relative error `3.29e-12` against the tagged
reference `0.1552195046628`. These tests established that the installed CDFT,
restart and force code worked. They did not validate carbon--water physics.

### Carbon-state initialization defects found and corrected

Early q=1 and q=4 trials exposed two workflow defects:

- CP2K's `KIND/BS NEL` applies half the listed shift to a spin occupation.
  The original q=1 input therefore made a 5.50-electron carbon guess instead
  of C+ with five electrons. The registry was corrected to even spin shifts;
  q=1 uses `BETA 2p NEL -2`.
- A fixed-multiplier diagnostic with CDFT `MAX_SCF 0` omitted an explicit
  optimizer. CP2K defaulted to `NONE` and aborted while printing setup. The
  renderer now names `BISECT`; `MAX_SCF 0` still prevents a multiplier update.

Those runs produced no accepted state. Correcting them removed code errors but
did not repair the underlying carbon electronic problem.

### Fixed-multiplier and reciprocal-root model

The replacement implementation independently evaluated active-CDFT
fixed-multiplier probes from fresh atomic guesses, sought two same-branch
residuals of opposite sign, and initialized two uninterrupted CP2K `BISECT`
runs from opposite endpoints. Wavefunction, multiplier, density, trace,
executable and input hashes were treated as one immutable state. Acceptance
required reciprocal agreement and local negative population response.

This machinery was deliberately stricter than merely observing normal CP2K
termination. No carbon complex passed it.

### What failed

- At q=4 and zero strength, the converged carbon population was approximately
  4.000 rather than the target 2.000 electrons: the calculation had relaxed
  to the charge-transferred branch.
- Fresh q=1 complex SCF calculations oscillated instead of contracting to the
  required `1e-7` residual.
- A restricted q=2 calculation with CDFT disabled reproduced the instability,
  showing that the problem was not solely the outer CDFT optimizer.
- OT/CG with `3PNT`, `GOLD` or `ADAPT`, `FULL_ALL` or `FULL_KINETIC`, and
  diagonalization with direct density mixing did not yield a reproducible
  accepted q=2 baseline. Best residuals could temporarily improve and then
  rebound into another energy basin.
- No pair of branch-consistent fixed-multiplier endpoints and reciprocal roots
  passed the population, energy, spin and density gates.

Increasing `MAX_SCF`, damping the mixer or restarting failed wavefunctions
would not establish the missing electronic identity. CP2K was therefore
closed as a carbon soft-potential route, although the software itself passed
its reference tests.

## Route 2: GPAW constrained DFT and orbital tracking

### What the model does

GPAW CDFT optimizes charge and optional spin constraints within the projector
augmented-wave formalism. We used PBE, GPAW 25.7.0 PAW datasets and the `dzp`
LCAO basis. Gaussian-Hirshfeld weights defined carbon charge/spin. GPAW's
bounded L-BFGS-B outer optimization was intended as an independent test of
whether CP2K's nested solver, rather than the state definition, caused the
failure.

For the open-shell C+ `2P` entrance state, direct orbital optimization (DO)
and maximum-overlap occupations (MOM) prepared explicit `p_x`, `p_y` and
`p_z` components. A fractional three-component state was used only as a
degeneracy diagnostic. DO-MOM prevents ordinary variational collapse by
tracking prescribed occupied orbitals; it does not itself impose an exact
fragment charge.

### Provenance

- GPAW 25.7.0 and official [CDFT](https://gpaw.readthedocs.io/documentation/cdft/cdft.html),
  [MOM](https://gpaw.readthedocs.io/documentation/mom/mom.html),
  [direct-optimization](https://gpaw.readthedocs.io/documentation/do/do.html)
  and [DO-GMF](https://gpaw.readthedocs.io/documentation/do-gmf/do-gmf.html)
  documentation.
- Melander *et al.*, *J. Chem. Theory Comput.* **12**, 5367--5378 (2016),
  [doi:10.1021/acs.jctc.6b00815](https://doi.org/10.1021/acs.jctc.6b00815),
  describes GPAW's PAW CDFT implementation.
- Ivanov, Levi and Jonsson, *J. Chem. Theory Comput.* **17**, 5034--5049
  (2021), [doi:10.1021/acs.jctc.1c00157](https://doi.org/10.1021/acs.jctc.1c00157),
  describes excited-state direct orbital optimization used for state
  preparation.

### Conventional SCF controls

Fresh q=1 atomic-density starts were tried with the documented He2-example
Davidson/Pulay settings, GPAW defaults, additional Davidson work, and the
documented difficult-SCF mixer. All reached noncontracting cycles or the
333-iteration limit before a usable CDFT state was produced. This independently
reproduced the initial electronic instability seen in CP2K.

### Constrained-first DO-MOM preparations

The next implementation removed GPAW's default unconstrained complex pre-SCF.
Isolated C+ and H2O fragments were prepared independently, embedded into the
full-system basis, and used to construct a full complex density matrix with a
selected carbon 2p occupation. The CDFT potential was active from the first
complex iteration.

At 12 Angstrom, four calculations terminated normally:

| State | Inner iterations | Carbon charge | Carbon spin | Selected 2p population |
|---|---:|---:|---:|---:|
| `p_z` | 19 | 1.00005945 | 0.99998683 | 0.99999885 |
| `p_x` | 23 | 1.00006751 | 0.99997674 | 0.99972264 |
| `p_y` | 23 | 1.00006475 | 0.99998016 | 0.99972337 |
| fractional `p_x/p_y/p_z` | 21 | 1.00006394 | 0.99998120 | about 1/3 each |

The pure states retained distinct orbital identities. Their determinant
`<S^2>` values were about 0.7525, close to 0.75 for a doublet. This was real
progress: the electronic-state preparation and checkpoint-rehydration problem
was solved numerically.

It was not yet a stationary multiplier-independent CDFT state. Independent
`p_z` preparations at 0.05 and 0.15 Ha retained the same SOMO but differed by
0.01624 eV and by 0.02274 electron in integrated density, failing the
preregistered agreement gate.

### Restart-retention controls

One live prepared MOM object and one rehydrated `.gpw` bundle were propagated
through genuinely tight inner solves. Rehydration restored spin-resolved
occupations, the DO eigensolver, MOM references and direct orbital matching.
Both retained `p_z`. Thus later failure could no longer be attributed to a
save/reload boundary or unnoticed 2p rotation.

### Multiplier calibration and fixed-lambda DO-GMF scan

Charge-only, charge-plus-spin and crossed multiplier starts were tested. Four
first inner solves retained `p_z`; their first charge-multiplier changes were
only about `-5.95e-5 Ha`. Nevertheless, the next solve changed the carbon
population by approximately four electrons and lost the selected branch. A
fifth calculation reached its bounded inner limit before a multiplier update.

To distinguish optimizer overshoot from discontinuous electronic response,
fixed multipliers `0.05, 0.075, 0.10, 0.125, 0.15 Ha` were evaluated from two
independent preparations each. The fragment-consistent target was
`4.999999720606075` electrons. The stable branch population changed by only
`4.60e-11` electron across the full 0.10 Ha interval, while retaining a
systematic residual near `-5.919e-5` electron. Repeatability was far tighter
than that residual.

### What failed

The prepared GPAW branch was reproducible but essentially flat with respect
to the charge multiplier. A tiny outer update then entered a different branch
instead of continuously approaching the root. Relaxing the tolerance would
have hidden a systematic error, not accommodated numerical noise. GPAW
therefore supplied no continuous, multiplier-independent q=1 diabatic state
for radial continuation or forces.

## Route 3: OpenMolcas multiconfigurational states

### Why this model was tried

CDFT defines fragment charge through a density partition. OpenMolcas was used
to construct entrance and charge-transfer states through explicit orbital
occupations and multiconfigurational wavefunctions instead. These calculations
were accurately described as occupation-restricted GASSCF or common-orbital
CASSCF calculations; they were not ALMO, BLW or a Q-Chem calculation.

### Provenance

- OpenMolcas v26.06, pinned upstream commit
  `8355057f32d65706a35996b5ab07cac2962bb728`; upstream verification test 000
  passed.
- Official OpenMolcas [RASSCF/GASSCF](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/rasscf.html),
  [RASSI](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/rassi.html)
  and [CASPT2](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/caspt2.html)
  documentation.
- Malmqvist *et al.*, *J. Phys. Chem.* **94**, 5477--5482 (1990),
  [doi:10.1021/j100377a011](https://doi.org/10.1021/j100377a011), for RASSCF.
- Andersson, Malmqvist and Roos, *J. Chem. Phys.* **96**, 1218--1226 (1992),
  [doi:10.1063/1.462209](https://doi.org/10.1063/1.462209), for CASPT2.
- ANO-RCC-VDZP and ANO-RCC-VTZP basis sets distributed with OpenMolcas were
  used; no basis exponent or contraction was fitted in this project.

### Occupation-restricted entrance and transfer preparations

At 20 and 12 Angstrom, separate calculations prepared:

- three C+ `2p^1` entrance components with neutral H2O; and
- the lowest C `2p^2` plus H2O+ transfer components coupled to total doublet
  spin.

Pure separately optimized p states falsely split the asymptotic carbon `2P`
manifold by about `0.0283 Ha`. This is an orbital-optimization bias, not a
physical asymptotic splitting. A generic UHF start also reached a higher
state-averaged local solution. Both paths were rejected.

RASSI between entrance and transfer JOBIPH files carrying different GAS
restrictions failed because OpenMolcas reset them as pure CAS spaces while the
CI expansions contained different configuration counts. More memory or MPI
ranks did not repair this incompatibility. This was an implementation/path
failure, not evidence of zero coupling.

### Common CAS(3,4) six-state model

A common active space containing H2O `1b1` and the three carbon 2p orbitals
retained six doublet roots:

- roots 1--3: H2O(`1b1`)^2 + C+(`2p`)^1 entrance components;
- roots 4--6: H2O+(`1b1`)^1 + C(`2p`)^2 transfer components.

Two independent orbital preparations converged to the same lower six-state
subspace. Boys-limit DQPhi/RASSI Hamiltonians agreed within `1.41e-7 Ha` at
20 Angstrom and `9.01e-7 Ha` at 12 Angstrom. The largest entrance--transfer
matrix elements were about `2e-14 Ha`, appropriately negligible at these large
separations. This validated a numerical manifold, not its energy accuracy.

The physical asymptotic test then failed. The CAS(3,4) fragment gap was
0.4586 eV, while the accepted 20 Angstrom common-CAS gap was 0.7602 eV. Their
0.3016 eV mismatch was about twenty times the leading 20 Angstrom
charge--dipole scale (about 0.0149 eV). Counterpoise and printed atomic
splittings did not explain it, so no 10 or 8 Angstrom calculation was launched
at this level.

### CAS(5,5) and manifold-specific averages

CAS(5,5) activated carbon 2s in addition to the CAS(3,4) orbitals. The
VDZP fragment threshold became 1.085 eV. Common-orbital results depended on
the initial p-component preparation. Separate entrance- and transfer-manifold
averages using the same configuration space did not both converge
reproducibly; transfer averages could converge while entrance averages failed
or reached preparation-dependent solutions. This quantified rather than
removed common-orbital/state-average bias.

### CAS(9,7) common six-state model

CAS(9,7) additionally activated water `3a1` and `1b2` orbitals. Eighteen roots
were retained to monitor intruding channels, while the physically relevant
first six were equally state averaged. Two independent preparations produced
the intended three entrance and three transfer identities.

RASSI comparison of the complete 18-root spaces showed important differences,
with singular values falling to 0.613. Restricting comparison to the selected
six-state spaces gave singular values

```text
0.99998846, 0.99996989, 0.99996341,
0.99992338, 0.99991719, 0.99989864.
```

Thus the selected six-state numerical subspace was reproducible even though
the larger root manifold was not. Bare VDZP supermolecule gaps were
0.849 and 0.823 eV for the two preparations, while the VDZP fragment limit was
about 1.085 eV. The internal disagreement remained much larger than the
finite-distance correction.

## Route 4: XMS-CASPT2 dynamic correlation

### What the model does

CASPT2 adds second-order dynamic correlation to a CASSCF reference. XMS-CASPT2
first rotates a common state-averaged manifold to reduce multistate artifacts,
then couples the perturbatively corrected states. We applied it only after the
six-state CAS(9,7) numerical subspace passed the overlap test.

The IPEA shift modifies the active-orbital part of the zeroth-order
Hamiltonian; it is a method choice, not a physical carbon parameter. We tested
the documented values 0.00 and 0.25 Ha. An imaginary shift of 0.1 Ha tested
weak-intruder sensitivity and was not fitted to the threshold.

XMS-CASPT2 follows the OpenMolcas implementation and the multistate CASPT2
literature; the underlying CASPT2 reference is Andersson, Malmqvist and Roos
([doi:10.1063/1.462209](https://doi.org/10.1063/1.462209)). The IPEA form was
introduced by Ghigo, Roos and Malmqvist, *Chem. Phys. Lett.* **396**, 142--149
(2004), [doi:10.1016/j.cplett.2004.08.032](https://doi.org/10.1016/j.cplett.2004.08.032).

### VDZP common-manifold and fragment results

Both 20 Angstrom VDZP supermolecule preparations and their matching
counterpoise fragments completed normally at IPEA 0.25. The corrected gaps
were:

| Quantity | Result (eV) |
|---|---:|
| 20 A supermolecule, preparation 1 | 1.562 |
| 20 A supermolecule, preparation 2 | 1.572 |
| Same-level counterpoise fragment limit | 1.814 |
| Physical threshold | 1.361 |

Dynamic correlation did not close the 0.24--0.25 eV internal mismatch and
placed the fragment threshold 0.453 eV above experiment.

### VTZP fragment and shift-sensitivity results

The VTZP carbon orbital ordering was independently inventoried before using
basis-specific active subspaces. All CASSCF and CASPT2 fragment calculations
terminated normally.

| Level | IPEA (Ha) | Imaginary shift (Ha) | Fragment threshold (eV) |
|---|---:|---:|---:|
| CASSCF | -- | -- | 0.812 |
| CASPT2 | 0.00 | 0.0 | 1.519686 |
| CASPT2 | 0.00 | 0.1 | 1.519700 |
| CASPT2 | 0.25 | 0.0 | 1.710885 |
| CASPT2 | 0.25 | 0.1 | 1.710886 |

The imaginary shift changed the threshold by less than 0.00002 eV; no weak
intruder sensitivity was evident. Basis and IPEA dependence were instead
large. The best result, 1.520 eV, remained 0.159 eV (about 12%) above the
physical threshold.

### What failed

XMS-CASPT2 produced numerically normal calculations but not a stable physical
asymptote. Agreement could not be claimed because:

- VDZP supermolecule and fragment results disagreed internally;
- VDZP and VTZP thresholds changed substantially;
- IPEA changed the VTZP threshold by about 0.191 eV; and
- even the closest tested result missed the experimental threshold by
  0.159 eV.

An empirical energy shift was not applied: the discrepancy was not small,
method-independent or isolated from common-orbital bias. A VTZP 20 Angstrom
supermolecule might have documented another mismatch, but could not satisfy
the already failed robustness gate. Development therefore stopped.

## Approaches discussed but not executed

ALMO/BLW in Q-Chem and separately optimized nonorthogonal diabatic-state
methods were considered as possible future approaches. Q-Chem was not
available, and no ALMO/BLW calculation was run. The OpenMolcas occupation-
restricted GASSCF tests must not be relabelled as ALMO or BLW. No experimental
soft potential was available and no empirical surrogate was introduced.

## Consequences for the transport framework

- No carbon soft-potential table exists.
- No q=1 result validates q=2--4; those surfaces were not developed.
- CTMC remains a charge-exchange model, not a soft-elastic replacement.
- NLH remains a hard-recoil model inside its validated short-range domain.
- The interval outside the NLH hard domain lacks a validated carbon elastic
  treatment and must be marked as a model limitation.
- The failed workflows must remain disconnected from production collectors and
  Geant4 table export.

## Archive contents and reproducibility

[`carbon_soft_collision_failed_validation_2026-08-10.zip`](carbon_soft_collision_failed_validation_2026-08-10.zip)
is the compact review archive. It contains:

- method and closure documentation;
- CP2K, GPAW and OpenMolcas workflow source snapshots;
- PBS launchers and software provenance receipts;
- manifests and representative textual inputs, outputs and logs; and
- machine-readable compact summaries.

It excludes GPW/WFN checkpoints, JOBIPH and other orbital files, cubes,
Molden files, NumPy binary arrays, installed software and duplicated scratch.
The archive is approximately 13 MiB; its SHA-256 checksum is stored beside it.

After checksum verification, approximately 5.4 GB of diagnostic run products,
6.3 GB of GPAW/OpenMolcas local installations, and duplicated root logs/orbital
files were removed. CP2K was retained because it supports separate workflows.

The shorter closure note is
[`../../nep_mbpol/soft_dft/benchmarking/CARBON_SOFT_COLLISION_CLOSURE_2026-08-10.md`](../../nep_mbpol/soft_dft/benchmarking/CARBON_SOFT_COLLISION_CLOSURE_2026-08-10.md).

## Reopening criteria

Future work should begin as a new specialist validation project using
independently optimized nonorthogonal diabatic states or independently
validated experimental data. Before any Geant4 export, it must repeat the
fragment asymptote, same-level supermolecule comparison, state-identity,
bidirectional continuation, active-space/basis/method convergence and
finite-difference force gates described above.
