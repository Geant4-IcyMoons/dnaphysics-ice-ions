# Carbon soft-collision validation record

## Status

The carbon fixed-charge soft-collision project is closed as **validation
failed**. No CP2K, GPAW or OpenMolcas result in this record is approved for a
Geant4 table or other production calculation.

This conclusion does not imply that soft elastic collisions are absent. It
means that the tested electronic-structure workflows did not produce a
reproducible, quantitatively validated potential for the prepared
`C^q+ + H2O` entrance state. Production work may continue independently for:

- NLH discrete hard nuclear collisions within the validated NLH domain; and
- CTMC charge exchange within its separately validated energy domain.

Those two processes do not replace the unresolved soft elastic contribution.
Any resulting transport model must therefore state this limitation rather
than assigning the missing contribution a value of zero.

## Physical quantity sought

The intended calculation was a charge-resolved diabatic interaction surface
for a carbon projectile approaching water before electron capture. Charge
exchange would then switch the trajectory to the surface for the new charge
state. The first validation channel was

```text
C+ + H2O -> C + H2O+
```

whose separated-fragment threshold is approximately 1.361 eV. Before any
short-distance surface, crossing, coupling or force could be accepted, the
method had to reproduce this limit and give consistent 20 Angstrom
supermolecule and fragment energies.

## What was tested

### CP2K constrained DFT

CP2K was tested with charge-localized carbon states, charge-specific atomic
occupations, fixed-multiplier probes, constrained outer optimization,
restart/continuation logic and solver controls. Reference calculations also
tested the installed density-Hirshfeld force implementation.

The carbon--water complexes did not establish a reproducible diabatic branch.
Observed problems included collapse toward charge-transferred states,
noncontracting or oscillatory inner SCF solutions, incorrect early occupation
initialization in obsolete trials, and an outer constraint response that could
not be accepted independently from both sides. Increasing iteration limits or
changing damping did not turn this into validated state following.

### GPAW constrained DFT with DO-MOM

GPAW was tested as an independent constrained-DFT implementation. Direct
orbital optimization and MOM were used to prepare and retain individual
carbon 2p components. Live-object and explicitly rehydrated checkpoint tests
showed that a selected component could survive a tightly converged inner
electronic solve.

This resolved the restart/state-retention question, but not the physical
outer problem. Independent multiplier starts did not yield a reproducible
accepted constrained root: calibration trajectories collapsed or showed an
unusable response. Thus the retained inner state was a useful numerical
benchmark, not an accepted soft potential.

### OpenMolcas CASSCF and XMS-CASPT2

OpenMolcas was used to avoid defining formal charge solely through a
Hirshfeld population. Tests included:

- separate entrance and charge-transfer preparations;
- three carbon 2p entrance components;
- common and manifold-specific state averages;
- CAS(3,4), CAS(5,5) and CAS(9,7) spaces;
- RASSI subspace-overlap and coupling diagnostics;
- ANO-RCC-VDZP and ANO-RCC-VTZP bases;
- XMS-CASPT2 with IPEA values 0.00 and 0.25 Ha; and
- imaginary shifts of 0.0 and 0.1 Ha.

A common six-state CAS(9,7) manifold was numerically reproducible, but the
physical asymptotic gates failed:

| Calculation | Threshold or gap (eV) |
|---|---:|
| Physical separated-fragment threshold | 1.361 |
| CAS(5,5)/VDZP fragment | 1.085 |
| CAS(9,7)/VDZP 20 A common supermolecule | 1.562--1.572 |
| CAS(9,7)/VDZP XMS-CASPT2 fragment, IPEA 0.25 | 1.814 |
| CAS(9,7)/VTZP CASSCF fragment | 0.812 |
| CAS(9,7)/VTZP CASPT2 fragment, IPEA 0.00 | 1.520 |
| CAS(9,7)/VTZP CASPT2 fragment, IPEA 0.25 | 1.711 |

The 0.1 Ha imaginary shift changed the VTZP thresholds by less than 0.00002
eV, whereas basis and IPEA choices produced large changes. The best tested
fragment result remained 0.159 eV above the physical threshold. At VDZP, the
20 A supermolecule and same-level fragment limits also differed by about
0.24--0.25 eV. Manifold-specific optimizations additionally exposed
common-orbital/state-average sensitivity.

## Why development stopped

The predeclared gate required all of the following:

1. reproducible electronic-state identity from independent preparations;
2. a stable separated-fragment threshold near 1.361 eV;
3. agreement between the corrected 20 A supermolecule and same-level fragment
   limit;
4. controlled active-space, basis and dynamic-correlation dependence; and
5. validated forces before any transport table was generated.

These conditions were not met for q=1. Extending the same unresolved method
to q=2--4 would require separate state development and would not repair the
failed q=1 asymptote. No shorter-separation grid, crossing location,
finite-difference force table or production potential was accepted.

No empirical threshold shift was applied. The discrepancies were too large
and method-dependent to justify presenting such a correction as an *ab
initio* prediction.

## Retained material

[`carbon_soft_collision_failed_validation_2026-08-10.zip`](carbon_soft_collision_failed_validation_2026-08-10.zip)
is a compact review archive containing:

- this closure documentation and method READMEs;
- the CP2K, GPAW and OpenMolcas workflow source used during the investigation;
- PBS launchers and software provenance receipts;
- manifests and representative textual inputs, outputs and logs; and
- compact numerical benchmark summaries.

The archive deliberately excludes large GPW/WFN checkpoints, JOBIPH and other
orbital files, cubes, Molden files, NumPy binary arrays and duplicated scratch
trees. The SHA-256 checksum is stored beside it.

## Future work

Reopening soft-collision development should be a separate specialist project.
The most defensible starting points are independently optimized nonorthogonal
diabatic states, or experimentally validated soft-stopping/transport data.
Any future method must repeat the asymptotic, state-identity, reverse
continuation and force gates before producing Geant4 tables.

Additional concise results are recorded in
[`../../nep_mbpol/soft_dft/benchmarking/CARBON_SOFT_COLLISION_CLOSURE_2026-08-10.md`](../../nep_mbpol/soft_dft/benchmarking/CARBON_SOFT_COLLISION_CLOSURE_2026-08-10.md).
