# Carbon soft-collision electronic-structure closure

## Outcome

No validated *ab initio* fixed-charge carbon--water soft-collision potential
was obtained. These calculations are numerical-method benchmarks only and
must not be used to generate Geant4 production tables. The soft elastic
interaction is physically real; its omission must not be described as a zero
cross section or zero stopping contribution.

The production path therefore retains the independent NLH hard-collision
model and CTMC charge-exchange model. It does not claim a complete elastic
model outside the validated NLH domain.

## Methods tested

1. CP2K constrained DFT exposed state-initialization, inner-SCF and constraint
   response problems. The intended charge-localized entrance branch was not
   reproducibly validated.
2. GPAW direct-orbital/MOM preparation could retain a selected carbon 2p
   component during a tight inner solve, but multiplier calibration did not
   establish a reproducible accepted constrained state.
3. OpenMolcas common-orbital CASSCF/XMS-CASPT2 produced a reproducible
   six-state CAS(9,7) numerical manifold, but failed the physical asymptotic
   energy gates and showed substantial common-orbital, basis and IPEA
   sensitivity.

## Decisive OpenMolcas results

The physical separated-fragment threshold is approximately 1.361 eV for
`C+ + H2O -> C + H2O+`.

| Calculation | Threshold or gap (eV) |
|---|---:|
| CAS(5,5), ANO-RCC-VDZP, fragment | 1.085 |
| CAS(9,7), ANO-RCC-VDZP, 20 A common six-state supermolecule | 1.562--1.572 |
| CAS(9,7), ANO-RCC-VDZP, XMS-CASPT2 fragment, IPEA 0.25 | 1.814 |
| CAS(9,7), ANO-RCC-VTZP, CASSCF fragment | 0.812 |
| CAS(9,7), ANO-RCC-VTZP, CASPT2 fragment, IPEA 0.00 | 1.520 |
| CAS(9,7), ANO-RCC-VTZP, CASPT2 fragment, IPEA 0.25 | 1.711 |

An imaginary shift of 0.1 Ha changed the VTZP fragment thresholds by less
than 0.00002 eV. IPEA and basis dependence were much larger. The best tested
fragment value remained 0.159 eV above the physical threshold. At VDZP, the
same-level 20 A supermolecule and fragment results also differed by roughly
0.24--0.25 eV, far exceeding the expected leading finite-distance scale.

## Decision gate

Development stopped before shorter separations, crossing locations, forces or
production tables. The validation gate required a stable physical threshold,
agreement between the corrected 20 A supermolecule and same-level fragment
limit, reproducibility across preparations, and controlled method/basis
dependence. Those conditions were not met, and q=2--4 were not attempted as
production surfaces.

Future work should begin as a separate specialist project using independently
optimized nonorthogonal diabatic states or validated experimental data. An
empirical asymptotic correction was not applied because the residual errors
were neither small nor method-independent.

## Archive policy

The companion review archive retains source workflows, PBS launchers,
manifests, inputs, textual logs and compact benchmark summaries. Large GPW,
WFN, JOBIPH, orbital, Molden, cube and duplicated scratch artifacts were
deliberately excluded. The original diagnostic run products were removed from
the production evidence tree after the archive checksum was verified.
