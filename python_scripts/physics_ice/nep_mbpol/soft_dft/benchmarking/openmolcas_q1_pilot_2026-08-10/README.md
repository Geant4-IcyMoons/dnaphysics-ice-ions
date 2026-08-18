# OpenMolcas q=1 C--H2O pilot (2026-08-10)

## Outcome

The OpenMolcas v26.06 runtime is pinned to upstream commit
`8355057f32d65706a35996b5ab07cac2962bb728`; upstream verification test 000
passes.  The 20 and 12 A orbital inventories, occupation-restricted GASSCF
states, common state-averaged CAS(3,4) controls, and Boys-limit DQPhi runs all
completed on PBS.

The common CAS active orbitals are the water 1b1 donor and the three carbon 2p
components.  The six retained doublet roots have the intended identities:

- roots 1--3: H2O(1b1)^2 + C+(2p)^1 entrance components;
- roots 4--6: H2O+(1b1)^1 + C(2p)^2 charge-transfer components, with the
  carbon triplet coupled to the water radical to total doublet spin.

Two independently prepared orbital guesses (the separately optimized pz and
px entrance states) converge to the same lower state-averaged branch.  Their
six-state DQPhi Hamiltonians agree to `1.41e-7` Ha at 20 A and `9.01e-7` Ha at
12 A.  The largest entrance--transfer matrix element is `1.91e-14` Ha at 20 A
and `2.20e-14` Ha at 12 A: numerically zero, as required in the separated
limit.  The DQPhi rotation is the identity at both distances because the
computed states are already charge localized.

## Rejected paths

- Separately optimized pure p states falsely split the asymptotic carbon 2P
  manifold by about 0.0283 Ha.  They are not a common-orbital diabatic basis.
- The generic-UHF common-CAS start converges to a higher state-averaged local
  solution and is rejected because independent starts find the lower branch.
- RASSI between entrance and transfer JOBIPH files with different GAS
  restrictions crashes in OpenMolcas v26.06.  The source resets both as pure
  CAS spaces while their CI expansions contain different numbers of
  configurations.  Increasing memory and changing MPI ranks does not fix it.
  No coupling from this path is accepted.

## Status and next gates

This validates a reproducible **numerical anchor**, not a production soft
potential or a Landau--Zener model.  Before inward continuation, the following
remain required:

1. compare the 20 A energies with independently calculated C+ and H2O / C and
   H2O+ fragment sums in the same basis, including counterpoise controls;
2. compare full one-particle densities between the two accepted orbital starts,
   not only energies, occupations, and Mulliken fragment identities;
3. continue bidirectionally through 10, 8, 7, and 6 A with root/occupation
   tracking and repeat the DQPhi analysis;
4. demonstrate nonzero, reproducible coupling where the fragments overlap;
5. validate finite-difference forces before constructing any soft-collision
   table.

The implementation follows the OpenMolcas v26.06 RASSCF/GASSCF and RASSI DQPhi
documentation.  The latter states that `ALPHA=0` and `BETA=0` reduce DQPhi to
Boys diabatization.  These calculations are occupation-restricted GASSCF or
common-CAS calculations; they are not BLW or ALMO.

## Evidence

- orbital inventory: job `113610`;
- initial occupation-restricted pilots: jobs `113611`--`113615`;
- reproducible lower common-CAS branches: jobs `113663`, `113664`, `113667`,
  and `113668`;
- reproducible lower-branch DQPhi: jobs `113669[]` and `113670[]`.

Machine-readable numerical results are in `summary.json`.  Full signed inputs,
outputs, JOBIPH files, and manifests are under
`process_evidence/soft_nuclear_collisions/validation/runs/`.

# Next-stage gate (2026-08-10)

The bounded next-stage calculation stopped at the required fragment-energy
gate.  The reproducible spherical C-fragment preparation gives a CAS fragment
gap of 0.4586 eV, whereas the accepted 20 A common-CAS calculation gives
0.7602 eV.  Their 0.3016 eV difference is far larger than the approximately
0.0149 eV leading charge--dipole scale at 20 A.  Counterpoise shifts and the
C atomic-manifold splitting are zero at the printed precision, so neither
explains the mismatch.  CAS(3,4) has therefore not established the correct
asymptotic model, and no 10 or 8 A point was launched.

Two independent preparations do span the same six-state subspace at 20 and
12 A to the printed overlap precision.  This is a numerical reproducibility
result, not physical validation.  A state-averaged density comparison remains
pending because this OpenMolcas build did not emit the requested HDF5 density
data.  Exact numerical provenance and the fail-closed decision are recorded in
`next_stage_summary.json`.
