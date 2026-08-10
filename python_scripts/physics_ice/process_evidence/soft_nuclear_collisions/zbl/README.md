# Universal-ZBL elastic baseline

This package is the independent Python reference for the
`G4DNAZBLFullElastic` Geant4 process. It evaluates classical binary collisions
with the universal Ziegler--Biersack--Littmark screening function for H, He, C,
O, and S projectiles against the H and O nuclei of water ice.

It is grouped with the retained evidence from the failed CP2K, GPAW, and
OpenMolcas soft-collision approaches. The compiled Geant4 implementation
remains in repository-level `include/G4DNAZBLFullElastic.hh` and
`src/G4DNAZBLFullElastic.cc`, where the existing CMake build expects runtime
processes.

## Scope

- Incident energy is **total projectile kinetic energy**, not energy per
  nucleon.
- The initial runtime range is 1 keV--100 MeV total kinetic energy.
- The interaction uses nuclear atomic number `Z`; it is independent of ionic
  charge state `q`.
- The model changes projectile direction and transfers energy to an H or O
  recoil. Sub-threshold recoil energy is local non-ionising deposition.
- Electronic stopping, excitation, ionisation, charge exchange, molecular
  polarization, ice orientation, and many-centre forces are outside this
  model.

The runtime is a **full nuclear-elastic diagnostic baseline**. It is not a
Barkas/effective-charge correction and is not a soft-only contribution that
can be added to NLH. `zbl_full` and `nlh_hard` are mutually exclusive until an
independently validated impact-parameter or potential-domain partition is
available.

The minimum represented recoil transfer is a numerical transport cutoff, not
the 10 or 30 eV NLH turning-potential boundary. Its convergence must be tested
separately.

## C/O/S atomistic backend procedure

The selected full-ZBL campaign covers carbon, oxygen, and sulfur. Hydrogen and
helium remain assigned to the complete HTran proton/helium elastic model and
must not be run through this procedure in the assembled physics list.

From the repository root, generate checksum-attested backends for both
accepted ice phases over 10 keV--100 MeV, with 1, 10, and 30 eV minimum recoil
transfers for cutoff convergence:

```bash
python -m \
  python_scripts.physics_ice.process_evidence.soft_nuclear_collisions.zbl.generate_backend \
  --phases hexagonal_ih_100k amorphous_lda_80k \
  --projectiles C O S \
  --energy-min-ev 1e4 \
  --energy-max-ev 1e8 \
  --transfer-cutoffs-ev 1 10 30 \
  --output-directory \
  python_scripts/physics_ice/process_evidence/soft_nuclear_collisions/validation/runs/zbl_full_atomistic
```

The command writes one phase manifest containing the three accepted hexagonal
replicas and one containing the published amorphous configuration.
`FullZBLKernel` supplies
analytic binary collisions to the existing periodic atomistic trajectory
engine, which determines the ordered H/O encounters from those structures.
The manifest records that C q=0..6, O q=0..8, and S q=0..16 share their
elemental ZBL kernel; these aliases are not independent calculations.

The retained domain is the complete ZBL disk from zero impact parameter to
the selected recoil cutoff. No mask or NLH partition is applied. Full ZBL
therefore replaces NLH for a diagnostic trajectory run and is never added to
it. HTran remains the complete elastic path for H and He. No backend becomes
the transport default until the validation gates in `../validation/README.md`
are satisfied.

## References

- J. F. Ziegler, J. P. Biersack, and U. Littmark, *The Stopping and Range of
  Ions in Solids* (Pergamon, 1985).
- M. H. Mendenhall and R. A. Weller, *Nucl. Instrum. Methods Phys. Res. B*
  **227**, 420--430 (2005), DOI `10.1016/j.nimb.2004.08.014`.
- Geant4 Physics Reference Manual, “Ion Scattering.”
