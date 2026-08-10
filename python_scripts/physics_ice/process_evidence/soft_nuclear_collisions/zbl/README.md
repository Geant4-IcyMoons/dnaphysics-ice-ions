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

## References

- J. F. Ziegler, J. P. Biersack, and U. Littmark, *The Stopping and Range of
  Ions in Solids* (Pergamon, 1985).
- M. H. Mendenhall and R. A. Weller, *Nucl. Instrum. Methods Phys. Res. B*
  **227**, 420--430 (2005), DOI `10.1016/j.nimb.2004.08.014`.
- Geant4 Physics Reference Manual, “Ion Scattering.”
