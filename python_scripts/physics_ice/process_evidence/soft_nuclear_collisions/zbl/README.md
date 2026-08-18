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

For the carbon soft-component study, `SoftZBLKernel` instead retains the
annulus between the energy-dependent NLH 30 eV hard impact parameter and the
outer ZBL recoil cutoff. This is exactly non-overlapping in impact area. It is
still validation-pending because NLH and ZBL are different potentials and
their force/angle continuity at the handoff is not assumed. Recoil cutoffs of
`1e-4`, `1e-5`, and `1e-6` eV are required for the annulus to remain represented
through 100 MeV; the earlier 1, 10, and 30 eV cutoffs become empty at high
energy and are not soft-boundary candidates.

## Restartable PBS campaign

Submit the carbon campaign from the repository root with:

```bash
bash pbs/launch_zbl_atomistic_carbon.sh
```

The launcher prepares 41 logarithmically spaced energies from 10 keV to
100 MeV for the `1e-4`, `1e-5`, and `1e-6` eV recoil cutoffs using the
production-matched hexagonal ice Ih structures. Each structure/orientation
case is a restart-safe PBS array task. Sampling proceeds in batches until both
the 95% relative confidence half-widths for the scalar cross-section,
stopping, and transport estimators and the DKW angular-CDF half-width are at
most 0.005, subject to the recorded trajectory ceiling.

The launcher partitions the cases into array shards of at most 50 elements;
each element requests 64 CPU cores and owns one case and checkpoint stream.

Outputs are written below the ignored
`validation/runs/zbl_soft_carbon/` directory. The reducer writes
`zbl_soft_cross_sections.csv`, `zbl_soft_angular_quantiles.csv`, and a
checksum-attested result manifest. Restart checkpoints retain the complete
weighted final-deflection sample for every case, so alternative angular bins
or quantiles can be reconstructed without rerunning trajectories. A failed or
walltime-limited case is resumed by running the launcher again. It submits only
case indices without a completion receipt; each incomplete case continues from
its last attested batch manifest. PBS output is retained under the campaign's
`pbs_logs/` directory, and the reducer refuses incomplete or statistically
unconverged campaigns.

The campaign was moved from the temporary ZBL worktree into this repository
after a clean PBS cancellation on 2026-08-18. `RELOCATION_PROVENANCE.json`
records the path rewrite and campaign-signature change. Numerical checkpoint
payloads and their configuration signatures were not changed.

## References

- J. F. Ziegler, J. P. Biersack, and U. Littmark, *The Stopping and Range of
  Ions in Solids* (Pergamon, 1985).
- M. H. Mendenhall and R. A. Weller, *Nucl. Instrum. Methods Phys. Res. B*
  **227**, 420--430 (2005), DOI `10.1016/j.nimb.2004.08.014`.
- Geant4 Physics Reference Manual, “Ion Scattering.”
