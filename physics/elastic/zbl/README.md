Relocated package: see [elastic layout](../README.md) and shared
[ice inputs](../../../models/ice/README.md). Historical validation references
below describe retained evidence in the legacy tree; they are not new results.

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
  physics.elastic.zbl.generate_backend \
  --phases hexagonal_ih_100k amorphous_lda_80k \
  --projectiles C O S \
  --energy-min-ev 1e4 \
  --energy-max-ev 1e8 \
  --transfer-cutoffs-ev 1 10 30 \
  --output-directory \
  physics/elastic/zbl/runs/zbl_full_atomistic
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
bash physics/elastic/jobs/launch_zbl_atomistic_carbon.sh
```

The launcher prepares 41 logarithmically spaced energies from 10 keV to
100 MeV for the `1e-4`, `1e-5`, and `1e-6` eV recoil cutoffs using the
production-matched hexagonal ice Ih structures. Each structure/orientation
case is a restart-safe PBS array task. Sampling now uses 5% estimated relative standard errors for the five whole-history scalar ratios, with 20% growth rounded to a trajectory block. The trajectory ceiling remains explicit. Zero means and zero empirical variance do not automatically qualify. There is no CDF stopping requirement or simultaneous confidence claim; exported angular quantiles are descriptive. Independent trajectory-level weighted ratio covariance is retained.

The launcher partitions the cases into array shards of at most 50 elements;
each element requests 64 CPU cores and owns one case and checkpoint stream.

Outputs are written below the ignored
`validation/runs/zbl_soft_carbon_scalar/` directory for new campaigns. The historical `zbl_soft_carbon/` campaign remains unchanged. The reducer writes
`zbl_soft_cross_sections.csv`, `zbl_soft_angular_quantiles.csv`, and a
checksum-attested result manifest. Restart checkpoints retain the complete
weighted final-deflection sample for every case, so alternative angular bins
or quantiles can be reconstructed without rerunning trajectories. A failed or
walltime-limited case is resumed by running the launcher again. It submits only
case indices without a completion receipt; each incomplete case continues from
its last attested batch manifest. PBS output is retained under the campaign's
`pbs_logs/` directory, and the reducer refuses incomplete or scalar-unconverged campaigns. It rechecks scalar SEs, exports explicitly named standard-error columns, and reports adjacent-cutoff stopping/transport comparisons. Neither scalar completion nor the reducer qualifies the physical handoff.

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


## NLH–ZBL handoff rationale (2026-09-14)

Retain **30 eV of NLH repulsive potential at closest approach** as the proposed
partition. It is neither projectile kinetic energy nor transferred recoil
energy. Nordlund, Lehtola & Hobler (2025), Physical Review A 111, 032818,
report good agreement of their fitted potentials with quantum-chemical
references above this potential scale. This supports the retained short-range
NLH domain, not an independently validated switch to universal ZBL.
Primary paper: https://www.mv.helsinki.fi/home/knordlun/pub/Nor25.pdf
DOI: https://doi.org/10.1103/PhysRevA.111.032818

The implemented central-potential boundary solves V_NLH(r30)=30 eV and uses
b_h=r30 sqrt(1-30/E_rel), with zero hard area when E_rel<=30 eV. Here E_rel
is the pair relative kinetic energy used by the retained scattering solver.
The hard model covers 0<=b<=b_h; the soft model covers b_h<b<=b_outer, where
b_outer is determined separately by the ZBL recoil-transfer cutoff. Both
must use identical projectile/target, energy convention and b_h. An empty
annulus when b_outer<=b_h is not evidence that omitted weak scattering is
negligible. Soft count cross sections are cutoff-dependent; convergence of
stopping and angular transport is the relevant omitted-tail diagnostic.

The Geant4 screened-scattering reference also treats the minimum represented
energy transfer as a distinct transport cutoff:
https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/elastic_scattering/nuclearrec.html
It does not prescribe our 30 eV NLH handoff or 5% acceptance target.

This partition avoids double counting in impact area. It does not make the
NLH and ZBL forces or scattering angles continuous at the boundary: each
retained binary solve uses its own potential over the full encounter.
No smoothing function or fitted blending factor has been introduced.
A responsible combined-model check compares angles, recoil and integrated
stopping/transport near b_h, and tests a common boundary shifted within the
supported short-range region (for example 30 versus 60 eV). Moving only the
soft boundary would create a gap and is not such a test. Existing hard maps
are fixed at 30 eV and cannot silently be reused as 60 eV maps.

For the existing 1e-4, 1e-5 and 1e-6 eV recoil-cutoff comparisons, the reducer
reports observed stopping/transport differences and independent-cohort
comparison SEs against 5% criteria. A noisy comparison is unresolved; more
samples can reduce its uncertainty. A resolved change requires a smaller
cutoff or a model review. The current reducer does not automatically launch
extra cutoff cases or declare an asymptotic tail bound.

For explicit ice, hard-only and soft-only histories do not constitute a
combined trajectory: a deflection changes subsequent encounters. A final
combined operator must choose the branch for each encounter in one history.
Only microscopic pair moments can be added directly over disjoint impact
regions. Do not add independently propagated finite-path ice tables and
claim a validated combined response.

The present change normalizes statistical policy and repairs the launcher’s
previously unsupported relative-tolerance argument. Schema 2 campaigns bind
the updated sources, including the convergence module. Old manifests fail
compatibility rather than being re-signed. The physical trajectory checkpoint
identity remains unchanged because the sampling law and trajectories are
unchanged; any reuse still requires matching physical identity and explicit
reassessment, not copying old completion receipts. Historical outputs were
not modified. No new simulations were launched or physical handoff validated.
The launcher covers ice Ih at 10 keV–100 MeV total carbon energy; amorphous
coverage is supported by preparation but is not selected in this launcher.
The hard campaign's 1–10 keV range therefore still needs matching soft coverage.
