# Generated NLH hard-collision kernels

This directory contains the retained-domain binary-collision tables for H,
He, C, O, and S projectiles against H and O targets from 1 keV to 100 MeV total
projectile energy. The numerical definition, 30 eV turning-potential boundary,
adaptive interpolation contract, and 0.5% combined numerical interpolation
tolerance are recorded in `nlh_collision_kernels.manifest.json`. The 0.5% value
is not a claim of physical accuracy; the published pair-potential RMS errors
must be reported separately.

The table has 207,689 data rows and SHA-256
`a7152349687fe69db74b2075bc083d28a8e188e5708d58075b817faf3b2d4119`.
PBS job 106765 generated the configuration-hashed checkpoints, assembled the
table, and ran the independent benchmark from source commit `51b366056`.
The manifest records numerical implementation version 3 and configuration hash
`080c407f5f167ab2`.
The benchmark covers all ten projectile--target pairs at six energies from
1 keV to 100 MeV and reports `accepted: true` at the 0.5% limit.

The `.checkpoints/` directory is deliberately excluded from Git because the
final table and manifest are sufficient runtime inputs and the restart shards
are deterministically reproducible. These tables describe NLH hard binary
events. They are not complete elastic cross sections, phase-resolved ice
transport results, secondary recoil cascades, or soft-scattering data. The
threshold-defined microscopic cross section is evaluated analytically at
runtime,

```text
sigma_Pt^hard = pi r_th,Pt^2 (1 - V_min/E_cm,Pt), E_cm,Pt > V_min,
                0,                                  otherwise,
```

and equals `pi b_max,Pt^2`; it is never interpolated. For pure water ice the
corresponding macroscopic rate is
`Sigma_P^hard = n_H2O (2 sigma_PH^hard + sigma_PO^hard)`.

`geant4/nlh_hard_elastic_C.dat` is the compact, checksum-linked carbon runtime
product. Reproduce it with `../export_nlh_geant4_table.py`. Its use as a
density-only continuum model remains conditional on atomistic validation:
appreciable amorphous--hexagonal or directional differences require a
validated phase/orientation correction.
