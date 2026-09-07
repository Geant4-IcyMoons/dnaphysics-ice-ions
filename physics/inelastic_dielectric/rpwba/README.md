# RPWBA

`kernels.py::RPWBAKernel` extends the PWBA state with relativistic momentum
bounds and finite-Q longitudinal/transverse integrands. It shares the target
and projectile models rather than maintaining a second set of ice parameters.

The projectile kernel follows Dominguez-Munoz et al., *Radiation Physics and
Chemistry* **199** (2022) 110363,
<https://doi.org/10.1016/j.radphyschem.2022.110363>. The medium transverse
coefficient follows the author's 2025 thesis, Eq. (2.287), not the discrepant
literal Eq. (8) in the article. Equations and source details are retained
beside `_rpwba_transverse_ratio` and in [validation](../validation/README.md).

Select `--relativistic-projectile-dcs=true`. The complete path includes both
longitudinal and transverse terms and the dielectric density effect. Use
`--no-rpwba-density-effect` only for the isolated-molecule diagnostic.
Incident energy defaults to total ion kinetic energy; `per_u` conversion
occurs before beta, momentum limits, and screening are evaluated.

The publication's proton validation range is 100-300 MeV. Application to
ice, other projectiles, lower energies, and screened charge states retains
the explicitly documented approximations; executable code is not validation.

```bash
python -m pytest -q tests/test_projectile_relativistic_dcs.py
```

These tests include dilute-limit recovery, transverse/longitudinal ratios,
charge scaling, mass/energy conventions, and stable momentum bounds.
