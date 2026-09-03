# Electronic excitation and ionisation validation

This gate records phase-specific optical-data provenance, numerical
convergence, Barkas/no-Barkas comparisons, energy-domain limits, and Geant4
table consistency. The SBETHE comparison is benchmark evidence; it does not by
itself validate the production optical oscillator-strength model.

No additional acceptance record is created merely because a PWBA table was
generated successfully.

## Relativistic projectile kernel

`generate_ice_cross_sections_ion.py --relativistic-projectile-dcs` uses the
finite-Q relativistic plane-wave Born approximation (RPWBA) from A. D.
Dominguez-Munoz et al., "A model for Geant4-DNA to simulate ionization and
excitation of liquid water by protons travelling above 100 MeV," *Radiation
Physics and Chemistry* **199** (2022) 110363,
<https://doi.org/10.1016/j.radphyschem.2022.110363>.

The implementation evaluates the longitudinal and transverse terms of Eqs.
(1)-(4) on the same finite-q grid and applies the condensed-medium dielectric
correction of Eqs. (7)-(9) by default. It does not use the earlier optical-q=0
Fano transverse approximation. The paper's liquid-water GOS is replaced by
the phase-specific ice dielectric response in this repository; consequently,
this is an application of the published projectile kernel, not a reproduction
of the paper's liquid-water tables. The hydrogenic K-shell GOS has no complex
K-shell dielectric counterpart in the current ice model, so its density-effect
screening factor uses the finite-q valence dielectric function; metadata marks
this approximation explicitly.

The publication validates protons from 100 to 300 MeV. Carbon, oxygen, and
sulfur calculations are explicitly marked in metadata as bare-ion first-Born
extrapolations. Their validity must be assessed at the projectile velocity in
question; table generation alone does not validate charge-state equilibrium,
charge exchange, Barkas terms, or other higher-order effects.

Example using incident energy per nucleon:

```bash
ICE_TYPE=hexagonal python -u \
  python_scripts/physics_ice/generate_ice_cross_sections_ion.py \
  --projectile carbon --energy-unit per_u \
  --energy-min-MeV 10 --energy-max-MeV 300 \
  --relativistic-projectile-dcs --include-kshell \
  --kshell-model hydrogenic-gos
```

Use `--no-rpwba-density-effect` only to produce an isolated-molecule RPWBA
diagnostic. It is encoded separately in output names and metadata.
