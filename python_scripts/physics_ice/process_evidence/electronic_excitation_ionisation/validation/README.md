# Electronic excitation and ionisation validation

This gate records phase-specific optical-data provenance, numerical
convergence, Barkas/no-Barkas comparisons, energy-domain limits, and Geant4
table consistency. The SBETHE comparison is benchmark evidence; it does not by
itself validate the production optical oscillator-strength model.

No additional acceptance record is created merely because a PWBA table was
generated successfully.

## Ion optical normalization

Ion PWBA and RPWBA generation use one physical density per ice phase, from
`constants.py`: 0.9343471678603292 g/cm3 for amorphous ice and 0.9335 g/cm3
for hexagonal ice. The corresponding H2O molecular densities are
3.1233320623e28 and 3.1205001529e28 m^-3. These are also the configured
Geant4 material densities. There is no water-density divisor or adjustable
target-density multiplier in the ion generator.

For the partitioned optical ELF, define

```
N_H2O = rho * N_A / M_H2O                       (with consistent SI units)
I_opt = integral W * ELF_valence(W,0) dW + (pi/2) * Ep_fit^2 * 0.179
K = (pi/2) * hbar_eVs^2 * e^2 / (m_e * epsilon_0)
ELF_scale = K * (10 * N_H2O) / I_opt
df/dW = W * ELF_scale * ELF_raw(W,0) / (K * N_H2O)
```

Thus the full optical oscillator-strength integral is 10 per H2O. The
physical plasma energies implied by the configured densities are 20.75231
and 20.74290 eV. The original `Ep_fit` values (20.82 and 20.59 eV) remain
parameters of the fitted spectral shapes, not additional material-density
settings. Their small f-sum residuals are included in `I_opt`; no optical
parameter is refitted and no ICRU/Matias stopping curve enters the calculation.

The generator applies `ELF_scale/N_H2O` in every ion valence and K-shell
Born kernel, including both RPWBA terms. The stored DCS remains microscopic
per H2O molecule. Geant4 already computes the macroscopic rate as
`N_H2O * sigma`, so no runtime density multiplier is added. The same
normalization is used with and without the K shell; omitting that channel
does not transfer its strength to valence. Barkas retains its separate
8-valence + 2-core optical OOS normalization and is not rescaled again.

The normalization integral uses 60001 logarithmic points from the valence
onset to 1e8 eV and the independently normalized optical K-shell moment.
Tests double the energy resolution and extend the upper bound. The existing
valence high-energy rolloff is unchanged; its small additional loss of
optical strength is tested separately. The K-shell B=543.4 eV, Zeff=7.7,
0.179 fractional target, finite-q shape, and absence of hydrogenic rolloff
are unchanged. In this Born partition the normalized core strength is about
1.79, not the Barkas OOS occupancy of 2.

This patch fixes the optical-to-molecular normalization only. It does not
repair finite-q sum-rule violations, restore Kramers--Kronig consistency,
or refit the complex dielectric screening used by RPWBA. Agreement with
stopping-power data remains a separate validation question.

### Regeneration and provenance

NPZ metadata and a Geant4-compatible comment in each DAT file record
`ion_normalization_version=optical-fsum-per-H2O-v1`, the material density,
ELF scale, optical moments, and electron sum. DAT metadata also records
the projectile, charge convention, K-shell selection, and PWBA/RPWBA mode.
Old NPZ caches are rejected; old or incompatible DAT patches cannot be
silently merged. Numeric DAT columns and their unit conversion are unchanged.

Regenerate the complete desired energy range once with
`--no-merge-energy-patches`, retaining the desired projectile, Barkas, and
RPWBA flags. Subsequent compatible energy patches can use the default merge
mode. Existing tables and simulation outputs are not corrected in place:
regenerate the ion DCS/TCS tables and rerun dependent simulations. Electron
generation and electron optical/K-shell behavior are unchanged.

Focused regression command:

```bash
python -m pytest -q tests/test_ion_optical_normalization.py \
  tests/test_barkas_dcs.py tests/test_projectile_relativistic_dcs.py \
  tests/test_ice_phase_density.py
```

On 2026-09-04 all 83 checks passed, including bitwise serial/10-worker
agreement and exported DCS/TCS consistency for both phases with PWBA/RPWBA
and Barkas off/on. Independent integration to 1e9 eV gave full optical sums
of 9.99999799 (amorphous) and 9.99999792 (hexagonal). Including the unchanged
valence rolloff gave 9.99995487 and 9.99995872. These are numerical
normalization checks, not experimental validation or regenerated production
tables.

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
