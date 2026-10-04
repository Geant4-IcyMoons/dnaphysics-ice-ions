# Optical target audit

Compare the production Born and oscillator optical inputs for amorphous and
hexagonal ice against the phase-specific ELF fits in Matias et al.,
[PRL 135, 148003 (2025), Fig. 1](https://doi.org/10.1103/ksdx-mnd7).
This is an optical consistency benchmark, not a finite-q validation or a
replacement target model. No cross sections or production parameters change.

## Run and output

From the repository root:

```bash
python -m physics.inelastic_dielectric.benchmarks.optical_elf.run
python -m pytest -q physics/inelastic_dielectric/benchmarks/optical_elf/test_audit.py
```

`plots/optical_elf_audit.pdf` compares the low-energy ELF, high-energy tail,
and cumulative strength. `plots/summary.md` contains the compact numerical
table; `plots/results.json` records component moments, convergence, native
normalization factors, and source hashes. Serial vectorized integration is
adequate: no projectile quadrature, nonlinear ODE, cluster job, or transport
simulation is required.

The retained reference CSVs can be reproduced with `pdfplumber`:

```bash
python -m physics.inelastic_dielectric.benchmarks.optical_elf.extract_reference physics/inelastic_dielectric/benchmarks/optical_elf/references/matias2025.pdf
```

`references/provenance.json` records the source URL, PDF hash, vector-path
selection, axis calibrations, and finite-figure limitations. Both main-panel
and inset clipping regions are enforced; off-frame inset paths must not be
mistaken for optical structure. Reference values are not renormalized.

## Normalization and moments

Let `U = pi*hbar^2*e^2/(2*m_e*epsilon_0)` in eV^2 m^3. Then
`df/dW = W*ELF/(N_H2O*U)` is in eV^-1 per molecule. The integral is the
effective electron number, and
`I = exp(integral log(W/eV) df / integral df)` eV.

- Born uses the generator's partitioned `epsilon_Eq` at q=0, joint outer/K
  allocation, unscaled hydrogenic continuum, global per-molecule conversion,
  and existing valence rolloff. Its allocation is approximately 8.262/1.738.
  The report also records the pre-rolloff normalization moment.
- Oscillator uses the actual `correction.oos_density`: the same partitioned
  q=0 ELF, common per-molecule factor, valence rolloff and K continuum as Born.
  The audit evaluates Born independently and checks pointwise agreement.
  Its reconstructed ELF is not a newly derived dielectric response.
- Material densities are imported from production. The Matias curves use
  the source's 0.94 g/cm^3 for both ice phases. Convert each to a per-molecule
  OOS before reconstructing its ELF at our phase density for the shape plot.
  This is a unit/density conversion, not an amplitude fit.

The numerical grid extends to 1 GeV transferred energy solely to converge
optical moments; this is not a projectile energy or a kinematic DCS cutoff.
Thresholds are explicit grid nodes, including both sides of discontinuities.
Refining 120001 to 240001 logarithmic nodes must change the total, I, and
window integrals by less than 1e-4 relatively. The shared normalization
retains Born's 240001-point reference integration. The independently refined
integral, including the valence rolloff, may differ slightly from ten; that
residual is reported rather than removed by another normalization.

Before this update, the separate unpartitioned 8+2 oscillator input gave
I = 86.38/81.16 eV (amorphous/hexagonal), versus Born's 77.13/72.47 eV.
Those are historical audit results, not alternative production inputs.

## Interpretation

### Thresholds versus fitted peak energies

Yoffe et al., ApJS 284:80 (2026), Table 2 and Appendix C.2
([paper](https://doi.org/10.3847/1538-4365/ae6b7a)), specify Bmin = 7 eV
and ionization thresholds Bi = 10, 13, 17, 32 eV for both ice phases.
The shared Born/oscillator input retains those thresholds. The hexagonal
ionization Drude centers are 15.8, 18.0, 24.5, 35.0 eV, not the Bi values.
The corresponding amorphous centers are 15.4, 18.6, 24.5, 38.0 eV.
Partitioning redistributes tails rather than deleting them at each Bi;
the total optical response need not show a step at an individual threshold.

The ion pipeline deliberately uses a different core from the electron paper:
the hydrogenic continuum starts at 543.4 eV, whereas the paper lists the
old optical K-shell threshold as 540 eV. This is not a valence-cutoff error.
Reference CSV endpoints are digitized curve endpoints, not fitted physical
thresholds; a difference in ELF onset/peak shape cannot by itself identify
a difference in the reference model's Bi.

Matias reports approximately seven electrons below 100 eV and three from
100 eV to 32 keV. **These are energy windows, not seven valence and three
core electrons.** Valence contributes above 100 eV; the K edge is higher.
The two points in the cumulative panels represent those approximate reported
sums, not experimentally measured error-free constraints.

`I (Matias, figure)` is estimated only over the digitized figure domain.
It is not an author-reported full Bethe I. Figure precision, sparse dash
endpoints, missing endpoint intervals, and the unresolved high-energy tail
limit this estimate. The extracted integral need not equal ten exactly.
Do not infer a normalization defect in the published calculation from this
figure extraction. No original analytic fit coefficients were available to
this benchmark.

The underlying ice optical measurements overlap those used by the existing
model (Matias Fig. 1 cites Emfietzoglou et al. 2007 fits). Agreement is therefore
not independent experimental validation. Differences in I and energy-window
strength identify an internal target-input discrepancy, but cannot establish
which finite-q extension or nonlinear-to-DCS mapping is correct.
