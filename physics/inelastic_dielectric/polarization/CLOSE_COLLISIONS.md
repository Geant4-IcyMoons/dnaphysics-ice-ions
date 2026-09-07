# Close-collision comparison

**Experimental benchmark only.** The production generator and its Salvat
cutoff are unchanged. The alternative reuses our atomic densities, screened
force-gradient kernel, and phase-specific ice optical oscillator strengths.
It is not a newly validated ice DCS model.

## Published source and adaptation

Schinner and Sigmund, NIMB 164-165 (2000), 220-229, Section 4,
DOI [10.1016/S0168-583X(99)01181-7](https://doi.org/10.1016/S0168-583X(99)01181-7),
is now available in full. We adopt its induced-potential argument,
shifted-Coulomb small-impact asymptote, and intersection-based matching.

Their Eq. (23) combines *exponential* screening radii. We do not invent a
radius for our HF densities. Instead, we evaluate Eq. (17) for each
dispersionless oscillator. This is an explicit adaptation, not a literal
port of Eq. (23) or a reproduction of their figures.

In Hartree atomic units let A(k)=Z-F(k), kappa=omega/v. With the auxiliary
oscillator dielectric epsilon=1-omega^2/(Omega+i0)^2, the real angular
principal-value integration of Eq. (17) yields

\[
\Delta V(0)=\frac{\kappa}{\pi}\int_0^\infty
 \frac{A(k)}{k}\ln\left|\frac{k+\kappa}{k-\kappa}\right|dk.
\]

This mode approximation uses the same frequency as the optical oscillator,
not a new plasma-density parameter or the full finite-q ice dielectric.
For a bare charge it gives pi*Z*kappa/2, Eq. (18) with a_ad=v/omega.
For the potential -[q+(Z-q)exp(-r/a)]/r it gives exactly
`kappa*[pi*q/2+(Z-q)*atan(kappa*a)]`. Tests check both limits independently.
The latter differs from the paper's approximate radius combination.
The shift vanishes as omega tends to zero, also for a neutral projectile.

## Matching equations

The actual static potential is -Z/r+C_static+... near the nucleus, with
C_static=4*pi*integral r*n(r)dr, calculated from the existing density.
Use V0=C_static+Delta V in Eq. (24), then subtract that same expression at
V0=C_static. Thus static screening is not counted again as polarization:

\[
\Delta T_{close}(b)=4\Delta V(v^2b/Z)^2.
\]

The existing nonrelativistic force-gradient result is

\[
\Delta T_{distant}(b)=(P_xD_x+P_zD_z)/(b^3v^4).
\]

Their intersection defines b_m. We integrate the close expression below
b_m and the unchanged distant expression above it, without a blending
function or fixed matching radius. The close moment is analytic:
`2*pi*integral_0^bm b*DeltaT_close db = 2*pi*DeltaV*v^4*bm^4/Z^2`.
Missing or multiple detected intersections and nonconvergence fail explicitly.
No values are clipped or fitted to stopping data.

Both alternatives use **identical nonrelativistic kinematics**, consistent
with Section 4: v=sqrt(2*T/(M*E_h)) and
Wmax=2*E_h*v^2/(1+1/M)^2, with M in electron masses and T in total eV
per ion. The baseline cutoff is a=0.5616*C_B/v, C_B=1.
This isolates the impact prescription. It is not a comparison to the fully
relativistic production spectrum and does not modify PWBA or RPWBA.

## Ice spectrum and units

The unchanged ice OOS supplies 8 valence plus 2 core electrons per H2O.
The same experimental spectral assignment divides a mode's stopping moment
by its energy and multiplies by df/dW. For dimensionless integrated K:

\[
d\sigma_{equivalent}/dW=(4\pi/v^5)K(df/dW)a_0^2.
\]

W and df/dW are in eV and inverse eV; a0_cm^2*1e-4 converts area to m2.
The moment integral W*d_sigma/dW dW is in eV m2 per molecule; multiplying
by 1e-2*N_A/M_H2O converts it to MeV cm2/g. Tests check this identity.
The finite-q K-shell and Born normalizations are unchanged. This does not
derive a quantum channel-resolved DCS. No production DCS/TCS/CDF is exported.
The matched close contribution must not be described as homogeneous cubic
charge scaling at fixed impact parameter.

## Validity checks

Numerical convergence does not establish physical accuracy. Every loss node
records b_m/b90, b90=Z/v^2, and 2*(C_static+DeltaV)/v^2. Matching at b_m of
order b90 or larger is outside a demonstrably small-impact domain. A shift
ratio at or above one makes the effective kinetic energy nonpositive. These
are retained as explicit invalidity diagnostics, not accepted predictions.
The CSV also gives the fraction of the matched stopping moment from loss
energies that trigger either warning. These are not fractions of individual
collisions or quantitative error estimates. Close and distant spectra are
interpolated separately and summed, preserving their exact additivity.
Section 4 itself warns that the approximate crossover cannot support precise
experimental comparisons. Salvat's separate Lindhard-Sorensen close
correction must not be added here without a double-counting analysis.
The close model is classical: it is not controlled in the Born regime
Z/v << 1, which includes the high-energy proton examples. Equal
nonrelativistic kinematics isolates the prescriptions mathematically;
it does not establish that classical matching is appropriate at those
energies or supply a quantum close-collision correction.

## Reproduce

```bash
python -m pytest -q tests/test_close_collisions.py
python -m physics.inelastic_dielectric.polarization.benchmarking.compare_close_collisions \
  --workers 10 --loss-nodes 65
```

Defaults cover H+, He0, He+, C3+, O4+, S8+ at 10, 30, 100 MeV **total per
ion**, in both phases. Use --states and --energies-MeV to change the sample.
Add --reference-pdf PATH to record the source PDF checksum. The output
directory is `benchmarking/plots/close_collisions/`: PDF, CSV moments and
JSON with all kernel nodes, matching diagnostics, failures, and source hashes.
--plot-only redraws the report without recomputing its provenance.

PASS is numerical only: refined kernels/matching within 0.5%, nested kernel
interpolation change below 1%, and stopping-moment loss-grid change below
0.5%. Failed cases are recorded, never replaced with cutoff results.
Increasing --loss-nodes tests interpolation, not a different physical model.

## Initial comparison

The default sample resolves all 1170 loss nodes and yields 36 phase/state/
energy rows. The following changes are **polarization stopping only**, not
total electronic stopping: `100*(S_matched/S_cutoff-1)` in amorphous ice.
Hexagonal results and all diagnostics are in the accompanying CSV/JSON.

| Projectile | 10 MeV | 30 MeV | 100 MeV |
| --- | ---: | ---: | ---: |
| H+ | +106.35% | +92.99% | +84.88% |
| He0 | +53.30% | +49.45% | +44.17% |
| He+ | +78.02% | +73.00% | +67.89% |
| C3+ | -53.30% | -17.96% | +12.88% |
| O4+ | -68.99% | -47.19% | -13.00% |
| S8+ | -92.63% | -87.18% | -68.36% |

These changes are far larger than numerical errors. They are not evidence
of improved physical accuracy. For example, the nonpositive shifted-energy
warning accounts for about 45% of the C3+ matched moment at 10 MeV and
all of the O4+ and S8+ moments at 10 MeV. For H+ the shifted energy stays
positive, but the crossover is approximately 1.1*b90, outside a controlled
small-impact expansion. The retained cutoff must therefore not be replaced
on the strength of this comparison. Neither alternative is newly validated
by their disagreement.
