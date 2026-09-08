# Screened optical-oscillator polarization correction

## Status and limits

The local cluster runner checkpoints each completed loss integral within an
incident-energy row, with values, numerical errors and a completion mask.
Progress bars count losses and completed incident energies. Failed workers
signal the other workers to stop between loss integrals; saved work remains.
These execution changes do not alter the physical or convergence guards below.

**Experimental. Not a validated all-charge-state ice DCS replacement.**
The force/force-gradient calculation covers all 38 frozen states of H, He,
C, O and S, including neutrals, using existing atomic densities and the
existing ice optical oscillator-strength density (OOS). It requires no
electron-gas target, imported stopping curves, fitted effective charge,
new atomic calculations, or missing nonlinear-response dataset.

The unweighted numerical point-charge integrals reproduce SBETHE's I1/I2
combination within its quoted 0.1% fit tolerance at the tested arguments.
This verifies an oscillator calculation, not its identification with the
finite-q target excitation/ionisation DCS.

The attempted additive coupling has a **recorded physical acceptance failure**:
at 10 MeV/u, among the losses 15, 50 and 1000 eV, the correction exceeds
the actual ice Born DCS for He0, C0, C+, O0, O+, and S0 through S5+.
For He0 at 40 MeV total and W=15 eV in amorphous ice, Born is
7.99187125e-25 m^2/eV and the correction is 2.26841649e-24 m^2/eV
(ratio 2.8384, Nq=256). These are numerical model outputs, not measurements.
They demonstrate that the raw optical spectral assignment is not yet a
general replacement for a finite-q DCS derivation. Neither a correction
smaller than Born nor a positive final DCS establishes physical accuracy.

Generation rejects nonconvergence, v at or below the Bohr velocity, and
|correction| >= Born on any row with positive Born. No clipping, effective
charge, target-density adjustment, or fit to ICRU repairs these failures.
The retained integration is experimental and records this status in every
new screened-correction table. Existing bare Salvat and correction-off
calculations are unchanged. No published validation for screened ice DCS
or arbitrary projectile energies is claimed.

## Definition

The screened-oscillator precedent is [Schinner and Sigmund (2000),
NIMB 164-165, 220-229](https://doi.org/10.1016/S0168-583X(99)01181-7).
The original implementation used the accessible abstract and section excerpts.
The full paper has since been supplied and read. Its close-collision treatment
is now examined separately in [CLOSE_COLLISIONS.md](CLOSE_COLLISIONS.md);
this does not retroactively validate the original screened extension.
The equations below are an explicit generalization of the accessible
[Salvat (2022), PRA 106, 032809, Eqs. (91)-(100)](https://doi.org/10.1103/PhysRevA.106.032809).
Do not describe this code as a verified port of Schinner and Sigmund.

For a spherical, frozen projectile, Gauss's law gives

\[
g_Q(r)=Z-4\pi\int_0^r n_Q(s)s^2 ds
      =Q+4\pi\int_r^\infty n_Q(s)s^2 ds,\qquad
r g'_Q(r)=-4\pi r^3n_Q(r).
\]

Lengths are in bohr. The Gaussian/HF and analytic hydrogenic densities are
the same as in [PROJECTILE_FORM_FACTORS.md](../projectile_potentials/README.md).
g(0)=Z and g(infinity)=Q. A neutral's short-range field remains nonzero.
The complementary radial integrals avoid cancellation for neutral tails.

For impact parameter b, write tau=gamma*v*t/b, h=sqrt(1+tau^2),
x=omega*b/(gamma*v), omega=W/Eh in atomic units, and r=b*h. The dimensionless
force and its gradient are

\[
f_x=g/h^3,\quad f_z=\tau g/h^3,
\quad A=((2-\tau^2)g-rg')/h^5,
\quad D=\tau(3g-rg')/h^5,
\quad C=((2\tau^2-1)g-\tau^2rg')/h^5.
\]

The first-order displacement solves Y''+x^2Y=f with retarded initial
conditions. Define P=integral exp(i*x*tau)*f(tau) dtau and the second-order
impulse from the Fourier transform of (A*Yx+D*Yz, D*Yx+C*Yz).
Parity makes Px and Dx real and Pz and Dz imaginary. The implementation
stores their real coefficients and evaluates

\[
K_Q=\frac12\int_{\xi_a}^{\infty}
   \frac{P_xD_x+\gamma^{-2}P_zD_z}{x^2}\,dx,
\quad \xi_a=0.5616 C_B W/(\gamma\beta^2m_ec^2),\quad C_B=1.
\]

The force appears twice through P and Y, and once through its gradient.
Scaling the entire potential by lambda therefore scales this term by
lambda^3; it is not [Z-F(k)]^3 at one momentum or Q^3 times Salvat.
For a point charge g=Z, g'=0, K_Q=Z^3*(I1+I2/gamma^2).

The gamma-weighted longitudinal/transverse oscillator prescription follows
the point-projectile Jackson-McCarthy/Salvat construction. Extending it to
a rigid spherical cloud is an approximation, not a full covariant composite
projectile treatment. In particular no additional magnetic-force, internal
projectile transition, close-collision, Bloch or Lindhard-Sorensen DCS is added.

## Spectral assignment and table integration

As in the existing optical Barkas table construction, oscillator energy is
identified with the table's loss W and its stopping integrand is divided by W:

\[
\frac{d\sigma_{\rm corr}}{dW}
=\frac{4\pi r_e^2\alpha}{\gamma^2\beta^5}\frac{df}{dW}K_Q.
\]

This is an **OOS-equivalent spectral correction**, not a derived
channel-resolved quantum second-Born cross section. The optical OOS remains
normalized separately to 8 valence plus 2 O K-shell electrons per molecule;
the finite-q K-shell normalization is untouched. Input loss is in eV,
df/dW in eV^-1, r_e in cm, and cm^2/eV is converted once to m^2/eV.
The oscillator atomic units use a0=r_e/alpha^2 and Eh=alpha^2*mec^2,
without modifying the constants used by the existing Born generator.

The existing exact Barkas Wmax cutoff is retained. No Born Wmax, target
dispersion, phase density, or incident-energy convention is changed.
Accepted experimental corrections are added once before the existing TCS
integration and sampling-table path. Allocation to excitation and ionisation
columns remains explicitly bookkeeping. No C++ charge-state transport ladder
is added by this feature.

## Reproduction

### Independent proton comparison

```bash
python physics/inelastic_dielectric/polarization/benchmarking/compare_screened_barkas_point_projectiles.py --projectile proton --workers 10
```

This bypasses the analytic bare-state dispatch and integrates the numerical
point-proton force/gradient kernel. At 0.1, 0.3, 1, 3, 10, 30 and 100 MeV
total energy, using identical phase-specific ice OOS and Barkas Wmax,
the stopping contributions differ from the existing Salvat kernel by at
most 0.01438%. The largest pointwise DCS difference is 0.08313%.
Impact/time refinement changes the integrals by at most 0.01024%; the
kernel interpolation check is 0.00156%, and loss-grid refinement changes
the stopping moment by at most 0.02094%. These are numerical checks of
the bare-proton limit, not validation of screened-state ice DCS.

The plot and JSON report, including code hashes and convergence diagnostics,
are saved under `physics/inelastic_dielectric/polarization/benchmarking/plots/point_proton/`.
The plotted quantity is the Barkas contribution, not total stopping power.
Conversion from eV m^2 per H2O to MeV cm^2/g multiplies by
`1e4 * 1e-6 * N_A / M_H2O = 1e-2 * N_A / M_H2O`.

Use `--projectile alpha` for the corresponding bare He2+ comparison. This
evaluates the He2+ force/gradient kernel directly, without scaling the proton
curve. The same incident energies are total MeV per alpha, not MeV/u.
Outputs are kept separately in `physics/inelastic_dielectric/polarization/benchmarking/plots/point_alpha/`.
An additional fixed-argument check compares the independently calculated
He2+ and H+ kernels against their expected 8:1 cubic charge ratio.
The alpha comparison passes: maximum stopping-moment difference 0.00910%,
maximum pointwise DCS difference 0.07220%, and zero observed deviation from
the 8:1 kernel ratio at the three tested arguments. Impact/time refinement
changes the integrals by at most 0.01038%; the interpolation check is
0.00170%, and loss-grid refinement changes the moment by at most 0.02207%.
The low-energy points test agreement between formulas, not the physical
validity of a perturbative correction for slowly moving bare alpha particles.
Neither bare-state comparison validates He+ or neutral He corrections.

### Independent nonlinear all-state check

```bash
python physics/inelastic_dielectric/polarization/benchmarking/check_screened_oscillator_nonlinear.py --workers 10
python -m pytest -q tests/test_screened_oscillator_nonlinear.py
```

This benchmark solves the full nonrelativistic classical oscillator equation
for all 38 states of H, He, C, O, and S. The field is evaluated at the displaced
electron position. Independent numerical quadrature of the electron tail
supplies the Gauss-law field, using the same frozen density input but not the
production radial-charge, gradient, or impulse routines. Quadrature on 8193
and 16385 log-radius nodes checks interpolation and integration error. The
initial oscillator is at rest; projectile capture, stripping, and internal
excitation are excluded, as in the kernel under test.

Let `u=r/b`, `tau=v*t/b`, `x=omega*b/v`, `R=(1,tau)`, and
`eta=lambda/(b*v^2)` in atomic units. The exact oscillator equation is
`u''+x^2*u=eta*g(b*|R-u|)*(R-u)/|R-u|^3`. The dimensionless final energy
is `E=E_physical/(m_e*v^2)=eta^2*H(eta)`. The limit of
`[H(eta)-H(-eta)]/(2*eta)` as eta approaches zero is the cubic coefficient
`Px*Dx+Pz*Dz`. Positive and negative lambda scale the entire potential for
this derivative check; they are not fitted effective charges or physical
projectile charge-state changes.

The test uses eta=(0.04, 0.02, 0.01)/Z with both signs and two overlapping
Richardson extrapolations in eta^2. Z sets only the numerical derivative
step; it is not an effective-charge adjustment. DOP853 uses nominal/tight
relative tolerances 2e-11/2e-12; the incoming/outgoing time bounds are doubled
for every test case. The implemented impulse kernel uses phase-panel
orders 12 and 16, with window parameters 128 and 256. The zero-coupling
energy is also checked against independent Fourier-force quadrature to
infinite time.

For each state, b/r_rms=(0.2, 1, 2) and x=(0.1, 0.3, 1), where r_rms is the
RMS electron radius. Bare states use an arbitrary geometric scale of 1 bohr.
These 342 encounters resolve the interior and exterior of the frozen cloud;
they are not a scan of physical projectile energies or an ice target spectrum.
In the original stored audit, all passed the preassigned 0.1% relative
tolerance for every check. Its maximum
difference between the extrapolated nonlinear coefficient and the refined
implemented kernel is 0.00240%; the largest refinement change is 0.00824%.
Independent field normalization and bare Salvat tests also pass for all
states. The maximum bare-kernel discrepancy is 0.06442%, within the 0.1%
SBETHE representation tolerance, at beta=0.01 and 0.4.
Reports, code hashes, settings and the plot are stored in
`physics/inelastic_dielectric/polarization/benchmarking/plots/nonlinear_oscillator/`.
Those stored results predate removal of the fixed-grid integrator; their
provenance is retained rather than relabelled. A fresh run checks the
current phase-panel implementation against the same independent ODE.

This is an independent numerical verification of the screened oscillator
kernel across the requested states. It does not reproduce the
Schinner-Sigmund curves, validate screened relativity, or establish the conversion
from oscillator stopping integrand to finite-q ice DCS. The production
generator and its rejection rules are unchanged.

At the illustrative physical speed v=5 atomic units and lambda=1, 139 of
the 342 cases have |cubic/leading| >= 1. These are not small corrections,
even though their coefficients are computed correctly. These ratios concern a single oscillator encounter,
not the pipeline's Born DCS or an integrated ice stopping power. A ratio
below one is not, by itself, a physical accuracy criterion.

### Screened-state checks

```bash
python -m pytest -q tests/test_screened_barkas.py tests/test_barkas_dcs.py
```

The tests distinguish mathematical checks, successful limited table exports,
and the named physical rejections above. Passing the rejection tests must not
be reported as successful physical validation for those states.

The production integrator is `oscillator_quadrature.py`. Adaptive
Gauss-Kronrod quadrature resolves the impact integral in log(x). Time panels
are split both logarithmically around the encounter and by oscillator phase,
with at most pi phase advance per panel. Gauss-Legendre polynomial
antiderivatives evaluate the parity-resolved displacements at quadrature
nodes. Prefix and tail sums are accumulated in their respective directions;
there is no subtraction of two large global prefix sums.

Each impact evaluation compares polynomial orders at a fixed time window,
then doubles the window at fixed order. Windows end at complete oscillator
cycles to reduce phase-dependent truncation error. Absolute local time and
tail differences are integrated separately, so opposite signed errors cannot
cancel. Their sum plus the adaptive impact error must be below 0.01% of the
correction plus the existing dimensionless floor 1e-10*Z^3. The Born DCS is
not used as an error floor. These are numerical error estimates, not rigorous
bounds or estimates of physical model uncertainty. Unresolved integrals
still fail rather than falling back to the older method.

There is one oscillator integrator for generation and benchmarks. The
obsolete fixed-grid implementation has been removed. Independent checks
remain the nonlinear ODE solver and the Salvat point-charge formulas.
The existing maximum x=50,
Salvat lower impact cutoff, physical velocity restriction, and correction
versus Born guard are unchanged. Full bare states use the existing analytic
SBETHE path exactly. Distinct incident energies use the requested worker
count, default 10, with deterministic ordering.

`barkas_quadrature_method` and `barkas_quadrature_rtol` identify the upgraded
numerics in screened-table metadata. Old screened-correction caches must be
regenerated before combining energy patches. No existing products are
rewritten, and correction-off or bare Salvat tables are not invalidated by
this numerical upgrade.

Retest the representative H/He campaign using the production adaptive
calculation and evaluating both phases and both PWBA/RPWBA baselines:

```bash
python -m physics.inelastic_dielectric.polarization.benchmarking.check_integration_failures --workers 10 --dq 1000
python -m pytest -q tests/test_oscillator_quadrature.py tests/test_screened_barkas.py
```

The report in `benchmarking/runs/integration/integration_report.json` retains
individual loss nodes, all three numerical error estimates, the computed
kernels, physical acceptance failures, timings, and source/data hashes.
It samples 0.1, 0.3, 1, 3, 10, 30, 40, and 100 MeV **total** energy. It is
not a complete production-grid or experimental validation, and no DCS/TCS
tables are exported by this benchmark.

Experimental generation retains the existing CLI:

```bash
ICE_TYPE=amorphous python -u physics/inelastic_dielectric/generate_cross_sections.py \
  --projectile H --charge-state 0 --include-barkas-dcs=true \
  --energy-min-MeV 10 --energy-max-MeV 100 --energy-unit total \
  --dE 1000 --dq 1000
```

Add `--relativistic-projectile-dcs=true` for the existing RPWBA Born baseline.
This example is not a certification of every loss/incident-energy point at
that grid density; the runtime acceptance checks still apply.

Use `--projectile H|He|C|O|S` (one element, not the literal list) and
`--charge-state Q`, with Q in 0..1, 0..2, 0..6, 0..8, or 0..16 respectively.
The same selection applies to PWBA and RPWBA. With
`--include-barkas-dcs=true`, Q=Z uses the existing Salvat Barkas kernel;
Q<Z uses the experimental frozen-screened oscillator kernel. Setting it to
false keeps the selected fixed-state Born kernel without the additive term.
Do not combine fixed-state screening with scalar Zeff or explicit-charge
rescaling. The incident-energy convention is total kinetic energy per ion
unless `--energy-unit per_u` is explicitly selected.

Charge state, PWBA/RPWBA selection, and correction presence are distinguished
in output filenames and checked in cache/energy-patch metadata. This is table
generation support, not a claim that the C++ simulation transports every
charge state. In particular, the all-state oscillator benchmark does not
remove the rejected ice-DCS combinations listed above. A requested grid
containing such rows fails rather than emitting an approved-looking table.

NPZ diagnostics retain Born, correction and final DCS/TCS, the correction's
stopping moment, the estimated quadrature errors, atomic-density provenance
and the experimental model identity. Incompatible caches/energy patches are
rejected. Do not relabel these tables as validated screened-projectile data.
