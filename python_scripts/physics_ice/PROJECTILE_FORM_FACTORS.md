# Fixed-state projectile screening in PWBA and RPWBA

## Scope

`--charge-state Q` selects a frozen, spherical projectile electronic density.
It covers H (Q=0..1), He (0..2), C (0..6), O (0..8), and S (0..16): 38
states, including neutrals and bare nuclei. Negative ions and excited or
metastable configurations are outside this library. Omitting the option
preserves the existing bare-projectile calculation.

This is the coherent, projectile-elastic contribution to **target electronic
excitation and ionisation**, not a complete collision or transport model.
The projectile retains its internal state during this contribution. Target
valence and O K-shell kernels, phase densities, and normalization are unchanged.

## PWBA definition

Let Z be nuclear charge, Q the integer net ionic charge, and N=Z-Q the bound
electron count. With r in bohr and k in inverse bohr, the form factor is

\[
F_Q(k)=4\pi\int_0^\infty r^2 n_Q(r)\frac{\sin kr}{kr}\,dr,
\qquad 4\pi\int_0^\infty r^2n_Q(r)\,dr=N.
\]

The existing nuclear-charge Born prefactor is modified **inside** every
momentum-transfer integral:

\[
Z^2\longrightarrow |Z-F_Q(k)|^2.
\]

This is the fixed-state charge-form-factor construction used in dielectric
collision theory; see [de Vera et al. (2023), Eq. (2)](https://doi.org/10.3389/fmats.2023.1249517).
It is neither an energy-only effective charge nor Q squared at every momentum.
At k=0 the amplitude is Q, whereas at high k it approaches Z. A neutral
projectile therefore has a nonzero interaction at finite k. Its low-k
amplitude is k squared times the second radial moment divided by six, so
its squared amplitude vanishes as k to the fourth power. The implementation
evaluates N-F directly to avoid cancellation in this limit.

Bare states have F=0. One-electron ground states use the analytic
nonrelativistic hydrogenic 1s density and transform:

\[
n(r)=\frac{Z^3}{\pi}e^{-2Zr},\qquad
F(k)=\left[1+\left(\frac{k}{2Z}\right)^2\right]^{-2}.
\]

No empirical target-dependent or stopping-power normalization is applied to
these projectile densities.

## Multielectron densities

`generate_projectile_atomic_data.py` computes the other 28 states with
[PySCF](https://pyscf.org/user/scf.html), version 2.14.0 for the supplied data.
Closed shells use RHF; open shells use high-spin ROHF with the ground
configuration filled through 3p. The orbital space includes occupied angular
momenta, s and p, and fully uncontracted radial primitives. The production
basis is the union of cc-pCV5Z and aug-cc-pV5Z primitives; He uses cc-pV5Z
and aug-cc-pV5Z. Duplicate exponents are removed by PySCF. The corresponding
QZ union provides a convergence comparison. No effective core potential is used.

The exported density is the spherical monopole of the converged determinant.
In open shells this is not an exact LS-coupled atomic term calculation:
orbital relaxation within the ROHF determinant precedes angular averaging.
Squaring this monopole does not include the full orientation-averaged
multipole contribution.
Electron correlation, spin-orbit structure, relativistic atomic contraction,
and an ensemble of metastable populations are not represented. In particular,
basis convergence does not bound these physical approximations.

The data are newly computed HF densities, **not** digitized Clementi--Roetti
coefficients. The classic [Clementi and Roetti (1974) compilation](https://doi.org/10.1016/S0092-640X(74)80016-1)
provides historical context, not the numerical input or an all-state external
validation. Do not describe these data as that compilation.

`projectile_atomic_data.json` stores the signed Gaussian expansion, basis
identity and checksum, electronic energy, configuration, spin, PySCF version,
SCF tolerances, electron count, and QZ/5Z comparison. Each term is proportional
to r^(2l) exp(-a r^2), with its integrated electron weight recorded explicitly.
Its Fourier transform is analytic. A log-grid PCHIP interpolator accelerates
production evaluation; outside its range the analytic expression is used.
Neither interpolation nor the export renormalizes the atomic density.

## RPWBA extension and its limitation

The target kernel remains the existing finite-Q Dominguez-Munoz implementation
and its medium transverse term. The published proton application is
[Dominguez-Munoz et al. (2022)](https://doi.org/10.1016/j.radphyschem.2022.110363),
not a validation of every ion charge state.

For an electron-bearing projectile we add an **elastic electric form-factor
approximation** to that kernel. Both longitudinal and transverse charge-current
terms are multiplied by the same squared charge amplitude, evaluated at

\[
\kappa=\sqrt{k^2-\left(\frac{W}{\hbar c}\right)^2}.
\]

This is the Breit/static-density prescription for spacelike four-momentum
transfer, not a new result taken verbatim from the proton paper. In the
straight-line, negligible-projectile-recoil limit W=\hbar v k_parallel,
it equals sqrt(k_perp^2+k_parallel^2/gamma^2), the rest-frame wave number
of a rigid Lorentz-contracted cloud. At finite recoil the static density is
an approximation to an elastic electric form factor. Magnetic form factors,
internal projectile transitions, and covariant many-electron response are
omitted. Do not claim a full relativistic composite-projectile calculation.

The helper uses W in eV, k in inverse bohr, EH in eV, and c in atomic units.
It rejects timelike transfer. The screening factor is dimensionless, leaving
the existing DCS units, m^2/eV, unchanged. Incident energy defaults to total
kinetic energy; `--energy-unit per_u` converts the input using the isotope
mass number before projectile kinematics. Explicit state runs add bound
electron masses and the HF binding energy to the repository's bare mass.
Existing Wmax, recoil bounds, target dielectric functions, and bare defaults
are not replaced by this feature.

## Generation and compatibility

Run from the repository root. For example, C3+ in amorphous ice, PWBA:

```bash
ICE_TYPE=amorphous python -u python_scripts/physics_ice/generate_ice_cross_sections_ion.py \
  --projectile C --charge-state 3 \
  --energy-min-MeV 1 --energy-max-MeV 100 --energy-unit total \
  --dE 1000 --dq 1000
```

Add `--relativistic-projectile-dcs=true` for the RPWBA extension. Select H,
He, C, O, or S and Q=0 for neutrals, Q=Z for bare nuclei. The projectile
name selects the nucleus; an explicit Q overrides the charge suggested by
legacy aliases such as alpha or C6+. Oxygen K-shell inclusion is the existing
default and is independent of the projectile's bound electrons.

State-specific filenames contain `_qQ_frozen_hf`; RPWBA keeps its existing
additional tag. DAT headers and NPZ caches record nuclear Z, net Q, bound
electron count, state model, atomic-data checksum, and momentum convention.
Cache reuse and energy-patch merging reject incompatible state metadata.
The screened final DCS follows the existing TCS integration and Geant4
sampling-table path. Both multiprocessing initializers carry the selected Q.

`--charge-state` cannot be combined with scalar `--charge-mode zeff` or
`explicit`, or with a non-bare input-reference rescaling. Electron-bearing
states also reject `--include-barkas-dcs=true`: substituting Z-F into a Z^3
point-charge Barkas formula is not derived here. The existing bare-nucleus
Barkas option remains available. No Bloch DCS is added.

These new files are **not automatically registered as a complete charge-state
ladder in the C++ transport model**. C++ particle definitions, state-specific
table selection, and capture/loss transitions must be connected separately.
Do not alias a screened table to a bare table filename to bypass that step.

## Reproduction and validation

PySCF is a build-only dependency; table generation needs the supplied JSON,
NumPy, and SciPy, not a quantum-chemistry solver.

```bash
python -m pip install pyscf==2.14.0
python python_scripts/physics_ice/generate_projectile_atomic_data.py --workers 10
python -m pytest -q tests/test_projectile_form_factors.py
python python_scripts/physics_ice/process_evidence/electronic_excitation_ionisation/benchmarking/benchmark_projectile_form_factors.py --workers 10
```

The builder fails on SCF failure, incorrect electron count, a negative
radial density, disagreement with independently evaluated PySCF AO densities,
or a QZ/5Z squared-charge change above 0.5% on the recorded grid. No density
is silently substituted for a missing or invalid state.

The benchmark checks all states against independent radial quadrature,
hydrogenic transforms where applicable, and analytic charge limits. It
compares production excitation, ionisation, and K-continuum DCS at Nq=1000
and 2000, in both phases, for PWBA and RPWBA at 1, 10, and 100 MeV/u.
Plots and the machine-readable report are written to the ignored directory
`plots/diagnostics/projectile_form_factors/`. Tests additionally check table
integration and incompatible-cache rejection.

An external neutral-He check uses [Koga (1990), Eq. (7) and Tables I--II](https://doi.org/10.1103/PhysRevA.41.1274):
the published three-exponential HF density approximation, near-HF-limit
energy, and per-electron radial moments. Its rounded coefficients are used
only in the benchmark. This is an independent atomic reference for He,
not external all-charge-state or experimental stopping validation.

Passing these checks establishes numerical consistency, not agreement with
measured fixed-charge stopping or neutral collision data. Capture, stripping,
projectile excitation/ionisation, dynamic polarization, projectile--target
electron exchange, and nuclear scattering remain separate physical channels.
The first-Born approximation is not made valid at low velocity by adding a
form factor. Charge-equilibrium ICRU stopping is not a direct benchmark for
a single fixed-Q contribution.
