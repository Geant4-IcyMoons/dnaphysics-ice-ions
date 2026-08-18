# Carbon, lithium, oxygen, and sulfur CTMC provenance

## Code separation

- `generate_carbon_charge_exchange_ctmc.py` is the carbon entry point.
- `generate_lithium_charge_exchange_ctmc.py` is the Li-7 entry point.
- `generate_oxygen_charge_exchange_ctmc.py` is the oxygen entry point.
- `generate_sulfur_charge_exchange_ctmc.py` is the sulfur entry point.
- `charge_exchange_ctmc.py` is the shared numerical/parallel engine.
- Each element has a separate PBS script, output directory, checkpoint prefix,
  probability archive, cross-section table, and metadata file.

The equations are shared deliberately: maintaining multiple copies of the
trajectory solver would allow the elements to acquire different numerical
physics accidentally.

## Shared collision model

The three-body trajectory model, spherical independent-H2O target, water
orbital data, microcanonical sampling, event classification, modified
independent-event model, independent-particle model, and impact-parameter
integration follow:

T. Liamsuwan and H. Nikjoo, *Cross sections for bare and dressed carbon ions
in water and neon*, Physics in Medicine and Biology **58** (2013) 641--672,
[doi:10.1088/0031-9155/58/3/641](https://doi.org/10.1088/0031-9155/58/3/641).

The implementation retains the published many-electron expressions without
clipping their approximate IEVM weights. Numerical tolerances, random seed,
grid sizes, work partitioning, and checkpoint cadence are implementation
controls and are recorded in every metadata file; they are not presented as
published physical parameters.

### Initial-electron ensemble and reproduction modes

Liamsuwan and Nikjoo define a microcanonical energy-shell density in equation
(4), use equation (5) for its radial marginal, and compute the momentum
magnitude from equation (3). Section 2.1 does not say that the position and
momentum directions are independent; it delegates their selection to Olson
and Salop (1977). Olson and Salop equations (6)--(7) make the sampled momentum
tangent to the sampled radius, so `r dot p = 0`. The directions are
individually rotationally symmetric but correlated.

The default `liamsuwan_olson_salop_tangent` mode implements that cited
historical prescription and is the only paper-reproduction mode. It consumes
four random variates per trajectory: radius, two position angles, and one
tangent-plane angle.

R. E. Olson and A. Salop, *Charge-transfer and impact-ionization cross
sections for fully and partially stripped positive ions colliding with atomic
hydrogen*, Physical Review A **16** (1977) 531--541,
[doi:10.1103/PhysRevA.16.531](https://doi.org/10.1103/PhysRevA.16.531).

Cohen (1982) showed that this Olson--Salop construction is not uniform over
the complete microcanonical energy shell: it removes one initial-condition
degree of freedom, starts the electron at an orbital extremum, and biases the
orbit distribution. The separate `independent_isotropic_sensitivity` mode
therefore retains the full equation-(4) interpretation by drawing position
and momentum directions independently after the equation-(5) radial sample.
It is a corrected-ensemble sensitivity calculation, not a reproduction of
the cited Olson--Salop initialization or CTMC81:

J. S. Cohen, *Comment on the classical-trajectory Monte Carlo method for
ion-atom collisions*, Physical Review A **26** (1982) 3008--3010,
[doi:10.1103/PhysRevA.26.3008](https://doi.org/10.1103/PhysRevA.26.3008).

The modes have different random-stream strides and configuration signatures;
their checkpoints cannot be combined. A 2,000-trajectory-per-impact
diagnostic changed only this angular prescription. Relative to the supplied
CTMC81 primitive tables, the tangent mode reduced the RMS z score from 12.466
to 1.390 and simultaneous-band failures from 560/1640 to 25/1640. This
identifies the dominant discrepancy but is not a release result; acceptance
requires the exact-size tangent-mode reproduction.

### Carbon benchmark observables

`../benchmarking/benchmark_carbon_charge_exchange_ctmc.py` reproduces the three published
observables that are directly available from the charge-exchange output:

| Paper result | Repository quantity |
|:---|:---|
| Figure 12, pure single electron capture | `sigma_SC_cm2`, C1+--C6+ |
| Figure 13, pure projectile electron loss | `sigma_SL_cm2`, C0--C5+ |
| Figure 14, equilibrium charge fractions | equation (23), using `SC+TI` for charge decrease and `SL+LI` for charge increase |

Figure 12's powers-of-ten display factors are reproduced only in the plot;
they never modify a stored cross section. The article provides plotted curves
but no machine-readable numerical table. The benchmark keeps its manually
digitized comparison points, source-pixel coordinates, and digitization
uncertainty in a dedicated provenance file; no point is fitted to the present
calculation. Numerical table acceptance remains a separate calculation.

The user-supplied `C3_100keVpu.zip` is a second, independent benchmark input.
It contains six `CTMC81` impact-probability tables for C3+ at 100 keV/u:
target ionization/capture for orbitals L1--L5 and projectile electron loss.
The retained archive SHA-256 is
`d6f40fd2da4b0acbda6626b1d08b9271a92d356c46aab28d4346e27eca3a80dd`.
The validation report records every member hash and header parameter before
comparing the primitive curves. The archive contains no citation or license
metadata, so it is identified as a supplied formal reference rather than
silently attributed to the paper. Its residuals remain informational until
that provenance and a physical acceptance threshold are documented.

The benchmark also verifies the reported channel identities, terminal charge
states, equation-(23) stationarity, adaptive error metadata, trajectory energy
acceptance, and reintegration of every table row from the saved primitive
probabilities. Impact-domain convergence is accepted only against an
independently calculated expanded-cutoff table. Probabilities at the supplied
cutoffs are reported as diagnostics but are not compared with an invented
probability threshold.

## Screened-core parameters

The screened potential is the Green--Sellin--Zachor/Garvey form. The complete
coefficient source is Table I of:

R. H. Garvey, C. H. Jackman, and A. E. S. Green,
*Independent-particle-model potentials for atoms and ions with 36 < Z <= 54
and a modified Thomas-Fermi atomic energy formula*, Physical Review A **12**
(1975) 1144--1152,
[doi:10.1103/PhysRevA.12.1144](https://doi.org/10.1103/PhysRevA.12.1144).

Garvey's `N` is the total number of electrons in the modeled system. The CTMC
core excludes its one active electron, so the engine indexes the same table
by `spectators = N - 1`. Garvey tabulates `10*xi1` and `10*eta1`; the stored
slopes are therefore divided by ten. In the code, Garvey's `xi` is named
`zeta`, matching the notation used in the carbon implementation.

The oxygen and sulfur additions are consequently direct table values:

| Spectators | Garvey `N` | Configuration | zeta0 | zeta1 | eta0 | eta1 |
|---:|---:|:---|---:|---:|---:|---:|
| 7 | 8 | 2p4 | 1.360 | 0.4613 | 2.410 | 0.3925 |
| 8 | 9 | 2p5 | 1.508 | 0.4602 | 2.590 | 0.3755 |
| 11 | 12 | 3s2 | 1.492 | 0.3452 | 3.010 | 0.3269 |
| 12 | 13 | 3p1 | 1.170 | 0.3191 | 3.170 | 0.3087 |
| 13 | 14 | 3p2 | 1.012 | 0.2933 | 3.260 | 0.2958 |
| 14 | 15 | 3p3 | 0.954 | 0.2659 | 3.330 | 0.2857 |
| 15 | 16 | 3p4 | 0.926 | 0.2478 | 3.392 | 0.2739 |
| 16 | 17 | 3p5 | 0.933 | 0.2368 | 3.447 | 0.2633 |

No coefficient is interpolated or fitted in this repository.
Lithium uses the already-tabulated spectator rows 0, 1, and 2; no new
screening coefficient is introduced for it.

## Ice-phase density

The Geant4 implementation follows the repository's phase-selection mechanism
introduced in commits
[`349f93b04`](https://github.com/Geant4-IcyMoons/dnaphysics-ice-ions/commit/349f93b0491eb978801b3b355e3fe7434679c077)
and
[`d8bf91720`](https://github.com/Geant4-IcyMoons/dnaphysics-ice-ions/commit/d8bf91720cfb72d94542809b99a8d4b4c7420413):

| `DNA_PHYSICS` | Geant4 material | Mass density (g/cm3) |
|:---|:---|---:|
| `ice_am` | `G4_WATER_ICE_AM` | 0.940 |
| `ice_hex` | `G4_WATER_ICE_HEX` | 0.9335 |
| `water` | `G4_WATER` | 1.000 |

The ice-Ih value is the rounded 100 K density obtained from the corrected
H2O lattice-volume polynomial of Rottger et al. (2012),
<https://doi.org/10.1107/S0108768111046908>.

The CTMC tables remain the same microscopic cross sections
`sigma(E,q)` in cm2 per H2O molecule for every phase. Geant4 obtains the
macroscopic interaction coefficient from the selected material:

`Sigma(E,q) = n_H2O sigma(E,q)`,

where `n_H2O = rho N_A / M_H2O`. Thus phase density changes the interaction
rate and mean free path, not the isolated-molecule collision physics. Applying
the density to the stored microscopic tables as well would double count it.
The shared output writer records the three densities and molecular number
densities in every carbon, lithium, oxygen, and sulfur metadata file.

### Runtime implementation status

The proton/alpha pipeline registers Geant4 11.3.2's Dingfelder charge-increase
and charge-decrease models. Those models calculate
`CrossSectionPerVolume` as the microscopic H2O cross section multiplied by
the H2O molecular number density returned for the current material by
`G4DNAMolecularMaterial`. Both the main application and proton/alpha pipeline
construct `G4_WATER_ICE_AM` and `G4_WATER_ICE_HEX` from `G4_WATER` at the
densities above. Proton/alpha charge-exchange rates therefore scale as
`1.000:0.940:0.9335` for water, amorphous ice, and hexagonal ice at the same
microscopic cross section; their mean free paths scale inversely.

The carbon, lithium, oxygen, and sulfur code in this repository currently
generates microscopic CTMC tables. It does **not** yet register a Geant4
process that reads those tables and changes the transported ion's charge
state. Consequently, density scaling for those ions is a required interface
contract recorded in their tables and metadata, not a claim that heavy-ion
charge exchange is already active in Geant4. A future table model must return
`n_H2O(material) * sigma(E,q)` from `CrossSectionPerVolume`; multiplying the
stored table itself by a phase density would be an error.

## Carbon data

The C0--C5+ binding energies and outer-shell occupancies are Table 1 of
Liamsuwan and Nikjoo (2013). C6+ is fully stripped. Sampler choice is part of
the checkpoint signature. Existing carbon tables generated with the
independent-isotropic sensitivity ensemble cannot be resumed or released as
paper-reproduction products.

The numerical engine uses 100,000,000 accepted plus rejected steps as the
liveness trigger for each ordinary DOP853 attempt. An exact adaptive-grid
regression at 1 keV/u required
53,384,632 attempts to return to the unchanged 20,000-a.u. boundary and had
relative total-energy drift `2.65e-6`; the ceiling exceeds that measured
requirement by 87%. A capped attempt supplies no event:
the identical initial phase is retried with the documented tighter tolerance
or positive Sundman parameterization. If every bounded attempt returns no
finite endpoint, the identical Sundman-regularized problem continues without
a step ceiling because the paper defines no such physical boundary. Only an
endpoint satisfying the paper's exit condition and the independent
energy-conservation check is accepted. The finite value therefore selects the
rare-event fallback without truncating the Monte Carlo ensemble.

The first failure exposed by the independent-isotropic sensitivity run is
retained as a deterministic recovery benchmark: 1 keV/u, C3+, zero impact parameter,
H2O 1a1 target electron, trajectory 0. The uncapped Sundman continuation
reached the unchanged outbound 20,000-a.u. boundary in 139,593,283 accepted
and 21,722,557 rejected steps, with relative total-energy drift `4.36e-5`.
This is below the unchanged `1e-3` acceptance limit; no event, phase,
potential, or physical boundary was altered.

DOP853 advances the autonomous Newtonian system relative to the current state,
while a compensated sum records elapsed physical time. Consequently, the
local step has no absolute machine-epsilon floor. Such a floor formerly
rejected two finite-periapsis production trajectories even though their state
increments remained representable. Removing it changes no equation, potential,
initial condition, or event definition. The retained deterministic regressions
complete those trajectories in 1,410,020 and 1,025,708 accepted-plus-rejected
steps, respectively, with relative total-energy drifts `9.43e-6` and
`2.89e-6`; the exact step counts are asserted only to remain below the
2,000,000-step regression budget because compiler-level arithmetic may alter
the detailed adaptive sequence.

Integrator policy version 3 corrects one failover condition in this declared
ladder. A finite physical-time endpoint outside the unchanged energy gate is
not an acceptable result and therefore no longer prevents the already-defined
uncapped Sundman continuation after every bounded regularized attempt is
exhausted. This changes no initial phase, equation, potential, event,
boundary, tolerance, or acceptance limit.

The restart controller treats numerical recovery as a versioned algorithm,
not an invitation to edit code automatically. A failure generated under an
older integrator policy is eligible for bounded restart with the current
predeclared fallback ladder. A failure generated after that ladder is
exhausted is quarantined. No automated path changes an initial condition,
potential, event definition, boundary, failure fraction, or energy-drift
threshold.

## Lithium data

The lithium process definitions come from:

N. D-Kondo et al., *Lithium inelastic cross-sections and their impact on
micro and nano dosimetry of boron neutron capture*, Physics in Medicine and
Biology **69** (2024) 145016,
[doi:10.1088/1361-6560/ad5f72](https://doi.org/10.1088/1361-6560/ad5f72).

The paper defines four charge states and six adjacent transitions:

| Direction | Published transitions |
|:---|:---|
| Electron capture / charge decrease | Li1+ -> Li0, Li2+ -> Li1+, Li3+ -> Li2+ |
| Electron loss / charge increase | Li0 -> Li1+, Li1+ -> Li2+, Li2+ -> Li3+ |

It calculates ten logarithmically evenly spaced energies from 1 keV/u through
10 MeV/u (7 keV through 70 MeV total Li-7 energy). These are the lithium
entry point's first ten points. The requested extension appends one directly
calculated endpoint at exactly `100000/7 keV/u`, corresponding to 100 MeV
total Li-7 energy. It does not move or replace any published point and does
not fit or extrapolate a cross section. The resulting 70--100 MeV interval is
identified in metadata as a new CTMC calculation rather than validation
claimed by D-Kondo et al.

The paper held charge-exchange cross sections constant below 1 keV/u in its
transport model. This repository does not turn that transport continuation
into CTMC data: generated tables declare 1 keV/u as their lower validity
limit.

The paper shows its cross sections in Figure 3 but does not provide numerical
values in a table, supplementary machine-readable data, or public source code.
The repository therefore recomputes the published process rather than
digitizing the plot or fitting an undocumented curve.

The Li0--Li2+ ground configurations and ionization energies are from NIST
Atomic Spectra Database 5.12, SRD 78,
[doi:10.18434/T4W30F](https://doi.org/10.18434/T4W30F):

| Charge | Ground outer subshell | Active electrons | Ionization energy (eV) |
|---:|:---|---:|---:|
| 0 | 2s1 | 1 | 5.391714996 |
| 1 | 1s2 | 2 | 75.6400970 |
| 2 | 1s1 | 1 | 122.45435913 |

Li3+ is fully stripped. The bare Li-7 nuclear mass is `12786.3922820`
electron masses. It is derived from the
[NIST Li-7 neutral-atom relative mass](https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ascii=ascii&ele=Li),
`7.0160034366(45) u`, by subtracting three electron masses and restoring the
summed `203.486171126 eV` electronic binding energy.

D-Kondo et al. used Boost's Runge--Kutta--Fehlberg 7/8 implementation and
reported global and local errors of `1e-10` and `1e-7`. The shared engine uses
adaptive DOP853. Its `rtol` and `atol` are numerical convergence controls, not
ad-hoc physical coefficients, and are not equated to the differently defined
RKF error controls.

The normal trajectory path uses physical-time DOP853. If that integrator
cannot cross an exceptional near-Coulomb encounter at any of its three
disclosed tolerances, or if all three finite endpoints fail the independent
energy-conservation limit, the identical initial phase is retried with the
standard positive Sundman reparameterization `dt/ds = min(1, r_min)`, where
`r_min` is the shortest instantaneous pair distance. This changes the
independent integration variable, not Newton's equations, the Garvey
potentials, the random phase, the event definition, or the acceptance limit.
To prevent long bound-orbit roundoff from consuming the energy budget, a
common relative-velocity projection onto the initial Hamiltonian surface
activates only when drift reaches one quarter of the configured final limit.
The endpoint is still rejected unless an independent total-energy calculation
satisfies the full configured limit. No failed trajectory is discarded or
replaced by a new random phase.

Luna et al., Physical Review A **93** (2016) 052705,
[doi:10.1103/PhysRevA.93.052705](https://doi.org/10.1103/PhysRevA.93.052705),
measured direct double capture Li3+ -> Li1+ in H2O at 0.75--5.8 MeV total
projectile energy. That channel is real, but it is not one of D-Kondo et al.'s
six charge-exchange processes. It is deliberately not inferred or mixed into
this implementation; adding it requires a separately sourced q -> q-2 table
and explicit Geant4 branching.

## Oxygen data

The O0--O7+ ground configurations and ionization energies are from NIST
Atomic Spectra Database 5.12, SRD 78,
[doi:10.18434/T4W30F](https://doi.org/10.18434/T4W30F):

| Charge | Ground outer subshell | Active electrons | Ionization energy (eV) |
|---:|:---|---:|---:|
| 0 | 2p4 | 4 | 13.618055 |
| 1 | 2p3 | 3 | 35.12112 |
| 2 | 2p2 | 2 | 54.93554 |
| 3 | 2p1 | 1 | 77.41350 |
| 4 | 2s2 | 2 | 113.8990 |
| 5 | 2s1 | 1 | 138.1189 |
| 6 | 1s2 | 2 | 739.32697 |
| 7 | 1s1 | 1 | 871.4099138 |

O8+ is fully stripped. The bare O-16 nuclear mass already used by the
projectile library is `29148.9497` electron masses. It is derived from the
NIST neutral-atom relative mass `15.99491461957(17) u` by subtracting eight
electron masses and restoring the total electronic binding energy,
`2043.8429988 eV`, obtained by summing the NIST O0--O7+ successive ionization
energies. This distinguishes the bare nucleus from the neutral atomic mass.

For non-carbon projectiles, the checkpoint signature fingerprints the nuclear
charge, bare nuclear mass, outer-orbital data, and every required Garvey
screening row. A physical-data correction therefore cannot silently resume a
checkpoint generated with different projectile physics. Model version 11
gives both declared ensembles explicit identities and rejects the earlier
tangent model-version-9 and independent-isotropic model-version-10
checkpoints. Non-carbon signatures inherit the same barrier, so no species can
silently combine ensembles.

CTMC has also been applied independently to bare O8+ impact on water by
A. Jorge et al., Physical Review A **99** (2019) 062701,
[doi:10.1103/PhysRevA.99.062701](https://doi.org/10.1103/PhysRevA.99.062701).
That paper is applicability evidence, not a source of substituted parameters
for the Liamsuwan--Nikjoo model.

## Sulfur data

The S0--S15+ ground configurations and ionization energies are from NIST
Atomic Spectra Database 5.12, SRD 78,
[doi:10.18434/T4W30F](https://doi.org/10.18434/T4W30F):

| Charge | Ground outer subshell | Active electrons | Ionization energy (eV) |
|---:|:---|---:|---:|
| 0 | 3p4 | 4 | 10.3600167 |
| 1 | 3p3 | 3 | 23.33788 |
| 2 | 3p2 | 2 | [34.86] |
| 3 | 3p1 | 1 | [47.222] |
| 4 | 3s2 | 2 | 72.5945 |
| 5 | 3s1 | 1 | 88.0529 |
| 6 | 2p6 | 6 | 280.954 |
| 7 | 2p5 | 5 | 328.794 |
| 8 | 2p4 | 4 | [379.84] |
| 9 | 2p3 | 3 | [447.7] |
| 10 | 2p2 | 2 | [504.55] |
| 11 | 2p1 | 1 | [564.41] |
| 12 | 2s2 | 2 | [651.96] |
| 13 | 2s1 | 1 | (706.994) |
| 14 | 1s2 | 2 | (3223.78057) |
| 15 | 1s1 | 1 | (3494.188518) |

Square brackets and parentheses reproduce the NIST ASD status notation; the
engine stores the corresponding numeric recommended values. S16+ is fully
stripped.

The bare S-32 nuclear mass is `58265.5417` electron masses. It is derived from
the NIST neutral-atom relative mass `31.9720711744(14) u` by subtracting
sixteen electron masses and restoring the total electronic binding energy,
`10859.5983847 eV`, obtained by summing the NIST S0--S15+ successive
ionization energies.

CTMC calculations spanning S- through S16+ impact on H2 and 1--25000 keV/u
provide independent element-level applicability evidence:

D. R. Schultz et al., *Data for secondary-electron production from ion
precipitation at Jupiter IV: Simultaneous and non-simultaneous target and
projectile processes in collisions of Sq+ + H2 (q=-1--16)*, Atomic Data and
Nuclear Data Tables **142** (2021) 101443,
[doi:10.1016/j.adt.2021.101443](https://doi.org/10.1016/j.adt.2021.101443).

Those sulfur-H2 calculations use a different target and event construction;
they are not substituted into this sulfur-H2O extension. A recent Europa
water-torus analysis explicitly reports finding no sulfur-ion/H2O
cross-section data:
[doi:10.1029/2024GL112110](https://doi.org/10.1029/2024GL112110).
Sulfur-H2O results from this engine must therefore be identified as
Liamsuwan--Nikjoo-model predictions rather than experimentally validated
cross sections.

## Numerical-domain convergence

Impact cutoffs, initial separations, and integration tolerances are numerical
domain controls, not new interaction definitions. The engine requires the
cutoffs and per-energy separations explicitly, rejects unphysical both-bound
endpoints, retries the identical initial phase at larger boundaries, and
rejects trajectories that exceed the disclosed energy-drift limit.

The lithium paper does not publish its impact cutoffs, per-energy start
separations, trajectory count, or numerical cross-section table. Its PBS
script therefore requires the cutoffs and all eleven start separations
explicitly. Paired boundary/statistical convergence runs are required before
lithium production; the published Figure 3 supplies an independent
order-of-magnitude and trend check.

For oxygen, using the largest published carbon-domain cutoffs for every
channel is conservative: it evaluates additional zero-probability domain
rather than removing a physical channel. Boundary probabilities in the
production archive must still be checked; a non-negligible boundary value
requires extending the domain and rerunning from a new output directory.

Sulfur has no published H2O impact cutoffs or initial separations for this
model. Its PBS script consequently requires these controls explicitly and
provides no sulfur production defaults. They must be selected by paired
boundary-convergence calculations, with the resulting controls and boundary
probabilities recorded before production.
