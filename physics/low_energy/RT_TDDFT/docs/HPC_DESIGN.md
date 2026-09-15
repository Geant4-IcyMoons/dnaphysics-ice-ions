# Scientific design for HPC collision calculations

Status: proposed design, 2026-09-15. The molecular capture-analysis code is implemented and tested on analytical
and synthetic data. No HPC jobs, periodic ice collision inputs, embedding, or
validated physical channel probabilities have been implemented.
The five-water execution attempts are incomplete. Completed
single-water checks establish execution and initial-state preparation only.


## Adopted workflow: reproduce, then extend

Follow one published protocol for each observable before modifying its target.
Create a parameter ledger linking geometry, functional, pseudopotentials,
propagator, nuclear dynamics, boundaries, initial conditions, and analysis to
specific equations/sections or supplied author inputs. Resolve missing settings
explicitly; current Octopus defaults are not evidence of the historical setup.
Use the publication's solver where available, or demonstrate the equivalence of
the implementation used. Preserve the reference case when adapting to ice.

For the molecular reference, Hong's Sections II--III use spin-resolved LDA,
eight active electrons, Ehrenfest nuclei, and determinant-overlap counting
with coordinate/momentum translations. At 1 keV, reproduce the Figure 2
impact-parameter curves before orientation-averaged cross sections. Their
reported grid and time steps are 0.33 bohr and 0.025 atomic time units. Our
sphere-integral prototype does not implement that protocol. Their ionization
estimate subtracts capture from electron loss and has acknowledged limitations;
it does not resolve every exclusive channel.

For bulk stopping, reproduce a periodic liquid-water reference and its sampling
and finite-size checks before preparing phase-specific ice counterparts.
For heavy ions, reproduce a specified published projectile-state preparation
before extrapolating its implementation to the required charge configurations.
No new embedding approximation is adopted as the starting production method.

## 1. Keep the physical objective explicit

The objective is ionization, excitation, and charge exchange for H and later
C/O/S in amorphous and hexagonal ice. Electronic stopping is an independent
validation observable, not a replacement for these channels. Interpret the
stated 100 eV--100 keV range as total projectile kinetic energy unless changed;
record both total energy and energy per nucleon, mass, velocity, initial charge,
and electronic configuration in every case.

For a finite collision target, outgoing projectile-bound electrons, outgoing
vacuum electrons, and the residual target can be separated asymptotically.
For periodic bulk ice, molecular ionization leaves holes and excited electrons
inside the material. There is no vacuum region into which an electron must
escape to count as ionized. Exciton formation, electron--hole separation,
localization, and the partition between projectile and medium complicate
classification. Specify the intended operational observables before generating
large datasets. A mean population is not an exclusive event probability.

## 2. Use our structures with traceable boundary conditions

The accepted amorphous model is one neutron-constrained EPSR realization with
3,000 waters at 80 K. The accepted hexagonal models contain 8,192 waters in
three 100 K replicas. Preserve these as the structural reference, including
source hashes and the distinction between EPSR and equilibrated MD structures.

For bulk stopping, use fully quantum periodic cells and test their size and
shape. Modest cells of order 100--200 waters are a starting cost estimate, not
a demonstrated convergence threshold for either ice phase or any projectile.
The existing archived structures do not provide an already validated smaller
periodic counterpart. A smaller cell must be prepared under consistent periodic
conditions and checked against the reference density, intermolecular structure,
hydrogen bonding, orientational order, and phase diagnostics. Cropping and
wrapping an arbitrary part of a larger snapshot is not sufficient. Preserve
hexagonal proton disorder and assess independent configurations; cuts from the
single amorphous realization are correlated samples, not independent replicas.

Finite targets or slabs can complement bulk runs when asymptotic charge
analysis or electron emission is needed. Verify surface and thickness effects.
Quantum/classical or frozen-density embedding is a possible later extension,
but it requires a separate boundary, polarization, charge-flow, and force
validation effort. It is not necessary to introduce embedding merely because
the full classical model has thousands of molecules.

## 3. Validate electronic structure and incoming states

First reproduce isolated H+--H2O reference conditions, including target
geometry, orientation, impact parameters, and reported observables. Use the
single-water runner only as an execution prototype; its coarse grid, standard
LDA pseudopotentials, and sphere populations are not a benchmark result.
Check target excitation/ionization energetics and the binding of the projectile
states used in capture analysis. Compare candidate exchange-correlation
approximations against relevant reference data rather than choosing by runtime.

Specify whether a bulk calculation measures a prescribed incoming charge state
or a steady-state projectile already travelling in matter. These are different
experiments. The neutral-target restart used for the finite H+ prototype
addresses an incoming bare proton, not steady-state bulk charge preparation.
Audit the solver's periodic charged-cell treatment and test initialization and
transient-length dependence before extracting steady-state stopping.

For C/O/S, specify each required charge and electronic configuration, include
spin where required, and propagate the active projectile electrons. Moving
bound orbitals require the appropriate translational phase. Verify initial
occupations and bound density in the projectile frame. Do not replace a heavy
ion by an arbitrary scaled proton potential or infer its rates by Z-scaling.
Core/semicore treatment must be tested for the chosen ion, energy, and close
encounter conditions.

## 4. Define and validate the measurements

- Stopping: compute transferred electronic energy per path length and compare
  with work inferred from the projectile force under consistent constraints.
  A prescribed constant velocity supplies external work; the uncorrected total
  energy is not expected to remain constant.
- Capture: develop moving projectile-bound-state projections and compare with
  spatial diagnostics at increasing separation. Test overlap with target states,
  partition dependence, and the velocity phase. In bulk, demonstrate that the
  resulting operational charge measure is stable enough for its intended use.
- Target excitation/ionization: define projections or state analyses that
  distinguish the required bound excitations and ionized configurations.
  Unoccupied Kohn--Sham population alone does not establish that distinction.
  Record joint outcomes where capture and target excitation coexist.
- Event probabilities: any reconstruction from a Kohn--Sham determinant uses
  an additional approximation. Compare that reconstruction with channel-resolved
  few-body references. If it fails, retain TDDFT for observables it supports
  and use an independently validated channel method; do not force probabilities
  out of density partitions.

No single stopping-power comparison validates all these measurements.

## 5. Convergence and sampling before production

Separate numerical error, sampling error, and electronic-model disagreement.
Agree on tolerances for the intended transport use before accepting a model.
Converge grid or basis, time step, unoccupied-state representation where used,
projectile potential, target size and shape, initial-state preparation,
trajectory length, and analysis partitions. For open systems also converge
vacuum, absorbers, and propagation time. For periodic systems test interaction
with repeated images and the previously excited electronic wake.

Use the large ice configurations to construct reference distributions of
projectile distances to O and H, select representative paths, and validate the
selection against independent paths. Retain rare close encounters and weights;
a selection that converges mean stopping need not converge rare channel yields.
Treat orientation dependence in hexagonal ice explicitly. Increase independent
structural sampling without relabelling correlated cutouts as replicas.

Fixed target nuclei and a prescribed path isolate the first electronic tests.
They are not assumed valid at 100 eV, especially for heavy projectiles. Compare
against coupled nuclear motion and deflection as the energy decreases. Ehrenfest
forces describe a mean-field trajectory and do not by themselves generate
separate nuclear branches for distinct electronic outcomes. Any later coupling
to the mechanical ice model must derive a single energy/force partition that
avoids double counting within the quantum region.

## 6. HPC implementation and first acceptance milestone

Build and test a pinned MPI-capable solver on the actual cluster. Octopus 16.4
is the installed prototype; the desktop binary has OpenMP but no MPI and is
not an HPC deployment. Benchmark memory and strong scaling on representative
inputs before choosing MPI ranks, OpenMP threads, node counts, and simultaneous
jobs. Distribute independent paths/configurations as scheduler-array tasks;
within each trajectory, time steps remain sequential. Checkpoint enough state
for reproducible continuation and record solver, build, libraries, inputs,
structure identities, and analysis versions.

The first acceptance milestone is a reproduced molecular H+ reference plus a
numerically converged bulk H+ calculation for each phase, with explicit bounds
on what the electronic analysis can identify. This is followed by one specified
heavy-ion charge state to test portability before a large H/C/O/S campaign.
The immediate deliverable is the validated measurement protocol and convergence
record, not a production transport table or an 8,192-water run for its own sake.

## Primary references informing the design

- Hong et al. (2016), isolated H+--water TDDFT/MD:
  https://doi.org/10.1103/PhysRevA.93.062706.
- Reeves & Kanai (2017), periodic liquid-water electron dynamics:
  https://doi.org/10.1038/srep40379.
- Gu et al. (2020), geometric trajectory pre-sampling for liquid-water stopping:
  https://doi.org/10.1063/5.0014276; accessible text
  https://arxiv.org/html/2006.12410.
- Kononov et al. (2023), trajectory-dependent sampling and finite-size errors
  demonstrated in aluminum, not an ice convergence result:
  https://doi.org/10.1038/s41524-023-01157-7.
- Shepard & Kanai (2023), heavy-ion initial states and large periodic systems:
  https://doi.org/10.1021/acs.jpcb.3c05446; accessible Methods
  https://arxiv.org/pdf/2401.04703.

These support the design choices and identify pitfalls. They do not validate
the proposed ice implementation, its lowest energy, or its exclusive channels.
