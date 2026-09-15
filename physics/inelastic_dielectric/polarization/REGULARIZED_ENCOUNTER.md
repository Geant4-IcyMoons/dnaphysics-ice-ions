# Diagnostic close-encounter regularization

## Integration and recovery

The lc-v4 integrator allows at most eight refinement passes. It refines the
dominant impact-quadrature, ODE, or time-window error without relaxing the
existing 0.1% relative target or absolute floor. The upper impact cutoff is
unchanged; this policy does not establish its convergence.

Strict generation preserves successful losses and continues independent
work after a numerical failure, but refuses final export while failures
remain. A compatible resume retries unfinished losses. Diagnostic tables
also retry numerically flagged points. Optional plotting failures are
reported separately and no longer invalidate the numerical calculation.

Set `ICE_POLARIZATION_PROGRESS_DIR` for per-worker refinement and encounter
status, or `ICE_ENCOUNTER_FAILURE_DIR` for direct-solver failure witnesses.
Progress reports are written at encounter/refinement boundaries, not as a
heartbeat during a single encounter. Regression checks are in
`tests/test_polarization_recovery.py`, `tests/test_regularized_encounter.py`,
and `tests/test_encounter_failure_diagnostics.py`.

## Close-encounter witness and equations

The privatex witness for H0 at total T=100 keV and loss 67.82012628321215 eV
failed at x=0.3720893743894298, b=0.2986193522410363 bohr,
eta=0.8370879566801157. Minimum accepted separation was
1.03559855321367e-6 bohr. The failed solver message was that the required
step is smaller than floating-point time spacing. This is evidence of an
extreme close encounter, not a demonstration that a physical cross-section
diverges. Rejected trial stages and accepted steps are recorded separately
under output/encounter_probe. Runtime-generated records are the primary evidence.

The diagnostic prototype uses planar Levi-Civita variables. In the existing
dimensionless encounter time t, r=R-eta*y satisfies

    r'' = -mu*r/|r|^3 + F(r,t), mu=eta*Z,
    F = x^2*(R-r) + eta*N_inside(b*|r|)*r/|r|^3.

Set r=(u1^2-u2^2,2*u1*u2), rho=u.u, and dt/ds=rho. With
L=[[u1,-u2],[u2,u1]], p=dr/dt, and h=p.p/2-mu/rho,

    du/ds = v,
    dv/ds = h*u/2 + rho*L.T*F/2,
    dh/ds = 2*v.L.T*F,
    dt/ds = rho.

These equations recover exactly r''=-mu*r/|r|^3+F away from collision;
the test suite checks this identity. No Coulomb softening is introduced.
N_inside is evaluated directly with incomplete gamma functions from the same
hydrogenic/Gaussian density, avoiding subtraction of almost equal charges at
small radius. The original solver instead interpolates the outer electronic
tail; agreement between these evaluations is also a numerical consideration.
The leading oscillator is integrated alongside the full solution. Returning
the difference of their energies is less cancellation-resistant than the
production amplitude-difference formulation at very weak coupling, so this
solver is not an unconditional replacement. It is called when the direct
encounter solver fails or matched-window solves disagree beyond the existing
ODE absolute/relative scale. If any of the three estimates requires this
path, all three use regularized encounters. The enclosing impact integral
still enforces its independent time-window and tolerance checks.
Numerical metadata is versioned lc-v4 to
reject reuse of pre-regularization polarization results.

This is the standard planar coordinate/time regularization, not a new force
model; see the mathematical discussion in
https://doi.org/10.1007/s10569-018-9862-4 . The equations above are specified
explicitly to make the implementation independently testable.

The fallback is connected to the generation integrator. Tests compare a weak point and screened encounter
against the original solver; compute-node tests vary both tolerance and final
time for the failed encounter. Solver success alone is not acceptance.
Finite nuclear size, quantum scattering/capture, relativistic target-electron
motion, and matching to the Born contribution are not supplied by this
transformation. Its validity is limited to the existing classical oscillator
equations. Persistent time-window dependence requires a model-level diagnosis.

The six privatex encounter probes (job 242473) all completed, at rtol=2e-8
and 2e-9 and time windows 1, 2, and 4 times the failed window. Their
full-minus-leading energy ranged from 10.876200748863784 to
10.876202246628255, a relative spread of 1.38e-7. This validates only this
encounter and these settings, not the entire impact integral or loss grid.
The complete three-point impact-integral probe is a separate job, 242474.
