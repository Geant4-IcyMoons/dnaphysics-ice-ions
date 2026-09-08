# Full nonlinear oscillator polarization

## Definition

    DCS_total(T,W) = DCS_Born(T,W) + DCS_polarization(T,W).

Born is the existing PWBA/RPWBA calculation with the frozen form factor.
The correction is the impact-integrated full oscillator energy minus its
own leading energy, not a charge-cubed multiplier or rescaled Born DCS.
Subtracting the leading oscillator response avoids adding it a second time.
It does not prove its equivalence to the finite-q ice Born response.

## Equations

Use atomic units for distance and velocity, omega=W/E_h, beta=v*alpha and
gamma from total kinetic energy per ion and M/m_e. The per_u CLI input is
converted to total energy first. The field is
g(r)=Z-4*pi*integral_0^r rho(s)*s^2 ds. Bare ions use g=Z; neutrals retain
their short-range field. Scalar charge/zeff modes use a point field of that
charge and cannot be combined with a state-specific frozen density.

    tau = gamma*v*t/b, x = omega*b/(gamma*v), eta = 1/(gamma*b*v^2)
    R = (1,tau)
    y0'' + x^2*y0 = g(b*|R|)*R/|R|^3
    y''  + x^2*y  = g(b*|R-eta*y|)*(R-eta*y)/|R-eta*y|^3
    H = [p_x^2+x^2*y_x^2 + gamma^-2*(p_z^2+x^2*y_z^2)]/2
    DeltaE = E_h * gamma^2*v^2*eta^2 * H.

Initial response vanishes in the remote past. Physical displacements are
(b*eta*y_x,b*eta*y_z/gamma). The gamma factors extend the previous
electric-field kinematic prescription to full displacement. They correspond
to a Lorentz-contracted frozen cloud driving a nonrelativistic oscillator.
This omits magnetic force on the moving target electron, relativistic electron
dynamics and dynamic projectile electrons. It is **not** a published, fully
covariant screened nonlinear RPWBA derivation. The RPWBA switch changes only
the Born longitudinal/transverse response.

## Optical assignment and units

The unchanged optical OOS convention is 8 valence plus 2 core electrons per
H2O, separate from the finite-q K continuum sum rule. Valence derives from
W*ELF(W,0); no finite-q ELF is inserted into the nonlinear oscillator.

    d_sigma_pol/dW = (df/dW)/W * 2*pi*a0^2 * integral b db (DeltaE_full-DeltaE_0)

With a0 in cm, energies in eV and df/dW in eV^-1 this is cm^2/eV.
The implementation uses the algebraically equivalent form

    d_sigma_pol/dW = 4*pi*r_e^2*alpha/(gamma^2*beta^5) * (df/dW) * K_full
    K_full = integral [(H_full-H_0)/(2*eta*x)] d(log x)
    xi = 0.5616*C_B*W/(gamma*beta^2*m_e*c^2), C_B=1.

The lower limit x=xi retains the Salvat cutoff b_min=0.5616*C_B/v bohr.
Its use for nonlinear response is an inherited assumption, not a new
close-collision derivation. The upper impact limit is x=50; sensitivity to
a larger limit is a separate convergence check. The classical oscillator
does not resolve quantum close collisions. Source for the inherited cutoff:
Salvat and Quesada, [SBETHE v2](https://doi.org/10.17632/7zw25f428t.2).

Multiply cm^2/eV by 1e-4 once to obtain m^2/eV. The final DAT conversion
retains `EMFI_DCS_SCALE_M2`; it is not another cm-to-m conversion.
Polarization is zero above the existing exact transfer maximum:

    Wmax = 2*beta^2*gamma^2*m_e*c^2 / (1+2*gamma/(M/m_e)+1/(M/m_e)^2).

Born kinematic limits are unchanged. TCS and sampling densities use the final
sum. S_polarization is integral W*DCS_polarization dW on the exported grid.
Assignment to excitation/ionization columns is bookkeeping only.

## Numerics and failures

DOP853 evolves rotating harmonic amplitudes, treating free oscillations
exactly. Full-minus-leading amplitudes are evolved together to reduce
cancellation. Its embedded error norm is evaluated with scaling before
squaring, avoiding underflow in vanishing neutral-field tails without changing
the estimator or the force. Adaptive Gauss-Kronrod integration is over log impact parameter.
Estimated errors combine impact quadrature, tighter ODE tolerance and doubled
time-window differences. The relative tolerance is 1e-3 on the correction,
with a dimensionless absolute floor 1e-9*max(1,Z^3). Z^3 sets an error scale,
not a response model. Three refinement levels are attempted. No Born-relative
error floor, cubic fallback, force softening or clipping is used.

Completed loss nodes are checkpointed with source/model/charge/kinematic
signatures. Nonconvergence, singular encounters and nonfinite values fail
strict generation. Negative final/channel DCS also fail transport export.
The v>1 atomic-unit domain guard remains. A finite correction larger than
Born is reported, not rejected merely because of its magnitude.
`--diagnostic-only` retains finite failed estimates and NaN for missing values
in separate products explicitly marked not transport-ready.

Full-minus-leading includes higher even and odd terms. No Bloch DCS is added.
A downstream stopping-only Bloch correction requires a double-counting
analysis; it is not automatically consistent with this full classical response.

## Status

Connected to DCS/TCS generation for both ice phases and PWBA/RPWBA, with one
nonlinear backend for every state. [Numerical checks](benchmarking/README.md)
test implementation, not the optical-to-DCS assignment, screened relativity
or agreement with ICRU/Matias. No complete production-energy matrix is implied.
