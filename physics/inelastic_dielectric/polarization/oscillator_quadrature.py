"""Phase-resolved quadrature of the existing screened oscillator equations.

The parity-resolved Green function solves Y''+x^2 Y=f with retarded initial
conditions. Even Yx and odd Yz determine the cubic energy coefficient.
Gauss polynomial antiderivatives evaluate these convolutions without
subtracting two large global prefix integrals. See SCREENED_BARKAS.md for
the force, gradient, relativistic weights, and retained x=50 boundary.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy.integrate import quad_vec

VERSION = "phase-panel-gauss-adaptive-impact-v1"
RELATIVE_TOLERANCE = 1e-4


@lru_cache(maxsize=16)
def legendre_rule(order):
    return leggauss(order)


@lru_cache(maxsize=8)
def _gauss_antiderivative(order):
    """Weights for integrals from -1 to each Gauss node, and node to +1."""
    nodes, weights = legendre_rule(order)
    basis = legvander(nodes, order)
    primitive = np.empty((order, order))
    primitive[:, 0] = nodes + 1
    for k in range(1, order):
        primitive[:, k] = (basis[:, k+1]-basis[:, k-1])/(2*k+1)
    coefficients = (np.arange(order)+.5)[:, None]*basis[:, :order].T*weights
    left = primitive @ coefficients
    return nodes, weights, left, weights[None, :]-left


def impulse_products(x, b_bohr, density, *, order=12, extent=64.,
                     log_step=.5, phase_step=np.pi):
    """Px*Dx, Pz*Dz with both encounter scale and oscillations resolved.

    Each panel spans at most log_step in log(1+tau) and phase_step in x*tau.
    The integration window is max(extent, extent/x); extent is numerical,
    not a change to the physical impact-parameter cutoff.
    Broadcast array inputs return shape (2, *broadcast_shape).
    """
    x, b_bohr = np.broadcast_arrays(np.asarray(x, float), np.asarray(b_bohr, float))
    if x.ndim:
        result = np.empty((2,)+x.shape)
        for idx in np.ndindex(x.shape):
            result[(slice(None),)+idx] = impulse_products(
                float(x[idx]), float(b_bohr[idx]), density, order=order,
                extent=extent, log_step=log_step, phase_step=phase_step)
        return result
    if (not np.isfinite(x) or x <= 0 or not np.isfinite(b_bohr) or b_bohr <= 0
            or order < 4 or extent <= 0 or log_step <= 0 or phase_step <= 0):
        raise ValueError("Invalid phase-resolved oscillator quadrature input.")
    # End on a complete cycle; arbitrary phases make the omitted tail error
    # oscillate rapidly as the outer integrator varies x.
    end = 2*np.pi*np.ceil(max(extent*x, extent)/(2*np.pi))/x
    logarithmic = np.expm1(np.arange(0., np.log1p(end), log_step))
    oscillatory = np.arange(1., np.ceil(x*end/phase_step))*phase_step/x
    edges = np.unique(np.r_[logarithmic, oscillatory, end])
    nodes, weights, left, right = _gauss_antiderivative(order)
    half = np.diff(edges)/2
    t = edges[:-1, None] + half[:, None]*(nodes+1)
    h = np.hypot(1., t)
    g, rgprime = density.radial_charge(b_bohr*h)
    fx, fz = g/h**3, t*g/h**3
    a = ((2-t*t)*g-rgprime)/h**5
    c = ((2*t*t-1)*g-t*t*rgprime)/h**5
    d = t*(3*g-rgprime)/h**5
    sn, cs = np.sin(x*t), np.cos(x*t)

    def convolution(values, backwards=False):
        areas = (values @ weights)*half
        matrix = right if backwards else left
        partial = (values @ matrix.T)*half[:, None]
        if backwards:
            partial[:-1] += np.cumsum(areas[:0:-1])[::-1, None]
        else:
            partial[1:] += np.cumsum(areas[:-1])[:, None]
        return partial

    yx = (sn*convolution(fx*cs)+cs*convolution(fx*sn, True))/x
    yz = (-sn*convolution(fz*cs, True)-cs*convolution(fz*sn))/x

    def integrate(values):
        return 2*np.sum((values @ weights)*half)

    px, pz = integrate(fx*cs), integrate(fz*sn)
    dx = integrate(cs*(a*yx+d*yz))
    dz = integrate(sn*(d*yx+c*yz))
    result = np.array([px*dx, pz*dz])
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("Nonfinite phase-resolved oscillator integral.")
    return result


def integrate_kernel(xi, b_per_x, gamma, density, *, rtol=RELATIVE_TOLERANCE,
                     atol=None, component=None, allow_unconverged=False):
    """Integrate in log(x), with separate impact, time, and tail errors.

    Positive local time/tail differences are integrated, so cancellations
    cannot hide their estimated errors. Adaptive Gauss-Kronrod error estimates
    and refinement differences are estimates, not rigorous error bounds.
    Neither the Born DCS nor its magnitude is used to accept this integral.
    component=0 or 1 isolates a Cartesian contribution for numerical
    benchmarks; at gamma=1 the point-charge integrals are Z^3 I1 and Z^3 I2.
    """
    if atol is None:
        atol = 1e-10*density.z**3
    if (not np.isfinite(xi) or xi <= 0 or not np.isfinite(b_per_x) or b_per_x <= 0
            or not np.isfinite(gamma) or gamma < 1 or not 0 < rtol < 1
            or not np.isfinite(atol) or atol <= 0 or component not in (None, 0, 1)):
        raise ValueError("Invalid adaptive oscillator integral input.")
    empty = dict(impact_error=0., time_error=0., tail_error=0.,
                 evaluations=0, refinement=0, converged=True)
    if xi >= 50.:
        return 0., 0., empty
    points = [np.log(p) for p in (.001, .01, .1, 1., 10.) if xi < p < 50.]
    if density.electrons:
        radius = np.sqrt(density.moment(2)/density.electrons)
        transition = radius/b_per_x
        if xi < transition < 50.:
            points.append(np.log(transition))
    total_evaluations = 0
    for level in range(3):
        order = 8+4*level
        extent = 32.*2**level

        def integrand(logx):
            x = np.exp(logx)
            b = x*b_per_x
            low = impulse_products(x, b, density, order=order, extent=extent)
            fine = impulse_products(x, b, density, order=order+4, extent=extent)
            tail = impulse_products(x, b, density, order=order+4, extent=2*extent)
            weight = np.array([.5/x, .5/(x*gamma**2)])
            if component is not None:
                weight[1-component] = 0.
            return np.array([tail @ weight, np.abs(fine-low) @ weight,
                             np.abs(tail-fine) @ weight])

        values, impact_error, info = quad_vec(
            integrand, np.log(xi), np.log(50.), points=sorted(set(points)),
            epsabs=atol/8, epsrel=rtol/8, norm="max", quadrature="gk21",
            limit=256, full_output=True)
        total_evaluations += info.neval
        value, time_error, tail_error = values
        error = float(impact_error+time_error+tail_error)
        diagnostics = dict(impact_error=float(impact_error), time_error=float(time_error),
                           tail_error=float(tail_error), evaluations=total_evaluations,
                           refinement=level)
        if (info.success and np.isfinite(error) and np.isfinite(value)
                and error <= atol+rtol*abs(value)):
            diagnostics["converged"] = True
            return float(value), error, diagnostics
    if allow_unconverged:
        diagnostics["converged"] = False
        return float(value), error, diagnostics
    raise RuntimeError(
        "Screened Barkas quadrature did not converge: "
        f"xi={xi:g}, value={value:g}, errors={diagnostics}; no tables were exported.")
