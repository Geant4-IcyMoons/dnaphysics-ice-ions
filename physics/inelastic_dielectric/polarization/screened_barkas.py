"""Frozen-projectile, optical-oscillator polarization correction.

This is a screened extension of the distant-oscillator construction in
Salvat, PRA 106, 032809 (2022), Eqs. (91)-(100), not an electron-gas kernel.
Schinner & Sigmund, NIMB 164-165, 220 (2000), DOI
10.1016/S0168-583X(99)01181-7, supplies the screened-oscillator precedent;
this implementation is derived from the force/force-gradient equations,
not a port of that paper's stopping-number parametrization.

The OOS-equivalent spectral correction assigns oscillator frequency W/hbar
to the table's loss W, as does the existing Salvat table construction.
It is not a microscopic, channel-resolved second-Born DCS. See
SCREENED_BARKAS.md for equations, relativistic prescription and limitations.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_simpson, simpson

from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import DEFAULT_WORKERS, load_density

MODEL = "frozen-screened-optical-oscillator-v1"
REFERENCE_DOI = "10.1016/S0168-583X(99)01181-7"
QUADRATURE_RTOL = 0.01


def metadata(density):
    return {
        "barkas_model": MODEL,
        "barkas_target": "ice-optical-OOS-independent-oscillators",
        "barkas_projectile": "frozen-spherical-field-and-gradient",
        "barkas_reference_doi": REFERENCE_DOI,
        "barkas_spectral_convention": "OOS-equivalent-not-channel-resolved",
        "barkas_relativistic_prescription": "Jackson-McCarthy-kinematic-extension",
        "barkas_close_collision_correction": False,
        "barkas_neutral_supported": True,
        "barkas_charge_state": density.charge,
        "barkas_CB": 1.0,
        "barkas_quadrature_rtol": QUADRATURE_RTOL,
        "barkas_experimental_validation": False,
        "barkas_status": "experimental-optical-spectral-mapping",
        "barkas_finite_q_target_matching_validated": False,
    }


def _impulse_products(x, b_bohr, density, n_time, tail_cycles):
    """Return Px*Dx and Pz*Dz for an oscillator, without charge rescaling.

    tau=gamma*v*t/b; h=sqrt(1+tau^2); f=(g/h^3,tau*g/h^3).
    Y''+x^2 Y=f, with retarded initial conditions. Parity selects Yx-even
    and Yz-odd in the final energy. Integrating these parts directly avoids
    subtracting large causal homogeneous oscillations at small x.
    """
    x, b = np.broadcast_arrays(np.atleast_1d(x), np.atleast_1d(b_bohr))
    u = np.linspace(0., 1., n_time)
    end = np.maximum(256., tail_cycles/x)
    logend = np.log1p(end)[:, None]
    t = np.expm1(logend*u)
    jac = logend*(1+t)
    h = np.hypot(1., t)
    g, rgprime = density.radial_charge(b[:, None]*h)
    fx, fz = g/h**3, t*g/h**3
    # rgprime is r*g'(r). Every derivative of the screening charge is kept.
    a = ((2-t*t)*g-rgprime)/h**5
    c = ((2*t*t-1)*g-t*t*rgprime)/h**5
    d = t*(3*g-rgprime)/h**5
    phase = x[:, None]*t
    sn, cs = np.sin(phase), np.cos(phase)

    def cumulative(v):
        return cumulative_simpson(v*jac, x=u, axis=1, initial=0.)

    def tail(v):
        return cumulative_simpson((v*jac)[:, ::-1], x=-u[::-1],
                                  axis=1, initial=0.)[:, ::-1]

    yx = (sn*cumulative(fx*cs)+cs*tail(fx*sn))/x[:, None]
    yz = (-sn*tail(fz*cs)-cs*cumulative(fz*sn))/x[:, None]
    px = 2*simpson(fx*cs*jac, x=u, axis=1)
    pz = 2*simpson(fz*sn*jac, x=u, axis=1)
    dx = 2*simpson(cs*(a*yx+d*yz)*jac, x=u, axis=1)
    dz = 2*simpson(sn*(d*yx+c*yz)*jac, x=u, axis=1)
    return px*dx, pz*dz


@lru_cache(maxsize=8)
def _legendre(n):
    return leggauss(n)


def oscillator_kernel(W_eV, beta, gamma, density, *, n_impact=64,
                      n_time=2049, tail_cycles=32.):
    """Dimensionless screened replacement for Z^3 [I1+I2/gamma^2].

    Atomic units use Eh=alpha^2*mec^2, a0=r_e/alpha^2, consistently with
    the Salvat constants, without changing the Born generator constants.
    a=0.5616*C_B/v, C_B=1; x=W*b/(gamma*v*Eh). The x integral is
    (1/2) integral (Px*Dx+Pz*Dz/gamma^2)/x^2 dx from xi to infinity.
    x=50 is the existing SBETHE numerical exponential-tail boundary.
    """
    from physics.inelastic_dielectric.polarization import barkas_dcs as bd
    w = np.atleast_1d(np.asarray(W_eV, float))
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Oscillator losses must be finite and positive, in eV.")
    if not (np.isfinite(beta) and 0 < beta < 1 and np.isfinite(gamma) and gamma >= 1):
        raise ValueError("Invalid projectile beta/gamma.")
    if not np.isclose(gamma, 1/np.sqrt(1-beta*beta), rtol=1e-12):
        raise ValueError("Inconsistent projectile beta/gamma.")
    if n_time < 5 or n_time % 2 != 1 or n_impact < 4 or tail_cycles <= 0:
        raise ValueError("Invalid oscillator quadrature settings.")
    v = beta/bd.ALPHA_FINE
    omega = w/(bd.ALPHA_FINE**2*bd.MEC2_EV)
    xi = 0.5616*bd.H2O_CB*omega/(gamma*v*v)
    out = np.zeros_like(w)
    nodes, weights = _legendre(n_impact)
    for j in np.flatnonzero(xi < 50.):
        span = np.log(50./xi[j])
        x = xi[j]*np.exp((nodes+1)*span/2)
        b = x*gamma*v/omega[j]
        px, pz = _impulse_products(x, b, density, n_time, tail_cycles)
        out[j] = span/4*np.sum(weights*(px+pz/gamma**2)/x)
    if np.any(~np.isfinite(out)):
        raise FloatingPointError("Nonfinite screened oscillator correction.")
    return out


def converged_kernel(W_eV, beta, gamma, density):
    """Refine the correction itself; never use the larger Born term as a floor."""
    w = np.asarray(W_eV, float)
    low = oscillator_kernel(w, beta, gamma, density)
    high = oscillator_kernel(w, beta, gamma, density, n_impact=96, n_time=4097,
                             tail_cycles=64.)
    error = np.abs(high-low)
    tolerance = QUADRATURE_RTOL*np.abs(high)+1e-10*density.z**3
    bad = error > tolerance
    if np.any(bad):
        refined = oscillator_kernel(w[bad], beta, gamma, density, n_impact=144,
                                    n_time=8193, tail_cycles=128.)
        error[bad] = np.abs(refined-high[bad])
        high[bad] = refined
        tolerance = QUADRATURE_RTOL*np.abs(high)+1e-10*density.z**3
    bad = error > tolerance
    if np.any(bad):
        raise RuntimeError("Screened Barkas quadrature did not converge at W/eV="
                           + repr(w[bad].tolist())+"; no tables were exported.")
    return high, error


def _energy_row(task):
    from physics.inelastic_dielectric.polarization import barkas_dcs as bd
    energy, w, mass, element, charge = task
    density = load_density(element, charge)
    beta, gamma = bd.projectile_beta_gamma(energy, mass)
    if beta/bd.ALPHA_FINE <= 1.:
        raise ValueError("Screened oscillator Barkas requires v > the Bohr velocity; "
                         f"T={energy:g} eV total is outside this model's domain.")
    value, error = converged_kernel(w, float(beta), float(gamma), density)
    pref = 4*np.pi*bd.RE_CLASSICAL_CM**2*bd.ALPHA_FINE/(gamma**2*beta**5)
    return bd.CM2_TO_M2*pref*value, bd.CM2_TO_M2*pref*error


def dcs_m2_per_eV(T_eV, W_eV, mass_me, df_dW, density, *, workers=DEFAULT_WORKERS):
    """Additive, OOS-equivalent Barkas-like correction for one frozen state.

    Returned arrays are DCS and estimated absolute quadrature error, both
    m^2/eV. Wmax is applied here; no Wmax in the Born path is changed.
    Bare states dispatch to the existing analytic implementation exactly.
    """
    from physics.inelastic_dielectric.polarization import barkas_dcs as bd
    t, w, oos = np.broadcast_arrays(np.asarray(T_eV, float), np.asarray(W_eV, float),
                                  np.asarray(df_dW, float))
    shape = t.shape
    t, w, oos = t.ravel(), w.ravel(), oos.ravel()
    if (not np.isfinite(mass_me) or mass_me <= 0 or np.any(~np.isfinite(t))
            or np.any(t <= 0) or np.any(~np.isfinite(w)) or np.any(w <= 0)
            or np.any(~np.isfinite(oos)) or np.any(oos < 0)):
        raise ValueError("Invalid screened Barkas input, mass, or OOS density.")
    out, error = np.zeros_like(w), np.zeros_like(w)
    mask = (w <= bd.wmax_eV(t, mass_me)) & (oos > 0)
    if density.electrons == 0:
        out[mask] = bd.barkas_dcs_m2_per_eV(t[mask], w[mask], density.z, mass_me, oos[mask])
    else:
        indices, tasks = [], []
        for energy in np.unique(t[mask]):
            idx = np.flatnonzero(mask & (t == energy))
            unique_w, inverse = np.unique(w[idx], return_inverse=True)
            indices.append((idx, inverse))
            tasks.append((float(energy), unique_w, mass_me, density.element, density.charge))
        if tasks and workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                     mp_context=get_context("spawn")) as pool:
                rows = list(pool.map(_energy_row, tasks))
        else:
            rows = list(map(_energy_row, tasks))
        for (idx, inverse), (value, err) in zip(indices, rows):
            out[idx], error[idx] = oos[idx]*value[inverse], oos[idx]*err[inverse]
    return out.reshape(shape), error.reshape(shape)
