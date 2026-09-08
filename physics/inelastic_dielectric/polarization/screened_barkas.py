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
from multiprocessing import get_context

import numpy as np

from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import DEFAULT_WORKERS, load_density
from physics.inelastic_dielectric.polarization import oscillator_quadrature

MODEL = "frozen-screened-optical-oscillator-v1"
REFERENCE_DOI = "10.1016/S0168-583X(99)01181-7"
QUADRATURE_RTOL = oscillator_quadrature.RELATIVE_TOLERANCE


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
        "barkas_quadrature_method": oscillator_quadrature.VERSION,
        "barkas_experimental_validation": False,
        "barkas_status": "experimental-optical-spectral-mapping",
        "barkas_finite_q_target_matching_validated": False,
    }


def converged_kernel(W_eV, beta, gamma, density, *, return_diagnostics=False):
    """Adaptive impact integration with independent time and tail checks."""
    from physics.inelastic_dielectric.polarization import barkas_dcs as bd
    w = np.asarray(W_eV, float)
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Oscillator losses must be finite and positive, in eV.")
    if not (np.isfinite(beta) and 0 < beta < 1 and np.isfinite(gamma) and gamma >= 1
            and np.isclose(gamma, 1/np.sqrt(1-beta*beta), rtol=1e-12)):
        raise ValueError("Invalid or inconsistent projectile beta/gamma.")
    v = beta/bd.ALPHA_FINE
    omega = w/(bd.ALPHA_FINE**2*bd.MEC2_EV)
    xi = .5616*bd.H2O_CB*omega/(gamma*v*v)
    result, error, diagnostics = np.empty_like(w), np.empty_like(w), []
    for idx in np.ndindex(w.shape):
        try:
            result[idx], error[idx], row = oscillator_quadrature.integrate_kernel(
                float(xi[idx]), float(gamma*v/omega[idx]), float(gamma), density,
                rtol=QUADRATURE_RTOL)
        except RuntimeError as exc:
            raise RuntimeError(f"Screened Barkas quadrature did not converge at "
                               f"W/eV={w[idx]:g}, beta={beta:g}: {exc}") from exc
        diagnostics.append(dict(W_eV=float(w[idx]), **row))
    if return_diagnostics:
        return result, error, diagnostics
    return result, error


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
