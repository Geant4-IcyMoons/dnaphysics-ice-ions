"""Nonlinear polarization spectra and restart-safe loss integration.

The historical barkas_* keys are retained for existing table consumers.
All correction-on states use the full-minus-leading nonlinear backend.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import json
from tqdm import tqdm
from physics.inelastic_dielectric import checkpoints

import numpy as np

from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import DEFAULT_WORKERS, load_density
from physics.inelastic_dielectric.polarization import nonlinear_oscillator

MODEL = "frozen-full-nonlinear-oscillator-v1"
QUADRATURE_RTOL = nonlinear_oscillator.RELATIVE_TOLERANCE
_STOP_EVENT = None


def _init_polarization_worker(stop_event):
    global _STOP_EVENT
    _STOP_EVENT = stop_event


def metadata(density=None):
    return {
        "barkas_model": MODEL,
        "polarization_response": "full_minus_leading",
        "polarization_contains_higher_even_and_odd": True,
        "polarization_cubic_backend_available": False,
        "barkas_target": "ice-optical-OOS-independent-oscillators",
        "barkas_projectile": "frozen-spherical-field" if density is not None else "point-interaction-charge",
        "barkas_spectral_convention": "OOS-equivalent-not-channel-resolved",
        "barkas_relativistic_prescription": "full-electric-field-kinematic-extension",
        "barkas_charge_state": density.charge if density is not None else -1,
        "barkas_CB": 1.0,
        "barkas_quadrature_rtol": QUADRATURE_RTOL,
        "barkas_quadrature_method": nonlinear_oscillator.VERSION,
        "barkas_experimental_validation": False,
        "barkas_status": "experimental-optical-spectral-mapping",
        "barkas_finite_q_target_matching_validated": False,
    }


def converged_kernel(W_eV, beta, gamma, density, *, interaction_charge=None, return_diagnostics=False, checkpoint_path=None, diagnostic_only=False):
    """Adaptive impact integration with independent time and tail checks."""
    from physics.inelastic_dielectric.polarization import correction as bd
    w = np.asarray(W_eV, float)
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Oscillator losses must be finite and positive, in eV.")
    if not (np.isfinite(beta) and 0 < beta < 1 and np.isfinite(gamma) and gamma >= 1
            and np.isclose(gamma, 1/np.sqrt(1-beta*beta), rtol=1e-12)):
        raise ValueError("Invalid or inconsistent projectile beta/gamma.")
    v = beta/bd.ALPHA_FINE
    omega = w/(bd.ALPHA_FINE**2*bd.MEC2_EV)
    xi = .5616*bd.H2O_CB*omega/(gamma*v*v)
    result, error, diagnostics = np.zeros_like(w), np.zeros_like(w), []
    done = np.zeros(w.shape, dtype=bool)
    field = (nonlinear_oscillator.frozen_field(density.element, density.charge) if density is not None
             else nonlinear_oscillator.PointField(interaction_charge))
    signature = json.dumps(dict(beta=float(beta), gamma=float(gamma), model=metadata(density),
                                interaction_charge=interaction_charge), sort_keys=True)
    if checkpoint_path is not None:
        if diagnostic_only:
            raise ValueError("Diagnostic rows use their separate diagnostic checkpoint store")
        if return_diagnostics:
            raise ValueError("Partial checkpoints store values/errors, not benchmark diagnostics")
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            with np.load(checkpoint_path, allow_pickle=False) as saved:
                if str(saved["signature"]) != signature or not np.array_equal(saved["W"], w):
                    raise ValueError("Incompatible polarization row checkpoint")
                result, error, done = saved["value"].copy(), saved["error"].copy(), saved["done"].copy()
                if (result.shape != w.shape or error.shape != w.shape or done.shape != w.shape
                        or done.dtype != np.dtype(bool) or np.any(~np.isfinite(result[done]))
                        or np.any(~np.isfinite(error[done])) or np.any(error[done] < 0)):
                    raise ValueError("Invalid polarization row checkpoint")
    pending = [idx for idx in np.ndindex(w.shape) if not done[idx]]
    for idx in tqdm(pending, total=w.size, initial=int(done.sum()), unit="loss",
                    desc=f"Polarization {Path(checkpoint_path).stem if checkpoint_path else ''}",
                    mininterval=60, disable=checkpoint_path is None):
        if _STOP_EVENT is not None and _STOP_EVENT.is_set():
            raise RuntimeError("Polarization stopped after another worker failed; completed losses are preserved")
        try:
            result[idx], error[idx], row = nonlinear_oscillator.integrate_kernel(
                float(xi[idx]), float(gamma*v/omega[idx]), float(gamma), field,
                rtol=QUADRATURE_RTOL,
                **({"allow_unconverged": True} if diagnostic_only else {}))
        except (FloatingPointError, OverflowError) as exc:
            if not diagnostic_only:
                raise
            result[idx], error[idx] = np.nan, np.nan
            row = dict(converged=False, failure=str(exc))
        except RuntimeError as exc:
            if not diagnostic_only:
                raise RuntimeError(f"Nonlinear polarization quadrature did not converge at "
                                   f"W/eV={w[idx]:g}, beta={beta:g}: {exc}") from exc
            result[idx], error[idx] = np.nan, np.nan
            row = dict(converged=False, failure=str(exc))
        diagnostics.append(dict(W_eV=float(w[idx]), **row))
        done[idx] = True
        if checkpoint_path is not None:
            checkpoints.atomic_savez(checkpoint_path, signature=signature, W=w,
                                     value=result, error=error, done=done)
    if return_diagnostics:
        return result, error, diagnostics
    return result, error


def _energy_row(task, *, diagnostic_only=False):
    from physics.inelastic_dielectric.polarization import correction as bd
    energy, w, mass, element, charge = task[:5]
    checkpoint_path = task[5] if len(task) > 5 else None
    density = load_density(element, charge) if element is not None else None
    beta, gamma = bd.projectile_beta_gamma(energy, mass)
    if beta/bd.ALPHA_FINE <= 1.:
        if diagnostic_only:
            return np.full_like(w, np.nan), np.full_like(w, np.nan), np.full(w.shape, 4, dtype=np.uint8)
        raise ValueError("Nonlinear oscillator polarization requires v > the Bohr velocity; "
                         f"T={energy:g} eV total is outside this model's domain.")
    calculated = converged_kernel(w, float(beta), float(gamma), density,
        interaction_charge=charge if density is None else None,
        **({"checkpoint_path": checkpoint_path} if checkpoint_path else {}),
        **({"diagnostic_only": True, "return_diagnostics": True} if diagnostic_only else {}))
    value, error = calculated[:2]
    pref = 4*np.pi*bd.RE_CLASSICAL_CM**2*bd.ALPHA_FINE/(gamma**2*beta**5)
    if diagnostic_only:
        flags = np.array([0 if row.get("converged", False) else 1 for row in calculated[2]], dtype=np.uint8).reshape(w.shape)
        flags[~np.isfinite(value) | ~np.isfinite(error)] |= 2
        return bd.CM2_TO_M2*pref*value, bd.CM2_TO_M2*pref*error, flags
    return bd.CM2_TO_M2*pref*value, bd.CM2_TO_M2*pref*error


def dcs_m2_per_eV(T_eV, W_eV, mass_me, df_dW, density=None, *, z_int=None, workers=DEFAULT_WORKERS, checkpoint_dir=None):
    """Full-minus-leading OOS-equivalent spectrum for every frozen/point state.

    Output is m2/eV. Exact Wmax is retained. Signed corrections are not
    clipped, and there is no analytic bare-ion or cubic fallback.
    """
    from physics.inelastic_dielectric.polarization import correction as bd
    t, w, oos = np.broadcast_arrays(np.asarray(T_eV, float), np.asarray(W_eV, float),
                                  np.asarray(df_dW, float))
    shape = t.shape
    t, w, oos = t.ravel(), w.ravel(), oos.ravel()
    if (not np.isfinite(mass_me) or mass_me <= 0 or np.any(~np.isfinite(t))
            or np.any(t <= 0) or np.any(~np.isfinite(w)) or np.any(w <= 0)
            or np.any(~np.isfinite(oos)) or np.any(oos < 0)):
        raise ValueError("Invalid nonlinear polarization input, mass, or OOS density.")
    charges = np.broadcast_to(np.asarray(z_int if z_int is not None else 0., float), shape).ravel()
    if density is None and (z_int is None or np.any(~np.isfinite(charges)) or np.any(charges < 0)):
        raise ValueError("Point nonlinear response requires finite nonnegative interaction charges")
    out, error = np.zeros_like(w), np.zeros_like(w)
    mask = (w <= bd.wmax_eV(t, mass_me)) & (oos > 0)
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoints.require_manifest(checkpoint_dir, dict(source=checkpoints.source_digest(),
            mass=float(mass_me), model=metadata(density),
            point_charge_signature=sorted(set(charges.tolist())) if density is None else None))
    indices, tasks = [], []
    for energy in np.unique(t[mask]):
        idx = np.flatnonzero(mask & (t == energy))
        unique_w, inverse = np.unique(w[idx], return_inverse=True)
        indices.append((idx, inverse))
        if density is None and np.unique(charges[idx]).size != 1:
            raise ValueError("Interaction charge must be constant within an incident-energy row")
        task = (float(energy), unique_w, mass_me, density.element if density is not None else None,
                density.charge if density is not None else float(charges[idx[0]]))
        if checkpoint_dir is not None:
            task += (checkpoint_dir / f"energy_{float(energy).hex()}.npz",)
        tasks.append(task)
    rows = [None] * len(tasks)
    if tasks and workers > 1 and len(tasks) > 1:
        context = get_context("spawn")
        stop_event = context.Event()
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                 mp_context=context, initializer=_init_polarization_worker,
                                 initargs=(stop_event,)) as pool:
            futures = {pool.submit(_energy_row, task): i for i, task in enumerate(tasks)}
            try:
                for future in tqdm(as_completed(futures), total=len(tasks), desc="Polarization energies", unit="energy"):
                    rows[futures[future]] = future.result()
            except BaseException:
                stop_event.set()
                for future in futures:
                    future.cancel()
                raise
    else:
        for i, task in enumerate(tqdm(tasks, desc="Polarization energies", unit="energy")):
            rows[i] = _energy_row(task)
    for (idx, inverse), (value, err) in zip(indices, rows):
        out[idx], error[idx] = oos[idx]*value[inverse], oos[idx]*err[inverse]
    return out.reshape(shape), error.reshape(shape)
