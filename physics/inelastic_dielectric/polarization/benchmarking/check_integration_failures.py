"""Retest numerical and physical acceptance separately; never export tables.

Representative H/He losses and total projectile energies are sampled in both
ice phases and both Born kernels. This is not a complete production-grid
validation. Only the production adaptive integrator is used.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import time

import numpy as np
import scipy

from physics.inelastic_dielectric import generate_cross_sections as gen
from physics.inelastic_dielectric.polarization import barkas_dcs as bd
from physics.inelastic_dielectric.polarization import screened_barkas as sb
from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import DEFAULT_WORKERS, DATA_PATH, load_density

ENERGIES_MEV = (.1, .3, 1., 3., 10., 30., 40., 100.)
STATES = (("H", 0), ("He", 0), ("He", 1), ("He", 2))


@contextmanager
def _reuse_channel_responses():
    """Reuse exactly identical calls within this benchmark, not production.

    Every channel requests the same full target response at a given W/q
    grid. Keys include all argument values, including phase and partition.
    Only the latest response is retained; no grids are approximated or fit.
    """
    names = ("epsilon1_valence_Eq", "epsilon2_valence_Eq")
    originals = {name: getattr(gen.model, name) for name in names}

    def cached(function):
        previous, response = None, None
        def evaluate(*args, **kwargs):
            nonlocal previous, response
            key = pickle.dumps((args, kwargs), protocol=5)
            if key != previous:
                response = function(*args, **kwargs)
                previous = key
            return response
        return evaluate

    for name, function in originals.items():
        setattr(gen.model, name, cached(function))
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(gen.model, name, function)


def _born_total(w, energy, s, c, nq):
    with _reuse_channel_responses():
        return np.array([
            sum(gen._selected_dsigma_excitation(loss, energy, i, s, c, nq)
                for i in range(len(s.excitations)))
            + sum(gen._selected_dsigma_ionization(loss, energy, i, s, c, nq)
                  for i in range(len(s.ionizations)))
            + gen._selected_dsigma_kshell(loss, energy, s, c, nq)
            for loss in w])


def _check(task):
    element, charge, energy, points, nq = task
    gen.set_projectile(element)
    gen._set_projectile_charge_state(charge)
    gen._set_kshell_model("hydrogenic-gos")
    density = load_density(element, charge)
    mass = gen.PROJECTILE_MASS_AU
    beta, gamma = bd.projectile_beta_gamma(energy, mass)
    wmax = float(bd.wmax_eV(energy, mass))
    # Include the published failure losses and the physical O K edge.
    special = np.array([15., 50., 1000., gen.KSHELL_B_EV*(1-1e-8),
                        gen.KSHELL_B_EV*(1+1e-8)])
    w = np.unique(np.r_[np.geomspace(8., wmax, points),
                         special[(special >= 8.) & (special <= wmax)]])
    result = dict(element=element, charge=charge, T_total_eV=energy,
                  beta=float(beta), gamma=float(gamma), Wmax_eV=wmax,
                  W_eV=w.tolist(), numerical_failures=[], comparisons=[])
    kernel, error = np.full_like(w, np.nan), np.full_like(w, np.nan)
    details = []
    start = time.monotonic()
    if density.electrons:
        for i, loss in enumerate(w):
            try:
                if beta <= bd.ALPHA_FINE:
                    raise ValueError("Outside screened model domain: v <= Bohr velocity")
                k, e, diagnostics = sb.converged_kernel([loss], beta, gamma, density,
                                                         return_diagnostics=True)
                kernel[i], error[i] = k[0], e[0]
                details.append(diagnostics[0])
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                result["numerical_failures"].append(dict(W_eV=float(loss), error=str(exc)))
        result["adaptive_seconds"] = time.monotonic()-start
    else:
        xi = bd.barkas_xi(w, energy, mass)
        kernel = density.z**3*(bd.arbi1(xi)+bd.arbi2(xi)/gamma**2)
        error.fill(0.)
        result["bare_dispatch"] = "unchanged Salvat"
    result["kernel"] = [float(v) if np.isfinite(v) else None for v in kernel]
    result["kernel_error"] = [float(v) if np.isfinite(v) else None for v in error]
    result["error_components"] = details
    pref = bd.CM2_TO_M2*4*np.pi*bd.RE_CLASSICAL_CM**2*bd.ALPHA_FINE/(gamma**2*beta**5)
    for phase in ("amorphous", "hexagonal"):
        gen.ICE_TYPE = phase
        s = gen.model.epsilon_optical(phase)
        c = gen.model.default_dispersion_coefficients()
        correction = pref*bd.oos_density(w, s, phase).df_dW_total*kernel
        for relativistic in (False, True):
            gen._set_projectile_relativistic_dcs(relativistic)
            born = _born_total(w, energy, s, c, nq)
            total = born+correction
            known = np.isfinite(correction)
            positive = born > 0
            ratios = np.divide(np.abs(correction), born, out=np.zeros_like(w), where=positive)
            uncontrolled = known & positive & (ratios >= 1)
            baseline_bad = (~np.isfinite(born)) | (born < 0)
            # Do not classify unresolved numerical nodes as physical failures.
            invalid_total = known & ((~np.isfinite(total)) | (total < 0))
            row = dict(phase=phase, kernel="rpwba" if relativistic else "pwba",
                       born_baseline_pass=not bool(np.any(baseline_bad)),
                       corrected_numerically_complete=bool(np.all(known)),
                       uncontrolled_W_eV=w[uncontrolled].tolist(),
                       uncontrolled_guard_applies=bool(density.electrons),
                       invalid_total_W_eV=w[invalid_total].tolist(),
                       born_m2_per_eV=born.tolist(),
                       correction_m2_per_eV=[float(v) if np.isfinite(v) else None for v in correction],
                       max_correction_over_born=float(np.max(ratios[known])) if np.any(known) else None)
            row["corrected_acceptance_pass"] = bool(
                np.all(known) and not np.any(baseline_bad | invalid_total)
                and (not density.electrons or not np.any(uncontrolled)))
            result["comparisons"].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--loss-points", type=int, default=24)
    parser.add_argument("--dq", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/
                        "runs/integration/integration_report.json")
    args = parser.parse_args()
    if args.workers < 1 or args.loss_points < 2 or args.dq < 2:
        parser.error("workers >= 1, loss-points >= 2, and dq >= 2 are required")
    tasks = [(e, q, t*1e6, args.loss_points, args.dq) for e, q in STATES for t in ENERGIES_MEV]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)),
                             mp_context=get_context("spawn")) as pool:
        rows = []
        for row in pool.map(_check, tasks):
            rows.append(row)
            failed = sum(not r["corrected_acceptance_pass"] for r in row["comparisons"])
            print(f"{row['element']}{row['charge']} T={row['T_total_eV']/1e6:g} MeV: "
                  f"numerical failures={len(row['numerical_failures'])}, "
                  f"corrected rejections={failed}/4", flush=True)
    paths = [Path(__file__), Path(sb.__file__), Path(sb.oscillator_quadrature.__file__),
             Path(bd.__file__), Path(gen.__file__), DATA_PATH]
    root = Path(gen.__file__).parents[2]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Representative losses/energies; not a complete production grid",
                  energy_unit="total_eV", physics_changed=False, tables_exported=False,
                  workers=args.workers, loss_points=args.loss_points, dq=args.dq,
                  numpy=np.__version__, scipy=scipy.__version__,
                  quadrature_method=sb.oscillator_quadrature.VERSION,
                  quadrature_rtol=sb.QUADRATURE_RTOL,
                  source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
