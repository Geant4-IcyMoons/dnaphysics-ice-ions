#!/usr/bin/env python3
"""Independent numerical bare-projectile oscillator vs Salvat Barkas comparison.

Bypasses screened_barkas.dcs_m2_per_eV's bare-state analytic dispatch.
Both calculations use the same ice OOS and exact Barkas Wmax. This is a
formula/implementation comparison, not an experimental ice validation.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

PHYSICS_ROOT = Path(__file__).resolve().parents[3]
from physics.inelastic_dielectric.polarization import barkas_dcs as bd
from physics.inelastic_dielectric.polarization import oscillator_quadrature as oq
from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import DEFAULT_WORKERS, load_density
from physics.constants import (AVOGADRO, H2O_MOLAR_MASS_G_MOL, PROJECTILE_LIBRARY,
                       AASTEX_FULL_WIDTH_IN, PAPER_FONTSIZE, FONT_COURIER,
                       RC_BASE_ELASTIC, rcparams_with_fontsize)

ENERGIES_MEV = np.array([.1, .3, 1., 3., 10., 30., 100.])
PHASES = ("amorphous", "hexagonal")


def _components(xi, projectile):
    # b_per_x is immaterial for a point charge; no analytic ARBI is called.
    config = PROJECTILE_LIBRARY[projectile]
    density = load_density(config["element"], int(config["charge"]))
    evaluated = [oq.integrate_kernel(xi, 1., 1., density, component=j)
                 for j in range(2)]
    return np.array([r[0] for r in evaluated]), np.array([r[1] for r in evaluated])


def _row(energy, phase, interpolators, points, projectile):
    config = PROJECTILE_LIBRARY[projectile]
    mass, charge = config["mass_au"], config["charge"]
    beta, gamma = bd.projectile_beta_gamma(energy, mass)
    wmax = float(bd.wmax_eV(energy, mass))
    s = bd.model.epsilon_optical(phase)
    w = np.geomspace(s.Bmin, wmax, points)
    edge = bd.model.OXYGEN_K_B_EV*np.array([1-1e-8, 1., 1+1e-8])
    w = np.unique(np.r_[w, edge[(edge > w[0]) & (edge < wmax)]])
    oos = bd.oos_density(w, s, phase)
    xi = bd.barkas_xi(w, energy, mass)
    numerical = (np.exp(interpolators[0](np.log(xi)))
                 + np.exp(interpolators[1](np.log(xi)))/gamma**2)
    pref = bd.CM2_TO_M2*4*np.pi*bd.RE_CLASSICAL_CM**2*bd.ALPHA_FINE/(gamma**2*beta**5)
    ours = pref*oos.df_dW_total*numerical
    salvat = bd.barkas_dcs_m2_per_eV(energy, w, charge, mass, oos.df_dW_total)
    ours_s = float(np.trapezoid(w*ours, w))
    reference_s = float(np.trapezoid(w*salvat, w))
    # eV m^2/molecule -> (1e4 cm^2/m^2)*(1e-6 MeV/eV)*(N_A/M) molecules/g.
    conversion = 1e-2*AVOGADRO/H2O_MOLAR_MASS_G_MOL
    mask = salvat > 0
    return dict(projectile=projectile, phase=phase, T_MeV=energy/1e6,
                T_MeV_per_u=energy/1e6/config["mass_number"],
                beta=float(beta), gamma=float(gamma),
                Wmax_eV=wmax, S_numerical_eV_m2=ours_s, S_salvat_eV_m2=reference_s,
                S_numerical_MeV_cm2_g=ours_s*conversion,
                S_salvat_MeV_cm2_g=reference_s*conversion,
                relative_difference=ours_s/reference_s-1.,
                max_dcs_relative_difference=float(np.max(np.abs(ours[mask]/salvat[mask]-1))),
                valence_integral=oos.valence_integral_raw*oos.valence_norm,
                K_integral=oos.ok_integral_raw*oos.ok_norm,
                total_oos_integral=oos.total_integral_norm_grid)


def _plot(rows, out, projectile):
    font = None
    for candidate in (FONT_COURIER, "Courier New", "Courier"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
        except ValueError:
            continue
        font = candidate
        break
    if font is None:
        raise RuntimeError("Install Courier, Courier New or Nimbus Mono PS to render the comparison")
    plt.rcParams.update(rcparams_with_fontsize(RC_BASE_ELASTIC, PAPER_FONTSIZE, {
        "font.family": font, "axes.grid": False,
        "figure.titlesize": PAPER_FONTSIZE, "pdf.fonttype": 42,
        "axes.titlepad": 4, "axes.labelpad": 4, "axes.linewidth": .8,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2, "ytick.minor.size": 2,
        "xtick.major.width": .7, "ytick.major.width": .7,
        "xtick.minor.width": .6, "ytick.minor.width": .6,
        "mathtext.fontset": "custom", "mathtext.rm": font,
        "mathtext.it": font, "mathtext.bf": font, "mathtext.fallback": None}))
    fig, axes = plt.subplots(2, 2, figsize=(AASTEX_FULL_WIDTH_IN, 4.4), sharex="col", sharey="row",
                             gridspec_kw={"height_ratios": [3, 1], "hspace": .07})
    colors = plt.get_cmap("plasma")(np.linspace(0., 1., 3))[:-1]
    extent = max(abs(100*r["relative_difference"]) for r in rows)
    for i, phase in enumerate(PHASES):
        subset = [r for r in rows if r["phase"] == phase]
        energy = np.array([r["T_MeV"] for r in subset])
        ax, residual = axes[:, i]
        ax.loglog(energy, [r["S_numerical_MeV_cm2_g"] for r in subset],
                  color=colors[0], lw=1.2, label="Numerical oscillator")
        ax.loglog(energy, [r["S_salvat_MeV_cm2_g"] for r in subset],
                  color=colors[1], ls="--", lw=1.2, label="Salvat/SBETHE Barkas")
        ax.set_title(f"({'ab'[i]})", loc="left")
        ax.text(.95, .92, phase.capitalize()+" ice", transform=ax.transAxes, ha="right", va="top")
        residual.semilogx(energy, [100*r["relative_difference"] for r in subset],
                         color=colors[0], lw=1.1)
        residual.axhline(0., color="0.4", lw=.7)
        residual.set_xlabel("Kinetic energy ($T$; MeV)")
        residual.set_ylim(-max(1e-6, extent*1.4), max(1e-6, extent*1.4))
        for panel in (ax, residual):
            panel.grid(False, which="both")
            panel.tick_params(which="both", direction="in", top=False, right=False)
    axes[0, 0].set_ylabel("Polarization stopping\n(MeV cm$^2$ g$^{-1}$)")
    axes[1, 0].set_ylabel("Difference (%)")
    fig.suptitle("Bare proton (H$^+$)" if projectile == "proton" else "Bare alpha (He$^{2+}$)", y=.985)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", frameon=False,
               bbox_to_anchor=(.55, .035))
    fig.text(.55, .015, "Difference = $100\\,(S_{\\rm oscillator}/S_{\\rm Salvat}-1)$", ha="center")
    fig.subplots_adjust(left=.13, right=.98, bottom=.20, top=.90, wspace=.10)
    path = out/f"{projectile}_oscillator_vs_salvat.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projectile", choices=("proton", "alpha"), default="proton")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--plot-only", action="store_true",
                        help="Render comparison.json without recomputation or provenance changes")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.out_dir is None:
        args.out_dir = Path(__file__).resolve().parent/f"plots/point_{args.projectile}"
    if args.plot_only:
        report = json.loads((args.out_dir/"comparison.json").read_text())
        if report["projectile"] != args.projectile:
            parser.error("Saved report projectile does not match --projectile")
        print(_plot(report["rows"], args.out_dir, args.projectile))
        return 0
    # All incident energies are total per ion, not MeV/u.
    config = PROJECTILE_LIBRARY[args.projectile]
    mass, charge = config["mass_au"], config["charge"]
    energies = ENERGIES_MEV*1e6
    lo = float(bd.barkas_xi(7., energies[-1], mass))
    hi = float(np.max(bd.barkas_xi(bd.wmax_eV(energies, mass), energies, mass)))
    xi = np.geomspace(lo*(1-1e-12), hi*(1+1e-12), 385)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        calculated = list(tqdm(pool.map(partial(_components, projectile=args.projectile), xi),
                               total=len(xi), desc="Independent oscillator integrals"))
    values = np.array([r[0] for r in calculated])
    error = np.array([r[1] for r in calculated])
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise RuntimeError("Invalid numerical oscillator kernel")
    interpolators = [PchipInterpolator(np.log(xi), np.log(values[:, j]), extrapolate=False)
                     for j in range(2)]
    interp_error = 0.
    for j in range(2):
        coarse = PchipInterpolator(np.log(xi[::2]), np.log(values[::2, j]))
        interp_error = max(interp_error, float(np.max(np.abs(
            np.exp(coarse(np.log(xi[1::2])))/values[1::2, j]-1))))
    rows = []
    for phase in PHASES:
        for energy in energies:
            coarse = _row(energy, phase, interpolators, 2049, args.projectile)
            row = _row(energy, phase, interpolators, 4097, args.projectile)
            row["W_grid_relative_change"] = abs(coarse["S_numerical_eV_m2"]/row["S_numerical_eV_m2"]-1)
            rows.append(row)
            print(f"{phase:10s} {row['T_MeV']:6g} MeV: {100*row['relative_difference']:+.6f}%", flush=True)
    quad_error = float(np.max(error/values))
    cubic_error = 0.
    if args.projectile == "alpha":
        for argument in (.001, .1, 1.):
            proton, _ = _components(argument, "proton")
            alpha, _ = _components(argument, "alpha")
            cubic_error = max(cubic_error, float(np.max(np.abs(alpha/(8*proton)-1))))
    passed = (quad_error < .001 and interp_error < .001
              and cubic_error < 1e-10
              and all(abs(r["relative_difference"]) < .001
                      and r["max_dcs_relative_difference"] < .0015
                      and r["W_grid_relative_change"] < .001 for r in rows))
    report = dict(status="PASS" if passed else "FAIL", projectile=args.projectile,
                  projectile_state=f"{config['element']} {charge:g}+ (bare)",
                  projectile_mass_me=mass, energy_convention="total kinetic energy per ion",
                  comparison="independent numerical oscillator vs existing Salvat ARBI kernel",
                  analytic_bare_shortcut_used=False, ice_physics_validation=False,
                  quadrature_method=oq.VERSION,
                  screened_state_validation=False, charge_mode="bare", z=charge,
                  xi_nodes=xi.tolist(), charge_cubed_I1_numerical=values[:, 0].tolist(),
                  charge_cubed_I2_numerical=values[:, 1].tolist(),
                  alpha_proton_cubic_check_relative_error=(cubic_error if args.projectile == "alpha" else None),
                  max_quadrature_relative_error=quad_error,
                  max_interpolation_relative_change=interp_error,
                  units="eV m^2 per molecule; multiply by 1e-2*N_A/M_H2O for MeV cm^2/g",
                  criteria="stopping moment <0.1%; DCS <0.15%; each numerical check <0.1%; alpha/proton cubic scaling error <1e-10",
                  code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path(__file__), Path(oq.__file__),
                                PHYSICS_ROOT/"inelastic_dielectric/polarization/barkas_dcs.py", PHYSICS_ROOT/"constants.py",
                                PHYSICS_ROOT/"inelastic_dielectric/finite_q/emfietzoglou_model_finite_q.py",
                                PHYSICS_ROOT/"inelastic_dielectric/k_shell/hydrogenic.py",
                                PHYSICS_ROOT/"inelastic_dielectric/projectile_potentials/projectile_form_factors.py"]}, rows=rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/"comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(_plot(rows, args.out_dir, args.projectile))
    print(report["status"], "quadrature", quad_error, "interpolation", interp_error)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
