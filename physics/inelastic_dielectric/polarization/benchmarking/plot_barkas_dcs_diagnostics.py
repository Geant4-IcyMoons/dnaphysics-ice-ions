#!/usr/bin/env python3
"""Lightweight demonstration plots for the Barkas DCS implementation.

These plots check the implemented Barkas machinery without regenerating any
Geant4 DCS/TCS tables and without comparing or tuning against ICRU curves.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics.inelastic_dielectric.polarization import barkas_dcs
from physics.inelastic_dielectric.finite_q import emfietzoglou_model_finite_q as model
from physics.constants import CROSS_SECTION_PLOTS_DIR, PROJECTILE_LIBRARY, RC_BASE_ELASTIC, rcparams_with_fontsize


DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "plots/dcs_diagnostics"


def _positive(y):
    y = np.asarray(y, dtype=float)
    return np.where(np.isfinite(y) & (y > 0.0), y, np.nan)


def _save(fig, out_dir, stem, formats):
    paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _projectile_config(name):
    key = str(name).strip().lower()
    for projectile, cfg in PROJECTILE_LIBRARY.items():
        aliases = {projectile, *cfg.get("aliases", ())}
        if key in aliases:
            return projectile, cfg
    choices = ", ".join(sorted(PROJECTILE_LIBRARY))
    raise ValueError(f"Unknown projectile '{name}'. Choices: {choices}.")


def _make_oos_plot(material, out_dir, formats):
    s = model.epsilon_optical(material)
    W = np.geomspace(1.0e-1, 1.0e6, 2500)
    oos = barkas_dcs.oos_density(W, s, material=material, include_kshell=True)
    valence_int = oos.valence_integral_raw * oos.valence_norm
    ok_int = oos.ok_integral_raw * oos.ok_norm
    total_int = oos.total_integral_norm_grid

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.loglog(W, _positive(oos.df_dW_valence), color="#34495e", lw=2.2, label=f"Valence OOS, integral={valence_int:.3g}")
    ax.loglog(W, _positive(oos.df_dW_OK), color="#c0392b", lw=2.2, ls="--", label=f"O K OOS, integral={ok_int:.3g}")
    ax.loglog(W, _positive(oos.df_dW_total), color="black", lw=2.5, label=f"Total OOS, integral={total_int:.3g}")
    ax.axvline(model.OXYGEN_K_B_EV, color="#c0392b", lw=1.5, alpha=0.55)
    ax.text(model.OXYGEN_K_B_EV * 1.08, ax.get_ylim()[0] * 1.8, "O K edge", color="#c0392b", fontsize=11)
    ax.set_xlabel("Energy loss W (eV)")
    ax.set_ylabel("Optical oscillator strength density df/dW (eV$^{-1}$)")
    ax.set_title(f"Barkas OOS normalization, {material} ice")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.35)
    return _save(fig, out_dir, f"barkas_oos_density_{material}", formats)


def _make_kernel_plot(projectile_key, projectile_cfg, T_total_eV, out_dir, formats):
    mass_me = float(projectile_cfg["mass_au"])
    beta, gamma = barkas_dcs.projectile_beta_gamma(T_total_eV, mass_me)
    Wmax = float(barkas_dcs.wmax_eV(T_total_eV, mass_me))
    W_min = max(1.0e-3, Wmax * 1.0e-5)
    W_max = max(W_min * 1.01, 0.999 * Wmax)
    W = np.geomspace(W_min, W_max, 1200)
    xi = barkas_dcs.barkas_xi(W, T_total_eV, mass_me)
    I1 = barkas_dcs.arbi1(xi)
    I2_rel = barkas_dcs.arbi2(xi) / gamma**2
    kernel = I1 + I2_rel

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.loglog(W, _positive(I1), color="#2c3e50", lw=2.2, label=r"$I_1(\xi)$")
    ax.loglog(W, _positive(I2_rel), color="#16a085", lw=2.2, ls="--", label=r"$\gamma^{-2} I_2(\xi)$")
    ax.loglog(W, _positive(kernel), color="black", lw=2.6, label=r"$I_1+\gamma^{-2} I_2$")
    ax.axvline(Wmax, color="#7f8c8d", lw=1.4, alpha=0.6, label=f"Wmax={Wmax:.3g} eV")
    ax.set_xlabel("Energy loss W (eV)")
    ax.set_ylabel("Barkas kernel factor")
    ax.set_title(f"Barkas kernel terms, {projectile_key}, T={T_total_eV:.3g} eV total")
    ax.text(
        0.03,
        0.04,
        f"beta={beta:.5g}, gamma={gamma:.7g}",
        transform=ax.transAxes,
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 4},
    )
    ax.legend(loc="best", frameon=False)
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.35)
    return _save(fig, out_dir, f"barkas_kernel_terms_{projectile_key}_{T_total_eV:.0f}eV", formats)


def _make_charge_scaling_plot(projectile_key, projectile_cfg, material, T_total_eV, out_dir, formats):
    mass_me = float(projectile_cfg["mass_au"])
    Wmax = float(barkas_dcs.wmax_eV(T_total_eV, mass_me))
    W0 = max(1.0, min(100.0, 0.25 * Wmax))
    s = model.epsilon_optical(material)
    oos = barkas_dcs.oos_density(np.asarray([W0]), s, material=material, include_kshell=True)
    z = np.asarray([1.0, 2.0, 3.0, 4.0, 6.0, 8.0], dtype=float)
    born_norm = z**2
    barkas = barkas_dcs.barkas_dcs_m2_per_eV(
        np.full_like(z, T_total_eV),
        np.full_like(z, W0),
        z_int=z,
        projectile_mass_me=mass_me,
        df_dW=np.full_like(z, oos.df_dW_total[0]),
    )
    barkas_norm = barkas / barkas[0]

    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    ax.plot(z, born_norm, "o-", color="#34495e", lw=2.2, label=r"Born scaling $z_{int}^2$")
    ax.plot(z, barkas_norm, "s--", color="#c0392b", lw=2.2, label=r"Barkas scaling $z_{int}^3$")
    ax.plot(z, z**2, color="#34495e", lw=1.0, alpha=0.35)
    ax.plot(z, z**3, color="#c0392b", lw=1.0, alpha=0.35)
    ax.set_yscale("log")
    ax.set_xlabel("Interaction charge $z_{int}$")
    ax.set_ylabel("DCS scaling relative to $z_{int}=1$")
    ax.set_title(f"Same interaction charge in Born and Barkas terms\n{projectile_key}, W={W0:.3g} eV")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.35)
    return _save(fig, out_dir, f"barkas_charge_scaling_{projectile_key}_{T_total_eV:.0f}eV", formats)


def _synthetic_dcs_diagnostic(projectile_cfg, material, T_total_eV):
    mass_me = float(projectile_cfg["mass_au"])
    z = float(projectile_cfg["charge"])
    s = model.epsilon_optical(material)
    Wmax = float(barkas_dcs.wmax_eV(T_total_eV, mass_me))
    W = np.geomspace(max(1.0e-1, Wmax * 1.0e-4), max(2.0, 0.95 * Wmax), 500)
    baseline = 3.0e-24 * (W / W[0]) ** -0.45 * np.exp(-W / max(1.0, 0.8 * Wmax))
    dcs_data = {
        "T_line": np.full(W.size, T_total_eV),
        "E_line": W,
        "exc_vals": (0.35 * baseline).reshape(-1, 1),
        "ion_vals": (0.65 * baseline).reshape(-1, 1),
    }
    _, diag = barkas_dcs.apply_barkas_correction_to_dcs_data(
        dcs_data,
        s,
        material=material,
        projectile_mass_me=mass_me,
        nuclear_charge=z,
        charge_mode="bare",
        include_barkas_dcs=True,
        include_kshell=True,
        dcs_scale_m2=1.0,
        born_reference_charge="bare_Z",
    )
    return W, diag


def _make_total_dcs_plot(projectile_key, projectile_cfg, material, T_total_eV, out_dir, formats):
    W, diag = _synthetic_dcs_diagnostic(projectile_cfg, material, T_total_eV)
    ratio = np.divide(
        diag.DCS_Barkas_m2_per_eV,
        diag.DCS_Born_m2_per_eV,
        out=np.full_like(diag.DCS_Barkas_m2_per_eV, np.nan),
        where=diag.DCS_Born_m2_per_eV > 0.0,
    )

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.15]})
    ax.loglog(W, _positive(diag.DCS_Born_m2_per_eV), color="#34495e", lw=2.2, label="Synthetic Born baseline")
    ax.loglog(W, _positive(diag.DCS_Barkas_m2_per_eV), color="#c0392b", lw=2.2, ls="--", label="Additive Barkas DCS")
    ax.loglog(W, _positive(diag.DCS_total_m2_per_eV), color="black", lw=2.5, label="Total used for TCS/CDF")
    ax.set_ylabel("DCS (m$^2$/eV)")
    ax.set_title(f"Born + Barkas assembly check, {projectile_key}, T={T_total_eV:.3g} eV total")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.35)

    rax.semilogx(W, ratio, color="#c0392b", lw=2.0)
    rax.axhline(0.0, color="black", lw=1.0)
    rax.set_xlabel("Energy loss W (eV)")
    rax.set_ylabel("Barkas/Born")
    rax.grid(True, which="both", ls=":", lw=0.6, alpha=0.35)
    return _save(fig, out_dir, f"barkas_total_dcs_synthetic_{projectile_key}_{T_total_eV:.0f}eV", formats), diag


def _make_summary_plot(material, diag, out_dir, formats):
    valence_int = diag.df_dW_valence_raw_integral * diag.df_dW_valence_norm
    ok_int = diag.df_dW_OK_raw_integral * diag.df_dW_OK_norm
    total_int = diag.df_dW_integral_norm_grid
    total_check = np.trapezoid(diag.DCS_total_m2_per_eV, diag.W_line)
    s_check = np.trapezoid(diag.W_line * diag.DCS_Barkas_m2_per_eV, diag.W_line)
    tcs_resid = abs(total_check - diag.TCS_total_m2[0]) / max(abs(diag.TCS_total_m2[0]), np.finfo(float).tiny)
    s_resid = abs(s_check - diag.S_Barkas_check_eV_m2[0]) / max(abs(diag.S_Barkas_check_eV_m2[0]), np.finfo(float).tiny)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"wspace": 0.42})
    labels = ["Valence", "O K", "Total"]
    vals = [valence_int, ok_int, total_int]
    targets = [8.0, 2.0, 10.0]
    x = np.arange(len(labels))
    ax1.bar(x - 0.17, vals, width=0.34, color="#34495e", label="Implemented")
    ax1.bar(x + 0.17, targets, width=0.34, color="#95a5a6", label="Target")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("OOS integral per H2O")
    ax1.set_title(f"OOS normalization, {material}")
    ax1.legend(frameon=False)
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.35)

    residuals = np.asarray([tcs_resid, s_resid], dtype=float)
    residual_floor = 1.0e-16
    residual_display = np.maximum(residuals, residual_floor)
    ax2.bar(["TCS integral", "Stopping moment"], residual_display, color=["#2c3e50", "#c0392b"])
    ax2.set_yscale("log")
    ax2.set_ylim(residual_floor / 3.0, 1.0e-12)
    ax2.set_ylabel("Relative residual")
    ax2.set_title("Numerical closure checks")
    ax2.grid(True, axis="y", ls=":", lw=0.6, alpha=0.35)
    for i, value in enumerate(residuals):
        ax2.text(i, residual_display[i] * 1.4, f"{value:.1e}", ha="center", va="bottom", fontsize=10)
    return _save(fig, out_dir, f"barkas_validation_summary_{material}", formats)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default="amorphous", choices=("amorphous", "hexagonal"))
    parser.add_argument("--projectile", default="proton", help="Projectile name or alias from PROJECTILE_LIBRARY.")
    parser.add_argument("--T-eV", type=float, default=1.0e6, help="Total kinetic energy per ion in eV.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--formats", nargs="+", default=("pdf",), choices=("pdf",))
    args = parser.parse_args(argv)

    if args.T_eV <= 0.0:
        raise ValueError("--T-eV must be positive and is interpreted as total kinetic energy per ion.")

    plt.rcParams.update(
        rcparams_with_fontsize(
            RC_BASE_ELASTIC,
            13,
            {
                "font.family": "DejaVu Sans",
                "figure.dpi": 120,
                "savefig.dpi": 300,
            },
        )
    )

    projectile_key, projectile_cfg = _projectile_config(args.projectile)
    paths = []
    paths.extend(_make_oos_plot(args.material, args.out_dir, args.formats))
    paths.extend(_make_kernel_plot(projectile_key, projectile_cfg, args.T_eV, args.out_dir, args.formats))
    paths.extend(_make_charge_scaling_plot(projectile_key, projectile_cfg, args.material, args.T_eV, args.out_dir, args.formats))
    total_paths, diag = _make_total_dcs_plot(projectile_key, projectile_cfg, args.material, args.T_eV, args.out_dir, args.formats)
    paths.extend(total_paths)
    paths.extend(_make_summary_plot(args.material, diag, args.out_dir, args.formats))

    print("Barkas diagnostic plots written:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
