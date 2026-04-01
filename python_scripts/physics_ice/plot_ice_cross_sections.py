#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from constants import (
    FONT_COURIER,
    FONTSIZE_24,
    OUTPUT_DIR,
    RC_BASE_ELASTIC,
    rcparams_with_fontsize,
)


def plot_full_cross_sections_per_channel(
    T_list,
    sigma_list,
    s,
    regime_flags_func,
    ax=None,
    linewidth=2,
    alpha=0.9,
    figsize=(12, 8),
):
    """Plot PWBA vs selected default model cross section for each channel."""
    if ax is None:
        fig_ion, ax_ion = plt.subplots(figsize=figsize)
        fig_exc, ax_exc = plt.subplots(figsize=figsize)
    else:
        ax_ion, ax_exc = ax

    T_arr = np.asarray(T_list, dtype=float)
    nT = len(T_arr)

    exc_colors = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6"]
    ion_colors = ["#67000d", "#a50f15", "#cb181d", "#ef3b2c", "#fb6a4a"]

    def _get_list(i, key, default):
        return sigma_list[i].get(key, default)

    n_ion = len(s.ionizations)
    for j in range(n_ion):
        y_pwba = []
        y_model = []
        for i in range(nT):
            Tj = float(T_arr[i])
            pwba_list = _get_list(i, "ionization_sigma_pwba", [0.0] * n_ion)
            pwba = pwba_list[j] if j < len(pwba_list) else 0.0

            use_mc, use_rel_long, use_rel_trans, _ = regime_flags_func(Tj)
            if use_mc:
                mc_list = _get_list(i, "ionization_sigma_mc", None)
                model = mc_list[j] if (mc_list is not None and j < len(mc_list)) else pwba
            elif use_rel_long:
                relL_list = _get_list(i, "ionization_sigma_rel", [0.0] * n_ion)
                relT_list = _get_list(i, "ionization_sigma_rel_trans", [0.0] * n_ion)
                relL = relL_list[j] if j < len(relL_list) else 0.0
                relT = relT_list[j] if j < len(relT_list) else 0.0
                model = relL + (relT if use_rel_trans else 0.0)
            else:
                model = pwba

            y_pwba.append(pwba)
            y_model.append(model)

        ax_ion.loglog(T_arr, y_pwba, color=ion_colors[j % len(ion_colors)], lw=linewidth, alpha=alpha, ls="--", label=f"Ion. {j+1} PWBA")
        ax_ion.loglog(T_arr, y_model, color=ion_colors[j % len(ion_colors)], lw=linewidth, alpha=alpha, ls="-", label=f"Ion. {j+1} Default model")

    ax_ion.set_xlabel("Electron Energy (T; eV)")
    ax_ion.set_ylabel("Cross section sigma(T)")
    ax_ion.set_title("Ionizations: PWBA baseline vs Default model")
    ax_ion.grid(True, which="both", ls="--", alpha=0.3)
    ax_ion.legend(loc="best", fontsize=8)

    n_exc = len(s.excitations)
    for k in range(n_exc):
        y_pwba = []
        y_model = []
        for i in range(nT):
            Tj = float(T_arr[i])
            pwba_list = _get_list(i, "excitation_sigma_pwba", [0.0] * n_exc)
            pwba = pwba_list[k] if k < len(pwba_list) else 0.0

            use_mc, use_rel_long, use_rel_trans, _ = regime_flags_func(Tj)
            if use_mc:
                mc_list = _get_list(i, "excitation_sigma_mc", None)
                model = mc_list[k] if (mc_list is not None and k < len(mc_list)) else pwba
            elif use_rel_long:
                relL_list = _get_list(i, "excitation_sigma_rel", [0.0] * n_exc)
                relT_list = _get_list(i, "excitation_sigma_rel_trans", [0.0] * n_exc)
                relL = relL_list[k] if k < len(relL_list) else 0.0
                relT = relT_list[k] if k < len(relT_list) else 0.0
                model = relL + (relT if use_rel_trans else 0.0)
            else:
                model = pwba

            y_pwba.append(pwba)
            y_model.append(model)

        ax_exc.loglog(T_arr, y_pwba, color=exc_colors[k % len(exc_colors)], lw=linewidth, alpha=alpha, ls="--", label=f"Exc. {k+1} PWBA")
        ax_exc.loglog(T_arr, y_model, color=exc_colors[k % len(exc_colors)], lw=linewidth, alpha=alpha, ls="-", label=f"Exc. {k+1} Default model")

    ax_exc.set_xlabel("Electron Energy (T; eV)")
    ax_exc.set_ylabel("Cross section sigma(T)")
    ax_exc.set_title("Excitations: PWBA baseline vs Default model")
    ax_exc.grid(True, which="both", ls="--", alpha=0.3)
    ax_exc.legend(loc="best", fontsize=8)
    return ax_ion, ax_exc


def plot_relativistic_component_per_channel(
    T_list,
    sigma_list,
    s,
    regime_ii_max_eV,
    ax=None,
    linewidth=2,
    alpha=0.9,
    figsize=(12, 8),
):
    """Plot longitudinal and transverse relativistic components per channel."""
    T_arr = np.asarray(T_list, dtype=float)
    mask = T_arr >= regime_ii_max_eV
    if not np.any(mask):
        raise ValueError("No T values >= REGIME_II_MAX_eV found in T_list.")

    Tm = T_arr[mask]
    idxs = np.where(mask)[0]

    exc_colors = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6"]
    ion_colors = ["#67000d", "#a50f15", "#cb181d", "#ef3b2c", "#fb6a4a"]

    n_exc = len(s.excitations)
    n_ion = len(s.ionizations)

    if ax is None:
        fig_ion, ax_ion = plt.subplots(figsize=figsize)
        fig_exc, ax_exc = plt.subplots(figsize=figsize)
    else:
        ax_ion, ax_exc = ax

    print(f"Plot IONIZATIONS (REL longitudinal solid, transverse dashed; T>={regime_ii_max_eV:.1e} eV)")
    for j in tqdm(range(n_ion)):
        y_long = []
        y_trans = []
        for i in idxs:
            long_list = sigma_list[i].get("ionization_sigma_rel", [0.0] * n_ion)
            trans_list = sigma_list[i].get("ionization_sigma_rel_trans", [0.0] * n_ion)
            y_long.append(long_list[j] if j < len(long_list) else 0.0)
            y_trans.append(trans_list[j] if j < len(trans_list) else 0.0)

        ax_ion.loglog(Tm, y_long, color=ion_colors[j % len(ion_colors)], lw=linewidth, alpha=alpha, label=f"Ion. {j+1} Long")
        ax_ion.loglog(Tm, y_trans, color=ion_colors[j % len(ion_colors)], lw=linewidth, alpha=alpha, ls="--", label=f"Ion. {j+1} Trans")

    ax_ion.set_xlabel("Electron energy (T; eV)")
    ax_ion.set_ylabel("Relativistic component sigma_rel(T)")
    ax_ion.legend(loc="best", fontsize=8)
    ax_ion.grid(True, which="both", ls="--", alpha=0.3)

    print(f"Plot EXCITATIONS (REL longitudinal solid, transverse dashed; T>={regime_ii_max_eV:.1e} eV)")
    for k in tqdm(range(n_exc)):
        y_long = []
        y_trans = []
        for i in idxs:
            long_list = sigma_list[i].get("excitation_sigma_rel", [0.0] * n_exc)
            trans_list = sigma_list[i].get("excitation_sigma_rel_trans", [0.0] * n_exc)
            y_long.append(long_list[k] if k < len(long_list) else 0.0)
            y_trans.append(trans_list[k] if k < len(trans_list) else 0.0)

        ax_exc.loglog(Tm, y_long, color=exc_colors[k % len(exc_colors)], lw=linewidth, alpha=alpha, label=f"Exc. {k+1} Long")
        ax_exc.loglog(Tm, y_trans, color=exc_colors[k % len(exc_colors)], lw=linewidth, alpha=alpha, ls="--", label=f"Exc. {k+1} Trans")

    ax_exc.set_xlabel("Electron energy (T; eV)")
    ax_exc.set_ylabel("Relativistic component sigma_rel(T)")
    ax_exc.legend(loc="best", fontsize=8)
    ax_exc.grid(True, which="both", ls="--", alpha=0.3)
    return ax_ion, ax_exc


def plot_total_cross_section(T_list, sigma_list, regime_flags_func, ax=None, linewidth=2, alpha=0.9, figsize=(12, 8)):
    """Plot total PWBA and total corrected cross section."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    T_arr = np.asarray(T_list, dtype=float)
    y_pwba = []
    y_corr = []

    for i, Tj in enumerate(T_arr):
        pwba = float(sigma_list[i].get("total_sigma_pwba", sigma_list[i].get("total_sigma", 0.0)) or 0.0)
        use_mc, use_rel_long, use_rel_trans, _ = regime_flags_func(Tj)
        if use_mc:
            corr = float(sigma_list[i].get("total_sigma_mc", pwba) or pwba)
        elif use_rel_long:
            corr = float(sigma_list[i].get("total_sigma_rel_total", pwba) or pwba) if use_rel_trans else float(sigma_list[i].get("total_sigma_rel", pwba) or pwba)
        else:
            corr = pwba
        y_pwba.append(pwba)
        y_corr.append(corr)

    ax.loglog(T_arr, y_pwba, lw=linewidth, alpha=alpha, ls=":", label="Total PWBA")
    ax.loglog(T_arr, y_corr, lw=linewidth + 1, alpha=alpha, ls="-", label="Total (all corrections)")
    ax.set_xlabel("Electron energy (T; eV)")
    ax.set_ylabel("Total cross section sigma(T)")
    ax.set_title("Total cross section: PWBA vs all corrections")
    ax.legend(loc="best", fontsize=9)
    return ax


def plot_corrected_exc_ion_scaled(T_list, sigma_list, regime_flags_func, ax=None, linewidth=2, alpha=0.9, figsize=(12, 8)):
    """Plot corrected excitation and ionization channel cross sections."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    T_arr = np.asarray(T_list, dtype=float)
    exc_colors = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6"]
    ion_colors = ["#67000d", "#a50f15", "#cb181d", "#ef3b2c", "#fb6a4a"]

    n_exc = len(sigma_list[0].get("excitation_sigma_pwba", [])) if sigma_list else 0
    n_ion = len(sigma_list[0].get("ionization_sigma_pwba", [])) if sigma_list else 0

    exc_scaled = np.zeros((len(T_arr), n_exc), float)
    ion_scaled = np.zeros((len(T_arr), n_ion), float)
    kshell_scaled = np.full(len(T_arr), np.nan, float)

    for i, Tj in enumerate(T_arr):
        sigma = sigma_list[i]
        use_mc, use_rel_long, use_rel_trans, _ = regime_flags_func(Tj)
        if sigma.get("total_sigma_mc", None) is None:
            use_mc = False

        if use_mc:
            exc_vals = sigma.get("excitation_sigma_mc", []) or []
            ion_vals = sigma.get("ionization_sigma_mc", []) or []
        elif use_rel_long:
            exc_vals = sigma.get("excitation_sigma_rel", []) or []
            ion_vals = sigma.get("ionization_sigma_rel", []) or []
            if use_rel_trans:
                exc_vals = [a + b for a, b in zip(exc_vals, (sigma.get("excitation_sigma_rel_trans", []) or []))]
                ion_vals = [a + b for a, b in zip(ion_vals, (sigma.get("ionization_sigma_rel_trans", []) or []))]
        else:
            exc_vals = sigma.get("excitation_sigma_pwba", []) or []
            ion_vals = sigma.get("ionization_sigma_pwba", []) or []

        for j in range(min(n_exc, len(exc_vals))):
            exc_scaled[i, j] = float(exc_vals[j])
        for j in range(min(n_ion, len(ion_vals))):
            ion_scaled[i, j] = float(ion_vals[j])

        kshell_val = sigma.get("kshell_sigma_rel", None) if use_rel_long and sigma.get("kshell_sigma_rel", None) is not None else sigma.get("kshell_sigma", None)
        if kshell_val is not None:
            try:
                kshell_scaled[i] = float(kshell_val)
            except Exception:
                kshell_scaled[i] = np.nan

    for j in range(n_exc):
        ax.loglog(T_arr, exc_scaled[:, j], lw=linewidth, alpha=alpha, ls='-', color=exc_colors[j % len(exc_colors)], label=f"{j+1}")
    for j in range(n_ion):
        ax.loglog(T_arr, ion_scaled[:, j], lw=linewidth, alpha=alpha, ls='-', color=ion_colors[j % len(ion_colors)], label=f"{j+1}")
    if np.any(np.isfinite(kshell_scaled)):
        ax.loglog(T_arr, kshell_scaled, lw=linewidth, alpha=alpha, ls='-', color='purple', label='K-shell')

    ax.set_xlabel("Electron Energy ($T$; eV)", labelpad=1)
    ax.set_ylabel(r"Cross-Section (cm$^2$)")
    ax.set_xlim(1.0, np.max(T_arr))
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=50))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(NullLocator())
    return ax


def plot_total_cross_section_corrections(T_list, sigma_list, compute_correction_row_func, out_path=None, ice_label=None):
    """Plot total PWBA/corrected curves and each correction term."""
    if out_path is None:
        if ice_label is None:
            ice_label = 'ice'
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"cross_section_corrections_{ice_label}.png"

    rows = [compute_correction_row_func(T, sigma) for T, sigma in zip(T_list, sigma_list)]
    T_arr = np.array([row["T_eV"] for row in rows], float)
    pwba = np.array([row["sigma_pwba"] for row in rows], float)
    corrected = np.array([row["sigma_corrected"] for row in rows], float)
    corr_mc = np.array([row["corr_stage1_mc"] for row in rows], float)
    corr_rel_long = np.array([row["corr_stage2_rel_long"] for row in rows], float)
    corr_rel_trans = np.array([row["corr_stage3_rel_trans"] for row in rows], float)
    corr_density = np.array([row["corr_stage4_density"] for row in rows], float)

    abs_vals = np.abs(np.concatenate([corr_mc, corr_rel_long, corr_rel_trans, corr_density]))
    abs_vals = abs_vals[abs_vals > 0.0]
    linthresh = float(np.min(abs_vals)) if abs_vals.size else 1e-40

    fig, (ax_tot, ax_corr) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax_tot.loglog(T_arr, pwba, lw=2, ls=':', label='PWBA total')
    ax_tot.loglog(T_arr, corrected, lw=2, ls='-', label='Corrected total')
    ax_tot.set_ylabel('Total cross section sigma(T)')
    ax_tot.grid(True, which='both', ls='--', alpha=0.3)
    ax_tot.legend(loc='best', fontsize=9)

    ax_corr.set_xscale('log')
    ax_corr.set_yscale('symlog', linthresh=linthresh)
    ax_corr.plot(T_arr, corr_mc, lw=1.8, label='Stage 1: MC')
    ax_corr.plot(T_arr, corr_rel_long, lw=1.8, label='Stage 2: Rel long')
    ax_corr.plot(T_arr, corr_rel_trans, lw=1.8, label='Stage 3: Rel trans')
    ax_corr.plot(T_arr, corr_density, lw=1.8, label='Stage 4: Density effect')
    ax_corr.set_xlabel('Electron energy (T; eV)')
    ax_corr.set_ylabel('Correction term')
    ax_corr.grid(True, which='both', ls='--', alpha=0.3)
    ax_corr.legend(loc='best', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved correction diagnostics to {out_path}")


def _load_total_sigma_npz(npz_path, density_scale_factor_from_npz_func, infer_ice_type_from_path_func, density_scale_factor_for_ice_func):
    with np.load(npz_path) as data:
        T = np.asarray(data['T_eV'], float)
        pwba = np.asarray(data.get('total_sigma_pwba', []), float)
        corrected = np.asarray(data.get('total_sigma_corrected', data.get('total_sigma', [])), float)
        stored = density_scale_factor_from_npz_func(data)
        inferred = infer_ice_type_from_path_func(npz_path)
        if inferred is not None:
            target = density_scale_factor_for_ice_func(inferred)
            adjust = target / stored if stored > 0.0 else target
            if np.isfinite(adjust) and adjust > 0.0 and (not np.isclose(adjust, 1.0)):
                pwba = pwba * adjust
                corrected = corrected * adjust
    return T, pwba, corrected


def plot_total_cross_section_two_panel(amorphous_npz, hexagonal_npz, density_scale_factor_from_npz_func, infer_ice_type_from_path_func, density_scale_factor_for_ice_func, out_path=None):
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    T_a, pwba_a, corr_a = _load_total_sigma_npz(amorphous_npz, density_scale_factor_from_npz_func, infer_ice_type_from_path_func, density_scale_factor_for_ice_func)
    T_h, pwba_h, corr_h = _load_total_sigma_npz(hexagonal_npz, density_scale_factor_from_npz_func, infer_ice_type_from_path_func, density_scale_factor_for_ice_func)

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2, height_ratios=[3.0, 0.8], width_ratios=[1.0, 1.0], hspace=0.30, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis('off')

    ln1 = ax_a.loglog(T_a, pwba_a, 'k:', linewidth=2, label='Total PWBA')[0]
    ln2 = ax_a.loglog(T_a, corr_a, 'k-', linewidth=2.5, label='Total (all corrections)')[0]
    ax_a.set_xlabel('Electron energy ($T$; eV)')
    ax_a.set_ylabel('Total cross section sigma(T)')
    ax_a.set_title('Amorphous ice')

    ax_h.loglog(T_h, pwba_h, 'k:', linewidth=2, label='Total PWBA')
    ax_h.loglog(T_h, corr_h, 'k-', linewidth=2.5, label='Total (all corrections)')
    ax_h.set_xlabel('Electron energy ($T$; eV)')
    ax_h.set_ylabel('')
    ax_h.set_title('Hexagonal ice')

    y_lo = min(ax_a.get_ylim()[0], ax_h.get_ylim()[0])
    y_hi = max(ax_a.get_ylim()[1], ax_h.get_ylim()[1])
    if not np.isfinite(y_lo) or y_lo <= 0.0:
        y_lo = 1e-30
    if not np.isfinite(y_hi) or y_hi <= y_lo:
        y_hi = y_lo * 10.0
    ax_a.set_ylim(y_lo, y_hi)
    ax_h.set_ylim(y_lo, y_hi)
    for ax in (ax_a, ax_h):
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=50))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(NullLocator())

    legend_ax.legend([ln1, ln2], ['Total PWBA', 'Total (all corrections)'], loc='center', ncol=2, frameon=False, columnspacing=1.5, handlelength=2.5, labelspacing=0.8)

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches='tight')
        print(f"Saved two-panel total cross section plot to {out_path}")
    plt.close(fig)


def plot_channel_cross_sections_two_panel(amorphous_npz, hexagonal_npz, sigma_list_from_npz_func, density_scale_factor_from_npz_func, density_scale_factor_for_ice_func, regime_flags_func, out_path=None):
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    with np.load(amorphous_npz) as data_a:
        stored_a = density_scale_factor_from_npz_func(data_a)
        target_a = density_scale_factor_for_ice_func('amorphous')
        adjust_a = target_a / stored_a if stored_a > 0.0 else target_a
        T_a, sigma_a = sigma_list_from_npz_func(data_a, scale_factor=adjust_a)

    with np.load(hexagonal_npz) as data_h:
        stored_h = density_scale_factor_from_npz_func(data_h)
        target_h = density_scale_factor_for_ice_func('hexagonal')
        adjust_h = target_h / stored_h if stored_h > 0.0 else target_h
        T_h, sigma_h = sigma_list_from_npz_func(data_h, scale_factor=adjust_h)

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2, height_ratios=[3.0, 0.8], width_ratios=[1.0, 1.0], hspace=0.30, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    plot_corrected_exc_ion_scaled(T_a, sigma_a, regime_flags_func=regime_flags_func, ax=ax_a)
    plot_corrected_exc_ion_scaled(T_h, sigma_h, regime_flags_func=regime_flags_func, ax=ax_h)
    ax_a.set_title('Amorphous ice')
    ax_h.set_title('Hexagonal ice')
    ax_a.set_xlabel('Electron energy ($T$; eV)')
    ax_h.set_xlabel('Electron energy ($T$; eV)')
    ax_a.set_ylabel(r'Total cross-section (cm$^2$)')
    ax_h.set_ylabel('')

    max_x = max(float(np.max(T_a)), float(np.max(T_h)))
    ax_a.set_xlim(1.0, max_x)
    ax_h.set_xlim(1.0, max_x)
    ax_a.set_ylim(bottom=1e-25)
    ax_h.set_ylim(bottom=1e-25)

    handles, labels = ax_a.get_legend_handles_labels()
    n_exc = len(sigma_a[0].get('excitation_sigma_pwba', [])) if sigma_a else 0
    n_ion = len(sigma_a[0].get('ionization_sigma_pwba', [])) if sigma_a else 0
    exc_handles = handles[:n_exc]
    ion_handles = handles[n_exc:n_exc + n_ion]
    kshell_handle = handles[labels.index('K-shell')] if 'K-shell' in labels else None

    gs_leg = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, :], height_ratios=[1.0, 1.0], hspace=0.30)
    ax_leg_exc = fig.add_subplot(gs_leg[0, 0])
    ax_leg_ion = fig.add_subplot(gs_leg[1, 0])
    ax_leg_exc.axis('off')
    ax_leg_ion.axis('off')

    if exc_handles:
        exc_labels = ['Excitation'] + [str(i + 1) for i in range(n_exc)]
        exc_handles = [Line2D([], [], color='none', linestyle='none')] + exc_handles
        ax_leg_exc.legend(exc_handles, exc_labels, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=max(1, len(exc_labels)), frameon=False, columnspacing=0.9, handlelength=2.2, handletextpad=0.6, borderaxespad=0.0)
    if ion_handles:
        ion_labels = ['Ionization'] + [str(i + 1) for i in range(n_ion)]
        ion_handles = [Line2D([], [], color='none', linestyle='none')] + ion_handles
        if kshell_handle is not None:
            ion_labels.append('K-shell')
            ion_handles.append(kshell_handle)
        ax_leg_ion.legend(ion_handles, ion_labels, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=max(1, len(ion_labels)), frameon=False, columnspacing=0.9, handlelength=2.2, handletextpad=0.6, borderaxespad=0.0)

    for ax in (ax_a, ax_h):
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=50))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.xaxis.set_minor_locator(NullLocator())

    y_lo = min(ax_a.get_ylim()[0], ax_h.get_ylim()[0])
    y_hi = max(ax_a.get_ylim()[1], ax_h.get_ylim()[1])
    if not np.isfinite(y_lo) or y_lo <= 0.0:
        y_lo = 1e-25
    if not np.isfinite(y_hi) or y_hi <= y_lo:
        y_hi = y_lo * 10.0
    ax_a.set_ylim(y_lo, y_hi)
    ax_h.set_ylim(y_lo, y_hi)
    for ax in (ax_a, ax_h):
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=50))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(NullLocator())

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches='tight')
        print(f"Saved two-panel channel cross section plot to {out_path}")
    plt.close(fig)


def generate_all_plots(
    T_list,
    sigma_list,
    s,
    output_dir,
    ice_label,
    regime_flags_func,
    compute_correction_row_func,
    sigma_list_from_npz_func,
    density_scale_factor_from_npz_func,
    infer_ice_type_from_path_func,
    density_scale_factor_for_ice_func,
    regime_ii_max_eV,
):
    output_dir = Path(output_dir)

    plot_total_cross_section_corrections(T_list, sigma_list, compute_correction_row_func=compute_correction_row_func, ice_label=ice_label)

    style = rcparams_with_fontsize(
        RC_BASE_ELASTIC,
        FONTSIZE_24,
        overrides={
            'font.family': FONT_COURIER,
            'mathtext.rm': FONT_COURIER,
            'mathtext.fontset': 'custom',
        },
    )
    with plt.rc_context(style):
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        fig = plt.figure(figsize=(9, 7.8))
        gs = GridSpec(2, 1, height_ratios=[4.7, 1.3], hspace=0.30)
        ax = fig.add_subplot(gs[0, 0])
        plot_corrected_exc_ion_scaled(T_list, sigma_list, regime_flags_func=regime_flags_func, ax=ax)

        gs_leg = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], height_ratios=[1.0, 1.0], hspace=0.30)
        ax_leg_exc = fig.add_subplot(gs_leg[0, 0])
        ax_leg_ion = fig.add_subplot(gs_leg[1, 0])
        ax_leg_exc.axis('off')
        ax_leg_ion.axis('off')
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            n_exc = len(s.excitations)
            n_ion = len(s.ionizations)
            exc_handles = handles[:n_exc]
            ion_handles = handles[n_exc:n_exc + n_ion]
            if exc_handles:
                ax_leg_exc.legend(exc_handles, [str(i + 1) for i in range(n_exc)], loc='center left', bbox_to_anchor=(0.0, 0.5), ncol=max(1, n_exc), frameon=False, columnspacing=0.9, handlelength=2.2, handletextpad=0.6, borderaxespad=0.0)
            if ion_handles:
                ax_leg_ion.legend(ion_handles, [str(i + 1) for i in range(n_ion)], loc='center left', bbox_to_anchor=(0.0, 0.5), ncol=max(1, n_ion), frameon=False, columnspacing=0.9, handlelength=2.2, handletextpad=0.6, borderaxespad=0.0)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f'corrected_excitation_ionization_scaled_{ice_label}.png'
        fig.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.06)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved corrected excitation/ionization plot to {out_path}")

    ax_ion, ax_exc = plot_full_cross_sections_per_channel(T_list, sigma_list, s, regime_flags_func=regime_flags_func)
    ax_ion.figure.tight_layout()
    ax_ion.figure.savefig(f'comparison_ionizations_pwba_vs_model_{ice_label}.png', dpi=300)
    plt.close(ax_ion.figure)
    ax_exc.figure.tight_layout()
    ax_exc.figure.savefig(f'comparison_excitations_pwba_vs_model_{ice_label}.png', dpi=300)
    plt.close(ax_exc.figure)
    print(f"Saved comparison plots to comparison_ionizations_pwba_vs_model_{ice_label}.png and comparison_excitations_pwba_vs_model_{ice_label}.png")

    fig_ion, ax_ion = plt.subplots(figsize=(14, 9))
    fig_exc, ax_exc = plt.subplots(figsize=(14, 9))
    plot_relativistic_component_per_channel(T_list, sigma_list, s, regime_ii_max_eV=regime_ii_max_eV, ax=(ax_ion, ax_exc))
    fig_ion.tight_layout()
    fig_exc.tight_layout()
    fig_ion.savefig(f'rel_cross_sections_ionizations_LT_{ice_label}.png', dpi=300)
    fig_exc.savefig(f'rel_cross_sections_excitations_LT_{ice_label}.png', dpi=300)
    plt.close(fig_ion)
    plt.close(fig_exc)
    print(f"Saved REL component plots to rel_cross_sections_ionizations_LT_{ice_label}.png and rel_cross_sections_excitations_LT_{ice_label}.png")

    fig, ax = plt.subplots()
    plot_total_cross_section(T_list, sigma_list, regime_flags_func=regime_flags_func, ax=ax)
    fig.tight_layout()
    fig.savefig(f'total_cross_section_all_corrections_{ice_label}.png', dpi=300)
    plt.close(fig)
    print(f"Saved total plot to total_cross_section_all_corrections_{ice_label}.png")

    amorphous_npz = output_dir / 'cross_section_corrections_amorphous_ice.npz'
    hexagonal_npz = output_dir / 'cross_section_corrections_hexagonal_ice.npz'
    if amorphous_npz.exists() and hexagonal_npz.exists():
        plot_channel_cross_sections_two_panel(
            amorphous_npz,
            hexagonal_npz,
            sigma_list_from_npz_func=sigma_list_from_npz_func,
            density_scale_factor_from_npz_func=density_scale_factor_from_npz_func,
            density_scale_factor_for_ice_func=density_scale_factor_for_ice_func,
            regime_flags_func=regime_flags_func,
            out_path=output_dir / 'channel_cross_section_two_panel_amorphous_hexagonal.png',
        )
        plot_total_cross_section_two_panel(
            amorphous_npz,
            hexagonal_npz,
            density_scale_factor_from_npz_func=density_scale_factor_from_npz_func,
            infer_ice_type_from_path_func=infer_ice_type_from_path_func,
            density_scale_factor_for_ice_func=density_scale_factor_for_ice_func,
            out_path=output_dir / 'total_cross_section_two_panel_amorphous_hexagonal.png',
        )
    else:
        missing = []
        if not amorphous_npz.exists():
            missing.append(amorphous_npz.name)
        if not hexagonal_npz.exists():
            missing.append(hexagonal_npz.name)
        if missing:
            print(f"Skipping two-panel comparison plots (missing {', '.join(missing)}).")
