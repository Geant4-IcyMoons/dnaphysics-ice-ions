"""Plot the generated Born baseline and additive polarization, without refitting.

Reads either the complete generation cache or DIAGNOSTIC_ONLY.npz. All
moments use the saved loss grid and a piecewise-linear DCS. Maps aggregate
only for display: the signed largest-magnitude ratio and union of warning
bits are retained in each log-energy bin. No interpolation fills failures.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap, SymLogNorm
import numpy as np

from physics.constants import (AASTEX_FULL_WIDTH_IN, PAPER_FONTSIZE, FONT_COURIER,
    RC_BASE_ELASTIC, rcparams_with_fontsize, PROJECTILE_LIBRARY, EMFI_DCS_SCALE_M2)


def load_components(path):
    """Load actual components; never infer Born from a corrected total."""
    with np.load(path, allow_pickle=False) as saved:
        if "rejection_flags" in saved:
            metadata = json.loads(str(saved["metadata"]))
            t, w = saved["T_eV"], saved["W_eV"]
            born, correction, total = (saved[key] for key in
                ("born_m2_eV", "polarization_m2_eV", "total_m2_eV"))
            flags = saved["rejection_flags"].astype(np.uint8)
        else:
            metadata = {key: saved[key].item() for key in (
                "projectile_key", "projectile_element", "projectile_charge_state",
                "projectile_charge", "charge_mode", "ice_type", "include_barkas_dcs",
                "projectile_relativistic_dcs", "include_kshell") if key in saved}
            t, w = saved["dcs_T_line"], saved["dcs_E_line"]
            if "DCS_Born_m2_per_eV" in saved:
                born, correction, total = (saved[key] for key in
                    ("DCS_Born_m2_per_eV", "DCS_Barkas_m2_per_eV", "DCS_total_m2_per_eV"))
            elif not metadata.get("include_barkas_dcs", False):
                born = EMFI_DCS_SCALE_M2 * (saved["dcs_exc_vals"].sum(axis=1)
                                + saved["dcs_ion_vals"].sum(axis=1))
                correction, total = np.zeros_like(born), born.copy()
            else:
                raise ValueError("Corrected cache lacks separate Born/polarization arrays; regenerate it.")
            flags = np.zeros(t.shape, np.uint8)
    arrays = [np.asarray(a, float) for a in (t, w, born, correction, total)]
    t, w, born, correction, total = arrays
    if not t.size or any(a.ndim != 1 or a.shape != t.shape for a in arrays + [flags]):
        raise ValueError("Empty or inconsistent DCS diagnostic arrays.")
    if np.any(~np.isfinite(t+w)) or np.any(t <= 0) or np.any(w <= 0):
        raise ValueError("Incident and loss energies must be finite and positive, in eV.")
    finite = np.isfinite(born) & np.isfinite(correction) & np.isfinite(total)
    tolerance = 5e-13 * np.maximum(np.abs(born) + np.abs(correction), np.finfo(float).tiny)
    if np.any(finite & (np.abs(total-born-correction) > tolerance)):
        raise ValueError("Saved DCS_total is not Born + polarization.")
    flags[~finite] |= 2
    flags[(born > 0) & (np.abs(correction) >= born)] |= 8
    flags[total < 0] |= 16
    # A zero baseline has no relative correction. Never plot it as zero.
    flags[(born <= 0) & (correction != 0)] |= 32
    order = np.lexsort((w, t))
    return dict(metadata=metadata, T=t[order], W=w[order], born=born[order],
                correction=correction[order], total=total[order], flags=flags[order])


def moments(data):
    """Use the generator's trapezoidal TCS and sampled W*DCS diagnostics.

    An undefined node makes its whole moment undefined; never integrate only
    the remaining nodes across the gap. These are loss-grid quadratures, not
    a new interpolation prescription or extrapolation outside the saved grid.
    """
    energy, starts = np.unique(data["T"], return_index=True)
    result = {key: np.full((2, energy.size), np.nan)
              for key in ("born", "correction", "total")}
    result["T"] = energy
    for i, idx in enumerate(np.split(np.arange(data["T"].size), starts[1:])):
        w = data["W"][idx]
        if w.size < 2:
            continue
        dw = np.diff(w)
        if np.any(dw <= 0):
            raise ValueError("Each incident-energy block needs unique ordered loss nodes.")
        for key in ("born", "correction", "total"):
            v = data[key][idx]
            if np.all(np.isfinite(v)):
                result[key][0, i] = np.sum(dw * (v[:-1]+v[1:]) / 2)
                result[key][1, i] = np.trapezoid(w*v, w)
    return result


def ratio_map(data, max_t_bins=300, max_w_bins=240):
    """Conservative display binning; even one failed node marks its bin."""
    def edges(values, limit):
        unique = np.unique(values)
        if unique.size == 1:
            return unique[0]*np.array([.95, 1.05])
        if unique.size <= limit:
            log = np.log(unique)
            return np.exp(np.r_[log[0]-(log[1]-log[0])/2,
                                (log[:-1]+log[1:])/2, log[-1]+(log[-1]-log[-2])/2])
        return np.geomspace(unique[0]*(1-1e-12), unique[-1]*(1+1e-12), limit+1)
    te, we = edges(data["T"], max_t_bins), edges(data["W"], max_w_bins)
    ti = np.clip(np.searchsorted(te, data["T"], side="right")-1, 0, te.size-2)
    wi = np.clip(np.searchsorted(we, data["W"], side="right")-1, 0, we.size-2)
    shape = (we.size-1, te.size-1)
    index = np.ravel_multi_index((wi, ti), shape)
    ratio = np.divide(data["correction"], data["born"],
                      out=np.full(data["T"].shape, np.nan), where=data["born"] > 0)
    finite = np.isfinite(ratio)
    positive, negative = np.zeros(np.prod(shape)), np.zeros(np.prod(shape))
    np.maximum.at(positive, index[finite], np.maximum(ratio[finite], 0))
    np.maximum.at(negative, index[finite], np.maximum(-ratio[finite], 0))
    count, flag_union = np.zeros(np.prod(shape), int), np.zeros(np.prod(shape), np.uint8)
    np.add.at(count, index[finite], 1)
    np.bitwise_or.at(flag_union, index, data["flags"])
    display = np.where(positive >= negative, positive, -negative)
    display[count == 0] = np.nan
    present = np.zeros(np.prod(shape), bool)
    present[index] = True
    status = np.zeros(np.prod(shape))
    status[(flag_union & 8) != 0] = 1
    status[(flag_union & 16) != 0] = 2
    status[(flag_union & (1 | 2 | 4 | 32)) != 0] = 3
    status[~present] = np.nan
    return te, we, display.reshape(shape), status.reshape(shape)


def _style():
    for font in (FONT_COURIER, "Courier New", "Courier"):
        try:
            font_manager.findfont(font, fallback_to_default=False)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("Install a Courier-compatible font for the correction figures.")
    return rcparams_with_fontsize(RC_BASE_ELASTIC, PAPER_FONTSIZE, {
        "font.family": font, "figure.titlesize": PAPER_FONTSIZE,
        "axes.grid": False, "pdf.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": font, "mathtext.it": font,
        "mathtext.bf": font, "mathtext.fallback": None, "axes.labelpad": 4,
        "axes.linewidth": .8, "axes.titlepad": 5, "xtick.major.size": 3,
        "ytick.major.size": 3, "xtick.minor.size": 1.7, "ytick.minor.size": 1.7,
        "xtick.major.width": .7, "ytick.major.width": .7,
        "xtick.minor.width": .5, "ytick.minor.width": .5})


def _title(metadata):
    config = PROJECTILE_LIBRARY.get(metadata.get("projectile_key", metadata.get("projectile")), {})
    element = metadata.get("projectile_element", config.get("element", "Projectile"))
    charge = metadata.get("projectile_charge_state", metadata.get("projectile_charge", "?"))
    charge_text = f"{float(charge):g}" if charge != "?" else charge
    state = element + "$^{" + charge_text + ("+" if charge_text not in ("0", "?") else "") + "}$"
    kernel = "RPWBA" if (metadata.get("projectile_relativistic_dcs", False)
        or metadata.get("projectile_kernel", "pwba") != "pwba") else "PWBA"
    mode = metadata.get("charge_mode", "bare")
    if mode != "bare":
        state += " (" + ("$z_{\\rm eff}$" if mode == "zeff" else "explicit charge") + ")"
    phase = metadata.get("ice_type", "unknown").capitalize()
    suffix = "Diagnostic only" if metadata.get("diagnostic_only", False) else "Generated tables"
    correction = "on" if metadata.get("include_barkas_dcs", True) else "off"
    core = "on" if metadata.get("include_kshell", True) else "off"
    return (f"{state} | {phase} ice | {kernel} | {suffix}\n"
            f"Polarization: {correction}; K shell: {core}; total kinetic energy per ion")


def _axis_style(ax):
    ax.tick_params(which="both", direction="in", top=False, right=False)
    ax.grid(False, which="both")


def _moment_figure(data):
    curves = moments(data)
    colors = plt.get_cmap("plasma")(np.linspace(0, 1, 4))[:-1]
    fig, axes = plt.subplots(2, 2, figsize=(AASTEX_FULL_WIDTH_IN, 4.8), sharex="col",
                            gridspec_kw={"height_ratios": [3, 1.2], "hspace": .10})
    for col in range(2):
        ax, relative = axes[:, col]
        for key, label, ls, color in zip(("born", "correction", "total"),
                ("Born", "Polarization", "Born + polarization"), ("--", ":", "-"), colors):
            ax.plot(curves["T"], curves[key][col], color=color, ls=ls, lw=1.2,
                    marker="o" if curves["T"].size == 1 else None, markersize=3, label=label)
        values = np.concatenate([curves[key][col] for key in ("born", "correction", "total")])
        nonzero = np.abs(values[np.isfinite(values) & (values != 0)])
        if nonzero.size:
            if np.all(values[np.isfinite(values)] > 0):
                ax.set_yscale("log")
            else:
                ax.set_yscale("symlog", linthresh=max(nonzero.min()*.1, np.finfo(float).tiny))
        ax.set_xscale("log")
        ax.set_title(("(a) Total cross section", "(b) Stopping cross section")[col], loc="left")
        ax.set_ylabel(("Cross section\n(m$^2$ / H$_2$O)",
                       "Stopping cross section\n(eV m$^2$ / H$_2$O)")[col])
        ratio = np.divide(100*curves["correction"][col], curves["born"][col],
            out=np.full(curves["T"].shape, np.nan), where=curves["born"][col] > 0)
        relative.plot(curves["T"], ratio, color=colors[1], lw=1.2,
                      marker="o" if curves["T"].size == 1 else None, markersize=3)
        finite_ratio = ratio[np.isfinite(ratio)]
        if finite_ratio.size and np.all(finite_ratio > 0):
            relative.set_yscale("log")
        else:
            relative.axhline(0, color=".5", lw=.7)
            if np.any(finite_ratio != 0):
                relative.set_yscale("symlog", linthresh=1.)
        relative.set_title(("(c)", "(d)")[col], loc="left", pad=2)
        relative.set_ylabel("Correction /\nBorn (%)")
        relative.set_xlabel("Kinetic energy ($T$; eV)")
        for panel in (ax, relative):
            _axis_style(panel)
    fig.suptitle(_title(data["metadata"]), y=.98)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.53, .025), borderpad=0, borderaxespad=0)
    fig.subplots_adjust(left=.115, right=.985, bottom=.19, top=.86, wspace=.36)
    return fig


def _map_figure(data):
    te, we, ratio, status = ratio_map(data)
    finite = ratio[np.isfinite(ratio)]
    lower = min(0., finite.min()) if finite.size else 0.
    upper = max(1., finite.max()) if finite.size else 1.
    fig = plt.figure(figsize=(AASTEX_FULL_WIDTH_IN, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, .025], height_ratios=[3, 1.4],
                          left=.10, right=.79, bottom=.12, top=.84, hspace=.22, wspace=.06)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    image = axes[0].pcolormesh(te, we, np.ma.masked_invalid(ratio), cmap="plasma",
        norm=SymLogNorm(linthresh=.01, vmin=lower, vmax=upper, base=10))
    colorbar = fig.colorbar(image, cax=fig.add_subplot(gs[0, 1]))
    colorbar.set_label("Polarization / Born")
    colors = plt.get_cmap("plasma")(np.linspace(0, 1, 4))[:-1]
    status_cmap = ListedColormap([".92", colors[2], colors[1], colors[0]])
    image = axes[1].pcolormesh(te, we, np.ma.masked_invalid(status), cmap=status_cmap,
        norm=BoundaryNorm(np.arange(-.5, 4.5), 4))
    colorbar = fig.colorbar(image, cax=fig.add_subplot(gs[1, 1]), ticks=range(4))
    colorbar.ax.set_yticklabels(["No flags", "|P| >= Born", "Negative total", "Unresolved"])
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Energy loss ($W$; eV)")
        ax.set_xlim(te[[0, -1]])
        ax.set_ylim(we[[0, -1]])
        _axis_style(ax)
    axes[0].tick_params(labelbottom=False)
    axes[0].set_title("(a) Largest-magnitude DCS ratio per display bin", loc="left")
    axes[1].set_title("(b) Warning map (worst flag per bin)", loc="left")
    axes[1].set_xlabel("Kinetic energy ($T$; eV)")
    fig.suptitle(_title(data["metadata"]), y=.98)
    return fig


def plot_correction(path, output=None):
    """Write a two-page PDF from saved values; never rerun a physics kernel."""
    path = Path(path)
    data = load_components(path)
    output = Path(output) if output is not None else path.parent / "plots" / (path.stem+"_polarization.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    try:
        with plt.rc_context(_style()), PdfPages(temporary) as pdf:
            for builder in (_moment_figure, _map_figure):
                fig = builder(data)
                try:
                    pdf.savefig(fig)
                finally:
                    plt.close(fig)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Polarization comparison: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot_correction(args.npz, args.output)


if __name__ == "__main__":
    main()
