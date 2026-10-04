"""Audit optical target inputs without changing production normalization or DCS."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.integrate import cumulative_trapezoid

from physics.constants import (AASTEX_FULL_WIDTH_IN, AVOGADRO, FONT_COURIER,
                               H2O_MOLAR_MASS_G_MOL, PAPER_FONTSIZE)
from physics.inelastic_dielectric import generate_cross_sections as generator
from physics.inelastic_dielectric.finite_q import emfietzoglou_model_finite_q as model
from physics.inelastic_dielectric.finite_q import optical_input
from physics.inelastic_dielectric.polarization import correction

HERE = Path(__file__).resolve().parent
PHASES = ("amorphous", "hexagonal")
UNIT = generator.OPTICAL_SUM_UNIT_EV2_M3


def grid(points):
    edges = np.array([7., 10., 13., 17., 21., 32., 100., model.OXYGEN_K_B_EV,
                      model.OXYGEN_K_ZEFF**2*model.RYD_ELECTRON_VOLT, 32000.])
    return np.unique(np.concatenate((np.geomspace(1., 1e9, points), edges,
                                      np.nextafter(edges, 0.), np.nextafter(edges, np.inf))))


def moments(w, density):
    """df/dW in eV^-1 per molecule; I = exp(integral ln(W/eV) df / integral df)."""
    if (np.any(~np.isfinite(density)) or np.any(density < 0)
            or np.any(np.diff(w) < 0) or np.any(w <= 0)):
        raise ValueError("Invalid OOS or energy grid")
    f = cumulative_trapezoid(density, w, initial=0.)
    logf = cumulative_trapezoid(density*np.log(w), w, initial=0.)
    if f[-1] <= 0:
        raise ValueError("Empty optical strength")
    result = {"electrons": float(f[-1]), "I_eV": float(np.exp(logf[-1]/f[-1]))}
    for edge in (21, 100, 32000):
        value = float(np.interp(edge, w, f))
        result[f"electrons_below_{edge}_eV"] = value
    result["electrons_100_to_32000_eV"] = result["electrons_below_32000_eV"]-result["electrons_below_100_eV"]
    edge = 32000
    result["I_to_32000_eV"] = float(np.exp(np.interp(edge, w, logf)/np.interp(edge, w, f)))
    return result, f


def inputs(phase, w):
    s = model.epsilon_optical(phase)
    meta = generator._ion_normalization_metadata(phase, s)
    n = meta["material_molecular_density_m3"]
    q = np.array([0.])
    c = model.DispersionCoeffs(0., 0., 0.)
    strength = model.oxygen_K_hydrogenic_gos_continuum_strength(q)
    e1 = model.epsilon1_valence_Eq(w, q, s, c, inner_shell_strength_electrons=strength)["total"][0]
    e2 = model.epsilon2_valence_Eq(w, q, s, c, partitioned=True,
                                 inner_shell_strength_electrons=strength)["total"][0]
    raw_val = e2 / (e1*e1 + e2*e2)
    raw_core = model.oxygen_K_ion_hydrogenic_gos_elf(w, 0., Ep_eV=s.Ep)
    factor = meta["optical_elf_fsum_scale"] / (n*UNIT)
    born_val, born_core = factor*w*raw_val, factor*w*raw_core
    osc = correction.oos_density(w, s, phase, include_kshell=True)
    values = {
        "Born": (born_val*generator._elf_rolloff_factor(w), born_core),
        "Oscillator": (osc.df_dW_valence, osc.df_dW_OK),
    }
    unrolled, _ = moments(w, born_val+born_core)
    meta["before_valence_rolloff"] = unrolled
    meta["raw_unscaled_elf_electrons_at_material_density"] = unrolled["electrons"]/meta["optical_elf_fsum_scale"]
    meta["oscillator_native_normalization"] = {
        "valence_scale": osc.valence_norm, "core_scale": osc.ok_norm,
        "native_grid_total": osc.total_integral_norm_grid}
    return values, meta


def reference(phase):
    """Finite-domain figure estimate; do not complete missing strength by rescaling."""
    parts = [np.loadtxt(HERE / f"references/matias2025_{phase}_{part}.csv", delimiter=",")
             for part in ("optical", "tail")]
    xy = np.vstack(parts)
    xy = xy[np.argsort(xy[:, 0], kind="stable")]
    # Keep vertical edges (duplicate energies); their integration width is zero.
    n = .94 * 1e6 * AVOGADRO / H2O_MOLAR_MASS_G_MOL
    return xy[:, 0], xy[:, 1]*xy[:, 0]/(n*UNIT)


def style():
    for font in (FONT_COURIER, "Courier New", "Courier"):
        try:
            font_manager.findfont(font, fallback_to_default=False)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("Courier font required")
    plt.rcParams.update({"font.family": font, "font.size": PAPER_FONTSIZE,
        "axes.labelsize": PAPER_FONTSIZE, "axes.titlesize": PAPER_FONTSIZE,
        "xtick.labelsize": PAPER_FONTSIZE, "ytick.labelsize": PAPER_FONTSIZE,
        "legend.fontsize": PAPER_FONTSIZE, "mathtext.fontset": "custom",
        "mathtext.rm": font, "mathtext.it": font, "mathtext.bf": font,
        "mathtext.fallback": None, "axes.grid": False, "pdf.fonttype": 42,
        "axes.linewidth": .8, "axes.titlepad": 6, "axes.labelpad": 4,
        "lines.linewidth": 1.1, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5})


def main():
    style()
    output = HERE / "plots"
    output.mkdir(exist_ok=True)
    fine, coarse = grid(240001), grid(120001)
    report = {"units": "df/dW: eV^-1 per H2O; ELF: dimensionless; I: eV",
        "definition": "df/dW = W ELF/(N_H2O * pi*hbar^2*e^2/(2*m_e*epsilon_0)); I=exp(integral ln(W/eV) df/integral df)",
        "energy_max_eV": 1e9, "grid_points": [len(coarse), len(fine)],
        "Matias_window_check": "Approximately 7 below 100 eV and 3 from 100 to 32000 eV; NOT a valence/core partition.",
        "reference_limits": "Digitized fitted ELF, not raw measurements. I and f-sum for Matias cover only the figure domain, not 0 to infinity. No reference renormalization.",
        "phases": {}}
    colors = plt.get_cmap("plasma")(np.linspace(0, 1, 3))[:2]
    fig, axes = plt.subplots(3, 2, figsize=(AASTEX_FULL_WIDTH_IN, 7.2), sharey="row")
    fig.subplots_adjust(left=.105, right=.98, bottom=.105, top=.95, wspace=.12, hspace=.36)
    for col, phase in enumerate(PHASES):
        values, meta = inputs(phase, fine)
        coarse_values, _ = inputs(phase, coarse)
        phase_report = {"production_normalization": meta, "inputs": {}}
        n = meta["material_molecular_density_m3"]
        for index, (name, (val, core)) in enumerate(values.items()):
            total = val+core
            summary, cumulative = moments(fine, total)
            summary["valence"], _ = moments(fine, val)
            summary["core"], _ = moments(fine, core)
            summary["deviation_from_ten_electrons"] = summary["electrons"]-10
            shorter, _ = moments(fine[fine <= 1e8], total[fine <= 1e8])
            summary["upper_limit_1e8_to_1e9_relative_change"] = {
                key: abs(summary[key]/shorter[key]-1) for key in ("electrons", "I_eV")}
            if max(summary["upper_limit_1e8_to_1e9_relative_change"].values()) > 1e-5:
                raise RuntimeError(f"Optical upper limit not converged: {phase} {name}")
            low, _ = moments(coarse, sum(coarse_values[name]))
            errors = {key: abs(summary[key]/low[key]-1) for key in ("electrons", "I_eV", "electrons_below_100_eV", "electrons_100_to_32000_eV")}
            summary["grid_refinement_relative_change"] = errors
            summary["numerically_converged_1e-4"] = bool(max(errors.values()) < 1e-4)
            if not summary["numerically_converged_1e-4"]:
                raise RuntimeError(f"Optical integration not converged: {phase} {name}: {errors}")
            phase_report["inputs"][name] = summary
            equivalent_elf = total*n*UNIT/fine
            for row in (0, 1):
                axes[row, col].plot(fine, equivalent_elf, color=colors[index],
                                    linestyle=("-", "--")[index], label=name)
            axes[2, col].plot(fine, cumulative, color=colors[index],
                              linestyle=("-", "--")[index], label=name)
        rw, rd = reference(phase)
        refsummary, rcum = moments(rw, rd)
        refsummary["integration_domain_eV"] = [float(rw[0]), float(rw[-1])]
        refsummary["I_is_figure_domain_only"] = True
        phase_report["Matias_figure_estimate"] = refsummary
        report["phases"][phase] = phase_report
        for row in (0, 1):
            axes[row, col].plot(rw, rd*n*UNIT/rw, color="black", lw=1., label="Matias+25")
        axes[2, col].plot(rw, rcum, color="black", lw=1., label="Matias+25")
        axes[0, col].set(xscale="log", xlim=(6, 100), ylim=(0, .88))
        axes[1, col].set(xscale="log", yscale="log", xlim=(100, 32000), ylim=(1e-10, .03))
        axes[2, col].set(xscale="log", xlim=(6, 32000), ylim=(0, 10.6))
        axes[2, col].plot([100, 32000], [7, 10], linestyle="none", marker="o",
                          color="black", markerfacecolor="black", markeredgecolor="black",
                          markersize=3, label="Reported sums")
        for row in range(3):
            ax = axes[row, col]
            ax.set_title(f"({chr(97+2*row+col)}) " + (phase.capitalize()+" ice" if row == 0 else ("High-energy tail" if row == 1 else "Cumulative strength")), loc="left")
            ax.tick_params(which="both", direction="in", top=False, right=False)
        vals = phase_report["inputs"]
        axes[2, col].text(.97, .07,
            f"I (Born): {vals['Born']['I_eV']:.1f} eV\n"
            f"I (osc.): {vals['Oscillator']['I_eV']:.1f} eV\n"
            f"I (Matias, figure): {refsummary['I_eV']:.0f} eV",
            transform=axes[2, col].transAxes, ha="right", va="bottom")
    axes[0, 0].set_ylabel("Optical ELF")
    axes[1, 0].set_ylabel("Optical ELF")
    axes[2, 0].set_ylabel("Electrons per H$_2$O")
    fig.supxlabel("Transferred energy (eV)", y=.065, fontsize=PAPER_FONTSIZE)
    handles, labels = axes[2, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .005))
    fig.savefig(output / "optical_elf_audit.pdf")
    plt.close(fig)
    files = [Path(generator.__file__), Path(model.__file__), Path(correction.__file__),
             Path(optical_input.__file__), Path(__file__),
             *sorted((HERE / "references").glob("*"))]
    report["source_sha256"] = {str(path.relative_to(HERE.parents[3])): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    lines = ["# Optical target audit", "", "All strengths are electrons per H2O. Born and oscillator now share the partitioned optical input, including the valence rolloff; the K continuum has no rolloff. No fit to stopping data is used.", "",
             "| Phase | Input | <100 eV | 100-32000 eV | Total | Valence | Core | I (eV) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for phase, results in report["phases"].items():
        for name, v in results["inputs"].items():
            lines.append(f"| {phase} | {name} | {v['electrons_below_100_eV']:.4f} | {v['electrons_100_to_32000_eV']:.4f} | {v['electrons']:.4f} | {v['valence']['electrons']:.4f} | {v['core']['electrons']:.4f} | {v['I_eV']:.2f} |")
        v = results["Matias_figure_estimate"]
        lines.append(f"| {phase} | Matias figure domain only | {v['electrons_below_100_eV']:.3f} | {v['electrons_100_to_32000_eV']:.3f} | {v['electrons']:.3f} | -- | -- | {v['I_eV']:.1f}* |")
    lines += ["", "*Matias I is a finite-domain digitization estimate, not an author-reported Bethe I. Missing energies and figure precision limit this comparison. Reference spectra were not forced to ten electrons. The reported 7/3 energy-window split is not a valence/core partition.", "", "The oscillator curve is its OOS converted to an equivalent ELF at the material density; this does not establish a causal dielectric function. Matias ELF amplitudes use the same per-molecule conversion (source density 0.94 g/cm3).", "", "Numerical convergence is separate from target validation. Matching optical moments cannot validate finite-q extrapolation, channel partition, or the oscillator-to-DCS mapping."]
    (output / "summary.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
