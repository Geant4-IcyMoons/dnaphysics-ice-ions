#!/usr/bin/env python3
"""Full-strength encounters compared with their weak-coupling expansion.

Use the production nonlinear solver and frozen fields. In atomic units,
x=omega*b/v, eta=1/(b*v^2), and the physical energy is v^2*eta^2*H(eta)
Hartree. The cubic curve is a benchmark-only derivative of that solver,
not an available production backend. See ../NONLINEAR_POLARIZATION.md.

The Salvat lower impact cutoff is retained in its nonrelativistic limit:
b_min=0.5616*C_B/v bohr. No integration below that cutoff, quantum
close-collision treatment, target OOS weighting, or DCS export is attempted.
Reversing the entire potential isolates odd/even energy terms; it is not
a change of the physical projectile charge state.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy

from physics import constants
from physics.inelastic_dielectric.checkpoints import atomic_text
from physics.inelastic_dielectric.polarization import correction as bd
from physics.inelastic_dielectric.polarization import nonlinear_oscillator as reference
from physics.inelastic_dielectric.projectile_potentials import projectile_form_factors

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "plots/full_strength_oscillator"
DEFAULT_STATES = ("H:1", "H:0", "He:1", "He:0")
HARTREE_EV = bd.ALPHA_FINE**2 * bd.MEC2_EV
CHECK_RTOL = reference.BENCHMARK_RTOL


def cubic_reference(x, b, field, *, tail=128., rtol=2e-12):
    """Weak-coupling derivative for this comparison only, never generation.

    Extract C=lim_eta->0 [H(eta)-H(-eta)]/(2*eta). Two Richardson
    estimates remove the leading eta^2 error. No old cubic kernel is used.
    """
    strengths=np.array([.04,.02,.01])/field.z
    odd=[]
    for eta in strengths:
        positive=reference.nonlinear_energy(x,b,eta,field,tail=tail,rtol=rtol)
        negative=reference.nonlinear_energy(x,b,-eta,field,tail=tail,rtol=rtol)
        odd.append((positive-negative)/(2*eta))
    coarse=(4*odd[1]-odd[0])/3
    fine=(4*odd[2]-odd[1])/3
    return float(fine),float(abs(fine-coarse))


def encounter_parameters(velocity, oscillator_eV, impact_ratio):
    """Nonrelativistic kinematics only, restricted to 1 < v <= 10 a.u."""
    if (not np.isfinite(velocity) or not 1 < velocity <= 10
            or not np.isfinite(oscillator_eV) or oscillator_eV <= 0
            or not np.isfinite(impact_ratio) or impact_ratio < 1):
        raise ValueError("Require 1 < v <= 10 a.u., positive oscillator energy, b/b_min >= 1")
    bmin = .5616 * bd.H2O_CB / velocity
    b = impact_ratio * bmin
    x = oscillator_eV / HARTREE_EV * b / velocity
    return b, x, 1 / (b * velocity**2)


def energy_components(velocity, eta, linear, cubic, positive, negative):
    """Return energies in eV, including signed odd and higher-even parts."""
    values = np.array([velocity, eta, linear, cubic, positive, negative])
    if (not np.all(np.isfinite(values)) or velocity <= 0 or eta <= 0
            or min(linear, positive, negative) < 0):
        raise ValueError("Invalid oscillator energy components")
    scale = HARTREE_EV * velocity**2 * eta**2
    leading = scale * linear
    correction = scale * eta * cubic
    full, reverse = scale * positive, scale * negative
    odd, even = (full - reverse) / 2, (full + reverse) / 2
    return dict(leading_eV=leading, cubic_eV=correction,
                leading_plus_cubic_eV=leading + correction,
                full_eV=full, reversed_potential_eV=reverse,
                full_odd_eV=odd, full_even_eV=even,
                higher_odd_eV=odd - correction, higher_even_eV=even - leading)


def relative_change(a, b):
    return abs(a-b) / max(abs(a), abs(b), np.finfo(float).tiny)


def evaluate_case(task):
    state, velocity, oscillator_eV, impact_ratio = task
    element, charge = state.split(":")
    charge = int(charge)
    b, x, eta = encounter_parameters(velocity, oscillator_eV, impact_ratio)
    field = reference.frozen_field(element, charge)
    refined_field = reference.frozen_field(element, charge, 16385)

    def solve_pair(selected_field, tail, rtol):
        return np.array([reference.nonlinear_energy(x, b, sign*eta, selected_field,
                         tail=tail, rtol=rtol) for sign in (1, -1)])

    nominal = solve_pair(field, 64., 2e-11)
    tight = solve_pair(field, 64., 2e-12)
    radial = solve_pair(refined_field, 64., 2e-12)
    pair_change = lambda a, c: max(relative_change(p, q) for p, q in zip(a, c))
    checks = dict(ode_tolerance=pair_change(nominal, tight),
                  radial_quadrature=pair_change(tight, radial))
    tail_history = []
    previous = radial
    for tail in (128., 256., 512.):
        full = solve_pair(refined_field, tail, 2e-12)
        window_difference = np.abs(full-previous)
        checks["integration_window"] = pair_change(full, previous)
        tail_history.append(dict(tail=tail, positive_H=float(full[0]),
                                 negative_H=float(full[1]), change=checks["integration_window"]))
        odd_uncertainty = np.sum(np.abs(tight-nominal)+np.abs(radial-tight)+window_difference)/2
        odd_value = abs(full[0]-full[1])/2
        if (checks["integration_window"] < CHECK_RTOL / 4
                and odd_uncertainty < CHECK_RTOL*odd_value):
            break
        previous = full

    linear = reference.linear_energy_quadrature(x, b, refined_field)
    linear_ode = reference.nonlinear_energy(x, b, 0., refined_field, tail=tail, rtol=2e-12)
    checks["linear_fourier_vs_ode"] = relative_change(linear, linear_ode)
    c0,_ = cubic_reference(x,b,refined_field,tail=64.)
    cubic,cubic_error = cubic_reference(x,b,refined_field,tail=128.)
    checks["reference_extrapolation"] = cubic_error/max(abs(cubic),np.finfo(float).tiny)
    checks["reference_window"] = relative_change(c0,cubic)
    energy = energy_components(velocity, eta, linear, cubic, *full)
    if energy["leading_eV"] <= 0 or energy["full_eV"] <= 0:
        raise RuntimeError("Nonpositive leading/full energy; relative comparison unresolved")
    # Full and sign-reversed energies can nearly cancel in their odd part.
    # Retain absolute uncertainty estimates rather than declaring the odd
    # residual converged just because the two total energies converged.
    scale = HARTREE_EV * velocity**2 * eta**2
    pair_error = np.abs(tight-nominal) + np.abs(radial-tight) + window_difference
    odd_error = float(scale * np.sum(pair_error) / 2)
    mass = next(v["mass_au"] for v in constants.PROJECTILE_LIBRARY.values()
                if v["element"] == element)
    return dict(state=state, velocity_au=velocity, oscillator_eV=oscillator_eV,
                impact_ratio=impact_ratio, b_bohr=b, x=x, eta=eta,
                T_total_MeV_NR=.5*mass*velocity**2*HARTREE_EV/1e6,
                beta=velocity*bd.ALPHA_FINE, **energy,
                cubic_over_leading=energy["cubic_eV"]/energy["leading_eV"],
                truncation_error_over_full=(energy["leading_plus_cubic_eV"]-energy["full_eV"])/energy["full_eV"],
                odd_absolute_error_estimate_eV=odd_error,
                full_odd_resolved=bool(odd_error < CHECK_RTOL*abs(energy["full_odd_eV"])),
                numerical_status="PASS" if max(checks.values()) < CHECK_RTOL else "FAIL",
                numerical_checks=checks, tail_history=tail_history)


def checked_case(task):
    try:
        return evaluate_case(task)
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        state, velocity, energy, ratio = task
        return dict(state=state, velocity_au=velocity, oscillator_eV=energy,
                    impact_ratio=ratio, numerical_status="FAIL", error=str(exc))


def summarize(rows):
    usable = [r for r in rows if r["numerical_status"] == "PASS"]
    return dict(case_count=len(rows), converged_count=len(usable),
                failed_count=len(rows)-len(usable),
                odd_unresolved_count=sum(not r["full_odd_resolved"] for r in usable),
                cubic_at_least_leading_count=int(sum(abs(r["cubic_over_leading"]) >= 1 for r in usable)),
                truncation_error_above_10_percent_count=int(sum(abs(r["truncation_error_over_full"]) > .1 for r in usable)),
                maximum_absolute_truncation_error_over_full=max(
                    (abs(r["truncation_error_over_full"]) for r in usable), default=None))


def plot_results(report, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    font = None
    for candidate in (constants.FONT_COURIER, "Courier New", "Courier"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            font = candidate
            break
        except ValueError:
            continue
    if font is None:
        raise RuntimeError("Install a Courier-compatible font for this figure")
    styles = constants.rcparams_with_fontsize(constants.RC_BASE_ELASTIC, constants.PAPER_FONTSIZE, {
        "font.family":font, "figure.titlesize":constants.PAPER_FONTSIZE,
        "axes.grid":False, "pdf.fonttype":42,
        "mathtext.fontset":"custom", "mathtext.rm":font, "mathtext.it":font,
        "mathtext.bf":font, "mathtext.fallback":None, "savefig.bbox":None,
        "axes.labelpad":4, "axes.titlepad":4, "axes.linewidth":.8,
        "xtick.major.size":3, "ytick.major.size":3,
        "xtick.minor.size":1.7, "ytick.minor.size":1.7})
    path = output / "full_strength_oscillator.pdf"
    with plt.rc_context(styles), PdfPages(path) as pdf:
        colors = plt.get_cmap("plasma")(np.linspace(0, 1, 4))[:-1]
        for velocity in report["velocities_au"]:
            for omega in report["oscillator_energies_eV"]:
                fig, axes = plt.subplots(2, 2, figsize=(constants.AASTEX_FULL_WIDTH_IN, 4.8), sharex=True)
                for ax, state, letter in zip(axes.flat, report["states"], "abcd"):
                    rows = [r for r in report["rows"] if r["state"] == state
                            and r["velocity_au"] == velocity and r["oscillator_eV"] == omega]
                    ratios = report["impact_ratios"]
                    lookup = {r["impact_ratio"]:r for r in rows}
                    for key, color, ls in zip(("leading_eV", "leading_plus_cubic_eV", "full_eV"), colors, (":", "--", "-")):
                        values = [lookup[b].get(key, np.nan)/lookup[b].get("leading_eV", np.nan)
                                  if lookup[b]["numerical_status"] == "PASS" else np.nan for b in ratios]
                        ax.plot(ratios, values, color=color, ls=ls, lw=1.2)
                    ax.set_xscale("log", base=2)
                    ax.set_xticks([1, 2, 4, 8, 16])
                    ax.set_xticklabels(["1", "2", "4", "8", "16"])
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                    element, charge = state.split(":")
                    label = element+"$^{"+(charge+"+" if int(charge) else "0")+"}$"
                    ax.set_title(f"({letter}) {label}", loc="left")
                    failures = sum(r["numerical_status"] != "PASS" for r in rows)
                    if failures:
                        ax.text(.98, .97, f"Unresolved: {failures}", transform=ax.transAxes, ha="right", va="top")
                    ax.tick_params(which="both", direction="in", top=True, right=True)
                    ax.grid(False, which="both")
                for ax in axes[0]:
                    ax.tick_params(labelbottom=False)
                for ax in axes[1]:
                    ax.set_xlabel("Impact parameter ($b/b_{min}$)")
                fig.text(.028, .55, "Transferred energy / leading energy", rotation=90, ha="center", va="center")
                fig.suptitle(f"$v={velocity:g}$ a.u.; $\\hbar\\omega={omega:g}$ eV", y=.985)
                handles = [Line2D([], [], color=c, ls=s, lw=1.2, label=l)
                           for c, s, l in zip(colors, (":", "--", "-"),
                                              ("Leading", "Leading + cubic", "Full nonlinear"))]
                fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(.53, .02), frameon=False)
                fig.subplots_adjust(left=.10, right=.985, bottom=.17, top=.89, hspace=.20, wspace=.25)
                pdf.savefig(fig)
                plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocities-au", nargs="+", type=float, default=[3., 5., 10.])
    parser.add_argument("--oscillator-energies-eV", nargs="+", type=float, default=[15., 50.])
    parser.add_argument("--impact-nodes", type=int, default=9)
    parser.add_argument("--workers", type=int, default=constants.DEFAULT_WORKERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)
    out = args.output_dir
    report_path = out / "comparison.json"
    if args.plot_only:
        print(plot_results(json.loads(report_path.read_text()), out))
        return 0
    if args.impact_nodes < 3 or args.workers < 1:
        parser.error("Require >= 3 impact nodes and >= 1 worker")
    speeds = sorted(set(args.velocities_au))
    energies = sorted(set(args.oscillator_energies_eV))
    try:
        for v in speeds:
            for w in energies:
                encounter_parameters(v, w, 1.)
    except ValueError as exc:
        parser.error(str(exc))
    ratios = np.geomspace(1., 16., args.impact_nodes).tolist()
    tasks = [(state, v, w, b) for state in DEFAULT_STATES for v in speeds for w in energies for b in ratios]
    sources = [Path(__file__), Path(reference.__file__), Path(bd.__file__),
               Path(constants.__file__),
               Path(projectile_form_factors.__file__),projectile_form_factors.DATA_PATH]
    repo = Path(__file__).resolve().parents[4]
    report = dict(numerical_status="RUNNING", created_utc=datetime.now(timezone.utc).isoformat(),
                  comparison="Same classical oscillator: leading, leading+cubic, full physical strength",
                  states=list(DEFAULT_STATES), velocities_au=speeds, oscillator_energies_eV=energies,
                  impact_ratios=ratios, workers=min(args.workers, len(tasks)), rows=[],
                  relative_tolerance=CHECK_RTOL, ode_rtols=[2e-11, 2e-12],
                  time_windows=[64., 128., 256., 512.], radial_nodes=[8193, 16385],
                  energy_units="eV per encounter; hbar*omega is not the actual transferred energy",
                  kinematics="Nonrelativistic: total T=M*v^2/2; mass in electron masses",
                  impact_cutoff="b_min=0.5616*C_B/v bohr; C_B=1; b sampled from b_min to 16*b_min",
                  finite_impact_sample_not_integrated_stopping=True,
                  oscillator_initial_state="At rest; harmonic binding; straight frozen projectile",
                  full_minus_leading_includes_higher_even_and_odd=True,
                  reversed_potential_is_physical_charge_state=False,
                  cubic_reference="Benchmark-only weak-coupling derivative of full solver; no cubic production backend",
                  production_generator_changed=False, ice_DCS_validation=False,
                  target_OOS_weighting=False, screened_relativity_validated=False,
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                  source_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  software=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    out.mkdir(parents=True, exist_ok=True)

    def save():
        with atomic_text(report_path) as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")

    save()
    with ProcessPoolExecutor(max_workers=report["workers"], mp_context=get_context("spawn")) as pool:
        for row in pool.map(checked_case, tasks):
            report["rows"].append(row)
            save()
            if len(report["rows"]) % args.impact_nodes == 0 or row["numerical_status"] != "PASS":
                print(f"{len(report['rows'])}/{len(tasks)} {row['state']} v={row['velocity_au']:g} "
                      f"omega={row['oscillator_eV']:g}: {row['numerical_status']}"
                      + (" "+row["error"] if "error" in row else ""), flush=True)
    report["summary"] = summarize(report["rows"])
    report["numerical_status"] = "PASS" if report["summary"]["failed_count"] == 0 else "FAIL"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(plot_results(report, out))
    print(json.dumps(report["summary"]))
    return 0 if report["numerical_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
