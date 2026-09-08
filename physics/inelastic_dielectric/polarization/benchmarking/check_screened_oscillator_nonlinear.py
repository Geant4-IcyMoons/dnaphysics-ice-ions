#!/usr/bin/env python3
"""All-state nonlinear oscillator benchmark, not a physical ice-DCS validation.

Let tau=v*t/b, u=r/b, x=omega*b/v and eta=lambda/(b*v^2), in atomic
units. The exact classical oscillator equation for a frozen projectile is
u''+x^2*u = eta*g(b*|R-u|)*(R-u)/|R-u|^3, R=(1,tau).
Gauss-law fields are independently integrated from the same frozen densities.

Solve for y=u/eta to retain accuracy at small eta. The final scaled energy
H(eta)=(|y'|^2+x^2*|y|^2)/2 gives E=eta^2*H. Hence
[H(eta)-H(-eta)]/(2*eta) tends to the cubic energy coefficient, which
must equal Px*Dx+Pz*Dz from oscillator_quadrature.impulse_products at gamma=1.
No derivative of the projectile field enters this independent ODE solver.

The oscillator is the same approximation as in SCREENED_BARKAS.md. This
tests its perturbative implementation, not its spectral assignment to an
ice DCS, relativistic extension, or Schinner & Sigmund's published curves.
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
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
from scipy.integrate import cumulative_simpson, quad, solve_ivp
from scipy.interpolate import CubicSpline

PHYSICS_ROOT = Path(__file__).resolve().parents[3]
from physics.constants import (AASTEX_FULL_WIDTH_IN, FONT_COURIER,
                       PAPER_FONTSIZE, RC_BASE_ELASTIC, THREE_PANEL_ROW_HEIGHT_IN,
                       rcparams_with_fontsize)
from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import DATA_PATH, DEFAULT_WORKERS, ELEMENTS, load_density
from physics.inelastic_dielectric.polarization import barkas_dcs as bd
from physics.inelastic_dielectric.polarization import screened_barkas as sb
from physics.inelastic_dielectric.polarization import oscillator_quadrature as oq

IMPACT_RADIUS_RATIOS = (.2, 1., 2.)
FREQUENCIES_X = (.1, .3, 1.)
SCALED_STRENGTHS = np.array([.04, .02, .01])
CHECK_RTOL = 1e-3
RATIO_DIAGNOSTIC_SPEED_AU = 5.
STATES = [(element, q) for element, z in ELEMENTS.items() for q in range(z+1)]


class IndependentField:
    """Gauss-law quadrature, without the production radial-charge routine.

    Only the atomic density input is shared. Integrating the electron tail
    in log radius avoids cancellation for neutrals. A log-tail spline retains
    positivity; truncation discards less than 1e-260 electrons. Its numerical
    convergence is checked independently of the oscillator time integration.
    """

    def __init__(self, element, charge, nodes=8193):
        self.density = load_density(element, charge)
        d = self.density
        self.radius = np.sqrt(d.moment(2)/d.electrons) if d.electrons else 1.
        if not d.electrons:
            self.electron_integral = 0.
            return
        upper = 100*self.radius
        if d.electrons > 1:
            upper = max(upper, np.sqrt(1500/np.min(d.terms[:, 1])))
        r = np.geomspace(1e-7*self.radius, upper, nodes)
        lr = np.log(r)
        integrand = 4*np.pi*r**3*d.density(r)
        if np.any(~np.isfinite(integrand)) or np.min(integrand) < -1e-12:
            raise RuntimeError("Nonphysical density in independent field quadrature.")
        tail = cumulative_simpson(integrand[::-1], x=-lr[::-1], initial=0.)[::-1]
        self.electron_integral = float(tail[0])
        if not np.isclose(self.electron_integral, d.electrons, rtol=1e-9, atol=1e-10):
            raise RuntimeError("Independent density quadrature fails electron normalization.")
        keep = tail > 1e-260
        self.lower, self.upper = r[0], r[keep][-1]
        self.spline = CubicSpline(lr[keep], np.log(tail[keep]), extrapolate=False)

    def scalar(self, r):
        d = self.density
        if not d.electrons or r < self.lower:
            return float(d.z)
        if r > self.upper:
            return float(d.charge)
        return float(d.charge+np.exp(self.spline(np.log(r))))

    def __call__(self, r):
        r = np.asarray(r, float)
        if np.any(~np.isfinite(r)) or np.any(r < 0):
            raise ValueError("Radius must be finite and nonnegative.")
        return np.array([self.scalar(v) for v in r.flat]).reshape(r.shape)


@lru_cache(maxsize=76)
def independent_field(element, charge, nodes=8193):
    return IndependentField(element, charge, nodes)


def nonlinear_energy(x, b, eta, field, *, tail=64., rtol=2e-11):
    """Final energy / eta^2; eta=0 evaluates the linear forced oscillator.

    s=asinh(tau) resolves the brief central encounter without losing the
    long incoming/outgoing tails. DOP853 advances displacements and their
    tau derivatives, with both initially zero at the finite incoming bound.
    """
    if not (np.isfinite(x) and x > 0 and np.isfinite(b) and b > 0
            and np.isfinite(eta) and np.isfinite(tail) and tail > 0 and 0 < rtol < 1):
        raise ValueError("Invalid nonlinear oscillator input.")
    extent = tail*max(4., 1/x)
    end = np.arcsinh(extent)

    def rhs(s, state):
        tau = np.sinh(s)
        dx, dz = 1-eta*state[0], tau-eta*state[1]
        distance = np.hypot(dx, dz)
        if distance < 1e-8:
            raise RuntimeError("Trajectory approaches the nuclear singularity; no softened force was substituted.")
        force = field.scalar(b*distance)/distance**3
        return np.cosh(s)*np.array([state[2], state[3],
            -x*x*state[0]+force*dx, -x*x*state[1]+force*dz])

    solved = solve_ivp(rhs, (-end, end), np.zeros(4), method="DOP853",
                       rtol=rtol, atol=rtol*1e-2, max_step=.1, t_eval=[end])
    if not solved.success or np.any(~np.isfinite(solved.y)):
        raise RuntimeError("Nonlinear oscillator integration failed: "+solved.message)
    y = solved.y[:, -1]
    return float((np.dot(y[2:], y[2:])+x*x*np.dot(y[:2], y[:2]))/2)


def extrapolate_cubic(positive, negative, strengths):
    """Two overlapping Richardson extrapolations remove the eta^2 bias."""
    strengths = np.asarray(strengths, float)
    if (strengths.shape != (3,) or np.any(~np.isfinite(strengths))
            or not np.allclose(strengths, strengths[0]/2**np.arange(3))):
        raise ValueError("Three positive successively halved strengths are required.")
    if strengths[0] <= 0:
        raise ValueError("Strength must be positive.")
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if (positive.shape != (3,) or negative.shape != (3,)
            or np.any(~np.isfinite(positive)) or np.any(~np.isfinite(negative))):
        raise ValueError("Three finite positive/negative-coupling energies are required.")
    odd = (positive-negative)/(2*strengths)
    coarse = (4*odd[1]-odd[0])/3
    fine = (4*odd[2]-odd[1])/3
    return float(fine), float(abs(fine-coarse)), odd.tolist()


def extracted_cubic(x, b, field, *, tail=64., rtol=2e-11):
    # Z only controls the numerical derivative step, not the physical charge.
    strengths = SCALED_STRENGTHS/field.density.z
    positive = [nonlinear_energy(x, b, e, field, tail=tail, rtol=rtol) for e in strengths]
    negative = [nonlinear_energy(x, b, -e, field, tail=tail, rtol=rtol) for e in strengths]
    value, error, raw = extrapolate_cubic(positive, negative, strengths)
    return dict(value=value, extrapolation_error=error, raw_odd_coefficients=raw,
                strengths=strengths.tolist(),
                positive_scaled_energies=positive, negative_scaled_energies=negative)


def linear_energy_quadrature(x, b, field):
    """Independent Fourier integrals of the unperturbed force, to infinity."""
    def force(tau, component):
        h = np.hypot(1., tau)
        return field.scalar(b*h)/h**3*(tau if component else 1.)
    px = 2*quad(force, 0, np.inf, args=(0,), weight="cos", wvar=x,
                epsabs=1e-11, limlst=150)[0]
    pz = 2*quad(force, 0, np.inf, args=(1,), weight="sin", wvar=x,
                epsabs=1e-11, limlst=150)[0]
    return (px*px+pz*pz)/2


def check_case(case):
    element, charge, radius_ratio, x = case
    field = independent_field(element, charge)
    field_refined = independent_field(element, charge, 16385)
    density = field.density
    b = radius_ratio*field.radius
    nominal = extracted_cubic(x, b, field)
    tight = extracted_cubic(x, b, field, rtol=2e-12)
    extended = extracted_cubic(x, b, field, rtol=2e-12, tail=128.)
    field_check = extracted_cubic(x, b, field_refined, rtol=2e-12, tail=128.)
    products = oq.impulse_products([x], [b], density, order=12, extent=128.)
    refined = oq.impulse_products([x], [b], density, order=16, extent=256.)
    kernel = float(refined[0][0]+refined[1][0])
    kernel_coarse = float(products[0][0]+products[1][0])
    linear = nonlinear_energy(x, b, 0., field_refined, rtol=2e-12, tail=128.)
    linear_quad = linear_energy_quadrature(x, b, field_refined)
    if not np.isfinite(kernel) or abs(kernel) < 1e-10:
        raise RuntimeError("Cubic coefficient too small for the relative comparison.")
    scale = abs(kernel)
    checks = dict(
        nonlinear_vs_kernel=abs(field_check["value"]-kernel)/scale,
        strength_extrapolation=field_check["extrapolation_error"]/scale,
        radial_quadrature_refinement=abs(field_check["value"]-extended["value"])/scale,
        ode_tolerance=abs(tight["value"]-nominal["value"])/scale,
        integration_tail=abs(extended["value"]-tight["value"])/scale,
        production_kernel_refinement=abs(kernel-kernel_coarse)/scale,
        linear_vs_fourier=abs(linear/linear_quad-1))
    # A diagnostic within this oscillator, not the pipeline's finite-q Born.
    # At v=5 bohr/atomic-time and physical lambda=1, eta=1/(b*v^2).
    physical_eta = 1/(b*RATIO_DIAGNOSTIC_SPEED_AU**2)
    ratio = physical_eta*kernel/linear_quad
    return dict(element=element, charge=charge, x=x, b_bohr=b,
                radius_bohr=field.radius, b_over_radius=radius_ratio,
                status="PASS" if max(checks.values()) < CHECK_RTOL else "FAIL",
                cubic_nonlinear=field_check["value"], cubic_production_kernel=kernel,
                relative_checks=checks, nominal=nominal, tight=tight, extended=extended,
                field_refined=field_check,
                linear_ode=linear, linear_fourier=linear_quad,
                cubic_over_leading_at_v5=ratio,
                perturbation_uncontrolled_at_v5=bool(abs(ratio) >= 1))


def checked_case(case):
    """Record unresolved cases without silently dropping them from coverage."""
    try:
        return check_case(case)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        element, charge, ratio, x = case
        return dict(element=element, charge=charge, b_over_radius=ratio, x=x,
                    status="FAIL", error=f"{type(exc).__name__}: {exc}")


def check_state(state):
    element, charge = state
    field = independent_field(element, charge, 16385)
    d = field.density
    r = field.radius*np.array([0., .03, .2, 1., 2., 4., 8., 100.])
    production = d.radial_charge_direct(r)[0]
    field_error = float(np.max(np.abs(field(r)-production))/d.z)
    normalization_error = abs(field.electron_integral-d.electrons)
    # The independent time/impact integration must recover Salvat for every
    # bare nucleus; do not test only the production analytic-dispatch shortcut.
    bare_checks = []
    if not d.electrons:
        for beta in (.01, .4):
            gamma = 1/np.sqrt(1-beta*beta)
            xi = np.array([1e-5, .001, .1, 1.])
            w = xi*gamma*beta*beta*bd.MEC2_EV/.5616
            actual, _ = sb.converged_kernel(w, beta, gamma, d)
            expected = d.z**3*(bd.arbi1(xi)+bd.arbi2(xi)/gamma**2)
            bare_checks.append(dict(beta=beta, xi=xi.tolist(),
                max_relative_error=float(np.max(np.abs(actual/expected-1)))))
    passed = (field_error < 1e-8 and normalization_error < 1e-8
              and all(v["max_relative_error"] < CHECK_RTOL for v in bare_checks))
    return dict(element=element, charge=charge, status="PASS" if passed else "FAIL",
                electron_integral=field.electron_integral, expected_electrons=d.electrons,
                field_absolute_error_over_Z=field_error, bare_salvat_checks=bare_checks)


def plot_results(rows, out):
    """Render stored comparisons; colors distinguish b/r0, not new physics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator
    font = None
    for candidate in (FONT_COURIER, "Courier New", "Courier"):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
        except ValueError:
            continue
        font = candidate
        break
    if font is None:
        raise RuntimeError("Install a Courier font to render the comparison.")
    plt.rcParams.update(rcparams_with_fontsize(RC_BASE_ELASTIC, PAPER_FONTSIZE, {
        "font.family": font, "figure.titlesize": PAPER_FONTSIZE,
        "axes.grid": False, "axes.titlepad": 4,
        "axes.labelpad": 4, "axes.linewidth": .8,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2, "ytick.minor.size": 2,
        "xtick.major.width": .7, "ytick.major.width": .7,
        "xtick.minor.width": .6, "ytick.minor.width": .6,
        "xtick.major.pad": 3, "ytick.major.pad": 3,
        "pdf.fonttype": 42, "savefig.bbox": None,
        "mathtext.fontset": "custom", "mathtext.rm": font,
        "mathtext.it": font, "mathtext.bf": font, "mathtext.fallback": None}))
    fig = plt.figure(figsize=(AASTEX_FULL_WIDTH_IN, 2*THREE_PANEL_ROW_HEIGHT_IN))
    grid = fig.add_gridspec(2, 3,
                           left=.095, right=.90, top=.95, bottom=.12,
                           hspace=.14, wspace=.10)
    axes = [fig.add_subplot(grid[i//3, i%3]) for i in range(5)]
    legend_ax = fig.add_subplot(grid[1, 2])
    legend_ax.set_axis_off()
    ratios = sorted({r["b_over_radius"] for r in rows})
    norm = Normalize(min(IMPACT_RADIUS_RATIOS), max(IMPACT_RADIUS_RATIOS))
    # N+1 equally spaced Plasma samples; omit the final bright-yellow color.
    colors = plt.get_cmap("plasma")(np.linspace(0., 1., len(ratios)+1))[:-1]
    stops = [(float(norm(r)), color) for r, color in zip(ratios, colors)]
    cmap = LinearSegmentedColormap.from_list("plasma_first_n", stops)
    elements = list(dict.fromkeys(r["element"] for r in rows))
    plotted_values = []
    for panel, element, letter in zip(axes, elements, "abcde"):
        subset = [r for r in rows if r["element"] == element]
        charges = sorted({r["charge"] for r in subset})
        for ratio, color in zip(ratios, colors):
            values = []
            for charge in charges:
                cases = [r for r in subset if r["charge"] == charge and r["b_over_radius"] == ratio]
                usable = [r for r in cases if "relative_checks" in r]
                values.append([
                    100*max(r["relative_checks"]["nonlinear_vs_kernel"] for r in usable),
                    100*max(v for r in usable for k, v in r["relative_checks"].items()
                            if k != "nonlinear_vs_kernel")
                ] if usable else [np.nan, np.nan])
                if any(r["status"] != "PASS" for r in cases):
                    panel.plot(charge, .95, "x", color="black", ms=5,
                               transform=panel.get_xaxis_transform())
            values = np.asarray(values)
            plotted_values.extend(values[np.isfinite(values) & (values > 0)])
            marker = "o" if len(charges) == 1 else None
            panel.semilogy(charges, values[:, 0], color=color, lw=1.1, marker=marker,
                           ms=3, mfc=color, mec=color)
            panel.semilogy(charges, values[:, 1], color=color, lw=1.1, ls="--", marker=marker,
                           ms=3, mfc=color, mec=color)
        panel.set_title(f"({letter})", loc="left")
        panel.text(.95, .055, element, transform=panel.transAxes, ha="right", va="bottom")
        index = axes.index(panel)
        if index not in (0, 1):
            panel.set_xlabel("Charge state ($Q$)")
        if index not in (0, 3):
            panel.tick_params(labelleft=False)
        panel.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        panel.xaxis.set_minor_locator(AutoMinorLocator(2))
        panel.tick_params(which="both", direction="in", top=False, right=False)
    if not plotted_values:
        plt.close(fig)
        raise ValueError("No finite positive benchmark differences to plot on a log axis")
    for panel in axes[:len(elements)]:
        panel.set_ylim(min(plotted_values)/2., max(plotted_values)*1.6)
    for panel in axes[len(elements):]:
        panel.set_visible(False)
    handles = [Line2D([], [], color="black", lw=1.1),
               Line2D([], [], color="black", lw=1.1, ls="--")]
    labels = ["ODE cubic vs kernel", "Numerical convergence"]
    if any(r["status"] != "PASS" for r in rows):
        handles.append(Line2D([], [], color="black", marker="x", ls="none", ms=5))
        labels.append("Failed check")
    legend_ax.legend(handles, labels, loc="upper left", frameon=False,
                     bbox_to_anchor=(0., .88), borderaxespad=0., labelspacing=1., handlelength=2.2)
    legend_ax.text(0., .52, "ODE: nonlinear trajectory\n"
                   "Kernel: force-gradient\n\n"
                   "Maximum over\n$x=0.1,\\,0.3,\\,1$.\n\n"
                   "$r_0$: RMS electron radius;\n$a_0$ for bare ions.",
                   transform=legend_ax.transAxes, ha="left", va="top", linespacing=1.5)
    # Anchor to the rightmost visible top panel; the bar has exactly its height.
    color_parent = axes[min(2, len(elements)-1)]
    color_ax = color_parent.inset_axes([1.012, 0., .035, 1.])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax, ticks=ratios)
    bar.set_label("Impact parameter ($b/r_0$)")
    bar.ax.tick_params(which="both", direction="out", length=3.5)
    fig.text(.035, .535, "Maximum difference (%)", rotation=90,
             ha="center", va="center")
    out.mkdir(parents=True, exist_ok=True)
    path = out/"all_charge_states_nonlinear_check.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--element", nargs="+", choices=list(ELEMENTS), default=list(ELEMENTS))
    parser.add_argument("--charge", type=int, help="Restrict to one charge; requires one --element")
    parser.add_argument("--plot-only", action="store_true",
                        help="Replot the existing comparison.json; do not rerun or overwrite benchmark results")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent/"plots/nonlinear_oscillator")
    args = parser.parse_args()
    if args.plot_only:
        report = json.loads((args.out_dir/"comparison.json").read_text())
        rows = [r for r in report["rows"] if r["element"] in args.element
                and (args.charge is None or r["charge"] == args.charge)]
        if not rows:
            parser.error("No saved benchmark rows match the requested state selection")
        print(plot_results(rows, args.out_dir))
        return 0
    if args.workers < 1:
        parser.error("--workers must be positive")
    states = [s for s in STATES if s[0] in args.element]
    if args.charge is not None:
        if len(args.element) != 1 or (args.element[0], args.charge) not in states:
            parser.error("--charge requires one element and a supported charge state")
        states = [(args.element[0], args.charge)]
    cases = [(element, q, ratio, x) for element, q in states
             for ratio in IMPACT_RADIUS_RATIOS for x in FREQUENCIES_X]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(cases)),
                             mp_context=get_context("spawn")) as pool:
        state_checks = list(pool.map(check_state, states))
        for state in state_checks:
            print(f"Field/bare limit {state['element']}:{state['charge']}: {state['status']}", flush=True)
        rows = []
        for row in pool.map(checked_case, cases):
            rows.append(row)
            message = (f"cubic difference={100*row['relative_checks']['nonlinear_vs_kernel']:.6g}%"
                       if "relative_checks" in row else row["error"])
            print(f"{row['element']}:{row['charge']}, b/r={row['b_over_radius']:g}, "
                  f"x={row['x']:g}: {row['status']}, {message}", flush=True)
    sources = [Path(__file__), PHYSICS_ROOT/"inelastic_dielectric/polarization/screened_barkas.py",
               Path(oq.__file__),
               PHYSICS_ROOT/"inelastic_dielectric/projectile_potentials/projectile_form_factors.py", PHYSICS_ROOT/"constants.py",
               PHYSICS_ROOT/"inelastic_dielectric/polarization/barkas_dcs.py",
               PHYSICS_ROOT/"inelastic_dielectric/k_shell/hydrogenic.py", DATA_PATH]
    passed = all(r["status"] == "PASS" for r in rows+state_checks)
    repo = PHYSICS_ROOT.parent
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    maxima = {k: max(r["relative_checks"][k] for r in rows if "relative_checks" in r)
              for k in next((r["relative_checks"] for r in rows if "relative_checks" in r), {})}
    report = dict(status="PASS" if passed else "FAIL", created_utc=datetime.now(timezone.utc).isoformat(),
                  purpose="Independent implementation check of nonrelativistic all-state oscillator cubic term",
                  ice_DCS_validation=False, external_paper_reproduction=False,
                  screened_relativistic_validation=False, ICRU_comparison_performed=False,
                  production_generator_changed=False, target="single classical harmonic oscillator",
                  state_count=len(states), case_count=len(cases), state_checks=state_checks,
                  max_relative_checks=maxima,
                  failed_case_count=sum(r["status"] != "PASS" for r in rows),
                  uncontrolled_oscillator_case_count=sum(r.get("perturbation_uncontrolled_at_v5", False) for r in rows),
                  relativity="strict nonrelativistic limit; bare Salvat checks also at beta=0.4",
                  git_commit=commit, code_identity="Hashes identify the working files, including uncommitted changes",
                  ode_method="scipy.integrate.solve_ivp DOP853", ode_max_step_asinh_tau=.1,
                  ode_nominal_rtol=2e-11, ode_tight_rtol=2e-12, ode_atol_over_rtol=.01,
                  ode_extent="tail*max(4,1/x)", ode_tail_parameters=[64., 128.],
                  kernel_quadrature_method=oq.VERSION,
                  kernel_comparison_settings=[dict(order=12, extent=128.), dict(order=16, extent=256.)],
                  workers=min(args.workers, len(cases)),
                  software_versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  physical_ratio_speed_au=RATIO_DIAGNOSTIC_SPEED_AU,
                  equation="u''+x^2*u=eta*g(b*|R-u|)*(R-u)/|R-u|^3; R=(1,tau)",
                  energy="E=eta^2*H; cubic coefficient=lim_eta->0 [H(eta)-H(-eta)]/(2*eta)",
                  energy_units="E is physical oscillator energy divided by m_e*v^2",
                  strengths_times_Z=SCALED_STRENGTHS.tolist(), relative_tolerance=CHECK_RTOL,
                  impact_parameter_radius_ratios=list(IMPACT_RADIUS_RATIOS),
                  impact_radius="RMS electron radius; bare states use 1 bohr as an arbitrary geometric scale",
                  radial_quadrature_nodes=[8193, 16385],
                  independent_field="Numerical electron-tail quadrature of shared atomic density; log-tail cubic spline",
                  physical_ratio_note="v=5 au, lambda=1: comparison with this oscillator's leading term, not finite-q Born DCS",
                  code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, rows=rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/"comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(plot_results(rows, args.out_dir))
    print(report["status"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
