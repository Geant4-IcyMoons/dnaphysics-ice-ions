#!/usr/bin/env python3
"""Numerical benchmark of all 38 frozen projectile states and their Born kernels.

This is not an experimental stopping-power validation. Reference checks are
analytic one-electron atoms, independent radial Fourier quadrature, direct
PySCF AO density export, and QZ/5Z radial-basis convergence. No ICRU fit enters.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
PHYSICS = Path(__file__).resolve().parents[3]

import numpy as np
from scipy.integrate import quad
import scipy
from physics.inelastic_dielectric.projectile_potentials import projectile_form_factors as ff
from physics.inelastic_dielectric import generate_cross_sections as gen
from physics.constants import DIAGNOSTIC_PLOTS_DIR, FONT_COURIER


def benchmark_state(state):
    element, charge = state
    density = ff.load_density(element, charge)
    k = np.geomspace(1.123e-6, 9.876e5, 611)
    direct = charge+density.electron_deficit(k, direct=True)
    interpolation_error = float(np.max(np.abs(density.charge_amplitude(k)/direct-1))) if density.electrons else 0.0
    transform_error = 0.0
    radial_count = 0.0
    if density.electrons:
        radial_count = quad(lambda r: 4*np.pi*r*r*density.density(r), 0, 60, epsabs=2e-10, epsrel=1e-11)[0]
        for wave in (0.3, 3.0, 30.0):
            integral = quad(lambda r: 4*np.pi*r*density.density(r)/wave,
                            0, 60, weight="sin", wvar=wave, epsabs=1e-9, limit=200)[0]
            transform_error = max(transform_error, abs(float(density.form_factor(wave, direct=True))-integral))
    one_electron_error = 0.0
    if density.electrons == 1:
        reference = (1+(k/(2*density.z))**2)**-2
        one_electron_error = float(np.max(np.abs(density.form_factor(k)-reference)))
    # Actual production excitation, valence-ionization and K-continuum DCS,
    # sampled in both phases at total energies corresponding to 1,10,100 MeV/u.
    gen.set_projectile(element)
    gen._set_projectile_charge_state(charge)
    gen._set_kshell_model("hydrogenic-gos")
    C = gen.model.default_dispersion_coefficients()
    samples = []
    for phase in ("amorphous", "hexagonal"):
        s = gen.model.epsilon_optical(phase)
        for relativistic in (False, True):
            gen._set_projectile_relativistic_dcs(relativistic)
            for energy in (1, 10, 100):
                T = gen.PROJECTILE_MASS_NUMBER*energy*1e6
                for channel, W in (("excitation", 15), ("ionization", 50), ("kshell", 1000)):
                    function = getattr(gen, "_selected_dsigma_"+channel)
                    args = (W, T, s, C) if channel == "kshell" else (W, T, 0, s, C)
                    coarse, fine = (float(function(*args, nq)) for nq in (1000, 2000))
                    if not np.isfinite(fine) or fine <= 0 or not np.isfinite(coarse) or coarse <= 0:
                        raise RuntimeError(f"Invalid production DCS for {state}, {phase}, {T}, {channel}.")
                    samples.append(dict(phase=phase, kernel="RPWBA" if relativistic else "PWBA",
                                        total_energy_eV=T, loss_eV=W, channel=channel,
                                        dcs_m2_per_eV=fine, nq_relative_difference=abs(coarse/fine-1)))
    comparison = density.record.get("basis_comparison", {})
    return dict(element=element, charge=charge, electrons=density.electrons,
                normalization_error=abs(density.moment(0)-density.electrons),
                radial_normalization_error=abs(radial_count-density.electrons),
                radial_fourier_max_absolute_error=transform_error,
                hydrogenic_max_absolute_error=one_electron_error,
                interpolation_max_amplitude_relative_error=interpolation_error,
                ao_density_export_error=density.record.get("ao_radial_export_max_scaled_error", 0),
                basis_max_squared_charge_relative_difference=comparison.get("max_squared_charge_relative_difference", 0),
                kernel_max_nq_relative_difference=max(row["nq_relative_difference"] for row in samples),
                samples=samples)


def benchmark_helium_reference():
    """Koga, Phys. Rev. A 41, 1274 (1990), Eq. (7) and Tables I--II.

    Table I gives an UNNORMALIZED three-exponential density, with rounded
    coefficients. Normalize that reference to two electrons, not our HF data.
    The tabulated radial moments are per electron. This reference is not used
    by the atomic solver or the production form factor.
    """
    density = ff.load_density("He", 0)
    a = np.array([2.8024, 3.5822, 5.2275])
    weights = np.array([1.0, 1.4190, 1.5099])/a**3
    weights *= 2/weights.sum()
    k = np.geomspace(1e-3, 100, 501)
    reference = np.sum(weights[:, None]/(1+(k[None, :]/a[:, None])**2)**2, axis=0)
    moments = {1: 0.92724, 2: 1.18464, 3: 1.93977, 4: 3.8838}
    actual = {order: density.moment(order)/2 for order in moments}
    energy_error = abs(density.record["energy_hartree"] - (-2.8616800))
    form_error = float(np.max(np.abs(density.form_factor(k)-reference)))
    moment_error = max(abs(actual[order]/ref-1) for order, ref in moments.items())
    return dict(reference="https://doi.org/10.1103/PhysRevA.41.1274",
                accessible_copy="https://muroran-it.repo.nii.ac.jp/record/5505/files/71_PhysRevA41_1990_1274.pdf",
                locations="p. 1275 Eq. (7); p. 1276 Tables I and II",
                reference_kind="Published HF density approximation and near-HF-limit energy/moments; not experiment",
                energy_error_hartree=energy_error, form_factor_max_absolute_error=form_error,
                radial_moment_max_relative_error=moment_error,
                radial_moments_per_electron=actual, reference_radial_moments_per_electron=moments,
                limits=dict(energy_hartree=1e-4, form_factor_absolute=1e-3, radial_moment_relative=0.005),
                status="PASS" if energy_error < 1e-4 and form_error < 1e-3 and moment_error < 0.005 else "FAIL")


def make_plot(output, states):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in (FONT_COURIER, "Courier", "Courier New") if name in available), None)
    if family is None:
        raise RuntimeError("Install a Courier family font before plotting; no proportional-font fallback is allowed.")
    plt.rcParams.update({"font.family": family, "font.size": 9,
                         "axes.titlesize": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "axes.grid": False})
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.8), layout="constrained")
    k = np.geomspace(0.01, 1000, 601)
    cmap = plt.get_cmap("viridis")
    for ax, (element, z), letter in zip(axes.flat, ff.ELEMENTS.items(), "abcde"):
        for q in range(z+1):
            ax.semilogx(k, ff.load_density(element, q).squared_charge(k)/z**2, color=cmap(q/z), lw=1)
        ax.set(title=f"({letter}) {element}", xlabel="Wave number (1/bohr)", ylabel="Squared charge / Z^2",
               xlim=(0.01, 1000), ylim=(-0.03, 1.03))
        ax.set_xticks([0.01, 1, 1000], ["0.01", "1", "1000"])
        ax.tick_params(direction="in", which="both", top=True, right=True)
    ax = axes.flat[-1]
    for element in ff.ELEMENTS:
        rows = [row for row in states if row["element"] == element and row["electrons"] > 1]
        if rows:
            ax.plot([row["charge"] for row in rows],
                    [100*row["basis_max_squared_charge_relative_difference"] for row in rows],
                    marker="o", markersize=2, label=element)
    ax.set(title="(f) QZ to 5Z change", xlabel="Charge state Q", ylabel="Maximum factor change (%)")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(frameon=False, loc="upper right")
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=list(axes.flat[:5]),
                 orientation="horizontal", shrink=0.6, label="Q/Z (0: neutral; 1: bare)", pad=0.03)
    fig.savefig(output/"projectile_form_factors.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=ff.DEFAULT_WORKERS)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"plots/form_factors")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    all_states = [(element, q) for element, z in ff.ELEMENTS.items() for q in range(z+1)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        states = list(pool.map(benchmark_state, all_states))
    # Numerical acceptance only. The basis bound is not an HF physical-error bar.
    limits = dict(normalization_error=1e-8, radial_normalization_error=3e-8,
                  radial_fourier_max_absolute_error=3e-8, hydrogenic_max_absolute_error=1e-14,
                  interpolation_max_amplitude_relative_error=2e-7, ao_density_export_error=1e-10,
                  basis_max_squared_charge_relative_difference=0.005,
                  kernel_max_nq_relative_difference=0.01)
    checks = {key: {"maximum": max(row[key] for row in states), "limit": limit,
                    "pass": all(row[key] <= limit for row in states)} for key, limit in limits.items()}
    external = benchmark_helium_reference()
    report = dict(model=ff.VERSION, atomic_data_sha256=hashlib.sha256(ff.DATA_PATH.read_bytes()).hexdigest(),
                  source_sha256={name: hashlib.sha256((PHYSICS/name).read_bytes()).hexdigest()
                                 for name in ('inelastic_dielectric/projectile_potentials/projectile_form_factors.py', 'inelastic_dielectric/generate_cross_sections.py', 'inelastic_dielectric/finite_q/emfietzoglou_model_finite_q.py', 'inelastic_dielectric/projectile_potentials/generate_projectile_atomic_data.py')},
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  workers=args.workers, checks=checks, states=states,
                  external_helium_reference=external,
                  status="PASS" if all(check["pass"] for check in checks.values()) and external["status"] == "PASS" else "FAIL",
                  experimental_stopping_validation="NOT PERFORMED", reference_scope=__doc__)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    make_plot(args.output, states)
    print(json.dumps({"status": report["status"], "checks": checks, "helium_reference": external, "output": str(args.output)}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
