#!/usr/bin/env python3
"""Plot ion stopping/IMFP diagnostics from dnaphysics_proton ROOT output.

The ROOT file must be written in full log mode so the ``step`` tree contains
``flagProcess``, ``stepLength``, ``kineticEnergyDifference`` and
``vibCrossSection``.  In the proton app the legacy ``vibCrossSection`` branch
stores the macroscopic cross section of the process that limited the step
in 1/cm, not only vibrational excitation.

This is the simulation/ROOT analogue of the table-based projectile plotter:
it derives stopping power and IMFP from simulated steps, and uses the recorded
process cross section only as a diagnostic overlay.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot
from matplotlib.ticker import LogLocator, NullFormatter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from physics_ice.constants import (  # noqa: E402
    DIAGNOSTIC_PLOTS_DIR,
    FONT_COURIER,
    FONTSIZE_24,
    RC_BASE_ELASTIC,
    rcparams_with_fontsize,
)
from physics_ice.root_utils import resolve_root_paths  # noqa: E402


ION_CONFIGS = {
    "proton": {
        "flag": 2,
        "flags": (2, 3),
        "excitation": (22, 32),
        "ionisation": (23, 33),
        "label": "Proton",
    },
    "hydrogen": {
        "flag": 3,
        "flags": (3,),
        "excitation": (32,),
        "ionisation": (33,),
        "label": "Hydrogen",
    },
    "alpha": {
        "flag": 4,
        "flags": (4, 5, 6),
        "excitation": (42, 52, 62),
        "ionisation": (43, 53, 63),
        "label": "Alpha",
    },
    "alpha+": {
        "flag": 5,
        "flags": (5,),
        "excitation": (52,),
        "ionisation": (53,),
        "label": "Alpha+",
    },
    "helium": {
        "flag": 6,
        "flags": (6,),
        "excitation": (62,),
        "ionisation": (63,),
        "label": "Helium",
    },
}
NM_TO_ANGSTROM = 10.0
EV_NM_TO_EV_ANG = 0.1
EV_NM_TO_MEV_CM = 10.0


def _load_step_arrays(root_arg: str | Path) -> dict[str, np.ndarray]:
    paths = resolve_root_paths(root_arg)
    if not paths:
        raise FileNotFoundError(root_arg)

    required = [
        "flagParticle",
        "flagProcess",
        "kineticEnergy",
        "kineticEnergyDifference",
        "stepLength",
        "vibCrossSection",
    ]
    tree_specs = [f"{path}:step" for path in paths]
    try:
        return uproot.concatenate(tree_specs, required, library="np")
    except uproot.KeyInFileError as exc:
        missing = str(exc)
        raise RuntimeError(
            "ROOT step tree is missing full diagnostic branches. "
            "Run the proton macro with /dna/test/setLogMode full."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Could not read ROOT step tree from {paths}") from exc


def _bin_mean(values: np.ndarray, bins: np.ndarray, nbins: int) -> np.ndarray:
    out = np.full(nbins, np.nan, dtype=float)
    valid = np.isfinite(values) & np.isfinite(bins) & (bins >= 0) & (bins < nbins)
    if not np.any(valid):
        return out
    sums = np.bincount(bins[valid], weights=values[valid], minlength=nbins)
    counts = np.bincount(bins[valid], minlength=nbins)
    good = counts > 0
    out[good] = sums[good] / counts[good]
    return out


def compute_observables(
    arrs: dict[str, np.ndarray],
    particle_key: str,
    emin_eV: float | None,
    emax_eV: float | None,
    nbins: int,
    n_h2o_cm3: float,
    density_g_cm3: float,
) -> dict[str, np.ndarray]:
    cfg = ION_CONFIGS[particle_key]
    particle_flags = tuple(int(flag) for flag in cfg.get("flags", (cfg["flag"],)))
    flag_particle = np.asarray(arrs["flagParticle"], dtype=int)
    flag_process = np.asarray(arrs["flagProcess"], dtype=int)
    energy_eV = np.asarray(arrs["kineticEnergy"], dtype=float)
    dE_eV = np.asarray(arrs["kineticEnergyDifference"], dtype=float)
    step_nm = np.asarray(arrs["stepLength"], dtype=float)
    macro_xs_cm_inv = np.asarray(arrs["vibCrossSection"], dtype=float)

    ion = (
        np.isin(flag_particle, particle_flags)
        & np.isfinite(energy_eV)
        & np.isfinite(dE_eV)
        & np.isfinite(step_nm)
        & (energy_eV > 0.0)
        & (step_nm > 0.0)
    )
    if not np.any(ion):
        raise RuntimeError(f"No {particle_key} steps found in ROOT step tree.")

    e = energy_eV[ion]
    if emin_eV is None:
        emin_eV = float(np.nanmin(e))
    if emax_eV is None:
        emax_eV = float(np.nanmax(e))
    if emin_eV <= 0.0 or emax_eV <= emin_eV:
        raise ValueError("Require 0 < emin_eV < emax_eV.")

    edges = np.logspace(np.log10(emin_eV), np.log10(emax_eV), nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    ion_bins = np.digitize(e, edges) - 1
    in_range = (ion_bins >= 0) & (ion_bins < nbins)
    ion_idx = np.flatnonzero(ion)[in_range]
    bins = ion_bins[in_range]

    step_in = step_nm[ion][in_range]
    dE_in = np.maximum(dE_eV[ion][in_range], 0.0)
    proc_in = flag_process[ion][in_range]

    path_nm = np.bincount(bins, weights=step_in, minlength=nbins)
    energy_loss_eV = np.bincount(bins, weights=dE_in, minlength=nbins)
    is_exc = np.isin(proc_in, list(cfg["excitation"]))
    is_ion = np.isin(proc_in, list(cfg["ionisation"]))
    inelastic = is_exc | is_ion
    n_inelastic = np.bincount(bins[inelastic], minlength=nbins)

    stopping_eV_per_A = np.divide(
        energy_loss_eV,
        path_nm * NM_TO_ANGSTROM,
        out=np.full(nbins, np.nan, dtype=float),
        where=path_nm > 0.0,
    )
    inverse_mfp_nm_inv = np.divide(
        n_inelastic,
        path_nm,
        out=np.full(nbins, np.nan, dtype=float),
        where=path_nm > 0.0,
    )
    imfp_nm = np.divide(
        1.0,
        inverse_mfp_nm_inv,
        out=np.full(nbins, np.nan, dtype=float),
        where=inverse_mfp_nm_inv > 0.0,
    )
    stopping_eV_per_nm = np.divide(
        energy_loss_eV,
        path_nm,
        out=np.full(nbins, np.nan, dtype=float),
        where=path_nm > 0.0,
    )
    stopping_MeV_cm2_g = (stopping_eV_per_nm * EV_NM_TO_MEV_CM) / density_g_cm3

    macro_in = macro_xs_cm_inv[ion_idx]
    micro_in = np.divide(
        macro_in,
        n_h2o_cm3,
        out=np.full_like(macro_in, np.nan, dtype=float),
        where=n_h2o_cm3 > 0.0,
    )
    sigma_exc = _bin_mean(micro_in[is_exc], bins[is_exc], nbins)
    sigma_ion = _bin_mean(micro_in[is_ion], bins[is_ion], nbins)
    sigma_total = np.nansum(np.vstack([sigma_exc, sigma_ion]), axis=0)
    sigma_total[~np.isfinite(sigma_exc) & ~np.isfinite(sigma_ion)] = np.nan

    return {
        "energy_eV": centers,
        "path_length_nm": path_nm,
        "n_inelastic": n_inelastic.astype(float),
        "stopping_power_eV_per_nm": stopping_eV_per_nm,
        "stopping_power_eV_per_A": stopping_eV_per_A,
        "mass_stopping_power_MeV_cm2_g": stopping_MeV_cm2_g,
        "inverse_mfp_nm_inv": inverse_mfp_nm_inv,
        "imfp_nm": imfp_nm,
        "sigma_excitation_cm2": sigma_exc,
        "sigma_ionisation_cm2": sigma_ion,
        "sigma_total_cm2": sigma_total,
    }


def write_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(data.keys())
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for row in zip(*(data[key] for key in keys)):
            writer.writerow([f"{float(value):.10e}" for value in row])


def _positive(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    return x[mask], y[mask]


def _stopping_values(data: dict[str, np.ndarray], units: str) -> tuple[np.ndarray, str]:
    if units == "ev_ang":
        return data["stopping_power_eV_per_nm"] * EV_NM_TO_EV_ANG, r"Stopping power (eV/$\AA$)"
    if units == "ev_nm":
        return data["stopping_power_eV_per_nm"], "Stopping power (eV/nm)"
    if units == "mev_cm2_g":
        return data["mass_stopping_power_MeV_cm2_g"], r"Mass stopping power (MeV cm$^2$/g)"
    return data["stopping_power_eV_per_nm"] * EV_NM_TO_MEV_CM, "Stopping power (MeV/cm)"


def plot_observables(
    path: Path,
    data: dict[str, np.ndarray],
    particle_label: str,
    units: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = FONT_COURIER
    plt.rcParams["mathtext.rm"] = FONT_COURIER
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams.update(rcparams_with_fontsize(RC_BASE_ELASTIC, FONTSIZE_24))

    e_mev = data["energy_eV"] * 1.0e-6
    stopping, stopping_label = _stopping_values(data, units)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, 12.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 2.0, 2.2], "hspace": 0.12},
    )

    x, y = _positive(e_mev, stopping)
    axes[0].loglog(x, y, "o", color="black", markerfacecolor="black", label="Simulation")
    axes[0].set_ylabel(stopping_label)

    x, y = _positive(e_mev, data["imfp_nm"])
    axes[1].loglog(x, y, "o", color="#2166ac", markerfacecolor="#2166ac")
    axes[1].set_ylabel("IMFP (nm)")

    for key, label, color in [
        ("sigma_excitation_cm2", "Excitation", "#1b9e77"),
        ("sigma_ionisation_cm2", "Ionisation", "#d95f02"),
        ("sigma_total_cm2", "Total", "black"),
    ]:
        x, y = _positive(e_mev, data[key])
        if x.size:
            axes[2].loglog(x, y, "o", color=color, markerfacecolor=color, label=label)
    axes[2].set_ylabel(r"Cross section (cm$^2$)")
    axes[2].set_xlabel(f"{particle_label} energy ($T$; MeV)")
    axes[2].legend(frameon=False, loc="best", ncol=1, fontsize=18)

    for ax in axes:
        ax.grid(False)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_formatter(NullFormatter())

    fig.subplots_adjust(left=0.17, right=0.98, top=0.98, bottom=0.11, hspace=0.10)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ion stopping power, IMFP, and process XS from ROOT output."
    )
    parser.add_argument("--root", default="dna.root", help="ROOT file or split ROOT glob.")
    parser.add_argument(
        "--particle",
        default="proton",
        choices=tuple(ION_CONFIGS),
        help="Ion species encoded in flagParticle/flagProcess.",
    )
    parser.add_argument("--out", default="proton_root_stopping_imfp_xs.png")
    parser.add_argument("--csv", default="proton_root_stopping_imfp_xs.csv")
    parser.add_argument("--emin-eV", type=float, default=None)
    parser.add_argument("--emax-eV", type=float, default=None)
    parser.add_argument("--nbins", type=int, default=40)
    parser.add_argument("--nH2O-cm3", type=float, default=3.343e22)
    parser.add_argument("--density-g-cm3", type=float, default=1.0)
    parser.add_argument(
        "--units",
        choices=("ev_ang", "ev_nm", "mev_cm", "mev_cm2_g"),
        default="ev_ang",
        help="Stopping-power units.",
    )
    args = parser.parse_args()

    if args.nbins < 2:
        raise ValueError("--nbins must be at least 2.")

    arrs = _load_step_arrays(args.root)
    data = compute_observables(
        arrs,
        args.particle,
        args.emin_eV,
        args.emax_eV,
        args.nbins,
        args.nH2O_cm3,
        args.density_g_cm3,
    )

    out_path = DIAGNOSTIC_PLOTS_DIR / Path(args.out).name
    csv_path = DIAGNOSTIC_PLOTS_DIR / Path(args.csv).name
    plot_observables(out_path, data, ION_CONFIGS[args.particle]["label"], args.units)
    write_csv(csv_path, data)
    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
