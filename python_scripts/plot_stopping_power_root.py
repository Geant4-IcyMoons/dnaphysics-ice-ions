#!/usr/bin/env python3
"""
Plot stopping power from a Geant4-DNA ROOT file (tree: "step").

Computes dE/dx from per-step data and bins it versus kinetic energy.
Uses the same plotting style as plot_elastic_cross_sections.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uproot

from constants import (
    FONT_COURIER,
    FONTSIZE_24,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RC_BASE_ELASTIC,
    rcparams_with_fontsize,
)

# Font configuration (match plot_elastic_cross_sections.py)
font = FONT_COURIER
plt.rcParams["font.family"] = font
plt.rcParams["mathtext.rm"] = font
plt.rcParams["mathtext.fontset"] = "custom"
FONTSIZE = FONTSIZE_24
plt.rcParams.update(rcparams_with_fontsize(RC_BASE_ELASTIC, FONTSIZE))

# Unit conversion: (eV / nm) -> (MeV / cm)
EV_NM_TO_MEV_CM = 10.0


def _load_step_tree(path: Path, tree_name: str) -> dict[str, np.ndarray]:
    with uproot.open(path) as f:
        if tree_name not in f:
            raise KeyError(f"Tree '{tree_name}' not found in {path}")
        tree = f[tree_name]
        cols = [
            "kineticEnergy",
            "kineticEnergyDifference",
            "totalEnergyDeposit",
            "stepLength",
            "trackID",
            "parentID",
            "flagParticle",
            "flagProcess",
            "processName",
        ]
        data = {name: tree[name].array(library="np") for name in cols if name in tree.keys()}
    return data


def _decode_str_array(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    out = []
    for val in arr:
        if isinstance(val, (bytes, bytearray)):
            s = val.decode(errors="ignore")
        else:
            s = str(val)
        if "\x00" in s:
            s = s.split("\x00", 1)[0]
        out.append(s.strip())
    return np.asarray(out, dtype=str)


def _binned_sum(x: np.ndarray, w: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return np.histogram(x, bins=bins, weights=w)[0]


def _units_and_label(units: str, density: float, dedx: np.ndarray) -> tuple[np.ndarray, str]:
    if units == "mev_cm":
        return dedx * EV_NM_TO_MEV_CM, "Stopping Power (MeV/cm)"
    if units == "mev_cm2_g":
        return (dedx * EV_NM_TO_MEV_CM) / density, r"Mass Stopping Power (MeV cm$^2$/g)"
    return dedx, "Stopping Power (eV/nm)"


def _process_category_masks(
    proc_names: np.ndarray | None, flag_proc: np.ndarray | None
) -> list[tuple[str, np.ndarray]]:
    categories: list[tuple[str, np.ndarray]] = []
    if proc_names is not None:
        pname = np.char.lower(proc_names)
        is_vib = np.char.find(pname, "vib") >= 0
        is_attach = np.char.find(pname, "attach") >= 0
        is_ion = (np.char.find(pname, "ionis") >= 0) | (np.char.find(pname, "ioniz") >= 0)
        is_exc = (np.char.find(pname, "excitation") >= 0) & ~is_vib
        is_elastic = np.char.find(pname, "elastic") >= 0
        is_solv = np.char.find(pname, "solvation") >= 0
        categories = [
            ("Vib. Exc.", is_vib),
            ("Excit.", is_exc),
            ("Ion.", is_ion),
            ("De-Att.", is_attach),
            ("Elastic", is_elastic),
            ("Solv.", is_solv),
        ]
    elif flag_proc is not None:
        fp = np.rint(flag_proc).astype(int)
        categories = [
            ("Vib. Exc.", fp == 15),
            ("Excit.", fp == 12),
            ("Ion.", fp == 13),
            ("De-Att.", fp == 14),
            ("Elastic", fp == 11),
            ("Solv.", fp == 10),
        ]
    return [(label, mask) for label, mask in categories if np.any(mask)]


def plot_stopping_power(
    root_path: Path,
    out_path: Path,
    tree_name: str,
    nbins: int,
    emin: float | None,
    emax: float | None,
    use_edep: bool,
    all_tracks: bool,
    particle: int | None,
    units: str,
    density: float,
    label: str,
    split_by_process: bool,
) -> None:
    data = _load_step_tree(root_path, tree_name)
    kin_e = data["kineticEnergy"]
    dE = data["totalEnergyDeposit"] if use_edep else data["kineticEnergyDifference"]
    dx = data["stepLength"]
    proc_names = _decode_str_array(data.get("processName"))
    flag_proc = data.get("flagProcess")

    mask = np.isfinite(kin_e) & np.isfinite(dE) & np.isfinite(dx)
    mask &= (kin_e > 0) & (dx > 0)
    if not all_tracks:
        if "trackID" in data and "parentID" in data:
            mask &= (data["trackID"] == 1) & (data["parentID"] == 0)
    if particle is not None and "flagParticle" in data:
        mask &= (data["flagParticle"] == particle)

    kin_e = kin_e[mask]
    dE = dE[mask]
    dx = dx[mask]
    if proc_names is not None:
        proc_names = proc_names[mask]
    if flag_proc is not None:
        flag_proc = np.asarray(flag_proc, dtype=float)[mask]

    if kin_e.size == 0:
        raise ValueError("No valid steps after filtering.")

    dE_use = np.where(dE > 0, dE, 0.0)

    if emin is None:
        emin = float(np.min(kin_e))
    if emax is None:
        emax = float(np.max(kin_e))
    if emin <= 0 or emax <= 0 or emin >= emax:
        raise ValueError(f"Invalid energy range: emin={emin}, emax={emax}")

    bins = np.logspace(np.log10(emin), np.log10(emax), nbins + 1)
    centers = np.sqrt(bins[:-1] * bins[1:])
    total_dx = _binned_sum(kin_e, dx, bins)
    total_dE = _binned_sum(kin_e, dE_use, bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        total_dedx = np.where(total_dx > 0, total_dE / total_dx, np.nan)

    dedx_plot_total, ylabel = _units_and_label(units, density, total_dedx)

    fig, ax = plt.subplots(figsize=(10, 6))
    if split_by_process:
        categories = _process_category_masks(proc_names, flag_proc)
        if categories:
            cmap = plt.cm.plasma
            n = len(categories)
            for i, (cat_label, cat_mask) in enumerate(categories):
                dE_cat = _binned_sum(kin_e[cat_mask], dE_use[cat_mask], bins)
                with np.errstate(divide="ignore", invalid="ignore"):
                    dedx_cat = np.where(total_dx > 0, dE_cat / total_dx, np.nan)
                dedx_plot, _ = _units_and_label(units, density, dedx_cat)
                valid = np.isfinite(dedx_plot) & (dedx_plot > 0)
                if not np.any(valid):
                    continue
                t = 0.5 if n == 1 else (i / (n - 1)) * 0.85
                color = cmap(t)
                ax.loglog(
                    centers[valid],
                    dedx_plot[valid],
                    color=color,
                    linewidth=3,
                    label=cat_label,
                    zorder=3,
                )
        else:
            split_by_process = False

    if not split_by_process:
        valid = np.isfinite(dedx_plot_total) & (dedx_plot_total > 0)
        ax.loglog(
            centers[valid],
            dedx_plot_total[valid],
            color="black",
            linewidth=2,
            label=label,
            zorder=3,
        )
    else:
        valid = np.isfinite(dedx_plot_total) & (dedx_plot_total > 0)
        ax.loglog(
            centers[valid],
            dedx_plot_total[valid],
            color="black",
            linewidth=4,
            label=label,
            zorder=4,
        )

    ax.set_xlabel("Electron Energy ($T$; eV)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(emin, emax)
    ax.legend(loc="best")
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.show()


def _default_root_path() -> Path:
    return PROJECT_ROOT / "build" / "dna.root"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stopping power from dnaphysics ROOT output.")
    parser.add_argument("--root", type=Path, default=_default_root_path(), help="Path to dna.root")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "stopping_power.png", help="Output PNG path")
    parser.add_argument("--tree", type=str, default="step", help="Tree name in ROOT file")
    parser.add_argument("--nbins", type=int, default=80, help="Number of log-spaced energy bins")
    parser.add_argument("--emin", type=float, default=None, help="Minimum energy (eV)")
    parser.add_argument("--emax", type=float, default=None, help="Maximum energy (eV)")
    parser.add_argument(
        "--use-edep",
        action="store_true",
        help="Use totalEnergyDeposit instead of kineticEnergyDifference (LET).",
    )
    parser.add_argument(
        "--all-tracks",
        action="store_true",
        help="Include all tracks (default: primary only).",
    )
    parser.add_argument(
        "--particle",
        type=int,
        default=None,
        help="Filter by flagParticle value (e.g., 1 for electron).",
    )
    parser.add_argument(
        "--units",
        type=str,
        default="mev_cm",
        choices=("ev_nm", "mev_cm", "mev_cm2_g"),
        help="Output units for stopping power.",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="Density (g/cm^3) for mass stopping power.",
    )
    parser.add_argument(
        "--total-only",
        action="store_true",
        help="Plot only total stopping power (no per-process breakdown).",
    )
    parser.add_argument("--label", type=str, default="Cumulative", help="Legend label for total")
    args = parser.parse_args()

    plot_stopping_power(
        root_path=args.root,
        out_path=args.out,
        tree_name=args.tree,
        nbins=args.nbins,
        emin=args.emin,
        emax=args.emax,
        use_edep=args.use_edep,
        all_tracks=args.all_tracks,
        particle=args.particle,
        units=args.units,
        density=args.density,
        label=args.label,
        split_by_process=not args.total_only,
    )


if __name__ == "__main__":
    main()
