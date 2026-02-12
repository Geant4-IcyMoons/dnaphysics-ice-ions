#!/usr/bin/env python3
"""Simple depth diagnostics from dna.root.

Plots:
- Deposited energy vs z
- Mean kinetic energy of primary electrons vs z
- Mean kinetic energy of secondary electrons vs z
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
    RC_BASE_ELASTIC,
    rcparams_with_fontsize,
)
from root_utils import resolve_root_paths

# Match elastic cross-sections plot style
plt.rcParams["font.family"] = FONT_COURIER
plt.rcParams["mathtext.rm"] = FONT_COURIER
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams.update(rcparams_with_fontsize(RC_BASE_ELASTIC, FONTSIZE_24))


def resolve_path(p: str) -> str:
    path = Path(p).expanduser()
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path)
    # assume run from repo root or python_scripts
    base = Path(__file__).resolve().parent.parent
    candidate = base / path
    if candidate.exists():
        return str(candidate)
    return str(path)


def iter_step_batches(root_path: str, step_size: int):
    paths = resolve_root_paths(root_path)
    if not paths:
        raise FileNotFoundError(root_path)
    branches = [
        "z",
        "kineticEnergy",
        "totalEnergyDeposit",
        "flagParticle",
        "parentID",
    ]
    tree_spec = [f"{p}:step" for p in paths]
    for chunk in uproot.iterate(tree_spec, branches, step_size=step_size, library="np"):
        yield chunk


def unit_factor(unit: str) -> float:
    unit = unit.lower()
    if unit == "nm":
        return 1.0
    if unit == "um" or unit == "micron":
        return 1e-3
    if unit == "mm":
        return 1e-6
    if unit == "cm":
        return 1e-7
    raise ValueError(f"Unsupported z unit: {unit}")


def binned_sum(z: np.ndarray, values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    idx = np.digitize(z, bins) - 1
    valid = (idx >= 0) & (idx < len(bins) - 1)
    sums = np.bincount(idx[valid], weights=values[valid], minlength=len(bins) - 1)
    return sums


def binned_mean(z: np.ndarray, values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    idx = np.digitize(z, bins) - 1
    valid = (idx >= 0) & (idx < len(bins) - 1)
    counts = np.bincount(idx[valid], minlength=len(bins) - 1)
    sums = np.bincount(idx[valid], weights=values[valid], minlength=len(bins) - 1)
    mean = np.full(len(bins) - 1, np.nan)
    nonzero = counts > 0
    mean[nonzero] = sums[nonzero] / counts[nonzero]
    return mean


def main() -> None:
    ap = argparse.ArgumentParser(description="Depth diagnostics from dna.root")
    ap.add_argument("--root", default="build/europa_test.root", help="Path to ROOT file")
    ap.add_argument("--bins", type=int, default=100, help="Number of z bins")
    ap.add_argument("--binning", choices=["log", "linear"], default="log", help="Bin spacing (default: log)")
    ap.add_argument("--step-size", type=int, default=200_000, help="Rows per batch when streaming ROOT")
    ap.add_argument("--zmin", type=float, default=None, help="Minimum z (in selected units)")
    ap.add_argument("--zmax", type=float, default=None, help="Maximum z (in selected units)")
    ap.add_argument("--z-unit", default="cm", choices=["nm", "um", "mm", "cm"], help="z-axis unit")
    ap.add_argument("--out", default="depth_diagnostics.png", help="Output plot")
    args = ap.parse_args()

    # Determine z range (one pass if not provided)
    zmin = args.zmin
    zmax = args.zmax
    if zmin is None or zmax is None:
        zmin = np.inf
        zmax = -np.inf
        for chunk in iter_step_batches(args.root, args.step_size):
            z_nm = np.asarray(chunk["z"], dtype=float)
            if z_nm.size == 0:
                continue
            z = z_nm * unit_factor(args.z_unit)
            zmin = min(zmin, float(np.nanmin(z)))
            zmax = max(zmax, float(np.nanmax(z)))
        if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
            raise RuntimeError("Invalid z range from data")
    if args.binning == "log":
        if zmin <= 0.0:
            raise RuntimeError("Log binning requires zmin > 0. Set --zmin > 0 or use --binning linear.")
        bins = np.logspace(np.log10(zmin), np.log10(zmax), args.bins + 1)
        centers = np.sqrt(bins[:-1] * bins[1:])
    else:
        bins = np.linspace(zmin, zmax, args.bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
    widths = bins[1:] - bins[:-1]

    # Accumulators
    dep = np.zeros(args.bins, dtype=float)
    cnt_electrons = np.zeros(args.bins, dtype=float)
    sum_ke_primary = np.zeros(args.bins, dtype=float)
    cnt_primary = np.zeros(args.bins, dtype=float)
    sum_ke_secondary = np.zeros(args.bins, dtype=float)
    cnt_secondary = np.zeros(args.bins, dtype=float)

    # Stream batches and accumulate
    for chunk in iter_step_batches(args.root, args.step_size):
        z_nm = np.asarray(chunk["z"], dtype=float)
        ke = np.asarray(chunk["kineticEnergy"], dtype=float)
        dE = np.asarray(chunk["totalEnergyDeposit"], dtype=float)
        flag = np.asarray(chunk.get("flagParticle", np.full_like(z_nm, -1)), dtype=float)
        parent = np.asarray(chunk.get("parentID", np.full_like(z_nm, -1)), dtype=int)

        z = z_nm * unit_factor(args.z_unit)
        idx = np.digitize(z, bins) - 1
        valid = (idx >= 0) & (idx < args.bins)

        # Deposited energy
        np.add.at(dep, idx[valid], dE[valid])

        # All electrons (primary + secondary)
        m_e = valid & (flag == 1)
        np.add.at(cnt_electrons, idx[m_e], 1.0)

        # Primary electrons
        m = valid & (flag == 1) & (parent == 0)
        np.add.at(sum_ke_primary, idx[m], ke[m])
        np.add.at(cnt_primary, idx[m], 1.0)

        # Secondary electrons
        m = valid & (flag == 1) & (parent > 0)
        np.add.at(sum_ke_secondary, idx[m], ke[m])
        np.add.at(cnt_secondary, idx[m], 1.0)

    ke_primary = np.full(args.bins, np.nan)
    ke_secondary = np.full(args.bins, np.nan)
    mask = cnt_primary > 0
    ke_primary[mask] = sum_ke_primary[mask] / cnt_primary[mask]
    mask = cnt_secondary > 0
    ke_secondary[mask] = sum_ke_secondary[mask] / cnt_secondary[mask]

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True, constrained_layout=True)

    m0 = np.isfinite(dep) & (dep > 0)
    axes[0].bar(centers[m0], dep[m0], width=widths[m0], color="gray", alpha=0.8)
    axes[0].set_ylabel("Dep. (E; eV)", labelpad=12)
    # axes[0].set_title("Deposited energy vs depth")

    m1 = np.isfinite(ke_primary) & (ke_primary > 0)
    axes[1].bar(centers[m1], ke_primary[m1], width=widths[m1], color="lightgray", alpha=0.8)
    axes[1].set_ylabel("<Prim.> (T; eV)", labelpad=12)
    # axes[1].set_title("Primary electron KE vs depth")

    m2 = np.isfinite(ke_secondary) & (ke_secondary > 0)
    axes[2].bar(centers[m2], ke_secondary[m2], width=widths[m2], color="slategray", alpha=0.8)
    axes[2].set_ylabel("<Sec.> (T; eV)", labelpad=12)
    # axes[2].set_title("Secondary electron KE vs depth")

    avg_dep_per_e = np.full(args.bins, np.nan)
    mcnt = cnt_electrons > 0
    avg_dep_per_e[mcnt] = dep[mcnt] / cnt_electrons[mcnt]
    m3 = np.isfinite(avg_dep_per_e) & (avg_dep_per_e > 0)
    axes[3].bar(centers[m3], avg_dep_per_e[m3], width=widths[m3], color="dimgray", alpha=0.8)
    axes[3].set_ylabel("Dep/E (eV)", labelpad=12)
    axes[3].set_title("Average deposited energy per electron vs depth")

    has_pos = [np.any(m0), np.any(m1), np.any(m2), np.any(m3)]
    for ax, ok in zip(axes, has_pos):
        if args.binning == "log":
            ax.set_xscale("log")
        if ok:
            ax.set_yscale("log")
        else:
            ax.set_yscale("linear")
            ax.set_ylim(0.0, 1.0)
    axes[3].set_xlabel(f"z ({args.z_unit})")
    # constrained_layout handles spacing
    out_path = resolve_path(args.out)
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
