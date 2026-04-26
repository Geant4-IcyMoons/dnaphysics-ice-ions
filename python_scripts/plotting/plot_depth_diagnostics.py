#!/usr/bin/env python3
"""Simple depth diagnostics from dna.root.

Plots:
- Deposited energy vs z
- Mean kinetic energy of primary electrons vs z
- Mean kinetic energy of secondary electrons vs z
- Cumulative deposited energy fraction vs z (percent of initial primary energy)
- Event-energy budget fractions (deposited, escaped back/lateral/forward)
- 3D voxelized deposited-energy map (green->red, red=high deposition)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
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
        "x",
        "y",
        "z",
        "kineticEnergy",
        "totalEnergyDeposit",
        "flagParticle",
        "parentID",
    ]
    tree_spec = [f"{p}:step" for p in paths]
    for chunk in uproot.iterate(tree_spec, branches, step_size=step_size, library="np"):
        yield chunk


def iter_event_batches(root_path: str, step_size: int):
    paths = resolve_root_paths(root_path)
    if not paths:
        raise FileNotFoundError(root_path)
    required = [
        "primaryEnergy",
        "depositedEnergy",
        "escapedBackEnergy",
        "escapedForwardEnergy",
        "escapedLateralEnergy",
    ]
    tree_spec: list[str] = []
    for p in paths:
        with uproot.open(p) as rootf:
            if "event" not in rootf:
                continue
            keys = set(rootf["event"].keys())
            if not all(name in keys for name in required):
                continue
            tree_spec.append(f"{p}:event")
    if not tree_spec:
        raise RuntimeError(
            "No ROOT file contains event tree with required branches: "
            + ", ".join(required)
        )
    for chunk in uproot.iterate(tree_spec, required, step_size=step_size, library="np"):
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


def label_every_step(step: float):
    def _formatter(val: float, _pos: int) -> str:
        if np.isclose((val / step) - round(val / step), 0.0, atol=1e-9):
            return f"{val:.1f}"
        return ""

    return FuncFormatter(_formatter)


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


def _update_first_two_steps(
    state: dict[int, tuple[int, float, float, float] | None],
    event_id: int,
    step_id: int,
    x: float,
    y: float,
    z: float,
) -> None:
    key_first = event_id * 2
    key_second = event_id * 2 + 1
    first = state.get(key_first)
    second = state.get(key_second)
    current = (step_id, x, y, z)
    if first is None:
        state[key_first] = current
        return
    if step_id < first[0]:
        state[key_second] = first
        state[key_first] = current
        return
    if step_id == first[0]:
        return
    if second is None or step_id < second[0]:
        state[key_second] = current


def extract_initial_angle_arrays(
    root_path: str,
    step_size: int,
    deposition_epsilon_frac: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    paths = resolve_root_paths(root_path)
    if not paths:
        raise FileNotFoundError(root_path)

    all_theta: list[float] = []
    incomplete_theta: list[float] = []
    n_events_used = 0

    for p in paths:
        with uproot.open(p) as rootf:
            if "event" not in rootf or "step" not in rootf:
                continue
            et = rootf["event"]
            st = rootf["step"]
            event_required = ["eventID", "primaryEnergy", "depositedEnergy"]
            step_required = ["eventID", "parentID", "flagParticle", "x", "y", "z"]
            has_track_id = "trackID" in st.keys()
            has_step_id = "stepID" in st.keys()
            if has_track_id:
                step_required.append("trackID")
            if has_step_id:
                step_required.append("stepID")
            if not all(name in et.keys() for name in event_required):
                continue
            if not all(name in st.keys() for name in step_required):
                continue

            earr = et.arrays(event_required, library="np")
            event_budget = {
                int(eid): (float(pe), float(de))
                for eid, pe, de in zip(
                    earr["eventID"], earr["primaryEnergy"], earr["depositedEnergy"]
                )
            }

            step_state: dict[int, tuple[int, float, float, float] | None] = {}
            step_state_seq: dict[int, list[tuple[float, float, float]]] = {}
            for chunk in st.iterate(step_required, library="np", step_size=step_size):
                m_primary = (
                    (np.asarray(chunk["parentID"], dtype=np.int64) == 0)
                    & (np.asarray(chunk["flagParticle"], dtype=np.int64) == 1)
                )
                if has_track_id:
                    m_primary &= np.asarray(chunk["trackID"], dtype=np.int64) == 1
                if not np.any(m_primary):
                    continue

                eids = np.asarray(chunk["eventID"], dtype=np.int64)[m_primary]
                xs = np.asarray(chunk["x"], dtype=np.float64)[m_primary]
                ys = np.asarray(chunk["y"], dtype=np.float64)[m_primary]
                zs = np.asarray(chunk["z"], dtype=np.float64)[m_primary]
                if has_step_id:
                    sids = np.asarray(chunk["stepID"], dtype=np.int64)[m_primary]
                    for eid, sid, x, y, z in zip(eids, sids, xs, ys, zs):
                        _update_first_two_steps(
                            step_state,
                            int(eid),
                            int(sid),
                            float(x),
                            float(y),
                            float(z),
                        )
                else:
                    for eid, x, y, z in zip(eids, xs, ys, zs):
                        eid_i = int(eid)
                        pts = step_state_seq.get(eid_i)
                        if pts is None:
                            step_state_seq[eid_i] = [(float(x), float(y), float(z))]
                        elif len(pts) < 2:
                            pts.append((float(x), float(y), float(z)))

            for eid, (primary_e, deposited_e) in event_budget.items():
                if has_step_id:
                    first = step_state.get(eid * 2)
                    second = step_state.get(eid * 2 + 1)
                    if first is None or second is None:
                        continue
                    x0, y0, z0 = first[1], first[2], first[3]
                    x1, y1, z1 = second[1], second[2], second[3]
                else:
                    pts = step_state_seq.get(eid)
                    if pts is None or len(pts) < 2:
                        continue
                    x0, y0, z0 = pts[0]
                    x1, y1, z1 = pts[1]

                dx = x1 - x0
                dy = y1 - y0
                dz = z1 - z0
                norm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                if norm <= 0.0:
                    continue
                uz = dz / norm
                theta = float(np.degrees(np.arccos(np.clip(uz, -1.0, 1.0))))
                all_theta.append(theta)
                n_events_used += 1

                if primary_e > 0.0:
                    missing_frac = max(0.0, (primary_e - deposited_e) / primary_e)
                    if missing_frac > deposition_epsilon_frac:
                        incomplete_theta.append(theta)

    return (
        np.asarray(all_theta, dtype=np.float64),
        np.asarray(incomplete_theta, dtype=np.float64),
        n_events_used,
    )


def _safe_edges(vmin: float, vmax: float, nbins: int) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise RuntimeError("Invalid voxel range for 3D map.")
    if vmax <= vmin:
        pad = max(abs(vmin) * 1e-6, 1e-6)
        vmin -= pad
        vmax += pad
    return np.linspace(vmin, vmax, int(nbins) + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Depth diagnostics from dna.root")
    ap.add_argument("--root", default="build/europa_test_e2.root", help="Path to ROOT file")
    ap.add_argument("--bins", type=int, default=100, help="Number of z bins")
    ap.add_argument("--binning", choices=["log", "linear"], default="log", help="Bin spacing (default: log)")
    ap.add_argument("--step-size", type=int, default=200_000, help="Rows per batch when streaming ROOT")
    ap.add_argument("--zmin", type=float, default=None, help="Minimum z (in selected units)")
    ap.add_argument("--zmax", type=float, default=None, help="Maximum z (in selected units)")
    ap.add_argument("--z-unit", default="cm", choices=["nm", "um", "mm", "cm"], help="z-axis unit")
    ap.add_argument("--out", default="depth_diagnostics.png", help="Output plot")
    ap.add_argument(
        "--out-cumulative",
        default="depth_cumulative_deposited_percent.png",
        help="Output cumulative deposited-energy fraction plot",
    )
    ap.add_argument(
        "--out-fractions",
        default="depth_energy_budget_fractions.png",
        help="Output deposited/escaped fractions bar plot",
    )
    ap.add_argument(
        "--out-initial-angles",
        default="depth_initial_angles.png",
        help="Output initial-angle diagnostics plot",
    )
    ap.add_argument(
        "--deposition-epsilon-frac",
        type=float,
        default=1e-6,
        help="Fractional tolerance on missing energy for classifying incomplete deposition",
    )
    ap.add_argument(
        "--out-3d",
        default="depth_deposition_3d.png",
        help="Output 3D deposited-energy voxel map",
    )
    ap.add_argument(
        "--xyz-bins",
        type=int,
        default=30,
        help="Number of bins per axis for 3D voxelization",
    )
    args = ap.parse_args()

    # Determine z range (one pass if not provided) + collect xyz range for 3D map.
    zmin = args.zmin
    zmax = args.zmax
    need_z_pass = zmin is None or zmax is None
    unit_fac = unit_factor(args.z_unit)
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    zmin_dep = np.inf
    zmax_dep = -np.inf
    if need_z_pass or args.out_3d:
        zmin = np.inf
        zmax = -np.inf
        for chunk in iter_step_batches(args.root, args.step_size):
            x_nm = np.asarray(chunk["x"], dtype=float)
            y_nm = np.asarray(chunk["y"], dtype=float)
            z_nm = np.asarray(chunk["z"], dtype=float)
            dE = np.asarray(chunk["totalEnergyDeposit"], dtype=float)
            if z_nm.size == 0:
                continue
            z = z_nm * unit_fac
            zmin = min(zmin, float(np.nanmin(z)))
            zmax = max(zmax, float(np.nanmax(z)))
            if args.out_3d:
                x = x_nm * unit_fac
                y = y_nm * unit_fac
                m_dep = (
                    np.isfinite(x)
                    & np.isfinite(y)
                    & np.isfinite(z)
                    & np.isfinite(dE)
                    & (dE > 0.0)
                )
                if np.any(m_dep):
                    xmin = min(xmin, float(np.nanmin(x[m_dep])))
                    xmax = max(xmax, float(np.nanmax(x[m_dep])))
                    ymin = min(ymin, float(np.nanmin(y[m_dep])))
                    ymax = max(ymax, float(np.nanmax(y[m_dep])))
                    zmin_dep = min(zmin_dep, float(np.nanmin(z[m_dep])))
                    zmax_dep = max(zmax_dep, float(np.nanmax(z[m_dep])))
        if need_z_pass and (not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin):
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

        z = z_nm * unit_fac
        idx = np.digitize(z, bins) - 1
        valid = (idx >= 0) & (idx < args.bins)

        # Deposited energy
        np.add.at(dep, idx[valid], dE[valid])

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

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True, constrained_layout=True)

    m0 = np.isfinite(dep) & (dep > 0)
    axes[0].bar(centers[m0], dep[m0], width=widths[m0], color="gray", alpha=0.8)
    axes[0].set_ylabel("Dep. ($E$; eV)", labelpad=12)
    # axes[0].set_title("Deposited energy vs depth")

    m1 = np.isfinite(ke_primary) & (ke_primary > 0)
    axes[1].bar(centers[m1], ke_primary[m1], width=widths[m1], color="lightgray", alpha=0.8)
    axes[1].set_ylabel("<Prim.> ($T$; eV)", labelpad=12)
    # axes[1].set_title("Primary electron KE vs depth")

    m2 = np.isfinite(ke_secondary) & (ke_secondary > 0)
    axes[2].bar(centers[m2], ke_secondary[m2], width=widths[m2], color="slategray", alpha=0.8)
    axes[2].set_ylabel("<Sec.> ($T$; eV)", labelpad=12)
    # axes[2].set_title("Secondary electron KE vs depth")

    has_pos = [np.any(m0), np.any(m1), np.any(m2)]
    for ax, ok in zip(axes, has_pos):
        if args.binning == "log":
            ax.set_xscale("log")
        if ok:
            ax.set_yscale("log")
        else:
            ax.set_yscale("linear")
            ax.set_ylim(0.0, 1.0)
    axes[2].set_xlabel(f"Depth ({args.z_unit})")
    # constrained_layout handles spacing
    out_path = resolve_path(args.out)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")

    # Event-level energy budget needed for the cumulative and fraction plots.
    sum_primary = 0.0
    sum_deposited = 0.0
    sum_escaped_back = 0.0
    sum_escaped_forward = 0.0
    sum_escaped_lateral = 0.0
    for chunk in iter_event_batches(args.root, args.step_size):
        sum_primary += float(np.sum(np.asarray(chunk["primaryEnergy"], dtype=np.float64)))
        sum_deposited += float(np.sum(np.asarray(chunk["depositedEnergy"], dtype=np.float64)))
        sum_escaped_back += float(np.sum(np.asarray(chunk["escapedBackEnergy"], dtype=np.float64)))
        sum_escaped_forward += float(np.sum(np.asarray(chunk["escapedForwardEnergy"], dtype=np.float64)))
        sum_escaped_lateral += float(np.sum(np.asarray(chunk["escapedLateralEnergy"], dtype=np.float64)))

    if sum_primary <= 0.0:
        raise RuntimeError("Total primary energy is non-positive; cannot build fraction plots.")

    # Cumulative deposited-energy percentage vs depth from step-level deposition profile.
    dep_cumulative_percent = 100.0 * np.cumsum(dep) / sum_primary
    fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax2.fill_between(
        centers,
        dep_cumulative_percent,
        0.0,
        color="lightgray",
        alpha=0.9,
    )
    ax2.plot(centers, dep_cumulative_percent, color="black", linewidth=2.2)
    if args.binning == "log":
        ax2.set_xscale("log")
    ax2.set_ylim(0.0, max(100.0, float(np.nanmax(dep_cumulative_percent)) * 1.05))
    ax2_right = ax2.twinx()
    ax2_right.set_ylim(ax2.get_ylim())
    ax2_right.set_yticks(ax2.get_yticks())
    ax2_right.yaxis.set_major_formatter(ax2.yaxis.get_major_formatter())
    ax2.set_ylabel("Cum. dep. energy (%)", labelpad=12)
    ax2.set_xlabel(f"Depth ({args.z_unit})")
    # ax2.grid(True, alpha=0.25)
    out_cum = resolve_path(args.out_cumulative)
    fig2.savefig(out_cum, bbox_inches="tight")
    print(f"Wrote {out_cum}")

    # Deposited/escaped fractions of initial kinetic energy.
    labels = ["Dep.", "Esc. (B)", "(L)", "(F)"]
    frac = np.asarray(
        [
            sum_deposited / sum_primary,
            sum_escaped_back / sum_primary,
            sum_escaped_lateral / sum_primary,
            sum_escaped_forward / sum_primary,
        ],
        dtype=float,
    )
    fig3, ax3 = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = ["black", "dimgray", "gray", "lightgray"]
    ax3.bar(labels, frac, color=colors, alpha=0.9)
    ax3.set_ylabel("E$_{\\rm{init}}$ (%)", labelpad=12)
    ax3.tick_params(axis="x", which="both", length=0)
    ax3.set_ylim(0.0, max(1.0, float(np.nanmax(frac)) * 1.15))
    tick_step = 0.1
    label_step = 0.2
    y_max = ax3.get_ylim()[1]
    y_ticks = np.arange(0.0, y_max + 0.5 * tick_step, tick_step)
    ax3.set_yticks(y_ticks)
    ax3.yaxis.set_major_formatter(label_every_step(label_step))
    # ax3.grid(True, axis="y", alpha=0.25)
    out_frac = resolve_path(args.out_fractions)
    fig3.savefig(out_frac, bbox_inches="tight")
    print(f"Wrote {out_frac}")

    # 3D deposited-energy voxel map (green->red, red means higher deposition).
    if args.out_3d:
        if not (
            np.isfinite(xmin)
            and np.isfinite(xmax)
            and np.isfinite(ymin)
            and np.isfinite(ymax)
            and np.isfinite(zmin_dep)
            and np.isfinite(zmax_dep)
        ):
            print("3D map skipped: no deposited-energy points found.")
        else:
            x_edges = _safe_edges(xmin, xmax, args.xyz_bins)
            y_edges = _safe_edges(ymin, ymax, args.xyz_bins)
            z_edges = _safe_edges(zmin_dep, zmax_dep, args.xyz_bins)
            dep3d = np.zeros((args.xyz_bins, args.xyz_bins, args.xyz_bins), dtype=np.float64)

            for chunk in iter_step_batches(args.root, args.step_size):
                x = np.asarray(chunk["x"], dtype=np.float64) * unit_fac
                y = np.asarray(chunk["y"], dtype=np.float64) * unit_fac
                z = np.asarray(chunk["z"], dtype=np.float64) * unit_fac
                dE = np.asarray(chunk["totalEnergyDeposit"], dtype=np.float64)
                m = (
                    np.isfinite(x)
                    & np.isfinite(y)
                    & np.isfinite(z)
                    & np.isfinite(dE)
                    & (dE > 0.0)
                )
                if not np.any(m):
                    continue
                coords = np.column_stack((x[m], y[m], z[m]))
                hist, _ = np.histogramdd(coords, bins=(x_edges, y_edges, z_edges), weights=dE[m])
                dep3d += hist

            vals = dep3d.ravel()
            mask = vals > 0.0
            if not np.any(mask):
                print("3D map skipped: no deposited-energy voxels after binning.")
            else:
                x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
                y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
                z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
                Xc, Yc, Zc = np.meshgrid(x_cent, y_cent, z_cent, indexing="ij")

                xv = Xc.ravel()[mask]
                yv = Yc.ravel()[mask]
                zv = Zc.ravel()[mask]
                vv = vals[mask]

                # Requested palette: green (low) -> red (high).
                cmap_gr = LinearSegmentedColormap.from_list(
                    "green_red_deposition",
                    ["#1a9850", "#d73027"],
                    N=256,
                )
                vmin = float(np.min(vv))
                vmax = float(np.max(vv))
                if vmax > vmin and vmin > 0.0:
                    norm = LogNorm(vmin=vmin, vmax=vmax)
                else:
                    norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-12))
                color_vals = norm(vv)
                rgba = cmap_gr(color_vals)
                # Alpha ramp to keep inner structure visible.
                rgba[:, 3] = 0.10 + 0.85 * np.clip(color_vals, 0.0, 1.0) ** 0.7

                fig5 = plt.figure(figsize=(10, 8), constrained_layout=True)
                ax6 = fig5.add_subplot(111, projection="3d")
                ax6.scatter(
                    xv,
                    yv,
                    zv,
                    c=rgba,
                    s=18,
                    marker="s",
                    linewidths=0.0,
                    depthshade=False,
                )
                ax6.set_xlabel(f"x ({args.z_unit})")
                ax6.set_ylabel(f"y ({args.z_unit})")
                ax6.set_zlabel(f"z ({args.z_unit})")
                ax6.set_xlim(x_edges[0], x_edges[-1])
                ax6.set_ylim(y_edges[0], y_edges[-1])
                ax6.set_zlim(z_edges[0], z_edges[-1])
                ax6.set_title("3D deposited energy map")
                sm = cm.ScalarMappable(norm=norm, cmap=cmap_gr)
                sm.set_array(vv)
                cbar = fig5.colorbar(sm, ax=ax6, pad=0.08, shrink=0.78)
                cbar.set_label("Deposited energy per voxel (eV)")
                out_3d = resolve_path(args.out_3d)
                fig5.savefig(out_3d, bbox_inches="tight")
                print(f"Wrote {out_3d}")

    # Initial-angle diagnostics: all events vs incomplete-deposition subset.
    theta_all, theta_incomplete, n_events_used = extract_initial_angle_arrays(
        args.root,
        args.step_size,
        float(args.deposition_epsilon_frac),
    )
    if theta_all.size > 0:
        fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
        bins_theta = np.linspace(0.0, 90.0, 46)
        ax4.hist(theta_all, bins=bins_theta, histtype="stepfilled", color="lightgray", alpha=0.9)
        ax4.set_xlim(0.0, 90.0)
        ax4.set_xlabel(r"$\theta_{\rm init}$ (deg)")
        ax4.set_ylabel("Counts")
        ax4.set_title(f"All events (N={n_events_used})")

        ax5.hist(theta_incomplete, bins=bins_theta, histtype="stepfilled", color="dimgray", alpha=0.9)
        ax5.set_xlim(0.0, 90.0)
        ax5.set_xlabel(r"$\theta_{\rm init}$ (deg)")
        ax5.set_title(
            f"Incomplete deposition (N={theta_incomplete.size}, eps={args.deposition_epsilon_frac:g})"
        )

        out_angles = resolve_path(args.out_initial_angles)
        fig4.savefig(out_angles, bbox_inches="tight")
        print(f"Wrote {out_angles}")
    else:
        print("No initial-angle diagnostics produced (missing required step/event branches).")

    plt.show()


if __name__ == "__main__":
    main()
