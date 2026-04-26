#!/usr/bin/env python3
"""Depth diagnostics for example_1.root."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot


def binned_sum(z_mm: np.ndarray, weights: np.ndarray, bins_mm: np.ndarray) -> np.ndarray:
    idx = np.digitize(z_mm, bins_mm) - 1
    valid = (idx >= 0) & (idx < len(bins_mm) - 1)
    return np.bincount(idx[valid], weights=weights[valid], minlength=len(bins_mm) - 1)


def binned_mean(z_mm: np.ndarray, values: np.ndarray, bins_mm: np.ndarray) -> np.ndarray:
    idx = np.digitize(z_mm, bins_mm) - 1
    valid = (idx >= 0) & (idx < len(bins_mm) - 1)
    sums = np.bincount(idx[valid], weights=values[valid], minlength=len(bins_mm) - 1)
    counts = np.bincount(idx[valid], minlength=len(bins_mm) - 1)
    mean = np.full(len(bins_mm) - 1, np.nan)
    good = counts > 0
    mean[good] = sums[good] / counts[good]
    return mean


def format_mev(value_mev: float) -> str:
    if value_mev >= 1.0:
        return f"{value_mev:g} MeV"
    return f"{value_mev:.3g} MeV"


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / "outputs" / "example_1.root"
    default_out = script_dir / "outputs" / "example_1_depth_diagnostics.png"

    parser = argparse.ArgumentParser(description="Plot depth diagnostics for example_1.root")
    parser.add_argument("--root", default=str(default_root), help="Path to example_1.root")
    parser.add_argument("--out", default=str(default_out), help="Output PNG path")
    parser.add_argument("--bins", type=int, default=150, help="Number of depth bins")
    parser.add_argument(
        "--z-max-mm",
        type=float,
        default=0.0,
        help="Manual maximum depth in mm; default infers from the ROOT file",
    )
    args = parser.parse_args()

    root_path = Path(args.root).expanduser()
    if not root_path.exists():
        raise SystemExit(f"ROOT file not found: {root_path}")

    with uproot.open(root_path) as root_file:
        if "step" not in root_file or "event" not in root_file:
            raise SystemExit("ROOT file must contain both 'step' and 'event' trees.")

        step_tree = root_file["step"]
        event_tree = root_file["event"]
        step_required = {"eventID", "z", "flagParticle", "parentID", "kineticEnergy", "totalEnergyDeposit"}
        event_required = {"eventID", "primaryEnergy"}
        if not step_required.issubset(step_tree.keys()):
            missing = sorted(step_required - set(step_tree.keys()))
            raise SystemExit("Missing step branches: " + ", ".join(missing))
        if not event_required.issubset(event_tree.keys()):
            missing = sorted(event_required - set(event_tree.keys()))
            raise SystemExit("Missing event branches: " + ", ".join(missing))

        step = step_tree.arrays(sorted(step_required), library="np")
        event = event_tree.arrays(sorted(event_required), library="np")

    event_ids = np.asarray(event["eventID"], dtype=int)
    primary_mev = np.asarray(event["primaryEnergy"], dtype=float) * 1.0e-6
    energy_by_event = {int(eid): float(energy) for eid, energy in zip(event_ids, primary_mev)}
    ordered_events = [int(eid) for eid in event_ids[np.argsort(primary_mev)]]

    z_mm_all = np.asarray(step["z"], dtype=float) * 1.0e-6
    z_max_mm = args.z_max_mm if args.z_max_mm > 0.0 else float(np.max(z_mm_all))
    z_max_mm = max(z_max_mm, 1.0e-6)
    bins_mm = np.linspace(0.0, z_max_mm, args.bins + 1)
    centers_mm = 0.5 * (bins_mm[:-1] + bins_mm[1:])

    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(ordered_events)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    ax_electron_dep, ax_total_dep, ax_ke, ax_cumulative = axes.ravel()

    step_event = np.asarray(step["eventID"], dtype=int)
    z_mm = np.asarray(step["z"], dtype=float) * 1.0e-6
    deposited_mev = np.asarray(step["totalEnergyDeposit"], dtype=float) * 1.0e-6
    flag_particle = np.asarray(step["flagParticle"], dtype=float)
    parent_id = np.asarray(step["parentID"], dtype=int)
    kinetic_mev = np.asarray(step["kineticEnergy"], dtype=float) * 1.0e-6

    for color, event_id in zip(colors, ordered_events):
        event_mask = step_event == event_id
        in_slab = event_mask & (z_mm >= 0.0)
        electron = flag_particle == 1.0
        primary_electron = electron & (parent_id == 0)
        secondary_electron = electron & (parent_id > 0)

        total_profile = binned_sum(z_mm[in_slab], deposited_mev[in_slab], bins_mm)
        electron_profile = binned_sum(
            z_mm[in_slab & electron],
            deposited_mev[in_slab & electron],
            bins_mm,
        )
        primary_ke_profile = binned_mean(
            z_mm[in_slab & primary_electron],
            kinetic_mev[in_slab & primary_electron],
            bins_mm,
        )
        secondary_ke_profile = binned_mean(
            z_mm[in_slab & secondary_electron],
            kinetic_mev[in_slab & secondary_electron],
            bins_mm,
        )
        cumulative = np.cumsum(total_profile)
        if cumulative.size and cumulative[-1] > 0.0:
            cumulative /= cumulative[-1]

        label = format_mev(energy_by_event[event_id])
        ax_electron_dep.plot(centers_mm, electron_profile, color=color, label=label)
        ax_total_dep.plot(centers_mm, total_profile, color=color, label=label)
        ax_ke.plot(centers_mm, primary_ke_profile, color=color, label=f"{label} primary")
        if np.any(np.isfinite(secondary_ke_profile)):
            ax_ke.plot(centers_mm, secondary_ke_profile, color=color, linestyle="--", label=f"{label} secondary")
        ax_cumulative.plot(centers_mm, cumulative, color=color, label=label)

    ax_electron_dep.set_title("Electron Deposited Energy")
    ax_electron_dep.set_ylabel("Deposited energy per bin (MeV)")
    ax_electron_dep.grid(True, alpha=0.3)

    ax_total_dep.set_title("All-Particle Deposited Energy")
    ax_total_dep.set_ylabel("Deposited energy per bin (MeV)")
    ax_total_dep.grid(True, alpha=0.3)

    ax_ke.set_title("Electron Kinetic Energy")
    ax_ke.set_ylabel("Mean kinetic energy (MeV)")
    ax_ke.grid(True, alpha=0.3)

    ax_cumulative.set_title("Cumulative Deposited Fraction")
    ax_cumulative.set_ylabel("Fraction")
    ax_cumulative.set_ylim(0.0, 1.05)
    ax_cumulative.grid(True, alpha=0.3)

    for axis in axes.ravel():
        axis.set_xlabel("Depth z (mm)")

    handles, labels = ax_ke.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2)

    fig.suptitle("Example 1: Depth diagnostics by event energy", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
