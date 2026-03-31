#!/usr/bin/env python3
"""Process diagnostics for example_2.root."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot


REQUIRED_STEP_BRANCHES = {
    "eventID",
    "flagParticle",
    "flagProcess",
    "processName",
    "totalEnergyDeposit",
    "kineticEnergyDifference",
    "stepLength",
}


def decode_strings(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="replace"))
        else:
            out.append(str(value))
    return out


def format_mev(value_mev: float) -> str:
    if value_mev >= 1.0:
        return f"{value_mev:g} MeV"
    return f"{value_mev:.3g} MeV"


def maybe_log_scale(axis: plt.Axes, values: np.ndarray) -> None:
    positive = values[values > 0.0]
    if positive.size < 2:
        return
    if float(np.max(positive) / np.min(positive)) >= 50.0:
        axis.set_xscale("log")


def load_config(root_file: uproot.reading.ReadOnlyDirectory) -> dict[str, str]:
    if "config" not in root_file:
        return {}
    tree = root_file["config"]
    if "key" not in tree or "value" not in tree:
        return {}
    keys = decode_strings(tree["key"].array(library="np"))
    values = decode_strings(tree["value"].array(library="np"))
    return {k: v for k, v in zip(keys, values) if k}


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / "outputs" / "example_2.root"
    default_out = script_dir / "outputs" / "example_2_process_diagnostics.png"

    parser = argparse.ArgumentParser(description="Plot process diagnostics for example_2.root")
    parser.add_argument("--root", default=str(default_root), help="Path to example_2.root")
    parser.add_argument("--out", default=str(default_out), help="Output PNG path")
    parser.add_argument("--top-processes", type=int, default=8, help="Number of named processes to keep before folding into Other")
    args = parser.parse_args()

    root_path = Path(args.root).expanduser()
    if not root_path.exists():
        raise SystemExit(f"ROOT file not found: {root_path}")

    with uproot.open(root_path) as root_file:
        if "step" not in root_file or "event" not in root_file:
            raise SystemExit("ROOT file must contain both 'step' and 'event' trees.")

        step_tree = root_file["step"]
        event_tree = root_file["event"]
        if not REQUIRED_STEP_BRANCHES.issubset(step_tree.keys()):
            missing = sorted(REQUIRED_STEP_BRANCHES - set(step_tree.keys()))
            raise SystemExit("Missing full-log step branches: " + ", ".join(missing))

        event_required = {
            "eventID",
            "primaryEnergy",
            "depositedEnergy",
            "escapedBackEnergy",
            "escapedForwardEnergy",
            "escapedLateralEnergy",
            "closureEnergy",
        }
        if not event_required.issubset(event_tree.keys()):
            missing = sorted(event_required - set(event_tree.keys()))
            raise SystemExit("Missing event branches: " + ", ".join(missing))

        step = step_tree.arrays(sorted(REQUIRED_STEP_BRANCHES), library="np")
        event = event_tree.arrays(sorted(event_required), library="np")
        config = load_config(root_file)

    event_ids = np.asarray(event["eventID"], dtype=int)
    primary_mev = np.asarray(event["primaryEnergy"], dtype=float) * 1.0e-6
    order = np.argsort(primary_mev)
    ordered_events = [int(event_ids[i]) for i in order]
    energy_by_event = {int(eid): float(energy) for eid, energy in zip(event_ids, primary_mev)}

    step_event = np.asarray(step["eventID"], dtype=int)
    electron = np.asarray(step["flagParticle"], dtype=float) == 1.0
    process_names = decode_strings(step["processName"])
    process_ids = np.asarray(step["flagProcess"], dtype=float)
    deposited_eV = np.asarray(step["totalEnergyDeposit"], dtype=float)
    kinetic_loss_eV = np.clip(np.asarray(step["kineticEnergyDifference"], dtype=float), 0.0, None)

    summaries: dict[int, dict[str, dict[str, float]]] = {}
    totals: dict[str, float] = defaultdict(float)
    for event_id in ordered_events:
        process_summary: dict[str, dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "deposited_eV": 0.0, "loss_eV": 0.0}
        )
        mask = (step_event == event_id) & electron
        for name, pid, dep, loss in zip(
            np.asarray(process_names, dtype=object)[mask],
            process_ids[mask],
            deposited_eV[mask],
            kinetic_loss_eV[mask],
        ):
            label = str(name).strip() or f"process_{int(pid)}"
            process_summary[label]["count"] += 1.0
            process_summary[label]["deposited_eV"] += float(dep)
            process_summary[label]["loss_eV"] += float(loss)
            totals[label] += 1.0
        summaries[event_id] = process_summary

    labels = sorted(totals, key=lambda key: (-totals[key], key))
    if len(labels) > args.top_processes:
        labels = labels[: args.top_processes] + ["Other"]

    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(ordered_events)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax_count, ax_dep, ax_loss, ax_budget = axes.ravel()

    y = np.arange(len(labels))
    width = 0.8 / max(len(ordered_events), 1)
    offsets = (np.arange(len(ordered_events)) - 0.5 * (len(ordered_events) - 1)) * width

    count_all: list[float] = []
    dep_all: list[float] = []
    loss_all: list[float] = []

    for idx, event_id in enumerate(ordered_events):
        process_summary = summaries[event_id]
        counts: list[float] = []
        deposited_mev: list[float] = []
        lost_mev: list[float] = []
        used: set[str] = set()
        for label in labels:
            if label == "Other":
                count_value = 0.0
                dep_value = 0.0
                loss_value = 0.0
                for name, stats in process_summary.items():
                    if name in used:
                        continue
                    count_value += float(stats["count"])
                    dep_value += float(stats["deposited_eV"])
                    loss_value += float(stats["loss_eV"])
            else:
                used.add(label)
                stats = process_summary.get(label, {})
                count_value = float(stats.get("count", 0.0))
                dep_value = float(stats.get("deposited_eV", 0.0))
                loss_value = float(stats.get("loss_eV", 0.0))
            counts.append(count_value)
            deposited_mev.append(dep_value * 1.0e-6)
            lost_mev.append(loss_value * 1.0e-6)

        count_arr = np.asarray(counts, dtype=float)
        dep_arr = np.asarray(deposited_mev, dtype=float)
        loss_arr = np.asarray(lost_mev, dtype=float)
        count_all.extend(count_arr.tolist())
        dep_all.extend(dep_arr.tolist())
        loss_all.extend(loss_arr.tolist())

        label = format_mev(energy_by_event[event_id])
        ax_count.barh(y + offsets[idx], count_arr, height=width * 0.95, color=colors[idx], label=label)
        ax_dep.barh(y + offsets[idx], dep_arr, height=width * 0.95, color=colors[idx], label=label)
        ax_loss.barh(y + offsets[idx], loss_arr, height=width * 0.95, color=colors[idx], label=label)

    for axis in (ax_count, ax_dep, ax_loss):
        axis.set_yticks(y)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.grid(True, axis="x", alpha=0.3)

    ax_count.set_title("Electron Step Counts By Process")
    ax_count.set_xlabel("Steps")
    maybe_log_scale(ax_count, np.asarray(count_all, dtype=float))

    ax_dep.set_title("Electron Deposited Energy By Process")
    ax_dep.set_xlabel("Deposited energy (MeV)")
    maybe_log_scale(ax_dep, np.asarray(dep_all, dtype=float))

    ax_loss.set_title("Electron Kinetic-Energy Loss By Process")
    ax_loss.set_xlabel("Energy loss (MeV)")
    maybe_log_scale(ax_loss, np.asarray(loss_all, dtype=float))

    x = np.arange(len(ordered_events))
    budget_categories = [
        ("Deposited", "depositedEnergy", "#2ca02c"),
        ("Backscatter", "escapedBackEnergy", "#d62728"),
        ("Lateral", "escapedLateralEnergy", "#ff7f0e"),
        ("Forward", "escapedForwardEnergy", "#1f77b4"),
    ]
    bottom = np.zeros(len(ordered_events), dtype=float)
    for title, branch, color in budget_categories:
        values = []
        for event_id in ordered_events:
            idx = int(np.where(event_ids == event_id)[0][0])
            primary = float(event["primaryEnergy"][idx])
            value = float(event[branch][idx]) / primary if primary > 0.0 else 0.0
            values.append(value)
        values_arr = np.asarray(values, dtype=float)
        ax_budget.bar(x, values_arr, bottom=bottom, color=color, label=title)
        bottom += values_arr

    closure = []
    for event_id in ordered_events:
        idx = int(np.where(event_ids == event_id)[0][0])
        primary = float(event["primaryEnergy"][idx])
        value = float(event["closureEnergy"][idx]) / primary if primary > 0.0 else 0.0
        closure.append(value)
    closure_arr = np.asarray(closure, dtype=float)
    if np.any(np.abs(closure_arr) > 0.0):
        ax_budget.bar(
            x,
            closure_arr,
            bottom=bottom,
            color="black",
            alpha=0.25,
            hatch="//",
            label="Closure",
        )

    ax_budget.set_xticks(x)
    ax_budget.set_xticklabels([format_mev(energy_by_event[event_id]) for event_id in ordered_events])
    ax_budget.set_ylim(0.0, 1.05)
    ax_budget.set_ylabel("Fraction of primary energy")
    ax_budget.set_title("Event Energy Budget")
    ax_budget.grid(True, axis="y", alpha=0.3)

    physics_label = config.get("DNA_PHYSICS", "unknown")
    log_mode = config.get("log_mode", "full")
    fig.suptitle(f"Example 2: Process diagnostics ({physics_label}, {log_mode})", fontsize=14)

    handles, legend_labels = ax_budget.get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="lower center", ncol=max(len(legend_labels), 1))

    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
