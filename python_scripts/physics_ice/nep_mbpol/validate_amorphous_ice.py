#!/usr/bin/env python3
"""Validate completed NEP-MB-pol candidate-LDA melt--quench replicas.

The report separates three questions that must not be conflated: whether the
trajectories completed, whether the final cells are amorphous, and whether the
result is consistent with low-density amorphous ice (LDA).  It writes two PNG
figures plus machine-readable JSON and CSV diagnostics; it never promotes a
candidate structure into the collision registry.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
PHYSICS_ICE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PHYSICS_ICE))

from constants import (  # noqa: E402
    AASTEX_FULL_WIDTH_IN,
    FONT_COURIER,
    PAPER_FONTSIZE,
    RC_BASE_STANDARD,
    THREE_PANEL_ROW_HEIGHT_IN,
    rcparams_with_fontsize,
)
from bca.config import (  # noqa: E402
    AVOGADRO_MOL_MINUS_ONE,
    WATER_MOLAR_MASS_G_MOL,
)
from bca.structure import file_sha256, iter_xyz_frames  # noqa: E402
from validate_hexagonal_ice import (  # noqa: E402
    THERMO_COLUMNS,
    _orthorhombic_lengths,
    _read_thermo,
    _wrapped_positions,
    chill_plus,
    hac_linear_trend,
)  # noqa: E402


SEEDS = (1000, 2000, 3000)
THERMO_INTERVAL_NS = 0.002
FINAL_HOLD_RECORDS = 500
TARGET_TEMPERATURE_K = 80.0
TARGET_PRESSURE_GPA = 0.0001
EXPERIMENTAL_LDA_DENSITY_G_CM3 = 0.94
EXPERIMENTAL_LDA_DENSITY_HALF_RANGE_G_CM3 = 0.01
PUBLISHED_QTIP4PF_LDA_DENSITY_G_CM3 = 0.98
RDF_BIN_WIDTH_A = 0.02
RDF_PLOT_MAXIMUM_A = 8.0
STRUCTURE_FACTOR_Q_A_INV = np.linspace(0.5, 8.0, 751)
LDA_FIRST_PEAK_RANGE_A_INV = (1.7, 1.8)
CHILL_NEIGHBOR_CUTOFF_A = 3.5
STAGES = (
    ("350 K hold", 0.0, 1.0),
    ("350--240 K ramp", 1.0, 2.0),
    ("240 K hold", 2.0, 3.0),
    ("240--80 K ramp", 3.0, 19.0),
    ("80 K hold", 19.0, 20.0),
)
REFERENCES = (
    {
        "purpose": "adapted cooling segment and q-TIP4P/F LDA comparator",
        "citation": (
            "Eltareb, Lopez, and Giovambattista, Communications Chemistry "
            "7, 36 (2024)"
        ),
        "doi": "10.1038/s42004-024-01117-2",
    },
    {
        "purpose": "experimental LDA/HDA partial pair structure",
        "citation": (
            "Finney et al., Physical Review Letters 88, 225503 (2002)"
        ),
        "doi": "10.1103/PhysRevLett.88.225503",
    },
    {
        "purpose": "experimental amorphous-ice density summary",
        "citation": "Loerting et al., PCCP 13, 8783 (2011)",
        "doi": "10.1039/C0CP02600J",
    },
    {
        "purpose": "CHILL+ local ice-polymorph definition",
        "citation": "Nguyen and Molinero, J. Phys. Chem. B 119, 9369 (2015)",
        "doi": "10.1021/jp510289t",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=HERE / "runs",
        help="Directory containing amorphous_seed1000/2000/3000.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE.parent
        / "ice_structures"
        / "rejected_candidates"
        / "amorphous_lda_80K_candidate"
        / "validation",
        help="Directory for compact validation products.",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable tqdm progress output."
    )
    return parser.parse_args()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o644)


def _json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _density_from_thermo(
    thermo: NDArray[np.float64], water_molecules: int
) -> NDArray[np.float64]:
    volume_a3 = np.prod(thermo[:, 9:12], axis=1)
    mass_g = (
        water_molecules
        * WATER_MOLAR_MASS_G_MOL
        / AVOGADRO_MOL_MINUS_ONE
    )
    return mass_g / (volume_a3 * 1.0e-24)


def _mean_normal_pressure(thermo: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.mean(thermo[:, 3:6], axis=1)


def _summary(values: NDArray[np.float64]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _stage_summaries(
    thermo: NDArray[np.float64],
    density: NDArray[np.float64],
    water_molecules: int,
) -> list[dict[str, object]]:
    time_ns = (np.arange(thermo.shape[0], dtype=np.float64) + 1.0) * THERMO_INTERVAL_NS
    pressure = _mean_normal_pressure(thermo)
    records: list[dict[str, object]] = []
    for label, start_ns, stop_ns in STAGES:
        mask = (time_ns > start_ns) & (time_ns <= stop_ns + 1.0e-12)
        records.append(
            {
                "stage": label,
                "start_ns": start_ns,
                "stop_ns": stop_ns,
                "records": int(np.count_nonzero(mask)),
                "temperature_k": _summary(thermo[mask, 0]),
                "density_g_cm3": _summary(density[mask]),
                "mean_normal_pressure_gpa": _summary(pressure[mask]),
                "potential_energy_ev_per_water": _summary(
                    thermo[mask, 2] / water_molecules
                ),
            }
        )
    return records


def _radial_distribution_and_structure_factor(
    oxygen: NDArray[np.float64], lengths: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return O--O g(r) and a Lorch-windowed oxygen S(q) diagnostic."""

    maximum_r = float(np.min(lengths) / 2.0)
    edges = np.arange(0.0, maximum_r + RDF_BIN_WIDTH_A, RDF_BIN_WIDTH_A)
    tree = cKDTree(oxygen, boxsize=lengths)
    pairs = tree.query_pairs(maximum_r, output_type="ndarray")
    delta = oxygen[pairs[:, 1]] - oxygen[pairs[:, 0]]
    delta -= np.rint(delta / lengths) * lengths
    distances = np.linalg.norm(delta, axis=1)
    counts = np.histogram(distances, bins=edges)[0]
    del pairs, delta, distances

    radii = 0.5 * (edges[1:] + edges[:-1])
    shells = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    number_density = oxygen.shape[0] / float(np.prod(lengths))
    normalization = 0.5 * oxygen.shape[0] * number_density * shells
    rdf = counts / normalization

    # This finite-r transform is an oxygen-only structural diagnostic.  The
    # Lorch window suppresses termination ripples; it is not a neutron-weighted
    # total experimental structure factor.
    window = np.sinc(radii / maximum_r)
    kernel = np.sinc(
        np.outer(STRUCTURE_FACTOR_Q_A_INV, radii) / np.pi
    )
    integrand = (rdf - 1.0) * radii * radii * window
    structure_factor = 1.0 + 4.0 * np.pi * number_density * np.trapezoid(
        kernel * integrand, radii, axis=1
    )
    return radii, rdf, structure_factor


def _first_peak(q: NDArray[np.float64], values: NDArray[np.float64]) -> dict[str, float]:
    mask = (q >= 1.2) & (q <= 2.5)
    indices = np.flatnonzero(mask)
    index = int(indices[np.argmax(values[mask])])
    return {
        "q_a_inverse": float(q[index]),
        "value": float(values[index]),
    }


def _rdf_feature(
    radii: NDArray[np.float64], rdf: NDArray[np.float64], low: float, high: float
) -> dict[str, float]:
    mask = (radii >= low) & (radii <= high)
    indices = np.flatnonzero(mask)
    index = int(indices[np.argmax(rdf[mask])])
    return {"r_a": float(radii[index]), "g_oo": float(rdf[index])}


def _configure_plot_style() -> None:
    for font_path in (
        Path("/usr/share/fonts/urw-base35/NimbusMonoPS-Regular.otf"),
        Path("/usr/share/fonts/urw-base35/NimbusMonoPS-Italic.otf"),
        Path("/usr/share/fonts/urw-base35/NimbusMonoPS-Bold.otf"),
        Path("/usr/share/fonts/urw-base35/NimbusMonoPS-BoldItalic.otf"),
    ):
        if font_path.is_file():
            font_manager.fontManager.addfont(font_path)
    plt.rcdefaults()
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.monospace"] = [FONT_COURIER]
    plt.rcParams["mathtext.rm"] = FONT_COURIER
    plt.rcParams["mathtext.it"] = f"{FONT_COURIER}:italic"
    plt.rcParams["mathtext.bf"] = f"{FONT_COURIER}:bold"
    plt.rcParams["mathtext.cal"] = FONT_COURIER
    plt.rcParams["mathtext.sf"] = FONT_COURIER
    plt.rcParams["mathtext.tt"] = FONT_COURIER
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams.update(
        rcparams_with_fontsize(
            RC_BASE_STANDARD,
            PAPER_FONTSIZE,
            overrides={
                "savefig.dpi": 300,
                "axes.linewidth": 0.8,
                "lines.linewidth": 1.0,
                "xtick.major.size": 4.0,
                "xtick.major.width": 0.8,
                "xtick.minor.size": 2.0,
                "xtick.minor.width": 0.6,
                "ytick.major.size": 4.0,
                "ytick.major.width": 0.8,
                "ytick.minor.size": 2.0,
                "ytick.minor.width": 0.6,
            },
        )
    )


def _style_axes(axes: NDArray[object]) -> None:
    for axis, label in zip(axes, ("(a)", "(b)", "(c)"), strict=True):
        axis.minorticks_on()
        axis.tick_params(
            axis="both",
            which="both",
            direction="in",
            bottom=True,
            left=True,
            top=False,
            right=False,
            labelsize=PAPER_FONTSIZE,
        )
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
        for text in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            text.set_fontfamily(FONT_COURIER)
            text.set_fontsize(PAPER_FONTSIZE)
        axis.xaxis.label.set_fontfamily(FONT_COURIER)
        axis.xaxis.label.set_fontsize(PAPER_FONTSIZE)
        axis.yaxis.label.set_fontfamily(FONT_COURIER)
        axis.yaxis.label.set_fontsize(PAPER_FONTSIZE)
        axis.text(
            0.0,
            1.04,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=PAPER_FONTSIZE,
            fontfamily=FONT_COURIER,
            clip_on=False,
        )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".png", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    fig.savefig(temporary, dpi=300, facecolor="white", format="png")
    plt.close(fig)
    os.replace(temporary, path)
    path.chmod(0o644)


def _seed_colors() -> dict[int, object]:
    colors = plt.get_cmap("plasma")(np.linspace(0.0, 1.0, 4))[:3]
    return dict(zip(SEEDS, colors, strict=True))


def _target_temperature(time_ns: NDArray[np.float64]) -> NDArray[np.float64]:
    target = np.empty_like(time_ns)
    target[time_ns <= 1.0] = 350.0
    ramp_one = (time_ns > 1.0) & (time_ns <= 2.0)
    target[ramp_one] = 350.0 - 110.0 * (time_ns[ramp_one] - 1.0)
    target[(time_ns > 2.0) & (time_ns <= 3.0)] = 240.0
    ramp_two = (time_ns > 3.0) & (time_ns <= 19.0)
    target[ramp_two] = 240.0 - 10.0 * (time_ns[ramp_two] - 3.0)
    target[time_ns > 19.0] = 80.0
    return target


def _plot_thermodynamics(
    path: Path,
    thermo_by_seed: dict[int, NDArray[np.float64]],
    density_by_seed: dict[int, NDArray[np.float64]],
    water_molecules_by_seed: dict[int, int],
) -> None:
    _configure_plot_style()
    colors = _seed_colors()
    fig, axes = plt.subplots(
        1, 3, figsize=(AASTEX_FULL_WIDTH_IN, THREE_PANEL_ROW_HEIGHT_IN)
    )
    for seed in SEEDS:
        thermo = thermo_by_seed[seed]
        time_ns = (
            np.arange(thermo.shape[0], dtype=np.float64) + 1.0
        ) * THERMO_INTERVAL_NS
        axes[0].plot(
            time_ns,
            thermo[:, 0],
            color=colors[seed],
            linewidth=0.55,
            alpha=0.78,
            label=f"Seed {seed}",
        )
        axes[1].plot(
            time_ns,
            density_by_seed[seed],
            color=colors[seed],
            linewidth=0.65,
            alpha=0.85,
        )
        axes[2].plot(
            time_ns,
            thermo[:, 2] / water_molecules_by_seed[seed],
            color=colors[seed],
            linewidth=0.65,
            alpha=0.85,
        )
    time_ns = (
        np.arange(next(iter(thermo_by_seed.values())).shape[0], dtype=np.float64)
        + 1.0
    ) * THERMO_INTERVAL_NS
    axes[0].plot(
        time_ns,
        _target_temperature(time_ns),
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="Target",
    )
    axes[1].axhspan(
        EXPERIMENTAL_LDA_DENSITY_G_CM3
        - EXPERIMENTAL_LDA_DENSITY_HALF_RANGE_G_CM3,
        EXPERIMENTAL_LDA_DENSITY_G_CM3
        + EXPERIMENTAL_LDA_DENSITY_HALF_RANGE_G_CM3,
        color="0.86",
        linewidth=0,
        zorder=0,
    )
    axes[1].axhline(
        PUBLISHED_QTIP4PF_LDA_DENSITY_G_CM3,
        color="black",
        linestyle=":",
        linewidth=0.8,
    )
    axes[0].set(xlabel="Time (ns)", ylabel="Temperature (K)")
    axes[1].set(xlabel="Time (ns)", ylabel=r"Density (g cm$^{-3}$)")
    axes[2].set(
        xlabel="Time (ns)", ylabel=r"Potential energy (eV H$_2$O$^{-1}$)"
    )
    for axis in axes:
        axis.set_xlim(0.0, 20.0)
        for boundary in (1.0, 2.0, 3.0, 19.0):
            axis.axvline(boundary, color="0.75", linewidth=0.45, zorder=0)
    _style_axes(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        prop={"family": FONT_COURIER, "size": PAPER_FONTSIZE},
        handlelength=1.6,
        columnspacing=1.4,
        handletextpad=0.45,
    )
    fig.subplots_adjust(
        left=0.077, right=0.992, bottom=0.285, top=0.91, wspace=0.38
    )
    _save_figure(fig, path)


def _plot_structure(
    path: Path,
    rdfs: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]],
    structure_factors: dict[int, NDArray[np.float64]],
    crystal_history: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> None:
    _configure_plot_style()
    colors = _seed_colors()
    fig, axes = plt.subplots(
        1, 3, figsize=(AASTEX_FULL_WIDTH_IN, THREE_PANEL_ROW_HEIGHT_IN)
    )
    for seed in SEEDS:
        radii, rdf = rdfs[seed]
        plot_mask = radii <= RDF_PLOT_MAXIMUM_A
        axes[0].plot(
            radii[plot_mask], rdf[plot_mask], color=colors[seed], label=f"Seed {seed}"
        )
        axes[1].plot(
            STRUCTURE_FACTOR_Q_A_INV,
            structure_factors[seed],
            color=colors[seed],
        )
        times, fractions = crystal_history[seed]
        axes[2].plot(
            times,
            fractions,
            color=colors[seed],
            marker="o",
            markersize=2.3,
            markerfacecolor=colors[seed],
            markeredgewidth=0.0,
        )
    axes[1].axvspan(
        *LDA_FIRST_PEAK_RANGE_A_INV,
        color="0.86",
        linewidth=0,
        zorder=0,
    )
    axes[0].set(
        xlim=(2.2, 7.0), xlabel=r"$r$ ($\AA$)", ylabel=r"$g_{\rm OO}(r)$"
    )
    axes[1].set(
        xlim=(0.8, 6.0),
        xlabel=r"$q$ ($\AA^{-1}$)",
        ylabel=r"Oxygen $S_{\rm OO}(q)$",
    )
    axes[2].set(
        xlim=(1.0, 20.0),
        ylim=(0.0, None),
        xlabel="Saved-frame time (ns)",
        ylabel="CHILL+ Ih + Ic fraction",
    )
    for boundary in (2.0, 3.0, 19.0):
        axes[2].axvline(boundary, color="0.75", linewidth=0.45, zorder=0)
    _style_axes(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        prop={"family": FONT_COURIER, "size": PAPER_FONTSIZE},
        handlelength=1.6,
        columnspacing=1.6,
        handletextpad=0.45,
    )
    fig.subplots_adjust(
        left=0.077, right=0.992, bottom=0.285, top=0.91, wspace=0.38
    )
    _save_figure(fig, path)


def _write_structural_csv(
    path: Path,
    rdfs: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]],
    structure_factors: dict[int, NDArray[np.float64]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "r_a",
                "g_oo",
                "q_a_inverse",
                "s_oo_lorch_diagnostic",
            ]
        )
        for seed in SEEDS:
            radii, rdf = rdfs[seed]
            rows = max(radii.size, STRUCTURE_FACTOR_Q_A_INV.size)
            for index in range(rows):
                writer.writerow(
                    [
                        seed,
                        radii[index] if index < radii.size else "",
                        rdf[index] if index < rdf.size else "",
                        (
                            STRUCTURE_FACTOR_Q_A_INV[index]
                            if index < STRUCTURE_FACTOR_Q_A_INV.size
                            else ""
                        ),
                        (
                            structure_factors[seed][index]
                            if index < STRUCTURE_FACTOR_Q_A_INV.size
                            else ""
                        ),
                    ]
                )
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o644)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    thermo_by_seed: dict[int, NDArray[np.float64]] = {}
    density_by_seed: dict[int, NDArray[np.float64]] = {}
    water_molecules_by_seed: dict[int, int] = {}
    rdfs: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    structure_factors: dict[int, NDArray[np.float64]] = {}
    crystal_history: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    replicas: dict[int, dict[str, object]] = {}

    progress = tqdm(
        total=len(SEEDS) * 22,
        desc="Validating amorphous replicas",
        unit="diagnostic",
        disable=args.no_progress,
    )
    for seed in SEEDS:
        run_dir = run_root / f"amorphous_seed{seed}"
        required = ("COMPLETED", "thermo.out", "dump.xyz", "restart.xyz", "run.in")
        missing = [name for name in required if not (run_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"{run_dir} is missing {missing}.")
        thermo = _read_thermo(run_dir / "thermo.out")
        frames = list(iter_xyz_frames(run_dir / "dump.xyz"))
        if thermo.shape[0] != 10_000 or len(frames) != 20:
            raise ValueError(
                f"{run_dir}: expected 10000 thermo records and 20 frames; "
                f"found {thermo.shape[0]} and {len(frames)}."
            )
        water_molecules = int(np.count_nonzero(frames[-1].species == "O"))
        density = _density_from_thermo(thermo, water_molecules)
        thermo_by_seed[seed] = thermo
        density_by_seed[seed] = density
        water_molecules_by_seed[seed] = water_molecules
        progress.update()

        fractions: list[float] = []
        frame_records: list[dict[str, object]] = []
        final_oxygen: NDArray[np.float64] | None = None
        final_lengths: NDArray[np.float64] | None = None
        final_chill: dict[str, object] | None = None
        final_tree: cKDTree | None = None
        for frame in frames:
            lengths = _orthorhombic_lengths(frame)
            positions = _wrapped_positions(frame.positions_angstrom, lengths)
            oxygen = positions[frame.species == "O"]
            chill, _, _, tree = chill_plus(oxygen, lengths)
            crystalline_fraction = float(chill["hexagonal_fraction"]) + float(
                chill["cubic_fraction"]
            )
            fractions.append(crystalline_fraction)
            frame_records.append(
                {
                    "frame_index": frame.frame_index,
                    "time_ns": float(frame.frame_index + 1),
                    "density_g_cm3": float(
                        water_molecules
                        * WATER_MOLAR_MASS_G_MOL
                        / AVOGADRO_MOL_MINUS_ONE
                        / (np.prod(lengths) * 1.0e-24)
                    ),
                    "chill_plus": chill,
                }
            )
            final_oxygen = oxygen
            final_lengths = lengths
            final_chill = chill
            final_tree = tree
            progress.update()
        assert final_oxygen is not None
        assert final_lengths is not None
        assert final_chill is not None
        assert final_tree is not None
        crystal_history[seed] = (
            np.arange(1.0, len(frames) + 1.0),
            np.asarray(fractions),
        )

        radii, rdf, structure_factor = _radial_distribution_and_structure_factor(
            final_oxygen, final_lengths
        )
        rdfs[seed] = (radii, rdf)
        structure_factors[seed] = structure_factor
        progress.update()

        fifth_distances = final_tree.query(final_oxygen, k=6)[0][:, 5]
        final_hold = thermo[-FINAL_HOLD_RECORDS:]
        final_density = density[-FINAL_HOLD_RECORDS:]
        final_pressure = _mean_normal_pressure(final_hold)
        final_potential_per_water = final_hold[:, 2] / water_molecules
        density_deviation = float(
            np.mean(final_density) - EXPERIMENTAL_LDA_DENSITY_G_CM3
        )
        replicas[seed] = {
            "run_directory": str(run_dir),
            "completed_marker_present": True,
            "trajectory_sha256": file_sha256(run_dir / "dump.xyz"),
            "restart_sha256": file_sha256(run_dir / "restart.xyz"),
            "water_molecules": water_molecules,
            "thermo_records": int(thermo.shape[0]),
            "saved_frames": len(frames),
            "stage_summaries": _stage_summaries(
                thermo, density, water_molecules
            ),
            "final_80k_hold": {
                "records": FINAL_HOLD_RECORDS,
                "duration_ns": FINAL_HOLD_RECORDS * THERMO_INTERVAL_NS,
                "temperature_k": _summary(final_hold[:, 0]),
                "density_g_cm3": _summary(final_density),
                "density_deviation_from_experimental_g_cm3": density_deviation,
                "density_relative_deviation_from_experimental": (
                    density_deviation / EXPERIMENTAL_LDA_DENSITY_G_CM3
                ),
                "mean_normal_pressure_gpa": _summary(final_pressure),
                "potential_energy_ev_per_water": _summary(
                    final_potential_per_water
                ),
                "trends": {
                    "temperature_k": hac_linear_trend(
                        final_hold[:, 0], THERMO_INTERVAL_NS
                    ),
                    "density_g_cm3": hac_linear_trend(
                        final_density, THERMO_INTERVAL_NS
                    ),
                    "mean_normal_pressure_gpa": hac_linear_trend(
                        final_pressure, THERMO_INTERVAL_NS
                    ),
                    "potential_energy_ev_per_water": hac_linear_trend(
                        final_potential_per_water, THERMO_INTERVAL_NS
                    ),
                },
            },
            "saved_frame_history": frame_records,
            "final_structure": {
                "chill_plus": final_chill,
                "chill_plus_ih_plus_ic_fraction": float(
                    final_chill["hexagonal_fraction"]
                )
                + float(final_chill["cubic_fraction"]),
                "fifth_oxygen_neighbor_within_3p5_a_fraction": float(
                    np.mean(fifth_distances <= CHILL_NEIGHBOR_CUTOFF_A)
                ),
                "rdf_first_peak": _rdf_feature(radii, rdf, 2.5, 3.1),
                "rdf_interstitial_region_maximum": _rdf_feature(
                    radii, rdf, 3.0, 3.5
                ),
                "rdf_second_peak": _rdf_feature(radii, rdf, 4.0, 5.2),
                "oxygen_structure_factor_first_peak": _first_peak(
                    STRUCTURE_FACTOR_Q_A_INV, structure_factor
                ),
            },
        }

    progress.close()
    final_densities = np.asarray(
        [replicas[seed]["final_80k_hold"]["density_g_cm3"]["mean"] for seed in SEEDS],
        dtype=np.float64,
    )
    crystalline = np.asarray(
        [
            replicas[seed]["final_structure"]["chill_plus_ih_plus_ic_fraction"]
            for seed in SEEDS
        ],
        dtype=np.float64,
    )
    first_peaks = np.asarray(
        [
            replicas[seed]["final_structure"]["oxygen_structure_factor_first_peak"][
                "q_a_inverse"
            ]
            for seed in SEEDS
        ],
        dtype=np.float64,
    )
    experimental_low = (
        EXPERIMENTAL_LDA_DENSITY_G_CM3
        - EXPERIMENTAL_LDA_DENSITY_HALF_RANGE_G_CM3
    )
    experimental_high = (
        EXPERIMENTAL_LDA_DENSITY_G_CM3
        + EXPERIMENTAL_LDA_DENSITY_HALF_RANGE_G_CM3
    )
    density_pass = bool(
        np.all((final_densities >= experimental_low) & (final_densities <= experimental_high))
    )
    amorphous_local_order_pass = bool(np.all(crystalline < 0.01))
    report: dict[str, object] = {
        "schema_version": 1,
        "analysis": "nep_mbpol_candidate_lda_80k_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "references": REFERENCES,
        "protocol": {
            "ensemble": "NPT at 0.1 MPa throughout",
            "total_duration_ns": 20.0,
            "time_step_fs": 0.2,
            "cooling_rate_k_per_ns": 10.0,
            "final_hold_temperature_k": TARGET_TEMPERATURE_K,
            "final_hold_duration_ns": 1.0,
            "experimental_lda_density_g_cm3": EXPERIMENTAL_LDA_DENSITY_G_CM3,
            "experimental_density_comparison_range_g_cm3": [
                experimental_low,
                experimental_high,
            ],
            "published_qtip4pf_lda_density_g_cm3": PUBLISHED_QTIP4PF_LDA_DENSITY_G_CM3,
            "published_lda_first_structure_factor_peak_range_a_inverse": list(
                LDA_FIRST_PEAK_RANGE_A_INV
            ),
        },
        "replicas": replicas,
        "cross_replica": {
            "final_density_g_cm3": _summary(final_densities),
            "final_chill_plus_ih_plus_ic_fraction": _summary(crystalline),
            "oxygen_structure_factor_first_peak_q_a_inverse": _summary(first_peaks),
        },
        "acceptance": {
            "trajectory_completion_pass": True,
            "low_crystalline_local_order_pass": amorphous_local_order_pass,
            "experimental_lda_density_pass": density_pass,
            "collision_ready": False,
            "decision": "rejected_as_experimental_lda_collision_target",
            "interpretation": (
                "The three trajectories completed and the final snapshots have very low "
                "CHILL+ Ih/Ic fractions, but their 80 K densities lie well above the "
                "experimental LDA range.  Their oxygen-only first diffraction diagnostic "
                "also lies above the published q-TIP4P/F LDA range.  They are reproducible "
                "dense amorphous candidates, not validated experimental-density LDA."
            ),
        },
        "limitations": [
            "Only one atomic frame was saved during the final 1 ns hold for each seed; thermodynamic stationarity has 500 records, but structural stationarity cannot be established.",
            "The plotted S_OO(q) is a finite-cell, oxygen-only, Lorch-windowed Fourier transform of g_OO(r), not the isotope-weighted experimental neutron structure factor.",
            "CHILL+ detects local Ih/Ic environments; a low CHILL+ fraction alone does not establish agreement with experimental LDA.",
            "No experimental RDF or structure-factor table was digitized or fitted in this analysis.",
            "The classical NEP-MB-pol trajectories omit nuclear quantum effects at 80 K.",
        ],
        "recommended_next_step": {
            "decision_gate": (
                "Run a short three-replica 80 K density--stress scan around 0.93--1.02 "
                "g/cm3, including 0.94 g/cm3, before another 20 ns quench."
            ),
            "reason": (
                "This determines whether NEP-MB-pol can mechanically sustain an "
                "experimental-density amorphous network or whether NPT will always "
                "collapse it toward approximately 1.02 g/cm3."
            ),
            "prohibited_shortcut": (
                "Do not simply rescale a final snapshot and label it LDA.  Any "
                "experimental-density cell must be equilibrated, its residual stress "
                "reported, and its RDF/order revalidated."
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_thermodynamics(
        output_dir / "amorphous_lda_80K_thermodynamics.png",
        thermo_by_seed,
        density_by_seed,
        water_molecules_by_seed,
    )
    _plot_structure(
        output_dir / "amorphous_lda_80K_structure.png",
        rdfs,
        structure_factors,
        crystal_history,
    )
    _write_structural_csv(
        output_dir / "amorphous_lda_80K_structural_diagnostics.csv",
        rdfs,
        structure_factors,
    )
    _atomic_write(
        output_dir / "amorphous_lda_80K_validation.json",
        json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
