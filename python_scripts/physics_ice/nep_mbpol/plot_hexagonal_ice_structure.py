#!/usr/bin/env python3
"""Render the generated ice-Ih structure in the project's paper style."""

from __future__ import annotations

import re
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


HERE = Path(__file__).resolve().parent
PHYSICS_ICE = HERE.parent
sys.path.insert(0, str(PHYSICS_ICE))

from constants import (  # noqa: E402
    FONT_COURIER,
    FONTSIZE_12,
    OUTPUT_DIR,
    RC_BASE_STANDARD,
    rcparams_with_fontsize,
)


STRUCTURE_PATH = (
    HERE / "structures" / "ice_ih_8x8x8_seed1000_initial.xyz"
)
PNG_PATH = OUTPUT_DIR / "hexagonal_ice_structure.png"
PDF_PATH = OUTPUT_DIR / "hexagonal_ice_structure.pdf"

# Full-width AASTeX figure*; compact enough for a two-column page.
FIGURE_WIDTH_IN = 7.1
FIGURE_HEIGHT_IN = 5.15

OXYGEN_COLOR = "slategray"
HYDROGEN_COLOR = "lightgray"
COVALENT_COLOR = "dimgray"
NETWORK_COLOR = "lightgray"


def _configure_style() -> None:
    plt.rcParams["font.family"] = FONT_COURIER
    plt.rcParams["mathtext.rm"] = FONT_COURIER
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams.update(
        rcparams_with_fontsize(
            RC_BASE_STANDARD,
            FONTSIZE_12,
            overrides={
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.dpi": 300,
            },
        )
    )


def _load_gpumd_xyz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[0])
    lattice_match = re.search(r'Lattice="([^"]+)"', lines[1])
    if lattice_match is None:
        raise ValueError(f"Missing Lattice field in {path}")
    lattice = np.asarray(
        [float(value) for value in lattice_match.group(1).split()], dtype=float
    ).reshape(3, 3)

    species = []
    positions = np.empty((atom_count, 3), dtype=float)
    for index, line in enumerate(lines[2 : atom_count + 2]):
        fields = line.split()
        species.append(fields[0])
        positions[index] = [float(value) for value in fields[1:4]]
    return np.asarray(species), positions, lattice


def _representative_volume(
    species: np.ndarray, positions: np.ndarray, lattice: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.all(species[0::3] == "O"):
        raise ValueError("Expected the GPUMD structure to be ordered O-H-H.")
    inverse_lattice = np.linalg.inv(lattice)
    oxygen_all = positions[0::3]
    oxygen_fractional = oxygen_all @ inverse_lattice
    selected = np.flatnonzero(
        np.all(
            (oxygen_fractional >= 0.375) & (oxygen_fractional < 0.625),
            axis=1,
        )
    )

    molecules = []
    for molecule_index in selected:
        oxygen = oxygen_all[molecule_index]
        molecule = [oxygen]
        for atom_index in (3 * molecule_index + 1, 3 * molecule_index + 2):
            displacement_fractional = (
                positions[atom_index] - oxygen
            ) @ inverse_lattice
            displacement_fractional -= np.rint(displacement_fractional)
            molecule.append(oxygen + displacement_fractional @ lattice)
        molecules.append(molecule)

    molecule_positions = np.asarray(molecules)
    molecule_positions -= molecule_positions.reshape(-1, 3).min(axis=0)
    oxygen = molecule_positions[:, 0, :]
    hydrogen = molecule_positions[:, 1:, :].reshape(-1, 3)
    return molecule_positions, oxygen, hydrogen


def _nearest_oxygen_segments(oxygen: np.ndarray) -> np.ndarray:
    segments = []
    for first in range(len(oxygen)):
        displacement = oxygen[first + 1 :] - oxygen[first]
        distance = np.linalg.norm(displacement, axis=1)
        for offset in np.flatnonzero((distance > 2.45) & (distance < 3.15)):
            second = first + 1 + int(offset)
            segments.append([oxygen[first], oxygen[second]])
    return np.asarray(segments)


def _covalent_segments(molecules: np.ndarray) -> np.ndarray:
    segments = []
    for molecule in molecules:
        segments.append([molecule[0], molecule[1]])
        segments.append([molecule[0], molecule[2]])
    return np.asarray(segments)


def _equal_3d_limits(ax, points: np.ndarray) -> float:
    minima = points.min(axis=0)
    maxima = points.max(axis=0)
    center = 0.5 * (minima + maxima)
    half_span = 0.52 * np.max(maxima - minima)
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)
    ax.set_box_aspect((1.0, 1.0, 1.0), zoom=1.35)
    return 2.0 * half_span


def _panel_label(ax, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=float(FONTSIZE_12),
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
            "pad": 0.15,
        },
        zorder=30,
    )


def _format_panel_frame(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def _draw_scale_bar(
    ax, x_limits: tuple[float, float], y_limits: tuple[float, float]
) -> None:
    bar_length = 5.0
    x_start = x_limits[1] - bar_length - 0.7
    y_start = y_limits[0] + 0.8
    ax.plot(
        [x_start, x_start + bar_length],
        [y_start, y_start],
        color="black",
        linewidth=1.5,
        solid_capstyle="butt",
        zorder=10,
    )
    ax.text(
        x_start + 0.5 * bar_length,
        y_start + 0.35,
        "5 Å",
        ha="center",
        va="bottom",
        fontsize=float(FONTSIZE_12),
        zorder=10,
    )


def _draw_3d_scale_bar(ax, scene_span: float) -> None:
    """Draw a projected 5-Angstrom reference for the 3D overview."""
    bar_fraction = 5.0 / scene_span
    x_end = 0.965
    x_start = x_end - bar_fraction
    y_start = 0.035
    ax.plot(
        [x_start, x_end],
        [y_start, y_start],
        transform=ax.transAxes,
        color="black",
        linewidth=1.5,
        solid_capstyle="butt",
        clip_on=False,
        zorder=30,
    )
    ax.text(
        0.5 * (x_start + x_end),
        y_start + 0.018,
        "5 Å",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=float(FONTSIZE_12),
        zorder=30,
    )


def _draw_projection(
    ax,
    oxygen: np.ndarray,
    hydrogen: np.ndarray,
    covalent: np.ndarray,
    network: np.ndarray,
    axes: tuple[int, int],
) -> None:
    projected_network = network[:, :, axes]
    projected_covalent = covalent[:, :, axes]
    ax.add_collection(
        LineCollection(
            projected_network,
            colors=NETWORK_COLOR,
            linewidths=0.65,
            alpha=0.75,
            zorder=1,
        )
    )
    ax.add_collection(
        LineCollection(
            projected_covalent,
            colors=COVALENT_COLOR,
            linewidths=0.7,
            alpha=0.7,
            zorder=2,
        )
    )
    ax.scatter(
        hydrogen[:, axes[0]],
        hydrogen[:, axes[1]],
        s=5.5,
        c=HYDROGEN_COLOR,
        edgecolors="black",
        linewidths=0.18,
        zorder=3,
    )
    ax.scatter(
        oxygen[:, axes[0]],
        oxygen[:, axes[1]],
        s=14,
        c=OXYGEN_COLOR,
        edgecolors="black",
        linewidths=0.25,
        zorder=4,
    )
    all_points = np.vstack((oxygen, hydrogen))
    x = all_points[:, axes[0]]
    y = all_points[:, axes[1]]
    x_span = float(np.ptp(x))
    y_span = float(np.ptp(y))
    plot_span = max(1.08 * x_span, 1.08 * y_span + 2.0)
    x_center = 0.5 * (x.min() + x.max())
    extra_y = plot_span - y_span
    x_limits = (x_center - 0.5 * plot_span, x_center + 0.5 * plot_span)
    y_limits = (y.min() - 0.67 * extra_y, y.max() + 0.33 * extra_y)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    _format_panel_frame(ax)
    _draw_scale_bar(ax, x_limits, y_limits)


def plot_hexagonal_ice_structure() -> tuple[Path, Path]:
    _configure_style()
    species, positions, lattice = _load_gpumd_xyz(STRUCTURE_PATH)
    molecules, oxygen, hydrogen = _representative_volume(
        species, positions, lattice
    )
    covalent = _covalent_segments(molecules)
    network = _nearest_oxygen_segments(oxygen)

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(2.0, 1.0),
        height_ratios=(1.0, 1.0),
        wspace=0.035,
        hspace=0.045,
    )
    frame_a = fig.add_subplot(grid[:, 0])
    ax_a = fig.add_subplot(grid[:, 0], projection="3d")
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    _format_panel_frame(frame_a)
    frame_a.set_xlim(0.0, 1.0)
    frame_a.set_ylim(0.0, 1.0)
    frame_a.set_zorder(0)
    ax_a.set_zorder(1)
    ax_a.patch.set_alpha(0.0)

    ax_a.add_collection3d(
        Line3DCollection(
            network,
            colors=NETWORK_COLOR,
            linewidths=0.45,
            alpha=0.35,
        )
    )
    ax_a.add_collection3d(
        Line3DCollection(
            covalent,
            colors=COVALENT_COLOR,
            linewidths=0.65,
            alpha=0.7,
        )
    )
    ax_a.scatter(
        hydrogen[:, 0],
        hydrogen[:, 1],
        hydrogen[:, 2],
        s=4.5,
        c=HYDROGEN_COLOR,
        edgecolors="black",
        linewidths=0.12,
        depthshade=True,
    )
    ax_a.scatter(
        oxygen[:, 0],
        oxygen[:, 1],
        oxygen[:, 2],
        s=12,
        c=OXYGEN_COLOR,
        edgecolors="black",
        linewidths=0.2,
        depthshade=True,
    )
    scene_span = _equal_3d_limits(ax_a, molecules.reshape(-1, 3))
    ax_a.view_init(elev=19, azim=-58)
    ax_a.set_axis_off()

    _draw_projection(ax_b, oxygen, hydrogen, covalent, network, (0, 1))
    _draw_projection(ax_c, oxygen, hydrogen, covalent, network, (0, 2))

    _panel_label(frame_a, "(a)")
    _draw_3d_scale_bar(frame_a, scene_span)
    _panel_label(ax_b, "(b)")
    _panel_label(ax_c, "(c)")

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=OXYGEN_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=5.5,
            label="Oxygen",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=HYDROGEN_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=4.0,
            label="Hydrogen",
        ),
        Line2D(
            [],
            [],
            color=NETWORK_COLOR,
            linewidth=1.5,
            label="O–O network",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.006),
        handlelength=1.6,
        columnspacing=1.6,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.012, right=0.995, bottom=0.08, top=0.97)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=300, facecolor="white")
    fig.savefig(PDF_PATH, facecolor="white")
    plt.close(fig)

    print(f"Representative volume: {len(molecules)} H2O molecules")
    print(f"Saved: {PNG_PATH}")
    print(f"Saved: {PDF_PATH}")
    return PNG_PATH, PDF_PATH


if __name__ == "__main__":
    plot_hexagonal_ice_structure()
