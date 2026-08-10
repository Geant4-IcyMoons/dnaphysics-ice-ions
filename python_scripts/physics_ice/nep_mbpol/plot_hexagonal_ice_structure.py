#!/usr/bin/env python3
"""Render a periodic molecular-ice structure in the project's paper style."""

from __future__ import annotations

import argparse
import gzip
import os
import re
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


HERE = Path(__file__).resolve().parent
PHYSICS_ICE = HERE.parent
sys.path.insert(0, str(PHYSICS_ICE))

from constants import (  # noqa: E402
    AASTEX_FULL_WIDTH_IN,
    FONT_COURIER,
    OUTPUT_DIR,
    PAPER_FONTSIZE,
    RC_BASE_STANDARD,
    THREE_PANEL_ROW_HEIGHT_IN,
    rcparams_with_fontsize,
)


STRUCTURE_PATH = (
    PHYSICS_ICE
    / "ice_structures"
    / "preparation"
    / "hexagonal_ih_genice2"
    / "ice_ih_8x8x8_seed1000_melt_start.xyz"
)
PNG_PATH = OUTPUT_DIR / "hexagonal_ice_structure.png"
ACCEPTED_HEXAGONAL_PATH = (
    PHYSICS_ICE
    / "ice_structures"
    / "hexagonal_ih_100K_experimental"
    / "seed1000_final.xyz.gz"
)
ACCEPTED_AMORPHOUS_PATH = (
    PHYSICS_ICE
    / "ice_structures"
    / "epsr_lda80k"
    / "artifacts"
    / "lda80k_epsr.xyz.gz"
)

OXYGEN_COLOR = "slategray"
HYDROGEN_COLOR = "lightgray"
COVALENT_COLOR = "dimgray"
NETWORK_COLOR = "lightgray"


def _configure_style() -> None:
    filenames = (
        "NimbusMonoPS-Regular.otf",
        "NimbusMonoPS-Italic.otf",
        "NimbusMonoPS-Bold.otf",
        "NimbusMonoPS-BoldItalic.otf",
    )
    configured_directory = os.environ.get("PHYSICS_ICE_PLOT_FONT_DIR")
    directories = [Path("/usr/share/fonts/urw-base35")]
    if configured_directory:
        directories.insert(0, Path(configured_directory))
    registered = 0
    for filename in filenames:
        for directory in directories:
            font_path = directory / filename
            if font_path.is_file():
                font_manager.fontManager.addfont(font_path)
                registered += 1
                break
    if registered != len(filenames):
        raise RuntimeError(
            "The four Nimbus Mono PS faces are required. Set "
            "PHYSICS_ICE_PLOT_FONT_DIR to a staged runtime copy."
        )
    plt.rcParams["font.family"] = FONT_COURIER
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
            },
        )
    )


def _load_gpumd_xyz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    else:
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
    fractional_all = positions @ inverse_lattice
    fractional_all -= np.floor(fractional_all)
    positions = fractional_all @ lattice
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


def _centered_physical_cube(
    species: np.ndarray,
    positions: np.ndarray,
    lattice: np.ndarray,
    side_angstrom: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return molecules whose oxygens occupy one centered physical cube."""
    if side_angstrom <= 0.0:
        raise ValueError("The physical crop side must be positive.")
    if not np.all(species[0::3] == "O"):
        raise ValueError("Expected the GPUMD structure to be ordered O-H-H.")
    inverse_lattice = np.linalg.inv(lattice)
    fractional = positions @ inverse_lattice
    fractional -= np.floor(fractional)
    oxygen_fractional = fractional[0::3]
    oxygen_displacement_fractional = oxygen_fractional - 0.5
    oxygen_displacement_fractional -= np.rint(oxygen_displacement_fractional)
    oxygen_displacement = oxygen_displacement_fractional @ lattice
    half_side = 0.5 * side_angstrom
    selected = np.flatnonzero(
        np.all(np.abs(oxygen_displacement) < half_side, axis=1)
    )
    if not len(selected):
        raise ValueError("The physical crop contains no water molecules.")

    molecules = []
    for molecule_index in selected:
        oxygen = oxygen_displacement[molecule_index]
        molecule = [oxygen]
        oxygen_atom = 3 * molecule_index
        for atom_index in (oxygen_atom + 1, oxygen_atom + 2):
            displacement_fractional = (
                fractional[atom_index] - fractional[oxygen_atom]
            )
            displacement_fractional -= np.rint(displacement_fractional)
            molecule.append(oxygen + displacement_fractional @ lattice)
        molecules.append(molecule)
    molecule_positions = np.asarray(molecules)
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
        fontsize=PAPER_FONTSIZE,
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
    ax.set_box_aspect(1.0)
    ax.set_anchor("C")
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
        fontsize=PAPER_FONTSIZE,
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
        fontsize=PAPER_FONTSIZE,
        zorder=30,
    )


def _draw_projection(
    ax,
    oxygen: np.ndarray,
    hydrogen: np.ndarray,
    covalent: np.ndarray,
    network: np.ndarray,
    axes: tuple[int, int],
    *,
    draw_scale_bar: bool = True,
    x_label: str | None = None,
    fixed_half_span: float | None = None,
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
    if fixed_half_span is None:
        x_span = float(np.ptp(x))
        y_span = float(np.ptp(y))
        plot_span = max(1.08 * x_span, 1.08 * y_span + 2.0)
        x_center = 0.5 * (x.min() + x.max())
        extra_y = plot_span - y_span
        x_limits = (x_center - 0.5 * plot_span, x_center + 0.5 * plot_span)
        y_limits = (y.min() - 0.67 * extra_y, y.max() + 0.33 * extra_y)
    else:
        x_limits = (-fixed_half_span, fixed_half_span)
        y_limits = (-fixed_half_span, fixed_half_span)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    _format_panel_frame(ax)
    if x_label is not None:
        ax.set_xlabel(x_label, labelpad=3.0)
    if draw_scale_bar:
        _draw_scale_bar(ax, x_limits, y_limits)


def _draw_3d_panel(
    ax,
    frame_ax,
    molecules,
    oxygen,
    hydrogen,
    covalent,
    network,
    *,
    fixed_half_span: float | None = None,
) -> float:
    """Draw one framed 3D molecular overview and return its scene span."""
    _format_panel_frame(frame_ax)
    frame_ax.set_xlim(0.0, 1.0)
    frame_ax.set_ylim(0.0, 1.0)
    frame_ax.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_alpha(0.0)
    ax.add_collection3d(
        Line3DCollection(
            network, colors=NETWORK_COLOR, linewidths=0.45, alpha=0.35
        )
    )
    ax.add_collection3d(
        Line3DCollection(
            covalent, colors=COVALENT_COLOR, linewidths=0.65, alpha=0.7
        )
    )
    ax.scatter(
        hydrogen[:, 0], hydrogen[:, 1], hydrogen[:, 2], s=4.5,
        c=HYDROGEN_COLOR, edgecolors="black", linewidths=0.12,
        depthshade=True,
    )
    ax.scatter(
        oxygen[:, 0], oxygen[:, 1], oxygen[:, 2], s=12,
        c=OXYGEN_COLOR, edgecolors="black", linewidths=0.2,
        depthshade=True,
    )
    if fixed_half_span is None:
        scene_span = _equal_3d_limits(ax, molecules.reshape(-1, 3))
    else:
        ax.set_xlim(-fixed_half_span, fixed_half_span)
        ax.set_ylim(-fixed_half_span, fixed_half_span)
        ax.set_zlim(-fixed_half_span, fixed_half_span)
        ax.set_box_aspect((1.0, 1.0, 1.0), zoom=1.35)
        scene_span = 2.0 * fixed_half_span
    ax.view_init(elev=19, azim=-58)
    ax.set_axis_off()
    return scene_span


def _structure_scene(structure_path: Path):
    species, positions, lattice = _load_gpumd_xyz(structure_path)
    molecules, oxygen, hydrogen = _representative_volume(species, positions, lattice)
    return (
        molecules,
        oxygen,
        hydrogen,
        _covalent_segments(molecules),
        _nearest_oxygen_segments(oxygen),
    )


def _comparison_scene(structure_path: Path, crop_side_angstrom: float):
    species, positions, lattice = _load_gpumd_xyz(structure_path)
    molecules, oxygen, hydrogen = _centered_physical_cube(
        species, positions, lattice, crop_side_angstrom
    )
    return (
        molecules,
        oxygen,
        hydrogen,
        _covalent_segments(molecules),
        _nearest_oxygen_segments(oxygen),
    )


def plot_ice_structure_comparison(
    hexagonal_path: Path,
    amorphous_path: Path,
    png_path: Path,
) -> Path:
    """Render matching three-view rows for accepted Ih and LDA structures."""
    _configure_style()
    crop_side_angstrom = 14.0
    scenes = (
        _comparison_scene(hexagonal_path, crop_side_angstrom),
        _comparison_scene(amorphous_path, crop_side_angstrom),
    )
    fig = plt.figure(figsize=(AASTEX_FULL_WIDTH_IN, 2.0 * THREE_PANEL_ROW_HEIGHT_IN))
    grid = fig.add_gridspec(2, 3, wspace=0.035, hspace=0.055)
    labels = (("(a)", "(b)", "(c)"), ("(d)", "(e)", "(f)"))

    for row, scene in enumerate(scenes):
        molecules, oxygen, hydrogen, covalent, network = scene
        frame = fig.add_subplot(grid[row, 0])
        overview = fig.add_subplot(grid[row, 0], projection="3d")
        xy = fig.add_subplot(grid[row, 1])
        xz = fig.add_subplot(grid[row, 2])
        _draw_3d_panel(
            overview,
            frame,
            molecules,
            oxygen,
            hydrogen,
            covalent,
            network,
        )
        bottom_label = r"$x$ (Å)" if row == 1 else None
        if bottom_label is not None:
            frame.set_xlabel(bottom_label, labelpad=3.0)
        _draw_projection(
            xy, oxygen, hydrogen, covalent, network, (0, 1),
            draw_scale_bar=False, x_label=bottom_label,
        )
        _draw_projection(
            xz, oxygen, hydrogen, covalent, network, (0, 2),
            draw_scale_bar=row == 1, x_label=bottom_label,
        )
        _panel_label(frame, labels[row][0])
        _panel_label(xy, labels[row][1])
        _panel_label(xz, labels[row][2])

    fig.text(0.008, 0.705, "Hexagonal ice Ih, 100 K", rotation=90,
             ha="left", va="center")
    fig.text(0.008, 0.315, "Amorphous LDA, 80 K", rotation=90,
             ha="left", va="center")
    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=OXYGEN_COLOR,
               markeredgecolor="black", markeredgewidth=0.35, markersize=5.5,
               label="Oxygen"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=HYDROGEN_COLOR,
               markeredgecolor="black", markeredgewidth=0.35, markersize=4.0,
               label="Hydrogen"),
        Line2D([], [], color=NETWORK_COLOR, linewidth=1.5, label="O–O network"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.006), handlelength=1.6, columnspacing=1.6,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.105, top=0.985)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, facecolor="white")
    plt.close(fig)
    png_path.chmod(0o644)
    print(f"Hexagonal structure: {hexagonal_path}")
    print(f"Amorphous structure: {amorphous_path}")
    print(f"Physical crop: {crop_side_angstrom:g} A cube in both structures")
    print(f"Displayed waters: Ih={len(scenes[0][0])}, LDA={len(scenes[1][0])}")
    print(f"Saved: {png_path}")
    return png_path


def plot_hexagonal_ice_structure(
    structure_path: Path = STRUCTURE_PATH,
    png_path: Path = PNG_PATH,
) -> Path:
    _configure_style()
    species, positions, lattice = _load_gpumd_xyz(structure_path)
    molecules, oxygen, hydrogen = _representative_volume(
        species, positions, lattice
    )
    covalent = _covalent_segments(molecules)
    network = _nearest_oxygen_segments(oxygen)

    fig = plt.figure(
        figsize=(AASTEX_FULL_WIDTH_IN, THREE_PANEL_ROW_HEIGHT_IN)
    )
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 1.0),
        wspace=0.035,
    )
    frame_a = fig.add_subplot(grid[0, 0])
    ax_a = fig.add_subplot(grid[0, 0], projection="3d")
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

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
    fig.subplots_adjust(left=0.012, right=0.995, bottom=0.16, top=0.97)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, facecolor="white")
    plt.close(fig)
    png_path.chmod(0o644)

    print(f"Structure: {structure_path}")
    print(f"Representative volume: {len(molecules)} H2O molecules")
    print(f"Saved: {png_path}")
    return png_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a representative volume of a periodic GPUMD ice cell."
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=STRUCTURE_PATH,
        help="GPUMD extended-XYZ structure or restart file.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=PNG_PATH.with_suffix(""),
        help="Output path without the .png suffix.",
    )
    parser.add_argument(
        "--amorphous-structure",
        type=Path,
        help=(
            "If supplied, render a two-row Ih/LDA comparison using --structure "
            "as the hexagonal input."
        ),
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.amorphous_structure is None:
        plot_hexagonal_ice_structure(
            structure_path=arguments.structure,
            png_path=arguments.output_stem.with_suffix(".png"),
        )
    else:
        plot_ice_structure_comparison(
            hexagonal_path=arguments.structure,
            amorphous_path=arguments.amorphous_structure,
            png_path=arguments.output_stem.with_suffix(".png"),
        )
