#!/usr/bin/env python3
"""Collect the three-density NEP-MB-pol amorphous-ice compatibility scan."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
PHYSICS_ICE = HERE.parent
sys.path[:0] = [str(HERE), str(PHYSICS_ICE)]

from constants import (  # noqa: E402
    AASTEX_FULL_WIDTH_IN,
    FONT_COURIER,
    PAPER_FONTSIZE,
    RC_BASE_STANDARD,
    THREE_PANEL_ROW_HEIGHT_IN,
    rcparams_with_fontsize,
)
from bca.structure import iter_xyz_frames  # noqa: E402
from validate_amorphous_ice import (  # noqa: E402
    EXPERIMENTAL_LDA_DENSITY_G_CM3,
    LDA_FIRST_PEAK_RANGE_A_INV,
    STRUCTURE_FACTOR_Q_A_INV,
    _first_peak,
    _radial_distribution_and_structure_factor,
)
from validate_hexagonal_ice import (  # noqa: E402
    THERMO_COLUMNS,
    _orthorhombic_lengths,
    _read_thermo,
    _wrapped_positions,
    chill_plus,
    hac_mean_interval,
    partial_rdfs,
)


SEEDS = (1000, 2000, 3000)
DENSITIES_G_CM3 = (0.94, 0.98, 1.02)
TARGET_PRESSURE_GPA = 0.0001
CONFIDENCE_ALPHA = 0.05


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o664)


def _json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _configure_style() -> None:
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
    for key in ("rm", "cal", "sf", "tt"):
        plt.rcParams[f"mathtext.{key}"] = FONT_COURIER
    plt.rcParams["mathtext.it"] = f"{FONT_COURIER}:italic"
    plt.rcParams["mathtext.bf"] = f"{FONT_COURIER}:bold"
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


def _style_axes(axes: np.ndarray) -> None:
    for axis, label in zip(axes, ("(a)", "(b)", "(c)")):
        axis.minorticks_on()
        axis.tick_params(
            axis="both", which="both", direction="in",
            bottom=True, left=True, top=False, right=False,
            labelsize=PAPER_FONTSIZE,
        )
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
        for text in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            text.set_fontfamily(FONT_COURIER)
            text.set_fontsize(PAPER_FONTSIZE)
        axis.xaxis.label.set_fontfamily(FONT_COURIER)
        axis.yaxis.label.set_fontfamily(FONT_COURIER)
        axis.text(
            0.0, 1.04, label, transform=axis.transAxes,
            ha="left", va="bottom", clip_on=False,
            fontsize=PAPER_FONTSIZE, fontfamily=FONT_COURIER,
        )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".png", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    fig.savefig(temporary, dpi=300, facecolor="white", format="png")
    plt.close(fig)
    os.replace(temporary, path)
    path.chmod(0o664)


def _average_sampling_rdf(frames: list[object]) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    radii: np.ndarray | None = None
    for frame in frames:
        lengths = _orthorhombic_lengths(frame)
        positions = _wrapped_positions(frame.positions_angstrom, lengths)
        oxygen = positions[frame.species == "O"]
        hydrogen = positions[frame.species == "H"]
        rdf = partial_rdfs(oxygen, hydrogen, lengths, cKDTree(oxygen, boxsize=lengths))
        radii = rdf["r_a"]
        values.append(rdf["oo"])
    assert radii is not None
    return radii, np.mean(np.stack(values), axis=0)


def collect(run_root: Path, output_dir: Path, no_progress: bool) -> dict[str, object]:
    records: list[dict[str, object]] = []
    curves: dict[tuple[float, int], dict[str, np.ndarray]] = {}
    progress = tqdm(
        total=len(DENSITIES_G_CM3) * len(SEEDS),
        desc="Collecting density scan",
        unit="case",
        disable=no_progress,
    )
    for density in DENSITIES_G_CM3:
        density_tag = str(density).replace(".", "p")
        for seed in SEEDS:
            case_dir = run_root / f"rho_{density_tag}" / f"seed{seed}"
            if not (case_dir / "COMPLETED").is_file():
                raise FileNotFoundError(f"Incomplete density-scan case: {case_dir}")
            mapping = json.loads((case_dir / "cell_mapping.json").read_text())
            thermo = _read_thermo(case_dir / "sampling" / "thermo.out")
            if thermo.shape != (250, len(THERMO_COLUMNS)):
                raise ValueError(f"Unexpected thermo shape {thermo.shape}: {case_dir}")
            pressure = np.mean(thermo[:, 3:6], axis=1)
            pressure_interval = hac_mean_interval(pressure, CONFIDENCE_ALPHA)
            pressure_pass = bool(
                pressure_interval["confidence_interval"][0]
                <= TARGET_PRESSURE_GPA
                <= pressure_interval["confidence_interval"][1]
            )
            frames = list(iter_xyz_frames(case_dir / "sampling" / "dump.xyz"))
            if len(frames) != 10:
                raise ValueError(f"Expected 10 sampling frames in {case_dir}; got {len(frames)}")
            radii, mean_rdf = _average_sampling_rdf(frames)
            final = frames[-1]
            lengths = _orthorhombic_lengths(final)
            positions = _wrapped_positions(final.positions_angstrom, lengths)
            oxygen = positions[final.species == "O"]
            chill, _, _, _ = chill_plus(oxygen, lengths)
            full_r, full_rdf, structure_factor = _radial_distribution_and_structure_factor(
                oxygen, lengths
            )
            curves[(density, seed)] = {
                "r_a": radii,
                "g_oo": mean_rdf,
                "full_r_a": full_r,
                "full_g_oo": full_rdf,
                "s_oo": structure_factor,
            }
            first_peak = _first_peak(STRUCTURE_FACTOR_Q_A_INV, structure_factor)
            records.append(
                {
                    "density_g_cm3": density,
                    "seed": seed,
                    "case_directory": str(case_dir),
                    "measured_input_density_g_cm3": mapping["measured_density_g_cm3"],
                    "isotropic_length_scale": mapping["isotropic_length_scale"],
                    "temperature_k": {
                        "mean": float(np.mean(thermo[:, 0])),
                        "standard_deviation": float(np.std(thermo[:, 0], ddof=1)),
                    },
                    "mean_normal_pressure_gpa": pressure_interval,
                    "ambient_pressure_compatibility_pass": pressure_pass,
                    "chill_plus_ih_plus_ic_fraction": (
                        float(chill["hexagonal_fraction"])
                        + float(chill["cubic_fraction"])
                    ),
                    "oxygen_structure_factor_first_peak": first_peak,
                    "rdf_sampling_frame_rms_change": float(
                        np.sqrt(
                            np.mean(
                                (
                                    partial_rdfs(
                                        positions[final.species == "O"],
                                        positions[final.species == "H"],
                                        lengths,
                                        cKDTree(oxygen, boxsize=lengths),
                                    )["oo"]
                                    - mean_rdf
                                )
                                ** 2
                            )
                        )
                    ),
                }
            )
            progress.update()
    progress.close()

    report: dict[str, object] = {
        "schema_version": 1,
        "analysis": "nep_mbpol_amorphous_density_stress_compatibility_scan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "temperature_k": 80.0,
        "target_pressure_gpa": TARGET_PRESSURE_GPA,
        "densities_g_cm3": DENSITIES_G_CM3,
        "seeds": SEEDS,
        "cases": records,
        "decision_rule": (
            "Mechanical compatibility requires the HAC 95% confidence interval of "
            "mean normal pressure to contain 0.0001 GPa. Structural compatibility "
            "requires a separately provenance-locked experimental O-O/S(Q) table; "
            "no empirical curve fitting is performed here."
        ),
        "collision_ready": False,
        "limitations": [
            "These cells begin from isotropically mapped dense-glass configurations and test local mechanical/structural compatibility, not preparation-path independence.",
            "The 0.5 ns equilibration and 0.5 ns sampling blocks are a preliminary gate, not production LDA validation.",
            "The oxygen structure factor is a finite-cell Lorch transform and is not a neutron-weighted total experimental structure factor.",
            "An uncertainty-bearing experimental O-O reference table has not yet been ingested, so no quantitative O-O acceptance decision is made.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        output_dir / "amorphous_density_stress_scan.json",
        json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n",
    )
    csv_path = output_dir / "amorphous_density_stress_scan.csv"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=output_dir, delete=False
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "density_g_cm3", "seed", "pressure_mean_gpa",
                "pressure_ci_low_gpa", "pressure_ci_high_gpa",
                "ambient_pressure_compatibility_pass", "chill_ih_plus_ic_fraction",
                "first_s_oo_peak_a_inverse",
            )
        )
        for record in records:
            interval = record["mean_normal_pressure_gpa"]
            writer.writerow(
                (
                    record["density_g_cm3"], record["seed"], interval["mean"],
                    interval["confidence_interval"][0],
                    interval["confidence_interval"][1],
                    record["ambient_pressure_compatibility_pass"],
                    record["chill_plus_ih_plus_ic_fraction"],
                    record["oxygen_structure_factor_first_peak"]["q_a_inverse"],
                )
            )
        temporary = Path(handle.name)
    os.replace(temporary, csv_path)
    csv_path.chmod(0o664)
    _plot(output_dir / "amorphous_density_stress_scan.png", records, curves)
    return report


def _plot(
    path: Path,
    records: list[dict[str, object]],
    curves: dict[tuple[float, int], dict[str, np.ndarray]],
) -> None:
    _configure_style()
    density_colors = dict(
        zip(DENSITIES_G_CM3, plt.get_cmap("plasma")(np.linspace(0.0, 1.0, 4))[:3])
    )
    seed_styles = dict(zip(SEEDS, ("-", "--", ":")))
    fig, axes = plt.subplots(
        1, 3, figsize=(AASTEX_FULL_WIDTH_IN, THREE_PANEL_ROW_HEIGHT_IN)
    )
    for seed in SEEDS:
        selected = sorted(
            (record for record in records if record["seed"] == seed),
            key=lambda record: record["density_g_cm3"],
        )
        x = np.asarray([record["density_g_cm3"] for record in selected])
        y = 1000.0 * np.asarray(
            [record["mean_normal_pressure_gpa"]["mean"] for record in selected]
        )
        low = 1000.0 * np.asarray(
            [record["mean_normal_pressure_gpa"]["confidence_interval"][0] for record in selected]
        )
        high = 1000.0 * np.asarray(
            [record["mean_normal_pressure_gpa"]["confidence_interval"][1] for record in selected]
        )
        axes[0].plot(x, y, color="0.35", linestyle=seed_styles[seed], linewidth=0.8)
        for index, density in enumerate(x):
            axes[0].errorbar(
                density, y[index], yerr=[[y[index] - low[index]], [high[index] - y[index]]],
                color=density_colors[float(density)], marker="o", markersize=3.2,
                markerfacecolor=density_colors[float(density)], markeredgewidth=0.0,
                capsize=1.8, linewidth=0.7,
            )
    axes[0].axhline(0.1, color="black", linestyle="--", linewidth=0.8)
    for density in DENSITIES_G_CM3:
        for seed in SEEDS:
            curve = curves[(density, seed)]
            mask = curve["r_a"] <= 7.0
            axes[1].plot(
                curve["r_a"][mask], curve["g_oo"][mask],
                color=density_colors[density], linestyle=seed_styles[seed],
            )
            axes[2].plot(
                STRUCTURE_FACTOR_Q_A_INV, curve["s_oo"],
                color=density_colors[density], linestyle=seed_styles[seed],
            )
    axes[2].axvspan(*LDA_FIRST_PEAK_RANGE_A_INV, color="0.86", linewidth=0, zorder=0)
    axes[0].set(
        xlabel=r"Fixed density (g cm$^{-3}$)", ylabel="Mean pressure (MPa)"
    )
    axes[1].set(xlim=(2.2, 7.0), xlabel=r"$r$ ($\AA$)", ylabel=r"$g_{\rm OO}(r)$")
    axes[2].set(
        xlim=(0.8, 6.0), xlabel=r"$q$ ($\AA^{-1}$)", ylabel=r"Oxygen $S_{\rm OO}(q)$"
    )
    _style_axes(axes)
    legend_handles = [
        Line2D([], [], color=density_colors[density], label=rf"{density:.2f} g cm$^{{-3}}$")
        for density in DENSITIES_G_CM3
    ] + [
        Line2D([], [], color="0.35", linestyle=seed_styles[seed], label=f"Seed {seed}")
        for seed in SEEDS
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=6, frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        prop={"family": FONT_COURIER, "size": PAPER_FONTSIZE},
        handlelength=1.6, columnspacing=1.0, handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.077, right=0.992, bottom=0.285, top=0.91, wspace=0.38)
    _save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path,
        default=HERE / "runs" / "amorphous_density_stress_scan",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=(
            HERE.parent
            / "ice_structures"
            / "rejected_candidates"
            / "amorphous_lda_80K_candidate"
            / "density_stress_scan"
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    collect(args.run_root.resolve(), args.output_dir.resolve(), args.no_progress)


if __name__ == "__main__":
    main()
