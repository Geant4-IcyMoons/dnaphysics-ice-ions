#!/usr/bin/env python3
"""Validate and convert the archived 80 K LDA EPSR collision structure."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
import platform
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
PHYSICS_ICE = HERE.parents[1]
NEP_MBPOL = PHYSICS_ICE / "nep_mbpol"
sys.path.insert(0, str(PHYSICS_ICE))
sys.path.insert(0, str(NEP_MBPOL))

from constants import (  # noqa: E402
    AASTEX_FULL_WIDTH_IN,
    FONT_COURIER,
    PAPER_FONTSIZE,
    RC_BASE_STANDARD,
    THREE_PANEL_ROW_HEIGHT_IN,
    rcparams_with_fontsize,
)
from ice_structures.epsr_lda80k.model import (  # noqa: E402
    EpsrAtoStructure,
    file_sha256,
    parse_ato,
    write_extended_xyz,
)
from nep_mbpol.bca.structure import iter_xyz_frames  # noqa: E402
from validate_hexagonal_ice import chill_plus  # noqa: E402


DEFAULT_RUN_ROOT = (
    PHYSICS_ICE
    / "process_evidence"
    / "ice_structures"
    / "validation"
    / "runs"
    / "epsr_lda80k"
)
DEFAULT_OUTPUT = HERE / "artifacts"
RDF_BIN_WIDTH_A = 0.03
RDF_MAXIMUM_A = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=DEFAULT_RUN_ROOT / "source" / "LDAneutron"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o644)


def _source_manifest() -> dict[str, Any]:
    value = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("source_manifest.json must contain one object.")
    return value


def _software_record(workers: int) -> dict[str, Any]:
    repository = PHYSICS_ICE.parents[1]
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    implementation_paths = (
        HERE / "model.py",
        HERE / "validate.py",
        HERE / "source_manifest.json",
        repository / "pbs" / "run_epsr_lda80k_validation.pbs",
    )
    font_directory = os.environ.get("PHYSICS_ICE_PLOT_FONT_DIR")
    font_manifest = Path(font_directory) / "manifest.json" if font_directory else None
    return {
        "repository_commit": commit,
        "implementation_sha256": {
            str(path.relative_to(repository)): file_sha256(path)
            for path in implementation_paths
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "host": socket.gethostname(),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "analysis_workers": workers,
        "plot_font_manifest_sha256": (
            file_sha256(font_manifest)
            if font_manifest is not None and font_manifest.is_file()
            else None
        ),
        "note": (
            "Per-file hashes identify the executed implementation even when the "
            "repository contains unrelated or uncommitted work."
        ),
    }


def _numeric_table(path: Path) -> NDArray[np.float64]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = stripped.split()
        try:
            rows.append([float(value) for value in values])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No numeric table was found in {path}.")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent numeric row widths in {path}.")
    return np.asarray(rows, dtype=np.float64)


def _epsr_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")

    def number(pattern: str, cast: type[int] | type[float] = float) -> int | float:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Could not parse {pattern!r} from {path}.")
        return cast(match.group(1).replace("D", "E"))

    r_factors = [
        float(value.replace("D", "E"))
        for value in re.findall(
            r"R-factor for data file\s+\d+:\s*([-+0-9.EeDd]+)", text
        )
    ]
    if len(r_factors) != 3:
        raise ValueError("The archived EPSR output must report three R factors.")
    return {
        "version": re.search(r"EPSR version\s+([^\n]+)", text).group(1).strip(),
        "accumulated_configurations": number(
            r"No\. of configurations in sum\s+(\d+)", int
        ),
        "chi_square": number(r"Chi-square\s+([-+0-9.EeDd]+)"),
        "quality_factor": number(r"Quality factor\s+([-+0-9.EeDd]+)"),
        "atomic_number_density_angstrom3": number(
            r"Number density\s+([-+0-9.EeDd]+)"
        ),
        "data_r_factors": r_factors,
    }


def _input_summary(path: Path) -> dict[str, Any]:
    records: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values or values[0].startswith("#"):
            continue
        records[values[0].lower()] = values[1:]
    try:
        return {
            "atomic_number_density_angstrom3": float(records["rho"][0]),
            "experimental_dataset_count": int(records["ndata"][0]),
            "accumulated_configurations": int(records["nsumt"][0]),
            "parallel_threads": int(records["num_threds"][0]),
            "potential_refinement_factor": float(records["potfac"][0]),
        }
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"Malformed EPSR input summary in {path}.") from exc


def _wrapped(structure: EpsrAtoStructure) -> NDArray[np.float64]:
    return np.mod(structure.positions_angstrom, structure.box_length_angstrom)


def _pair_rdf(
    pair: str,
    positions: NDArray[np.float64],
    molecule_indices: NDArray[np.int64],
    length: float,
) -> tuple[str, NDArray[np.float64], NDArray[np.float64]]:
    species_a, species_b = pair
    # O-H-H order is an invariant of the strict ATO parser.
    oxygen_indices = np.arange(0, positions.shape[0], 3, dtype=np.int64)
    hydrogen_indices = np.setdiff1d(
        np.arange(positions.shape[0], dtype=np.int64), oxygen_indices
    )
    first_indices = oxygen_indices if species_a == "O" else hydrogen_indices
    second_indices = oxygen_indices if species_b == "O" else hydrogen_indices
    first = positions[first_indices]
    second = positions[second_indices]
    first_molecules = molecule_indices[first_indices]
    second_molecules = molecule_indices[second_indices]
    first_tree = cKDTree(first, boxsize=length)
    second_tree = cKDTree(second, boxsize=length)

    if pair in {"OO", "HH"}:
        pairs = first_tree.query_pairs(RDF_MAXIMUM_A, output_type="ndarray")
        keep = first_molecules[pairs[:, 0]] != first_molecules[pairs[:, 1]]
        pairs = pairs[keep]
        displacement = first[pairs[:, 1]] - first[pairs[:, 0]]
        normalization_factor = 0.5 * first.shape[0] * (first.shape[0] / length**3)
    else:
        sparse = first_tree.sparse_distance_matrix(
            second_tree, RDF_MAXIMUM_A, output_type="coo_matrix"
        )
        keep = first_molecules[sparse.row] != second_molecules[sparse.col]
        displacement = second[sparse.col[keep]] - first[sparse.row[keep]]
        normalization_factor = first.shape[0] * (second.shape[0] / length**3)
    displacement -= np.rint(displacement / length) * length
    distances = np.linalg.norm(displacement, axis=1)
    edges = np.arange(0.0, RDF_MAXIMUM_A + RDF_BIN_WIDTH_A, RDF_BIN_WIDTH_A)
    counts = np.histogram(distances, bins=edges)[0]
    shells = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    return pair, 0.5 * (edges[1:] + edges[:-1]), counts / (normalization_factor * shells)


def _chill_metrics(oxygen: NDArray[np.float64], length: float) -> dict[str, Any]:
    metrics, _, _, _ = chill_plus(oxygen, np.repeat(length, 3))
    return metrics


def _geometry(structure: EpsrAtoStructure) -> dict[str, Any]:
    molecules = structure.positions_angstrom.reshape((-1, 3, 3))
    oh1 = np.linalg.norm(molecules[:, 1] - molecules[:, 0], axis=1)
    oh2 = np.linalg.norm(molecules[:, 2] - molecules[:, 0], axis=1)
    hh = np.linalg.norm(molecules[:, 2] - molecules[:, 1], axis=1)
    oh = np.concatenate((oh1, oh2))
    cosine = np.sum(
        (molecules[:, 1] - molecules[:, 0])
        * (molecules[:, 2] - molecules[:, 0]),
        axis=1,
    ) / (oh1 * oh2)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def summary(values: NDArray[np.float64]) -> dict[str, float]:
        return {
            "minimum": float(np.min(values)),
            "mean": float(np.mean(values)),
            "maximum": float(np.max(values)),
            "standard_deviation": float(np.std(values)),
        }

    return {
        "oh_distance_angstrom": summary(oh),
        "hh_distance_angstrom": summary(hh),
        "hoh_angle_degree": summary(angles),
    }


def _rdf_comparison(
    coordinate: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    archived: NDArray[np.float64],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, pair in enumerate(("OO", "OH", "HH")):
        radii, values = coordinate[pair]
        reference = np.interp(radii, archived[:, 0], archived[:, 1 + 2 * index])
        mask = (radii >= 1.2) & (radii <= RDF_MAXIMUM_A)
        result[pair] = {
            "r_range_angstrom": [1.2, RDF_MAXIMUM_A],
            "single_configuration_vs_1721_configuration_mean_rmse": float(
                np.sqrt(np.mean((values[mask] - reference[mask]) ** 2))
            ),
            "interpretation": "diagnostic only; no physical-accuracy threshold applied",
        }
    return result


def _fit_diagnostics(source_dir: Path, fitted: NDArray[np.float64]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, label in enumerate(("D2O", "HDO", "H2O")):
        measured = _numeric_table(source_dir / f"LDA_{label}.mdcs01")
        q = measured[:, 0]
        mask = (q >= fitted[:, 0].min()) & (q <= fitted[:, 0].max())
        interpolated = np.interp(q[mask], fitted[:, 0], fitted[:, 1 + 2 * index])
        residual = measured[mask, 1] - interpolated
        result[label] = {
            "points": int(np.count_nonzero(mask)),
            "q_range_angstrom_inverse": [float(q[mask].min()), float(q[mask].max())],
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mean_absolute_residual": float(np.mean(np.abs(residual))),
            "interpretation": (
                "independent file-alignment diagnostic; the archived EPSR R factor "
                "is the source calculation's fit statistic"
            ),
        }
    return result


def _write_rdf_csv(
    path: Path,
    coordinate: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    archived: NDArray[np.float64],
) -> None:
    rows = []
    for pair_index, pair in enumerate(("OO", "OH", "HH")):
        radii, values = coordinate[pair]
        reference = np.interp(radii, archived[:, 0], archived[:, 1 + 2 * pair_index])
        for radius, value, reference_value in zip(radii, values, reference, strict=True):
            rows.append((pair, radius, value, reference_value))
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("pair", "r_angstrom", "single_configuration_g", "epsr_mean_g"))
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o644)


def _configure_plot_style() -> None:
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
            "The four Nimbus Mono PS faces are required; stage them with "
            "stage_plot_fonts.py and set PHYSICS_ICE_PLOT_FONT_DIR."
        )
    plt.rcdefaults()
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
            overrides={"savefig.dpi": 300},
        )
    )


def _plot(
    path: Path,
    coordinate: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    archived_rdf: NDArray[np.float64],
    fitted: NDArray[np.float64],
    source_dir: Path,
    chill: dict[str, Any],
) -> None:
    _configure_plot_style()
    colors = plt.get_cmap("plasma")(np.linspace(0.0, 1.0, 4))[:3]
    fig, axes = plt.subplots(
        1, 3, figsize=(AASTEX_FULL_WIDTH_IN, THREE_PANEL_ROW_HEIGHT_IN)
    )
    for index, pair in enumerate(("OO", "OH", "HH")):
        axes[0].plot(
            archived_rdf[:, 0],
            archived_rdf[:, 1 + 2 * index],
            color=colors[index],
            label=pair,
        )
        radii, values = coordinate[pair]
        axes[0].plot(radii, values, color=colors[index], alpha=0.35, linewidth=0.7)
    for index, label in enumerate(("D2O", "HDO", "H2O")):
        measured = _numeric_table(source_dir / f"LDA_{label}.mdcs01")
        axes[1].errorbar(
            measured[:, 0],
            measured[:, 1],
            yerr=measured[:, 2],
            color=colors[index],
            linestyle="none",
            marker=".",
            markersize=1.3,
            elinewidth=0.25,
            alpha=0.7,
        )
        axes[1].plot(
            fitted[:, 0], fitted[:, 1 + 2 * index], color=colors[index], label=label
        )
    categories = ("Ih", "Ic", "Clath.", "Other")
    fractions = (
        float(chill["hexagonal_fraction"]),
        float(chill["cubic_fraction"]),
        float(chill["clathrate_count"]) / float(chill["water_molecules"]),
        float(chill["other_count"]) / float(chill["water_molecules"]),
    )
    axes[2].bar(categories, fractions, color=plt.get_cmap("plasma")(np.linspace(0, 1, 4)))
    axes[0].set(xlim=(1.2, 8.0), xlabel=r"$r$ ($\AA$)", ylabel=r"$g_{\alpha\beta}(r)$")
    axes[1].set(xlim=(0.4, 20.0), xlabel=r"$Q$ ($\AA^{-1}$)", ylabel="Neutron interference")
    axes[2].set(ylim=(0.0, 1.0), ylabel="CHILL+ fraction")
    for axis, label in zip(axes, ("(a)", "(b)", "(c)"), strict=True):
        axis.text(0.0, 1.04, label, transform=axis.transAxes, ha="left", va="bottom")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(top=False, right=False)
    axes[0].legend(frameon=False, prop={"family": FONT_COURIER, "size": PAPER_FONTSIZE})
    axes[1].legend(frameon=False, prop={"family": FONT_COURIER, "size": PAPER_FONTSIZE})
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.22, top=0.88, wspace=0.39)
    temporary = path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=300, facecolor="white")
    plt.close(fig)
    os.replace(temporary, path)
    path.chmod(0o644)


def validate(source_dir: Path, output_dir: Path, workers: int) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive.")
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _source_manifest()
    required = manifest["required_files"]
    if not isinstance(required, dict):
        raise ValueError("Malformed required-files manifest.")
    source_hashes = {name: file_sha256(source_dir / name) for name in required}
    hash_gate = all(source_hashes[name] == expected for name, expected in required.items())
    if not hash_gate:
        mismatched = [name for name, expected in required.items() if source_hashes[name] != expected]
        raise ValueError(f"Source checksum mismatch: {', '.join(mismatched)}")

    structure = parse_ato(source_dir / "LDA80K.ato")
    output_xyz = output_dir / "lda80k_epsr.xyz.gz"
    write_extended_xyz(structure, output_xyz)
    converted_frame = next(iter_xyz_frames(output_xyz))
    wrapped = _wrapped(structure)
    converted_wrapped = np.mod(
        converted_frame.positions_angstrom, structure.box_length_angstrom
    )
    conversion_maximum_error = float(np.max(np.abs(wrapped - converted_wrapped)))
    geometry = _geometry(structure)
    epsr_output = _epsr_summary(source_dir / "LDA80K.EPSR.out")
    epsr_input = _input_summary(source_dir / "LDA80K.EPSR.inp")

    futures = {}
    coordinate_rdfs: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    chill: dict[str, Any] | None = None
    with ProcessPoolExecutor(max_workers=min(workers, 4)) as executor:
        for pair in ("OO", "OH", "HH"):
            future = executor.submit(
                _pair_rdf,
                pair,
                wrapped,
                structure.molecule_indices,
                structure.box_length_angstrom,
            )
            futures[future] = pair
        future = executor.submit(
            _chill_metrics, wrapped[0::3], structure.box_length_angstrom
        )
        futures[future] = "CHILL+"
        for future in tqdm(as_completed(futures), total=len(futures), desc="structural checks"):
            label = futures[future]
            value = future.result()
            if label == "CHILL+":
                chill = value
            else:
                pair, radii, rdf = value
                coordinate_rdfs[pair] = (radii, rdf)
    if chill is None:
        raise RuntimeError("CHILL+ task did not return a result.")

    archived_rdf = _numeric_table(source_dir / "LDA80K.EPSR.g01")
    fitted = _numeric_table(source_dir / "LDA80K.EPSR.t01")
    geometry_gate = (
        structure.molecule_count == 3000
        and structure.species.size == 9000
        and math.isclose(structure.temperature_k, 80.0, rel_tol=0.0, abs_tol=1.0e-10)
        and all(
            math.isfinite(float(value))
            for distribution in geometry.values()
            for value in distribution.values()
        )
    )
    density_gate = abs(
        structure.atomic_number_density_angstrom3
        - float(epsr_input["atomic_number_density_angstrom3"])
    ) < 1.0e-8
    conversion_gate = conversion_maximum_error < 5.0e-10
    fit_artifact_gate = (
        int(epsr_input["experimental_dataset_count"]) == 3
        and int(epsr_input["accumulated_configurations"])
        == int(epsr_output["accumulated_configurations"])
        and int(epsr_output["accumulated_configurations"]) > 0
        and len(epsr_output["data_r_factors"]) == 3
        and all(math.isfinite(value) for value in epsr_output["data_r_factors"])
    )
    accepted = hash_gate and geometry_gate and density_gate and conversion_gate and fit_artifact_gate
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "accepted" if accepted else "rejected",
        "decision": (
            "accepted_exact_published_epsr_lda80k_structure_for_static_collision_sampling"
            if accepted
            else "rejected_for_collision_sampling"
        ),
        "scope": (
            "Exact conversion and structural audit of the archived published EPSR model; "
            "not a new EPSR refinement, thermodynamic trajectory, or independent replica."
        ),
        "source": {
            "manifest": manifest,
            "source_directory": str(source_dir),
            "verified_sha256": source_hashes,
        },
        "software": _software_record(min(workers, 4)),
        "gates": {
            "exact_source_checksums": hash_gate,
            "ato_water_geometry_and_state": geometry_gate,
            "ato_density_matches_epsr_input": density_gate,
            "lossless_extended_xyz_conversion": conversion_gate,
            "archived_three_dataset_fit_is_complete": fit_artifact_gate,
        },
        "structure": {
            "temperature_k": structure.temperature_k,
            "water_molecules": structure.molecule_count,
            "atoms": int(structure.species.size),
            "box_length_angstrom": structure.box_length_angstrom,
            "density_g_cm3": structure.density_g_cm3,
            "molecular_number_density_angstrom3": structure.molecular_number_density_angstrom3,
            "atomic_number_density_angstrom3": structure.atomic_number_density_angstrom3,
            "geometry": geometry,
            "intramolecular_restraint_targets_angstrom": {
                "OH": 0.976,
                "HH": 1.55,
                "note": (
                    "Verified in every ATO connectivity record. EPSR uses harmonic "
                    "restraints, so instantaneous distances are distributed around "
                    "these targets rather than fixed exactly."
                ),
            },
            "converted_path": str(output_xyz),
            "converted_sha256": file_sha256(output_xyz),
            "maximum_conversion_coordinate_error_angstrom": conversion_maximum_error,
        },
        "archived_epsr_calculation": {"input": epsr_input, "output": epsr_output},
        "independent_diagnostics": {
            "rdf": _rdf_comparison(coordinate_rdfs, archived_rdf),
            "neutron_fit_file_alignment": _fit_diagnostics(source_dir, fitted),
            "chill_plus": chill,
            "note": (
                "These diagnostics are reported without an invented acceptance cutoff. "
                "Physical phase provenance and diffraction agreement come from the exact "
                "published EPSR artifact and its archived fit statistics."
            ),
        },
        "limitations": [
            "The archive supplies one retained coordinate realization, not three independent replicas.",
            "EPSR Monte Carlo configurations are static structural models, not an 80 K dynamics trajectory.",
            "The archived EPSR pressure estimator is not used as a thermodynamic-pressure gate.",
            "The structure constrains phase geometry; it does not validate NLH or soft-DFT collision physics.",
        ],
        "workers": min(workers, 4),
    }
    report_path = output_dir / "epsr_lda80k_validation.json"
    _atomic_write_text(report_path, json.dumps(report, indent=2) + "\n")
    _write_rdf_csv(output_dir / "epsr_lda80k_partial_rdf.csv", coordinate_rdfs, archived_rdf)
    _plot(
        output_dir / "epsr_lda80k_validation.png",
        coordinate_rdfs,
        archived_rdf,
        fitted,
        source_dir,
        chill,
    )
    if not accepted:
        raise RuntimeError("EPSR LDA 80 K validation gates failed; see JSON report.")
    return report


def main() -> None:
    args = parse_args()
    report = validate(args.source_dir, args.output_dir, args.workers)
    print(json.dumps({"status": report["status"], "structure": report["structure"]}, indent=2))


if __name__ == "__main__":
    main()
