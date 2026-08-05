#!/usr/bin/env python3
"""Validate the three completed 100 K NEP-MB-pol ice-Ih replicas.

The acceptance decision combines trajectory provenance, autocorrelation-aware
thermodynamic trend tests, the published CHILL+ definition of hexagonal ice,
the Bernal--Fowler ice rules, and explicit low-order ice-Ih Bragg maxima.  RDFs
and cross-replica summaries are retained as auditable numerical diagnostics.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import special, stats
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
PHYSICS_ICE = HERE.parent
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
from bca.structure import XYZFrame, file_sha256, iter_xyz_frames  # noqa: E402
from prepare_hexagonal_ice_experimental_cell import (  # noqa: E402
    experimental_cell,
)


SEEDS = (1000, 2000, 3000)
TARGET_TEMPERATURE_K = 100.0
SAMPLING_THERMO_RECORDS = 500
SAMPLING_FRAMES = 10
THERMO_INTERVAL_NS = 0.002
CHILL_NEIGHBOR_CUTOFF_A = 3.5
CHILL_MINIMUM_HEXAGONAL_FRACTION = 0.99
FAMILYWISE_ALPHA = 0.05
RDF_MAXIMUM_A = 8.0
RDF_BIN_WIDTH_A = 0.02
THERMO_COLUMNS = (
    "temperature_k",
    "kinetic_energy_ev",
    "potential_energy_ev",
    "pxx_gpa",
    "pyy_gpa",
    "pzz_gpa",
    "pyz_gpa",
    "pxz_gpa",
    "pxy_gpa",
    "lx_a",
    "ly_a",
    "lz_a",
)
TREND_COLUMNS = (
    "temperature_k",
    "potential_energy_ev",
    "pxx_gpa",
    "pyy_gpa",
    "pzz_gpa",
    "pyz_gpa",
    "pxz_gpa",
    "pxy_gpa",
)
REFLECTIONS = (
    (1, 0, 0),
    (0, 0, 2),
    (1, 0, 1),
    (1, 0, 2),
    (1, 1, 0),
    (1, 0, 3),
    (1, 1, 2),
)
REFERENCES = (
    {
        "purpose": "100 K ice-Ih unit-cell volume and c/a ratio",
        "citation": "Rottger et al., Acta Cryst. B 68, 91 (2012)",
        "doi": "10.1107/S0108768111046908",
    },
    {
        "purpose": "local hexagonal/cubic ice classification",
        "citation": "Nguyen and Molinero, J. Phys. Chem. B 119, 9369 (2015)",
        "doi": "10.1021/jp510289t",
    },
    {
        "purpose": "two-protons-per-water and one-proton-per-O--O-bond rules",
        "citation": "Bernal and Fowler, J. Chem. Phys. 1, 515 (1933)",
        "doi": "10.1063/1.1749327",
    },
    {
        "purpose": "experimental low-temperature ice-Ih structure",
        "citation": "Kuhs and Lehmann, J. Phys. Chem. 87, 4312 (1983)",
        "doi": "10.1021/j100244a063",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=HERE / "runs",
        help="Directory containing the three completed run directories.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=HERE / "structures" / "hexagonal_ih_100K_experimental",
        help="Directory containing manifest.json and compact final snapshots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to ARCHIVE_DIR/validation.",
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


def _decompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _read_thermo(path: Path) -> NDArray[np.float64]:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(THERMO_COLUMNS):
        raise ValueError(
            f"{path} has shape {values.shape}; expected N x {len(THERMO_COLUMNS)}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains non-finite values.")
    return values


def _newey_west_lag(record_count: int) -> int:
    return max(1, int(math.floor(4.0 * (record_count / 100.0) ** (2.0 / 9.0))))


def _hac_covariance(
    design: NDArray[np.float64], residual: NDArray[np.float64], lag: int
) -> NDArray[np.float64]:
    meat = (design * residual[:, None]).T @ (design * residual[:, None])
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1.0)
        right = design[offset:] * residual[offset:, None]
        left = design[:-offset] * residual[:-offset, None]
        cross = right.T @ left
        meat += weight * (cross + cross.T)
    bread = np.linalg.inv(design.T @ design)
    return bread @ meat @ bread


def hac_linear_trend(values: NDArray[np.float64], interval_ns: float) -> dict[str, float]:
    """Return an OLS slope with a Bartlett-kernel Newey--West uncertainty."""

    if np.ptp(values) == 0.0:
        return {
            "slope_per_ns": 0.0,
            "slope_standard_error_per_ns": 0.0,
            "raw_two_sided_p_value": 1.0,
            "newey_west_lag_records": _newey_west_lag(values.size),
        }
    time_ns = np.arange(values.size, dtype=np.float64) * interval_ns
    centered_time = time_ns - np.mean(time_ns)
    design = np.column_stack((np.ones(values.size), centered_time))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design @ coefficients
    lag = _newey_west_lag(values.size)
    covariance = _hac_covariance(design, residual, lag)
    standard_error = math.sqrt(max(float(covariance[1, 1]), 0.0))
    if standard_error == 0.0:
        p_value = 0.0 if coefficients[1] != 0.0 else 1.0
    else:
        p_value = 2.0 * stats.norm.sf(abs(float(coefficients[1])) / standard_error)
    return {
        "slope_per_ns": float(coefficients[1]),
        "slope_standard_error_per_ns": standard_error,
        "raw_two_sided_p_value": float(p_value),
        "newey_west_lag_records": lag,
    }


def hac_mean_interval(values: NDArray[np.float64], alpha: float) -> dict[str, float]:
    design = np.ones((values.size, 1), dtype=np.float64)
    mean = float(np.mean(values))
    residual = values - mean
    lag = _newey_west_lag(values.size)
    covariance = _hac_covariance(design, residual, lag)
    standard_error = math.sqrt(max(float(covariance[0, 0]), 0.0))
    critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return {
        "mean": mean,
        "standard_deviation": float(np.std(values, ddof=1)),
        "hac_standard_error": standard_error,
        "confidence_level": 1.0 - alpha,
        "confidence_interval": [
            mean - critical * standard_error,
            mean + critical * standard_error,
        ],
        "newey_west_lag_records": lag,
    }


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty(values.size, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (values.size - rank) * float(values[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def _orthorhombic_lengths(frame: XYZFrame) -> NDArray[np.float64]:
    diagonal = np.diag(np.diag(frame.lattice_angstrom))
    if not np.allclose(frame.lattice_angstrom, diagonal, rtol=0.0, atol=1.0e-10):
        raise ValueError("The validation implementation requires an orthorhombic cell.")
    return np.diag(frame.lattice_angstrom).copy()


def _wrapped_positions(
    positions: NDArray[np.float64], lengths: NDArray[np.float64]
) -> NDArray[np.float64]:
    return positions - np.floor(positions / lengths) * lengths


def chill_plus(
    oxygen: NDArray[np.float64], lengths: NDArray[np.float64]
) -> tuple[dict[str, object], NDArray[np.int64], NDArray[np.float64], cKDTree]:
    """Apply the published l=3 CHILL+ bulk-polymorph definitions."""

    tree = cKDTree(oxygen, boxsize=lengths)
    distances, indices = tree.query(oxygen, k=5)
    neighbors = np.asarray(indices[:, 1:], dtype=np.int64)
    neighbor_distances = np.asarray(distances[:, 1:], dtype=np.float64)
    displacement = oxygen[neighbors] - oxygen[:, None, :]
    displacement -= np.rint(displacement / lengths) * lengths
    radii = np.linalg.norm(displacement, axis=2)
    theta = np.arccos(np.clip(displacement[:, :, 2] / radii, -1.0, 1.0))
    phi = np.mod(np.arctan2(displacement[:, :, 1], displacement[:, :, 0]), 2.0 * np.pi)
    q_lm = np.stack(
        [
            np.mean(special.sph_harm_y(3, m, theta, phi), axis=1)
            for m in range(-3, 4)
        ],
        axis=1,
    )
    norms = np.linalg.norm(q_lm, axis=1)
    correlations = np.empty_like(neighbor_distances)
    for neighbor_slot in range(4):
        neighbor_index = neighbors[:, neighbor_slot]
        correlations[:, neighbor_slot] = (
            np.sum(q_lm * np.conjugate(q_lm[neighbor_index]), axis=1).real
            / (norms * norms[neighbor_index])
        )
    staggered = np.sum(correlations <= -0.8, axis=1)
    eclipsed = np.sum(
        (correlations >= -0.35) & (correlations <= 0.25), axis=1
    )
    coordination = tree.query_ball_point(
        oxygen, CHILL_NEIGHBOR_CUTOFF_A, return_length=True
    ) - 1
    four_neighbors = coordination == 4
    hexagonal = four_neighbors & (staggered == 3) & (eclipsed == 1)
    cubic = four_neighbors & (staggered == 4) & (eclipsed == 0)
    clathrate = four_neighbors & (staggered == 0) & (eclipsed == 4)
    metrics: dict[str, object] = {
        "water_molecules": int(oxygen.shape[0]),
        "hexagonal_count": int(np.count_nonzero(hexagonal)),
        "hexagonal_fraction": float(np.mean(hexagonal)),
        "cubic_count": int(np.count_nonzero(cubic)),
        "cubic_fraction": float(np.mean(cubic)),
        "clathrate_count": int(np.count_nonzero(clathrate)),
        "other_count": int(
            oxygen.shape[0]
            - np.count_nonzero(hexagonal)
            - np.count_nonzero(cubic)
            - np.count_nonzero(clathrate)
        ),
        "four_neighbor_count": int(np.count_nonzero(four_neighbors)),
        "coordination_minimum": int(np.min(coordination)),
        "coordination_maximum": int(np.max(coordination)),
        "correlation_minimum": float(np.min(correlations)),
        "correlation_maximum": float(np.max(correlations)),
        "four_nearest_oo_distance_a": {
            "minimum": float(np.min(neighbor_distances)),
            "mean": float(np.mean(neighbor_distances)),
            "maximum": float(np.max(neighbor_distances)),
        },
        "passes_published_bulk_ih_benchmark": bool(
            np.mean(hexagonal) >= CHILL_MINIMUM_HEXAGONAL_FRACTION
            and np.all(four_neighbors)
        ),
    }
    return metrics, neighbors, neighbor_distances, tree


def bernal_fowler_topology(
    oxygen: NDArray[np.float64],
    hydrogen: NDArray[np.float64],
    neighbors: NDArray[np.int64],
    oxygen_tree: cKDTree,
) -> tuple[dict[str, object], bytes | None]:
    """Test the two local Bernal--Fowler rules on the periodic O network."""

    oh_distances, oh_indices = oxygen_tree.query(hydrogen, k=2)
    donors = np.asarray(oh_indices[:, 0], dtype=np.int64)
    acceptors = np.asarray(oh_indices[:, 1], dtype=np.int64)
    donor_counts = np.bincount(donors, minlength=oxygen.shape[0])

    directed = np.column_stack(
        (np.repeat(np.arange(oxygen.shape[0]), 4), neighbors.reshape(-1))
    )
    oxygen_edges, directed_counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_counts=True
    )
    symmetric_network = bool(
        oxygen_edges.shape[0] == 2 * oxygen.shape[0]
        and np.all(directed_counts == 2)
    )

    hydrogen_edges = np.sort(np.column_stack((donors, acceptors)), axis=1)
    occupied_edges, proton_counts = np.unique(
        hydrogen_edges, axis=0, return_counts=True
    )
    donor_is_neighbor = np.any(neighbors[donors] == acceptors[:, None], axis=1)
    one_proton_per_edge = bool(
        symmetric_network
        and occupied_edges.shape == oxygen_edges.shape
        and np.array_equal(occupied_edges, oxygen_edges)
        and np.all(proton_counts == 1)
        and np.all(donor_is_neighbor)
    )
    two_protons_per_oxygen = bool(np.all(donor_counts == 2))
    passed = two_protons_per_oxygen and one_proton_per_edge

    orientation: bytes | None = None
    orientation_sha256: str | None = None
    if passed:
        edge_to_donor = {
            tuple(int(value) for value in edge): int(donor)
            for edge, donor in zip(hydrogen_edges, donors, strict=True)
        }
        bits = np.asarray(
            [edge_to_donor[tuple(edge)] == int(edge[0]) for edge in oxygen_edges],
            dtype=np.uint8,
        )
        orientation = bits.tobytes()
        orientation_sha256 = hashlib.sha256(orientation).hexdigest()

    return (
        {
            "two_nearest_oxygen_oh_distance_a": {
                "nearest_minimum": float(np.min(oh_distances[:, 0])),
                "nearest_mean": float(np.mean(oh_distances[:, 0])),
                "nearest_maximum": float(np.max(oh_distances[:, 0])),
                "second_nearest_minimum": float(np.min(oh_distances[:, 1])),
                "second_nearest_mean": float(np.mean(oh_distances[:, 1])),
                "second_nearest_maximum": float(np.max(oh_distances[:, 1])),
            },
            "oxygen_network_edges": int(oxygen_edges.shape[0]),
            "oxygen_network_is_symmetric_and_four_connected": symmetric_network,
            "oxygen_with_exactly_two_nearest_hydrogens": int(
                np.count_nonzero(donor_counts == 2)
            ),
            "hydrogen_bond_edges_with_exactly_one_proton": int(
                np.count_nonzero(proton_counts == 1)
            ),
            "all_hydrogens_link_nearest_oxygen_neighbors": bool(
                np.all(donor_is_neighbor)
            ),
            "orientation_sha256": orientation_sha256,
            "passes_bernal_fowler_rules": passed,
        },
        orientation,
    )


def partial_rdfs(
    oxygen: NDArray[np.float64],
    hydrogen: NDArray[np.float64],
    lengths: NDArray[np.float64],
    oxygen_tree: cKDTree,
) -> dict[str, NDArray[np.float64]]:
    edges = np.arange(
        0.0, RDF_MAXIMUM_A + RDF_BIN_WIDTH_A * 0.5, RDF_BIN_WIDTH_A
    )
    shell_volumes = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    volume = float(np.prod(lengths))
    hydrogen_tree = cKDTree(hydrogen, boxsize=lengths)

    oo_pairs = oxygen_tree.query_pairs(RDF_MAXIMUM_A, output_type="ndarray")
    oo_delta = oxygen[oo_pairs[:, 1]] - oxygen[oo_pairs[:, 0]]
    oo_delta -= np.rint(oo_delta / lengths) * lengths
    oo_distance = np.linalg.norm(oo_delta, axis=1)

    hh_pairs = hydrogen_tree.query_pairs(RDF_MAXIMUM_A, output_type="ndarray")
    hh_delta = hydrogen[hh_pairs[:, 1]] - hydrogen[hh_pairs[:, 0]]
    hh_delta -= np.rint(hh_delta / lengths) * lengths
    hh_distance = np.linalg.norm(hh_delta, axis=1)

    oh_distance = oxygen_tree.sparse_distance_matrix(
        hydrogen_tree, RDF_MAXIMUM_A, output_type="coo_matrix"
    ).data
    oo_counts = np.histogram(oo_distance, bins=edges)[0]
    oh_counts = np.histogram(oh_distance, bins=edges)[0]
    hh_counts = np.histogram(hh_distance, bins=edges)[0]
    oo_normalization = (
        0.5 * oxygen.shape[0] * (oxygen.shape[0] / volume) * shell_volumes
    )
    oh_normalization = (
        oxygen.shape[0] * (hydrogen.shape[0] / volume) * shell_volumes
    )
    hh_normalization = (
        0.5 * hydrogen.shape[0] * (hydrogen.shape[0] / volume) * shell_volumes
    )
    return {
        "r_a": 0.5 * (edges[1:] + edges[:-1]),
        "oo": oo_counts / oo_normalization,
        "oh": oh_counts / oh_normalization,
        "hh": hh_counts / hh_normalization,
    }


def _reflection_vector_sets(
    lengths: NDArray[np.float64], a_a: float, c_a: float
) -> dict[str, dict[str, object]]:
    spacing = 2.0 * np.pi / lengths
    result: dict[str, dict[str, object]] = {}
    for h, k, l_value in REFLECTIONS:
        q_target = 2.0 * np.pi * math.sqrt(
            (4.0 / 3.0) * (h * h + h * k + k * k) / (a_a * a_a)
            + l_value * l_value / (c_a * c_a)
        )
        limits = np.ceil(q_target / spacing).astype(int) + 1
        integer_vectors: set[tuple[int, int, int]] = set()
        tolerance = max(1.0e-8, q_target * 1.0e-8)
        for nx in range(-int(limits[0]), int(limits[0]) + 1):
            for ny in range(-int(limits[1]), int(limits[1]) + 1):
                remaining = (
                    q_target * q_target
                    - (nx * spacing[0]) ** 2
                    - (ny * spacing[1]) ** 2
                )
                if remaining < -tolerance:
                    continue
                nz_magnitude = math.sqrt(max(remaining, 0.0)) / spacing[2]
                nz_integer = int(round(nz_magnitude))
                for nz in {nz_integer, -nz_integer}:
                    vector = np.asarray((nx, ny, nz), dtype=np.float64) * spacing
                    if abs(float(np.linalg.norm(vector)) - q_target) <= tolerance:
                        integer_vectors.add((nx, ny, nz))
        if not integer_vectors:
            raise ValueError(f"No supercell reciprocal vectors found for {(h, k, l_value)}.")
        local_neighbors: set[tuple[int, int, int]] = set()
        for vector in integer_vectors:
            for axis in range(3):
                for direction in (-1, 1):
                    shifted = list(vector)
                    shifted[axis] += direction
                    local_neighbors.add(tuple(shifted))
        local_neighbors -= integer_vectors
        label = f"{h}{k}{l_value}"
        result[label] = {
            "hkl": [h, k, l_value],
            "q_a_inverse": q_target,
            "peak_vectors": np.asarray(sorted(integer_vectors), dtype=np.float64)
            * spacing,
            "neighbor_vectors": np.asarray(sorted(local_neighbors), dtype=np.float64)
            * spacing,
        }
    return result


def _normalized_intensities(
    oxygen: NDArray[np.float64], vectors: NDArray[np.float64]
) -> NDArray[np.float64]:
    phase = oxygen @ vectors.T
    amplitudes = np.mean(np.exp(1j * phase), axis=0)
    return np.abs(amplitudes) ** 2


def bragg_diagnostics(
    oxygen: NDArray[np.float64],
    vector_sets: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for label, specification in vector_sets.items():
        peak = _normalized_intensities(
            oxygen, np.asarray(specification["peak_vectors"])
        )
        neighbors = _normalized_intensities(
            oxygen, np.asarray(specification["neighbor_vectors"])
        )
        peak_mean = float(np.mean(peak))
        neighbor_mean = float(np.mean(neighbors))
        result[label] = {
            "hkl": specification["hkl"],
            "q_a_inverse": specification["q_a_inverse"],
            "symmetry_equivalent_vector_count": int(peak.size),
            "normalized_peak_intensity_mean": peak_mean,
            "normalized_peak_intensity_minimum": float(np.min(peak)),
            "adjacent_supercell_bin_intensity_mean": neighbor_mean,
            "adjacent_supercell_bin_intensity_maximum": float(np.max(neighbors)),
            "peak_to_adjacent_mean_ratio": peak_mean / neighbor_mean,
            "is_local_reciprocal_space_maximum": bool(peak_mean > neighbor_mean),
        }
    return result


def _frame_density(frame: XYZFrame) -> float:
    oxygen_count = int(np.count_nonzero(frame.species == "O"))
    volume_a3 = abs(float(np.linalg.det(frame.lattice_angstrom)))
    mass_g = oxygen_count * WATER_MOLAR_MASS_G_MOL / AVOGADRO_MOL_MINUS_ONE
    return mass_g / (volume_a3 * 1.0e-24)


def _lineage_checks(
    seed: int,
    run_dir: Path,
    archive_dir: Path,
    replica_manifest: dict[str, object],
) -> dict[str, object]:
    archive = archive_dir / str(replica_manifest["final_snapshot"])
    source_80k = HERE / "runs" / f"hexagonal_ih_80K_seed{seed}" / "restart.xyz"
    generated = HERE / "structures" / f"ice_ih_8x8x8_seed{seed}_melt_start.xyz"
    generated_metadata = generated.with_suffix(".json")
    generator_record = _read_json(generated_metadata)
    experimental_record = _read_json(run_dir / "experimental_cell.json")
    checks = {
        "dynamics_completion_marker": (run_dir / "DYNAMICS_COMPLETED").is_file(),
        "restart_matches_manifest": file_sha256(run_dir / "restart.xyz")
        == replica_manifest["final_snapshot_uncompressed_sha256"],
        "archive_compressed_hash_matches_manifest": file_sha256(archive)
        == replica_manifest["final_snapshot_gzip_sha256"],
        "archive_content_matches_restart": _decompressed_sha256(archive)
        == file_sha256(run_dir / "restart.xyz"),
        "source_80k_restart_matches_manifest": file_sha256(source_80k)
        == replica_manifest["source_80k_restart_sha256"],
        "source_80k_model_matches_seeded_genice_structure": file_sha256(
            source_80k.parent / "model.xyz"
        )
        == file_sha256(generated),
        "prepared_100k_cell_matches_manifest": file_sha256(run_dir / "model.xyz")
        == replica_manifest["prepared_cell_sha256"],
        "generated_structure_hash_matches_sidecar": file_sha256(generated)
        == generator_record.get("sha256"),
        "genice_seed_matches": generator_record.get("seed") == seed,
        "genice_strict_depolarization_recorded": generator_record.get(
            "depolarization"
        )
        == "strict",
        "genice_state_records_proton_disorder": "proton-disordered"
        in str(generator_record.get("state", "")),
        "experimental_source_matches_80k_restart": (
            isinstance(experimental_record.get("source"), dict)
            and experimental_record["source"].get("sha256")
            == file_sha256(source_80k)
        ),
        "potential_matches_manifest": file_sha256(run_dir / "nep.txt")
        == replica_manifest["potential_sha256"],
        "run_input_matches_manifest": file_sha256(run_dir / "run.in")
        == replica_manifest["run_input_sha256"],
        "explicit_velocity_seed_recorded": f"velocity        100 seed {seed}"
        in (run_dir / "run.in").read_text(encoding="utf-8"),
    }
    return {
        "checks": checks,
        "all_pass": bool(all(checks.values())),
        "input_sha256": {
            "dump_xyz": file_sha256(run_dir / "dump.xyz"),
            "thermo_out": file_sha256(run_dir / "thermo.out"),
            "restart_xyz": file_sha256(run_dir / "restart.xyz"),
            "archived_final_xyz_gz": file_sha256(archive),
        },
    }


def _write_rdf_csv(
    path: Path, radii: NDArray[np.float64], replica_rdfs: dict[int, dict[str, NDArray[np.float64]]]
) -> None:
    fields = ["r_a"]
    for pair in ("oo", "oh", "hh"):
        fields.extend([f"seed{seed}_{pair}" for seed in SEEDS])
        fields.extend([f"replica_mean_{pair}", f"replica_sd_{pair}"])
    rows: list[dict[str, float]] = []
    for index, radius in enumerate(radii):
        row: dict[str, float] = {"r_a": float(radius)}
        for pair in ("oo", "oh", "hh"):
            values = np.asarray(
                [replica_rdfs[seed][pair][index] for seed in SEEDS]
            )
            for seed, value in zip(SEEDS, values, strict=True):
                row[f"seed{seed}_{pair}"] = float(value)
            row[f"replica_mean_{pair}"] = float(np.mean(values))
            row[f"replica_sd_{pair}"] = float(np.std(values, ddof=1))
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_bragg_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "seed",
        "frame",
        "hkl",
        "q_a_inverse",
        "normalized_peak_intensity_mean",
        "adjacent_supercell_bin_intensity_mean",
        "peak_to_adjacent_mean_ratio",
        "is_local_reciprocal_space_maximum",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _configure_plot_style() -> None:
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


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.02,
        0.98,
        label,
        transform=axis.transAxes,
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


def _plot_validation(
    path: Path,
    thermo_by_seed: dict[int, NDArray[np.float64]],
    replica_rdfs: dict[int, dict[str, NDArray[np.float64]]],
    bragg_rows: list[dict[str, object]],
) -> None:
    _configure_plot_style()
    plasma_four = plt.get_cmap("plasma")(np.linspace(0.0, 1.0, 4))
    colors = dict(zip(SEEDS, plasma_four[:3], strict=True))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(AASTEX_FULL_WIDTH_IN, THREE_PANEL_ROW_HEIGHT_IN),
    )
    time = np.arange(SAMPLING_THERMO_RECORDS) * THERMO_INTERVAL_NS
    for seed in SEEDS:
        axes[0].plot(
            time,
            thermo_by_seed[seed][:, 0],
            color=colors[seed],
            linewidth=0.75,
            alpha=0.85,
            label=f"Seed {seed}",
        )
    axes[0].axhline(
        TARGET_TEMPERATURE_K,
        color="black",
        linewidth=0.8,
        linestyle="--",
    )
    axes[0].set(xlabel="Sampling time (ns)", ylabel="Temperature (K)")

    radii = replica_rdfs[SEEDS[0]]["r_a"]
    for seed in SEEDS:
        axes[1].plot(
            radii,
            replica_rdfs[seed]["oo"],
            color=colors[seed],
            linewidth=1.0,
            label=f"seed {seed}",
        )
    axes[1].axvline(
        CHILL_NEIGHBOR_CUTOFF_A,
        color="0.45",
        linewidth=0.8,
        linestyle=":",
    )
    axes[1].set(
        xlim=(2.2, 6.0),
        xlabel=r"$r$ ($\AA$)",
        ylabel=r"$g_{\mathrm{OO}}(r)$",
    )

    labels = ["".join(str(value) for value in hkl) for hkl in REFLECTIONS]
    x = np.arange(len(labels), dtype=float)
    width = 0.22
    for seed_index, seed in enumerate(SEEDS):
        ratios = []
        for label in labels:
            values = [
                float(row["peak_to_adjacent_mean_ratio"])
                for row in bragg_rows
                if row["seed"] == seed and row["hkl"] == label
            ]
            ratios.append(float(np.mean(values)))
        axes[2].bar(
            x + (seed_index - 1) * width,
            ratios,
            width,
            color=colors[seed],
            label=f"seed {seed}",
        )
    axes[2].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_yscale("log")
    axes[2].set_xticks(x, labels)
    axes[2].set(
        xlabel="Ice-Ih reflection (hkl)",
        ylabel="Peak / adjacent intensity",
    )
    for axis_index, (axis, label) in enumerate(
        zip(axes, ("(a)", "(b)", "(c)"), strict=True)
    ):
        axis.set_facecolor("white")
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
        if axis_index < 2:
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
        axis.xaxis.label.set_fontfamily(FONT_COURIER)
        axis.xaxis.label.set_fontsize(PAPER_FONTSIZE)
        axis.yaxis.label.set_fontfamily(FONT_COURIER)
        axis.yaxis.label.set_fontsize(PAPER_FONTSIZE)
        for tick_label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            tick_label.set_fontfamily(FONT_COURIER)
            tick_label.set_fontsize(PAPER_FONTSIZE)
        _panel_label(axis, label)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.006),
        prop={"family": FONT_COURIER, "size": PAPER_FONTSIZE},
        handlelength=1.6,
        columnspacing=1.6,
        handletextpad=0.45,
    )
    fig.subplots_adjust(
        left=0.078,
        right=0.992,
        bottom=0.285,
        top=0.975,
        wspace=0.38,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".png", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    fig.savefig(temporary, dpi=300, facecolor="white", format="png")
    plt.close(fig)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    archive_dir = args.archive_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else archive_dir / "validation"
    )
    manifest_path = archive_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    replica_manifest_by_seed = {
        int(record["structure_seed"]): {**record, "potential_sha256": manifest["potential_sha256"]}
        for record in manifest["replicas"]
    }
    target = experimental_cell(TARGET_TEMPERATURE_K)
    expected_lengths = np.asarray(
        [target["length_x_a"], target["length_y_a"], target["length_z_a"]]
    )
    expected_lattice = np.diag(expected_lengths)
    vector_sets = _reflection_vector_sets(
        expected_lengths,
        float(target["volume_constrained_a_a"]),
        float(target["volume_constrained_c_a"]),
    )

    report: dict[str, object] = {
        "schema_version": 1,
        "analysis": "hexagonal_ice_ih_100K_structural_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "references": REFERENCES,
        "protocol_constants": {
            "target_temperature_k": TARGET_TEMPERATURE_K,
            "sampling_thermo_records": SAMPLING_THERMO_RECORDS,
            "thermo_interval_ns": THERMO_INTERVAL_NS,
            "sampling_frames": SAMPLING_FRAMES,
            "chill_plus_l": 3,
            "chill_plus_staggered_c_maximum": -0.8,
            "chill_plus_eclipsed_c_range": [-0.35, 0.25],
            "chill_plus_neighbor_cutoff_a": CHILL_NEIGHBOR_CUTOFF_A,
            "published_chill_plus_bulk_ih_benchmark_fraction": CHILL_MINIMUM_HEXAGONAL_FRACTION,
            "trend_test": "two-sided OLS slope with Bartlett Newey-West HAC covariance",
            "multiple_testing": "Holm familywise correction",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "rdf_maximum_a": RDF_MAXIMUM_A,
            "rdf_bin_width_a": RDF_BIN_WIDTH_A,
        },
        "density_interpretation": (
            "The Rottger-derived density and cell are imposed NVT inputs. Their "
            "agreement is a provenance/integrity check, not independent structural evidence."
        ),
        "limitations": [
            "The trajectories are classical and do not include nuclear quantum effects.",
            "RDFs and reciprocal-space intensities are simulation diagnostics, not a new Rietveld refinement against raw experimental counts.",
            "Three independent proton configurations establish repeatability of the phase assignment but provide only a coarse ensemble uncertainty.",
        ],
        "replicas": {},
    }
    replica_results: dict[int, dict[str, object]] = {}
    thermo_by_seed: dict[int, NDArray[np.float64]] = {}
    replica_rdfs: dict[int, dict[str, NDArray[np.float64]]] = {}
    orientation_by_seed: dict[int, bytes] = {}
    trend_records: list[dict[str, object]] = []
    bragg_rows: list[dict[str, object]] = []

    progress = tqdm(
        total=len(SEEDS) * SAMPLING_FRAMES,
        desc="Validating saved 100 K frames",
        unit="frame",
        disable=args.no_progress,
    )
    try:
        for seed in SEEDS:
            run_dir = run_root / f"hexagonal_ih_100K_experimental_seed{seed}"
            replica_manifest = replica_manifest_by_seed[seed]
            lineage = _lineage_checks(seed, run_dir, archive_dir, replica_manifest)
            thermo = _read_thermo(run_dir / "thermo.out")
            record_count_pass = thermo.shape[0] == 2 * SAMPLING_THERMO_RECORDS
            sampling_thermo = thermo[-SAMPLING_THERMO_RECORDS:]
            thermo_by_seed[seed] = sampling_thermo
            thermodynamic_summary = {
                column_name: {
                    "mean": float(np.mean(sampling_thermo[:, column_index])),
                    "standard_deviation": float(
                        np.std(sampling_thermo[:, column_index], ddof=1)
                    ),
                }
                for column_index, column_name in enumerate(THERMO_COLUMNS)
            }
            thermodynamic_summary["mean_normal_stress_gpa"] = {
                "mean": float(np.mean(sampling_thermo[:, 3:6])),
                "standard_deviation": float(
                    np.std(np.mean(sampling_thermo[:, 3:6], axis=1), ddof=1)
                ),
            }
            temperature_interval = hac_mean_interval(
                sampling_thermo[:, 0], FAMILYWISE_ALPHA
            )
            temperature_target_pass = bool(
                temperature_interval["confidence_interval"][0]
                <= TARGET_TEMPERATURE_K
                <= temperature_interval["confidence_interval"][1]
            )
            trends: dict[str, dict[str, float]] = {}
            for column_name in TREND_COLUMNS:
                column_index = THERMO_COLUMNS.index(column_name)
                trend = hac_linear_trend(
                    sampling_thermo[:, column_index], THERMO_INTERVAL_NS
                )
                trends[column_name] = trend
                trend_records.append(
                    {"seed": seed, "observable": column_name, "trend": trend}
                )

            frames = list(iter_xyz_frames(run_dir / "dump.xyz"))
            frame_count_pass = len(frames) == 2 * SAMPLING_FRAMES
            sampling = frames[-SAMPLING_FRAMES:]
            frame_results: list[dict[str, object]] = []
            rdfs: list[dict[str, NDArray[np.float64]]] = []
            orientation_hashes: list[str] = []
            final_orientation: bytes | None = None
            for frame in sampling:
                lengths = _orthorhombic_lengths(frame)
                coordinates = _wrapped_positions(frame.positions_angstrom, lengths)
                oxygen = coordinates[frame.species == "O"]
                hydrogen = coordinates[frame.species == "H"]
                chill, neighbors, _, oxygen_tree = chill_plus(oxygen, lengths)
                topology, orientation = bernal_fowler_topology(
                    oxygen, hydrogen, neighbors, oxygen_tree
                )
                if orientation is not None:
                    orientation_hashes.append(hashlib.sha256(orientation).hexdigest())
                    final_orientation = orientation
                rdf = partial_rdfs(oxygen, hydrogen, lengths, oxygen_tree)
                rdfs.append(rdf)
                bragg = bragg_diagnostics(oxygen, vector_sets)
                for label, values in bragg.items():
                    bragg_rows.append(
                        {
                            "seed": seed,
                            "frame": frame.frame_index,
                            "hkl": label,
                            "q_a_inverse": values["q_a_inverse"],
                            "normalized_peak_intensity_mean": values[
                                "normalized_peak_intensity_mean"
                            ],
                            "adjacent_supercell_bin_intensity_mean": values[
                                "adjacent_supercell_bin_intensity_mean"
                            ],
                            "peak_to_adjacent_mean_ratio": values[
                                "peak_to_adjacent_mean_ratio"
                            ],
                            "is_local_reciprocal_space_maximum": values[
                                "is_local_reciprocal_space_maximum"
                            ],
                        }
                    )
                cell_matches = bool(
                    all(frame.pbc)
                    and np.allclose(
                        frame.lattice_angstrom,
                        expected_lattice,
                        rtol=0.0,
                        atol=5.0e-8,
                    )
                )
                frame_results.append(
                    {
                        "frame_index": frame.frame_index,
                        "density_g_cm3": _frame_density(frame),
                        "experimental_cell_matches_file_precision": cell_matches,
                        "chill_plus": chill,
                        "bernal_fowler": topology,
                        "bragg": bragg,
                    }
                )
                progress.update(1)
            if final_orientation is not None:
                orientation_by_seed[seed] = final_orientation
            replica_rdfs[seed] = {
                "r_a": rdfs[0]["r_a"],
                "oo": np.mean([item["oo"] for item in rdfs], axis=0),
                "oh": np.mean([item["oh"] for item in rdfs], axis=0),
                "hh": np.mean([item["hh"] for item in rdfs], axis=0),
            }
            structural_pass = bool(
                all(
                    item["experimental_cell_matches_file_precision"]
                    and item["chill_plus"]["passes_published_bulk_ih_benchmark"]
                    and item["bernal_fowler"]["passes_bernal_fowler_rules"]
                    and all(
                        reflection["is_local_reciprocal_space_maximum"]
                        for reflection in item["bragg"].values()
                    )
                    for item in frame_results
                )
            )
            topology_constant = len(set(orientation_hashes)) == 1
            replica_results[seed] = {
                "seed": seed,
                "lineage": lineage,
                "thermodynamic_records": int(thermo.shape[0]),
                "expected_thermodynamic_record_count_pass": record_count_pass,
                "sampling_thermodynamic_summary": thermodynamic_summary,
                "temperature": temperature_interval,
                "target_temperature_in_95pct_hac_interval": temperature_target_pass,
                "trends": trends,
                "saved_frames": len(frames),
                "expected_saved_frame_count_pass": frame_count_pass,
                "sampling_frame_indices": [frame.frame_index for frame in sampling],
                "sampling_frames": frame_results,
                "proton_orientation_constant_during_sampling": topology_constant,
                "proton_orientation_sha256": orientation_hashes[-1]
                if orientation_hashes
                else None,
                "all_frame_structural_checks_pass": structural_pass,
            }
    finally:
        progress.close()

    adjusted = holm_adjust(
        record["trend"]["raw_two_sided_p_value"] for record in trend_records
    )
    for record, adjusted_p in zip(trend_records, adjusted, strict=True):
        trend = record["trend"]
        trend["holm_adjusted_p_value"] = adjusted_p
        trend["significant_drift_at_familywise_alpha"] = adjusted_p < FAMILYWISE_ALPHA
    for seed in SEEDS:
        replica_results[seed]["no_resolved_thermodynamic_drift"] = not any(
            item["trend"]["significant_drift_at_familywise_alpha"]
            for item in trend_records
            if item["seed"] == seed
        )

    orientation_hashes_distinct = len(orientation_by_seed) == len(SEEDS) and len(
        {hashlib.sha256(value).hexdigest() for value in orientation_by_seed.values()}
    ) == len(SEEDS)
    pairwise_orientation_difference: dict[str, float] = {}
    for left_index, left_seed in enumerate(SEEDS):
        for right_seed in SEEDS[left_index + 1 :]:
            if left_seed not in orientation_by_seed or right_seed not in orientation_by_seed:
                continue
            left = np.frombuffer(orientation_by_seed[left_seed], dtype=np.uint8)
            right = np.frombuffer(orientation_by_seed[right_seed], dtype=np.uint8)
            pairwise_orientation_difference[f"{left_seed}-{right_seed}"] = float(
                np.mean(left != right)
            )

    for seed in SEEDS:
        item = replica_results[seed]
        item["accepted"] = bool(
            item["lineage"]["all_pass"]
            and item["expected_thermodynamic_record_count_pass"]
            and item["target_temperature_in_95pct_hac_interval"]
            and item["no_resolved_thermodynamic_drift"]
            and item["expected_saved_frame_count_pass"]
            and item["proton_orientation_constant_during_sampling"]
            and item["all_frame_structural_checks_pass"]
            and orientation_hashes_distinct
        )

    rdf_csv = output_dir / "hexagonal_ih_100K_partial_rdf.csv"
    bragg_csv = output_dir / "hexagonal_ih_100K_bragg.csv"
    figure_png = output_dir / "hexagonal_ih_100K_validation.png"
    _write_rdf_csv(rdf_csv, replica_rdfs[SEEDS[0]]["r_a"], replica_rdfs)
    _write_bragg_csv(bragg_csv, bragg_rows)
    _plot_validation(figure_png, thermo_by_seed, replica_rdfs, bragg_rows)

    potential_energy_means = {
        str(seed): float(np.mean(thermo_by_seed[seed][:, 2])) for seed in SEEDS
    }
    minimum_hexagonal_fraction = {
        str(seed): min(
            frame["chill_plus"]["hexagonal_fraction"]
            for frame in replica_results[seed]["sampling_frames"]
        )
        for seed in SEEDS
    }
    minimum_bragg_ratio = {
        str(seed): min(
            float(row["peak_to_adjacent_mean_ratio"])
            for row in bragg_rows
            if row["seed"] == seed
        )
        for seed in SEEDS
    }
    accepted = all(replica_results[seed]["accepted"] for seed in SEEDS)
    report["replicas"] = {str(seed): replica_results[seed] for seed in SEEDS}
    report["cross_replica"] = {
        "proton_orientation_hashes_are_distinct": orientation_hashes_distinct,
        "pairwise_proton_orientation_hamming_fraction": pairwise_orientation_difference,
        "sampling_potential_energy_mean_ev": potential_energy_means,
        "sampling_potential_energy_range_per_water_mev": 1000.0
        * (max(potential_energy_means.values()) - min(potential_energy_means.values()))
        / int(manifest["water_molecules"]),
        "minimum_chill_plus_hexagonal_fraction": minimum_hexagonal_fraction,
        "minimum_bragg_peak_to_adjacent_mean_ratio": minimum_bragg_ratio,
        "all_three_replicas_independently_pass": accepted,
    }
    report["acceptance"] = {
        "status": "accepted" if accepted else "rejected",
        "collision_ready": accepted,
        "criteria": [
            "complete checksum-linked GenIce2 -> 80 K -> 100 K lineage with strict GenIce2 depolarization",
            "1000 thermodynamic records and 20 saved frames, with the final halves constituting the documented 1 ns sampling block",
            "100 K lies inside each replica's 95% Newey-West HAC mean-temperature interval",
            "no Holm-significant linear drift among temperature, potential energy, or six stress components at familywise alpha=0.05",
            "every sampling frame meets the published CHILL+ >=99% bulk-Ih benchmark and has four O neighbors within 3.5 A",
            "every sampling frame satisfies both Bernal-Fowler ice rules exactly",
            "all seven low-order expected Ih reflections are local reciprocal-space intensity maxima in every sampling frame",
            "the three proton configurations are independently seeded, distinct, and topologically unchanged during sampling",
        ],
    }
    report["artifacts"] = {
        "partial_rdf_csv": {"path": rdf_csv.name, "sha256": file_sha256(rdf_csv)},
        "bragg_csv": {"path": bragg_csv.name, "sha256": file_sha256(bragg_csv)},
        "validation_png": {
            "path": figure_png.name,
            "sha256": file_sha256(figure_png),
        },
    }
    report_path = output_dir / "hexagonal_ih_100K_validation.json"
    _atomic_write(report_path, json.dumps(_json_value(report), indent=2) + "\n")
    print(f"Wrote {report_path}")
    print(f"Wrote {rdf_csv}")
    print(f"Wrote {bragg_csv}")
    print(f"Wrote {figure_png}")
    print(f"Validation status: {'ACCEPTED' if accepted else 'REJECTED'}")
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
