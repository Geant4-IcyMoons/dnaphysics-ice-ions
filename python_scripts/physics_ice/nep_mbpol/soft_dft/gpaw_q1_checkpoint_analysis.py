"""Post-process final GPAW q=1 wavefunctions without rerunning SCF."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .gpaw_prepared_q1 import (
    _initialize_combined_calculator,
    _metric_normalize,
    PreparedQ1Settings,
    carbon_ao_indices,
    select_carbon_p_components,
    sha256_file,
)


PURE_COMPONENTS = (
    "p_parallel",
    "p_perpendicular_in_plane",
    "p_perpendicular_normal",
)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".next")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _task_by_component(run_root: Path) -> dict[str, Path]:
    found = {}
    for result_path in sorted((run_root / "tasks").glob("*/result.json")):
        result = json.loads(result_path.read_text())
        component = str(result["component"])
        if not result.get("numerical_gate_passed"):
            raise RuntimeError(f"Checkpoint did not pass its numerical gate: {result_path}")
        found[component] = result_path.parent
    missing = set(PURE_COMPONENTS) - set(found)
    if missing:
        raise RuntimeError(f"Missing pure components: {sorted(missing)}")
    return found


def _raw_checkpoint(path: Path) -> dict[str, np.ndarray]:
    from ase.io.ulm import open as ulm_open

    reader = ulm_open(path)
    return {
        "coefficients": np.asarray(reader.wave_functions.coefficients)[:, 0],
        "occupations": np.asarray(reader.wave_functions.occupations)[:, 0],
        "density": np.asarray(reader.density.density),
        "positions": np.asarray(reader.atoms.positions),
        "cell": np.asarray(reader.atoms.cell),
    }


def _s2(coefficients: np.ndarray, occupations: np.ndarray, overlap: np.ndarray) -> float:
    """Return determinant <S^2>; requires integer occupations."""

    if not np.allclose(occupations, np.rint(occupations), atol=1.0e-12):
        raise ValueError("Determinant S^2 is undefined for fractional occupations.")
    alpha = coefficients[0][occupations[0] > 0.5]
    beta = coefficients[1][occupations[1] > 0.5]
    nalpha, nbeta = len(alpha), len(beta)
    sz = 0.5 * (nalpha - nbeta)
    cross = alpha.conj() @ overlap @ beta.T
    return float(sz * (sz + 1.0) + nbeta - np.sum(np.abs(cross) ** 2))


def _normalize_orbitals(coefficients: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    normalized = np.empty_like(coefficients)
    for spin in range(coefficients.shape[0]):
        for band in range(coefficients.shape[1]):
            normalized[spin, band] = _metric_normalize(coefficients[spin, band], overlap)
    return normalized


def _orbital_density(calculator: object, coefficients: np.ndarray) -> np.ndarray:
    grid = calculator.wfs.gd.empty(dtype=float)
    calculator.wfs.basis_functions.lcao_to_grid(coefficients, grid, q=-1)
    density = np.abs(grid) ** 2
    density /= float(np.sum(density))
    return density


def analyze_checkpoints(run_root: Path, output: Path) -> dict[str, object]:
    from ase import Atoms
    from ase.io import write
    from gpaw import GPAW

    run_root = run_root.resolve()
    tasks = _task_by_component(run_root)
    manifest = json.loads((run_root / "manifest.json").read_text())
    task0 = manifest["tasks"][0]
    settings = PreparedQ1Settings(**task0["settings"])
    raw = {name: _raw_checkpoint(root / "final.gpw") for name, root in tasks.items()}

    coordinates = task0["geometry"]["coordinates_angstrom"]
    atoms = Atoms(
        symbols=[row[0] for row in coordinates],
        positions=[[float(x) for x in row[1:]] for row in coordinates],
        pbc=False,
    )
    atoms.center(vacuum=settings.vacuum_angstrom)
    atoms.set_initial_magnetic_moments([1.0, 0.0, 0.0, 0.0])
    calculator = _initialize_combined_calculator(atoms, settings, output.with_suffix(".log"))
    overlap = np.asarray(calculator.wfs.S_qMM[0])

    # Use one immutable isolated-C checkpoint to define a common x/y/z frame.
    reference_task = tasks["p_parallel"]
    carbon = GPAW(str(reference_task / "carbon_fragment.gpw"), txt=None)
    carbon.set_positions(carbon.atoms)
    local_refs = select_carbon_p_components(carbon)
    carbon_nao = int(carbon.wfs.setups.nao)
    fixed_refs = {}
    for axis, vector in local_refs.items():
        embedded = np.zeros(overlap.shape[0], dtype=vector.dtype)
        embedded[:carbon_nao] = vector
        fixed_refs[axis] = _metric_normalize(embedded, overlap)

    overlaps = np.empty((3, 3))
    s2_values = {}
    orbital_products = {}
    array_products = {}
    for row, component in enumerate(PURE_COMPONENTS):
        data = raw[component]
        coeff = _normalize_orbitals(data["coefficients"], overlap)
        occ = data["occupations"]
        # The singly occupied orbital is the occupied alpha state with the
        # smallest projection onto the occupied beta space.
        alpha_indices = np.flatnonzero(occ[0] > 0.5)
        beta = coeff[1][occ[1] > 0.5]
        scores = []
        for index in alpha_indices:
            coupling = beta.conj() @ overlap @ coeff[0, index]
            scores.append(float(np.sum(np.abs(coupling) ** 2)))
        somo_index = int(alpha_indices[int(np.argmin(scores))])
        somo = coeff[0, somo_index]
        for col, axis in enumerate(("x", "y", "z")):
            overlaps[row, col] = float(abs(somo.conj() @ overlap @ fixed_refs[axis]) ** 2)
        s2_values[component] = _s2(coeff, occ, overlap)
        somo_density = _orbital_density(calculator, somo)
        carbon_2p_density = sum(
            _orbital_density(calculator, fixed_refs[axis]) for axis in ("x", "y", "z")
        ) / 3.0
        array_products[f"somo_density_{component}"] = somo_density
        array_products[f"carbon_2p_density_{component}"] = carbon_2p_density
        orbital_products[component] = {
            "somo_band": somo_index,
            "somo_density_integral_normalized_to_electron": 1.0,
            "carbon_2p_reference_density_definition": "equal mean of fixed px/py/pz densities",
        }

    density_pairs = {}
    volume = abs(float(np.linalg.det(raw[PURE_COMPONENTS[0]]["cell"])))
    grid_shape = raw[PURE_COMPONENTS[0]]["density"].shape[1:]
    dv = volume / float(np.prod(grid_shape))
    for i, left in enumerate(PURE_COMPONENTS):
        rho_left = np.sum(raw[left]["density"], axis=0)
        for right in PURE_COMPONENTS[i + 1 :]:
            delta = rho_left - np.sum(raw[right]["density"], axis=0)
            density_pairs[f"{left}__{right}"] = {
                "integrated_absolute_difference_electrons": float(np.sum(abs(delta)) * dv),
                "rms_difference_electrons_per_angstrom3": float(np.sqrt(np.mean(delta**2))),
                "maximum_absolute_difference_electrons_per_angstrom3": float(np.max(abs(delta))),
            }

    output.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        overlap_squared=overlaps,
        components=np.asarray(PURE_COMPONENTS),
        references=np.asarray(("p_x", "p_y", "p_z")),
        **{f"density_{name}": raw[name]["density"] for name in PURE_COMPONENTS},
        **array_products,
    )
    result = {
        "schema_version": 1,
        "analysis_kind": "immutable_final_checkpoint_analysis",
        "electronic_iterations_performed": False,
        "source_run": str(run_root),
        "source_final_gpw_sha256": {
            name: sha256_file(root / "final.gpw") for name, root in tasks.items()
        },
        "overlap_squared_rows": list(PURE_COMPONENTS),
        "overlap_squared_columns": ["p_x", "p_y", "p_z"],
        "overlap_squared_3x3": overlaps.tolist(),
        "pairwise_total_pseudo_density_differences": density_pairs,
        "total_s2": s2_values,
        "orbital_diagnostics": orbital_products,
        "array_products": str(npz_path.resolve()),
        "acceptance_summary": {
            "all_pure_checkpoints_analyzed": True,
            "s2_expected_doublet": 0.75,
            "physical_state_status": "validation_pending",
        },
    }
    _atomic_json(output, result)
    return result
