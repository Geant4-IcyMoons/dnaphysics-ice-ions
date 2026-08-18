"""Constrained-first GPAW DO-MOM diagnostics for the carbon q=1 manifold.

This is a bounded state-generation experiment.  It prepares isolated C+ and
H2O fragments in the same cell and atom-centred dzp basis, embeds their AO
coefficient blocks into the corresponding full-system basis, assembles a
orbital density matrix with an explicit carbon 2p component, installs GPAW's
cDFT potential, restores that density matrix, and only then permits the first
complex electronic iteration.  Results are diagnostics, never production
soft-collision surfaces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import traceback
from typing import Mapping, Sequence

import numpy as np

from .geometry import ORIENTATIONS, build_scan_geometry


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = 1
REVIEWED_GPAW_VERSION = "25.7.0"
Q1_COMPONENTS = (
    "p_parallel",
    "p_perpendicular_in_plane",
    "p_perpendicular_normal",
    "p_fractional_ensemble",
)
COMPONENT_DIRECTIONS = {
    "p_parallel": (0.0, 0.0, 1.0),
    "p_perpendicular_in_plane": (1.0, 0.0, 0.0),
    "p_perpendicular_normal": (0.0, 1.0, 0.0),
}
SPHERICAL_P_ORDER = ("y", "z", "x")
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW_SOURCES = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parent / "geometry.py",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def signature(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".next")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _source_records() -> dict[str, dict[str, object]]:
    records = {}
    for source in WORKFLOW_SOURCES:
        resolved = source.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing prepared-state source: {resolved}")
        records[str(resolved.relative_to(REPOSITORY_ROOT))] = {
            "sha256": sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    return records


@dataclass(frozen=True)
class PreparedQ1Settings:
    gpaw_version: str = REVIEWED_GPAW_VERSION
    separation_angstrom: float = 12.0
    orientation: str = "oxygen_back"
    mode: str = "lcao"
    basis: str = "dzp"
    xc: str = "PBE"
    grid_spacing_angstrom: float = 0.20
    vacuum_angstrom: float = 6.0
    fragment_energy_tolerance_ev_per_electron: float = 1.0e-6
    fragment_eigenstate_tolerance_ev2_per_electron: float = 1.0e-8
    direct_gradient_tolerance_ev_per_electron: float = 1.0e-7
    maximum_scf_iterations: int = 333
    charge_coefficient_hartree: float = 0.1
    spin_coefficient_hartree: float = 0.1
    constraint_tolerance_electrons: float = 0.01
    multiplier_bound_ev: float = 100.0
    maximum_multiplier_iterations: int = 200
    optimizer_tolerance: float = 1.0e-10
    optimizer_ftol: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.gpaw_version != REVIEWED_GPAW_VERSION:
            raise ValueError("Only GPAW 25.7.0 has been reviewed for this pilot.")
        if self.orientation != "oxygen_back":
            raise ValueError("The bounded q=1 experiment is oxygen-back only.")
        if self.mode != "lcao" or self.basis != "dzp" or self.xc != "PBE":
            raise ValueError("The reviewed preparation requires LCAO/dzp/PBE.")
        for name in (
            "separation_angstrom",
            "grid_spacing_angstrom",
            "vacuum_angstrom",
            "fragment_energy_tolerance_ev_per_electron",
            "fragment_eigenstate_tolerance_ev2_per_electron",
            "direct_gradient_tolerance_ev_per_electron",
            "charge_coefficient_hartree",
            "spin_coefficient_hartree",
            "constraint_tolerance_electrons",
            "multiplier_bound_ev",
            "optimizer_tolerance",
            "optimizer_ftol",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.maximum_scf_iterations < 1 or self.maximum_multiplier_iterations < 1:
            raise ValueError("Iteration limits must be positive.")


DEFAULT_SETTINGS = PreparedQ1Settings()


def build_multiplier_calibration_manifest(
    output_root: Path, reference_task_root: Path,
    retention_bundle_path: Path | None = None,
    gmf_anchor_result_path: Path | None = None,
) -> Path:
    """Plan the five requested pz outer-root calibration trials."""

    trials = (
        ("charge_only_005", "charge_only", 0.05, None),
        ("charge_only_015", "charge_only", 0.15, None),
        ("charge_spin_005_005", "charge_spin", 0.05, 0.05),
        ("charge_spin_015_015", "charge_spin", 0.15, 0.15),
        ("charge_spin_005_015", "charge_spin", 0.05, 0.15),
    )
    source_records = _source_records()
    reference_task_root = reference_task_root.resolve()
    reference_files = {
        name: {
            "path": str((reference_task_root / name).resolve()),
            "sha256": sha256_file(reference_task_root / name),
        }
        for name in ("prepared_complex.gpw", "carbon_fragment.gpw", "water_fragment.gpw")
    }
    retention_bundle = None
    if retention_bundle_path is not None:
        retention_bundle_path = retention_bundle_path.resolve()
        retention_bundle = json.loads(retention_bundle_path.read_text())
        claimed = retention_bundle.pop("bundle_signature")
        if signature(retention_bundle) != claimed:
            raise RuntimeError("Retention bundle signature mismatch.")
        retention_bundle["bundle_signature"] = claimed
        retention_bundle["manifest_path"] = str(retention_bundle_path)
        retention_result_path = retention_bundle_path.with_name("result.json")
        retention_result = json.loads(retention_result_path.read_text())
        if not (retention_result.get("tight_inner_converged")
                and retention_result.get("retention_gate_passed")):
            raise RuntimeError("Retention bundle lacks a passed retention result.")
        retention_bundle["validation_result"] = {
            "path": str(retention_result_path),
            "sha256": sha256_file(retention_result_path),
        }
    gmf_anchor = None
    if gmf_anchor_result_path is not None:
        gmf_anchor_result_path = gmf_anchor_result_path.resolve()
        gmf_anchor = json.loads(gmf_anchor_result_path.read_text())
        if not (gmf_anchor.get("gmf_converged")
                and gmf_anchor.get("gmf_anchor_gate_passed")
                and int(gmf_anchor.get("saddle_order", 0)) > 0):
            raise RuntimeError("GMF calibration requires a passed GMF anchor.")
        gmf_anchor = {
            "path": str(gmf_anchor_result_path),
            "sha256": sha256_file(gmf_anchor_result_path),
            "saddle_order": int(gmf_anchor["saddle_order"]),
            "method": gmf_anchor["method"],
        }
    tasks = []
    for index, (name, mode, charge_start, spin_start) in enumerate(trials):
        settings = PreparedQ1Settings(
            charge_coefficient_hartree=charge_start,
            spin_coefficient_hartree=charge_start if spin_start is None else spin_start,
            constraint_tolerance_electrons=1.0e-5,
            maximum_multiplier_iterations=1000,
        )
        orientation = next(item for item in ORIENTATIONS if item.name == settings.orientation)
        geometry = build_scan_geometry(orientation, settings.separation_angstrom, "C")
        identity = {
            "schema_version": SCHEMA_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "component": "p_parallel",
            "calibration_trial": name,
            "constraint_mode": mode,
            "charge": 1,
            "geometry": geometry.as_dict(),
            "settings": asdict(settings),
            "sources": source_records,
            "reference_prepared_state": reference_files,
            "validated_retention_bundle": retention_bundle,
            "validated_gmf_anchor": gmf_anchor,
        }
        tasks.append({
            **identity,
            "task_index": index,
            "task_id": f"c_q1_pz_{name}_oxygen_back_r12A",
            "task_signature": signature(identity),
            "scientific_status": "multiplier_calibration_diagnostic",
            "production_eligible": False,
        })
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "task_count": len(tasks),
        "tasks": tasks,
        "sources": source_records,
        "parallel_model": "five independent fresh pz preparations",
        "production_eligible": False,
    }
    manifest = {**unsigned, "manifest_signature": signature(unsigned)}
    path = output_root.resolve() / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(f"Refusing to overwrite incompatible manifest: {path}")
        return path
    _atomic_json(path, manifest)
    return path


def build_manifest(
    output_root: Path,
    settings: PreparedQ1Settings = DEFAULT_SETTINGS,
) -> Path:
    orientation = next(item for item in ORIENTATIONS if item.name == settings.orientation)
    geometry = build_scan_geometry(orientation, settings.separation_angstrom, "C")
    source_records = _source_records()
    tasks = []
    for index, component in enumerate(Q1_COMPONENTS):
        identity = {
            "schema_version": SCHEMA_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "component": component,
            "charge": 1,
            "geometry": geometry.as_dict(),
            "settings": asdict(settings),
            "sources": source_records,
        }
        tasks.append(
            {
                **identity,
                "task_index": index,
                "task_id": f"c_q1_{component}_oxygen_back_r12A",
                "task_signature": signature(identity),
                "scientific_status": "state_preparation_diagnostic",
                "production_eligible": False,
            }
        )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "task_count": len(tasks),
        "tasks": tasks,
        "sources": source_records,
        "parallel_model": (
            "one independent q=1 component per PBS array task; eight MPI ranks "
            "inside each GPAW task"
        ),
        "production_eligible": False,
    }
    manifest = {**unsigned, "manifest_signature": signature(unsigned)}
    path = output_root.resolve() / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(f"Refusing to overwrite incompatible manifest: {path}")
        return path
    _atomic_json(path, manifest)
    return path


def load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.resolve().read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_signature", None)
    actual = signature(manifest)
    manifest["manifest_signature"] = claimed
    if claimed != actual:
        raise ValueError("Prepared q=1 manifest signature mismatch.")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported prepared q=1 manifest schema.")
    if manifest.get("sources") != _source_records():
        raise RuntimeError("Prepared q=1 workflow source changed after planning.")
    return manifest


def task_root(manifest_path: Path, task: Mapping[str, object]) -> Path:
    return (
        manifest_path.resolve().parent
        / "tasks"
        / f"{int(task['task_index']):04d}_{task['component']}_{str(task['task_signature'])[:12]}"
    )


def carbon_ao_indices(calc: object) -> dict[str, list[int]]:
    """Return carbon s/x/y/z AO indices using GPAW's documented real harmonics."""

    start = int(calc.wfs.setups.M_a[0])
    result: dict[str, list[int]] = {"s": [], "x": [], "y": [], "z": []}
    offset = start
    for basis_function in calc.wfs.setups[0].basis.bf_j:
        angular_momentum = int(basis_function.l)
        if angular_momentum == 0:
            result["s"].append(offset)
        elif angular_momentum == 1:
            for local, direction in enumerate(SPHERICAL_P_ORDER):
                result[direction].append(offset + local)
        offset += 2 * angular_momentum + 1
    if not all(result[key] for key in ("s", "x", "y", "z")):
        raise RuntimeError("The carbon dzp basis lacks a required s/p component.")
    return result


def _metric_normalize(vector: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    norm2 = float(np.real(vector.conj() @ overlap @ vector))
    if not math.isfinite(norm2) or norm2 <= 1.0e-14:
        raise RuntimeError("A prepared orbital has zero metric norm.")
    return vector / math.sqrt(norm2)


def _reference_ao(
    overlap: np.ndarray,
    indices: Mapping[str, Sequence[int]],
    direction: str,
) -> np.ndarray:
    vector = np.zeros(overlap.shape[0], dtype=overlap.dtype)
    vector[int(indices[direction][0])] = 1.0
    return _metric_normalize(vector, overlap)


def _orbital_scores(
    coefficients: np.ndarray,
    overlap: np.ndarray,
    references: Sequence[np.ndarray],
) -> np.ndarray:
    values = [np.abs(coefficients.conj() @ overlap @ ref) ** 2 for ref in references]
    return np.sum(values, axis=0)


def select_carbon_p_components(calc: object) -> dict[str, np.ndarray]:
    """Extract the three valence 2p components from converged isolated C+."""

    overlap = np.asarray(calc.wfs.S_qMM[0])
    indices = carbon_ao_indices(calc)
    references = {
        direction: _reference_ao(overlap, indices, direction)
        for direction in ("x", "y", "z")
    }
    alpha = calc.wfs.kpt_u[0]
    coefficients = np.asarray(alpha.C_nM)
    occupations = np.asarray(alpha.f_n)
    occupied = np.flatnonzero(occupations > 0.5)
    if len(occupied) != 2:
        raise RuntimeError(f"Expected two occupied C+ alpha orbitals, found {len(occupied)}.")
    p_scores = _orbital_scores(coefficients, overlap, tuple(references.values()))
    occupied_p = int(occupied[np.argmax(p_scores[occupied])])
    if p_scores[occupied_p] < 0.25:
        raise RuntimeError("Could not identify the occupied isolated-carbon 2p orbital.")
    energy = float(alpha.eps_n[occupied_p])
    ranked = sorted(
        range(len(coefficients)),
        key=lambda band: (abs(float(alpha.eps_n[band]) - energy), -p_scores[band]),
    )
    manifold = [band for band in ranked if p_scores[band] > 0.20][:3]
    if len(manifold) != 3:
        raise RuntimeError("Could not identify a three-dimensional carbon 2p manifold.")
    subspace = coefficients[manifold]
    components: dict[str, np.ndarray] = {}
    for direction, reference in references.items():
        amplitudes = subspace.conj() @ overlap @ reference
        projected = amplitudes.conj() @ subspace
        components[direction] = _metric_normalize(projected, overlap)
    gram = np.asarray(
        [[left.conj() @ overlap @ right for right in components.values()]
         for left in components.values()]
    )
    if not np.allclose(gram, np.eye(3), atol=5.0e-3, rtol=0.0):
        raise RuntimeError("The isolated-carbon x/y/z components are not orthonormal.")
    return components


def _occupied_rows(calc: object, spin: int) -> list[np.ndarray]:
    kpoint = calc.wfs.kpt_u[spin]
    return [
        np.asarray(kpoint.C_nM[index]).copy()
        for index in np.flatnonzero(np.asarray(kpoint.f_n) > 0.5)
    ]


def _complete_metric_basis(
    occupied: Sequence[np.ndarray], overlap: np.ndarray
) -> np.ndarray:
    """Preserve the ordered occupied span and complete an S-orthonormal basis."""

    dimension = overlap.shape[0]
    lower = np.linalg.cholesky(overlap)
    transformed = np.asarray(occupied) @ lower
    if transformed.shape[0] == 0:
        raise ValueError("At least one prepared orbital is required.")
    q_occupied, _ = np.linalg.qr(transformed.T, mode="reduced")
    # QR of the occupied directions followed by the canonical basis preserves
    # their ordered span and supplies a numerically stable orthogonal
    # complement. QR of the rank-deficient projector itself does not promise
    # that ordering.
    q_full, _ = np.linalg.qr(
        np.concatenate((q_occupied, np.eye(dimension)), axis=1), mode="complete"
    )
    coefficients = np.linalg.solve(lower.T, q_full).T
    metric = coefficients @ overlap @ coefficients.conj().T
    if not np.allclose(metric, np.eye(dimension), atol=1.0e-9, rtol=0.0):
        raise RuntimeError("Full-system orbital completion lost orthonormality.")
    return coefficients


def assemble_component_orbitals(
    carbon_calc: object,
    water_calc: object,
    combined_calc: object,
    component: str,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, np.ndarray]]:
    """Assemble alpha/beta coefficient matrices and fixed occupations."""

    if component not in Q1_COMPONENTS:
        raise ValueError(f"Unknown q=1 component: {component}")
    overlap = np.asarray(combined_calc.wfs.S_qMM[0])
    local_p_components = select_carbon_p_components(carbon_calc)
    carbon_alpha = _occupied_rows(carbon_calc, 0)
    carbon_beta = _occupied_rows(carbon_calc, 1)
    carbon_overlap = np.asarray(carbon_calc.wfs.S_qMM[0])
    p_refs = tuple(local_p_components.values())
    scores = [
        float(_orbital_scores(np.asarray([row]), carbon_overlap, p_refs)[0])
        for row in carbon_alpha
    ]
    carbon_s = carbon_alpha[int(np.argmin(scores))]
    if len(carbon_beta) != 1:
        raise RuntimeError("Expected one occupied beta orbital for isolated C+.")
    water_alpha = _occupied_rows(water_calc, 0)
    water_beta = _occupied_rows(water_calc, 1)
    if len(water_alpha) != 4 or len(water_beta) != 4:
        raise RuntimeError("Neutral water fragment must have four orbitals per spin.")

    carbon_dimension = int(carbon_calc.wfs.setups.nao)
    water_dimension = int(water_calc.wfs.setups.nao)
    if carbon_dimension + water_dimension != overlap.shape[0]:
        raise RuntimeError("Fragment and combined AO dimensions do not match.")

    def embed_carbon(row: np.ndarray) -> np.ndarray:
        embedded = np.zeros(overlap.shape[0], dtype=row.dtype)
        embedded[:carbon_dimension] = row
        return embedded

    def embed_water(row: np.ndarray) -> np.ndarray:
        embedded = np.zeros(overlap.shape[0], dtype=row.dtype)
        embedded[carbon_dimension:] = row
        return embedded

    p_components = {
        name: embed_carbon(row) for name, row in local_p_components.items()
    }
    carbon_s = embed_carbon(carbon_s)
    carbon_beta = [embed_carbon(row) for row in carbon_beta]
    water_alpha = [embed_water(row) for row in water_alpha]
    water_beta = [embed_water(row) for row in water_beta]

    if component == "p_fractional_ensemble":
        alpha_occupied = [*water_alpha, carbon_s, *p_components.values()]
        alpha_occupations = [1.0] * 5 + [1.0 / 3.0] * 3
    else:
        axis = max(
            range(3), key=lambda index: abs(COMPONENT_DIRECTIONS[component][index])
        )
        direction = ("x", "y", "z")[axis]
        alpha_occupied = [*water_alpha, carbon_s, p_components[direction]]
        alpha_occupations = [1.0] * 6
    beta_occupied = [*water_beta, carbon_beta[0]]
    beta_occupations = [1.0] * 5
    alpha_coefficients = _complete_metric_basis(alpha_occupied, overlap)
    beta_coefficients = _complete_metric_basis(beta_occupied, overlap)
    nao = overlap.shape[0]
    alpha_numbers = np.zeros(nao)
    beta_numbers = np.zeros(nao)
    alpha_numbers[: len(alpha_occupations)] = alpha_occupations
    beta_numbers[: len(beta_occupations)] = beta_occupations
    diagnostic_references = {"s": carbon_s, **p_components}
    return (
        [alpha_coefficients, beta_coefficients],
        [alpha_numbers, beta_numbers],
        diagnostic_references,
    )


def _make_calculator(
    settings: PreparedQ1Settings,
    *,
    charge: int,
    setups: Mapping[object, object],
    log_path: Path,
) -> object:
    from gpaw import GPAW, LCAO
    from gpaw.directmin.etdm_lcao import LCAOETDM

    return GPAW(
        mode=LCAO(),
        basis=settings.basis,
        h=settings.grid_spacing_angstrom,
        xc=settings.xc,
        charge=charge,
        spinpol=True,
        symmetry="off",
        nbands="nao",
        setups=dict(setups),
        eigensolver=LCAOETDM(
            searchdir_algo={"name": "l-bfgs-p"},
            linesearch_algo={"name": "swc-awc"},
            representation="u-invar",
        ),
        occupations={"name": "fixed-uniform"},
        mixer={"backend": "no-mixing"},
        convergence={
            "energy": settings.fragment_energy_tolerance_ev_per_electron,
            "eigenstates": settings.fragment_eigenstate_tolerance_ev2_per_electron,
            "density": settings.direct_gradient_tolerance_ev_per_electron,
            "bands": "occupied",
        },
        maxiter=settings.maximum_scf_iterations,
        txt=str(log_path),
    )


def _run_fragment(
    atoms: object,
    settings: PreparedQ1Settings,
    *,
    fragment: str,
    path: Path,
) -> object:
    from gpaw import GPAW

    if path.is_file():
        calc = GPAW(str(path), txt=str(path.with_suffix(".restart.log")))
        if not bool(calc.scf.converged):
            raise RuntimeError(f"Stored {fragment} fragment is not converged.")
        calc.set_positions(calc.atoms)
        return calc
    if fragment == "carbon":
        fragment_atoms = atoms[[0]]
        setups = {"default": "paw"}
        charge = 1
        fragment_atoms.set_initial_magnetic_moments([1.0])
    elif fragment == "water":
        fragment_atoms = atoms[[1, 2, 3]]
        setups = {"default": "paw"}
        charge = 0
        fragment_atoms.set_initial_magnetic_moments([0.0, 0.0, 0.0])
    else:
        raise ValueError(fragment)
    calc = _make_calculator(
        settings, charge=charge, setups=setups, log_path=path.with_suffix(".log")
    )
    fragment_atoms.calc = calc
    fragment_atoms.get_potential_energy()
    if not bool(calc.scf.converged):
        raise RuntimeError(f"The isolated {fragment} fragment did not converge.")
    calc.write(str(path), mode="all")
    return calc


def _initialize_combined_calculator(
    atoms: object,
    settings: PreparedQ1Settings,
    log_path: Path,
) -> object:
    from gpaw import GPAW, LCAO
    from gpaw.directmin.etdm_lcao import LCAOETDM

    calc = GPAW(
        mode=LCAO(),
        basis=settings.basis,
        h=settings.grid_spacing_angstrom,
        xc=settings.xc,
        charge=1,
        spinpol=True,
        symmetry="off",
        nbands="nao",
        eigensolver=LCAOETDM(
            excited_state=True,
            searchdir_algo={"name": "l-sr1p"},
            linesearch_algo={"name": "max-step", "max_step": 0.20},
            # GPAW 25.7.0 implements the real-valued unequal-occupation path
            # with the documented sparse representation; ``full`` is only
            # initialized for complex-valued orbitals in this release.
            representation="sparse",
            need_init_orbs=False,
            update_ref_orbs_counter=1000,
        ),
        occupations={"name": "fixed-uniform"},
        mixer={"backend": "no-mixing"},
        convergence={
            "energy": settings.fragment_energy_tolerance_ev_per_electron,
            "eigenstates": settings.direct_gradient_tolerance_ev_per_electron,
            "density": settings.direct_gradient_tolerance_ev_per_electron,
            "bands": "occupied",
        },
        maxiter=settings.maximum_scf_iterations,
        txt=str(log_path),
    )
    calc.initialize(atoms)
    calc.set_positions(atoms)
    return calc


def _install_orbitals(
    calc: object,
    atoms: object,
    coefficients: Sequence[np.ndarray],
    occupations: Sequence[np.ndarray],
    *,
    use_projections: bool = True,
) -> None:
    from gpaw.mom import prepare_mom_calculation

    if calc.wfs is None:
        calc.initialize(atoms)
        calc.set_positions(atoms)
    for spin, kpoint in enumerate(calc.wfs.kpt_u):
        kpoint.C_nM = np.asarray(coefficients[spin]).copy()
        # FixedOccupationNumbers still receives an eigenvalue array through
        # GPAW's generic occupation interface.  DO-MOM does not use these
        # placeholders as physical eigenvalues; its first orbital-gradient
        # evaluation constructs the Lagrange matrix from the prepared state.
        kpoint.eps_n = np.zeros(len(coefficients[spin]), dtype=float)
        calc.wfs.atomic_correction.calculate_projections(calc.wfs, kpoint)
    calc.wfs.set_orthonormalized(True)
    prepare_mom_calculation(
        calc,
        atoms,
        [np.asarray(values).tolist() for values in occupations],
        use_projections=use_projections,
        update_numbers=False,
        use_fixed_occupations=True,
    )
    # Changing the occupation calculator invalidates GPAW's SCFLoop object but
    # deliberately preserves the wavefunctions. Recreate only the calculator
    # infrastructure before publishing the prepared checkpoint.
    calc.initialize(atoms)
    calc.initialize_positions(atoms)
    for kpoint in calc.wfs.kpt_u:
        calc.wfs.atomic_correction.calculate_projections(calc.wfs, kpoint)
    calc.wfs.calculate_occupation_numbers(calc.density.fixed)
    calc.density.mixer.reset()
    calc.density.initialize_from_wavefunctions(calc.wfs)
    calc.hamiltonian.update(calc.density)
    if calc.scf is not None:
        calc.scf.reset()


def _iteration_observer(
    calc: object,
    cdft: object,
    references: Mapping[str, np.ndarray],
    trace_path: Path,
) -> None:
    from ase.units import Hartree

    overlap = np.asarray(calc.wfs.S_qMM[0])
    projections: dict[str, float] = {}
    for name, reference in references.items():
        population = 0.0
        for kpoint in calc.wfs.kpt_u:
            coefficients = np.asarray(kpoint.C_nM)
            occupations = np.asarray(kpoint.f_n)
            weights = np.abs(coefficients.conj() @ overlap @ reference) ** 2
            population += float(np.sum(occupations * weights))
        projections[name] = population
    charge = spin = None
    population_error = None
    try:
        cdft.get_atomic_density_correction()
        correction = np.asarray(cdft.get_energy_correction(return_density=True))
        density = calc.density.nt_sg
        charge_electrons = float(
            cdft.gd.integrate(cdft.ext.w_ig[0] * (density[0] + density[1]),
                              global_integral=True)
            + correction[0]
        )
        if cdft.n_spin_regions:
            spin = float(
                cdft.gd.integrate(cdft.ext.w_ig[1] * (density[0] - density[1]),
                                  global_integral=True)
                + correction[1]
            )
        else:
            # Never call a public Calculator ``get_*`` method from an SCF
            # observer: ASE may re-enter calculate() and recursively launch
            # another electronic solve.  GPAW has already populated these
            # results for the current completed inner iteration.
            magnetic_moments = calc.results.get("magmoms")
            spin = (None if magnetic_moments is None
                    else float(np.asarray(magnetic_moments)[0]))
        charge = 6.0 - charge_electrons
    except Exception as exc:
        population_error = f"{type(exc).__name__}: {exc}"
    energy = getattr(calc.hamiltonian, "e_total_extrapolated", None)
    record = {
        "scf_iteration": int(calc.scf.niter),
        "energy_ev": None if energy is None else float(energy * Hartree),
        "carbon_charge": charge,
        "carbon_spin_electrons": spin,
        "population_diagnostic_error": population_error,
        "reference_orbital_populations": projections,
        "external_cdft_potential_active": calc.parameters.external is cdft.ext,
        "constraint_coefficients_ev": (np.asarray(cdft.v_i) * Hartree).tolist(),
        "constraint_population_residuals_electrons": (
            [None if charge is None else 1.0 - charge]
            + ([None if spin is None else spin - 1.0] if cdft.n_spin_regions else [])
        ),
        "total_magnetic_moment_electrons": (
            None if calc.results.get("magmom") is None
            else float(calc.results["magmom"])
        ),
    }
    if calc.world.rank == 0:
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def execute_task(manifest_path: Path, task_index: int) -> dict[str, object]:
    from ase import Atoms
    from ase.units import Hartree
    from gpaw import GPAW
    from gpaw.cdft.cdft import CDFT
    from gpaw.mpi import world
    import gpaw
    from tqdm import tqdm

    manifest = load_manifest(manifest_path)
    tasks = manifest["tasks"]
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(task_index)
    task = tasks[task_index]
    settings = PreparedQ1Settings(**task["settings"])
    if gpaw.__version__ != settings.gpaw_version:
        raise RuntimeError(f"Expected GPAW {settings.gpaw_version}, found {gpaw.__version__}.")
    root = task_root(manifest_path, task)
    if world.rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    world.barrier()
    result_path = root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("task_signature") == task["task_signature"]:
            return result
        raise RuntimeError("Existing prepared q=1 result has incompatible identity.")
    coordinates = task["geometry"]["coordinates_angstrom"]
    atoms = Atoms(
        symbols=[row[0] for row in coordinates],
        positions=[[float(value) for value in row[1:]] for row in coordinates],
        pbc=False,
    )
    atoms.center(vacuum=settings.vacuum_angstrom)
    atoms.set_initial_magnetic_moments([1.0, 0.0, 0.0, 0.0])
    progress = tqdm(
        total=4,
        desc=f"q1 {task['component']}",
        unit="stage",
        disable=world.rank != 0,
    )
    try:
        reference = task.get("reference_prepared_state")
        if reference:
            for item in reference.values():
                if sha256_file(Path(item["path"])) != item["sha256"]:
                    raise RuntimeError("Reference prepared-state checksum mismatch.")
            retention_bundle = task.get("validated_retention_bundle")
            if retention_bundle:
                for key in ("gpw", "arrays"):
                    if sha256_file(Path(retention_bundle[key]["path"])) != retention_bundle[key]["sha256"]:
                        raise RuntimeError("Validated retention-bundle checksum mismatch.")
                combined = GPAW(retention_bundle["gpw"]["path"],
                                txt=str(root / "retention_bundle.restart.log"))
                combined.set_positions(combined.atoms)
                atoms = combined.atoms
                with np.load(retention_bundle["arrays"]["path"], allow_pickle=False) as data:
                    coefficients = [data["alpha_coefficients"], data["beta_coefficients"]]
                    occupations = [data["alpha_occupations"], data["beta_occupations"]]
                    orbital_references = {"x": data["reference_px"],
                                          "y": data["reference_py"],
                                          "z": data["reference_pz"],
                                          "s": data["reference_s"]}
                _install_orbitals(combined, atoms, coefficients, occupations,
                                  use_projections=False)
                prepared_hash = retention_bundle["gpw"]["sha256"]
                progress.update(3)
            else:
                # A .gpw file preserves the density and occupations but not the
            # in-memory DO orbital gauge/reference state.  Reloading its
            # arbitrarily rotated occupied coefficients caused tight MOM
            # continuation to lose the selected pz root.  Reconstruct the
            # exact reviewed orbital gauge from the signed fragment
            # checkpoints and geometry, then install cDFT before the first
            # complex iteration, as in the successful preparation pilot.
                carbon = GPAW(reference["carbon_fragment.gpw"]["path"], txt=None)
                carbon.set_positions(carbon.atoms)
                water = GPAW(reference["water_fragment.gpw"]["path"], txt=None)
                water.set_positions(water.atoms)
                combined = _initialize_combined_calculator(
                    atoms, settings, root / "prepared_complex.reconstructed.log"
                )
                coefficients, occupations, orbital_references = assemble_component_orbitals(
                    carbon, water, combined, str(task["component"])
                )
                _install_orbitals(combined, atoms, coefficients, occupations,
                                  use_projections=False)
                reconstructed_path = root / "prepared_complex.reconstructed.gpw"
                combined.write(str(reconstructed_path), mode="all")
                prepared_hash = reference["prepared_complex.gpw"]["sha256"]
                progress.update(3)
        else:
            carbon = _run_fragment(
                atoms.copy(), settings, fragment="carbon", path=root / "carbon_fragment.gpw"
            )
            progress.update(1)
            water = _run_fragment(
                atoms.copy(), settings, fragment="water", path=root / "water_fragment.gpw"
            )
            progress.update(1)
            combined = _initialize_combined_calculator(
                atoms, settings, root / "prepared_complex.log"
            )
            coefficients, occupations, orbital_references = assemble_component_orbitals(
                carbon, water, combined, str(task["component"])
            )
            _install_orbitals(combined, atoms, coefficients, occupations)
            prepared_path = root / "prepared_complex.gpw"
            combined.write(str(prepared_path), mode="all")
            prepared_hash = sha256_file(prepared_path)
            progress.update(1)

        # CDFT.set(external=...) resets GPAW's wavefunctions. Construct cDFT
        # first, then restore the signed prepared state before any energy call.
        constraint_mode = str(task.get("constraint_mode", "charge_spin"))
        include_spin_constraint = constraint_mode == "charge_spin"
        if constraint_mode not in {"charge_only", "charge_spin"}:
            raise ValueError(f"Unknown constraint mode: {constraint_mode}")
        cdft = CDFT(
            calc=combined,
            atoms=atoms,
            charge_regions=[[0]],
            charges=[1.0],
            spin_regions=[[0]] if include_spin_constraint else [],
            spins=[1.0] if include_spin_constraint else None,
            charge_coefs=[settings.charge_coefficient_hartree * Hartree],
            spin_coefs=[settings.spin_coefficient_hartree * Hartree]
            if include_spin_constraint else None,
            method="L-BFGS-B",
            minimizer_options={
                "gtol": settings.constraint_tolerance_electrons,
                "maxiter": settings.maximum_multiplier_iterations,
                "ftol": settings.optimizer_ftol,
            },
            bounds=[(-settings.multiplier_bound_ev, settings.multiplier_bound_ev)]
            * (2 if include_spin_constraint else 1),
            maxstep=settings.multiplier_bound_ev,
            tol=settings.optimizer_tolerance,
            compute_forces=False,
            restart=True,
            txt=str(root / "cdft.log"),
        )
        if combined.wfs is not None:
            raise RuntimeError("CDFT external-potential installation did not reset GPAW as reviewed.")
        combined.initialize(atoms)
        combined.set_positions(atoms)
        _install_orbitals(combined, atoms, coefficients, occupations,
                          use_projections=False if task.get("validated_retention_bundle") else True)
        if combined.parameters.external is not cdft.ext:
            raise RuntimeError("The cDFT external potential is not active.")
        gmf_anchor = task.get("validated_gmf_anchor")
        if gmf_anchor:
            if sha256_file(Path(gmf_anchor["path"])) != gmf_anchor["sha256"]:
                raise RuntimeError("Validated GMF-anchor checksum mismatch.")
            from gpaw.directmin.etdm_lcao import LCAOETDM
            coefficients = [np.asarray(kpoint.C_nM).copy()
                            for kpoint in combined.wfs.kpt_u]
            occupations = [np.asarray(kpoint.f_n).copy()
                           for kpoint in combined.wfs.kpt_u]
            combined.set(
                eigensolver=LCAOETDM(
                    excited_state=True,
                    searchdir_algo={"name": "l-bfgs-p_gmf"},
                    linesearch_algo={"name": "max-step", "max_step": 0.20},
                    partial_diagonalizer={
                        "name": "Davidson",
                        "logfile": str(root / "gmf_davidson.log"),
                        "seed": 42,
                        "sp_order": int(gmf_anchor["saddle_order"]),
                        "remember_sp_order": True,
                    },
                    update_ref_orbs_counter=1000,
                    representation="u-invar",
                    need_init_orbs=False,
                ),
                occupations={"name": "mom", "numbers": occupations,
                             "use_projections": False,
                             "update_numbers": False,
                             "use_fixed_occupations": True},
                txt=str(root / "gmf_inner.log"),
            )
            combined.initialize(atoms)
            combined.set_positions(atoms)
            _install_orbitals(combined, atoms, coefficients, occupations,
                              use_projections=False)
            if combined.parameters.external is not cdft.ext:
                raise RuntimeError("GMF setup displaced the active cDFT potential.")
        trace_path = root / "iteration_trace.jsonl"
        combined.attach(
            _iteration_observer,
            1,
            combined,
            cdft,
            {
                "s": orbital_references["s"],
                "p_x": orbital_references["x"],
                "p_y": orbital_references["y"],
                "p_z": orbital_references["z"],
            },
            trace_path,
        )
        atoms.calc = cdft
        energy = float(atoms.get_potential_energy())
        progress.update(1)
        residuals = np.asarray(cdft.dn_i, dtype=float)
        multipliers = np.asarray(cdft.v_i, dtype=float) * Hartree
        final_path = root / "final.gpw"
        combined.write(str(final_path), mode="all")
        accepted = bool(
            combined.scf.converged
            and np.all(np.isfinite(residuals))
            and np.max(np.abs(residuals)) <= settings.constraint_tolerance_electrons
            and np.all(np.abs(multipliers) < settings.multiplier_bound_ev - 1.0e-6)
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "task_id": task["task_id"],
            "task_signature": task["task_signature"],
            "component": task["component"],
            "calibration_trial": task.get("calibration_trial"),
            "constraint_mode": constraint_mode,
            "inner_solver": "do_gmf" if gmf_anchor else "do_mom",
            "gmf_saddle_order": (None if not gmf_anchor else int(gmf_anchor["saddle_order"])),
            "normal_return": True,
            "numerical_gate_passed": accepted,
            "production_eligible": False,
            "physical_state_status": "validation_pending",
            "energy_ev": energy,
            "constraint_residuals_electrons": residuals.tolist(),
            "computed_carbon_charge": float(1.0 - residuals[0]),
            "computed_carbon_spin_electrons": (
                float(1.0 + residuals[1]) if include_spin_constraint
                else (
                    None if combined.results.get("magmoms") is None
                    else float(np.asarray(combined.results["magmoms"])[0])
                )
            ),
            "constraint_coefficients_ev": multipliers.tolist(),
            "prepared_complex_sha256": prepared_hash,
            "final_gpw_sha256": sha256_file(final_path),
            "external_constraint_active_from_first_complex_iteration": True,
            "outer_history_path": str(trace_path.resolve()),
            "mpi_ranks": int(world.size),
        }
        if world.rank == 0:
            _atomic_json(root / ("result.json" if accepted else "rejected_result.json"), record)
        world.barrier()
        if not accepted:
            raise RuntimeError("Prepared q=1 calculation failed its numerical gate.")
        return record
    except Exception as exc:
        if world.rank == 0:
            _atomic_json(
                root / "failure.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "task_id": task["task_id"],
                    "task_signature": task["task_signature"],
                    "accepted_state": False,
                    "restart_wavefunction_allowed": False,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    finally:
        progress.close()
