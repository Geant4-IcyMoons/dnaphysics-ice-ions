"""Restart-safe GPAW cDFT pilot for charge-localized ion--water states.

This module deliberately does not produce soft-collision potentials.  It tests
whether GPAW can construct reproducible Gaussian-Hirshfeld charge/spin
constrained states where the present CP2K workflow has been numerically
fragile.  Accepted records remain physical-validation pending.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable, Mapping

import numpy as np

from .config import ProjectileDefinition, load_builtin_projectile
from .geometry import ORIENTATIONS, ScanGeometry, build_scan_geometry


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = 1
REVIEWED_GPAW_VERSION = "25.7.0"
MAXIMUM_STANDARD_PAW_CARBON_CHARGE = 4
PHYSICAL_STATE_STATUS = "validation_pending"
SCF_PROFILES = (
    "default",
    "official_eigensolver5",
    "official_damped_pulay",
    "official_damped_pulay_eigensolver5",
)
NEP_MBPOL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = NEP_MBPOL_ROOT.parents[2]
WORKFLOW_SOURCE_PATHS = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parent / "config.py",
    Path(__file__).resolve().parent / "geometry.py",
)

GPAW_CDFT_PROVENANCE = {
    "software": "GPAW",
    "reviewed_version": REVIEWED_GPAW_VERSION,
    "documentation": "https://gpaw.readthedocs.io/documentation/cdft/cdft.html",
    "parallel_documentation": (
        "https://gpaw.readthedocs.io/documentation/parallel_runs/parallel_runs.html"
    ),
    "implementation": (
        "Melander, Jónsson, and Mortensen, Journal of Chemical Theory and "
        "Computation 12, 5367-5378 (2016)."
    ),
    "implementation_doi": "10.1021/acs.jctc.6b00815",
    "partition": (
        "GPAW atom-centred Gaussian-Hirshfeld charge and spin constraints; "
        "the default GPAW weight parameters are retained and must later be "
        "tested against alternative population definitions."
    ),
    "scope": (
        "Independent numerical pilot only; not an ALMO calculation, not a "
        "validated diabatic state, and not a production soft-collision table."
    ),
}


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def signature(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workflow_source_records() -> dict[str, dict[str, object]]:
    records = {}
    for path in WORKFLOW_SOURCE_PATHS:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing GPAW workflow source: {resolved}")
        records[str(resolved.relative_to(REPOSITORY_ROOT))] = {
            "sha256": sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    return records


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".next")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class GPAWCDFTSettings:
    """Reviewed numerical settings for the first isolated-water pilot."""

    gpaw_version: str = REVIEWED_GPAW_VERSION
    mode: str = "fd"
    xc_functional: str = "PBE"
    grid_spacing_angstrom: float = 0.20
    vacuum_angstrom: float = 6.0
    occupations_width_ev: float = 0.0
    fix_total_magnetic_moment: bool = True
    spin_polarized: bool = True
    symmetry: str = "off"
    scf_profile: str = "default"
    cdft_method: str = "L-BFGS-B"
    constraint_tolerance_electrons: float = 0.01
    optimizer_tolerance: float = 0.001
    optimizer_max_iterations: int = 200
    maximum_multiplier_step_ev: float = 100.0
    multiplier_bound_ev: float = 100.0
    analytical_constraint_forces: bool = True
    checkpoint_every_outer_evaluation: bool = True

    def __post_init__(self) -> None:
        if self.gpaw_version != REVIEWED_GPAW_VERSION:
            raise ValueError(
                f"The pilot is reviewed only for GPAW {REVIEWED_GPAW_VERSION}."
            )
        for name in (
            "grid_spacing_angstrom",
            "vacuum_angstrom",
            "constraint_tolerance_electrons",
            "optimizer_tolerance",
            "maximum_multiplier_step_ev",
            "multiplier_bound_ev",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.optimizer_max_iterations < 1:
            raise ValueError("The optimizer iteration limit must be positive.")
        if self.cdft_method != "L-BFGS-B":
            raise ValueError("The reviewed GPAW pilot requires L-BFGS-B.")
        if self.scf_profile not in SCF_PROFILES:
            raise ValueError(f"Unsupported GPAW SCF profile: {self.scf_profile}")
        if not self.spin_polarized or not self.fix_total_magnetic_moment:
            raise ValueError("GPAW cDFT requires the reviewed fixed-spin setup.")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_GPAW_CDFT_SETTINGS = GPAWCDFTSettings()


def _state_payload(projectile: ProjectileDefinition, charge: int) -> dict[str, object]:
    state = projectile.state(charge)
    if projectile.symbol == "C" and charge > MAXIMUM_STANDARD_PAW_CARBON_CHARGE:
        raise ValueError(
            "The standard GPAW carbon PAW dataset freezes the 1s2 core; "
            "C5+ and C6+ therefore require a separately validated all-electron "
            "or custom-setup calculation."
        )
    return {
        "projectile": projectile.symbol,
        "atomic_number": projectile.atomic_number,
        "charge": state.charge,
        "electrons_on_projectile": state.electrons_on_projectile,
        "multiplicity": state.multiplicity,
        "target_local_spin_electrons": state.multiplicity - 1,
        "configuration": state.configuration,
        "term": state.term,
        "state_provenance": dict(projectile.state_provenance),
    }


def build_pilot_manifest(
    output_root: Path,
    *,
    projectile: ProjectileDefinition | None = None,
    charges: Iterable[int] = (1, 2),
    orientations: Iterable[str] = tuple(item.name for item in ORIENTATIONS),
    separation_angstrom: float = 12.0,
    settings: GPAWCDFTSettings = DEFAULT_GPAW_CDFT_SETTINGS,
) -> Path:
    """Write a deterministic q/orientation task manifest and return its path."""

    projectile = projectile or load_builtin_projectile("C")
    selected_charges = tuple(dict.fromkeys(int(value) for value in charges))
    if not selected_charges:
        raise ValueError("At least one charge state is required.")
    orientation_by_name = {item.name: item for item in ORIENTATIONS}
    selected_orientations = tuple(dict.fromkeys(str(value) for value in orientations))
    if not selected_orientations:
        raise ValueError("At least one orientation is required.")
    unknown = sorted(set(selected_orientations) - set(orientation_by_name))
    if unknown:
        raise ValueError(f"Unknown water orientations: {', '.join(unknown)}")
    if not math.isfinite(separation_angstrom) or separation_angstrom <= 0.0:
        raise ValueError("The ion--water separation must be positive and finite.")

    tasks: list[dict[str, object]] = []
    for charge in selected_charges:
        state = _state_payload(projectile, charge)
        for orientation_name in selected_orientations:
            geometry = build_scan_geometry(
                orientation_by_name[orientation_name],
                separation_angstrom,
                projectile.symbol,
            )
            task_core = {
                "state": state,
                "geometry": geometry.as_dict(),
                "settings": settings.as_dict(),
                "constraint": {
                    "charge_region_atom_indices": [0],
                    "target_projectile_charge": charge,
                    "spin_region_atom_indices": [0],
                    "target_projectile_spin_electrons": (
                        state["target_local_spin_electrons"]
                    ),
                    "weight_definition": "GPAW default Gaussian-Hirshfeld",
                },
            }
            task_signature = signature(task_core)
            tasks.append(
                {
                    "task_index": len(tasks),
                    "task_id": (
                        f"{projectile.symbol.lower()}_q{charge}_"
                        f"{orientation_name}_r{separation_angstrom:g}A"
                    ),
                    "task_signature": task_signature,
                    **task_core,
                }
            )

    configuration = {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "projectile": projectile.symbol,
        "charges": list(selected_charges),
        "orientations": list(selected_orientations),
        "separation_angstrom": float(separation_angstrom),
        "settings": settings.as_dict(),
        "provenance": GPAW_CDFT_PROVENANCE,
        "workflow_source_files": workflow_source_records(),
        "scientific_status": "numerical_cdft_pilot",
        "physical_state_status": PHYSICAL_STATE_STATUS,
        "production_eligible": False,
        "task_count": len(tasks),
        "tasks": tasks,
    }
    manifest = {**configuration, "manifest_signature": signature(configuration)}
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                f"Refusing to overwrite an incompatible GPAW pilot: {manifest_path}"
            )
    else:
        _atomic_write_json(manifest_path, manifest)
    return manifest_path


def load_pilot_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported GPAW pilot manifest schema.")
    recorded = payload.get("manifest_signature")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_signature"}
    if recorded != signature(unsigned):
        raise ValueError("GPAW pilot manifest signature mismatch.")
    if payload.get("workflow_source_files") != workflow_source_records():
        raise ValueError(
            "GPAW pilot workflow sources changed; prepare a new signed manifest."
        )
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or payload.get("task_count") != len(tasks):
        raise ValueError("GPAW pilot task count is inconsistent.")
    for index, task in enumerate(tasks):
        if task.get("task_index") != index:
            raise ValueError("GPAW pilot task indices are not contiguous.")
        core = {
            key: task[key]
            for key in ("state", "geometry", "settings", "constraint")
        }
        if task.get("task_signature") != signature(core):
            raise ValueError(f"Task {index} signature mismatch.")
    return payload


def task_directory(manifest_path: Path, task: Mapping[str, object]) -> Path:
    return (
        manifest_path.parent
        / "tasks"
        / f"{int(task['task_index']):04d}_{task['task_id']}_{str(task['task_signature'])[:12]}"
    )


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    return repr(value)


def validate_result_record(
    record: Mapping[str, object],
    task: Mapping[str, object],
    *,
    manifest_signature: str | None = None,
    task_root: Path | None = None,
) -> tuple[bool, list[str]]:
    """Apply fail-closed numerical gates; no physical-state claim is made."""

    reasons: list[str] = []
    if record.get("task_signature") != task.get("task_signature"):
        reasons.append("task_signature_mismatch")
    if (
        manifest_signature is not None
        and record.get("manifest_signature") != manifest_signature
    ):
        reasons.append("manifest_signature_mismatch")
    if record.get("gpaw_version") != REVIEWED_GPAW_VERSION:
        reasons.append("unreviewed_gpaw_version")
    if record.get("normal_return") is not True:
        reasons.append("no_normal_return")
    if record.get("inner_scf_converged") is not True:
        reasons.append("inner_scf_not_converged")
    residuals = record.get("constraint_residuals_electrons")
    tolerance = float(task["settings"]["constraint_tolerance_electrons"])
    if not isinstance(residuals, list) or len(residuals) != 2:
        reasons.append("missing_charge_spin_residuals")
    elif any(
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or abs(float(value)) > tolerance
        for value in residuals
    ):
        reasons.append("constraint_tolerance_not_met")
    energy = record.get("dft_energy_ev")
    if not isinstance(energy, (int, float)) or not math.isfinite(float(energy)):
        reasons.append("nonfinite_energy")
    forces = record.get("forces_ev_per_angstrom")
    try:
        force_array = np.asarray(forces, dtype=float)
    except (TypeError, ValueError):
        force_array = np.asarray([], dtype=float)
    if force_array.shape != (4, 3) or not np.all(np.isfinite(force_array)):
        reasons.append("invalid_forces")
    if record.get("multiplier_bound_contact") is not False:
        reasons.append("multiplier_bound_contact")
    state = task["state"]
    total_moment = record.get("total_magnetic_moment_electrons")
    expected_moment = float(state["multiplicity"]) - 1.0
    if (
        not isinstance(total_moment, (int, float))
        or not math.isfinite(float(total_moment))
        or abs(float(total_moment) - expected_moment) > tolerance
    ):
        reasons.append("total_magnetic_moment_mismatch")
    populations = record.get("gaussian_hirshfeld_electrons_by_atom")
    if not isinstance(populations, list) or len(populations) != 4:
        reasons.append("missing_atomic_population_diagnostics")
    elif isinstance(residuals, list) and len(residuals) == 2:
        expected_population = float(state["electrons_on_projectile"]) + float(
            residuals[0]
        )
        if (
            not math.isfinite(float(populations[0]))
            or abs(float(populations[0]) - expected_population) > 1.0e-6
        ):
            reasons.append("projectile_population_residual_inconsistent")
    if int(state["charge"]) > MAXIMUM_STANDARD_PAW_CARBON_CHARGE:
        reasons.append("frozen_core_invalid_for_charge")
    setups = record.get("paw_setups")
    if not isinstance(setups, list) or len(setups) != 4:
        reasons.append("missing_paw_setup_provenance")
    else:
        projectile_valence = setups[0].get("valence_electrons")
        if not isinstance(projectile_valence, (int, float)):
            reasons.append("missing_projectile_valence_count")
        elif int(state["charge"]) > int(round(float(projectile_valence))):
            reasons.append("charge_exceeds_projectile_paw_valence")
    if task_root is not None:
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, Mapping):
            reasons.append("missing_artifact_provenance")
        else:
            for path_key, hash_key in (
                ("gpw_path", "gpw_sha256"),
                ("density_path", "density_sha256"),
            ):
                try:
                    candidate = (task_root / str(artifacts[path_key])).resolve()
                    candidate.relative_to(task_root.resolve())
                    expected_hash = str(artifacts[hash_key])
                except (KeyError, ValueError):
                    reasons.append(f"invalid_{path_key}")
                    continue
                if not candidate.is_file() or sha256_file(candidate) != expected_hash:
                    reasons.append(f"{path_key}_checksum_mismatch")
    return not reasons, reasons


def _attempt_directory(root: Path, world: object) -> Path:
    attempts = root / "attempts"
    if world.rank == 0:
        attempts.mkdir(parents=True, exist_ok=True)
        existing = [
            int(path.name.split("_", 1)[1])
            for path in attempts.glob("attempt_[0-9][0-9][0-9][0-9]")
            if path.is_dir()
        ]
        number = max(existing, default=0) + 1
        (attempts / f"attempt_{number:04d}").mkdir()
    world.barrier()
    existing = sorted(attempts.glob("attempt_[0-9][0-9][0-9][0-9]"))
    if not existing:
        raise RuntimeError("The GPAW attempt directory was not created.")
    return existing[-1]


def _checkpoint_is_compatible(
    path: Path,
    task: Mapping[str, object],
    manifest_signature: str,
    runtime_file_signature: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        gpw_path = path.parent / str(checkpoint["gpw_path"])
        return (
            checkpoint.get("schema_version") == SCHEMA_VERSION
            and checkpoint.get("task_signature") == task.get("task_signature")
            and checkpoint.get("manifest_signature") == manifest_signature
            and checkpoint.get("runtime_file_signature") == runtime_file_signature
            and gpw_path.is_file()
            and checkpoint.get("gpw_sha256") == sha256_file(gpw_path)
            and len(checkpoint.get("coefficients_ev", [])) == 2
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _runtime_file_records() -> dict[str, dict[str, object]]:
    import ase
    import gpaw
    import scipy
    import _gpaw
    from gpaw.cdft import cdft as gpaw_cdft_module

    files = {}
    for name, module in (
        ("gpaw", gpaw),
        ("gpaw_cdft", gpaw_cdft_module),
        ("ase", ase),
        ("scipy", scipy),
        ("_gpaw", _gpaw),
    ):
        source = Path(module.__file__).resolve()
        files[name] = {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        }
    return files


def _runtime_provenance(calc: object, world: object) -> dict[str, object]:
    import ase
    import gpaw
    import scipy

    return {
        "python_version": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "gpaw_version": gpaw.__version__,
        "ase_version": ase.__version__,
        "scipy_version": scipy.__version__,
        "mpi_backend": getattr(world, "backend", "unknown"),
        "mpi_ranks": int(world.size),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "files": _runtime_file_records(),
        "calculator_parameters": _jsonable(calc.parameters),
    }


def _paw_setup_records(calc: object, atoms: object) -> list[dict[str, object]]:
    records = []
    for atom, setup in zip(atoms, calc.wfs.setups):
        record = {
            "atom_index": int(atom.index),
            "symbol": atom.symbol,
            "valence_electrons": float(setup.Nv),
            "setup_type": type(setup).__name__,
        }
        for attribute in ("fingerprint", "filename", "name", "type"):
            if hasattr(setup, attribute):
                record[attribute] = _jsonable(getattr(setup, attribute))
        records.append(record)
    return records


def _build_atoms(task: Mapping[str, object], settings: GPAWCDFTSettings) -> object:
    from ase import Atoms

    coordinates = task["geometry"]["coordinates_angstrom"]
    atoms = Atoms(
        symbols=[row[0] for row in coordinates],
        positions=[[float(value) for value in row[1:]] for row in coordinates],
        pbc=False,
    )
    atoms.center(vacuum=settings.vacuum_angstrom)
    spin = float(task["constraint"]["target_projectile_spin_electrons"])
    atoms.set_initial_magnetic_moments([spin, 0.0, 0.0, 0.0])
    return atoms


def execute_pilot_task(
    manifest_path: Path,
    task_index: int,
    *,
    resume: bool = True,
) -> dict[str, object]:
    """Run one manifest task collectively on the active GPAW MPI world."""

    from ase.units import Hartree
    from gpaw import GPAW
    from gpaw.cdft.cdft import CDFT
    from gpaw.mpi import world
    from gpaw.occupations import FermiDirac
    import gpaw

    manifest = load_pilot_manifest(manifest_path)
    tasks = manifest["tasks"]
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(f"Task index {task_index} lies outside 0..{len(tasks) - 1}.")
    task = tasks[task_index]
    manifest_signature = str(manifest["manifest_signature"])
    settings = GPAWCDFTSettings(**task["settings"])
    if gpaw.__version__ != settings.gpaw_version:
        raise RuntimeError(
            f"Expected GPAW {settings.gpaw_version}, found {gpaw.__version__}."
        )
    root = task_directory(manifest_path, task)
    if world.rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    world.barrier()
    accepted_path = root / "result.json"
    if accepted_path.is_file():
        record = json.loads(accepted_path.read_text(encoding="utf-8"))
        accepted, reasons = validate_result_record(
            record,
            task,
            manifest_signature=manifest_signature,
            task_root=root,
        )
        if accepted:
            return record
        raise RuntimeError(
            "Existing GPAW result failed revalidation: " + ", ".join(reasons)
        )

    attempt = _attempt_directory(root, world)
    checkpoint_path = root / "checkpoint.json"
    checkpoint_gpw = root / "checkpoint.gpw"
    runtime_file_signature = signature(_runtime_file_records())
    use_checkpoint = resume and _checkpoint_is_compatible(
        checkpoint_path,
        task,
        manifest_signature,
        runtime_file_signature,
    )
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if use_checkpoint
        else None
    )

    atoms = _build_atoms(task, settings)
    dft_log = attempt / "gpaw_dft.log"
    cdft_log = attempt / "gpaw_cdft.log"
    if use_checkpoint:
        calc = GPAW(str(checkpoint_gpw), txt=str(dft_log))
        atoms = calc.get_atoms()
        restart = True
        coefficients = [float(value) for value in checkpoint["coefficients_ev"]]
    else:
        calculator_kwargs: dict[str, object] = {}
        if settings.scf_profile in (
            "official_eigensolver5",
            "official_damped_pulay_eigensolver5",
        ):
            calculator_kwargs["eigensolver"] = {
                "name": "davidson",
                "niter": 5,
            }
        if settings.scf_profile in (
            "official_damped_pulay",
            "official_damped_pulay_eigensolver5",
        ):
            # Values documented in GPAW's convergence guide for difficult SCF
            # calculations. GPAW 25.7.0 lacks the newer ``msr1`` backend, so
            # retain its available Pulay backend and vary no undocumented knob.
            calculator_kwargs["mixer"] = {
                "beta": 0.04,
                "method": "difference",
                "nmaxold": 8,
                "weight": 100.0,
            }
        calc = GPAW(
            mode=settings.mode,
            h=settings.grid_spacing_angstrom,
            xc=settings.xc_functional,
            charge=int(task["state"]["charge"]),
            spinpol=settings.spin_polarized,
            occupations=FermiDirac(
                settings.occupations_width_ev,
                fixmagmom=settings.fix_total_magnetic_moment,
            ),
            symmetry=settings.symmetry,
            txt=str(dft_log),
            **calculator_kwargs,
        )
        restart = False
        coefficients = [None, None]

    class CheckpointingCDFT(CDFT):
        def jacobian(self, values):  # type: ignore[no-untyped-def]
            if settings.checkpoint_every_outer_evaluation:
                next_gpw = root / "checkpoint.next.gpw"
                self.calc.write(str(next_gpw), mode="all")
                self.calc.world.barrier()
                if self.calc.world.rank == 0:
                    os.replace(next_gpw, checkpoint_gpw)
                    payload = {
                        "schema_version": SCHEMA_VERSION,
                        "task_signature": task["task_signature"],
                        "manifest_signature": manifest_signature,
                        "runtime_file_signature": runtime_file_signature,
                        "gpaw_version": gpaw.__version__,
                        "outer_evaluation": int(self.iteration),
                        "coefficients_ev": [
                            float(value) for value in self.v_i * Hartree
                        ],
                        "constraint_residuals_electrons": [
                            float(value) for value in self.dn_i
                        ],
                        "dft_energy_ev": float(self.Edft),
                        "gpw_path": checkpoint_gpw.name,
                        "gpw_sha256": sha256_file(checkpoint_gpw),
                        "resume_semantics": (
                            "Paired wavefunction and multiplier; SciPy L-BFGS-B "
                            "history is restarted and final acceptance is rechecked."
                        ),
                    }
                    _atomic_write_json(checkpoint_path, payload)
                self.calc.world.barrier()
            return super().jacobian(values)

    bounds = [
        (-settings.multiplier_bound_ev, settings.multiplier_bound_ev),
        (-settings.multiplier_bound_ev, settings.multiplier_bound_ev),
    ]
    cdft_kwargs: dict[str, object] = {
        "calc": calc,
        "atoms": atoms,
        "charge_regions": [[0]],
        "charges": [float(task["constraint"]["target_projectile_charge"])],
        "spin_regions": [[0]],
        "spins": [float(task["constraint"]["target_projectile_spin_electrons"])],
        "txt": str(cdft_log),
        "method": settings.cdft_method,
        "minimizer_options": {
            "gtol": settings.constraint_tolerance_electrons,
            "maxiter": settings.optimizer_max_iterations,
        },
        "tol": settings.optimizer_tolerance,
        "maxstep": settings.maximum_multiplier_step_ev,
        "bounds": bounds,
        "forces": "analytical",
        "compute_forces": settings.analytical_constraint_forces,
        "restart": restart,
    }
    if restart:
        cdft_kwargs["charge_coefs"] = [coefficients[0]]
        cdft_kwargs["spin_coefs"] = [coefficients[1]]

    try:
        cdft = CheckpointingCDFT(**cdft_kwargs)
        atoms.calc = cdft
        atoms.get_potential_energy()
        forces = np.asarray(atoms.get_forces(), dtype=float)
        residuals = np.asarray(cdft.dn_i, dtype=float)
        coefficients_ev = np.asarray(cdft.v_i, dtype=float) * Hartree
        electron_populations = np.asarray(
            cdft.get_number_of_electrons_on_atoms(), dtype=float
        )
        spin_up = np.asarray(calc.get_pseudo_density(spin=0), dtype=float)
        spin_down = np.asarray(calc.get_pseudo_density(spin=1), dtype=float)
        density_path = attempt / "density_fingerprint.npz"
        if world.rank == 0:
            density_next = attempt / "density_fingerprint.next.npz"
            np.savez_compressed(
                density_next,
                total_density=spin_up + spin_down,
                spin_density=spin_up - spin_down,
            )
            os.replace(density_next, density_path)
        world.barrier()
        final_gpw = attempt / "final.gpw"
        final_next = attempt / "final.next.gpw"
        calc.write(str(final_next), mode="all")
        if world.rank == 0:
            os.replace(final_next, final_gpw)
        world.barrier()

        bound_distance = min(
            min(abs(value - lower), abs(upper - value))
            for value, (lower, upper) in zip(coefficients_ev, bounds)
        )
        bound_contact = bound_distance <= (
            1.0e-6 * max(1.0, settings.multiplier_bound_ev)
        )
        record: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "task_id": task["task_id"],
            "task_index": task_index,
            "task_signature": task["task_signature"],
            "manifest_signature": manifest_signature,
            "normal_return": True,
            "inner_scf_converged": bool(getattr(calc.scf, "converged", False)),
            "gpaw_version": gpaw.__version__,
            "resumed_from_checkpoint": use_checkpoint,
            "constraint_residuals_electrons": residuals.tolist(),
            "computed_projectile_charge": float(
                task["state"]["charge"] - residuals[0]
            ),
            "computed_projectile_spin_electrons": float(
                task["constraint"]["target_projectile_spin_electrons"]
                + residuals[1]
            ),
            "constraint_coefficients_ev": coefficients_ev.tolist(),
            "multiplier_bounds_ev": bounds,
            "multiplier_bound_distance_ev": float(bound_distance),
            "multiplier_bound_contact": bool(bound_contact),
            "dft_energy_ev": float(cdft.dft_energy()),
            "cdft_lagrangian_ev": float(cdft.cdft_free_energy()),
            "forces_ev_per_angstrom": forces.tolist(),
            "gaussian_hirshfeld_electrons_by_atom": electron_populations.tolist(),
            "total_magnetic_moment_electrons": float(
                calc.get_magnetic_moment()
            ),
            "atomic_magnetic_moments_electrons": np.asarray(
                calc.get_magnetic_moments(), dtype=float
            ).tolist(),
            "resolved_weight_parameters_bohr": {
                "cutoff_radius": _jsonable(cdft.Rc),
                "gaussian_width": _jsonable(cdft.mu),
            },
            "paw_setups": _paw_setup_records(calc, atoms),
            "centered_positions_angstrom": atoms.positions.tolist(),
            "cell_angstrom": atoms.cell.array.tolist(),
            "runtime_provenance": _runtime_provenance(calc, world),
            "artifacts": {
                "gpw_path": str(final_gpw.relative_to(root)),
                "gpw_sha256": sha256_file(final_gpw),
                "density_path": str(density_path.relative_to(root)),
                "density_sha256": sha256_file(density_path),
                "dft_log_path": str(dft_log.relative_to(root)),
                "cdft_log_path": str(cdft_log.relative_to(root)),
            },
            "physical_state_status": PHYSICAL_STATE_STATUS,
            "production_eligible": False,
        }
        accepted, reasons = validate_result_record(
            record,
            task,
            manifest_signature=manifest_signature,
            task_root=root,
        )
        record["numerical_gate_passed"] = accepted
        record["rejection_reasons"] = reasons
        if world.rank == 0:
            _atomic_write_json(attempt / "result.json", record)
            if accepted:
                _atomic_write_json(accepted_path, record)
            else:
                _atomic_write_json(root / "latest_rejected_result.json", record)
        world.barrier()
        if not accepted:
            raise RuntimeError("GPAW numerical gate failed: " + ", ".join(reasons))
        return record
    except Exception as exc:
        if world.rank == 0:
            failure = {
                "schema_version": SCHEMA_VERSION,
                "task_id": task["task_id"],
                "task_signature": task["task_signature"],
                "accepted_state": False,
                "restart_wavefunction_allowed": _checkpoint_is_compatible(
                    checkpoint_path,
                    task,
                    manifest_signature,
                    runtime_file_signature,
                ),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
            _atomic_write_json(attempt / "failure.json", failure)
            _atomic_write_json(root / "latest_failure.json", failure)
        raise


def collect_pilot(manifest_path: Path) -> dict[str, object]:
    """Revalidate every task record and write a deterministic summary."""

    from tqdm import tqdm

    manifest = load_pilot_manifest(manifest_path)
    rows = []
    for task in tqdm(manifest["tasks"], desc="GPAW cDFT validation", unit="task"):
        result_path = task_directory(manifest_path, task) / "result.json"
        if not result_path.is_file():
            rows.append(
                {
                    "task_index": task["task_index"],
                    "task_id": task["task_id"],
                    "status": "pending",
                    "reasons": ["missing_result"],
                }
            )
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        root = task_directory(manifest_path, task)
        accepted, reasons = validate_result_record(
            result,
            task,
            manifest_signature=str(manifest["manifest_signature"]),
            task_root=root,
        )
        rows.append(
            {
                "task_index": task["task_index"],
                "task_id": task["task_id"],
                "status": "numerically_passed" if accepted else "rejected",
                "reasons": reasons,
                "dft_energy_ev": result.get("dft_energy_ev"),
                "constraint_residuals_electrons": result.get(
                    "constraint_residuals_electrons"
                ),
            }
        )
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("numerically_passed", "rejected", "pending")
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest_signature": manifest["manifest_signature"],
        "counts": counts,
        "tasks": rows,
        "physical_state_status": PHYSICAL_STATE_STATUS,
        "production_eligible": False,
        "next_gate": (
            "Repeatability, density/orbital character, asymptotic, population-"
            "definition, grid/cell/setup and finite-difference force validation."
        ),
    }
    _atomic_write_json(manifest_path.parent / "summary.json", summary)
    return summary
