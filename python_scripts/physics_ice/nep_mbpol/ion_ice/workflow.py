"""Dependency-aware status, planning, submission, and assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable

from bca.runtime import AdaptiveKernelTable, KernelTableError
from bca.structure import file_sha256, load_ice_structure
from nlh import get_coefficients

from .registry import (
    NEP_MBPOL_ROOT,
    PROCESS_EVIDENCE_ROOT,
    REPOSITORY_ROOT,
    get_phase,
    get_projectile,
)
from .resources import RESOURCE_PATH, load_resource_profiles
from .schema import PhaseDefinition, ProjectileDefinition


REPORT_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = 1
HARD_COLLISION_BENCHMARK_ROOT = (
    PROCESS_EVIDENCE_ROOT
    / "hard_nuclear_collisions"
    / "benchmarking"
    / "reference_results"
)
STATES = frozenset(
    (
        "complete",
        "running",
        "runnable",
        "validation_pending",
        "missing_input",
        "blocked",
    )
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class StageStatus:
    stage: str
    state: str
    summary: str
    evidence: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    next_action: str | None = None

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError(f"Invalid workflow state {self.state!r}.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON product {path}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON product is not an object: {path}")
    return value


@lru_cache(maxsize=64)
def _active_pbs_job_ids(job_name: str) -> tuple[str, ...]:
    """Return active jobs with one exact submitted PBS name, if queryable."""

    qselect = shutil.which("qselect")
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if qselect is None or not user:
        return ()
    completed = subprocess.run(
        [qselect, "-u", user, "-N", job_name],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return ()
    return tuple(
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    )


def _structure_status(phase: PhaseDefinition) -> StageStatus:
    registry_path = phase.resolve(phase.structure_registry)
    if registry_path is None:
        active = _active_pbs_job_ids(str(phase.preparation["pbs_job_name"]))
        if active:
            return StageStatus(
                "ice_structure",
                "running",
                f"The {phase.phase_id} preparation is active in PBS.",
                evidence=active,
                blockers=(
                    str(
                        phase.validation.get(
                            "missing", "phase validation is incomplete"
                        )
                    ),
                ),
                next_action=(
                    "Wait for dynamics, then run the phase validator and "
                    "register only accepted snapshots."
                ),
            )
        return StageStatus(
            "ice_structure",
            "missing_input",
            f"No accepted collision registry exists for {phase.phase_id}.",
            blockers=(
                str(phase.validation.get("missing", "phase validation is incomplete")),
            ),
            next_action=(
                "Run the phase preparation, its documented structural validator, "
                "and register only accepted snapshots."
            ),
        )
    if not registry_path.is_file():
        active = _active_pbs_job_ids(str(phase.preparation["pbs_job_name"]))
        if active:
            return StageStatus(
                "ice_structure",
                "running",
                f"The {phase.phase_id} preparation is active in PBS.",
                evidence=active,
                next_action="Validate and register its completed snapshots.",
            )
        return StageStatus(
            "ice_structure",
            "missing_input",
            f"Declared structure registry is absent: {registry_path}",
            next_action="Run and validate the phase preparation.",
        )
    try:
        registry = _read_json(registry_path)
        records = registry.get("structures")
        if registry.get("all_collision_ready") is not True or not isinstance(
            records, list
        ) or not records:
            raise ValueError("registry does not attest every structure")
        for record in records:
            if not isinstance(record, dict) or record.get("collision_ready") is not True:
                raise ValueError("registry contains a non-ready structure")
            structure_path = (registry_path.parent / str(record["path"])).resolve()
            metadata_path = structure_path.with_suffix(".json")
            structure = load_ice_structure(
                structure_path, metadata_path=metadata_path
            )
            if structure.source_sha256 != record.get("sha256"):
                raise ValueError("structure digest differs from registry")
    except (KeyError, ValueError, FileNotFoundError) as exc:
        return StageStatus(
            "ice_structure",
            "blocked",
            f"Structure registry failed validation: {exc}",
            evidence=(str(registry_path),),
            next_action="Repair and re-attest the phase registry.",
        )
    return StageStatus(
        "ice_structure",
        "complete",
        (
            f"{len(records)} accepted {phase.phase_id} structure"
            f"{'s' if len(records) != 1 else ''} available."
        ),
        evidence=(str(registry_path),),
    )


def _nlh_status(projectile: ProjectileDefinition) -> StageStatus:
    component = projectile.components["nlh_pair_potential"]
    if component["status"] != "implemented":
        return StageStatus(
            "nlh_pair_potential",
            "missing_input",
            f"No validated projectile--H/O hard pair potential for {projectile.symbol}.",
            blockers=(str(component["reason"]),),
            next_action=(
                "Supply independently validated pair-potential coefficients and "
                "their provenance; do not extrapolate another element."
            ),
        )
    missing = []
    for target in ("H", "O"):
        try:
            get_coefficients(projectile.symbol, target)
        except (ValueError, RuntimeError) as exc:
            missing.append(f"{projectile.symbol}-{target}: {exc}")
    if missing:
        return StageStatus(
            "nlh_pair_potential",
            "blocked",
            "The registry declares NLH support, but backend coefficients are missing.",
            blockers=tuple(missing),
            next_action="Add and regression-test the published coefficient rows.",
        )
    source = (projectile.source_path.parent / str(component["coefficient_source"])).resolve()
    return StageStatus(
        "nlh_pair_potential",
        "complete",
        f"Validated NLH coefficient pairs exist for {projectile.symbol}-H/O.",
        evidence=(str(source),),
    )


def _kernel_candidates(symbol: str) -> tuple[tuple[Path, Path], ...]:
    return (
        (
            NEP_MBPOL_ROOT
            / "collision_kernels"
            / "by_species"
            / symbol
            / "nlh_collision_kernels.manifest.json",
            HARD_COLLISION_BENCHMARK_ROOT
            / "by_species"
            / symbol
            / "nlh_adaptive_kernel_benchmark.json",
        ),
        (
            NEP_MBPOL_ROOT
            / "collision_kernels"
            / "nlh_collision_kernels.manifest.json",
            HARD_COLLISION_BENCHMARK_ROOT
            / "nlh_adaptive_kernel_benchmark.json",
        ),
    )


@lru_cache(maxsize=16)
def _validated_kernel(path: str) -> AdaptiveKernelTable:
    return AdaptiveKernelTable(path)


def _kernel_status(
    projectile: ProjectileDefinition, nlh: StageStatus
) -> StageStatus:
    if nlh.state != "complete":
        return StageStatus(
            "nlh_kernel",
            "blocked",
            "Kernel generation requires both projectile--H/O pair potentials.",
            blockers=(nlh.summary,),
        )
    for manifest_path, benchmark_path in _kernel_candidates(projectile.symbol):
        if not manifest_path.is_file():
            continue
        try:
            table = _validated_kernel(str(manifest_path.resolve()))
            required = {(projectile.symbol, "H"), (projectile.symbol, "O")}
            if not required.issubset(set(table.pairs)):
                raise KernelTableError("both projectile--H/O pairs are not present")
            if not benchmark_path.is_file():
                return StageStatus(
                    "nlh_kernel",
                    "validation_pending",
                    "Kernel table exists but its independent benchmark is absent.",
                    evidence=(str(manifest_path),),
                    next_action="Run the independent dense-grid kernel benchmark.",
                )
            benchmark = _read_json(benchmark_path)
            if benchmark.get("accepted") is not True or projectile.symbol not in benchmark.get(
                "projectiles", []
            ):
                return StageStatus(
                    "nlh_kernel",
                    "validation_pending",
                    "Kernel benchmark does not record acceptance for this projectile.",
                    evidence=(str(manifest_path), str(benchmark_path)),
                    next_action="Complete the 0.5% independent benchmark.",
                )
        except (FileNotFoundError, ValueError, KernelTableError) as exc:
            return StageStatus(
                "nlh_kernel",
                "blocked",
                f"Kernel product failed validation: {exc}",
                evidence=(str(manifest_path),),
                next_action="Regenerate from checksum-compatible checkpoints.",
            )
        return StageStatus(
            "nlh_kernel",
            "complete",
            "Adaptive H/O kernels and the independent 0.5% benchmark are accepted.",
            evidence=(str(manifest_path), str(benchmark_path)),
        )
    active = _active_pbs_job_ids(f"nlh_k_{projectile.symbol}")
    if active:
        return StageStatus(
            "nlh_kernel",
            "running",
            "The projectile kernel/benchmark job is active in PBS.",
            evidence=active,
        )
    return StageStatus(
        "nlh_kernel",
        "runnable",
        "NLH inputs are complete; the adaptive kernel and benchmark job can run.",
        next_action="Submit one restart-safe kernel/benchmark job.",
    )


def _hard_state_candidates(symbol: str, phase: PhaseDefinition) -> list[Path]:
    roots = (
        PROCESS_EVIDENCE_ROOT
        / "hard_nuclear_collisions"
        / "validation"
        / "runs"
        / "sharded_adaptive_particles"
        / symbol,
        PROCESS_EVIDENCE_ROOT
        / "hard_nuclear_collisions"
        / "validation"
        / "runs"
        / "adaptive_particles"
        / symbol,
        NEP_MBPOL_ROOT / "hard_collision_runs" / "adaptive_particles" / symbol,
    )
    result = []
    structure_registry = phase.resolve(phase.structure_registry)
    structure_directory = structure_registry.parent if structure_registry else None
    for path in sorted(
        path
        for root in roots
        for pattern in (
            "*/adaptive_particle_state.json",
            "*/adaptive_particle_shards.state.json",
        )
        for path in root.glob(pattern)
    ):
        try:
            value = _read_json(path)
            configured = Path(
                str(value["configuration"]["structure_directory"])
            ).resolve()
        except (KeyError, ValueError):
            continue
        if structure_directory is not None and configured == structure_directory:
            result.append(path)
    return result


def _hard_transport_status(
    projectile: ProjectileDefinition,
    phase: PhaseDefinition,
    structure: StageStatus,
    kernel: StageStatus,
) -> StageStatus:
    component = projectile.components["hard_transport"]
    if component["status"] != "implemented":
        return StageStatus(
            "hard_transport",
            "blocked",
            "No hard-transport backend is registered.",
            blockers=(str(component["reason"]),),
        )
    candidates = _hard_state_candidates(projectile.symbol, phase)
    for path in reversed(candidates):
        state = _read_json(path).get("status")
        if state == "complete":
            return StageStatus(
                "hard_transport",
                "complete",
                "Adaptive phase/orientation transport is numerically complete.",
                evidence=(str(path),),
            )
        if state in {"running", "awaiting_calculations", "refining"}:
            return StageStatus(
                "hard_transport",
                "running",
                "A restart-safe adaptive hard-transport workflow is active/incomplete.",
                evidence=(str(path),),
                next_action="Resume the identical workflow until its state is complete.",
            )
        if state == "refinement_limit_reached":
            return StageStatus(
                "hard_transport",
                "blocked",
                "Hard transport reached its declared refinement limit.",
                evidence=(str(path),),
                next_action="Inspect the failed interval before changing a limit.",
            )
    active = _active_pbs_job_ids(
        f"nlh_{projectile.symbol}_{phase.phase_id[:8]}"
    )
    if active:
        return StageStatus(
            "hard_transport",
            "running",
            "A matching adaptive hard-transport job is active in PBS.",
            evidence=active,
        )
    blockers = tuple(
        item.summary for item in (structure, kernel) if item.state != "complete"
    )
    if blockers:
        return StageStatus(
            "hard_transport",
            "blocked",
            "Hard transport is waiting for its structure and kernel inputs.",
            blockers=blockers,
        )
    return StageStatus(
        "hard_transport",
        "runnable",
        "All inputs are accepted; one adaptive transport job can be submitted.",
        next_action="Submit one restart-safe job for this projectile and phase.",
    )


def _hard_validation_status(
    projectile: ProjectileDefinition,
    phase: PhaseDefinition,
    hard_transport: StageStatus,
) -> StageStatus:
    path = (
        NEP_MBPOL_ROOT
        / "hard_collision_validation"
        / phase.phase_id
        / f"{projectile.symbol}.json"
    )
    if path.is_file():
        value = _read_json(path)
        if (
            value.get("projectile") == projectile.symbol
            and value.get("phase_id") == phase.phase_id
            and value.get("accepted") is True
        ):
            return StageStatus(
                "hard_phase_validation",
                "complete",
                "The phase/orientation hard-collision decision gate is accepted.",
                evidence=(str(path),),
            )
        return StageStatus(
            "hard_phase_validation",
            "blocked",
            "The hard-validation record exists but is not an accepted matching result.",
            evidence=(str(path),),
        )
    if hard_transport.state == "complete":
        return StageStatus(
            "hard_phase_validation",
            "validation_pending",
            "Transport converged, but no accepted phase/orientation decision exists.",
            next_action=(
                "Compare replicas, orientations, and phases; record whether a "
                "density-only continuum table is adequate."
            ),
        )
    return StageStatus(
        "hard_phase_validation",
        "blocked",
        "The phase/orientation gate requires completed hard transport.",
        blockers=(hard_transport.summary,),
    )


def _soft_definition_status(projectile: ProjectileDefinition) -> StageStatus:
    component = projectile.components["soft_dft"]
    if component["status"] != "implemented":
        return StageStatus(
            "soft_dft_definition",
            "missing_input",
            f"No complete charge-resolved soft-DFT definition for {projectile.symbol}.",
            blockers=(str(component["reason"]),),
            next_action=(
                "Register all q=0..Z states, multiplicities, provenance, an "
                "all-electron basis, and sourced H/O scan grids."
            ),
        )
    return StageStatus(
        "soft_dft_definition",
        "complete",
        f"The complete q=0..{projectile.atomic_number} soft-DFT definition is registered.",
        evidence=(str(projectile.source_path),),
    )


def _soft_state_candidates(symbol: str) -> list[Path]:
    root = PROCESS_EVIDENCE_ROOT / "soft_nuclear_collisions" / "validation" / "runs"
    return sorted(
        path
        for path in root.glob(
            f"{symbol.lower()}_adaptive_pilot/*/adaptive_charge_resolved_dft.state.json"
        )
    )


def _soft_molecular_status(
    projectile: ProjectileDefinition, definition: StageStatus
) -> StageStatus:
    for path in reversed(_soft_state_candidates(projectile.symbol)):
        state = _read_json(path)
        if state.get("status") == "complete":
            final = Path(str(state.get("final_manifest", "")))
            if final.is_file():
                return StageStatus(
                    "soft_molecular_dft",
                    "validation_pending",
                    "The adaptive molecular table converged numerically; physical gates remain.",
                    evidence=(str(path), str(final)),
                    next_action=(
                        "Complete basis/grid/cell/functional, charge-localization, "
                        "asymptotic, and NLH-overlap validation."
                    ),
                )
        if state.get("status") in {"awaiting_calculations", "refining"}:
            return StageStatus(
                "soft_molecular_dft",
                "running",
                "A restart-safe adaptive molecular CDFT workflow is incomplete.",
                evidence=(str(path),),
            )
    active = _active_pbs_job_ids(f"soft_dft_{projectile.symbol}")
    if active:
        return StageStatus(
            "soft_molecular_dft",
            "running",
            "A matching adaptive molecular CDFT job is active in PBS.",
            evidence=active,
        )
    if definition.state != "complete":
        return StageStatus(
            "soft_molecular_dft",
            "blocked",
            "Molecular CDFT requires a complete projectile definition.",
            blockers=(definition.summary,),
        )
    return StageStatus(
        "soft_molecular_dft",
        "runnable",
        "The adaptive molecular CDFT workflow can run where CP2K is available.",
        blockers=("CP2K executable/module and all-electron basis availability required",),
        next_action="Submit an independent restart-safe adaptive CP2K job.",
    )


def _soft_phase_status(
    phase: PhaseDefinition, molecular: StageStatus, structure: StageStatus
) -> StageStatus:
    blockers = []
    if molecular.state != "complete":
        blockers.append("molecular DFT has not passed its physical acceptance gates")
    if structure.state != "complete":
        blockers.append(structure.summary)
    blockers.append(
        "the representative ice-environment DFT dataset, residual fit, and validator are not implemented"
    )
    return StageStatus(
        "soft_phase_model",
        "blocked",
        f"No validated charge-resolved soft model exists for {phase.phase_id}.",
        blockers=tuple(blockers),
        next_action=(
            "After molecular acceptance, calculate representative local ice "
            "environments, fit the declared residual model, and validate moments."
        ),
    )


def _hard_geant4_status(
    projectile: ProjectileDefinition, kernel: StageStatus
) -> StageStatus:
    manifest = (
        NEP_MBPOL_ROOT
        / "collision_kernels"
        / "geant4"
        / f"nlh_hard_elastic_{projectile.symbol}.manifest.json"
    )
    if manifest.is_file():
        try:
            value = _read_json(manifest)
            table_path = manifest.parent / str(value["table"])
            if not table_path.is_file():
                raise ValueError(f"declared table is absent: {table_path}")
            if file_sha256(table_path) != value.get("table_sha256"):
                raise ValueError("declared table checksum does not match")
        except (KeyError, ValueError) as exc:
            return StageStatus(
                "geant4_hard_table",
                "blocked",
                f"The Geant4 hard-table product failed validation: {exc}",
                evidence=(str(manifest),),
            )
        if value.get("projectile") != projectile.symbol:
            return StageStatus(
                "geant4_hard_table",
                "blocked",
                "The Geant4 hard-table manifest names the wrong projectile.",
                evidence=(str(manifest),),
            )
        state = (
            "complete" if value.get("release_status") == "accepted" else "validation_pending"
        )
        return StageStatus(
            "geant4_hard_table",
            state,
            f"A checksum-linked hard table exists with release status "
            f"{value.get('release_status')}.",
            evidence=(str(manifest),),
        )
    if kernel.state == "complete":
        return StageStatus(
            "geant4_hard_table",
            "runnable",
            "A validation-pending Geant4-format hard table can be exported from kernels.",
            next_action="Run the deterministic hard-table exporter.",
        )
    return StageStatus(
        "geant4_hard_table",
        "blocked",
        "The Geant4 hard table requires an accepted NLH kernel benchmark.",
        blockers=(kernel.summary,),
    )


def _soft_geant4_status(projectile: ProjectileDefinition) -> StageStatus:
    manifest = (
        NEP_MBPOL_ROOT
        / "soft_collision_tables"
        / "geant4"
        / f"soft_elastic_{projectile.symbol}.manifest.json"
    )
    if manifest.is_file():
        try:
            value = _read_json(manifest)
            table_path = manifest.parent / str(value["table"])
            valid_product = (
                table_path.is_file()
                and file_sha256(table_path) == value.get("table_sha256")
            )
        except (KeyError, ValueError):
            valid_product = False
        if valid_product and value.get("projectile") == projectile.symbol and value.get(
            "release_status"
        ) == "accepted":
            return StageStatus(
                "geant4_soft_table",
                "complete",
                "An accepted charge-resolved soft Geant4 table exists.",
                evidence=(str(manifest),),
            )
    return StageStatus(
        "geant4_soft_table",
        "blocked",
        "No accepted charge-resolved soft Geant4 table/exporter exists.",
        next_action="Complete and validate the soft phase model before export.",
    )


def _runtime_status(
    projectile: ProjectileDefinition, component_name: str
) -> StageStatus:
    component = projectile.components[component_name]
    label = "hard" if component_name == "geant4_hard_runtime" else "soft"
    if component["status"] == "implemented":
        backend = str(component.get("backend", "registered runtime"))
        release = component.get("release_status")
        suffix = f"; release status {release}" if release else ""
        return StageStatus(
            component_name,
            "complete",
            f"The Geant4 {label} consumer is implemented as {backend}{suffix}.",
            evidence=(str(projectile.source_path),),
        )
    state = "blocked" if component["status"] == "blocked" else "missing_input"
    return StageStatus(
        component_name,
        state,
        f"The Geant4 {label} consumer is not available for {projectile.symbol}.",
        blockers=(str(component["reason"]),),
    )


def inspect_model(
    projectile_value: str, phase_id: str
) -> dict[str, Any]:
    projectile = get_projectile(projectile_value)
    phase = get_phase(phase_id)
    structure = _structure_status(phase)
    nlh = _nlh_status(projectile)
    kernel = _kernel_status(projectile, nlh)
    hard = _hard_transport_status(projectile, phase, structure, kernel)
    hard_validation = _hard_validation_status(projectile, phase, hard)
    soft_definition = _soft_definition_status(projectile)
    soft_molecular = _soft_molecular_status(projectile, soft_definition)
    soft_phase = _soft_phase_status(phase, soft_molecular, structure)
    hard_table = _hard_geant4_status(projectile, kernel)
    soft_table = _soft_geant4_status(projectile)
    hard_runtime = _runtime_status(projectile, "geant4_hard_runtime")
    soft_runtime = _runtime_status(projectile, "geant4_soft_runtime")
    bundle_blockers = tuple(
        stage.summary
        for stage in (
            hard_validation,
            hard_table,
            soft_phase,
            soft_table,
            hard_runtime,
            soft_runtime,
        )
        if stage.state != "complete"
    )
    bundle = StageStatus(
        "geant4_bundle",
        "complete" if not bundle_blockers else "blocked",
        "All accepted component tables and runtimes can be assembled."
        if not bundle_blockers
        else "The complete Geant4 ion--ice model is not yet assemblable.",
        blockers=bundle_blockers,
    )
    stages = (
        structure,
        nlh,
        kernel,
        hard,
        hard_validation,
        soft_definition,
        soft_molecular,
        soft_phase,
        hard_table,
        soft_table,
        hard_runtime,
        soft_runtime,
        bundle,
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "projectile": projectile.symbol,
        "phase_id": phase.phase_id,
        "species_definition": str(projectile.source_path),
        "species_definition_sha256": file_sha256(projectile.source_path),
        "phase_definition": str(phase.source_path),
        "phase_definition_sha256": file_sha256(phase.source_path),
        "stages": [stage.as_dict() for stage in stages],
        "ready_for_geant4": bundle.state == "complete",
        "ctmc": {
            "included": False,
            "interface": (
                "Charge exchange is a separate process; a q transition only "
                "selects another independently validated soft table."
            ),
        },
    }
    report["report_signature"] = _sha256_payload(report)
    return report


def _stage(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in report["stages"] if item["stage"] == name)


def build_plan(
    projectiles: Iterable[str],
    phases: Iterable[str],
    *,
    cp2k_exe: str | None = None,
    cp2k_module: str | None = None,
    resource_profile_path: str | Path = RESOURCE_PATH,
) -> dict[str, Any]:
    resources, resource_limits, resource_fallback = load_resource_profiles(
        resource_profile_path
    )

    def resource_record(stage: str) -> dict[str, Any]:
        return resources[stage].as_dict()

    reports = [
        inspect_model(projectile, phase)
        for projectile in projectiles
        for phase in phases
    ]
    tasks: list[dict[str, Any]] = []
    task_ids: set[str] = set()

    def add(task: dict[str, Any]) -> None:
        if task["task_id"] not in task_ids:
            task_ids.add(task["task_id"])
            tasks.append(task)

    for report in reports:
        symbol = report["projectile"]
        phase_id = report["phase_id"]
        projectile = get_projectile(symbol)
        phase = get_phase(phase_id)
        structure = _stage(report, "ice_structure")
        kernel = _stage(report, "nlh_kernel")
        hard = _stage(report, "hard_transport")
        soft_definition = _stage(report, "soft_dft_definition")
        soft_molecular = _stage(report, "soft_molecular_dft")
        hard_table = _stage(report, "geant4_hard_table")

        structure_task = f"structure:{phase_id}"
        if structure["state"] == "missing_input":
            pbs = REPOSITORY_ROOT / str(phase.preparation["pbs_script"])
            allocation = resources["ice_structure_generation"]
            add(
                {
                    "task_id": structure_task,
                    "stage": "ice_structure_generation",
                    "projectile": None,
                    "phase_id": phase_id,
                    "kind": "pbs",
                    "state": "runnable",
                    "dependencies": [],
                    "qsub_argv": [
                        "qsub",
                        "-l",
                        allocation.select,
                        *[str(value) for value in phase.preparation["qsub_arguments"]],
                        str(pbs),
                    ],
                    "resources": resource_record("ice_structure_generation"),
                    "completion_gate": (
                        "Dynamics must pass phase validation and be registered; "
                        "PBS success alone does not complete this stage."
                    ),
                }
            )

        kernel_task = f"kernel:{symbol}"
        if kernel["state"] == "runnable":
            allocation = resources["nlh_kernel_and_benchmark"]
            add(
                {
                    "task_id": kernel_task,
                    "stage": "nlh_kernel_and_benchmark",
                    "projectile": symbol,
                    "phase_id": None,
                    "kind": "pbs",
                    "state": "runnable",
                    "dependencies": [],
                    "qsub_argv": [
                        "qsub",
                        "-l",
                        allocation.select,
                        "-N",
                        f"nlh_k_{symbol}",
                        "-v",
                        f"PROJECTILE={symbol}",
                        str(REPOSITORY_ROOT / "pbs/run_ion_ice_nlh_kernel.pbs"),
                    ],
                    "resources": resource_record("nlh_kernel_and_benchmark"),
                    "completion_gate": "Kernel and independent benchmark must both pass.",
                }
            )

        if hard["state"] == "runnable" or (
            hard["state"] == "blocked"
            and structure["state"] == "complete"
            and kernel["state"] in {"complete", "runnable"}
            and projectile.component_status("hard_transport") == "implemented"
        ):
            dependencies = []
            if kernel_task in task_ids:
                dependencies.append(kernel_task)
            structure_registry = phase.resolve(phase.structure_registry)
            kernel_dir = (
                NEP_MBPOL_ROOT / "collision_kernels" / "by_species" / symbol
                if kernel_task in task_ids
                else NEP_MBPOL_ROOT / "collision_kernels"
            )
            if structure_registry is not None:
                structure_dir = structure_registry.parent
                allocation = resources["hard_transport"]
                add(
                    {
                        "task_id": f"hard_transport:{symbol}:{phase_id}",
                        "stage": "hard_transport",
                        "projectile": symbol,
                        "phase_id": phase_id,
                        "kind": "pbs",
                        "state": "runnable_after_dependencies"
                        if dependencies
                        else "runnable",
                        "dependencies": dependencies,
                        "qsub_argv": [
                            "qsub",
                            "-l",
                            allocation.select,
                            "-N",
                            f"nlh_{symbol}_{phase_id[:8]}",
                            "-v",
                            (
                                f"PROJECTILE={symbol},PHASE_ID={phase_id},"
                                f"STRUCTURE_DIRECTORY={structure_dir},KERNELS={kernel_dir}"
                            ),
                            str(
                                REPOSITORY_ROOT
                                / "pbs/run_ion_ice_hard_transport.pbs"
                            ),
                        ],
                        "resources": resource_record("hard_transport"),
                        "completion_gate": (
                            "Adaptive statistical and energy interpolation gates; "
                            "phase/orientation acceptance remains separate."
                        ),
                    }
                )

        if (
            soft_definition["state"] == "complete"
            and soft_molecular["state"] == "runnable"
        ):
            environment = []
            if cp2k_exe:
                environment.append(f"CP2K_EXE={Path(cp2k_exe).expanduser().resolve()}")
            if cp2k_module:
                environment.append(f"CP2K_MODULE={cp2k_module}")
            if environment:
                environment.append(f"PROJECTILE_DEFINITION={projectile.source_path}")
                allocation = resources["soft_molecular_dft"]
                add(
                    {
                        "task_id": f"soft_molecular_dft:{symbol}",
                        "stage": "soft_molecular_dft",
                        "projectile": symbol,
                        "phase_id": None,
                        "kind": "pbs",
                        "state": "runnable",
                        "dependencies": [],
                        "qsub_argv": [
                            "qsub",
                            "-l",
                            allocation.select,
                            "-N",
                            f"soft_dft_{symbol}",
                            "-v",
                            ",".join(environment),
                            str(
                                REPOSITORY_ROOT
                                / "pbs/run_adaptive_charge_resolved_dft.pbs"
                            ),
                        ],
                        "resources": resource_record("soft_molecular_dft"),
                        "completion_gate": (
                            "0.5% interpolation convergence, followed by all "
                            "documented physical DFT acceptance gates."
                        ),
                    }
                )
            else:
                add(
                    {
                        "task_id": f"soft_molecular_dft:{symbol}",
                        "stage": "soft_molecular_dft",
                        "projectile": symbol,
                        "phase_id": None,
                        "kind": "pbs",
                        "state": "blocked",
                        "dependencies": [],
                        "qsub_argv": [],
                        "resources": resource_record("soft_molecular_dft"),
                        "blockers": [
                            "Provide --cp2k-exe or --cp2k-module when building the plan."
                        ],
                    }
                )

        if hard_table["state"] == "runnable" and kernel["state"] == "complete":
            kernel_manifest = kernel["evidence"][0]
            add(
                {
                    "task_id": f"export_hard_table:{symbol}",
                    "stage": "geant4_hard_table",
                    "projectile": symbol,
                    "phase_id": None,
                    "kind": "local",
                    "state": "runnable",
                    "dependencies": [],
                    "command_argv": [
                        str(NEP_MBPOL_ROOT / ".venv/bin/python"),
                        str(NEP_MBPOL_ROOT / "export_nlh_geant4_table.py"),
                        "--projectile",
                        symbol,
                        "--source",
                        kernel_manifest,
                        "--release-status",
                        "atomistic_validation_pending",
                    ],
                    "completion_gate": "Output remains validation-pending.",
                }
            )

    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "repository": str(REPOSITORY_ROOT),
        "reports": reports,
        "tasks": tasks,
        "blocked_task_count": sum(task["state"] == "blocked" for task in tasks),
        "resource_policy": {
            "profile": str(Path(resource_profile_path).expanduser().resolve()),
            "profile_sha256": file_sha256(
                Path(resource_profile_path).expanduser().resolve()
            ),
            "limits": resource_limits,
            "uncalibrated_default": resource_fallback.as_dict(),
            "interpretation": (
                "Requested GB/CPU is recorded per PBS task. Profiles are "
                "promoted only after PBS accounting review; no task may exceed "
                "256 CPUs or 512 GB."
            ),
        },
        "submission_policy": (
            "Only explicit `submit --confirm` mutates PBS. Independent roots are "
            "submitted separately; dependent jobs use afterok."
        ),
    }
    payload["plan_signature"] = _sha256_payload(payload)
    return payload


def save_plan(plan: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    signature = plan.get("plan_signature")
    unsigned = dict(plan)
    unsigned.pop("plan_signature", None)
    if signature != _sha256_payload(unsigned):
        raise ValueError("Refusing to save an invalid workflow plan checksum.")
    _atomic_json(destination, plan)
    return destination


def _load_verified_plan(plan_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(plan_path).expanduser().resolve()
    plan = _read_json(path)
    signature = plan.pop("plan_signature", None)
    if signature != _sha256_payload(plan):
        raise ValueError("Workflow plan checksum mismatch.")
    plan["plan_signature"] = signature
    return path, plan


def _task_job_name(task: dict[str, Any]) -> str:
    if task["stage"] == "ice_structure_generation":
        return str(get_phase(task["phase_id"]).preparation["pbs_job_name"])
    command = task.get("qsub_argv", [])
    try:
        return str(command[command.index("-N") + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"PBS task {task['task_id']} has no exact job name.") from exc


def _require_task_still_runnable(task: dict[str, Any]) -> None:
    job_name = _task_job_name(task)
    active = _active_pbs_job_ids(job_name)
    if active:
        raise RuntimeError(
            f"Refusing duplicate {task['task_id']}: active PBS job(s) "
            + ", ".join(active)
        )
    if task.get("dependencies"):
        return
    stage = task["stage"]
    if stage == "ice_structure_generation":
        current = _structure_status(get_phase(task["phase_id"]))
        expected = "missing_input"
    else:
        report = inspect_model(task["projectile"], task.get("phase_id") or "hexagonal_ih_100k")
        report_stage = {
            "nlh_kernel_and_benchmark": "nlh_kernel",
            "hard_transport": "hard_transport",
            "soft_molecular_dft": "soft_molecular_dft",
        }[stage]
        current = StageStatus(**_stage(report, report_stage))
        expected = "runnable"
    if current.state != expected:
        raise RuntimeError(
            f"Plan is stale for {task['task_id']}: current state is "
            f"{current.state}, expected {expected}. Build a fresh plan."
        )


def submit_plan(plan_path: str | Path, *, confirm: bool) -> Path:
    if not confirm:
        raise ValueError("Submission requires explicit confirm=True.")
    path, plan = _load_verified_plan(plan_path)
    signature = plan["plan_signature"]
    if shutil.which("qsub") is None:
        raise RuntimeError("qsub is unavailable in this environment.")
    receipt_path = path.with_name(path.stem + ".submission.json")
    if receipt_path.exists():
        raise FileExistsError(
            f"Refusing duplicate submission: receipt already exists at {receipt_path}."
        )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "plan": str(path),
        "plan_signature": signature,
        "started_utc": _utc_now(),
        "jobs": [],
    }
    job_ids: dict[str, str] = {}
    _active_pbs_job_ids.cache_clear()
    for task in plan["tasks"]:
        if task["kind"] != "pbs" or task["state"] == "blocked":
            continue
        missing_dependencies = [
            dependency
            for dependency in task["dependencies"]
            if dependency not in job_ids
        ]
        if missing_dependencies:
            raise RuntimeError(
                f"Task {task['task_id']} has unsatisfied plan dependencies: "
                + ", ".join(missing_dependencies)
            )
        _require_task_still_runnable(task)
        command = list(task["qsub_argv"])
        dependency_ids = [job_ids[value] for value in task["dependencies"]]
        if dependency_ids:
            command[1:1] = ["-W", "depend=afterok:" + ":".join(dependency_ids)]
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        record = {
            "task_id": task["task_id"],
            "command": command,
            "return_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "submitted_utc": _utc_now(),
        }
        receipt["jobs"].append(record)
        _atomic_json(receipt_path, receipt)
        if completed.returncode != 0:
            raise RuntimeError(
                f"qsub failed for {task['task_id']}: {completed.stderr.strip()}"
            )
        job_id = completed.stdout.strip().splitlines()[-1]
        if not job_id:
            raise RuntimeError(f"qsub returned no job ID for {task['task_id']}.")
        job_ids[task["task_id"]] = job_id
        record["job_id"] = job_id
        _atomic_json(receipt_path, receipt)
    receipt["completed_utc"] = _utc_now()
    _atomic_json(receipt_path, receipt)
    return receipt_path


def run_local_tasks(plan_path: str | Path, *, confirm: bool) -> Path:
    """Run deterministic local reducers/exporters from a signed plan."""

    if not confirm:
        raise ValueError("Local execution requires explicit confirm=True.")
    path, plan = _load_verified_plan(plan_path)
    receipt_path = path.with_name(path.stem + ".local.json")
    if receipt_path.exists():
        raise FileExistsError(
            f"Refusing duplicate local execution: receipt already exists at {receipt_path}."
        )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "plan": str(path),
        "plan_signature": plan["plan_signature"],
        "started_utc": _utc_now(),
        "tasks": [],
    }
    for task in plan["tasks"]:
        if task["kind"] != "local" or task["state"] == "blocked":
            continue
        dependencies = task.get("dependencies", [])
        if dependencies:
            raise RuntimeError(
                f"Local task {task['task_id']} has unresolved PBS dependencies: "
                + ", ".join(dependencies)
            )
        command = list(task["command_argv"])
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        record = {
            "task_id": task["task_id"],
            "command": command,
            "return_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "completed_utc": _utc_now(),
        }
        receipt["tasks"].append(record)
        _atomic_json(receipt_path, receipt)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Local task {task['task_id']} failed: {completed.stderr.strip()}"
            )
    receipt["completed_utc"] = _utc_now()
    _atomic_json(receipt_path, receipt)
    return receipt_path


def assemble_geant4_bundle(
    projectile_value: str, phase_id: str, output_directory: str | Path
) -> Path:
    report = inspect_model(projectile_value, phase_id)
    if not report["ready_for_geant4"]:
        bundle = _stage(report, "geant4_bundle")
        raise RuntimeError(
            "Geant4 assembly refused: " + "; ".join(bundle["blockers"])
        )
    projectile = get_projectile(projectile_value)
    phase = get_phase(phase_id)
    hard_manifest = Path(_stage(report, "geant4_hard_table")["evidence"][0])
    soft_manifest = Path(_stage(report, "geant4_soft_table")["evidence"][0])
    hard_validation = Path(
        _stage(report, "hard_phase_validation")["evidence"][0]
    )
    payload = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "projectile": projectile.symbol,
        "phase_id": phase.phase_id,
        "species_definition": str(projectile.source_path),
        "species_definition_sha256": file_sha256(projectile.source_path),
        "phase_definition": str(phase.source_path),
        "phase_definition_sha256": file_sha256(phase.source_path),
        "hard_table_manifest": str(hard_manifest),
        "hard_table_manifest_sha256": file_sha256(hard_manifest),
        "hard_phase_validation": str(hard_validation),
        "hard_phase_validation_sha256": file_sha256(hard_validation),
        "soft_table_manifest": str(soft_manifest),
        "soft_table_manifest_sha256": file_sha256(soft_manifest),
        "ctmc_included": False,
        "release_status": "accepted",
    }
    payload["bundle_signature"] = _sha256_payload(payload)
    destination = Path(output_directory).expanduser().resolve()
    path = destination / f"ion_ice_{projectile.symbol}_{phase.phase_id}.json"
    _atomic_json(path, payload)
    return path
