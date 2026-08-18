"""Execute one CP2K CDFT branch gate with immutable, restart-safe evidence."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from .branch_solver import BranchValidationSettings, solve_cdft_branch
from .cdft_branch import (
    ProbeRecord,
    build_state_identity,
    load_validated_state,
    sha256_file,
)
from .config import CP2KSettings
from .cp2k import (
    CDFT_MODE_FIXED_LAMBDA,
    CDFT_MODE_OPTIMIZED,
    SCF_MODE_UNCONSTRAINED,
    parse_cp2k_output,
    render_cp2k_input,
)
from .density_validation import compare_density_cube_files


BRANCH_EXECUTION_VERSION = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _safe_label(label: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
    if not value or value in {".", ".."}:
        raise ValueError("CDFT execution label is empty or unsafe.")
    return value


def _project_name(task_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]", "_", task_id)[:100]
    if not value:
        raise ValueError("CDFT task ID produces no CP2K project name.")
    return value


def _final_trace_lagrangian(parsed: Mapping[str, Any]) -> float:
    trace = parsed.get("cdft_trace")
    if not isinstance(trace, list) or not trace or not isinstance(trace[-1], dict):
        raise RuntimeError("CDFT output has no final constrained-Lagrangian point.")
    value = trace[-1].get("energy")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError("CDFT output has no finite constrained Lagrangian.")
    return float(value)


def _verified_provenance_file(
    payload: Mapping[str, Any], label: str
) -> Path:
    path_value = payload.get("path")
    checksum = payload.get("sha256")
    size = payload.get("size_bytes")
    if (
        not isinstance(path_value, str)
        or not isinstance(checksum, str)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
    ):
        raise RuntimeError(f"CDFT {label} provenance is incomplete.")
    path = Path(path_value).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != size
        or sha256_file(path) != checksum
    ):
        raise RuntimeError(f"CDFT {label} provenance checksum mismatch.")
    return path


class CP2KBranchExecutor:
    """CP2K callback implementation for :func:`solve_cdft_branch`."""

    def __init__(
        self,
        *,
        task: Mapping[str, Any],
        settings: CP2KSettings,
        validation_settings: BranchValidationSettings,
        output_directory: Path,
        cp2k_command: str | Sequence[str],
        run_configuration_signature: str,
        state_family_configuration_signature: str,
        density_cube_stride: int,
        executable_provenance: Mapping[str, Any],
        parent_state: Mapping[str, Any] | None = None,
        progress: bool = True,
    ) -> None:
        if task.get("role") != "complex_cdft":
            raise ValueError("The CDFT branch executor accepts only complex tasks.")
        if density_cube_stride < 1:
            raise ValueError("CDFT density validation stride must be positive.")
        command = (
            shlex.split(cp2k_command)
            if isinstance(cp2k_command, str)
            else [str(value) for value in cp2k_command]
        )
        if not command:
            raise ValueError("CP2K command cannot be empty.")
        if not str(run_configuration_signature).strip():
            raise ValueError("CDFT run configuration signature cannot be empty.")
        if not str(state_family_configuration_signature).strip():
            raise ValueError("CDFT state-family signature cannot be empty.")
        if settings.cdft_optimizer != "BISECT":
            raise ValueError("Validated branch execution requires CP2K BISECT.")
        self.task = dict(task)
        self.settings = settings
        self.validation_settings = validation_settings
        self.output_directory = output_directory.resolve()
        self.command = command
        self.run_configuration_signature = run_configuration_signature
        self.state_family_configuration_signature = (
            state_family_configuration_signature
        )
        self.density_cube_stride = density_cube_stride
        self.executable_provenance = dict(executable_provenance)
        self._verify_executable_provenance()
        self.state_identity = build_state_identity(
            self.task,
            configuration_signature=state_family_configuration_signature,
            cp2k_settings=settings.as_dict(),
        )
        self.parent_state = self._validated_parent_state(parent_state)
        self.calibration_center_hartree = (
            0.0
            if self.parent_state is None
            else float(self.parent_state["multiplier_hartree"])
        )
        if not math.isfinite(self.calibration_center_hartree):
            raise RuntimeError("Parent CDFT multiplier is not finite.")
        self.progress = progress
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.solver_signature = _sha256_payload(
            {
                "branch_execution_version": BRANCH_EXECUTION_VERSION,
                "task": self.task,
                "cp2k_settings": settings.as_dict(),
                "validation_settings": validation_settings.as_dict(),
                "run_configuration_signature": run_configuration_signature,
                "state_family_configuration_signature": (
                    state_family_configuration_signature
                ),
                "state_identity": self.state_identity,
                "density_cube_stride": density_cube_stride,
                "command": command,
                "executable_provenance": self.executable_provenance,
                "parent_state_id": (
                    self.parent_state.get("state_id") if self.parent_state else None
                ),
                "calibration_center_hartree": self.calibration_center_hartree,
                "calibration_center_probe": self.parent_state is not None,
            }
        )

    def _verify_executable_provenance(self) -> None:
        provenance = self.executable_provenance
        if provenance.get("schema_version") != 1:
            raise RuntimeError("Unsupported CDFT executable-provenance schema.")
        recorded_command = provenance.get("command")
        if recorded_command != self.command:
            raise RuntimeError("CDFT command differs from its signed provenance.")

        launcher_payload = provenance.get("launcher")
        image_payload = provenance.get("image")
        python_payload = provenance.get("python")
        if not all(
            isinstance(value, dict)
            for value in (launcher_payload, image_payload, python_payload)
        ):
            raise RuntimeError("CDFT launcher, image, or Python provenance is absent.")
        launcher = _verified_provenance_file(launcher_payload, "launcher")
        image = _verified_provenance_file(image_payload, "container image")
        python = _verified_provenance_file(python_payload, "Python executable")
        command_launcher = Path(shutil.which(self.command[0]) or self.command[0]).resolve()
        if command_launcher != launcher:
            raise RuntimeError("CDFT command launcher path is not provenance-bound.")
        if Path(sys.executable).resolve() != python:
            raise RuntimeError("CDFT Python executable is not provenance-bound.")
        configured_image = os.environ.get("CP2K_IMAGE")
        if configured_image is None or Path(configured_image).resolve() != image:
            raise RuntimeError("CP2K_IMAGE is not the signed container image.")

        command_artifacts = provenance.get("command_artifacts")
        if not isinstance(command_artifacts, dict):
            raise RuntimeError("CDFT command-artifact provenance is malformed.")
        for raw_index, artifact in command_artifacts.items():
            if not isinstance(artifact, dict):
                raise RuntimeError("CDFT command-artifact provenance is malformed.")
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("CDFT command-artifact index is invalid.") from exc
            if index < 1 or index >= len(self.command):
                raise RuntimeError("CDFT command-artifact index is out of range.")
            artifact_path = _verified_provenance_file(
                artifact, f"command artifact {index}"
            )
            if Path(self.command[index]).resolve() != artifact_path:
                raise RuntimeError("CDFT command artifact path changed.")

        source_files = provenance.get("workflow_source_files")
        if not isinstance(source_files, dict) or not source_files:
            raise RuntimeError("CDFT workflow source hashes are absent.")
        for relative, artifact in source_files.items():
            if not isinstance(relative, str) or not isinstance(artifact, dict):
                raise RuntimeError("CDFT workflow source provenance is malformed.")
            _verified_provenance_file(artifact, f"workflow source {relative}")

        expected_cp2k = provenance.get("expected_cp2k")
        if not isinstance(expected_cp2k, dict):
            raise RuntimeError("Expected CP2K identity is absent.")
        expected_version = expected_cp2k.get("version")
        expected_revision = expected_cp2k.get("source_revision")
        if (
            expected_version != f"CP2K version {self.settings.cp2k_series}"
            or not isinstance(expected_revision, str)
            or not expected_revision
        ):
            raise RuntimeError("Expected CP2K version/revision is incompatible.")
        self.expected_cp2k_version = expected_version
        self.expected_cp2k_source_revision = expected_revision

        runtime = provenance.get("runtime")
        environment_names = {
            "cp2k_mpi_ranks": "CP2K_MPI_RANKS",
            "omp_num_threads": "OMP_NUM_THREADS",
            "openblas_num_threads": "OPENBLAS_NUM_THREADS",
            "mkl_num_threads": "MKL_NUM_THREADS",
        }
        if not isinstance(runtime, dict):
            raise RuntimeError("CDFT runtime provenance is absent.")
        expected_binary = runtime.get("cp2k_container_binary")
        actual_binary = os.environ.get(
            "CP2K_CONTAINER_BINARY", "/opt/cp2k/bin/cp2k"
        )
        if (
            not isinstance(expected_binary, str)
            or not expected_binary
            or actual_binary != expected_binary
        ):
            raise RuntimeError(
                "CDFT runtime setting CP2K_CONTAINER_BINARY is not provenance-bound."
            )
        for key, environment_name in environment_names.items():
            expected = runtime.get(key)
            actual_text = os.environ.get(environment_name, "1")
            if (
                isinstance(expected, bool)
                or not isinstance(expected, int)
                or expected < 1
                or not actual_text.isdigit()
                or int(actual_text) != expected
            ):
                raise RuntimeError(
                    f"CDFT runtime setting {environment_name} is not provenance-bound."
                )

    def _validate_cp2k_identity(self, parsed: Mapping[str, Any]) -> None:
        if parsed.get("cp2k_version") != self.expected_cp2k_version:
            raise RuntimeError(
                "CP2K output version differs from the signed executable identity."
            )
        if parsed.get("cp2k_source_revision") != self.expected_cp2k_source_revision:
            raise RuntimeError(
                "CP2K output source revision differs from the signed identity."
            )

    def _validated_parent_state(
        self, parent_state: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        if parent_state is None:
            return None
        state_path_value = parent_state.get("_state_path")
        if not isinstance(state_path_value, str) or not state_path_value:
            raise RuntimeError(
                "Parent CDFT state requires its immutable state-record path."
            )
        state_path = Path(state_path_value).resolve()
        loaded = load_validated_state(
            state_path,
            expected_identity=self.state_identity,
            same_geometry=False,
        )
        if loaded.get("state_id") != parent_state.get("state_id"):
            raise RuntimeError("Parent CDFT state identity does not match its record.")
        return {**loaded, "_state_path": str(state_path)}

    @property
    def checkpoint_path(self) -> Path:
        return self.output_directory / "branch.checkpoint.json"

    def _completed_runs(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        runs = self.output_directory / "runs"
        if not runs.is_dir():
            return records
        for path in sorted(runs.glob("*/run.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Malformed CDFT execution record: {path}"
                ) from exc
            if record.get("status") != "complete":
                continue
            self._verify_completed_record(record)
            record = self._reparse_completed_record(record)
            record["_record_path"] = str(path)
            records.append(record)
        return records

    def _verify_completed_record(self, record: Mapping[str, Any]) -> None:
        if record.get("solver_signature") != self.solver_signature:
            raise RuntimeError("Completed CDFT run has incompatible state identity.")
        execution_definition = {
            key: record.get(key)
            for key in (
                "solver_signature",
                "label",
                "kind",
                "strength_hartree",
                "mode",
                "step_size",
                "direction",
                "restart_sha256",
                "density_cube_stride",
                "scf_spin_mode",
            )
        }
        if _sha256_payload(execution_definition) != record.get(
            "execution_signature"
        ):
            raise RuntimeError("Completed CDFT run definition was modified.")
        for name in ("input", "output", "wavefunction"):
            path = Path(str(record.get(f"{name}_path", "")))
            expected = record.get(f"{name}_sha256")
            if (
                not path.is_file()
                or not isinstance(expected, str)
                or sha256_file(path) != expected
            ):
                raise RuntimeError(
                    f"Checkpointed CDFT {name} checksum mismatch."
                )
        restart_sha256 = record.get("restart_sha256")
        restart_path_value = record.get("restart_path")
        if restart_sha256 is not None:
            restart_path = Path(str(restart_path_value or ""))
            if (
                not isinstance(restart_sha256, str)
                or not restart_path.is_file()
                or sha256_file(restart_path) != restart_sha256
            ):
                raise RuntimeError(
                    "Checkpointed CDFT starting-wavefunction checksum mismatch."
                )
        elif restart_path_value is not None:
            raise RuntimeError("CDFT run has inconsistent restart provenance.")
        density_cubes = record.get("density_cubes", [])
        if not isinstance(density_cubes, list):
            raise RuntimeError("Checkpointed CDFT density provenance is malformed.")
        for cube in density_cubes:
            if not isinstance(cube, dict):
                raise RuntimeError(
                    "Checkpointed CDFT density provenance is malformed."
                )
            path = Path(str(cube.get("path", "")))
            if not path.is_file() or sha256_file(path) != cube.get("sha256"):
                raise RuntimeError("Checkpointed density-cube checksum mismatch.")

    def _reparse_completed_record(
        self, record: Mapping[str, Any]
    ) -> dict[str, Any]:
        mode = record.get("mode")
        if mode not in {
            CDFT_MODE_FIXED_LAMBDA,
            CDFT_MODE_OPTIMIZED,
            SCF_MODE_UNCONSTRAINED,
        }:
            raise RuntimeError("Checkpointed CDFT mode is invalid.")
        output = Path(str(record["output_path"]))
        parsed = parse_cp2k_output(
            output.read_text(encoding="utf-8", errors="replace"),
            require_cdft=mode != SCF_MODE_UNCONSTRAINED,
            cdft_mode=str(mode),
            cdft_tolerance=self.validation_settings.population_tolerance_electrons,
        )
        self._validate_cp2k_identity(parsed)
        accepted = (
            parsed["fixed_lambda_probe_accepted"]
            if mode == CDFT_MODE_FIXED_LAMBDA
            else parsed["valid_completion"]
        )
        if not accepted:
            raise RuntimeError("Checkpointed CDFT output no longer passes parsing.")
        lagrangian = (
            _final_trace_lagrangian(parsed)
            if mode != SCF_MODE_UNCONSTRAINED
            else None
        )
        if lagrangian is not None and not math.isclose(
            lagrangian, float(record.get("cdft_lagrangian_hartree")),
            rel_tol=0.0, abs_tol=1.0e-12,
        ):
            raise RuntimeError("Checkpointed constrained Lagrangian is inconsistent.")
        return {
            **record,
            **parsed,
            "cdft_lagrangian_hartree": lagrangian,
        }

    def _restart_wavefunction_for_probe(
        self,
        strength: float | None,
        label: str,
    ) -> Path | None:
        runs = self._completed_runs()
        if label in {"lower", "upper"}:
            direction = label
            candidates = [
                record
                for record in runs
                if record.get("kind") == "root"
                and record.get("direction") == direction
            ]
            if candidates:
                return self._verified_wavefunction(candidates[-1])
        if self.parent_state is None:
            # Independent fresh-anchor probes must each rebuild the declared
            # charge-specific atomic occupation.  Reusing a prior probe WFN
            # would silently bypass that initialization after the first run.
            return None
        sign = self._calibration_side(strength)
        candidates = [
            record
            for record in runs
            if record.get("kind") == "probe"
            and self._calibration_side(float(record["strength_hartree"])) == sign
            and self._calibration_side(float(record["strength_hartree"])) != 0
        ]
        if candidates:
            nearest = min(
                candidates,
                key=lambda record: abs(float(record["strength_hartree"]) - strength),
            )
            return self._verified_wavefunction(nearest)
        wavefunction = self.parent_state.get("wavefunction", {})
        path = Path(str(self.parent_state.get("_state_path", ""))).parent / str(
            wavefunction.get("path", "")
        )
        if path.is_file() and sha256_file(path) == wavefunction.get("sha256"):
            return path
        raise RuntimeError("Parent CDFT state wavefunction is incompatible.")

    def _calibration_side(self, strength: float) -> int:
        offset = strength - self.calibration_center_hartree
        if math.isclose(
            offset,
            0.0,
            rel_tol=0.0,
            abs_tol=self.validation_settings.endpoint_strength_tolerance_hartree,
        ):
            return 0
        return 1 if offset > 0.0 else -1

    def _restart_wavefunction_for_endpoint(self, probe: ProbeRecord) -> Path:
        matches = [
            record
            for record in self._completed_runs()
            if record.get("kind") == "probe"
            and record.get("output_sha256") == probe.output_sha256
            and math.isclose(
                float(record.get("strength_hartree")),
                probe.strength_hartree,
                rel_tol=0.0,
                abs_tol=self.validation_settings.endpoint_strength_tolerance_hartree,
            )
        ]
        if len(matches) != 1:
            raise RuntimeError("Cannot identify the paired bracket-endpoint WFN.")
        return self._verified_wavefunction(matches[0])

    @staticmethod
    def _verified_wavefunction(record: Mapping[str, Any]) -> Path:
        path = Path(str(record.get("wavefunction_path", "")))
        expected = record.get("wavefunction_sha256")
        if (
            not path.is_file()
            or not isinstance(expected, str)
            or sha256_file(path) != expected
        ):
            raise RuntimeError("Checkpointed CP2K wavefunction checksum mismatch.")
        return path

    def _load_existing_run(
        self,
        record_path: Path,
        execution_signature: str,
        *,
        mode: str,
    ) -> dict[str, Any] | None:
        if not record_path.is_file():
            return None
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("execution_signature") != execution_signature:
            raise RuntimeError("Existing CDFT execution directory is incompatible.")
        if record.get("status") != "complete":
            return None
        self._verify_completed_record(record)
        reparsed = self._reparse_completed_record(record)
        if reparsed.get("mode") != mode:
            raise RuntimeError("Checkpointed CDFT mode changed.")
        return reparsed

    @staticmethod
    def _discard_unaccepted_generated_artifacts(directory: Path) -> None:
        candidates = [
            directory / ".cp2k.out.tmp",
            directory / "cp2k.out",
            *directory.glob("*-RESTART.wfn"),
            *directory.glob("*.cube"),
        ]
        for path in candidates:
            if path.is_file():
                path.unlink()

    def _run_once(
        self,
        *,
        label: str,
        kind: str,
        strength: float,
        mode: str,
        restart_wavefunction: Path | None,
        step_size: float | None = None,
        direction: str | None = None,
        density_cube: bool = False,
    ) -> dict[str, Any]:
        safe_label = _safe_label(label)
        directory = self.output_directory / "runs" / safe_label
        directory.mkdir(parents=True, exist_ok=True)
        restart_sha256 = (
            sha256_file(restart_wavefunction)
            if restart_wavefunction is not None
            else None
        )
        execution_definition = {
            "solver_signature": self.solver_signature,
            "label": label,
            "kind": kind,
            "strength_hartree": strength,
            "mode": mode,
            "step_size": step_size,
            "direction": direction,
            "restart_sha256": restart_sha256,
            "density_cube_stride": self.density_cube_stride if density_cube else None,
            "scf_spin_mode": self.task.get("scf_spin_mode"),
        }
        execution_signature = _sha256_payload(execution_definition)
        record_path = directory / "run.json"
        existing = self._load_existing_run(
            record_path, execution_signature, mode=mode
        )
        if existing is not None:
            return existing
        self._discard_unaccepted_generated_artifacts(directory)

        start_wfn: Path | None = None
        start_wfn_name: str | None = None
        if restart_wavefunction is not None:
            start_wfn = directory / "starting.wfn"
            temporary = directory / ".starting.wfn.tmp"
            shutil.copyfile(restart_wavefunction, temporary)
            os.replace(temporary, start_wfn)
            start_wfn_name = start_wfn.name
        task = dict(self.task)
        task["task_id"] = f"{self.task['task_id']}_{safe_label}"
        input_text = render_cp2k_input(
            task,
            self.settings,
            cdft_strength=strength,
            cdft_mode=mode,
            cdft_step_size=step_size,
            wavefunction_restart=start_wfn_name,
            density_cube_stride=(self.density_cube_stride if density_cube else None),
            enable_cdft=mode != SCF_MODE_UNCONSTRAINED,
        )
        input_path = directory / "input.inp"
        _atomic_text(input_path, input_text)
        temporary_output = directory / ".cp2k.out.tmp"
        started = _utc_now()
        with temporary_output.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                [*self.command, "-i", input_path.name],
                cwd=directory,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            return_code = process.wait()
        text = temporary_output.read_text(encoding="utf-8", errors="replace")
        require_cdft = mode != SCF_MODE_UNCONSTRAINED
        parsed = parse_cp2k_output(
            text,
            require_cdft=require_cdft,
            cdft_mode=mode,
            cdft_tolerance=self.validation_settings.population_tolerance_electrons,
        )
        identity_error: str | None = None
        try:
            self._validate_cp2k_identity(parsed)
        except RuntimeError as exc:
            identity_error = str(exc)
        accepted = (
            parsed["fixed_lambda_probe_accepted"]
            if mode == CDFT_MODE_FIXED_LAMBDA
            else parsed["valid_completion"]
        ) and identity_error is None
        if return_code != 0 or not accepted:
            output_path = directory / "cp2k.failed.out"
            os.replace(temporary_output, output_path)
            _atomic_json(
                record_path,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "execution_signature": execution_signature,
                    **execution_definition,
                    "return_code": return_code,
                    "started_utc": started,
                    "finished_utc": _utc_now(),
                    "input_path": str(input_path),
                    "input_sha256": sha256_file(input_path),
                    "output_path": str(output_path),
                    "output_sha256": sha256_file(output_path),
                    "parsed": parsed,
                    "identity_error": identity_error,
                },
            )
            raise RuntimeError(f"CP2K CDFT execution failed: {output_path}")

        output_path = directory / "cp2k.out"
        os.replace(temporary_output, output_path)
        expected_wfn = directory / f"{_project_name(str(task['task_id']))}-RESTART.wfn"
        if not expected_wfn.is_file() or expected_wfn.stat().st_size == 0:
            alternatives = sorted(directory.glob("*-RESTART.wfn"))
            if len(alternatives) != 1:
                raise RuntimeError("CP2K did not produce one identifiable restart WFN.")
            expected_wfn = alternatives[0]
        density_paths = sorted(
            path
            for path in directory.glob("*.cube")
            if "density" in path.name.lower()
        )
        if density_cube and not density_paths:
            raise RuntimeError("CP2K root produced no requested density cube.")
        record = {
            "schema_version": 1,
            "status": "complete",
            "execution_signature": execution_signature,
            **execution_definition,
            "return_code": return_code,
            "started_utc": started,
            "finished_utc": _utc_now(),
            "input_path": str(input_path),
            "input_sha256": sha256_file(input_path),
            "output_path": str(output_path),
            "output_sha256": sha256_file(output_path),
            "wavefunction_path": str(expected_wfn),
            "wavefunction_sha256": sha256_file(expected_wfn),
            "restart_path": str(start_wfn) if start_wfn is not None else None,
            "density_cubes": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in density_paths
            ],
            "cdft_lagrangian_hartree": (
                _final_trace_lagrangian(parsed) if require_cdft else None
            ),
            **parsed,
        }
        _atomic_json(record_path, record)
        return record

    def _probe_record_from_run(self, run: Mapping[str, Any]) -> ProbeRecord:
        trace = run.get("cdft_trace")
        if not isinstance(trace, list) or not trace or not isinstance(trace[-1], dict):
            raise RuntimeError("Fixed-lambda probe has no CDFT trace.")
        point = trace[-1]
        branch_fingerprint = _sha256_payload(
            {
                "solver_signature": self.solver_signature,
                "state_identity": self.state_identity,
                "electron_count_alpha": run.get("electron_count_alpha"),
                "electron_count_beta": run.get("electron_count_beta"),
            }
        )
        return ProbeRecord(
            strength_hartree=float(point["strength"]),
            target_electrons=float(point["target"]),
            current_electrons=float(point["current"]),
            residual_electrons=float(point["residual"]),
            energy_hartree=float(run["cdft_lagrangian_hartree"]),
            inner_scf_converged=point.get("inner_scf_converged") is True,
            electron_count_alpha=run.get("electron_count_alpha"),
            electron_count_beta=run.get("electron_count_beta"),
            spin_squared=run.get("spin_squared_single_determinant"),
            branch_fingerprint=branch_fingerprint,
            output_sha256=str(run["output_sha256"]),
        )

    def probe(self, strength: float, label: str) -> ProbeRecord:
        restart = self._restart_wavefunction_for_probe(strength, label)
        run = self._run_once(
            label=f"probe_{label}",
            kind="probe",
            strength=strength,
            mode=CDFT_MODE_FIXED_LAMBDA,
            restart_wavefunction=restart,
        )
        return self._probe_record_from_run(run)

    def unconstrained_diagnostic(self) -> Mapping[str, Any]:
        """Run one fresh complex SCF with no CDFT section or restart WFN."""

        return self._run_once(
            label="unconstrained_scf_diagnostic",
            kind="unconstrained_scf_diagnostic",
            strength=None,
            mode=SCF_MODE_UNCONSTRAINED,
            restart_wavefunction=None,
        )

    def root(
        self,
        direction: str,
        start: ProbeRecord,
        opposite: ProbeRecord,
        step_size: float,
    ) -> Mapping[str, Any]:
        restart = self._restart_wavefunction_for_endpoint(start)
        return self._run_once(
            label=f"root_{direction}",
            kind="root",
            strength=start.strength_hartree,
            mode=CDFT_MODE_OPTIMIZED,
            restart_wavefunction=restart,
            step_size=step_size,
            direction=direction,
            density_cube=True,
        )

    @staticmethod
    def compare_root_densities(
        lower: Mapping[str, Any], upper: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        lower_cubes = lower.get("density_cubes")
        upper_cubes = upper.get("density_cubes")
        if not isinstance(lower_cubes, list) or not isinstance(upper_cubes, list):
            raise RuntimeError("Reciprocal roots lack density-cube provenance.")
        if len(lower_cubes) != len(upper_cubes) or not lower_cubes:
            raise RuntimeError("Reciprocal roots produced different density fields.")
        comparisons = []
        for first, second in zip(lower_cubes, upper_cubes):
            first_path = Path(str(first["path"]))
            second_path = Path(str(second["path"]))
            if sha256_file(first_path) != first.get("sha256") or sha256_file(
                second_path
            ) != second.get("sha256"):
                raise RuntimeError("Density-cube checksum mismatch.")
            comparisons.append(compare_density_cube_files(first_path, second_path))
        return {
            "field_count": len(comparisons),
            "fields": comparisons,
            "relative_rms": max(float(item["relative_rms"]) for item in comparisons),
        }

    def _reconstruct_checkpoint_from_run_evidence(self) -> None:
        if not self.checkpoint_path.is_file():
            return
        try:
            checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("Malformed CDFT branch checkpoint.") from exc
        if checkpoint.get("solver_signature") != self.solver_signature:
            raise RuntimeError("CDFT branch checkpoint belongs to another solver.")
        runs = self._completed_runs()
        indexed: dict[tuple[str, str], dict[str, Any]] = {}
        for run in runs:
            key = (str(run.get("kind")), str(run.get("label")))
            if key in indexed:
                raise RuntimeError("Duplicate immutable CDFT run evidence.")
            indexed[key] = run

        reconstructed_probes: dict[str, dict[str, dict[str, Any]]] = {}
        for category in ("calibration_probes", "curvature_probes"):
            values = checkpoint.get(category)
            if not isinstance(values, dict):
                raise RuntimeError(f"Malformed CDFT checkpoint field: {category}")
            rebuilt: dict[str, dict[str, Any]] = {}
            for label in values:
                if not isinstance(label, str):
                    raise RuntimeError("CDFT checkpoint probe label is invalid.")
                run = indexed.get(("probe", f"probe_{label}"))
                if run is None:
                    raise RuntimeError(
                        "CDFT checkpoint probe lacks checksum-verified run evidence."
                    )
                rebuilt[label] = self._probe_record_from_run(run).as_dict()
            reconstructed_probes[category] = rebuilt

        roots = checkpoint.get("roots")
        if not isinstance(roots, dict):
            raise RuntimeError("Malformed CDFT checkpoint roots.")
        rebuilt_roots: dict[str, dict[str, Any]] = {}
        for direction in roots:
            if direction not in {"lower", "upper"}:
                raise RuntimeError("CDFT checkpoint root direction is invalid.")
            run = indexed.get(("root", f"root_{direction}"))
            if run is None or run.get("direction") != direction:
                raise RuntimeError(
                    "CDFT checkpoint root lacks checksum-verified run evidence."
                )
            rebuilt_roots[direction] = {
                key: value for key, value in run.items() if key != "_record_path"
            }
            rebuilt_roots[direction]["injection_step_size"] = run.get("step_size")

        checkpoint.update(reconstructed_probes)
        checkpoint["roots"] = rebuilt_roots
        for derived in (
            "bracket",
            "reciprocal_validation",
            "curvature_validation",
            "directions",
        ):
            checkpoint.pop(derived, None)
        checkpoint["status"] = "calibrating"
        _atomic_json(self.checkpoint_path, checkpoint)

    def _validate_solver_evidence(self, result: Mapping[str, Any]) -> None:
        runs = self._completed_runs()
        indexed = {
            (str(record.get("kind")), str(record.get("label"))): record
            for record in runs
        }
        for category in ("calibration_probes", "curvature_probes"):
            probes = result.get(category)
            if not isinstance(probes, dict):
                raise RuntimeError("CDFT branch checkpoint has malformed probes.")
            for label, probe in probes.items():
                if not isinstance(probe, dict):
                    raise RuntimeError("CDFT branch checkpoint has malformed probes.")
                record = indexed.get(("probe", f"probe_{label}"))
                expected = (
                    self._probe_record_from_run(record).as_dict()
                    if record is not None
                    else None
                )
                if probe != expected:
                    raise RuntimeError(
                        "CDFT branch probe differs from immutable run evidence."
                    )
        roots = result.get("roots")
        if not isinstance(roots, dict) or set(roots) != {"lower", "upper"}:
            raise RuntimeError("CDFT branch checkpoint has incomplete roots.")
        for direction, root in roots.items():
            record = indexed.get(("root", f"root_{direction}"))
            if not isinstance(root, dict) or record is None:
                raise RuntimeError(
                    "CDFT root is detached from immutable run evidence."
                )
            expected = {
                key: value for key, value in record.items() if key != "_record_path"
            }
            if any(root.get(key) != value for key, value in expected.items()):
                raise RuntimeError("CDFT root differs from immutable run evidence.")
            injected = root.get("injection_step_size")
            if (
                isinstance(injected, bool)
                or not isinstance(injected, (int, float))
                or not math.isclose(
                    float(injected),
                    float(expected["step_size"]),
                    rel_tol=0.0,
                    abs_tol=(
                        self.validation_settings.endpoint_strength_tolerance_hartree
                    ),
                )
            ):
                raise RuntimeError("CDFT root injection step is inconsistent.")

    def solve(self) -> dict[str, Any]:
        self._verify_executable_provenance()
        self._reconstruct_checkpoint_from_run_evidence()
        result = solve_cdft_branch(
            self.checkpoint_path,
            solver_signature=self.solver_signature,
            settings=self.validation_settings,
            calibration_center_hartree=self.calibration_center_hartree,
            calibration_center_probe=self.parent_state is not None,
            probe_callback=self.probe,
            root_callback=self.root,
            density_comparison_callback=self.compare_root_densities,
            progress=self.progress,
        )
        self._validate_solver_evidence(result)
        return result
