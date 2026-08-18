"""State and numerical invariants for reproducible one-constraint CDFT branches.

This module contains no CP2K execution policy.  It defines the immutable
wavefunction--multiplier pair, the empirical bracket algebra, and the
continuation checks used by the runner.  A fixed-strength probe is diagnostic;
only reciprocal, fully converged CP2K outer-CDFT roots can publish a state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping


CDFT_BRANCH_SCHEMA_VERSION = 1
RECIPROCAL_VALIDATION_SCOPE = "reciprocal_numerical_cdft_root"
PHYSICAL_STATE_STATUS = "validation_pending"
RECIPROCAL_VALIDATION_LIMITATION = (
    "Static electronic-sector, reciprocal-root, spin, and density agreement "
    "does not establish orbital or charge-localization branch uniqueness."
)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        with Path(temporary).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class CalibrationSettings:
    """Exposed numerical limits for empirical multiplier bracketing.

    These values control a root search; they are not physical parameters or
    estimates of the converged CDFT multiplier.
    """

    initial_step_hartree: float
    expansion_factor: float
    max_abs_strength_hartree: float
    max_probe_count: int
    include_zero_diagnostic: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_step_hartree) or (
            self.initial_step_hartree <= 0.0
        ):
            raise ValueError("Initial CDFT calibration step must be positive.")
        if not math.isfinite(self.expansion_factor) or self.expansion_factor <= 1.0:
            raise ValueError("CDFT calibration expansion factor must exceed one.")
        if not math.isfinite(self.max_abs_strength_hartree) or (
            self.max_abs_strength_hartree < self.initial_step_hartree
        ):
            raise ValueError(
                "Maximum CDFT calibration strength must include the first step."
            )
        if self.max_probe_count < 2:
            raise ValueError("CDFT calibration requires at least two probes.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeRecord:
    """One inner-converged population measurement at a fixed multiplier."""

    strength_hartree: float
    target_electrons: float
    current_electrons: float
    residual_electrons: float
    energy_hartree: float
    inner_scf_converged: bool
    electron_count_alpha: int | None = None
    electron_count_beta: int | None = None
    spin_squared: float | None = None
    # A static electronic-sector identifier.  Equality does not establish
    # orbital or localization-branch identity.
    branch_fingerprint: str | None = None
    output_sha256: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("strength", self.strength_hartree),
            ("target", self.target_electrons),
            ("population", self.current_electrons),
            ("residual", self.residual_electrons),
            ("energy", self.energy_hartree),
        ):
            if not math.isfinite(value):
                raise ValueError(f"Probe {name} must be finite.")
        if not math.isclose(
            self.current_electrons - self.target_electrons,
            self.residual_electrons,
            rel_tol=1.0e-10,
            abs_tol=1.0e-10,
        ):
            raise ValueError("Probe residual is inconsistent with population-target.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CDFTBracket:
    """Two same-sector probes enclosing one negative-response root."""

    lower: ProbeRecord
    upper: ProbeRecord
    secant_response_electrons_per_hartree: float

    def __post_init__(self) -> None:
        if not self.lower.strength_hartree < self.upper.strength_hartree:
            raise ValueError("CDFT bracket strengths must be strictly ordered.")
        if self.lower.residual_electrons * self.upper.residual_electrons >= 0.0:
            raise ValueError("CDFT bracket residuals do not enclose zero.")
        if not self.secant_response_electrons_per_hartree < 0.0:
            raise ValueError("CDFT population response must be negative.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower.as_dict(),
            "upper": self.upper.as_dict(),
            "secant_response_electrons_per_hartree": (
                self.secant_response_electrons_per_hartree
            ),
        }


def calibration_strengths(settings: CalibrationSettings) -> tuple[float, ...]:
    """Return a deterministic two-sided search without a physical seed value."""

    values: list[float] = [0.0] if settings.include_zero_diagnostic else []
    magnitude = settings.initial_step_hartree
    while (
        len(values) < settings.max_probe_count
        and magnitude <= settings.max_abs_strength_hartree
    ):
        for sign in (1.0, -1.0):
            if len(values) >= settings.max_probe_count:
                break
            values.append(sign * magnitude)
        magnitude *= settings.expansion_factor
    return tuple(values)


def find_negative_response_bracket(
    records: Iterable[ProbeRecord],
) -> CDFTBracket:
    """Select the narrowest adjacent sign change with the expected response.

    Only inner-converged probes participate.  A shared non-null static-sector
    fingerprint is required when any record provides one.  This excludes
    explicitly different spin/electron sectors but does not prove orbital or
    localization-branch identity.
    """

    usable = sorted(
        (
            record
            for record in records
            if record.inner_scf_converged and record.residual_electrons != 0.0
        ),
        key=lambda record: record.strength_hartree,
    )
    if len(usable) < 2:
        raise ValueError("Fewer than two inner-converged CDFT probes are available.")
    candidates: list[CDFTBracket] = []
    for lower, upper in zip(usable[:-1], usable[1:]):
        if lower.strength_hartree == upper.strength_hartree:
            continue
        fingerprints = {
            value
            for value in (lower.branch_fingerprint, upper.branch_fingerprint)
            if value is not None
        }
        if len(fingerprints) > 1:
            continue
        if lower.residual_electrons * upper.residual_electrons >= 0.0:
            continue
        slope = (
            upper.current_electrons - lower.current_electrons
        ) / (upper.strength_hartree - lower.strength_hartree)
        if slope >= 0.0:
            continue
        candidates.append(CDFTBracket(lower, upper, slope))
    if not candidates:
        raise ValueError(
            "No same-sector sign-changing bracket with negative dN/dlambda was found."
        )
    return min(
        candidates,
        key=lambda bracket: (
            bracket.upper.strength_hartree - bracket.lower.strength_hartree,
            abs(bracket.lower.residual_electrons)
            + abs(bracket.upper.residual_electrons),
        ),
    )


def bisect_injection_step_size(
    start: ProbeRecord,
    opposite: ProbeRecord,
) -> float:
    """Return CP2K's source-defined first SD step that visits the other side.

    CP2K 2025.2 falls from BISECT to DIIS to steepest descent until its live
    history contains opposite residual signs, and then applies
    ``lambda_next = lambda - STEP_SIZE * residual``.
    """

    if not (start.inner_scf_converged and opposite.inner_scf_converged):
        raise ValueError("BISECT endpoints must have converged inner SCF probes.")
    if start.residual_electrons == 0.0:
        raise ValueError("An exact-root endpoint does not need bracket injection.")
    if start.residual_electrons * opposite.residual_electrons >= 0.0:
        raise ValueError("BISECT injection endpoints must have opposite signs.")
    return (
        start.strength_hartree - opposite.strength_hartree
    ) / start.residual_electrons


def continuation_midpoint(
    accepted_separation_angstrom: float,
    failed_separation_angstrom: float,
    *,
    minimum_step_angstrom: float,
) -> float:
    """Bisect one failed mesh interval without embedding a distance ladder."""

    values = (
        accepted_separation_angstrom,
        failed_separation_angstrom,
        minimum_step_angstrom,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Continuation distances must be finite.")
    if minimum_step_angstrom <= 0.0:
        raise ValueError("Minimum continuation step must be positive.")
    interval = abs(accepted_separation_angstrom - failed_separation_angstrom)
    if interval <= minimum_step_angstrom:
        raise ValueError("Failed continuation interval is already at its step limit.")
    return 0.5 * (accepted_separation_angstrom + failed_separation_angstrom)


def build_state_identity(
    task: Mapping[str, Any],
    *,
    configuration_signature: str,
    cp2k_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a state to its electronic family and exact nuclear geometry."""

    coordinates = task.get("coordinates_angstrom")
    if not isinstance(coordinates, (list, tuple)) or not coordinates:
        raise ValueError("CDFT state identity requires ordered coordinates.")
    atom_order = [str(row[0]) for row in coordinates]
    geometry = {
        "orientation": task.get("orientation"),
        "separation_angstrom": task.get("separation_angstrom"),
        "coordinates_angstrom": coordinates,
        "atom_order": atom_order,
    }
    family = {
        "configuration_signature": configuration_signature,
        "projectile": task.get("projectile"),
        "charge": task.get("charge"),
        "electrons_on_projectile": task.get("electrons_on_projectile"),
        "multiplicity": task.get("multiplicity"),
        "scf_spin_mode": task.get("scf_spin_mode"),
        "constraint_type": cp2k_settings.get("cdft_constraint_type"),
        "constraint_target": task.get("electrons_on_projectile"),
        "atom_order": atom_order,
        "cp2k_settings": dict(cp2k_settings),
    }
    return {
        "task_id": task.get("task_id"),
        "family": family,
        "geometry": geometry,
        "family_signature": _sha256_bytes(_canonical_json(family)),
        "geometry_signature": _sha256_bytes(_canonical_json(geometry)),
    }


def state_identity_compatible(
    predecessor: Mapping[str, Any],
    successor: Mapping[str, Any],
    *,
    same_geometry: bool,
) -> bool:
    """Check whether one accepted pair may initialize another calculation."""

    if predecessor.get("family_signature") != successor.get("family_signature"):
        return False
    if same_geometry and (
        predecessor.get("geometry_signature")
        != successor.get("geometry_signature")
    ):
        return False
    return True


def _validated_state_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    state_id = record.pop("state_id", None)
    if not isinstance(state_id, str) or state_id != _sha256_bytes(
        _canonical_json(record)
    ):
        raise RuntimeError("CDFT state-record checksum mismatch.")
    return record


def load_validated_state(
    state_path: Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    same_geometry: bool = True,
) -> dict[str, Any]:
    """Load an immutable WFN--multiplier pair and verify every checksum."""

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    record = _validated_state_record(payload)
    if record.get("schema_version") != CDFT_BRANCH_SCHEMA_VERSION:
        raise RuntimeError("Unsupported CDFT state schema.")
    if record.get("status") != "validated":
        raise RuntimeError("Only a validated CDFT branch may be restarted.")
    identity = record.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("CDFT state identity is absent.")
    if expected_identity is not None and not state_identity_compatible(
        identity, expected_identity, same_geometry=same_geometry
    ):
        raise RuntimeError("CDFT restart state is incompatible with the task.")
    for label in ("wavefunction", "input", "output"):
        artifact = record.get(label, {})
        name = artifact.get("path") if isinstance(artifact, dict) else None
        if not isinstance(name, str) or Path(name).name != name:
            raise RuntimeError(f"CDFT state {label} path is invalid.")
        artifact_path = state_path.parent / name
        if not artifact_path.is_file() or sha256_file(
            artifact_path
        ) != artifact.get("sha256"):
            raise RuntimeError(f"CDFT state {label} checksum mismatch.")
    return {**payload, "_state_path": str(state_path.resolve())}


def publish_validated_state(
    state_path: Path,
    source_wavefunction: Path,
    *,
    identity: Mapping[str, Any],
    multiplier_hartree: float,
    target_electrons: float,
    current_electrons: float,
    residual_electrons: float,
    energy_hartree: float,
    electron_count_alpha: int | None,
    electron_count_beta: int | None,
    spin_squared: float | None,
    cdft_trace: Iterable[Mapping[str, Any]],
    branch_validation: Mapping[str, Any],
    cp2k_identity: Mapping[str, Any],
    input_path: Path,
    output_path: Path,
    parent_state_id: str | None,
) -> dict[str, Any]:
    """Atomically publish an accepted state after reciprocal branch validation."""

    numeric = (
        multiplier_hartree,
        target_electrons,
        current_electrons,
        residual_electrons,
        energy_hartree,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Accepted CDFT state values must be finite.")
    if not math.isclose(
        current_electrons - target_electrons,
        residual_electrons,
        rel_tol=1.0e-10,
        abs_tol=1.0e-10,
    ):
        raise ValueError("Accepted CDFT residual is inconsistent.")
    if branch_validation.get("status") != "validated":
        raise ValueError("Reciprocal branch validation has not passed.")
    if branch_validation.get("validation_scope") != RECIPROCAL_VALIDATION_SCOPE:
        raise ValueError("CDFT branch validation scope is absent or incompatible.")
    if branch_validation.get("physical_state_status") != PHYSICAL_STATE_STATUS:
        raise ValueError("CDFT physical-state status must remain validation_pending.")
    if (
        branch_validation.get("validation_limitation")
        != RECIPROCAL_VALIDATION_LIMITATION
    ):
        raise ValueError("CDFT branch-validation limitation is absent.")
    directions = branch_validation.get("directions")
    if (
        not isinstance(directions, list)
        or len(directions) != 2
        or set(directions) != {"lower", "upper"}
    ):
        raise ValueError("Validation must contain lower- and upper-start roots.")
    if not source_wavefunction.is_file() or source_wavefunction.stat().st_size == 0:
        raise FileNotFoundError("Accepted CP2K wavefunction is absent or empty.")
    if (
        not input_path.is_file()
        or input_path.stat().st_size == 0
        or not output_path.is_file()
        or output_path.stat().st_size == 0
    ):
        raise FileNotFoundError("Accepted CDFT input/output evidence is absent.")

    state_path.parent.mkdir(parents=True, exist_ok=True)
    destinations = {
        "wavefunction": state_path.parent / "state.wfn",
        "input": state_path.parent / "state.inp",
        "output": state_path.parent / "state.out",
    }
    sources = {
        "wavefunction": source_wavefunction,
        "input": input_path,
        "output": output_path,
    }
    artifact_records = {
        label: {
            "path": destinations[label].name,
            "sha256": sha256_file(source),
        }
        for label, source in sources.items()
    }
    immutable_record: dict[str, Any] = {
        "schema_version": CDFT_BRANCH_SCHEMA_VERSION,
        "status": "validated",
        "identity": dict(identity),
        "multiplier_hartree": multiplier_hartree,
        "target_electrons": target_electrons,
        "current_electrons": current_electrons,
        "residual_electrons": residual_electrons,
        "energy_hartree": energy_hartree,
        "electron_count_alpha": electron_count_alpha,
        "electron_count_beta": electron_count_beta,
        "spin_squared": spin_squared,
        "cdft_trace": [dict(point) for point in cdft_trace],
        "branch_validation": dict(branch_validation),
        "cp2k_identity": dict(cp2k_identity),
        "parent_state_id": parent_state_id,
        "validation_scope": RECIPROCAL_VALIDATION_SCOPE,
        "physical_state_status": PHYSICAL_STATE_STATUS,
        "validation_limitation": RECIPROCAL_VALIDATION_LIMITATION,
        **artifact_records,
    }
    if state_path.exists():
        existing = load_validated_state(state_path)
        existing.pop("_state_path", None)
        comparable = dict(existing)
        comparable.pop("state_id", None)
        comparable.pop("published_utc", None)
        if comparable != immutable_record:
            raise RuntimeError("Refusing to overwrite a different CDFT state pair.")
        return existing

    for label, source in sources.items():
        _atomic_copy(source, destinations[label])
        if sha256_file(destinations[label]) != artifact_records[label]["sha256"]:
            raise RuntimeError(f"Published CDFT {label} checksum mismatch.")
    record = {
        **immutable_record,
        "published_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload = {**record, "state_id": _sha256_bytes(_canonical_json(record))}
    _atomic_json(state_path, payload)
    return payload
