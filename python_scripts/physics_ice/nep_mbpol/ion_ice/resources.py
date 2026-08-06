"""Strict PBS resource profiles and accounting-based calibration reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable


RESOURCE_SCHEMA_VERSION = 1
CALIBRATION_SCHEMA_VERSION = 1
RESOURCE_PATH = Path(__file__).resolve().with_name("resources.json")
PARALLEL_STAGES = frozenset(
    (
        "ice_structure_generation",
        "nlh_kernel_and_benchmark",
        "hard_transport",
        "soft_molecular_dft",
        "soft_dft_shard",
    )
)


@dataclass(frozen=True)
class ResourceProfile:
    stage: str
    ncpus: int
    memory_gb: int
    ngpus: int
    select_extras: tuple[tuple[str, str], ...]
    calibration_status: str
    evidence: str
    source_path: Path

    @property
    def requested_gb_per_cpu(self) -> float:
        return self.memory_gb / self.ncpus

    @property
    def select(self) -> str:
        value = f"select=1:ncpus={self.ncpus}:mem={self.memory_gb}gb"
        if self.ngpus:
            value += f":ngpus={self.ngpus}"
        for key, item in self.select_extras:
            value += f":{key}={item}"
        return value

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_path"] = str(self.source_path)
        value["select_extras"] = dict(self.select_extras)
        value["requested_gb_per_cpu"] = self.requested_gb_per_cpu
        value["select"] = self.select
        return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def load_resource_profiles(
    path: str | Path = RESOURCE_PATH,
) -> tuple[dict[str, ResourceProfile], dict[str, int], ResourceProfile]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RESOURCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported resource-profile schema in {source}.")
    limits = payload.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("Resource profiles require a limits object.")
    maximum_ncpus = _positive_integer(limits.get("maximum_ncpus"), "CPU limit")
    maximum_memory_gb = _positive_integer(
        limits.get("maximum_memory_gb"), "Memory limit"
    )
    if maximum_ncpus > 256 or maximum_memory_gb > 512:
        raise ValueError("Resource limits cannot exceed 256 CPUs or 512 GB.")

    def parse(stage: str, record: object) -> ResourceProfile:
        if not isinstance(record, dict):
            raise ValueError(f"Resource profile {stage} must be an object.")
        raw_extras = record.get("select_extras", {})
        if not isinstance(raw_extras, dict) or any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key))
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(value))
            for key, value in raw_extras.items()
        ):
            raise ValueError(f"Invalid PBS select extras for {stage}.")
        profile = ResourceProfile(
            stage=stage,
            ncpus=_positive_integer(record.get("ncpus"), f"{stage} CPUs"),
            memory_gb=_positive_integer(
                record.get("memory_gb"), f"{stage} memory"
            ),
            ngpus=_nonnegative_integer(record.get("ngpus", 0), f"{stage} GPUs"),
            select_extras=tuple(
                sorted((str(key), str(value)) for key, value in raw_extras.items())
            ),
            calibration_status=str(record.get("calibration_status", "")).strip(),
            evidence=str(record.get("evidence", "")).strip(),
            source_path=source,
        )
        if not profile.calibration_status or not profile.evidence:
            raise ValueError(f"Resource profile {stage} requires status and evidence.")
        if profile.ncpus > maximum_ncpus or profile.memory_gb > maximum_memory_gb:
            raise ValueError(f"Resource profile {stage} exceeds the declared limits.")
        return profile

    fallback = parse("uncalibrated_default", payload.get("uncalibrated_default"))
    if fallback.ncpus != 64 or fallback.memory_gb != 128:
        raise ValueError("The uncalibrated default must remain 64 CPUs and 128 GB.")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, dict) or set(raw_profiles) != PARALLEL_STAGES:
        raise ValueError(
            "Resource profiles must define exactly: "
            + ", ".join(sorted(PARALLEL_STAGES))
        )
    profiles = {stage: parse(stage, raw_profiles[stage]) for stage in raw_profiles}
    return profiles, {
        "maximum_ncpus": maximum_ncpus,
        "maximum_memory_gb": maximum_memory_gb,
    }, fallback


def _pbs_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        match = re.match(r"^\s{4}([A-Za-z_][A-Za-z0-9_.-]*)\s=\s(.*)$", line)
        if match:
            current = match.group(1)
            fields[current] = match.group(2).strip()
        elif current is not None and line.startswith("\t"):
            fields[current] += line.strip()
    return fields


_MEMORY_UNITS_GB = {
    "b": 1.0 / 1024**3,
    "kb": 1.0 / 1024**2,
    "mb": 1.0 / 1024,
    "gb": 1.0,
    "tb": 1024.0,
}


def _memory_gb(value: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([kmgt]?b)", value.lower())
    if match is None:
        raise ValueError(f"Unsupported PBS memory value {value!r}.")
    return float(match.group(1)) * _MEMORY_UNITS_GB[match.group(2)]


def _duration_seconds(value: str) -> float:
    fields = value.split(":")
    if len(fields) != 3:
        raise ValueError(f"Unsupported PBS duration {value!r}.")
    hours, minutes, seconds = (float(item) for item in fields)
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_pbs_accounting(text: str, requested_job_id: str) -> dict[str, Any]:
    fields = _pbs_fields(text)
    required = (
        "job_state",
        "Resource_List.ncpus",
        "Resource_List.mem",
        "resources_used.mem",
        "resources_used.cput",
        "resources_used.walltime",
    )
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(
            f"PBS accounting for {requested_job_id} lacks: " + ", ".join(missing)
        )
    ncpus = int(fields["Resource_List.ncpus"])
    used_memory_gb = _memory_gb(fields["resources_used.mem"])
    walltime_seconds = _duration_seconds(fields["resources_used.walltime"])
    cput_seconds = _duration_seconds(fields["resources_used.cput"])
    average_used_cores = (
        cput_seconds / walltime_seconds if walltime_seconds > 0.0 else 0.0
    )
    return {
        "job_id": fields.get("Job_Id", requested_job_id),
        "job_name": fields.get("Job_Name"),
        "state": fields["job_state"],
        "exit_status": (
            int(fields["Exit_status"]) if "Exit_status" in fields else None
        ),
        "allocated_ncpus": ncpus,
        "requested_memory_gb": _memory_gb(fields["Resource_List.mem"]),
        "used_memory_gb": used_memory_gb,
        "used_gb_per_allocated_cpu": used_memory_gb / ncpus,
        "walltime_seconds": walltime_seconds,
        "cpu_time_seconds": cput_seconds,
        "average_used_cores": average_used_cores,
        "average_cpu_utilization": average_used_cores / ncpus,
        "accounting_complete": fields["job_state"] == "F"
        and fields.get("Exit_status") is not None,
    }


def query_pbs_accounting(job_id: str) -> dict[str, Any]:
    if shutil.which("qstat") is None:
        raise RuntimeError("qstat is unavailable in this environment.")
    completed = subprocess.run(
        ["qstat", "-xf", job_id],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"qstat failed for {job_id}: {completed.stderr.strip()}"
        )
    return parse_pbs_accounting(completed.stdout, job_id)


def _round_up_multiple(value: float, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _power_of_two_at_least(value: float) -> int:
    return 1 << max(0, math.ceil(math.log2(max(1.0, value))))


def calibrate_resources(stage: str, job_ids: Iterable[str]) -> dict[str, Any]:
    profiles, limits, _ = load_resource_profiles()
    if stage not in profiles:
        raise ValueError(f"Unknown parallel stage {stage!r}.")
    identifiers = tuple(dict.fromkeys(str(value).strip() for value in job_ids))
    if not identifiers or any(not value for value in identifiers):
        raise ValueError("At least one non-empty PBS job ID is required.")
    observations = [query_pbs_accounting(job_id) for job_id in identifiers]
    maximum_allocated = max(item["allocated_ncpus"] for item in observations)
    maximum_average_cores = max(item["average_used_cores"] for item in observations)
    recommended_ncpus = min(
        limits["maximum_ncpus"],
        maximum_allocated,
        max(1, _round_up_multiple(maximum_average_cores / 0.85, 4)),
    )
    maximum_per_cpu = max(
        item["used_gb_per_allocated_cpu"] for item in observations
    )
    maximum_total_memory = max(item["used_memory_gb"] for item in observations)
    # Twofold total and per-core RSS headroom plus a 4 GB total floor. Running
    # observations are lower bounds and are never promoted automatically.
    recommended_memory_gb = min(
        limits["maximum_memory_gb"],
        _power_of_two_at_least(
            max(
                4.0,
                2.0 * maximum_total_memory,
                2.0 * maximum_per_cpu * recommended_ncpus,
            )
        ),
    )
    completed_successfully = all(
        item["accounting_complete"] and item["exit_status"] == 0
        for item in observations
    )
    current = profiles[stage]
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "observations": observations,
        "all_jobs_completed_successfully": completed_successfully,
        "current_profile": current.as_dict(),
        "proposed_profile": {
            "ncpus": recommended_ncpus,
            "memory_gb": recommended_memory_gb,
            "requested_gb_per_cpu": recommended_memory_gb / recommended_ncpus,
            "ngpus": current.ngpus,
            "select_extras": dict(current.select_extras),
            "calibration_status": (
                "completed_pbs_accounting"
                if completed_successfully
                else "provisional_incomplete_pbs_accounting"
            ),
            "headroom_rule": (
                "2x observed total peak RSS and per-allocated-CPU RSS, "
                "4 GB total floor; "
                "CPU count targets at most 85% average utilization"
            ),
        },
        "limits": limits,
        "promotion_policy": (
            "Review scaling and task concurrency, then update resources.json "
            "manually. Incomplete-job observations are lower bounds and cannot "
            "replace a production profile."
        ),
    }
