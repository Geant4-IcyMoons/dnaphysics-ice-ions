#!/usr/bin/env python3
"""Reproduce the supplied CTMC81 C3+ 100-keV/u primitive tables.

This is a validation run, not a replacement for the production energy grid.
Every reference impact parameter, trajectory count, initial separation,
minimum integration interval and terminal-distance gate is read from the
immutable user-supplied archive.
Independent exact-size replicas provide a precision improvement without
changing any one replica's parameter match.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402
from tqdm import tqdm  # noqa: E402


HERE = Path(__file__).resolve().parent
PHYSICS_ICE_ROOT = HERE.parents[2]
sys.path.insert(0, str(PHYSICS_ICE_ROOT))

import charge_exchange_ctmc as ctmc  # noqa: E402
from benchmark_carbon_charge_exchange_ctmc import (  # noqa: E402
    FORMAL_REFERENCE_ARCHIVE,
    FORMAL_REFERENCE_SHA256,
    PAPER_DOI,
    _read_formal_reference_archive,
)
from constants import (  # noqa: E402
    AASTEX_FULL_WIDTH_IN,
    PAPER_FONTSIZE,
    RC_BASE_STANDARD,
    rcparams_with_fontsize,
)


IMPLEMENTATION_VERSION = 4
DEFAULT_OUTPUT_DIR = (
    HERE / "runs" / "ctmc81_c3_100keV_u_olson_salop_reproduction"
)
REFERENCE_LABELS = ("L1", "L2", "L3", "L4", "L5", "loss")
_WORKER_CONFIG: ctmc.CTMCConfig | None = None
_WORKER_SPECS: tuple[dict[str, Any], ...] = ()
_STOP_REQUESTED = False


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _build_specs(reference: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    point_index = 0
    for dataset_index, label in enumerate(REFERENCE_LABELS):
        dataset = reference["datasets"][label]
        metadata = dataset["metadata"]
        for impact_index, impact in enumerate(dataset["impact_au"]):
            specs.append(
                {
                    "point_index": point_index,
                    "dataset_index": dataset_index,
                    "dataset": label,
                    "impact_index": impact_index,
                    "impact_au": float(impact),
                    "projectile_velocity_au": float(
                        metadata["projectile_velocity_au"]
                    ),
                    "orbital_index": (
                        None
                        if label == "loss"
                        else int(metadata["orbital_index"]) - 1
                    ),
                    "trajectory_count": int(metadata["trajectory_count"]),
                    "start_separation_au": float(metadata["initial_z_au"]),
                    "integration_time_au": float(
                        metadata["integration_time_au"]
                    ),
                    "limit_distance_au": float(
                        metadata["limit_distance_au"]
                    ),
                }
            )
            point_index += 1
    return tuple(specs)


def _configuration(
    *,
    trajectory_chunk_size: int,
    seed: int,
    rtol: float,
    atol: float,
    retry_rtol: float,
    retry_atol: float,
    maximum_relative_energy_drift: float,
    maximum_integration_steps: int,
) -> ctmc.CTMCConfig:
    return ctmc.CTMCConfig(
        backend="numba",
        trajectories=20_000,
        trajectory_chunk_size=trajectory_chunk_size,
        target_bmax_au=(10.0, 10.0, 5.0, 5.0, 1.0),
        loss_bmax_au=(10.0,) * 6,
        radial_grid_points=20_000,
        start_separation_au=(1_000.0,),
        boundary_extension_factor=2.0,
        minimum_integration_time_au=(1_000.0,),
        rtol=rtol,
        atol=atol,
        retry_rtol=retry_rtol,
        retry_atol=retry_atol,
        max_step_au=math.inf,
        minimum_radius_au=1.0e-10,
        maximum_relative_energy_drift=maximum_relative_energy_drift,
        max_failure_fraction=0.0,
        maximum_integration_steps=maximum_integration_steps,
        seed=seed,
        projectile="carbon",
        initial_ensemble=ctmc.INITIAL_ENSEMBLE_LIAMSUWAN_OLSON_SALOP,
        projectile_loss_initialization=(
            ctmc.PROJECTILE_LOSS_INITIALIZATION_CTMC81_NUCLEUS
        ),
    )


def _signature(
    reference: dict[str, Any],
    specs: tuple[dict[str, Any], ...],
    config: ctmc.CTMCConfig,
    replicas: int,
) -> str:
    config_payload = asdict(config)
    config_payload["max_step_au"] = None
    payload = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "reference_sha256": reference["sha256"],
        "engine_sha256": _file_sha256(PHYSICS_ICE_ROOT / "charge_exchange_ctmc.py"),
        "numba_backend_sha256": _file_sha256(
            PHYSICS_ICE_ROOT / "ctmc_numba_backend.py"
        ),
        "replicas": replicas,
        "specs": specs,
        "config": config_payload,
        "endpoint": "minimum_tEND_then_disclosed_limit_distance",
        "ambiguous_endpoint": "retained_as_null",
        "initial_ensemble": config.initial_ensemble,
        "projectile_loss_initialization": (
            config.projectile_loss_initialization
        ),
        "random_draws_per_trajectory": ctmc.random_draws_per_trajectory(
            config.initial_ensemble
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _init_worker(
    config: ctmc.CTMCConfig,
    specs: tuple[dict[str, Any], ...],
) -> None:
    global _WORKER_CONFIG, _WORKER_SPECS
    _WORKER_CONFIG = config
    _WORKER_SPECS = specs
    ctmc.select_projectile("carbon")
    ctmc.warm_up_initial_state_sampling(
        np.asarray([3], dtype=int), config.radial_grid_points
    )
    for signal_name in ("SIGTERM", "SIGUSR1"):
        worker_signal = getattr(signal, signal_name, None)
        if worker_signal is not None:
            signal.signal(worker_signal, signal.SIG_DFL)


def _rng(replica: int, point_index: int, trajectory_start: int) -> np.random.Generator:
    if _WORKER_CONFIG is None:
        raise RuntimeError("CTMC81 worker is not initialized")
    seed = np.random.SeedSequence(
        [_WORKER_CONFIG.seed, replica, point_index, 81]
    )
    bit_generator = np.random.PCG64(seed)
    bit_generator.advance(
        trajectory_start
        * ctmc.random_draws_per_trajectory(
            _WORKER_CONFIG.initial_ensemble
        )
    )
    return np.random.Generator(bit_generator)


def _event_index(dataset: str, outcome: str) -> int:
    if outcome == "ionized":
        return 0
    if dataset == "loss":
        if outcome == "retained_target":
            return 1
        if outcome == "captured_projectile":
            return 2
    else:
        if outcome == "captured_projectile":
            return 1
        if outcome == "retained_target":
            return 2
    if outcome == "ambiguous_bound":
        return 3
    raise RuntimeError(f"Unexpected CTMC outcome {outcome!r}")


def _run_chunk(task: tuple[int, int, int, int]) -> tuple[Any, ...]:
    replica, point_index, trajectory_start, trajectory_count = task
    if _WORKER_CONFIG is None:
        raise RuntimeError("CTMC81 worker is not initialized")
    spec = _WORKER_SPECS[point_index]
    dataset = str(spec["dataset"])
    generator = _rng(replica, point_index, trajectory_start)
    events = np.zeros(4, dtype=np.int64)
    maximum_drift = 0.0
    for offset in range(trajectory_count):
        outcome, success, drift = _simulate_at_config(
            spec, _WORKER_CONFIG, generator
        )
        if not success:
            raise ctmc.TrajectoryIntegrationFailure(
                {
                    "replica": replica,
                    "dataset": dataset,
                    "impact_index": spec["impact_index"],
                    "impact_parameter_au": spec["impact_au"],
                    "trajectory_index": trajectory_start + offset,
                    "failure_outcome": outcome,
                    "relative_total_energy_drift": drift,
                }
            )
        events[_event_index(dataset, outcome)] += 1
        maximum_drift = max(maximum_drift, float(drift))
    return (
        replica,
        point_index,
        trajectory_start,
        trajectory_count,
        events,
        maximum_drift,
    )


def _simulate_at_config(
    spec: dict[str, Any],
    config: ctmc.CTMCConfig,
    generator: np.random.Generator,
) -> tuple[str, bool, float]:
    dataset = str(spec["dataset"])
    if dataset == "loss":
        bound_to = "projectile"
        binding_eV = ctmc.PROJECTILE_OUTER_ORBITAL[3].binding_eV
    else:
        bound_to = "target"
        binding_eV = ctmc.WATER_ORBITALS[int(spec["orbital_index"])].binding_eV
    return ctmc.simulate_one_trajectory(
        energy_keV_u=100.0,
        charge_state=3,
        impact_parameter_au=float(spec["impact_au"]),
        bound_to=bound_to,
        binding_eV=binding_eV,
        start_separation_au=float(spec["start_separation_au"]),
        minimum_integration_time_au=float(spec["integration_time_au"]),
        config=config,
        rng=generator,
        fixed_time_endpoint=False,
        retain_ambiguous_endpoint=True,
        projectile_velocity_override_au=float(
            spec["projectile_velocity_au"]
        ),
        minimum_terminal_core_separation_au=float(
            spec["limit_distance_au"]
        ),
        garvey_role_term=bool(spec.get("garvey_role_term", False)),
    )


def _run_tolerance_pair(task: tuple[int, int]) -> dict[str, Any]:
    """Compare identical phases at the production and tenfold tighter tolerances."""
    point_index, trajectory_count = task
    if _WORKER_CONFIG is None:
        raise RuntimeError("CTMC81 worker is not initialized")
    spec = _WORKER_SPECS[point_index]
    tight = replace(
        _WORKER_CONFIG,
        rtol=0.1 * _WORKER_CONFIG.rtol,
        atol=0.1 * _WORKER_CONFIG.atol,
        retry_rtol=0.1 * _WORKER_CONFIG.retry_rtol,
        retry_atol=0.1 * _WORKER_CONFIG.retry_atol,
    )
    mismatches = 0
    failures = 0
    primary_maximum_drift = 0.0
    tight_maximum_drift = 0.0
    for trajectory_index in range(trajectory_count):
        primary_rng = _rng(10_000, point_index, trajectory_index)
        tight_rng = _rng(10_000, point_index, trajectory_index)
        primary = _simulate_at_config(spec, _WORKER_CONFIG, primary_rng)
        tighter = _simulate_at_config(spec, tight, tight_rng)
        failures += int(not primary[1]) + int(not tighter[1])
        if primary[1] and tighter[1]:
            mismatches += int(primary[0] != tighter[0])
            primary_maximum_drift = max(primary_maximum_drift, primary[2])
            tight_maximum_drift = max(tight_maximum_drift, tighter[2])
    return {
        "point_index": point_index,
        "dataset": spec["dataset"],
        "impact_index": spec["impact_index"],
        "impact_au": spec["impact_au"],
        "paired_trajectories": trajectory_count,
        "classification_mismatches": mismatches,
        "failed_integrations": failures,
        "primary_maximum_relative_energy_drift": primary_maximum_drift,
        "tight_maximum_relative_energy_drift": tight_maximum_drift,
    }


def _tolerance_audit(
    *,
    workers: int,
    trajectories_per_point: int,
    specs: tuple[dict[str, Any], ...],
    config: ctmc.CTMCConfig,
) -> dict[str, Any]:
    if trajectories_per_point < 1:
        return {"status": "not_evaluated", "reason": "disabled"}
    selected: list[int] = []
    for label in REFERENCE_LABELS:
        candidates = [
            int(spec["point_index"])
            for spec in specs
            if spec["dataset"] == label
        ]
        selected.extend(
            (candidates[0], candidates[len(candidates) // 2], candidates[-1])
        )
    with ProcessPoolExecutor(
        max_workers=min(workers, len(selected)),
        initializer=_init_worker,
        initargs=(config, specs),
    ) as executor:
        records = list(
            tqdm(
                executor.map(
                    _run_tolerance_pair,
                    ((point, trajectories_per_point) for point in selected),
                ),
                total=len(selected),
                desc="Tenfold-tighter audit",
                unit="point",
                dynamic_ncols=True,
            )
        )
    failures = sum(int(record["failed_integrations"]) for record in records)
    mismatches = sum(
        int(record["classification_mismatches"]) for record in records
    )
    return {
        "status": "pass" if failures == 0 and mismatches == 0 else "fail",
        "criterion": (
            "Identical microcanonical phases retain their event class at "
            "tenfold tighter DOP853 rtol/atol and pass the unchanged energy gate."
        ),
        "primary_rtol": config.rtol,
        "primary_atol": config.atol,
        "tight_rtol": 0.1 * config.rtol,
        "tight_atol": 0.1 * config.atol,
        "paired_trajectory_count": len(selected) * trajectories_per_point,
        "classification_mismatches": mismatches,
        "failed_integrations": failures,
        "records": records,
    }


def _checkpoint(
    path: Path,
    signature_value: str,
    events: np.ndarray,
    completed: np.ndarray,
    maximum_drift: np.ndarray,
) -> None:
    _atomic_npz(
        path,
        signature=np.asarray(signature_value),
        events=events,
        completed=completed,
        maximum_relative_energy_drift=maximum_drift,
    )


def _load_checkpoint(
    path: Path,
    signature_value: str,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        return (
            np.zeros(shape + (4,), dtype=np.int64),
            np.zeros(shape, dtype=np.int64),
            np.zeros(shape, dtype=float),
        )
    with np.load(path, allow_pickle=False) as archive:
        found = str(np.asarray(archive["signature"]).item())
        if found != signature_value:
            raise RuntimeError(
                "Existing CTMC81 checkpoint has a different configuration; "
                "use a new output directory"
            )
        events = np.asarray(archive["events"], dtype=np.int64)
        completed = np.asarray(archive["completed"], dtype=np.int64)
        maximum_drift = np.asarray(
            archive["maximum_relative_energy_drift"], dtype=float
        )
    if events.shape != shape + (4,) or completed.shape != shape:
        raise RuntimeError("Malformed CTMC81 checkpoint array shape")
    return events, completed, maximum_drift


def _request_stop(signum: int, _: object) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    tqdm.write(f"Received signal {signum}; checkpointing bounded chunks")


def _point_order(specs: tuple[dict[str, Any], ...]) -> list[int]:
    grouped = {
        label: [int(spec["point_index"]) for spec in specs if spec["dataset"] == label]
        for label in REFERENCE_LABELS
    }
    order: list[int] = []
    for row in range(max(map(len, grouped.values()))):
        for label in REFERENCE_LABELS:
            if row < len(grouped[label]):
                order.append(grouped[label][row])
    return order


def _run_reproduction(
    *,
    workers: int,
    replicas: int,
    specs: tuple[dict[str, Any], ...],
    config: ctmc.CTMCConfig,
    signature_value: str,
    checkpoint_path: Path,
    checkpoint_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (replicas, len(specs))
    events, completed, maximum_drift = _load_checkpoint(
        checkpoint_path, signature_value, shape
    )
    for point_index, spec in enumerate(specs):
        expected = int(spec["trajectory_count"])
        if np.any(completed[:, point_index] > expected):
            raise RuntimeError("Checkpoint exceeds the disclosed trajectory count")
        if np.any(np.sum(events[:, point_index], axis=1) != completed[:, point_index]):
            raise RuntimeError("Checkpoint event counts do not equal completed work")

    remaining = sum(
        int(specs[point_index]["trajectory_count"] - completed[replica, point_index])
        for replica in range(replicas)
        for point_index in range(len(specs))
    )
    if remaining == 0:
        return events, completed, maximum_drift

    chain_order = [
        (replica, point_index)
        for point_index in _point_order(specs)
        for replica in range(replicas)
        if completed[replica, point_index] < specs[point_index]["trajectory_count"]
    ]
    chain_cursor = 0
    active: dict[Any, tuple[int, int, int, int]] = {}
    last_checkpoint = time.monotonic()
    chunks_since_checkpoint = 0

    def submit_next(
        executor: ProcessPoolExecutor, replica: int, point_index: int
    ) -> None:
        start = int(completed[replica, point_index])
        total = int(specs[point_index]["trajectory_count"])
        if start >= total:
            return
        count = min(config.trajectory_chunk_size, total - start)
        task = (replica, point_index, start, count)
        active[executor.submit(_run_chunk, task)] = task

    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(config, specs),
    )
    try:
        while chain_cursor < len(chain_order) and len(active) < workers:
            submit_next(executor, *chain_order[chain_cursor])
            chain_cursor += 1
        with tqdm(
            total=remaining,
            initial=0,
            desc="CTMC81 exact replicas",
            unit="traj",
            dynamic_ncols=True,
        ) as progress:
            while active:
                done, _ = wait(active, timeout=5.0, return_when=FIRST_COMPLETED)
                for future in done:
                    task = active.pop(future)
                    (
                        replica,
                        point_index,
                        start,
                        count,
                        chunk_events,
                        drift,
                    ) = future.result()
                    if start != int(completed[replica, point_index]):
                        raise RuntimeError("Non-contiguous CTMC81 chunk completion")
                    events[replica, point_index] += chunk_events
                    completed[replica, point_index] += count
                    maximum_drift[replica, point_index] = max(
                        maximum_drift[replica, point_index], drift
                    )
                    chunks_since_checkpoint += 1
                    progress.update(count)
                    if (
                        not _STOP_REQUESTED
                        and completed[replica, point_index]
                        < specs[point_index]["trajectory_count"]
                    ):
                        submit_next(executor, replica, point_index)
                    elif not _STOP_REQUESTED and chain_cursor < len(chain_order):
                        submit_next(executor, *chain_order[chain_cursor])
                        chain_cursor += 1
                now = time.monotonic()
                if (
                    chunks_since_checkpoint >= 100
                    or now - last_checkpoint >= checkpoint_seconds
                ):
                    _checkpoint(
                        checkpoint_path,
                        signature_value,
                        events,
                        completed,
                        maximum_drift,
                    )
                    last_checkpoint = now
                    chunks_since_checkpoint = 0
                if _STOP_REQUESTED:
                    break
    except BaseException:
        _checkpoint(
            checkpoint_path,
            signature_value,
            events,
            completed,
            maximum_drift,
        )
        ctmc.terminate_process_pool(executor)
        raise
    finally:
        if _STOP_REQUESTED:
            _checkpoint(
                checkpoint_path,
                signature_value,
                events,
                completed,
                maximum_drift,
            )
            ctmc.terminate_process_pool(executor)
        else:
            executor.shutdown(wait=True)

    if _STOP_REQUESTED:
        raise RuntimeError("CTMC81 reproduction stopped after a safe checkpoint")
    _checkpoint(
        checkpoint_path,
        signature_value,
        events,
        completed,
        maximum_drift,
    )
    return events, completed, maximum_drift


def _comparison_rows(
    reference: dict[str, Any],
    specs: tuple[dict[str, Any], ...],
    events: np.ndarray,
    completed: np.ndarray,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    comparison_count = len(specs) * 4
    simultaneous_z = float(norm.ppf(1.0 - 0.05 / (2.0 * comparison_count)))
    for spec in specs:
        point = int(spec["point_index"])
        dataset = str(spec["dataset"])
        dataset_reference = reference["datasets"][dataset]
        reference_probability = np.asarray(
            dataset_reference["probabilities"][int(spec["impact_index"])],
            dtype=float,
        )
        reference_n = int(spec["trajectory_count"])
        current_counts = np.sum(events[:, point], axis=0)
        current_n = int(np.sum(completed[:, point]))
        current_probability = current_counts / current_n
        for column in range(4):
            p_ref = float(reference_probability[column])
            p_cur = float(current_probability[column])
            variance = (
                p_ref * (1.0 - p_ref) / reference_n
                + p_cur * (1.0 - p_cur) / current_n
                + (0.5e-4) ** 2 / 3.0
            )
            difference = p_cur - p_ref
            z_score = difference / math.sqrt(variance) if variance > 0.0 else 0.0
            rows.append(
                {
                    "dataset": dataset,
                    "impact_index": int(spec["impact_index"]),
                    "impact_au": float(spec["impact_au"]),
                    "event": str(dataset_reference["column_names"][column]),
                    "reference_probability": p_ref,
                    "reference_trajectories": reference_n,
                    "calculated_probability": p_cur,
                    "calculated_trajectories": current_n,
                    "difference": difference,
                    "combined_standard_error": math.sqrt(variance),
                    "z_score": z_score,
                    "within_95pct_simultaneous_sampling_band": bool(
                        abs(z_score) <= simultaneous_z
                    ),
                }
            )
    return rows, simultaneous_z


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = tuple(rows[0])
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(",".join(columns) + "\n")
        for row in rows:
            values = []
            for column in columns:
                value = row[column]
                if isinstance(value, float):
                    values.append(f"{value:.17g}")
                elif isinstance(value, bool):
                    values.append("1" if value else "0")
                else:
                    values.append(str(value))
            stream.write(",".join(values) + "\n")
    os.replace(temporary, path)


def _derived_cross_sections(
    reference: dict[str, Any],
    specs: tuple[dict[str, Any], ...],
    events: np.ndarray,
    completed: np.ndarray,
    grid_points: int,
) -> dict[str, dict[str, float]]:
    impact = np.linspace(0.0, 10.0, grid_points)
    reference_pi = np.zeros((grid_points, 5), dtype=float)
    reference_pc = np.zeros_like(reference_pi)
    calculated_pi = np.zeros_like(reference_pi)
    calculated_pc = np.zeros_like(reference_pi)
    calculated_by_label: dict[str, np.ndarray] = {}
    for label in REFERENCE_LABELS:
        points = [
            int(spec["point_index"])
            for spec in specs
            if spec["dataset"] == label
        ]
        calculated_by_label[label] = np.sum(events[:, points], axis=0) / np.sum(
            completed[:, points], axis=0
        )[:, None]
    for orbital in range(5):
        label = f"L{orbital + 1}"
        dataset = reference["datasets"][label]
        source_impact = np.asarray(dataset["impact_au"], dtype=float)
        source_reference = np.asarray(dataset["probabilities"], dtype=float)
        source_calculated = calculated_by_label[label]
        reference_pi[:, orbital] = np.interp(
            impact, source_impact, source_reference[:, 0], right=0.0
        )
        reference_pc[:, orbital] = np.interp(
            impact, source_impact, source_reference[:, 1], right=0.0
        )
        calculated_pi[:, orbital] = np.interp(
            impact, source_impact, source_calculated[:, 0], right=0.0
        )
        calculated_pc[:, orbital] = np.interp(
            impact, source_impact, source_calculated[:, 1], right=0.0
        )
    loss = reference["datasets"]["loss"]
    loss_impact = np.asarray(loss["impact_au"], dtype=float)
    reference_pl = np.interp(
        impact,
        loss_impact,
        np.asarray(loss["probabilities"], dtype=float)[:, 0],
        right=0.0,
    )
    calculated_pl = np.interp(
        impact,
        loss_impact,
        calculated_by_label["loss"][:, 0],
        right=0.0,
    )
    result: dict[str, dict[str, float]] = {}
    for name, pi, pc, pl in (
        ("reference", reference_pi, reference_pc, reference_pl),
        ("calculated", calculated_pi, calculated_pc, calculated_pl),
    ):
        probabilities = ctmc.many_electron_probabilities(pi, pc, pl, 3)
        result[name] = {
            channel: ctmc.integrate_impact_parameter(impact, probability)
            for channel, probability in probabilities.items()
            if channel in {"SC", "TI", "SL", "LI", "SI", "DI"}
        }
    return result


def _cross_section_summary(
    reference: dict[str, Any],
    specs: tuple[dict[str, Any], ...],
    events: np.ndarray,
    completed: np.ndarray,
) -> dict[str, Any]:
    primary = _derived_cross_sections(
        reference, specs, events, completed, grid_points=20_001
    )
    control = _derived_cross_sections(
        reference, specs, events, completed, grid_points=40_001
    )
    rows: list[dict[str, Any]] = []
    maximum_quadrature_change = 0.0
    for channel in ("SC", "TI", "SL", "LI", "SI", "DI"):
        ref = primary["reference"][channel]
        calc = primary["calculated"][channel]
        for source in ("reference", "calculated"):
            value = primary[source][channel]
            tighter = control[source][channel]
            maximum_quadrature_change = max(
                maximum_quadrature_change,
                abs(tighter - value) / max(abs(tighter), np.finfo(float).tiny),
            )
        rows.append(
            {
                "channel": channel,
                "reference_cm2_per_h2o": ref,
                "calculated_cm2_per_h2o": calc,
                "relative_difference": (calc - ref) / ref,
            }
        )
    return {
        "rows": rows,
        "interpolation": "piecewise linear on each disclosed impact grid",
        "integration_grid_points": 20_001,
        "control_grid_points": 40_001,
        "maximum_relative_quadrature_change": maximum_quadrature_change,
        "quadrature_0p5pct_gate_passed": bool(
            maximum_quadrature_change <= 5.0e-3
        ),
    }


def _plot_comparison(
    path: Path,
    reference: dict[str, Any],
    specs: tuple[dict[str, Any], ...],
    events: np.ndarray,
    completed: np.ndarray,
) -> None:
    styles = rcparams_with_fontsize(RC_BASE_STANDARD, PAPER_FONTSIZE)
    with plt.rc_context(styles):
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(AASTEX_FULL_WIDTH_IN, 4.4),
            sharex=False,
            sharey=True,
        )
        colors = plt.get_cmap("plasma")(np.linspace(0.12, 0.88, 4))
        panels = zip(axes.flat, REFERENCE_LABELS, strict=True)
        for panel, (axis, label) in enumerate(panels):
            dataset = reference["datasets"][label]
            impact = np.asarray(dataset["impact_au"], dtype=float)
            reference_values = np.asarray(dataset["probabilities"], dtype=float)
            points = [
                int(spec["point_index"])
                for spec in specs
                if spec["dataset"] == label
            ]
            calculated = np.sum(events[:, points], axis=0) / np.sum(
                completed[:, points], axis=0
            )[:, None]
            for column, color in enumerate(colors):
                name = str(dataset["column_names"][column])
                axis.plot(
                    impact,
                    reference_values[:, column],
                    color=color,
                    lw=1.1,
                    label=name,
                )
                axis.plot(
                    impact,
                    calculated[:, column],
                    linestyle="none",
                    marker="o",
                    markersize=1.5,
                    markerfacecolor="none",
                    markeredgewidth=0.45,
                    color=color,
                )
            axis.text(
                0.0,
                1.04,
                f"({chr(97 + panel)}) {label}",
                transform=axis.transAxes,
            )
            axis.set_xlabel(r"Impact parameter ($b$; a.u.)")
            if panel % 3 == 0:
                axis.set_ylabel("Probability")
            axis.set_ylim(-0.03, 1.03)
            axis.tick_params(top=False, right=False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
        fig.subplots_adjust(
            left=0.09,
            right=0.985,
            bottom=0.17,
            top=0.94,
            wspace=0.27,
            hspace=0.40,
        )
        fig.savefig(path, dpi=300)
        plt.close(fig)


def _replica_consistency(
    events: np.ndarray,
    completed: np.ndarray,
) -> dict[str, Any]:
    if events.shape[0] < 2:
        return {"status": "not_evaluated", "reason": "one replica"}
    probabilities = events / completed[:, :, None]
    z_values: list[float] = []
    for left in range(events.shape[0] - 1):
        for right in range(left + 1, events.shape[0]):
            p_left = probabilities[left]
            p_right = probabilities[right]
            variance = (
                p_left * (1.0 - p_left) / completed[left, :, None]
                + p_right * (1.0 - p_right) / completed[right, :, None]
            )
            valid = variance > 0.0
            z_values.extend(
                np.abs((p_left - p_right)[valid] / np.sqrt(variance[valid])).tolist()
            )
    count = max(len(z_values), 1)
    threshold = float(norm.ppf(1.0 - 0.05 / (2.0 * count)))
    maximum = max(z_values, default=0.0)
    return {
        "status": "pass" if maximum <= threshold else "fail",
        "maximum_absolute_z_score": maximum,
        "simultaneous_95pct_z_threshold": threshold,
        "comparison_count": len(z_values),
    }


def _write_results(
    *,
    output_dir: Path,
    reference: dict[str, Any],
    specs: tuple[dict[str, Any], ...],
    config: ctmc.CTMCConfig,
    signature_value: str,
    events: np.ndarray,
    completed: np.ndarray,
    maximum_drift: np.ndarray,
    tolerance_audit: dict[str, Any],
) -> None:
    rows, threshold = _comparison_rows(reference, specs, events, completed)
    csv_path = output_dir / "ctmc81_pointwise_comparison.csv"
    _write_csv(csv_path, rows)
    plot_path = output_dir / "ctmc81_c3_100keV_u_reproduction.png"
    _plot_comparison(plot_path, reference, specs, events, completed)
    absolute_z = np.asarray([abs(float(row["z_score"])) for row in rows])
    maximum_drift_value = float(np.max(maximum_drift))
    replica_check = _replica_consistency(events, completed)
    cross_sections = _cross_section_summary(
        reference, specs, events, completed
    )
    cross_section_csv = output_dir / "ctmc81_derived_cross_sections.csv"
    _write_csv(cross_section_csv, cross_sections["rows"])
    parameter_match = {
        label: {
            **reference["datasets"][label]["metadata"],
            "calculated_replica_count": int(events.shape[0]),
            "calculated_trajectories_per_point_per_replica": int(
                reference["datasets"][label]["metadata"]["trajectory_count"]
            ),
            "endpoint": "t >= tEND, separation >= disclosed limit, receding",
        }
        for label in REFERENCE_LABELS
    }
    config_payload = asdict(config)
    config_payload["max_step_au"] = None
    summary = {
        "status": (
            "pass"
            if np.all(absolute_z <= threshold)
            and maximum_drift_value <= config.maximum_relative_energy_drift
            and tolerance_audit.get("status") == "pass"
            and replica_check.get("status") in {"pass", "not_evaluated"}
            and cross_sections["quadrature_0p5pct_gate_passed"]
            else "fail"
        ),
        "interpretation": (
            "Pass/fail tests independent Monte Carlo consistency with the "
            "supplied primitive CTMC81 probabilities; it is not a claim of "
            "physical-model accuracy."
        ),
        "paper": {"doi": PAPER_DOI},
        "reference": {
            "provenance": "user supplied directly as Liamsuwan CTMC81 output",
            "archive": reference["path"],
            "sha256": reference["sha256"],
            "expected_sha256": FORMAL_REFERENCE_SHA256,
        },
        "configuration_signature": signature_value,
        "initial_ensemble": {
            "mode": config.initial_ensemble,
            "role": "historical_paper_reproduction",
            "momentum_direction": (
                "uniform in tangent plane (Olson--Salop equation 7)"
            ),
            "random_draws_per_trajectory": (
                ctmc.random_draws_per_trajectory(config.initial_ensemble)
            ),
        },
        "projectile_loss_initialization": (
            ctmc.projectile_loss_initialization_metadata(
                config.projectile_loss_initialization
            )
        ),
        "parameter_match": parameter_match,
        "numerical_controls": {
            "ctmc81_recorded_TOL": 1.0e-6,
            "note": (
                "CTMC81 TOL and DOP853 rtol/atol are implementation-specific "
                "and are not asserted to be mathematically identical."
            ),
            "calculated": config_payload,
            "maximum_observed_relative_total_energy_drift": maximum_drift_value,
            "energy_drift_gate_passed": bool(
                maximum_drift_value <= config.maximum_relative_energy_drift
            ),
            "tenfold_tighter_paired_audit": tolerance_audit,
        },
        "sampling_comparison": {
            "point_event_comparison_count": len(rows),
            "simultaneous_95pct_z_threshold": threshold,
            "maximum_absolute_z_score": float(np.max(absolute_z)),
            "rms_z_score": float(np.sqrt(np.mean(absolute_z**2))),
            "failed_simultaneous_points": int(np.count_nonzero(absolute_z > threshold)),
            "replica_consistency": replica_check,
        },
        "derived_cross_sections": cross_sections,
        "outputs": {
            "pointwise_csv": str(csv_path),
            "cross_section_csv": str(cross_section_csv),
            "plot": str(plot_path),
        },
    }
    _atomic_json(output_dir / "ctmc81_reproduction_summary.json", summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-archive", type=Path, default=FORMAL_REFERENCE_ARCHIVE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--trajectory-chunk-size", type=int, default=100)
    parser.add_argument("--checkpoint-seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--rtol", type=float, default=1.0e-11)
    parser.add_argument("--atol", type=float, default=1.0e-13)
    parser.add_argument("--retry-rtol", type=float, default=1.0e-12)
    parser.add_argument("--retry-atol", type=float, default=1.0e-14)
    parser.add_argument("--maximum-relative-energy-drift", type=float, default=1.0e-4)
    parser.add_argument("--maximum-integration-steps", type=int, default=100_000_000)
    parser.add_argument("--tolerance-audit-trajectories", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers < 1 or args.replicas < 1 or args.trajectory_chunk_size < 1:
        raise ValueError("workers, replicas and trajectory chunk size must be positive")
    reference = _read_formal_reference_archive(args.reference_archive)
    if reference["sha256"] != FORMAL_REFERENCE_SHA256:
        raise RuntimeError(
            "The CTMC81 archive hash does not match the reviewed dataset"
        )
    specs = _build_specs(reference)
    config = _configuration(
        trajectory_chunk_size=args.trajectory_chunk_size,
        seed=args.seed,
        rtol=args.rtol,
        atol=args.atol,
        retry_rtol=args.retry_rtol,
        retry_atol=args.retry_atol,
        maximum_relative_energy_drift=args.maximum_relative_energy_drift,
        maximum_integration_steps=args.maximum_integration_steps,
    )
    # Populate immutable CDF arrays before fork so all workers share them
    # copy-on-write instead of rebuilding them independently.
    ctmc.select_projectile("carbon")
    ctmc.warm_up_initial_state_sampling(
        np.asarray([3], dtype=int), config.radial_grid_points
    )
    signature_value = _signature(reference, specs, config, args.replicas)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for signal_name in ("SIGTERM", "SIGUSR1"):
        value = getattr(signal, signal_name, None)
        if value is not None:
            signal.signal(value, _request_stop)
    events, completed, maximum_drift = _run_reproduction(
        workers=args.workers,
        replicas=args.replicas,
        specs=specs,
        config=config,
        signature_value=signature_value,
        checkpoint_path=output_dir / "ctmc81_reproduction_checkpoint.npz",
        checkpoint_seconds=args.checkpoint_seconds,
    )
    expected = np.asarray([spec["trajectory_count"] for spec in specs])
    if not np.all(completed == expected[None, :]):
        raise RuntimeError("CTMC81 reproduction is incomplete")
    tolerance_audit = _tolerance_audit(
        workers=args.workers,
        trajectories_per_point=args.tolerance_audit_trajectories,
        specs=specs,
        config=config,
    )
    _write_results(
        output_dir=output_dir,
        reference=reference,
        specs=specs,
        config=config,
        signature_value=signature_value,
        events=events,
        completed=completed,
        maximum_drift=maximum_drift,
        tolerance_audit=tolerance_audit,
    )
    print(f"Completed {int(np.sum(completed)):,} trajectories")
    print(output_dir / "ctmc81_reproduction_summary.json")


if __name__ == "__main__":
    main()
