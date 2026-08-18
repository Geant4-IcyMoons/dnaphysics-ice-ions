#!/usr/bin/env python3
"""Safely rebalance and resubmit the tail of a PBS CTMC base-grid run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Sequence

import numpy as np
from tqdm import tqdm

from constants import CTMC_INTEGRATOR_POLICY_VERSION


ACTIVE_WRITER_STATES = frozenset({"R", "E"})
WAITING_WRITER_STATES = frozenset({"Q", "H", "W", "S"})
UNFINISHED_STATES = frozenset({"Q", "H", "W", "S", "R", "E", "B"})


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_qstat_states(output: str) -> list[str]:
    """Return scalar/array-element states, excluding array parent rows."""
    states: list[str] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 6 or re.search(r"\[\]", fields[0]):
            continue
        if not re.match(r"^\d+(?:\[\d+\])?\.", fields[0]):
            continue
        state = fields[4]
        if len(state) == 1:
            states.append(state)
    return states


def qstat_states(job_id: str) -> list[str]:
    completed = subprocess.run(
        ["qstat", "-t", job_id],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []
    return parse_qstat_states(completed.stdout)


def pbs_parent_job_id(job_id: str) -> str:
    """Return the scalar PBS parent identifier for a job or array."""
    return re.sub(r"\[\]$", "", job_id)


def failure_shard(path: Path, generation: str | None) -> int | None:
    """Return a current-generation shard index encoded by a diagnostic."""
    match = re.search(
        r"_failure_shard_(\d{4})(?:_rebalance_([^/]+))?\.json$",
        path.name,
    )
    if match is None:
        return None
    recorded_generation = match.group(2)
    if recorded_generation != generation:
        return None
    return int(match.group(1))


def failure_diagnostics(
    output_dir: Path,
    atom: str,
    generation: str | None,
) -> list[tuple[int, Path, dict[str, object]]]:
    records: list[tuple[int, Path, dict[str, object]]] = []
    for path in sorted(output_dir.glob(
        f"{atom}_charge_exchange_ctmc_failure_shard_*.json"
    )):
        shard = failure_shard(path, generation)
        if shard is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append((shard, path, payload))
    return records


def validate_failure_checkpoint(payload: dict[str, object]) -> str | None:
    """Return a quarantine reason, or None for a resumable diagnostic."""
    checkpoint_value = payload.get("checkpoint")
    signature_value = payload.get("configuration_signature")
    if not isinstance(checkpoint_value, str) or not isinstance(
        signature_value, str
    ):
        return "Failure diagnostic lacks a checkpoint or signature."
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_file():
        return f"Failure checkpoint is missing: {checkpoint}"
    try:
        with np.load(checkpoint, allow_pickle=False) as data:
            signature = str(np.asarray(data["signature"]).item())
    except (OSError, KeyError, ValueError) as error:
        return f"Failure checkpoint is unreadable: {error}"
    if signature != signature_value:
        return "Failure diagnostic and checkpoint signatures differ."
    return None


def recovery_action(
    failure_policy_version: int,
    attempts: int,
    maximum_attempts: int,
) -> str:
    """Classify a deterministic failure without weakening any gate."""
    if failure_policy_version >= CTMC_INTEGRATOR_POLICY_VERSION:
        return "quarantine_current_policy"
    if attempts >= maximum_attempts:
        return "quarantine_attempts"
    return "resubmit"


def should_rebalance(
    *,
    fraction: float,
    trigger_fraction: float,
    active: int,
    waiting: int,
    active_threshold: int,
) -> bool:
    """Return whether a generation is an executable, depleted tail."""
    return (
        fraction >= trigger_fraction
        and active <= active_threshold
        and waiting == 0
    )


def submit_recovery(
    *,
    pbs_script: Path,
    shard_index: int,
    shard_count: int,
    ownership_manifest: str | None,
    queue: str,
    memory: str,
    job_name: str,
) -> str:
    variables = [
        f"SHARD_COUNT={shard_count}",
        f"SHARD_OFFSET={shard_index}",
    ]
    if ownership_manifest:
        variables.append(f"OWNERSHIP_MANIFEST={ownership_manifest}")
    completed = subprocess.run(
        [
            "qsub", "-V", "-N", f"{job_name}_fix{shard_index}",
            "-q", queue,
            "-l", f"select=1:ncpus=64:mem={memory}",
            "-v", ",".join(variables), str(pbs_script),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().split(".", 1)[0].replace("[]", "")


def checkpoint_paths(
    output_dir: Path,
    atom: str,
    shard_count: int,
    generation: str | None,
) -> list[Path]:
    suffix = "" if generation is None else f".rebalance-{generation}"
    return [
        output_dir
        / (
            f"{atom}_charge_exchange_ctmc_checkpoint{suffix}."
            f"shard-{index:05d}-of-{shard_count:05d}.npz"
        )
        for index in range(shard_count)
    ]


def checkpoint_progress(paths: Sequence[Path]) -> tuple[int, int, int]:
    """Return committed trajectories, finished points, and all owned points."""
    completed = finished_points = owned_points = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            completed += int(np.sum(data["channel_completed"], dtype=np.int64))
            finished_points += int(np.count_nonzero(data["done"]))
            # Checkpoints contain zeros outside their immutable ownership.
            owned_points += int(np.count_nonzero(
                np.any(data["channel_completed"] != 0, axis=-1)
                | data["done"]
            ))
    return completed, finished_points, owned_points


def manifest_generation(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        return str(data["generation"].item())


def stable_stats(paths: Sequence[Path]) -> tuple[tuple[int, int], ...]:
    return tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in paths)


def wait_until_frozen(paths: Sequence[Path], seconds: float) -> None:
    previous = stable_stats(paths)
    deadline = time.monotonic() + seconds
    while True:
        time.sleep(min(30.0, max(1.0, seconds / 4.0)))
        current = stable_stats(paths)
        if current != previous:
            previous = current
            deadline = time.monotonic() + seconds
        if time.monotonic() >= deadline:
            return


def submit_arrays(
    *,
    pbs_script: Path,
    shard_count: int,
    manifest: Path,
    array_max: int,
    queues: Sequence[str],
    memory: str,
    job_name: str,
) -> list[str]:
    job_ids: list[str] = []
    for batch, offset in enumerate(range(0, shard_count, array_max)):
        batch_size = min(array_max, shard_count - offset)
        queue = queues[batch % len(queues)]
        variables = (
            f"SHARD_COUNT={shard_count},SHARD_OFFSET={offset},"
            f"OWNERSHIP_MANIFEST={manifest}"
        )
        completed = subprocess.run(
            [
                "qsub", "-V", "-N", f"{job_name}_r{batch}", "-q", queue,
                "-J", f"0-{batch_size - 1}",
                "-l", f"select=1:ncpus=64:mem={memory}",
                "-v", variables, str(pbs_script),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        job_ids.append(completed.stdout.strip().split(".", 1)[0].replace("[]", ""))
    return job_ids


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--atom", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--pbs-script", type=Path, required=True)
    result.add_argument("--source-shards", type=int, required=True)
    result.add_argument("--source-job-id", action="append", required=True)
    result.add_argument("--target-shards", type=int, required=True)
    result.add_argument("--array-max", type=int, default=42)
    result.add_argument("--queue", action="append", default=[])
    result.add_argument("--memory", default="8gb")
    result.add_argument("--trigger-fraction", type=float, default=0.95)
    result.add_argument("--active-fraction", type=float, default=0.25)
    result.add_argument("--poll-seconds", type=float, default=300.0)
    result.add_argument("--freeze-seconds", type=float, default=180.0)
    result.add_argument("--max-recovery-attempts", type=int, default=3)
    result.add_argument("--state-file", type=Path, required=True)
    result.add_argument("--job-name", default="ctmc_tail")
    result.add_argument(
        "generator_command", nargs=argparse.REMAINDER,
        help="Exact generator command after --, without shard/ownership options",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    command = list(args.generator_command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("An exact generator command is required after --")
    if not 0.0 < args.trigger_fraction < 1.0:
        raise SystemExit("--trigger-fraction must lie between zero and one")
    if not 0.0 < args.active_fraction <= 1.0:
        raise SystemExit("--active-fraction must lie in (0, 1]")
    if args.max_recovery_attempts < 1:
        raise SystemExit("--max-recovery-attempts must be positive")
    queues = args.queue or ["idle"]
    state: dict[str, object] = {
        "format_version": 2,
        "source_shards": args.source_shards,
        "source_job_ids": args.source_job_id,
        "source_manifest": None,
        "generation": None,
        "rebalances": 0,
        "recoveries": {},
        "quarantined_failures": {},
    }
    if args.state_file.is_file():
        state = json.loads(args.state_file.read_text())
    state["format_version"] = 2
    state.setdefault("recoveries", {})
    state.setdefault("quarantined_failures", {})
    atomic_json(args.state_file, state)

    progress = tqdm(total=1.0, unit="fraction", desc=f"{args.atom} CTMC")
    while True:
        shard_count = int(state["source_shards"])
        generation = state.get("generation")
        generation_text = None if generation is None else str(generation)
        recoveries = dict(state.get("recoveries", {}))
        quarantined = dict(state.get("quarantined_failures", {}))
        for shard, diagnostic_path, payload in failure_diagnostics(
            args.output_dir, args.atom, generation_text
        ):
            key = f"{generation_text or 'base'}:{shard}"
            policy_version = int(payload.get("integrator_policy_version", 0))
            previous = dict(recoveries.get(key, {}))
            recovery_job = str(previous.get("job_id", ""))
            recovery_states = qstat_states(recovery_job) if recovery_job else []
            if any(code in UNFINISHED_STATES for code in recovery_states):
                continue
            invalid_reason = validate_failure_checkpoint(payload)
            if invalid_reason is not None:
                quarantined[key] = {
                    "diagnostic": str(diagnostic_path),
                    "integrator_policy_version": policy_version,
                    "reason": invalid_reason,
                }
                continue
            attempts = int(previous.get("attempts", 0))
            action = recovery_action(
                policy_version, attempts, args.max_recovery_attempts
            )
            if action != "resubmit":
                quarantined[key] = {
                    "diagnostic": str(diagnostic_path),
                    "integrator_policy_version": policy_version,
                    "reason": (
                        "Exact trajectory exhausted the current validated "
                        "numerical fallback ladder; automatic repetition is "
                        "disabled."
                        if action == "quarantine_current_policy"
                        else "Bounded automatic recovery attempts exhausted."
                    ),
                }
                continue
            recovery_job = submit_recovery(
                pbs_script=args.pbs_script.resolve(),
                shard_index=shard,
                shard_count=shard_count,
                ownership_manifest=(
                    None
                    if state.get("source_manifest") is None
                    else str(state["source_manifest"])
                ),
                queue=queues[attempts % len(queues)],
                memory=args.memory,
                job_name=args.job_name,
            )
            recoveries[key] = {
                "attempts": attempts + 1,
                "job_id": recovery_job,
                "submitted_policy_version": CTMC_INTEGRATOR_POLICY_VERSION,
                "diagnostic": str(diagnostic_path),
            }
            source_jobs = [str(job) for job in state["source_job_ids"]]
            if recovery_job not in source_jobs:
                source_jobs.append(recovery_job)
                state["source_job_ids"] = source_jobs
            tqdm.write(
                f"Submitted policy-v{CTMC_INTEGRATOR_POLICY_VERSION} recovery "
                f"for shard {shard}: {recovery_job}"
            )
        state["recoveries"] = recoveries
        state["quarantined_failures"] = quarantined
        atomic_json(args.state_file, state)
        paths = checkpoint_paths(
            args.output_dir, args.atom, shard_count,
            generation_text,
        )
        if not all(path.is_file() for path in paths):
            missing = sum(not path.is_file() for path in paths)
            tqdm.write(f"Waiting for {missing}/{len(paths)} source checkpoints")
            time.sleep(args.poll_seconds)
            continue

        committed, finished_points, observed_points = checkpoint_progress(paths)
        # Every owned point eventually contains at least one channel. Until then,
        # use the manifest's exact initial committed count only for reporting.
        with np.load(paths[0], allow_pickle=False) as first:
            trajectories = _option_int(command, "--trajectories")
            exact_total = _total_trajectories(
                first["energies_keV_u"], first["charges"], first["impact_au"],
                trajectories,
                _option_values(command, "--target-bmax-au"),
                _option_values(command, "--loss-bmax-au"),
            )
        fraction = min(1.0, committed / exact_total)
        progress.n = fraction
        progress.set_postfix(
            active=sum(
                state_code in ACTIVE_WRITER_STATES
                for job in state["source_job_ids"]
                for state_code in qstat_states(str(job))
            ),
            points=finished_points,
            refresh=True,
        )
        if fraction >= 1.0:
            tqdm.write("Base-grid generation is complete; no redistribution needed")
            progress.close()
            return 0

        job_states = [
            state_code
            for job in state["source_job_ids"]
            for state_code in qstat_states(str(job))
        ]
        active = sum(code in ACTIVE_WRITER_STATES for code in job_states)
        waiting = sum(code in WAITING_WRITER_STATES for code in job_states)
        unfinished = sum(code in UNFINISHED_STATES for code in job_states)
        threshold = max(1, int(args.active_fraction * shard_count))
        if not should_rebalance(
            fraction=fraction,
            trigger_fraction=args.trigger_fraction,
            active=active,
            waiting=waiting,
            active_threshold=threshold,
        ):
            if unfinished == 0:
                raise RuntimeError(
                    "All PBS jobs ended before completion; inspect failures before restart"
                )
            time.sleep(args.poll_seconds)
            continue

        tqdm.write(
            f"Tail trigger: {fraction:.3%} committed with {active} active writers; "
            "freezing the current generation"
        )
        while True:
            for job in state["source_job_ids"]:
                subprocess.run(
                    ["qdel", pbs_parent_job_id(str(job))],
                    check=False,
                )
            remaining_states = [
                code
                for job in state["source_job_ids"]
                for code in qstat_states(str(job))
            ]
            if not any(code in UNFINISHED_STATES for code in remaining_states):
                break
            # PBS may temporarily reject deletion of a running array element
            # while its job record is locked. Retry the parent cancellation so
            # neither a locked writer nor a queued sibling can survive the
            # checkpoint-freeze boundary.
            time.sleep(15.0)
        wait_until_frozen(paths, args.freeze_seconds)

        before = set(args.output_dir.glob(
            f"{args.atom}_charge_exchange_ctmc_rebalance-*.npz"
        ))
        prepare = command + [
            "--prepare-rebalance-from-shards", str(shard_count),
            "--shard-count", str(args.target_shards),
        ]
        if state.get("source_manifest"):
            prepare += ["--source-ownership-manifest", str(state["source_manifest"])]
        subprocess.run(prepare, check=True)
        manifests = set(args.output_dir.glob(
            f"{args.atom}_charge_exchange_ctmc_rebalance-*.npz"
        ))
        created = sorted(manifests - before, key=lambda path: path.stat().st_mtime_ns)
        if not created:
            # A deterministic re-run may reproduce an already published manifest.
            created = sorted(manifests, key=lambda path: path.stat().st_mtime_ns)
        if not created:
            raise RuntimeError("Rebalance completed without an ownership manifest")
        manifest = created[-1].resolve()
        new_jobs = submit_arrays(
            pbs_script=args.pbs_script.resolve(),
            shard_count=args.target_shards,
            manifest=manifest,
            array_max=args.array_max,
            queues=queues,
            memory=args.memory,
            job_name=args.job_name,
        )
        state.update(
            source_shards=args.target_shards,
            source_job_ids=new_jobs,
            source_manifest=str(manifest),
            generation=manifest_generation(manifest),
            rebalances=int(state["rebalances"]) + 1,
        )
        atomic_json(args.state_file, state)
        tqdm.write(f"Submitted redistributed generation: {', '.join(new_jobs)}")


def _option_values(command: Sequence[str], option: str) -> list[float]:
    try:
        raw = command[command.index(option) + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"Generator command lacks {option}") from exc
    return [float(value) for value in raw.split(";")]


def _option_int(command: Sequence[str], option: str) -> int:
    try:
        return int(command[command.index(option) + 1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"Generator command lacks {option}") from exc


def _total_trajectories(
    energies: np.ndarray,
    charges: np.ndarray,
    impact: np.ndarray,
    trajectories: int,
    target_bmax: Sequence[float],
    loss_bmax: Sequence[float],
) -> int:
    channels = 0
    for charge in charges:
        for b_value in impact:
            channels += sum(b_value <= cutoff for cutoff in target_bmax)
            q = int(charge)
            channels += int(q < len(loss_bmax) and b_value <= loss_bmax[q])
    return int(len(energies) * channels * trajectories)


if __name__ == "__main__":
    raise SystemExit(main())
