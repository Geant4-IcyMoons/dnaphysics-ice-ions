#!/usr/bin/env python3
"""Screen the proposed Garvey role term at selected CTMC81 loss points.

This diagnostic leaves the production convention unchanged.  It applies
``a=1`` only to the perturbing core, runs selected projectile-loss impact
parameters, and compares them with both CTMC81 and the completed baseline.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import signal

import numpy as np
from scipy.stats import norm

import run_ctmc81_c3_reproduction as reproduction


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    HERE / "runs" / "ctmc81_c3_100keV_u_garvey_role_sensitivity"
)
DEFAULT_BASELINE_CSV = (
    HERE
    / "runs"
    / "ctmc81_c3_100keV_u_olson_salop_com_balanced_failed"
    / "ctmc81_pointwise_comparison.csv"
)
DEFAULT_IMPACT_INDICES = (0, 1, 5, 10, 15, 20, 25, 30)


def _parse_indices(text: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(value) for value in text.split(",")))
    if not values or any(value < 0 or value >= 100 for value in values):
        raise argparse.ArgumentTypeError(
            "impact indices must be unique integers from 0 through 99"
        )
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-archive",
        type=Path,
        default=reproduction.FORMAL_REFERENCE_ARCHIVE,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--trajectories-per-replica", type=int, default=2_500)
    parser.add_argument("--trajectory-chunk-size", type=int, default=100)
    parser.add_argument("--checkpoint-seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--impact-indices",
        type=_parse_indices,
        default=DEFAULT_IMPACT_INDICES,
    )
    return parser.parse_args()


def _selected_specs(
    reference: dict[str, object],
    impact_indices: tuple[int, ...],
    trajectories_per_replica: int,
) -> tuple[dict[str, object], ...]:
    selected = []
    for source in reproduction._build_specs(reference):
        if source["dataset"] != "loss":
            continue
        if int(source["impact_index"]) not in impact_indices:
            continue
        spec = dict(source)
        spec["source_point_index"] = int(source["point_index"])
        spec["point_index"] = len(selected)
        spec["reference_trajectory_count"] = int(source["trajectory_count"])
        spec["trajectory_count"] = int(trajectories_per_replica)
        spec["garvey_role_term"] = True
        selected.append(spec)
    if len(selected) != len(impact_indices):
        raise RuntimeError("Not every requested loss impact index was found")
    selected.sort(key=lambda spec: impact_indices.index(int(spec["impact_index"])))
    for point_index, spec in enumerate(selected):
        spec["point_index"] = point_index
    return tuple(selected)


def _read_baseline(path: Path) -> dict[tuple[int, str], dict[str, float]]:
    result: dict[tuple[int, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] != "loss":
                continue
            result[(int(row["impact_index"]), row["event"])] = {
                "probability": float(row["calculated_probability"]),
                "trajectories": float(row["calculated_trajectories"]),
            }
    return result


def _variance(
    reference_probability: float,
    reference_count: int,
    calculated_probability: float,
    calculated_count: int,
) -> float:
    return (
        reference_probability * (1.0 - reference_probability) / reference_count
        + calculated_probability
        * (1.0 - calculated_probability)
        / calculated_count
        + (0.5e-4) ** 2 / 3.0
    )


def _comparison(
    *,
    reference: dict[str, object],
    specs: tuple[dict[str, object], ...],
    events: np.ndarray,
    completed: np.ndarray,
    baseline: dict[tuple[int, str], dict[str, float]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    simultaneous_z = float(norm.ppf(1.0 - 0.05 / (2.0 * len(specs) * 4)))
    dataset = reference["datasets"]["loss"]
    column_names = tuple(dataset["column_names"])
    for spec in specs:
        point_index = int(spec["point_index"])
        impact_index = int(spec["impact_index"])
        reference_probability = np.asarray(
            dataset["probabilities"][impact_index], dtype=float
        )
        reference_count = int(spec["reference_trajectory_count"])
        role_counts = np.sum(events[:, point_index], axis=0)
        role_count = int(np.sum(completed[:, point_index]))
        role_probability = role_counts / role_count
        for column, event in enumerate(column_names):
            p_ref = float(reference_probability[column])
            p_role = float(role_probability[column])
            role_standard_error = math.sqrt(
                _variance(p_ref, reference_count, p_role, role_count)
            )
            baseline_record = baseline[(impact_index, str(event))]
            p_baseline = float(baseline_record["probability"])
            baseline_count = int(baseline_record["trajectories"])
            baseline_standard_error = math.sqrt(
                _variance(p_ref, reference_count, p_baseline, baseline_count)
            )
            rows.append(
                {
                    "impact_index": impact_index,
                    "impact_au": float(spec["impact_au"]),
                    "event": str(event),
                    "reference_probability": p_ref,
                    "reference_trajectories": reference_count,
                    "baseline_probability": p_baseline,
                    "baseline_trajectories": baseline_count,
                    "baseline_z_score": (p_baseline - p_ref)
                    / baseline_standard_error,
                    "role_probability": p_role,
                    "role_trajectories": role_count,
                    "role_z_score": (p_role - p_ref) / role_standard_error,
                    "role_within_selected_95pct_simultaneous_band": bool(
                        abs(p_role - p_ref)
                        <= simultaneous_z * role_standard_error
                    ),
                }
            )

    physical = [row for row in rows if row["event"] != "null"]
    loss = [row for row in rows if row["event"] == "loss"]

    def rmse(records: list[dict[str, object]], key: str) -> float:
        return math.sqrt(
            sum(
                (float(row[key]) - float(row["reference_probability"])) ** 2
                for row in records
            )
            / len(records)
        )

    metrics = {
        "selected_simultaneous_z": simultaneous_z,
        "selected_comparison_count": len(rows),
        "baseline_physical_event_rmse": rmse(
            physical, "baseline_probability"
        ),
        "role_physical_event_rmse": rmse(physical, "role_probability"),
        "baseline_loss_rmse": rmse(loss, "baseline_probability"),
        "role_loss_rmse": rmse(loss, "role_probability"),
        "baseline_selected_band_failures": sum(
            abs(float(row["baseline_z_score"])) > simultaneous_z
            for row in rows
        ),
        "role_selected_band_failures": sum(
            not bool(row["role_within_selected_95pct_simultaneous_band"])
            for row in rows
        ),
    }
    metrics["hypothesis_supported_on_selected_points"] = bool(
        metrics["role_physical_event_rmse"]
        < metrics["baseline_physical_event_rmse"]
        and metrics["role_loss_rmse"] < metrics["baseline_loss_rmse"]
    )
    return rows, metrics


def _screening_parameters() -> dict[str, dict[str, list[float]]]:
    ctmc = reproduction.ctmc
    baseline_target = ctmc._trajectory_cores(3, "target")
    baseline_loss = ctmc._trajectory_cores(3, "projectile")
    role_target = ctmc._trajectory_cores(3, "target", True)
    role_loss = ctmc._trajectory_cores(3, "projectile", True)

    def pair(core: object) -> list[float]:
        return [float(core.eta), float(core.zeta)]

    return {
        "baseline": {
            "target_h2o_host": pair(baseline_target[0]),
            "target_carbon_perturber": pair(baseline_target[1]),
            "loss_carbon_host": pair(baseline_loss[1]),
            "loss_h2o_perturber": pair(baseline_loss[0]),
        },
        "role_sensitivity": {
            "target_h2o_host": pair(role_target[0]),
            "target_carbon_perturber": pair(role_target[1]),
            "loss_carbon_host": pair(role_loss[1]),
            "loss_h2o_perturber": pair(role_loss[0]),
        },
    }


def main() -> None:
    args = _parse_args()
    if min(
        args.workers,
        args.replicas,
        args.trajectories_per_replica,
        args.trajectory_chunk_size,
    ) < 1:
        raise ValueError("worker and trajectory counts must be positive")
    reference = reproduction._read_formal_reference_archive(
        args.reference_archive
    )
    if reference["sha256"] != reproduction.FORMAL_REFERENCE_SHA256:
        raise RuntimeError("The CTMC81 archive hash does not match")
    specs = _selected_specs(
        reference,
        args.impact_indices,
        args.trajectories_per_replica,
    )
    config = reproduction._configuration(
        trajectory_chunk_size=args.trajectory_chunk_size,
        seed=args.seed,
        rtol=1.0e-11,
        atol=1.0e-13,
        retry_rtol=1.0e-12,
        retry_atol=1.0e-14,
        maximum_relative_energy_drift=1.0e-4,
        maximum_integration_steps=100_000_000,
    )
    config = replace(
        config,
        projectile_loss_initialization=(
            reproduction.ctmc.PROJECTILE_LOSS_INITIALIZATION_COM_BALANCED
        ),
    )
    reproduction.ctmc.select_projectile("carbon")
    reproduction.ctmc.warm_up_initial_state_sampling(
        np.asarray([3], dtype=int), config.radial_grid_points
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = reproduction._signature(reference, specs, config, args.replicas)
    for signal_name in ("SIGTERM", "SIGUSR1"):
        value = getattr(signal, signal_name, None)
        if value is not None:
            signal.signal(value, reproduction._request_stop)
    events, completed, maximum_drift = reproduction._run_reproduction(
        workers=min(args.workers, len(specs) * args.replicas),
        replicas=args.replicas,
        specs=specs,
        config=config,
        signature_value=signature,
        checkpoint_path=output_dir / "garvey_role_sensitivity_checkpoint.npz",
        checkpoint_seconds=args.checkpoint_seconds,
    )
    baseline = _read_baseline(args.baseline_csv)
    rows, metrics = _comparison(
        reference=reference,
        specs=specs,
        events=events,
        completed=completed,
        baseline=baseline,
    )
    reproduction._write_csv(output_dir / "garvey_role_comparison.csv", rows)
    summary = {
        "status": "diagnostic_complete",
        "interpretation": (
            "selected-point sensitivity only; not a production-model decision"
        ),
        "signature": signature,
        "reference_sha256": reference["sha256"],
        "impact_indices": list(args.impact_indices),
        "replicas": args.replicas,
        "trajectories_per_replica": args.trajectories_per_replica,
        "combined_trajectories_per_point": int(np.sum(completed[:, 0])),
        "screening_parameters_eta_zeta": _screening_parameters(),
        "maximum_relative_energy_drift": float(np.max(maximum_drift)),
        "metrics": metrics,
    }
    reproduction._atomic_json(
        output_dir / "garvey_role_sensitivity_summary.json", summary
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
