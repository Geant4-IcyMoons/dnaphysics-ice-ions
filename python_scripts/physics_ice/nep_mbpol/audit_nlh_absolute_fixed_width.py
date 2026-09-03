#!/usr/bin/env python3
"""Reassess a checkpointed NLH campaign with absolute fixed-width gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bca.convergence import (  # noqa: E402
    DEFAULT_MEANINGFUL_SIGNIFICANT_DIGITS,
    OBSERVABLES,
    RatioStatistics,
    absolute_tolerances_from_estimates,
    convergence_report,
    doubling_schedule,
    merge_statistics,
    simultaneous_dkw_half_width,
)


BATCH_PATTERN = re.compile(r"batch_(\d+)_(\d+)\.manifest\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument(
        "--meaningful-significant-digits",
        type=int,
        default=DEFAULT_MEANINGFUL_SIGNIFICANT_DIGITS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: CAMPAIGN_ROOT/absolute_fixed_width_audit.json",
    )
    parser.add_argument(
        "--reuse-existing-results",
        action="store_true",
        help=(
            "Reuse completed per-case assessments only when their immutable "
            "checkpointed trajectory counts are unchanged."
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _batch_bounds(path: Path) -> tuple[int, int]:
    match = BATCH_PATTERN.fullmatch(path.name)
    if match is None:
        raise RuntimeError(f"Malformed batch-manifest name: {path}")
    return int(match.group(1)), int(match.group(2))


def _checkpoint_batches(case_directory: Path) -> list[Path]:
    checkpoint_roots = list(case_directory.glob(".trajectory_checkpoints/*"))
    if len(checkpoint_roots) != 1:
        raise RuntimeError(
            f"Expected one checkpoint configuration below {case_directory}; "
            f"found {len(checkpoint_roots)}."
        )
    return sorted(
        checkpoint_roots[0].glob("batch_*.manifest.json"),
        key=lambda path: _batch_bounds(path)[0],
    )


def _recommended_look(
    projected: int | None, schedule: tuple[int, ...]
) -> int | None:
    if projected is None:
        return None
    return next((count for count in schedule if count >= projected), None)


def _calibration_contract(
    calibration_manifest: Path, significant_digits: int
) -> tuple[dict[str, float], dict[str, float]]:
    manifest = _load_json(calibration_manifest)
    raw_report = manifest.get("raw_statistical_convergence")
    if not isinstance(raw_report, dict):
        raise RuntimeError(f"Missing raw calibration report: {calibration_manifest}")
    observables = raw_report.get("observables")
    if not isinstance(observables, dict) or set(observables) != set(OBSERVABLES):
        raise RuntimeError(f"Malformed raw calibration report: {calibration_manifest}")
    estimates = {
        name: float(observables[name]["estimate"]) for name in OBSERVABLES
    }
    return estimates, absolute_tolerances_from_estimates(
        estimates, significant_digits
    )


def _assess_case(
    root: Path,
    case_key: str,
    *,
    schedule: tuple[int, ...],
    confidence: float,
    cdf_tolerance: float,
    significant_digits: int,
) -> dict[str, Any]:
    calibration_manifest = (
        root / "estimator_calibration" / case_key / "hard_collision_run.manifest.json"
    )
    production_directory = root / "production" / case_key
    reference_estimates, tolerances = _calibration_contract(
        calibration_manifest, significant_digits
    )
    batches = _checkpoint_batches(production_directory)
    if not batches:
        raise RuntimeError(f"No production batches found for {case_key}")
    expected_start = 0
    signature: str | None = None
    completed = 0
    for path in batches:
        start, stop = _batch_bounds(path)
        if start != expected_start or stop <= start:
            raise RuntimeError(f"Non-contiguous batches at {path}")
        expected_start = stop
        completed = stop
    eligible_looks = [
        (index, count)
        for index, count in enumerate(schedule, start=1)
        if count <= completed
    ]
    if not eligible_looks:
        return {
            "case_key": case_key,
            "checkpointed_trajectories": completed,
            "assessed_trajectories": 0,
            "converged": False,
            "reason": "minimum scheduled look not reached",
            "absolute_tolerances": tolerances,
            "tolerance_reference_estimates": reference_estimates,
        }
    look_index, assessed = eligible_looks[-1]
    batch_statistics: list[dict[str, RatioStatistics]] = []
    cdf_sample_count = 0
    expected_start = 0
    for path in batches:
        start, stop = _batch_bounds(path)
        if stop > assessed:
            break
        record = _load_json(path)
        record_signature = str(record.get("configuration_signature", ""))
        if not record_signature or (
            signature is not None and record_signature != signature
        ):
            raise RuntimeError(f"Checkpoint signature mismatch: {path}")
        signature = record_signature
        if int(record["trajectory_start"]) != start or int(
            record["trajectory_stop"]
        ) != stop or start != expected_start:
            raise RuntimeError(f"Checkpoint range mismatch: {path}")
        raw_statistics = record.get("statistics")
        if not isinstance(raw_statistics, dict) or set(raw_statistics) != set(
            OBSERVABLES
        ):
            raise RuntimeError(f"Incomplete sufficient statistics: {path}")
        batch_statistics.append(
            {
                name: RatioStatistics.from_dict(raw_statistics[name])
                for name in OBSERVABLES
            }
        )
        summary = record.get("summary")
        if not isinstance(summary, dict):
            raise RuntimeError(f"Missing checkpoint summary: {path}")
        cdf_sample_count += int(summary["target_distribution_sample_count"])
        expected_start = stop
    if expected_start != assessed:
        raise RuntimeError(f"Scheduled look {assessed:,} is not batch-aligned.")
    statistics = merge_statistics(batch_statistics)
    report = convergence_report(
        statistics,
        absolute_tolerances=tolerances,
        confidence=confidence,
        scheduled_look_count=len(schedule),
        look_index=look_index,
    )
    cdf_half_width, cdf_individual_confidence = simultaneous_dkw_half_width(
        cdf_sample_count,
        confidence,
        distribution_count=2,
        scheduled_look_count=len(schedule),
    )
    cdf_passes = cdf_half_width <= cdf_tolerance
    report["trajectory_cdf_convergence"] = {
        "passes": cdf_passes,
        "absolute_confidence_half_width": cdf_half_width,
        "absolute_tolerance": cdf_tolerance,
        "target_distribution_sample_count": cdf_sample_count,
        "individual_band_confidence": cdf_individual_confidence,
        "method": "Dvoretzky-Kiefer-Wolfowitz-Massart band with Bonferroni correction",
    }
    report["converged"] = bool(report["converged"] and cdf_passes)
    next_look = next((count for count in schedule if count > assessed), None)
    maximum_normalized_width = report["maximum_normalized_confidence_width"]
    projected = None
    recommended_look = None
    if maximum_normalized_width is not None:
        projected = math.ceil(
            assessed * max(1.0, float(maximum_normalized_width) ** 2)
        )
        recommended_look = _recommended_look(projected, schedule)
    return {
        "case_key": case_key,
        "configuration_signature": signature,
        "checkpointed_trajectories": completed,
        "assessed_trajectories": assessed,
        "unassessed_checkpointed_trajectories": completed - assessed,
        "next_scheduled_look": next_look,
        "projected_trajectories_from_inverse_sqrt_scaling": projected,
        "recommended_scheduled_look": recommended_look,
        "meaningful_significant_digits": significant_digits,
        "tolerance_reference": "independent raw estimator calibration",
        "tolerance_reference_estimates": reference_estimates,
        "statistical_convergence": report,
        "converged": bool(report["converged"]),
    }


def main() -> int:
    args = parse_args()
    if args.meaningful_significant_digits < 1:
        raise ValueError("--meaningful-significant-digits must be positive.")
    root = args.campaign_root.expanduser().resolve()
    state = _load_json(root / "adaptive_particle_shards.state.json")
    waves = state.get("waves")
    if not isinstance(waves, list) or not waves:
        raise RuntimeError("Campaign state contains no waves.")
    wave_manifests = [_load_json(Path(str(wave["manifest"]))) for wave in waves]
    configuration = wave_manifests[0]["configuration"]
    runtime = wave_manifests[0]["runtime"]
    confidence = float(configuration["confidence"])
    cdf_tolerance = float(
        configuration.get("trajectory_cdf_tolerance", configuration["tolerance"])
    )
    schedule = doubling_schedule(
        int(configuration["minimum_trajectories"]),
        int(runtime["maximum_trajectories"]),
        int(configuration["trajectory_batch_size"]),
    )
    case_keys = sorted(
        {
            str(item["case_key"])
            for wave in wave_manifests
            for item in wave["cases"]
        }
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "absolute_fixed_width_audit.json"
    )
    partial_output = output.with_name(output.name + ".partial")
    cached_cases: dict[str, dict[str, Any]] = {}
    cache_path = (
        output
        if args.reuse_existing_results and output.is_file()
        else partial_output
    )
    if cache_path.is_file():
        cached = _load_json(cache_path)
        cached_digits = cached.get("policy", {}).get(
            "meaningful_significant_digits"
        )
        if cached_digits == args.meaningful_significant_digits:
            cached_cases = {
                str(case["case_key"]): case for case in cached.get("cases", [])
            }
    cases = []
    for case_key in tqdm(case_keys, desc="Absolute fixed-width audit", unit="case"):
        cached_case = cached_cases.get(case_key)
        checkpointed = _batch_bounds(
            _checkpoint_batches(root / "production" / case_key)[-1]
        )[1]
        if (
            cached_case is not None
            and int(cached_case["checkpointed_trajectories"]) == checkpointed
        ):
            case_result = dict(cached_case)
            if "recommended_scheduled_look" not in case_result:
                case_result["recommended_scheduled_look"] = _recommended_look(
                    case_result.get(
                        "projected_trajectories_from_inverse_sqrt_scaling"
                    ),
                    schedule,
                )
        else:
            case_result = _assess_case(
                root,
                case_key,
                schedule=schedule,
                confidence=confidence,
                cdf_tolerance=cdf_tolerance,
                significant_digits=args.meaningful_significant_digits,
            )
        cases.append(case_result)
        _atomic_json(
            partial_output,
            {
                "schema_version": 1,
                "policy": {
                    "meaningful_significant_digits": (
                        args.meaningful_significant_digits
                    )
                },
                "cases": cases,
            },
        )
    converged = sum(bool(case["converged"]) for case in cases)
    assessed = sum(int(case["assessed_trajectories"]) for case in cases)
    checkpointed = sum(int(case["checkpointed_trajectories"]) for case in cases)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(root),
        "policy": {
            "criterion": "simultaneous absolute fixed-width confidence intervals",
            "familywise_confidence": confidence,
            "scheduled_look_count": len(schedule),
            "meaningful_significant_digits": args.meaningful_significant_digits,
            "precision_selection_basis": (
                "predeclared project reporting requirement; the literature "
                "does not prescribe the number of meaningful digits"
            ),
            "absolute_width_source": (
                "independent raw 10000-trajectory estimator calibration; "
                "decimal tolerance defined by JCGM 101:2008 section 7.9.2"
            ),
            "references": [
                "Glynn and Whitt (1992), doi:10.1214/aoap/1177005770",
                "Flegal and Gong (2015), doi:10.5705/ss.2013.209",
                "JCGM 101:2008, section 7.9",
                "Massart (1990), doi:10.1214/aop/1176990746",
            ],
        },
        "summary": {
            "case_count": len(cases),
            "converged_case_count": converged,
            "additional_sampling_case_count": len(cases) - converged,
            "checkpointed_trajectories": checkpointed,
            "assessed_trajectories": assessed,
        },
        "cases": cases,
    }
    _atomic_json(output, payload)
    if partial_output.is_file():
        partial_output.unlink()
    print(f"Converged cases: {converged}/{len(cases)}")
    print(f"Checkpointed trajectories: {checkpointed:,}")
    print(f"Assessed at scheduled looks: {assessed:,}")
    print(f"Audit: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
