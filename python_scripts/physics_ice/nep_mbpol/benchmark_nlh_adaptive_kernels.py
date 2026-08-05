#!/usr/bin/env python3
"""Benchmark adaptive NLH impact meshes against dense direct-solver references."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bca.adaptivity import adaptive_energy_mesh, adaptive_impact_mesh  # noqa: E402
from bca.config import (  # noqa: E402
    DEFAULT_AXIS_RELATIVE_TOLERANCE,
    DEFAULT_BASE_ENERGY_POINTS,
    DEFAULT_ENERGY_MAX_EV,
    DEFAULT_ENERGY_MIN_EV,
    DEFAULT_MAX_ENERGY_POINTS,
    DEFAULT_MAX_IMPACT_POINTS,
    DEFAULT_MINIMUM_TURNING_POTENTIAL_EV,
    DEFAULT_QUADRATURE_ORDER,
    DEFAULT_WORKERS,
    ICE_TARGETS,
    canonical_element,
)
from bca.scattering import (  # noqa: E402
    NLHCollisionKernel,
    pair_kinematics,
    two_body_observables_from_cm_angles,
)


ALL_PROJECTILES = ("H", "He", "C", "O", "S")


@dataclass(frozen=True)
class BenchmarkCase:
    projectile: str
    target: str
    projectile_energy_ev: float
    adaptive_points: int
    reference_points: int
    mean_recoil_relative_error: float
    recoil_l1_relative_error: float
    transport_relative_error: float
    transport_l1_relative_error: float
    theta_cm_l1_relative_error: float
    theta_cm_max_error_over_pi: float


@dataclass(frozen=True)
class EnergyBenchmarkCase:
    projectile: str
    target: str
    adaptive_energy_points: int
    validation_points: int
    maximum_theta_cm_relative_error: float
    maximum_recoil_relative_error: float


@dataclass(frozen=True)
class QuadratureBenchmarkCase:
    projectile: str
    target: str
    projectile_energy_ev: float
    production_order: int
    reference_order: int
    validation_points: int
    theta_cm_max_error_over_pi: float
    maximum_recoil_relative_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projectiles", nargs="+", default=ALL_PROJECTILES)
    parser.add_argument("--energy-min-ev", type=float, default=DEFAULT_ENERGY_MIN_EV)
    parser.add_argument("--energy-max-ev", type=float, default=DEFAULT_ENERGY_MAX_EV)
    parser.add_argument("--energy-cases", type=int, default=6)
    parser.add_argument(
        "--base-energy-points", type=int, default=DEFAULT_BASE_ENERGY_POINTS
    )
    parser.add_argument(
        "--max-energy-points", type=int, default=DEFAULT_MAX_ENERGY_POINTS
    )
    parser.add_argument(
        "--axis-relative-tolerance",
        type=float,
        default=DEFAULT_AXIS_RELATIVE_TOLERANCE,
    )
    parser.add_argument(
        "--acceptance-tolerance",
        type=float,
        default=2.0 * DEFAULT_AXIS_RELATIVE_TOLERANCE,
        help="Independent combined acceptance limit (default: 0.005 = 0.5%%).",
    )
    parser.add_argument("--reference-log-points", type=int, default=1025)
    parser.add_argument("--reference-linear-points", type=int, default=1025)
    parser.add_argument(
        "--minimum-turning-potential-ev",
        type=float,
        default=DEFAULT_MINIMUM_TURNING_POTENTIAL_EV,
    )
    parser.add_argument(
        "--quadrature-order", type=int, default=DEFAULT_QUADRATURE_ORDER
    )
    parser.add_argument(
        "--reference-quadrature-order",
        type=int,
        help="Independent quadrature order (default: twice --quadrature-order).",
    )
    parser.add_argument(
        "--max-impact-points", type=int, default=DEFAULT_MAX_IMPACT_POINTS
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=HERE / "collision_benchmarks",
    )
    return parser.parse_args()


def _direct_values(
    kernel: NLHCollisionKernel, quantiles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recoil = np.empty(len(quantiles))
    transport = np.empty(len(quantiles))
    theta_cm = np.empty(len(quantiles))
    maximum_impact = kernel.maximum_impact_parameter_angstrom
    for index, quantile in enumerate(quantiles):
        result = kernel.solve(maximum_impact * math.sqrt(float(quantile)))
        recoil[index] = result.recoil_energy_ev
        transport[index] = 1.0 - math.cos(result.theta_projectile_lab_rad)
        theta_cm[index] = result.theta_cm_rad
    return recoil, transport, theta_cm


def _relative_integral_error(
    interpolated: np.ndarray, reference: np.ndarray, quantiles: np.ndarray
) -> float:
    reference_integral = float(np.trapezoid(reference, quantiles))
    return abs(float(np.trapezoid(interpolated, quantiles)) / reference_integral - 1.0)


def _relative_l1_error(
    interpolated: np.ndarray, reference: np.ndarray, quantiles: np.ndarray
) -> float:
    denominator = float(np.trapezoid(reference, quantiles))
    return float(np.trapezoid(np.abs(interpolated - reference), quantiles)) / denominator


def _run_case(task: tuple[object, ...]) -> BenchmarkCase:
    (
        projectile,
        target,
        energy_ev,
        axis_tolerance,
        max_impact_points,
        minimum_turning_potential_ev,
        quadrature_order,
        reference_log_points,
        reference_linear_points,
    ) = task
    kernel = NLHCollisionKernel(
        str(projectile),
        str(target),
        float(energy_ev),
        minimum_turning_potential_ev=float(minimum_turning_potential_ev),
        quadrature_order=int(quadrature_order),
    )
    mesh = adaptive_impact_mesh(
        kernel,
        relative_tolerance=float(axis_tolerance),
        max_points=int(max_impact_points),
    )
    reference_quantiles = np.unique(
        np.concatenate(
            (
                np.asarray((0.0,)),
                np.geomspace(1.0e-24, 1.0e-2, int(reference_log_points)),
                np.linspace(1.0e-2, 1.0, int(reference_linear_points)),
            )
        )
    )
    reference_recoil, reference_transport, reference_theta = _direct_values(
        kernel, reference_quantiles
    )
    adaptive_theta = np.asarray([value.theta_cm_rad for value in mesh.collisions])
    interpolated_theta = np.interp(
        reference_quantiles, mesh.area_quantiles, adaptive_theta
    )
    interpolated_lab_theta, interpolated_recoil = (
        two_body_observables_from_cm_angles(
            kernel.kinematics, interpolated_theta
        )
    )
    interpolated_transport = 1.0 - np.cos(interpolated_lab_theta)
    return BenchmarkCase(
        projectile=str(projectile),
        target=str(target),
        projectile_energy_ev=float(energy_ev),
        adaptive_points=mesh.point_count,
        reference_points=len(reference_quantiles),
        mean_recoil_relative_error=_relative_integral_error(
            interpolated_recoil, reference_recoil, reference_quantiles
        ),
        recoil_l1_relative_error=_relative_l1_error(
            interpolated_recoil, reference_recoil, reference_quantiles
        ),
        transport_relative_error=_relative_integral_error(
            interpolated_transport, reference_transport, reference_quantiles
        ),
        transport_l1_relative_error=_relative_l1_error(
            interpolated_transport, reference_transport, reference_quantiles
        ),
        theta_cm_l1_relative_error=_relative_l1_error(
            interpolated_theta, reference_theta, reference_quantiles
        ),
        theta_cm_max_error_over_pi=float(
            np.max(np.abs(interpolated_theta - reference_theta))
        )
        / math.pi,
    )


def _energy_values(
    projectile: str,
    target: str,
    energy_ev: float,
    quantiles: np.ndarray,
    minimum_turning_potential_ev: float,
    quadrature_order: int,
) -> np.ndarray:
    kernel = NLHCollisionKernel(
        projectile,
        target,
        energy_ev,
        minimum_turning_potential_ev=minimum_turning_potential_ev,
        quadrature_order=quadrature_order,
    )
    rows = []
    for quantile in quantiles:
        result = kernel.solve(
            kernel.maximum_impact_parameter_angstrom * math.sqrt(float(quantile))
        )
        rows.append((result.theta_cm_rad, result.recoil_energy_ev))
    return np.asarray(rows)


def _run_energy_case(task: tuple[object, ...]) -> EnergyBenchmarkCase:
    (
        projectile,
        target,
        energy_min_ev,
        energy_max_ev,
        base_energy_points,
        axis_tolerance,
        max_energy_points,
        minimum_turning_potential_ev,
        quadrature_order,
    ) = task
    projectile = str(projectile)
    target = str(target)
    mesh = adaptive_energy_mesh(
        projectile,
        target,
        energy_min_ev=float(energy_min_ev),
        energy_max_ev=float(energy_max_ev),
        base_points=int(base_energy_points),
        minimum_turning_potential_ev=float(minimum_turning_potential_ev),
        quadrature_order=int(quadrature_order),
        relative_tolerance=float(axis_tolerance),
        max_points=int(max_energy_points),
    )
    quantiles = np.unique(
        np.concatenate(
            (
                np.asarray((0.0,)),
                np.geomspace(1.0e-20, 1.0e-2, 16),
                np.linspace(1.0e-2, 1.0, 17),
            )
        )
    )
    cache = {
        float(energy): _energy_values(
            projectile,
            target,
            float(energy),
            quantiles,
            float(minimum_turning_potential_ev),
            int(quadrature_order),
        )
        for energy in mesh.energies_ev
    }
    maximum_theta_error = 0.0
    maximum_recoil_error = 0.0
    validation_points = 0
    independent_fractions = (0.125, 0.375, 0.625, 0.875)
    for lower_energy, upper_energy in zip(
        mesh.energies_ev[:-1], mesh.energies_ev[1:], strict=True
    ):
        lower = cache[float(lower_energy)]
        upper = cache[float(upper_energy)]
        log_lower = math.log(float(lower_energy))
        log_upper = math.log(float(upper_energy))
        for fraction in independent_fractions:
            probe_energy = math.exp(
                (1.0 - fraction) * log_lower + fraction * log_upper
            )
            actual = _energy_values(
                projectile,
                target,
                probe_energy,
                quantiles,
                float(minimum_turning_potential_ev),
                int(quadrature_order),
            )
            predicted_theta = np.exp(
                (1.0 - fraction) * np.log(lower[:, 0])
                + fraction * np.log(upper[:, 0])
            )
            probe_kinematics = pair_kinematics(
                projectile, target, probe_energy
            )
            _, predicted_recoil = two_body_observables_from_cm_angles(
                probe_kinematics, predicted_theta
            )
            maximum_theta_error = max(
                maximum_theta_error,
                float(np.max(np.abs(predicted_theta / actual[:, 0] - 1.0))),
            )
            maximum_recoil_error = max(
                maximum_recoil_error,
                float(np.max(np.abs(predicted_recoil / actual[:, 1] - 1.0))),
            )
            validation_points += len(quantiles)
    return EnergyBenchmarkCase(
        projectile=projectile,
        target=target,
        adaptive_energy_points=mesh.point_count,
        validation_points=validation_points,
        maximum_theta_cm_relative_error=maximum_theta_error,
        maximum_recoil_relative_error=maximum_recoil_error,
    )


def _run_quadrature_case(task: tuple[object, ...]) -> QuadratureBenchmarkCase:
    (
        projectile,
        target,
        energy_ev,
        production_order,
        reference_order,
        minimum_turning_potential_ev,
    ) = task
    projectile = str(projectile)
    target = str(target)
    energy_ev = float(energy_ev)
    production = NLHCollisionKernel(
        projectile,
        target,
        energy_ev,
        minimum_turning_potential_ev=float(minimum_turning_potential_ev),
        quadrature_order=int(production_order),
    )
    reference = NLHCollisionKernel(
        projectile,
        target,
        energy_ev,
        minimum_turning_potential_ev=float(minimum_turning_potential_ev),
        quadrature_order=int(reference_order),
    )
    quantiles = np.unique(
        np.concatenate(
            (
                np.asarray((0.0,)),
                np.geomspace(1.0e-20, 1.0e-2, 24),
                np.linspace(1.0e-2, 1.0, 25),
            )
        )
    )
    production_recoil, _, production_theta = _direct_values(
        production, quantiles
    )
    reference_recoil, _, reference_theta = _direct_values(reference, quantiles)
    return QuadratureBenchmarkCase(
        projectile=projectile,
        target=target,
        projectile_energy_ev=energy_ev,
        production_order=int(production_order),
        reference_order=int(reference_order),
        validation_points=len(quantiles),
        theta_cm_max_error_over_pi=float(
            np.max(np.abs(production_theta - reference_theta))
        )
        / math.pi,
        maximum_recoil_relative_error=float(
            np.max(np.abs(production_recoil / reference_recoil - 1.0))
        ),
    )


def _parallel_cases(function, tasks, *, workers: int, description: str, unit: str):
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(function, task) for task in tasks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            unit=unit,
            desc=description,
        ):
            results.append(future.result())
    return results


def main() -> None:
    args = parse_args()
    projectiles = tuple(canonical_element(value) for value in args.projectiles)
    energies = np.geomspace(
        args.energy_min_ev, args.energy_max_ev, args.energy_cases
    )
    tasks = [
        (
            projectile,
            target,
            energy,
            args.axis_relative_tolerance,
            args.max_impact_points,
            args.minimum_turning_potential_ev,
            args.quadrature_order,
            args.reference_log_points,
            args.reference_linear_points,
        )
        for projectile in projectiles
        for target in ICE_TARGETS
        for energy in energies
    ]
    cases = _parallel_cases(
        _run_case,
        tasks,
        workers=args.workers,
        description="Dense-reference benchmark",
        unit="case",
    )
    cases.sort(key=lambda value: (value.projectile, value.target, value.projectile_energy_ev))

    energy_tasks = [
        (
            projectile,
            target,
            args.energy_min_ev,
            args.energy_max_ev,
            args.base_energy_points,
            args.axis_relative_tolerance,
            args.max_energy_points,
            args.minimum_turning_potential_ev,
            args.quadrature_order,
        )
        for projectile in projectiles
        for target in ICE_TARGETS
    ]
    energy_cases = _parallel_cases(
        _run_energy_case,
        energy_tasks,
        workers=args.workers,
        description="Independent energy benchmark",
        unit="pair",
    )
    energy_cases.sort(key=lambda value: (value.projectile, value.target))

    reference_quadrature_order = (
        args.reference_quadrature_order or 2 * args.quadrature_order
    )
    if reference_quadrature_order <= args.quadrature_order:
        raise ValueError("Reference quadrature order must exceed production order.")
    quadrature_tasks = [
        (
            projectile,
            target,
            energy,
            args.quadrature_order,
            reference_quadrature_order,
            args.minimum_turning_potential_ev,
        )
        for projectile in projectiles
        for target in ICE_TARGETS
        for energy in energies
    ]
    quadrature_cases = _parallel_cases(
        _run_quadrature_case,
        quadrature_tasks,
        workers=args.workers,
        description="Quadrature benchmark",
        unit="case",
    )
    quadrature_cases.sort(
        key=lambda value: (
            value.projectile,
            value.target,
            value.projectile_energy_ev,
        )
    )

    metrics = (
        "mean_recoil_relative_error",
        "recoil_l1_relative_error",
        "transport_relative_error",
        "transport_l1_relative_error",
        "theta_cm_l1_relative_error",
        "theta_cm_max_error_over_pi",
    )
    impact_maxima = {
        name: max(getattr(case, name) for case in cases) for name in metrics
    }
    energy_maxima = {
        "maximum_theta_cm_relative_error": max(
            case.maximum_theta_cm_relative_error for case in energy_cases
        ),
        "maximum_recoil_relative_error": max(
            case.maximum_recoil_relative_error for case in energy_cases
        ),
    }
    quadrature_maxima = {
        "theta_cm_max_error_over_pi": max(
            case.theta_cm_max_error_over_pi for case in quadrature_cases
        ),
        "maximum_recoil_relative_error": max(
            case.maximum_recoil_relative_error for case in quadrature_cases
        ),
    }
    accepted = all(
        value <= args.acceptance_tolerance
        for value in (
            *impact_maxima.values(),
            *energy_maxima.values(),
            *quadrature_maxima.values(),
        )
    )
    output = args.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "nlh_adaptive_kernel_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=asdict(cases[0]).keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(asdict(case) for case in cases)
    energy_csv_path = output / "nlh_adaptive_energy_benchmark.csv"
    with energy_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=asdict(energy_cases[0]).keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(asdict(case) for case in energy_cases)
    quadrature_csv_path = output / "nlh_quadrature_benchmark.csv"
    with quadrature_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=asdict(quadrature_cases[0]).keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(case) for case in quadrature_cases)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
        "axis_relative_tolerance": args.axis_relative_tolerance,
        "acceptance_tolerance": args.acceptance_tolerance,
        "projectiles": list(projectiles),
        "targets": list(ICE_TARGETS),
        "energies_ev": energies.tolist(),
        "case_count": len(cases),
        "maximum_impact_errors": impact_maxima,
        "maximum_energy_errors": energy_maxima,
        "maximum_quadrature_errors": quadrature_maxima,
        "adaptive_points": {
            "minimum": min(case.adaptive_points for case in cases),
            "median": float(np.median([case.adaptive_points for case in cases])),
            "maximum": max(case.adaptive_points for case in cases),
        },
        "dense_reference": {
            "logarithmic_q_points": args.reference_log_points,
            "linear_q_points": args.reference_linear_points,
            "q_minimum_positive": 1.0e-24,
            "q_split": 1.0e-2,
        },
        "csv": csv_path.name,
        "energy_csv": energy_csv_path.name,
        "quadrature_csv": quadrature_csv_path.name,
    }
    report_path = output / "nlh_adaptive_kernel_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {energy_csv_path}")
    print(f"Wrote {quadrature_csv_path}")
    print(f"Wrote {report_path}")
    print(
        f"Accepted: {accepted}; impact maxima: {impact_maxima}; "
        f"energy maxima: {energy_maxima}; quadrature maxima: {quadrature_maxima}"
    )
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    DEFAULT_MAX_ENERGY_POINTS,
