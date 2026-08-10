from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


NEP_MBPOL = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "nep_mbpol"
)
if str(NEP_MBPOL) not in sys.path:
    sys.path.insert(0, str(NEP_MBPOL))

from bca.convergence import (  # noqa: E402
    OBSERVABLES,
    RatioStatistics,
    convergence_report,
    doubling_schedule,
    merge_statistics,
    simultaneous_critical_value,
    simultaneous_dkw_half_width,
)
import simulate_nlh_hard_collisions as simulator  # noqa: E402
from bca.trajectory import (  # noqa: E402
    CollisionTubeImportanceSample,
    HardTrajectoryResult,
    StraightLineControlVariate,
)


def test_parallel_execution_bounds_pending_futures(monkeypatch) -> None:
    monkeypatch.setattr(simulator, "_run_one", lambda task: (task, task))
    real_wait = simulator.wait
    pending_sizes: list[int] = []

    def recording_wait(pending, **kwargs):
        pending_sizes.append(len(pending))
        return real_wait(pending, **kwargs)

    monkeypatch.setattr(simulator, "wait", recording_wait)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = simulator._execute_tasks(
            list(range(100)), executor, worker_count=2
        )

    assert [index for index, _ in results] == list(range(100))
    assert pending_sizes
    assert max(pending_sizes) <= 8


def _statistics(scale: float = 0.001) -> dict[str, RatioStatistics]:
    result = {name: RatioStatistics() for name in OBSERVABLES}
    for index in range(2_000):
        path = 100.0
        collision_count = 10.0 + scale * ((index % 5) - 2)
        recoil = 25.0 + scale * ((index % 7) - 3)
        transport = 0.2 + scale * 0.01 * ((index % 3) - 1)
        result["hard_collision_rate_per_angstrom"].add(
            collision_count, path
        )
        result["hard_nuclear_stopping_ev_per_angstrom"].add(recoil, path)
        result["hard_transport_rate_per_angstrom"].add(transport, path)
        result["mean_recoil_energy_ev_per_collision"].add(
            recoil, collision_count
        )
        result["mean_one_minus_cosine_per_collision"].add(
            transport, collision_count
        )
    return result


@pytest.mark.parametrize("projectile", ("H", "He"))
def test_trajectory_cli_accepts_light_projectiles(monkeypatch, projectile) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simulate_nlh_hard_collisions.py",
            "ice.xyz",
            "--projectile",
            projectile,
            "--energy-ev",
            "1000",
        ],
    )
    assert simulator.parse_args().projectile == projectile


def test_run_signature_rejects_obsolete_numerical_checkpoints() -> None:
    args = SimpleNamespace(
        projectile="C",
        energy_ev=1.0e5,
        path_length_angstrom=100.0,
        isotropic_directions=False,
        seed=1000,
        search_window_angstrom=4.0,
        max_collisions=10_000,
    )
    structure = SimpleNamespace(source_sha256="structure", frame_index=0)
    kernels = SimpleNamespace(csv_sha256="kernels")

    _, configuration = simulator._run_signature(
        args, structure, kernels, (0.0, 0.0, 1.0)
    )

    assert configuration["trajectory_implementation_version"] == (
        simulator.TRAJECTORY_IMPLEMENTATION_VERSION
    )


def test_run_signature_records_importance_proposal() -> None:
    args = SimpleNamespace(
        projectile="C",
        energy_ev=1.0e8,
        path_length_angstrom=100.0,
        isotropic_directions=False,
        seed=1000,
        search_window_angstrom=4.0,
        max_collisions=10_000,
        control_variate=True,
        sampling_mode="collision_tube_mixture",
        tube_mixture_fraction=0.5,
        output_detail="summary",
    )
    structure = SimpleNamespace(source_sha256="structure", frame_index=0)
    kernels = SimpleNamespace(csv_sha256="kernels")

    _, configuration = simulator._run_signature(
        args, structure, kernels, (0.0, 0.0, 1.0)
    )

    assert configuration["initial_condition_sampling"] == (
        "collision_tube_mixture"
    )
    assert configuration["tube_mixture_fraction"] == 0.5


def test_importance_weighted_control_variate_statistics_are_exact() -> None:
    importance = CollisionTubeImportanceSample(
        component="collision_tube",
        tube_mixture_fraction=0.5,
        tube_density_over_uniform=6.0,
        target_over_proposal_weight=0.25,
        selected_target="O",
        selected_atom_index=0,
        selected_area_quantile=0.1,
        selected_stratum_index=0,
        selected_stratum_count=3,
    )
    reference = StraightLineControlVariate(
        collision_count=1.0,
        recoil_energy_ev=2.0,
        transport_moment=0.1,
        expected_collision_count=4.0,
        expected_recoil_energy_ev=10.0,
        expected_transport_moment=2.0,
        maximum_quadrature_relative_error=1.0e-5,
    )
    events = (
        SimpleNamespace(recoil_energy_ev=3.0, theta_projectile_lab_rad=math.acos(0.8)),
        SimpleNamespace(recoil_energy_ev=5.0, theta_projectile_lab_rad=math.acos(0.5)),
    )
    result = HardTrajectoryResult(
        projectile="C",
        initial_energy_ev=1.0e8,
        final_energy_ev=1.0e8 - 8.0,
        requested_path_length_angstrom=100.0,
        traveled_path_length_angstrom=100.0,
        initial_position_angstrom=(0.0, 0.0, 0.0),
        final_position_angstrom=(0.0, 0.0, 100.0),
        initial_direction=(0.0, 0.0, 1.0),
        final_direction=(0.0, 0.0, 1.0),
        termination="path_complete",
        events=events,
        control_variate=reference,
        importance_sampling=importance,
    )

    statistics = simulator._batch_statistics(
        [(0, result)], control_variate=True
    )

    assert statistics["hard_collision_rate_per_angstrom"].numerator_sum == (
        pytest.approx(4.25)
    )
    assert statistics["hard_nuclear_stopping_ev_per_angstrom"].numerator_sum == (
        pytest.approx(11.5)
    )
    assert statistics["hard_transport_rate_per_angstrom"].numerator_sum == (
        pytest.approx(2.15)
    )
    assert statistics["hard_collision_rate_per_angstrom"].denominator_sum == (
        pytest.approx(25.0)
    )


def test_doubling_schedule_is_bounded_and_batch_aligned() -> None:
    assert doubling_schedule(1_001, 10_500, 1_000) == (
        2_000,
        4_000,
        8_000,
        10_500,
    )


def test_ratio_standard_error_uses_trajectory_clusters() -> None:
    numerators = np.asarray((1.0, 3.0, 2.0, 6.0, 5.0))
    denominators = np.asarray((2.0, 4.0, 2.5, 5.0, 4.5))
    statistics = RatioStatistics()
    for numerator, denominator in zip(numerators, denominators, strict=True):
        statistics.add(float(numerator), float(denominator))
    report = statistics.interval(critical_value=1.0)
    estimate = numerators.sum() / denominators.sum()
    residuals = numerators - estimate * denominators
    expected_standard_error = math.sqrt(
        len(numerators) * np.sum(residuals**2) / (len(numerators) - 1)
    ) / denominators.sum()
    assert report["estimate"] == pytest.approx(estimate)
    assert report["standard_error"] == pytest.approx(expected_standard_error)


def test_confidence_budget_covers_observables_and_scheduled_looks() -> None:
    critical, individual_confidence = simultaneous_critical_value(
        1_000, 0.95, len(OBSERVABLES), 11
    )
    assert critical > 1.96
    individual_alpha = 1.0 - individual_confidence
    assert individual_alpha * len(OBSERVABLES) * 11 == pytest.approx(0.05)


def test_dkw_budget_covers_two_distributions_and_scheduled_looks() -> None:
    width, individual_confidence = simultaneous_dkw_half_width(
        200_000, 0.95, 2, 11
    )
    assert width < 0.005
    individual_alpha = 1.0 - individual_confidence
    assert individual_alpha * 2 * 11 == pytest.approx(0.05)


def test_convergence_requires_every_rate_and_moment() -> None:
    passing = convergence_report(
        _statistics(scale=0.001),
        tolerance=0.005,
        confidence=0.95,
        scheduled_look_count=4,
        look_index=1,
    )
    assert passing["converged"]
    failing = convergence_report(
        _statistics(scale=2.0),
        tolerance=0.005,
        confidence=0.95,
        scheduled_look_count=4,
        look_index=1,
    )
    assert not failing["converged"]
    assert any(
        not observable["passes"]
        for observable in failing["observables"].values()
    )


def test_statistics_round_trip_and_merge() -> None:
    first = _statistics(scale=0.1)
    restored = {
        name: RatioStatistics.from_dict(value.to_dict())
        for name, value in first.items()
    }
    merged = merge_statistics((first, restored))
    for name in OBSERVABLES:
        assert merged[name].count == 4_000
        assert merged[name].numerator_sum == pytest.approx(
            2.0 * first[name].numerator_sum
        )


def test_zero_event_sample_is_reported_as_unresolved_not_infinite() -> None:
    statistics = {name: RatioStatistics() for name in OBSERVABLES}
    for _ in range(10):
        statistics["hard_collision_rate_per_angstrom"].add(0.0, 100.0)
        statistics["hard_nuclear_stopping_ev_per_angstrom"].add(0.0, 100.0)
        statistics["hard_transport_rate_per_angstrom"].add(0.0, 100.0)
        statistics["mean_recoil_energy_ev_per_collision"].add(0.0, 0.0)
        statistics["mean_one_minus_cosine_per_collision"].add(0.0, 0.0)
    report = convergence_report(
        statistics,
        tolerance=0.005,
        confidence=0.95,
        scheduled_look_count=1,
        look_index=1,
    )
    assert not report["converged"]
    assert report["observables"][
        "hard_collision_rate_per_angstrom"
    ]["relative_confidence_half_width"] is None
    assert report["observables"][
        "mean_recoil_energy_ev_per_collision"
    ]["relative_confidence_half_width"] is None
