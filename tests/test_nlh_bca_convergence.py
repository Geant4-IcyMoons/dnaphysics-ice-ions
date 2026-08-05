from __future__ import annotations

import math
from pathlib import Path
import sys

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
)


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
