"""Sequential Monte Carlo convergence for independent ion trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

from scipy.stats import t as student_t


DEFAULT_STATISTICAL_RELATIVE_TOLERANCE = 0.005
DEFAULT_STATISTICAL_CONFIDENCE = 0.95
DEFAULT_MINIMUM_TRAJECTORIES = 1_000
DEFAULT_MAXIMUM_TRAJECTORIES = 1_024_000
DEFAULT_TRAJECTORY_BATCH_SIZE = 1_000

OBSERVABLES = (
    "hard_collision_rate_per_angstrom",
    "hard_nuclear_stopping_ev_per_angstrom",
    "hard_transport_rate_per_angstrom",
    "mean_recoil_energy_ev_per_collision",
    "mean_one_minus_cosine_per_collision",
)


@dataclass
class RatioStatistics:
    """Sufficient statistics for a ratio of trajectory-level expectations."""

    count: int = 0
    numerator_sum: float = 0.0
    denominator_sum: float = 0.0
    numerator_square_sum: float = 0.0
    denominator_square_sum: float = 0.0
    numerator_denominator_sum: float = 0.0

    def add(self, numerator: float, denominator: float) -> None:
        if not math.isfinite(numerator) or not math.isfinite(denominator):
            raise ValueError("Ratio observations must be finite.")
        # Difference-estimator control variates can make an individual
        # corrected denominator negative while preserving a positive expected
        # denominator. The ratio delta method only requires the accumulated
        # denominator to be positive at assessment time.
        self.count += 1
        self.numerator_sum += numerator
        self.denominator_sum += denominator
        self.numerator_square_sum += numerator * numerator
        self.denominator_square_sum += denominator * denominator
        self.numerator_denominator_sum += numerator * denominator

    def merge(self, other: "RatioStatistics") -> None:
        self.count += other.count
        self.numerator_sum += other.numerator_sum
        self.denominator_sum += other.denominator_sum
        self.numerator_square_sum += other.numerator_square_sum
        self.denominator_square_sum += other.denominator_square_sum
        self.numerator_denominator_sum += other.numerator_denominator_sum

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "numerator_sum": self.numerator_sum,
            "denominator_sum": self.denominator_sum,
            "numerator_square_sum": self.numerator_square_sum,
            "denominator_square_sum": self.denominator_square_sum,
            "numerator_denominator_sum": self.numerator_denominator_sum,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RatioStatistics":
        return cls(
            count=int(payload["count"]),
            numerator_sum=float(payload["numerator_sum"]),
            denominator_sum=float(payload["denominator_sum"]),
            numerator_square_sum=float(payload["numerator_square_sum"]),
            denominator_square_sum=float(payload["denominator_square_sum"]),
            numerator_denominator_sum=float(
                payload["numerator_denominator_sum"]
            ),
        )

    def interval(self, critical_value: float) -> dict[str, float | bool | None]:
        """Return a trajectory-clustered delta-method confidence interval."""

        if self.count < 2 or self.denominator_sum <= 0.0:
            return {
                "estimate": None,
                "standard_error": None,
                "confidence_half_width": None,
                "relative_confidence_half_width": None,
                "finite": False,
            }
        estimate = self.numerator_sum / self.denominator_sum
        residual_square_sum = (
            self.numerator_square_sum
            - 2.0 * estimate * self.numerator_denominator_sum
            + estimate * estimate * self.denominator_square_sum
        )
        numerical_scale = max(
            self.numerator_square_sum,
            estimate * estimate * self.denominator_square_sum,
            1.0,
        )
        if residual_square_sum < -1.0e-12 * numerical_scale:
            raise ArithmeticError("Negative trajectory residual variance.")
        residual_square_sum = max(0.0, residual_square_sum)
        standard_error = math.sqrt(
            self.count * residual_square_sum / (self.count - 1)
        ) / self.denominator_sum
        half_width = critical_value * standard_error
        relative = half_width / abs(estimate) if estimate != 0.0 else None
        return {
            "estimate": estimate,
            "standard_error": standard_error,
            "confidence_half_width": half_width,
            "relative_confidence_half_width": relative,
            "finite": relative is not None and math.isfinite(relative),
        }


def doubling_schedule(
    minimum_trajectories: int,
    maximum_trajectories: int,
    batch_size: int,
) -> tuple[int, ...]:
    """Return bounded, batch-aligned trajectory counts for sequential checks."""

    if minimum_trajectories < 2:
        raise ValueError("minimum_trajectories must be at least two.")
    if maximum_trajectories < minimum_trajectories:
        raise ValueError(
            "maximum_trajectories cannot be below minimum_trajectories."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    first = min(
        maximum_trajectories,
        math.ceil(minimum_trajectories / batch_size) * batch_size,
    )
    schedule = [first]
    while schedule[-1] < maximum_trajectories:
        doubled = math.ceil((2 * schedule[-1]) / batch_size) * batch_size
        schedule.append(min(maximum_trajectories, doubled))
    return tuple(schedule)


def simultaneous_critical_value(
    trajectory_count: int,
    confidence: float,
    observable_count: int,
    scheduled_look_count: int,
) -> tuple[float, float]:
    """Student-t critical value with a family-wise Bonferroni correction."""

    if trajectory_count < 2:
        raise ValueError("At least two trajectories are required.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")
    if observable_count < 1 or scheduled_look_count < 1:
        raise ValueError("Observable and scheduled-look counts must be positive.")
    individual_alpha = (1.0 - confidence) / (
        observable_count * scheduled_look_count
    )
    critical = float(
        student_t.ppf(1.0 - 0.5 * individual_alpha, trajectory_count - 1)
    )
    return critical, 1.0 - individual_alpha


def simultaneous_dkw_half_width(
    trajectory_count: int,
    confidence: float,
    distribution_count: int,
    scheduled_look_count: int,
) -> tuple[float, float]:
    """Distribution-free CDF band across distributions and sequential looks."""

    if trajectory_count < 1:
        raise ValueError("At least one trajectory is required.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")
    if distribution_count < 1 or scheduled_look_count < 1:
        raise ValueError(
            "Distribution and scheduled-look counts must be positive."
        )
    individual_alpha = (1.0 - confidence) / (
        distribution_count * scheduled_look_count
    )
    half_width = math.sqrt(
        math.log(2.0 / individual_alpha) / (2.0 * trajectory_count)
    )
    return half_width, 1.0 - individual_alpha


def convergence_report(
    statistics: Mapping[str, RatioStatistics],
    *,
    tolerance: float,
    confidence: float,
    scheduled_look_count: int,
    look_index: int,
) -> dict[str, object]:
    """Assess all registered observables at one pre-scheduled sample size."""

    if set(statistics) != set(OBSERVABLES):
        raise ValueError(f"Statistics must contain exactly {OBSERVABLES}.")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    counts = {value.count for value in statistics.values()}
    if len(counts) != 1:
        raise ValueError("All observables must contain the same trajectories.")
    trajectory_count = counts.pop()
    if not 1 <= look_index <= scheduled_look_count:
        raise ValueError("look_index is outside the scheduled looks.")
    critical, individual_confidence = simultaneous_critical_value(
        trajectory_count,
        confidence,
        len(OBSERVABLES),
        scheduled_look_count,
    )
    observable_reports: dict[str, dict[str, object]] = {}
    for name in OBSERVABLES:
        interval = statistics[name].interval(critical)
        interval["passes"] = bool(
            interval["finite"]
            and interval["relative_confidence_half_width"] is not None
            and float(interval["relative_confidence_half_width"]) <= tolerance
        )
        observable_reports[name] = interval
    converged = all(
        bool(report["passes"]) for report in observable_reports.values()
    )
    finite_widths = [
        float(report["relative_confidence_half_width"])
        for report in observable_reports.values()
        if bool(report["finite"])
        and report["relative_confidence_half_width"] is not None
    ]
    return {
        "converged": converged,
        "trajectory_count": trajectory_count,
        "look_index": look_index,
        "scheduled_look_count": scheduled_look_count,
        "familywise_confidence": confidence,
        "individual_interval_confidence": individual_confidence,
        "critical_value": critical,
        "relative_tolerance": tolerance,
        "maximum_finite_relative_confidence_half_width": (
            max(finite_widths) if finite_widths else None
        ),
        "observables": observable_reports,
        "method": (
            "trajectory-clustered ratio delta method with Student-t intervals; "
            "Bonferroni correction across observables and scheduled looks"
        ),
    }


def merge_statistics(
    batches: Iterable[Mapping[str, RatioStatistics]],
) -> dict[str, RatioStatistics]:
    merged = {name: RatioStatistics() for name in OBSERVABLES}
    for batch in batches:
        if set(batch) != set(OBSERVABLES):
            raise ValueError(f"Batch statistics must contain exactly {OBSERVABLES}.")
        for name in OBSERVABLES:
            merged[name].merge(batch[name])
    return merged
