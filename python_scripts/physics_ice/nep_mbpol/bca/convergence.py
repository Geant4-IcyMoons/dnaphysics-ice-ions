"""Sequential Monte Carlo convergence for independent ion trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

from scipy.stats import t as student_t


DEFAULT_STATISTICAL_CONFIDENCE = 0.95
DEFAULT_MEANINGFUL_SIGNIFICANT_DIGITS = 2
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

OBSERVABLE_UNITS = {
    "hard_collision_rate_per_angstrom": "angstrom^-1",
    "hard_nuclear_stopping_ev_per_angstrom": "eV angstrom^-1",
    "hard_transport_rate_per_angstrom": "angstrom^-1",
    "mean_recoil_energy_ev_per_collision": "eV collision^-1",
    "mean_one_minus_cosine_per_collision": "collision^-1",
}


def numerical_tolerance(value: float, significant_digits: int) -> float:
    """Return the JCGM 101:2008 section 7.9.2 decimal tolerance for ``value``.

    The caller, rather than JCGM, is responsible for choosing how many digits
    the numerical result must retain.
    """

    if not math.isfinite(value) or value == 0.0:
        raise ValueError("A finite nonzero calibration value is required.")
    if significant_digits < 1:
        raise ValueError("significant_digits must be positive.")
    leading_exponent = math.floor(math.log10(abs(value)))
    significand = abs(value) / 10.0**leading_exponent
    rounded_significand = round(significand, significant_digits - 1)
    if rounded_significand >= 10.0:
        leading_exponent += 1
    last_place_exponent = leading_exponent - significant_digits + 1
    return 0.5 * 10.0**last_place_exponent


def absolute_tolerances_from_estimates(
    estimates: Mapping[str, float], significant_digits: int
) -> dict[str, float]:
    """Freeze one absolute width per observable from calibration estimates."""

    if set(estimates) != set(OBSERVABLES):
        raise ValueError(f"Estimates must contain exactly {OBSERVABLES}.")
    return {
        name: numerical_tolerance(float(estimates[name]), significant_digits)
        for name in OBSERVABLES
    }


def validate_absolute_tolerances(
    tolerances: Mapping[str, float],
) -> dict[str, float]:
    """Validate and normalize a complete absolute fixed-width contract."""

    if set(tolerances) != set(OBSERVABLES):
        raise ValueError(f"Absolute tolerances must contain exactly {OBSERVABLES}.")
    normalized = {name: float(tolerances[name]) for name in OBSERVABLES}
    if any(not math.isfinite(value) or value <= 0.0 for value in normalized.values()):
        raise ValueError("Absolute tolerances must be finite and positive.")
    return normalized


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
                "confidence_width": None,
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
        width = 2.0 * half_width
        finite = all(
            math.isfinite(value)
            for value in (estimate, standard_error, half_width, width)
        ) and not (estimate == 0.0 and standard_error == 0.0)
        return {
            "estimate": estimate,
            "standard_error": standard_error,
            "confidence_half_width": half_width,
            "confidence_width": width,
            "finite": finite,
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
    absolute_tolerances: Mapping[str, float],
    confidence: float,
    scheduled_look_count: int,
    look_index: int,
) -> dict[str, object]:
    """Apply a simultaneous absolute fixed-width rule at one scheduled look."""

    if set(statistics) != set(OBSERVABLES):
        raise ValueError(f"Statistics must contain exactly {OBSERVABLES}.")
    tolerances = validate_absolute_tolerances(absolute_tolerances)
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
        width = interval["confidence_width"]
        interval["absolute_tolerance"] = tolerances[name]
        interval["unit"] = OBSERVABLE_UNITS[name]
        interval["normalized_confidence_width"] = (
            float(width) / tolerances[name] if width is not None else None
        )
        interval["passes"] = bool(
            interval["finite"]
            and width is not None
            and float(width) <= tolerances[name]
        )
        observable_reports[name] = interval
    converged = all(
        bool(report["passes"]) for report in observable_reports.values()
    )
    normalized_widths = [
        float(report["normalized_confidence_width"])
        for report in observable_reports.values()
        if bool(report["finite"])
        and report["normalized_confidence_width"] is not None
    ]
    return {
        "converged": converged,
        "trajectory_count": trajectory_count,
        "look_index": look_index,
        "scheduled_look_count": scheduled_look_count,
        "familywise_confidence": confidence,
        "individual_interval_confidence": individual_confidence,
        "critical_value": critical,
        "absolute_tolerances": tolerances,
        "maximum_normalized_confidence_width": (
            max(normalized_widths) if normalized_widths else None
        ),
        "observables": observable_reports,
        "method": (
            "trajectory-clustered ratio delta method with simultaneous "
            "Student-t confidence intervals; absolute fixed-width stopping; "
            "Bonferroni correction across observables and scheduled looks"
        ),
        "references": (
            "Glynn and Whitt (1992), doi:10.1214/aoap/1177005770; "
            "Flegal and Gong (2015), doi:10.5705/ss.2013.209"
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
