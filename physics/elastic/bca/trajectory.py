"""Periodic hard-collision trajectories through explicit ice snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from .runtime import AdaptiveKernelTable, KernelTableError
from .structure import IceStructure, StructureValidationError


Vector = NDArray[np.float64]
ImageKey = tuple[int, int, int, int]
_IMAGE_NEIGHBORS = np.asarray(
    tuple(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64
)


@dataclass(frozen=True)
class HardCollisionEvent:
    """One structure-selected hard binary collision."""

    collision_index: int
    target_atom_index: int
    target: str
    target_image: tuple[int, int, int]
    path_distance_angstrom: float
    collision_position_angstrom: tuple[float, float, float]
    target_position_angstrom: tuple[float, float, float]
    direction_in: tuple[float, float, float]
    direction_out: tuple[float, float, float]
    recoil_direction: tuple[float, float, float]
    projectile_energy_in_ev: float
    projectile_energy_out_ev: float
    recoil_energy_ev: float
    impact_parameter_angstrom: float
    maximum_impact_parameter_angstrom: float
    theta_cm_rad: float
    theta_projectile_lab_rad: float
    competing_hard_candidates: int


@dataclass(frozen=True)
class HardTrajectoryResult:
    """Primary-projectile trajectory and the recoils it emitted."""

    projectile: str
    initial_energy_ev: float
    final_energy_ev: float
    requested_path_length_angstrom: float
    traveled_path_length_angstrom: float
    initial_position_angstrom: tuple[float, float, float]
    final_position_angstrom: tuple[float, float, float]
    initial_direction: tuple[float, float, float]
    final_direction: tuple[float, float, float]
    termination: str
    events: tuple[HardCollisionEvent, ...]
    control_variate: "StraightLineControlVariate | None" = None
    importance_sampling: "CollisionTubeImportanceSample | None" = None

    @property
    def recoil_energy_ev(self) -> float:
        return float(sum(event.recoil_energy_ev for event in self.events))

    @property
    def ambiguous_event_count(self) -> int:
        return sum(event.competing_hard_candidates > 0 for event in self.events)


@dataclass(frozen=True)
class StraightLineControlVariate:
    """Unperturbed-line hard moments and their exact translation average."""

    collision_count: float
    recoil_energy_ev: float
    transport_moment: float
    expected_collision_count: float
    expected_recoil_energy_ev: float
    expected_transport_moment: float
    maximum_quadrature_relative_error: float


@dataclass(frozen=True)
class CollisionTubeImportanceSample:
    """Exact likelihood record for one cell-translation proposal.

    The target distribution is uniform translation through the periodic cell.
    The proposal is a declared mixture of that target and a collision-tube
    density.  ``target_over_proposal_weight`` is therefore the complete
    Radon--Nikodym derivative used by every rate and moment estimator; it is a
    numerical sampling weight, not a physical correction.
    """

    component: str
    tube_mixture_fraction: float
    tube_density_over_uniform: float
    target_over_proposal_weight: float
    selected_target: str | None
    selected_atom_index: int | None
    selected_area_quantile: float | None
    selected_stratum_index: int | None
    selected_stratum_count: int | None


@dataclass(frozen=True)
class _Candidate:
    atom_index: int
    target: str
    image: tuple[int, int, int]
    projection_angstrom: float
    impact_parameter_angstrom: float
    target_position_angstrom: Vector
    impact_to_target_angstrom: Vector
    maximum_impact_parameter_angstrom: float
    retained_half_span_angstrom: float

    @property
    def key(self) -> ImageKey:
        return (self.atom_index, *self.image)


def _unit_vector(value: ArrayLike, name: str) -> Vector:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values.")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} cannot be the zero vector.")
    return vector / norm


def _tuple3(value: Vector) -> tuple[float, float, float]:
    return float(value[0]), float(value[1]), float(value[2])


def _random_transverse(direction: Vector, rng: np.random.Generator) -> Vector:
    trial = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(np.dot(direction, trial))) > 0.8:
        trial = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    first = np.cross(direction, trial)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    phi = 2.0 * math.pi * float(rng.random())
    return math.cos(phi) * first + math.sin(phi) * second


def _transverse_basis(direction: Vector) -> tuple[Vector, Vector]:
    """Return a deterministic right-handed basis normal to ``direction``."""

    trial = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(np.dot(direction, trial))) > 0.8:
        trial = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    first = np.cross(direction, trial)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return first, second


class PeriodicHardCollisionTransport:
    """Sequence retained NLH collisions in an infinite periodic ice snapshot.

    The target coordinates remain static. Target recoils are returned as events
    for downstream transport; they are not inserted back into this lattice.
    Distant soft scattering and simultaneous many-atom forces are deliberately
    absent and are diagnosed rather than silently approximated.
    """

    def __init__(
        self,
        structure: IceStructure,
        kernels: AdaptiveKernelTable,
        *,
        search_window_angstrom: float = 4.0,
        allow_unvalidated_structure: bool = False,
    ) -> None:
        if not structure.collision_ready and not allow_unvalidated_structure:
            raise StructureValidationError(
                "Periodic collision transport requires an attested structure."
            )
        if (
            not math.isfinite(search_window_angstrom)
            or search_window_angstrom <= 0.0
        ):
            raise ValueError("search_window_angstrom must be finite and positive.")
        self.structure = structure
        self.kernels = kernels
        self.search_window_angstrom = float(search_window_angstrom)
        self._lattice = np.asarray(structure.lattice_angstrom, dtype=np.float64)
        self._inverse_lattice = np.linalg.inv(self._lattice)
        singular_values = np.linalg.svd(self._lattice, compute_uv=False)
        self._minimum_lattice_scale = float(np.min(singular_values))
        fractional = structure.positions_angstrom @ self._inverse_lattice
        fractional -= np.floor(fractional)
        self._fractional_positions = fractional
        self._tree = cKDTree(fractional, boxsize=1.0)
        self._moment_cache: dict[
            tuple[str, float], tuple[float, float, float, float]
        ] = {}

    def independent_atom_rate_per_angstrom(
        self, projectile: str, projectile_energy_ev: float
    ) -> float:
        """Return the uncorrelated H/O hard-collision reference rate."""

        rate = 0.0
        for target in ("H", "O"):
            atom_count = int(np.count_nonzero(self.structure.species == target))
            rate += (
                atom_count
                / self.structure.volume_angstrom3
                * self.kernels.hard_cross_section_angstrom2(
                    projectile, target, projectile_energy_ev
                )
            )
        return rate

    def independent_atom_moment_rates(
        self, projectile: str, projectile_energy_ev: float
    ) -> tuple[float, float, float, float]:
        """Return exact translation-averaged hard count/recoil/transport rates."""

        key = (projectile, float(projectile_energy_ev))
        cached = self._moment_cache.get(key)
        if cached is not None:
            return cached
        count_rate = 0.0
        recoil_rate = 0.0
        transport_rate = 0.0
        quadrature_error = 0.0
        for target in ("H", "O"):
            number_density = (
                int(np.count_nonzero(self.structure.species == target))
                / self.structure.volume_angstrom3
            )
            moments = self.kernels.hard_moment_cross_sections(
                projectile, target, projectile_energy_ev
            )
            count_rate += number_density * moments.cross_section_angstrom2
            recoil_rate += (
                number_density
                * moments.recoil_energy_cross_section_ev_angstrom2
            )
            transport_rate += (
                number_density * moments.transport_cross_section_angstrom2
            )
            quadrature_error = max(
                quadrature_error, moments.quadrature_relative_error
            )
        result = (
            count_rate,
            recoil_rate,
            transport_rate,
            quadrature_error,
        )
        self._moment_cache[key] = result
        return result

    def straight_line_control_variate(
        self,
        projectile: str,
        projectile_energy_ev: float,
        initial_position_angstrom: ArrayLike,
        initial_direction: ArrayLike,
        path_length_angstrom: float,
    ) -> StraightLineControlVariate:
        """Evaluate an unbiased difference-estimator control variate.

        The reference is the sum of retained binary encounters along the
        unperturbed initial line at fixed initial energy.  Uniform translation
        of that line through a periodic cell has the exact independent-atom
        expectation n_t*L*integral(2*pi*b db), irrespective of crystal order.
        Subtracting this sampled reference and adding its analytic expectation
        therefore changes variance, not the atomistic trajectory expectation.
        """

        if not math.isfinite(path_length_angstrom) or path_length_angstrom <= 0.0:
            raise ValueError("path_length_angstrom must be finite and positive.")
        position = np.asarray(initial_position_angstrom, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(
                "initial_position_angstrom must contain three finite values."
            )
        direction = _unit_vector(initial_direction, "initial_direction")
        candidates = self._straight_line_candidates(
            position,
            direction,
            path_length_angstrom,
            projectile,
            projectile_energy_ev,
        )
        collisions = [
            self.kernels.collide(
                projectile,
                candidate.target,
                projectile_energy_ev,
                candidate.impact_parameter_angstrom,
            )
            for candidate in candidates
        ]
        count_rate, recoil_rate, transport_rate, quadrature_error = (
            self.independent_atom_moment_rates(projectile, projectile_energy_ev)
        )
        return StraightLineControlVariate(
            collision_count=float(len(collisions)),
            recoil_energy_ev=math.fsum(
                collision.recoil_energy_ev for collision in collisions
            ),
            transport_moment=math.fsum(
                1.0 - math.cos(collision.theta_projectile_lab_rad)
                for collision in collisions
            ),
            expected_collision_count=count_rate * path_length_angstrom,
            expected_recoil_energy_ev=recoil_rate * path_length_angstrom,
            expected_transport_moment=transport_rate * path_length_angstrom,
            maximum_quadrature_relative_error=quadrature_error,
        )

    def _straight_line_candidates(
        self,
        initial_position_angstrom: Vector,
        initial_direction: Vector,
        path_length_angstrom: float,
        projectile: str,
        projectile_energy_ev: float,
    ) -> list[_Candidate]:
        """Return every retained encounter along one unperturbed finite line."""

        candidates: list[_Candidate] = []
        traveled = 0.0
        while traveled < path_length_angstrom:
            segment = min(
                self.search_window_angstrom,
                path_length_angstrom - traveled,
            )
            segment_position = (
                initial_position_angstrom + traveled * initial_direction
            )
            segment_candidates = self._candidates(
                segment_position,
                initial_direction,
                segment,
                projectile,
                projectile_energy_ev,
                None,
            )
            for candidate in segment_candidates:
                if candidate.projection_angstrom > segment:
                    continue
                candidates.append(
                    _Candidate(
                        atom_index=candidate.atom_index,
                        target=candidate.target,
                        image=candidate.image,
                        projection_angstrom=traveled
                        + candidate.projection_angstrom,
                        impact_parameter_angstrom=(
                            candidate.impact_parameter_angstrom
                        ),
                        target_position_angstrom=(
                            candidate.target_position_angstrom
                        ),
                        impact_to_target_angstrom=(
                            candidate.impact_to_target_angstrom
                        ),
                        maximum_impact_parameter_angstrom=(
                            candidate.maximum_impact_parameter_angstrom
                        ),
                        retained_half_span_angstrom=(
                            candidate.retained_half_span_angstrom
                        ),
                    )
                )
            traveled += segment
        candidates.sort(
            key=lambda item: (
                item.projection_angstrom,
                item.impact_parameter_angstrom,
                item.atom_index,
                item.image,
            )
        )
        return candidates

    def _area_quantile_density(
        self,
        projectile: str,
        target: str,
        projectile_energy_ev: float,
        area_quantile: float,
    ) -> float:
        """Density from equal allocation over the real adaptive-q intervals."""

        breakpoints = self.kernels.area_quantile_breakpoints(
            projectile, target, projectile_energy_ev
        )
        index = min(
            len(breakpoints) - 2,
            int(np.searchsorted(breakpoints, area_quantile, side="right")) - 1,
        )
        index = max(0, index)
        width = float(breakpoints[index + 1] - breakpoints[index])
        return 1.0 / ((len(breakpoints) - 1) * width)

    def _collision_tube_density_over_uniform(
        self,
        projectile: str,
        projectile_energy_ev: float,
        initial_position_angstrom: Vector,
        initial_direction: Vector,
        path_length_angstrom: float,
    ) -> float:
        """Evaluate the exact periodic collision-tube proposal ratio q(x)/p(x)."""

        candidates = self._straight_line_candidates(
            initial_position_angstrom,
            initial_direction,
            path_length_angstrom,
            projectile,
            projectile_energy_ev,
        )
        weighted_area = math.fsum(
            self._area_quantile_density(
                projectile,
                candidate.target,
                projectile_energy_ev,
                self.kernels.area_quantile_from_impact_parameter(
                    projectile,
                    candidate.target,
                    projectile_energy_ev,
                    candidate.impact_parameter_angstrom,
                ),
            )
            for candidate in candidates
        )
        cross_section_sum = math.fsum(
            int(np.count_nonzero(self.structure.species == target))
            * self.kernels.hard_cross_section_angstrom2(
                projectile, target, projectile_energy_ev
            )
            for target in ("H", "O")
        )
        if cross_section_sum <= 0.0:
            raise KernelTableError("Collision-tube sampling has zero support.")
        return (
            self.structure.volume_angstrom3
            * weighted_area
            / (path_length_angstrom * cross_section_sum)
        )

    def sample_collision_tube_mixture(
        self,
        projectile: str,
        projectile_energy_ev: float,
        initial_direction: ArrayLike,
        path_length_angstrom: float,
        tube_mixture_fraction: float,
        rng: np.random.Generator,
    ) -> tuple[Vector, CollisionTubeImportanceSample]:
        """Sample an exactly weighted uniform/collision-tube mixture.

        A tube draw first selects H or O in proportion to ``N_t sigma_t``,
        then one real target atom, one longitudinal collision location, and
        one adaptive impact-area interval.  The inverse map is the complete
        list of retained straight-line candidates.  Summing those preimages
        gives the exact proposal density even where tubes overlap.
        """

        if not 0.0 < tube_mixture_fraction < 1.0:
            raise ValueError("tube_mixture_fraction must lie strictly in (0, 1).")
        if not math.isfinite(path_length_angstrom) or path_length_angstrom <= 0.0:
            raise ValueError("path_length_angstrom must be finite and positive.")
        direction = _unit_vector(initial_direction, "initial_direction")
        use_tube = float(rng.random()) < tube_mixture_fraction
        selected_target: str | None = None
        selected_atom: int | None = None
        selected_quantile: float | None = None
        selected_stratum: int | None = None
        selected_stratum_count: int | None = None

        if not use_tube:
            fractional_position = rng.random(3)
            initial_position = fractional_position @ self._lattice
            component = "uniform"
        else:
            target_records: list[tuple[str, NDArray[np.int64], float]] = []
            total_area = 0.0
            for target in ("H", "O"):
                indices = np.flatnonzero(self.structure.species == target)
                cross_section = self.kernels.hard_cross_section_angstrom2(
                    projectile, target, projectile_energy_ev
                )
                area = float(indices.size) * cross_section
                if area > 0.0:
                    target_records.append((target, indices, area))
                    total_area += area
            if total_area <= 0.0:
                raise KernelTableError("Collision-tube sampling has zero support.")
            draw = float(rng.random()) * total_area
            cumulative = 0.0
            target = target_records[-1][0]
            indices = target_records[-1][1]
            for candidate_target, candidate_indices, area in target_records:
                cumulative += area
                if draw < cumulative:
                    target = candidate_target
                    indices = candidate_indices
                    break
            atom_index = int(indices[int(rng.integers(indices.size))])
            breakpoints = self.kernels.area_quantile_breakpoints(
                projectile, target, projectile_energy_ev
            )
            stratum_count = len(breakpoints) - 1
            stratum = int(rng.integers(stratum_count))
            lower = float(breakpoints[stratum])
            upper = float(breakpoints[stratum + 1])
            quantile = lower + (upper - lower) * float(rng.random())
            maximum_impact = self.kernels.maximum_impact_parameter_angstrom(
                projectile, target, projectile_energy_ev
            )
            impact = self.kernels.impact_parameter_from_area_quantile(
                projectile, target, projectile_energy_ev, quantile
            )
            phi = 2.0 * math.pi * float(rng.random())
            first, second = _transverse_basis(direction)
            transverse = impact * (math.cos(phi) * first + math.sin(phi) * second)
            longitudinal = path_length_angstrom * float(rng.random())
            unwrapped = (
                self.structure.positions_angstrom[atom_index]
                - longitudinal * direction
                + transverse
            )
            fractional_position = unwrapped @ self._inverse_lattice
            fractional_position -= np.floor(fractional_position)
            initial_position = fractional_position @ self._lattice
            component = "collision_tube"
            selected_target = target
            selected_atom = atom_index
            selected_quantile = quantile
            selected_stratum = stratum
            selected_stratum_count = stratum_count

        tube_ratio = self._collision_tube_density_over_uniform(
            projectile,
            projectile_energy_ev,
            initial_position,
            direction,
            path_length_angstrom,
        )
        proposal_over_target = (
            1.0
            - tube_mixture_fraction
            + tube_mixture_fraction * tube_ratio
        )
        if not math.isfinite(proposal_over_target) or proposal_over_target <= 0.0:
            raise RuntimeError("Invalid collision-tube mixture density.")
        record = CollisionTubeImportanceSample(
            component=component,
            tube_mixture_fraction=tube_mixture_fraction,
            tube_density_over_uniform=tube_ratio,
            target_over_proposal_weight=1.0 / proposal_over_target,
            selected_target=selected_target,
            selected_atom_index=selected_atom,
            selected_area_quantile=selected_quantile,
            selected_stratum_index=selected_stratum,
            selected_stratum_count=selected_stratum_count,
        )
        return initial_position, record

    def _energy_bounds(self, projectile: str) -> tuple[float, float]:
        bounds = [
            self.kernels.energy_bounds_ev(projectile, target)
            for target in ("H", "O")
        ]
        return max(value[0] for value in bounds), min(value[1] for value in bounds)

    def _maximum_impacts(
        self, projectile: str, energy_ev: float
    ) -> dict[str, float]:
        return {
            target: self.kernels.maximum_impact_parameter_angstrom(
                projectile, target, energy_ev
            )
            for target in ("H", "O")
        }

    def _candidates(
        self,
        position: Vector,
        direction: Vector,
        segment_length: float,
        projectile: str,
        energy_ev: float,
        excluded_image: ImageKey | None,
    ) -> list[_Candidate]:
        maximum_impacts = self._maximum_impacts(projectile, energy_ev)
        minimum_impacts = {
            target: self.kernels.minimum_impact_parameter_angstrom(
                projectile, target, energy_ev
            )
            for target in ("H", "O")
        }
        threshold_radii = {
            target: self.kernels.turning_threshold_radius_angstrom(
                projectile, target
            )
            for target in ("H", "O")
        }
        maximum_radius = max(maximum_impacts.values())
        if maximum_radius <= 0.0:
            return []
        query_length = segment_length + 2.0 * max(threshold_radii.values())
        midpoint = position + 0.5 * query_length * direction
        midpoint_fractional = midpoint @ self._inverse_lattice
        wrapped_midpoint = midpoint_fractional - np.floor(midpoint_fractional)
        cartesian_bound = math.hypot(0.5 * query_length, maximum_radius)
        fractional_bound = cartesian_bound / self._minimum_lattice_scale
        atom_indices = self._tree.query_ball_point(
            wrapped_midpoint, fractional_bound
        )
        candidates: list[_Candidate] = []
        tolerance = 1.0e-10 * max(1.0, query_length)
        for atom_index in atom_indices:
            target = str(self.structure.species[atom_index])
            maximum_impact = maximum_impacts[target]
            minimum_impact = minimum_impacts[target]
            if maximum_impact <= 0.0:
                continue
            base_fractional = self._fractional_positions[atom_index]
            central_image = np.rint(
                midpoint_fractional - base_fractional
            ).astype(np.int64)
            for image_array in central_image + _IMAGE_NEIGHBORS:
                image = tuple(int(value) for value in image_array)
                key: ImageKey = (atom_index, *image)
                if key == excluded_image:
                    continue
                atom_position = (base_fractional + image_array) @ self._lattice
                displacement = atom_position - position
                projection = float(np.dot(displacement, direction))
                if projection <= tolerance or projection > query_length + tolerance:
                    continue
                closest = position + projection * direction
                impact_to_target = atom_position - closest
                impact = float(np.linalg.norm(impact_to_target))
                impact_tolerance = 1.0e-12 * max(1.0, maximum_impact)
                if (
                    impact < minimum_impact - impact_tolerance
                    or impact > maximum_impact + impact_tolerance
                ):
                    continue
                candidates.append(
                    _Candidate(
                        atom_index=atom_index,
                        target=target,
                        image=image,
                        projection_angstrom=projection,
                        impact_parameter_angstrom=min(impact, maximum_impact),
                        target_position_angstrom=atom_position,
                        impact_to_target_angstrom=impact_to_target,
                        maximum_impact_parameter_angstrom=maximum_impact,
                        retained_half_span_angstrom=math.sqrt(
                            max(0.0, threshold_radii[target] ** 2 - impact**2)
                        ),
                    )
                )
        candidates.sort(
            key=lambda item: (
                item.projection_angstrom,
                item.impact_parameter_angstrom,
                item.atom_index,
                item.image,
            )
        )
        return candidates

    @staticmethod
    def _competing_candidates(
        selected: _Candidate, candidates: list[_Candidate]
    ) -> int:
        count = 0
        for candidate in candidates[1:]:
            longitudinal_gap = abs(
                candidate.projection_angstrom - selected.projection_angstrom
            )
            if longitudinal_gap <= (
                selected.retained_half_span_angstrom
                + candidate.retained_half_span_angstrom
            ):
                count += 1
        return count

    def _recoil_direction(
        self,
        projectile: str,
        target: str,
        energy_in_ev: float,
        energy_out_ev: float,
        direction_in: Vector,
        direction_out: Vector,
    ) -> Vector:
        context = self.kernels.pair_kinematics(projectile, target, energy_in_ev)
        mass = context.projectile_mass_c2_ev
        momentum_in = math.sqrt(energy_in_ev * (energy_in_ev + 2.0 * mass))
        momentum_out = math.sqrt(
            max(0.0, energy_out_ev * (energy_out_ev + 2.0 * mass))
        )
        recoil_momentum = momentum_in * direction_in - momentum_out * direction_out
        norm = float(np.linalg.norm(recoil_momentum))
        if norm <= 0.0:
            return np.zeros(3, dtype=np.float64)
        return recoil_momentum / norm

    def trace(
        self,
        projectile: str,
        projectile_energy_ev: float,
        initial_position_angstrom: ArrayLike,
        initial_direction: ArrayLike,
        path_length_angstrom: float,
        *,
        rng: np.random.Generator | None = None,
        max_collisions: int = 10_000,
    ) -> HardTrajectoryResult:
        if not math.isfinite(path_length_angstrom) or path_length_angstrom <= 0.0:
            raise ValueError("path_length_angstrom must be finite and positive.")
        if max_collisions < 1:
            raise ValueError("max_collisions must be positive.")
        random = rng or np.random.default_rng()
        position = np.asarray(initial_position_angstrom, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("initial_position_angstrom must contain three finite values.")
        position = position.copy()
        direction = _unit_vector(initial_direction, "initial_direction")
        initial_position = position.copy()
        initial_unit_direction = direction.copy()
        initial_energy = float(projectile_energy_ev)
        energy = initial_energy
        minimum_energy, maximum_energy = self._energy_bounds(projectile)
        if (
            not math.isfinite(energy)
            or energy < minimum_energy
            or energy > maximum_energy
        ):
            raise KernelTableError(
                f"Projectile energy must lie in {minimum_energy:g}-{maximum_energy:g} eV."
            )

        traveled = 0.0
        events: list[HardCollisionEvent] = []
        excluded_image: ImageKey | None = None
        termination = "path_complete"
        while traveled < path_length_angstrom:
            if len(events) >= max_collisions:
                termination = "maximum_collisions"
                break
            if energy < minimum_energy:
                termination = "energy_below_kernel_table"
                break
            remaining = path_length_angstrom - traveled
            segment = min(self.search_window_angstrom, remaining)
            candidates = self._candidates(
                position,
                direction,
                segment,
                projectile,
                energy,
                excluded_image,
            )
            if not candidates or candidates[0].projection_angstrom > segment:
                position = position + segment * direction
                traveled += segment
                excluded_image = None
                continue

            selected = candidates[0]
            distance = min(selected.projection_angstrom, remaining)
            collision_position = position + distance * direction
            traveled += distance
            collision = self.kernels.collide(
                projectile,
                selected.target,
                energy,
                selected.impact_parameter_angstrom,
            )
            direction_in = direction.copy()
            if selected.impact_parameter_angstrom > 1.0e-12:
                transverse = -selected.impact_to_target_angstrom
                transverse /= np.linalg.norm(transverse)
            else:
                transverse = _random_transverse(direction_in, random)
            theta = collision.theta_projectile_lab_rad
            direction = (
                math.cos(theta) * direction_in + math.sin(theta) * transverse
            )
            direction /= np.linalg.norm(direction)
            recoil_direction = self._recoil_direction(
                projectile,
                selected.target,
                energy,
                collision.projectile_out_energy_ev,
                direction_in,
                direction,
            )
            events.append(
                HardCollisionEvent(
                    collision_index=len(events),
                    target_atom_index=selected.atom_index,
                    target=selected.target,
                    target_image=selected.image,
                    path_distance_angstrom=traveled,
                    collision_position_angstrom=_tuple3(collision_position),
                    target_position_angstrom=_tuple3(
                        selected.target_position_angstrom
                    ),
                    direction_in=_tuple3(direction_in),
                    direction_out=_tuple3(direction),
                    recoil_direction=_tuple3(recoil_direction),
                    projectile_energy_in_ev=energy,
                    projectile_energy_out_ev=collision.projectile_out_energy_ev,
                    recoil_energy_ev=collision.recoil_energy_ev,
                    impact_parameter_angstrom=collision.impact_parameter_angstrom,
                    maximum_impact_parameter_angstrom=(
                        collision.maximum_impact_parameter_angstrom
                    ),
                    theta_cm_rad=collision.theta_cm_rad,
                    theta_projectile_lab_rad=collision.theta_projectile_lab_rad,
                    competing_hard_candidates=self._competing_candidates(
                        selected, candidates
                    ),
                )
            )
            energy = collision.projectile_out_energy_ev
            position = collision_position
            excluded_image = selected.key

        return HardTrajectoryResult(
            projectile=projectile,
            initial_energy_ev=initial_energy,
            final_energy_ev=energy,
            requested_path_length_angstrom=float(path_length_angstrom),
            traveled_path_length_angstrom=traveled,
            initial_position_angstrom=_tuple3(initial_position),
            final_position_angstrom=_tuple3(position),
            initial_direction=_tuple3(initial_unit_direction),
            final_direction=_tuple3(direction),
            termination=termination,
            events=tuple(events),
        )
