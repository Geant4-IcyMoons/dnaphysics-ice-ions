"""Full-domain ZBL kernel for atomistic periodic-ice trajectories."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from functools import lru_cache
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad_vec

NEP_MBPOL_ROOT = Path(__file__).resolve().parents[3] / "nep_mbpol"
if str(NEP_MBPOL_ROOT) not in sys.path:
    sys.path.insert(0, str(NEP_MBPOL_ROOT))

from bca.runtime import (  # noqa: E402
    HardMomentCrossSections,
    KernelTableError,
    RuntimeCollision,
)
from bca.scattering import (  # noqa: E402
    PairKinematics,
    maximum_impact_parameter_angstrom as nlh_maximum_impact_parameter_angstrom,
    pair_kinematics,
    two_body_outcome_from_cm_angle,
)

from .kernel import PROJECTILES, TARGETS, cos_theta_cm, pair_cross_section_angstrom2


DEFAULT_PROJECTILES = ("C", "O", "S")
DEFAULT_ENERGY_BOUNDS_EV = (1.0e4, 1.0e8)


class FullZBLKernel:
    """Analytic full-ZBL collision kernel with a recoil transport cutoff.

    The retained disk is ``0 <= b <= b_max``. It therefore contains both
    close and distant nuclear encounters above ``minimum_transfer_ev`` and
    must replace, rather than accompany, NLH in one trajectory calculation.
    """

    def __init__(
        self,
        *,
        minimum_transfer_ev: float,
        projectiles: Sequence[str] = DEFAULT_PROJECTILES,
        energy_bounds_ev: tuple[float, float] = DEFAULT_ENERGY_BOUNDS_EV,
    ) -> None:
        if not math.isfinite(minimum_transfer_ev) or minimum_transfer_ev <= 0.0:
            raise ValueError("minimum_transfer_ev must be finite and positive.")
        selected = tuple(projectiles)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Projectiles must be a non-empty unique sequence.")
        unknown = sorted(set(selected) - set(PROJECTILES))
        if unknown:
            raise ValueError(f"Unsupported ZBL projectiles: {', '.join(unknown)}")
        lower, upper = energy_bounds_ev
        if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower < upper):
            raise ValueError("energy_bounds_ev must satisfy 0 < lower < upper.")
        self.minimum_transfer_ev = float(minimum_transfer_ev)
        self._projectiles = selected
        self._energy_bounds_ev = float(lower), float(upper)
        configuration = {
            "model": "universal_zbl_full_nuclear_elastic",
            "minimum_transfer_ev": self.minimum_transfer_ev,
            "projectiles": list(selected),
            "energy_bounds_ev": list(self._energy_bounds_ev),
        }
        self.csv_sha256 = hashlib.sha256(
            json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.manifest = configuration
        self.manifest_path = Path("analytic-full-zbl")

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (projectile, target)
            for projectile in self._projectiles
            for target in TARGETS
        )

    @property
    def projectiles(self) -> tuple[str, ...]:
        return self._projectiles

    def _validate_pair(self, projectile: str, target: str) -> None:
        if projectile not in self._projectiles:
            raise KernelTableError(f"ZBL backend does not contain {projectile}.")
        if target not in TARGETS:
            raise KernelTableError("Ice target must be H or O.")

    def energy_bounds_ev(self, projectile: str, target: str) -> tuple[float, float]:
        self._validate_pair(projectile, target)
        return self._energy_bounds_ev

    def pair_kinematics(
        self, projectile: str, target: str, projectile_energy_ev: float
    ) -> PairKinematics:
        self._validate_pair(projectile, target)
        lower, upper = self._energy_bounds_ev
        if (
            not math.isfinite(projectile_energy_ev)
            or not lower <= projectile_energy_ev <= upper
        ):
            raise KernelTableError(
                f"Energy must lie in the full-ZBL backend range {lower:g}-{upper:g} eV."
            )
        return pair_kinematics(projectile, target, float(projectile_energy_ev))

    @lru_cache(maxsize=4096)
    def maximum_impact_parameter_angstrom(
        self, projectile: str, target: str, projectile_energy_ev: float
    ) -> float:
        self.pair_kinematics(projectile, target, projectile_energy_ev)
        cross_section = pair_cross_section_angstrom2(
            PROJECTILES[projectile],
            TARGETS[target],
            projectile_energy_ev,
            self.minimum_transfer_ev,
        )
        return math.sqrt(cross_section / math.pi)

    def minimum_impact_parameter_angstrom(
        self, projectile: str, target: str, projectile_energy_ev: float
    ) -> float:
        self.pair_kinematics(projectile, target, projectile_energy_ev)
        return 0.0

    def impact_parameter_from_area_quantile(
        self,
        projectile: str,
        target: str,
        projectile_energy_ev: float,
        area_quantile: float,
    ) -> float:
        if not math.isfinite(area_quantile) or not 0.0 <= area_quantile <= 1.0:
            raise KernelTableError("area_quantile must lie in [0, 1].")
        lower = self.minimum_impact_parameter_angstrom(
            projectile, target, projectile_energy_ev
        )
        upper = self.maximum_impact_parameter_angstrom(
            projectile, target, projectile_energy_ev
        )
        return math.sqrt(lower * lower + area_quantile * (upper * upper - lower * lower))

    def area_quantile_from_impact_parameter(
        self,
        projectile: str,
        target: str,
        projectile_energy_ev: float,
        impact_parameter_angstrom: float,
    ) -> float:
        lower = self.minimum_impact_parameter_angstrom(
            projectile, target, projectile_energy_ev
        )
        upper = self.maximum_impact_parameter_angstrom(
            projectile, target, projectile_energy_ev
        )
        area = upper * upper - lower * lower
        if area <= 0.0:
            raise KernelTableError("The retained collision domain is empty.")
        return (impact_parameter_angstrom**2 - lower * lower) / area

    def turning_threshold_radius_angstrom(
        self, projectile: str, target: str
    ) -> float:
        """Return a conservative search extent for the retained ZBL disk."""
        return self.maximum_impact_parameter_angstrom(
            projectile, target, self._energy_bounds_ev[0]
        )

    def hard_cross_section_angstrom2(
        self, projectile: str, target: str, projectile_energy_ev: float
    ) -> float:
        minimum_impact = self.minimum_impact_parameter_angstrom(
            projectile, target, projectile_energy_ev
        )
        maximum_impact = self.maximum_impact_parameter_angstrom(
            projectile, target, projectile_energy_ev
        )
        return math.pi * max(0.0, maximum_impact**2 - minimum_impact**2)

    def area_quantile_breakpoints(
        self, projectile: str, target: str, projectile_energy_ev: float
    ) -> NDArray[np.float64]:
        self.pair_kinematics(projectile, target, projectile_energy_ev)
        values = np.asarray((0.0, 1.0), dtype=np.float64)
        values.setflags(write=False)
        return values

    def theta_cm_rad(
        self,
        projectile: str,
        target: str,
        projectile_energy_ev: float,
        area_quantile: float,
    ) -> float:
        self.pair_kinematics(projectile, target, projectile_energy_ev)
        if not math.isfinite(area_quantile) or not 0.0 <= area_quantile <= 1.0:
            raise KernelTableError("area_quantile must lie in [0, 1].")
        impact = self.impact_parameter_from_area_quantile(
            projectile, target, projectile_energy_ev, area_quantile
        )
        cosine = cos_theta_cm(
            PROJECTILES[projectile], TARGETS[target], projectile_energy_ev, impact
        )
        return math.acos(max(-1.0, min(1.0, cosine)))

    def hard_moment_cross_sections(
        self,
        projectile: str,
        target: str,
        projectile_energy_ev: float,
        *,
        quadrature_order: int = 16,
        quadrature_relative_tolerance: float = 5.0e-4,
    ) -> HardMomentCrossSections:
        if quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two.")
        if quadrature_relative_tolerance <= 0.0:
            raise ValueError("quadrature_relative_tolerance must be positive.")
        cross_section = self.hard_cross_section_angstrom2(
            projectile, target, projectile_energy_ev
        )
        if cross_section == 0.0:
            return HardMomentCrossSections(0.0, 0.0, 0.0, 0.0)
        context = self.pair_kinematics(projectile, target, projectile_energy_ev)

        def integrand(area_quantile: float) -> NDArray[np.float64]:
            theta = self.theta_cm_rad(
                projectile, target, projectile_energy_ev, area_quantile
            )
            outcome = two_body_outcome_from_cm_angle(context, theta)
            return np.asarray(
                (
                    outcome.recoil_energy_ev,
                    1.0 - math.cos(outcome.theta_projectile_lab_rad),
                ),
                dtype=np.float64,
            )

        # High-energy ZBL moments are concentrated in a very small head-on
        # impact-area interval. Logarithmic breakpoints prevent a nominally
        # high-order global rule from silently missing that contribution.
        moments, absolute_error = quad_vec(
            integrand,
            0.0,
            1.0,
            epsabs=1.0e-14,
            epsrel=0.1 * quadrature_relative_tolerance,
            points=[10.0 ** (-index) for index in range(1, 16)],
            limit=max(128, 8 * quadrature_order),
        )
        scale = max(float(np.linalg.norm(moments)), 1.0e-300)
        error = float(absolute_error) / scale
        if error > quadrature_relative_tolerance:
            raise KernelTableError(
                f"Full-ZBL moment quadrature error {error:.3g} exceeds "
                f"{quadrature_relative_tolerance:.3g}."
            )
        return HardMomentCrossSections(
            cross_section_angstrom2=cross_section,
            recoil_energy_cross_section_ev_angstrom2=(
                cross_section * float(moments[0])
            ),
            transport_cross_section_angstrom2=(
                cross_section * float(moments[1])
            ),
            quadrature_relative_error=error,
        )

    def collide(
        self,
        projectile: str,
        target: str,
        projectile_energy_ev: float,
        impact_parameter_angstrom: float,
    ) -> RuntimeCollision:
        context = self.pair_kinematics(projectile, target, projectile_energy_ev)
        maximum_impact = self.maximum_impact_parameter_angstrom(
            projectile, target, projectile_energy_ev
        )
        minimum_impact = self.minimum_impact_parameter_angstrom(
            projectile, target, projectile_energy_ev
        )
        tolerance = 1.0e-12 * max(1.0, maximum_impact)
        if (
            not math.isfinite(impact_parameter_angstrom)
            or impact_parameter_angstrom < minimum_impact - tolerance
            or impact_parameter_angstrom > maximum_impact + tolerance
        ):
            raise KernelTableError(
                "Impact parameter must lie in "
                f"[{minimum_impact:g}, {maximum_impact:g}] angstrom."
            )
        impact = min(max(float(impact_parameter_angstrom), minimum_impact), maximum_impact)
        theta = math.acos(
            max(
                -1.0,
                min(
                    1.0,
                    cos_theta_cm(
                        PROJECTILES[projectile],
                        TARGETS[target],
                        projectile_energy_ev,
                        impact,
                    ),
                ),
            )
        )
        outcome = two_body_outcome_from_cm_angle(context, theta)
        return RuntimeCollision(
            projectile=context.projectile,
            target=context.target,
            projectile_energy_ev=context.projectile_energy_ev,
            relative_kinetic_energy_ev=context.relative_kinetic_energy_ev,
            impact_parameter_angstrom=impact,
            maximum_impact_parameter_angstrom=maximum_impact,
            theta_cm_rad=theta,
            theta_projectile_lab_rad=outcome.theta_projectile_lab_rad,
            recoil_energy_ev=outcome.recoil_energy_ev,
            projectile_out_energy_ev=outcome.projectile_out_energy_ev,
            energy_conservation_error_ev=outcome.energy_conservation_error_ev,
        )


class SoftZBLKernel(FullZBLKernel):
    """ZBL annulus complementary to the 30 eV retained NLH hard disk.

    The inner boundary is the energy- and target-dependent NLH impact
    parameter whose turning point reaches ``nlh_boundary_ev``. The outer
    boundary is the ZBL impact parameter for ``minimum_transfer_ev``.
    This construction is exactly non-overlapping in impact area, but remains
    a validation-pending model because the two potentials need not match at
    the handoff.
    """

    def __init__(
        self,
        *,
        minimum_transfer_ev: float,
        nlh_boundary_ev: float = 30.0,
        projectiles: Sequence[str] = ("C",),
        energy_bounds_ev: tuple[float, float] = DEFAULT_ENERGY_BOUNDS_EV,
    ) -> None:
        if not math.isfinite(nlh_boundary_ev) or nlh_boundary_ev < 30.0:
            raise ValueError("nlh_boundary_ev cannot be below 30 eV.")
        self.nlh_boundary_ev = float(nlh_boundary_ev)
        super().__init__(
            minimum_transfer_ev=minimum_transfer_ev,
            projectiles=projectiles,
            energy_bounds_ev=energy_bounds_ev,
        )
        configuration = {
            "model": "universal_zbl_soft_annulus_nlh_complement",
            "minimum_transfer_ev": self.minimum_transfer_ev,
            "nlh_boundary_ev": self.nlh_boundary_ev,
            "projectiles": list(self.projectiles),
            "energy_bounds_ev": list(self._energy_bounds_ev),
            "partition": "b_nlh_30eV(E,target) < b <= b_zbl_transfer(E,target)",
        }
        self.csv_sha256 = hashlib.sha256(
            json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.manifest = configuration
        self.manifest_path = Path("analytic-zbl-soft-nlh-complement")

    @lru_cache(maxsize=4096)
    def minimum_impact_parameter_angstrom(
        self, projectile: str, target: str, projectile_energy_ev: float
    ) -> float:
        self.pair_kinematics(projectile, target, projectile_energy_ev)
        inner = nlh_maximum_impact_parameter_angstrom(
            projectile,
            target,
            projectile_energy_ev,
            minimum_turning_potential_ev=self.nlh_boundary_ev,
        )
        outer = super().maximum_impact_parameter_angstrom(
            projectile, target, projectile_energy_ev
        )
        return min(inner, outer)

    def hard_moment_cross_sections(
        self,
        projectile: str,
        target: str,
        projectile_energy_ev: float,
        *,
        quadrature_order: int = 32,
        quadrature_relative_tolerance: float = 5.0e-4,
    ) -> HardMomentCrossSections:
        """Integrate the smooth retained annulus with an n/2n Gauss check."""

        if quadrature_order < 8:
            raise ValueError("quadrature_order must be at least eight.")
        if quadrature_relative_tolerance <= 0.0:
            raise ValueError("quadrature_relative_tolerance must be positive.")
        cross_section = self.hard_cross_section_angstrom2(
            projectile, target, projectile_energy_ev
        )
        if cross_section == 0.0:
            return HardMomentCrossSections(0.0, 0.0, 0.0, 0.0)
        context = self.pair_kinematics(projectile, target, projectile_energy_ev)

        def integrate(order: int) -> NDArray[np.float64]:
            nodes, weights = np.polynomial.legendre.leggauss(order)
            total = np.zeros(2, dtype=np.float64)
            for node, weight in zip(nodes, weights, strict=True):
                quantile = 0.5 * (float(node) + 1.0)
                theta = self.theta_cm_rad(
                    projectile, target, projectile_energy_ev, quantile
                )
                outcome = two_body_outcome_from_cm_angle(context, theta)
                total += float(weight) * np.asarray(
                    (
                        outcome.recoil_energy_ev,
                        1.0 - math.cos(outcome.theta_projectile_lab_rad),
                    )
                )
            return 0.5 * total

        coarse = integrate(quadrature_order)
        fine = integrate(2 * quadrature_order)
        component_errors = np.abs(fine - coarse) / np.maximum(np.abs(fine), 1.0e-300)
        error = float(np.max(component_errors))
        if error > quadrature_relative_tolerance:
            raise KernelTableError(
                f"Soft-ZBL moment quadrature error {error:.3g} exceeds "
                f"{quadrature_relative_tolerance:.3g}."
            )
        return HardMomentCrossSections(
            cross_section_angstrom2=cross_section,
            recoil_energy_cross_section_ev_angstrom2=(
                cross_section * float(fine[0])
            ),
            transport_cross_section_angstrom2=(
                cross_section * float(fine[1])
            ),
            quadrature_relative_error=error,
        )
