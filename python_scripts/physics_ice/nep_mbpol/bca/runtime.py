"""Validated runtime interpolation of adaptive hard-collision kernels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import canonical_element
from .scattering import (
    PairKinematics,
    pair_kinematics,
    two_body_observables_from_cm_angles,
    two_body_outcome_from_cm_angle,
)


class KernelTableError(ValueError):
    """Raised when a collision table or interpolation request is invalid."""


@dataclass(frozen=True)
class RuntimeCollision:
    """One tabulated central-potential collision and its exact lab outcome."""

    projectile: str
    target: str
    projectile_energy_ev: float
    relative_kinetic_energy_ev: float
    impact_parameter_angstrom: float
    maximum_impact_parameter_angstrom: float
    theta_cm_rad: float
    theta_projectile_lab_rad: float
    recoil_energy_ev: float
    projectile_out_energy_ev: float
    energy_conservation_error_ev: float


@dataclass(frozen=True)
class HardMomentCrossSections:
    """Area-integrated moments of the interpolated retained hard kernel."""

    cross_section_angstrom2: float
    recoil_energy_cross_section_ev_angstrom2: float
    transport_cross_section_angstrom2: float
    quadrature_relative_error: float


@dataclass(frozen=True)
class _PairTable:
    energies_ev: NDArray[np.float64]
    area_quantiles: tuple[NDArray[np.float64], ...]
    theta_cm_rad: tuple[NDArray[np.float64], ...]
    threshold_radius_angstrom: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KernelTableError(f"Could not parse kernel manifest {path}.") from exc
    if not isinstance(value, dict):
        raise KernelTableError("Kernel manifest must contain a JSON object.")
    return value


def _resolve_paths(source: str | Path) -> tuple[Path, Path, dict[str, object]]:
    path = Path(source).expanduser().resolve()
    manifest_path = (
        path / "nlh_collision_kernels.manifest.json" if path.is_dir() else path
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _load_manifest(manifest_path)
    csv_name = manifest.get("csv")
    if not isinstance(csv_name, str) or not csv_name:
        raise KernelTableError("Kernel manifest does not name its CSV file.")
    csv_path = (manifest_path.parent / csv_name).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    expected_sha = manifest.get("csv_sha256")
    if not isinstance(expected_sha, str) or _sha256(csv_path) != expected_sha:
        raise KernelTableError("Kernel CSV checksum does not match its manifest.")
    return manifest_path, csv_path, manifest


def _pair_radii(manifest: dict[str, object]) -> dict[tuple[str, str], float]:
    energy_mesh = manifest.get("energy_mesh")
    if not isinstance(energy_mesh, dict):
        raise KernelTableError("Kernel manifest has no energy_mesh object.")
    records = energy_mesh.get("pairs")
    if not isinstance(records, list):
        raise KernelTableError("Kernel manifest has no pair metadata.")
    result: dict[tuple[str, str], float] = {}
    for record in records:
        if not isinstance(record, dict):
            raise KernelTableError("Malformed pair metadata in kernel manifest.")
        try:
            pair = (
                canonical_element(str(record["projectile"])),
                canonical_element(str(record["target"])),
            )
            radius = float(record["threshold_radius_angstrom"])
        except (KeyError, TypeError, ValueError) as exc:
            raise KernelTableError("Malformed threshold-radius record.") from exc
        if not math.isfinite(radius) or radius <= 0.0 or pair in result:
            raise KernelTableError("Invalid or duplicate threshold-radius record.")
        result[pair] = radius
    return result


def _load_csv(
    path: Path,
    expected_rows: int,
    radii: dict[tuple[str, str], float],
) -> dict[tuple[str, str], _PairTable]:
    grouped: dict[
        tuple[str, str],
        dict[float, tuple[list[float], list[float]]],
    ] = {}
    row_count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "projectile",
            "target",
            "projectile_energy_ev",
            "area_quantile",
            "theta_cm_rad",
        }
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise KernelTableError("Kernel CSV columns do not match schema version 4.")
        for row in reader:
            row_count += 1
            try:
                pair = (
                    canonical_element(row["projectile"]),
                    canonical_element(row["target"]),
                )
                energy = float(row["projectile_energy_ev"])
                quantile = float(row["area_quantile"])
                theta = float(row["theta_cm_rad"])
            except (TypeError, ValueError) as exc:
                raise KernelTableError(f"Invalid kernel CSV row {row_count}.") from exc
            values = grouped.setdefault(pair, {}).setdefault(energy, ([], []))
            values[0].append(quantile)
            values[1].append(theta)
    if row_count != expected_rows:
        raise KernelTableError(
            f"Kernel row count is {row_count}, manifest records {expected_rows}."
        )
    if set(grouped) != set(radii):
        raise KernelTableError("Kernel CSV pairs do not match the manifest.")

    result: dict[tuple[str, str], _PairTable] = {}
    for pair, energy_groups in grouped.items():
        energies = np.asarray(sorted(energy_groups), dtype=np.float64)
        if (
            energies.size < 2
            or not np.all(np.isfinite(energies))
            or not np.all(np.diff(energies) > 0.0)
        ):
            raise KernelTableError(f"Invalid energy mesh for {pair}.")
        quantiles: list[NDArray[np.float64]] = []
        angles: list[NDArray[np.float64]] = []
        for energy in energies:
            q_values, theta_values = energy_groups[float(energy)]
            q = np.asarray(q_values, dtype=np.float64)
            theta = np.asarray(theta_values, dtype=np.float64)
            if not (
                q.size >= 2
                and q.shape == theta.shape
                and q[0] == 0.0
                and q[-1] == 1.0
                and np.all(np.isfinite(q))
                and np.all(np.isfinite(theta))
                and np.all(np.diff(q) > 0.0)
                and np.all(np.diff(theta) < 0.0)
                and np.all((theta >= 0.0) & (theta <= math.pi))
            ):
                raise KernelTableError(
                    f"Invalid impact mesh for {pair} at {energy:g} eV."
                )
            q.setflags(write=False)
            theta.setflags(write=False)
            quantiles.append(q)
            angles.append(theta)
        energies.setflags(write=False)
        result[pair] = _PairTable(
            energies_ev=energies,
            area_quantiles=tuple(quantiles),
            theta_cm_rad=tuple(angles),
            threshold_radius_angstrom=radii[pair],
        )
    return result


class AdaptiveKernelTable:
    """Read and interpolate one converged adaptive collision-table product."""

    def __init__(self, source: str | Path) -> None:
        manifest_path, csv_path, manifest = _resolve_paths(source)
        if manifest.get("schema_version") != 4:
            raise KernelTableError("Only adaptive kernel schema version 4 is supported.")
        configuration = manifest.get("configuration")
        if not isinstance(configuration, dict):
            raise KernelTableError("Kernel manifest has no configuration object.")
        try:
            threshold = float(configuration["minimum_turning_potential_ev"])
            expected_rows = int(manifest["row_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise KernelTableError("Incomplete numerical kernel configuration.") from exc
        if not math.isfinite(threshold) or threshold <= 0.0 or expected_rows <= 0:
            raise KernelTableError("Invalid kernel threshold or row count.")
        self.manifest_path = manifest_path
        self.csv_path = csv_path
        self.manifest = manifest
        self.minimum_turning_potential_ev = threshold
        self.csv_sha256 = str(manifest["csv_sha256"])
        self._pairs = _load_csv(
            csv_path,
            expected_rows,
            _pair_radii(manifest),
        )

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._pairs))

    @property
    def projectiles(self) -> tuple[str, ...]:
        return tuple(sorted({pair[0] for pair in self._pairs}))

    def _pair(self, projectile: str, target: str) -> _PairTable:
        pair = (canonical_element(projectile), canonical_element(target))
        try:
            return self._pairs[pair]
        except KeyError as exc:
            raise KernelTableError(f"Kernel table does not contain {pair[0]}-{pair[1]}.") from exc

    def energy_bounds_ev(self, projectile: str, target: str) -> tuple[float, float]:
        pair = self._pair(projectile, target)
        return float(pair.energies_ev[0]), float(pair.energies_ev[-1])

    def turning_threshold_radius_angstrom(
        self, projectile: str, target: str
    ) -> float:
        """Radius at the tabulated turning-potential validity boundary."""

        return self._pair(projectile, target).threshold_radius_angstrom

    def pair_kinematics(
        self, projectile: str, target: str, projectile_energy_ev: float
    ) -> PairKinematics:
        pair = self._pair(projectile, target)
        minimum, maximum = float(pair.energies_ev[0]), float(pair.energies_ev[-1])
        if (
            not math.isfinite(projectile_energy_ev)
            or projectile_energy_ev < minimum
            or projectile_energy_ev > maximum
        ):
            raise KernelTableError(
                f"Energy {projectile_energy_ev:g} eV lies outside the {minimum:g}-"
                f"{maximum:g} eV table for {projectile}-{target}."
            )
        return pair_kinematics(projectile, target, float(projectile_energy_ev))

    def maximum_impact_parameter_angstrom(
        self, projectile: str, target: str, projectile_energy_ev: float
    ) -> float:
        pair = self._pair(projectile, target)
        kinematics = self.pair_kinematics(
            projectile, target, projectile_energy_ev
        )
        relative_energy = kinematics.relative_kinetic_energy_ev
        if relative_energy <= self.minimum_turning_potential_ev:
            return 0.0
        return pair.threshold_radius_angstrom * math.sqrt(
            1.0 - self.minimum_turning_potential_ev / relative_energy
        )

    def minimum_impact_parameter_angstrom(
        self, projectile: str, target: str, projectile_energy_ev: float
    ) -> float:
        """Lower edge of the retained impact-area domain (a disk for NLH)."""

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
        maximum = self.maximum_impact_parameter_angstrom(
            projectile, target, projectile_energy_ev
        )
        return maximum * math.sqrt(area_quantile)

    def area_quantile_from_impact_parameter(
        self,
        projectile: str,
        target: str,
        projectile_energy_ev: float,
        impact_parameter_angstrom: float,
    ) -> float:
        maximum = self.maximum_impact_parameter_angstrom(
            projectile, target, projectile_energy_ev
        )
        if maximum <= 0.0:
            raise KernelTableError("The retained collision domain is empty.")
        return (impact_parameter_angstrom / maximum) ** 2

    def hard_cross_section_angstrom2(
        self, projectile: str, target: str, projectile_energy_ev: float
    ) -> float:
        maximum_impact = self.maximum_impact_parameter_angstrom(
            projectile, target, projectile_energy_ev
        )
        return math.pi * maximum_impact * maximum_impact

    def area_quantile_breakpoints(
        self, projectile: str, target: str, projectile_energy_ev: float
    ) -> NDArray[np.float64]:
        """Return every impact-grid breakpoint used at one interpolated energy."""

        pair = self._pair(projectile, target)
        self.pair_kinematics(projectile, target, projectile_energy_ev)
        energies = pair.energies_ev
        upper = int(np.searchsorted(energies, projectile_energy_ev, side="left"))
        if upper < len(energies) and energies[upper] == projectile_energy_ev:
            values = pair.area_quantiles[upper].copy()
        else:
            if upper == 0 or upper == len(energies):
                raise KernelTableError(
                    "Energy interpolation attempted outside the table."
                )
            values = np.union1d(
                pair.area_quantiles[upper - 1], pair.area_quantiles[upper]
            )
        values.setflags(write=False)
        return values

    def hard_moment_cross_sections(
        self,
        projectile: str,
        target: str,
        projectile_energy_ev: float,
        *,
        quadrature_order: int = 8,
        quadrature_relative_tolerance: float = 5.0e-4,
    ) -> HardMomentCrossSections:
        """Integrate recoil and transport moments over retained collision area.

        The area quantile q=(b/b_max)^2 is uniform under 2*pi*b*db.  Every
        adaptive table breakpoint is integrated separately, and orders n and
        2n are compared so quadrature error is distinct from the kernel's
        interpolation budget.
        """

        if quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two.")
        if (
            not math.isfinite(quadrature_relative_tolerance)
            or quadrature_relative_tolerance <= 0.0
        ):
            raise ValueError(
                "quadrature_relative_tolerance must be finite and positive."
            )
        cross_section = self.hard_cross_section_angstrom2(
            projectile, target, projectile_energy_ev
        )
        if cross_section == 0.0:
            return HardMomentCrossSections(0.0, 0.0, 0.0, 0.0)
        breakpoints = self.area_quantile_breakpoints(
            projectile, target, projectile_energy_ev
        )
        kinematics = self.pair_kinematics(
            projectile, target, projectile_energy_ev
        )

        def integrate(order: int) -> tuple[float, float]:
            nodes, weights = np.polynomial.legendre.leggauss(order)
            recoil_total = 0.0
            transport_total = 0.0
            for lower, upper in zip(
                breakpoints[:-1], breakpoints[1:], strict=True
            ):
                half_width = 0.5 * float(upper - lower)
                midpoint = 0.5 * float(upper + lower)
                quantiles = midpoint + half_width * nodes
                theta_cm = np.asarray(
                    [
                        self.theta_cm_rad(
                            projectile,
                            target,
                            projectile_energy_ev,
                            float(quantile),
                        )
                        for quantile in quantiles
                    ],
                    dtype=np.float64,
                )
                theta_lab, recoil = two_body_observables_from_cm_angles(
                    kinematics, theta_cm
                )
                recoil_total += half_width * float(np.dot(weights, recoil))
                transport_total += half_width * float(
                    np.dot(weights, 1.0 - np.cos(theta_lab))
                )
            return recoil_total, transport_total

        coarse = integrate(quadrature_order)
        fine = integrate(2 * quadrature_order)
        relative_errors = [
            abs(fine_value - coarse_value) / max(abs(fine_value), 1.0e-300)
            for coarse_value, fine_value in zip(coarse, fine, strict=True)
        ]
        maximum_error = max(relative_errors)
        if maximum_error > quadrature_relative_tolerance:
            raise KernelTableError(
                "Hard-moment quadrature did not converge: "
                f"relative error {maximum_error:.3g} exceeds "
                f"{quadrature_relative_tolerance:.3g}."
            )
        return HardMomentCrossSections(
            cross_section_angstrom2=cross_section,
            recoil_energy_cross_section_ev_angstrom2=cross_section * fine[0],
            transport_cross_section_angstrom2=cross_section * fine[1],
            quadrature_relative_error=maximum_error,
        )

    @staticmethod
    def _angle_at_quantile(pair: _PairTable, index: int, quantile: float) -> float:
        return float(
            np.interp(
                quantile,
                pair.area_quantiles[index],
                pair.theta_cm_rad[index],
            )
        )

    def theta_cm_rad(
        self,
        projectile: str,
        target: str,
        projectile_energy_ev: float,
        area_quantile: float,
    ) -> float:
        pair = self._pair(projectile, target)
        self.pair_kinematics(projectile, target, projectile_energy_ev)
        if not math.isfinite(area_quantile) or not 0.0 <= area_quantile <= 1.0:
            raise KernelTableError("area_quantile must lie in [0, 1].")
        energies = pair.energies_ev
        upper = int(np.searchsorted(energies, projectile_energy_ev, side="left"))
        if upper < len(energies) and energies[upper] == projectile_energy_ev:
            return self._angle_at_quantile(pair, upper, area_quantile)
        if upper == 0 or upper == len(energies):
            raise KernelTableError("Energy interpolation attempted outside the table.")
        lower = upper - 1
        theta_lower = self._angle_at_quantile(pair, lower, area_quantile)
        theta_upper = self._angle_at_quantile(pair, upper, area_quantile)
        fraction = math.log(projectile_energy_ev / energies[lower]) / math.log(
            energies[upper] / energies[lower]
        )
        if theta_lower > 0.0 and theta_upper > 0.0:
            return math.exp(
                (1.0 - fraction) * math.log(theta_lower)
                + fraction * math.log(theta_upper)
            )
        return (1.0 - fraction) * theta_lower + fraction * theta_upper

    def collide(
        self,
        projectile: str,
        target: str,
        projectile_energy_ev: float,
        impact_parameter_angstrom: float,
    ) -> RuntimeCollision:
        kinematics = self.pair_kinematics(
            projectile, target, projectile_energy_ev
        )
        maximum_impact = self.maximum_impact_parameter_angstrom(
            projectile, target, projectile_energy_ev
        )
        if maximum_impact <= 0.0:
            raise KernelTableError("This pair cannot reach the retained hard domain.")
        tolerance = 1.0e-12 * max(1.0, maximum_impact)
        if (
            not math.isfinite(impact_parameter_angstrom)
            or impact_parameter_angstrom < 0.0
            or impact_parameter_angstrom > maximum_impact + tolerance
        ):
            raise KernelTableError(
                f"Impact parameter must lie in [0, {maximum_impact:g}] angstrom."
            )
        impact = min(float(impact_parameter_angstrom), maximum_impact)
        quantile = (impact / maximum_impact) ** 2
        theta_cm = self.theta_cm_rad(
            projectile, target, projectile_energy_ev, quantile
        )
        outcome = two_body_outcome_from_cm_angle(kinematics, theta_cm)
        return RuntimeCollision(
            projectile=kinematics.projectile,
            target=kinematics.target,
            projectile_energy_ev=kinematics.projectile_energy_ev,
            relative_kinetic_energy_ev=kinematics.relative_kinetic_energy_ev,
            impact_parameter_angstrom=impact,
            maximum_impact_parameter_angstrom=maximum_impact,
            theta_cm_rad=theta_cm,
            theta_projectile_lab_rad=outcome.theta_projectile_lab_rad,
            recoil_energy_ev=outcome.recoil_energy_ev,
            projectile_out_energy_ev=outcome.projectile_out_energy_ev,
            energy_conservation_error_ev=outcome.energy_conservation_error_ev,
        )
