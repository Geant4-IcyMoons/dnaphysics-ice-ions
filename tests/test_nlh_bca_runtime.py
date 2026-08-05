from __future__ import annotations

import csv
import hashlib
import json
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

from bca.runtime import AdaptiveKernelTable, KernelTableError  # noqa: E402
from bca.scattering import (  # noqa: E402
    NLHCollisionKernel,
    turning_threshold_radius_angstrom,
)
from bca.structure import IceStructure  # noqa: E402
from bca.trajectory import PeriodicHardCollisionTransport  # noqa: E402


def _kernel_product(
    directory: Path, projectiles: tuple[str, ...] = ("C",)
) -> Path:
    energies = (10_000.0, 100_000.0)
    quantiles = (0.0, 0.25, 0.75, 1.0)
    csv_path = directory / "nlh_collision_kernels.csv"
    rows: list[tuple[object, ...]] = []
    pair_records = []
    for projectile in projectiles:
        for target in ("H", "O"):
            pair_records.append(
                {
                    "projectile": projectile,
                    "target": target,
                    "threshold_radius_angstrom": turning_threshold_radius_angstrom(
                        projectile, target, minimum_turning_potential_ev=30.0
                    ),
                    "point_count": len(energies),
                    "maximum_theta_cm_relative_error": 0.0,
                    "maximum_recoil_relative_error": 0.0,
                    "validation_evaluations": 0,
                }
            )
            for energy in energies:
                kernel = NLHCollisionKernel(
                    projectile,
                    target,
                    energy,
                    minimum_turning_potential_ev=30.0,
                )
                for quantile in quantiles:
                    collision = kernel.solve(
                        kernel.maximum_impact_parameter_angstrom
                        * math.sqrt(quantile)
                    )
                    rows.append(
                        (
                            projectile,
                            target,
                            energy,
                            quantile,
                            collision.theta_cm_rad,
                        )
                    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "projectile",
                "target",
                "projectile_energy_ev",
                "area_quantile",
                "theta_cm_rad",
            )
        )
        writer.writerows(rows)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 4,
        "configuration": {
            "projectiles": list(projectiles),
            "minimum_turning_potential_ev": 30.0,
        },
        "energy_mesh": {"pairs": pair_records},
        "row_count": len(rows),
        "csv": csv_path.name,
        "csv_sha256": digest,
    }
    manifest_path = directory / "nlh_collision_kernels.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _structure() -> IceStructure:
    lattice = np.diag((10.0, 10.0, 10.0))
    return IceStructure(
        species=np.asarray(("O", "H", "H")),
        positions_angstrom=np.asarray(
            ((0.2, 5.0, 5.0), (5.0, 1.0, 1.0), (7.0, 8.0, 8.0)),
            dtype=np.float64,
        ),
        lattice_angstrom=lattice,
        pbc=(True, True, True),
        source_path=Path("accepted.xyz"),
        source_sha256="0" * 64,
        frame_index=0,
        metadata={"phase": "test Ih"},
        collision_ready=True,
    )


def test_runtime_reader_matches_direct_grid_collision(tmp_path):
    table = AdaptiveKernelTable(_kernel_product(tmp_path))
    direct = NLHCollisionKernel("C", "O", 10_000.0)
    impact = direct.maximum_impact_parameter_angstrom * math.sqrt(0.25)
    expected = direct.solve(impact)
    actual = table.collide("C", "O", 10_000.0, impact)
    assert actual.theta_cm_rad == pytest.approx(expected.theta_cm_rad, rel=1.0e-14)
    assert actual.recoil_energy_ev == pytest.approx(
        expected.recoil_energy_ev, rel=1.0e-14
    )
    assert actual.projectile_out_energy_ev + actual.recoil_energy_ev == pytest.approx(
        10_000.0, abs=1.0e-10
    )


def test_runtime_integrates_hard_moments_over_exact_cross_section(tmp_path):
    table = AdaptiveKernelTable(_kernel_product(tmp_path))
    moments = table.hard_moment_cross_sections("C", "O", 10_000.0)
    assert moments.cross_section_angstrom2 == pytest.approx(
        table.hard_cross_section_angstrom2("C", "O", 10_000.0), rel=1.0e-15
    )
    assert moments.recoil_energy_cross_section_ev_angstrom2 > 0.0
    assert moments.transport_cross_section_angstrom2 > 0.0
    # The integration budget is one tenth of the 0.5% table tolerance.
    assert moments.quadrature_relative_error <= 5.0e-4


def test_runtime_reader_rejects_tampered_csv(tmp_path):
    manifest = _kernel_product(tmp_path)
    csv_path = tmp_path / "nlh_collision_kernels.csv"
    csv_path.write_text(csv_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(KernelTableError, match="checksum"):
        AdaptiveKernelTable(manifest)


def test_periodic_transport_links_structure_to_kernel(tmp_path):
    table = AdaptiveKernelTable(_kernel_product(tmp_path))
    transport = PeriodicHardCollisionTransport(_structure(), table)
    result = transport.trace(
        "C",
        10_000.0,
        (9.8, 5.0, 5.0),
        (1.0, 0.0, 0.0),
        1.0,
        rng=np.random.default_rng(1234),
        max_collisions=1,
    )
    assert result.termination == "maximum_collisions"
    assert len(result.events) == 1
    event = result.events[0]
    assert event.target == "O"
    assert event.target_image == (1, 0, 0)
    assert event.path_distance_angstrom == pytest.approx(0.4, abs=1.0e-12)
    assert event.impact_parameter_angstrom == pytest.approx(0.0, abs=1.0e-12)
    assert event.projectile_energy_out_ev + event.recoil_energy_ev == pytest.approx(
        event.projectile_energy_in_ev, abs=1.0e-10
    )
    assert np.linalg.norm(event.recoil_direction) == pytest.approx(1.0)


def test_straight_line_control_variate_has_exact_periodic_mean(tmp_path):
    table = AdaptiveKernelTable(_kernel_product(tmp_path))
    transport = PeriodicHardCollisionTransport(_structure(), table)
    reference = transport.straight_line_control_variate(
        "C", 10_000.0, (9.8, 5.0, 5.0), (1.0, 0.0, 0.0), 1.0
    )
    assert reference.collision_count == 1.0
    assert reference.recoil_energy_ev > 0.0
    assert reference.transport_moment > 0.0
    assert reference.expected_collision_count == pytest.approx(
        transport.independent_atom_rate_per_angstrom("C", 10_000.0)
    )
    assert reference.expected_recoil_energy_ev > 0.0
    assert reference.expected_transport_moment > 0.0


@pytest.mark.parametrize("projectile", ("H", "He"))
def test_periodic_transport_supports_proton_and_helium_projectiles(
    tmp_path, projectile
):
    table = AdaptiveKernelTable(_kernel_product(tmp_path, (projectile,)))
    transport = PeriodicHardCollisionTransport(_structure(), table)
    result = transport.trace(
        projectile,
        10_000.0,
        (9.8, 5.0, 5.0),
        (1.0, 0.0, 0.0),
        1.0,
        rng=np.random.default_rng(2025),
        max_collisions=1,
    )
    assert result.projectile == projectile
    assert len(result.events) == 1
    event = result.events[0]
    assert event.target == "O"
    assert event.projectile_energy_out_ev + event.recoil_energy_ev == pytest.approx(
        event.projectile_energy_in_ev, abs=1.0e-10
    )
    assert np.linalg.norm(event.recoil_direction) == pytest.approx(1.0)


def test_search_window_does_not_change_first_collision(tmp_path):
    table = AdaptiveKernelTable(_kernel_product(tmp_path))
    results = []
    for window in (0.2, 4.0):
        transport = PeriodicHardCollisionTransport(
            _structure(), table, search_window_angstrom=window
        )
        results.append(
            transport.trace(
                "C",
                10_000.0,
                (9.8, 5.0, 5.0),
                (1.0, 0.0, 0.0),
                1.0,
                rng=np.random.default_rng(42),
                max_collisions=1,
            )
        )
    first, second = results[0].events[0], results[1].events[0]
    assert first.target_atom_index == second.target_atom_index
    assert first.target_image == second.target_image
    assert first.path_distance_angstrom == pytest.approx(
        second.path_distance_angstrom, abs=1.0e-12
    )
    assert first.recoil_energy_ev == pytest.approx(second.recoil_energy_ev)
    assert first.direction_out == pytest.approx(second.direction_out)


def test_path_that_misses_hard_cylinders_is_unchanged(tmp_path):
    table = AdaptiveKernelTable(_kernel_product(tmp_path))
    transport = PeriodicHardCollisionTransport(_structure(), table)
    result = transport.trace(
        "C",
        10_000.0,
        (9.8, 6.5, 5.0),
        (1.0, 0.0, 0.0),
        1.0,
        rng=np.random.default_rng(7),
    )
    assert result.termination == "path_complete"
    assert not result.events
    assert result.final_energy_ev == result.initial_energy_ev
    assert result.traveled_path_length_angstrom == pytest.approx(1.0)
