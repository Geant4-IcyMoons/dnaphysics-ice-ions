from __future__ import annotations

from pathlib import Path
import sys


NEP_MBPOL = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "nep_mbpol"
)
if str(NEP_MBPOL) not in sys.path:
    sys.path.insert(0, str(NEP_MBPOL))

from bca.adaptivity import adaptive_energy_mesh, adaptive_impact_mesh  # noqa: E402
from bca.scattering import NLHCollisionKernel  # noqa: E402


def test_impact_mesh_discovers_narrow_high_energy_head_on_region():
    tolerance = 0.02
    mesh = adaptive_impact_mesh(
        NLHCollisionKernel("H", "H", 1.0e8, quadrature_order=32),
        relative_tolerance=tolerance,
        max_points=2048,
    )
    assert mesh.area_quantiles[1] < 1.0e-12
    assert mesh.recoil_l1_relative_error <= tolerance
    assert mesh.transport_l1_relative_error <= tolerance
    assert mesh.theta_cm_l1_relative_error <= tolerance
    assert mesh.theta_cm_max_error_over_pi <= tolerance


def test_energy_mesh_refines_sulfur_hydrogen_threshold_region():
    tolerance = 0.02
    mesh = adaptive_energy_mesh(
        "S",
        "H",
        energy_min_ev=1.0e3,
        energy_max_ev=2.0e3,
        base_points=3,
        minimum_turning_potential_ev=30.0,
        quadrature_order=32,
        relative_tolerance=tolerance,
        max_points=64,
    )
    assert mesh.point_count > 3
    assert mesh.maximum_theta_cm_relative_error <= tolerance
    assert mesh.maximum_recoil_relative_error <= tolerance
