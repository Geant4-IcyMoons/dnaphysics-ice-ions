from __future__ import annotations

import json
import math

import numpy as np
import pytest

from python_scripts.physics_ice.process_evidence.soft_nuclear_collisions.zbl \
    import FullZBLKernel
from python_scripts.physics_ice.process_evidence.soft_nuclear_collisions.zbl \
    import generate_backend
from python_scripts.physics_ice.process_evidence.soft_nuclear_collisions.zbl \
    import kernel


@pytest.mark.parametrize("projectile", ("C", "O", "S"))
@pytest.mark.parametrize("target", ("H", "O"))
@pytest.mark.parametrize("energy_ev", (1.0e4, 1.0e6, 1.0e8))
def test_full_zbl_kernel_matches_pair_cross_section(
    projectile: str, target: str, energy_ev: float
) -> None:
    backend = FullZBLKernel(minimum_transfer_ev=10.0)
    expected = kernel.pair_cross_section_angstrom2(
        kernel.PROJECTILES[projectile], kernel.TARGETS[target], energy_ev, 10.0
    )
    assert backend.hard_cross_section_angstrom2(
        projectile, target, energy_ev
    ) == pytest.approx(expected, rel=1.0e-12)
    assert backend.maximum_impact_parameter_angstrom(
        projectile, target, energy_ev
    ) > 0.0


def test_full_zbl_collision_retains_exact_two_body_energy() -> None:
    backend = FullZBLKernel(minimum_transfer_ev=10.0)
    maximum_impact = backend.maximum_impact_parameter_angstrom("C", "O", 1.0e6)
    collision = backend.collide("C", "O", 1.0e6, 0.5 * maximum_impact)
    assert collision.recoil_energy_ev > 10.0
    assert collision.projectile_out_energy_ev > 0.0
    assert collision.energy_conservation_error_ev == pytest.approx(
        0.0, abs=1.0e-8
    )
    assert (
        collision.projectile_out_energy_ev + collision.recoil_energy_ev
        == pytest.approx(collision.projectile_energy_ev, rel=1.0e-12)
    )


def test_high_energy_moments_resolve_the_head_on_area() -> None:
    backend = FullZBLKernel(minimum_transfer_ev=10.0)
    moments = backend.hard_moment_cross_sections("C", "O", 1.0e8)
    assert moments.cross_section_angstrom2 > 0.0
    assert moments.recoil_energy_cross_section_ev_angstrom2 > 0.0
    assert moments.transport_cross_section_angstrom2 > 0.0
    assert moments.quadrature_relative_error <= 5.0e-4


def test_phase_manifests_bind_only_accepted_structures() -> None:
    hexagonal = generate_backend.build_manifest("hexagonal_ih_100k")
    amorphous = generate_backend.build_manifest("amorphous_lda_80k")
    assert len(hexagonal["structures"]) == 3
    assert len(amorphous["structures"]) == 1
    assert hexagonal["orientations"] == ["c_axis", "basal_a_axis", "isotropic"]
    assert amorphous["orientations"] == ["isotropic"]
    assert hexagonal["charge_state_aliases"] == {
        "C": list(range(7)),
        "O": list(range(9)),
        "S": list(range(17)),
    }
    assert hexagonal["interaction_domain"].startswith("full retained ZBL disk")
    assert hexagonal["production_enabled"] is False


def test_written_manifest_loads_atomistic_transport(tmp_path) -> None:
    paths = generate_backend.write_manifests(
        tmp_path,
        ("amorphous_lda_80k",),
        minimum_transfer_cutoffs_ev=(10.0,),
    )
    manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    transport = generate_backend.load_transport(
        paths[0], structure_index=0, minimum_transfer_ev=10.0
    )
    assert manifest["phase_id"] == "amorphous_lda_80k"
    assert transport.structure.collision_ready
    assert transport.structure.water_molecule_count == 3000
    assert transport.independent_atom_rate_per_angstrom("C", 1.0e6) > 0.0
    result = transport.trace(
        "C",
        1.0e6,
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((1.0, 0.0, 0.0)),
        1.0,
        rng=np.random.default_rng(7),
    )
    assert result.termination == "path_complete"
    assert math.isfinite(result.final_energy_ev)
