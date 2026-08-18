from __future__ import annotations

import json
from pathlib import Path

import pytest

from python_scripts.physics_ice.process_evidence.soft_nuclear_collisions.zbl.run_campaign import (
    prepare_campaign,
)


def test_prepare_carbon_campaign_covers_both_atomistic_phases(tmp_path: Path) -> None:
    path = prepare_campaign(
        tmp_path / "campaign",
        projectile="C",
        phases=("hexagonal_ih_100k", "amorphous_lda_80k"),
        cutoffs_ev=(10.0,),
        energy_min_ev=1.0e4,
        energy_max_ev=1.0e8,
        energy_points=2,
        tolerance=0.005,
        confidence=0.95,
        minimum_trajectories=2,
        maximum_trajectories=4,
        trajectory_batch_size=2,
        path_length_angstrom=1.0,
    )
    campaign = json.loads(path.read_text(encoding="utf-8"))

    assert campaign["case_count"] == 20
    assert len({case["case_key"] for case in campaign["cases"]}) == 20
    assert {case["phase_id"] for case in campaign["cases"]} == {
        "hexagonal_ih_100k",
        "amorphous_lda_80k",
    }
    configuration = campaign["configuration"]
    assert configuration["statistical_relative_tolerance"] == 0.005
    assert configuration["trajectory_cdf_absolute_tolerance"] == 0.005
    assert len(configuration["backend_manifests"]) == 2
    assert all(
        len(record["sha256"]) == 64
        for record in configuration["backend_manifests"]
    )


def test_prepare_rejects_light_projectile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="restricted to C, O, and S"):
        prepare_campaign(
            tmp_path,
            projectile="H",
            phases=("amorphous_lda_80k",),
            cutoffs_ev=(10.0,),
            energy_min_ev=1.0e4,
            energy_max_ev=1.0e8,
            energy_points=2,
            tolerance=0.005,
            confidence=0.95,
            minimum_trajectories=2,
            maximum_trajectories=4,
            trajectory_batch_size=2,
            path_length_angstrom=1.0,
        )


def test_prepare_soft_campaign_records_nlh_complement(tmp_path: Path) -> None:
    path = prepare_campaign(
        tmp_path / "soft_campaign",
        projectile="C",
        phases=("amorphous_lda_80k",),
        cutoffs_ev=(1.0e-4,),
        energy_min_ev=1.0e4,
        energy_max_ev=1.0e8,
        energy_points=2,
        tolerance=0.005,
        confidence=0.95,
        minimum_trajectories=2,
        maximum_trajectories=4,
        trajectory_batch_size=2,
        path_length_angstrom=1.0,
        interaction_model="zbl_soft",
        nlh_boundary_ev=30.0,
    )
    campaign = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "zbl_soft_campaign.manifest.json"
    assert campaign["configuration"]["interaction_model"] == "zbl_soft"
    assert campaign["configuration"]["nlh_turning_potential_boundary_ev"] == 30.0
    backend_path = Path(
        campaign["configuration"]["backend_manifests"][0]["path"]
    )
    backend = json.loads(backend_path.read_text(encoding="utf-8"))
    assert backend["backend"] == "atomistic_periodic_zbl_soft"
    assert backend["production_enabled"] is False
