from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest


NEP_DIR = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "nep_mbpol"
)
if str(NEP_DIR) not in sys.path:
    sys.path.insert(0, str(NEP_DIR))

from soft_dft.config import load_builtin_projectile  # noqa: E402
from soft_dft.gpaw_cdft import (  # noqa: E402
    DEFAULT_GPAW_CDFT_SETTINGS,
    GPAWCDFTSettings,
    SCF_PROFILES,
    build_pilot_manifest,
    load_pilot_manifest,
    task_directory,
    validate_result_record,
)


def test_default_carbon_pilot_has_two_charges_and_five_orientations(tmp_path):
    path = build_pilot_manifest(tmp_path / "pilot")
    manifest = load_pilot_manifest(path)
    assert manifest["task_count"] == 10
    assert [task["task_index"] for task in manifest["tasks"]] == list(range(10))
    assert {task["state"]["charge"] for task in manifest["tasks"]} == {1, 2}
    assert len({task["geometry"]["orientation"] for task in manifest["tasks"]}) == 5
    assert all(task["geometry"]["separation_angstrom"] == 12.0 for task in manifest["tasks"])
    assert manifest["production_eligible"] is False
    assert manifest["physical_state_status"] == "validation_pending"


def test_manifest_is_immutable_and_signature_checked(tmp_path):
    path = build_pilot_manifest(tmp_path / "pilot", charges=(1,), orientations=("oxygen_back",))
    original = json.loads(path.read_text(encoding="utf-8"))
    original["tasks"][0]["state"]["charge"] = 2
    path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest signature mismatch"):
        load_pilot_manifest(path)


def test_standard_carbon_paw_scope_rejects_core_ionization(tmp_path):
    with pytest.raises(ValueError, match="freezes the 1s2 core"):
        build_pilot_manifest(
            tmp_path / "pilot",
            projectile=load_builtin_projectile("C"),
            charges=(5,),
            orientations=("oxygen_back",),
        )


def _passing_record(task):
    return {
        "task_signature": task["task_signature"],
        "gpaw_version": "25.7.0",
        "normal_return": True,
        "inner_scf_converged": True,
        "constraint_residuals_electrons": [0.002, -0.003],
        "dft_energy_ev": -100.0,
        "forces_ev_per_angstrom": [[0.0, 0.0, 0.0]] * 4,
        "multiplier_bound_contact": False,
        "total_magnetic_moment_electrons": 1.0,
        "gaussian_hirshfeld_electrons_by_atom": [5.002, 8.0, 1.0, 1.0],
        "paw_setups": [
            {"symbol": "C", "valence_electrons": 4.0},
            {"symbol": "O", "valence_electrons": 6.0},
            {"symbol": "H", "valence_electrons": 1.0},
            {"symbol": "H", "valence_electrons": 1.0},
        ],
    }


def test_numerical_gate_fails_closed_on_residual_or_bound(tmp_path):
    path = build_pilot_manifest(
        tmp_path / "pilot", charges=(1,), orientations=("oxygen_back",)
    )
    task = load_pilot_manifest(path)["tasks"][0]
    record = _passing_record(task)
    assert validate_result_record(record, task) == (True, [])
    record["constraint_residuals_electrons"] = [0.011, 0.0]
    accepted, reasons = validate_result_record(record, task)
    assert not accepted
    assert "constraint_tolerance_not_met" in reasons
    record = _passing_record(task)
    record["multiplier_bound_contact"] = True
    accepted, reasons = validate_result_record(record, task)
    assert not accepted
    assert "multiplier_bound_contact" in reasons


def test_settings_reject_unreviewed_version_and_bad_tolerance():
    with pytest.raises(ValueError, match="reviewed only"):
        replace(DEFAULT_GPAW_CDFT_SETTINGS, gpaw_version="26.7.0")
    with pytest.raises(ValueError, match="positive and finite"):
        GPAWCDFTSettings(constraint_tolerance_electrons=0.0)
    with pytest.raises(ValueError, match="Unsupported GPAW SCF profile"):
        GPAWCDFTSettings(scf_profile="invented")


def test_reviewed_scf_profiles_include_damped_pulay_controls():
    assert "official_damped_pulay" in SCF_PROFILES
    assert "official_damped_pulay_eigensolver5" in SCF_PROFILES
    for profile in SCF_PROFILES:
        assert GPAWCDFTSettings(scf_profile=profile).scf_profile == profile


def test_task_directory_binds_index_identity_and_signature(tmp_path):
    path = build_pilot_manifest(
        tmp_path / "pilot", charges=(1,), orientations=("plane_normal",)
    )
    task = load_pilot_manifest(path)["tasks"][0]
    directory = task_directory(path, task)
    assert directory.name.startswith("0000_c_q1_plane_normal_r12A_")
    assert directory.name.endswith(task["task_signature"][:12])
