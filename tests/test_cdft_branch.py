from __future__ import annotations

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

from soft_dft.cdft_branch import (
    CalibrationSettings,
    PHYSICAL_STATE_STATUS,
    ProbeRecord,
    RECIPROCAL_VALIDATION_LIMITATION,
    RECIPROCAL_VALIDATION_SCOPE,
    bisect_injection_step_size,
    build_state_identity,
    calibration_strengths,
    continuation_midpoint,
    find_negative_response_bracket,
    load_validated_state,
    publish_validated_state,
    state_identity_compatible,
)


def _validated_branch() -> dict[str, object]:
    return {
        "status": "validated",
        "directions": ["lower", "upper"],
        "validation_scope": RECIPROCAL_VALIDATION_SCOPE,
        "physical_state_status": PHYSICAL_STATE_STATUS,
        "validation_limitation": RECIPROCAL_VALIDATION_LIMITATION,
    }


def _probe(strength: float, current: float, **kwargs: object) -> ProbeRecord:
    return ProbeRecord(
        strength_hartree=strength,
        target_electrons=2.0,
        current_electrons=current,
        residual_electrons=current - 2.0,
        energy_hartree=-100.0 + 0.01 * strength,
        inner_scf_converged=bool(kwargs.pop("inner_scf_converged", True)),
        **kwargs,
    )


def _task(separation: float = 6.0) -> dict[str, object]:
    return {
        "task_id": "C_q4_oxygen_back_r000_complex",
        "projectile": "C",
        "charge": 4,
        "electrons_on_projectile": 2,
        "multiplicity": 1,
        "scf_spin_mode": "RESTRICTED",
        "orientation": "oxygen_back",
        "separation_angstrom": separation,
        "coordinates_angstrom": [
            ["C", 0.0, 0.0, -separation],
            ["O", 0.0, 0.0, 0.0],
            ["H", 0.75, 0.0, 0.58],
            ["H", -0.75, 0.0, 0.58],
        ],
    }


def test_calibration_grid_is_explicit_two_sided_and_contains_no_3ha_seed():
    settings = CalibrationSettings(
        initial_step_hartree=0.25,
        expansion_factor=2.0,
        max_abs_strength_hartree=2.0,
        max_probe_count=8,
    )
    assert calibration_strengths(settings) == (
        0.25,
        -0.25,
        0.5,
        -0.5,
        1.0,
        -1.0,
        2.0,
        -2.0,
    )
    assert 0.0 not in calibration_strengths(settings)
    assert 3.0 not in calibration_strengths(settings)


def test_zero_strength_can_only_be_an_explicit_diagnostic_probe():
    settings = CalibrationSettings(
        initial_step_hartree=0.5,
        expansion_factor=2.0,
        max_abs_strength_hartree=1.0,
        max_probe_count=5,
        include_zero_diagnostic=True,
    )
    assert calibration_strengths(settings)[0] == 0.0


def test_bracket_requires_inner_convergence_same_sector_and_negative_response():
    records = [
        _probe(0.0, 3.0, branch_fingerprint="A"),
        _probe(1.0, 2.2, inner_scf_converged=False, branch_fingerprint="A"),
        _probe(2.0, 1.5, branch_fingerprint="A"),
    ]
    bracket = find_negative_response_bracket(records)
    assert bracket.lower.strength_hartree == 0.0
    assert bracket.upper.strength_hartree == 2.0
    assert bracket.secant_response_electrons_per_hartree == pytest.approx(-0.75)

    with pytest.raises(ValueError, match="No same-sector"):
        find_negative_response_bracket(
            [
                _probe(0.0, 3.0, branch_fingerprint="A"),
                _probe(2.0, 1.5, branch_fingerprint="B"),
            ]
        )
    with pytest.raises(ValueError, match="negative dN"):
        find_negative_response_bracket([_probe(0.0, 1.0), _probe(2.0, 3.0)])


def test_bisect_step_injects_external_endpoint_using_cp2k_sd_equation():
    lower = _probe(1.0, 2.5)
    upper = _probe(3.0, 1.5)
    step = bisect_injection_step_size(lower, upper)
    assert lower.strength_hartree - step * lower.residual_electrons == pytest.approx(
        upper.strength_hartree
    )
    reverse = bisect_injection_step_size(upper, lower)
    assert upper.strength_hartree - reverse * upper.residual_electrons == pytest.approx(
        lower.strength_hartree
    )


def test_state_identity_separates_family_from_geometry():
    settings = {
        "cp2k_series": "2025.2",
        "cdft_constraint_type": "HIRSHFELD",
        "cell_angstrom": 30.0,
    }
    first = build_state_identity(
        _task(6.0), configuration_signature="config", cp2k_settings=settings
    )
    second = build_state_identity(
        _task(5.0), configuration_signature="config", cp2k_settings=settings
    )
    assert first["family_signature"] == second["family_signature"]
    assert first["geometry_signature"] != second["geometry_signature"]
    assert state_identity_compatible(first, second, same_geometry=False)
    assert not state_identity_compatible(first, second, same_geometry=True)

    changed = build_state_identity(
        {**_task(5.0), "charge": 3, "electrons_on_projectile": 3},
        configuration_signature="config",
        cp2k_settings=settings,
    )
    assert not state_identity_compatible(first, changed, same_geometry=False)


def test_only_reciprocally_validated_pair_is_published_and_checksum_loaded(tmp_path):
    source_wfn = tmp_path / "source.wfn"
    input_path = tmp_path / "input.inp"
    output_path = tmp_path / "cp2k.out"
    source_wfn.write_bytes(b"paired-wavefunction")
    input_path.write_text("input", encoding="utf-8")
    output_path.write_text("output", encoding="utf-8")
    identity = build_state_identity(
        _task(),
        configuration_signature="configuration",
        cp2k_settings={
            "cp2k_series": "2025.2",
            "cdft_constraint_type": "HIRSHFELD",
        },
    )
    state_path = tmp_path / "accepted" / "state.json"
    with pytest.raises(ValueError, match="Reciprocal"):
        publish_validated_state(
            state_path,
            source_wfn,
            identity=identity,
            multiplier_hartree=2.1,
            target_electrons=2.0,
            current_electrons=2.0,
            residual_electrons=0.0,
            energy_hartree=-100.0,
            electron_count_alpha=6,
            electron_count_beta=6,
            spin_squared=0.0,
            cdft_trace=[],
            branch_validation={"status": "pending"},
            cp2k_identity={"version": "2025.2"},
            input_path=input_path,
            output_path=output_path,
            parent_state_id=None,
        )

    payload = publish_validated_state(
        state_path,
        source_wfn,
        identity=identity,
        multiplier_hartree=2.1,
        target_electrons=2.0,
        current_electrons=2.0,
        residual_electrons=0.0,
        energy_hartree=-100.0,
        electron_count_alpha=6,
        electron_count_beta=6,
        spin_squared=0.0,
        cdft_trace=[{"iteration": 4, "strength": 2.1, "residual": 0.0}],
        branch_validation=_validated_branch(),
        cp2k_identity={"version": "2025.2", "revision": "test"},
        input_path=input_path,
        output_path=output_path,
        parent_state_id=None,
    )
    assert load_validated_state(
        state_path, expected_identity=identity
    )["state_id"] == payload["state_id"]
    assert payload["input"]["path"] == "state.inp"
    assert payload["output"]["path"] == "state.out"
    assert payload["validation_scope"] == RECIPROCAL_VALIDATION_SCOPE
    assert payload["physical_state_status"] == PHYSICAL_STATE_STATUS
    assert payload["validation_limitation"] == RECIPROCAL_VALIDATION_LIMITATION
    assert (state_path.parent / "state.inp").read_text(encoding="utf-8") == "input"
    assert (state_path.parent / "state.out").read_text(encoding="utf-8") == "output"

    repeated = publish_validated_state(
        state_path,
        source_wfn,
        identity=identity,
        multiplier_hartree=2.1,
        target_electrons=2.0,
        current_electrons=2.0,
        residual_electrons=0.0,
        energy_hartree=-100.0,
        electron_count_alpha=6,
        electron_count_beta=6,
        spin_squared=0.0,
        cdft_trace=[{"iteration": 4, "strength": 2.1, "residual": 0.0}],
        branch_validation=_validated_branch(),
        cp2k_identity={"version": "2025.2", "revision": "test"},
        input_path=input_path,
        output_path=output_path,
        parent_state_id=None,
    )
    assert repeated["state_id"] == payload["state_id"]

    published_input = state_path.parent / "state.inp"
    published_input.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="input checksum"):
        load_validated_state(state_path)
    published_input.write_bytes(b"input")

    published_output = state_path.parent / "state.out"
    published_output.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="output checksum"):
        load_validated_state(state_path)
    published_output.write_bytes(b"output")

    (state_path.parent / "state.wfn").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="wavefunction checksum"):
        load_validated_state(state_path)


def test_partial_or_mismatched_state_records_are_rejected(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "partial",
                "identity": {},
                "state_id": "not-a-valid-checksum",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="record checksum"):
        load_validated_state(state_path)


def test_continuation_refines_the_actual_interval_not_a_fixed_ladder():
    assert continuation_midpoint(
        5.75, 4.9, minimum_step_angstrom=0.05
    ) == pytest.approx(5.325)
    with pytest.raises(ValueError, match="step limit"):
        continuation_midpoint(5.0, 4.98, minimum_step_angstrom=0.05)
