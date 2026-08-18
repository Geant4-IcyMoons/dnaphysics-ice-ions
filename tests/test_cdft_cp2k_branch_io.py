from __future__ import annotations

from dataclasses import replace
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

from soft_dft.config import DEFAULT_CP2K_SETTINGS
from soft_dft.cp2k import (
    CDFT_MODE_FIXED_LAMBDA,
    CDFT_MODE_OPTIMIZED,
    COMPLEX_ROLE,
    parse_cp2k_output,
    render_cp2k_input,
)


def _complex_task(
    *, multiplicity: int = 1, spin_mode: str = "RESTRICTED"
) -> dict[str, object]:
    return {
        "task_id": "C_q2_oxygen_back_r000_complex",
        "role": COMPLEX_ROLE,
        "charge": 2,
        "multiplicity": multiplicity,
        "electrons_on_projectile": 4,
        "scf_spin_mode": spin_mode,
        "coordinates_angstrom": [
            ["C", 8.0, 0.0, 0.0],
            ["O", 0.0, 0.0, 0.0],
            ["H", 0.9572, 0.0, 0.0],
            ["H", -0.2399872, 0.927297, 0.0],
        ],
    }


def test_optimized_cdft_requires_explicit_lambda_and_step_size() -> None:
    task = _complex_task()
    with pytest.raises(ValueError, match="strength must be explicit"):
        render_cp2k_input(task, DEFAULT_CP2K_SETTINGS, cdft_step_size=0.25)
    with pytest.raises(ValueError, match="explicit STEP_SIZE"):
        render_cp2k_input(task, DEFAULT_CP2K_SETTINGS, cdft_strength=1.75)

    rendered = render_cp2k_input(
        task,
        DEFAULT_CP2K_SETTINGS,
        cdft_strength=1.75,
        cdft_step_size=0.25,
    )
    assert "STRENGTH 1.75" in rendered
    assert "STEP_SIZE 0.25" in rendered
    assert "STEP_SIZE -1" not in rendered
    assert "MAX_SCF 60" in rendered
    assert "UKS FALSE" in rendered


def test_fixed_lambda_keeps_cdft_active_but_disables_outer_optimization() -> None:
    rendered = render_cp2k_input(
        _complex_task(),
        DEFAULT_CP2K_SETTINGS,
        cdft_strength=2.5,
        cdft_mode=CDFT_MODE_FIXED_LAMBDA,
    )
    assert "&CDFT" in rendered
    assert "STRENGTH 2.5" in rendered
    assert "TYPE CDFT_CONSTRAINT" in rendered
    assert "MAX_SCF 0" in rendered
    assert "OPTIMIZER BISECT" in rendered
    assert "STEP_SIZE" not in rendered
    with pytest.raises(ValueError, match="does not use STEP_SIZE"):
        render_cp2k_input(
            _complex_task(),
            DEFAULT_CP2K_SETTINGS,
            cdft_strength=2.5,
            cdft_mode=CDFT_MODE_FIXED_LAMBDA,
            cdft_step_size=0.25,
        )


def test_unconstrained_complex_input_contains_no_cdft_section() -> None:
    rendered = render_cp2k_input(
        _complex_task(),
        DEFAULT_CP2K_SETTINGS,
        enable_cdft=False,
    )
    assert "&CDFT" not in rendered
    assert "CDFT_CONSTRAINT" not in rendered
    assert "STRENGTH" not in rendered
    assert "SCF_GUESS ATOMIC" in rendered
    assert "UKS FALSE" in rendered


def test_unconstrained_ot_diagnostic_uses_reviewed_difficult_scf_route() -> None:
    settings = replace(DEFAULT_CP2K_SETTINGS, complex_scf_solver="OT")
    rendered = render_cp2k_input(
        _complex_task(),
        settings,
        enable_cdft=False,
    )
    assert "&CDFT" not in rendered
    assert "&MIXING" not in rendered
    assert "&OT ON" in rendered
    assert "MINIMIZER CG" in rendered
    assert "LINESEARCH 3PNT" in rendered
    assert "PRECONDITIONER FULL_ALL" in rendered
    assert "ENERGY_GAP 0.001" in rendered
    assert "ALGORITHM STRICT" in rendered
    assert "SCF_GUESS ATOMIC" in rendered


@pytest.mark.parametrize(
    ("linesearch", "preconditioner", "expected_gap"),
    [
        ("GOLD", "FULL_ALL", True),
        ("ADAPT", "FULL_ALL", True),
        ("GOLD", "FULL_KINETIC", False),
    ],
)
def test_unconstrained_ot_matrix_is_rendered_exactly(
    linesearch: str, preconditioner: str, expected_gap: bool
) -> None:
    settings = replace(
        DEFAULT_CP2K_SETTINGS,
        complex_scf_solver="OT",
        ot_linesearch=linesearch,
        ot_preconditioner=preconditioner,
    )
    rendered = render_cp2k_input(_complex_task(), settings, enable_cdft=False)
    assert f"LINESEARCH {linesearch}" in rendered
    assert f"PRECONDITIONER {preconditioner}" in rendered
    assert ("ENERGY_GAP 0.001" in rendered) is expected_gap
    assert "&MIXING" not in rendered
    assert "&CDFT" not in rendered


def test_unconstrained_direct_mixing_matrix_control_is_rendered_exactly() -> None:
    settings = replace(
        DEFAULT_CP2K_SETTINGS,
        complex_scf_solver="DIAGONALIZATION",
        complex_mixing_method="DIRECT_P_MIXING",
        complex_mixing_alpha=0.2,
    )
    rendered = render_cp2k_input(_complex_task(), settings, enable_cdft=False)
    assert "&DIAGONALIZATION ON" in rendered
    assert "ALGORITHM STANDARD" in rendered
    assert "METHOD DIRECT_P_MIXING" in rendered
    assert "ALPHA 0.2" in rendered
    assert "NPULAY" not in rendered
    assert "&OT" not in rendered
    assert "&CDFT" not in rendered


def test_task_spin_mode_controls_uks_and_restricted_requires_singlet() -> None:
    restricted = render_cp2k_input(
        _complex_task(multiplicity=1, spin_mode="RESTRICTED"),
        DEFAULT_CP2K_SETTINGS,
        cdft_strength=1.0,
        cdft_step_size=0.2,
    )
    unrestricted = render_cp2k_input(
        _complex_task(multiplicity=2, spin_mode="UNRESTRICTED"),
        DEFAULT_CP2K_SETTINGS,
        cdft_strength=1.0,
        cdft_step_size=0.2,
    )
    assert "UKS FALSE" in restricted
    assert "UKS TRUE" in unrestricted
    with pytest.raises(ValueError, match="requires multiplicity 1"):
        render_cp2k_input(
            _complex_task(multiplicity=2, spin_mode="RESTRICTED"),
            DEFAULT_CP2K_SETTINGS,
            cdft_strength=1.0,
            cdft_step_size=0.2,
        )


def test_density_cube_is_an_explicit_validation_only_output() -> None:
    plain = render_cp2k_input(
        _complex_task(),
        DEFAULT_CP2K_SETTINGS,
        cdft_strength=1.0,
        cdft_step_size=0.2,
    )
    rendered = render_cp2k_input(
        _complex_task(),
        DEFAULT_CP2K_SETTINGS,
        cdft_strength=1.0,
        cdft_step_size=0.2,
        density_cube_stride=4,
    )
    assert "&E_DENSITY_CUBE" not in plain
    assert "DENSITY_INCLUDE TOTAL_HARD_APPROX" in rendered
    assert "STRIDE 4 4 4" in rendered
    with pytest.raises(ValueError, match="Density-cube stride"):
        render_cp2k_input(
            _complex_task(),
            DEFAULT_CP2K_SETTINGS,
            cdft_strength=1.0,
            cdft_step_size=0.2,
            density_cube_stride=0,
        )
    with pytest.raises(ValueError, match="scf_spin_mode"):
        render_cp2k_input(
            _complex_task(spin_mode="BROKEN"),
            DEFAULT_CP2K_SETTINGS,
            cdft_strength=1.0,
            cdft_step_size=0.2,
        )


OPTIMIZED_OUTPUT = """
 CP2K| version string:                                       CP2K version 2025.2
 CP2K| source code revision number:                                   c3a8adfec5
 SCF WAVEFUNCTION OPTIMIZATION
 *** SCF run converged in 9 steps ***
 CDFT SCF iter =     1 RMS gradient = 2.0E-02 energy = -75.1000000000
 Target value of constraint  : 4.000000000000
 Current value of constraint : 4.020000000000
 Deviation from target       : 2.000E-02
 Strength of constraint      : 1.500000000000
 SCF WAVEFUNCTION OPTIMIZATION
 *** SCF run converged in 7 steps ***
 CDFT SCF iter =     2 RMS gradient = 2.0E-06 energy = -75.1234567890
 CDFT SCF loop converged in 2 iterations or 16 steps
 Target value of constraint  : 4.000000000000
 Current value of constraint : 4.000002000000
 Deviation from target       : 2.000E-06
 Strength of constraint      : 1.725000000000
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -75.123456789
 PROGRAM ENDED AT 2026-08-09 18:00:00
"""


def test_parser_returns_ordered_branch_trace_and_all_acceptance_gates() -> None:
    result = parse_cp2k_output(
        OPTIMIZED_OUTPUT,
        require_cdft=True,
        cdft_mode=CDFT_MODE_OPTIMIZED,
        cdft_tolerance=1.0e-5,
    )
    assert result["cp2k_version"] == "CP2K version 2025.2"
    assert result["cp2k_source_revision"] == "c3a8adfec5"
    assert result["last_inner_scf_converged"] is True
    assert result["cdft_outer_converged"] is True
    assert result["cdft_target_met"] is True
    assert result["normal_end"] is True
    assert result["valid_completion"] is True
    assert [entry["iteration"] for entry in result["cdft_trace"]] == [1, 2]
    assert result["cdft_trace"][0] == {
        "iteration": 1,
        "energy": pytest.approx(-75.1),
        "target": pytest.approx(4.0),
        "current": pytest.approx(4.02),
        "residual": pytest.approx(2.0e-2),
        "strength": pytest.approx(1.5),
        "inner_scf_converged": True,
    }
    assert result["cdft_trace"][1]["strength"] == pytest.approx(1.725)


@pytest.mark.parametrize(
    ("text", "failed_gate"),
    [
        (
            OPTIMIZED_OUTPUT.replace(
                "*** SCF run converged in 7 steps ***",
                "*** SCF run NOT converged after 100 steps ***",
            ),
            "last_inner_scf_converged",
        ),
        (
            OPTIMIZED_OUTPUT.replace(
                "CDFT SCF loop converged in 2 iterations or 16 steps", ""
            ),
            "cdft_outer_converged",
        ),
        (
            OPTIMIZED_OUTPUT.replace(
                "Deviation from target       : 2.000E-06",
                "Deviation from target       : 2.000E-04",
            ),
            "cdft_target_met",
        ),
        (
            OPTIMIZED_OUTPUT.replace("PROGRAM ENDED AT", "PROGRAM ABORTED AT"),
            "normal_end",
        ),
    ],
)
def test_optimized_acceptance_rejects_each_failed_gate(
    text: str, failed_gate: str
) -> None:
    result = parse_cp2k_output(text, require_cdft=True, cdft_tolerance=1.0e-5)
    assert result[failed_gate] is False
    assert result["valid_completion"] is False


def test_fixed_lambda_probe_has_separate_nonproduction_acceptance() -> None:
    probe_output = OPTIMIZED_OUTPUT.replace(
        "CDFT SCF loop converged in 2 iterations or 16 steps", ""
    ).replace(
        "Deviation from target       : 2.000E-06",
        "Deviation from target       : -0.125",
    )
    result = parse_cp2k_output(
        probe_output,
        require_cdft=True,
        cdft_mode=CDFT_MODE_FIXED_LAMBDA,
    )
    assert result["fixed_lambda_probe_accepted"] is True
    assert result["cdft_target_met"] is False
    assert result["valid_completion"] is False


def test_unfinished_latest_inner_scf_overrides_an_earlier_success() -> None:
    unfinished = OPTIMIZED_OUTPUT.replace(
        " PROGRAM ENDED AT 2026-08-09 18:00:00",
        " SCF WAVEFUNCTION OPTIMIZATION\n",
    )
    result = parse_cp2k_output(unfinished, require_cdft=True)
    assert result["last_inner_scf_converged"] is None
    assert result["scf_converged"] is False
    assert result["valid_completion"] is False
