from __future__ import annotations

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

from soft_dft.branch_solver import BranchValidationSettings, solve_cdft_branch
from soft_dft.cdft_branch import CalibrationSettings, ProbeRecord


def _settings() -> BranchValidationSettings:
    return BranchValidationSettings(
        calibration=CalibrationSettings(
            initial_step_hartree=1.0,
            expansion_factor=2.0,
            max_abs_strength_hartree=4.0,
            max_probe_count=6,
        ),
        population_tolerance_electrons=1.0e-5,
        endpoint_strength_tolerance_hartree=1.0e-9,
        endpoint_residual_tolerance_electrons=1.0e-8,
        endpoint_population_tolerance_electrons=1.0e-8,
        endpoint_lagrangian_tolerance_hartree=1.0e-8,
        reciprocal_strength_tolerance_hartree=1.0e-6,
        reciprocal_energy_tolerance_hartree=1.0e-7,
        reciprocal_population_tolerance_electrons=1.0e-6,
        reciprocal_spin_squared_tolerance=1.0e-6,
        reciprocal_density_relative_rms_tolerance=1.0e-6,
        curvature_delta_hartree=0.1,
        curvature_negative_margin_hartree=0.0,
    )


def _probe(strength: float, _label: str) -> ProbeRecord:
    # Concave W(lambda) with dW/dlambda = N(lambda)-target = 2-lambda.
    residual = 2.0 - strength
    return ProbeRecord(
        strength_hartree=strength,
        target_electrons=2.0,
        current_electrons=2.0 + residual,
        residual_electrons=residual,
        energy_hartree=10.0 - 0.5 * (strength - 2.0) ** 2,
        inner_scf_converged=True,
        electron_count_alpha=6,
        electron_count_beta=6,
        spin_squared=0.0,
        branch_fingerprint="same",
    )


def _root(direction, start, opposite, step):
    assert start.strength_hartree - step * start.residual_electrons == pytest.approx(
        opposite.strength_hartree
    )
    return {
        "status": "complete",
        "valid_completion": True,
        "cdft_outer_converged": True,
        "last_inner_scf_converged": True,
        "normal_end": True,
        "cdft_trace": [
            {
                "strength": start.strength_hartree,
                "target": 2.0,
                "current": 2.0 + start.residual_electrons,
                "residual": start.residual_electrons,
                "energy": 10.0
                - 0.5 * (start.strength_hartree - 2.0) ** 2,
                "inner_scf_converged": True,
            },
            {
                "strength": opposite.strength_hartree,
                "target": 2.0,
                "current": 2.0 + opposite.residual_electrons,
                "residual": opposite.residual_electrons,
                "energy": 10.0
                - 0.5 * (opposite.strength_hartree - 2.0) ** 2,
                "inner_scf_converged": True,
            },
            {
                "strength": 2.0,
                "target": 2.0,
                "current": 2.0,
                "residual": 0.0,
                "energy": 10.0,
                "inner_scf_converged": True,
            },
        ],
        "cdft_strength": 2.0,
        "cdft_target_electrons": 2.0,
        "cdft_current_electrons": 2.0,
        "cdft_deviation_electrons": 0.0,
        # This unconstrained ENERGY| diagnostic is deliberately distinct from
        # the CDFT trace's constrained Lagrangian and must not enter its gates.
        "energy_hartree": -123.0 if direction == "lower" else 456.0,
        "scf_spin_mode": "RESTRICTED",
        "spin_squared_single_determinant": 0.0,
        "direction": direction,
    }


def test_solver_brackets_injects_both_directions_and_publishes_validation(tmp_path):
    calls: list[tuple[float, str]] = []

    def probe(strength, label):
        calls.append((strength, label))
        return _probe(strength, label)

    result = solve_cdft_branch(
        tmp_path / "branch.json",
        solver_signature="task-signature",
        settings=_settings(),
        calibration_center_hartree=0.0,
        calibration_center_probe=False,
        probe_callback=probe,
        root_callback=_root,
        density_comparison_callback=lambda _a, _b: {"relative_rms": 0.0},
        progress=False,
    )
    assert result["status"] == "validated"
    assert result["validation_scope"] == "reciprocal_numerical_cdft_root"
    assert result["physical_state_status"] == "validation_pending"
    assert "does not establish" in result["validation_limitation"]
    assert set(result["roots"]) == {"lower", "upper"}
    assert result["curvature_validation"][
        "local_response_electrons_per_hartree"
    ] == pytest.approx(-1.0)
    assert result["reciprocal_validation"]["differences"][
        "constrained_lagrangian_hartree"
    ] == pytest.approx(0.0)
    assert result["curvature_validation"][
        "second_differences_hartree"
    ] == {"lower": pytest.approx(-0.01), "upper": pytest.approx(-0.01)}
    prior_calls = list(calls)
    resumed = solve_cdft_branch(
        tmp_path / "branch.json",
        solver_signature="task-signature",
        settings=_settings(),
        calibration_center_hartree=0.0,
        calibration_center_probe=False,
        probe_callback=probe,
        root_callback=_root,
        density_comparison_callback=lambda _a, _b: {"relative_rms": 0.0},
        progress=False,
    )
    assert resumed == result
    assert calls == prior_calls


def test_restricted_reciprocal_roots_may_both_omit_s_squared(tmp_path):
    def rks_root(direction, start, opposite, step):
        result = dict(_root(direction, start, opposite, step))
        result["spin_squared_single_determinant"] = None
        return result

    result = solve_cdft_branch(
        tmp_path / "rks.json",
        solver_signature="rks-no-s2",
        settings=_settings(),
        calibration_center_hartree=0.0,
        calibration_center_probe=False,
        probe_callback=_probe,
        root_callback=rks_root,
        density_comparison_callback=lambda _a, _b: {"relative_rms": 0.0},
        progress=False,
    )
    assert result["status"] == "validated"
    assert result["reciprocal_validation"]["differences"]["spin_squared"] is None


def test_solver_rejects_false_bisect_history_and_never_validates(tmp_path):
    def bad_root(direction, start, opposite, step):
        result = dict(_root(direction, start, opposite, step))
        result["cdft_trace"] = [result["cdft_trace"][0], result["cdft_trace"][0]]
        return result

    with pytest.raises(ValueError, match="both endpoints"):
        solve_cdft_branch(
            tmp_path / "branch.json",
            solver_signature="task-signature",
            settings=_settings(),
            calibration_center_hartree=0.0,
            calibration_center_probe=False,
            probe_callback=_probe,
            root_callback=bad_root,
            density_comparison_callback=lambda _a, _b: {"relative_rms": 0.0},
            progress=False,
        )
    assert '"status": "validated"' not in (tmp_path / "branch.json").read_text()


@pytest.mark.parametrize(
    ("field", "match"),
    (
        ("residual", "residual was not reproduced"),
        ("current", "population was not reproduced"),
        ("energy", "Lagrangian was not reproduced"),
    ),
)
def test_injected_trace_must_reproduce_calibrated_endpoint_observables(
    tmp_path, field, match
):
    def switched_root(direction, start, opposite, step):
        result = dict(_root(direction, start, opposite, step))
        trace = [dict(point) for point in result["cdft_trace"]]
        trace[0][field] += 0.01
        result["cdft_trace"] = trace
        return result

    with pytest.raises(ValueError, match=match):
        solve_cdft_branch(
            tmp_path / f"{field}.json",
            solver_signature=f"endpoint-{field}",
            settings=_settings(),
            calibration_center_hartree=0.0,
            calibration_center_probe=False,
            probe_callback=_probe,
            root_callback=switched_root,
            density_comparison_callback=lambda _a, _b: {"relative_rms": 0.0},
            progress=False,
        )


def test_exact_zero_probe_is_not_misused_as_a_bisect_endpoint(tmp_path):
    result = solve_cdft_branch(
        tmp_path / "branch.json",
        solver_signature="exact-zero",
        settings=_settings(),
        calibration_center_hartree=0.0,
        calibration_center_probe=False,
        probe_callback=_probe,
        root_callback=_root,
        density_comparison_callback=lambda _a, _b: {"relative_rms": 0.0},
        progress=False,
    )
    assert result["calibration_probes"]["calibration_002"][
        "residual_electrons"
    ] == 0.0
    assert result["bracket"]["lower"]["strength_hartree"] == 1.0
    assert result["bracket"]["upper"]["strength_hartree"] == 4.0


def test_calibration_is_centered_on_signature_bound_parent_multiplier(tmp_path):
    calls: list[float] = []

    def probe(strength, label):
        calls.append(strength)
        return _probe(strength, label)

    checkpoint = tmp_path / "centered.json"
    result = solve_cdft_branch(
        checkpoint,
        solver_signature="centered",
        settings=_settings(),
        calibration_center_hartree=2.0,
        calibration_center_probe=True,
        probe_callback=probe,
        root_callback=_root,
        density_comparison_callback=lambda _a, _b: {"relative_rms": 0.0},
        progress=False,
    )
    assert result["calibration_center_hartree"] == 2.0
    assert [
        result["calibration_probes"][label]["strength_hartree"]
        for label in sorted(result["calibration_probes"])
    ] == [3.0, 1.0, 2.0]
    assert calls[:3] == [2.0, 3.0, 1.0]

    with pytest.raises(RuntimeError, match="calibration center"):
        solve_cdft_branch(
            checkpoint,
            solver_signature="centered",
            settings=_settings(),
            calibration_center_hartree=2.1,
            calibration_center_probe=True,
            probe_callback=probe,
            root_callback=_root,
            density_comparison_callback=lambda _a, _b: {"relative_rms": 0.0},
            progress=False,
        )


def test_solver_requires_explicit_root_gates_and_same_sector_probes(tmp_path):
    def partial_root(direction, start, opposite, step):
        result = dict(_root(direction, start, opposite, step))
        result["status"] = "partial"
        return result

    with pytest.raises(ValueError, match="completion gates"):
        solve_cdft_branch(
            tmp_path / "partial.json",
            solver_signature="partial",
            settings=_settings(),
            calibration_center_hartree=0.0,
            calibration_center_probe=False,
            probe_callback=_probe,
            root_callback=partial_root,
            density_comparison_callback=lambda _a, _b: {"relative_rms": 0.0},
            progress=False,
        )

    def switched_probe(strength, label):
        record = _probe(strength, label)
        return ProbeRecord(
            **{
                **record.as_dict(),
                "branch_fingerprint": (
                    "positive-residual"
                    if record.residual_electrons >= 0.0
                    else "negative-residual"
                ),
            }
        )

    with pytest.raises(RuntimeError, match="without a bracket"):
        solve_cdft_branch(
            tmp_path / "branch-switch.json",
            solver_signature="branch-switch",
            settings=_settings(),
            calibration_center_hartree=0.0,
            calibration_center_probe=False,
            probe_callback=switched_probe,
            root_callback=_root,
            density_comparison_callback=lambda _a, _b: {"relative_rms": 0.0},
            progress=False,
        )


def test_solver_rejects_incompatible_resume_and_density_hysteresis(tmp_path):
    checkpoint = tmp_path / "branch.json"
    solve_cdft_branch(
        checkpoint,
        solver_signature="first",
        settings=_settings(),
        calibration_center_hartree=0.0,
        calibration_center_probe=False,
        probe_callback=_probe,
        root_callback=_root,
        density_comparison_callback=lambda _a, _b: {"relative_rms": 0.0},
        progress=False,
    )
    with pytest.raises(RuntimeError, match="another task"):
        solve_cdft_branch(
            checkpoint,
            solver_signature="second",
            settings=_settings(),
            calibration_center_hartree=0.0,
            calibration_center_probe=False,
            probe_callback=_probe,
            root_callback=_root,
            density_comparison_callback=lambda _a, _b: {"relative_rms": 0.0},
            progress=False,
        )
    changed_settings = BranchValidationSettings(
        **{
            **_settings().as_dict(),
            "calibration": _settings().calibration,
            "curvature_delta_hartree": 0.2,
        }
    )
    with pytest.raises(RuntimeError, match="settings are incompatible"):
        solve_cdft_branch(
            checkpoint,
            solver_signature="first",
            settings=changed_settings,
            calibration_center_hartree=0.0,
            calibration_center_probe=False,
            probe_callback=_probe,
            root_callback=_root,
            density_comparison_callback=lambda _a, _b: {"relative_rms": 0.0},
            progress=False,
        )

    with pytest.raises(ValueError, match="different electronic densities"):
        solve_cdft_branch(
            tmp_path / "density.json",
            solver_signature="density",
            settings=_settings(),
            calibration_center_hartree=0.0,
            calibration_center_probe=False,
            probe_callback=_probe,
            root_callback=_root,
            density_comparison_callback=lambda _a, _b: {"relative_rms": 1.0e-3},
            progress=False,
        )


def test_solver_requires_explicit_positive_tolerances():
    with pytest.raises(ValueError, match="density tolerance"):
        BranchValidationSettings(
            **{
                **_settings().as_dict(),
                "calibration": _settings().calibration,
                "reciprocal_density_relative_rms_tolerance": 0.0,
            }
        )
