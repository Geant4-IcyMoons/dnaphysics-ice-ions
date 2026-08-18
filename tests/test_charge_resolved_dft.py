from __future__ import annotations

from dataclasses import replace
import hashlib
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

from soft_dft.config import (
    DEFAULT_CP2K_SETTINGS,
    DEFAULT_PROJECTILE,
    load_projectile_definition,
)
from ion_ice import projectile_registry
from soft_dft.adaptive import (
    AdaptiveSettings,
    interval_interpolation_report,
    point_key,
    refine_completed_intervals,
    required_probe_geometries,
)
from soft_dft.cp2k import (
    CDFT_MODE_FIXED_LAMBDA,
    COMPLEX_ROLE,
    PROJECTILE_COUNTERPOISE_ROLE,
    WATER_COUNTERPOISE_ROLE,
    parse_cp2k_output,
    render_cp2k_input,
)
from soft_dft.geometry import ORIENTATIONS, build_scan_geometries, scan_distances
from soft_dft.workflow import (
    VALIDATED_BRANCH_EXECUTION,
    VALIDATED_BRANCH_RUNNER,
    audit_workflow_electronic_states,
    build_workflow,
    collect_workflow,
    load_workflow_manifest,
    pending_workflow_tasks,
    reuse_compatible_workflow_results,
    run_workflow_tasks,
)
import adaptive_charge_resolved_dft as adaptive_controller


def _write_lithium_definition(path: Path, *, omit_charge: int | None = None) -> Path:
    states = [
        {
            "charge": charge,
            "electrons_on_projectile": 3 - charge,
            "multiplicity": (2, 1, 2, 1)[charge],
            "configuration": ("1s2 2s1", "1s2", "1s1", "bare nucleus")[charge],
            "term": ("2S1/2", "1S0", "2S1/2", "1S0")[charge],
        }
        for charge in range(4)
        if charge != omit_charge
    ]
    payload = {
        "schema_version": 1,
        "symbol": "Li",
        "name": "lithium",
        "atomic_number": 3,
        "aliases": ["li", "lithium"],
        "default_isotope": {
            "mass_number": 7,
            "neutral_atomic_mass_u": 7.0160034366,
            "provenance": {
                "source": "test fixture only",
                "url": "https://example.invalid/test-mass",
            },
        },
        "components": {
            "nlh_pair_potential": {
                "status": "missing",
                "reason": "Test fixture has no NLH data.",
            },
            "hard_transport": {
                "status": "missing",
                "reason": "Test fixture has no hard transport.",
            },
            "soft_dft": {
                "status": "implemented",
                "complete_charge_ladder": True,
                "states": states,
                "state_provenance": {
                    "source": "test fixture only",
                    "url": "https://example.invalid/test-fixture",
                },
                "scan_grid": {
                    "kind": "explicit",
                    "distances_angstrom": {
                        "H": [1.0, 2.0, 4.0],
                        "O": [1.1, 2.1, 4.1],
                    },
                    "provenance": {
                        "source": "test fixture only",
                        "url": "https://example.invalid/test-grid",
                    },
                },
                "cp2k": {
                    "projectile_basis_set": "TEST-AE-BASIS",
                    "basis_file": "TEST-BASIS-FILE",
                    "basis_provenance": {
                        "source": "test fixture only",
                        "url": "https://example.invalid/test-basis",
                    },
                },
            },
            "geant4_hard_runtime": {
                "status": "missing",
                "reason": "Test fixture has no hard runtime.",
            },
            "geant4_soft_runtime": {
                "status": "missing",
                "reason": "Test fixture has no soft runtime.",
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_carbon_states_are_complete_ground_term_definitions():
    assert [state.charge for state in DEFAULT_PROJECTILE.states] == list(range(7))
    assert [
        state.electrons_on_projectile for state in DEFAULT_PROJECTILE.states
    ] == list(
        reversed(range(7))
    )
    assert [state.multiplicity for state in DEFAULT_PROJECTILE.states] == [
        3,
        2,
        1,
        2,
        1,
        2,
        1,
    ]
    assert DEFAULT_PROJECTILE.states[-1].configuration == "bare nucleus"


def test_every_registered_projectile_has_a_complete_sourced_charge_ladder():
    expected_multiplicities = {
        "H": [2, 1],
        "He": [1, 2, 1],
        "C": [3, 2, 1, 2, 1, 2, 1],
        "O": [3, 4, 3, 2, 1, 2, 1, 2, 1],
        "S": [3, 4, 3, 2, 1, 2, 1, 2, 3, 4, 3, 2, 1, 2, 1, 2, 1],
    }
    for symbol, projectile in projectile_registry().items():
        assert [state.charge for state in projectile.states] == list(
            range(projectile.atomic_number + 1)
        )
        assert [state.electrons_on_projectile for state in projectile.states] == list(
            reversed(range(projectile.atomic_number + 1))
        )
        assert [state.multiplicity for state in projectile.states] == (
            expected_multiplicities[symbol]
        )
        assert projectile.states[-1].configuration == "bare nucleus"
        assert projectile.scan_grid["kind"] == "nlh_overlap"
        assert projectile.projectile_basis_set == "TZVPP-MOLOPT-GGA-ae"
        assert projectile.basis_file == "BASIS_MOLOPT_UZH"


def test_external_ion_definition_drives_element_charge_ladder_and_grid(tmp_path):
    lithium = load_projectile_definition(_write_lithium_definition(tmp_path / "Li.json"))
    assert lithium.symbol == "Li"
    assert [state.charge for state in lithium.states] == [0, 1, 2, 3]
    assert scan_distances(lithium, "H") == (1.0, 2.0, 4.0)
    assert scan_distances(lithium, "O") == (1.1, 2.1, 4.1)
    geometries = build_scan_geometries(lithium)
    assert len(geometries) == 15
    assert {geometry.coordinates_angstrom[0][0] for geometry in geometries} == {
        "Li"
    }
    manifest_path = build_workflow(
        tmp_path / "lithium", projectile=lithium, charges=(0, 3)
    )
    manifest = load_workflow_manifest(manifest_path, verify_inputs=True)
    assert manifest["configuration"]["projectile"]["symbol"] == "Li"
    assert manifest["point_count"] == 30
    assert manifest["task_count"] == 75
    assert manifest["analytic_task_count"] == 15
    complex_task = next(
        task
        for task in manifest["tasks"]
        if task["role"] == COMPLEX_ROLE and task["charge"] == 0
    )
    assert complex_task["execution"] == VALIDATED_BRANCH_EXECUTION
    assert complex_task["validated_branch_runner"] == VALIDATED_BRANCH_RUNNER
    assert "input_path" not in complex_task
    assert "input_sha256" not in complex_task


def test_external_ion_definition_rejects_incomplete_charge_ladder(tmp_path):
    path = _write_lithium_definition(tmp_path / "Li.json", omit_charge=2)
    with pytest.raises(ValueError, match="complete ordered charge ladder"):
        load_projectile_definition(path)


def test_external_explicit_grid_collection_waits_for_validated_branches(tmp_path):
    lithium = load_projectile_definition(_write_lithium_definition(tmp_path / "Li.json"))
    manifest_path = build_workflow(
        tmp_path / "workflow",
        projectile=lithium,
        charges=(3,),
        geometries=build_scan_geometries(lithium)[:1],
    )
    manifest = load_workflow_manifest(manifest_path)
    for task in manifest["tasks"]:
        if task["execution"] == "analytic":
            continue
        if task["role"] == COMPLEX_ROLE:
            continue
        result = {
            "schema_version": manifest["schema_version"],
            "status": "complete",
            "execution": "cp2k",
            "task_id": task["task_id"],
            "configuration_signature": manifest["configuration_signature"],
            "input_sha256": task["input_sha256"],
            "energy_hartree": -2.0,
            "valid_completion": True,
        }
        path = manifest_path.parent / task["result_path"]
        path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="CP2KBranchExecutor"):
        collect_workflow(manifest_path)


def test_scan_contains_exact_nlh_overlap_boundaries_and_soft_probes():
    for anchor in ("H", "O"):
        distances = scan_distances(DEFAULT_PROJECTILE, anchor)
        assert distances == tuple(sorted(set(distances)))
        assert 1.0 in distances
        assert 6.0 in distances
        assert len(distances) == 12
    geometries = build_scan_geometries(DEFAULT_PROJECTILE)
    assert len(geometries) == len(ORIENTATIONS) * 12
    for geometry in geometries:
        if geometry.orientation in {"oxygen_back", "hydrogen_out"}:
            assert geometry.minimum_pair_distance_angstrom == pytest.approx(
                geometry.separation_angstrom, abs=2e-12
            )


def _task(role: str, charge: int = 5) -> dict[str, object]:
    geometry = build_scan_geometries(DEFAULT_PROJECTILE)[0]
    state = DEFAULT_PROJECTILE.states[charge]
    return {
        "task_id": f"test_q{charge}_{role}",
        "role": role,
        "charge": state.charge,
        "multiplicity": state.multiplicity,
        "scf_spin_mode": state.scf_spin_mode,
        "electrons_on_projectile": state.electrons_on_projectile,
        "cp2k_atomic_guess": state.atomic_guess_dict(),
        "coordinates_angstrom": [list(row) for row in geometry.coordinates_angstrom],
    }


def _render_test_complex(
    task: dict[str, object], settings=DEFAULT_CP2K_SETTINGS
) -> str:
    """Render explicit numerical values used only by input-format tests."""

    return render_cp2k_input(
        task,
        settings,
        cdft_strength=1.0,
        cdft_step_size=0.25,
    )


def test_carbon_q4_uses_the_declared_atomic_occupation_and_active_cdft():
    rendered = _render_test_complex(_task(COMPLEX_ROLE, charge=4))
    assert "&BS ON" in rendered
    expected_spin_shells = "N 2 2\n          L 0 1\n          NEL -2 -2"
    assert rendered.count(expected_spin_shells) == 2
    assert "&CDFT" in rendered
    assert "&OUTER_SCF ON" in rendered
    assert "TARGET 2" in rendered

    counterpoise = render_cp2k_input(
        _task(PROJECTILE_COUNTERPOISE_ROLE, charge=4), DEFAULT_CP2K_SETTINGS
    )
    assert "&BS ON" not in counterpoise


def test_carbon_q1_renders_cp2k_half_shift_convention_exactly():
    rendered = render_cp2k_input(
        _task(COMPLEX_ROLE, charge=1),
        DEFAULT_CP2K_SETTINGS,
        cdft_strength=0.25,
        cdft_mode=CDFT_MODE_FIXED_LAMBDA,
    )
    assert "&ALPHA\n          N 2\n          L 1\n          NEL 0" in rendered
    assert "&BETA\n          N 2\n          L 1\n          NEL -2" in rendered
    assert "CHARGE 1" in rendered
    assert "MULTIPLICITY 2" in rendered
    assert "UKS TRUE" in rendered
    assert "TARGET 5" in rendered
    assert "MAX_SCF 0" in rendered
    assert "OPTIMIZER BISECT" in rendered
    assert "STEP_SIZE" not in rendered


def test_cp2k_inputs_use_all_electron_gapw_and_separate_counterpoise_roles():
    complex_input = _render_test_complex(
        _task(COMPLEX_ROLE),
        replace(DEFAULT_CP2K_SETTINGS, complex_scf_solver="OT"),
    )
    assert "METHOD GAPW" in complex_input
    assert "POTENTIAL ALL" in complex_input
    assert "MINIMIZER CG" in complex_input
    assert "LINESEARCH 3PNT" in complex_input
    assert "ALGORITHM STRICT" in complex_input
    assert "MAX_SCF 24" in complex_input
    assert "BASIS_SET TZVPP-MOLOPT-GGA-ae" in complex_input
    assert "CHARGE 5" in complex_input
    assert "MULTIPLICITY 2" in complex_input
    assert "TARGET 1" in complex_input
    assert "TYPE_OF_CONSTRAINT HIRSHFELD" in complex_input
    assert "SHAPE_FUNCTION DENSITY" in complex_input
    assert "PERIODIC NONE" in complex_input

    carbon_cp = render_cp2k_input(
        _task(PROJECTILE_COUNTERPOISE_ROLE), DEFAULT_CP2K_SETTINGS
    )
    assert "&DIAGONALIZATION ON" in carbon_cp
    assert "ALGORITHM STANDARD" in carbon_cp
    assert "METHOD PULAY_MIXING" in carbon_cp
    assert "&OT ON" not in carbon_cp
    assert "&CDFT" not in carbon_cp
    assert "&KIND O_G" in carbon_cp
    assert "GHOST TRUE" in carbon_cp

    water_cp = render_cp2k_input(
        _task(WATER_COUNTERPOISE_ROLE), DEFAULT_CP2K_SETTINGS
    )
    assert "CHARGE 0" in water_cp
    assert "MULTIPLICITY 1" in water_cp
    assert "&OT ON" in water_cp
    assert "ALGORITHM IRAC" in water_cp
    assert "MAX_SCF 100" in water_cp
    assert "&DIAGONALIZATION ON" not in water_cp
    assert "&KIND P_G" in water_cp


def test_complex_ot_inner_cycle_is_explicit_and_validated():
    rendered = _render_test_complex(
        _task(COMPLEX_ROLE),
        replace(
            DEFAULT_CP2K_SETTINGS,
            complex_scf_solver="OT",
            complex_ot_inner_scf_max=32,
        ),
    )
    assert "MAX_SCF 32" in rendered
    with pytest.raises(ValueError, match="Complex OT inner-SCF"):
        replace(DEFAULT_CP2K_SETTINGS, complex_ot_inner_scf_max=0)


def test_counterpoise_diagonalization_does_not_change_cdft():
    settings = replace(
        DEFAULT_CP2K_SETTINGS,
        water_counterpoise_scf_solver="DIAGONALIZATION",
        projectile_counterpoise_scf_solver="DIAGONALIZATION",
        complex_scf_solver="OT",
    )
    carbon_cp = render_cp2k_input(
        _task(PROJECTILE_COUNTERPOISE_ROLE), settings
    )
    assert "&DIAGONALIZATION ON" in carbon_cp
    assert "ALGORITHM STANDARD" in carbon_cp
    assert "METHOD PULAY_MIXING" in carbon_cp
    assert "ALPHA 0.5" in carbon_cp
    assert "NPULAY 5" in carbon_cp
    assert "&OT ON" not in carbon_cp

    complex_input = _render_test_complex(_task(COMPLEX_ROLE), settings)
    assert "&OT ON" in complex_input
    assert "ALGORITHM STRICT" in complex_input
    assert "&DIAGONALIZATION ON" not in complex_input


def test_complex_diagonalization_is_an_explicit_signature_bound_option():
    settings = replace(
        DEFAULT_CP2K_SETTINGS,
        complex_scf_solver="DIAGONALIZATION",
    )
    complex_input = _render_test_complex(_task(COMPLEX_ROLE), settings)
    assert "&DIAGONALIZATION ON" in complex_input
    assert "METHOD PULAY_MIXING" in complex_input
    assert "&OUTER_SCF ON" in complex_input
    assert "&OT ON" not in complex_input


def test_complex_and_counterpoise_mixing_controls_are_independent():
    settings = replace(
        DEFAULT_CP2K_SETTINGS,
        complex_scf_solver="DIAGONALIZATION",
        complex_mixing_alpha=0.2,
        complex_mixing_npulay=7,
        complex_mixing_method="BROYDEN_MIXING",
        counterpoise_mixing_alpha=0.4,
        counterpoise_mixing_npulay=3,
    )
    complex_input = _render_test_complex(_task(COMPLEX_ROLE), settings)
    projectile_input = render_cp2k_input(
        _task(PROJECTILE_COUNTERPOISE_ROLE), settings
    )
    assert "METHOD BROYDEN_MIXING" in complex_input
    assert "ALPHA 0.2" in complex_input
    assert "NPULAY 7" in complex_input
    assert "ALPHA 0.4" in projectile_input
    assert "NPULAY 3" in projectile_input
    with pytest.raises(ValueError, match="Complex mixing alpha"):
        replace(DEFAULT_CP2K_SETTINGS, complex_mixing_alpha=0.0)
    with pytest.raises(ValueError, match="Complex NPULAY"):
        replace(DEFAULT_CP2K_SETTINGS, complex_mixing_npulay=0)
    with pytest.raises(ValueError, match="Unsupported complex mixing method"):
        replace(DEFAULT_CP2K_SETTINGS, complex_mixing_method="INVENTED")


def test_direct_complex_mixing_omits_irrelevant_history_buffer():
    settings = replace(
        DEFAULT_CP2K_SETTINGS,
        complex_scf_solver="DIAGONALIZATION",
        complex_mixing_method="DIRECT_P_MIXING",
        complex_mixing_alpha=0.4,
    )
    rendered = _render_test_complex(_task(COMPLEX_ROLE), settings)
    assert "METHOD DIRECT_P_MIXING" in rendered
    assert "ALPHA 0.4" in rendered
    assert "NPULAY" not in rendered


def test_bisect_cdft_optimizer_is_explicit_and_has_no_newton_options():
    settings = replace(DEFAULT_CP2K_SETTINGS, cdft_optimizer="BISECT")
    rendered = _render_test_complex(_task(COMPLEX_ROLE), settings)
    assert "OPTIMIZER BISECT" in rendered
    assert "BISECT_TRUST_COUNT 10" in rendered
    assert "&CDFT_OPT ON" not in rendered


def test_reviewed_cdft_uses_density_hirshfeld_and_bisect():
    rendered = _render_test_complex(_task(COMPLEX_ROLE))
    assert "TYPE_OF_CONSTRAINT HIRSHFELD" in rendered
    assert "&HIRSHFELD_CONSTRAINT" in rendered
    assert "SHAPE_FUNCTION DENSITY" in rendered
    assert "&BECKE_CONSTRAINT" not in rendered
    assert "OPTIMIZER BISECT" in rendered
    assert "BISECT_TRUST_COUNT 10" in rendered
    assert "MAX_SCF 60" in rendered


def test_becke_constraint_remains_an_explicit_sensitivity_option():
    settings = replace(DEFAULT_CP2K_SETTINGS, cdft_constraint_type="BECKE")
    rendered = _render_test_complex(_task(COMPLEX_ROLE), settings)
    assert "TYPE_OF_CONSTRAINT BECKE" in rendered
    assert "&BECKE_CONSTRAINT" in rendered
    assert "&HIRSHFELD_CONSTRAINT" not in rendered
    with pytest.raises(ValueError, match="Unsupported CDFT constraint type"):
        replace(DEFAULT_CP2K_SETTINGS, cdft_constraint_type="INVENTED")


def test_counterpoise_ot_remains_an_explicit_validation_alternative():
    settings = replace(
        DEFAULT_CP2K_SETTINGS,
        projectile_counterpoise_scf_solver="OT",
    )
    carbon_cp = render_cp2k_input(
        _task(PROJECTILE_COUNTERPOISE_ROLE), settings
    )
    assert "MINIMIZER CG" in carbon_cp
    assert "LINESEARCH 2PNT" in carbon_cp
    assert "ALGORITHM IRAC" in carbon_cp
    assert "&DIAGONALIZATION ON" not in carbon_cp


def test_counterpoise_pulay_parameters_are_explicit_and_validated():
    settings = replace(
        DEFAULT_CP2K_SETTINGS,
        counterpoise_mixing_alpha=0.4,
        counterpoise_mixing_npulay=7,
    )
    carbon_cp = render_cp2k_input(
        _task(PROJECTILE_COUNTERPOISE_ROLE), settings
    )
    assert "ALPHA 0.4" in carbon_cp
    assert "NPULAY 7" in carbon_cp
    with pytest.raises(ValueError, match="mixing alpha"):
        replace(DEFAULT_CP2K_SETTINGS, counterpoise_mixing_alpha=0.0)
    with pytest.raises(ValueError, match="NPULAY"):
        replace(DEFAULT_CP2K_SETTINGS, counterpoise_mixing_npulay=0)


def test_cp2k_parser_requires_both_scf_and_cdft_completion():
    text = """
    *** SCF run converged in 9 steps ***
    ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.): -75.123456789
    CDFT SCF iter = 4 RMS gradient = 2.0E-06 energy = -75.123456789
    CDFT SCF loop converged in 4 iterations or 20 steps
    Target value of constraint  : 5.000000000000
    Current value of constraint : 5.000002000000
    Deviation from target       : 2.000E-06
    Strength of constraint      : 1.250000000000
    PROGRAM ENDED AT 2026-08-05 18:00:00
    """
    result = parse_cp2k_output(text, require_cdft=True)
    assert result["valid_completion"]
    assert result["energy_hartree"] == pytest.approx(-75.123456789)
    assert result["cdft_target_electrons"] == pytest.approx(5.0)
    assert result["cdft_deviation_electrons"] == pytest.approx(2.0e-6)
    assert not parse_cp2k_output(
        text.replace("CDFT SCF loop converged", "CDFT did not converge"),
        require_cdft=True,
    )["valid_completion"]


def test_cp2k_parser_accepts_2025_2_hartree_energy_line():
    text = """
    *** SCF run converged in 23 steps ***
    ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -76.359117871621663
    PROGRAM ENDED AT 2026-08-06 00:49:44.619
    """
    result = parse_cp2k_output(text, require_cdft=False)
    assert result["valid_completion"]
    assert result["energy_hartree"] == pytest.approx(-76.359117871621663)


def test_workflow_is_deterministic_and_bare_carbon_reference_is_analytic(tmp_path):
    first = build_workflow(tmp_path / "workflow", charges=(6,))
    second = build_workflow(tmp_path / "workflow", charges=(6,))
    assert first == second
    manifest = load_workflow_manifest(first, verify_inputs=True)
    assert manifest["task_count"] == 180
    assert manifest["point_count"] == 60
    analytic = [task for task in manifest["tasks"] if task["execution"] == "analytic"]
    assert len(analytic) == 60
    assert all(task["role"] == PROJECTILE_COUNTERPOISE_ROLE for task in analytic)
    for task in analytic:
        result = json.loads(
            (first.parent / task["result_path"]).read_text(encoding="utf-8")
        )
        assert result["energy_hartree"] == 0.0


def test_full_workflow_shares_charge_independent_water_references(tmp_path):
    manifest_path = build_workflow(tmp_path / "workflow")
    manifest = load_workflow_manifest(manifest_path, verify_inputs=True)
    assert manifest["point_count"] == 420
    assert manifest["task_count"] == 900
    assert manifest["cp2k_task_count"] == 420
    assert manifest["validated_branch_task_count"] == 420
    assert manifest["analytic_task_count"] == 60
    water_tasks = [
        task
        for task in manifest["tasks"]
        if task["role"] == WATER_COUNTERPOISE_ROLE
    ]
    assert len(water_tasks) == 60
    referenced_water_ids = {
        point["component_task_ids"][WATER_COUNTERPOISE_ROLE]
        for point in manifest["points"]
    }
    assert referenced_water_ids == {task["task_id"] for task in water_tasks}


def test_custom_workflow_batch_is_crossed_with_selected_charges(tmp_path):
    geometries = build_scan_geometries(DEFAULT_PROJECTILE)[:2]
    manifest_path = build_workflow(
        tmp_path / "workflow",
        charges=(0, 1),
        geometries=geometries,
        workflow_context={"kind": "adaptive_test_batch", "batch_index": 1},
    )
    manifest = load_workflow_manifest(manifest_path, verify_inputs=True)
    assert manifest["point_count"] == 4
    assert manifest["task_count"] == 10
    assert manifest["cp2k_task_count"] == 6
    assert manifest["validated_branch_task_count"] == 4
    assert manifest["configuration"]["workflow_context"]["batch_index"] == 1


def _adaptive_values(function):
    values = {}
    for charge in (0, 1):
        for separation in (1.0, 1.5, 2.0, 2.5, 3.0):
            values[point_key(charge, "test", separation)] = function(
                charge, separation
            )
    return values


def _single_interval_mesh():
    return {
        "status": "awaiting_calculations",
        "settings": {},
        "curves": {
            "test": {
                "mesh_separations_angstrom": [1.0, 3.0],
                "active_intervals": [{"lower": 1.0, "upper": 3.0, "depth": 0}],
                "accepted_intervals": [],
                "refined_intervals": [],
            }
        },
        "maximum_scaled_interpolation_error": None,
        "sign_mismatch_count": 0,
        "historical_maximum_scaled_interpolation_error": 0.0,
        "historical_sign_mismatch_count": 0,
    }


def test_adaptive_dft_mesh_accepts_directly_validated_linear_potential():
    mesh = _single_interval_mesh()
    values = _adaptive_values(lambda charge, radius: (charge + 1) * radius)
    assert required_probe_geometries(mesh, values, (0, 1)) == ()
    refine_completed_intervals(mesh, values, (0, 1), AdaptiveSettings())
    assert mesh["status"] == "complete"
    assert mesh["maximum_scaled_interpolation_error"] == pytest.approx(0.0)
    assert mesh["sign_mismatch_count"] == 0


def test_adaptive_dft_mesh_bisects_failed_interval_and_reuses_probes():
    mesh = _single_interval_mesh()
    values = _adaptive_values(lambda charge, radius: (charge + 1) * radius**2)
    refine_completed_intervals(mesh, values, (0, 1), AdaptiveSettings())
    assert mesh["status"] == "refining"
    assert mesh["curves"]["test"]["mesh_separations_angstrom"] == [1.0, 2.0, 3.0]
    assert required_probe_geometries(mesh, values, (0, 1)) == (
        ("test", 1.25),
        ("test", 1.75),
        ("test", 2.25),
        ("test", 2.75),
    )


def test_adaptive_dft_sign_change_is_an_explicit_failure():
    values = _adaptive_values(lambda _charge, _radius: 1.0)
    values[point_key(0, "test", 2.0)] = -1.0
    report = interval_interpolation_report(
        orientation="test",
        lower=1.0,
        upper=3.0,
        depth=0,
        charges=(0,),
        values_ev=values,
        tolerance=0.005,
    )
    assert report["passes"] is False
    assert report["sign_mismatch_count"] == 1


def test_adaptive_controller_remains_blocked_without_validated_branches(
    tmp_path, monkeypatch
):
    def fake_parallel_runner(manifest_path, **_kwargs):
        manifest = load_workflow_manifest(manifest_path)
        for task in manifest["tasks"]:
            if task["execution"] in {"analytic", VALIDATED_BRANCH_EXECUTION}:
                continue
            energy = {
                WATER_COUNTERPOISE_ROLE: -62.0,
                PROJECTILE_COUNTERPOISE_ROLE: -37.0,
            }[task["role"]]
            result = {
                "schema_version": manifest["schema_version"],
                "status": "complete",
                "execution": "cp2k",
                "task_id": task["task_id"],
                "configuration_signature": manifest["configuration_signature"],
                "input_sha256": task["input_sha256"],
                "energy_hartree": energy,
                "valid_completion": True,
            }
            path = Path(manifest_path).parent / task["result_path"]
            path.write_text(json.dumps(result), encoding="utf-8")

    monkeypatch.setattr(
        adaptive_controller, "_run_manifest_parallel", fake_parallel_runner
    )
    output_root = tmp_path / "adaptive"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adaptive_charge_resolved_dft.py",
            "--output-root",
            str(output_root),
            "--charges",
            "0",
        ],
    )
    with pytest.raises(RuntimeError, match="Batch remains incomplete"):
        adaptive_controller.main()
    assert not list(output_root.glob("*/collected/c_cdft_adaptive.manifest.json"))


def test_runner_is_task_restart_safe_with_a_fake_cp2k(tmp_path):
    manifest_path = build_workflow(tmp_path / "workflow", charges=(0,))
    fake = tmp_path / "fake_cp2k.py"
    fake.write_text(
        "print('*** SCF run converged in 2 steps ***')\n"
        "print('ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.): -100.0')\n"
        "print('CDFT SCF loop converged in 2 iterations or 4 steps')\n"
        "print('Target value of constraint  : 6.000000000000')\n"
        "print('Current value of constraint : 6.000001000000')\n"
        "print('Deviation from target       : 1.000E-06')\n"
        "print('PROGRAM ENDED AT 2026-08-05 18:00:00')\n",
        encoding="utf-8",
    )
    command = f"{sys.executable} {fake}"
    completed, skipped = run_workflow_tasks(
        manifest_path,
        shard_count=180,
        shard_index=0,
        cp2k_command=command,
    )
    assert (completed, skipped) == (1, 0)
    completed, skipped = run_workflow_tasks(
        manifest_path,
        shard_count=180,
        shard_index=0,
        cp2k_command=command,
    )
    assert (completed, skipped) == (0, 1)


def test_runner_recovers_output_rejected_by_an_older_parser(tmp_path):
    manifest_path = build_workflow(tmp_path / "workflow", charges=(0,))
    manifest = load_workflow_manifest(manifest_path, verify_inputs=True)
    task = manifest["tasks"][0]
    task_directory = manifest_path.parent / task["task_directory"]
    failed_output = task_directory / "cp2k.failed.old-parser.out"
    failed_output.write_text(
        "*** SCF run converged in 2 steps ***\n"
        "ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -100.0\n"
        "CDFT SCF loop converged in 2 iterations or 4 steps\n"
        "Target value of constraint  : 6.000000000000\n"
        "Current value of constraint : 6.000001000000\n"
        "Deviation from target       : 1.000E-06\n"
        "PROGRAM ENDED AT 2026-08-06 00:00:00\n",
        encoding="utf-8",
    )
    failure_path = task_directory / "failure.json"
    failure_path.write_text(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "status": "failed",
                "task_id": task["task_id"],
                "configuration_signature": manifest["configuration_signature"],
                "input_sha256": task["input_sha256"],
                "command": ["cp2k"],
                "return_code": 0,
                "started_utc": "2026-08-06T00:00:00+00:00",
                "failed_utc": "2026-08-06T00:01:00+00:00",
                "output_path": failed_output.name,
            }
        ),
        encoding="utf-8",
    )

    completed, skipped = run_workflow_tasks(
        manifest_path,
        shard_count=manifest["task_count"],
        shard_index=task["task_index"],
        cp2k_command="/bin/false",
    )
    assert (completed, skipped) == (0, 1)
    result = json.loads(
        (manifest_path.parent / task["result_path"]).read_text(encoding="utf-8")
    )
    assert result["valid_completion"]
    assert result["recovered_from_parser_failure"]
    assert result["energy_hartree"] == pytest.approx(-100.0)
    assert (task_directory / "cp2k.out").is_file()
    assert not failed_output.exists()
    assert not failure_path.exists()


def test_runner_continues_failed_scf_from_checksum_recorded_wavefunction(tmp_path):
    manifest_path = build_workflow(tmp_path / "workflow", charges=(0,))
    manifest = load_workflow_manifest(manifest_path, verify_inputs=True)
    task = manifest["tasks"][0]
    task_directory = manifest_path.parent / task["task_directory"]
    fake = tmp_path / "fake_cp2k_restart.py"
    fake.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "inp = Path(sys.argv[sys.argv.index('-i') + 1])\n"
        f"wfn = Path({(task['task_id'] + '-RESTART.wfn')!r})\n"
        "if inp.name == 'input.inp':\n"
        "    wfn.write_bytes(b'bounded CP2K wavefunction checkpoint')\n"
        "    print('SCF run NOT converged')\n"
        "    raise SystemExit(1)\n"
        "text = inp.read_text(encoding='utf-8')\n"
        "assert 'SCF_GUESS RESTART' in text\n"
        "assert f'WFN_RESTART_FILE_NAME {wfn.name}' in text\n"
        "print('*** SCF run converged in 2 steps ***')\n"
        "print('ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -100.0')\n"
        "print('PROGRAM ENDED AT 2026-08-08 06:00:00')\n",
        encoding="utf-8",
    )
    command = f"{sys.executable} {fake}"
    with pytest.raises(RuntimeError, match="failed"):
        run_workflow_tasks(
            manifest_path,
            shard_count=manifest["task_count"],
            shard_index=task["task_index"],
            cp2k_command=command,
        )

    completed, skipped = run_workflow_tasks(
        manifest_path,
        shard_count=manifest["task_count"],
        shard_index=task["task_index"],
        cp2k_command=command,
    )
    assert (completed, skipped) == (1, 0)
    result = json.loads(
        (manifest_path.parent / task["result_path"]).read_text(encoding="utf-8")
    )
    restart_input = task_directory / result["execution_input_path"]
    wavefunction = task_directory / result["restarted_from_wavefunction"]
    assert restart_input.name == "input.restart.inp"
    assert result["execution_input_sha256"] == hashlib.sha256(
        restart_input.read_bytes()
    ).hexdigest()
    assert result["wavefunction_restart_sha256"] == hashlib.sha256(
        wavefunction.read_bytes()
    ).hexdigest()
    assert not (task_directory / "failure.json").exists()


def test_reuse_requires_identical_task_input_and_verified_output(tmp_path):
    geometry = build_scan_geometries(DEFAULT_PROJECTILE)[:1]
    source_path = build_workflow(
        tmp_path / "source", charges=(0,), geometries=geometry
    )
    target_path = build_workflow(
        tmp_path / "target",
        charges=(0,),
        geometries=geometry,
        settings=replace(
            DEFAULT_CP2K_SETTINGS,
            water_counterpoise_scf_solver="DIAGONALIZATION",
            projectile_counterpoise_scf_solver="DIAGONALIZATION",
        ),
    )
    source = load_workflow_manifest(source_path, verify_inputs=True)
    target = load_workflow_manifest(target_path, verify_inputs=True)
    source_task = next(
        task
        for task in source["tasks"]
        if task["role"] == PROJECTILE_COUNTERPOISE_ROLE
    )
    target_task = next(
        task
        for task in target["tasks"]
        if task["role"] == PROJECTILE_COUNTERPOISE_ROLE
    )
    assert source_task["input_sha256"] == target_task["input_sha256"]
    assert any(
        source_task["input_sha256"] != task["input_sha256"]
        for task in target["tasks"]
        if task["execution"] == "cp2k"
        and task["role"] != PROJECTILE_COUNTERPOISE_ROLE
    )
    output_text = (
        "*** SCF run converged in 2 steps ***\n"
        "ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -100.0\n"
        "PROGRAM ENDED AT 2026-08-06 00:00:00\n"
    )
    source_directory = source_path.parent / source_task["task_directory"]
    source_output = source_directory / "cp2k.out"
    source_output.write_text(output_text, encoding="utf-8")
    output_sha256 = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    source_result = {
        "schema_version": source["schema_version"],
        "status": "complete",
        "execution": "cp2k",
        "task_id": source_task["task_id"],
        "configuration_signature": source["configuration_signature"],
        "input_sha256": source_task["input_sha256"],
        "output_path": source_output.name,
        "output_sha256": output_sha256,
        "return_code": 0,
        "energy_hartree": -100.0,
        "valid_completion": True,
    }
    (source_path.parent / source_task["result_path"]).write_text(
        json.dumps(source_result), encoding="utf-8"
    )

    reused, incompatible = reuse_compatible_workflow_results(
        source_path, target_path
    )
    assert reused == 1
    assert incompatible == 1
    target_result = json.loads(
        (target_path.parent / target_task["result_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert target_result["configuration_signature"] == target[
        "configuration_signature"
    ]
    assert target_result["reused_from_configuration_signature"] == source[
        "configuration_signature"
    ]
    assert target_result["output_sha256"] == output_sha256
    assert len(pending_workflow_tasks(target_path)) == 2


def test_electronic_state_audit_rejects_generic_complex_completion(tmp_path):
    manifest_path = build_workflow(
        tmp_path / "workflow",
        charges=(0,),
        geometries=build_scan_geometries(DEFAULT_PROJECTILE)[:1],
    )
    manifest = load_workflow_manifest(manifest_path)
    task = next(
        item for item in manifest["tasks"] if item["role"] == COMPLEX_ROLE
    )
    result = {
        "schema_version": manifest["schema_version"],
        "status": "complete",
        "execution": VALIDATED_BRANCH_EXECUTION,
        "task_id": task["task_id"],
        "configuration_signature": manifest["configuration_signature"],
        "energy_hartree": -100.0,
        "valid_completion": True,
    }
    result_path = manifest_path.parent / task["result_path"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result), encoding="utf-8"
    )

    audit = audit_workflow_electronic_states(manifest_path)
    assert task["task_id"] in audit["missing_task_ids"]
    assert not audit["complete_workflow_audited"]
    assert all(record["task_id"] != task["task_id"] for record in audit["records"])


def test_adaptive_controller_fails_closed_on_unintegrated_complex_tasks(tmp_path):
    manifest_path = build_workflow(
        tmp_path / "workflow",
        settings=DEFAULT_CP2K_SETTINGS,
        charges=(0,),
        geometries=build_scan_geometries(DEFAULT_PROJECTILE)[:2],
    )
    fake = tmp_path / "fake_cp2k.py"
    fake.write_text(
        "print('*** SCF run converged in 2 steps ***')\n"
        "print('ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.): -100.0')\n"
        "print('CDFT SCF loop converged in 2 iterations or 4 steps')\n"
        "print('Target value of constraint  : 6.000000000000')\n"
        "print('Current value of constraint : 6.000001000000')\n"
        "print('Deviation from target       : 1.000E-06')\n"
        "print('PROGRAM ENDED AT 2026-08-05 18:00:00')\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Adaptive CP2K batch"):
        adaptive_controller._run_manifest_parallel(
            manifest_path,
            run_root=tmp_path,
            batch_index=0,
            cp2k_command=f"{sys.executable} {fake}",
            cpus_per_calculation=1,
            requested_parallel=2,
        )
    assert any(
        task["role"] == COMPLEX_ROLE
        for task in adaptive_controller.pending_workflow_tasks(manifest_path)
    )


def test_legacy_runner_never_executes_a_complex_cdft_task(tmp_path):
    manifest_path = build_workflow(
        tmp_path / "workflow",
        charges=(0,),
        geometries=build_scan_geometries(DEFAULT_PROJECTILE)[:1],
    )
    marker = tmp_path / "legacy_complex_was_executed"
    fake = tmp_path / "fake_seed_cp2k.py"
    fake.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    manifest = load_workflow_manifest(manifest_path)
    complex_task = next(
        task for task in manifest["tasks"] if task["role"] == COMPLEX_ROLE
    )
    assert complex_task["execution"] == VALIDATED_BRANCH_EXECUTION
    assert "input_path" not in complex_task
    with pytest.raises(RuntimeError, match="CP2KBranchExecutor"):
        run_workflow_tasks(
            manifest_path,
            shard_count=manifest["task_count"],
            shard_index=complex_task["task_index"],
            cp2k_command=f"{sys.executable} {fake}",
            progress=False,
        )
    assert not marker.exists()


def test_collection_rejects_generic_complex_result(tmp_path):
    manifest_path = build_workflow(
        tmp_path / "workflow",
        charges=(0,),
        geometries=build_scan_geometries(DEFAULT_PROJECTILE)[:1],
    )
    manifest = load_workflow_manifest(manifest_path)
    for task in manifest["tasks"]:
        energy = {
            COMPLEX_ROLE: -100.0,
            PROJECTILE_COUNTERPOISE_ROLE: -37.0,
            WATER_COUNTERPOISE_ROLE: -62.0,
        }[task["role"]]
        result = {
            "schema_version": manifest["schema_version"],
            "status": "complete",
            "execution": task["execution"],
            "task_id": task["task_id"],
            "configuration_signature": manifest["configuration_signature"],
            "energy_hartree": energy,
            "valid_completion": True,
        }
        if "input_sha256" in task:
            result["input_sha256"] = task["input_sha256"]
        if task["role"] == COMPLEX_ROLE:
            result.update(
                {
                    "cdft_target_electrons": 6.0,
                    "cdft_current_electrons": 6.0,
                    "cdft_deviation_electrons": 0.0,
                }
            )
        path = manifest_path.parent / task["result_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(RuntimeError, match="CP2KBranchExecutor"):
        collect_workflow(manifest_path)
