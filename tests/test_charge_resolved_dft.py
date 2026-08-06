from __future__ import annotations

import csv
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
    HARTREE_TO_EV,
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
    COMPLEX_ROLE,
    PROJECTILE_COUNTERPOISE_ROLE,
    WATER_COUNTERPOISE_ROLE,
    parse_cp2k_output,
    render_cp2k_input,
)
from soft_dft.geometry import ORIENTATIONS, build_scan_geometries, scan_distances
from soft_dft.workflow import (
    build_workflow,
    collect_workflow,
    load_workflow_manifest,
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
    complex_input = next(
        (manifest_path.parent / task["input_path"]).read_text(encoding="utf-8")
        for task in manifest["tasks"]
        if task["role"] == COMPLEX_ROLE and task["charge"] == 0
    )
    assert "ELEMENT Li" in complex_input
    assert "BASIS_SET TEST-AE-BASIS" in complex_input
    assert "TARGET 3" in complex_input


def test_external_ion_definition_rejects_incomplete_charge_ladder(tmp_path):
    path = _write_lithium_definition(tmp_path / "Li.json", omit_charge=2)
    with pytest.raises(ValueError, match="complete ordered charge ladder"):
        load_projectile_definition(path)


def test_external_explicit_grid_collects_without_inventing_nlh_values(tmp_path):
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
        if task["role"] == COMPLEX_ROLE:
            result.update(
                {
                    "cdft_target_electrons": 0.0,
                    "cdft_current_electrons": 0.0,
                    "cdft_deviation_electrons": 0.0,
                }
            )
        path = manifest_path.parent / task["result_path"]
        path.write_text(json.dumps(result), encoding="utf-8")
    csv_path, collection_path = collect_workflow(manifest_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["projectile"] == "Li"
    assert row["anchor_nlh_potential_ev_if_in_domain"] == ""
    assert row["nlh_retained_pair_count"] == "0"
    assert row["nlh_max_pair_potential_ev"] == ""
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    assert collection["projectile"]["symbol"] == "Li"


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
        "electrons_on_projectile": state.electrons_on_projectile,
        "coordinates_angstrom": [list(row) for row in geometry.coordinates_angstrom],
    }


def test_cp2k_inputs_use_all_electron_gapw_and_separate_counterpoise_roles():
    complex_input = render_cp2k_input(_task(COMPLEX_ROLE), DEFAULT_CP2K_SETTINGS)
    assert "METHOD GAPW" in complex_input
    assert "POTENTIAL ALL" in complex_input
    assert "BASIS_SET TZVPP-MOLOPT-GGA-ae" in complex_input
    assert "CHARGE 5" in complex_input
    assert "MULTIPLICITY 2" in complex_input
    assert "TARGET 1" in complex_input
    assert "TYPE_OF_CONSTRAINT BECKE" in complex_input
    assert "ADJUST_SIZE FALSE" in complex_input
    assert "PERIODIC NONE" in complex_input

    carbon_cp = render_cp2k_input(
        _task(PROJECTILE_COUNTERPOISE_ROLE), DEFAULT_CP2K_SETTINGS
    )
    assert "&CDFT" not in carbon_cp
    assert "&KIND O_G" in carbon_cp
    assert "GHOST TRUE" in carbon_cp

    water_cp = render_cp2k_input(
        _task(WATER_COUNTERPOISE_ROLE), DEFAULT_CP2K_SETTINGS
    )
    assert "CHARGE 0" in water_cp
    assert "MULTIPLICITY 1" in water_cp
    assert "&KIND P_G" in water_cp


def test_cp2k_parser_requires_both_scf_and_cdft_completion():
    text = """
    *** SCF run converged in 9 steps ***
    ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.): -75.123456789
    CDFT SCF loop converged in 4 iterations or 20 steps
    Target value of constraint  : 5.000000000000
    Current value of constraint : 5.000002000000
    Deviation from target       : 2.000E-06
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
    assert manifest["cp2k_task_count"] == 840
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
    assert manifest["cp2k_task_count"] == 10
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


def test_adaptive_controller_completes_linear_surface_without_refinement(
    tmp_path, monkeypatch
):
    def fake_parallel_runner(manifest_path, **_kwargs):
        manifest = load_workflow_manifest(manifest_path)
        for task in manifest["tasks"]:
            if task["execution"] == "analytic":
                continue
            energy = {
                WATER_COUNTERPOISE_ROLE: -62.0,
                PROJECTILE_COUNTERPOISE_ROLE: -37.0,
                COMPLEX_ROLE: (
                    -99.0
                    + float(task["separation_angstrom"]) / HARTREE_TO_EV
                ),
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
            if task["role"] == COMPLEX_ROLE:
                result.update(
                    {
                        "cdft_target_electrons": 6.0,
                        "cdft_current_electrons": 6.0,
                        "cdft_deviation_electrons": 0.0,
                    }
                )
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
    assert adaptive_controller.main() == 0
    final_manifests = list(
        output_root.glob("*/collected/c_cdft_adaptive.manifest.json")
    )
    assert len(final_manifests) == 1
    final = json.loads(final_manifests[0].read_text(encoding="utf-8"))
    assert final["numerical_status"] == "adaptive_interpolation_converged"
    assert final["mesh_row_count"] == 60
    assert final["evaluation_row_count"] == 225
    assert final["maximum_scaled_interpolation_error"] < 1.0e-10


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


def test_adaptive_controller_runs_independent_cp2k_tasks_concurrently(tmp_path):
    manifest_path = build_workflow(
        tmp_path / "workflow",
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
    adaptive_controller._run_manifest_parallel(
        manifest_path,
        run_root=tmp_path,
        batch_index=0,
        cp2k_command=f"{sys.executable} {fake}",
        cpus_per_calculation=1,
        requested_parallel=2,
    )
    assert not adaptive_controller.pending_workflow_tasks(manifest_path)


def test_collection_applies_counterpoise_definition_without_shifting(tmp_path):
    manifest_path = build_workflow(tmp_path / "workflow", charges=(0,))
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
            "execution": "cp2k",
            "task_id": task["task_id"],
            "configuration_signature": manifest["configuration_signature"],
            "input_sha256": task["input_sha256"],
            "energy_hartree": energy,
            "valid_completion": True,
        }
        if task["role"] == COMPLEX_ROLE:
            result.update(
                {
                    "cdft_target_electrons": 6.0,
                    "cdft_current_electrons": 6.0,
                    "cdft_deviation_electrons": 0.0,
                }
            )
        path = manifest_path.parent / task["result_path"]
        path.write_text(json.dumps(result), encoding="utf-8")

    csv_path, collection_path = collect_workflow(manifest_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 60
    assert float(rows[0]["interaction_energy_ev_counterpoise"]) == pytest.approx(
        -HARTREE_TO_EV
    )
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    assert collection["numerical_status"] == "calculations_complete"
    assert collection["physics_status"] == "validation_pending"
    assert collection["no_asymptotic_shift_applied"] is True
    assert collection["no_nlh_blend_or_scaling_applied"] is True
