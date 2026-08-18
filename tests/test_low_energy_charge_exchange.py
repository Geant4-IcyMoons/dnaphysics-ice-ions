from __future__ import annotations

import json
import math
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

from ion_ice import get_projectile, projectile_registry  # noqa: E402
from low_energy_charge_exchange import (  # noqa: E402
    BRANCH_HANDOFF_PENDING,
    build_single_capture_channels,
    build_workflow,
    collect_workflow,
    coupled_multiplicities,
    impact_parameter_cross_section_cm2,
    load_workflow_manifest,
    run_workflow_tasks,
    transition_probability,
)
from low_energy_charge_exchange.cp2k import (  # noqa: E402
    parse_mixed_cdft_output,
    render_mixed_cdft_input,
)
from soft_dft import DEFAULT_CP2K_SETTINGS, build_scan_geometries  # noqa: E402
from soft_dft.workflow import (  # noqa: E402
    VALIDATED_BRANCH_EXECUTION,
    VALIDATED_BRANCH_RUNNER,
)


def test_carbon_has_every_positive_charge_capture_step_to_neutral():
    channels = build_single_capture_channels(get_projectile("C"))
    assert [channel.incident_state.charge for channel in channels] == list(
        range(1, 7)
    )
    assert [channel.product_state.charge for channel in channels] == list(range(6))
    assert channels[0].reaction == "C+ + H2O -> C + H2O+"
    assert channels[-1].reaction == "C6+ + H2O -> C5+ + H2O+"
    for channel in channels:
        assert channel.spin_allowed
        assert channel.total_charge == channel.incident_state.charge
        assert (
            channel.product_state.electrons_on_projectile
            == channel.incident_state.electrons_on_projectile + 1
        )
        assert channel.total_multiplicity in (
            channel.allowed_product_total_multiplicities
        )


def test_every_registered_species_builds_its_complete_capture_ladder():
    for projectile in projectile_registry().values():
        channels = build_single_capture_channels(projectile)
        assert len(channels) == projectile.atomic_number
        assert {channel.incident_state.charge for channel in channels} == set(
            range(1, projectile.atomic_number + 1)
        )
        assert all(channel.spin_allowed for channel in channels)


def test_neutral_is_an_endpoint_not_an_implicit_negative_ion_channel():
    with pytest.raises(ValueError, match="q=0 is the neutral endpoint"):
        build_single_capture_channels(get_projectile("C"), (0,))


def test_spin_coupling_uses_exact_fragment_multiplicity_triangle():
    assert coupled_multiplicities(1, 2) == (2,)
    assert coupled_multiplicities(2, 2) == (1, 3)
    assert coupled_multiplicities(3, 2) == (2, 4)


def test_carbon_smoke_manifest_defers_exact_diabatic_branches(tmp_path):
    projectile = get_projectile("C")
    geometry = next(
        geometry
        for geometry in build_scan_geometries(projectile)
        if geometry.orientation == "oxygen_back"
        and geometry.separation_angstrom == 6.0
    )
    manifest_path = build_workflow(
        tmp_path / "capture", projectile=projectile, geometries=(geometry,)
    )
    manifest = load_workflow_manifest(manifest_path, verify_inputs=True)
    assert manifest["channel_count"] == 6
    assert manifest["geometry_count"] == 1
    assert manifest["work_unit_count"] == 6
    assert manifest["required_validated_branch_state_count"] == 12
    assert manifest["integration_status"] == BRANCH_HANDOFF_PENDING
    assert manifest["configuration"]["represented_projectile_charge_ladder"] == list(
        range(7)
    )
    coverage = manifest["configuration"]["charge_state_coverage"]
    assert coverage[0]["role"] == "product_endpoint"
    assert all(item["role"] == "incident_and_product" for item in coverage[1:6])
    assert coverage[6]["role"] == "incident_only"
    for unit in manifest["work_units"]:
        entrance_task = unit["state_tasks"]["entrance"]
        product_task = unit["state_tasks"]["capture_product"]
        for task in (entrance_task, product_task):
            assert task["execution"] == VALIDATED_BRANCH_EXECUTION
            assert task["validated_branch_runner"] == VALIDATED_BRANCH_RUNNER
            assert task["integration_status"] == BRANCH_HANDOFF_PENDING
            assert task["total_charge"] == unit["total_charge"]
            assert task["total_multiplicity"] == unit["total_multiplicity"]
            assert task["coordinates_angstrom"] == unit["coordinates_angstrom"]
            assert "input_path" not in task
            assert "input_sha256" not in task
            assert "expected_wavefunction" not in task
            assert not (manifest_path.parent / task["task_directory"]).exists()
        assert (
            entrance_task["electrons_on_projectile"]
            == unit["entrance_electrons_on_projectile"]
        )
        assert (
            product_task["electrons_on_projectile"]
            == unit["product_electrons_on_projectile"]
        )
        assert not (manifest_path.parent / unit["mixed_task"]["input_path"]).exists()


def test_mixed_input_restarts_both_states_at_common_charge_and_spin(tmp_path):
    projectile = get_projectile("C")
    geometry = build_scan_geometries(projectile)[0]
    manifest_path = build_workflow(
        tmp_path / "capture",
        projectile=projectile,
        incident_charges=(2,),
        geometries=(geometry,),
    )
    unit = load_workflow_manifest(manifest_path)["work_units"][0]
    text = render_mixed_cdft_input(
        {**unit, "task_id": unit["mixed_task"]["task_id"]},
        DEFAULT_CP2K_SETTINGS,
        entrance_wavefunction="../entrance/entrance.wfn",
        product_wavefunction="../capture_product/product.wfn",
        entrance_strength=0.25,
        product_strength=-0.30,
    )
    assert text.count("&FORCE_EVAL") == 3
    assert "METHOD MIXED" in text
    assert "MIXING_TYPE MIXED_CDFT" in text
    assert "COUPLING 1" in text
    assert "LOWDIN TRUE" in text
    assert text.count("CHARGE 2") == 2
    assert text.count("MULTIPLICITY 1") == 2
    assert "TARGET 4" in text
    assert "TARGET 5" in text
    assert "STRENGTH 0.25" in text
    assert "STRENGTH -0.3" in text
    assert text.count("MAX_SCF 0") == 2
    assert "STEP_SIZE" not in text
    assert "WFN_RESTART_FILE_NAME ../entrance/entrance.wfn" in text
    assert "WFN_RESTART_FILE_NAME ../capture_product/product.wfn" in text


def test_mixed_output_parser_requires_two_states_and_reports_ev():
    text = """
    CDFT SCF loop converged in 1 iterations or 1 steps
    CDFT SCF loop converged in 1 iterations or 1 steps
    MIXED_CDFT| Activating mixed CDFT calculation
    Overlap between states I and J: 0.030261294466
    Charge transfer energy (J-I) (Hartree): 0.000739539045
    Diabatic electronic coupling (rotation, mHartree): 5.674875246867
    Diabatic electronic coupling (Lowdin, mHartree): 5.674714192287
    PROGRAM ENDED AT 2026-08-06 12:00:00
    """
    result = parse_mixed_cdft_output(text)
    assert result["valid_completion"]
    assert result["charge_transfer_energy_ev"] == pytest.approx(
        0.000739539045 * 27.211386245981
    )
    assert result["coupling_lowdin_ev"] == pytest.approx(
        5.674714192287e-3 * 27.211386245981
    )
    assert not parse_mixed_cdft_output(
        text.replace("CDFT SCF loop converged", "CDFT did not converge", 1)
    )["valid_completion"]


def test_runner_and_collector_fail_before_using_wfn_only_states(tmp_path):
    projectile = get_projectile("C")
    geometry = build_scan_geometries(projectile)[0]
    manifest_path = build_workflow(
        tmp_path / "capture",
        projectile=projectile,
        incident_charges=(1,),
        geometries=(geometry,),
    )

    manifest = load_workflow_manifest(manifest_path)
    unit = manifest["work_units"][0]
    entrance = unit["state_tasks"]["entrance"]
    legacy_directory = manifest_path.parent / entrance["task_directory"]
    legacy_directory.mkdir(parents=True)
    (legacy_directory / "legacy-RESTART.wfn").write_bytes(b"wfn only")
    (legacy_directory / "task_result.json").write_text(
        json.dumps({"status": "complete", "cdft_strength": 0.25}),
        encoding="utf-8",
    )
    marker = tmp_path / "cp2k_was_started"
    fake_cp2k = tmp_path / "fake_cp2k.py"
    fake_cp2k.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('started', encoding='utf-8')\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Accepted-state handoff"):
        run_workflow_tasks(
            manifest_path,
            cp2k_command=f"{sys.executable} {fake_cp2k}",
            progress=False,
        )
    assert not marker.exists()
    collection_directory = tmp_path / "collected"
    with pytest.raises(RuntimeError, match="Accepted-state handoff"):
        collect_workflow(manifest_path, collection_directory)
    assert not collection_directory.exists()


def test_landau_zener_primitives_are_bounded_and_unit_explicit():
    assert transition_probability(0.0, 1.0, 1.0e5) == 0.0
    weak = transition_probability(0.01, 1.0, 1.0e5)
    strong = transition_probability(0.1, 1.0, 1.0e5)
    slow = transition_probability(0.1, 1.0, 1.0e4)
    assert 0.0 < weak < strong < slow < 1.0
    cross_section = impact_parameter_cross_section_cm2(
        (0.0, 1.0, 2.0), (1.0, 1.0, 1.0)
    )
    assert cross_section == pytest.approx(math.pi * 4.0e-16)
