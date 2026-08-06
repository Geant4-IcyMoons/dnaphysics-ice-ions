from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

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


def test_carbon_smoke_workflow_keeps_total_charge_fixed_between_diabats(tmp_path):
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
    assert manifest["prepared_cdft_state_count"] == 12
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
        entrance = (manifest_path.parent / entrance_task["input_path"]).read_text()
        product = (manifest_path.parent / product_task["input_path"]).read_text()
        charge_line = f"CHARGE {unit['incident_charge']}"
        assert charge_line in entrance
        assert charge_line in product
        assert (
            f"TARGET {unit['entrance_electrons_on_projectile']}" in entrance
        )
        assert f"TARGET {unit['product_electrons_on_projectile']}" in product
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


def test_runner_resumes_states_then_mixed_and_verifies_wavefunctions(
    tmp_path, monkeypatch
):
    projectile = get_projectile("C")
    geometry = build_scan_geometries(projectile)[0]
    manifest_path = build_workflow(
        tmp_path / "capture",
        projectile=projectile,
        incident_charges=(1,),
        geometries=(geometry,),
    )

    def fake_cp2k(args, *, cwd, stdout, **_kwargs):
        input_text = (Path(cwd) / args[-1]).read_text(encoding="utf-8")
        if "METHOD MIXED" in input_text:
            stdout.write(
                """
                CDFT SCF loop converged in 1 iterations or 1 steps
                CDFT SCF loop converged in 1 iterations or 1 steps
                MIXED_CDFT| Activating mixed CDFT calculation
                Overlap between states I and J: 0.03
                Charge transfer energy (J-I) (Hartree): 0.001
                Diabatic electronic coupling (rotation, mHartree): 2.1
                Diabatic electronic coupling (Lowdin, mHartree): 2.0
                PROGRAM ENDED AT 2026-08-06 12:00:00
                """
            )
        else:
            project = next(
                line.split(maxsplit=1)[1]
                for line in input_text.splitlines()
                if line.strip().startswith("PROJECT ")
            )
            target = next(
                line.split()[1]
                for line in input_text.splitlines()
                if line.strip().startswith("TARGET ")
            )
            (Path(cwd) / f"{project}-RESTART.wfn").write_bytes(
                f"wavefunction target={target}".encode()
            )
            stdout.write(
                f"""
                *** SCF run converged in 1 steps ***
                ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.): -75.0
                CDFT SCF loop converged in 1 iterations or 1 steps
                Target value of constraint  : {target}
                Current value of constraint : {target}
                Deviation from target       : 0.0
                Strength of constraint      : 0.25
                PROGRAM ENDED AT 2026-08-06 12:00:00
                """
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "low_energy_charge_exchange.workflow.subprocess.run", fake_cp2k
    )
    assert run_workflow_tasks(
        manifest_path, cp2k_command="fake-cp2k", progress=False
    ) == (3, 0)
    assert run_workflow_tasks(
        manifest_path, cp2k_command="fake-cp2k", progress=False
    ) == (0, 3)
    manifest = load_workflow_manifest(manifest_path)
    unit = manifest["work_units"][0]
    mixed_result = json.loads(
        (manifest_path.parent / unit["mixed_task"]["result_path"]).read_text()
    )
    assert mixed_result["valid_completion"]
    table_path, collection_path = collect_workflow(
        manifest_path, tmp_path / "collected"
    )
    assert table_path.read_text().count("\n") == 2
    collection = json.loads(collection_path.read_text())
    assert collection["numerical_status"] == "complete"
    assert collection["cross_section_status"] == "not_computed"
    entrance = unit["state_tasks"]["entrance"]
    entrance_result = json.loads(
        (manifest_path.parent / entrance["result_path"]).read_text()
    )
    assert entrance_result["wavefunction_sha256"]
    wavefunction = manifest_path.parent / entrance_result["wavefunction_path"]
    wavefunction.write_bytes(b"changed after convergence")
    with pytest.raises(RuntimeError, match="wavefunction changed"):
        run_workflow_tasks(
            manifest_path, cp2k_command="fake-cp2k", progress=False
        )


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
