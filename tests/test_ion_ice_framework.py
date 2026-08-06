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

from ion_ice import (  # noqa: E402
    canonical_projectile,
    get_phase,
    get_projectile,
    projectile_registry,
)
from ion_ice.compatibility import backend_compatibility_checks  # noqa: E402
from ion_ice.resources import (  # noqa: E402
    PARALLEL_STAGES,
    load_resource_profiles,
    parse_pbs_accounting,
)
from ion_ice.schema import load_phase_definition  # noqa: E402
from ion_ice.workflow import (  # noqa: E402
    assemble_geant4_bundle,
    build_plan,
    inspect_model,
    save_plan,
)


def _stage(report, name):
    return next(item for item in report["stages"] if item["stage"] == name)


def test_registry_is_atomic_number_ordered_and_aliases_are_canonical():
    assert tuple(projectile_registry()) == ("H", "He", "C", "O", "S")
    assert canonical_projectile("proton") == "H"
    assert canonical_projectile("alpha") == "He"
    assert canonical_projectile("sulphur") == "S"
    assert canonical_projectile("C6+") == "C"


def test_registry_and_active_backends_have_no_duplicate_constant_drift():
    checks = backend_compatibility_checks()
    assert checks
    assert all(check.passed for check in checks), [
        check.as_dict() for check in checks if not check.passed
    ]


def test_species_registry_exposes_real_component_gaps():
    projectiles = projectile_registry()
    proton = get_projectile("H")
    assert all(
        projectile.component_status("soft_dft") == "implemented"
        for projectile in projectiles.values()
    )
    assert proton.component_status("geant4_hard_runtime") == "blocked"


def test_phase_registry_separates_accepted_and_candidate_structures():
    hexagonal = get_phase("hexagonal_ih_100k")
    amorphous = get_phase("amorphous_lda_80k")
    assert hexagonal.resolve(hexagonal.structure_registry).is_file()
    assert amorphous.structure_registry is None
    assert amorphous.density["value_g_cm3"] is None


def test_phase_cannot_bypass_bounded_resource_profiles(tmp_path):
    source = get_phase("amorphous_lda_80k").source_path
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["preparation"]["qsub_arguments"] = [
        "-l",
        "select=1:ncpus=512:mem=1tb",
    ]
    path = tmp_path / "phase.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot override bounded"):
        load_phase_definition(path)


def test_status_is_component_resolved_and_never_overstates_geant4_readiness():
    carbon = inspect_model("C", "hexagonal_ih_100k")
    oxygen = inspect_model("O", "hexagonal_ih_100k")
    assert _stage(carbon, "ice_structure")["state"] == "complete"
    assert _stage(carbon, "nlh_pair_potential")["state"] == "complete"
    assert _stage(carbon, "nlh_kernel")["state"] == "complete"
    assert _stage(carbon, "hard_transport")["state"] in {"running", "complete"}
    assert _stage(carbon, "geant4_hard_table")["state"] == "validation_pending"
    assert _stage(carbon, "soft_dft_definition")["state"] == "complete"
    assert _stage(carbon, "geant4_soft_runtime")["state"] == "missing_input"
    assert not carbon["ready_for_geant4"]
    assert _stage(oxygen, "soft_dft_definition")["state"] == "complete"
    assert _stage(oxygen, "geant4_hard_runtime")["state"] == "missing_input"
    assert not oxygen["ready_for_geant4"]
    assert carbon["ctmc"]["included"] is False


def test_plan_does_not_chain_unvalidated_structure_dynamics_into_transport(
    monkeypatch,
):
    monkeypatch.setattr("ion_ice.workflow._active_pbs_job_ids", lambda _: ())
    plan = build_plan(("C",), ("amorphous_lda_80k",))
    task_ids = {task["task_id"] for task in plan["tasks"]}
    assert "structure:amorphous_lda_80k" in task_ids
    assert "hard_transport:C:amorphous_lda_80k" not in task_ids
    structure = next(
        task
        for task in plan["tasks"]
        if task["task_id"] == "structure:amorphous_lda_80k"
    )
    assert "PBS success alone does not complete" in structure["completion_gate"]


def test_active_phase_preparation_is_reported_and_never_planned_again(monkeypatch):
    monkeypatch.setattr(
        "ion_ice.workflow._active_pbs_job_ids",
        lambda name: ("123[0].pbs",) if name == "nep_mbpol_lda" else (),
    )
    report = inspect_model("C", "amorphous_lda_80k")
    assert _stage(report, "ice_structure")["state"] == "running"
    plan = build_plan(("C",), ("amorphous_lda_80k",))
    assert "structure:amorphous_lda_80k" not in {
        task["task_id"] for task in plan["tasks"]
    }


def test_plan_is_signed_and_records_missing_cp2k_without_submitting(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("ion_ice.workflow._active_pbs_job_ids", lambda _: ())
    monkeypatch.setattr("ion_ice.workflow._soft_state_candidates", lambda _: [])
    plan = build_plan(("C",), ("hexagonal_ih_100k",))
    soft = next(
        task for task in plan["tasks"] if task["task_id"] == "soft_molecular_dft:C"
    )
    assert soft["state"] == "blocked"
    assert "cp2k" in soft["blockers"][0].lower()
    path = save_plan(plan, tmp_path / "plan.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["plan_signature"] == plan["plan_signature"]


def test_every_parallel_task_has_a_bounded_resource_profile(monkeypatch):
    monkeypatch.setattr("ion_ice.workflow._active_pbs_job_ids", lambda _: ())
    profiles, limits, fallback = load_resource_profiles()
    assert set(profiles) == PARALLEL_STAGES
    assert limits == {"maximum_ncpus": 256, "maximum_memory_gb": 512}
    assert (fallback.ncpus, fallback.memory_gb) == (64, 128)
    plan = build_plan(("C",), ("amorphous_lda_80k",))
    for task in plan["tasks"]:
        if task["kind"] != "pbs":
            continue
        resources = task["resources"]
        assert 1 <= resources["ncpus"] <= 256
        assert 1 <= resources["memory_gb"] <= 512
        assert resources["requested_gb_per_cpu"] == pytest.approx(
            resources["memory_gb"] / resources["ncpus"]
        )
        assert resources["select"] in task.get("qsub_argv", []) or task[
            "state"
        ] == "blocked"


def test_pbs_accounting_reports_memory_per_cpu_and_utilization():
    text = """
    Job_Id = 123.pbs02
    Job_Name = measured
    resources_used.cput = 01:00:00
    resources_used.mem = 2097152kb
    resources_used.walltime = 00:15:00
    job_state = F
    Resource_List.mem = 128gb
    Resource_List.ncpus = 64
    Exit_status = 0
"""
    record = parse_pbs_accounting(text, "123")
    assert record["used_memory_gb"] == pytest.approx(2.0)
    assert record["used_gb_per_allocated_cpu"] == pytest.approx(0.03125)
    assert record["average_used_cores"] == pytest.approx(4.0)
    assert record["average_cpu_utilization"] == pytest.approx(0.0625)
    assert record["accounting_complete"]


def test_final_assembly_refuses_missing_scientific_gates(tmp_path):
    with pytest.raises(RuntimeError, match="assembly refused"):
        assemble_geant4_bundle("C", "hexagonal_ih_100k", tmp_path)
    assert not list(tmp_path.iterdir())
