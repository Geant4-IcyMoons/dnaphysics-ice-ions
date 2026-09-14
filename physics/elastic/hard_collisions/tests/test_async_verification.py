"""Deterministic verification scheduling; synthetic products are not physics evidence."""
from dataclasses import asdict
from concurrent.futures import Future
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("hard_async_verification", Path(__file__).parents[1] / "certify.py")
c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c
spec.loader.exec_module(c)


def fixture(root, selected="validation"):
    policy = c.Policy(batch_size=2, training_histories=4, initial_validation_histories=4)
    manifest = {"signature": "abc", "policy": asdict(policy), "projectile": "C"}
    sample = {"raw": np.tile([1., 1., .2, .1, 0., 0.], (4, 1)), "cpu_seconds": np.ones(4)}
    case = {"status": "pending", "kernel": {"passes": True}, "kernel_mode": "qualified_pair_maps",
            "structure": 0, "direction": 0, "energy_ev": 1000., "phase": selected, "round": 0,
            "selected_cohort": selected, "target": 4, "design_sha256": "design", "cohorts": {},
            "proposal": {"boxes": [], "alpha": [1.]}}
    for phase in ("baseline", "validation"):
        identity = {"campaign": "abc", "case": "one", "phase": phase, "round": 0, "start": 0, "stop": 4,
                    "proposal": {"boxes": [], "alpha": [1.]}, "design_sha256": "design",
                    "kernel_mode": "qualified_pair_maps"}
        entry = c.save_block(root, identity, sample)
        case["cohorts"][phase + "_0"] = c.refresh_cohort({"blocks": [entry]})
    state = {"cases": {"one": case}, "status": "prepared", "intervals": [], "reserved_histories": 8,
             "differential": {"index_sha256": "synthetic_pair_index", "tables_verified": True}}
    return manifest, state, case, sample


def test_atomic_verification_proof_is_reusable_and_adds_no_samples(tmp_path, monkeypatch):
    manifest, state, case, sample = fixture(tmp_path, "baseline")
    identity = c.verification_identity(manifest, case, "one")
    assert identity["propagations"] == 4
    calls = []
    def replay(task):
        calls.append(task[3]["phase"])
        return {name: values.copy() * (2 if name == "cpu_seconds" else 1) for name, values in sample.items()}
    monkeypatch.setattr(c, "execute_block", replay)
    result = c.execute_verification(tmp_path, manifest, case, identity)
    assert c.execute_verification(tmp_path, manifest, case, identity) == result
    assert calls == ["baseline"]
    case["phase"] = "verification"
    c.advance_case(tmp_path, manifest, state, "one")
    c.advance_case(tmp_path, manifest, state, "one")
    assert case["status"] == "qualified" and case["phase"] == "baseline"
    assert state["reserved_histories"] == 12  # Adopted orphan proof, accounted once.
    assert c.summary(state)["new_independent_histories"] == 8
    assert set(case["cohorts"]) == {"baseline_0", "validation_0"}


def test_verification_mismatch_cannot_commit_a_passing_proof(tmp_path, monkeypatch):
    manifest, _, case, sample = fixture(tmp_path)
    identity = c.verification_identity(manifest, case, "one")
    assert identity["propagations"] == 6  # Four replays plus two search-window checks.
    changed = {name: value.copy() for name, value in sample.items()}
    changed["raw"][0, 1] += 1
    monkeypatch.setattr(c, "execute_block", lambda task: changed)
    with pytest.raises(RuntimeError, match="differs in raw"):
        c.execute_verification(tmp_path, manifest, case, identity)
    assert c.verification_result(tmp_path, identity) is None


def test_scheduler_reserves_only_one_bounded_verification_task(tmp_path):
    manifest, state, case, _ = fixture(tmp_path)
    case["phase"] = "verification"
    task, _ = c.next_case_task(tmp_path, manifest, state, {}, smoke_remaining=6)
    assert task[3]["phase"] == "verification" and task[3]["propagations"] == 6
    pending = {("one", "verification", 0): [(0, 4)]}
    assert c.next_case_task(tmp_path, manifest, state, pending)[0] is None
    assert c.next_case_task(tmp_path, manifest, state, {}, smoke_remaining=5)[0] is None


def test_export_cannot_reuse_another_cohorts_passing_replay_flag(tmp_path, monkeypatch):
    manifest, state, case, sample = fixture(tmp_path)
    monkeypatch.setattr(c, "execute_block", lambda *args: sample)
    case["replay"] = c.execute_verification(tmp_path, manifest, case,
                                            c.verification_identity(manifest, case, "one"))
    case.update(selected_cohort="baseline", phase="baseline", design_path="synthetic")
    c.atomic_json(tmp_path / "state.json", state)
    monkeypatch.setattr(c, "verify", lambda *args: manifest)
    monkeypatch.setattr(c, "verify_coverage", lambda *args: None)
    monkeypatch.setattr(c, "assess", lambda *args: {"statistical_pass": True, "grid_pass": True})
    with pytest.raises(ValueError, match="current selected cohort"):
        c.export(tmp_path)
    assert not (tmp_path / "tables").exists()


def test_worker_verification_uses_queue_without_new_trajectory_blocks(tmp_path, monkeypatch):
    manifest, state, case, sample = fixture(tmp_path)
    report = {"statistical_pass": True, "grid_pass": True, "importance_normalization_interval": [.9, 1.1]}
    c.atomic_json(tmp_path / "state.json", state)
    original_paths = sorted(p.name for p in (tmp_path / "blocks").glob("*"))
    outer = []
    def execute(task):
        if task[3]["phase"] == "verification":
            return c.execute_verification(*task)
        return sample
    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, function, task):
            outer.append(task[3]["phase"])
            result = Future(); result.set_result(function(task)); return result
    monkeypatch.setattr(c, "verify", lambda *args: manifest)
    monkeypatch.setattr(c, "verify_coverage", lambda *args: None)
    monkeypatch.setattr(c, "run_differential", lambda *args: True)
    monkeypatch.setattr(c, "feasibility", lambda *args: {"blocked_cases": []})
    monkeypatch.setattr(c, "assess", lambda *args: report)
    monkeypatch.setattr(c, "efficiency_diagnostic", lambda *args: {"estimated_limiting_observable_gain": 2.})
    monkeypatch.setattr(c, "observations_compatible", lambda *args: True)
    monkeypatch.setattr(c, "advance_grid", lambda *args: None)
    monkeypatch.setattr(c, "export", lambda *args: {})
    monkeypatch.setattr(c, "readiness", lambda *args: {})
    monkeypatch.setattr(c, "execute_block", execute)
    monkeypatch.setattr(c, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(c.signal, "signal", lambda *args: None)
    result = c.run(tmp_path, workers=2, max_new_histories=8)
    assert outer == ["verification"]
    assert result["qualified_cases"] == 1
    assert result["new_independent_histories"] == result["committed_histories"] == 8
    assert result["reserved_or_consumed_histories"] == 14
    assert sorted(p.name for p in (tmp_path / "blocks").glob("*")) == original_paths
    assert len(list((tmp_path / "verifications").glob("*.json"))) == 1
