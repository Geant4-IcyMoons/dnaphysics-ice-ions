"""Stage boundary, independent samples and explicitly estimated scalar SE."""
from concurrent.futures import Future
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("staged_hard_runner", Path(__file__).parents[1]/"certify.py")
c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c
spec.loader.exec_module(c)


def setup(root, monkeypatch):
    p = c.Policy(max_histories=128, max_case_histories=16, max_cases=8, batch_size=2,
        training_histories=4, initial_validation_histories=4, max_proposal_rounds=1,
        base_bins=2, max_bin_level=1)
    m = {"signature": "stage-test", "policy": asdict(p), "historical_seeds": [], "seed": 7,
         "path_length_angstrom": 1., "projectile": "C", "structures": [{"name": "ice"}],
         "directions": [{"name": "fixed", "vector": [1, 0, 0]}], "energies_ev": [10., 100.]}
    state = {"cases": {}, "intervals": [], "reserved_histories": 0,
             "differential": {"index_sha256": "pair-index", "tables_verified": True}}
    for energy in m["energies_ev"]:
        key = c.add_case(state, m, 0, 0, energy)
        state["cases"][key]["kernel"] = {"passes": True}
    keys = list(state["cases"])
    state["intervals"] = [{"a": keys[0], "b": keys[1], "depth": 0, "status": "pending"}]
    c.atomic_json(root/"state.json", state)
    calls = []
    def execute(task):
        identity = task[3]; calls.append(identity)
        n = identity["stop"]-identity["start"]
        return {"raw": np.tile([1., 1., .2, .1, 0., 0.], (n, 1)),
            "points": np.zeros((n, 3)), "weights": np.ones(n), "density": np.ones(n),
            "base": np.ones(n), "cpu_seconds": np.ones(n),
            "events": np.array([[i, 0, .05, .02, 1., 0., 0., 0.] for i in range(n)])}
    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, task):
            f = Future(); f.set_result(execute(task)); return f
    monkeypatch.setattr(c, "verify", lambda root: m)
    monkeypatch.setattr(c, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(c, "run_differential", lambda *args: True)
    monkeypatch.setattr(c, "fit_proposal", lambda *args, **kwargs: {"boxes": [], "alpha": [1.]})
    monkeypatch.setattr(c.signal, "signal", lambda *args: None)
    return m, calls


def test_calibration_freezes_without_consuming_production_and_resumes(tmp_path, monkeypatch):
    m, calls = setup(tmp_path, monkeypatch)
    first = c.run(tmp_path, 2, 32, stage="calibrate")
    assert first["status"] == "calibration_ready"
    assert first["committed_histories"] == 8
    assert first["reserved_or_consumed_histories"] == 12  # Four geometry replays.
    assert all(i["phase"] == "training" and i["calibration_geometry_check"] for i in calls)
    handoff = c.read(tmp_path/"calibration/index.json")
    assert handoff["ready_for_independent_sampling"]
    before = {p.name: c.file_hash(p) for p in (tmp_path/"designs").glob("*.json")}
    assert c.run(tmp_path, 1, 8, stage="calibrate")["committed_histories"] == 8
    second = c.run(tmp_path, 1, 8, stage="produce")
    assert second["committed_histories"] == 16
    assert all(c.file_hash(tmp_path/"designs"/name) == sha for name, sha in before.items())
    assert {i["phase"] for i in calls} == {"training", "baseline"}
    assert not c.read(tmp_path/"production/report.json")["calibration_samples_pooled"]
    assert c.read(tmp_path/"calibration/index.json") == handoff


def test_production_refuses_missing_or_tampered_calibration(tmp_path, monkeypatch):
    _, calls = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="calibrate before produce"):
        c.run(tmp_path, 1, 8, stage="produce")
    assert not calls
    c.run(tmp_path, 1, 32, stage="calibrate")
    path = tmp_path/"calibration/index.json"
    handoff = c.read(path); handoff["pair_index_sha256"] = "changed"
    c.atomic_json(path, handoff)
    with pytest.raises(ValueError, match="handoff changed"):
        c.run(tmp_path, 1, 8, stage="produce")


def test_weighted_ratio_se_matches_centred_influence_formula(monkeypatch):
    raw = np.array([[1., 1., .2, .1], [2., 2., 1., .4], [3., 1., .3, .2], [1., 3., 2., .8]])
    weights = np.array([.5, 1.5, 1., 1.])
    monkeypatch.setattr(c, "load_block", lambda *args: {"raw": raw, "weights": weights})
    result = c.scalar_standard_errors(None, [{}], .05)
    x = raw*weights[:, None]
    for name, (a, b) in zip(c.OBSERVABLES, c.RATIOS):
        ratio = x[:, a].mean()/x[:, b].mean()
        expected = np.std(x[:, a]-ratio*x[:, b], ddof=1)/np.sqrt(len(x))/x[:, b].mean()
        assert result[name]["standard_error"] == pytest.approx(expected)
        assert result[name]["relative_standard_error"] == pytest.approx(expected/ratio)
    c.canonical(result)  # Atomic reports require ordinary JSON scalar types.


def test_zero_variance_is_not_claimed_to_resolve_unseen_events(monkeypatch):
    monkeypatch.setattr(c, "load_block", lambda *args: {"raw": np.ones((8, 4)), "weights": np.ones(8)})
    result = c.scalar_standard_errors(None, [{}], .05)
    assert all(r["zero_variance_requires_review"] and not r["standard_error_passes"] for r in result.values())
    assert c.Policy().scalar_precision_mode == "relative_standard_error"
    assert c.Policy().distribution_tv_tolerance == .005
