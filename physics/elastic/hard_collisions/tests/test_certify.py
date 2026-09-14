"""Software tests; synthetic integration is not validation of ice physics."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("hard_certify", Path(__file__).parents[1] / "certify.py")
c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c
spec.loader.exec_module(c)


def test_policy_rejects_zero_budget_and_invalid_floor():
    with pytest.raises(ValueError):
        c.Policy(max_histories=0).validate()
    with pytest.raises(ValueError):
        c.Policy(defensive_fraction=0).validate()
    c.Policy().validate()


def test_periodic_box_normalizes_and_wraps():
    boxes = [{"centre": [.98], "half_width": [.05]}]
    x = ((np.arange(10000) + .5) / 10000)[:, None]
    density = c.box_density(x, boxes)[:, 0]
    assert density.mean() == pytest.approx(1.)
    assert c.box_density([[.001]], boxes)[0, 0] == pytest.approx(10.)


def test_overlap_uses_complete_mixture_density():
    box = {"centre": [.5, .5, .5], "half_width": [.1, .1, .1]}
    proposal = {"boxes": [box, box], "alpha": [.2, .3, .5]}
    assert c.mixture_density([1.], [[.5] * 3], proposal)[0] == pytest.approx(.2 + .8 / .008)


def test_targeted_integration_recovers_known_rare_event():
    rng = np.random.default_rng(1)
    proposal = {"boxes": [{"centre": [.01], "half_width": [.001]}], "alpha": [.2, .8]}
    n = 100000
    x = rng.random((n, 1)); selected = rng.random(n) > .2
    x[selected, 0] = .009 + .002 * rng.random(selected.sum())
    q = c.mixture_density(np.ones(n), x, proposal)
    y = ((x[:, 0] >= .009) & (x[:, 0] < .011)) / q
    assert abs(y.mean() - .002) < 5 * y.std(ddof=1) / np.sqrt(n)
    assert (1 / q).max() <= 5.


def test_defensive_sampling_does_not_delete_undiscovered_modes():
    proposal = {"boxes": [{"centre": [.01], "half_width": [.001]}], "alpha": [.2, .8]}
    assert c.mixture_density([1.], [[.8]], proposal)[0] == .2


def test_isotropic_coordinate_roundtrip():
    for p in ([.1, .3], [.9, .98], [.5, .5]):
        assert np.allclose(c.isotropic_point(c.direction_from_point(p)), p)


def test_clustered_event_moments_not_independent_event_errors():
    raw = np.array([[1., 2., .4, .4, 0., 0.], [1., 0., 0., 0., 0., 0.]])
    events = np.array([[0, 0, .1, .2, 1, 0, 0, 0], [0, 0, .1, .2, 1, 0, 0, 0]])
    features, _ = c.feature_rows(raw, events, np.ones(2), [[0, 1], [0, 1]])
    assert features[:, 6].sum() == 2
    assert features[:, 6].power(2).sum() == 4  # Not 2 independent Bernoulli events.


def test_rebinning_retains_counts_and_target_identity():
    raw = np.array([[1., 2., .4, .4, 0., 0.]])
    events = np.array([[0, 0, .1, .2, 1, 0, 0, 0], [0, 1, .9, 1, 1, 0, 0, 0]])
    features, _ = c.feature_rows(raw, events, [2.], [[0, .5, 1], [0, .5, 1]])
    assert features[:, 6:14].sum() == 4
    assert features[:, 6:10].sum() == 2
    assert features[:, 10:14].sum() == 2


def test_zero_events_have_positive_uncertainty():
    low, high = c.empirical_interval(100, [0.], [0.], [1.], .05)
    assert low[0] == 0 and high[0] > 0


def test_bernstein_interval_covers_deterministic_mean():
    low, high = c.empirical_interval(10000, [5000.], [2500.], [1.], .01)
    assert low[0] < .5 < high[0]


def test_smaller_error_budget_widens_intervals():
    a = c.empirical_interval(10000, [100.], [100.], [1.], .05)
    b = c.empirical_interval(10000, [100.], [100.], [1.], 1e-8)
    assert b[1][0] > a[1][0]


def test_zero_denominator_refuses_ratio():
    assert c.ratio_interval([0, 0], [1, 2], 1, 0) == [0., None]


def test_five_percent_relative_error_uses_positive_confidence_endpoint():
    assert c.relative_precision(100., [96., 104.], .05)["passes"]
    assert not c.relative_precision(100., [95., 105.], .05)["passes"]
    assert not c.relative_precision(.001, [0., .002], .05)["passes"]
    assert not c.relative_precision(0., [0., None], .05)["passes"]
    assert c.relative_precision(1e-20, [.96e-20, 1.04e-20], .05)["passes"]


def block(n=4):
    return {"raw": np.tile([1., 1., .2, .1, 0., 0.], (n, 1)),
            "points": np.full((n, 3), .5), "weights": np.ones(n), "density": np.ones(n),
            "base": np.ones(n), "cpu_seconds": np.ones(n), "window_delta": np.zeros(n),
            "events": np.array([[i, 0, .05, .2, 1., 0., 0., 0.] for i in range(n)])}


def identity(start=0, stop=4):
    return {"campaign": "abc", "case": "one", "phase": "training", "round": 0,
            "start": start, "stop": stop, "proposal": {"boxes": [], "alpha": [1.]}, "kernel_mode": "table"}


def test_checkpoint_roundtrip_and_corruption(tmp_path):
    entry = c.save_block(tmp_path, identity(), block())
    assert np.array_equal(c.load_block(tmp_path, entry)["raw"], block()["raw"])
    with (tmp_path / entry["path"]).open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        c.load_block(tmp_path, entry)


def test_orphan_committed_receipt_is_adopted_once(tmp_path):
    c.save_block(tmp_path, identity(), block())
    state = {"cases": {"one": {"cohorts": {}}}, "reserved_histories": 4}
    c.reconcile(tmp_path, {"signature": "abc"}, state)
    c.reconcile(tmp_path, {"signature": "abc"}, state)
    assert len(state["cases"]["one"]["cohorts"]["training_0"]["blocks"]) == 1
    assert state["reserved_histories"] == 4


def test_noncontiguous_resume_preserves_later_commit_and_fills_gap(tmp_path):
    late = c.save_block(tmp_path, identity(4, 8), block())
    checksum = c.file_hash(tmp_path / late["path"])
    state = {"cases": {"one": {"cohorts": {}}}, "reserved_histories": 4}
    c.reconcile(tmp_path, {"signature": "abc"}, state)
    cohort = state["cases"]["one"]["cohorts"]["training_0"]
    assert cohort["count"] == 4 and cohort["prefix"] == 0
    assert c.missing_ranges(cohort, 12) == [(0, 4), (8, 12)]
    c.save_block(tmp_path, identity(0, 4), block())
    c.reconcile(tmp_path, {"signature": "abc"}, state)
    c.reconcile(tmp_path, {"signature": "abc"}, state)
    assert cohort["count"] == 8 and cohort["prefix"] == 8
    assert c.missing_ranges(cohort, 12) == [(8, 12)]
    assert [entry["identity"]["start"] for entry in cohort["blocks"]] == [0, 4]
    assert c.file_hash(tmp_path / late["path"]) == checksum
    assert state["reserved_histories"] == 8


def test_overlapping_resume_is_rejected(tmp_path):
    c.save_block(tmp_path, identity(0, 4), block())
    c.save_block(tmp_path, identity(2, 6), block())
    state = {"cases": {"one": {"cohorts": {}}}, "reserved_histories": 8}
    with pytest.raises(ValueError, match="Overlapping"):
        c.reconcile(tmp_path, {"signature": "abc"}, state)


def test_missing_ranges_excludes_pending_tasks_without_treating_them_as_evidence():
    cohort = {"blocks": [{"identity": identity(8, 12)}, {"identity": identity(0, 2)}]}
    assert c.missing_ranges(cohort, 16, pending=[(2, 4), (12, 14)]) == [(4, 8), (14, 16)]
    assert cohort["count"] == 6 and cohort["prefix"] == 2
    # An interrupted in-flight task is missing again; its receipt never committed.
    assert c.missing_ranges(cohort, 16) == [(2, 8), (12, 16)]
    assert c.missing_ranges(cohort, 1) == []
    assert c.missing_ranges(cohort, 0) == []
    with pytest.raises(ValueError, match="Overlapping"):
        c.missing_ranges(cohort, 16, pending=[(1, 4)])


def test_missing_ranges_empty_cohort_and_invalid_intervals():
    cohort = {"blocks": []}
    assert c.missing_ranges(cohort, 7) == [(0, 7)]
    assert cohort["count"] == cohort["prefix"] == 0
    for bad in [(-1, 2), (3, 3), (4, 2), (False, 2)]:
        with pytest.raises(ValueError, match="Invalid"):
            c.missing_ranges(cohort, 7, pending=[bad])
    with pytest.raises(ValueError, match="nonnegative"):
        c.missing_ranges(cohort, -1)


def test_scheduler_is_fair_and_skips_committed_and_live_ranges():
    from dataclasses import asdict
    policy = c.Policy(batch_size=2)
    cases = {key: {"structure": 0, "direction": 0, "energy_ev": 1000., "kernel_mode": "table",
                   "kernel": {"passes": True}, "status": "pending", "phase": "training", "round": 0,
                   "target": 8, "proposal": {"boxes": [], "alpha": [1.]}, "cohorts": {}}
             for key in ("a", "b")}
    cases["a"]["cohorts"]["training_0"] = {"blocks": [{"identity": identity(2, 4)}]}
    manifest = {"signature": "abc", "policy": asdict(policy)}
    state = {"cases": cases, "reserved_histories": 0, "kernel_checks": {}}
    task, cursor = c.next_case_task("root", manifest, state, {})
    assert (task[3]["case"], task[3]["start"], task[3]["stop"]) == ("a", 0, 2)
    pending = {("a", "training", 0): [(0, 2)]}
    task, cursor = c.next_case_task("root", manifest, state, pending, cursor)
    assert task[3]["case"] == "b"
    task, _ = c.next_case_task("root", manifest, state, pending, cursor)
    assert (task[3]["case"], task[3]["start"], task[3]["stop"]) == ("a", 4, 6)
    assert c.next_case_task("root", manifest, state, pending, smoke_remaining=0)[0] is None


def test_live_queue_replenishes_before_straggler_finishes(tmp_path, monkeypatch):
    from dataclasses import asdict
    from concurrent.futures import Future
    policy = c.Policy(max_histories=64, max_case_histories=8, max_cases=8, batch_size=2,
                      training_histories=4, initial_validation_histories=4, max_proposal_rounds=1)
    manifest = {"signature": "abc", "policy": asdict(policy), "historical_seeds": [], "seed": 10,
                "path_length_angstrom": 1., "projectile": "C", "structures": [{"name": "ice"}],
                "directions": [{"name": "fixed", "vector": [1, 0, 0]}], "energies_ev": [10., 100.]}
    state = {"cases": {}, "intervals": [], "reserved_histories": 0, "status": "prepared",
             "differential": {"index_sha256": "synthetic_pair_index", "tables_verified": True}}
    keys = [c.add_case(state, manifest, 0, 0, energy) for energy in manifest["energies_ev"]]
    state["intervals"] = [{"a": keys[0], "b": keys[1], "depth": 0, "status": "pending"}]
    for case in state["cases"].values(): case["kernel"] = {"passes": True}
    c.atomic_json(tmp_path / "state.json", state)
    calls, delayed = [], []
    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, function, task):
            future = Future(); calls.append(task[3])
            if len(calls) == 1:
                delayed.append(future)
            else:
                future.set_result(block(2))
                if len(calls) == 3:
                    # A wave-barrier scheduler would never submit this task.
                    assert not delayed[0].done()
                    delayed[0].set_result(block(2))
            return future
    monkeypatch.setattr(c, "verify", lambda root: manifest)
    monkeypatch.setattr(c, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(c, "run_differential", lambda *args: True)  # This test isolates trajectory scheduling.
    monkeypatch.setattr(c.signal, "signal", lambda *args: None)
    result = c.run(tmp_path, workers=2, max_new_histories=8)
    assert result["committed_histories"] == 8
    assert [item["case"] for item in calls[:3]] == [keys[0], keys[1], keys[0]]
    assert len({c.digest(item) for item in calls}) == 4


def test_inefficient_targeted_pilot_uses_independent_baseline_without_relaxation(tmp_path, monkeypatch):
    from dataclasses import asdict
    policy = c.Policy(batch_size=2, initial_validation_histories=4, training_histories=4,
                      max_case_histories=16)
    case = {"status": "pending", "kernel": {"passes": True}, "phase": "validation", "round": 0,
            "target": 4, "level": 0, "proposal": {"alpha": [.2, .8], "boxes": [{"kept": True}]},
            "cohorts": {phase + "_0": {"blocks": [{"identity": {**identity(), "phase": phase}}]}
                        for phase in ("baseline", "validation")}}
    original_proposal = c.digest(case["proposal"])
    manifest = {"policy": asdict(policy)}
    report = {"statistical_pass": False, "grid_pass": True, "importance_normalization_interval": [.9, 1.1]}
    weights = []
    monkeypatch.setattr(c, "assess", lambda *args: report)
    monkeypatch.setattr(c, "observations_compatible", lambda *args: True)
    monkeypatch.setattr(c, "efficiency_diagnostic", lambda *args: {"estimated_limiting_observable_gain": .5})
    def feasibility(*args, weight_bound=None):
        weights.append(weight_bound)
        return {"sampling_at_pilot_means_can_fit_budget": True}
    monkeypatch.setattr(c, "range_budget_diagnostic", feasibility)
    state = {"cases": {"one": case}}
    c.advance_case(tmp_path, manifest, state, "one")
    assert case["status"] == "pending" and case["phase"] == "baseline"
    assert c.production_cohort(case) == "baseline"
    assert case["sampling_decision"]["tolerances_unchanged"]
    assert c.digest(case["proposal"]) == original_proposal
    assert c.cohort_weight_bound(case) == 2.
    assert c.cohort_weight_bound(case, "validation") == 10.
    c.advance_case(tmp_path, manifest, state, "one")
    assert case["target"] == 8 and weights == [2.]


def test_frozen_design_cannot_be_relaxed(tmp_path):
    case = {k: [] for k in ("structure", "direction", "energy_ev", "round", "proposal", "kernel_mode",
                             "kernel", "statistical_relative_tolerance", "grids")}
    case.update(design_path="design.json", cohorts={})
    case["design_sha256"] = c.digest(c.frozen_design(case))
    c.atomic_json(tmp_path / "design.json", c.frozen_design(case))
    c.verify_design(tmp_path, case)
    case["statistical_relative_tolerance"] = .99
    with pytest.raises(ValueError, match="Changed"):
        c.verify_design(tmp_path, case)


def report(value, radius=.001):
    return {"scalar": {name: {"estimate": value, "interval": [value-radius, value+radius]} for name in c.OBSERVABLES}}


def test_midpoint_requires_equivalence_not_nonsignificance():
    assert c.interpolation_bound(report(1), report(1), report(1), .01)["passes"]
    assert c.interpolation_bound(report(1), report(2), report(1), .01)["resolved_failure"]
    unresolved = c.interpolation_bound(report(1, .5), report(1, .5), report(1, .5), .01)
    assert not unresolved["passes"] and not unresolved["resolved_failure"]


def test_matrix_contains_every_declared_combination():
    state = {"cases": {}}
    manifest = {"policy": {"max_cases": 12, "training_histories": 4}, "historical_seeds": [],
                "structures": [{}, {}], "directions": [{"vector": None}] * 3}
    for si in range(2):
        for di in range(3):
            for energy in (1000, 10000):
                assert c.add_case(state, manifest, si, di, energy) is not None
    assert len(state["cases"]) == 12
    assert c.add_case(state, manifest, 0, 0, 5000) is None


def test_output_bins_are_frozen_from_training():
    events = block()["events"]
    first = c.grid_edges(events, 8)
    changed_validation = block()["events"] * 100
    assert c.grid_edges(events, 8) == first
    assert changed_validation.shape == events.shape


def test_source_changes_fail_before_sampling(tmp_path):
    c.atomic_json(tmp_path / "manifest.json", {"schema": c.SCHEMA, "signature": "not a signature"})
    with pytest.raises(ValueError, match="signature"):
        c.verify(tmp_path)


def test_export_refuses_empty_scientific_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "verify", lambda root: {})
    monkeypatch.setattr(c, "verify_coverage", lambda *args: None)
    monkeypatch.setattr(c, "reconcile", lambda *args: None)
    monkeypatch.setattr(c, "assess", lambda *args: None)
    c.atomic_json(tmp_path / "state.json", {"cases": {"one": {}}, "intervals": []})
    with pytest.raises(ValueError, match="not qualified"):
        c.export(tmp_path)
    assert not (tmp_path / "tables").exists()


def test_empty_matrix_cannot_be_certified():
    with pytest.raises(ValueError, match="matrix is missing"):
        c.verify_coverage({}, {"cases": {}, "intervals": []})


def test_controller_advances_and_resumes_without_resubmitting_old_work(tmp_path, monkeypatch):
    from dataclasses import asdict
    from concurrent.futures import Future
    policy = c.Policy(max_histories=64, max_case_histories=8, max_cases=8, batch_size=2,
                      training_histories=4, initial_validation_histories=4, max_proposal_rounds=1,
                      base_bins=2, max_bin_level=1)
    manifest = {"signature": "abc", "policy": asdict(policy), "historical_seeds": [], "seed": 10,
                "path_length_angstrom": 1., "projectile": "C", "structures": [{"name": "ice"}],
                "directions": [{"name": "fixed", "vector": [1, 0, 0]}], "energies_ev": [10., 100.]}
    state = {"cases": {}, "intervals": [], "reserved_histories": 0, "status": "prepared",
             "differential": {"index_sha256": "synthetic_pair_index", "tables_verified": True}}
    keys = [c.add_case(state, manifest, 0, 0, e) for e in manifest["energies_ev"]]
    state["intervals"] = [{"a": keys[0], "b": keys[1], "depth": 0, "status": "pending"}]
    for case in state["cases"].values(): case["kernel"] = {"passes": True}
    c.atomic_json(tmp_path / "state.json", state)
    calls = []
    def execute(task):
        identity = task[3]
        calls.append(c.digest(identity))
        return block(identity["stop"] - identity["start"])
    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, function, task):
            future = Future()
            future.set_result(execute(task))
            return future
    monkeypatch.setattr(c, "verify", lambda root: manifest)
    monkeypatch.setattr(c, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(c, "run_differential", lambda *args: True)  # Differential-stage integration is tested separately.
    monkeypatch.setattr(c, "fit_proposal", lambda *args, **kwargs: {"boxes": [], "alpha": [1.]})
    monkeypatch.setattr(c.signal, "signal", lambda *args: None)
    first = c.run(tmp_path, workers=2, max_new_histories=8)
    assert first["committed_histories"] == 8
    second = c.run(tmp_path, workers=2, max_new_histories=32)
    assert len(calls) == len(set(calls))
    assert second["committed_histories"] > first["committed_histories"]
    assert second["qualified_cases"] == 0
    assert second["reserved_or_consumed_histories"] <= 64
    assert any(case.get("assessment") for case in c.read(tmp_path / "state.json")["cases"].values())


def test_weighted_tv_interpolation_retains_uncertainty(tmp_path):
    from dataclasses import asdict
    case = {"round": 0, "energy_ev": 1., "proposal": {"alpha": [1.]}, "cohorts": {"validation_0": {"blocks": []}}}
    sample = block(50)
    sample["weights"] = np.linspace(.5, 1.5, 50)
    entry = c.save_block(tmp_path, identity(0, 50), sample)
    case["cohorts"]["validation_0"]["blocks"].append(entry)
    manifest = {"policy": asdict(c.Policy(max_collisions=1)), "path_length_angstrom": 1.}
    result = c.tv_interpolation_check(tmp_path, manifest, [case] * 3, [[0., .5, 1.], [0., .5, 1.]])
    assert not result["passes"]
    assert not result["resolved_failure"]


def test_historical_batch_replay_is_verified_before_training(tmp_path):
    from dataclasses import asdict
    sample = block()
    sample["historical"] = sample["raw"][:, :4]
    entry = c.save_block(tmp_path, {**identity(), "phase": "replay"}, sample)
    expected = {}
    for name, (a, b) in zip(c.OBSERVABLES, c.RATIOS):
        x, y = sample["historical"][:, a], sample["historical"][:, b]
        expected[name] = {"numerator_sum": x.sum(), "denominator_sum": y.sum(),
                          "numerator_square_sum": (x*x).sum(), "denominator_square_sum": (y*y).sum(),
                          "numerator_denominator_sum": (x*y).sum()}
    case = {"status": "pending", "kernel": {"passes": True}, "phase": "replay", "round": 0,
            "target": 4, "cohorts": {"replay_0": {"count": 4, "blocks": [entry]}}}
    manifest = {"policy": asdict(c.Policy()), "historical_batch_replay": {"batch": {"raw_statistics": expected}, "batch_sha256": "test"}}
    state = {"cases": {"one": case}}
    c.advance_case(tmp_path, manifest, state, "one")
    assert case["phase"] == "training"
    assert case["historical_replay"]["new_independent_samples"] == 0
    assert case["historical_replay"]["passes"]


def test_scheduler_refuses_unqualified_pair_domain():
    from dataclasses import asdict
    manifest = {"signature": "abc", "policy": asdict(c.Policy()), "historical_seeds": [],
                "structures": [{}], "directions": [{"vector": None}]}
    state = {"cases": {}, "reserved_histories": 0}
    key = c.add_case(state, manifest, 0, 0, 1000.)
    assert "kernel" not in state["cases"][key]
    assert c.next_case_task("root", manifest, state, {})[0] is None
    state["cases"][key]["kernel"] = {"passes": False}
    assert c.next_case_task("root", manifest, state, {})[0] is None


def test_range_penalty_stops_futile_escalation():
    from dataclasses import asdict
    manifest = {"policy": asdict(c.Policy()), "path_length_angstrom": 100.}
    case = {"energy_ev": 1e8, "proposal": {"alpha": [.2]}}
    result = c.range_budget_diagnostic(manifest, case, {"means": [100., 9., .8, 1e-8]})
    assert not result["sampling_at_pilot_means_can_fit_budget"]


def test_total_variation_is_not_maximum_cell_or_cdf_error():
    p = np.full(100, .01)
    q = p + np.tile([.001, -.001], 50)
    assert np.max(np.abs(p - q)) < .005
    assert np.max(np.abs(np.cumsum(p - q))) < .005
    assert c.total_variation_bound(p, q, q) == pytest.approx(.05)


def test_empty_joint_bins_remain_in_tv_bound():
    p = np.array([1., 0., 0.])
    assert c.total_variation_bound(p, [.98, 0, 0], [1., .01, .01]) == pytest.approx(.02)


def test_tv_bound_encloses_all_tested_simplex_probabilities():
    rng = np.random.default_rng(16)
    p = np.array([.1, .2, .3, .4])
    lo, hi = np.maximum(0., p - .08), np.minimum(1., p + .1)
    bound = c.total_variation_bound(p, lo, hi)
    for q in rng.dirichlet(np.ones(4) * 10, size=20000):
        if np.all(q >= lo) and np.all(q <= hi):
            assert .5 * np.abs(q - p).sum() <= bound + 1e-15


def test_tv_refuses_nonprobability_or_bad_intervals():
    with pytest.raises(ValueError):
        c.total_variation_bound([.4, .4], [0, 0], [1, 1])
    with pytest.raises(ValueError):
        c.total_variation_bound([.5, .5], [.7, 0], [.3, 1])


def test_declared_materials_do_not_force_extra_amorphous_directions():
    manifest = {"directions": [{"name": "basal"}, {"name": "c"}, {"name": "isotropic"}],
                "structures": [{}, {"directions": ["isotropic"]}]}
    assert c.case_directions(manifest, 0) == [0, 1, 2]
    assert c.case_directions(manifest, 1) == [2]


def test_pilot_gate_does_not_unlock_broad_sampling_on_completion_alone():
    state = {"cases": {"pilot": {"status": "pending"}, "other": {"status": "pending"}}, "pilot_cases": ["pilot"]}
    assert [k for k, _ in c.active_cases(state)] == ["pilot"]
    state["cases"]["pilot"]["status"] = "blocked"
    assert [k for k, _ in c.active_cases(state)] == ["pilot"]
    state["cases"]["pilot"].update(status="pending", feasibility_passed=True)
    assert len(c.active_cases(state)) == 2


def test_proposal_scale_family_reaches_small_basins_without_duplicate_centres():
    boxes = c.proposal_boxes([[.1, .2, .3], [.1, .2, .3]], c.Policy())
    assert len(boxes) == len(c.Policy().proposal_scale_powers)
    assert min(box["half_width"][0] for box in boxes) <= 1e-6


def test_optimizer_with_analytic_gradient_preserves_support():
    rng = np.random.default_rng(80)
    points = rng.random((512, 3))
    raw = np.tile([1., 1., .1, .01, 0., 0.], (512, 1))
    raw[:, 2] *= 1 + 100 * (points[:, 0] < .02)
    proposal = c.fit_proposal(points, np.ones(512), np.ones(512), raw, [],
                             c.Policy(local_centres=2, proposal_scale_powers=(2, 5)))
    assert sum(proposal["alpha"]) == pytest.approx(1.)
    assert proposal["alpha"][0] >= c.Policy().defensive_fraction - 1e-10
    assert np.all(c.mixture_density(np.ones(512), points, proposal) > 0)


def test_old_scalar_policy_is_rejected_instead_of_silently_kept():
    with pytest.raises(TypeError):
        c.Policy(meaningful_digits=2)
    with pytest.raises(TypeError):
        c.Policy(distribution_absolute_tolerance=.005)


def test_new_cases_require_the_shared_qualified_pair_maps():
    from dataclasses import asdict
    manifest = {"policy": asdict(c.Policy(kernel_relative_tolerance=.001)), "historical_seeds": [],
                "structures": [{}], "directions": [{"vector": None}], "kernel_generation_nominal_error": .005}
    state = {"cases": {}}
    key = c.add_case(state, manifest, 0, 0, 1000.)
    assert state["cases"][key]["kernel_mode"] == "qualified_pair_maps"
    assert "kernel" not in state["cases"][key]
    state["differential"] = {"tables_verified": True, "index_sha256": "qualified_index"}
    midpoint = c.add_case(state, manifest, 0, 0, 2000.)
    assert state["cases"][midpoint]["kernel"]["index_sha256"] == "qualified_index"


def test_no_transfer_of_historical_feasibility_to_another_material(tmp_path):
    from dataclasses import asdict
    manifest = {"signature": "x", "policy": asdict(c.Policy()), "path_length_angstrom": 100.,
                "structures": [{"sha256": "new"}], "directions": [{"vector": None}],
                "historical_planning": [{"structure_sha256": "old", "direction": None, "energy_ev": 1e8}]}
    state = {"cases": {"one": {"structure": 0, "direction": 0, "energy_ev": 1e8}}}
    result = c.feasibility(tmp_path, manifest, state)
    assert not result["blocked_cases"] and not result["checks"]
    assert not result["broad_sampling_authorized_by_feasibility"]


def test_feasibility_result_is_atomic_json_and_never_an_acceptance(tmp_path):
    from dataclasses import asdict
    manifest = {"signature": "x", "policy": asdict(c.Policy()), "path_length_angstrom": 100.,
                "structures": [{"sha256": "old"}], "directions": [{"vector": None}],
                "historical_planning": [{"structure_sha256": "old", "direction": None, "energy_ev": 1e8,
                  "means": [100., 9., .8, 1e-8], "case": "old", "report_sha256": "abc", "trajectories": 400000000}]}
    state = {"cases": {"one": {"structure": 0, "direction": 0, "energy_ev": 1e8}}}
    result = c.feasibility(tmp_path, manifest, state)
    assert result["blocked_cases"] == ["one"]
    assert c.read(tmp_path / "feasibility.json") == result
    assert result["checks"]["one"]["new_trajectories"] == 0


def test_sparse_aggregate_matches_dense_with_correlated_events(tmp_path):
    sample = block(5)
    sample["events"] = np.concatenate((sample["events"], sample["events"]))
    sample["raw"][:, 1] = 2
    sample["weights"] = np.linspace(.5, 1.5, 5)
    entry = c.save_block(tmp_path, identity(0, 5), sample)
    edges = [np.linspace(0, 1, 100).tolist()] * 2
    features, _ = c.feature_rows(sample["raw"], sample["events"], sample["weights"], edges)
    n, total, total2, _, _ = c.aggregate(tmp_path, [entry], edges)
    assert n == 5
    np.testing.assert_allclose(total, np.asarray(features.sum(0)).ravel())
    np.testing.assert_allclose(total2, np.asarray(features.power(2).sum(0)).ravel())


def test_table_handoff_refuses_missing_case(tmp_path, monkeypatch):
    manifest = {"signature": "abc"}
    c.atomic_json(tmp_path / "state.json", {"cases": {"one": {}}})
    c.atomic_json(tmp_path / "tables/index.json", {"schema": c.SCHEMA, "campaign_signature": "abc", "tables": []})
    monkeypatch.setattr(c, "verify_coverage", lambda *args: None)
    with pytest.raises(ValueError, match="missing or duplicating"):
        c.verify_tables(tmp_path, manifest)


def synthetic_handoff(tmp_path, monkeypatch):
    from dataclasses import asdict
    record = {"water_molecules": 1, "volume_angstrom3": 1., "phase": "synthetic"}
    direction = {"vector": [1., 0., 0.]}
    manifest = {"signature": "test", "projectile": "C", "policy": asdict(c.Policy()),
                "structures": [record], "directions": [direction], "estimand": "synthetic software test"}
    state = {"cases": {"one": {"structure": 0, "direction": 0, "energy_ev": 1000.}}, "intervals": []}
    table = {"campaign_signature": "test", "case": "one", "projectile": "C", "structure": record,
             "entrance_energy_ev": 1000., "entrance_direction": direction,
             "joint_probability": [.25, .75], "macroscopic_bin_rate_per_cm": [2.5e7, 7.5e7],
             "effective_bin_area_per_water_cm2": [2.5e-17, 7.5e-17],
             "nonzero_bin_indices_target_angle_recoil": [[0, 0, 0], [1, 1, 1]],
             "shape": [2, 2, 2], "angular_edges": [0., .5, 1.], "recoil_fraction_edges": [0., .5, 1.],
             "uncertainty": {"statistical_pass": True, "grid_pass": True,
               "joint_distribution_total_variation_upper_bound": .004,
               "scalar": {name: {"estimate": 1., "interval": [.99, 1.01], "relative_standard_error": .01} for name in c.OBSERVABLES}}}
    c.atomic_json(tmp_path / "state.json", state)
    c.atomic_json(tmp_path / "tables/one.json", table)
    index = {"schema": c.SCHEMA, "campaign_signature": "test", "energy_intervals": [],
             "tables": [{"case": "one", "path": "one.json", "sha256": c.file_hash(tmp_path / "tables/one.json")}]}
    c.atomic_json(tmp_path / "tables/index.json", index)
    monkeypatch.setattr(c, "verify_coverage", lambda *args: None)
    return manifest, table, index


def test_valid_response_handoff_never_claims_local_cross_sections(tmp_path, monkeypatch):
    manifest, _, _ = synthetic_handoff(tmp_path, monkeypatch)
    result = c.verify_tables(tmp_path, manifest)
    assert result["response_tables_ready"]
    assert not result["local_elastic_collision_law_ready"] and not result["geant4_drop_in"]


def test_baseline_fallback_export_uses_selected_blocks_and_weight_bound(tmp_path, monkeypatch):
    """Exercise export selection, not scientific qualification of synthetic data."""
    from dataclasses import asdict
    policy = c.Policy()
    record = {"water_molecules": 1, "volume_angstrom3": 1.}
    direction = {"name": "test", "vector": [1., 0., 0.]}
    manifest = {"signature": "abc", "projectile": "C", "policy": asdict(policy), "structures": [record],
                "directions": [direction], "estimand": "synthetic selection test", "path_length_angstrom": 1.,
                "orientation_domain": "one synthetic direction"}
    base = c.save_block(tmp_path, {**identity(), "phase": "baseline"}, block())
    other = block(); other["raw"][:, :4] *= 2
    targeted = c.save_block(tmp_path, {**identity(), "phase": "validation"}, other)
    scalar = {name: {"estimate": value, "interval": [.99 * value, 1.01 * value], "relative_standard_error": .01}
              for name, value in zip(c.OBSERVABLES, [1., .2, .1, .2, .1])}
    report = {"statistical_pass": True, "grid_pass": True, "scalar": scalar,
              "physical_history_support": [1., policy.max_collisions, 1., 2.],
              "importance_normalization_interval": [.99, 1.01], "individual_alpha": .001,
              "joint_distribution_total_variation_upper_bound": .004,
              "scalar_lower": [.99, .99, .198, .099, 0., 0.],
              "scalar_upper": [1.01, 1.01, .202, .101, 0., 0.]}
    case = {"structure": 0, "direction": 0, "energy_ev": 1., "round": 0, "phase": "baseline",
            "selected_cohort": "baseline", "proposal": {"alpha": [.2, .8]}, "design_path": "synthetic",
            "design_sha256": "synthetic",
            "replay": {"passes": True}, "level": 0, "grids": [[[0., 1.], [0., 1.]]],
            "cohorts": {"baseline_0": {"blocks": [base]}, "validation_0": {"blocks": [targeted]}}}
    c.atomic_json(tmp_path / "state.json", {"cases": {"one": case}, "intervals": []})
    monkeypatch.setattr(c, "verify", lambda *args: manifest)
    monkeypatch.setattr(c, "verify_coverage", lambda *args: None)
    monkeypatch.setattr(c, "reconcile", lambda *args: None)
    monkeypatch.setattr(c, "assess", lambda *args: report)
    monkeypatch.setattr(c, "advance_grid", lambda *args: None)
    monkeypatch.setattr(c, "readiness", lambda *args: {})
    monkeypatch.setattr(c, "execute_block", lambda *args: c.load_block(tmp_path, base))
    c.execute_verification(tmp_path, manifest, case, c.verification_identity(manifest, case, "one"))
    original_interval, bounds = c.empirical_interval, []
    def interval(n, total, total2, bound, alpha):
        bounds.append(bound)
        return original_interval(n, total, total2, bound, alpha)
    monkeypatch.setattr(c, "empirical_interval", interval)
    c.export(tmp_path)
    table = c.read(tmp_path / "tables/one.json")
    assert table["selected_evidence_cohort"] == "baseline"
    assert table["raw_weighted_event_blocks"][0]["path"] == base["path"]
    assert table["macroscopic_bin_rate_per_cm"] == [1e8]
    assert bounds == [2. * policy.max_collisions]
    assert c.read(tmp_path / "production_handoff.json")["response_tables_ready"]


@pytest.mark.parametrize("fault", ["units", "tv", "relative", "joint_cell", "particle"])
def test_handoff_rechecks_scientific_contract_not_just_pass_flags(tmp_path, monkeypatch, fault):
    manifest, table, index = synthetic_handoff(tmp_path, monkeypatch)
    if fault == "units": table["effective_bin_area_per_water_cm2"][0] *= 1e8
    if fault == "tv": table["uncertainty"]["joint_distribution_total_variation_upper_bound"] = .01
    if fault == "relative": table["uncertainty"]["scalar"][c.OBSERVABLES[0]]["relative_standard_error"] = .5
    if fault == "joint_cell": table["nonzero_bin_indices_target_angle_recoil"][1] = [0, 0, 0]
    if fault == "particle": table["projectile"] = "O"
    c.atomic_json(tmp_path / "tables/one.json", table)
    index["tables"][0]["sha256"] = c.file_hash(tmp_path / "tables/one.json")
    c.atomic_json(tmp_path / "tables/index.json", index)
    with pytest.raises(ValueError): c.verify_tables(tmp_path, manifest)


@pytest.mark.skipif(os.environ.get("NLH_REAL_RUNTIME_TESTS") != "1", reason="Explicit opt-in: real retained-runtime checks, at most two local workers")
def test_real_runtime_materials_policy_replay_and_resume(tmp_path):
    source = Path(c.__file__).resolve()
    config = source.with_name("carbon_ice.json")
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    def cli(*args):
        result = subprocess.run([sys.executable, "-B", str(source), *map(str, args)], env=env, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    carbon = tmp_path / "carbon"
    assert cli("prepare", carbon, config)["cases"] == 60
    plan = cli("feasibility", carbon)
    assert len(plan["checks"]) == 18 and len(plan["blocked_cases"]) == 18
    stopped = cli("run", carbon, "--workers", 2, "--max-new-histories", 4)
    assert stopped["committed_histories"] == 0
    assert stopped["reason"] == "binary_differential_stage_incomplete"
    assert stopped["binary_dcs"]["qualified_pair_energy_maps"] == 2
    # A narrow real energy interval exercises the complete pair stage without
    # running a physical full-domain campaign on a login node.
    narrow = c.read(config)
    narrow["energies_ev"] = [1000., 1001.]
    narrow["kernel_manifest"] = str((config.parent / narrow["kernel_manifest"]).resolve())
    for structure in narrow["structures"]:
        for key in ("path", "metadata"):
            if key in structure:
                structure[key] = str((config.parent / structure[key]).resolve())
    narrow_path = tmp_path / "narrow.json"
    c.atomic_json(narrow_path, narrow)
    oxygen = tmp_path / "oxygen"
    assert cli("prepare", oxygen, narrow_path, "--projectile", "O", "--material", "amorphous")["cases"] == 2
    for attempt in range(8):
        result = cli("run", oxygen, "--workers", 2, "--max-new-histories", 4)
        if result["binary_dcs"]["tables_verified"]:
            break
        assert result["binary_dcs"]["status"] == "pending"
    assert result["binary_dcs"]["tables_verified"]
    assert result["committed_histories"] == 4
    before = {p.name: c.file_hash(p) for p in (oxygen / "blocks").glob("*.npz")}
    pair_before = {p.name: c.file_hash(p) for p in (oxygen / "differential").glob("*.json") if p.name != "index.json"}
    assert cli("run", oxygen, "--workers", 1, "--max-new-histories", 4)["committed_histories"] == 8
    assert all(c.file_hash(oxygen / "blocks" / name) == sha for name, sha in before.items())
    assert all(c.file_hash(oxygen / "differential" / name) == sha for name, sha in pair_before.items())
    reused = tmp_path / "oxygen_reused"
    assert cli("prepare", reused, narrow_path, "--projectile", "O", "--material", "amorphous",
               "--reuse-pairs", oxygen)["committed_histories"] == 0
    imported = c.read(reused / "state.json")["differential"]
    assert len(imported["nodes"]) == len(pair_before)
    assert not imported["tables_verified"]  # Fresh consumer checks, not relabelled acceptance.
    assert cli("run", reused, "--workers", 1, "--max-new-histories", 4)["committed_histories"] == 4
    new_manifest = c.read(reused / "manifest.json")
    support = c.assessment_physics(reused, new_manifest, 1001.)
    assert support["tail_support"]["minimum_recoil_fraction"] > 0
    assert c.physical_scalar_bounds(support, 1001.)[1] >= 1
    assert not (reused / "differential/work").exists()  # No new pair solves.
    assert all(c.file_hash(oxygen / "differential" / name) == sha for name, sha in pair_before.items())
    result = subprocess.run([sys.executable, "-B", str(source), "export", str(oxygen)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "not qualified" in result.stderr
    # Fresh worker imports avoid mixing retained runtimes between campaigns.
    script = '''
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import certify as c
root = c.Path(sys.argv[2]); manifest = c.verify(root); state = c.read(root / "state.json")
for case in state["cases"].values():
    for cohort in case["cohorts"].values():
        for entry in cohort["blocks"]:
            expected = c.load_block(root, entry)
            actual = c.execute_block((root, manifest, case, c.full_identity(root, entry)))
            for name in expected:
                if name != "cpu_seconds":
                    assert np.array_equal(expected[name], actual[name]), name
            segments = actual["segments"]
            assert np.all(segments[:, 2] >= segments[:, 1])
            for row, raw in enumerate(actual["raw"]):
                selected = segments[segments[:, 0] == row]
                assert np.isclose(np.sum(selected[:, 2] - selected[:, 1]), raw[0])
print("exact replay")
'''
    checked = subprocess.run([sys.executable, "-B", "-c", script, str(source.parent), str(oxygen)], env=env, text=True, capture_output=True, check=True)
    assert "exact replay" in checked.stdout
