"""Physical support, exact score accounting and independent tail estimation."""
from dataclasses import asdict
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("tail_runner", Path(__file__).parents[1] / "certify.py")
c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c
spec.loader.exec_module(c)


def manifest(m=12e9, targets=(1e9, 16e9), floor=1000.):
    return {"policy": asdict(c.Policy()), "path_length_angstrom": 100., "tail_support": {
        "projectile_mass_c2_ev": m, "target_mass_c2_ev": list(targets),
        "energy_floor_ev": floor, "model": "stationary_targets_nonenergizing_primary_sequential_binary"}}


@pytest.mark.parametrize("mass", [1e9, 4e9, 12e9, 16e9, 32e9])
def test_angular_budget_encloses_exact_relativistic_multicollision_paths(mass):
    c.differential_backend()
    backend = sys.modules["_differential"]
    rng = np.random.default_rng(518)
    for initial in (1000., 1e4, 1e6, 1e8):
        for mode in ("weak", "mixed", "strong"):
            energy, angular, recoil = initial, 0., 0.
            for _ in range(128):
                if energy < 1000.:
                    break
                target = float(rng.choice([1e9, 16e9]))
                momentum = math.sqrt(energy * (energy + 2*mass))
                invariant = math.sqrt((mass+target)**2 + 2*target*energy)
                kin = SimpleNamespace(projectile_mass_c2_ev=mass, target_mass_c2_ev=target,
                    projectile_energy_ev=energy, momentum_cm_ev_c=target*momentum/invariant,
                    beta_cm=momentum/(mass+target+energy), gamma_cm=(mass+target+energy)/invariant)
                theta = float(rng.uniform(.8, 1.)*math.pi if mode == "strong" else
                              10**rng.uniform(-6, -2 if mode == "weak" else .49))
                angle, transfer = backend.two_body_observables(kin, theta)
                angular += 2*math.sin(float(angle)/2)**2
                recoil += float(transfer)
                energy -= float(transfer)
            bounds = c.physical_scalar_bounds(manifest(mass), initial)
            assert recoil <= bounds[2]
            assert angular <= bounds[3]


def test_angular_bound_does_not_depend_on_runtime_collision_cap():
    one = manifest()
    bound = c.physical_scalar_bounds(one, 1e8)[3]
    assert 35 < bound < 36
    one["policy"]["max_collisions"] = 100
    assert c.physical_scalar_bounds(one, 1e8)[3] == bound
    assert c.physical_scalar_bounds(one, 1000.)[3] == pytest.approx(2.)


def test_physical_count_bound_uses_energy_loss_not_computational_cap():
    one = manifest()
    one["tail_support"]["minimum_recoil_fraction"] = .001
    upper = c.physical_scalar_bounds(one, 1e8)[1]
    assert upper >= math.log(1e5)/-math.log1p(-.001)
    one["policy"]["max_collisions"] = 8
    assert c.physical_scalar_bounds(one, 1e8)[1] == upper
    assert upper > 10000
    assert c.physical_scalar_bounds(one, 1000.)[1] == pytest.approx(1.)


@pytest.mark.parametrize("edges", [(0., .2, .1, 1.), (0., .1, .99), (0., float("nan"), 1.)])
def test_tail_policy_rejects_missing_or_overlapping_support(edges):
    with pytest.raises(ValueError, match="Tail score"):
        c.Policy(tail_score_fractions=edges).validate()


def test_partition_is_exact_including_edges_zero_and_extremes():
    x = np.array([0., 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.])
    scores, bounds = c.tail_scores(x, 1., c.Policy().tail_score_fractions)
    assert np.array_equal(scores.sum(axis=1), x)
    assert np.all(scores <= bounds)
    assert np.count_nonzero(scores[-1]) == 1
    with pytest.raises(ValueError, match="physical tail support"):
        c.tail_scores(np.array([1.001]), 1., c.Policy().tail_score_fractions)


def test_tail_checkpoint_accounting_zero_tail_limits_and_resume(monkeypatch):
    physical = c.physical_scalar_bounds(manifest(), 1e8)
    raw = np.array([[100., 1., 1e-4, 1e-8, 0, 0], [100., 2., 2e6, .03, 0, 0],
                    [100., 0., 0., 0., 0, 0], [100., 1., 1., 1e-7, 0, 0]])
    weights = np.array([1., .5, 1.5, 1.])
    block = {"raw": raw, "weights": weights}
    monkeypatch.setattr(c, "load_block", lambda *args: block)
    entries = [{"identity": {"start": 0, "stop": 4}}]
    def evaluate():
        return c.scalar_betting_bounds(None, entries, np.r_[physical*2, 2.], .001, 2,
            physical_bounds=physical, tail_fractions=c.Policy().tail_score_fractions)
    lo, hi, tails = evaluate()
    point = np.r_[(raw[:, :4]*weights[:, None]).mean(axis=0), weights.mean()]
    assert np.all(lo <= point) and np.all(point <= hi)
    for col, name in ((2, "recoil"), (3, "angular")):
        assert sum(tails[name]["means"]) == pytest.approx(point[col])
        for count, interval in zip(tails[name]["nonzero_histories"], tails[name]["intervals"]):
            if count == 0:
                assert interval[0] == 0 and interval[1] > 0
    again = evaluate()
    assert np.array_equal(lo, again[0]) and np.array_equal(hi, again[1]) and tails == again[2]


def test_tail_confidence_error_budget_includes_all_methods():
    p = c.Policy()
    family = 5 + 2*(len(p.tail_score_fractions)-1)
    assert c.scalar_confidence_alpha(p)*family*2*p.max_cases*p.max_proposal_rounds == pytest.approx((1-p.confidence)/4)


def test_tail_training_uses_total_precision_and_retains_defensive_sampling(monkeypatch):
    n = 200
    points = ((np.arange(n)+.5)/n)[:, None]
    raw = np.tile([100., 1., 1., 1e-8, 0., 0.], (n, 1))
    raw[:20, 2] = 1e6
    monkeypatch.setattr(c, "proposal_boxes", lambda *args: [{"centre": [.05], "half_width": [.05]}])
    result = c.fit_proposal(points, np.ones(n), np.ones(n), raw, [], c.Policy(),
                            tail_support=c.physical_scalar_bounds(manifest(), 1e8))
    assert result["alpha"][0] >= c.Policy().defensive_fraction
    assert result["fit"]["tail_score_objectives_included"]
    assert result["fit"]["independent_validation_required"]


def test_pair_reuse_rejects_changed_physics_or_numerical_policy(tmp_path):
    original = {"projectile": "C", "runtime_commit": "abc", "kernel_csv_sha256": "123",
        "minimum_turning_potential_ev": 30., "pair_energies_ev": {"H": [1000., 10000.]},
        "implementation_modules": {"_differential.py": "sha"}, "environment": {},
        "policy": asdict(c.Policy())}
    original["signature"] = c.digest(original)
    c.atomic_json(tmp_path / "manifest.json", original)
    c.atomic_json(tmp_path / "differential/index.json", {"campaign_signature": original["signature"],
        "status": "numerically_qualified_binary_differential_cross_sections"})
    record, _ = c.pair_reuse_source(tmp_path, original)
    assert record["source_signature"] == original["signature"]
    changed = {**original, "projectile": "O"}
    with pytest.raises(ValueError, match="projectile"):
        c.pair_reuse_source(tmp_path, changed)
    changed = {**original, "policy": {**original["policy"], "pair_distribution_tv_tolerance": .01}}
    with pytest.raises(ValueError, match="numerical policy"):
        c.pair_reuse_source(tmp_path, changed)
