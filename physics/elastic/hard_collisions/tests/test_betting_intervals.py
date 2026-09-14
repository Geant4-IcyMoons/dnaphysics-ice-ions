"""Betting bounds: exact finite laws, weighted samples and checkpoint integration."""
from dataclasses import asdict
import importlib.util
import itertools
import math
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("betting_runner", Path(__file__).parents[1] / "certify.py")
c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c
spec.loader.exec_module(c)


@pytest.mark.parametrize("value", [0., 1e-20, .3, 1.])
def test_single_bet_matches_constant_sequence_analytic_boundary(value):
    n, alpha = 1000, .05
    low, high = c.betting_interval([value], 1., alpha, 1, counts=np.array([n]))
    factor = math.exp(-math.log(2 / alpha) / n)
    expected = [value * factor, value + (1 - value) * -math.expm1(-math.log(2 / alpha) / n)]
    assert low <= expected[0] + 1e-35
    assert high >= expected[1] - 1e-15
    assert low == pytest.approx(expected[0], rel=1e-10, abs=1e-35)
    assert high == pytest.approx(expected[1], rel=1e-10)


def test_zero_events_never_certify_zero_relative_error():
    interval = c.betting_interval([0.], 100., .01, counts=np.array([100000]))
    assert interval[0] == 0. and interval[1] > 0.
    assert not c.relative_precision(0., interval, .05)["passes"]


def test_compression_order_units_and_more_confidence():
    x = np.array([0., .1, .1, .7, .7, .7, 1.] * 12)
    unique, count = np.unique(x, return_counts=True)
    first = c.betting_interval(x, 1., .05)
    assert c.betting_interval(unique, 1., .05, counts=count) == first
    assert c.betting_interval(x[::-1], 1., .05) == first
    assert c.betting_interval(x * 1e8, 1e8, .05) == pytest.approx(np.array(first) * 1e8)
    tighter_confidence = c.betting_interval(x, 1., .001)
    assert tighter_confidence[0] <= first[0] <= first[1] <= tighter_confidence[1]


@pytest.mark.parametrize("values,bound,counts", [([-.1], 1., None), ([1.1], 1., None),
    ([float("nan")], 1., None), ([0.], 0., None), ([.2], 1., np.array([1.5])),
    ([.2], 1., np.array([0]))])
def test_invalid_support_or_importance_weights_as_counts_rejected(values, bound, counts):
    with pytest.raises(ValueError):
        c.betting_interval(values, bound, .05, counts=counts)


def test_exact_anytime_coverage_for_small_bernoulli_laws():
    # Enumerate all paths, not a Monte Carlo approximation to test coverage.
    horizon, alpha = 10, .1
    intervals = {}
    for n in range(1, horizon + 1):
        for k in range(n + 1):
            x, count = ([0., 1.], np.array([n-k, k]))
            keep = count > 0
            intervals[n, k] = c.betting_interval(np.array(x)[keep], 1., alpha, counts=count[keep])
    for p in (.01, .1, .5, .9, .99):
        rejected_probability = 0.
        for path in itertools.product((0, 1), repeat=horizon):
            k, rejected = 0, False
            for n, value in enumerate(path, 1):
                k += value
                low, high = intervals[n, k]
                rejected |= not low <= p <= high
            if rejected:
                rejected_probability += p ** k * (1-p) ** (horizon-k)
        assert rejected_probability <= alpha + 1e-12


def test_weighted_rare_event_mean_uses_generating_density():
    # Physical event probability .002; proposal visits it with probability .8.
    # A likelihood-weighted observation is .002/.8 or zero, not a Bernoulli.
    n = 10000
    values, counts = np.array([0., .002/.8]), np.array([2000, 8000])
    interval = c.betting_interval(values, 1., .001, counts=counts)
    assert interval[0] <= .002 <= interval[1]
    assert interval[1] < c.empirical_interval(n, np.dot(values, counts), np.dot(values**2, counts), 1., .001)[1]
    # A physically justified stratum/support bound, unlike an observed maximum,
    # can be smaller; this demonstrates why support matters even for betting.
    stratum = c.betting_interval(values, .002/.8, .001, counts=counts)
    assert stratum[0] <= .002 <= stratum[1]
    assert stratum[1] - stratum[0] < interval[1] - interval[0]


def test_scalar_budget_is_grid_and_look_independent_and_globally_spent():
    policy = c.Policy()
    alpha = c.scalar_confidence_alpha(policy)
    family = 5 + 2 * (len(policy.tail_score_fractions) - 1)
    assert alpha * 2 * policy.max_cases * policy.max_proposal_rounds * family == pytest.approx((1-policy.confidence)/4)
    changed = c.Policy(max_bin_level=20, max_case_histories=1000000)
    assert c.scalar_confidence_alpha(changed) == alpha


def test_distribution_tolerances_are_independent():
    p = c.Policy(distribution_tv_tolerance=.05)
    assert p.pair_distribution_tv_tolerance == .005
    p.validate()
    with pytest.raises(ValueError):
        c.Policy(pair_distribution_tv_tolerance=0).validate()


def test_realistic_optimistic_recoil_remains_unqualified():
    manifest = {"policy": asdict(c.Policy()), "path_length_angstrom": 100.}
    case = {"energy_ev": 1e8}
    # Full-path importance-normalization fluctuation must not invalidate the
    # ideal planning calculation; ratio estimates remain unchanged.
    result = c.range_budget_diagnostic(manifest, case, {"means": [100.01, 9., .8, 1e-8]}, weight_bound=1.)
    assert not result["sampling_at_pilot_means_can_fit_budget"]
    assert result["zero_variance_precision_at_case_limit"][c.OBSERVABLES[0]]["passes"]


def test_scalar_checkpoint_gap_refuses_certification(tmp_path):
    entries = [{"identity": {"start": 4, "stop": 8}}]
    with pytest.raises(ValueError, match="contiguous"):
        c.scalar_betting_bounds(tmp_path, entries, np.ones(5), .05, 20)


def test_saved_weighted_histories_resume_exactly_and_ignore_histogram_level(tmp_path):
    policy = c.Policy(base_bins=2, max_bin_level=1)
    manifest = {"signature": "test", "policy": asdict(policy), "path_length_angstrom": 1.}
    case = {"structure": 0, "direction": 0, "energy_ev": 1., "round": 0,
            "proposal": {"boxes": [], "alpha": [1.]}, "kernel_mode": "qualified_pair_maps",
            "kernel": {"passes": True}, "statistical_relative_tolerance": .05,
            "level": 0, "grids": [[[0., .5, 1.]] * 2, [[0., .25, .5, 1.]] * 2],
            "design_path": "designs/test.json", "cohorts": {"validation_0": {"blocks": []}}}
    case["design_sha256"] = c.digest(c.frozen_design(case))
    c.atomic_json(tmp_path / case["design_path"], c.frozen_design(case))
    for start in (0, 4):
        raw = np.tile([1., 1., .2, .1, 0., 0.], (4, 1))
        weights = np.array([.5, 1., 1.5, 1.])
        sample = {"raw": raw, "weights": weights, "cpu_seconds": np.ones(4),
                  "events": np.array([[i, 0, .05, .2, 1., 0., 0., 0.] for i in range(4)])}
        identity = {"campaign": "test", "case": "one", "phase": "validation", "round": 0,
                    "start": start, "stop": start+4, "proposal": case["proposal"],
                    "design_sha256": case["design_sha256"]}
        case["cohorts"]["validation_0"]["blocks"].append(c.save_block(tmp_path, identity, sample))
    before = {p.name: c.file_hash(p) for p in (tmp_path / "blocks").glob("*")}
    first = c.assess(tmp_path, manifest, case)
    assert c.assess(tmp_path, manifest, case) == first
    case["level"] = 1
    finer = c.assess(tmp_path, manifest, case)
    assert first["scalar"] == finer["scalar"]
    assert first["scalar_alpha"] == finer["scalar_alpha"]
    assert first["n"] == finer["n"] == 8
    assert {p.name: c.file_hash(p) for p in (tmp_path / "blocks").glob("*")} == before
