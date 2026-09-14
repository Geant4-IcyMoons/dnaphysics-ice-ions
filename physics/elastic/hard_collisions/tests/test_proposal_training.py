"""Synthetic cost/variance optimization, not demonstrated NLH acceleration."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("hard_proposal_certify", Path(__file__).parents[1] / "certify.py")
c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c
spec.loader.exec_module(c)


def setup(monkeypatch, points=None):
    points = ((np.arange(2000) + .5) / 2000)[:, None] if points is None else points
    raw = np.tile([1., 1., 1., .01, 0., 0.], (len(points), 1))
    raw[:, 2] = np.where(points[:, 0] < .2, 10., 1.)
    box = {"centre": [.1], "half_width": [.1]}
    monkeypatch.setattr(c, "proposal_boxes", lambda *args: [box])
    return points, raw, c.Policy(local_centres=1)


def fit(points, raw, policy, *, density=None, costs=None, events=None):
    return c.fit_proposal(points, np.ones(len(raw)), np.ones(len(raw)) if density is None else density,
                          raw, [], policy, costs=costs, events=events)


def test_cost_objective_reduces_expensive_component_allocation(monkeypatch):
    points, raw, policy = setup(monkeypatch)
    constant = fit(points, raw, policy, costs=np.ones(len(raw)))
    expensive = fit(points, raw, policy, costs=np.where(points[:, 0] < .2, 10., 1.))
    # Exact two-region scalar influence optimum for equal CPU cost.
    assert constant["alpha"][0] == pytest.approx(.625, abs=1e-5)
    assert expensive["alpha"][0] > constant["alpha"][0] + .2
    assert expensive["fit"]["estimated_gain"] > 1
    assert constant["fit"]["estimated_gain"] > expensive["fit"]["estimated_gain"]
    assert expensive["fit"]["independent_validation_required"]


def test_cpu_units_do_not_change_the_proposal(monkeypatch):
    points, raw, policy = setup(monkeypatch)
    costs = np.where(points[:, 0] < .2, 4., 1.)
    one, another = fit(points, raw, policy, costs=costs), fit(points, raw, policy, costs=1000 * costs)
    assert np.allclose(one["alpha"], another["alpha"], atol=1e-8, rtol=0)


def test_generating_density_weights_recover_same_physical_objective(monkeypatch):
    points, raw, policy = setup(monkeypatch)
    uniform = fit(points, raw, policy)
    # Half the proposal mass is in the inner fifth: g=2.5 inside, .625 outside.
    x = (np.arange(1000) + .5) / 1000
    points, raw, policy = setup(monkeypatch, np.r_[.2 * x, .2 + .8 * x][:, None])
    density = np.where(points[:, 0] < .2, 2.5, .625)
    before = density.copy()
    targeted = fit(points, raw, policy, density=density)
    assert np.allclose(uniform["alpha"], targeted["alpha"], atol=1e-7, rtol=0)
    assert np.array_equal(density, before)  # Past proposal weights are not overwritten.


def test_joint_categories_can_drive_training_with_constant_scalar_moments(monkeypatch):
    points, raw, policy = setup(monkeypatch)
    raw[:, 2:4] = .2
    assert fit(points, raw, policy) == {"boxes": [], "alpha": [1.]}
    events = np.array([[i, int(x >= .2), .1, .2, 1., 0., 0., 0.]
                       for i, x in enumerate(points[:, 0])])
    proposal = fit(points, raw, policy, events=events)
    assert proposal["fit"]["joint_tv_surrogate_included"]
    assert proposal["alpha"][0] == pytest.approx(.625, abs=1e-5)
    assert proposal["fit"]["estimated_gain"] > 1


def test_invalid_costs_and_event_history_mapping_refused(monkeypatch):
    points, raw, policy = setup(monkeypatch)
    with pytest.raises(ValueError, match="CPU"):
        fit(points, raw, policy, costs=np.zeros(len(raw)))
    with pytest.raises(ValueError, match="complete trajectory"):
        fit(points, raw, policy, events=np.array([[0, 0, .1, .2, 1, 0, 0, 0]]))


def test_training_concatenation_offsets_history_indices_and_preserves_weights(monkeypatch):
    def part(hit, weights):
        raw = np.tile([1., 0., 0., 0., 0., 0.], (2, 1)); raw[hit, 1] = 1
        return {"points": np.zeros((2, 3)), "base": np.ones(2), "density": 1 / np.asarray(weights),
                "raw": raw, "weights": np.asarray(weights), "cpu_seconds": np.ones(2),
                "events": np.array([[hit, 0, .1, .2, 1., 0., 0., 0.]])}
    parts = [part(1, [1., 2.]), part(0, [3., 4.])]
    monkeypatch.setattr(c, "load_block", lambda root, entry: parts[entry["index"]])
    case = {"cohorts": {"training_1": {"blocks": [{"index": 1, "identity": {"start": 0}}]},
                        "training_0": {"blocks": [{"index": 0, "identity": {"start": 0}}]}}}
    combined = c.training_data(None, case)
    assert combined["events"][:, 0].tolist() == [1., 2.]
    assert combined["events"][:, 8].tolist() == [2., 3.]
    assert parts[1]["events"][0, 0] == 0
