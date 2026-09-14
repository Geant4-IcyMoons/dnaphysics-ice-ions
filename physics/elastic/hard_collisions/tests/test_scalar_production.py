"""Scalar production gates and handoff; synthetic evidence is not ice validation."""
from dataclasses import asdict
import copy
import pytest
from test_certify import c


def report(value=1., se=.02):
    return {"scalar_pass": True, "statistical_pass": False, "grid_pass": False,
            "importance_normalization_interval": [.9, 1.1],
            "scalar": {name: {"estimate": value, "standard_error": se,
                              "relative_standard_error": se/value} for name in c.OBSERVABLES}}


def test_separate_acceptance_without_falsifying_distribution():
    r = report()
    assert c.acceptance_passes(r, c.Policy(output_scope="scalar"))
    assert not c.acceptance_passes(r, c.Policy())
    assert not r["statistical_pass"] and not r["grid_pass"]


def test_scalar_interpolation_pass_refine_and_sample():
    p = c.Policy(output_scope="scalar")
    assert c.scalar_interpolation_check(report(), report(), report(), p)["passes"]
    failed = c.scalar_interpolation_check(report(), report(2.), report(), p)
    assert failed["resolved_failure"] and not failed["passes"]
    noisy = c.scalar_interpolation_check(report(se=.2), report(se=.2), report(se=.2), p)
    assert not noisy["passes"] and not noisy["resolved_failure"]
    assert not c.scalar_interpolation_check(report(se=0), report(se=0), report(se=0), p)["passes"]


def test_growth_uses_final_budget_remainder():
    p = c.Policy(output_scope="scalar", max_case_histories=3000)
    assert c.next_sample_target(2048, p) == 2560
    assert c.next_sample_target(2560, p) == 3000
    assert c.next_sample_target(3000, p) == 3000


def test_scalar_export_and_readonly_handoff(tmp_path, monkeypatch):
    p = c.Policy(output_scope="scalar")
    manifest = {"signature": "synthetic", "projectile": "C", "policy": asdict(p),
                "structures": [{}], "directions": [{}], "path_length_angstrom": 100., "estimand": "test"}
    case = {"structure": 0, "direction": 0, "energy_ev": 1000., "assessment": report(),
            "phase": "baseline", "selected_cohort": "baseline", "round": 0,
            "cohorts": {"baseline_0": {"blocks": []}}}
    state = {"cases": {"one": case}, "intervals": []}
    monkeypatch.setattr(c, "verify_coverage", lambda *a: None)
    monkeypatch.setattr(c, "readiness", lambda *a: {})
    index = c.export_scalars(tmp_path, manifest, state)
    assert not index["joint_distribution_qualified"]
    handoff = c.verify_tables(tmp_path, manifest)
    assert handoff["scalar_response_ready"] and not handoff["response_tables_ready"]
    table = c.read(tmp_path / "tables/one.json")
    assert "joint_probability" not in table
    assert not table["uncertainty"]["grid_pass"]
    for se in (0., .1, float('nan')):
        bad = copy.deepcopy(table)
        bad["uncertainty"]["scalar"][c.OBSERVABLES[0]]["standard_error"] = se
        with pytest.raises(ValueError): c.verify_scalar_table(bad, case, manifest)
    bad = copy.deepcopy(table); bad["joint_distribution_qualified"] = True
    with pytest.raises(ValueError): c.verify_scalar_table(bad, case, manifest)


def test_scalar_advance_does_not_call_histogram_interpolation(tmp_path, monkeypatch):
    p = c.Policy(output_scope="scalar")
    cases = {k: {"status": "qualified", "assessment": report(), "energy_ev": e}
             for k,e in zip(('a','m','b'), (100.,1000.,10000.))}
    state = {"cases": cases, "intervals": [{"a": "a", "mid": "m", "b": "b", "depth": 0, "status": "pending"}]}
    def forbidden(*args): raise AssertionError("Scalar production requested TV interpolation")
    monkeypatch.setattr(c, "tv_interpolation_check", forbidden)
    c.advance_grid(tmp_path, {"policy": asdict(p)}, state)
    assert state["intervals"][0]["status"] == "qualified"
    assert state["intervals"][0]["assessment"]["joint_distribution"]["qualified"] is False
