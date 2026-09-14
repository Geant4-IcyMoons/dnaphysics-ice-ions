"""Lightweight orchestration checks; synthetic pair products are not physics evidence."""
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("pair_protocol_runner", Path(__file__).parents[1] / "certify.py")
c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c
spec.loader.exec_module(c)


@pytest.fixture
def campaign():
    manifest = {"signature": "synthetic", "projectile": "C", "runtime_commit": "synthetic-runtime",
                "pair_energies_ev": {"H": [1., 16.], "O": [1., 16.]},
                "energies_ev": [8., 16.], "policy": asdict(c.Policy()),
                "kernel_manifest": "synthetic-table", "minimum_turning_potential_ev": 30., "structures": [],
                "production_contracts": {"structured_ice": "Explicit frozen geometry; no density-only closure"}}
    return manifest, {"differential": c.initialize_differential(manifest)}


@pytest.fixture
def product_factory(monkeypatch):
    build, _ = c.differential_backend()
    module = sys.modules[build.__module__]

    class Table:
        minimum_turning_potential_ev = 30.
        def __init__(self, source=None):
            pass
        def pair_kinematics(self, projectile, target, energy):
            m1, m2 = 12., 1. if target == "H" else 16.
            invariant = math.sqrt((m1 + m2) ** 2 + 2 * m2 * energy)
            momentum = math.sqrt(energy * (energy + 2 * m1))
            beta = momentum / (m1 + m2 + energy)
            return SimpleNamespace(projectile_mass_c2_ev=m1, target_mass_c2_ev=m2,
                    projectile_energy_ev=energy, momentum_cm_ev_c=m2 * momentum / invariant,
                    beta_cm=beta, gamma_cm=1 / math.sqrt(1 - beta * beta))
        def maximum_impact_parameter_angstrom(self, *args):
            return 1.
        def area_quantile_breakpoints(self, *args):
            return np.linspace(0., 1., 17)
        def turning_threshold_radius_angstrom(self, *args):
            return 1.

    class Kernel:
        def solve(self, b):
            return SimpleNamespace(theta_cm_rad=math.pi / (1 + 40 * b * b))

    monkeypatch.setattr(module, "_direct_kernels", lambda *args: [Kernel(), Kernel()])
    monkeypatch.setattr(c, "load_runtime", lambda root: (Table, None, None))
    cache = {}
    def make(target, energy):
        key = target, energy
        if key not in cache:
            cache[key] = build(Table(), "C", target, energy, relative_tolerance=.005,
                    tv_tolerance=.005, quadrature_orders=[32, 64], max_points=20000,
                    max_refinements=12, bin_count=8)
            assert cache[key]["passes"]
        return deepcopy(cache[key])
    return make


def write_product(root, manifest, key, product):
    envelope = {"campaign": manifest["signature"], "product": product}
    envelope["sha256"] = c.digest(envelope)
    c.atomic_json(root / "differential" / (key + ".json"), envelope)


def seed_products(root, manifest, stage, make):
    for key, node in stage["nodes"].items():
        path = root / "differential" / (key + ".json")
        if not path.exists():
            write_product(root, manifest, key, make(node["target"], node["energy_ev"]))


def ready_stage(root, campaign, monkeypatch, make):
    manifest, state = campaign
    monkeypatch.setattr(c, "differential_backend", lambda: (None, lambda *args, **kwargs: {"passes": True}))
    seed_products(root, manifest, state["differential"], make)
    assert not c.advance_differential(root, manifest, state)
    seed_products(root, manifest, state["differential"], make)
    assert c.advance_differential(root, manifest, state)
    return manifest, state


def test_pair_domain_includes_reachable_energies_below_entrance(campaign):
    manifest, state = campaign
    assert min(n["energy_ev"] for n in state["differential"]["nodes"].values()) == 1.
    assert min(manifest["energies_ev"]) == 8.
    assert {n["target"] for n in state["differential"]["nodes"].values()} == {"H", "O"}
    c.verify_pair_coverage(manifest, state["differential"])


def test_missing_base_pair_energy_or_interval_is_rejected(campaign):
    manifest, state = campaign
    stage = deepcopy(state["differential"])
    stage["nodes"].pop(c.pair_key("H", 1.))
    with pytest.raises(ValueError, match="Missing requested pair energies"):
        c.verify_pair_coverage(manifest, stage)
    stage = deepcopy(state["differential"])
    stage["intervals"].pop()
    with pytest.raises(ValueError, match="Missing requested pair energy interval"):
        c.verify_pair_coverage(manifest, stage)


def test_pair_receipt_detects_corruption_and_campaign_mismatch(tmp_path, campaign, product_factory):
    manifest, state = campaign
    key = c.pair_key("H", 1.)
    write_product(tmp_path, manifest, key, product_factory("H", 1.))
    assert c.pair_product(tmp_path, manifest, key)["passes"]
    path = tmp_path / "differential" / (key + ".json")
    changed = c.read(path); changed["product"]["hard_cross_section_cm2"] *= 2
    c.atomic_json(path, changed)
    with pytest.raises(ValueError, match="Changed or incompatible"):
        c.pair_product(tmp_path, manifest, key)
    write_product(tmp_path, manifest, key, product_factory("H", 1.))
    with pytest.raises(ValueError, match="Changed or incompatible"):
        c.pair_product(tmp_path, dict(manifest, signature="other"), key)


def test_product_identity_is_checked_even_with_valid_checksum(tmp_path, campaign, product_factory):
    manifest, _ = campaign
    key = c.pair_key("H", 1.)
    write_product(tmp_path, manifest, key, product_factory("O", 1.))
    with pytest.raises(ValueError, match="physical identity"):
        c.pair_product(tmp_path, manifest, key)


def test_committed_orphan_products_are_adopted_without_backend_runs(tmp_path, campaign, product_factory, monkeypatch):
    manifest, state = campaign
    seed_products(tmp_path, manifest, state["differential"], product_factory)
    monkeypatch.setattr(c, "differential_backend", lambda: (lambda *a: pytest.fail("No build needed"), None))
    assert not c.advance_differential(tmp_path, manifest, state)
    for target in ("H", "O"):
        assert state["differential"]["nodes"][c.pair_key(target, 1.)]["status"] == "qualified"
    assert len(state["differential"]["nodes"]) == 10
    assert all(len(i["checks"]) == 3 for i in state["differential"]["intervals"])


def test_failed_energy_check_splits_all_quarter_intervals(tmp_path, campaign, product_factory, monkeypatch):
    manifest, state = ready_stage(tmp_path, campaign, monkeypatch, product_factory)
    for interval in state["differential"]["intervals"]:
        interval["status"] = "pending"
    monkeypatch.setattr(c, "differential_backend", lambda: (None, lambda *a, **k: {"passes": False}))
    assert not c.advance_differential(tmp_path, manifest, state)
    stage = state["differential"]
    assert len(stage["intervals"]) == 10  # Two parents and eight required children.
    c.verify_pair_coverage(manifest, stage)
    stage["intervals"].pop()
    with pytest.raises(ValueError, match="Missing pair refinement child"):
        c.verify_pair_coverage(manifest, stage)


def test_changed_interior_energy_and_duplicate_intervals_rejected(tmp_path, campaign, product_factory, monkeypatch):
    manifest, state = ready_stage(tmp_path, campaign, monkeypatch, product_factory)
    stage = deepcopy(state["differential"])
    stage["intervals"].append(deepcopy(stage["intervals"][0]))
    with pytest.raises(ValueError, match="Duplicated"):
        c.verify_pair_coverage(manifest, stage)
    stage = deepcopy(state["differential"])
    stage["intervals"][0]["checks"][0] = stage["intervals"][0]["checks"][1]
    with pytest.raises(ValueError, match="Changed independent"):
        c.verify_pair_coverage(manifest, stage)


def test_product_limit_blocks_without_false_completion(tmp_path, campaign, product_factory, monkeypatch):
    manifest, state = campaign
    manifest["policy"]["pair_max_products"] = 4
    seed_products(tmp_path, manifest, state["differential"], product_factory)
    monkeypatch.setattr(c, "differential_backend", lambda: (None, None))
    assert not c.advance_differential(tmp_path, manifest, state)
    assert state["differential"]["status"] == "not_qualified"
    assert len(state["differential"]["nodes"]) == 4


def test_export_verifies_bin_units_and_keeps_phase_contract(tmp_path, campaign, product_factory, monkeypatch):
    manifest, state = ready_stage(tmp_path, campaign, monkeypatch, product_factory)
    index = c.export_differential(tmp_path, manifest, state)
    assert state["differential"]["tables_verified"]
    assert not index["phase_operator"]["homogeneous_density_only_closure_qualified"]
    assert index["units"]["area"] == "cm2"
    key = c.pair_key("H", 1.)
    product = c.pair_product(tmp_path, manifest, key)
    product["joint_bins"]["integrated_cross_section_cm2"][0] *= 2
    write_product(tmp_path, manifest, key, product)
    with pytest.raises(ValueError, match="area units"):
        c.export_differential(tmp_path, manifest, state)


def test_export_rechecks_cached_interpolation_acceptance(tmp_path, campaign, product_factory, monkeypatch):
    manifest, state = ready_stage(tmp_path, campaign, monkeypatch, product_factory)
    monkeypatch.setattr(c, "differential_backend", lambda: (None, lambda *a, **k: {"passes": False}))
    with pytest.raises(ValueError):
        c.export_differential(tmp_path, manifest, state)


def test_export_checks_exact_threshold_area_not_only_bin_sum(tmp_path, campaign, product_factory, monkeypatch):
    manifest, state = ready_stage(tmp_path, campaign, monkeypatch, product_factory)
    key = c.pair_key("H", 1.)
    product = c.pair_product(tmp_path, manifest, key)
    product["hard_cross_section_cm2"] *= 2
    product["joint_bins"]["integrated_cross_section_cm2"] = [x * 2 for x in product["joint_bins"]["integrated_cross_section_cm2"]]
    write_product(tmp_path, manifest, key, product)
    with pytest.raises(ValueError):
        c.export_differential(tmp_path, manifest, state)


def test_export_rejects_nonmonotone_cm_map(tmp_path, campaign, product_factory, monkeypatch):
    manifest, state = ready_stage(tmp_path, campaign, monkeypatch, product_factory)
    key = c.pair_key("H", 1.)
    product = c.pair_product(tmp_path, manifest, key)
    product["theta_cm_rad"][1] = product["theta_cm_rad"][0] + .01
    write_product(tmp_path, manifest, key, product)
    with pytest.raises(ValueError):
        c.export_differential(tmp_path, manifest, state)


def test_execute_pair_checkpoints_partial_work_on_exception(tmp_path, campaign, monkeypatch):
    manifest, state = campaign
    key = c.pair_key("H", 1.)
    monkeypatch.setattr(c, "_CACHE", {})
    monkeypatch.setattr(c, "load_runtime", lambda root: (lambda path: object(), None, None))
    def fail(table, *args, reference_cache, **kwargs):
        reference_cache["retained_partial_work"] = [1, 2, 3]
        raise RuntimeError("synthetic interrupted quadrature")
    monkeypatch.setattr(c, "differential_backend", lambda: (fail, None))
    with pytest.raises(RuntimeError, match="interrupted"):
        c.execute_pair((tmp_path, manifest, key, state["differential"]["nodes"][key]))
    checkpoint = c.read(tmp_path / "differential" / "work" / (key + ".json"))
    assert checkpoint["reference_cache"]["retained_partial_work"] == [1, 2, 3]
    checksum = checkpoint.pop("sha256")
    assert checksum == c.digest(checkpoint)
    assert not (tmp_path / "differential" / (key + ".json")).exists()


def test_differential_login_smoke_cap_limits_submissions(tmp_path, campaign, product_factory, monkeypatch):
    manifest, state = campaign
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.setattr(c, "_STOP", False)
    submitted = []
    class InlinePool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, task):
            root, m, key, node = task
            submitted.append(key)
            product = product_factory(node["target"], node["energy_ev"])
            write_product(root, m, key, product)
            future = Future(); future.set_result({"status": "qualified"})
            return future
    monkeypatch.setattr(c, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(c, "differential_backend", lambda: (None, lambda *a, **k: {"passes": True}))
    assert not c.run_differential(tmp_path, manifest, state, 3, c.time.monotonic())
    assert len(submitted) == 2
