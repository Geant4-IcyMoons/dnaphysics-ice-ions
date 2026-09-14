"""Pair-product software evidence; synthetic potentials are not ice validation."""
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("hard_differential", Path(__file__).parents[1] / "_differential.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def kinematics(energy=10., m1=12., m2=1.):
    invariant = math.sqrt((m1 + m2) ** 2 + 2 * m2 * energy)
    momentum_lab = math.sqrt(energy * (energy + 2 * m1))
    beta = momentum_lab / (m1 + m2 + energy)
    return SimpleNamespace(projectile_mass_c2_ev=m1, target_mass_c2_ev=m2,
                           projectile_energy_ev=energy, momentum_cm_ev_c=m2 * momentum_lab / invariant,
                           beta_cm=beta, gamma_cm=1 / math.sqrt(1 - beta * beta))


class Table:
    minimum_turning_potential_ev = 30.
    def pair_kinematics(self, projectile, target, energy):
        return kinematics(energy)
    def maximum_impact_parameter_angstrom(self, *args):
        return 1.
    def area_quantile_breakpoints(self, *args):
        return np.linspace(0., 1., 17)


class Kernel:
    def solve(self, impact):
        return SimpleNamespace(theta_cm_rad=math.pi / (1 + 40 * impact * impact))


def build(monkeypatch, **overrides):
    monkeypatch.setattr(d, "_direct_kernels", lambda *args: [Kernel(), Kernel()])
    options = dict(relative_tolerance=.005, tv_tolerance=.005, quadrature_orders=[32, 64],
                   max_points=20000, max_refinements=12, bin_count=8)
    options.update(overrides)
    return d.build_pair_differential(Table(), "C", "H", 10., **options)


def test_kinematics_keeps_tiny_recoil_and_energy_conservation():
    kin = kinematics()
    angle, recoil = d.two_body_observables(kin, np.array([0., 1e-10, math.pi]))
    assert recoil[0] == 0 and recoil[1] > 0
    assert np.all(recoil <= kin.projectile_energy_ev)
    assert np.allclose((kin.projectile_energy_ev - recoil) + recoil, kin.projectile_energy_ev)
    assert angle[1] > 0


def test_joint_preimage_includes_both_heavy_projectile_angle_branches():
    kin = kinematics()
    q = np.linspace(0., 1., 101)
    theta = math.pi * (1 - q)
    ae, re = [0., .04, math.pi], [0., 10.]
    mass = d.joint_bin_masses(kin, q, theta, ae, re)
    sampled_angle, _ = d.two_body_observables(kin, math.pi * (1 - (np.arange(100000) + .5) / 100000))
    assert mass[0, 0] == pytest.approx(np.mean(sampled_angle < .04), abs=3e-5)
    roots = d._cm_boundaries(kin, np.array(ae), np.array(re))
    assert len(roots) == 4  # Two internal roots, not one monotone lab branch.


def test_joint_bins_are_correlated_and_normalized():
    kin = kinematics(m1=1., m2=16.)
    q = np.linspace(0., 1., 101)
    theta = math.pi * (1 - q)
    ae, re = np.linspace(0., math.pi, 9), np.linspace(0., 10., 9)
    mass = d.joint_bin_masses(kin, q, theta, ae, re)
    assert mass.sum() == pytest.approx(1.)
    assert np.count_nonzero(mass) < mass.size / 2
    independent = mass.sum(axis=0)[None, :] * mass.sum(axis=1)[:, None]
    assert not np.allclose(mass, independent)


def test_pair_backend_refines_reuses_knots_and_emits_json(monkeypatch):
    product = build(monkeypatch)
    assert product["passes"], product
    assert set(Table().area_quantile_breakpoints()).issubset(set(product["area_quantile"]))
    assert product["hard_cross_section_cm2"] == pytest.approx(math.pi * 1e-16)
    assert sum(product["joint_bins"]["probability_mass"]) == pytest.approx(1.)
    assert sum(product["joint_bins"]["integrated_cross_section_cm2"]) == pytest.approx(math.pi * 1e-16)
    assert product["numerical_checks"]["bin_TV_with_root_allowance"] <= .005
    assert not product["physical_model_validated"]
    json.dumps(product, allow_nan=False)


def test_budget_exhaustion_does_not_certify(monkeypatch):
    product = build(monkeypatch, max_points=5)
    assert not product["passes"] and "budget" in product["reason"]
    assert product["evaluations"]["area_points"] == 5


def test_reference_cache_resumes_without_repeating_completed_solves(monkeypatch):
    cache = {}
    observed = []
    stopped = build(monkeypatch, max_points=25, reference_cache=cache,
                    progress=lambda completed, total: observed.append(len(cache["points"])))
    assert not stopped["passes"] and observed == list(range(1, 26))
    before = json.loads(json.dumps(cache))
    resumed = build(monkeypatch, reference_cache=cache)
    fresh = build(monkeypatch)
    assert resumed["passes"]
    assert resumed["area_quantile"] == fresh["area_quantile"]
    assert resumed["theta_cm_rad"] == fresh["theta_cm_rad"]
    assert resumed["maximum_direct_quadrature_relative_change"] == fresh["maximum_direct_quadrature_relative_change"]
    assert resumed["evaluations"]["new_kernel_solves"] == fresh["evaluations"]["kernel_solves"] - 50
    assert all(cache["points"][k] == v for k, v in before["points"].items())


def test_reference_cache_rejects_changed_tolerance_and_corrupt_evidence(monkeypatch):
    cache = {}
    build(monkeypatch, max_points=5, reference_cache=cache)
    with pytest.raises(ValueError, match="different source"):
        build(monkeypatch, relative_tolerance=.01, reference_cache=cache)
    next(iter(cache["points"].values()))["quadrature_relative_change"] = 1.
    with pytest.raises(ValueError, match="Invalid completed"):
        build(monkeypatch, reference_cache=cache)


def test_equal_mass_head_on_has_no_nonzero_outgoing_energy():
    kin = kinematics(m1=1., m2=1.)
    _, recoil = d.two_body_observables(kin, math.pi)
    assert recoil == pytest.approx(kin.projectile_energy_ev, rel=1e-14)
    # The outgoing direction of a stopped projectile is not physically defined;
    # it must not be used as an independent transport event with finite energy.


def test_one_ulp_duplicate_preimage_knot_is_compressed_without_losing_area():
    first = 2.1535572730464816e-13
    q = np.array([0., first, np.nextafter(first, math.inf), 1.])
    theta = np.array([math.pi, 3.12731841175192, 3.12731841175192, .01])
    compressed, values, width = d._compress_roundoff_plateaus(q, theta)
    assert compressed[0] == 0 and compressed[-1] == 1
    assert len(compressed) == 3 and np.all(np.diff(values) < 0)
    assert width == q[2] - q[1]
    assert np.diff(compressed).sum() == 1.


def test_roundoff_plateau_handling_does_not_hide_broad_flat_reference():
    with pytest.raises(d._Limit, match="plateau wider"):
        d._compress_roundoff_plateaus(np.array([0., .1, .2, 1.]), np.array([math.pi, 2., 2., .01]))
    q, theta, width = d._compress_roundoff_plateaus(np.array([0., .1, .2, 1.]), np.array([math.pi, 2., 2.001, .01]))
    assert np.any(np.diff(theta) > 0) and width == 0  # Caller must still refuse it.


def test_preimage_validation_does_not_insert_subprecision_neighbour_knots():
    kin = kinematics()
    q = np.linspace(0., 1., 33)
    theta = math.pi / (1 + 40 * q)
    ae, re = d._bin_edges(kin, q, theta, 32)
    reference = lambda value: math.pi / (1 + 40 * value)
    mass, allowance, roots = d._reference_bins(reference, kin, q, theta, ae, re, .005)
    boundaries = d._cm_boundaries(kin, ae, re)
    q_tolerance = .005 / (16 * max(1, len(boundaries) - 2))
    assert all(np.min(np.abs(q - root)) > q_tolerance for root in roots)
    assert mass.sum() == pytest.approx(1.) and allowance < .005


def test_positive_integral_matches_analytic_coulomb_over_many_decades():
    nodes, weights = np.polynomial.legendre.leggauss(192)
    angles, weights = (nodes + 1) * math.pi / 4, weights * math.pi / 4
    for energy in (1., 1e6):
        for impact in np.geomspace(1e-8, 1e4, 25):
            radius = 1 / (2 * energy) + math.hypot(1 / (2 * energy), impact)
            got, residual, bound = d._positive_deflection(impact, radius, energy, [1.], [0.], 1., angles, weights)
            expected = 2 * math.atan(1 / (2 * energy * impact))
            assert got == pytest.approx(expected, rel=2e-13, abs=0)
            assert residual <= bound
    assert d._positive_deflection(0., 1., 1., [1.], [0.], 1., angles, weights)[0] == math.pi


def test_positive_integral_refuses_inaccurate_turning_radius():
    nodes, weights = np.polynomial.legendre.leggauss(32)
    with pytest.raises(ValueError, match="Turning-point identity"):
        d._positive_deflection(1., 1., 1., [1.], [0.], 1., (nodes + 1) * math.pi / 4, weights * math.pi / 4)


def test_reference_cache_rejects_old_cancellation_integral(monkeypatch):
    cache = {}
    build(monkeypatch, max_points=5, reference_cache=cache)
    cache["identity"].pop("angular_integral")
    with pytest.raises(ValueError, match="different source"):
        build(monkeypatch, reference_cache=cache)


def test_unconverged_quadrature_blocks_product(monkeypatch):
    class Shifted(Kernel):
        def solve(self, impact):
            return SimpleNamespace(theta_cm_rad=.7 * super().solve(impact).theta_cm_rad)
    monkeypatch.setattr(d, "_direct_kernels", lambda *args: [Kernel(), Shifted()])
    product = d.build_pair_differential(Table(), "C", "H", 10., relative_tolerance=.005,
            tv_tolerance=.005, quadrature_orders=[32, 64], max_points=100,
            max_refinements=2, bin_count=8)
    assert not product["passes"] and "quadrature" in product["reason"]


def test_zero_hard_cross_section_is_analytic_not_sampling(monkeypatch):
    table = Table()
    table.maximum_impact_parameter_angstrom = lambda *args: 0.
    monkeypatch.setattr(d, "_direct_kernels", lambda *args: pytest.fail("Closed channel must not integrate"))
    product = d.build_pair_differential(table, "C", "H", 10., relative_tolerance=.005,
            tv_tolerance=.005, quadrature_orders=[32, 64], max_points=100,
            max_refinements=2, bin_count=8)
    assert product["passes"] and product["status"] == "analytically_closed"
    assert product["hard_cross_section_cm2"] == 0


def test_joint_map_rejects_nonmonotone_or_incomplete_input():
    with pytest.raises(ValueError):
        d.joint_bin_masses(kinematics(), [0., .5, 1.], [3., 1., 2.], [0., math.pi], [0., 10.])
    with pytest.raises(ValueError):
        d.joint_bin_masses(kinematics(), [0., 1.], [3., 1.], [0., 1.], [0., 10.])


def test_energy_comparison_checks_interpolated_map_and_moments(monkeypatch):
    middle = build(monkeypatch)
    left, right = dict(middle), dict(middle)
    left["energy_ev"], right["energy_ev"] = 5., 20.
    comparison = d.compare_pair_energy(left, middle, right, relative_tolerance=.005, tv_tolerance=.005)
    assert comparison["passes"]
    perturbed = dict(right, theta_cm_rad=(np.asarray(right["theta_cm_rad"]) * .7).tolist())
    assert not d.compare_pair_energy(left, middle, perturbed, relative_tolerance=.005, tv_tolerance=.005)["passes"]


def test_unqualified_energy_map_cannot_pass_interpolation():
    assert not d.compare_pair_energy({"passes": False}, {}, {}, relative_tolerance=.005, tv_tolerance=.005)["passes"]


def adapter_inputs(monkeypatch):
    product = build(monkeypatch)
    left, right = dict(product), dict(product)
    left["energy_ev"], right["energy_ev"] = 5., 20.
    right["theta_cm_rad"] = (np.asarray(product["theta_cm_rad"]) * .5).tolist()
    @dataclass
    class Collision:
        theta_cm_rad: float
        recoil_energy_ev: float = 0.
        projectile_out_energy_ev: float = 0.
        energy_conservation_error_ev: float = 0.
    class ConsumerTable(Table):
        def theta_cm_rad(self, *args):
            return -99.  # Any accidental delegation is visible.
        def collide(self, projectile, target, energy, impact):
            return Collision(self.theta_cm_rad(projectile, target, energy, impact * impact))
        def hard_cross_section_angstrom2(self, *args):
            return math.pi
        def hard_moment_cross_sections(self, projectile, target, energy, *, quadrature_order=8, quadrature_relative_tolerance=5e-4):
            raise AssertionError("Original moment map must not be used")
    monkeypatch.setattr(d, "_hard_moment_result", lambda *values: SimpleNamespace(
        cross_section_angstrom2=values[0], recoil_energy_cross_section_ev_angstrom2=values[1],
        transport_cross_section_angstrom2=values[2], quadrature_relative_error=values[3]))
    intervals = [{"a": "left", "b": "right", "target": "H", "status": "qualified"}]
    return ConsumerTable(), {"left": left, "right": right}, intervals


def test_pair_adapter_uses_checked_leaves_in_existing_collision_methods(monkeypatch):
    table, products, intervals = adapter_inputs(monkeypatch)
    adapter = d.PairMapTable(table, products, intervals)
    q = .5
    left = np.interp(q, products["left"]["area_quantile"], products["left"]["theta_cm_rad"])
    expected = left / math.sqrt(2)
    assert adapter.theta_cm_rad("C", "H", 10., q) == pytest.approx(expected)
    assert adapter.collide("C", "H", 10., math.sqrt(q)).theta_cm_rad == pytest.approx(expected)
    moments = adapter.hard_moment_cross_sections("C", "H", 10.)
    assert moments.recoil_energy_cross_section_ev_angstrom2 > 0
    assert moments.transport_cross_section_angstrom2 > 0
    assert adapter.maximum_impact_parameter_angstrom("C", "H", 10.) == 1.
    assert adapter.energy_bounds_ev("C", "H") == (5., 20.)
    assert adapter.pairs == (("C", "H"),)


def test_pair_adapter_rejects_extrapolation_unqualified_pairs_and_quantiles(monkeypatch):
    adapter = d.PairMapTable(*adapter_inputs(monkeypatch))
    for energy in (4., 21., float("nan")):
        with pytest.raises(ValueError, match="domain"):
            adapter.theta_cm_rad("C", "H", energy, .5)
    with pytest.raises(ValueError, match="not qualified"):
        adapter.theta_cm_rad("C", "O", 10., .5)
    with pytest.raises(ValueError, match="quantile"):
        adapter.theta_cm_rad("C", "H", 10., 1.01)


def test_pair_adapter_uses_leaf_endpoints_not_validation_only_maps(monkeypatch):
    table, products, intervals = adapter_inputs(monkeypatch)
    check = dict(products["left"], energy_ev=10., theta_cm_rad=(np.asarray(products["left"]["theta_cm_rad"]) * .3).tolist())
    products["check"] = check
    adapter = d.PairMapTable(table, products, intervals)
    assert not math.isclose(adapter.theta_cm_rad("C", "H", 10., .5), np.interp(.5, check["area_quantile"], check["theta_cm_rad"]))


def test_pair_adapter_rejects_gap_and_unqualified_leaf(monkeypatch):
    table, products, intervals = adapter_inputs(monkeypatch)
    intervals[0]["status"] = "pending"
    with pytest.raises(ValueError, match="qualified energy leaves"):
        d.PairMapTable(table, products, intervals)
    intervals[0]["status"] = "qualified"
    products["lateleft"] = dict(products["left"], energy_ev=25.)
    products["lateright"] = dict(products["right"], energy_ev=30.)
    intervals.append({"a": "lateleft", "b": "lateright", "target": "H", "status": "qualified"})
    with pytest.raises(ValueError, match="gap or overlap"):
        d.PairMapTable(table, products, intervals)


def test_pair_adapter_preserves_tiny_recoil_and_transport(monkeypatch):
    table, products, intervals = adapter_inputs(monkeypatch)
    for product in products.values():
        product["theta_cm_rad"] = (np.asarray(product["theta_cm_rad"]) * 1e-10).tolist()
    adapter = d.PairMapTable(table, products, intervals)
    collision = adapter.collide("C", "H", 10., .5)
    assert collision.recoil_energy_ev > 0
    assert collision.recoil_energy_ev + collision.projectile_out_energy_ev == 10.
    exact = float(d.two_body_observables(kinematics(10.), collision.theta_cm_rad)[1])
    assert collision.recoil_energy_ev == pytest.approx(exact, rel=1e-14, abs=0)
    moments = adapter.hard_moment_cross_sections("C", "H", 10.)
    assert moments.transport_cross_section_angstrom2 > 0
    assert moments.recoil_energy_cross_section_ev_angstrom2 > 0
