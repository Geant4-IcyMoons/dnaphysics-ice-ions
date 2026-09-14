"""Deterministic, model-conditional pair cross sections for the shared runner.

The area measure is d sigma = pi b_max**2 dq.  Its pushforward through the
two-body kinematics preserves angle--recoil correlation and every lab-angle
branch.  Numerical refinement estimates are not rigorous error enclosures or
validation of an independent-atom description of an ice phase.
"""

from __future__ import annotations

import math
from dataclasses import replace
import inspect
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq


class _Limit(RuntimeError):
    pass


class PairMapTable:
    """Use the exported leaf maps in the retained geometry-aware runtime.

    The caller verifies checksums and numerical product evidence. This adapter
    validates the supported interpolation topology and never falls through to
    old angle tables outside the qualified leaf intervals.
    """

    def __init__(self, table, products, intervals):
        self.table = table
        self._maps = {}
        self._leaves = {}
        for interval in intervals:
            if interval["status"] == "split":
                continue
            if interval["status"] != "qualified":
                raise ValueError("Pair adapter requires qualified energy leaves")
            endpoints = []
            for key in (interval["a"], interval["b"]):
                product = products[key]
                if not product.get("passes") or product["status"] != "numerically_qualified_pair":
                    raise ValueError("Pair adapter needs qualified nonzero endpoint maps")
                q, theta = np.asarray(product["area_quantile"], float), np.asarray(product["theta_cm_rad"], float)
                if (q.ndim != 1 or q.shape != theta.shape or len(q) < 2 or q[0] != 0 or q[-1] != 1
                        or np.any(~np.isfinite(q)) or np.any(~np.isfinite(theta))
                        or np.any(np.diff(q) <= 0) or np.any(np.diff(theta) >= 0)
                        or np.any(theta <= 0) or np.any(theta > math.pi)):
                    raise ValueError("Invalid qualified endpoint area map")
                q.setflags(write=False); theta.setflags(write=False)
                self._maps.setdefault(key, (q, theta))
                endpoints.append(product)
            a, b = endpoints
            pair = (a["projectile"], a["target"])
            if (pair != (b["projectile"], b["target"]) or a["target"] != interval["target"]
                    or not 0 < a["energy_ev"] < b["energy_ev"]
                    or a["minimum_turning_potential_ev"] != table.minimum_turning_potential_ev
                    or b["minimum_turning_potential_ev"] != table.minimum_turning_potential_ev):
                raise ValueError("Mismatched pair interpolation endpoints")
            self._leaves.setdefault(pair, []).append((a["energy_ev"], b["energy_ev"], interval["a"], interval["b"]))
        if not self._leaves:
            raise ValueError("No qualified pair interpolation intervals")
        for pair, leaves in self._leaves.items():
            leaves.sort()
            if any(previous[1] != following[0] for previous, following in zip(leaves[:-1], leaves[1:])):
                raise ValueError("Pair interpolation leaves have a gap or overlap")
        self._upper = {pair: np.array([leaf[1] for leaf in leaves]) for pair, leaves in self._leaves.items()}

    def __getattr__(self, name):
        return getattr(self.table, name)

    @property
    def pairs(self):
        return tuple(sorted(self._leaves))

    def energy_bounds_ev(self, projectile, target):
        leaves = self._leaves.get((projectile, target))
        if leaves is None:
            raise ValueError("Requested projectile-target pair is not qualified")
        return leaves[0][0], leaves[-1][1]

    def _leaf(self, projectile, target, energy):
        low, high = self.energy_bounds_ev(projectile, target)
        if not math.isfinite(energy) or not low <= energy <= high:
            raise ValueError("Energy outside qualified pair interpolation domain")
        pair = projectile, target
        return self._leaves[pair][int(np.searchsorted(self._upper[pair], energy, side="left"))]

    def theta_cm_rad(self, projectile, target, energy, quantile):
        low, high, a, b = self._leaf(projectile, target, energy)
        if not math.isfinite(quantile) or not 0 <= quantile <= 1:
            raise ValueError("Area quantile must lie in [0,1]")
        left = float(np.interp(quantile, *self._maps[a]))
        if energy == low:
            return left
        right = float(np.interp(quantile, *self._maps[b]))
        if energy == high:
            return right
        fraction = math.log(energy / low) / math.log(high / low)
        return math.exp((1 - fraction) * math.log(left) + fraction * math.log(right))

    def area_quantile_breakpoints(self, projectile, target, energy):
        low, high, a, b = self._leaf(projectile, target, energy)
        if energy == low:
            return self._maps[a][0]
        if energy == high:
            return self._maps[b][0]
        result = np.union1d(self._maps[a][0], self._maps[b][0])
        result.setflags(write=False)
        return result

    def collide(self, projectile, target, projectile_energy_ev, impact_parameter_angstrom):
        # The retained method calls self.theta_cm_rad: do not delegate the
        # bound table method, which would silently use the original map.
        collision = type(self.table).collide(self, projectile, target, projectile_energy_ev, impact_parameter_angstrom)
        kin = self.pair_kinematics(projectile, target, projectile_energy_ev)
        recoil = 2 * kin.momentum_cm_ev_c ** 2 / kin.target_mass_c2_ev * math.sin(collision.theta_cm_rad / 2) ** 2
        if recoil > projectile_energy_ev * (1 + 32 * np.finfo(float).eps):
            raise ValueError("Two-body recoil exceeds projectile energy")
        recoil = min(recoil, projectile_energy_ev)
        outgoing = projectile_energy_ev - recoil
        return replace(collision, recoil_energy_ev=recoil, projectile_out_energy_ev=outgoing,
                       energy_conservation_error_ev=projectile_energy_ev - (outgoing + recoil))

    def hard_moment_cross_sections(self, projectile, target, projectile_energy_ev, *,
                                    quadrature_order=None, quadrature_relative_tolerance=None):
        # Defaults remain owned by the retained common runtime. Here only the
        # angle lookup and cancellation-prone moment evaluation change.
        parameters = inspect.signature(type(self.table).hard_moment_cross_sections).parameters
        order = quadrature_order if quadrature_order is not None else parameters["quadrature_order"].default
        tolerance = quadrature_relative_tolerance if quadrature_relative_tolerance is not None else parameters["quadrature_relative_tolerance"].default
        if type(order) is not int or order < 2 or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Invalid pair-moment quadrature settings")
        low, high, a, b = self._leaf(projectile, target, projectile_energy_ev)
        sigma = self.hard_cross_section_angstrom2(projectile, target, projectile_energy_ev)
        if sigma == 0:
            return _hard_moment_result(0., 0., 0., 0.)
        q = self.area_quantile_breakpoints(projectile, target, projectile_energy_ev)
        kin = self.pair_kinematics(projectile, target, projectile_energy_ev)
        fraction = math.log(projectile_energy_ev / low) / math.log(high / low)
        def integrate(n):
            x, weights = np.polynomial.legendre.leggauss(n)
            width = np.diff(q) / 2
            probe = (q[:-1] + q[1:])[:, None] / 2 + width[:, None] * x
            left, right = np.interp(probe, *self._maps[a]), np.interp(probe, *self._maps[b])
            theta = np.exp((1 - fraction) * np.log(left) + fraction * np.log(right))
            values = _features(kin, theta)
            return np.einsum("i,j,ijk->k", width, weights, values)
        coarse, fine = integrate(order), integrate(2 * order)
        error = float(np.max(_relative_difference(coarse, fine)))
        if error > tolerance:
            raise ValueError("Stable pair-moment quadrature did not converge")
        return _hard_moment_result(sigma, float(sigma * fine[0]), float(sigma * fine[1]), error)


def _hard_moment_result(*values):
    from bca.runtime import HardMomentCrossSections
    return HardMomentCrossSections(*values)


def two_body_observables(kinematics, theta_cm):
    """Exact stationary-target kinematics, avoiding 1-cos cancellation."""
    theta = np.asarray(theta_cm, dtype=float)
    if np.any(~np.isfinite(theta)) or np.any((theta < 0) | (theta > math.pi)):
        raise ValueError("CM angles must be finite and in [0, pi]")
    p = kinematics.momentum_cm_ev_c
    e = math.hypot(kinematics.projectile_mass_c2_ev, p)
    longitudinal = kinematics.gamma_cm * (p * np.cos(theta) + kinematics.beta_cm * e)
    angle = np.arctan2(p * np.sin(theta), longitudinal)
    recoil = 2 * p * p / kinematics.target_mass_c2_ev * np.sin(theta / 2) ** 2
    energy = kinematics.projectile_energy_ev
    if np.any(recoil > energy * (1 + 32 * np.finfo(float).eps)):
        raise ValueError("Two-body recoil exceeds projectile energy")
    return angle, np.minimum(recoil, energy)


def _features(kinematics, theta):
    angle, recoil = two_body_observables(kinematics, theta)
    return np.stack((recoil, 2 * np.sin(angle / 2) ** 2), axis=-1)


def _relative_difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    result = np.full(np.broadcast_shapes(a.shape, b.shape), np.inf)
    np.divide(np.abs(a - b), np.abs(b), out=result, where=b != 0)
    return np.where((a == 0) & (b == 0), 0., result)


def _cm_boundaries(kinematics, angle_edges, recoil_edges):
    """Find all CM preimages of joint-bin boundaries, including lab folds."""
    p = kinematics.momentum_cm_ev_c
    boost = kinematics.beta_cm * math.hypot(kinematics.projectile_mass_c2_ev, p)
    turning = math.acos(-p / boost) if boost > p else math.pi
    branches = [(0., turning)]
    if turning < math.pi:
        branches.append((turning, math.pi))
    roots = [0., math.pi]
    for edge in angle_edges[1:-1]:
        for low, high in branches:
            def residual(theta):
                return float(two_body_observables(kinematics, theta)[0]) - edge
            a, b = residual(low), residual(high)
            if a == 0:
                roots.append(low)
            if b == 0:
                roots.append(high)
            if a * b < 0:
                roots.append(brentq(residual, low, high, xtol=2e-14))
    maximum_recoil = 2 * p * p / kinematics.target_mass_c2_ev
    roots.extend(2 * math.asin(math.sqrt(edge / maximum_recoil))
                 for edge in recoil_edges[1:-1] if 0 < edge < maximum_recoil)
    return np.unique(roots)


def _validate_edges(angle_edges, recoil_edges, energy):
    angle_edges, recoil_edges = np.asarray(angle_edges, float), np.asarray(recoil_edges, float)
    for edges, maximum in ((angle_edges, math.pi), (recoil_edges, energy)):
        if edges.ndim != 1 or len(edges) < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError("Joint-bin edges must be finite and strictly increasing")
        if edges[0] != 0 or edges[-1] != maximum:
            raise ValueError("Joint-bin partitions must cover [0, pi] and [0, E]")
    return angle_edges, recoil_edges


def _masses_from_inverse(kinematics, inverse, boundaries, angle_edges, recoil_edges):
    q = np.asarray([inverse(float(theta)) for theta in boundaries])
    if np.any(np.diff(q) > 1e-12) or np.any((q < 0) | (q > 1)):
        raise ValueError("CM inverse must decrease from q=1 to q=0")
    midpoint = (boundaries[:-1] + boundaries[1:]) / 2
    angle, recoil = two_body_observables(kinematics, midpoint)
    ai = np.clip(np.searchsorted(angle_edges, angle, side="right") - 1, 0, len(angle_edges) - 2)
    ri = np.clip(np.searchsorted(recoil_edges, recoil, side="right") - 1, 0, len(recoil_edges) - 2)
    mass = np.zeros((len(angle_edges) - 1, len(recoil_edges) - 1))
    np.add.at(mass, (ai, ri), np.maximum(0., q[:-1] - q[1:]))
    if not np.isclose(mass.sum(), 1., rtol=0, atol=2e-12):
        raise ValueError("Joint pushforward does not conserve collision area")
    return mass


def joint_bin_masses(kinematics, q, theta_cm, angle_edges, recoil_edges):
    """Exact bin areas of a piecewise-linear CM-angle map, not bin centres."""
    q, theta = np.asarray(q, float), np.asarray(theta_cm, float)
    if (q.ndim != 1 or q.shape != theta.shape or len(q) < 2 or q[0] != 0 or q[-1] != 1
            or np.any(np.diff(q) <= 0) or np.any(np.diff(theta) >= 0)
            or np.any(~np.isfinite(theta)) or np.any((theta < 0) | (theta > math.pi))):
        raise ValueError("The pair map must span q=[0,1] with strictly decreasing CM angle")
    ae, re = _validate_edges(angle_edges, recoil_edges, kinematics.projectile_energy_ev)
    boundaries = _cm_boundaries(kinematics, ae, re)
    inverse = lambda x: float(np.interp(x, theta[::-1], q[::-1], left=1., right=0.))
    return _masses_from_inverse(kinematics, inverse, boundaries, ae, re)


def _direct_kernels(table, projectile, target, energy, orders):
    from bca.scattering import NLHCollisionKernel, _turning_radius, _gauss_legendre
    from nlh.potential import get_coefficients, COULOMB_EV_ANGSTROM
    context = NLHCollisionKernel(projectile, target, energy,
              minimum_turning_potential_ev=table.minimum_turning_potential_ev,
              quadrature_order=orders[0])
    coefficients = get_coefficients(projectile, target)
    prefactor = COULOMB_EV_ANGSTROM * coefficients.z1 * coefficients.z2
    relative_energy = context.kinematics.relative_kinetic_energy_ev
    roots = {}

    class StableKernel:
        def __init__(self, order):
            self.angles, self.weights = _gauss_legendre(order)
        def solve(self, impact):
            if impact not in roots:
                roots[impact] = _turning_radius(impact, relative_energy, projectile, target)
            theta, residual, limit = _positive_deflection(impact, roots[impact], relative_energy,
                coefficients.a, coefficients.b_per_angstrom, prefactor, self.angles, self.weights)
            return SimpleNamespace(theta_cm_rad=theta, turning_identity_residual=residual,
                                   turning_identity_residual_limit=limit)

    return [StableKernel(order) for order in orders]


def _positive_deflection(impact, radius, energy, amplitudes, decays, prefactor, angles, weights):
    """The same classical integral, conditioned for very small deflections.

    With A=b/r0, c=cos(alpha), D=[V(r0)-V(r0/c)]/E and the checked turning
    identity 1=A**2+V(r0)/E, F=A**2*sin(alpha)**2+D. Rationalizing 1-J, where
    J=A*sin(alpha)/sqrt(F), gives theta=2 integral D/[sqrt(F)*(sqrt(F)+A*sin)]
    d alpha. Thus neither pi-2*half_orbit nor nearly equal radial terms are
    subtracted. NLH component differences use expm1 and 1-c=2*sin(alpha/2)**2.
    No shape constraint, fitted correction, or change of potential is applied.
    """
    if not all(math.isfinite(v) for v in (impact, radius, energy, prefactor)) or impact < 0 or radius <= 0 or energy <= 0:
        raise ValueError("Invalid positive-integral collision arguments")
    amplitudes, decays = np.asarray(amplitudes, float), np.asarray(decays, float)
    a = impact / radius
    components = prefactor / (radius * energy) * amplitudes * np.exp(-decays * radius)
    residual = float(abs(1 - a * a - components.sum()))
    # A roundoff guard on the normalized algebraic root identity, not a
    # physical convergence tolerance or an asserted rigorous error enclosure.
    limit = float(32 * np.finfo(float).eps * (1 + a * a + np.abs(components).sum()))
    if not math.isfinite(residual) or residual > limit:
        raise ValueError("Turning-point identity is insufficiently resolved for stable quadrature")
    if impact == 0:
        return math.pi, residual, limit
    cosine = np.cos(angles)
    one_minus_cosine = 2 * np.sin(np.asarray(angles) / 2) ** 2
    difference_r = radius * one_minus_cosine / cosine
    difference_v = np.sum(components[:, None] * (one_minus_cosine + cosine *
                         (-np.expm1(-decays[:, None] * difference_r))), axis=0)
    sine = np.sin(angles)
    radial = a * a * sine * sine + difference_v
    if np.any(~np.isfinite(radial)) or np.any(radial <= 0) or np.any(difference_v < 0):
        raise ValueError("Repulsive positive-integral radicand is invalid")
    root = np.sqrt(radial)
    theta = float(2 * np.sum(weights * difference_v / (root * (root + a * sine))))
    if not math.isfinite(theta) or theta < 0 or theta > math.pi * (1 + 32 * np.finfo(float).eps):
        raise ValueError("Stable deflection lies outside [0,pi]")
    return min(math.pi, theta), residual, limit


class _Reference:
    def __init__(self, kernels, kinematics, bmax, tolerance, max_points, progress, saved):
        self.kernels, self.kinematics, self.bmax = kernels, kinematics, bmax
        self.tolerance, self.max_points, self.progress = tolerance, max_points, progress
        self.cache = {}
        self.maximum_quadrature_change = 0.
        self.maximum_root_residual = 0.
        self.maximum_root_residual_ratio = 0.
        self.saved = saved
        self.solve_count = saved.get("kernel_solves_total", 0)
        self.initial_solve_count = self.solve_count
        for key, record in saved["points"].items():
            q = float.fromhex(key)
            theta, change, solves = record["theta_cm_rad"], record["quadrature_relative_change"], record["kernel_solves"]
            if (not math.isfinite(q) or not 0 <= q <= 1 or not math.isfinite(theta) or not 0 <= theta <= math.pi
                    or not math.isfinite(change) or not 0 <= change <= tolerance / 4
                    or type(solves) is not int or not 2 <= solves <= len(kernels)):
                raise ValueError("Invalid completed pair-reference cache entry")
            residual, bound = record["turning_identity_residual"], record["turning_identity_residual_limit"]
            if not math.isfinite(residual) or not math.isfinite(bound) or not 0 <= residual <= bound or bound <= 0:
                raise ValueError("Invalid cached turning-point evidence")
            self.cache[q] = theta
            self.maximum_quadrature_change = max(self.maximum_quadrature_change, change)
            self.maximum_root_residual = max(self.maximum_root_residual, residual)
            self.maximum_root_residual_ratio = max(self.maximum_root_residual_ratio, residual / bound)
        if type(self.solve_count) is not int or self.solve_count < sum(r["kernel_solves"] for r in saved["points"].values()):
            raise ValueError("Pair-reference cache has inconsistent solve accounting")

    def __call__(self, quantile):
        q = float(quantile)
        if q in self.cache:
            return self.cache[q]
        if len(self.cache) >= self.max_points:
            raise _Limit("Direct area-point budget exhausted")
        previous = None
        for used, kernel in enumerate(self.kernels, start=1):
            outcome = kernel.solve(self.bmax * math.sqrt(q))
            current = float(outcome.theta_cm_rad)
            self.solve_count += 1
            self.saved["kernel_solves_total"] = self.solve_count
            if previous is not None:
                change = max(float(np.max(_relative_difference(
                    _features(self.kinematics, previous), _features(self.kinematics, current)))),
                    abs(previous - current) / math.pi)
                if change <= self.tolerance / 4:
                    residual = float(getattr(outcome, "turning_identity_residual", 0.))
                    bound = float(getattr(outcome, "turning_identity_residual_limit", 32 * np.finfo(float).eps))
                    self.cache[q] = current
                    self.saved["points"][q.hex()] = {"theta_cm_rad": current,
                        "quadrature_relative_change": change, "kernel_solves": used,
                        "turning_identity_residual": residual, "turning_identity_residual_limit": bound}
                    self.maximum_quadrature_change = max(self.maximum_quadrature_change, change)
                    self.maximum_root_residual = max(self.maximum_root_residual, residual)
                    self.maximum_root_residual_ratio = max(self.maximum_root_residual_ratio, residual / bound)
                    if self.progress is not None:
                        self.progress(len(self.cache), self.max_points)
                    return current
            previous = current
        raise _Limit(f"Direct quadrature ladder did not converge at area quantile {q:.17g}")


def _area_checks(reference, kinematics, q, theta):
    fractions = np.array([0., .25, .5, .75, 1.])
    probe = q[:-1, None] + np.diff(q)[:, None] * fractions
    direct_theta = np.array([[reference(x) for x in row] for row in probe])
    interpolated = theta[:-1, None] + np.diff(theta)[:, None] * fractions
    values, approximated = _features(kinematics, direct_theta), _features(kinematics, interpolated)
    width = np.diff(q)
    fine_weights = np.array([1., 4., 2., 4., 1.]) / 12
    fine = np.einsum("j,ijk,i->ik", fine_weights, values, width)
    coarse = np.einsum("j,ijk,i->ik", np.array([1., 4., 1.]) / 6, values[:, ::2], width)
    differences = np.einsum("j,ijk,i->ik", fine_weights, np.abs(values - approximated), width)
    total = fine.sum(axis=0)
    if np.any(total <= 0):
        raise _Limit("Positive hard area has a numerically unresolved recoil or transport moment")
    l1 = differences.sum(axis=0) / total
    integration = np.abs(fine - coarse).sum(axis=0) / total
    angular = np.max(np.abs(direct_theta - interpolated), axis=1) / math.pi
    return {"recoil_relative_L1": float(l1[0]), "transport_relative_L1": float(l1[1]),
            "moment_quadrature_relative_change": integration.tolist(),
            "maximum_cm_angle_error_over_pi": float(angular.max())}, total, differences / total, np.abs(fine - coarse) / total, angular


def _bin_edges(kinematics, q, theta, count):
    # Quantile-derived partitions resolve the forward peak without wasting
    # most cells in forbidden angle--recoil combinations. Bounds remain exact.
    cm = np.interp(np.linspace(0., 1., count + 1), q, theta)
    angle, recoil = two_body_observables(kinematics, cm)
    ae = np.unique(np.r_[0., angle, math.pi])
    re = np.unique(np.r_[0., recoil, kinematics.projectile_energy_ev])
    return ae, re


def _compress_roundoff_plateaus(q, theta):
    """Remove indistinguishable knots spanning at most one unit-q ulp.

    Root insertion can place two q values one ulp apart while their computed
    angles are identical. Removing the redundant knot preserves q=[0,1]; no
    probability mass is discarded. Broad plateaus and every increase remain
    numerical failures, and the independent area/bin checks still apply.
    """
    keep = np.ones(len(q), dtype=bool)
    width = 0.
    start = 0
    while start < len(q) - 1:
        end = start
        while end + 1 < len(q) and theta[end + 1] == theta[start]:
            end += 1
        if end > start:
            span = float(q[end] - q[start])
            if span > np.spacing(1.):
                raise _Limit("Direct CM map has a plateau wider than unit-q roundoff")
            width += span
            retained = end if end == len(q) - 1 else start
            keep[start:end + 1] = False
            keep[retained] = True
        start = end + 1
    return q[keep], theta[keep], width


def _reference_bins(reference, kinematics, q, theta, angle_edges, recoil_edges, tv_tolerance):
    boundaries = _cm_boundaries(kinematics, angle_edges, recoil_edges)
    roots = []
    bracket_total = 0.
    theta_low, theta_high = reference(1.), reference(0.)
    q_tolerance = tv_tolerance / (16 * max(1, len(boundaries) - 2))

    def inverse(value):
        nonlocal bracket_total
        if value <= theta_low:
            return 1.
        if value >= theta_high:
            return 0.
        low, high = 0., 1.
        # Retained knots give tight valid brackets; no local branch inversion.
        index = int(np.searchsorted(-theta, -value))
        if 0 < index < len(q):
            low, high = float(q[index - 1]), float(q[index])
        while high - low > q_tolerance:
            mid = (low + high) / 2
            if reference(mid) > value:
                low = mid
            else:
                high = mid
        bracket_total += high - low
        root = (low + high) / 2
        neighbour = int(np.searchsorted(q, root))
        distance = min(abs(root - float(q[i])) for i in
                       (max(0, neighbour - 1), min(len(q) - 1, neighbour)))
        # A root already located within its allocated q uncertainty of a
        # retained knot cannot improve the declared bin accuracy by adding an
        # indistinguishable neighbour. The reference mass and bracket allowance
        # still include it; only redundant map refinement is avoided.
        if distance > q_tolerance:
            roots.append(root)
        return root

    mass = _masses_from_inverse(kinematics, inverse, boundaries, angle_edges, recoil_edges)
    return mass, bracket_total, roots


def build_pair_differential(table, projectile, target, energy_ev, *, relative_tolerance,
                            tv_tolerance, quadrature_orders, max_points, max_refinements,
                            bin_count, progress=None, reference_cache=None):
    """Build one bounded pair-energy product; the caller owns persistence.

    All accuracy/resource settings come from the runner's shared policy.
    `passes` denotes passed numerical refinement tests, not an analytic bound,
    a confidence statement, or qualification of phase-dependent transport.
    `reference_cache` is updated before each progress callback; the caller may
    atomically commit it there and must validate runtime/source provenance on
    resume.  Completed quadrature values and their measured changes are reused.
    """
    for value in (relative_tolerance, tv_tolerance):
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("Numerical tolerances must lie in (0,1)")
    if any(type(n) is not int or n < 1 for n in (max_points, max_refinements, bin_count)):
        raise ValueError("Computational limits must be positive integers")
    if len(quadrature_orders) < 2 or sorted(set(quadrature_orders)) != list(quadrature_orders):
        raise ValueError("At least two strictly increasing quadrature orders are required")
    kin = table.pair_kinematics(projectile, target, energy_ev)
    bmax = float(table.maximum_impact_parameter_angstrom(projectile, target, energy_ev))
    sigma = math.pi * bmax * bmax
    result = {"schema_version": 1, "projectile": projectile, "target": target,
              "energy_ev": float(energy_ev), "maximum_impact_parameter_angstrom": bmax,
              "hard_cross_section_cm2": sigma * 1e-16,
              "minimum_turning_potential_ev": table.minimum_turning_potential_ev,
              "passes": False, "status": "not_qualified",
              "physical_scope": "Independent stationary-target NLH pair; no ice-phase encounter law",
              "statistical_sampling_required": False, "physical_model_validated": False,
              "continuous_density_TV_certified": False,
              "angular_integral": "positive_turning_difference_v1",
              "certificate_kind": "deterministic numerical refinement evidence, not a rigorous error enclosure"}
    result["kinematics"] = {name: float(getattr(kin, name)) for name in
                            ("momentum_cm_ev_c", "projectile_mass_c2_ev", "target_mass_c2_ev",
                             "projectile_energy_ev", "beta_cm", "gamma_cm")}
    if bmax == 0:
        result.update(passes=True, status="analytically_closed", area_quantile=[], theta_cm_rad=[],
                      mean_recoil_energy_ev=0., mean_transport=0., evaluations={"area_points": 0, "kernel_solves": 0})
        return result
    kernels = _direct_kernels(table, projectile, target, energy_ev, quadrature_orders)
    identity = {"projectile": projectile, "target": target, "energy_ev": float(energy_ev),
                "angular_integral": "positive_turning_difference_v1",
                "relative_tolerance": relative_tolerance, "quadrature_orders": list(quadrature_orders),
                "minimum_turning_potential_ev": table.minimum_turning_potential_ev,
                "maximum_impact_parameter_angstrom": bmax,
                "kernel_csv_sha256": getattr(table, "csv_sha256", None)}
    saved = reference_cache if reference_cache is not None else {}
    if not saved:
        saved.update(identity=identity, points={}, kernel_solves_total=0)
    elif saved.get("identity") != identity:
        raise ValueError("Pair-reference cache belongs to a different source or numerical configuration")
    reference = _Reference(kernels, kin, bmax, relative_tolerance, max_points, progress, saved)
    q = np.asarray(table.area_quantile_breakpoints(projectile, target, energy_ev), float)
    # Explicit logarithmic support protects rare large-transfer contributions
    # if an incoming table was generated on an inadequate coarse area grid.
    q = np.unique(np.r_[q, 0., np.geomspace(np.finfo(float).eps, .01, 33), 1.])
    compressed_width = 0.
    try:
        for refinement in range(max_refinements):
            theta = np.array([reference(x) for x in q])
            q, theta, width = _compress_roundoff_plateaus(q, theta)
            compressed_width += width
            if np.any(np.diff(theta) >= 0):
                raise _Limit("Direct CM-angle map is not strictly monotone; numerical reference unresolved")
            checks, moments, l1_cells, quadrature_cells, angular = _area_checks(reference, kin, q, theta)
            result["numerical_checks"] = checks
            result["refinement_rounds"] = refinement + 1
            precision = max(checks["recoil_relative_L1"], checks["transport_relative_L1"],
                            checks["maximum_cm_angle_error_over_pi"]) <= relative_tolerance / 2
            area_passes = max(checks["moment_quadrature_relative_change"]) <= relative_tolerance / 4
            if precision and area_passes:
                ae, re = _bin_edges(kin, q, theta, bin_count)
                mass = joint_bin_masses(kin, q, theta, ae, re)
                direct_mass, root_width, roots = _reference_bins(reference, kin, q, theta, ae, re, tv_tolerance)
                tv = float(np.abs(mass - direct_mass).sum() / 2)
                checks.update(joint_bin_TV_against_direct_preimages=tv,
                              direct_preimage_bracket_mass_bound=root_width,
                              bin_TV_with_root_allowance=tv + root_width)
                if tv + root_width <= tv_tolerance:
                    angle, recoil = two_body_observables(kin, theta)
                    indices = np.argwhere(mass > 0)
                    selected = tuple(indices.T)
                    result.update(passes=True, status="numerically_qualified_pair", area_quantile=q.tolist(),
                                  theta_cm_rad=theta.tolist(), theta_lab_rad=angle.tolist(), recoil_energy_ev=recoil.tolist(),
                                  projectile_out_energy_ev=(energy_ev - recoil).tolist(),
                                  mean_recoil_energy_ev=float(moments[0]), mean_transport=float(moments[1]),
                                  recoil_moment_cross_section_ev_cm2=float(moments[0] * sigma * 1e-16),
                                  transport_cross_section_cm2=float(moments[1] * sigma * 1e-16),
                                  sampling_contract="Draw uniform q; interpolate only theta_CM in q; derive correlated lab angle/recoil by exact two-body kinematics; azimuth uniform only for isolated pair",
                                  joint_bins={"angle_edges_rad": ae.tolist(), "recoil_edges_ev": re.tolist(),
                                              "indices_angle_recoil": indices.tolist(), "probability_mass": mass[selected].tolist(),
                                              "integrated_cross_section_cm2": (sigma * 1e-16 * mass[selected]).tolist(),
                                              "scope": "Finite-bin integrated DCS; not an independent pair of marginals or a regular 2-D density"})
                    break
                q = np.unique(np.r_[q, roots])
            else:
                n = len(q) - 1
                selected = ((l1_cells > relative_tolerance / (4 * n)).any(axis=1)
                            | (quadrature_cells > relative_tolerance / (8 * n)).any(axis=1)
                            | (angular > relative_tolerance / 2))
                if not np.any(selected):
                    raise _Limit("No resolvable refinement direction for failed numerical checks")
                q = np.unique(np.r_[q, (q[:-1][selected] + q[1:][selected]) / 2])
        else:
            raise _Limit("Pair area-refinement limit exhausted")
    except _Limit as error:
        result["reason"] = str(error)
    result["evaluations"] = {"area_points": len(reference.cache), "kernel_solves": reference.solve_count,
                             "new_kernel_solves": reference.solve_count - reference.initial_solve_count}
    result["maximum_direct_quadrature_relative_change"] = reference.maximum_quadrature_change
    result["maximum_turning_identity_residual"] = reference.maximum_root_residual
    result["maximum_turning_identity_residual_fraction_of_roundoff_guard"] = reference.maximum_root_residual_ratio
    result["roundoff_plateau_q_width"] = compressed_width
    return result


def compare_pair_energy(left, middle, right, *, relative_tolerance, tv_tolerance):
    """Check one independently produced interior energy, with no new solves.

    Endpoint theta_CM is interpolated log-linearly in energy at each q, as
    in the retained runtime.  This checks that specified energy only; it does
    not assert an enclosure throughout the surrounding energy interval.
    """
    for item in (left, middle, right):
        if not item.get("passes") or item.get("status") != "numerically_qualified_pair":
            return {"passes": False, "reason": "Interior energy check requires three qualified nonzero pair maps"}
    if len({(item["projectile"], item["target"], item["minimum_turning_potential_ev"])
            for item in (left, middle, right)}) != 1:
        raise ValueError("Energy interpolation must use the same pair and hard boundary")
    energies = [item["energy_ev"] for item in (left, middle, right)]
    if not 0 < energies[0] < energies[1] < energies[2]:
        raise ValueError("Energy check needs ordered positive energies")
    weight = math.log(energies[1] / energies[0]) / math.log(energies[2] / energies[0])
    kin = SimpleNamespace(**middle["kinematics"])
    q = np.unique(np.concatenate([item["area_quantile"] for item in (left, middle, right)]))

    def theta_at(item, quantile):
        return np.interp(quantile, item["area_quantile"], item["theta_cm_rad"])

    def interpolated(quantile):
        a, b = theta_at(left, quantile), theta_at(right, quantile)
        if np.any(a <= 0) or np.any(b <= 0):
            raise ValueError("Log-energy interpolation requires positive CM angles")
        return np.exp((1 - weight) * np.log(a) + weight * np.log(b))

    estimates = []
    for order in (4, 8):
        x, weights = np.polynomial.legendre.leggauss(order)
        width = np.diff(q) / 2
        probe = (q[:-1] + q[1:])[:, None] / 2 + width[:, None] * x
        interpolated_theta, direct_theta = interpolated(probe), theta_at(middle, probe)
        direct = _features(kin, direct_theta)
        candidate = _features(kin, interpolated_theta)
        integrate = lambda values: np.einsum("i,j,ijk->k", width, weights, values)
        reference = integrate(direct)
        error = integrate(np.abs(candidate - direct)) / reference
        estimates.append((reference, error))
    integration_change = float(max(np.max(_relative_difference(estimates[0][0], estimates[1][0])),
                                   np.max(np.abs(estimates[0][1] - estimates[1][1]))))
    ae = np.asarray(middle["joint_bins"]["angle_edges_rad"])
    re = np.asarray(middle["joint_bins"]["recoil_edges_ev"])
    boundaries = _cm_boundaries(kin, ae, re)
    a, b = float(interpolated(0.)), float(interpolated(1.))

    def inverse(theta):
        if theta <= b:
            return 1.
        if theta >= a:
            return 0.
        return brentq(lambda value: float(interpolated(value)) - theta, 0., 1., xtol=2e-14)

    candidate_mass = _masses_from_inverse(kin, inverse, boundaries, ae, re)
    reference_mass = joint_bin_masses(kin, middle["area_quantile"], middle["theta_cm_rad"], ae, re)
    tv = float(np.abs(candidate_mass - reference_mass).sum() / 2)
    error = estimates[-1][1]
    angle_error = float(np.max(np.abs(interpolated(probe) - theta_at(middle, probe))) / math.pi)
    return {"passes": bool(max(error) <= relative_tolerance and angle_error <= relative_tolerance
                            and integration_change <= relative_tolerance / 4 and tv <= tv_tolerance),
            "energy_ev": energies[1], "energy_interval_ev": [energies[0], energies[2]],
            "recoil_relative_L1": float(error[0]), "transport_relative_L1": float(error[1]),
            "maximum_cm_angle_error_over_pi": angle_error, "joint_bin_TV": tv,
            "area_integration_change": integration_change,
            "scope": "Numerical interpolation evidence at this tested energy only; no physical or continuous-energy error bound"}
