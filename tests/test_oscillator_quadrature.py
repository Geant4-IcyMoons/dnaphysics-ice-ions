"""Integration accuracy, separate error checks, and unchanged physics guards."""

from types import SimpleNamespace

import numpy as np
import pytest

from physics.inelastic_dielectric.polarization import barkas_dcs as bd
from physics.inelastic_dielectric.polarization import oscillator_quadrature as oq
from physics.inelastic_dielectric.polarization import screened_barkas as sb
from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import load_density


def test_panel_antiderivatives_integrate_polynomials_exactly():
    x, w, left, right = oq._gauss_antiderivative(12)
    for k in range(12):
        np.testing.assert_allclose(left @ x**k, (x**(k+1)-(-1)**(k+1))/(k+1), atol=2e-14)
        np.testing.assert_allclose(right @ x**k, (1-x**(k+1))/(k+1), atol=2e-14)
    np.testing.assert_allclose(left+right, np.broadcast_to(w, left.shape), atol=2e-16)


@pytest.mark.parametrize("element,charge", [("H", 0), ("He", 0), ("He", 1), ("C", 2), ("S", 0)])
@pytest.mark.parametrize("x,b", [(1e-4, .05), (.1, .5), (1., 1.), (10., 2.)])
def test_phase_panel_refinement(element, charge, x, b):
    d = load_density(element, charge)
    reference = oq.impulse_products(x, b, d, order=20, extent=1024., log_step=.25, phase_step=np.pi/2)
    actual = oq.impulse_products(x, b, d, order=16, extent=512.)
    np.testing.assert_allclose(actual, reference, rtol=5e-4, atol=1e-8*d.z**3)


@pytest.mark.parametrize("beta", [.01, .4])
@pytest.mark.parametrize("xi", [.001, .1, 1.])
def test_adaptive_point_kernel_matches_salvat_without_analytic_dispatch(beta, xi):
    gamma = 1/np.sqrt(1-beta*beta)
    value, error, details = oq.integrate_kernel(xi, .5, gamma, load_density("H", 1), rtol=sb.QUADRATURE_RTOL)
    assert value == pytest.approx(bd.arbi1(xi)+bd.arbi2(xi)/gamma**2, rel=.001)
    assert error <= sb.QUADRATURE_RTOL*abs(value)+1e-10
    assert error == pytest.approx(sum(details[k] for k in ("impact_error", "time_error", "tail_error")))


def test_phase_panel_cubic_matches_independent_nonlinear_ode():
    from test_screened_oscillator_nonlinear import check
    d = load_density("He", 0)
    b = np.sqrt(d.moment(2)/d.electrons)
    expected = check.extracted_cubic(.3, b, check.independent_field("He", 0), tail=128., rtol=2e-12)
    actual = np.sum(oq.impulse_products(.3, b, d, order=16, extent=128.))
    assert actual == pytest.approx(expected["value"], rel=1e-4)


def test_broadcast_impulses_and_separate_bare_components():
    d = load_density("H", 1)
    x = np.array([.001, .1, 1.])
    actual = oq.impulse_products(x, .5, d)
    np.testing.assert_array_equal(actual, np.array([oq.impulse_products(v, .5, d) for v in x]).T)
    for xi in (.001, .1):
        components = [oq.integrate_kernel(xi, 1., 1., d, component=j)[0] for j in (0, 1)]
        np.testing.assert_allclose(components, [bd.arbi1(xi), bd.arbi2(xi)], rtol=.001)


def test_legacy_quadrature_is_absent():
    assert not hasattr(sb, "oscillator_kernel")
    assert not hasattr(sb, "_impulse_products")


@pytest.mark.parametrize("xi", [.001, .1, 1.3])
def test_point_benchmark_uses_numerical_components_and_cubic_scaling(xi):
    from physics.inelastic_dielectric.polarization.benchmarking import compare_screened_barkas_point_projectiles as benchmark
    proton, error = benchmark._components(xi, "proton")
    alpha, alpha_error = benchmark._components(xi, "alpha")
    np.testing.assert_allclose(proton, [bd.arbi1(xi), bd.arbi2(xi)], rtol=.001)
    np.testing.assert_allclose(alpha, 8*proton, rtol=1e-10)
    assert np.all(error <= oq.RELATIVE_TOLERANCE*np.abs(proton)+1e-10)
    assert np.all(alpha_error <= oq.RELATIVE_TOLERANCE*np.abs(alpha)+8e-10)


@pytest.mark.parametrize("component", ["impact", "time", "tail"])
def test_separate_error_sources_cannot_be_hidden(monkeypatch, component):
    def unresolved(*args, **kwargs):
        errors = np.array([1., float(component == "time"), float(component == "tail")])
        return errors, float(component == "impact"), SimpleNamespace(neval=1, success=True)
    monkeypatch.setattr(oq, "quad_vec", unresolved)
    with pytest.raises(RuntimeError, match="did not converge"):
        oq.integrate_kernel(.1, 1., 1., load_density("H", 0))


def test_nonfinite_integrand_is_not_accepted(monkeypatch):
    monkeypatch.setattr(oq, "quad_vec", lambda *a, **k: (
        np.array([np.nan, 0., 0.]), 0., SimpleNamespace(neval=1, success=True)))
    with pytest.raises(RuntimeError, match="did not converge"):
        oq.integrate_kernel(.1, 1., 1., load_density("H", 0))


def test_cutoff_and_new_cache_provenance():
    value, error, details = oq.integrate_kernel(50., 1., 1., load_density("He", 0))
    assert value == error == details["evaluations"] == 0
    assert sb.metadata(load_density("He", 0))["barkas_quadrature_method"] == oq.VERSION
    assert sb.QUADRATURE_RTOL == 1e-4


@pytest.mark.parametrize("w", [[0.], [-1.], [np.nan], [np.inf]])
def test_invalid_losses_fail(w):
    with pytest.raises(ValueError):
        sb.converged_kernel(w, .1, 1/np.sqrt(.99), load_density("He", 0))


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
@pytest.mark.parametrize("relativistic", [False, True])
def test_benchmark_response_reuse_is_exact_and_restores_functions(phase, relativistic):
    from physics.inelastic_dielectric.polarization.benchmarking import check_integration_failures as check
    g = check.gen
    g.set_projectile("He")
    g._set_projectile_charge_state(0)
    g._set_projectile_relativistic_dcs(relativistic)
    s, c = g.model.epsilon_optical(phase), g.model.default_dispersion_coefficients()
    loss, energy, nq = 15., 4e7, 16
    functions = (g.model.epsilon1_valence_Eq, g.model.epsilon2_valence_Eq)
    try:
        expected = (sum(g._selected_dsigma_excitation(loss, energy, i, s, c, nq)
                        for i in range(len(s.excitations)))
                    + sum(g._selected_dsigma_ionization(loss, energy, i, s, c, nq)
                          for i in range(len(s.ionizations)))
                    + g._selected_dsigma_kshell(loss, energy, s, c, nq))
        assert check._born_total([loss], energy, s, c, nq)[0] == expected
        assert functions == (g.model.epsilon1_valence_Eq, g.model.epsilon2_valence_Eq)
    finally:
        g.set_projectile("proton")
        g._set_projectile_relativistic_dcs(False)
