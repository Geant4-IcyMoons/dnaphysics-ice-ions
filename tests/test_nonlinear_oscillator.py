"""Numerical oscillator checks, independent of the synthetic assembly fixture."""
import numpy as np
import pytest
from scipy.integrate import DOP853, solve_ivp
from scipy.special import k0, k1

from physics.inelastic_dielectric.polarization import nonlinear_oscillator as no
from physics.inelastic_dielectric.polarization import nonlinear_polarization as npol
from physics.inelastic_dielectric.polarization import correction as pc
from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import ELEMENTS, load_density


def test_scaled_dop853_error_norm_avoids_neutral_tail_underflow():
    stages = np.random.default_rng(731).normal(size=(len(DOP853.E5), 8))
    scale = np.ones(8)
    original, stable = object.__new__(DOP853), object.__new__(no._StableDOP853)
    reference = original._estimate_error_norm(stages, .1, scale)
    assert stable._estimate_error_norm(stages, .1, scale) == pytest.approx(reference, rel=1e-14)
    for factor in (1e-160, 1e-200, 1e-260):
        actual = stable._estimate_error_norm(stages*factor, .1, scale)
        assert actual == pytest.approx(reference*factor, rel=1e-13, abs=0)
    assert stable._estimate_error_norm(stages*0, .1, scale) == 0


@pytest.mark.parametrize("gamma", [1., 1.2])
def test_rotating_amplitudes_against_displacement_equations(gamma):
    x, b, eta, tail = .3, .5, .1, 32.
    field = no.frozen_field("He", 0)
    end = np.arcsinh(tail*max(4., 1/x))

    def rhs(s, state):
        tau = np.sinh(s)
        y, p = state[:2], state[2:]
        r = np.array([1., tau])-eta*y
        h = np.linalg.norm(r)
        return np.cosh(s)*np.r_[p, -x*x*y+field.scalar(b*h)*r/h**3]

    reference = solve_ivp(rhs, [-end, end], np.zeros(4), method="DOP853",
                          rtol=2e-11, atol=2e-13, max_step=.05)
    assert reference.success
    y, p = reference.y[:2, -1], reference.y[2:, -1]
    expected = np.dot([1., gamma**-2], p*p+x*x*y*y)/2
    actual = no.encounter(x, b, eta, field, gamma=gamma, tail=tail)
    assert actual["full"] == pytest.approx(expected, rel=2e-8)
    assert actual["full"]-actual["leading"] == pytest.approx(actual["difference"], abs=1e-14)


@pytest.mark.parametrize("gamma", [1., 1.2])
def test_point_linear_limit_is_fourier_bessel(gamma):
    x, z = .3, 2.
    expected = 2*z*z*x*x*(k1(x)**2+k0(x)**2/gamma**2)
    assert no.linear_energy_quadrature(x, .5, no.PointField(z), gamma=gamma) == pytest.approx(expected, rel=1e-10)
    result = no.encounter(x, .5, 0., no.PointField(z), gamma=gamma, tail=128)
    assert result["leading"] == pytest.approx(expected, rel=1e-5)
    assert result["difference"] == 0.


@pytest.mark.parametrize("element,charge", [(e,q) for e,z in ELEMENTS.items() for q in range(z+1)])
def test_all_states_use_normalized_frozen_fields(element, charge):
    field = no.frozen_field(element, charge)
    density = load_density(element, charge)
    assert field.electron_integral == pytest.approx(density.electrons, abs=2e-9)
    radii = np.geomspace(1e-4, 20., 100)
    expected = density.radial_charge_direct(radii)[0]
    np.testing.assert_allclose(field(radii), expected, rtol=1e-6, atol=2e-8)
    assert npol.metadata(density)["polarization_response"] == "full_minus_leading"
    assert not npol.metadata(density)["polarization_cubic_backend_available"]
    result = no.encounter(.3, 1., .01, field, tail=16, rtol=1e-8)
    assert result["full"] >= 0 and np.isfinite(result["difference"])


def test_impact_jacobian_and_velocity_reconstruction(monkeypatch):
    xi, scale, gamma = .01, 3., 1.2
    velocity = .5616/(xi*scale)

    def response(x, b, eta, field, **kwargs):
        assert b == pytest.approx(x*scale)
        assert eta == pytest.approx(1/(gamma*b*velocity**2))
        return dict(difference=2*eta*x)

    monkeypatch.setattr(no, "encounter", response)
    value, error, report = no.integrate_kernel(xi, scale, gamma, no.PointField(1.))
    assert value == pytest.approx(np.log(no.MAX_X/xi), rel=1e-12)
    assert report["converged"] and error < 1e-9


def test_oos_energy_area_conversion_matches_prefactor():
    beta, gamma = pc.projectile_beta_gamma(1e6, 1836.152673)
    v = beta/pc.ALPHA_FINE
    hartree = pc.ALPHA_FINE**2*pc.MEC2_EV
    # a0 = r_e/alpha^2, derived in the same Gaussian-unit convention.
    a0_cm = pc.RE_CLASSICAL_CM/pc.ALPHA_FINE**2
    w, x, difference = 15., .1, .002
    b = x*gamma*v/(w/hartree)
    eta = 1/(gamma*b*v*v)
    direct = 2*np.pi*a0_cm**2*b*b*hartree*gamma**2*v*v*eta*eta*difference/w
    pref = 4*np.pi*pc.RE_CLASSICAL_CM**2*pc.ALPHA_FINE/(gamma**2*beta**5)
    assert direct == pytest.approx(pref*difference/(2*eta*x), rel=1e-14)


def test_cm2_to_m2_once_and_exact_cutoff(synthetic_nonlinear_kernel):
    t, mass, charge = 1e6, 1836.152673, 2.5
    w = np.array([15., 2*pc.wmax_eV(t, mass)])
    oos = np.array([.2, .2])
    actual, error = npol.dcs_m2_per_eV(t,w,mass,oos,z_int=charge,workers=1)
    beta,gamma = pc.projectile_beta_gamma(t,mass)
    xi = .5616*w[0]/(gamma*beta**2*pc.MEC2_EV)
    expected = 1e-4*4*np.pi*pc.RE_CLASSICAL_CM**2*pc.ALPHA_FINE/(gamma**2*beta**5)*oos[0]*charge/(1+xi)
    assert actual[0] == pytest.approx(expected, rel=1e-14, abs=0)
    assert actual[1] == 0 and np.all(error == 0)
    assert synthetic_nonlinear_kernel[0][3] == charge


def test_zero_point_charge_has_no_polarization():
    result, error = npol.dcs_m2_per_eV(1e6,[15.],1836.152673,[1.],z_int=0.,workers=1)
    np.testing.assert_array_equal(result, [0.])
    np.testing.assert_array_equal(error, [0.])


def test_negative_correction_not_clipped_and_negative_total_flagged(monkeypatch):
    monkeypatch.setattr(npol,"dcs_m2_per_eV",lambda *a,**k: (np.array([-.5,-3.]),np.zeros(2)))
    data = dict(T_line=np.full(2,1e6),E_line=np.array([15.,20.]),
                exc_vals=np.ones((2,1)),ion_vals=np.ones((2,1)))
    _, diag = pc.apply_barkas_correction_to_dcs_data(data,pc.model.epsilon_optical("amorphous"),
        "amorphous",1836.152673,1.,include_barkas_dcs=True,born_reference_charge="bare_Z",workers=1)
    np.testing.assert_array_equal(diag.DCS_total_m2_per_eV,[1.5,-1.])
    np.testing.assert_array_equal(diag.negative_or_unstable_T_eV,[1e6])
