"""Analytic limits and numerical matching, not validation of ice DCS."""

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.integrate import quad

from physics.inelastic_dielectric.polarization import close_collisions as cc
from physics.inelastic_dielectric.polarization import barkas_dcs as bd
from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import ELEMENTS, load_density


@dataclass
class Yukawa:
    z: int
    charge: float
    radius: float

    @property
    def electrons(self):
        return self.z-self.charge

    def electron_deficit(self, k, *, direct=False):
        x = (np.asarray(k)*self.radius)**2
        return self.electrons*x/(1+x)


@pytest.mark.parametrize("q", [0, 1, 3])
@pytest.mark.parametrize("kappa", [0.001, 0.1, 1, 10])
def test_induced_potential_yukawa_exact(q, kappa):
    density = Yukawa(3, q, 0.4)
    exact = kappa*(q*np.pi/2 + (3-q)*np.arctan(kappa*density.radius))
    assert cc.induced_shift(kappa, density) == pytest.approx(exact, rel=2e-7)


@pytest.mark.parametrize("element", ELEMENTS)
def test_all_state_static_and_dynamic_potential_limits(element):
    for charge in range(ELEMENTS[element]+1):
        density = load_density(element, charge)
        static = quad(lambda r: 4*np.pi*r*float(density.density(r)), 0, np.inf,
                      epsabs=1e-9, epsrel=1e-10)[0]
        assert cc.static_core_offset(density) == pytest.approx(static, abs=1e-8)
        assert cc.induced_shift(0, density) == 0
        for kappa in (0.01, 1):
            shift = cc.induced_shift(kappa, density)
            assert density.charge*np.pi*kappa/2 - 1e-10 <= shift <= density.z*np.pi*kappa/2 + 1e-10
            assert shift == pytest.approx(cc.induced_shift(kappa, density, order=384), rel=2e-6)


def test_eq24_subtraction_and_integrated_close_moment():
    v, z, c_static, shift, match = 10., 2., 2., .1, .03
    b = np.linspace(0, match, 21)
    def total(offset):
        return 2*v*v*(1-(1-2*offset/v**2)*(v*v*b/z)**2)
    np.testing.assert_allclose(total(c_static+shift)-total(c_static),
                               cc.close_transfer(b,v,z,shift), atol=1e-13)
    integral = quad(lambda p: 2*np.pi*p*cc.close_transfer(p,v,z,shift),0,match)[0]
    assert integral == pytest.approx(2*np.pi*shift*v**4*match**4/z**2, rel=1e-12)


@pytest.mark.parametrize("element,charge", [("H",1),("He",0),("He",1),("C",3)])
def test_matching_and_cutoff_convergence(element, charge):
    density = load_density(element,charge)
    row = cc.compare_kernels(30,10,density)
    assert row["crossovers_found"] == 1
    assert row["matching_relative_mismatch"] < 1e-5
    assert row["K_matched"] == pytest.approx(row["K_close"]+row["K_distant"])
    assert row["numerical_convergence"]
    assert not row["physical_accuracy_validated"]
    if element == "H":
        xi = .5616*(30/cc.HARTREE_EV)/100
        assert row["K_cutoff"] == pytest.approx(bd.arbi1(xi)+bd.arbi2(xi), rel=.001)


def test_spectral_units_and_stopping_moment():
    w = np.linspace(7,100,1001)
    df = np.ones_like(w)/93
    k = np.ones_like(w)*2
    dcs = cc.equivalent_dcs(k,10,df)
    assert np.all(dcs > 0)
    expected = 4*np.pi*2/10**5*cc.BOHR_CM**2*bd.CM2_TO_M2*np.trapezoid(w*df,w)
    assert np.trapezoid(w*dcs,w) == pytest.approx(expected, rel=1e-12)


def test_invalid_inputs_rejected():
    density = load_density("H",1)
    with pytest.raises(ValueError):
        cc.compare_kernels(30,1,density)
    with pytest.raises(ValueError):
        cc.induced_shift(-1,density)


def test_interpolated_matched_spectrum_remains_additive():
    from physics.inelastic_dielectric.polarization.benchmarking import compare_close_collisions as bench

    source_w = np.geomspace(7,1000,9)
    nodes = [dict(W_eV=w,K_close=np.sqrt(w),K_distant=1000/w,
                  shift_over_incident_electron_energy=.5,match_over_b90=2.) for w in source_w]
    w = np.geomspace(7,1000,501)
    oos = np.ones_like(w)/993
    close = bench._moment(nodes,10,w,oos,"K_close")
    distant = bench._moment(nodes,10,w,oos,"K_distant")
    assert bench._moment(nodes,10,w,oos,"K_matched") == pytest.approx(close+distant,rel=1e-14)
    fractions = bench._validity_fractions(nodes,10,w,oos)
    assert fractions["matched_moment_fraction_with_nonpositive_close_energy"] == 0
    assert fractions["matched_moment_fraction_with_match_beyond_b90"] == pytest.approx(1)
