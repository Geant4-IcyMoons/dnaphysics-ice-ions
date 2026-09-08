from __future__ import annotations

import sys

import numpy as np
import pytest


from physics.inelastic_dielectric.polarization import correction
from physics.inelastic_dielectric.finite_q import emfietzoglou_model_finite_q as model
from physics.inelastic_dielectric import generate_cross_sections as ion_generator


def test_barkas_constants():
    assert np.isclose(correction.RE_CLASSICAL_CM, 2.8179403205e-13)
    assert np.isclose(correction.ALPHA_FINE, 7.2973525643e-3)
    assert np.isclose(correction.MEC2_EV, 510998.95069)
    assert correction.H2O_Z_MOL == 10.0
    assert np.isclose(correction.H2O_CB, 1.0)
    assert np.isclose(correction.mass_u_to_electron_mass(1.0), 1822.888486217313)


def test_barkas_zeff_formula():
    T = np.array([1.0e6, 1.0e8])
    Z = 8.0
    mass_au = 29156.9469
    beta, _ = correction.projectile_beta_gamma(T, mass_au)
    expected = Z * (1.0 - np.exp(-125.0 * beta * Z ** (-2.0 / 3.0)))
    assert np.allclose(correction.barkas_effective_charge(T, Z, mass_au), expected)


def test_projectile_mass_convention_uses_electron_mass_units():
    proton_mass_me = 1836.152673
    beta, gamma = correction.projectile_beta_gamma(1.0e6, proton_mass_me)
    expected_gamma = 1.0 + 1.0e6 / (proton_mass_me * correction.MEC2_EV)
    assert np.isclose(gamma, expected_gamma)
    assert np.isclose(beta, np.sqrt(1.0 - 1.0 / expected_gamma**2))
    assert np.isclose(correction.mass_number_to_electron_mass(16.0), 16.0 * correction.U_TO_ELECTRON_MASS)
    assert np.isclose(correction.mass_number_to_electron_mass(32.0), 32.0 * correction.U_TO_ELECTRON_MASS)


def test_wmax_for_proton_and_oxygen_uses_mass_ratio():
    proton_mass_me = 1836.152673
    oxygen_mass_me = 29156.9469
    for mass_me in (proton_mass_me, oxygen_mass_me):
        beta, gamma = correction.projectile_beta_gamma(1.0e6, mass_me)
        expected = (
            2.0
            * beta**2
            * gamma**2
            * correction.MEC2_EV
            / (1.0 + 2.0 * gamma / mass_me + 1.0 / mass_me**2)
        )
        assert np.isclose(correction.wmax_eV(1.0e6, mass_me), expected)


def test_total_and_per_u_conversion_equivalence():
    mass_number = 16.0
    mass_me = 29156.9469
    T_per_u = 1.0e6
    T_total = mass_number * T_per_u
    beta_total, gamma_total = correction.projectile_beta_gamma(T_total, mass_me)
    beta_converted, gamma_converted = correction.projectile_beta_gamma(mass_number * T_per_u, mass_me)
    assert np.isclose(beta_total, beta_converted)
    assert np.isclose(gamma_total, gamma_converted)


def test_oos_normalization_per_h2o():
    s = model.epsilon_optical("amorphous")
    W = np.geomspace(1.0e-3, 1.0e6, 1000)
    oos = correction.oos_density(W, s, material="amorphous", include_kshell=True)
    assert np.isclose(oos.valence_integral_raw * oos.valence_norm, 8.0, rtol=1e-12)
    assert np.isclose(oos.ok_integral_raw * oos.ok_norm, 2.0, rtol=1e-12)
    assert np.isclose(oos.total_integral_norm_grid, 10.0, rtol=1e-5)
    assert np.isfinite(oos.total_integral_unique_grid)
    assert oos.df_dW_total_unique.size == np.unique(W).size


def test_oos_normalization_is_separate_from_finite_q_kshell_fsum():
    s = model.epsilon_optical("amorphous")
    W = np.geomspace(1.0e-6, 1.0e8, 5000)
    oos = correction.oos_density(W, s, material="amorphous", include_kshell=True)
    assert np.isclose(oos.valence_integral_raw * oos.valence_norm, 8.0, rtol=1e-12)
    assert np.isclose(oos.ok_integral_raw * oos.ok_norm, 2.0, rtol=1e-12)
    assert np.isclose(oos.total_integral_norm_grid, 10.0, rtol=1e-12)

    continuum_strength = model.oxygen_K_hydrogenic_gos_continuum_strength(0.0)
    assert np.isclose(continuum_strength, 1.736914215348305, rtol=3e-6)
    assert not np.isclose(continuum_strength, 2.0)


def test_born_reference_charge_scaling_modes():
    z_int = np.array([2.0, 4.0])
    scale, q_ref = correction.born_reference_scale(z_int, 8.0, "bare_Z")
    assert q_ref == 8.0
    assert np.allclose(scale, (z_int / 8.0) ** 2)
    scale, q_ref = correction.born_reference_scale(z_int, 8.0, "unit_charge")
    assert q_ref == 1.0
    assert np.allclose(scale, z_int**2)
    scale, q_ref = correction.born_reference_scale(z_int, 8.0, "explicit_q", q_reference=2.0)
    assert q_ref == 2.0
    assert np.allclose(scale, (z_int / 2.0) ** 2)
    try:
        correction.born_reference_scale(z_int, 8.0, None)
    except ValueError:
        pass
    else:
        raise AssertionError("missing born_reference_charge should fail")


@pytest.mark.parametrize("mode", ["bare", "zeff", "explicit"])
def test_same_point_charge_enters_born_and_nonlinear(mode, synthetic_nonlinear_kernel):
    t, w = np.repeat([1e6, 1e7], 2), np.tile([15., 50.], 2)
    data = dict(T_line=t, E_line=w, exc_vals=np.ones((4,1)), ion_vals=np.ones((4,1)))
    _, diag = correction.apply_barkas_correction_to_dcs_data(data,
        model.epsilon_optical("amorphous"), "amorphous", 1836.152673, 8.,
        charge_mode=mode, explicit_charge=2.5, include_barkas_dcs=True,
        born_reference_charge="unit_charge", workers=1)
    np.testing.assert_allclose(diag.DCS_Born_m2_per_eV, 2*diag.z_int**2, rtol=1e-14, atol=0)
    np.testing.assert_array_equal([row[3] for row in synthetic_nonlinear_kernel], diag.z_int)


def test_barkas_dcs_total_integrates_to_tcs_total(synthetic_nonlinear_kernel):
    s = model.epsilon_optical("amorphous")
    T = 1.0e6
    W = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    dcs_data = {
        "T_line": np.full(W.size, T),
        "E_line": W,
        "exc_vals": np.full((W.size, 1), 1.0e-6),
        "ion_vals": np.full((W.size, 1), 2.0e-6),
    }
    corrected, diag = correction.apply_barkas_correction_to_dcs_data(
        dcs_data,
        s,
        material="amorphous",
        projectile_mass_me=1836.152673,
        nuclear_charge=1.0,
        charge_mode="bare",
        include_barkas_dcs=True,
        include_kshell=True,
        dcs_scale_m2=1.0,
        born_reference_charge="bare_Z",
    )
    total_file = np.sum(corrected["exc_vals"], axis=1) + np.sum(corrected["ion_vals"], axis=1)
    expected = np.trapezoid(total_file, W)
    assert np.allclose(diag.TCS_T_eV, np.array([T]))
    assert np.isclose(diag.TCS_total_m2[0], expected)
    assert np.isclose(diag.S_Barkas_check_eV_m2[0], np.trapezoid(W * diag.DCS_Barkas_m2_per_eV, W))
    assert diag.negative_or_unstable_T_eV.size == 0
    assert np.all(np.isfinite(diag.DCS_total_m2_per_eV))
    assert np.all(diag.DCS_total_m2_per_eV >= 0.0)
    assert diag.barkas_channel_distribution == "bookkeeping_weighted_across_existing_channels"


def test_bloch_dcs_cannot_be_enabled_silently():
    old_argv = sys.argv[:]
    try:
        sys.argv = ["generate_ice_cross_sections_ion.py", "--include-bloch-dcs"]
        with pytest.raises(ValueError, match="Bloch corrections"):
            ion_generator._include_bloch_dcs_from_argv(default=False)
        sys.argv = ["generate_ice_cross_sections_ion.py", "--include-bloch-dcs=true"]
        with pytest.raises(ValueError, match="Bloch corrections"):
            ion_generator._include_bloch_dcs_from_argv(default=False)
    finally:
        sys.argv = old_argv


def test_energy_unit_per_u_converts_to_total_before_kinematics():
    previous = ion_generator.PROJECTILE_KEY
    try:
        ion_generator.set_projectile("oxygen")
        assert ion_generator.PROJECTILE_MASS_NUMBER == 16.0
        assert ion_generator._energy_range_to_total_eV(1.0e6, 2.0e6, "total") == (1.0e6, 2.0e6)
        assert ion_generator._energy_range_to_total_eV(1.0e6, 2.0e6, "per_u") == (16.0e6, 32.0e6)
    finally:
        ion_generator.set_projectile(previous)
