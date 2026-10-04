"""Common target spectrum, distinct dynamical models; no fit to stopping data."""
import numpy as np
import pytest

from physics.inelastic_dielectric import generate_cross_sections as gen
from physics.inelastic_dielectric.finite_q import optical_input as optical
from physics.inelastic_dielectric.polarization import correction


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
def test_oscillator_equals_independent_born_optical_limit(phase):
    s = gen.model.epsilon_optical(phase)
    w = np.unique(np.r_[np.geomspace(7., 1e6, 2001), 10., 13., 17., 32.])
    q = np.array([0.])
    c = gen.model.DispersionCoeffs(0., 0., 0.)
    strength = gen.model.oxygen_K_hydrogenic_gos_continuum_strength(q)
    e1 = gen.model.epsilon1_valence_Eq(w, q, s, c, inner_shell_strength_electrons=strength)["total"][0]
    e2 = gen.model.epsilon2_valence_Eq(w, q, s, c, partitioned=True,
                                     inner_shell_strength_electrons=strength)["total"][0]
    factor = w*gen._ion_elf_per_molecule_factor(s)/gen.OPTICAL_SUM_UNIT_EV2_M3
    valence = factor*e2/(e1*e1+e2*e2)*gen._elf_rolloff_factor(w)
    core = factor*gen.model.oxygen_K_ion_hydrogenic_gos_elf(w, 0., Ep_eV=s.Ep)
    actual = correction.oos_density(w, s, phase)
    np.testing.assert_allclose(actual.df_dW_valence, valence, rtol=2e-14, atol=1e-18)
    np.testing.assert_allclose(actual.df_dW_OK, core, rtol=2e-14, atol=0)
    assert actual.valence_norm == actual.ok_norm


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
def test_query_grid_does_not_change_target(phase):
    s = gen.model.epsilon_optical(phase)
    w = np.array([40., 31.99999, 32.00001, 15., 40., 600., 3.])
    actual = correction.oos_density(w, s, phase)
    independent = [correction.oos_density(np.array([loss]), s, phase).df_dW_total[0] for loss in w]
    np.testing.assert_array_equal(actual.df_dW_total, independent)
    assert actual.df_dW_total[-1] == 0
    assert abs(actual.df_dW_total[1]/actual.df_dW_total[2]-1) < 1e-5
    without_core = correction.oos_density(w, s, phase, include_kshell=False, fail_if_missing_k=False)
    np.testing.assert_array_equal(actual.df_dW_valence, without_core.df_dW_valence)
    assert np.all(without_core.df_dW_OK == 0)


def test_wrong_phase_rejected():
    with pytest.raises(ValueError, match="phase"):
        correction.oos_density([20.], gen.model.epsilon_optical("amorphous"), "hexagonal")


def test_shared_born_normalizer_is_single_source():
    assert gen._optical_normalization is optical.optical_normalization


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
def test_thresholds_match_icymoons_table2_and_appendix_c(phase):
    # Yoffe et al. 2026, ApJS 284:80, Table 2 and Eqs. C3-C6.
    # The ion hydrogenic core intentionally replaces the paper's optical core.
    s = gen.model.epsilon_optical(phase)
    assert s.Bmin == 7.
    assert [o.Bth for o in s.ionizations] == [10., 13., 17., 32.]
    assert [o.E0 for o in s.excitations] == [8.65, 10.50, 12.60, 14.10, 14.50]
    w = np.unique(np.r_[6.9, 7., 9.99, 10., 12.99, 13., 16.99, 17., 31.99, 32., 40.])
    oos = correction.oos_density(w, s, phase)
    assert np.all(oos.df_dW_valence[w < 7.] == 0.)
    assert np.all(oos.df_dW_valence[(w >= 7.) & (w < 10.)] > 0.)
    assert gen.model.OXYGEN_K_B_EV == 543.4
