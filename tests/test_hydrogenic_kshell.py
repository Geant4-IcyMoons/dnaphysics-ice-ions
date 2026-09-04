from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from scipy.integrate import quad

PHYSICS_DIR = Path(__file__).resolve().parents[1] / "python_scripts" / "physics_ice"
sys.path.insert(0, str(PHYSICS_DIR))
import emfietzoglou_model_finite_q as model
import generate_ice_cross_sections_ion as generator
import audit_finite_q_sum_rule as sum_rule_audit

H = model.OXYGEN_K_ZEFF**2 * model.RYD_ELECTRON_VOLT


# Independent 80-decimal mpmath evaluation of Heredia-Avalos 2005,
# Appendix A.4-A.9, A.13, n=1. Negative branch used the power of the
# log-ratio directly, not the production log1p rearrangement. E in eV,
# q in a0^-1, df/dE in eV^-1; the factor 256 includes both 1s electrons.
@pytest.mark.parametrize("q,E,expected", [
    (0, 543.400001, 0.005444268919092308702),
    (0, 600, 0.004217393985131209286),
    (0, 800, 0.001980920361764171329),
    (0, 806.6815452623143, 0.0019374863943330775),
    (0, 1000, 0.00108579354603461404),
    (0, 5000, 0.00001004516697931803181),
    (0, 100000, 6.070034104326799435e-10),
    (5, 543.400001, 0.001670086237754265193),
    (5, 600, 0.001743789680130876412),
    (5, 800, 0.001746492032976457495),
    (5, 806.6815452623143, 0.001740891500482019863),
    (5, 1000, 0.001488227562200152899),
    (5, 5000, 0.00001666954218378588348),
    (5, 100000, 6.257237133167591602e-10),
    (10, 543.400001, 0.0002151214503846148632),
    (10, 600, 0.0002486940532406665958),
    (10, 800, 0.0003799880942150948081),
    (10, 806.6815452623143, 0.0003846233121983197017),
    (10, 1000, 0.0005205382857887287181),
    (10, 5000, 0.00006419287081397517776),
    (10, 100000, 6.852794424371070379e-10),
])
def test_published_pointwise_reference(q, E, expected):
    assert model.oxygen_K_hydrogenic_gos_df_dE(E, q) == pytest.approx(
        expected, rel=3e-13, abs=0.0,
    )


def test_threshold_is_a_gate_not_an_energy_shift():
    E = np.array([-1., 0., 543.4, np.nextafter(543.4, np.inf), 600.])
    with np.errstate(all="raise"):
        df = model.oxygen_K_hydrogenic_gos_df_dE(E, 0.)
        elf = model.oxygen_K_hydrogenic_gos_elf(E, 0.)
    np.testing.assert_array_equal(df[:3], 0.)
    np.testing.assert_array_equal(elf[:3], 0.)
    assert np.all(df[3:] > 0.)
    assert model.oxygen_K_hydrogenic_gos_df_dE(600., 0., B_K_eV=580.) == df[-1]
    assert model.oxygen_K_hydrogenic_gos_df_dE(600., 0., B_K_eV=650.) == 0.


@pytest.mark.parametrize("q", [0., 1., 10., 1000.])
def test_branch_boundary_continuity_and_analytic_limit(q):
    Q = (q / model.OXYGEN_K_ZEFF)**2
    expected = 256. * (Q + 1./3.) * np.exp(-4./(Q+1.)) / (Q+1.)**6 / H
    E = H * (1. + np.array([-1e-8, -1e-12, 0., 1e-12, 1e-8]))
    result = model.oxygen_K_hydrogenic_gos_df_dE(E, q)
    np.testing.assert_allclose(result, expected, rtol=4e-8, atol=0.)


def _integrated_strength(q):
    below = quad(lambda E: float(model.oxygen_K_hydrogenic_gos_df_dE(E, q)),
                 model.OXYGEN_K_B_EV, H, epsabs=1e-10, epsrel=1e-10)[0]
    above = quad(
        lambda t: float(model.oxygen_K_hydrogenic_gos_df_dE(H*(1+np.exp(t)), q))
        * H * np.exp(t), -28, 28, epsabs=1e-10, epsrel=1e-10, limit=200,
    )[0]
    return below + above


@pytest.mark.parametrize("q,expected", [
    (0., 1.736914215348305), (2., 1.7712652255570225),
    (5., 1.8901138200678191), (10., 1.9847230845648867),
    (30., 1.99999097109283),
])
def test_integrated_published_strength_not_forced_to_optical_target(q, expected):
    assert _integrated_strength(q) == pytest.approx(expected, rel=2e-9)


@pytest.mark.parametrize("q", [0., 0.1, 1., 5., 10., 30.])
def test_continuum_strength_lookup_matches_direct_quadrature(q):
    lookup = model.oxygen_K_hydrogenic_gos_continuum_strength(q)
    assert lookup == pytest.approx(_integrated_strength(q), rel=2e-7, abs=1e-8)


def test_occupancy_remainder_is_not_misidentified_as_bound_spectrum():
    q = np.array([0., 1., 5., 10., 30.])
    continuum = model.oxygen_K_hydrogenic_gos_continuum_strength(q)
    remainder = model.oxygen_K_hydrogenic_occupancy_remainder(q)
    np.testing.assert_allclose(continuum + remainder, model.OXYGEN_K_OCCUPANCY,
                               rtol=0., atol=2e-15)
    assert np.all(remainder >= 0.)
    assert remainder[0] == pytest.approx(0.263085784651, rel=2e-9)


def test_broadcast_symmetry_and_finite_positive_spectrum():
    E = np.geomspace(np.nextafter(543.4, np.inf), 1e10, 2000)[None, :]
    q = np.array([0., 1e-9, 1., 10., 100., 1e5])[:, None]
    result = model.oxygen_K_hydrogenic_gos_df_dE(E, q)
    assert result.shape == (6, 2000)
    assert np.all(np.isfinite(result)) and np.all(result >= 0.)
    np.testing.assert_array_equal(result, model.oxygen_K_hydrogenic_gos_df_dE(E, -q))


def test_ten_worker_results_match_serial():
    energies = np.tile([600., 800., H, 1000., 5000.], 4)
    momenta = np.repeat([0., 1., 5., 10.], 5)
    expected = model.oxygen_K_hydrogenic_gos_df_dE(energies, momenta)
    with ProcessPoolExecutor(max_workers=10) as pool:
        actual = list(pool.map(model.oxygen_K_hydrogenic_gos_df_dE, energies, momenta))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kwargs", [
    {"B_K_eV": 0.}, {"B_K_eV": np.nan}, {"Zeff": -1.}, {"Zeff": np.inf},
])
def test_invalid_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        model.oxygen_K_hydrogenic_gos_df_dE(600., 0., **kwargs)


@pytest.mark.parametrize("E,q", [(np.nan, 0.), (np.inf, 0.), (600., np.nan), (600., np.inf)])
def test_nonfinite_inputs_fail(E, q):
    with pytest.raises(ValueError):
        model.oxygen_K_hydrogenic_gos_df_dE(E, q)


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
def test_generator_uses_unscaled_gos_and_no_rolloff(monkeypatch, phase):
    monkeypatch.setattr(generator, "KSHELL_MODEL", "hydrogenic-gos")
    s = model.epsilon_optical(phase)
    E, q = np.array([600., 1000., 5000.]), np.array([0., 5., 10.])
    expected = (np.pi/2.) * s.Ep**2 / 10. * model.oxygen_K_hydrogenic_gos_df_dE(E, q) / E
    np.testing.assert_allclose(generator._kshell_hydrogenic_gos_elf(E, q, s), expected,
                               rtol=2e-15, atol=0.)
    assert model.oxygen_K_hydrogenic_gos_fsum(Ep_eV=s.Ep) == pytest.approx(
        1.736914215348305/10., rel=3e-6,
    )

    def forbidden_rolloff(_):
        raise AssertionError("Hydrogenic GOS must not use ELF rolloff")

    monkeypatch.setattr(generator, "_elf_rolloff_factor", forbidden_rolloff)
    assert generator._integrate_kshell_single_E(600., 1e7, s, Nq=80) > 0.
    C = model.default_dispersion_coefficients()
    assert generator._integrate_kshell_single_E_rel(600., 1e7, s, C, Nq=80) > 0.


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
def test_joint_outer_k_continuum_amplitude_budget_is_ten(phase):
    s = model.epsilon_optical(phase)
    C = sum_rule_audit.dispersion_coefficients()
    q = np.array([0., 0.1, 1., 5., 10., 30.])
    fj0 = np.array([entry.f for entry in s.excitations])
    fi0 = np.array([entry.f for entry in s.ionizations])
    fjq_sum = np.sum(model._fj_q(fj0, q, C), axis=1)
    k_strength = model.oxygen_K_hydrogenic_gos_continuum_strength(q)
    fiq = model._renorm_fi_q(
        fi0,
        float(np.sum(fj0)),
        fjq_sum,
        inner_shell_strength_electrons=k_strength,
    )
    budget = 10. * (fjq_sum + np.sum(fiq, axis=1)) + k_strength
    np.testing.assert_allclose(budget, 10., rtol=0., atol=2e-15)

    legacy = model._renorm_fi_q(fi0, float(np.sum(fj0)), fjq_sum)
    expected_legacy = fi0[None, :] * (
        (1. - fjq_sum) / (1. - np.sum(fj0))
    )[:, None]
    np.testing.assert_array_equal(legacy, expected_legacy)


def test_generator_uses_joint_allocation_only_for_hydrogenic_ion_path(monkeypatch):
    s = model.epsilon_optical("amorphous")
    C = sum_rule_audit.dispersion_coefficients()
    E = np.array([20., 50., 600.])
    q = np.array([0., 1., 10.])
    monkeypatch.setattr(generator, "KSHELL_MODEL", "hydrogenic-gos")
    actual = generator._ion_epsilon2_valence(E, q, s, C, partitioned=False)
    expected = model.epsilon2_valence_Eq(
        E,
        q,
        s,
        C,
        partitioned=False,
        inner_shell_strength_electrons=(
            model.oxygen_K_hydrogenic_gos_continuum_strength(q)
        ),
    )
    np.testing.assert_allclose(actual["total"], expected["total"], rtol=0., atol=0.)

    excluded = generator._ion_epsilon2_valence(
        E, q, s, C, partitioned=False, include_kshell=False
    )
    expected_legacy = model.epsilon2_valence_Eq(E, q, s, C, partitioned=False)
    np.testing.assert_allclose(excluded["total"], expected_legacy["total"], rtol=0., atol=0.)

    monkeypatch.setattr(generator, "KSHELL_MODEL", "old-optical")
    legacy = generator._ion_epsilon2_valence(E, q, s, C, partitioned=False)
    np.testing.assert_allclose(legacy["total"], expected_legacy["total"], rtol=0., atol=0.)


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
@pytest.mark.parametrize("q", [0., 0.3, 1., 5., 10., 30.])
def test_component_elf_sum_rule_is_numerically_reconciled(phase, q):
    audit = sum_rule_audit.component_audit(phase, q, 20001)
    assert audit["amplitude_budget_electrons"] == pytest.approx(10., abs=2e-15)
    assert audit["s1_total_modeled_strength_electrons"] == pytest.approx(10., abs=0.05)
    assert audit["s2_total_modeled_strength_electrons"] == pytest.approx(10., abs=0.015)
    assert audit["k_bound_excitation_strength_electrons"] is None
    assert np.isfinite(audit["total_first_energy_moment_eV"])
    assert audit["total_first_energy_moment_eV"] > 0.


def test_component_strength_and_first_energy_moment_converge():
    coarse = sum_rule_audit.component_audit("amorphous", 1., 10001)
    fine = sum_rule_audit.component_audit("amorphous", 1., 20001)
    assert sum_rule_audit.relative_change(
        coarse, fine, "s2_total_modeled_strength_electrons"
    ) < 5e-6
    assert sum_rule_audit.relative_change(
        coarse, fine, "total_first_energy_moment_eV"
    ) < 5e-6


def test_proton_dcs_stopping_moment_converges():
    coarse = sum_rule_audit.dcs_stopping_moment("amorphous", 1.e7, 30, 30)
    fine = sum_rule_audit.dcs_stopping_moment("amorphous", 1.e7, 60, 60)
    assert np.isfinite(fine["stopping_cross_section_eV_m2"])
    assert fine["stopping_cross_section_eV_m2"] > 0.
    assert sum_rule_audit.relative_change(
        coarse, fine, "stopping_cross_section_eV_m2"
    ) < 0.011


def test_cache_provenance_rejects_old_gos(monkeypatch, tmp_path):
    monkeypatch.setattr(generator, "KSHELL_MODEL", "hydrogenic-gos")
    path = tmp_path / "corrections.npz"
    generator.save_cross_section_corrections_npz([1e6], [{}], out_path=path, NE=20, Nq=20)
    with np.load(path, allow_pickle=False) as loaded:
        cache = dict(loaded)
    assert generator._npz_matches_params(cache, 20, 20, T_list=[1e6])
    assert cache["kshell_normalization"].item() == "published-unscaled"
    assert cache["finite_q_sum_rule"].item() == model.ION_FINITE_Q_SUM_RULE_VERSION
    assert np.isnan(cache["kshell_fsum_target"])
    assert cache["kshell_optical_fsum"] == pytest.approx(0.1736914215348305, rel=3e-6)
    for field, bad in [("kshell_gos_version", "threshold-shifted"),
                       ("kshell_normalization", "optical-fsum"),
                       ("finite_q_sum_rule", "legacy")]:
        changed = dict(cache, **{field: bad})
        assert not generator._npz_matches_params(changed, 20, 20)
        changed.pop(field)
        assert not generator._npz_matches_params(changed, 20, 20)


def test_dat_provenance_and_energy_patch_protection(monkeypatch, tmp_path):
    monkeypatch.setattr(generator, "KSHELL_MODEL", "hydrogenic-gos")
    s = model.epsilon_optical("amorphous")
    metadata = generator._ion_normalization_metadata("amorphous", s)
    metadata.update(generator._kshell_generation_metadata())
    data = {"T_line": np.array([1e6, 1e6, 1e6]), "E_line": np.array([10., 20., 30.]),
            "exc_vals": np.ones((3, 1)), "ion_vals": np.ones((3, 1))*2.}
    exc, ion = tmp_path / "exc.dat", tmp_path / "ion.dat"
    generator._write_dcs_tables_from_data(
        data, exc, ion, table_metadata=metadata
    )
    generator._check_energy_patch_normalization(exc, ion, metadata)
    np.testing.assert_array_equal(np.loadtxt(ion)[:, 2:], data["ion_vals"])
    merged = generator._prepare_dcs_data_for_output(
        data, exc, ion, table_metadata=metadata
    )
    np.testing.assert_array_equal(merged["ion_vals"], data["ion_vals"])
    # A legacy table has no formula metadata, even if its numeric grid matches.
    exc.write_text("1e6 10 1\n1e6 20 1\n1e6 30 1\n")
    with pytest.raises(ValueError, match="no-merge-energy-patches"):
        generator._prepare_dcs_data_for_output(
            data, exc, ion, table_metadata=metadata
        )
    replacement = generator._prepare_dcs_data_for_output(
        data,
        exc,
        ion,
        merge_energy_patches=False,
        table_metadata=metadata,
    )
    np.testing.assert_array_equal(replacement["ion_vals"], data["ion_vals"])


def test_final_dcs_integrates_to_exported_tcs(monkeypatch, tmp_path):
    monkeypatch.setattr(generator, "KSHELL_MODEL", "hydrogenic-gos")
    s = model.epsilon_optical("amorphous")
    W = np.unique(np.concatenate((np.geomspace(543.4, 2000., 101), [H])))
    values = np.array([generator._integrate_kshell_single_E(w, 1e6, s, Nq=80) for w in W])
    data = {"T_line": np.full(W.shape, 1e6), "E_line": W,
            "exc_vals": np.zeros((len(W), 1)), "ion_vals": values[:, None]}
    assert np.all(np.isfinite(values)) and np.all(values >= 0.)
    exc, ion = tmp_path / "exc.dat", tmp_path / "ion.dat"
    generator._write_total_tables_from_dcs(data, exc, ion)
    row = np.loadtxt(ion)
    assert row[1] == pytest.approx(generator._simpson_integrate(values, W), rel=1e-9)
