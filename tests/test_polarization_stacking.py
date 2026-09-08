"""Real Born kernels with synthetic polarization: assembly, not physics validation."""
import json

import numpy as np
import pytest

from test_projectile_relativistic_dcs import MODULE as gen
from physics.inelastic_dielectric.polarization import correction as bd
from physics.inelastic_dielectric.polarization import diagnostic_tables as dt
from physics.inelastic_dielectric.polarization import plot_correction as pc


@pytest.fixture(autouse=True)
def reset_state():
    yield
    gen.set_projectile("proton")
    gen._set_kshell_model("hydrogenic-gos")
    gen._set_projectile_relativistic_dcs(False)


@pytest.mark.parametrize("phase", ["amorphous", "hexagonal"])
@pytest.mark.parametrize("element,charge", [
    ("H", 0), ("H", 1), ("He", 0), ("He", 1), ("He", 2),
    ("C", 3), ("O", 4), ("S", 8)])
def test_pwba_rpwba_and_polarization_stack_once(tmp_path, monkeypatch, phase, element, charge, synthetic_nonlinear_kernel):
    monkeypatch.setattr(gen, "ICE_TYPE", phase)
    gen.set_projectile(element)
    gen._set_projectile_charge_state(charge)
    gen._set_kshell_model("hydrogenic-gos")
    s, c = gen.model.epsilon_optical(phase), gen.model.default_dispersion_coefficients()
    w, energy, nq = np.array([15., 50., 1000.]), 4e7, 256
    t = np.full(w.shape, energy)
    baselines, corrections = [], []
    for relativistic in (False, True):
        gen._set_projectile_relativistic_dcs(relativistic)
        exc = np.array([[gen._selected_dsigma_excitation(loss, energy, k, s, c, nq)
                         for k in range(len(s.excitations))] for loss in w])
        ion = np.array([[gen._selected_dsigma_ionization(loss, energy, k, s, c, nq)
                         for k in range(len(s.ionizations))]
                        + [gen._selected_dsigma_kshell(loss, energy, s, c, nq)] for loss in w])
        baseline = exc.sum(axis=1)+ion.sum(axis=1)
        assert ion[-1, -1] > 0  # The test exercises the K continuum, not only valence.
        if relativistic:
            manual = []
            for loss in w:
                components = [gen._integrate_channel_single_E_rpwba_components(
                    loss, energy, k, channel, s, c, Nq=nq, use_density_effect=True)
                    for channel, count in (("excitation", len(s.excitations)),
                                           ("ionization", len(s.ionizations)))
                    for k in range(count)]
                components.append(gen._integrate_kshell_single_E_rpwba_components(
                    loss, energy, s, c, Nq=nq, use_density_effect=True))
                manual.append(np.sum(components))
            np.testing.assert_allclose(baseline, manual, rtol=2e-14, atol=0)
        data = dict(T_line=t, E_line=w, exc_vals=exc/gen.EMFI_DCS_SCALE_M2,
                    ion_vals=ion/gen.EMFI_DCS_SCALE_M2)
        kwargs = dict(include_barkas_dcs=True, born_reference_charge="bare_Z",
                      projectile_density=gen.PROJECTILE_DENSITY, workers=1,
                      dcs_scale_m2=gen.EMFI_DCS_SCALE_M2)
        if gen.PROJECTILE_DENSITY.electrons:
            destination = tmp_path / str(relativistic)
            metadata = dict(gen._projectile_state_metadata(), include_kshell=True,
                ice_type=phase, projectile_kernel=gen.RPWBA_MODEL_NAME if relativistic else "pwba")
            dt.write_diagnostic_tables(data, s, gen.PROJECTILE_DENSITY, gen.PROJECTILE_MASS_AU,
                destination, destination / "checkpoints", 1, metadata, gen.EMFI_DCS_SCALE_M2,
                dat_names=("exc.dat", "ion.dat", "exc_total.dat", "ion_total.dat"))
            plotted = pc.load_components(destination / "DIAGNOSTIC_ONLY.npz")
            assert not np.any(plotted["flags"] & 7), "Unexpected numerical/domain failure"
            correction = plotted["correction"]
            np.testing.assert_allclose(plotted["born"], baseline, rtol=2e-14, atol=0)
            np.testing.assert_allclose(plotted["total"], baseline+correction, rtol=2e-14, atol=0)
            exported = sum(np.loadtxt(destination / name)[:, 2:].sum(axis=1)
                           for name in ("exc.dat", "ion.dat"))*gen.EMFI_DCS_SCALE_M2
            np.testing.assert_allclose(exported, plotted["total"], rtol=2e-14, atol=0)
            tcs = sum(np.loadtxt(destination / name)[1:].sum()
                      for name in ("exc_total.dat", "ion_total.dat"))*gen.EMFI_DCS_SCALE_M2
            assert tcs == pytest.approx(pc.moments(plotted)["total"][0, 0], rel=2e-14)
            report = json.loads((destination / "DIAGNOSTIC_ONLY.json").read_text())
            assert not report["transport_ready"]
        _, diag = bd.apply_barkas_correction_to_dcs_data(data, s, phase,
            gen.PROJECTILE_MASS_AU, gen.PROJECTILE_CHARGE, **kwargs)
        if gen.PROJECTILE_DENSITY.electrons:
            np.testing.assert_array_equal(diag.DCS_Barkas_m2_per_eV, correction)
        np.testing.assert_allclose(diag.DCS_total_m2_per_eV,
            baseline+diag.DCS_Barkas_m2_per_eV, rtol=2e-14, atol=0)
        assert diag.TCS_total_m2[0] == pytest.approx(np.trapezoid(diag.DCS_total_m2_per_eV, w), rel=2e-14)
        assert diag.S_Barkas_check_eV_m2[0] == pytest.approx(np.trapezoid(w*diag.DCS_Barkas_m2_per_eV, w), rel=2e-14)
        baselines.append(baseline)
        corrections.append(diag.DCS_Barkas_m2_per_eV)
    # The oscillator correction already uses relativistic beta/gamma. The
    # RPWBA switch replaces only Born; it must not multiply polarization again.
    assert np.max(np.abs(baselines[1]/baselines[0]-1)) > 1e-5
    np.testing.assert_array_equal(corrections[0], corrections[1])
