"""Regression tests for the actual ion export grid and linear-W table measure."""
import json

import numpy as np
import pytest

from test_projectile_relativistic_dcs import MODULE as gen, _optical_and_dispersion


@pytest.mark.parametrize("projectile", tuple(gen.PROJECTILE_LIBRARY))
def test_requested_grid_and_cutoff(projectile):
    gen.set_projectile(projectile)
    s, _ = _optical_and_dispersion()
    T = np.geomspace(1e5, 1e8, 11)
    grid = gen._ion_dcs_grid(T, s, 300, True)
    np.testing.assert_array_equal(list(grid), T)
    for t, w in grid.items():
        assert len(w) >= 300
        assert np.all(np.diff(w) > 0)
        assert w[-1] == gen._projectile_energy_loss_upper_eV(t)
        if w[0] < gen.KSHELL_B_EV < w[-1]:
            assert gen.KSHELL_B_EV in w
    gen.set_projectile("proton")


def test_uniform_grid_uses_linear_density_not_simpson():
    gen.set_projectile("proton")
    T, total = gen._integrate_dcs_to_totals([1e6]*3, [10, 20, 30], [0, 1, 0])
    assert total[0, 0] == pytest.approx(10.0)


@pytest.mark.parametrize("barkas", [False, True])
@pytest.mark.parametrize("relativistic", [False, True])
def test_writer_honours_requested_nodes(tmp_path, monkeypatch, barkas, relativistic, synthetic_nonlinear_kernel):
    gen.set_projectile("proton")
    gen._set_projectile_relativistic_dcs(relativistic)
    s, C = _optical_and_dispersion()
    # A known positive Born density isolates export/integration from q quadrature.
    for name in ("excitation", "ionization", "kshell"):
        monkeypatch.setattr(gen, "_selected_dsigma_" + name, lambda *a: 1e-23)
    monkeypatch.setattr(gen, "_export_to_custom_geant4", lambda paths: None)
    T = np.array([1.1e5, 3.33e6, 1e8])
    data = gen.write_emfietzoglou_dcs_tables(
        s, C, out_dir=tmp_path, T_list=T, NE=37, Nq=20, return_data=True,
        parallel_channels=False, include_kshell=True, include_barkas_dcs=barkas,
        charge_mode="bare", merge_energy_patches=False,
    )
    np.testing.assert_array_equal(np.unique(data["T_line"]), T)
    for channel in ("excitation", "ionisation"):
        path = next(tmp_path.glob(f"sigmadiff_{channel}_*.dat"))
        dcs = np.loadtxt(path)
        total = np.loadtxt(next(tmp_path.glob(f"sigma_{channel}_*.dat")))
        meta = json.loads(path.read_text().splitlines()[0].split(": ", 1)[1])
        assert meta["dcs_numerics_version"] == gen.DCS_NUMERICS_VERSION
        assert meta["ice_type"] == s.material
        assert s.material in path.name
        for i, t in enumerate(T):
            rows = dcs[dcs[:, 0] == t]
            assert 37 <= len(rows) < 65
            assert rows[-1, 1] == gen._projectile_energy_loss_upper_eV(t)
            np.testing.assert_allclose(
                np.trapezoid(rows[:, 2:], rows[:, 1], axis=0), total[i, 1:],
                rtol=1e-14, atol=0.0,
            )
        assert np.all(np.isfinite(dcs)) and np.all(dcs[:, 2:] >= 0)
    gen._set_projectile_relativistic_dcs(False, False, use_density_effect=False)


@pytest.mark.parametrize("energies", [[], [1, 1], [2, 1], [float("nan")], [-1]])
def test_invalid_incident_grids_fail(energies):
    s, _ = _optical_and_dispersion()
    with pytest.raises(ValueError, match="T_list"):
        gen._ion_dcs_grid(energies, s, 30, True)


def test_old_numerics_metadata_is_rejected():
    metadata = gen._ion_normalization_metadata("amorphous")
    metadata.pop("dcs_numerics_version")
    with pytest.raises(ValueError, match="dcs_numerics_version"):
        gen._require_current_normalization(metadata)


def test_selected_totals_preserve_born_diagnostics_and_separate_core():
    gen.set_projectile("proton")
    original = dict(
        excitation_sigma_pwba=[123.0], ionization_sigma_pwba=[456.0],
        valence_sigma_pwba=579.0, total_sigma_pwba=600.0,
    )
    data = dict(T_line=[1e6]*3, E_line=[10, 20, 30],
                exc_vals=[[0], [1], [0]], ion_vals=[[1, 0], [1, 2], [1, 0]])
    selected = gen._replace_sigma_list_with_dcs_totals([1e6], [original], data)[0]
    for key, value in original.items():
        assert selected[key] == value
    assert selected["total_sigma"] == pytest.approx(50 * gen.EMFI_DCS_SCALE_M2)
    assert selected["valence_sigma"] == pytest.approx(30 * gen.EMFI_DCS_SCALE_M2)
    assert selected["kshell_sigma"] == pytest.approx(20 * gen.EMFI_DCS_SCALE_M2)
    assert len(selected["ionization_sigma"]) == 1
    with pytest.raises(ValueError, match="match"):
        gen._replace_sigma_list_with_dcs_totals([2e6], [original], data)


def test_energy_grid_preserves_exact_cli_endpoints():
    assert gen._energy_grid(1.1e5, 1e8, 1000)[0] == 1.1e5
    assert gen._energy_grid(1.1e5, 1e8, 1000)[-1] == 1e8
