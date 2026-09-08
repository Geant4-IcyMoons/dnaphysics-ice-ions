"""Plot arithmetic, failure visibility and PDF style, using synthetic inputs."""
import json

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pytest

from physics.constants import EMFI_DCS_SCALE_M2, PAPER_FONTSIZE
from physics.inelastic_dielectric.polarization import plot_correction as pc


def diagnostic(tmp_path, *, correction=None):
    t = np.repeat([1e6, 1e7, 1e8], 3)
    w = np.tile([10., 20., 40.], 3)
    born = np.tile([1., 2., 3.], 3)*1e-22
    correction = .2*born if correction is None else np.asarray(correction)*1e-22
    path = tmp_path / "diagnostic.npz"
    np.savez(path, T_eV=t, W_eV=w, born_m2_eV=born, polarization_m2_eV=correction,
        total_m2_eV=born+correction, rejection_flags=np.zeros(9, np.uint8),
        metadata=json.dumps(dict(projectile_element="He", projectile_charge_state=0,
            ice_type="hexagonal", projectile_kernel="rpwba", diagnostic_only=True)))
    return path


def test_moments_use_saved_components_and_generator_quadrature(tmp_path):
    data = pc.load_components(diagnostic(tmp_path))
    result = pc.moments(data)
    for i in range(3):
        idx = data["T"] == result["T"][i]
        for key in ("born", "correction", "total"):
            np.testing.assert_allclose(result[key][:, i], [
                np.trapezoid(data[key][idx], data["W"][idx]),
                np.trapezoid(data["W"][idx]*data[key][idx], data["W"][idx])], rtol=1e-15)
    np.testing.assert_allclose(result["total"], 1.2*result["born"], rtol=1e-15)


def test_missing_node_is_not_bridged_and_negative_values_survive(tmp_path):
    data = pc.load_components(diagnostic(tmp_path, correction=[.2, np.nan, .6, -2, -3, -4, 5, 6, 7]))
    values = pc.moments(data)
    assert np.all(np.isnan(values["total"][:, 0]))
    assert np.all(np.isfinite(values["born"]))
    assert np.all(values["total"][:, 1] < 0)
    assert data["flags"][1] & 2
    assert np.all(data["flags"][3:6] & 16)
    assert np.all(data["flags"][6:] & 8)


def test_conservative_map_retains_extremes_and_worst_flags(tmp_path):
    data = pc.load_components(diagnostic(tmp_path, correction=[.2, np.nan, .6, -20, -3, -4, 5, 6, 7]))
    _, _, ratio, status = pc.ratio_map(data, max_t_bins=1, max_w_bins=1)
    assert ratio.shape == (1, 1)
    assert ratio[0, 0] == pytest.approx(-20)
    assert status[0, 0] == 3  # Missing node cannot disappear under bin aggregation.


def test_zero_born_nonzero_correction_is_undefined_not_zero(tmp_path):
    path = diagnostic(tmp_path)
    with np.load(path) as saved:
        data = dict(saved)
    data["born_m2_eV"][0] = 0
    data["total_m2_eV"][0] = data["polarization_m2_eV"][0]
    np.savez(path, **data)
    plotted = pc.load_components(path)
    assert plotted["flags"][0] & 32
    _, _, ratio, status = pc.ratio_map(plotted)
    assert np.isnan(ratio[0, 0]) and status[0, 0] == 3


def test_inconsistent_corrected_total_is_rejected(tmp_path):
    path = diagnostic(tmp_path)
    with np.load(path) as saved:
        data = dict(saved)
    data["total_m2_eV"] += data["born_m2_eV"]
    np.savez(path, **data)
    with pytest.raises(ValueError, match="not Born"):
        pc.load_components(path)


def test_bare_cache_uses_shared_units_and_corrected_cache_never_infers_born(tmp_path):
    path = tmp_path / "cache.npz"
    fields = dict(dcs_T_line=np.repeat(1e7, 3), dcs_E_line=[10., 20., 40.],
                  dcs_exc_vals=np.ones((3, 1)), dcs_ion_vals=np.ones((3, 1)),
                  include_barkas_dcs=False)
    np.savez(path, **fields)
    data = pc.load_components(path)
    np.testing.assert_array_equal(data["born"], np.full(3, 2*EMFI_DCS_SCALE_M2))
    np.testing.assert_array_equal(data["correction"], np.zeros(3))
    fields["include_barkas_dcs"] = True
    np.savez(path, **fields)
    with pytest.raises(ValueError, match="lacks separate"):
        pc.load_components(path)


def test_pdf_is_two_pages_and_uses_uniform_courier_style(tmp_path):
    path = diagnostic(tmp_path)
    data = pc.load_components(path)
    with plt.rc_context(pc._style()):
        for build in (pc._moment_figure, pc._map_figure):
            fig = build(data)
            fig.canvas.draw()
            for text in fig.findobj(Text):
                if text.get_visible() and text.get_text():
                    assert text.get_fontsize() == PAPER_FONTSIZE
                    assert text.get_fontfamily() == plt.rcParams["font.family"]
            assert fig.get_figwidth() == 7.1
            assert all(not line.get_visible() for ax in fig.axes for line in ax.get_xgridlines())
            plt.close(fig)
    output = pc.plot_correction(path)
    assert output.parent.name == "plots" and output.suffix == ".pdf"
    content = output.read_bytes()
    assert content.startswith(b"%PDF") and b"/Count 2" in content


def test_charge_zero_is_not_relabelled_nuclear_charge():
    title = pc._title(dict(projectile="alpha", projectile_charge=2,
        projectile_charge_state=0, ice_type="amorphous", projectile_kernel="rpwba"))
    assert "He$^{0}$" in title and "RPWBA" in title
