"""Small real nonlinear exports, not a production-grid validation campaign."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json

import numpy as np


def _generate_state(task):
    root, element, charge = task
    from physics.inelastic_dielectric import generate_cross_sections as gen
    from physics.inelastic_dielectric.polarization.nonlinear_polarization import MODEL
    gen.set_projectile(element)
    gen._set_projectile_charge_state(charge)
    t, w = 1e7, np.array([15., 50.])
    reports = []
    for phase in ("amorphous", "hexagonal"):
        s, c = gen.model.epsilon_optical(phase), gen.model.default_dispersion_coefficients()
        for relativistic in (False, True):
            gen._set_projectile_relativistic_dcs(relativistic)
            exc = np.array([[gen._selected_dsigma_excitation(loss,t,k,s,c,128)
                             for k in range(len(s.excitations))] for loss in w])
            ion = np.array([[gen._selected_dsigma_ionization(loss,t,k,s,c,128)
                             for k in range(len(s.ionizations))]+[0.] for loss in w])
            data = dict(T_line=np.full(w.size,t), E_line=w,
                exc_vals=exc/gen.EMFI_DCS_SCALE_M2, ion_vals=ion/gen.EMFI_DCS_SCALE_M2,
                projectile_state_metadata=gen._projectile_state_metadata())
            destination = root/f"{element}_{charge}_{phase}_{relativistic}"
            result = gen.write_emfietzoglou_dcs_tables(s,c,out_dir=destination,
                dcs_data=data,T_list=[t],NE=2,Nq=128,return_data=True,
                max_workers=1,include_barkas_dcs=True,merge_energy_patches=False,
                checkpoint_dir=root/f"checkpoints_{element}_{charge}")
            diag = result["barkas_diagnostics"]
            assert diag.negative_or_unstable_T_eV.size == 0
            np.testing.assert_allclose(diag.DCS_total_m2_per_eV,
                diag.DCS_Born_m2_per_eV+diag.DCS_Barkas_m2_per_eV,rtol=2e-14,atol=0)
            assert np.isclose(diag.TCS_total_m2[0],np.trapezoid(diag.DCS_total_m2_per_eV,w),rtol=1e-13,atol=0)
            for kind in ("excitation","ionisation"):
                path = next(destination.glob(f"sigmadiff_{kind}_*.dat"))
                meta = json.loads(path.read_text().splitlines()[0].split(": ",1)[1])
                assert meta["barkas_model"] == MODEL
                table = np.loadtxt(path)
                totals = np.loadtxt(next(destination.glob(f"sigma_{kind}_*.dat")))
                assert np.all(np.isfinite(table)) and np.all(table[:,2:] >= 0)
                area = .5*(table[1:,2:]+table[:-1,2:])*np.diff(table[:,1])[:,None]
                cdf = np.cumsum(area,axis=0)
                np.testing.assert_allclose(cdf[-1],totals[1:],rtol=2e-14,atol=0)
            reports.append((phase,relativistic,diag.DCS_Barkas_m2_per_eV.tolist()))
    return reports


def test_real_h_and_he_tables_both_phases_and_born_kernels(tmp_path):
    tasks = [(tmp_path,e,q) for e,q in (("H",1),("H",0),("He",2),("He",0))]
    # Four independent states; additional workers would be idle.
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context("spawn")) as pool:
        results = list(pool.map(_generate_state,tasks))
    for state in results:
        assert len(state) == 4
        for i in (0,2):
            np.testing.assert_array_equal(state[i][2],state[i+1][2])
