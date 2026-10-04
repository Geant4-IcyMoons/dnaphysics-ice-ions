"""Numerical checks of the optical audit, not validation of finite-q physics."""
import unittest

import numpy as np

from physics.inelastic_dielectric.benchmarks.optical_elf.run import (
    UNIT, AVOGADRO, H2O_MOLAR_MASS_G_MOL, grid, inputs, moments, reference,
)


class OpticalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = grid(120001)
        cls.data = {phase: inputs(phase, cls.w) for phase in ("amorphous", "hexagonal")}

    def test_log_moment_analytic(self):
        w = np.linspace(1, 100, 100001)
        result, cumulative = moments(w, np.full_like(w, 10/99))
        self.assertAlmostEqual(result["electrons"], 10, places=9)
        exact = np.exp((100*np.log(100)-100+1)/99)
        self.assertLess(abs(result["I_eV"]/exact-1), 1e-8)
        self.assertTrue(np.all(np.diff(cumulative) >= 0))

    def test_invalid_density_rejected(self):
        for values in ([1, -1], [1, np.nan], [0, 0]):
            with self.assertRaises(ValueError):
                moments(np.array([1., 2.]), np.array(values))

    def test_strengths_and_no_forced_oscillator_closure(self):
        for phase, (data, meta) in self.data.items():
            for name, (val, core) in data.items():
                result, _ = moments(self.w, val+core)
                self.assertLess(abs(result["electrons"]-10), .001)
                v, _ = moments(self.w, val)
                k, _ = moments(self.w, core)
                self.assertAlmostEqual(v["electrons"]+k["electrons"], result["electrons"], places=9)
                self.assertLess(abs(k["electrons"]-meta["optical_kshell_continuum_electrons_per_H2O"]), 1e-4)
            np.testing.assert_allclose(data["Born"], data["Oscillator"], rtol=2e-14)

    def test_window_split_is_not_core_partition(self):
        for data, _ in self.data.values():
            for val, core in data.values():
                self.assertTrue(np.all(core[self.w < 100] == 0))
                self.assertGreater(np.trapezoid(val[self.w > 100], self.w[self.w > 100]), .9)

    def test_per_molecule_conversion_and_density_invariance(self):
        for data, meta in self.data.values():
            n = meta["material_molecular_density_m3"]
            df = sum(data["Born"])
            elf = df*n*UNIT/self.w
            np.testing.assert_allclose(elf*self.w/(n*UNIT), df, rtol=1e-14)
            np.testing.assert_allclose(2*elf*self.w/(2*n*UNIT), df, rtol=1e-14)

    def test_figure_extraction_has_no_inset_leak_or_forced_sum(self):
        for phase in self.data:
            w, df = reference(phase)
            n = .94e6*AVOGADRO/H2O_MOLAR_MASS_G_MOL
            elf = df*n*UNIT/w
            self.assertLess(np.max(elf[(w > 33) & (w < 40)]), .4)
            result, _ = moments(w, df)
            self.assertGreater(result["electrons"], 9.5)
            self.assertLess(result["electrons"], 10.5)
            self.assertGreater(abs(result["electrons"]-10), .01)


if __name__ == "__main__":
    unittest.main()
