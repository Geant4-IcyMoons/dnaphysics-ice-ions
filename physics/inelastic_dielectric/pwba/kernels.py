"""PWBA integrands and shared target/projectile state with explicit projectile/target state.

No generator globals are read here. The caller supplies molecular ELF
normalization and the frozen-projectile screening amplitude. See README.md
for sources, units, and the distinct Born/Barkas energy-loss cutoffs.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from physics.constants import C_AU, EH, EV_TO_HA, MC2_eV, a0, ELF_ROLLOFF_COEF, ELF_ROLLOFF_E0_eV
from physics.inelastic_dielectric.finite_q import emfietzoglou_model_finite_q as model
from physics.inelastic_dielectric.numerics import _simpson_integrate

@dataclass(frozen=True)
class PWBAKernel:
    mass_au: float
    nuclear_charge: float
    kshell_model: str
    kshell_B_eV: float
    kshell_Zeff: float
    elf_per_molecule: float | None
    screening_ratio: Callable

    def _elf_rolloff_factor(self, Ei):
        Ei = np.asarray(Ei, dtype=float)
        factor = np.ones_like(Ei)
        mask = Ei > ELF_ROLLOFF_E0_eV
        if np.any(mask):
            factor[mask] = 1.0 - ELF_ROLLOFF_COEF * np.log10(Ei[mask] / ELF_ROLLOFF_E0_eV)
        return factor


    def projectile_rest_energy_eV(self, projectile_mass_au=None):
        if projectile_mass_au is None:
            projectile_mass_au = self.mass_au
        return float(projectile_mass_au) * MC2_eV


    def projectile_beta2(self, Tp_eV, projectile_mass_au=None):
        Tp_eV = float(Tp_eV)
        if Tp_eV <= 0.0:
            return 0.0
        gamma = 1.0 + Tp_eV / self.projectile_rest_energy_eV(projectile_mass_au)
        beta2 = 1.0 - 1.0 / (gamma * gamma)
        return float(np.clip(beta2, 0.0, 1.0 - np.finfo(float).eps))


    def heavy_projectile_Emax(self, Tp_eV, projectile_mass_au=None):
        beta2 = self.projectile_beta2(Tp_eV, projectile_mass_au)
        if beta2 <= 0.0:
            return 0.0
        return float(2.0 * MC2_eV * beta2 / max(1.0 - beta2, np.finfo(float).tiny))


    def _projectile_energy_loss_upper_eV(self, Tp_eV):
        return float(min(float(Tp_eV), self.heavy_projectile_Emax(Tp_eV)))


    def beta2_rel(self, Tj):
        return self.projectile_beta2(Tj)


    def _q_bounds_scalar(self, Ei, Tj, projectile_mass_au=None):
        """Return (q_lo, q_hi) for scalar Ei, Tj (both in eV)."""
        if projectile_mass_au is None:
            projectile_mass_au = self.mass_au
        Ei_H = Ei * EV_TO_HA
        T_H = Tj * EV_TO_HA
        if Ei_H >= T_H:
            return 0.0, 0.0

        d = T_H - Ei_H
        if d <= 0.0:
            return 0.0, 0.0

        sqrtT = np.sqrt(T_H)
        sqrt_d = np.sqrt(d)
        qlo = np.sqrt(2.0 * projectile_mass_au) * (sqrtT - sqrt_d)
        qhi = np.sqrt(2.0 * projectile_mass_au) * (sqrtT + sqrt_d)

        if not np.isfinite(qlo) or not np.isfinite(qhi) or qhi <= qlo:
            return 0.0, 0.0
        return float(qlo), float(qhi)


    def _integrate_channel_single_E(self,
        Ei,
        Tj,
        idx,
        channel_type,
        s,
        C,
        Nq=400,
        use_rel_bounds=False,
        include_kshell_sum_rule=None,
    ):
        """
        Compute inner integral over q:

            ∫ dq [ ELF_channel(Ei, q) / q ]

        for one excitation or ionization channel, at fixed Ei, Tj.
        """
        if use_rel_bounds:
            raise RuntimeError("Relativistic electron q-bounds are disabled for heavy projectiles.")
        if Ei > self._projectile_energy_loss_upper_eV(Tj):
            return 0.0
        qlo, qhi = self._q_bounds_scalar(Ei, Tj, projectile_mass_au=self.mass_au)
        if qhi <= qlo or qlo <= 0.0:
            return 0.0

        # q-grid
        xi = np.linspace(np.log(qlo), np.log(qhi), Nq)
        qvals = np.exp(xi)
        E_arr = np.array([Ei], float)

        # Vectorized dielectric functions at (Ei, qvals)
        e1 = self._ion_epsilon1_valence(
            E_arr, qvals, s, C, include_kshell=include_kshell_sum_rule
        )
        e2 = self._ion_epsilon2_valence(
            E_arr, qvals, s, C, include_kshell=include_kshell_sum_rule
        )

        e1t = e1["total"][:, 0]   # shape (Nq,)
        e2t = e2["total"][:, 0]
        denom = e1t**2 + e2t**2
        denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)

        if channel_type == "excitation":
            vals = e2["excitations"][idx][:, 0] / denom
        elif channel_type == "ionization":
            vals = e2["ionizations"][idx][:, 0] / denom
        else:
            raise ValueError("channel_type must be 'excitation' or 'ionization'")

        vals = vals * self._elf_rolloff_factor(Ei)
        vals = vals * self.screening_ratio(qvals, Ei)
        accum = float(_simpson_integrate(vals, xi))

        T_scaled = Tj / self.mass_au
        int_cons = self.nuclear_charge**2 * self.elf_per_molecule / (
            np.pi * a0 * T_scaled
        )

        return float(int_cons * accum)


    def _kshell_threshold_eV(self, s):
        if self.kshell_model == "hydrogenic-gos":
            return float(self.kshell_B_eV)
        if self.kshell_model == "old-optical" and s.kshell is not None:
            return float(s.kshell.Bth)
        return None


    def _kshell_old_optical_elf(self, Ei, s):
        ks_arr = model.oxygen_K_electron_optical_elf(np.array([Ei], float), s, fsum_corrected=False)
        return float(ks_arr[0])


    def _kshell_hydrogenic_gos_elf(self, Ei, qvals, s):
        return model.oxygen_K_ion_hydrogenic_gos_elf(
            Ei,
            qvals,
            B_K_eV=self.kshell_B_eV,
            Zeff=self.kshell_Zeff,
            Ep_eV=float(s.Ep),
        )


    def _ion_inner_shell_strength_electrons(self, qvals, include_kshell=None):
        """K-continuum strength used in the ion finite-q molecular sum rule."""
        if include_kshell is None:
            include_kshell = self.kshell_model != "none"
        if not include_kshell or self.kshell_model != "hydrogenic-gos":
            return None
        return model.oxygen_K_hydrogenic_gos_continuum_strength(
            qvals,
            B_K_eV=self.kshell_B_eV,
            Zeff=self.kshell_Zeff,
        )


    def _ion_epsilon1_valence(self, E_arr, qvals, s, C, include_kshell=None):
        return model.epsilon1_valence_Eq(
            E_arr,
            qvals,
            s,
            C,
            inner_shell_strength_electrons=self._ion_inner_shell_strength_electrons(
                qvals, include_kshell=include_kshell
            ),
        )


    def _ion_epsilon2_valence(self,
        E_arr, qvals, s, C, partitioned=True, include_kshell=None
    ):
        return model.epsilon2_valence_Eq(
            E_arr,
            qvals,
            s,
            C,
            partitioned=partitioned,
            inner_shell_strength_electrons=self._ion_inner_shell_strength_electrons(
                qvals, include_kshell=include_kshell
            ),
        )


    def _integrate_kshell_single_E(self,
        Ei, Tj, s, Nq=400, include_kshell=True, use_rel_bounds=False
    ):
        """
        Inner integral over q for K-shell:
        """
        if not include_kshell or (s.kshell is None) or self.kshell_model == "none":
            return 0.0

        if use_rel_bounds:
            raise RuntimeError("Relativistic electron q-bounds are disabled for heavy projectiles.")
        kshell_B = self._kshell_threshold_eV(s)
        if kshell_B is None or Ei <= kshell_B:
            return 0.0
        if Ei > self._projectile_energy_loss_upper_eV(Tj):
            return 0.0
        qlo, qhi = self._q_bounds_scalar(Ei, Tj, projectile_mass_au=self.mass_au)
        if qhi <= qlo or qlo <= 0.0:
            return 0.0

        xi = np.linspace(np.log(qlo), np.log(qhi), Nq)
        qvals = np.exp(xi)

        if self.kshell_model == "old-optical":
            ks_val = self._kshell_old_optical_elf(Ei, s)
            if ks_val == 0.0:
                return 0.0
            vals = np.full_like(xi, ks_val)
            vals = vals * self._elf_rolloff_factor(Ei)
        elif self.kshell_model == "hydrogenic-gos":
            vals = self._kshell_hydrogenic_gos_elf(Ei, qvals, s)
        else:
            return 0.0

        vals = vals * self.screening_ratio(qvals, Ei)
        accum = float(_simpson_integrate(vals, xi))

        T_scaled = Tj / self.mass_au
        int_cons = self.nuclear_charge**2 * self.elf_per_molecule / (
            np.pi * a0 * T_scaled
        )

        return float(int_cons * accum)
