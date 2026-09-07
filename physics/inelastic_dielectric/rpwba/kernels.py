"""Finite-Q Dominguez-Munoz RPWBA longitudinal and transverse integrands.

Inherits the same target, K-shell and projectile state as PWBA; no copies of
the optical response or projectile screening data are maintained here.
"""
import numpy as np
from physics.constants import C_AU, EH, EV_TO_HA, MC2_eV, a0
from physics.inelastic_dielectric.numerics import _simpson_integrate
from physics.inelastic_dielectric.pwba.kernels import PWBAKernel

class RPWBAKernel(PWBAKernel):
    def _q_bounds_scalar_rel(self, Ei, Tj):
        """Return exact relativistic ion momentum-transfer bounds in a0^-1.

        For ion rest energy M c^2 and energy loss W,
            q_- = [p(T) - p(T-W)] / (hbar/a0),
            q_+ = [p(T) + p(T-W)] / (hbar/a0),
        where p c = sqrt[T(T + 2 M c^2)]. The selected projectile supplies M.
        """
        Ei = float(Ei)
        Tj = float(Tj)
        if Ei <= 0.0 or Ei >= Tj:
            return 0.0, 0.0
        rest_H = self.projectile_rest_energy_eV() * EV_TO_HA
        T_H = Tj * EV_TO_HA
        final_H = (Tj - Ei) * EV_TO_HA
        p_initial = np.sqrt(T_H * (T_H + 2.0 * rest_H)) / C_AU
        p_final = np.sqrt(final_H * (final_H + 2.0 * rest_H)) / C_AU
        qhi = p_initial + p_final
        loss_H = Ei * EV_TO_HA
        # Rationalized p(T)-p(T-W), avoiding cancellation for W << T.
        qlo = (
            loss_H * (2.0 * (T_H + rest_H) - loss_H)
            / (C_AU**2 * qhi)
        )
        if (not np.isfinite(qlo)) or (not np.isfinite(qhi)) or qlo <= 0.0 or qhi <= qlo:
            return 0.0, 0.0
        return float(qlo), float(qhi)


    def _projectile_relativistic_longitudinal_prefactor(self, Tj, s):
        """Microscopic Dominguez-Munoz RPWBA longitudinal DCS prefactor.

        d sigma_L/dW = 2 z^2 / (pi a0 N m_e c^2 beta^2)
                       * integral[dq/q Im(-1/epsilon(W,q))].

        This is Eq. (3) of Dominguez-Munoz et al. (2022), after substituting
        their Eq. (9), changing variables from recoil energy Q to momentum q,
        and dividing the sum-rule-normalized DIMFP by the phase molecular
        density N. _ion_elf_per_molecule_factor supplies the ELF scale/N. The selected
        projectile supplies z and beta. The established high-mass energy-loss
        cutoff remains unchanged in heavy_projectile_Emax().
        """
        beta2 = self.projectile_beta2(Tj)
        if beta2 <= 0.0:
            return 0.0
        return float(
            2.0 * self.nuclear_charge**2 * self.elf_per_molecule
            / (np.pi * a0 * MC2_eV * beta2)
        )


    def _rpwba_transverse_ratio(self,
        W_eV,
        q_au,
        beta2,
        epsilon1=None,
        epsilon2=None,
        use_density_effect=False,
    ):
        """Return the finite-Q transverse/longitudinal integrand ratio.

        With R = Q(Q + 2 m_e c^2) = (q c)^2 and rho = W^2/R, the second
        term in braces in Dominguez-Munoz et al. Eq. (3), divided by the first,
        is

            rho * (beta^2 - rho) / (1 - rho)^2.

        For the medium use Dominguez-Munoz's thesis Eq. (2.287), with the
        stated approximation epsilon_T = epsilon_L and transverse GOS = GOS:

            rho * |epsilon|^2 * (beta^2 - rho)
            / [(1 - rho epsilon_1)^2 + (rho epsilon_2)^2].

        This follows by dividing its transverse coefficient by 2*m_e*c^2/(W*R).
        It recovers the vacuum expression as epsilon -> 1. The earlier
        W/(2*m_e*c^2) implementation did not. We do not identify this with the
        literal printed Eq. (8) of the 2022 paper, whose normalization differs.
        Thesis source (pp. 71-72, Eqs. 2.282-2.289):
        https://idus.us.es/bitstreams/d1adf440-b21e-41dd-bc17-d44c484bc576/download
        """
        q_au = np.asarray(q_au, dtype=float)
        W_eV = float(W_eV)
        beta2 = float(beta2)
        qc_eV = C_AU * EH * q_au
        recoil_product = qc_eV * qc_eV
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            rho = (W_eV * W_eV) / recoil_product

        tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(beta2))
        if np.any(rho > beta2 + tolerance):
            raise FloatingPointError("RPWBA q grid violates beta^2 - W^2/(qc)^2 >= 0.")
        beta_minus_rho = np.maximum(beta2 - rho, 0.0)

        if use_density_effect:
            if epsilon1 is None or epsilon2 is None:
                raise ValueError("Finite-Q epsilon1 and epsilon2 are required for the density effect.")
            epsilon1 = np.asarray(epsilon1, dtype=float)
            epsilon2 = np.asarray(epsilon2, dtype=float)
            denominator = (1.0 - rho * epsilon1) ** 2 + (rho * epsilon2) ** 2
            denominator = np.maximum(denominator, np.finfo(float).tiny)
            ratio = (
                rho
                * (epsilon1 * epsilon1 + epsilon2 * epsilon2)
                * beta_minus_rho
                / denominator
            )
        else:
            denominator = np.maximum((1.0 - rho) ** 2, np.finfo(float).tiny)
            ratio = rho * beta_minus_rho / denominator

        if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
            raise FloatingPointError("Non-finite or negative RPWBA transverse kernel.")
        return ratio


    def _integrate_channel_single_E_rpwba_components(self,
        Ei,
        Tj,
        idx,
        channel_type,
        s,
        C,
        Nq=400,
        use_density_effect=False,
        include_kshell_sum_rule=None,
    ):
        """Return finite-Q longitudinal and transverse RPWBA DCS components.

        The integration is Eq. (4) of Dominguez-Munoz et al. (2022), evaluated
        on logarithmic q after applying Eqs. (2), (3), and (9). Both terms use
        the same channel-resolved finite-q GOS/ELF. The density option applies
        thesis Eq. (2.287) through the complex ice dielectric function; see
        _rpwba_transverse_ratio for the source and approximations.
        """
        if Ei > self._projectile_energy_loss_upper_eV(Tj):
            return 0.0, 0.0
        qlo, qhi = self._q_bounds_scalar_rel(Ei, Tj)
        if qhi <= qlo or qlo <= 0.0:
            return 0.0, 0.0

        xi = np.linspace(np.log(qlo), np.log(qhi), Nq)
        qvals = np.exp(xi)
        E_arr = np.array([Ei], float)
        e1 = self._ion_epsilon1_valence(
            E_arr, qvals, s, C, include_kshell=include_kshell_sum_rule
        )
        e2 = self._ion_epsilon2_valence(
            E_arr, qvals, s, C, include_kshell=include_kshell_sum_rule
        )
        denominator = e1["total"][:, 0] ** 2 + e2["total"][:, 0] ** 2
        denominator = np.where(denominator == 0.0, np.finfo(float).tiny, denominator)
        if channel_type == "excitation":
            vals = e2["excitations"][idx][:, 0] / denominator
        elif channel_type == "ionization":
            vals = e2["ionizations"][idx][:, 0] / denominator
        else:
            raise ValueError("channel_type must be 'excitation' or 'ionization'")
        vals = vals * self._elf_rolloff_factor(Ei)
        vals = vals * self.screening_ratio(qvals, Ei, relativistic=True)
        beta2 = self.projectile_beta2(Tj)
        transverse_ratio = self._rpwba_transverse_ratio(
            Ei,
            qvals,
            beta2,
            epsilon1=e1["total"][:, 0],
            epsilon2=e2["total"][:, 0],
            use_density_effect=use_density_effect,
        )
        prefactor = self._projectile_relativistic_longitudinal_prefactor(Tj, s)
        longitudinal = prefactor * _simpson_integrate(vals, xi)
        transverse = prefactor * _simpson_integrate(vals * transverse_ratio, xi)
        return float(longitudinal), float(transverse)


    def _integrate_channel_single_E_rel(self,
        Ei, Tj, idx, channel_type, s, C, Nq=400, include_kshell_sum_rule=None
    ):
        longitudinal, _ = self._integrate_channel_single_E_rpwba_components(
            Ei,
            Tj,
            idx,
            channel_type,
            s,
            C,
            Nq=Nq,
            use_density_effect=False,
            include_kshell_sum_rule=include_kshell_sum_rule,
        )
        return longitudinal


    def _integrate_channel_single_E_trans(self,
        Ei,
        Tj,
        idx,
        channel_type,
        s,
        C,
        Nq=400,
        use_density_effect=False,
        include_kshell_sum_rule=None,
    ):
        _, transverse = self._integrate_channel_single_E_rpwba_components(
            Ei,
            Tj,
            idx,
            channel_type,
            s,
            C,
            Nq=Nq,
            use_density_effect=use_density_effect,
            include_kshell_sum_rule=include_kshell_sum_rule,
        )
        return transverse


    def _integrate_kshell_single_E_rpwba_components(self,
        Ei,
        Tj,
        s,
        C,
        Nq=400,
        include_kshell=True,
        use_density_effect=False,
    ):
        """Return longitudinal and transverse RPWBA O K-shell DCS components.

        The hydrogenic K-shell GOS is additive and has no corresponding complex
        K-shell epsilon in the current ice model. The medium screening factor is
        therefore evaluated with the finite-q valence epsilon. This approximation
        is recorded in output metadata and is relevant only when the density
        correction is enabled.
        """
        if not include_kshell or (s.kshell is None) or self.kshell_model == "none":
            return 0.0, 0.0
        threshold = self._kshell_threshold_eV(s)
        if threshold is None or Ei <= threshold or Ei > self._projectile_energy_loss_upper_eV(Tj):
            return 0.0, 0.0
        qlo, qhi = self._q_bounds_scalar_rel(Ei, Tj)
        if qhi <= qlo or qlo <= 0.0:
            return 0.0, 0.0
        xi = np.linspace(np.log(qlo), np.log(qhi), Nq)
        qvals = np.exp(xi)
        if self.kshell_model == "old-optical":
            vals = np.full_like(xi, self._kshell_old_optical_elf(Ei, s))
            vals = vals * self._elf_rolloff_factor(Ei)
        elif self.kshell_model == "hydrogenic-gos":
            vals = self._kshell_hydrogenic_gos_elf(Ei, qvals, s)
        else:
            return 0.0, 0.0

        vals = vals * self.screening_ratio(qvals, Ei, relativistic=True)
        E_arr = np.array([Ei], float)
        e1 = self._ion_epsilon1_valence(E_arr, qvals, s, C, include_kshell=True)
        e2 = self._ion_epsilon2_valence(E_arr, qvals, s, C, include_kshell=True)
        transverse_ratio = self._rpwba_transverse_ratio(
            Ei,
            qvals,
            self.projectile_beta2(Tj),
            epsilon1=e1["total"][:, 0],
            epsilon2=e2["total"][:, 0],
            use_density_effect=use_density_effect,
        )
        prefactor = self._projectile_relativistic_longitudinal_prefactor(Tj, s)
        longitudinal = prefactor * _simpson_integrate(vals, xi)
        transverse = prefactor * _simpson_integrate(vals * transverse_ratio, xi)
        return float(longitudinal), float(transverse)


    def _integrate_kshell_single_E_rel(self, Ei, Tj, s, C, Nq=400, include_kshell=True):
        longitudinal, _ = self._integrate_kshell_single_E_rpwba_components(
            Ei,
            Tj,
            s,
            C,
            Nq=Nq,
            include_kshell=include_kshell,
            use_density_effect=False,
        )
        return longitudinal
