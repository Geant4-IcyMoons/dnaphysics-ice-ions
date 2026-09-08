"""Barkas single-differential cross-section helpers for ion projectiles.

The Ashley-Ritchie-Brandt functions ARBI1/ARBI2 are ported from the
official SBETHE Fortran source distribution:

    Francesc Salvat and Pedro Andreo,
    "SBETHE: Stopping powers of materials for swift charged particles from
    the corrected Bethe formula (new version announcement)",
    Mendeley Data, version 2, DOI: 10.17632/7zw25f428t.2.

Source file: sbethe/sbethe.f, SHA-256 of sbethe.zip:
    d5d4879c2073ada3bd799fe0727054549cd6ec65cc699acbd0ff25fd5c3c4144

Dataset licence: CC BY-NC 3.0. Keep this provenance with the ported code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from physics.inelastic_dielectric.finite_q import emfietzoglou_model_finite_q as model

RE_CLASSICAL_CM = 2.8179403205e-13
ALPHA_FINE = 7.2973525643e-3
MEC2_EV = 510998.95069
H2O_Z_MOL = 10.0
H2O_ZBAR = 10.0 / 3.0
H2O_CB = max(1.0, H2O_ZBAR / 10.0)
BARKAS_ZEFF_COEFF = 125.0
CM2_TO_M2 = 1.0e-4
U_TO_ELECTRON_MASS = 1822.888486217313
ARBI_SOURCE_SHA256 = "d5d4879c2073ada3bd799fe0727054549cd6ec65cc699acbd0ff25fd5c3c4144"
BORN_REFERENCE_CHOICES = ("bare_Z", "unit_charge", "explicit_q")


@dataclass(frozen=True)
class OOSDensity:
    W_eV: np.ndarray
    W_unique_eV: np.ndarray
    df_dW_total: np.ndarray
    df_dW_valence: np.ndarray
    df_dW_OK: np.ndarray
    df_dW_total_unique: np.ndarray
    df_dW_valence_unique: np.ndarray
    df_dW_OK_unique: np.ndarray
    valence_integral_raw: float
    ok_integral_raw: float
    total_integral_norm_grid: float
    total_integral_unique_grid: float
    valence_norm: float
    ok_norm: float


@dataclass(frozen=True)
class DCSDiagnostics:
    T_line: np.ndarray
    W_line: np.ndarray
    z_int: np.ndarray
    DCS_Born_m2_per_eV: np.ndarray
    DCS_Barkas_m2_per_eV: np.ndarray
    DCS_total_m2_per_eV: np.ndarray
    TCS_T_eV: np.ndarray
    TCS_Born_m2: np.ndarray
    TCS_Barkas_m2: np.ndarray
    TCS_total_m2: np.ndarray
    S_Barkas_check_eV_m2: np.ndarray
    negative_or_unstable_T_eV: np.ndarray
    negative_channel_T_eV: np.ndarray
    nonfinite_T_eV: np.ndarray
    dcs_total_min_m2_per_eV: float
    dcs_total_max_m2_per_eV: float
    df_dW_total: np.ndarray
    df_dW_valence: np.ndarray
    df_dW_OK: np.ndarray
    df_dW_total_unique: np.ndarray
    df_dW_valence_unique: np.ndarray
    df_dW_OK_unique: np.ndarray
    df_dW_integral: float
    df_dW_integral_norm_grid: float
    df_dW_integral_unique_grid: float
    df_dW_valence_raw_integral: float
    df_dW_OK_raw_integral: float
    df_dW_valence_norm: float
    df_dW_OK_norm: float
    born_reference_charge: str
    born_reference_q: float
    barkas_channel_distribution: str
    barkas_model_metadata: dict
    barkas_quadrature_error_m2_per_eV: np.ndarray


def mass_u_to_electron_mass(mass_u):
    """Convert a projectile mass in atomic-mass units u to M_projectile/m_e."""
    return float(mass_u) * U_TO_ELECTRON_MASS


def mass_number_to_electron_mass(mass_number):
    """Approximate M_projectile/m_e from mass number A."""
    return mass_u_to_electron_mass(float(mass_number))


def _maybe_scalar(x, y):
    arr = np.asarray(y, dtype=float)
    if np.ndim(x) == 0:
        return float(arr)
    return arr


def arbi1(x):
    """SBETHE ARBI1(X), Ashley-Ritchie-Brandt I1(X)."""
    x_in = x
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)

    m = x < 1.0e-10
    with np.errstate(divide="ignore", invalid="ignore"):
        out[m] = 1.5 * np.pi * np.log(0.375 / x[m])

    m = (x >= 1.0e-10) & (x < 0.25)
    if np.any(m):
        P1 = 1.796676e1
        P2 = 4.825194e1
        P3 = 4.661380e1
        P4 = 2.269807e1
        P5 = 5.409180e0
        P6 = 5.116173e-1
        XL = np.log(x[m])
        out[m] = (
            1.5 * np.pi * np.log(0.375 / x[m])
            - 2.0 * np.pi * x[m] ** 2 * (P1 + XL * (P2 + XL * (P3 + XL * (P4 + XL * (P5 + XL * P6)))))
        )

    m = (x >= 0.25) & (x < 2.0)
    if np.any(m):
        P1 = -3.062544e0
        P2 = 1.787672e1
        P3 = -4.100572e1
        P4 = 4.652797e1
        P5 = -2.728692e1
        P6 = 8.070905e0
        P7 = -9.587701e-1
        XR = np.sqrt(1.0 / x[m])
        out[m] = (
            P1 + XR * (P2 + XR * (P3 + XR * (P4 + XR * (P5 + XR * (P6 + XR * P7)))))
        ) * XR**1.25

    m = (x >= 2.0) & (x < 15.01)
    if np.any(m):
        P1 = -1.018033e-3
        P2 = 1.080448e-1
        P3 = 1.568923e0
        P4 = -2.779110e0
        P5 = 8.079930e0
        P6 = -7.244712e0
        XR = 1.0 / x[m]
        out[m] = (
            P1 + XR * (P2 + XR * (P3 + XR * (P4 + XR * (P5 + XR * P6))))
        ) * np.exp(-2.0 * x[m])

    m = (x >= 15.01) & (x < 50.01)
    if np.any(m):
        P1 = -2.798361e1
        P2 = 3.964525e0
        P3 = -5.706711e-1
        P4 = 2.652851e-2
        P5 = -6.442942e-4
        P6 = 8.003948e-6
        P7 = -4.016094e-8
        out[m] = np.exp(P1 + x[m] * (P2 + x[m] * (P3 + x[m] * (P4 + x[m] * (P5 + x[m] * (P6 + x[m] * P7))))))

    return _maybe_scalar(x_in, out)


def arbi2(x):
    """SBETHE ARBI2(X), Ashley-Ritchie-Brandt I2(X)."""
    x_in = x
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)

    m = x < 1.0e-9
    out[m] = 2.17759

    m = (x >= 1.0e-9) & (x < 0.25)
    if np.any(m):
        P1 = 2.177590e0
        P2 = 5.689823e-1
        P3 = 1.038828e0
        P4 = 2.877808e-4
        XL = np.log(x[m]) ** 2
        out[m] = P1 - 2.0 * np.pi * x[m] ** 2 * (P2 + XL * (P3 + XL * P4))

    m = (x >= 0.25) & (x < 2.0)
    if np.any(m):
        P1 = 3.768512e0
        P2 = -2.135702e1
        P3 = 4.601060e1
        P4 = -4.714780e1
        P5 = 2.551946e1
        P6 = -7.102352e0
        P7 = 8.042960e-1
        XR = np.sqrt(1.0 / x[m])
        out[m] = (
            P1 + XR * (P2 + XR * (P3 + XR * (P4 + XR * (P5 + XR * (P6 + XR * P7)))))
        ) * XR**2 * np.exp(-1.5 * x[m])

    m = (x >= 2.0) & (x < 15.01)
    if np.any(m):
        P1 = 7.431717e-5
        P2 = 6.662051e-2
        P3 = 2.142710e0
        P4 = -7.407167e0
        P5 = 2.327532e1
        P6 = -3.497742e1
        P7 = 1.950319e1
        XR = 1.0 / x[m]
        out[m] = (
            P1 + XR * (P2 + XR * (P3 + XR * (P4 + XR * (P5 + XR * (P6 + XR * P7)))))
        ) * np.exp(-2.0 * x[m])

    m = (x >= 15.01) & (x < 50.01)
    if np.any(m):
        P1 = -2.808307e1
        P2 = 3.977364e0
        P3 = -5.714987e-1
        P4 = 2.655846e-2
        P5 = -6.449171e-4
        P6 = 8.010890e-6
        P7 = -4.019306e-8
        out[m] = np.exp(P1 + x[m] * (P2 + x[m] * (P3 + x[m] * (P4 + x[m] * (P5 + x[m] * (P6 + x[m] * P7))))))

    return _maybe_scalar(x_in, out)


def projectile_beta_gamma(T_total_eV, projectile_mass_me):
    T = np.asarray(T_total_eV, dtype=float)
    gamma = 1.0 + np.maximum(T, 0.0) / (float(projectile_mass_me) * MEC2_EV)
    beta2 = 1.0 - 1.0 / np.square(gamma)
    beta2 = np.clip(beta2, 0.0, 1.0 - np.finfo(float).eps)
    return np.sqrt(beta2), gamma


def wmax_eV(T_total_eV, projectile_mass_me):
    beta, gamma = projectile_beta_gamma(T_total_eV, projectile_mass_me)
    mass = float(projectile_mass_me)
    denom = 1.0 + 2.0 * gamma / mass + 1.0 / (mass * mass)
    return 2.0 * beta * beta * gamma * gamma * MEC2_EV / denom


def barkas_effective_charge(T_total_eV, nuclear_charge, projectile_mass_me):
    z = float(nuclear_charge)
    beta, _ = projectile_beta_gamma(T_total_eV, projectile_mass_me)
    if z <= 0.0:
        return np.zeros_like(np.asarray(T_total_eV, dtype=float))
    zeff = z * (1.0 - np.exp(-BARKAS_ZEFF_COEFF * beta * z ** (-2.0 / 3.0)))
    return np.clip(zeff, 0.0, z)


def interaction_charge(T_total_eV, nuclear_charge, projectile_mass_me, charge_mode="bare", explicit_charge=None):
    mode = str(charge_mode).strip().lower()
    if mode == "bare":
        return np.full_like(np.asarray(T_total_eV, dtype=float), float(nuclear_charge), dtype=float)
    if mode == "zeff":
        return barkas_effective_charge(T_total_eV, nuclear_charge, projectile_mass_me)
    if mode == "explicit":
        if explicit_charge is None:
            raise ValueError("charge_mode='explicit' requires explicit_charge.")
        return np.full_like(np.asarray(T_total_eV, dtype=float), float(explicit_charge), dtype=float)
    raise ValueError("charge_mode must be bare, zeff, or explicit.")


def barkas_xi(W_eV, T_total_eV, projectile_mass_me, C_B=H2O_CB):
    W = np.asarray(W_eV, dtype=float)
    beta, gamma = projectile_beta_gamma(T_total_eV, projectile_mass_me)
    beta2 = np.maximum(beta * beta, np.finfo(float).tiny)
    return 0.5616 * float(C_B) * W / (gamma * beta2 * MEC2_EV)


def _valence_oos_raw(W_eV, s):
    W = np.asarray(W_eV, dtype=float)
    e1 = model.epsilon1_valence_E0(W, s)["total"]
    e2 = model.epsilon2_valence_E0(W, s)["total"]
    denom = np.where(e1 * e1 + e2 * e2 > 0.0, e1 * e1 + e2 * e2, np.finfo(float).tiny)
    elf = e2 / denom
    df = 2.0 * H2O_Z_MOL * W * elf / (np.pi * float(s.Ep) ** 2)
    return np.where(np.isfinite(df) & (df > 0.0), df, 0.0)


def _ok_oos_raw(W_eV):
    W = np.asarray(W_eV, dtype=float)
    df = model.oxygen_K_hydrogenic_gos_df_dE(W, 0.0)
    return np.where(np.isfinite(df) & (df > 0.0), df, 0.0)


@lru_cache(maxsize=16)
def _oos_norms(material, Ep_eV, include_kshell):
    s = model.epsilon_optical(material)
    W = np.geomspace(1.0e-6, 1.0e8, 50000)
    raw_val = _valence_oos_raw(W, s)
    val_int = float(np.trapezoid(raw_val, W))
    if not np.isfinite(val_int) or val_int <= 0.0:
        raise RuntimeError("Cannot normalize Barkas valence OOS density.")
    val_norm = 8.0 / val_int

    ok_int = 0.0
    ok_norm = 0.0
    if include_kshell:
        raw_ok = _ok_oos_raw(W)
        ok_int = float(np.trapezoid(raw_ok, W))
        if not np.isfinite(ok_int) or ok_int <= 0.0:
            raise RuntimeError("Cannot normalize Barkas O K-shell OOS density.")
        ok_norm = 2.0 / ok_int
    total_norm_grid = float(np.trapezoid(val_norm * raw_val + ok_norm * _ok_oos_raw(W), W))
    return val_int, ok_int, val_norm, ok_norm, total_norm_grid


def oos_density(W_eV, s, material, include_kshell=True, fail_if_missing_k=True):
    W = np.asarray(W_eV, dtype=float)
    if W.size == 0:
        raise ValueError("Barkas OOS requires a nonempty W grid.")
    if np.any(~np.isfinite(W)) or np.any(W <= 0.0):
        raise ValueError("Barkas OOS requires finite positive W values.")
    if fail_if_missing_k and (not include_kshell) and np.nanmax(W) >= model.OXYGEN_K_B_EV:
        raise RuntimeError(
            "Barkas OOS normalization requires O K-shell strength when W reaches the O K edge."
        )
    val_int, ok_int, val_norm, ok_norm, total_norm_grid = _oos_norms(material, float(s.Ep), bool(include_kshell))
    val = val_norm * _valence_oos_raw(W, s)
    ok = ok_norm * _ok_oos_raw(W) if include_kshell else np.zeros_like(W)
    total = val + ok
    W_unique = np.unique(W[np.isfinite(W) & (W > 0.0)])
    val_unique = val_norm * _valence_oos_raw(W_unique, s)
    ok_unique = ok_norm * _ok_oos_raw(W_unique) if include_kshell else np.zeros_like(W_unique)
    total_unique = val_unique + ok_unique
    total_unique_int = float(np.trapezoid(total_unique, W_unique)) if W_unique.size > 1 else float("nan")
    return OOSDensity(
        W_eV=W,
        W_unique_eV=W_unique,
        df_dW_total=total,
        df_dW_valence=val,
        df_dW_OK=ok,
        df_dW_total_unique=total_unique,
        df_dW_valence_unique=val_unique,
        df_dW_OK_unique=ok_unique,
        valence_integral_raw=val_int,
        ok_integral_raw=ok_int,
        total_integral_norm_grid=total_norm_grid,
        total_integral_unique_grid=total_unique_int,
        valence_norm=val_norm,
        ok_norm=ok_norm,
    )


def barkas_unit_dcs_cm2_per_eV(T_total_eV, W_eV, projectile_mass_me, df_dW):
    T = np.asarray(T_total_eV, dtype=float)
    W = np.asarray(W_eV, dtype=float)
    if np.any(~np.isfinite(T)) or np.any(T <= 0.0):
        raise ValueError("Barkas DCS requires finite positive projectile kinetic energies.")
    if np.any(~np.isfinite(W)) or np.any(W <= 0.0):
        raise ValueError("Barkas DCS requires finite positive energy losses W.")
    beta, gamma = projectile_beta_gamma(T, projectile_mass_me)
    beta = np.maximum(beta, np.finfo(float).tiny)
    gamma2 = gamma * gamma
    xi = barkas_xi(W, T, projectile_mass_me)
    kernel = arbi1(xi) + arbi2(xi) / gamma2
    pref = 4.0 * np.pi * RE_CLASSICAL_CM**2 * ALPHA_FINE / (gamma2 * beta**5)
    out = pref * np.asarray(df_dW, dtype=float) * kernel
    if np.any(~np.isfinite(out)):
        raise FloatingPointError("Barkas DCS produced non-finite values.")
    return out


def barkas_dcs_m2_per_eV(T_total_eV, W_eV, z_int, projectile_mass_me, df_dW):
    return CM2_TO_M2 * np.asarray(z_int, dtype=float) ** 3 * barkas_unit_dcs_cm2_per_eV(
        T_total_eV, W_eV, projectile_mass_me, df_dW
    )


def integrate_by_T(T_line, W_line, values):
    T_line = np.asarray(T_line, dtype=float)
    W_line = np.asarray(W_line, dtype=float)
    values = np.asarray(values, dtype=float)
    unique_T = np.unique(T_line[np.isfinite(T_line)])
    out_T = []
    out = []
    for T in unique_T:
        idx = np.flatnonzero(T_line == T)
        if idx.size < 2:
            continue
        order = np.argsort(W_line[idx])
        W = W_line[idx][order]
        V = values[idx][order]
        out_T.append(float(T))
        out.append(float(np.trapezoid(V, W)))
    return np.asarray(out_T, dtype=float), np.asarray(out, dtype=float)


def born_reference_scale(z_int, nuclear_charge, born_reference_charge, q_reference=None):
    if born_reference_charge is None:
        raise ValueError("born_reference_charge is required.")
    ref = str(born_reference_charge).strip()
    if ref not in BORN_REFERENCE_CHOICES:
        choices = ", ".join(BORN_REFERENCE_CHOICES)
        raise ValueError(f"born_reference_charge must be one of: {choices}.")

    z_int = np.asarray(z_int, dtype=float)
    if ref == "bare_Z":
        q_ref = float(nuclear_charge)
    elif ref == "unit_charge":
        q_ref = 1.0
    else:
        if q_reference is None:
            raise ValueError("born_reference_charge='explicit_q' requires q_reference.")
        q_ref = float(q_reference)
    if not np.isfinite(q_ref) or q_ref <= 0.0:
        raise ValueError("Born reference charge must be finite and positive.")
    return np.square(z_int / q_ref), q_ref


def apply_barkas_correction_to_dcs_data(
    dcs_data,
    s,
    material,
    projectile_mass_me,
    nuclear_charge,
    charge_mode="bare",
    include_barkas_dcs=False,
    explicit_charge=None,
    include_kshell=True,
    dcs_scale_m2=1.0,
    born_reference_charge=None,
    born_reference_q=None,
    projectile_density=None,
    workers=None,
    checkpoint_dir=None,
):
    T_line = np.asarray(dcs_data["T_line"], dtype=float)
    W_line = np.asarray(dcs_data["E_line"], dtype=float)
    exc_vals = np.asarray(dcs_data["exc_vals"], dtype=float)
    ion_vals = np.asarray(dcs_data["ion_vals"], dtype=float)
    if exc_vals.ndim == 1:
        exc_vals = exc_vals.reshape(-1, 1)
    if ion_vals.ndim == 1:
        ion_vals = ion_vals.reshape(-1, 1)

    z_int = interaction_charge(
        T_line,
        nuclear_charge=nuclear_charge,
        projectile_mass_me=projectile_mass_me,
        charge_mode=charge_mode,
        explicit_charge=explicit_charge,
    )
    z0 = float(nuclear_charge)
    if z0 <= 0.0:
        raise ValueError("nuclear_charge must be positive.")
    if projectile_density is not None and (
        charge_mode != "bare" or born_reference_charge != "bare_Z"
        or projectile_density.z != z0
    ):
        raise ValueError("Frozen projectile screening cannot be combined with scalar charge rescaling.")
    born_scale, born_q_ref = born_reference_scale(
        z_int,
        nuclear_charge=z0,
        born_reference_charge=born_reference_charge,
        q_reference=born_reference_q,
    )

    exc_born_file = exc_vals * born_scale[:, None]
    ion_born_file = ion_vals * born_scale[:, None]
    born_total_m2 = dcs_scale_m2 * (np.sum(exc_born_file, axis=1) + np.sum(ion_born_file, axis=1))

    oos = oos_density(W_line, s, material=material, include_kshell=include_kshell)
    barkas_m2 = np.zeros_like(born_total_m2)
    quadrature_error = np.zeros_like(born_total_m2)
    model_metadata = {}
    screened = projectile_density is not None and projectile_density.electrons > 0
    if include_barkas_dcs and screened:
        from physics.inelastic_dielectric.polarization import screened_barkas
        from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import DEFAULT_WORKERS
        barkas_m2, quadrature_error = screened_barkas.dcs_m2_per_eV(
            T_line, W_line, projectile_mass_me, oos.df_dW_total, projectile_density,
            workers=DEFAULT_WORKERS if workers is None else workers,
            **({"checkpoint_dir": checkpoint_dir} if checkpoint_dir is not None else {}))
        model_metadata = screened_barkas.metadata(projectile_density)
        uncontrolled = (born_total_m2 > 0) & (np.abs(barkas_m2) >= born_total_m2)
        if np.any(uncontrolled):
            raise RuntimeError(
                "Screened Barkas correction is as large as the Born DCS; "
                "the additive correction is uncontrolled relative to this baseline at T/eV="
                + repr(np.unique(T_line[uncontrolled]).tolist())
                + ". No clipping or fitted scaling was applied.")
    elif include_barkas_dcs:
        barkas_m2 = barkas_dcs_m2_per_eV(
            T_line,
            W_line,
            z_int=z_int,
            projectile_mass_me=projectile_mass_me,
            df_dW=oos.df_dW_total,
        )
        Wmax = wmax_eV(T_line, projectile_mass_me)
        barkas_m2 = np.where(W_line <= Wmax, barkas_m2, 0.0)

    all_born_file = np.hstack([exc_born_file, ion_born_file])
    weights = np.where(all_born_file > 0.0, all_born_file, 0.0)
    weight_sum = np.sum(weights, axis=1)
    barkas_file = barkas_m2 / float(dcs_scale_m2)
    all_total_file = all_born_file.copy()
    good = weight_sum > 0.0
    all_total_file[good] += weights[good] * (barkas_file[good] / weight_sum[good])[:, None]
    fallback = (~good) & (barkas_file != 0.0)
    if np.any(fallback):
        target_col = exc_born_file.shape[1] if ion_born_file.shape[1] else 0
        all_total_file[fallback, target_col] += barkas_file[fallback]

    n_exc = exc_born_file.shape[1]
    exc_total_file = all_total_file[:, :n_exc]
    ion_total_file = all_total_file[:, n_exc:]
    total_m2 = dcs_scale_m2 * np.sum(all_total_file, axis=1)

    T_born, tcs_born = integrate_by_T(T_line, W_line, born_total_m2)
    T_bar, tcs_bar = integrate_by_T(T_line, W_line, barkas_m2)
    T_tot, tcs_tot = integrate_by_T(T_line, W_line, total_m2)
    _, s_bar = integrate_by_T(T_line, W_line, W_line * barkas_m2)
    if not (np.array_equal(T_born, T_bar) and np.array_equal(T_born, T_tot)):
        raise RuntimeError("TCS diagnostic energy grids are inconsistent.")
    channel_negative = np.any(all_total_file < 0.0, axis=1)
    channel_nonfinite = np.any(~np.isfinite(all_total_file), axis=1)
    total_nonfinite = ~np.isfinite(total_m2)
    unstable = total_nonfinite | (total_m2 < 0.0) | channel_negative | channel_nonfinite
    diag = DCSDiagnostics(
        T_line=T_line,
        W_line=W_line,
        z_int=z_int,
        DCS_Born_m2_per_eV=born_total_m2,
        DCS_Barkas_m2_per_eV=barkas_m2,
        DCS_total_m2_per_eV=total_m2,
        TCS_T_eV=T_tot,
        TCS_Born_m2=tcs_born,
        TCS_Barkas_m2=tcs_bar,
        TCS_total_m2=tcs_tot,
        S_Barkas_check_eV_m2=s_bar,
        negative_or_unstable_T_eV=np.unique(T_line[unstable]),
        negative_channel_T_eV=np.unique(T_line[channel_negative]),
        nonfinite_T_eV=np.unique(T_line[total_nonfinite | channel_nonfinite]),
        dcs_total_min_m2_per_eV=float(np.nanmin(total_m2)) if total_m2.size else float("nan"),
        dcs_total_max_m2_per_eV=float(np.nanmax(total_m2)) if total_m2.size else float("nan"),
        df_dW_total=oos.df_dW_total,
        df_dW_valence=oos.df_dW_valence,
        df_dW_OK=oos.df_dW_OK,
        df_dW_total_unique=oos.df_dW_total_unique,
        df_dW_valence_unique=oos.df_dW_valence_unique,
        df_dW_OK_unique=oos.df_dW_OK_unique,
        df_dW_integral=oos.total_integral_unique_grid,
        df_dW_integral_norm_grid=oos.total_integral_norm_grid,
        df_dW_integral_unique_grid=oos.total_integral_unique_grid,
        df_dW_valence_raw_integral=oos.valence_integral_raw,
        df_dW_OK_raw_integral=oos.ok_integral_raw,
        df_dW_valence_norm=oos.valence_norm,
        df_dW_OK_norm=oos.ok_norm,
        born_reference_charge=str(born_reference_charge),
        born_reference_q=float(born_q_ref),
        barkas_channel_distribution="bookkeeping_weighted_across_existing_channels",
        barkas_model_metadata=model_metadata,
        barkas_quadrature_error_m2_per_eV=quadrature_error,
    )
    corrected = dict(dcs_data)
    corrected["exc_vals"] = exc_total_file
    corrected["ion_vals"] = ion_total_file
    corrected["barkas_diagnostics"] = diag
    corrected["charge_mode"] = str(charge_mode)
    corrected["include_barkas_dcs"] = bool(include_barkas_dcs)
    corrected["explicit_charge"] = np.nan if explicit_charge is None else float(explicit_charge)
    corrected["born_reference_charge"] = str(born_reference_charge)
    corrected["born_reference_q"] = float(born_q_ref)
    corrected["barkas_channel_distribution"] = "bookkeeping_weighted_across_existing_channels"
    corrected["barkas_model_metadata"] = model_metadata
    return corrected, diag
