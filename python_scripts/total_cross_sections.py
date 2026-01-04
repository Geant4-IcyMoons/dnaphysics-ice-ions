#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import emfietzoglou_model_finite_q as model
from constants import EV_TO_HA, N, a0, mass


# ----------------------------------------------------------------------
# Plotting: full cross sections per channel vs T
# ----------------------------------------------------------------------
def plot_full_cross_sections_per_channel(
        T_list, sigma_list, s,
        ax=None, linewidth=2, alpha=0.9, figsize=(12, 8)):
    """
    Plot full (double-integrated) cross section per channel vs T.

    Parameters
    ----------
    T_list : list or array of T values
    sigma_list : list of dicts
        Output of integrate_elf_double_integral(s, C, T)
        computed for each T in T_list.
        Must have same length as T_list.
    s : IceOpticalSet
        For channel counts (excitations & ionizations).
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    T_arr = np.asarray(T_list)

    exc_colors = ["#1f77b4", "#2ca02c", "#17becf", "#8c564b", "#9467bd"]
    ion_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2", "#7f7f7f"]

    # ----------------- Excitations -----------------
    n_exc = len(s.excitations)

    print("Integrate EXCITATIONS")
    for k in tqdm(range(n_exc)):
        y = [sigma_list[i]["excitation_sigma"][k] for i in range(len(T_list))]
        ax.loglog(
            T_arr,
            y,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth,
            alpha=alpha,
            label=f"Exc {k+1}",
        )

    # ----------------- Ionizations -----------------
    n_ion = len(s.ionizations)

    print("Integrate IONIZATIONS")
    for j in tqdm(range(n_ion)):
        y = [sigma_list[i]["ionization_sigma"][j] for i in range(len(T_list))]
        ax.loglog(
            T_arr,
            y,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth,
            alpha=alpha,
            label=f"Ion {j+1}",
        )

    print("Integrate K-SHELL")
    # ----------------- K-shell -----------------
    if sigma_list[0]["kshell_sigma"] is not None:
        y = [sigma_list[i]["kshell_sigma"] for i in range(len(T_list))]
        ax.loglog(
            T_arr,
            y,
            color="0.3",
            ls="--",
            lw=linewidth,
            label="K-shell",
        )

    # ----------------- Totals -----------------
    y_valence = [sigma_list[i]["valence_sigma"] for i in range(len(T_list))]
    y_total = [sigma_list[i]["total_sigma"] for i in range(len(T_list))]

    ax.loglog(
        T_arr,
        y_valence,
        color="0.2",
        lw=2.4,
        ls=":",
        label="Valence total",
    )

    ax.loglog(
        T_arr,
        y_total,
        color="k",
        lw=3.0,
        label="Total cross section",
    )

    # ----------------- Labels -----------------
    ax.set_xlabel("Incident energy T (eV)")
    ax.set_ylabel("Total cross section σ(T)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    return ax


# ----------------------------------------------------------------------
# Helper: q-bounds for scalar Ei, Tj (eV)
# ----------------------------------------------------------------------
def _q_bounds_scalar(Ei, Tj, mass=1.0):
    """Return (q_lo, q_hi) for scalar Ei, Tj (both in eV)."""
    Ei_H = Ei * EV_TO_HA
    T_H = Tj * EV_TO_HA
    if Ei_H >= T_H:
        return 0.0, 0.0

    d = T_H - Ei_H
    if d <= 0.0:
        return 0.0, 0.0

    sqrtT = np.sqrt(T_H)
    sqrt_d = np.sqrt(d)
    qlo = np.sqrt(2.0 * mass) * (sqrtT - sqrt_d)
    qhi = np.sqrt(2.0 * mass) * (sqrtT + sqrt_d)

    if not np.isfinite(qlo) or not np.isfinite(qhi) or qhi <= qlo:
        return 0.0, 0.0
    return qlo, qhi


# ----------------------------------------------------------------------
# Inner q-integral at fixed Ei for channels (excitation / ionization)
# ----------------------------------------------------------------------
def _integrate_channel_single_E(Ei, Tj, idx, channel_type, s, C, Nq=400):
    """
    Compute inner integral over q:

        ∫ dq [ ELF_channel(Ei, q) / q ]

    for one excitation or ionization channel, at fixed Ei, Tj.
    """
    qlo, qhi = _q_bounds_scalar(Ei, Tj, mass)
    if qhi <= qlo:
        return 0.0

    # q-grid
    qvals = np.linspace(qlo, qhi, Nq)
    E_arr = np.array([Ei], float)

    # Vectorized dielectric functions at (Ei, qvals)
    e1 = model.epsilon1_valence_Eq(E_arr, qvals, s, C)
    e2 = model.epsilon2_valence_Eq(E_arr, qvals, s, C)

    e1t = e1["total"][:, 0]   # shape (Nq,)
    e2t = e2["total"][:, 0]
    denom = e1t**2 + e2t**2
    denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)

    if channel_type == "excitation":
        vals = e2["excitations"][idx][:, 0] / denom  # ELF_channel(Ei, q)
    elif channel_type == "ionization":
        vals = e2["ionizations"][idx][:, 0] / denom
    else:
        raise ValueError("channel_type must be 'excitation' or 'ionization'")

    dq = np.diff(qvals)
    prev = vals[:-1]
    cur = vals[1:]
    accum = np.sum(0.5 * dq * (prev + cur) / qvals[1:])

    int_cons = 1.0 / (np.pi * a0 * N * Tj)
    return float(int_cons * accum)


# ----------------------------------------------------------------------
# Inner q-integral at fixed Ei for K-shell channel
# ----------------------------------------------------------------------
def _integrate_kshell_single_E(Ei, Tj, s, Nq=400, include_kshell=True):
    """
    Inner integral over q for K-shell:
    """
    if not include_kshell or (s.kshell is None):
        return 0.0

    qlo, qhi = _q_bounds_scalar(Ei, Tj, mass)
    if qhi <= qlo:
        return 0.0

    # K-shell imaginary part at optical limit, with threshold gating
    ks_arr = model.epsilon2_Kshell_E0(np.array([Ei], float), s)
    ks_val = float(ks_arr[0])
    if ks_val == 0.0:
        return 0.0

    qvals = np.linspace(qlo, qhi, Nq)
    dq = np.diff(qvals)

    # ks_val is constant in q
    accum = np.sum(0.5 * dq * (ks_val + ks_val) / qvals[1:])  # == ks_val * sum(dq/q)

    int_cons = 1.0 / (np.pi * a0 * N * Tj)
    return float(int_cons * accum)


# ----------------------------------------------------------------------
# Main: q-integrated ELF per channel, on its own E-grid
# ----------------------------------------------------------------------
def integrate_elf_channels_per_channel_q(s, C, T, NE=400, Nq=400, include_kshell=True):
    """
    Integrate ELF(E,q)/q over q for each excitation, ionization, and K-shell.

    Energy bounds for each channel are determined as:

        Excitation k:
            Emin_k = E0_k           (osc.E0)
            Emax_k = T

        Ionization j:
            Emin_j = B_j            (osc.Bth)
            Emax_j = (T + B_j) / 2

        K-shell:
            Emin_K = B_K
            Emax_K = T

    Returns each channel evaluated on its own E grid:

        {
          "excitation_E":   [E_k arrays],
          "excitation_int": [vals_k arrays],
          "ionization_E":   [E_j arrays],
          "ionization_int": [vals_j arrays],
          "kshell_E":       E_K array or None,
          "kshell_int":     vals_K array or None,
        }
    """

    # --- Channel energy windows ---
    exc_Emin = np.array([osc.E0 for osc in s.excitations], float)
    exc_Emax = np.array([T for _ in s.excitations], float)

    ion_Emin = np.array([osc.Bth for osc in s.ionizations], float)
    ion_Emax = np.array([(T + osc.Bth) / 2.0 for osc in s.ionizations], float)

    if include_kshell and (s.kshell is not None):
        kshell_Emin = s.kshell.Bth
        kshell_Emax = T
    else:
        kshell_Emin = None
        kshell_Emax = None

    results = {
        "excitation_E": [],
        "excitation_int": [],
        "ionization_E": [],
        "ionization_int": [],
        "kshell_E": None,
        "kshell_int": None,
    }

    # ------------------- Excitations -------------------
    for k in range(len(s.excitations)):
        Emin, Emax = exc_Emin[k], exc_Emax[k]
        if Emin >= Emax:
            results["excitation_E"].append(np.array([], float))
            results["excitation_int"].append(np.array([], float))
            continue

        Egrid = np.linspace(Emin, Emax, NE)
        vals = np.empty_like(Egrid)

        for i, Ei in enumerate(Egrid):
            vals[i] = _integrate_channel_single_E(Ei, T, k, "excitation", s, C, Nq=Nq)

        results["excitation_E"].append(Egrid)
        results["excitation_int"].append(vals)

    # ------------------- Ionizations -------------------
    for j in range(len(s.ionizations)):
        Emin, Emax = ion_Emin[j], ion_Emax[j]
        if Emin >= Emax:
            results["ionization_E"].append(np.array([], float))
            results["ionization_int"].append(np.array([], float))
            continue

        Egrid = np.linspace(Emin, Emax, NE)
        vals = np.empty_like(Egrid)

        for i, Ei in enumerate(Egrid):
            vals[i] = _integrate_channel_single_E(Ei, T, j, "ionization", s, C, Nq=Nq)

        results["ionization_E"].append(Egrid)
        results["ionization_int"].append(vals)

    # ------------------- K-shell -----------------------
    if (kshell_Emin is not None) and (kshell_Emin < kshell_Emax):
        Egrid = np.linspace(kshell_Emin, kshell_Emax, NE)
        vals = np.empty_like(Egrid)

        for i, Ei in enumerate(Egrid):
            vals[i] = _integrate_kshell_single_E(Ei, T, s, Nq=Nq, include_kshell=include_kshell)

        results["kshell_E"] = Egrid
        results["kshell_int"] = vals

    return results


# ----------------------------------------------------------------------
# Full double integral over E and q: σ(T) per channel and totals
# ----------------------------------------------------------------------
def integrate_elf_double_integral(s, C, T,
                                  NE=400, Nq=400,
                                  include_kshell=True):
    """
    Compute full double integral:
        sigma(T) = ∫ dE ∫ dq [ELF(E,q)/q]
    for all channels.

    Returns:
        {
          "excitation_sigma": [values],   # per excitation band
          "ionization_sigma": [values],   # per ionization shell
          "kshell_sigma": value or None,
          "valence_sigma": scalar,
          "total_sigma": scalar
        }
    """

    # 1) Inner q-integrals as functions of E
    integ = integrate_elf_channels_per_channel_q(
        s, C, T=T, NE=NE, Nq=Nq,
        include_kshell=include_kshell
    )

    exc_sigma = []
    ion_sigma = []
    kshell_sigma = None

    # ------------- Excitations -------------
    for k in range(len(s.excitations)):
        E_k = integ["excitation_E"][k]
        y_k = integ["excitation_int"][k]

        if E_k.size == 0:
            exc_sigma.append(0.0)
        else:
            exc_sigma.append(float(np.trapezoid(y_k, E_k)))

    # ------------- Ionizations -------------
    for j in range(len(s.ionizations)):
        E_j = integ["ionization_E"][j]
        y_j = integ["ionization_int"][j]

        if E_j.size == 0:
            ion_sigma.append(0.0)
        else:
            ion_sigma.append(float(np.trapezoid(y_j, E_j)))

    # ------------- K-shell -------------
    if integ["kshell_E"] is not None:
        E_K = integ["kshell_E"]
        y_K = integ["kshell_int"]

        if E_K.size == 0:
            kshell_sigma = 0.0
        else:
            kshell_sigma = float(np.trapezoid(y_K, E_K))

    # ------------- Totals -------------
    valence_sigma = float(np.sum(exc_sigma) + np.sum(ion_sigma))

    if kshell_sigma is not None:
        total_sigma = valence_sigma + kshell_sigma
    else:
        total_sigma = valence_sigma

    return {
        "excitation_sigma": exc_sigma,
        "ionization_sigma": ion_sigma,
        "kshell_sigma": kshell_sigma,
        "valence_sigma": valence_sigma,
        "total_sigma": total_sigma,
    }


# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------
def main():
    # Optical model / dispersion coefficients
    s = model.epsilon_optical('amorphous')
    a_vec = np.array([3.82, 2.47, 2.47, 3.01, 2.44])
    b_vec = np.array([0.0272, 0.0295, 0.0311, 0.0111, 0.0633])
    c_vec = np.array([0.098, 0.075, 0.074, 0.765, 0.425])

    # RR2017 defaults for c_disp, d_disp, b1, b2
    C = model.DispersionCoeffs(a_fj=a_vec, b_fj=b_vec, c_fj=c_vec)

    # Energy grid (eV)
    # NOTE: original snippet used logspace(1, 6, 20)
    T_list = np.logspace(1, 6, 20)

    sigma_list = []
    print("Computing double-integrated cross sections...")
    for T in tqdm(T_list):
        sigma = integrate_elf_double_integral(
            s, C, T,
            NE=10, Nq=10,
            include_kshell=True
        )
        sigma_list.append(sigma)

    # ----------------- Plot and save figure -----------------
    fig, ax = plt.subplots(figsize=(14, 9))
    plot_full_cross_sections_per_channel(T_list, sigma_list, s, ax=ax)

    ax.set_title("Full (q,E) Double-Integrated Cross Sections Per Channel")
    fig.tight_layout()

    fig.savefig("cross_sections_per_channel.png", dpi=300)
    print("Saved plot to cross_sections_per_channel.png")

    plt.close(fig)

    # ----------------- Build and save .dat file -------------
    T_arr = np.asarray(T_list, dtype=float)
    n_exc = len(s.excitations)
    n_ion = len(s.ionizations)

    n_rows = len(T_arr)
    n_cols = 1 + n_exc + n_ion  # T + all excitations + all ionizations

    data = np.zeros((n_rows, n_cols), dtype=float)
    data[:, 0] = T_arr

    for i in range(n_rows):
        exc = np.asarray(sigma_list[i]["excitation_sigma"], dtype=float)
        ion = np.asarray(sigma_list[i]["ionization_sigma"], dtype=float)

        data[i, 1:1 + n_exc] = exc
        data[i, 1 + n_exc:1 + n_exc + n_ion] = ion

    header_cols = (
        ["T_eV"] +
        [f"exc_{k+1}" for k in range(n_exc)] +
        [f"ion_{j+1}" for j in range(n_ion)]
    )
    header = " ".join(header_cols)

    np.savetxt(
        "cross_sections_per_channel.dat",
        data,
        header=header,
        comments="",
    )
    print("Saved data to cross_sections_per_channel.dat")
    print("Columns:", header)


if __name__ == "__main__":
    main()
