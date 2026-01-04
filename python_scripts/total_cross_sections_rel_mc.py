#!/usr/bin/env python3

import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import (
    C_AU,
    EH,
    EV_TO_HA,
    MC2_HA,
    MC2_eV,
    MC_T_THRESHOLD_eV,
    N,
    REL_T_THRESHOLD_eV,
    TRANS_T_THRESHOLD_eV,
    a0,
    mass,
)
import emfietzoglou_model_finite_q as model

# ----------------------------------------------------------------------
# Constants for integration (from constants.py)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Helpers: Relativistic corrections
# ----------------------------------------------------------------------
def beta2_rel(Tj):
    # Use exactly the form you specified:
    return 1.0 - 1.0 / (Tj / MC2_eV + 1.0)**2

def Q_q(q_au):
    """
    Q(q) in eV.
    """
    q_au = np.asarray(q_au, dtype=float)
    Q_Ha = np.sqrt((C_AU * q_au)**2 + (MC2_HA)**2) - MC2_HA
    return Q_Ha * EH

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
    return float(qlo), float(qhi)

def _q_bounds_scalar_rel(Ei, Tj):
    if Ei >= Tj:
        return 0.0, 0.0

    Ei_H = Ei * EV_TO_HA
    T_H  = Tj * EV_TO_HA
    d = T_H - Ei_H
    if d <= 0.0:
        return 0.0, 0.0

    term1 = np.sqrt(T_H * (T_H + 2.0 * MC2_HA))
    term2 = np.sqrt(d   * (d   + 2.0 * MC2_HA))
    qlo = (term1 - term2) / C_AU
    qhi = (term1 + term2) / C_AU
    if (not np.isfinite(qlo)) or (not np.isfinite(qhi)) or (qhi <= qlo):
        return 0.0, 0.0
    return float(qlo), float(qhi)

# ----------------------------------------------------------------------
# Inner q-integral at fixed Ei for channels (excitation / ionization)
# ----------------------------------------------------------------------
def _integrate_channel_single_E(Ei, Tj, idx, channel_type, s, C, Nq=400):
    """
    Compute inner integral over q:

        ∫ dq [ ELF_channel(Ei, q) / q ]

    for one excitation or ionization channel, at fixed Ei, Tj.
    """
    qlo, qhi = _q_bounds_scalar_rel(Ei, Tj)
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
        vals = e2["excitations"][idx][:, 0] / denom
    elif channel_type == "ionization":
        vals = e2["ionizations"][idx][:, 0] / denom
    else:
        raise ValueError("channel_type must be 'excitation' or 'ionization'")

    integrand = vals
    dq = np.diff(qvals)

    # Use mid/edge trapezoid form and divide by qvals[1:], avoids small q-value infinities
    accum = float(np.sum(0.5 * dq * (integrand[:-1] + integrand[1:]) / qvals[1:]))

    int_cons = 1.0 / (np.pi * a0 * N * Tj)
    return float(int_cons * accum)

def _integrate_channel_single_E_rel(Ei, Tj, idx, channel_type, s, C, Nq=400):
    # Only defined/used above threshold (or return 0 below)
    if Tj < REL_T_THRESHOLD_eV:
        return 0.0

    qlo, qhi = _q_bounds_scalar_rel(Ei, Tj)
    if qhi <= qlo:
        return 0.0

    qvals = np.linspace(qlo, qhi, Nq)
    E_arr = np.array([Ei], float)

    e1 = model.epsilon1_valence_Eq(E_arr, qvals, s, C)
    e2 = model.epsilon2_valence_Eq(E_arr, qvals, s, C)

    e1t = e1["total"][:, 0]
    e2t = e2["total"][:, 0]
    denom = e1t**2 + e2t**2
    denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)

    if channel_type == "excitation":
        vals = e2["excitations"][idx][:, 0] / denom
    else:
        vals = e2["ionizations"][idx][:, 0] / denom

    # Q(q) in eV
    Q_eV = Q_q(qvals)
    Q_eV = np.where(Q_eV == 0.0, np.finfo(float).tiny, Q_eV)

    factor1 = (C_AU**2 * qvals) / np.sqrt((C_AU * qvals)**2 + (MC2_HA**2))
    factor1 *= EH  # dQ/dq in eV per a0^-1
    factor2 = (1.0 + Q_eV / MC2_eV) / (1.0 + Q_eV / (2.0 * MC2_eV))
    factor3 = 1 / Q_eV
    kernel = factor1 * factor2 * factor3

    integrand = vals * kernel
    accum = float(np.trapezoid(integrand, qvals))

    b2 = beta2_rel(Tj)
    b2 = max(b2, np.finfo(float).tiny)

    int_cons = 1.0 / (np.pi * a0 * N * MC2_eV * b2)
    return float(int_cons * accum)

def _integrate_channel_single_E_trans(Ei, Tj, idx, channel_type, s, C, Nq=0):
    """ Transverse RPWBA term (Fano approximation at q=0)"""
    if Tj < TRANS_T_THRESHOLD_eV:
        return 0.0

    # beta^2 and transverse bracket
    b2 = beta2_rel(Tj)
    b2 = max(b2, np.finfo(float).tiny)
    bracket = np.log(1.0 / max(1.0 - b2, np.finfo(float).tiny)) - b2

    # Optical (q=0) channel-resolved ELF via epsilon2_channel / (epsilon1_total^2 + epsilon2_total^2)
    E_arr = np.array([Ei], float)
    e1 = model.epsilon1_valence_E0(E_arr, s)
    e2 = model.epsilon2_valence_E0(E_arr, s)

    e1t = e1["total"]
    e2t = e2["total"]
    denom = e1t**2 + e2t**2
    denom = np.where(denom == 0.0, np.finfo(float).tiny, denom)

    if channel_type == "excitation":
        elf0 = e2["excitations"][idx] / denom
    elif channel_type == "ionization":
        elf0 = e2["ionizations"][idx] / denom
    else:
        raise ValueError("channel_type must be 'excitation' or 'ionization'")

    elf0 = float(np.asarray(elf0).ravel()[0])

    int_cons = 1.0 / (np.pi * a0 * N * MC2_eV * b2)
    return float(int_cons * elf0 * bracket)

# ----------------------------------------------------------------------
# Low-energy Mott–Coulomb (MC) corrections using PWBA kernel evaluations
# ----------------------------------------------------------------------
def _dsigma_pwba_dE(Ei, Tj, idx, channel_type, s, C, Nq=400, use_rel=False):
    """
    Return the differential cross section d sigma/dE at (Ei,Tj) for one channel.
    If use_rel=True (and Tj >= REL_T_THRESHOLD_eV), this uses the longitudinal relativistic
    kernel; otherwise it uses the nonrelativistic PWBA kernel.
    """
    if use_rel and (float(Tj) >= float(REL_T_THRESHOLD_eV)):
        return _integrate_channel_single_E_rel(Ei, Tj, idx, channel_type, s, C, Nq=Nq)
    return _integrate_channel_single_E(Ei, Tj, idx, channel_type, s, C, Nq=Nq)

def _dsigma_mc_ionization_dE(Ei, Tj, j, s, C, Nq=400, use_rel=False):
    """Mott–Coulomb exchange-style correction for *ionization* shells. """
    osc = s.ionizations[j]
    B = float(getattr(osc, "Bth"))
    U = float(getattr(osc, "U"))

    Tprime = float(Tj + B + U)
    E2 = float(Tj + 2.0*B + U - Ei)

    a = _dsigma_pwba_dE(Ei, Tprime, j, "ionization", s, C, Nq=Nq, use_rel=use_rel)
    b = _dsigma_pwba_dE(E2, Tprime, j, "ionization", s, C, Nq=Nq, use_rel=use_rel)

    # Guard against tiny negative numerical noise
    a = max(a, 0.0)
    b = max(b, 0.0)

    return float(a + b - np.sqrt(a*b))

def _sigma_pwba_excitation_shifted_T(s, C, Tshift, Tj, k, NE=400, Nq=400, use_rel=False):
    """sigma_PWBA for excitation k, with shifted kernel but kinematic E-window from Tj."""
    Emin = float(s.excitations[k].Bth)
    Emax = float(Tj)
    if Emin >= Emax:
        return 0.0

    Egrid = np.linspace(Emin, Emax, NE)
    vals = np.empty_like(Egrid)
    for i, Ei in enumerate(Egrid):
        vals[i] = _dsigma_pwba_dE(Ei, Tshift, k, "excitation", s, C, Nq=Nq, use_rel=use_rel)

    vals = np.where(vals < 0.0, 0.0, vals)
    return float(np.trapezoid(vals, Egrid))

def _sigma_mc_ionization(s, C, Tj, j, NE=400, Nq=400, use_rel=False):
    """sigma_MC(T) for ionization shell j: integrate d sigma_MC/dE over E."""
    osc = s.ionizations[j]
    B = float(getattr(osc, "Bth"))
    U = float(getattr(osc, "U"))

    Emin = B
    Emax = 0.5 * (Tj + B)
    if Emin >= Emax:
        return 0.0

    Egrid = np.linspace(Emin, Emax, NE)
    vals = np.empty_like(Egrid)
    for i, Ei in enumerate(Egrid):
        vals[i] = _dsigma_mc_ionization_dE(Ei, Tj, j, s, C, Nq=Nq, use_rel=use_rel)

    vals = np.where(vals < 0.0, 0.0, vals)
    return float(np.trapezoid(vals, Egrid))

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

    ks_arr = model.epsilon2_Kshell_E0(np.array([Ei], float), s)
    ks_val = float(ks_arr[0])
    if ks_val == 0.0:
        return 0.0

    qvals = np.linspace(qlo, qhi, Nq)

    dq = np.diff(qvals)
    accum = float(np.sum(0.5 * dq * (ks_val + ks_val) / qvals[1:]))

    int_cons = 1.0 / (np.pi * a0 * N * Tj)
    return float(int_cons * accum)

def _integrate_kshell_single_E_rel(Ei, Tj, s, Nq=400, include_kshell=True):
    """
    Inner integral over q for K-shell:
    """
    if not include_kshell or (s.kshell is None):
        return 0.0

    qlo, qhi = _q_bounds_scalar_rel(Ei, Tj)
    if qhi <= qlo:
        return 0.0

    # K-shell imaginary part at optical limit, with threshold gating
    ks_arr = model.epsilon2_Kshell_E0(np.array([Ei], float), s)
    ks_val = float(ks_arr[0])
    if ks_val == 0.0:
        return 0.0

    qvals = np.linspace(qlo, qhi, Nq)

    # Q(q) in eV
    Q_eV = Q_q(qvals)
    Q_eV = np.where(Q_eV == 0.0, np.finfo(float).tiny, Q_eV)

    factor1 = (C_AU**2 * qvals) / np.sqrt((C_AU * qvals) ** 2 + (MC2_HA ** 2))
    factor1 *= EH  # dQ/dq in eV per a0^-1
    factor2 = (1.0 + Q_eV / MC2_eV) / (1.0 + Q_eV / (2.0 * MC2_eV))
    factor3 = 1 / Q_eV
    kernel = factor1 * factor2 * factor3

    integrand = ks_val * kernel
    accum = float(np.trapezoid(integrand, qvals))

    b2 = beta2_rel(Tj)
    b2 = max(b2, np.finfo(float).tiny)

    int_cons = 1.0 / (np.pi * a0 * N * MC2_eV * b2)
    return float(int_cons * accum)

# ----------------------------------------------------------------------
# Q-integrated ELF per channel, on its own E-grid
# ----------------------------------------------------------------------
def integrate_elf_channels_per_channel_q(s, C, T, NE=400, Nq=400, include_kshell=True):
    """
    Integrate ELF(E,q)/q over q for each excitation, ionization, and K-shell,
    and ALSO compute a second "relativistic longitudinal-corrected" version
    of the same inner-q integrals (stored with *_rel keys).
    """

    # --- Channel energy windows ---
    exc_Emin = np.array([osc.Bth for osc in s.excitations], float)
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
        "excitation_int_rel": [],
        "excitation_int_rel_trans": [],

        "ionization_E": [],
        "ionization_int": [],
        "ionization_int_rel": [],
        "ionization_int_rel_trans": [],

        "kshell_E": None,
        "kshell_int": None,
        "kshell_int_rel": None,
    }

    # ------------------- Excitations -------------------
    for k in range(len(s.excitations)):
        Emin, Emax = exc_Emin[k], exc_Emax[k]
        if Emin >= Emax:
            results["excitation_E"].append(np.array([], float))
            results["excitation_int"].append(np.array([], float))
            results["excitation_int_rel"].append(np.array([], float))
            continue

        Egrid = np.linspace(Emin, Emax, NE)
        vals = np.empty_like(Egrid)
        vals_rel = np.empty_like(Egrid)
        vals_rel_trans = np.empty_like(Egrid)

        for i, Ei in enumerate(Egrid):
            # Nonrelativistic inner-q integral
            vals[i] = _integrate_channel_single_E(Ei, T, k, "excitation", s, C, Nq=Nq)

            # Relativistic-longitudinal corrected inner-q integral
            vals_rel[i] = _integrate_channel_single_E_rel(Ei, T, k, "excitation", s, C, Nq=Nq)

            vals_rel_trans[i] = _integrate_channel_single_E_trans(Ei, T, k, "excitation", s, C)


        results["excitation_E"].append(Egrid)
        results["excitation_int"].append(vals)
        results["excitation_int_rel"].append(vals_rel)
        results["excitation_int_rel_trans"].append(vals_rel_trans)

    # ------------------- Ionizations -------------------
    for j in range(len(s.ionizations)):
        Emin, Emax = ion_Emin[j], ion_Emax[j]
        if Emin >= Emax:
            results["ionization_E"].append(np.array([], float))
            results["ionization_int"].append(np.array([], float))
            results["ionization_int_rel"].append(np.array([], float))
            continue

        Egrid = np.linspace(Emin, Emax, NE)
        vals = np.empty_like(Egrid)
        vals_rel = np.empty_like(Egrid)
        vals_rel_trans = np.empty_like(Egrid)

        for i, Ei in enumerate(Egrid):
            # Nonrelativistic inner-q integral
            vals[i] = _integrate_channel_single_E(Ei, T, j, "ionization", s, C, Nq=Nq)

            # Relativistic-longitudinal corrected inner-q integral
            vals_rel[i] = _integrate_channel_single_E_rel(Ei, T, j, "ionization", s, C, Nq=Nq)

            vals_rel_trans[i] = _integrate_channel_single_E_trans(Ei, T, j, "ionization", s, C)


        results["ionization_E"].append(Egrid)
        results["ionization_int"].append(vals)
        results["ionization_int_rel"].append(vals_rel)
        results["ionization_int_rel_trans"].append(vals_rel_trans)

    # ------------------- K-shell -----------------------
    if (kshell_Emin is not None) and (kshell_Emin < kshell_Emax):
        Egrid = np.linspace(kshell_Emin, kshell_Emax, NE)
        vals = np.empty_like(Egrid)
        vals_rel = np.empty_like(Egrid)

        for i, Ei in enumerate(Egrid):
            # Nonrelativistic inner-q integral
            vals[i] = _integrate_kshell_single_E(
                Ei, T, s, Nq=Nq, include_kshell=include_kshell
            )

            # Relativistic-longitudinal corrected inner-q integral
            vals_rel[i] = _integrate_kshell_single_E_rel(
                Ei, T, s, Nq=Nq, include_kshell=include_kshell
            )

        results["kshell_E"] = Egrid
        results["kshell_int"] = vals
        results["kshell_int_rel"] = vals_rel

    return results

# ----------------------------------------------------------------------
# Full double integral over E and q: sigma(T) per channel and totals
# ----------------------------------------------------------------------
def integrate_elf_double_integral(s, C, T,
                                  NE=1000, Nq=1000,
                                  include_kshell=True,
                                  use_mott_coulomb=False,
                                  mc_T_max_eV=MC_T_THRESHOLD_eV):
    """
    Compute full double integral sigma(T) per channel and totals.
    """

    # 1) Inner q-integrals as functions of E
    integ = integrate_elf_channels_per_channel_q(
        s, C, T=T, NE=NE, Nq=Nq,
        include_kshell=include_kshell
    )

    # -------------------- Nonrelativistic --------------------
    exc_sigma = []
    ion_sigma = []
    kshell_sigma = None

    #Activate Mott-Colomb Corrections
    use_mc_here = bool(use_mott_coulomb) and (float(T) <= float(mc_T_max_eV))
    use_rel_in_mc = bool(use_mc_here) and (float(T) >= float(REL_T_THRESHOLD_eV))

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


    # -------------------- Low-energy Mott–Coulomb (optional) -------------
    exc_sigma_mc = None
    ion_sigma_mc = None
    valence_sigma_mc = None
    total_sigma_mc = None

    if use_mc_here:
        # Excitations: sigma_MC(T) = sum_k sigma_PWBA(T + 2*B_k)
        exc_sigma_mc = []
        for k in range(len(s.excitations)):
            Bk = float(s.excitations[k].Bth)
            Tshift = float(T + 2.0 * Bk)
            exc_sigma_mc.append(_sigma_pwba_excitation_shifted_T(s, C, Tshift, T, k, NE=NE, Nq=Nq, use_rel=use_rel_in_mc))

        # Ionizations: integrate the specified MC differential form
        ion_sigma_mc = []
        for j in range(len(s.ionizations)):
            ion_sigma_mc.append(_sigma_mc_ionization(s, C, T, j, NE=NE, Nq=Nq, use_rel=use_rel_in_mc))

        valence_sigma_mc = float(np.sum(exc_sigma_mc) + np.sum(ion_sigma_mc))
        if kshell_sigma is not None:
            total_sigma_mc = valence_sigma_mc + kshell_sigma
        else:
            total_sigma_mc = valence_sigma_mc

    elif use_mott_coulomb and (not use_mc_here):
        # Above the MC validity range: fall back to PWBA results
        exc_sigma_mc = exc_sigma
        ion_sigma_mc = ion_sigma
        valence_sigma_mc = valence_sigma
        total_sigma_mc = total_sigma

    # Keep raw PWBA results for debugging/comparison
    exc_sigma_pwba = list(exc_sigma)
    ion_sigma_pwba = list(ion_sigma)
    valence_sigma_pwba = float(valence_sigma)
    total_sigma_pwba = float(total_sigma)

    if use_mc_here and (exc_sigma_mc is not None) and (ion_sigma_mc is not None):
        exc_sigma = list(exc_sigma_mc)
        ion_sigma = list(ion_sigma_mc)
        valence_sigma = float(valence_sigma_mc)
        total_sigma = float(total_sigma_mc)

    # -------------------- Relativistic-longitudinal (additional) --------
    exc_sigma_rel = []
    exc_sigma_rel_trans = []

    ion_sigma_rel = []
    ion_sigma_rel_trans = []

    kshell_sigma_rel = None

    # ------------- Excitations (REL) -------------
    for k in range(len(s.excitations)):
        E_k = integ["excitation_E"][k]
        y_k_rel = integ.get("excitation_int_rel", [])[k] if "excitation_int_rel" in integ else None

        if (E_k.size == 0) or (y_k_rel is None) or (len(y_k_rel) == 0):
            exc_sigma_rel.append(0.0)
            exc_sigma_rel_trans.append(0.0)
        else:
            exc_sigma_rel.append(float(np.trapezoid(y_k_rel, E_k)))
            y_k_trans = integ.get("excitation_int_rel_trans", [])[k] if "excitation_int_rel_trans" in integ else None
            if (y_k_trans is None) or (len(y_k_trans) == 0):
                exc_sigma_rel_trans.append(0.0)
            else:
                exc_sigma_rel_trans.append(float(np.trapezoid(y_k_trans, E_k)))

    # ------------- Ionizations (REL) -------------
    for j in range(len(s.ionizations)):
        E_j = integ["ionization_E"][j]
        y_j_rel = integ.get("ionization_int_rel", [])[j] if "ionization_int_rel" in integ else None

        if (E_j.size == 0) or (y_j_rel is None) or (len(y_j_rel) == 0):
            ion_sigma_rel.append(0.0)
            ion_sigma_rel_trans.append(0.0)
        else:
            ion_sigma_rel.append(float(np.trapezoid(y_j_rel, E_j)))
            y_j_trans = integ.get("ionization_int_rel_trans", [])[j] if "ionization_int_rel_trans" in integ else None
            if (y_j_trans is None) or (len(y_j_trans) == 0):
                ion_sigma_rel_trans.append(0.0)
            else:
                ion_sigma_rel_trans.append(float(np.trapezoid(y_j_trans, E_j)))


    # ------------- K-shell (REL) -------------
    if integ["kshell_E"] is not None:
        E_K = integ["kshell_E"]
        y_K_rel = integ.get("kshell_int_rel", None)

        if (y_K_rel is None) or (E_K.size == 0):
            kshell_sigma_rel = 0.0
        else:
            kshell_sigma_rel = float(np.trapezoid(y_K_rel, E_K))

    # ------------- Totals (REL) -------------
    valence_sigma_rel = float(np.sum(exc_sigma_rel) + np.sum(ion_sigma_rel))
    valence_sigma_rel_trans = float(np.sum(exc_sigma_rel_trans) + np.sum(ion_sigma_rel_trans))

    if kshell_sigma_rel is not None:
        total_sigma_rel = valence_sigma_rel + kshell_sigma_rel
    else:
        total_sigma_rel = valence_sigma_rel

    # Transverse totals (K-shell transverse not included)
    if kshell_sigma_rel is not None:
        total_sigma_rel_trans = valence_sigma_rel_trans  # keep kshell separate
    else:
        total_sigma_rel_trans = valence_sigma_rel_trans

    total_sigma_rel_total = total_sigma_rel + total_sigma_rel_trans

    if float(T) >= float(mc_T_max_eV):
        exc_sigma = list(exc_sigma_rel)  # longitudinal only
        ion_sigma = list(ion_sigma_rel)  # longitudinal only
        valence_sigma = float(np.sum(exc_sigma) + np.sum(ion_sigma))
        if kshell_sigma is not None:
            total_sigma = valence_sigma + float(kshell_sigma)
        else:
            total_sigma = valence_sigma

    # Optional convenience: combined
    total_sigma_plus_rel = total_sigma_pwba + total_sigma_rel  # Long Only (PWBA + long)
    total_sigma_plus_rel_total = total_sigma_pwba + total_sigma_rel_total  # (PWBA + long + trans)

    return {
        # PWBA Baseline
        "excitation_sigma_pwba": exc_sigma_pwba,
        "ionization_sigma_pwba": ion_sigma_pwba,
        "valence_sigma_pwba": valence_sigma_pwba,
        "total_sigma_pwba": total_sigma_pwba,

        "excitation_sigma": exc_sigma,
        "ionization_sigma": ion_sigma,
        "kshell_sigma": kshell_sigma,
        "valence_sigma": valence_sigma,
        "total_sigma": total_sigma,

        # Low-energy Mott–Coulomb
        "excitation_sigma_mc": exc_sigma_mc,
        "ionization_sigma_mc": ion_sigma_mc,
        "valence_sigma_mc": valence_sigma_mc,
        "total_sigma_mc": total_sigma_mc,

        # Longitudinal relativistic outputs
        "excitation_sigma_rel": exc_sigma_rel,
        "excitation_sigma_rel_trans": exc_sigma_rel_trans,

        "ionization_sigma_rel": ion_sigma_rel,
        "ionization_sigma_rel_trans": ion_sigma_rel_trans,

        "kshell_sigma_rel": kshell_sigma_rel,
        "valence_sigma_rel": valence_sigma_rel,
        "valence_sigma_rel_trans": valence_sigma_rel_trans,

        "total_sigma_rel": total_sigma_rel,
        "total_sigma_rel_trans": total_sigma_rel_trans,
        "total_sigma_rel_total": total_sigma_rel_total,

        "total_sigma_plus_rel": total_sigma_plus_rel,
        "total_sigma_plus_rel_total": total_sigma_plus_rel_total,
    }

# ----------------------------------------------------------------------
# Plotting: full cross sections per channel vs T
# ----------------------------------------------------------------------
def plot_full_cross_sections_per_channel(
        T_list, sigma_list, s,
        ax=None, linewidth=2, alpha=0.9, figsize=(12, 8),
        mc_T_max_eV=1.0e5,
        rel_T_min_eV=REL_T_THRESHOLD_eV):
    """
    Plot a *comparison* of PWBA vs the default model (per channel)

    Baseline (PWBA):
        uses keys '*_sigma_pwba' computed by the integrator.

    Default model :
        - T < rel_T_min_eV (default 1 keV): Mott–Coulomb (MC) using PWBA kernel
        - rel_T_min_eV <= T < mc_T_max_eV (default 100 keV): MC using longitudinal-relativistic kernel
        - T >= mc_T_max_eV: longitudinal-relativistic replacement + transverse relativistic add-on
    """

    # Allow passing (ax_ion, ax_exc) or None
    if ax is None:
        fig_ion, ax_ion = plt.subplots(figsize=figsize)
        fig_exc, ax_exc = plt.subplots(figsize=figsize)
    else:
        ax_ion, ax_exc = ax

    T_arr = np.asarray(T_list, dtype=float)
    nT = len(T_arr)

    exc_colors = ["#1f77b4", "#2ca02c", "#17becf", "#8c564b", "#9467bd"]
    ion_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2", "#7f7f7f"]

    def _get_list(i, key, default):
        return sigma_list[i].get(key, default)

    # ----------------- Ionizations: PWBA vs Default model -----------------
    n_ion = len(s.ionizations)
    for j in range(n_ion):
        y_pwba = []
        y_model = []
        for i in range(nT):
            Tj = float(T_arr[i])

            pwba_list = _get_list(i, "ionization_sigma_pwba", [0.0] * n_ion)
            pwba = pwba_list[j] if j < len(pwba_list) else 0.0

            # Default model logic by energy range
            if Tj < float(rel_T_min_eV):
                mc_list = _get_list(i, "ionization_sigma_mc", None)
                model = (mc_list[j] if (mc_list is not None and j < len(mc_list)) else pwba)
            elif Tj < float(mc_T_max_eV):
                mc_list = _get_list(i, "ionization_sigma_mc", None)
                model = (mc_list[j] if (mc_list is not None and j < len(mc_list)) else pwba)
            else:
                relL_list = _get_list(i, "ionization_sigma_rel", [0.0] * n_ion)
                relT_list = _get_list(i, "ionization_sigma_rel_trans", [0.0] * n_ion)
                relL = relL_list[j] if j < len(relL_list) else 0.0
                relT = relT_list[j] if j < len(relT_list) else 0.0
                model = relL + relT

            y_pwba.append(pwba)
            y_model.append(model)

        ax_ion.loglog(
            T_arr, y_pwba,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth, alpha=alpha,
            ls="--",
            label=f"Ion {j+1} PWBA",
        )
        ax_ion.loglog(
            T_arr, y_model,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth, alpha=alpha,
            ls="-",
            label=f"Ion {j+1} Default model",
        )

    ax_ion.set_xlabel("Incident energy T (eV)")
    ax_ion.set_ylabel("Cross section sigma(T)")
    ax_ion.set_title("Ionizations: PWBA baseline vs Default model")
    ax_ion.grid(True, which="both", ls="--", alpha=0.3)
    ax_ion.legend(loc="best", fontsize=8)

    # ----------------- Excitations: PWBA vs Default model -----------------
    n_exc = len(s.excitations)
    for k in range(n_exc):
        y_pwba = []
        y_model = []
        for i in range(nT):
            Tj = float(T_arr[i])

            pwba_list = _get_list(i, "excitation_sigma_pwba", [0.0] * n_exc)
            pwba = pwba_list[k] if k < len(pwba_list) else 0.0

            if Tj < float(rel_T_min_eV):
                mc_list = _get_list(i, "excitation_sigma_mc", None)
                model = (mc_list[k] if (mc_list is not None and k < len(mc_list)) else pwba)
            elif Tj < float(mc_T_max_eV):
                mc_list = _get_list(i, "excitation_sigma_mc", None)
                model = (mc_list[k] if (mc_list is not None and k < len(mc_list)) else pwba)
            else:
                relL_list = _get_list(i, "excitation_sigma_rel", [0.0] * n_exc)
                relT_list = _get_list(i, "excitation_sigma_rel_trans", [0.0] * n_exc)
                relL = relL_list[k] if k < len(relL_list) else 0.0
                relT = relT_list[k] if k < len(relT_list) else 0.0
                model = relL + relT

            y_pwba.append(pwba)
            y_model.append(model)

        ax_exc.loglog(
            T_arr, y_pwba,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth, alpha=alpha,
            ls="--",
            label=f"Exc {k+1} PWBA",
        )
        ax_exc.loglog(
            T_arr, y_model,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth, alpha=alpha,
            ls="-",
            label=f"Exc {k+1} Default model",
        )

    ax_exc.set_xlabel("Incident energy T (eV)")
    ax_exc.set_ylabel("Cross section sigma(T)")
    ax_exc.set_title("Excitations: PWBA baseline vs Default model")
    ax_exc.grid(True, which="both", ls="--", alpha=0.3)
    ax_exc.legend(loc="best", fontsize=8)

    return (ax_ion, ax_exc)


def plot_relativistic_component_per_channel(
        T_list, sigma_list, s,
        ax=None, linewidth=2, alpha=0.9, figsize=(12, 8),
        include_kshell=False, include_totals=False):
    """ Make two plots:
      (1) Ionizations: longitudinal (solid) and transverse (dashed)
      (2) Excitations: longitudinal (solid) and transverse (dashed)

    Uses keys:
      - ionization_sigma_rel (longitudinal)
      - ionization_sigma_rel_trans (transverse)
      - excitation_sigma_rel (longitudinal)
      - excitation_sigma_rel_trans (transverse)

    Plots only for T >= 100 keV by default.
    """

    T_arr = np.asarray(T_list, dtype=float)
    mask = T_arr >= 1.0e5
    if not np.any(mask):
        raise ValueError("No T values >= 100 keV found in T_list.")

    Tm = T_arr[mask]
    idxs = np.where(mask)[0]

    exc_colors = ["#1f77b4", "#2ca02c", "#17becf", "#8c564b", "#9467bd"]
    ion_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#e377c2", "#7f7f7f"]

    n_exc = len(s.excitations)
    n_ion = len(s.ionizations)

    # Ax handling: allow ax=None or ax=(ax_ion, ax_exc)
    if ax is None:
        fig_ion, ax_ion = plt.subplots(figsize=figsize)
        fig_exc, ax_exc = plt.subplots(figsize=figsize)
    else:
        if not (isinstance(ax, (tuple, list)) and len(ax) == 2):
            raise ValueError("ax must be None or a (ax_ion, ax_exc) tuple/list")
        ax_ion, ax_exc = ax

    # ----------------- Ionizations -----------------
    print("Plot IONIZATIONS (REL longitudinal solid, transverse dashed; T>=100keV)")
    for j in tqdm(range(n_ion)):
        y_long = []
        y_trans = []
        for i in idxs:
            long_list = sigma_list[i].get("ionization_sigma_rel", [0.0] * n_ion)
            trans_list = sigma_list[i].get("ionization_sigma_rel_trans", [0.0] * n_ion)
            y_long.append(long_list[j] if j < len(long_list) else 0.0)
            y_trans.append(trans_list[j] if j < len(trans_list) else 0.0)

        ax_ion.loglog(
            Tm, y_long,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth, alpha=alpha,
            label=f"Ion {j+1} Long",
        )
        ax_ion.loglog(
            Tm, y_trans,
            color=ion_colors[j % len(ion_colors)],
            lw=linewidth, alpha=alpha, ls="--",
            label=f"Ion {j+1} Trans",
        )

    ax_ion.set_xlabel("Incident energy T (eV)")
    ax_ion.set_ylabel("Relativistic component sigma_rel(T)")
    ax_ion.legend(loc="best", fontsize=8)
    ax_ion.grid(True, which="both", ls="--", alpha=0.3)

    # ----------------- Excitations -----------------
    print("Plot EXCITATIONS (REL longitudinal solid, transverse dashed; T>=100keV)")
    for k in tqdm(range(n_exc)):
        y_long = []
        y_trans = []
        for i in idxs:
            long_list = sigma_list[i].get("excitation_sigma_rel", [0.0] * n_exc)
            trans_list = sigma_list[i].get("excitation_sigma_rel_trans", [0.0] * n_exc)
            y_long.append(long_list[k] if k < len(long_list) else 0.0)
            y_trans.append(trans_list[k] if k < len(trans_list) else 0.0)

        ax_exc.loglog(
            Tm, y_long,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth, alpha=alpha,
            label=f"Exc {k+1} Long",
        )
        ax_exc.loglog(
            Tm, y_trans,
            color=exc_colors[k % len(exc_colors)],
            lw=linewidth, alpha=alpha, ls="--",
            label=f"Exc {k+1} Trans",
        )

    ax_exc.set_xlabel("Incident energy T (eV)")
    ax_exc.set_ylabel("Relativistic component sigma_rel(T)")
    ax_exc.legend(loc="best", fontsize=8)
    ax_exc.grid(True, which="both", ls="--", alpha=0.3)

    return (ax_ion, ax_exc)

def plot_total_cross_section(
        T_list, sigma_list, s,
        ax=None, linewidth=2, alpha=0.9, figsize=(12, 8),
        mc_T_max_eV=1.0e5):
    """
    Plot TOTAL cross section with *all* corrections (as available in sigma_list).

    The corrected total is chosen per T as:
      * if 'total_sigma_mc' exists and T <= mc_T_max_eV: use it
      * else if 'total_sigma_plus_rel_total' exists: use it (PWBA + REL_long + REL_trans)
      * else fall back to 'total_sigma_plus_rel' if present, else 'total_sigma'
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    T_arr = np.asarray(T_list, dtype=float)

    y_pwba = []
    y_corr = []

    for i, Tj in enumerate(T_arr):
        pwba = float(sigma_list[i].get("total_sigma_pwba", sigma_list[i].get("total_sigma", 0.0)) or 0.0)

        # choose corrected total
        if ("total_sigma_mc" in sigma_list[i]) and (float(Tj) <= float(mc_T_max_eV)):
            corr = float(sigma_list[i].get("total_sigma_mc", pwba) or pwba)
        else:
            if "total_sigma_plus_rel_total" in sigma_list[i]:
                corr = float(sigma_list[i].get("total_sigma_plus_rel_total", pwba) or pwba)
            elif "total_sigma_plus_rel" in sigma_list[i]:
                corr = float(sigma_list[i].get("total_sigma_plus_rel", pwba) or pwba)
            else:
                corr = pwba

        y_pwba.append(pwba)
        y_corr.append(corr)

    ax.loglog(T_arr, y_pwba, lw=linewidth, alpha=alpha, ls=":", label="Total PWBA")
    ax.loglog(T_arr, y_corr, lw=linewidth+1, alpha=alpha, ls="-", label="Total (all corrections)")

    ax.set_xlabel("Incident energy T (eV)")
    ax.set_ylabel("Total cross section sigma(T)")
    ax.set_title("Total cross section: PWBA vs all corrections")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return ax

# ----------------------------------------------------------------------
# Workers for parallel computing
# ----------------------------------------------------------------------
# Globals set once per worker process
_WORKER_S = None
_WORKER_C = None
_WORKER_KW = None

def _init_worker(a_vec, b_vec, c_vec, NE, Nq, include_kshell, use_mott_coulomb, mc_T_max_eV):
    """
    Runs once inside each worker process. Builds s and C once to avoid repeated pickling.
    """
    global _WORKER_S, _WORKER_C, _WORKER_KW

    import emfietzoglou_model_finite_q as model
    _WORKER_S = model.epsilon_optical('amorphous')
    _WORKER_C = model.DispersionCoeffs(a_fj=a_vec, b_fj=b_vec, c_fj=c_vec)
    _WORKER_KW = dict(
        NE=NE,
        Nq=Nq,
        include_kshell=include_kshell,
        use_mott_coulomb=use_mott_coulomb,
        mc_T_max_eV=mc_T_max_eV,
    )

def _compute_for_T(T):
    """
    Compute sigma dict for one T.
    Returns (T, sigma_dict)
    """
    sigma = integrate_elf_double_integral(_WORKER_S, _WORKER_C, float(T), **_WORKER_KW)
    return float(T), sigma

def main():
    # Optical model / dispersion coefficients
    s = model.epsilon_optical('amorphous')
    a_vec = np.array([3.82, 2.47, 2.47, 3.01, 2.44])
    b_vec = np.array([0.0272, 0.0295, 0.0311, 0.0111, 0.0633])
    c_vec = np.array([0.098, 0.075, 0.074, 0.765, 0.425])

    # RR2017 defaults for c_disp, d_disp, b1, b2
    C = model.DispersionCoeffs(a_fj=a_vec, b_fj=b_vec, c_fj=c_vec)

    # Energy grid (eV)
    T_list = np.logspace(1, 6, 40)

    print("Computing double-integrated cross sections (parallel over T)...")

    # Computing Choices
    NE = 100
    Nq = 100
    include_kshell = True
    use_mott_coulomb = True
    mc_T_max_eV = 1.0e5

    # Use ~ (CPU cores) workers;
    max_workers = max(1, (os.cpu_count() or 4) - 1)
    print("Using %i workers" %(max_workers))

    results_by_T = {}

    with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(a_vec, b_vec, c_vec, NE, Nq, include_kshell, use_mott_coulomb, mc_T_max_eV),
    ) as ex:
        futures = [ex.submit(_compute_for_T, T) for T in T_list]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            T_val, sigma = fut.result()
            results_by_T[T_val] = sigma

    # Restore original T order
    sigma_list = [results_by_T[float(T)] for T in T_list]

    # ----------------- Plot 1: comparison per channel (PWBA vs selected corrections) -----------------
    fig_ion, ax_ion = plt.subplots(figsize=(14, 9))
    fig_exc, ax_exc = plt.subplots(figsize=(14, 9))
    ax_ion, ax_exc = plot_full_cross_sections_per_channel(
        T_list, sigma_list, s,
        mc_T_max_eV=1.0e5,
        rel_T_min_eV=REL_T_THRESHOLD_eV
    )

    ax_ion.figure.tight_layout()
    ax_ion.figure.savefig("comparison_ionizations_pwba_vs_model.png", dpi=300)
    plt.close(ax_ion.figure)

    ax_exc.figure.tight_layout()
    ax_exc.figure.savefig("comparison_excitations_pwba_vs_model.png", dpi=300)
    plt.close(ax_exc.figure)
    print("Saved comparison plots to comparison_ionizations_pwba_vs_model.png and comparison_excitations_pwba_vs_model.png")


    # ----------------- Plot 2: REL components per channel (L solid, T dashed) -----------------
    fig_ion, ax_ion = plt.subplots(figsize=(14, 9))
    fig_exc, ax_exc = plt.subplots(figsize=(14, 9))
    plot_relativistic_component_per_channel(T_list, sigma_list, s, ax=(ax_ion, ax_exc))
    fig_ion.tight_layout()
    fig_exc.tight_layout()
    fig_ion.savefig("rel_cross_sections_ionizations_LT.png", dpi=300)
    fig_exc.savefig("rel_cross_sections_excitations_LT.png", dpi=300)
    plt.close(fig_ion)
    plt.close(fig_exc)
    print("Saved REL component plots to rel_cross_sections_ionizations_LT.png and rel_cross_sections_excitations_LT.png")

    # ----------------- Plot 3: TOTAL cross section with all corrections -----------------
    fig, ax = plt.subplots()
    plot_total_cross_section(T_list, sigma_list, s, ax=ax, mc_T_max_eV=1.0e5)
    fig.tight_layout()
    fig.savefig("total_cross_section_all_corrections.png", dpi=300)
    plt.close(fig)
    print("Saved total plot to total_cross_section_all_corrections.png")

if __name__ == "__main__":
    main()
