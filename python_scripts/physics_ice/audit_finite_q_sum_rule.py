#!/usr/bin/env python3
"""Audit the ion finite-q oscillator-strength and DCS stopping moments.

This is a numerical diagnostic, not a table generator. It reports the
valence-excitation, outer-ionization, and Heredia-Avalos K-continuum
contributions separately. The quantity 2-F_K is also reported as an occupancy-
closure diagnostic, but is not identified with or added as a bound spectrum:
the adopted source provides no bound-excitation energies or finite-q GOSs.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PHYSICS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHYSICS_DIR))

import emfietzoglou_model_finite_q as model


Q_VALUES = (0.0, 0.1, 0.3, 1.0, 2.0, 5.0, 10.0, 30.0)


def dispersion_coefficients():
    return model.default_dispersion_coefficients()


def audit_energy_grid(points):
    if points < 1001:
        raise ValueError("The sum-rule audit requires at least 1001 energy points.")
    fixed = np.array([
        7.0,
        10.0,
        13.0,
        17.0,
        32.0,
        model.OXYGEN_K_B_EV,
        model.OXYGEN_K_ZEFF**2 * model.RYD_ELECTRON_VOLT,
    ])
    k_offsets = model.OXYGEN_K_B_EV + np.geomspace(
        1.0e-9, 1.0e9, max(points // 4, 1001)
    )
    return np.unique(np.concatenate((
        np.geomspace(1.0e-5, 1.0e9, points),
        k_offsets,
        fixed,
    )))


def _strength_and_energy_moment(E, spectrum, Ep_eV):
    conversion = (
        model.WATER_TOTAL_OSCILLATOR_STRENGTH
        * 2.0
        / (np.pi * float(Ep_eV) ** 2)
    )
    strength = conversion * np.trapezoid(E * spectrum, E)
    first_energy_moment = conversion * np.trapezoid(E * E * spectrum, E)
    return float(strength), float(first_energy_moment)


def component_audit(phase, q_value, points):
    s = model.epsilon_optical(phase)
    C = dispersion_coefficients()
    E = audit_energy_grid(points)
    q = np.array([float(q_value)])
    k_continuum_strength = float(
        model.oxygen_K_hydrogenic_gos_continuum_strength(q)[0]
    )
    e1 = model.epsilon1_valence_Eq(
        E,
        q,
        s,
        C,
        inner_shell_strength_electrons=np.array([k_continuum_strength]),
    )
    e2 = model.epsilon2_valence_Eq(
        E,
        q,
        s,
        C,
        partitioned=True,
        inner_shell_strength_electrons=np.array([k_continuum_strength]),
    )
    denominator = e1["total"][0] ** 2 + e2["total"][0] ** 2
    denominator = np.maximum(denominator, np.finfo(float).tiny)
    excitation_epsilon2 = sum(
        (entry[0] for entry in e2["excitations"]), np.zeros_like(E)
    )
    ionization_epsilon2 = sum(
        (entry[0] for entry in e2["ionizations"]), np.zeros_like(E)
    )
    excitation_elf = excitation_epsilon2 / denominator
    ionization_elf = ionization_epsilon2 / denominator
    k_elf = model.oxygen_K_ion_hydrogenic_gos_elf(
        E,
        float(q_value),
        Ep_eV=s.Ep,
    )

    exc_strength, exc_moment = _strength_and_energy_moment(E, excitation_elf, s.Ep)
    ion_strength, ion_moment = _strength_and_energy_moment(E, ionization_elf, s.Ep)
    k_strength_integrated, k_moment = _strength_and_energy_moment(E, k_elf, s.Ep)
    total_strength = exc_strength + ion_strength + k_strength_integrated
    s1_exc_strength, _ = _strength_and_energy_moment(E, excitation_epsilon2, s.Ep)
    s1_ion_strength, _ = _strength_and_energy_moment(E, ionization_epsilon2, s.Ep)
    s1_total_strength = s1_exc_strength + s1_ion_strength + k_strength_integrated

    fj0 = np.array([entry.f for entry in s.excitations])
    fjq = model._fj_q(fj0, q, C)[0]
    allocated_outer = (
        model.WATER_TOTAL_OSCILLATOR_STRENGTH
        - model.WATER_TOTAL_OSCILLATOR_STRENGTH * float(np.sum(fjq))
        - k_continuum_strength
    )
    occupancy_remainder = float(
        model.oxygen_K_hydrogenic_occupancy_remainder(float(q_value))
    )
    return {
        "phase": phase,
        "q_a0_inverse": float(q_value),
        "s1_valence_excitation_strength_electrons": s1_exc_strength,
        "s1_valence_ionization_strength_electrons": s1_ion_strength,
        "s1_total_modeled_strength_electrons": s1_total_strength,
        "s2_valence_excitation_strength_electrons": exc_strength,
        "s2_valence_ionization_strength_electrons": ion_strength,
        "s2_valence_total_strength_electrons": exc_strength + ion_strength,
        "k_continuum_strength_electrons": k_strength_integrated,
        "k_continuum_lookup_strength_electrons": k_continuum_strength,
        "k_occupancy_closure_remainder_electrons": occupancy_remainder,
        "k_bound_excitation_strength_electrons": None,
        "k_bound_excitation_model": "not_available_from_heredia_avalos_gos",
        "s2_total_modeled_strength_electrons": total_strength,
        "total_if_occupancy_remainder_were_misidentified_as_bound_electrons": (
            total_strength + occupancy_remainder
        ),
        "outer_ionization_amplitude_allocation_electrons": allocated_outer,
        "amplitude_budget_electrons": (
            model.WATER_TOTAL_OSCILLATOR_STRENGTH * float(np.sum(fjq))
            + allocated_outer
            + k_continuum_strength
        ),
        "excitation_first_energy_moment_eV": exc_moment,
        "ionization_first_energy_moment_eV": ion_moment,
        "k_continuum_first_energy_moment_eV": k_moment,
        "total_first_energy_moment_eV": exc_moment + ion_moment + k_moment,
    }


def dcs_stopping_moment(phase, incident_eV, NE, Nq):
    import generate_ice_cross_sections_ion as generator

    generator.set_projectile("proton")
    generator._set_kshell_model("hydrogenic-gos")
    generator._set_projectile_relativistic_dcs(False, False)
    s = model.epsilon_optical(phase)
    data = generator.integrate_elf_channels_per_channel_q(
        s,
        dispersion_coefficients(),
        T=float(incident_eV),
        NE=int(NE),
        Nq=int(Nq),
        include_kshell=True,
    )

    def integrate_group(energy_grids, dcs_grids):
        tcs = 0.0
        stopping = 0.0
        for energy, dcs in zip(energy_grids, dcs_grids):
            if energy.size:
                tcs += generator._simpson_integrate(dcs, energy)
                stopping += generator._simpson_integrate(energy * dcs, energy)
        return float(tcs), float(stopping)

    exc_tcs, exc_stopping = integrate_group(
        data["excitation_E"], data["excitation_int"]
    )
    ion_tcs, ion_stopping = integrate_group(
        data["ionization_E"], data["ionization_int"]
    )
    k_tcs = 0.0
    k_stopping = 0.0
    if data["kshell_E"] is not None:
        k_tcs = generator._simpson_integrate(data["kshell_int"], data["kshell_E"])
        k_stopping = generator._simpson_integrate(
            data["kshell_E"] * data["kshell_int"], data["kshell_E"]
        )
    return {
        "phase": phase,
        "projectile": "proton",
        "incident_energy_eV": float(incident_eV),
        "NE": int(NE),
        "Nq": int(Nq),
        "tcs_m2": exc_tcs + ion_tcs + k_tcs,
        "stopping_cross_section_eV_m2": exc_stopping + ion_stopping + k_stopping,
        "excitation_stopping_cross_section_eV_m2": exc_stopping,
        "ionization_stopping_cross_section_eV_m2": ion_stopping,
        "k_continuum_stopping_cross_section_eV_m2": k_stopping,
    }


def relative_change(coarse, fine, key):
    denominator = float(fine[key])
    return abs(float(coarse[key]) - denominator) / abs(denominator)


def run_audit(points, stopping_energy_eV, coarse_resolution, fine_resolution):
    components = [
        component_audit(phase, q_value, points)
        for phase in ("amorphous", "hexagonal")
        for q_value in Q_VALUES
    ]
    component_convergence = []
    for phase in ("amorphous", "hexagonal"):
        for q_value in (0.0, 1.0, 10.0, 30.0):
            coarse = component_audit(phase, q_value, points // 2)
            fine = component_audit(phase, q_value, points)
            component_convergence.append({
                "phase": phase,
                "q_a0_inverse": q_value,
                "strength_relative_change": relative_change(
                    coarse, fine, "s2_total_modeled_strength_electrons"
                ),
                "first_energy_moment_relative_change": relative_change(
                    coarse, fine, "total_first_energy_moment_eV"
                ),
            })

    stopping_convergence = []
    for phase in ("amorphous", "hexagonal"):
        coarse = dcs_stopping_moment(
            phase, stopping_energy_eV, coarse_resolution, coarse_resolution
        )
        fine = dcs_stopping_moment(
            phase, stopping_energy_eV, fine_resolution, fine_resolution
        )
        stopping_convergence.append({
            "phase": phase,
            "coarse": coarse,
            "fine": fine,
            "tcs_relative_change": relative_change(coarse, fine, "tcs_m2"),
            "stopping_moment_relative_change": relative_change(
                coarse, fine, "stopping_cross_section_eV_m2"
            ),
        })
    return {
        "model": model.ION_FINITE_Q_SUM_RULE_VERSION,
        "components": components,
        "component_convergence": component_convergence,
        "dcs_stopping_convergence": stopping_convergence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=40001)
    parser.add_argument("--stopping-energy-eV", type=float, default=1.0e7)
    parser.add_argument("--coarse-resolution", type=int, default=160)
    parser.add_argument("--fine-resolution", type=int, default=320)
    args = parser.parse_args()
    report = run_audit(
        args.points,
        args.stopping_energy_eV,
        args.coarse_resolution,
        args.fine_resolution,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
