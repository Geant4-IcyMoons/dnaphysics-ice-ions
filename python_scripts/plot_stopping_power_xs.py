#!/usr/bin/env python3
"""
Compute stopping power directly from Geant4-DNA cross-section tables (no MC).

This mirrors the water and ice physics lists in this project:
- Vib excitation: Michaud (water + ice)
- Excitation: Emfietzoglou
- Ionisation: Emfietzoglou (from differential tables)
- Attachment: Melton (water) vs Michaud (ice)

Notes:
- The Emfietzoglou excitation/ionisation tables stop at 10 keV.
- Vib and attachment are limited to lower energies.
  Outside the model ranges the stopping power is set to 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from constants import (
    CROSS_SECTIONS_DIR,
    CUSTOM_DATA_ROOT_GEANT4,
    CUSTOM_DATA_ROOT_PROJECT,
    EMFI_EXCITATION_EEV,
    EMFIETZOGLOU_SCALE_1E16,
    FONT_COURIER,
    FONTSIZE_24,
    N,
    OUTPUT_DIR,
    RC_BASE_ELASTIC,
    MICHAUD_SIGMA_SCALE_CM2,
    rcparams_with_fontsize,
)

# --- Plot style (match plot_stopping_power_root.py) ---
plt.rcParams["font.family"] = FONT_COURIER
plt.rcParams["mathtext.rm"] = FONT_COURIER
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams.update(rcparams_with_fontsize(RC_BASE_ELASTIC, FONTSIZE_24))

# --- Unit conversion ---
EV_NM_TO_MEV_CM = 10.0
EV_NM_TO_EV_ANG = 0.1

CM_PER_NM = 1.0e7

WATER_DENSITY_G_CM3 = 1.0
ICE_DENSITY_G_CM3 = 0.917

N_CM3_WATER = N / 1.0e6
N_CM3_ICE = N_CM3_WATER * (ICE_DENSITY_G_CM3 / WATER_DENSITY_G_CM3)

VIB_ELOSS_EEV = np.array(
    [0.024, 0.061, 0.092, 0.205, 0.417, 0.460, 0.510, 0.834], dtype=float
)


def _data_root() -> Path:
    if CUSTOM_DATA_ROOT_GEANT4.exists():
        return CUSTOM_DATA_ROOT_GEANT4
    return CUSTOM_DATA_ROOT_PROJECT


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required data file: {path}")


def _interp_loglog(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    if x.size == 0:
        return np.zeros_like(x_new)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    positive = y > 0
    if np.count_nonzero(positive) < 2:
        return np.zeros_like(x_new)

    x_pos = x[positive]
    y_pos = y[positive]

    out = np.zeros_like(x_new)
    in_range = (x_new >= x_pos[0]) & (x_new <= x_pos[-1])
    if np.any(in_range):
        out[in_range] = 10 ** np.interp(
            np.log10(x_new[in_range]),
            np.log10(x_pos),
            np.log10(y_pos),
        )
    out[~np.isfinite(out)] = 0.0
    return out


def _units_and_label(units: str, density: float, dedx: np.ndarray) -> tuple[np.ndarray, str]:
    if units == "ev_ang":
        return dedx * EV_NM_TO_EV_ANG, "Stopping Power (eV/Angstrom)"
    if units == "mev_cm":
        return dedx * EV_NM_TO_MEV_CM, "Stopping Power (MeV/cm)"
    if units == "mev_cm2_g":
        return (dedx * EV_NM_TO_MEV_CM) / density, "Mass Stopping Power (MeV cm^2/g)"
    return dedx, "Stopping Power (eV/nm)"


def _load_table(path: Path) -> np.ndarray:
    _ensure_exists(path)
    return np.loadtxt(path, dtype=float)


def _vib_energy_loss_xs(energy_grid: np.ndarray, path: Path) -> np.ndarray:
    data = _load_table(path)
    energies = data[:, 0]
    sigma_levels = data[:, 1:]
    if sigma_levels.shape[1] != VIB_ELOSS_EEV.size:
        raise ValueError(f"Unexpected vib table shape in {path}")

    sigma_cm2 = sigma_levels * MICHAUD_SIGMA_SCALE_CM2
    eloss_xs = np.zeros_like(energy_grid)
    for i, omega in enumerate(VIB_ELOSS_EEV):
        sigma_i = _interp_loglog(energies, sigma_cm2[:, i], energy_grid)
        eloss_xs += sigma_i * omega
    return eloss_xs


def _attachment_energy_loss_xs(energy_grid: np.ndarray, path: Path) -> np.ndarray:
    data = _load_table(path)
    energies = data[:, 0]
    sigma = data[:, 1] * MICHAUD_SIGMA_SCALE_CM2
    sigma_interp = _interp_loglog(energies, sigma, energy_grid)
    return sigma_interp * energy_grid


def _excitation_energy_loss_xs(energy_grid: np.ndarray, path: Path) -> np.ndarray:
    data = _load_table(path)
    energies = data[:, 0]
    sigma_levels = data[:, 1:1 + EMFI_EXCITATION_EEV.size]
    if sigma_levels.shape[1] != EMFI_EXCITATION_EEV.size:
        raise ValueError(f"Unexpected excitation table shape in {path}")

    scale_cm2 = EMFIETZOGLOU_SCALE_1E16 * 1.0e-16
    sigma_cm2 = sigma_levels * scale_cm2

    eloss_xs = np.zeros_like(energy_grid)
    for i, e_loss in enumerate(EMFI_EXCITATION_EEV):
        sigma_i = _interp_loglog(energies, sigma_cm2[:, i], energy_grid)
        eloss_xs += sigma_i * e_loss
    return eloss_xs


def _trapz_nonuniform(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))


def _simpson_nonuniform(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return 0.0

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if np.any(np.diff(x) == 0.0):
        unique_x, inv = np.unique(x, return_inverse=True)
        y_acc = np.zeros_like(unique_x)
        counts = np.zeros_like(unique_x)
        for i, idx in enumerate(inv):
            y_acc[idx] += y[i]
            counts[idx] += 1.0
        y = y_acc / np.where(counts == 0.0, 1.0, counts)
        x = unique_x

    n = x.size
    if n == 2:
        return _trapz_nonuniform(x, y)

    def _simpson_segment(x0, x1, x2, y0, y1, y2):
        h0 = x1 - x0
        h1 = x2 - x1
        if h0 <= 0.0 or h1 <= 0.0:
            return 0.0
        denom = h0 * h1
        if denom == 0.0:
            return 0.0
        w0 = 2.0 - (h1 / h0)
        w1 = ((h0 + h1) ** 2) / denom
        w2 = 2.0 - (h0 / h1)
        return (h0 + h1) / 6.0 * (w0 * y0 + w1 * y1 + w2 * y2)

    end = n if n % 2 == 1 else n - 1
    total = 0.0
    for i in range(0, end - 2, 2):
        total += _simpson_segment(
            x[i], x[i + 1], x[i + 2], y[i], y[i + 1], y[i + 2]
        )
    if n % 2 == 0:
        total += _trapz_nonuniform(x[-2:], y[-2:])
    return total


def _integrate_ion_eloss(w_vals: list[float], sigma_vals: list[list[float]]) -> float:
    if len(w_vals) < 2:
        return 0.0
    w = np.asarray(w_vals, dtype=float)
    sigma = np.asarray(sigma_vals, dtype=float)
    scale_cm2 = EMFIETZOGLOU_SCALE_1E16 * 1.0e-16
    sigma *= scale_cm2

    eloss = 0.0
    for j in range(sigma.shape[0]):
        eloss += _simpson_nonuniform(w, w * sigma[j])
    return eloss


def _ion_energy_loss_xs_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    _ensure_exists(path)
    energies = []
    eloss_xs = []

    current_e = None
    w_vals: list[float] = []
    sigma_vals = [[] for _ in range(5)]

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            e = float(parts[0])
            w = float(parts[1])
            sigs = [float(val) for val in parts[2:7]]

            if current_e is None:
                current_e = e
            if e != current_e:
                energies.append(current_e)
                eloss_xs.append(_integrate_ion_eloss(w_vals, sigma_vals))
                current_e = e
                w_vals = []
                sigma_vals = [[] for _ in range(5)]

            w_vals.append(w)
            for j in range(5):
                sigma_vals[j].append(sigs[j])

    if current_e is not None:
        energies.append(current_e)
        eloss_xs.append(_integrate_ion_eloss(w_vals, sigma_vals))

    return np.asarray(energies, dtype=float), np.asarray(eloss_xs, dtype=float)


def _to_dedx_ev_nm(eloss_xs: np.ndarray, n_cm3: float) -> np.ndarray:
    return (eloss_xs * n_cm3) / CM_PER_NM


def _compute_stopping_power(
    energy_grid: np.ndarray,
    n_cm3: float,
    vib_path: Path,
    attach_path: Path,
    exc_eloss_xs: np.ndarray,
    ion_energy: np.ndarray,
    ion_eloss_xs: np.ndarray,
) -> np.ndarray:
    vib_eloss = _vib_energy_loss_xs(energy_grid, vib_path)
    attach_eloss = _attachment_energy_loss_xs(energy_grid, attach_path)
    ion_eloss = _interp_loglog(ion_energy, ion_eloss_xs, energy_grid)

    total_eloss_xs = vib_eloss + exc_eloss_xs + ion_eloss + attach_eloss
    return _to_dedx_ev_nm(total_eloss_xs, n_cm3)


def _plot(
    energy_grid: np.ndarray,
    water_dedx: np.ndarray,
    ice_dedx: np.ndarray,
    units: str,
    out_path: Path,
    rho_water: float,
    rho_ice: float,
) -> None:
    water_plot, ylabel = _units_and_label(units, rho_water, water_dedx)
    ice_plot, _ = _units_and_label(units, rho_ice, ice_dedx)

    fig, ax = plt.subplots(figsize=(10, 6))
    valid_w = np.isfinite(water_plot) & (water_plot > 0) & np.isfinite(energy_grid)
    valid_i = np.isfinite(ice_plot) & (ice_plot > 0) & np.isfinite(energy_grid)

    ax.loglog(
        energy_grid[valid_w],
        water_plot[valid_w],
        color="black",
        linewidth=3,
        label="Water (physics list)",
        zorder=3,
    )
    ax.loglog(
        energy_grid[valid_i],
        ice_plot[valid_i],
        color="0.35",
        linewidth=3,
        label="Ice (physics list)",
        zorder=2,
    )

    ax.set_xlabel("Electron Energy (T; eV)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(energy_grid.min(), energy_grid.max())
    ax.legend(loc="best")
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute stopping power from cross sections for water vs ice."
    )
    parser.add_argument("--emin", type=float, default=1.0, help="Minimum energy (eV)")
    parser.add_argument("--emax", type=float, default=1.0e6, help="Maximum energy (eV)")
    parser.add_argument("--nbins", type=int, default=500, help="Number of log bins")
    parser.add_argument(
        "--units",
        type=str,
        default="ev_ang",
        choices=("ev_ang", "ev_nm", "mev_cm", "mev_cm2_g"),
        help="Output units for stopping power.",
    )
    parser.add_argument(
        "--rho-water",
        type=float,
        default=WATER_DENSITY_G_CM3,
        help="Water density (g/cm^3) for mass stopping power.",
    )
    parser.add_argument(
        "--rho-ice",
        type=float,
        default=ICE_DENSITY_G_CM3,
        help="Ice density (g/cm^3) for mass stopping power.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR / "stopping_power_xs_water_vs_ice.png",
        help="Output plot path.",
    )
    args = parser.parse_args()

    if args.emin <= 0 or args.emax <= 0 or args.emin >= args.emax:
        raise ValueError("Invalid energy range.")

    data_root = _data_root()
    dna_dir = data_root / "G4EMLOW8.6.1" / "dna"

    water_vib = dna_dir / "sigma_excitationvib_e_michaud.dat"
    ice_vib = CROSS_SECTIONS_DIR / "sigma_excitationvib_e_michaud.dat"
    water_attach = dna_dir / "sigma_attachment_e_melton.dat"
    ice_attach = CROSS_SECTIONS_DIR / "sigma_attachment_e_michaud.dat"
    exc_path = dna_dir / "sigma_excitation_e_emfietzoglou.dat"
    ion_dcs_path = dna_dir / "sigmadiff_ionisation_e_emfietzoglou.dat"

    energy = np.logspace(np.log10(args.emin), np.log10(args.emax), args.nbins)

    exc_eloss_xs = _excitation_energy_loss_xs(energy, exc_path)
    ion_energy, ion_eloss_xs = _ion_energy_loss_xs_table(ion_dcs_path)

    water_dedx = _compute_stopping_power(
        energy_grid=energy,
        n_cm3=N_CM3_WATER,
        vib_path=water_vib,
        attach_path=water_attach,
        exc_eloss_xs=exc_eloss_xs,
        ion_energy=ion_energy,
        ion_eloss_xs=ion_eloss_xs,
    )
    ice_dedx = _compute_stopping_power(
        energy_grid=energy,
        n_cm3=N_CM3_ICE,
        vib_path=ice_vib,
        attach_path=ice_attach,
        exc_eloss_xs=exc_eloss_xs,
        ion_energy=ion_energy,
        ion_eloss_xs=ion_eloss_xs,
    )

    _plot(
        energy_grid=energy,
        water_dedx=water_dedx,
        ice_dedx=ice_dedx,
        units=args.units,
        out_path=args.out,
        rho_water=args.rho_water,
        rho_ice=args.rho_ice,
    )


if __name__ == "__main__":
    main()
