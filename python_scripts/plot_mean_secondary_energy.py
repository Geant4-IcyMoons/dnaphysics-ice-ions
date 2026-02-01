#!/usr/bin/env python3
"""
Plot mean secondary-electron energy <W> vs incident energy T
from Geant4-DNA ionisation DCS tables for water and ice.
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
    EMFI_ION_BINDING_EEV,
    FONT_COURIER,
    FONTSIZE_24,
    OUTPUT_DIR,
    RC_BASE_ELASTIC,
    rcparams_with_fontsize,
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
        return float(0.5 * (y[0] + y[1]) * (x[1] - x[0]))

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
        total += float(0.5 * (y[-2] + y[-1]) * (x[-1] - x[-2]))
    return total


def _mean_w_from_dcs(path: Path, bindings_eV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _ensure_exists(path)
    energies = []
    mean_w = []

    current_e = None
    e_vals: list[float] = []
    sigma_vals: list[list[float]] = []
    n_channels = None

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            t_val = float(parts[0])
            transfer = float(parts[1])
            if n_channels is None:
                n_channels = len(parts) - 2
                sigma_vals = [[] for _ in range(n_channels)]
            if len(parts) < 2 + n_channels:
                continue
            sigs = [float(val) for val in parts[2 : 2 + n_channels]]

            if current_e is None:
                current_e = t_val
            if t_val != current_e:
                energies.append(current_e)
                mean_w.append(_mean_w_group(e_vals, sigma_vals, bindings_eV))
                current_e = t_val
                e_vals = []
                sigma_vals = [[] for _ in range(n_channels)]

            e_vals.append(transfer)
            for j in range(n_channels):
                sigma_vals[j].append(sigs[j])

    if current_e is not None:
        energies.append(current_e)
        mean_w.append(_mean_w_group(e_vals, sigma_vals, bindings_eV))

    return np.asarray(energies, dtype=float), np.asarray(mean_w, dtype=float)

def _merge_low_high_tables(
    low_energy: np.ndarray,
    low_vals: np.ndarray,
    high_energy: np.ndarray,
    high_vals: np.ndarray,
    switch_e: float,
) -> tuple[np.ndarray, np.ndarray]:
    low_energy = np.asarray(low_energy, dtype=float)
    low_vals = np.asarray(low_vals, dtype=float)
    high_energy = np.asarray(high_energy, dtype=float)
    high_vals = np.asarray(high_vals, dtype=float)

    if low_energy.size == 0:
        return high_energy, high_vals
    if high_energy.size == 0:
        return low_energy, low_vals

    low_mask = low_energy <= switch_e
    high_mask = high_energy > switch_e
    merged_energy = np.concatenate([low_energy[low_mask], high_energy[high_mask]])
    merged_vals = np.concatenate([low_vals[low_mask], high_vals[high_mask]])

    order = np.argsort(merged_energy)
    return merged_energy[order], merged_vals[order]


def _mean_w_group(e_vals: list[float], sigma_vals: list[list[float]], bindings_eV: np.ndarray) -> float:
    if len(e_vals) < 2:
        return 0.0
    e = np.asarray(e_vals, dtype=float)
    sigma = np.asarray(sigma_vals, dtype=float)
    n_channels = sigma.shape[0]

    b = np.asarray(bindings_eV, dtype=float)
    if b.size < n_channels:
        b = np.pad(b, (0, n_channels - b.size), mode="edge")
    elif b.size > n_channels:
        b = b[:n_channels]

    numerator = 0.0
    denom = 0.0
    for j in range(n_channels):
        s = sigma[j]
        total = _simpson_nonuniform(e, s)
        if total <= 0.0:
            continue
        e_weight = _simpson_nonuniform(e, e * s)
        numerator += e_weight - b[j] * total
        denom += total

    if denom <= 0.0:
        return 0.0
    return numerator / denom


def _plot(
    energy_grid: np.ndarray,
    water_w: np.ndarray,
    ice_series: list[dict],
    out_path: Path,
    water_emax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    valid_w = np.isfinite(water_w) & (water_w > 0) & np.isfinite(energy_grid)
    if water_emax is not None:
        valid_w &= energy_grid <= water_emax

    ax.loglog(
        energy_grid[valid_w],
        water_w[valid_w],
        color="black",
        linewidth=3,
        label="Water",
        zorder=3,
    )

    for idx, series in enumerate(ice_series):
        ice_w = series["mean_w"]
        valid_i = np.isfinite(ice_w) & (ice_w > 0) & np.isfinite(energy_grid)
        ice_emax = series.get("emax")
        if ice_emax is not None:
            valid_i &= energy_grid <= ice_emax
        ax.loglog(
            energy_grid[valid_i],
            ice_w[valid_i],
            color=series.get("color", f"0.{35 + idx * 20:02d}"),
            linewidth=3,
            ls=series.get("ls", "-"),
            label=series.get("label", "Ice"),
            zorder=2,
        )

    ax.set_xlabel("Electron energy (T; eV)")
    ax.set_ylabel("$<W>$ (eV)")
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
        description="Compute mean secondary energy <W> from ionisation DCS tables."
    )
    parser.add_argument("--emin", type=float, default=1.0, help="Minimum energy (eV)")
    parser.add_argument("--emax", type=float, default=1.0e7, help="Maximum energy (eV)")
    parser.add_argument("--nbins", type=int, default=1000, help="Number of log bins")
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR / "mean_secondary_energy_water_vs_ice.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--ice-types",
        type=str,
        default="amorphous,hexagonal",
        help="Comma-separated ice types to plot (amorphous,hexagonal).",
    )
    args = parser.parse_args()

    if args.emin <= 0 or args.emax <= 0 or args.emin >= args.emax:
        raise ValueError("Invalid energy range.")

    plt.rcParams["font.family"] = FONT_COURIER
    plt.rcParams["mathtext.rm"] = FONT_COURIER
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams.update(rcparams_with_fontsize(RC_BASE_ELASTIC, FONTSIZE_24))

    data_root = _data_root()
    dna_dir = data_root / "G4EMLOW8.6.1" / "dna"
    water_dcs_path = dna_dir / "sigmadiff_ionisation_e_emfietzoglou.dat"

    ice_types = [t.strip().lower() for t in args.ice_types.split(",") if t.strip()]
    ice_types = [t for t in ice_types if t in ("amorphous", "hexagonal")]
    if not ice_types:
        raise ValueError("No valid ice types specified (use amorphous and/or hexagonal).")

    born_dcs_path = dna_dir / "sigmadiff_ionisation_e_born.dat"
    water_T_emfi, water_mean_w_emfi = _mean_w_from_dcs(
        water_dcs_path, EMFI_ION_BINDING_EEV
    )
    water_T = water_T_emfi
    water_mean_w = water_mean_w_emfi
    if born_dcs_path.exists():
        water_T_born, water_mean_w_born = _mean_w_from_dcs(
            born_dcs_path, EMFI_ION_BINDING_EEV
        )
        water_T, water_mean_w = _merge_low_high_tables(
            water_T_emfi,
            water_mean_w_emfi,
            water_T_born,
            water_mean_w_born,
            1.0e4,
        )
    max_water = float(np.max(water_T)) if water_T.size else None

    ice_series = []
    ice_styles = {
        "amorphous": {"color": "slategray", "ls": "-"},
        "hexagonal": {"color": "0.55", "ls": "-"},
    }
    max_ice = args.emax
    for ice_type in ice_types:
        ice_label = f"{ice_type}_ice"
        ice_path = CROSS_SECTIONS_DIR / f"sigmadiff_ionisation_e_{ice_label}_emfietzoglou_kyriakou.dat"
        if not ice_path.exists():
            fallback = CROSS_SECTIONS_DIR / "sigmadiff_ionisation_e_ice_emfietzoglou_kyriakou.dat"
            if fallback.exists():
                print(f"Missing {ice_path.name}; falling back to {fallback.name}.")
                ice_path = fallback
        ice_T, ice_mean_w = _mean_w_from_dcs(ice_path, EMFI_ION_BINDING_EEV)
        if ice_T.size:
            max_ice = max(max_ice, float(np.max(ice_T)))
        style = ice_styles.get(ice_type, {"color": "0.6", "ls": "-"})
        ice_series.append(
            {
                "label": f"{ice_type.capitalize()} ice",
                "T": ice_T,
                "mean_w_raw": ice_mean_w,
                "color": style["color"],
                "ls": style["ls"],
                "emax": float(np.max(ice_T)) if ice_T.size else None,
            }
        )

    energy = np.logspace(np.log10(args.emin), np.log10(args.emax), args.nbins)
    water_w = _interp_loglog(water_T, water_mean_w, energy)
    for series in ice_series:
        series["mean_w"] = _interp_loglog(series["T"], series["mean_w_raw"], energy)

    if max_water is not None and args.emax > max_water:
        print(
            f"Requested emax={args.emax:.3e} eV exceeds water table coverage; "
            f"water curve will be truncated at {max_water:.3e} eV."
        )

    _plot(
        energy_grid=energy,
        water_w=water_w,
        ice_series=ice_series,
        out_path=args.out,
        water_emax=max_water,
    )


if __name__ == "__main__":
    main()
