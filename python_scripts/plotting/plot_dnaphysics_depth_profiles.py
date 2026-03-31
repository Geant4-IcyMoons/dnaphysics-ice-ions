#!/usr/bin/env python3
"""
Plot/print depth profiles from dnaphysics ROOT output:
  - energy deposition vs depth [MGy/yr]
  - average primary electron energy vs depth [MeV]
  - average secondary electron energy vs depth [MeV]

Assumes:
  - step ntuple contains totalEnergyDeposit (eV), kineticEnergy (eV), z (nm)
  - eventID/trackID/parentID/stepID identify primary energies per event
  - flux weighting via electron_spectrum_fit with deltaE and n_per_bin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uproot

from generate_europa_electron_bins import electron_spectrum_fit
from root_utils import resolve_root_paths
from constants import (
    FONT_COURIER,
    FONTSIZE_24,
    OUTPUT_DIR,
    RC_BASE_ELASTIC,
    rcparams_with_fontsize,
)


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
MEV_TO_MGY_PER_YEAR = 1.602176634e-10 * SECONDS_PER_YEAR / 1.0e6


plt.rcParams["font.family"] = FONT_COURIER
plt.rcParams["mathtext.rm"] = FONT_COURIER
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams.update(rcparams_with_fontsize(RC_BASE_ELASTIC, FONTSIZE_24))


def _legacy_z_range_mm() -> np.ndarray:
    return np.round(
        np.array(
            list(np.arange(0.0, 0.05, 0.0001))
            + list(np.arange(0.05, 0.5, 0.001))
            + list(np.arange(0.5, 1.0, 0.01))
            + list(np.arange(1.0, 10.0, 0.1))
            + list(np.arange(10.0, 100.0, 0.5))
            + list(np.arange(100.0, 1000.0, 1.0)),
            dtype=float,
        ),
        8,
    )


def _load_step_arrays(root_path: Path, max_entries: int | None = None) -> dict[str, np.ndarray]:
    paths = resolve_root_paths(root_path)
    if not paths:
        raise FileNotFoundError(root_path)
    fields = [
        "flagParticle",
        "z",
        "totalEnergyDeposit",
        "kineticEnergy",
        "eventID",
        "trackID",
        "parentID",
        "stepID",
    ]
    tree_spec = [f"{p}:step" for p in paths]
    return uproot.concatenate(tree_spec, fields, entry_stop=max_entries, library="np")


def _event_energy_map(arrs: dict[str, np.ndarray]) -> dict[int, float]:
    event_id = arrs["eventID"].astype(int)
    track_id = arrs["trackID"].astype(int)
    parent_id = arrs["parentID"].astype(int)
    step_id = arrs["stepID"].astype(int)
    kin_e = arrs["kineticEnergy"].astype(float)

    mask = (track_id == 1) & (parent_id == 0) & (step_id == 0)
    if not np.any(mask):
        return {}

    event_ids = event_id[mask]
    energies_mev = kin_e[mask] * 1.0e-6
    return {int(eid): float(e) for eid, e in zip(event_ids, energies_mev)}


def _infer_delta_e(energies_mev: np.ndarray) -> float:
    uniq = np.unique(energies_mev)
    if uniq.size < 2:
        return 0.0
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if diffs.size else 0.0


def _z_edges_from_file(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    if len(values) < 1:
        raise ValueError(f"No usable z_range values in {path}")
    values = np.unique(np.asarray(values, dtype=float))
    if values.size < 2:
        last = values[-1]
        return np.array([last, last + 1.0], dtype=float)
    delta = values[-1] - values[-2]
    return np.append(values, values[-1] + delta)


def main() -> None:
    ap = argparse.ArgumentParser(description="Depth profiles from dnaphysics ROOT output.")
    ap.add_argument("--root", type=Path, default=Path("build/dna.root"), help="Path to dna.root")
    ap.add_argument("--density-g-cm3", type=float, default=0.917, help="Ice density (g/cm^3)")
    ap.add_argument("--n-per-bin", type=int, default=1000, help="Electrons per energy bin in the macro")
    ap.add_argument("--delta-e", type=float, default=None, help="Energy bin width (MeV). If omitted, inferred.")
    ap.add_argument("--z-range-file", type=Path, default=None,
                    help="Optional z_range file (mm per line). If set, overrides legacy grid.")
    ap.add_argument("--use-legacy-z", action="store_true", help="Use legacy depth grid (default).")
    ap.add_argument("--dz-mm", type=float, default=0.1, help="Uniform depth bin (mm) if not using legacy grid.")
    ap.add_argument("--depth-origin-mm", type=float, default=0.0, help="Depth origin in mm (z=0 by default).")
    ap.add_argument("--max-entries", type=int, default=None, help="Limit number of rows for quick tests.")
    ap.add_argument("--print", action="store_true", help="Print depth profile table to stdout.")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "depth_profiles.png",
                    help="Output plot path.")
    args = ap.parse_args()

    arrs = _load_step_arrays(args.root, max_entries=args.max_entries)

    # Filter electrons
    flag_particle = arrs["flagParticle"].astype(int)
    m_e = (flag_particle == 1)
    if not np.any(m_e):
        raise RuntimeError("No electron steps found in ROOT file.")

    z_nm = arrs["z"].astype(float)[m_e]
    z_mm = z_nm * 1.0e-6
    depth_mm = z_mm - float(args.depth_origin_mm)
    m_depth = depth_mm >= 0.0
    depth_mm = depth_mm[m_depth]

    edep_eV = arrs["totalEnergyDeposit"].astype(float)[m_e][m_depth]
    kin_e_eV = arrs["kineticEnergy"].astype(float)[m_e][m_depth]
    event_id = arrs["eventID"].astype(int)[m_e][m_depth]
    track_id = arrs["trackID"].astype(int)[m_e][m_depth]
    parent_id = arrs["parentID"].astype(int)[m_e][m_depth]

    # Map event -> primary energy
    event_energy = _event_energy_map(arrs)
    if not event_energy:
        raise RuntimeError("Could not determine primary energy per event.")

    energies_mev = np.array(list(event_energy.values()), dtype=float)
    delta_e = float(args.delta_e) if args.delta_e is not None else _infer_delta_e(energies_mev)
    if delta_e <= 0.0:
        raise RuntimeError("deltaE could not be inferred; pass --delta-e explicitly.")

    # Flux weight per event (per cm^2 per s)
    weight_per_event = {}
    for eid, e0 in event_energy.items():
        weight_per_event[int(eid)] = float(electron_spectrum_fit(e0) * delta_e / float(args.n_per_bin))

    weights = np.array([weight_per_event.get(int(eid), 0.0) for eid in event_id], dtype=float)

    # Depth bins
    if args.z_range_file is not None:
        z_edges = _z_edges_from_file(args.z_range_file)
    elif args.use_legacy_z or args.dz_mm <= 0:
        z_edges = _legacy_z_range_mm()
        z_edges = np.append(z_edges, z_edges[-1] + (z_edges[-1] - z_edges[-2]))
    else:
        z_max = float(np.nanmax(depth_mm)) if depth_mm.size else 0.0
        z_edges = np.arange(0.0, z_max + args.dz_mm, args.dz_mm)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    # Histogram deposits (MeV/cm^2/s)
    edep_mev = edep_eV * 1.0e-6
    edep_flux = edep_mev * weights
    dep_mev_per_cm2_s, _ = np.histogram(depth_mm, bins=z_edges, weights=edep_flux)

    # Convert to MGy/yr per depth bin (mass per cm^2 slice)
    dz_cm = (z_edges[1:] - z_edges[:-1]) * 0.1  # mm -> cm
    dose_mev_per_g_s = np.where(
        dz_cm > 0.0,
        dep_mev_per_cm2_s / (args.density_g_cm3 * dz_cm),
        0.0,
    )
    dose_mgy_per_yr = dose_mev_per_g_s * MEV_TO_MGY_PER_YEAR

    # Average primary/secondary electron kinetic energy per depth (weighted by flux)
    kin_mev = kin_e_eV * 1.0e-6
    prim_mask = (track_id == 1) & (parent_id == 0)
    sec_mask = parent_id > 0

    sum_w_prim, _ = np.histogram(depth_mm[prim_mask], bins=z_edges, weights=weights[prim_mask])
    sum_wE_prim, _ = np.histogram(depth_mm[prim_mask], bins=z_edges, weights=weights[prim_mask] * kin_mev[prim_mask])
    avg_prim_mev = np.divide(sum_wE_prim, sum_w_prim, out=np.zeros_like(sum_wE_prim), where=sum_w_prim > 0)

    sum_w_sec, _ = np.histogram(depth_mm[sec_mask], bins=z_edges, weights=weights[sec_mask])
    sum_wE_sec, _ = np.histogram(depth_mm[sec_mask], bins=z_edges, weights=weights[sec_mask] * kin_mev[sec_mask])
    avg_sec_mev = np.divide(sum_wE_sec, sum_w_sec, out=np.zeros_like(sum_wE_sec), where=sum_w_sec > 0)

    if args.print:
        print("depth_mm  dose_MGy_yr  avg_primary_MeV  avg_secondary_MeV")
        for zc, dose, avgP, avgS in zip(z_centers, dose_mgy_per_yr, avg_prim_mev, avg_sec_mev):
            print(f"{zc:.6g} {dose:.6g} {avgP:.6g} {avgS:.6g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(z_centers, dose_mgy_per_yr, color="k")
    axes[0].set_ylabel("Dose (MGy/yr)")
    axes[0].set_title("Energy deposition vs depth")

    axes[1].plot(z_centers, avg_prim_mev, color="tab:blue")
    axes[1].set_ylabel("Avg E$_{primary}$ (MeV)")
    axes[1].set_title("Average primary energy vs depth")

    axes[2].plot(z_centers, avg_sec_mev, color="tab:red")
    axes[2].set_ylabel("Avg E$_{secondary}$ (MeV)")
    axes[2].set_xlabel("Depth (mm)")
    axes[2].set_title("Average secondary energy vs depth")

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
