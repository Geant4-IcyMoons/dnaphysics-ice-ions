#!/usr/bin/env python3
"""
Plot stopping power from a Geant4-DNA ROOT file (tree: "step").

Computes dE/dx from per-step data and bins it versus kinetic energy.
Uses the same plotting style as plot_elastic_cross_sections.py.
Stores per-material stopping power ranges in NPZ and stitches ranges for plotting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uproot

from root_utils import resolve_root_paths
from constants import (
    FONT_COURIER,
    FONTSIZE_24,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RC_BASE_ELASTIC,
    rcparams_with_fontsize,
)

# Font configuration (match plot_elastic_cross_sections.py)
font = FONT_COURIER
plt.rcParams["font.family"] = font
plt.rcParams["mathtext.rm"] = font
plt.rcParams["mathtext.fontset"] = "custom"
FONTSIZE = FONTSIZE_24
plt.rcParams.update(rcparams_with_fontsize(RC_BASE_ELASTIC, FONTSIZE))

# Unit conversion: (eV / nm) -> (MeV / cm), (eV / nm) -> (eV / angstrom)
EV_NM_TO_MEV_CM = 10.0
EV_NM_TO_EV_ANG = 0.1
PROC_ORDER = ("Vib. Exc.", "Excit.", "Ion.", "De-Att.", "Elastic", "Solv.")
# False -> ignore cached NPZ ranges and plot only the current ROOT run.
LOAD_EXISTING_RANGES = False


def _load_step_tree(path: Path, tree_name: str) -> dict[str, np.ndarray]:
    paths = resolve_root_paths(path)
    if not paths:
        raise FileNotFoundError(path)
    with uproot.open(paths[0]) as f:
        if tree_name not in f:
            raise KeyError(f"Tree '{tree_name}' not found in {paths[0]}")
        tree = f[tree_name]
        cols = [
            "kineticEnergy",
            "kineticEnergyDifference",
            "totalEnergyDeposit",
            "stepLength",
            "trackID",
            "parentID",
            "flagParticle",
            "flagProcess",
            "processName",
        ]
        available = [name for name in cols if name in tree.keys()]
    tree_spec = [f"{p}:{tree_name}" for p in paths]
    data = uproot.concatenate(tree_spec, available, library="np")
    return data


def _decode_str_array(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    out = []
    for val in arr:
        if isinstance(val, (bytes, bytearray)):
            s = val.decode(errors="ignore")
        else:
            s = str(val)
        if "\x00" in s:
            s = s.split("\x00", 1)[0]
        out.append(s.strip())
    return np.asarray(out, dtype=str)


def _binned_sum(x: np.ndarray, w: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return np.histogram(x, bins=bins, weights=w)[0]


def _units_and_label(units: str, density: float, dedx: np.ndarray) -> tuple[np.ndarray, str]:
    if units == "ev_ang":
        return dedx * EV_NM_TO_EV_ANG, "Stopping Power (eV/Å)"
    if units == "mev_cm":
        return dedx * EV_NM_TO_MEV_CM, "Stopping Power (MeV/cm)"
    if units == "mev_cm2_g":
        return (dedx * EV_NM_TO_MEV_CM) / density, r"Mass Stopping Power (MeV cm$^2$/g)"
    return dedx, "Stopping Power (eV/nm)"


def _process_category_masks(
    proc_names: np.ndarray | None, flag_proc: np.ndarray | None
) -> list[tuple[str, np.ndarray]]:
    categories: list[tuple[str, np.ndarray]] = []
    if proc_names is not None:
        pname = np.char.lower(proc_names)
        is_vib = np.char.find(pname, "vib") >= 0
        is_attach = np.char.find(pname, "attach") >= 0
        is_ion = (np.char.find(pname, "ionis") >= 0) | (np.char.find(pname, "ioniz") >= 0)
        is_exc = (np.char.find(pname, "excitation") >= 0) & ~is_vib
        is_elastic = np.char.find(pname, "elastic") >= 0
        is_solv = np.char.find(pname, "solvation") >= 0
        categories = [
            ("Vib. Exc.", is_vib),
            ("Excit.", is_exc),
            ("Ion.", is_ion),
            ("De-Att.", is_attach),
            ("Elastic", is_elastic),
            ("Solv.", is_solv),
        ]
    elif flag_proc is not None:
        fp = np.rint(flag_proc).astype(int)
        categories = [
            ("Vib. Exc.", fp == 15),
            ("Excit.", fp == 12),
            ("Ion.", fp == 13),
            ("De-Att.", fp == 14),
            ("Elastic", fp == 11),
            ("Solv.", fp == 10),
        ]
    return [(label, mask) for label, mask in categories if np.any(mask)]


def _compute_stopping_power_data(
    root_path: Path,
    tree_name: str,
    nbins: int,
    emin: float | None,
    emax: float | None,
    use_edep: bool,
    all_tracks: bool,
    particle: int | None,
    split_by_process: bool,
) -> dict[str, object]:
    data = _load_step_tree(root_path, tree_name)
    kin_e = data["kineticEnergy"]
    dE = data["totalEnergyDeposit"] if use_edep else data["kineticEnergyDifference"]
    dx = data["stepLength"]
    proc_names = _decode_str_array(data.get("processName"))
    flag_proc = data.get("flagProcess")

    mask = np.isfinite(kin_e) & np.isfinite(dE) & np.isfinite(dx)
    mask &= (kin_e > 0) & (dx > 0)
    if not all_tracks:
        if "trackID" in data and "parentID" in data:
            mask &= (data["trackID"] == 1) & (data["parentID"] == 0)
    if particle is not None and "flagParticle" in data:
        mask &= (data["flagParticle"] == particle)

    kin_e = kin_e[mask]
    dE = dE[mask]
    dx = dx[mask]
    if proc_names is not None:
        proc_names = proc_names[mask]
    if flag_proc is not None:
        flag_proc = np.asarray(flag_proc, dtype=float)[mask]

    if kin_e.size == 0:
        raise ValueError("No valid steps after filtering.")

    dE_use = np.where(dE > 0, dE, 0.0)

    if emin is None:
        emin = float(np.min(kin_e))
    if emax is None:
        emax = float(np.max(kin_e))
    if emin <= 0 or emax <= 0 or emin >= emax:
        raise ValueError(f"Invalid energy range: emin={emin}, emax={emax}")

    bins = np.logspace(np.log10(emin), np.log10(emax), nbins + 1)
    centers = np.sqrt(bins[:-1] * bins[1:])
    total_dx = _binned_sum(kin_e, dx, bins)
    total_dE = _binned_sum(kin_e, dE_use, bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        total_dedx = np.where(total_dx > 0, total_dE / total_dx, np.nan)

    process_dedx: dict[str, np.ndarray] = {}
    if split_by_process:
        categories = _process_category_masks(proc_names, flag_proc)
        for cat_label, cat_mask in categories:
            dE_cat = _binned_sum(kin_e[cat_mask], dE_use[cat_mask], bins)
            with np.errstate(divide="ignore", invalid="ignore"):
                dedx_cat = np.where(total_dx > 0, dE_cat / total_dx, np.nan)
            process_dedx[cat_label] = dedx_cat

    return {
        "emin": float(emin),
        "emax": float(emax),
        "nbins": int(nbins),
        "centers_eV": centers,
        "total_dedx": total_dedx,
        "process_dedx": process_dedx,
    }


def _ordered_process_labels(labels: list[str]) -> list[str]:
    ordered = [name for name in PROC_ORDER if name in labels]
    extras = sorted(name for name in labels if name not in PROC_ORDER)
    return ordered + extras


def _plot_stopping_power_series(
    total_energy: np.ndarray,
    total_dedx: np.ndarray,
    process_series: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    units: str,
    density: float,
    label: str,
    split_by_process: bool,
) -> None:
    if total_energy.size == 0:
        raise ValueError("No stopping power data to plot.")

    dedx_plot_total, ylabel = _units_and_label(units, density, total_dedx)

    fig, ax = plt.subplots(figsize=(10, 6))
    if split_by_process and process_series:
        labels = _ordered_process_labels(list(process_series.keys()))
        cmap = plt.cm.plasma
        n = len(labels)
        for i, cat_label in enumerate(labels):
            cat_energy, cat_dedx = process_series[cat_label]
            dedx_plot, _ = _units_and_label(units, density, cat_dedx)
            valid = np.isfinite(dedx_plot) & (dedx_plot > 0) & np.isfinite(cat_energy)
            if not np.any(valid):
                continue
            t = 0.5 if n == 1 else (i / (n - 1)) * 0.85
            color = cmap(t)
            ax.loglog(
                cat_energy[valid],
                dedx_plot[valid],
                color=color,
                linewidth=3,
                label=cat_label,
                zorder=3,
            )
    else:
        split_by_process = False

    valid = np.isfinite(dedx_plot_total) & (dedx_plot_total > 0) & np.isfinite(total_energy)
    ax.loglog(
        total_energy[valid],
        dedx_plot_total[valid],
        color="black",
        linewidth=4 if split_by_process else 2,
        label=label,
        zorder=4 if split_by_process else 3,
    )

    ax.set_xlabel("Electron Energy ($T$; eV)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1., np.max(total_energy))
    ax.legend(loc="best")
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.show()


def plot_stopping_power(
    root_path: Path,
    out_path: Path,
    tree_name: str,
    nbins: int,
    emin: float | None,
    emax: float | None,
    use_edep: bool,
    all_tracks: bool,
    particle: int | None,
    units: str,
    density: float,
    label: str,
    split_by_process: bool,
) -> None:
    range_data = _compute_stopping_power_data(
        root_path=root_path,
        tree_name=tree_name,
        nbins=nbins,
        emin=emin,
        emax=emax,
        use_edep=use_edep,
        all_tracks=all_tracks,
        particle=particle,
        split_by_process=split_by_process,
    )
    centers = np.asarray(range_data["centers_eV"])
    total_dedx = np.asarray(range_data["total_dedx"])
    process_dedx = range_data["process_dedx"]
    process_series = {label: (centers, arr) for label, arr in process_dedx.items()}

    _plot_stopping_power_series(
        total_energy=centers,
        total_dedx=total_dedx,
        process_series=process_series,
        out_path=out_path,
        units=units,
        density=density,
        label=label,
        split_by_process=split_by_process,
    )
    out_path_total = _total_only_out_path(out_path)
    _plot_stopping_power_series(
        total_energy=centers,
        total_dedx=total_dedx,
        process_series=process_series,
        out_path=out_path_total,
        units=units,
        density=density,
        label=label,
        split_by_process=False,
    )


def _default_root_path() -> Path:
    return PROJECT_ROOT / "build" / "dna.root"


def _default_npz_path(material: str) -> Path:
    return OUTPUT_DIR / f"stopping_power_{material}.npz"


def _total_only_out_path(out_path: Path) -> Path:
    if out_path.suffix:
        return out_path.with_name(f"{out_path.stem}_total{out_path.suffix}")
    return out_path.with_name(f"{out_path.name}_total")


def _format_range_key(emin: float, emax: float) -> str:
    return f"{emin:.6e}_{emax:.6e}"


def _find_matching_range_key(
    ranges: dict[str, dict[str, object]], emin: float, emax: float
) -> str | None:
    for key, info in ranges.items():
        stored_emin = float(info.get("emin", np.nan))
        stored_emax = float(info.get("emax", np.nan))
        if np.isclose(emin, stored_emin, rtol=1.0e-8, atol=0.0) and np.isclose(
            emax, stored_emax, rtol=1.0e-8, atol=0.0
        ):
            return key
    return None


def _load_stopping_power_store(npz_path: Path) -> dict[str, object]:
    if not npz_path.exists():
        return {"ranges": {}}
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "data" not in data:
                return {"ranges": {}}
            store = data["data"].item()
    except Exception:
        return {"ranges": {}}
    if not isinstance(store, dict):
        return {"ranges": {}}
    if "ranges" not in store or not isinstance(store["ranges"], dict):
        store["ranges"] = {}
    return store


def _save_stopping_power_store(npz_path: Path, store: dict[str, object]) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, data=store)


def _update_stopping_power_store(
    store: dict[str, object],
    material: str,
    range_data: dict[str, object],
) -> str:
    store["material"] = material
    ranges = store.setdefault("ranges", {})
    emin = float(range_data["emin"])
    emax = float(range_data["emax"])
    key = _find_matching_range_key(ranges, emin, emax)
    if key is None:
        key = _format_range_key(emin, emax)
    ranges[key] = range_data
    return key


def _collect_combined_series(
    store: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    ranges = list(store.get("ranges", {}).values())
    if not ranges:
        raise ValueError("No stored stopping power ranges found.")

    total_energy = np.concatenate([np.asarray(r["centers_eV"]) for r in ranges])
    total_dedx = np.concatenate([np.asarray(r["total_dedx"]) for r in ranges])
    total_sort = np.argsort(total_energy)
    total_energy = total_energy[total_sort]
    total_dedx = total_dedx[total_sort]

    proc_map: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for r in ranges:
        centers = np.asarray(r["centers_eV"])
        process_dedx = r.get("process_dedx", {})
        if not isinstance(process_dedx, dict):
            continue
        for label, dedx in process_dedx.items():
            proc_map.setdefault(label, []).append((centers, np.asarray(dedx)))

    proc_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, chunks in proc_map.items():
        energies = np.concatenate([c[0] for c in chunks])
        values = np.concatenate([c[1] for c in chunks])
        order = np.argsort(energies)
        proc_series[label] = (energies[order], values[order])

    return total_energy, total_dedx, proc_series


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stopping power from dnaphysics ROOT output.")
    parser.add_argument("--root", type=Path, default=None, help="Path or glob to ROOT file(s)")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument(
        "--material",
        type=str,
        default="ice",
        choices=("water", "ice"),
        help="Material label for saving and merging stopping power ranges.",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Optional NPZ path for stopping power storage (overrides material default).",
    )
    parser.add_argument("--tree", type=str, default="step", help="Tree name in ROOT file")
    parser.add_argument("--nbins", type=int, default=80, help="Number of log-spaced energy bins")
    parser.add_argument("--emin", type=float, default=None, help="Minimum energy (eV)")
    parser.add_argument("--emax", type=float, default=None, help="Maximum energy (eV)")
    parser.add_argument(
        "--use-edep",
        action="store_true",
        help="Use totalEnergyDeposit instead of kineticEnergyDifference (LET).",
    )
    parser.add_argument(
        "--all-tracks",
        action="store_true",
        help="Include all tracks (default: primary only).",
    )
    parser.add_argument(
        "--particle",
        type=int,
        default=None,
        help="Filter by flagParticle value (e.g., 1 for electron).",
    )
    parser.add_argument(
        "--units",
        type=str,
        default="ev_ang",
        choices=("ev_ang", "ev_nm", "mev_cm", "mev_cm2_g"),
        help="Output units for stopping power.",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="Density (g/cm^3) for mass stopping power.",
    )
    parser.add_argument(
        "--total-only",
        action="store_true",
        help="Plot only total stopping power (no per-process breakdown).",
    )
    parser.add_argument("--label", type=str, default="Cumulative", help="Legend label for total")
    args = parser.parse_args()

    root_path = args.root if args.root is not None else _default_root_path()
    material = args.material.lower()
    npz_path = args.npz if args.npz is not None else _default_npz_path(material)
    out_path = (
        args.out if args.out is not None else OUTPUT_DIR / f"stopping_power_{material}.png"
    )

    range_data = _compute_stopping_power_data(
        root_path=root_path,
        tree_name=args.tree,
        nbins=args.nbins,
        emin=args.emin,
        emax=args.emax,
        use_edep=args.use_edep,
        all_tracks=args.all_tracks,
        particle=args.particle,
        split_by_process=True,
    )

    if LOAD_EXISTING_RANGES:
        store = _load_stopping_power_store(npz_path)
    else:
        store = {"ranges": {}}
    key = _update_stopping_power_store(store, material, range_data)
    _save_stopping_power_store(npz_path, store)
    print(
        f"\nSaved stopping power range {range_data['emin']:.6e}–{range_data['emax']:.6e} eV "
        f"to {npz_path} (key: {key})."
    )

    total_energy, total_dedx, process_series = _collect_combined_series(store)
    _plot_stopping_power_series(
        total_energy=total_energy,
        total_dedx=total_dedx,
        process_series=process_series,
        out_path=out_path,
        units=args.units,
        density=args.density,
        label=args.label,
        split_by_process=True,
    )
    out_path_total = _total_only_out_path(out_path)
    _plot_stopping_power_series(
        total_energy=total_energy,
        total_dedx=total_dedx,
        process_series=process_series,
        out_path=out_path_total,
        units=args.units,
        density=args.density,
        label=args.label,
        split_by_process=False,
    )


if __name__ == "__main__":
    main()
