#!/usr/bin/env python3
"""Plot simulation vs reference cross sections per channel.

Minimal, model-agnostic comparator:
- Reads ROOT step tree.
- Uses model_ref entries from ROOT config (no fallbacks).
- For each (process, model) pair, plots per-channel XS vs reference.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import uproot

from root_utils import resolve_root_paths
FONTSIZE = 18
EMFIETZOGLOU_SCALE_1E16 = 1e16

plt.rcParams.update({
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE * 0.8,
})


def _resolve_path(path: str | Path) -> str:
    p = Path(path).expanduser()
    return str(p)


def _decode_to_str_array(arr: np.ndarray) -> np.ndarray | None:
    if arr is None:
        return None
    if arr.dtype.kind in {"S", "U"}:
        return arr.astype(str)
    try:
        return np.array([bytes(x).decode("utf-8", "ignore").strip("\x00") if isinstance(x, (bytes, bytearray))
                         else str(x) for x in arr], dtype=object)
    except Exception:
        return None


def load_arrays(root_path: str, tree_name: str = "step") -> dict:
    paths = resolve_root_paths(root_path)
    if not paths:
        raise FileNotFoundError(root_path)
    with uproot.open(paths[0]) as f:
        if tree_name not in f:
            raise RuntimeError(f"Tree '{tree_name}' not found in {paths[0]}")
        t = f[tree_name]
        wanted = [
            "flagProcess",
            "kineticEnergy",
            "macroCrossSection",
            "processCrossSection",
            "vibCrossSection",
            "channelIndex",
            "channelMicroXS",
            "modelName",
        ]
        available = [k for k in wanted if k in t.keys()]
    tree_spec = [f"{p}:{tree_name}" for p in paths]
    return uproot.concatenate(tree_spec, available, library="np")


def load_config(root_path: str, tree_name: str = "config") -> dict[str, str]:
    paths = resolve_root_paths(root_path)
    if not paths:
        return {}
    with uproot.open(paths[0]) as f:
        if tree_name not in f:
            return {}
        t = f[tree_name]
        if "key" not in t.keys() or "value" not in t.keys():
            return {}
        keys = t["key"].array(library="np")
        vals = t["value"].array(library="np")
    cfg: dict[str, str] = {}
    for k, v in zip(keys, vals):
        key = _clean_string(k)
        val = _clean_string(v)
        if key:
            cfg[key] = val
    return cfg


def _clean_string(val: object) -> str:
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8", "ignore").strip("\x00").strip()
        except Exception:
            return ""
    if val is None:
        return ""
    return str(val).strip("\x00").strip()


def resolve_model_refs(config: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, val in config.items():
        if not key.startswith("model_ref:"):
            continue
        model = key.split("model_ref:", 1)[1].strip()
        if not model:
            continue
        path = _resolve_reference_from_value(val)
        if path:
            out[model] = path
    return out


def _resolve_reference_from_value(value: str | None) -> str | None:
    if not value:
        return None
    val = _clean_string(value)
    if not val:
        return None
    if os.path.isabs(val) and os.path.exists(val):
        return val
    candidate = _resolve_path(val)
    if os.path.exists(candidate):
        return candidate
    # As a last resort, try basename in G4LEDATA/dna
    base = os.path.basename(val)
    ledata = os.environ.get("G4LEDATA")
    if ledata:
        cand = Path(ledata) / "dna" / base
        if cand.exists():
            return str(cand)
    return None


def load_reference(path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Load 1D (E + channels) or 2D (E, W, channels) tables."""
    path = _resolve_path(path)
    E: list[float] = []
    rows: list[list[float]] = []
    max_cols = 0
    with open(path, "r") as fin:
        for line in fin:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            e = float(parts[0])
            vals = [float(x) for x in parts[1:]]
            if not vals:
                continue
            E.append(e)
            rows.append(vals)
            max_cols = max(max_cols, len(vals))

    if not rows:
        return np.asarray([], dtype=float), []

    E_arr = np.asarray(E, dtype=float)
    unique_e = np.unique(E_arr)
    has_repeats = unique_e.size < E_arr.size
    is_diff = has_repeats and max_cols >= 2

    if is_diff:
        from collections import defaultdict
        grouped: dict[float, list[list[float]]] = defaultdict(list)
        for e, vals in zip(E_arr.tolist(), rows):
            grouped[e].append(vals)
        e_sorted = np.array(sorted(grouped.keys()), dtype=float)
        n_ch = max(len(v) for vv in grouped.values() for v in vv) - 1
        ref_by_ch = [np.zeros_like(e_sorted, dtype=float) for _ in range(max(n_ch, 0))]
        for i, e in enumerate(e_sorted):
            block = grouped[e]
            w = np.array([v[0] for v in block], dtype=float)
            order = np.argsort(w)
            w = w[order]
            for ch in range(n_ch):
                y = np.array([v[ch + 1] if (ch + 1) < len(v) else 0.0 for v in block], dtype=float)[order]
                ref_by_ch[ch][i] = np.trapezoid(y, w) if w.size > 1 else 0.0
        return e_sorted, ref_by_ch

    ref_by_ch: list[np.ndarray] = []
    for j in range(max_cols):
        col = np.array([row[j] if j < len(row) else 0.0 for row in rows], dtype=float)
        ref_by_ch.append(col)
    return E_arr, ref_by_ch


def scale_reference_if_needed(path: str, ref_by_ch: List[np.ndarray]) -> List[np.ndarray]:
    base = os.path.basename(path).lower()
    if "emfietzoglou" in base:
        return [arr * EMFIETZOGLOU_SCALE_1E16 for arr in ref_by_ch]
    return ref_by_ch


def to_micro_cm2(xs_macro_mm_inv: np.ndarray, nH2O_cm3: float) -> np.ndarray:
    return (xs_macro_mm_inv * 10.0) / float(nH2O_cm3)


def bin_means(x: np.ndarray, y: np.ndarray, nbins: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.asarray([]), np.asarray([])
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    if not np.any(valid):
        return np.asarray([]), np.asarray([])
    x = x[valid]
    y = y[valid]
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if x_max <= x_min:
        return np.asarray([]), np.asarray([])
    if x_max / max(x_min, 1e-30) > 1.2:
        bins = np.geomspace(x_min, x_max, nbins + 1)
    else:
        bins = np.linspace(x_min, x_max, nbins + 1)
    idx = np.digitize(x, bins) - 1
    in_range = (idx >= 0) & (idx < nbins)
    idx = idx[in_range]
    x = x[in_range]
    y = y[in_range]
    centers = 0.5 * (bins[:-1] + bins[1:])
    xs: list[float] = []
    ys: list[float] = []
    for b in range(nbins):
        m = (idx == b)
        if not np.any(m):
            continue
        xs.append(centers[b])
        ys.append(float(np.nanmean(y[m])))
    return np.asarray(xs), np.asarray(ys)


def estimate_by_counts(
    ke: np.ndarray,
    total_xs: np.ndarray,
    ch_idx: np.ndarray,
    n_channels: int,
    nbins: int = 60,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    valid = np.isfinite(ke) & np.isfinite(total_xs) & (ke > 0.0) & (total_xs >= 0.0) & (ch_idx >= 0)
    if not np.any(valid):
        return {}
    ke = ke[valid]
    total_xs = total_xs[valid]
    ch_idx = ch_idx[valid]

    if ke.size == 0:
        return {}
    e_min = float(np.nanmin(ke))
    e_max = float(np.nanmax(ke))
    if e_max <= e_min:
        return {}
    bins = np.geomspace(e_min, e_max, nbins + 1) if e_max / max(e_min, 1e-30) > 1.2 else np.linspace(e_min, e_max, nbins + 1)
    bin_idx = np.digitize(ke, bins) - 1
    in_range = (bin_idx >= 0) & (bin_idx < nbins)
    bin_idx = bin_idx[in_range]
    ke = ke[in_range]
    total_xs = total_xs[in_range]
    ch_idx = ch_idx[in_range]
    centers = 0.5 * (bins[:-1] + bins[1:])

    series: Dict[int, Tuple[List[float], List[float]]] = {c: ([], []) for c in range(n_channels)}
    for b in range(nbins):
        m = (bin_idx == b)
        if not np.any(m):
            continue
        mean_total = float(np.nanmean(total_xs[m]))
        counts = np.bincount(ch_idx[m], minlength=n_channels).astype(float)
        count_total = float(counts.sum())
        if count_total <= 0.0:
            continue
        for c in range(n_channels):
            series[c][0].append(centers[b])
            series[c][1].append(mean_total * (counts[c] / count_total))

    return {c: (np.asarray(x), np.asarray(y)) for c, (x, y) in series.items() if len(x)}


def estimate_by_micro(
    ke: np.ndarray,
    ch_idx: np.ndarray,
    ch_micro: np.ndarray,
    n_channels: int,
    nbins: int = 60,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    valid = np.isfinite(ke) & (ke > 0.0) & (ch_idx >= 0) & (ch_micro > 0.0)
    if not np.any(valid):
        return {}
    ke = ke[valid]
    ch_idx = ch_idx[valid]
    ch_micro = ch_micro[valid]

    series: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for c in range(n_channels):
        m = (ch_idx == c)
        if not np.any(m):
            continue
        x, y = bin_means(ke[m], ch_micro[m], nbins=nbins)
        if x.size:
            series[c] = (x, y)
    return series


def get_macro_xs(arrs: dict, mask: np.ndarray) -> np.ndarray:
    for key in ("macroCrossSection", "processCrossSection", "vibCrossSection"):
        if key in arrs:
            return np.asarray(arrs[key], dtype=float)[mask]
    raise RuntimeError("No macroscopic cross section field found in ROOT.")


def sanitize_label(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def plot_process(
    ke: np.ndarray,
    total_xs_micro: np.ndarray,
    ch_idx: np.ndarray,
    ch_micro: np.ndarray | None,
    ref_E: np.ndarray,
    ref_by_ch: List[np.ndarray],
    scale: float,
    out_path: str,
    title: str,
    loglog: bool,
    channel_method: str,
):
    n_ref = len(ref_by_ch)
    n_sim = int(np.max(ch_idx)) + 1 if ch_idx.size and np.any(ch_idx >= 0) else 0
    n_channels = max(n_ref, n_sim)

    if n_channels <= 0:
        raise RuntimeError("No channels found in simulation or reference.")

    if channel_method == "micro" and ch_micro is None:
        raise RuntimeError("Requested channel_method=micro but channelMicroXS missing.")

    use_micro = channel_method == "micro"
    if channel_method == "auto":
        use_micro = (ch_micro is not None) and np.any(ch_micro > 0.0)

    if use_micro:
        sim_by_ch = estimate_by_micro(ke, ch_idx, ch_micro, n_channels)
    else:
        sim_by_ch = estimate_by_counts(ke, total_xs_micro, ch_idx, n_channels)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.tab10 if n_channels <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, n_channels))

    # Reference
    for ch in range(n_channels):
        if ch >= len(ref_by_ch):
            continue
        ax.plot(ref_E, ref_by_ch[ch], color=colors[ch], linewidth=2.0, label=f"ch{ch} ref")

    # Simulation
    for ch in range(n_channels):
        if ch not in sim_by_ch:
            continue
        x, y = sim_by_ch[ch]
        ax.plot(x, y * scale, linestyle="--", marker="o", markersize=3, color=colors[ch], label=f"ch{ch} sim")

    ax.set_xlabel("Kinetic Energy (eV)")
    ax.set_ylabel("Cross Section (10$^{-16}$ cm$^{2}$)")
    ax.set_title(title)
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.legend(loc="best", frameon=True)

    out_path = _resolve_path(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot simulation vs reference cross sections per channel.")
    ap.add_argument("--root", default="build/dna.root", help="ROOT file path")
    ap.add_argument("--processes", default="all", help="Comma list of process codes or 'all'")
    ap.add_argument("--out", default="xs_channels.png", help="Base output filename")
    ap.add_argument("--scale", type=float, default=1e16, help="Scale factor for sim XS (default: 1e16)")
    ap.add_argument("--nH2O_cm3", type=float, default=3.343e22, help="Number density (cm^-3)")
    ap.add_argument("--channel-method", choices=["auto", "counts", "micro"], default="auto",
                    help="How to derive per-channel sim XS")
    ap.add_argument("--dat", default=None, help="Explicit reference .dat (overrides model_ref)")
    args = ap.parse_args()

    arrs = load_arrays(args.root, "step")
    config = load_config(args.root)
    model_refs = resolve_model_refs(config)

    if not model_refs and not args.dat:
        raise RuntimeError("ROOT is missing model_ref entries and no --dat provided.")

    flag_proc = np.asarray(arrs["flagProcess"], dtype=float)
    if args.processes.strip().lower() == "all":
        proc_list = np.unique(flag_proc).astype(int).tolist()
    else:
        proc_list = [int(x.strip()) for x in args.processes.split(",") if x.strip()]

    model_names = _decode_to_str_array(arrs.get("modelName")) if "modelName" in arrs else None
    base, ext = os.path.splitext(args.out)
    ext = ext or ".png"

    only_ref = next(iter(model_refs.values())) if len(model_refs) == 1 else None

    for pcode in proc_list:
        mask_proc = (flag_proc == float(pcode))
        if not np.any(mask_proc):
            continue

        # Group by model name if available
        if model_names is not None:
            models = np.unique(model_names[mask_proc])
            models = [m for m in models if m]
            if not models:
                models = [None]
        else:
            models = [None]

        if models == [None] and not args.dat and only_ref is None:
            raise RuntimeError(\"modelName missing/empty in ROOT and multiple/no model_ref entries present.\")

        for mname in models:
            mask = mask_proc
            if model_names is not None and mname is not None:
                mask = mask & (model_names == mname)

            if not np.any(mask):
                continue

            ref_path = args.dat
            if ref_path is None:
                if mname is None:
                    ref_path = only_ref
                else:
                    ref_path = model_refs.get(mname)
            if not ref_path:
                raise RuntimeError(f"Missing model_ref for model '{mname}' (process {pcode}).")

            ref_E, ref_by_ch = load_reference(ref_path)
            ref_by_ch = scale_reference_if_needed(ref_path, ref_by_ch)
            if ref_E.size == 0 or not ref_by_ch:
                raise RuntimeError(f"Reference file is empty: {ref_path}")

            ke = np.asarray(arrs["kineticEnergy"], dtype=float)[mask]
            if "channelIndex" in arrs:
                ch_idx = np.asarray(arrs["channelIndex"], dtype=int)[mask]
            else:
                ch_idx = np.full_like(ke, -1, dtype=int)
            if not np.any(ch_idx >= 0):
                if len(ref_by_ch) <= 1:
                    ch_idx = np.zeros_like(ke, dtype=int)
                else:
                    raise RuntimeError("channelIndex missing/invalid but reference has multiple channels.")

            ch_micro = None
            if "channelMicroXS" in arrs:
                ch_micro = np.asarray(arrs["channelMicroXS"], dtype=float)[mask]

            macro = get_macro_xs(arrs, mask)
            total_xs_micro = to_micro_cm2(macro, args.nH2O_cm3)

            msafe = sanitize_label(mname) if mname else "nomodel"
            out_path = f"{base}_proc{pcode}_{msafe}{ext}"
            title = f"proc{pcode} {mname or ''}".strip()
            loglog = int(pcode) in (12, 13)

            plot_process(
                ke=ke,
                total_xs_micro=total_xs_micro,
                ch_idx=ch_idx,
                ch_micro=ch_micro,
                ref_E=ref_E,
                ref_by_ch=ref_by_ch,
                scale=args.scale,
                out_path=out_path,
                title=title,
                loglog=loglog,
                channel_method=args.channel_method,
            )


if __name__ == "__main__":
    main()
