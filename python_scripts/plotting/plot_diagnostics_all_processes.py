#!/usr/bin/env python3
"""
Diagnostics plotting utilities for dnaphysics-ice.

Features
- Per-process cross-section plots (one panel per process; optional reference overlay).
- Excitation/ionisation overlays (one panel each; all channels + cumulative, sim vs reference).
- Elastic XS reference vs simulation multi-panel (one panel per elastic process found).
- Vibrational energy-loss histograms per channel (for process 15).
- Deflection-angle distributions for all (process, channel) pairs.
- Compact 4-panel simulation summary.

Inputs
- ROOT: Expects a tree named "step" with columns:
  flagParticle, kineticEnergy, flagProcess, vibCrossSection, channelIndex,
  channelMicroXS, kineticEnergyDifference, cosTheta, x, y, z.
- Reference .dat (optional): First column energy (eV) followed by M partial XS
  columns (10^-16 cm^2). For vib, M=8 (channels 0..7). For elastic, M=1.

Usage
- python plot_root_vibExcitation.py --root build/dna.root --processes 15 --out xs_channels.png
- python plot_root_vibExcitation.py --root build/dna.root --processes all --out xs_channels.png
- python plot_root_vibExcitation.py --dat /abs/path/to/sigma_excitationvib_e_michaud.dat --out xs_channels.png

Notes
- If --dat is not provided, the script tries to locate reference files under
  $G4LEDATA/dna or a local g4_custom_ice install.
- Designed to be non-interactive (saves figures; no GUI required).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List
import warnings

import numpy as np
import matplotlib.pyplot as plt
import uproot

from root_utils import resolve_root_paths, resolve_first_root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from physics_ice.constants import (
    CROSS_SECTIONS_DIR,
    CUSTOM_DATA_ROOT_GEANT4,
    EMFI_EXCITATION_EEV,
    EMFI_EXCITATION_TOL_EEV,
    EMFI_ION_BINDING_EEV,
    EV_TO_MEV,
    FM2_TO_CM2,
    FONT_COURIER,
    FONTSIZE_16,
    GEANT4_PROJECTS_ROOT,
    HFONT_COURIER,
    MICHAUD_TABLE2_PATH,
    MICHAUD_TABLE3_PATH,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RC_BASE_STANDARD,
    SR_ALPHA_1,
    SR_BETA_1,
    SR_CONST_K,
    SR_E_SQUARED_MEV_FM,
    SR_ELECTRON_MASS_MEV,
    SR_Z_WATER,
    rcparams_with_fontsize,
)

font = FONT_COURIER
hfont = HFONT_COURIER
plt.rcParams['font.family'] = font
plt.rcParams['mathtext.rm'] = font
plt.rcParams['mathtext.fontset'] = 'custom'
FONTSIZE = FONTSIZE_16
plt.rcParams.update(rcparams_with_fontsize(RC_BASE_STANDARD, FONTSIZE))

# Emfietzoglou tables use a model scale factor in Geant4 DNA.
# Convert raw table values to cm^2, then apply the same plot scaling as sim.
EMFI_REF_TO_CM2 = (1e-22 / 3.343) * 1e4

# -------- Paths and data locations --------
# This script lives under dnaphysics-ice/python_scripts
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Roots for locating data
CUSTOM_DATA_ROOT = CUSTOM_DATA_ROOT_GEANT4

MICHAUD_TABLE2 = str(MICHAUD_TABLE2_PATH)
MICHAUD_TABLE3 = str(MICHAUD_TABLE3_PATH)

# -------- Small helpers --------
def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return str(PROJECT_ROOT / p)

def _resolve_output(name: str) -> str:
    base = os.path.basename(name) if name else "output.png"
    return str(OUTPUT_DIR / base)


def _find_g4ledata_file(basename: str) -> str | None:
    """Find a file under G4LEDATA/dna or in the local g4_custom_ice install.

    Order:
      1) $G4LEDATA/dna/<basename>
      2) <TOP_ROOT>/g4_custom_ice/install/share/Geant4/data/G4EMLOW*/dna/<basename>
    Returns a string path if found, else None.
    """
    p = CROSS_SECTIONS_DIR / basename
    if p.exists():
        return str(p)
    led = os.environ.get("G4LEDATA")
    if led:
        p = Path(led) / "dna" / basename
        if p.exists():
            return str(p)
    # 2) Local custom install with versioned G4EMLOW dir under project tree
    data_root = CUSTOM_DATA_ROOT
    if data_root.exists():
        for sub in sorted(data_root.iterdir()):
            if sub.is_dir() and sub.name.startswith("G4EMLOW"):
                candidate = sub / "dna" / basename
                if candidate.exists():
                    return str(candidate)
    return None

def _find_backup_dat(basename: str) -> str | None:
    """Fallback: look for reference .dat in backup/geant4_icyMoons."""
    p = GEANT4_PROJECTS_ROOT / "backup" / "geant4_icyMoons" / basename
    return str(p) if p.exists() else None

def load_config_from_root(path: str, tree_name: str = "config") -> dict[str, str]:
    """Load key/value metadata from the ROOT config tree, if present."""
    paths = resolve_root_paths(path)
    if not paths:
        return {}
    with uproot.open(paths[0]) as f:
        if tree_name not in f:
            return {}
        t = f[tree_name]
        if "key" not in t.keys() or "value" not in t.keys():
            return {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="overflow encountered in scalar add", category=RuntimeWarning)
            keys_arr = t["key"].array(library="np")
            vals_arr = t["value"].array(library="np")
        cfg: dict[str, str] = {}
        for k, v in zip(keys_arr, vals_arr):
            key = _clean_string(k)
            val = _clean_string(v)
            if key:
                cfg[key] = val
        return cfg

def _resolve_reference_from_value(value: str | None) -> str | None:
    if not value:
        return None
    val = _clean_string(value)
    if not val:
        return None
    if val.lower() in {"none", "default", "null"}:
        return None
    if os.path.isabs(val) and os.path.exists(val):
        return val
    candidate = _resolve_path(val)
    if os.path.exists(candidate):
        return candidate
    # If this looks like a path fragment, fall back to basename lookup.
    base = os.path.basename(val)
    p = _find_g4ledata_file(base)
    if not p:
        p = _find_backup_dat(base)
    return p


def _prefer_total_ref(path: str | None, pcode: int) -> str | None:
    """If a differential ref is provided for ionisation/excitation, prefer total sigma file."""
    if not path or int(pcode) not in (12, 13):
        return path
    base = os.path.basename(path)
    if base.startswith("sigmadiff_") or base.startswith("sigmadiff_cumulated_"):
        repl = base
        if base.startswith("sigmadiff_cumulated_"):
            repl = base.replace("sigmadiff_cumulated_", "sigma_", 1)
        elif base.startswith("sigmadiff_"):
            repl = base.replace("sigmadiff_", "sigma_", 1)
        cand = _resolve_reference_from_value(repl)
        if cand:
            return cand
    return path

def resolve_config_reference_paths(config: dict[str, str]) -> dict[str, str]:
    """Resolve reference file paths from config keys."""
    keys = [
        "ref_elastic_low",
        "ref_elastic_high",
        "ref_elastic_blend",
        "ref_vib",
        "ref_excitation",
        "ref_excitation_born",
        "ref_ionisation",
        "ref_ionisation_born",
        "ref_attachment",
        "model_ionisation_diff",
        "model_ionisation_diff_born",
    ]
    out: dict[str, str] = {}
    for key in keys:
        path = _resolve_reference_from_value(config.get(key, ""))
        if path:
            out[key] = path
    return out

def resolve_model_reference_paths(config: dict[str, str]) -> dict[str, str]:
    """Extract model->reference mapping from config keys like 'model_ref:<ModelName>'."""
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

def resolve_model_lineshapes(config: dict[str, str]) -> dict[str, dict]:
    """Extract model->lineshape metadata from config keys like 'model_lineshape:<ModelName>'."""
    out: dict[str, dict] = {}
    if not config:
        return out
    for key, val in config.items():
        if not key.startswith("model_lineshape:"):
            continue
        model = key.split("model_lineshape:", 1)[1].strip()
        if not model:
            continue
        try:
            data = json.loads(val)
        except Exception:
            continue
        if isinstance(data, dict):
            out[model] = data
    return out

def _scale_reference_if_needed(path: str | None, ref_by_ch: list[np.ndarray], scale: float) -> list[np.ndarray]:
    """Apply unit fixes for known table formats (e.g., Emfietzoglou) and plot scaling."""
    if not path or not ref_by_ch:
        return ref_by_ch
    base = os.path.basename(path).lower()
    if "emfietzoglou" in base:
        factor = EMFI_REF_TO_CM2 * scale
        return [arr * factor for arr in ref_by_ch]
    return ref_by_ch


def _clean_string(val: object) -> str:
    """Decode bytes, drop null-terminated tail, and strip whitespace."""
    if val is None:
        return ""
    if isinstance(val, (bytes, bytearray)):
        s = val.decode(errors="ignore")
    else:
        s = str(val)
    if "\x00" in s:
        s = s.split("\x00", 1)[0]
    return s.strip()


def _process_name_map() -> dict:
    return {
        10: "Solvation", 11: "Elastic", 12: "Excitation", 13: "Ionisation", 14: "Attachment", 15: "VibExc",
        21: "pElastic", 22: "pExcitation", 23: "pIonisation", 24: "pChgDec",
        31: "HElastic", 32: "HExcitation", 33: "HIonisation", 35: "HChgInc",
        41: "aElastic", 42: "aExcitation", 43: "aIonisation", 44: "aChgDec",
        51: "a+Elastic", 52: "a+Excitation", 53: "a+Ionisation", 54: "a+ChgDec", 55: "a+ChgInc",
        61: "HeElastic", 62: "HeExcitation", 63: "HeIonisation", 65: "HeChgInc",
        73: "GIonIonis", 110: "mscEl", 710: "msc", 720: "Coulomb", 730: "ionIoni", 740: "nucStop"
    }


def _decode_to_str_array(arr: np.ndarray | None) -> np.ndarray | None:
    """Convert a numpy array of bytes/strings into a str array (None -> None)."""
    if arr is None:
        return None
    # Handle fixed-width byte arrays cleanly (e.g., char[32])
    if isinstance(arr, np.ndarray):
        if arr.dtype.kind == "S":
            return np.char.decode(arr, errors="ignore").astype(object)
        if arr.dtype.kind == "U":
            return arr.astype(object)
        if arr.dtype.kind in ("u", "i") and arr.ndim == 2:
            out = []
            for row in arr:
                try:
                    s = bytes(row).decode(errors="ignore")
                except Exception:
                    s = str(row)
                if "\x00" in s:
                    s = s.split("\x00", 1)[0]
                out.append(s.strip())
            return np.asarray(out, dtype=object)
    out: list[str] = []
    for v in np.asarray(arr, dtype=object):
        out.append(_clean_string(v))
    return np.asarray(out, dtype=object)

def _canonicalize_model_names(names: list[str] | set[str]) -> tuple[list[str], dict[str, str]]:
    """Deduplicate model names by collapsing substrings/rotations into a canonical form."""
    cleaned: list[str] = []
    alias: dict[str, str] = {}
    for n in sorted([x for x in names if x], key=len, reverse=True):
        if not n:
            continue
        matched = None
        for c in cleaned:
            if len(n) == len(c) and n in (c + c):
                matched = c
                break
            if n in c:
                matched = c
                break
        if matched is None:
            cleaned.append(n)
            alias[n] = n
        else:
            alias[n] = matched
    return cleaned, alias

def _apply_model_alias_to_meta(meta: dict) -> dict:
    """Collapse model-name aliases inside meta (ke_min/max, chan_max, models_by_proc)."""
    models_all: set[str] = set()
    for names in meta.get("models_by_proc", {}).values():
        models_all.update([n for n in names if n])
    if not models_all:
        return meta
    canon, alias = _canonicalize_model_names(models_all)
    meta["model_alias"] = alias
    # Update models_by_proc
    new_models_by_proc = {}
    for p, names in meta.get("models_by_proc", {}).items():
        new_models_by_proc[p] = set(alias.get(n, n) for n in names if n)
    meta["models_by_proc"] = new_models_by_proc
    # Remap keyed dicts
    def _remap_dict(d: dict, combine):
        out = {}
        for (p, m), v in d.items():
            m2 = alias.get(m, m)
            key = (p, m2)
            if key in out:
                out[key] = combine(out[key], v)
            else:
                out[key] = v
        return out
    meta["ke_min"] = _remap_dict(meta.get("ke_min", {}), min)
    meta["ke_max"] = _remap_dict(meta.get("ke_max", {}), max)
    meta["chan_max"] = _remap_dict(meta.get("chan_max", {}), max)
    meta["has_chan_micro"] = _remap_dict(meta.get("has_chan_micro", {}), lambda a, b: a or b)
    meta = _apply_model_alias_to_meta(meta)
    return meta


def _resolve_elastic_reference_path(
    hints: list[str],
    fallback: str | None = None,
    config_refs: dict[str, str] | None = None,
) -> str | None:
    """Pick the best-matching elastic reference .dat based on model/process hints."""
    if fallback:
        candidate = _resolve_path(fallback)
        if os.path.exists(candidate):
            return candidate
    if not config_refs:
        return None
    for h in hints:
        if not h:
            continue
        m = h.lower()
        if "elsepa_low" in m:
            if "ref_elastic_low" in config_refs:
                return config_refs["ref_elastic_low"]
        elif "elsepa_high" in m:
            if "ref_elastic_high" in config_refs:
                return config_refs["ref_elastic_high"]
        elif "michaud" in m or "champion" in m or "muffin" in m:
            if "ref_elastic_blend" in config_refs:
                return config_refs["ref_elastic_blend"]
    return None


def _sanitize_label(label: str) -> str:
    """Return a filesystem-friendly label."""
    if not label:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(label))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")

def _select_sim_cross_section(arrs, mask, nH2O_cm3: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (ke, xs_micro_cm2) for a given boolean mask."""
    ke = np.asarray(arrs["kineticEnergy"], dtype=float)[mask]
    macro_keys = ["vibCrossSection", "macroCrossSection", "processCrossSection"]
    xs_macro = None
    for k in macro_keys:
        if k in arrs:
            xs_macro = np.asarray(arrs[k], dtype=float)[mask]
            break
    if xs_macro is None:
        xs_macro = np.zeros_like(ke)
    xs_micro_cm2 = _to_micro_cm2(xs_macro, nH2O_cm3)
    if "channelMicroXS" in arrs:
        xs_micro = np.asarray(arrs["channelMicroXS"], dtype=float)[mask]
        xs_micro_cm2 = np.where(xs_micro > 0.0, xs_micro, xs_micro_cm2)
    return ke, xs_micro_cm2

def print_root_processes(arrs) -> None:
    """Print unique processes present in the ROOT file."""
    if arrs is None:
        print("No ROOT data loaded; nothing to list.")
        return
    if "flagProcess" not in arrs:
        print("flagProcess column not found; cannot list processes.")
        return

    flag_proc = np.asarray(arrs["flagProcess"], dtype=int)
    proc_names = _decode_to_str_array(arrs.get("processName"))
    model_names = _decode_to_str_array(arrs.get("modelName"))
    uniq, counts = np.unique(flag_proc, return_counts=True)

    print(f"Found {uniq.size} unique flagProcess entries in ROOT:")
    for code, cnt in zip(uniq, counts):
        label = _process_name_map().get(int(code), f"proc{int(code)}")
        mask = (flag_proc == code)
        names = None
        models = None
        if proc_names is not None:
            names = sorted([n for n in np.unique(proc_names[mask]) if n])
        if model_names is not None:
            models = sorted([m for m in np.unique(model_names[mask]) if m])
        names_str = f" | processName: {', '.join(names)}" if names else ""
        models_str = f" | modelName: {', '.join(models)}" if models else ""
        print(f"  {int(code):>4d}  {label:<12} steps={cnt}{names_str}{models_str}")

# -------- Data I/O --------
def load_arrays(path: str, tree_name: str = "step"):
    """Load ROOT ntuple arrays from the given file, selecting known columns."""
    paths = resolve_root_paths(path)
    if not paths:
        raise FileNotFoundError(path)
    wanted = [
        "flagParticle",
        "kineticEnergy",
        "flagProcess",
        "vibCrossSection",
        "channelIndex",
        "channelMicroXS",
        "kineticEnergyDifference",
        "cosTheta",
        "x","y","z",
        "processName",
        "modelName",
    ]
    # Determine available columns from first file
    with uproot.open(paths[0]) as f:
        if tree_name not in f:
            raise RuntimeError(f"Tree '{tree_name}' not found in {paths[0]}")
        t = f[tree_name]
        available = [name for name in wanted if name in t.keys()]
    tree_spec = [f"{p}:{tree_name}" for p in paths]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow encountered in scalar add", category=RuntimeWarning)
        return uproot.concatenate(tree_spec, available, library="np")

def _iter_root_arrays(path: str, tree_name: str, columns: list[str], step_size: int):
    """Yield ROOT arrays in batches to avoid loading the full tree into memory."""
    paths = resolve_root_paths(path)
    if not paths:
        raise FileNotFoundError(path)
    tree_spec = [f"{p}:{tree_name}" for p in paths]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow encountered in scalar add", category=RuntimeWarning)
        for chunk in uproot.iterate(tree_spec, columns, library="np", step_size=step_size):
            yield chunk


def _update_first_two_primary_steps(
    state: dict[int, tuple[int, float, float, float] | None],
    event_id: int,
    step_id: int,
    x: float,
    y: float,
    z: float,
) -> None:
    key_first = event_id * 2
    key_second = event_id * 2 + 1
    first = state.get(key_first)
    second = state.get(key_second)
    current = (step_id, x, y, z)
    if first is None:
        state[key_first] = current
        return
    if step_id < first[0]:
        state[key_second] = first
        state[key_first] = current
        return
    if step_id == first[0]:
        return
    if second is None or step_id < second[0]:
        state[key_second] = current


def extract_initial_angle_distributions(
    root_arg: str,
    deposition_epsilon_frac: float,
    step_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    paths = resolve_root_paths(root_arg)
    if not paths:
        raise FileNotFoundError(root_arg)

    theta_all: list[float] = []
    theta_incomplete: list[float] = []
    n_events_used = 0

    for path in paths:
        with uproot.open(path) as rootf:
            if "event" not in rootf or "step" not in rootf:
                continue
            et = rootf["event"]
            st = rootf["step"]
            event_required = ["eventID", "primaryEnergy", "depositedEnergy"]
            step_required = ["eventID", "parentID", "flagParticle", "x", "y", "z"]
            has_track_id = "trackID" in st.keys()
            has_step_id = "stepID" in st.keys()
            if has_track_id:
                step_required.append("trackID")
            if has_step_id:
                step_required.append("stepID")
            if not all(name in et.keys() for name in event_required):
                continue
            if not all(name in st.keys() for name in step_required):
                continue

            event_arr = et.arrays(event_required, library="np")
            event_budget = {
                int(eid): (float(pe), float(de))
                for eid, pe, de in zip(
                    event_arr["eventID"],
                    event_arr["primaryEnergy"],
                    event_arr["depositedEnergy"],
                )
            }

            step_state: dict[int, tuple[int, float, float, float] | None] = {}
            step_state_seq: dict[int, list[tuple[float, float, float]]] = {}
            for chunk in st.iterate(step_required, library="np", step_size=step_size):
                m_primary = (
                    (np.asarray(chunk["parentID"], dtype=np.int64) == 0)
                    & (np.asarray(chunk["flagParticle"], dtype=np.int64) == 1)
                )
                if has_track_id:
                    m_primary &= np.asarray(chunk["trackID"], dtype=np.int64) == 1
                if not np.any(m_primary):
                    continue
                event_ids = np.asarray(chunk["eventID"], dtype=np.int64)[m_primary]
                xs = np.asarray(chunk["x"], dtype=np.float64)[m_primary]
                ys = np.asarray(chunk["y"], dtype=np.float64)[m_primary]
                zs = np.asarray(chunk["z"], dtype=np.float64)[m_primary]
                if has_step_id:
                    step_ids = np.asarray(chunk["stepID"], dtype=np.int64)[m_primary]
                    for eid, sid, x, y, z in zip(event_ids, step_ids, xs, ys, zs):
                        _update_first_two_primary_steps(
                            step_state,
                            int(eid),
                            int(sid),
                            float(x),
                            float(y),
                            float(z),
                        )
                else:
                    for eid, x, y, z in zip(event_ids, xs, ys, zs):
                        eid_i = int(eid)
                        pts = step_state_seq.get(eid_i)
                        if pts is None:
                            step_state_seq[eid_i] = [(float(x), float(y), float(z))]
                        elif len(pts) < 2:
                            pts.append((float(x), float(y), float(z)))

            for event_id, (primary_e, deposited_e) in event_budget.items():
                if has_step_id:
                    first = step_state.get(event_id * 2)
                    second = step_state.get(event_id * 2 + 1)
                    if first is None or second is None:
                        continue
                    x0, y0, z0 = first[1], first[2], first[3]
                    x1, y1, z1 = second[1], second[2], second[3]
                else:
                    pts = step_state_seq.get(event_id)
                    if pts is None or len(pts) < 2:
                        continue
                    x0, y0, z0 = pts[0]
                    x1, y1, z1 = pts[1]
                dx = x1 - x0
                dy = y1 - y0
                dz = z1 - z0
                norm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                if norm <= 0.0:
                    continue
                uz = dz / norm
                theta = float(np.degrees(np.arccos(np.clip(uz, -1.0, 1.0))))
                theta_all.append(theta)
                n_events_used += 1

                if primary_e > 0.0:
                    missing_frac = max(0.0, (primary_e - deposited_e) / primary_e)
                    if missing_frac > deposition_epsilon_frac:
                        theta_incomplete.append(theta)

    return (
        np.asarray(theta_all, dtype=np.float64),
        np.asarray(theta_incomplete, dtype=np.float64),
        n_events_used,
    )


def plot_initial_angle_diagnostics(
    root_arg: str,
    out_path: str,
    deposition_epsilon_frac: float,
    step_size: int,
) -> str | None:
    if not out_path:
        return None
    try:
        theta_all, theta_incomplete, n_events_used = extract_initial_angle_distributions(
            root_arg=root_arg,
            deposition_epsilon_frac=deposition_epsilon_frac,
            step_size=step_size,
        )
    except Exception as exc:
        print(f"Initial-angle diagnostics skipped: {exc}")
        return None

    if theta_all.size == 0:
        print("Initial-angle diagnostics skipped: no valid primary launch angles found.")
        return None

    fig, (ax_all, ax_incomplete) = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True, constrained_layout=True
    )
    bins = np.linspace(0.0, 90.0, 46)

    ax_all.hist(theta_all, bins=bins, histtype="stepfilled", alpha=0.85, color="lightgray")
    ax_all.set_xlim(0.0, 90.0)
    ax_all.set_xlabel(r"$\theta$ (deg)")
    ax_all.set_ylabel("Counts")
    ax_all.set_title(f"Initial angle (all, N={n_events_used})")

    ax_incomplete.hist(
        theta_incomplete,
        bins=bins,
        histtype="stepfilled",
        alpha=0.85,
        color="dimgray",
    )
    ax_incomplete.set_xlim(0.0, 90.0)
    ax_incomplete.set_xlabel(r"$\theta$ (deg)")
    ax_incomplete.set_title(
        f"Incomplete deposition (N={theta_incomplete.size}, eps={deposition_epsilon_frac:g})"
    )

    out_resolved = _resolve_output(out_path)
    fig.savefig(out_resolved, bbox_inches="tight")
    print(f"Wrote {out_resolved}")
    return out_resolved


def _make_energy_bins(e_min: float, e_max: float, nbins: int = 60) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Create energy bins and centers; returns (bins, centers, is_log) or None."""
    if not (np.isfinite(e_min) and np.isfinite(e_max)):
        return None
    if e_min <= 0 or e_max <= 0 or e_max <= e_min:
        return None
    if e_max / max(e_min, 1e-30) > 1e3:
        bins = np.logspace(np.log10(e_min), np.log10(e_max), nbins + 1)
        centers = np.sqrt(bins[:-1] * bins[1:])
        return bins, centers, True
    bins = np.linspace(e_min, e_max, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return bins, centers, False

def _stream_collect_meta(path: str, tree_name: str, step_size: int) -> dict:
    """Collect metadata (min/max, channels, models) in a streaming pass."""
    meta = {
        "proc_counts": {},
        "models_by_proc": {},
        "process_names": {},
        "ke_min": {},
        "ke_max": {},
        "chan_max": {},
        "has_chan_micro": {},
        "kmax_electron": 0.0,
        "vib_channels": set(),
        "vib_dE_max": {},
    }
    columns = [
        "flagProcess",
        "kineticEnergy",
        "channelIndex",
        "channelMicroXS",
        "kineticEnergyDifference",
        "flagParticle",
        "modelName",
        "processName",
    ]
    for chunk in _iter_root_arrays(path, tree_name, columns, step_size):
        if "flagProcess" not in chunk or "kineticEnergy" not in chunk:
            continue
        flag_proc = np.asarray(chunk["flagProcess"], dtype=int)
        kinE = np.asarray(chunk["kineticEnergy"], dtype=float)
        model_arr = _decode_to_str_array(chunk.get("modelName"))
        proc_names = _decode_to_str_array(chunk.get("processName"))

        uniq, counts = np.unique(flag_proc, return_counts=True)
        for p, c in zip(uniq.tolist(), counts.tolist()):
            meta["proc_counts"][p] = meta["proc_counts"].get(p, 0) + int(c)

        if model_arr is None:
            model_arr = np.full(flag_proc.shape, "", dtype=object)

        for p in uniq:
            m = (flag_proc == int(p))
            if proc_names is not None:
                names = np.unique(proc_names[m])
                if p not in meta["process_names"]:
                    meta["process_names"][p] = set()
                meta["process_names"][p].update([n for n in names if n])
            if p not in meta["models_by_proc"]:
                meta["models_by_proc"][p] = set()
            meta["models_by_proc"][p].update([n for n in np.unique(model_arr[m]) if n or n == ""])

        # Per (process, model) min/max and channels
        for p in uniq:
            m_p = (flag_proc == int(p))
            if not np.any(m_p):
                continue
            models = np.unique(model_arr[m_p])
            for mdl in models:
                key = (int(p), str(mdl))
                m = m_p & (model_arr == mdl)
                if not np.any(m):
                    continue
                ke = kinE[m]
                ke_pos = ke[np.isfinite(ke) & (ke > 0)]
                if ke_pos.size:
                    meta["ke_min"][key] = min(meta["ke_min"].get(key, float("inf")), float(np.min(ke_pos)))
                    meta["ke_max"][key] = max(meta["ke_max"].get(key, 0.0), float(np.max(ke_pos)))
                if "channelIndex" in chunk:
                    ch = np.asarray(chunk["channelIndex"], dtype=int)[m]
                    if np.any(ch >= 0):
                        meta["chan_max"][key] = max(meta["chan_max"].get(key, -1), int(np.max(ch[ch >= 0])))
                if "channelMicroXS" in chunk:
                    cm = np.asarray(chunk["channelMicroXS"], dtype=float)[m]
                    if np.any(cm > 0.0):
                        meta["has_chan_micro"][key] = True

        # Electron kmax for summary
        if "flagParticle" in chunk:
            fp = np.asarray(chunk["flagParticle"], dtype=int)
            m_e = (fp == 1)
            if np.any(m_e):
                meta["kmax_electron"] = max(meta["kmax_electron"], float(np.nanmax(kinE[m_e])))

        # Vib energy loss ranges
        m_vib = (flag_proc == 15)
        if np.any(m_vib) and "kineticEnergyDifference" in chunk and "channelIndex" in chunk:
            dE = np.asarray(chunk["kineticEnergyDifference"], dtype=float)[m_vib]
            ch = np.asarray(chunk["channelIndex"], dtype=int)[m_vib]
            valid = ch >= 0
            if np.any(valid):
                for cidx in np.unique(ch[valid]).tolist():
                    meta["vib_channels"].add(int(cidx))
                    vmax = float(np.nanmax(dE[ch == cidx])) if np.any(ch == cidx) else 0.0
                    meta["vib_dE_max"][int(cidx)] = max(meta["vib_dE_max"].get(int(cidx), 0.0), vmax)

    return meta

def _print_root_processes_stream(meta: dict) -> None:
    if not meta or "proc_counts" not in meta:
        print("No ROOT data loaded; nothing to list.")
        return
    uniq = sorted(meta["proc_counts"].keys())
    print(f"Found {len(uniq)} unique flagProcess entries in ROOT:")
    for code in uniq:
        cnt = meta["proc_counts"].get(code, 0)
        label = _process_name_map().get(int(code), f"proc{int(code)}")
        names = sorted(meta.get("process_names", {}).get(code, []))
        models = sorted(meta.get("models_by_proc", {}).get(code, []))
        names_str = f" | processName: {', '.join(names)}" if names else ""
        models_str = f" | modelName: {', '.join(models)}" if models else ""
        print(f"  {int(code):>4d}  {label:<12} steps={cnt}{names_str}{models_str}")

def _init_stream_acc(meta: dict, nH2O_cm3: float) -> dict:
    acc = {
        "xs": {},
        "deflection": {},
        "vib": {"bins": {}, "hist": {}, "hist_by_model": {}, "models": []},
        "summary": {},
    }

    # Energy bins per (process, model)
    for key, e_min in meta.get("ke_min", {}).items():
        e_max = meta.get("ke_max", {}).get(key, 0.0)
        bins_info = _make_energy_bins(e_min, e_max, nbins=60)
        if bins_info is None:
            continue
        bins, centers, is_log = bins_info
        nbins = len(bins) - 1
        chan_max = meta.get("chan_max", {}).get(key, -1)
        n_channels = max(1, int(chan_max) + 1)
        acc["xs"][key] = {
            "bins": bins,
            "centers": centers,
            "is_log": is_log,
            "sum_total": np.zeros(nbins, dtype=float),
            "count_total": np.zeros(nbins, dtype=int),
            "sum_ch": {ch: np.zeros(nbins, dtype=float) for ch in range(n_channels)},
            "count_ch": {ch: np.zeros(nbins, dtype=int) for ch in range(n_channels)},
            "use_counts": int(key[0]) in (12, 13),
        }
        acc["deflection"][key] = np.zeros(90, dtype=int)

    # Vib hist bins
    vib_models = sorted([m for m in meta.get("models_by_proc", {}).get(15, set()) if m])
    acc["vib"]["models"] = vib_models
    for ch in sorted(meta.get("vib_channels", [])):
        vmax = meta.get("vib_dE_max", {}).get(int(ch), 1.0)
        vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
        bins = np.linspace(0.0, vmax, 81)
        acc["vib"]["bins"][int(ch)] = bins
        acc["vib"]["hist"][int(ch)] = np.zeros(len(bins) - 1, dtype=int)
        for m in vib_models:
            acc["vib"]["hist_by_model"].setdefault(m, {})[int(ch)] = np.zeros(len(bins) - 1, dtype=int)

    # Summary accumulators
    summary = {}
    summary["proc_counts"] = {}
    summary["x_bins"] = np.linspace(0, 2000, 101)
    summary["x_hist_by_proc"] = {pid: np.zeros(100, dtype=int) for pid in (10, 11, 12, 13, 14, 15)}
    kmax = meta.get("kmax_electron", 0.0)
    if not np.isfinite(kmax) or kmax <= 0:
        kmax = 2000.0
    summary["k_bins"] = np.linspace(0, float(kmax), 101)
    summary["k_hist"] = np.zeros(100, dtype=int)
    summary["scatter"] = {"x": np.array([], dtype=float), "y": np.array([], dtype=float), "z": np.array([], dtype=float)}
    summary["scatter_max"] = 50000
    summary["rng"] = np.random.default_rng(0)
    acc["summary"] = summary
    return acc

def _stream_accumulate(path: str, tree_name: str, step_size: int, nH2O_cm3: float, meta: dict, acc: dict) -> None:
    macro_keys = ["vibCrossSection", "macroCrossSection", "processCrossSection"]
    columns = [
        "flagProcess",
        "kineticEnergy",
        "channelIndex",
        "channelMicroXS",
        "kineticEnergyDifference",
        "cosTheta",
        "flagParticle",
        "x","y","z",
        "processName",
        "modelName",
        *macro_keys,
    ]
    theta_bins = np.linspace(0, 180, 91)

    for chunk in _iter_root_arrays(path, tree_name, columns, step_size):
        if "flagProcess" not in chunk or "kineticEnergy" not in chunk:
            continue
        flag_proc = np.asarray(chunk["flagProcess"], dtype=int)
        kinE = np.asarray(chunk["kineticEnergy"], dtype=float)
        model_arr = _decode_to_str_array(chunk.get("modelName"))
        if model_arr is None:
            model_arr = np.full(flag_proc.shape, "", dtype=object)
        alias = meta.get("model_alias")
        if alias:
            model_arr = np.asarray([alias.get(m, m) for m in model_arr], dtype=object)

        # Total cross-section (macro->micro)
        xs_macro = None
        for k in macro_keys:
            if k in chunk:
                xs_macro = np.asarray(chunk[k], dtype=float)
                break
        if xs_macro is None:
            xs_macro = np.zeros_like(kinE)
        total_xs_micro = _to_micro_cm2(xs_macro, nH2O_cm3)

        chan_idx_all = np.asarray(chunk["channelIndex"], dtype=int) if "channelIndex" in chunk else None
        chan_micro_all = np.asarray(chunk["channelMicroXS"], dtype=float) if "channelMicroXS" in chunk else None
        dE_all = np.asarray(chunk["kineticEnergyDifference"], dtype=float) if "kineticEnergyDifference" in chunk else None

        # Update per (process, model) accumulators
        for key, data in acc["xs"].items():
            pcode, model = key
            m = (flag_proc == int(pcode))
            if model:
                m = m & (model_arr == model)
            if not np.any(m):
                continue
            ke = kinE[m]
            bins = data["bins"]
            nbins = len(bins) - 1
            idx = np.searchsorted(bins, ke, side="right") - 1
            valid = (idx >= 0) & (idx < nbins) & np.isfinite(ke)
            if not np.any(valid):
                continue
            idx_v = idx[valid]

            xs_tot = total_xs_micro[m][valid]
            np.add.at(data["sum_total"], idx_v, xs_tot)
            np.add.at(data["count_total"], idx_v, 1)

            ch = None
            if chan_idx_all is not None:
                ch = chan_idx_all[m]
            if ch is None or not np.any(ch >= 0):
                if int(pcode) in (12, 13) and dE_all is not None:
                    ch = _infer_channel_indices(int(pcode), dE_all[m])
                else:
                    ch = np.zeros_like(ke, dtype=int)

            if data["use_counts"]:
                for cidx in data["count_ch"].keys():
                    m_ch = (ch == int(cidx)) & valid
                    if not np.any(m_ch):
                        continue
                    np.add.at(data["count_ch"][cidx], idx[m_ch], 1)
                continue

            micro = total_xs_micro[m]
            if chan_micro_all is not None:
                cm = chan_micro_all[m]
                micro = np.where(cm > 0.0, cm, micro)
            micro = micro[valid]
            ch_valid = ch[valid]
            for cidx in data["sum_ch"].keys():
                m_ch = (ch_valid == int(cidx))
                if not np.any(m_ch):
                    continue
                np.add.at(data["sum_ch"][cidx], idx_v[m_ch], micro[m_ch])
                np.add.at(data["count_ch"][cidx], idx_v[m_ch], 1)

        # Deflection angles
        if "cosTheta" in chunk:
            cos_theta = np.asarray(chunk["cosTheta"], dtype=float)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta_deg = np.degrees(np.arccos(cos_theta))
            for key in acc["deflection"].keys():
                pcode, model = key
                m = (flag_proc == int(pcode))
                if model:
                    m = m & (model_arr == model)
                if not np.any(m):
                    continue
                hist, _ = np.histogram(theta_deg[m], bins=theta_bins)
                acc["deflection"][key] += hist

        # Vib energy-loss histograms
        if dE_all is not None and chan_idx_all is not None:
            m_vib = (flag_proc == 15)
            if np.any(m_vib):
                dE = dE_all[m_vib]
                ch = chan_idx_all[m_vib]
                models_vib = model_arr[m_vib] if model_arr is not None else None
                for cidx, bins in acc["vib"]["bins"].items():
                    m_ch = (ch == int(cidx))
                    if not np.any(m_ch):
                        continue
                    h, _ = np.histogram(dE[m_ch], bins=bins)
                    acc["vib"]["hist"][int(cidx)] += h
                    if models_vib is not None and acc["vib"]["hist_by_model"]:
                        for mname, hist_by_ch in acc["vib"]["hist_by_model"].items():
                            m_model = (models_vib == mname) & m_ch
                            if not np.any(m_model):
                                continue
                            hm, _ = np.histogram(dE[m_model], bins=bins)
                            hist_by_ch[int(cidx)] += hm

        # Summary accumulators
        summary = acc["summary"]
        uniq, counts = np.unique(flag_proc, return_counts=True)
        for p, c in zip(uniq.tolist(), counts.tolist()):
            summary["proc_counts"][p] = summary["proc_counts"].get(p, 0) + int(c)

        if "x" in chunk:
            x_nm = np.asarray(chunk["x"], dtype=float)
            for pid in summary["x_hist_by_proc"].keys():
                m = (flag_proc == int(pid))
                if not np.any(m):
                    continue
                h, _ = np.histogram(x_nm[m], bins=summary["x_bins"])
                summary["x_hist_by_proc"][pid] += h

        if "flagParticle" in chunk:
            fp = np.asarray(chunk["flagParticle"], dtype=int)
            m_e = (fp == 1)
            if np.any(m_e):
                h, _ = np.histogram(kinE[m_e], bins=summary["k_bins"])
                summary["k_hist"] += h
                # Reservoir-ish sample for scatter
                if "x" in chunk and "y" in chunk and "z" in chunk:
                    x = np.asarray(chunk["x"], dtype=float)[m_e]
                    y = np.asarray(chunk["y"], dtype=float)[m_e]
                    z = np.asarray(chunk["z"], dtype=float)[m_e]
                    if x.size:
                        sx = summary["scatter"]["x"]
                        sy = summary["scatter"]["y"]
                        sz = summary["scatter"]["z"]
                        max_n = summary["scatter_max"]
                        combined_n = sx.size + x.size
                        if combined_n <= max_n:
                            summary["scatter"]["x"] = np.concatenate([sx, x])
                            summary["scatter"]["y"] = np.concatenate([sy, y])
                            summary["scatter"]["z"] = np.concatenate([sz, z])
                        else:
                            combined_x = np.concatenate([sx, x])
                            combined_y = np.concatenate([sy, y])
                            combined_z = np.concatenate([sz, z])
                            idx = summary["rng"].choice(combined_x.size, size=max_n, replace=False)
                            summary["scatter"]["x"] = combined_x[idx]
                            summary["scatter"]["y"] = combined_y[idx]
                            summary["scatter"]["z"] = combined_z[idx]

def load_reference_from_path(fpath: str):
    """Load reference partial XS from a .dat file.

    Format: first column is energy (eV), followed by M partial XS columns.

    By default the columns are interpreted as being in 1e-16 cm^2 (Michaud
    vibrational / elastic tables).  For some data sets (notably the
    Emfietzoglou ionisation table used by G4DNAEmfietzoglouIonisationModel),
    the caller applies an extra scale factor to convert the raw table units
    to 1e-16 cm^2 before plotting.

    Returns (E_eV, ref_by_channel) where len(ref_by_channel)=M and each
    element is a NumPy array of the same length as E_eV.
    """
    fpath = _resolve_path(fpath)
    E: list[float] = []
    rows: list[list[float]] = []
    max_cols = 0
    with open(fpath, "r") as fin:
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
            if len(vals) > max_cols:
                max_cols = len(vals)
    E_arr = np.asarray(E, dtype=float)
    ref_by_ch: list[np.ndarray] = []
    for j in range(max_cols):
        col = np.array([row[j] if j < len(row) else 0.0 for row in rows], dtype=float)
        ref_by_ch.append(col)
    return E_arr, ref_by_ch

# -------- Physics helpers --------

def _to_micro_cm2(xs_macro_mm_inv_subset: np.ndarray, nH2O_cm3: float) -> np.ndarray:
    """Convert macroscopic mm^-1 to microscopic cm^2 using number density."""
    return (xs_macro_mm_inv_subset * 10.0) / float(nH2O_cm3)

def _energy_grid_for_ref(energy_eV: np.ndarray) -> np.ndarray:
    """Build a smooth energy grid from simulation energies."""
    e = np.asarray(energy_eV, dtype=float)
    e = e[np.isfinite(e) & (e > 0.0)]
    if e.size == 0:
        return np.logspace(0.0, 7.0, 240)  # 1 eV to 10 MeV
    e_min = float(np.nanmin(e))
    e_max = float(np.nanmax(e))
    if e_max <= e_min:
        return np.array([e_min], dtype=float)
    if e_max / max(e_min, 1e-30) > 100.0:
        return np.logspace(np.log10(e_min), np.log10(e_max), 240)
    return np.linspace(e_min, e_max, 240)

def _screened_rutherford_sigma_cm2(energy_eV: np.ndarray) -> np.ndarray:
    """Screened Rutherford total elastic σ (cm^2)."""
    E = np.asarray(energy_eV, dtype=float)
    out = np.zeros_like(E)
    pos = E > 0.0
    if not np.any(pos):
        return out
    k_MeV = E[pos] * EV_TO_MEV
    m = SR_ELECTRON_MASS_MEV
    length_fm = (SR_E_SQUARED_MEV_FM * (k_MeV + m)) / (k_MeV * (k_MeV + 2.0 * m))
    sigma_ruth_fm2 = SR_Z_WATER * (SR_Z_WATER + 1.0) * length_fm**2
    numerator = (SR_ALPHA_1 + SR_BETA_1 * np.log(E[pos])) * SR_CONST_K * (SR_Z_WATER ** (2.0 / 3.0))
    denom = (k_MeV / m) * (2.0 + (k_MeV / m))
    n = np.where(denom > 0.0, numerator / denom, 0.0)
    sigma_fm2 = np.pi * sigma_ruth_fm2 / (n * (n + 1.0))
    out[pos] = sigma_fm2 * FM2_TO_CM2
    return out

def _analytic_reference(pcode: int, model_hints: list[str], energy_eV: np.ndarray, scale: float):
    """Fallback analytic/reference curves for analytic models."""
    hints = [m.lower() for m in model_hints if m]
    if int(pcode) == 11 and any(("screenedrutherford" in m) or ("rutherfordelastic" in m) for m in hints):
        egrid = _energy_grid_for_ref(energy_eV)
        sigma_cm2 = _screened_rutherford_sigma_cm2(egrid)
        # Reference is plotted in 1e-16 cm^2 units (same as sim*scale)
        return egrid, [sigma_cm2 * scale], "ScreenedRutherford (analytic)"
    if int(pcode) == 14 and any("melton" in m for m in hints):
        p = _find_g4ledata_file("sigma_attachment_e_melton.dat")
        if p:
            ref_E, ref_by_ch = load_reference_from_path(p)
            ref_by_ch = _scale_reference_if_needed(p, ref_by_ch, scale)
            return ref_E, ref_by_ch, Path(p).name
    return None, None, None

def _infer_channel_indices(pcode: int, dE_eV: np.ndarray) -> np.ndarray:
    """Infer channel indices from energy loss for excitation/ionisation."""
    if dE_eV.size == 0:
        return np.asarray([], dtype=int)
    dE = np.asarray(dE_eV, dtype=float)
    out = np.full(dE.shape, -1, dtype=int)
    valid = np.isfinite(dE) & (dE > 0.0)
    if not np.any(valid):
        return out
    dE_valid = dE[valid]
    if int(pcode) == 12:
        diff = np.abs(dE_valid[:, None] - EMFI_EXCITATION_EEV[None, :])
        best = np.argmin(diff, axis=1)
        best_diff = diff[np.arange(diff.shape[0]), best]
        out_valid = np.where(best_diff <= EMFI_EXCITATION_TOL_EEV, best, -1)
        out[valid] = out_valid
        return out
    if int(pcode) == 13:
        bind = EMFI_ION_BINDING_EEV[None, :]
        diff = np.where(dE_valid[:, None] >= bind, dE_valid[:, None] - bind, np.inf)
        diff = np.where(diff < 1000.0, diff, np.inf)
        best = np.argmin(diff, axis=1)
        best_diff = diff[np.arange(diff.shape[0]), best]
        out_valid = np.where(np.isfinite(best_diff), best, -1)
        out[valid] = out_valid
        return out
    return out

def _estimate_partial_xs_by_counts(
    ke_eV: np.ndarray,
    total_xs_cm2: np.ndarray,
    chan_idx: np.ndarray,
    n_channels: int,
    nbins: int = 60,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    """Estimate per-channel microscopic XS by channel fractions in energy bins."""
    ke = np.asarray(ke_eV, dtype=float)
    xs = np.asarray(total_xs_cm2, dtype=float)
    ch = np.asarray(chan_idx, dtype=int)
    valid = np.isfinite(ke) & np.isfinite(xs) & (ke > 0.0) & (xs >= 0.0) & (ch >= 0)
    if not np.any(valid):
        return {}, (np.asarray([]), np.asarray([]))
    ke = ke[valid]
    xs = xs[valid]
    ch = ch[valid]
    e_min = float(np.nanmin(ke))
    e_max = float(np.nanmax(ke))
    if e_max <= e_min:
        return {}, (np.asarray([]), np.asarray([]))
    if e_max / max(e_min, 1e-30) > 1.2:
        bins = np.geomspace(e_min, e_max, nbins + 1)
    else:
        bins = np.linspace(e_min, e_max, nbins + 1)
    bin_idx = np.digitize(ke, bins) - 1
    in_range = (bin_idx >= 0) & (bin_idx < nbins)
    bin_idx = bin_idx[in_range]
    ke = ke[in_range]
    xs = xs[in_range]
    ch = ch[in_range]
    centers = 0.5 * (bins[:-1] + bins[1:])

    series_by_ch: dict[int, tuple[list[float], list[float]]] = {
        c: ([], []) for c in range(n_channels)
    }
    total_x = []
    total_y = []
    for b in range(nbins):
        m = (bin_idx == b)
        if not np.any(m):
            continue
        mean_total = float(np.nanmean(xs[m]))
        counts = np.bincount(ch[m], minlength=n_channels).astype(float)
        count_total = float(counts.sum())
        if count_total <= 0.0:
            continue
        total_x.append(centers[b])
        total_y.append(mean_total)
        for c in range(n_channels):
            series_by_ch[c][0].append(centers[b])
            series_by_ch[c][1].append(mean_total * (counts[c] / count_total))

    series_out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for c, (xs_list, ys_list) in series_by_ch.items():
        series_out[c] = (np.asarray(xs_list, dtype=float), np.asarray(ys_list, dtype=float))
    return series_out, (np.asarray(total_x, dtype=float), np.asarray(total_y, dtype=float))

# HG helpers for deflection overlays
def _hg_p_per_deg(theta_deg: np.ndarray, g: np.ndarray) -> np.ndarray:
    th = np.deg2rad(theta_deg)[:, None]
    cos_t = np.cos(th)
    sin_t = np.sin(th)
    g = np.clip(np.asarray(g, dtype=float)[None, :], -0.9999, 0.9999)
    denom = np.power(1.0 + g*g - 2.0*g*cos_t, 1.5)
    p_cos = (1.0 - g*g) / (4.0 * np.pi * denom)
    p_per_rad = p_cos * sin_t
    return p_per_rad / (np.pi / 180.0)

def _hg_forward_fraction(g: float) -> float:
    g = float(np.clip(g, -0.999999, 0.999999))
    if abs(g) < 1e-12:
        return 0.5
    A0 = 1.0 + g*g
    term = (1.0 - g) / np.sqrt(A0)
    return (1.0 + g) / (2.0 * g) * (1.0 - term)

def _invert_forward_fraction_to_g(Y: float, tol: float = 1e-10, maxit: int = 100) -> float:
    Y = float(np.clip(Y, 0.0, 1.0))
    if Y <= 0.0:
        return -0.999999
    if Y >= 1.0:
        return 0.999999
    lo, hi = -0.999999, 0.999999
    f_lo = _hg_forward_fraction(lo) - Y
    f_hi = _hg_forward_fraction(hi) - Y
    if f_lo * f_hi > 0:
        return 0.0
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        f_mid = _hg_forward_fraction(mid) - Y
        if abs(f_mid) < tol or (hi - lo) < 1e-12:
            return float(np.clip(mid, -0.999999, 0.999999))
        if f_lo * f_mid <= 0:
            hi = mid; f_hi = f_mid
        else:
            lo = mid; f_lo = f_mid
    return float(np.clip(0.5 * (lo + hi), -0.999999, 0.999999))

def _load_michaud_gamma_for_channel(idx: int):
    import pandas as pd
    mapping = [
        ("table2", 5, 6), ("table2", 7, 8), ("table2", 9, 10),
        ("table3", 1, 2), ("table3", 3, 4), ("table3", 5, 6),
        ("table3", 7, 8), ("table3", 9, 10)
    ]
    if not (0 <= idx <= 7):
        return None, None
    table, _, col_g = mapping[idx]
    path = MICHAUD_TABLE2 if table == "table2" else MICHAUD_TABLE3
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        return None, None
    e = pd.to_numeric(df.iloc[3:, 0], errors='coerce').values.astype(float)
    g = pd.to_numeric(df.iloc[3:, col_g], errors='coerce').values.astype(float)
    m = ~(np.isnan(e) | np.isnan(g))
    e = e[m]; g = g[m]
    if e.size == 0:
        return None, None
    order = np.argsort(e)
    return e[order], g[order]

# -------- Plotters --------
def plot_cross_sections_for_process(arrs, pcode: int, ncols: int, scale: float,
                                    nH2O_cm3: float, out_path: str, dat_path: str | None = None,
                                    model_filter: str | None = None,
                                    config_refs: dict[str, str] | None = None):
    """Plot XS vs KE per channel for a given process (optionally filtered by modelName)."""
    if arrs is None:
        return None
    mask_proc = (np.asarray(arrs["flagProcess"], dtype=float) == float(pcode))
    if model_filter and "modelName" in arrs:
        models = _decode_to_str_array(arrs["modelName"])
        if models is not None:
            mask_proc = mask_proc & (models == model_filter)
    if not np.any(mask_proc):
        return None

    ke_all = np.asarray(arrs["kineticEnergy"], dtype=float)[mask_proc]
    chan_idx = (np.asarray(arrs["channelIndex"], dtype=int)[mask_proc]
                if "channelIndex" in arrs else None)
    chan_micro = (np.asarray(arrs["channelMicroXS"], dtype=float)[mask_proc]
                  if "channelMicroXS" in arrs else None)

    macro_keys = ["vibCrossSection", "macroCrossSection", "processCrossSection"]

    # Reference data
    model_hints: list[str] = []
    if "modelName" in arrs and np.any(mask_proc):
        models = np.asarray(arrs["modelName"], dtype=object)[mask_proc]
        unique = np.unique(models[models != b""])
        for u in unique:
            model_hints.append(u.decode(errors="ignore") if isinstance(u, (bytes, bytearray)) else str(u))
    ref_E = None
    ref_by_ch = None
    ref_label = None
    if dat_path:
        dat_path = _prefer_total_ref(dat_path, int(pcode))
        ref_E, ref_by_ch = load_reference_from_path(dat_path)
        ref_by_ch = _scale_reference_if_needed(dat_path, ref_by_ch, scale)
        ref_label = Path(dat_path).name
    elif int(pcode) == 11 and model_hints:
        p = _resolve_elastic_reference_path(model_hints, config_refs=config_refs)
        if p:
            ref_E, ref_by_ch = load_reference_from_path(p)
            ref_by_ch = _scale_reference_if_needed(p, ref_by_ch, scale)
            ref_label = Path(p).name
    elif int(pcode) == 15:
        p = config_refs.get("ref_vib") if config_refs else None
        if p:
            ref_E, ref_by_ch = load_reference_from_path(p)
            ref_by_ch = _scale_reference_if_needed(p, ref_by_ch, scale)
            ref_label = Path(p).name
    elif int(pcode) == 12:
        p = None
        if model_hints and any("bornexcitationmodel" in m.lower() for m in model_hints):
            p = config_refs.get("ref_excitation_born") if config_refs else None
        if not p:
            p = config_refs.get("ref_excitation") if config_refs else None
        if p:
            p = _prefer_total_ref(p, int(pcode))
            ref_E, ref_by_ch = load_reference_from_path(p)
            ref_by_ch = _scale_reference_if_needed(p, ref_by_ch, scale)
            ref_label = Path(p).name
    elif int(pcode) == 13:
        p = None
        if model_hints and any("bornionisationmodel" in m.lower() for m in model_hints):
            p = config_refs.get("ref_ionisation_born") if config_refs else None
        if not p:
            p = config_refs.get("ref_ionisation") if config_refs else None
        if p:
            p = _prefer_total_ref(p, int(pcode))
            ref_E, ref_by_ch = load_reference_from_path(p)
            ref_by_ch = _scale_reference_if_needed(p, ref_by_ch, scale)
            ref_label = Path(p).name
    elif int(pcode) == 14:
        p = config_refs.get("ref_attachment") if config_refs else None
        if p:
            ref_E, ref_by_ch = load_reference_from_path(p)
            ref_by_ch = _scale_reference_if_needed(p, ref_by_ch, scale)
            ref_label = Path(p).name
    if ref_E is None or not ref_by_ch:
        ref_E, ref_by_ch, ref_label = _analytic_reference(int(pcode), model_hints, ke_all, scale)

    channels: List[int] = []
    if chan_idx is not None:
        pos = chan_idx[chan_idx >= 0]
        if pos.size:
            channels = np.unique(pos).tolist()
    if not channels:
        channels = [0]

    n = len(channels)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    figsize = (10, 6) if n == 1 else (5 * ncols, 4.5 * nrows)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize,
        squeeze=False, sharex=True, sharey='row', constrained_layout=False
    )

    pname = _process_name_map().get(int(pcode), f"proc{int(pcode)}")
    legend_added = False

    # Precompute counts-based series for ionisation/excitation (more robust than per-step micro XS)
    use_counts = int(pcode) in (12, 13)
    series_by_ch = None
    total_series = None
    if use_counts and chan_idx is not None and np.any(chan_idx >= 0):
        xs_macro_all = None
        for k in macro_keys:
            if k in arrs:
                xs_macro_all = np.asarray(arrs[k], dtype=float)[mask_proc]
                break
        if xs_macro_all is None:
            xs_macro_all = np.zeros_like(ke_all)
        total_xs_micro = _to_micro_cm2(xs_macro_all, nH2O_cm3)
        n_channels = int(np.max(chan_idx[chan_idx >= 0])) + 1
        series_by_ch, _ = _estimate_partial_xs_by_counts(
            ke_all, total_xs_micro, chan_idx, n_channels
        )

    for i, ch in enumerate(channels):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if use_counts:
            if series_by_ch is None or ch not in series_by_ch:
                ax.set_visible(False)
                continue
            x, y = series_by_ch[ch]
            if x.size == 0:
                ax.set_visible(False)
                continue
            line_sim, = ax.plot(
                x,
                y * scale,
                "--",
                linewidth=2,
                c="dodgerblue",
                label="Simulation (counts)",
                zorder=2,
            )
        else:
            if chan_idx is not None and np.any(chan_idx >= 0):
                m = (chan_idx == ch)
            else:
                m = np.ones_like(ke_all, dtype=bool)
            if not np.any(m):
                ax.set_visible(False)
                continue

            ke = ke_all[m]
            if ke.size == 0:
                ax.set_visible(False)
                continue
            ke_order = np.argsort(ke)

            xs_macro = None
            for k in macro_keys:
                if k in arrs:
                    xs_macro = np.asarray(arrs[k], dtype=float)[mask_proc][m]
                    break
            if xs_macro is None:
                xs_macro = np.zeros_like(ke)
            xs_micro_cm2 = _to_micro_cm2(xs_macro, nH2O_cm3)
            if chan_micro is not None:
                xs_micro_sel = chan_micro[m]
                xs_micro_cm2 = np.where(xs_micro_sel > 0.0, xs_micro_sel, xs_micro_cm2)

            line_sim, = ax.plot(
                ke[ke_order],
                xs_micro_cm2[ke_order] * scale,
                "--",
                linewidth=2,
                c="dodgerblue",
                label="Simulation",
                zorder=2,
            )

        lines_ref = []
        if ref_E is not None and ref_by_ch:
            if 0 <= ch < len(ref_by_ch):
                y_ref = ref_by_ch[ch]
            else:
                y_ref = np.sum(np.vstack(ref_by_ch), axis=0)
            lr, = ax.plot(ref_E, y_ref, color="black", linewidth=2.5, alpha=0.9, label=ref_label or "Reference", zorder=1)
            lines_ref.append(lr)

        ax.set_xlabel("Kinetic Energy (eV)")
        ax.set_ylabel("Cross Section (10$^{-16}$ cm$^{2}$)")
        panel_label = f"{pname}_ch{ch}" if (chan_idx is not None and np.any(chan_idx >= 0)) else pname
        ax.set_title(panel_label)
        span_vals = []
        if ref_E is not None and ref_E.size:
            pos_ref = ref_E[ref_E > 0]
            if pos_ref.size:
                span_vals.append(np.nanmin(pos_ref))
                span_vals.append(np.nanmax(pos_ref))
        ke_panel = x if use_counts else ke
        if ke_panel.size:
            kpos = ke_panel[ke_panel > 0]
            if kpos.size:
                span_vals.append(np.nanmin(kpos))
                span_vals.append(np.nanmax(kpos))
        if span_vals:
            e_min = float(np.nanmin(span_vals))
            e_max = float(np.nanmax(span_vals))
            if e_max / max(e_min, 1e-30) > 1e3:
                ax.set_xscale("log")

        if not legend_added and (lines_ref or line_sim is not None):
            handles = []
            labels = []
            if lines_ref:
                handles.extend(lines_ref)
                labels.extend([lr.get_label() for lr in lines_ref])
            if line_sim is not None:
                handles.append(line_sim); labels.append(line_sim.get_label())
            if handles:
                ax.legend(handles, labels, loc="best", frameon=True)
                legend_added = True

    total_axes = nrows * ncols
    for j in range(n, total_axes):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            if not ax.get_visible():
                continue
            if c != 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
    for r in range(nrows - 1):
        for c in range(ncols):
            ax = axes[r][c]
            if not ax.get_visible():
                continue
            ax.set_xlabel("")
            ax.tick_params(labelbottom=True)

    fig.align_ylabels([axes[r][0] for r in range(nrows) if axes[r][0].get_visible()])
    fig.tight_layout()
    outpath = _resolve_output(out_path)
    plt.show()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Wrote {outpath}")
    plt.close(fig)
    return outpath

def plot_channel_overlay_for_process(arrs, pcode: int, scale: float,
                                     nH2O_cm3: float, out_path: str,
                                     dat_path: str | None = None):
    """Plot all channels + cumulative in a single panel for excitation/ionisation."""
    if arrs is None:
        return None
    flag_proc = np.asarray(arrs["flagProcess"], dtype=float)
    mask_proc = (flag_proc == float(pcode))
    if not np.any(mask_proc):
        return None

    ke_all = np.asarray(arrs["kineticEnergy"], dtype=float)[mask_proc]
    macro_keys = ["vibCrossSection", "macroCrossSection", "processCrossSection"]
    xs_macro = None
    for k in macro_keys:
        if k in arrs:
            xs_macro = np.asarray(arrs[k], dtype=float)[mask_proc]
            break
    if xs_macro is None:
        return None
    total_xs_micro = _to_micro_cm2(xs_macro, nH2O_cm3)

    chan_idx = None
    if "channelIndex" in arrs:
        chan_idx = np.asarray(arrs["channelIndex"], dtype=int)[mask_proc]
    if chan_idx is None or not np.any(chan_idx >= 0):
        if "kineticEnergyDifference" in arrs:
            dE = np.asarray(arrs["kineticEnergyDifference"], dtype=float)[mask_proc]
            chan_idx = _infer_channel_indices(int(pcode), dE)
    if chan_idx is None:
        chan_idx = np.full_like(ke_all, -1, dtype=int)

    chan_micro = None
    if "channelMicroXS" in arrs:
        chan_micro = np.asarray(arrs["channelMicroXS"], dtype=float)[mask_proc]
    use_chan_micro = chan_micro is not None and np.any(chan_micro > 0.0)
    if int(pcode) in (12, 13):
        use_chan_micro = False

    # Reference data
    ref_E = None
    ref_by_ch = None
    ref_path = None
    if dat_path:
        ref_path = dat_path
    if ref_path:
        ref_path = _prefer_total_ref(ref_path, int(pcode))
        ref_E, ref_by_ch = load_reference_from_path(ref_path)
        ref_by_ch = _scale_reference_if_needed(ref_path, ref_by_ch, scale)

    sim_channels = np.unique(chan_idx[chan_idx >= 0]) if chan_idx is not None else np.array([], dtype=int)
    n_ref = len(ref_by_ch) if ref_by_ch else 0
    channels = sorted(set(range(n_ref)) | set(sim_channels.tolist()))
    if not channels:
        channels = [0]
    n_channels = max(channels) + 1

    if use_chan_micro:
        series_by_ch = {}
        for ch in channels:
            m = (chan_idx == ch) & (chan_micro > 0.0)
            if not np.any(m):
                continue
            x = ke_all[m]
            y = chan_micro[m]
            order = np.argsort(x)
            series_by_ch[ch] = (x[order], y[order])
        total_series = (ke_all, total_xs_micro)
    else:
        series_by_ch, total_series = _estimate_partial_xs_by_counts(
            ke_all, total_xs_micro, chan_idx, n_channels
        )
        if total_series[0].size == 0 and ke_all.size:
            total_series = (ke_all, total_xs_micro)

    fig, ax = plt.subplots(figsize=(8, 6))
    pname = _process_name_map().get(int(pcode), f"proc{int(pcode)}")
    cmap = plt.cm.tab10 if len(channels) <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, len(channels)))

    # Reference per-channel
    if ref_E is not None and ref_by_ch:
        for i, ch in enumerate(channels):
            if ch >= len(ref_by_ch):
                continue
            ax.plot(
                ref_E,
                ref_by_ch[ch],
                color=colors[i],
                linewidth=2.0,
                label=f"ch{ch} ref",
            )

    # Simulation per-channel
    for i, ch in enumerate(channels):
        if ch not in series_by_ch:
            continue
        x_sim, y_sim = series_by_ch[ch]
        if x_sim.size == 0:
            continue
        ax.plot(
            x_sim,
            y_sim * scale,
            linestyle="--",
            marker="o",
            markersize=3,
            color=colors[i],
            label=f"ch{ch} sim",
        )

    # Cumulative totals
    if ref_E is not None and ref_by_ch:
        ref_cum = np.sum(np.vstack(ref_by_ch), axis=0)
        ax.plot(ref_E, ref_cum, color="gray", linewidth=4, label="total ref")
    if total_series[0].size:
        x_tot, y_tot = total_series
        order = np.argsort(x_tot)
        ax.plot(
            x_tot[order],
            y_tot[order] * scale,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="total sim",
        )

    ax.set_xlabel("Kinetic Energy (eV)")
    ax.set_ylabel("Cross Section (10$^{-16}$ cm$^{2}$)")
    ax.set_title(pname)

    span_vals = []
    if ref_E is not None and ref_E.size:
        pos_ref = ref_E[ref_E > 0]
        if pos_ref.size:
            span_vals.append(np.nanmin(pos_ref))
            span_vals.append(np.nanmax(pos_ref))
    if ke_all.size:
        kpos = ke_all[ke_all > 0]
        if kpos.size:
            span_vals.append(np.nanmin(kpos))
            span_vals.append(np.nanmax(kpos))
    if span_vals:
        e_min = float(np.nanmin(span_vals))
        e_max = float(np.nanmax(span_vals))
        if e_max / max(e_min, 1e-30) > 1e3:
            ax.set_xscale("log")

    ax.legend(loc="best", frameon=True, ncol=2)
    fig.tight_layout()
    outpath = _resolve_output(out_path)
    plt.show()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Wrote {outpath}")
    plt.close(fig)
    return outpath

def plot_cross_sections_for_process_stream(acc: dict, pcode: int, model: str, ncols: int,
                                           scale: float, out_path: str,
                                           dat_path: str | None = None,
                                           config_refs: dict[str, str] | None = None):
    """Streaming version of per-channel XS plots."""
    key = (int(pcode), str(model))
    if "xs" not in acc or key not in acc["xs"]:
        return None
    data = acc["xs"][key]
    centers = data["centers"]
    use_counts = data["use_counts"]

    count_total = data["count_total"].astype(float)
    sum_total = data["sum_total"]
    mean_total = np.divide(sum_total, count_total, out=np.zeros_like(sum_total), where=count_total > 0)

    channels = [ch for ch in data["count_ch"].keys() if np.any(data["count_ch"][ch] > 0)]
    if not channels:
        channels = list(data["count_ch"].keys())
    if not channels:
        return None

    ref_E = None
    ref_by_ch = None
    ref_label = None
    if dat_path:
        dat_path = _prefer_total_ref(dat_path, int(pcode))
        ref_E, ref_by_ch = load_reference_from_path(dat_path)
        ref_by_ch = _scale_reference_if_needed(dat_path, ref_by_ch, scale)
        ref_label = Path(dat_path).name
    elif int(pcode) == 11 and config_refs:
        hints = [model, _process_name_map().get(int(pcode), "")]
        p = _resolve_elastic_reference_path(hints, config_refs=config_refs)
        if p:
            ref_E, ref_by_ch = load_reference_from_path(p)
            ref_by_ch = _scale_reference_if_needed(p, ref_by_ch, scale)
            ref_label = Path(p).name
    if ref_E is None or not ref_by_ch:
        ref_E, ref_by_ch, ref_label = _analytic_reference(int(pcode), [model], centers, scale)

    n = len(channels)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    figsize = (10, 6) if n == 1 else (5 * ncols, 4.5 * nrows)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize,
        squeeze=False, sharex=True, sharey='row', constrained_layout=False
    )

    pname = _process_name_map().get(int(pcode), f"proc{int(pcode)}")
    legend_added = False
    for i, ch in enumerate(channels):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        count_ch = data["count_ch"][ch].astype(float)
        if use_counts:
            frac = np.divide(count_ch, count_total, out=np.zeros_like(count_total), where=count_total > 0)
            y = frac * mean_total
        else:
            sum_ch = data["sum_ch"][ch]
            y = np.divide(sum_ch, count_ch, out=np.zeros_like(sum_ch), where=count_ch > 0)

        mask = (count_ch > 0) & np.isfinite(y)
        if not np.any(mask):
            ax.set_visible(False)
            continue
        line_sim, = ax.plot(
            centers[mask],
            y[mask] * scale,
            "--",
            linewidth=2,
            c="dodgerblue",
            label="Simulation (binned)",
            zorder=2,
        )

        lines_ref = []
        if ref_E is not None and ref_by_ch:
            if 0 <= ch < len(ref_by_ch):
                y_ref = ref_by_ch[ch]
            else:
                y_ref = np.sum(np.vstack(ref_by_ch), axis=0)
            lr, = ax.plot(ref_E, y_ref, color="black", linewidth=2.5, alpha=0.9, label=ref_label or "Reference", zorder=1)
            lines_ref.append(lr)

        ax.set_xlabel("Kinetic Energy (eV)")
        ax.set_ylabel("Cross Section (10$^{-16}$ cm$^{2}$)")
        ax.set_title(f"{pname}_ch{ch}")
        if data["is_log"]:
            ax.set_xscale("log")

        if not legend_added and (lines_ref or line_sim is not None):
            handles = []
            labels = []
            if lines_ref:
                handles.extend(lines_ref)
                labels.extend([lr.get_label() for lr in lines_ref])
            if line_sim is not None:
                handles.append(line_sim); labels.append(line_sim.get_label())
            if handles:
                ax.legend(handles, labels, loc="best", frameon=True)
                legend_added = True

    total_axes = nrows * ncols
    for j in range(n, total_axes):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.align_ylabels([axes[r][0] for r in range(nrows) if axes[r][0].get_visible()])
    fig.tight_layout()
    outpath = _resolve_output(out_path)
    plt.show()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Wrote {outpath}")
    plt.close(fig)
    return outpath

def plot_channel_overlay_for_process_stream(acc: dict, pcode: int, model: str, scale: float,
                                            out_path: str, dat_path: str | None = None):
    """Streaming overlay for ionisation/excitation: all channels + total in one panel."""
    key = (int(pcode), str(model))
    if "xs" not in acc or key not in acc["xs"]:
        return None
    data = acc["xs"][key]
    centers = data["centers"]
    count_total = data["count_total"].astype(float)
    sum_total = data["sum_total"]
    mean_total = np.divide(sum_total, count_total, out=np.zeros_like(sum_total), where=count_total > 0)

    ref_E = None
    ref_by_ch = None
    if dat_path:
        dat_path = _prefer_total_ref(dat_path, int(pcode))
        ref_E, ref_by_ch = load_reference_from_path(dat_path)
        ref_by_ch = _scale_reference_if_needed(dat_path, ref_by_ch, scale)

    channels = sorted(data["count_ch"].keys())
    if not channels:
        channels = [0]

    fig, ax = plt.subplots(figsize=(8, 6))
    pname = _process_name_map().get(int(pcode), f"proc{int(pcode)}")
    cmap = plt.cm.tab10 if len(channels) <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, len(channels)))

    # Reference per-channel
    if ref_E is not None and ref_by_ch:
        for i, ch in enumerate(channels):
            if ch >= len(ref_by_ch):
                continue
            ax.plot(ref_E, ref_by_ch[ch], color=colors[i], linewidth=2.0, label=f"ch{ch} ref")

    # Simulation per-channel (counts-based)
    for i, ch in enumerate(channels):
        count_ch = data["count_ch"][ch].astype(float)
        frac = np.divide(count_ch, count_total, out=np.zeros_like(count_total), where=count_total > 0)
        y = frac * mean_total
        mask = (count_ch > 0) & np.isfinite(y)
        if not np.any(mask):
            continue
        ax.plot(
            centers[mask],
            y[mask] * scale,
            linestyle="--",
            marker="o",
            markersize=3,
            color=colors[i],
            label=f"ch{ch} sim",
        )

    if ref_E is not None and ref_by_ch:
        ref_cum = np.sum(np.vstack(ref_by_ch), axis=0)
        ax.plot(ref_E, ref_cum, color="gray", linewidth=4, label="total ref")

    if np.any(count_total > 0):
        mask = count_total > 0
        ax.plot(
            centers[mask],
            mean_total[mask] * scale,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="total sim",
        )

    ax.set_xlabel("Kinetic Energy (eV)")
    ax.set_ylabel("Cross Section (10$^{-16}$ cm$^{2}$)")
    ax.set_title(pname)
    if data["is_log"]:
        ax.set_xscale("log")
    ax.legend(loc="best", frameon=True, ncol=2)
    fig.tight_layout()
    outpath = _resolve_output(out_path)
    plt.show()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Wrote {outpath}")
    plt.close(fig)
    return outpath

def _overlay_lineshape(ax, shape: str, centers: list[float], widths: list[float] | None, bins):
    if not centers:
        return
    if shape == "gaussian" and widths:
        vmax = float(bins[-1]) if len(bins) else max(centers)
        for omega, b in zip(centers, widths):
            xg = np.linspace(0.0, vmax, 500)
            g = (1.0 / (np.sqrt(np.pi) * b)) * np.exp(-((xg - omega) ** 2) / (b * b))
            if np.max(g) > 0:
                g = g / np.max(g)
            ax.plot(xg, g, color="black", linewidth=2.5, alpha=0.95, zorder=3)
    elif shape == "delta":
        for omega in centers:
            ax.axvline(omega, color="black", linewidth=2.5, alpha=0.95, zorder=3)
        ax.set_ylim(0, 1.05)

def _plot_vib_hist(chans, bins_by_ch, hist_by_ch, ncols, out_path, model, lineshape):
    n = len(chans)
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False, sharex=True)
    for i, cidx in enumerate(chans):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        bins = bins_by_ch[int(cidx)]
        hist = hist_by_ch[int(cidx)]
        if hist.size:
            hmax = float(np.max(hist)) if np.max(hist) > 0 else 1.0
            hist_norm = hist / hmax
            centers = 0.5 * (bins[:-1] + bins[1:])
            ax.bar(centers, hist_norm, width=(bins[1] - bins[0]), color="lightgray", alpha=0.6, zorder=1)
            if lineshape:
                shape = lineshape.get("shape", "")
                centers_ls = lineshape.get("centers", [])
                widths_ls = lineshape.get("widths", None)
                if isinstance(centers_ls, list) and centers_ls:
                    if shape == "gaussian" and widths_ls and int(cidx) < len(centers_ls):
                        _overlay_lineshape(ax, "gaussian", [centers_ls[int(cidx)]], [widths_ls[int(cidx)]], bins)
                    elif shape == "delta" and int(cidx) < len(centers_ls):
                        _overlay_lineshape(ax, "delta", [centers_ls[int(cidx)]], None, bins)
            ax.set_ylim(0, 1.05)
        title = f"vib_{int(cidx)}"
        if model:
            title = f"{title} [{model}]"
        ax.set_title(title)
        ax.set_ylabel("Counts")
        ax.tick_params(axis="x", which="both", length=4, width=1.5, labelbottom=True)

    total_axes = nrows * ncols
    for j in range(len(chans), total_axes):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    for c in range(ncols):
        bottom_r = None
        for r in range(nrows - 1, -1, -1):
            if axes[r][c].get_visible():
                bottom_r = r; break
        if bottom_r is not None:
            axb = axes[bottom_r][c]
            axb.set_xlabel("Energy loss dE (eV)")
            axb.tick_params(axis="x", which="both", labelbottom=True)

    fig.tight_layout()
    outpath = _resolve_output(out_path)
    plt.show()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Wrote {outpath}")
    plt.close(fig)
    return outpath

def plot_vib_energy_loss_hist_stream(acc: dict, ncols: int, out_path: str, lineshapes: dict[str, dict] | None = None):
    """Streaming vib energy-loss histogram plotter."""
    vib = acc.get("vib", {})
    bins_by_ch = vib.get("bins", {})
    hist_by_ch = vib.get("hist", {})
    hist_by_model = vib.get("hist_by_model", {})
    models = vib.get("models", [])
    if lineshapes is None:
        lineshapes = {}
    if models and hist_by_model:
        for model in models:
            hbm = hist_by_model.get(model, {})
            chans = sorted([c for c in bins_by_ch.keys() if c in hbm])
            if not chans:
                continue
            _plot_vib_hist(chans, bins_by_ch, hbm, ncols, out_path, model, lineshapes.get(model))
        return out_path
    chans = sorted([c for c in bins_by_ch.keys() if c in hist_by_ch])
    if not chans:
        return None
    _plot_vib_hist(chans, bins_by_ch, hist_by_ch, ncols, out_path, None, None)
    return out_path

def plot_summary_stream(acc: dict, out_path: str, fontsize: float):
    """Streaming summary plot (4 panels)."""
    summary = acc.get("summary", {})
    if not summary:
        return None
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), squeeze=True, constrained_layout=True)

    # Panel 1: histogram of flagProcess
    ax1 = axs[0]
    proc_counts = summary.get("proc_counts", {})
    if proc_counts:
        uniq = np.array(sorted(proc_counts.keys()), dtype=float)
        counts = np.array([proc_counts[k] for k in uniq], dtype=float)
        ax1.bar(uniq, counts, width=0.9, color="#dddddd", edgecolor="none", label="All")
        cats = {
            "Excitation": [12,15,22,32,42,52,62],
            "Elastic": [11,21,31,41,51,61,110,210,410,510,710,120,220,420,520,720],
            "Ionisation": [13,23,33,43,53,63,73,130,230,430,530,730],
        }
        colors = {"Excitation": "#2ca02c", "Elastic": "#1f77b4", "Ionisation": "#d62728"}
        for name, ids in cats.items():
            mask = np.isin(uniq, ids)
            ax1.bar(uniq[mask], counts[mask], width=0.9, color=colors[name], alpha=0.7, label=name)
    ax1.set_yscale("log"); ax1.set_xlabel("flagProcess"); ax1.set_ylabel("Counts")

    # Panel 2: 3D scatter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    sx = summary["scatter"]["x"]; sy = summary["scatter"]["y"]; sz = summary["scatter"]["z"]
    if sx.size:
        ax2.scatter(sx, sy, sz, s=1, c="black", alpha=0.6)
    ax2.set_xlabel("x (nm)"); ax2.set_ylabel("y (nm)"); ax2.set_zlabel("z (nm)")

    # Panel 3: position histogram along x by process types
    ax3 = axs[2]
    centers = 0.5 * (summary["x_bins"][1:] + summary["x_bins"][:-1])
    proc_sets = {10:("Solv",'#9467bd'),11:("Elastic",'#d62728'),12:("Excit",'#2ca02c'),13:("Ionis",'#1f77b4'),14:("Attach",'#8c564b'),15:("Vib",'#e377c2')}
    for pid, (lab, col) in proc_sets.items():
        h = summary["x_hist_by_proc"].get(pid)
        if h is None:
            continue
        ax3.plot(centers, h, label=lab, color=col)
    ax3.set_xlabel("x (nm)"); ax3.set_yscale("log"); ax3.set_ylabel("Counts")

    # Panel 4: kinetic energy histogram for electrons
    ax4 = axs[3]
    k_centers = 0.5 * (summary["k_bins"][1:] + summary["k_bins"][:-1])
    ax4.hist(k_centers, bins=summary["k_bins"], weights=summary["k_hist"], histtype="stepfilled", alpha=0.7, color="#d62728")
    ax4.set_yscale("log"); ax4.set_xlabel("Kinetic Energy (eV)"); ax4.set_ylabel("Counts")

    outpath = _resolve_output(out_path)
    plt.show()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Wrote {outpath}")
    plt.close(fig)
    return outpath

def plot_deflection_angles_all_stream(acc: dict, out_path: str, fontsize: float = FONTSIZE):
    """Streaming deflection-angle plots per (process, model)."""
    if out_path is None:
        return None
    if "deflection" not in acc:
        return None
    base, ext = os.path.splitext(out_path)
    theta_bins = np.linspace(0, 180, 91)
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    for key, hist in acc["deflection"].items():
        proc_code, model_name = key
        if hist is None or not np.any(hist):
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(theta_centers, bins=theta_bins, weights=hist, histtype="stepfilled", alpha=0.8, color="lightgray")
        ax.set_xlim(0, 180)
        ax.set_xlabel(r"$\theta$ (deg)")
        ax.set_ylabel("Counts")
        title = model_name if model_name else f"proc {int(proc_code)}"
        ax.set_title(title)
        fig.tight_layout()
        suffix = f"proc{int(proc_code)}"
        if model_name:
            suffix += f"_{_sanitize_label(model_name)}"
        outname = f"{base}_{suffix}{ext or '.png'}"
        out_resolved = _resolve_output(outname)
        plt.show()
        fig.savefig(out_resolved, bbox_inches="tight")
        print(f"Wrote {out_resolved}")
        plt.close(fig)
    return out_path

def plot_elastic_reference_vs_sim(arrs, ncols: int, scale: float, nH2O_cm3: float,
                                  out_path: str, dat_path: str | None = None,
                                  config_refs: dict[str, str] | None = None):
    """Multi-panel elastic XS comparison: reference (black) vs simulation (blue dashed)."""
    if arrs is None or "kineticEnergy" not in arrs:
        return None

    proc_names = _decode_to_str_array(arrs.get("processName"))
    model_names = _decode_to_str_array(arrs.get("modelName"))
    flag_proc = np.asarray(arrs["flagProcess"], dtype=int) if "flagProcess" in arrs else None

    macro_keys = [
        "vibCrossSection",
        "macroCrossSection",
        "processCrossSection",
    ]

    groups: list[dict] = []
    if proc_names is not None:
        elastic_mask = np.array([("elastic" in str(p).lower()) for p in proc_names], dtype=bool)
        uniq_names = np.unique(proc_names[elastic_mask])
        for name in uniq_names:
            if not name:
                continue
            mask = (proc_names == name)
            groups.append({"label": name, "mask": mask})
    if not groups and flag_proc is not None:
        elastic_codes = [11, 21, 31, 41, 51, 61, 110, 210, 410, 510, 710]
        uniq_codes = np.unique(flag_proc[np.isin(flag_proc, elastic_codes)])
        for code in uniq_codes:
            mask = (flag_proc == code)
            label = _process_name_map().get(int(code), f"proc{int(code)}")
            groups.append({"label": label, "mask": mask})
    if not groups:
        return None

    n = len(groups)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows),
        squeeze=False, sharex=True, sharey='row', constrained_layout=False
    )

    legend_added = False
    for i, g in enumerate(groups):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        mask = g["mask"]

        ke = np.asarray(arrs["kineticEnergy"], dtype=float)[mask]
        if ke.size == 0:
            ax.set_visible(False)
            continue

        xs_macro = None
        for k in macro_keys:
            if k in arrs:
                xs_macro = np.asarray(arrs[k], dtype=float)[mask]
                break
        if xs_macro is None:
            ax.set_visible(False)
            continue

        xs_micro_cm2 = _to_micro_cm2(xs_macro, nH2O_cm3)
        if "channelMicroXS" in arrs:
            xs_micro = np.asarray(arrs["channelMicroXS"], dtype=float)[mask]
            xs_micro_cm2 = np.where(xs_micro > 0.0, xs_micro, xs_micro_cm2)

        order = np.argsort(ke)
        line_sim, = ax.plot(
            ke[order],
            xs_micro_cm2[order] * scale,
            "--",
            linewidth=2,
            c="dodgerblue",
            zorder=2,
            label="Simulation",
        )

        model_hints: list[str] = []
        if model_names is not None:
            uniq_models = np.unique(model_names[mask])
            model_hints.extend([m for m in uniq_models if m])
        model_hints.append(g.get("label", ""))

        ref_line = None
        ref_E = None
        ref_path = _resolve_elastic_reference_path(
            model_hints, fallback=dat_path, config_refs=config_refs
        )
        if ref_path:
            try:
                ref_E, ref_by_ch = load_reference_from_path(ref_path)
                if ref_by_ch:
                    y_ref = ref_by_ch[0] if len(ref_by_ch) else None
                    if y_ref is not None and y_ref.size:
                        ref_line, = ax.plot(
                            ref_E,
                            y_ref,
                            color="black",
                            linewidth=2,
                            alpha=0.9,
                            zorder=1,
                            label=Path(ref_path).name if ref_path else "Reference",
                        )
            except Exception:
                ref_line = None

        ax.set_xlabel("Kinetic Energy (eV)")
        ax.set_ylabel("Cross Section (10$^{-16}$ cm$^{2}$)")
        ax.set_title(g["label"])

        span_vals = []
        if ref_E is not None and ref_E.size:
            pos_ref = ref_E[ref_E > 0]
            if pos_ref.size:
                span_vals.append(np.nanmin(pos_ref))
                span_vals.append(np.nanmax(ref_E))
        if ke.size:
            kpos = ke[ke > 0]
            if kpos.size:
                span_vals.append(np.nanmin(kpos))
                span_vals.append(np.nanmax(kpos))
        if span_vals:
            e_min = float(np.nanmin(span_vals))
            e_max = float(np.nanmax(span_vals))
            if e_max / max(e_min, 1e-30) > 1e3:
                ax.set_xscale("log")

        if not legend_added and (ref_line is not None or line_sim is not None):
            handles = []
            labels = []
            if ref_line is not None:
                handles.append(ref_line); labels.append(ref_line.get_label())
            if line_sim is not None:
                handles.append(line_sim); labels.append(line_sim.get_label())
            if handles:
                ax.legend(handles, labels, loc="best", frameon=True)
                legend_added = True

    total_axes = nrows * ncols
    for j in range(n, total_axes):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            if not ax.get_visible():
                continue
            if c != 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
    for r in range(nrows - 1):
        for c in range(ncols):
            ax = axes[r][c]
            if not ax.get_visible():
                continue
            ax.set_xlabel("")
            ax.tick_params(labelbottom=True)

    fig.align_ylabels([axes[r][0] for r in range(nrows) if axes[r][0].get_visible()])
    fig.tight_layout()
    outpath = _resolve_output(out_path)
    plt.show()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Wrote {outpath}")
    plt.close(fig)
    return outpath

def plot_vib_energy_loss_hist(arrs, ncols: int, out_path: str, lineshapes: dict[str, dict] | None = None):
    """Plot energy-loss histograms per vib channel using kineticEnergyDifference."""
    if arrs is None or "kineticEnergyDifference" not in arrs or "channelIndex" not in arrs:
        return None
    if lineshapes is None:
        lineshapes = {}
    m_vib = (np.asarray(arrs["flagProcess"], dtype=float) == 15.0)
    if not np.any(m_vib):
        return None
    dE_all = np.asarray(arrs["kineticEnergyDifference"], dtype=float)[m_vib]
    ch_all = np.asarray(arrs["channelIndex"], dtype=int)[m_vib]
    model_all = _decode_to_str_array(arrs.get("modelName"))
    if model_all is None:
        model_all = np.full(m_vib.shape, "", dtype=object)
    model_all = model_all[m_vib]

    models = sorted([m for m in np.unique(model_all) if m])
    if not models:
        models = [""]

    for model in models:
        m_model = (model_all == model) if model else np.ones_like(ch_all, dtype=bool)
        if not np.any(m_model):
            continue
        dE = dE_all[m_model]
        ch = ch_all[m_model]
        chans = np.unique(ch[ch >= 0])
        if chans.size == 0:
            continue
        bins_by_ch = {}
        hist_by_ch = {}
        for cidx in chans:
            vals = dE[ch == cidx]
            vmax = float(np.nanmax(vals)) if vals.size and np.isfinite(np.nanmax(vals)) else 1.0
            bins = np.linspace(0.0, vmax, 81)
            h, _ = np.histogram(vals, bins=bins)
            bins_by_ch[int(cidx)] = bins
            hist_by_ch[int(cidx)] = h
        _plot_vib_hist(chans.tolist(), bins_by_ch, hist_by_ch, ncols, out_path, model or None, lineshapes.get(model))
    return out_path

def plot_summary(arrs, out_path: str, fontsize: float):
    """Create a compact 4-panel summary of the simulation arrays."""
    if arrs is None:
        return None
    fp   = arrs.get("flagParticle"); fl_p = arrs.get("flagProcess")
    kinE = arrs.get("kineticEnergy"); x_nm = arrs.get("x"); y_nm = arrs.get("y"); z_nm = arrs.get("z")
    if fp is None or fl_p is None or kinE is None or x_nm is None or y_nm is None or z_nm is None:
        return None
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), squeeze=True, constrained_layout=True)

    # Panel 1: histogram of flagProcess with category overlays
    ax1 = axs[0]
    vals = np.asarray(fl_p, dtype=float)
    uniq, counts = np.unique(vals, return_counts=True)
    ax1.bar(uniq, counts, width=0.9, color="#dddddd", edgecolor="none", label="All")
    cats = {
        "Excitation": [12,15,22,32,42,52,62],
        "Elastic": [11,21,31,41,51,61,110,210,410,510,710,120,220,420,520,720],
        "Ionisation": [13,23,33,43,53,63,73,130,230,430,530,730],
    }
    colors = {"Excitation": "#2ca02c", "Elastic": "#1f77b4", "Ionisation": "#d62728"}
    for name, ids in cats.items():
        mask = np.isin(uniq, ids)
        ax1.bar(uniq[mask], counts[mask], width=0.9, color=colors[name], alpha=0.7, label=name)
    ax1.set_yscale('log'); ax1.set_xlabel("flagProcess"); ax1.set_ylabel("Counts")

    # Panel 2: 3D scatter x:y:z for electrons (downsample)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    idx = np.where(np.asarray(fp)==1)[0]
    step = max(1, idx.size // 50000) if idx.size > 50000 else 1
    idx = idx[::step]
    ax2.scatter(x_nm[idx], y_nm[idx], z_nm[idx], s=1, c='black', alpha=0.6)
    ax2.set_xlabel("x (nm)"); ax2.set_ylabel("y (nm)"); ax2.set_zlabel("z (nm)")

    # Panel 3: position histogram along x by process types
    ax3 = axs[2]
    bins = np.linspace(0, 2000, 101)
    proc_sets = {10:("Solv",'#9467bd'),11:("Elastic",'#d62728'),12:("Excit",'#2ca02c'),13:("Ionis",'#1f77b4'),14:("Attach",'#8c564b'),15:("Vib",'#e377c2')}
    for pid, (lab, col) in proc_sets.items():
        m = (np.asarray(fl_p)==pid)
        h, be = np.histogram(x_nm[m], bins=bins)
        centers = 0.5*(be[1:]+be[:-1])
        ax3.plot(centers, h, label=lab, color=col)
    ax3.set_xlabel("x (nm)"); ax3.set_yscale('log'); ax3.set_ylabel("Counts")

    # Panel 4: kinetic energy histogram for electrons
    ax4 = axs[3]
    m = (np.asarray(fp)==1)
    kmax = np.nanmax(np.asarray(kinE)[m]) if np.any(m) else np.nanmax(np.asarray(kinE))
    rng = (0, float(kmax) if np.isfinite(kmax) and kmax>0 else 2000)
    ax4.hist(np.asarray(kinE)[m], bins=100, range=rng, histtype='stepfilled', alpha=0.7, color='#d62728')
    ax4.set_yscale('log'); ax4.set_xlabel("Kinetic Energy (eV)"); ax4.set_ylabel("Counts")

    outpath = _resolve_output(out_path)
    plt.show()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Wrote {outpath}")
    plt.close(fig)
    return outpath

def _born_angular_distribution(theta_deg, E_kin_eV, E_sec_eV):
    """
    Compute theoretical Born approximation angular distribution for ionisation.
    
    Based on G4DNABornAngle::SampleDirectionForShell:
    - E_sec < 50 eV: Isotropic (uniform in cos(theta))
    - 50 eV <= E_sec <= 200 eV: 90% forward peaked (0-45°), 10% isotropic
    - E_sec > 200 eV: Born approximation formula
    
    Args:
        theta_deg: array of angles in degrees
        E_kin_eV: incident kinetic energy in eV
        E_sec_eV: secondary electron energy in eV
    
    Returns:
        Normalized PDF values at each theta
    """
    theta_rad = np.deg2rad(theta_deg)
    
    if E_sec_eV < 50:
        # Isotropic: uniform in cos(theta)
        pdf = np.sin(theta_rad) / 2.0  # d(cos)/dtheta = sin(theta), integral over hemisphere = 2
    elif E_sec_eV <= 200:
        # Mixed forward/isotropic
        # 90% in forward cone (0-45°), 10% isotropic
        forward_mask = theta_deg <= 45
        pdf = np.zeros_like(theta_rad)
        pdf[forward_mask] = 0.9 / (2 * np.pi * (1 - np.cos(np.pi/4))) * np.sin(theta_rad[forward_mask])
        pdf[~forward_mask] = 0.1 * np.sin(theta_rad[~forward_mask]) / 2.0
    else:
        # Born approximation: P(theta) ~ sin^3(theta) / (1 + E_sec/(2*m_e*c^2) - cos(theta))^2
        # where sin^2(theta) = (1 - E_sec/E_kin) / (1 + E_sec/(2*m_e*c^2))
        m_e_c2 = 510998.95  # electron rest mass in eV
        sin2_max = (1 - E_sec_eV/E_kin_eV) / (1 + E_sec_eV/(2*m_e_c2))
        sin2 = np.sin(theta_rad)**2
        
        # Born formula (approximate)
        pdf = (np.sin(theta_rad) * sin2) / (1 + E_sec_eV/(2*m_e_c2) - np.cos(theta_rad))**2
        
        # Suppress unphysical angles where sin^2 > sin2_max
        pdf[sin2 > sin2_max] = 0
    
    # Normalize
    integral = np.trapz(pdf, theta_rad)
    if integral > 0:
        pdf /= integral
    
    return pdf

def plot_deflection_angles_all(arrs, out_path: str, fontsize: float = FONTSIZE):
    """Plot deflection-angle histograms per (process, model) pair (channels aggregated), one figure each."""
    if out_path is None:
        return None
    if arrs is None or "cosTheta" not in arrs or "flagProcess" not in arrs:
        return None

    cos_theta = np.asarray(arrs["cosTheta"], dtype=float)
    flag_process = np.asarray(arrs["flagProcess"], dtype=int)
    model_names = _decode_to_str_array(arrs.get("modelName"))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))

    # Build groups keyed by (proc, model)
    keys_list = []
    if model_names is not None:
        for p, m in zip(flag_process, model_names):
            keys_list.append((int(p), m if m else ""))
    else:
        for p in flag_process:
            keys_list.append((int(p), ""))
    unique_keys = []
    seen = set()
    for k in keys_list:
        if k in seen:
            continue
        seen.add(k)
        unique_keys.append(k)

    if not unique_keys:
        return None

    base, ext = os.path.splitext(out_path)
    for proc_code, model_name in unique_keys:
        fig, ax = plt.subplots(figsize=(6, 4))
        if model_names is not None:
            mask = (flag_process == int(proc_code)) & (model_names == model_name)
        else:
            mask = (flag_process == int(proc_code))
        thetas = theta_deg[mask]
        if thetas.size == 0:
            plt.close(fig)
            continue
        ax.hist(thetas, bins=90, range=(0, 180), histtype="stepfilled", alpha=0.8, color='lightgray')
        ax.set_xlim(0, 180)
        ax.set_xlabel(r"$\theta$ (deg)")
        ax.set_ylabel("Counts")
        title = model_name if model_name else f"proc {int(proc_code)}"
        ax.set_title(title)
        fig.tight_layout()
        suffix = f"proc{int(proc_code)}"
        if model_name:
            suffix += f"_{_sanitize_label(model_name)}"
        outname = f"{base}_{suffix}{ext or '.png'}"
        out_resolved = _resolve_output(outname)
        plt.show()
        fig.savefig(out_resolved, bbox_inches="tight")
        print(f"Wrote {out_resolved}")
        plt.close(fig)
    return out_path

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Plot cross-sections, deflection angles, and summaries from dna.root and optional reference .dat")
    ap.add_argument("--root", default="build/europa_test_e2.root", help="Path to ROOT file (default: build/dna.root)")
    ap.add_argument("--process", type=int, default=15, help="Single process code to plot when --processes is not given (default: 15)")
    ap.add_argument("--processes", default=None, help="Comma-separated process codes or 'all' to iterate over all present in ROOT")
    ap.add_argument("--dat", default=None, help="Absolute path to a reference .dat file (energy + partial XS columns). Overrides auto lookup")
    ap.add_argument("--out", default="xs_channels.png", help="Base output filename for XS plots (suffixes per process)")
    ap.add_argument("--summary_out", default="summary_panels.png", help="Output image for 4-panel summary")
    ap.add_argument("--deflection_out", default="deflection_angles.png", help="Output image for deflection-angle distributions")
    ap.add_argument("--de_hist_out", default="de_hist_vib.png", help="Output image for vib energy-loss histograms")
    ap.add_argument(
        "--initial_angles_out",
        default="initial_angle_diagnostics.png",
        help="Output image for initial-angle diagnostics (all vs incomplete deposition)",
    )
    ap.add_argument(
        "--deposition-epsilon-frac",
        type=float,
        default=1e-6,
        help="Fractional tolerance on missing energy for classifying incomplete deposition",
    )
    ap.add_argument("--ncols", type=int, default=5, help="Max panels per row (default: 5)")
    ap.add_argument("--scale", type=float, default=1e16, help="Y-scale multiplier for microscopic XS (default: 1e16)")
    ap.add_argument("--nH2O_cm3", type=float, default=3.343e22, help="Number density (cm^-3) for macro→micro conversion")
    ap.add_argument("--fontsize", type=float, default=18, help="Base font size for ticks, labels, titles, legend")
    ap.add_argument("--summary", action="store_true", help="Also write the 4-panel summary figure")
    ap.add_argument("--stream", action="store_true", help="Stream ROOT in batches (use for very large files)")
    ap.add_argument("--step-size", type=int, default=200000, help="Entries per batch for --stream (default: 200000)")
    args = ap.parse_args()

    # Font sizes are controlled globally via FONTSIZE

    root_arg = args.root
    root_paths = resolve_root_paths(root_arg)
    if not root_paths:
        raise FileNotFoundError(root_arg)
    root_path = root_paths[0]
    config = load_config_from_root(root_arg)
    config_refs = resolve_config_reference_paths(config) if config else {}
    model_refs = resolve_model_reference_paths(config) if config else {}
    lineshapes = resolve_model_lineshapes(config) if config else {}

    if config_refs:
        print("ROOT config references:")
        for key in sorted(config_refs.keys()):
            print(f"  {key}: {config_refs[key]}")
    if model_refs:
        print("ROOT model references:")
        for key in sorted(model_refs.keys()):
            print(f"  {key}: {model_refs[key]}")
    if lineshapes:
        print("ROOT model lineshapes:")
        for key in sorted(lineshapes.keys()):
            shape = lineshapes[key].get("shape", "")
            print(f"  {key}: {shape}")

    if args.stream:
        meta = _stream_collect_meta(root_arg, "step", args.step_size)
        _print_root_processes_stream(meta)
        acc = _init_stream_acc(meta, args.nH2O_cm3)
        _stream_accumulate(root_arg, "step", args.step_size, args.nH2O_cm3, meta, acc)

        # Determine processes to plot
        if args.processes is None or str(args.processes).strip().lower() == "all":
            proc_list = sorted(meta.get("proc_counts", {}).keys())
        elif args.processes:
            proc_list = [int(x.strip()) for x in str(args.processes).split(",") if x.strip()]
        else:
            proc_list = [int(args.process)]

        base, ext = os.path.splitext(args.out)
        pname_map = _process_name_map()
        ref_by_proc: dict[int, str | None] = {}
        if config_refs and args.dat is None:
            ref_by_proc = {
                12: config_refs.get("ref_excitation") or config_refs.get("ref_excitation_born"),
                13: config_refs.get("ref_ionisation") or config_refs.get("ref_ionisation_born"),
                14: config_refs.get("ref_attachment"),
                15: config_refs.get("ref_vib"),
            }

        for pcode in proc_list:
            dat_path = args.dat if args.dat else ref_by_proc.get(int(pcode))
            suffix = pname_map.get(int(pcode), str(int(pcode)))
            models = meta.get("models_by_proc", {}).get(int(pcode), set())
            if not models:
                models = {""}

            if int(pcode) in (12, 13):
                for model in sorted(models):
                    msafe = f"_{_sanitize_label(model)}" if model else ""
                    outname = f"{base}_{suffix}{msafe}{ext or '.png'}"
                    ref_for_model = _prefer_total_ref(model_refs.get(model, dat_path), int(pcode))
                    plot_channel_overlay_for_process_stream(
                        acc=acc,
                        pcode=int(pcode),
                        model=model,
                        scale=args.scale,
                        out_path=outname,
                        dat_path=ref_for_model,
                    )
                continue

            for model in sorted(models):
                msafe = f"_{_sanitize_label(model)}" if model else ""
                outname = f"{base}_{suffix}{msafe}{ext or '.png'}"
                ref_for_model = _prefer_total_ref(model_refs.get(model, dat_path), int(pcode))
                plot_cross_sections_for_process_stream(
                    acc=acc,
                    pcode=int(pcode),
                    model=model,
                    ncols=args.ncols,
                    scale=args.scale,
                    out_path=outname,
                    dat_path=ref_for_model,
                    config_refs=config_refs,
                )

        plot_vib_energy_loss_hist_stream(acc, args.ncols, args.de_hist_out, lineshapes)
        plot_summary_stream(acc, args.summary_out, fontsize=FONTSIZE)
        plot_deflection_angles_all_stream(acc, args.deflection_out, fontsize=FONTSIZE)
        plot_initial_angle_diagnostics(
            root_arg=root_arg,
            out_path=args.initial_angles_out,
            deposition_epsilon_frac=float(args.deposition_epsilon_frac),
            step_size=int(args.step_size),
        )
        return

    # ---- non-streaming path ----
    arrs = None
    if os.path.exists(root_path):
        arrs = load_arrays(root_arg, "step")

    print_root_processes(arrs)

    # Determine processes to plot
    if arrs is not None and (args.processes is None or str(args.processes).strip().lower() == 'all'):
        proc_list = np.unique(np.asarray(arrs["flagProcess"], dtype=float)).astype(int).tolist()
    elif args.processes:
        proc_list = [int(x.strip()) for x in str(args.processes).split(',') if x.strip()]
    else:
        proc_list = [int(args.process)]

    # Plot per-process XS
    base, ext = os.path.splitext(args.out)
    pname_map = _process_name_map()
    ref_by_proc: dict[int, str | None] = {}
    if config_refs and args.dat is None:
        ref_by_proc = {
            12: config_refs.get("ref_excitation") or config_refs.get("ref_excitation_born"),
            13: config_refs.get("ref_ionisation") or config_refs.get("ref_ionisation_born"),
            14: config_refs.get("ref_attachment"),
            15: config_refs.get("ref_vib"),
        }
    for pcode in proc_list:
        dat_path = args.dat if args.dat else ref_by_proc.get(int(pcode))
        suffix = pname_map.get(int(pcode), str(int(pcode)))
        if int(pcode) in (12, 13):
            # If model names are present, emit one plot per model using model_ref
            if arrs is not None and "modelName" in arrs:
                models_all = _decode_to_str_array(arrs["modelName"])
                if models_all is not None:
                    mproc = (np.asarray(arrs["flagProcess"], dtype=float) == float(pcode))
                    uniq_models = np.unique(models_all[mproc])
                    for mname in uniq_models:
                        if not mname:
                            continue
                        msafe = _sanitize_label(mname)
                        outname = f"{base}_{suffix}_{msafe}{ext or '.png'}"
                        ref_for_model = _prefer_total_ref(model_refs.get(mname, dat_path), int(pcode))
                        plot_channel_overlay_for_process(
                            arrs=arrs,
                            pcode=int(pcode),
                            scale=args.scale,
                            nH2O_cm3=args.nH2O_cm3,
                            out_path=outname,
                            dat_path=ref_for_model,
                        )
                    continue
            outname = f"{base}_{suffix}{ext or '.png'}"
            plot_channel_overlay_for_process(
                arrs=arrs,
                pcode=int(pcode),
                scale=args.scale,
                nH2O_cm3=args.nH2O_cm3,
                out_path=outname,
                dat_path=dat_path,
            )
            continue
        # If multiple models exist, emit one plot per model using model_ref when available
        if arrs is not None and "modelName" in arrs:
            models_all = _decode_to_str_array(arrs["modelName"])
            if models_all is not None:
                mproc = (np.asarray(arrs["flagProcess"], dtype=float) == float(pcode))
                uniq_models = np.unique(models_all[mproc])
                uniq_models = [m for m in uniq_models if m]
                if uniq_models:
                    for mname in uniq_models:
                        msafe = _sanitize_label(mname)
                        outname = f"{base}_{suffix}_{msafe}{ext or '.png'}"
                        ref_for_model = _prefer_total_ref(model_refs.get(mname, dat_path), int(pcode))
                        plot_cross_sections_for_process(
                            arrs=arrs,
                            pcode=int(pcode),
                            ncols=args.ncols,
                            scale=args.scale,
                            nH2O_cm3=args.nH2O_cm3,
                            out_path=outname,
                            dat_path=ref_for_model,
                            model_filter=mname,
                            config_refs=config_refs,
                        )
                    continue

        outname = f"{base}_{suffix}{ext or '.png'}"
        plot_cross_sections_for_process(
            arrs=arrs,
            pcode=int(pcode),
            ncols=args.ncols,
            scale=args.scale,
            nH2O_cm3=args.nH2O_cm3,
            out_path=outname,
            dat_path=dat_path,
            config_refs=config_refs,
        )

    # Vib energy-loss histograms (if available)
    plot_vib_energy_loss_hist(arrs, args.ncols, args.de_hist_out, lineshapes)

    # Summary and deflection angles
    plot_summary(arrs, args.summary_out, fontsize=FONTSIZE)
    plot_deflection_angles_all(arrs, args.deflection_out, fontsize=FONTSIZE)
    plot_initial_angle_diagnostics(
        root_arg=root_arg,
        out_path=args.initial_angles_out,
        deposition_epsilon_frac=float(args.deposition_epsilon_frac),
        step_size=int(args.step_size),
    )


if __name__ == "__main__":
    main()
