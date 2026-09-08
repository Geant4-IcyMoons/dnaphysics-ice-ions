"""Non-transport tables retaining rejected finite polarization estimates."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import json

import numpy as np
from tqdm import tqdm
from physics.inelastic_dielectric import checkpoints
from physics.inelastic_dielectric.polarization import barkas_dcs as bd, screened_barkas as sb

FLAGS = {1: "quadrature_not_converged", 2: "nonfinite_value_or_error",
         4: "outside_velocity_domain", 8: "abs_correction_at_least_born",
         16: "negative_total"}


def _diagnostic_row(task):
    energy, w, mass, element, charge, path = task
    signature = json.dumps(dict(energy=energy, mass=mass, element=element, charge=charge), sort_keys=True)
    value, error = np.full(w.shape, np.nan), np.full(w.shape, np.nan)
    flags, done = np.zeros(w.shape, np.uint8), np.zeros(w.shape, bool)
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["signature"]) != signature or not np.array_equal(saved["W"], w):
                raise ValueError("Incompatible diagnostic checkpoint")
            value, error, flags, done = (saved[key].copy() for key in ("value", "error", "flags", "done"))
            if any(array.shape != w.shape for array in (value, error, flags, done)) or done.dtype != np.dtype(bool):
                raise ValueError("Invalid diagnostic checkpoint")
    for i in tqdm(np.flatnonzero(~done), total=w.size, initial=int(done.sum()),
                  desc=f"Diagnostic polarization T={energy:g}", unit="loss", mininterval=60):
        v, e, f = sb._energy_row((energy, w[i:i+1], mass, element, charge), diagnostic_only=True)
        value[i] = v[0] if np.isfinite(v[0]) else np.nan
        error[i] = e[0] if np.isfinite(e[0]) else np.nan
        flags[i], done[i] = f[0], True
        checkpoints.atomic_savez(path, signature=signature, W=w, value=value, error=error, flags=flags, done=done)
    return value, error, flags


def write_diagnostic_tables(data, s, density, mass, out_dir, checkpoint_dir, workers, metadata, scale, dat_names=None):
    """Retain raw estimates in diagnostic products and optional standard-layout DAT."""
    if density is None or density.electrons == 0:
        raise ValueError("Diagnostic polarization mode requires an electron-bearing projectile")
    out_dir, checkpoint_dir = Path(out_dir), Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints.require_manifest(checkpoint_dir, dict(source=checkpoints.source_digest(),
        metadata=metadata, mass=mass, diagnostic_only=True))
    t, w = np.asarray(data["T_line"]), np.asarray(data["E_line"])
    born = scale*(np.sum(data["exc_vals"], axis=1)+np.sum(data["ion_vals"], axis=1))
    oos = bd.oos_density(w, s, material=s.material, include_kshell=metadata["include_kshell"])
    active = (w <= bd.wmax_eV(t, mass)) & (oos.df_dW_total > 0)
    correction, error, flags = np.zeros_like(w), np.zeros_like(w), np.zeros(w.shape, np.uint8)
    tasks, indices = [], []
    for energy in np.unique(t[active]):
        idx = np.flatnonzero(active & (t == energy))
        indices.append(idx)
        tasks.append((float(energy), w[idx], mass, density.element, density.charge,
                      checkpoint_dir / f"energy_{float(energy).hex()}.npz"))

    def accept(i, row):
        idx = indices[i]
        correction[idx], error[idx] = row[0]*oos.df_dW_total[idx], row[1]*oos.df_dW_total[idx]
        flags[idx] = row[2]

    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(_diagnostic_row, task): i for i, task in enumerate(tasks)}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Diagnostic energies", unit="energy"):
                accept(futures[future], future.result())
    else:
        for i, task in enumerate(tqdm(tasks, desc="Diagnostic energies", unit="energy")):
            accept(i, _diagnostic_row(task))
    flags[(born > 0) & (np.abs(correction) >= born)] |= 8
    total = born + correction
    flags[total < 0] |= 16
    ratio = np.divide(correction, born, out=np.full_like(born, np.nan), where=born > 0)
    report = dict(metadata, diagnostic_only=True, transport_ready=False,
                  dat_scale_m2=scale, dat_convention="standard legacy DCS/TCS scaling; multiply by dat_scale_m2",
                  flag_bits=FLAGS, failure_counts={reason: int(np.count_nonzero(flags & bit)) for bit, reason in FLAGS.items()},
                  units="energy eV; DCS and error m2/eV", rows=int(w.size))
    # Use exactly the existing Born-proportional channel allocation, including
    # its zero-Born fallback. Do not clip negative values or replace missing ones.
    channels = scale*np.hstack((data["exc_vals"], data["ion_vals"]))
    good = born > 0
    channels[good] += channels[good]*(correction[good]/born[good])[:, None]
    fallback = (~good) & (correction != 0)
    first_ion = data["exc_vals"].shape[1]
    channels[fallback, first_ion if data["ion_vals"].shape[1] else 0] += correction[fallback]
    if dat_names is not None:
        exc_name, ion_name, exc_total_name, ion_total_name = dat_names
        for name, values in ((exc_name, channels[:, :first_ion]), (ion_name, channels[:, first_ion:])):
            with checkpoints.atomic_text(out_dir / name) as stream:
                stream.write("# ion_table_metadata: "+json.dumps(report, sort_keys=True)+"\n")
                for i in tqdm(range(w.size), desc=f"Writing {name}", unit="row"):
                    stream.write(" ".join(f"{x:.16E}" for x in (t[i], w[i], *(values[i]/scale)))+"\n")
        for name, values in ((exc_total_name, channels[:, :first_ion]), (ion_total_name, channels[:, first_ion:])):
            with checkpoints.atomic_text(out_dir / name) as stream:
                stream.write("# ion_table_metadata: "+json.dumps(report, sort_keys=True)+"\n")
                for energy in tqdm(np.unique(t), desc=f"Writing {name}", unit="energy"):
                    idx = np.flatnonzero(t == energy)
                    totals = np.trapezoid(values[idx]/scale, w[idx], axis=0)
                    stream.write(" ".join(f"{x:.16E}" for x in (energy, *totals))+"\n")
    with checkpoints.atomic_text(out_dir / "DIAGNOSTIC_ONLY.json") as stream:
        json.dump(report, stream, indent=2)
    checkpoints.atomic_savez(out_dir / "DIAGNOSTIC_ONLY.npz", metadata=json.dumps(report),
        T_eV=t, W_eV=w, born_m2_eV=born, polarization_m2_eV=correction, total_m2_eV=total,
        correction_over_born=ratio, quadrature_error_m2_eV=error, rejection_flags=flags,
        born_excitation_channels_m2_eV=scale*data["exc_vals"], born_ionization_channels_m2_eV=scale*data["ion_vals"])
    with checkpoints.atomic_text(out_dir / "DIAGNOSTIC_ONLY.csv") as stream:
        stream.write("# DIAGNOSTIC ONLY — NOT FOR TRANSPORT; flag definitions in DIAGNOSTIC_ONLY.json\n")
        stream.write("T_eV,W_eV,born_m2_eV,polarization_m2_eV,total_m2_eV,correction_over_born,quadrature_error_m2_eV,rejection_flags\n")
        for row in tqdm(zip(t, w, born, correction, total, ratio, error, flags), total=w.size, desc="Writing diagnostic rows"):
            stream.write(",".join(format(float(v), ".17g") for v in row)+"\n")
    print(f"Diagnostic-only tables saved to {out_dir}; rejected rows retained and flagged.")
