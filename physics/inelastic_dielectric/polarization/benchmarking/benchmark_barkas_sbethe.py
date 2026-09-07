#!/usr/bin/env python3
"""External SBETHE benchmark for the Barkas DCS implementation.

The benchmark downloads the official SBETHE v2 distribution from Mendeley
Data, verifies the ZIP SHA-256, compiles the Fortran source with gfortran,
and runs the corrected-Bethe calculation for liquid water/H2O.

Two comparisons are reported:

1. Formula parity:
   our Barkas DCS kernel is integrated using SBETHE's own water OOS table.
   This isolates the Barkas formula, ARBI1/ARBI2 functions, Wmax, charge,
   and unit conversion from differences in target optical data.

2. Production target:
   our Barkas DCS kernel is integrated using the Geant4-table-path H2O OOS
   density from barkas_dcs.oos_density(). This is the actual pipeline target
   model and is expected to differ from SBETHE's DHFS water OOS.

Unit conversion:
    S_eV_cm2 = integral W * (d sigma_B / dW) dW
is a stopping cross section per H2O molecule. The mass stopping-power unit is
    S_MeV_cm2_g = S_eV_cm2 * (N_A / M_H2O) * 1.0e-6,
because N_A / M_H2O is the number of molecules per gram and 1 MeV = 1e6 eV.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PHYSICS_ICE_ROOT = HERE.parents[2]

from physics.inelastic_dielectric.polarization import barkas_dcs  # noqa: E402
from physics.inelastic_dielectric.finite_q import emfietzoglou_model_finite_q as model  # noqa: E402
from physics.constants import (  # noqa: E402
    AVOGADRO,
    FONT_COURIER,
    H2O_MOLAR_MASS_G_MOL,
    RC_BASE_ELASTIC,
    rcparams_with_fontsize,
)


SBETHE_DATASET_ID = "7zw25f428t"
SBETHE_DATASET_VERSION = 2
SBETHE_FILE_ID = "a7d2eed6-9aab-4462-936d-b15adf0e7c16"
SBETHE_ZIP_SHA256 = "d5d4879c2073ada3bd799fe0727054549cd6ec65cc699acbd0ff25fd5c3c4144"
SBETHE_DIRECT_URL = (
    "https://data.mendeley.com/public-files/datasets/"
    f"{SBETHE_DATASET_ID}/files/{SBETHE_FILE_ID}/file_downloaded"
)
SBETHE_FILE_LIST_URL = (
    "https://data.mendeley.com/public-api/datasets/"
    f"{SBETHE_DATASET_ID}/files?folder_id=root&version={SBETHE_DATASET_VERSION}"
)
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 BarkasDCSBenchmark/1.0",
    "Accept": "application/vnd.mendeley-public-dataset.1+json,*/*",
}

SBETHE_HREV_EV = 27.211386245988
SBETHE_A0B_CM = 5.29177210903e-9
SBETHE_REV_EV = 510.9989500e3
PROTON_MASS_ME_SBETHE = 1836.15267343
PROTON_REST_EV_SBETHE = PROTON_MASS_ME_SBETHE * SBETHE_REV_EV

DEFAULT_ENERGIES_MEV = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DEFAULT_OUT_DIR = HERE / "plots" / "sbethe_v2"


@dataclass(frozen=True)
class SbetheOutputs:
    rows: np.ndarray
    oos_W_eV: np.ndarray
    oos_df_dW: np.ndarray
    electrons_per_molecule: float
    molar_mass_g_mol: float
    density_g_cm3: float
    mean_excitation_eV: float


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        with open(path, "wb") as f:
            shutil.copyfileobj(response, f)


def _sbethe_download_url():
    try:
        request = urllib.request.Request(SBETHE_FILE_LIST_URL, headers=HTTP_HEADERS)
        with urllib.request.urlopen(request, timeout=60) as response:
            files = json.loads(response.read().decode("utf-8"))
        for item in files:
            if item.get("filename") == "sbethe.zip":
                return item["content_details"]["download_url"]
    except Exception:
        pass
    return SBETHE_DIRECT_URL


def _download_sbethe_zip(work_dir):
    zip_path = work_dir / "sbethe.zip"
    if zip_path.exists() and _sha256(zip_path) == SBETHE_ZIP_SHA256:
        return zip_path
    _download(_sbethe_download_url(), zip_path)
    actual = _sha256(zip_path)
    if actual != SBETHE_ZIP_SHA256:
        raise RuntimeError(
            f"SBETHE ZIP SHA-256 mismatch: expected {SBETHE_ZIP_SHA256}, got {actual}."
        )
    return zip_path


def _prepare_sbethe(work_dir, keep_existing=False):
    zip_path = _download_sbethe_zip(work_dir)
    source_dir = work_dir / "sbethe"
    exe_path = source_dir / "sbethe"
    if (not keep_existing) or (not source_dir.exists()):
        if source_dir.exists():
            shutil.rmtree(source_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(work_dir)

    if not exe_path.exists():
        gfortran = shutil.which("gfortran")
        if gfortran is None:
            raise RuntimeError("gfortran is required to compile official SBETHE source.")
        subprocess.run(
            [gfortran, "-O2", "sbethe.f", "-o", "sbethe"],
            cwd=source_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    return source_dir, exe_path


def _run_sbethe_bare_proton(source_dir, exe_path):
    material_name = "water_bare_p"
    for name in ("OOS.dat", "stp.dat", "stplogb.dat", "PENstp.dat", "gnuinfo.dat", f"{material_name}.mat"):
        path = source_dir / name
        if path.exists():
            path.unlink()

    # Material: liquid water ICRU90 entry 278.  I value unchanged, water treated
    # as an insulator with a 5 eV gap.  Projectile: "other" bare proton, so
    # SBETHE does not silently apply its built-in proton effective charge.
    stdin = "\n".join(
        [
            material_name,
            "2",
            "278",
            "n",
            "y",
            "5",
            "8",
            "protonbare",
            f"{PROTON_REST_EV_SBETHE:.12e}",
            "1",
            "",
        ]
    )
    proc = subprocess.run(
        [str(exe_path)],
        cwd=source_dir,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "SBETHE run failed.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )
    return _parse_sbethe_outputs(source_dir)


def _metadata_value(lines, label):
    for line in lines:
        if label in line:
            tail = line.split(label, 1)[1]
            match = re.search(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?", tail)
            if match:
                return float(match.group(0))
    raise RuntimeError(f"Could not parse SBETHE metadata field: {label}")


def _parse_numeric_table(path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        parts = stripped.split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    if not rows:
        raise RuntimeError(f"No numeric rows found in {path}.")
    return np.asarray(rows, dtype=float)


def _parse_sbethe_outputs(source_dir):
    stplogb_path = source_dir / "stplogb.dat"
    oos_path = source_dir / "OOS.dat"
    stplogb_lines = stplogb_path.read_text(errors="replace").splitlines()
    oos_lines = oos_path.read_text(errors="replace").splitlines()
    rows = _parse_numeric_table(stplogb_path)
    oos_rows = _parse_numeric_table(oos_path)
    return SbetheOutputs(
        rows=rows,
        oos_W_eV=oos_rows[:, 0],
        oos_df_dW=oos_rows[:, 1],
        electrons_per_molecule=_metadata_value(stplogb_lines, "Electrons/molecule"),
        molar_mass_g_mol=_metadata_value(stplogb_lines, "Molecular weight"),
        density_g_cm3=_metadata_value(stplogb_lines, "Density"),
        mean_excitation_eV=_metadata_value(stplogb_lines, "Mean excitation energy"),
    )


def _log_interp_positive(x, xp, fp):
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    floor = 1.0e-90
    y = np.exp(np.interp(np.log(x), np.log(xp), np.log(np.maximum(fp, floor))))
    return np.where(y <= 10.0 * floor, 0.0, y)


def _mass_stopping_unit(s_eV_cm2, molar_mass_g_mol=H2O_MOLAR_MASS_G_MOL):
    return np.asarray(s_eV_cm2, dtype=float) * (AVOGADRO / float(molar_mass_g_mol)) * 1.0e-6


def _sbethe_s1bar_eV_cm2(T_eV, outputs):
    E = outputs.rows[:, 0]
    cbar = outputs.rows[:, 7]
    if T_eV < E[0] or T_eV > E[-1]:
        return float("nan")
    cbar_i = float(np.interp(np.log(T_eV), np.log(E), cbar))
    beta, _ = barkas_dcs.projectile_beta_gamma(T_eV, PROTON_MASS_ME_SBETHE)
    beta2 = float(beta * beta)
    cons = 2.0 * math.pi * (SBETHE_HREV_EV * SBETHE_A0B_CM) ** 2 / (SBETHE_REV_EV * beta2)
    sfact = 2.0 * cons * outputs.electrons_per_molecule
    return cbar_i * sfact


def _integrate_barkas_on_oos(T_eV, W_eV, df_dW, z_int=1.0, n_dense=100000):
    W_eV = np.asarray(W_eV, dtype=float)
    df_dW = np.asarray(df_dW, dtype=float)
    mass_me = PROTON_MASS_ME_SBETHE
    wmax = float(barkas_dcs.wmax_eV(T_eV, mass_me))
    w_min = max(float(np.nanmin(W_eV[W_eV > 0.0])), 1.0e-6)
    if wmax <= w_min:
        return 0.0, wmax
    dense = np.geomspace(w_min, wmax, int(n_dense))
    knots = W_eV[(W_eV > w_min) & (W_eV < wmax)]
    grid = np.unique(np.concatenate([dense, knots, np.asarray([wmax])]))
    df = _log_interp_positive(grid, W_eV, df_dW)
    unit_cm2 = barkas_dcs.barkas_unit_dcs_cm2_per_eV(
        np.full(grid.size, T_eV),
        grid,
        projectile_mass_me=mass_me,
        df_dW=df,
    )
    dcs_cm2 = float(z_int) ** 3 * unit_cm2
    return float(np.trapezoid(grid * dcs_cm2, grid)), wmax


def _integrate_barkas_production_oos(T_eV, material="amorphous", z_int=1.0, n_dense=100000):
    mass_me = PROTON_MASS_ME_SBETHE
    wmax = float(barkas_dcs.wmax_eV(T_eV, mass_me))
    if wmax <= 1.0e-6:
        return 0.0, wmax
    grid = np.geomspace(1.0e-3, wmax, int(n_dense))
    extra = [wmax]
    if 1.0e-3 < model.OXYGEN_K_B_EV < wmax:
        extra.append(model.OXYGEN_K_B_EV)
    grid = np.unique(np.concatenate([grid, np.asarray(extra)]))
    s = model.epsilon_optical(material)
    oos = barkas_dcs.oos_density(grid, s, material=material, include_kshell=True)
    unit_cm2 = barkas_dcs.barkas_unit_dcs_cm2_per_eV(
        np.full(grid.size, T_eV),
        grid,
        projectile_mass_me=mass_me,
        df_dW=oos.df_dW_total,
    )
    dcs_cm2 = float(z_int) ** 3 * unit_cm2
    return float(np.trapezoid(grid * dcs_cm2, grid)), wmax


def _charge_scaling_check(outputs):
    T_eV = 10.0e6
    W0 = 100.0
    df0 = _log_interp_positive(
        np.asarray([W0], dtype=float),
        outputs.oos_W_eV,
        outputs.oos_df_dW,
    )[0]
    z = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
    unit = barkas_dcs.barkas_unit_dcs_cm2_per_eV(
        np.full(z.size, T_eV),
        np.full(z.size, W0),
        projectile_mass_me=PROTON_MASS_ME_SBETHE,
        df_dW=np.full(z.size, df0),
    )
    barkas_ratio = (z**3 * unit) / unit[0]
    born_ratio = z**2
    return {
        "T_eV": T_eV,
        "W_eV": W0,
        "z": z.tolist(),
        "born_ratio": born_ratio.tolist(),
        "barkas_ratio": barkas_ratio.tolist(),
        "born_max_abs_error_vs_z2": float(np.max(np.abs(born_ratio - z**2))),
        "barkas_max_rel_error_vs_z3": float(np.max(np.abs(barkas_ratio / (z**3) - 1.0))),
    }


def _make_rows(energies_MeV, outputs, material):
    rows = []
    for T_MeV in energies_MeV:
        T_eV = float(T_MeV) * 1.0e6
        beta, gamma = barkas_dcs.projectile_beta_gamma(T_eV, PROTON_MASS_ME_SBETHE)
        beta = float(beta)
        gamma = float(gamma)
        ours_sbethe_oos, wmax = _integrate_barkas_on_oos(
            T_eV, outputs.oos_W_eV, outputs.oos_df_dW
        )
        ours_prod_oos, _ = _integrate_barkas_production_oos(T_eV, material=material)
        sbethe = _sbethe_s1bar_eV_cm2(T_eV, outputs)
        rel_formula = (ours_sbethe_oos - sbethe) / sbethe if np.isfinite(sbethe) else float("nan")
        rel_prod = (ours_prod_oos - sbethe) / sbethe if np.isfinite(sbethe) else float("nan")
        status = "valid" if np.isfinite(sbethe) else "outside_sbethe_corrected_bethe_grid"
        rows.append(
            {
                "projectile": "proton_bare",
                "T_MeV": T_MeV,
                "T_eV": T_eV,
                "beta": beta,
                "gamma": gamma,
                "Wmax_eV": wmax,
                "S_Barkas_ours_sbethe_oos_eV_cm2": ours_sbethe_oos,
                "S_Barkas_ours_sbethe_oos_MeV_cm2_g": float(_mass_stopping_unit(ours_sbethe_oos)),
                "S_Barkas_ours_geant4_oos_eV_cm2": ours_prod_oos,
                "S_Barkas_ours_geant4_oos_MeV_cm2_g": float(_mass_stopping_unit(ours_prod_oos)),
                "S_Barkas_SBETHE_eV_cm2": sbethe,
                "S_Barkas_SBETHE_MeV_cm2_g": float(_mass_stopping_unit(sbethe)) if np.isfinite(sbethe) else float("nan"),
                "relative_difference_formula": rel_formula,
                "relative_difference_geant4_oos": rel_prod,
                "reference_status": status,
            }
        )
    return rows


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, rows):
    T = np.asarray([r["T_MeV"] for r in rows], dtype=float)
    ours_formula = np.asarray([r["S_Barkas_ours_sbethe_oos_MeV_cm2_g"] for r in rows], dtype=float)
    ours_prod = np.asarray([r["S_Barkas_ours_geant4_oos_MeV_cm2_g"] for r in rows], dtype=float)
    ref = np.asarray([r["S_Barkas_SBETHE_MeV_cm2_g"] for r in rows], dtype=float)
    rel_formula = np.asarray([r["relative_difference_formula"] for r in rows], dtype=float)
    rel_prod = np.asarray([r["relative_difference_geant4_oos"] for r in rows], dtype=float)
    valid = np.isfinite(ref)

    plt.rcParams.update(
        rcparams_with_fontsize(
            RC_BASE_ELASTIC,
            14,
            {
                "font.family": FONT_COURIER,
                "figure.dpi": 120,
                "savefig.dpi": 300,
            },
        )
    )
    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(8.3, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2]},
    )
    ax.loglog(T[valid], ref[valid], "k-", lw=2.4, label="SBETHE Barkas")
    ax.loglog(T, ours_formula, color="#c0392b", ls="--", marker="s", lw=2.0, label="Our kernel + SBETHE OOS")
    ax.loglog(T, ours_prod, color="#2c7fb8", ls=":", marker="^", lw=2.1, label="Our kernel + ice OOS")
    ax.set_ylabel(r"$S_\mathrm{Barkas}$ (MeV cm$^2$/g)")
    ax.set_title("Barkas stopping-moment check, bare proton in H2O")
    ax.legend(loc="best", frameon=False)

    rax.axhline(0.0, color="black", lw=1.0)
    rax.semilogx(T[valid], rel_formula[valid], color="#c0392b", marker="s", lw=2.0, label="kernel")
    rax.semilogx(T[valid], rel_prod[valid], color="#2c7fb8", marker="^", lw=2.0, label="ice OOS")
    rax.set_xlabel("Proton kinetic energy (MeV)")
    rax.set_ylabel("fractional diff.")
    rax.legend(loc="best", frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _pass_fail(rows, tolerance):
    valid = [r for r in rows if np.isfinite(r["relative_difference_formula"])]
    if not valid:
        return "FAIL", float("nan"), "No energies inside the SBETHE corrected-Bethe grid."
    max_abs = max(abs(r["relative_difference_formula"]) for r in valid)
    if max_abs <= tolerance:
        return "PASS", max_abs, f"formula-parity max relative difference {max_abs:.3e} <= {tolerance:.3e}"
    return "FAIL", max_abs, f"formula-parity max relative difference {max_abs:.3e} > {tolerance:.3e}"


def _print_table(rows):
    header = (
        "projectile,T_MeV,beta,gamma,Wmax_eV,"
        "S_ours_SBETHE_OOS_MeV_cm2_g,S_SBETHE_MeV_cm2_g,rel_diff_formula,"
        "S_ours_Geant4_OOS_MeV_cm2_g,rel_diff_geant4_oos,status"
    )
    print(header)
    for r in rows:
        print(
            f"{r['projectile']},{r['T_MeV']:.6g},{r['beta']:.8g},{r['gamma']:.10g},"
            f"{r['Wmax_eV']:.8g},{r['S_Barkas_ours_sbethe_oos_MeV_cm2_g']:.8e},"
            f"{r['S_Barkas_SBETHE_MeV_cm2_g']:.8e},"
            f"{r['relative_difference_formula']:.8e},"
            f"{r['S_Barkas_ours_geant4_oos_MeV_cm2_g']:.8e},"
            f"{r['relative_difference_geant4_oos']:.8e},{r['reference_status']}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--material", choices=("amorphous", "hexagonal"), default="amorphous")
    parser.add_argument("--energies-MeV", nargs="+", type=float, default=DEFAULT_ENERGIES_MEV)
    parser.add_argument("--formula-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--keep-sbethe-work", action="store_true")
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    work_dir = args.work_dir if args.work_dir is not None else HERE / "runs" / "sbethe_v2"
    source_dir, exe_path = _prepare_sbethe(work_dir, keep_existing=args.keep_sbethe_work)
    outputs = _run_sbethe_bare_proton(source_dir, exe_path)
    rows = _make_rows(args.energies_MeV, outputs, args.material)
    charge_scaling = _charge_scaling_check(outputs)
    status, max_abs, reason = _pass_fail(rows, args.formula_tolerance)

    csv_path = out_dir / "barkas_sbethe_benchmark_proton.csv"
    plot_path = out_dir / "barkas_sbethe_benchmark_proton.pdf"
    json_path = out_dir / "barkas_sbethe_benchmark_proton_summary.json"
    _write_csv(csv_path, rows)
    _plot(plot_path, rows)
    summary = {
        "status": status,
        "reason": reason,
        "formula_tolerance": args.formula_tolerance,
        "max_abs_relative_difference_formula": max_abs,
        "sbethe_dataset": {
            "doi": "10.17632/7zw25f428t.2",
            "zip_sha256": SBETHE_ZIP_SHA256,
            "file_url": SBETHE_DIRECT_URL,
        },
        "sbethe_water": {
            "material_id": 278,
            "description": "WATER, LIQUID -- ICRU90",
            "electrons_per_molecule": outputs.electrons_per_molecule,
            "molar_mass_g_mol": outputs.molar_mass_g_mol,
            "density_g_cm3": outputs.density_g_cm3,
            "mean_excitation_eV": outputs.mean_excitation_eV,
        },
        "unit_conversion": (
            "S_MeV_cm2_g = S_eV_cm2 * (N_A / M_H2O) * 1e-6, "
            "with S_eV_cm2 = integral W * dSigma_B/dW dW per H2O molecule."
        ),
        "charge_scaling_check": charge_scaling,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    _print_table(rows)
    print(f"PASS_FAIL,{status},{reason}")
    print(f"CSV,{csv_path}")
    print(f"PLOT,{plot_path}")
    print(f"SUMMARY,{json_path}")


if __name__ == "__main__":
    main()
