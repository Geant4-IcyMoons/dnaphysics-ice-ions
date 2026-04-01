#!/usr/bin/env python3
"""
Build a global monoenergetic simulation library for Europa and per-cell scaling tables.

Workflow:
1) Split Europa into an N_lat x N_lon grid (default 90 x 90).
2) For each cell, derive the allowed local energy range from the leading/trailing maps.
3) Build ONE global logarithmic energy grid (unique energies) from global E_min to E_max,
   with bin count auto-selected from the Riemann/integral tolerance.
4) Write Geant4 macros for each global energy independently (wvalue-style) so each
   energy can be run into its own ROOT output file.
5) Write per-cell tables that map local allowed energies to global energy-bin indices and
   provide scaling factors derived directly from electron_spectrum_fit(E)*dE.

Notes:
- This script does not run Geant4; it prepares inputs and scaling metadata.
- Use DNA_PHYSICS=ice_hex or DNA_PHYSICS=ice_am at runtime unless you override
  density manually via --manual-density-gcm3.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
import sys
import os


import numpy as np

from generate_europa_electron_bins import (
    _allowed_energy_range_from_value,
    _energy_bins,
    _infer_hemisphere,
    _integrate_spectrum,
    _interpolate_map_value,
    _load_map,
    _normalize_lon,
    electron_spectrum_fit,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from physics_ice.constants import TOP_ROOT


def _lat_lon_centers(n_lat: int, n_lon: int) -> tuple[np.ndarray, np.ndarray]:
    lat_edges = np.linspace(90.0, -90.0, n_lat + 1)
    lon_edges = np.linspace(0.0, 360.0, n_lon + 1)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lon_centers = np.mod(lon_centers, 360.0)
    return lat_centers, lon_centers


def _resolve_cell_range(
    lat_deg: float,
    lon_w_deg: float,
    leading_map: np.ndarray,
    trailing_map: np.ndarray,
    leading_cap: float,
    trailing_min: float,
) -> tuple[str, float, float]:
    hemisphere = _infer_hemisphere(lon_w_deg)
    if hemisphere == "leading":
        value = _interpolate_map_value(leading_map, lat_deg, lon_w_deg, hemisphere)
    else:
        value = _interpolate_map_value(trailing_map, lat_deg, lon_w_deg, hemisphere)
    e_min, e_max = _allowed_energy_range_from_value(
        hemisphere, value, leading_cap, trailing_min
    )
    return hemisphere, float(e_min), float(e_max)


def _find_log_nbins(
    e_min: float,
    e_max: float,
    initial_bins: int,
    tol: float,
    max_iter: int,
    max_bins: int,
    cell_e_min: np.ndarray | None = None,
    cell_e_max: np.ndarray | None = None,
) -> tuple[int, float, float, float, int, float]:
    integral_sum = _integrate_spectrum(e_min, e_max)
    if integral_sum <= 0.0:
        return 0, 0.0, 0.0, 0.0, -1, float("nan")

    use_cell_criterion = cell_e_min is not None or cell_e_max is not None
    if use_cell_criterion:
        if cell_e_min is None or cell_e_max is None:
            raise ValueError("Both cell_e_min and cell_e_max are required for per-cell tolerance.")
        if cell_e_min.shape != cell_e_max.shape:
            raise ValueError("cell_e_min and cell_e_max must have the same shape.")
        if cell_e_min.size == 0:
            raise ValueError("At least one valid cell range is required for per-cell tolerance.")
        if np.any(cell_e_max <= cell_e_min):
            raise ValueError("All per-cell ranges must satisfy E_max > E_min.")

        lookup_grid, lookup_cumulative = _build_spectrum_integral_lookup(e_min, e_max)
        cell_integrals = _integrals_from_lookup(
            cell_e_min, cell_e_max, lookup_grid, lookup_cumulative
        )
        if np.any(cell_integrals <= 0.0):
            raise ValueError("Per-cell spectrum integrals must be positive.")
    else:
        cell_integrals = None

    nbins = max(1, int(initial_bins))
    ratio = 0.0
    max_cell_abs_err = 0.0
    worst_cell_index = -1
    worst_cell_ratio = float("nan")
    for _ in range(max_iter):
        if nbins > max_bins:
            raise ValueError(
                f"Auto log-bin search exceeded max bins ({max_bins}); "
                f"last nbins={nbins}."
            )
        edges, centers = _energy_bins(e_min, e_max, nbins=nbins, delta_e=None)
        widths = edges[1:] - edges[:-1]
        spectrum_vals = electron_spectrum_fit(centers)
        discrete_sum = float(np.sum(spectrum_vals * widths))
        ratio = discrete_sum / integral_sum
        if cell_integrals is None:
            if abs(ratio - 1.0) <= tol:
                return nbins, ratio, integral_sum, 0.0, -1, float("nan")
        else:
            max_cell_abs_err, worst_cell_index, worst_cell_ratio = _max_cell_ratio_error(
                edges,
                centers,
                cell_e_min,
                cell_e_max,
                cell_integrals,
            )
            if max_cell_abs_err <= tol:
                return (
                    nbins,
                    ratio,
                    integral_sum,
                    max_cell_abs_err,
                    worst_cell_index,
                    worst_cell_ratio,
                )
        nbins *= 2

    if cell_integrals is None:
        raise ValueError(
            f"Auto log-bin search failed to reach tol={tol:.2e} after {max_iter} iterations."
        )

    raise ValueError(
        f"Auto log-bin search failed to reach per-cell tol={tol:.2e} after {max_iter} iterations. "
        f"Last max cell |ratio-1|={max_cell_abs_err:.3e} (index={worst_cell_index})."
    )


def _build_spectrum_integral_lookup(
    e_min: float,
    e_max: float,
    samples: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    if e_min <= 0.0 or e_max <= e_min:
        raise ValueError(f"Invalid lookup range: E_min={e_min}, E_max={e_max}.")
    n = max(2_000, int(samples))
    grid = np.logspace(np.log10(e_min), np.log10(e_max), n)
    vals = electron_spectrum_fit(grid)
    cumulative = np.zeros_like(grid)
    cumulative[1:] = np.cumsum(0.5 * (vals[1:] + vals[:-1]) * (grid[1:] - grid[:-1]))
    return grid, cumulative


def _integrals_from_lookup(
    e_low: np.ndarray,
    e_high: np.ndarray,
    lookup_grid: np.ndarray,
    lookup_cumulative: np.ndarray,
) -> np.ndarray:
    low = np.clip(np.asarray(e_low, dtype=float), lookup_grid[0], lookup_grid[-1])
    high = np.clip(np.asarray(e_high, dtype=float), lookup_grid[0], lookup_grid[-1])
    hi_int = np.interp(high, lookup_grid, lookup_cumulative)
    lo_int = np.interp(low, lookup_grid, lookup_cumulative)
    out = hi_int - lo_int
    out = np.where(high > low, out, 0.0)
    return np.asarray(out, dtype=float)


def _max_cell_ratio_error(
    edges: np.ndarray,
    centers: np.ndarray,
    cell_e_min: np.ndarray,
    cell_e_max: np.ndarray,
    cell_integrals: np.ndarray,
    chunk_size: int = 256,
) -> tuple[float, int, float]:
    low_edges = edges[:-1]
    high_edges = edges[1:]
    spectrum_vals = electron_spectrum_fit(centers)

    max_abs_err = -1.0
    worst_idx = -1
    worst_ratio = float("nan")
    n_cells = int(cell_e_min.size)

    for start in range(0, n_cells, chunk_size):
        stop = min(start + chunk_size, n_cells)
        low = cell_e_min[start:stop, None]
        high = cell_e_max[start:stop, None]
        overlap = np.maximum(0.0, np.minimum(high_edges, high) - np.maximum(low_edges, low))
        discrete = overlap @ spectrum_vals
        ratios = discrete / cell_integrals[start:stop]
        abs_err = np.abs(ratios - 1.0)
        local = int(np.argmax(abs_err))
        local_abs_err = float(abs_err[local])
        if local_abs_err > max_abs_err:
            max_abs_err = local_abs_err
            worst_idx = start + local
            worst_ratio = float(ratios[local])

    return max_abs_err, worst_idx, worst_ratio


def _normalize_direction(x: float, y: float, z: float) -> tuple[float, float, float]:
    vec = np.asarray([x, y, z], dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("Source direction vector must be non-zero.")
    unit = vec / norm
    return float(unit[0]), float(unit[1]), float(unit[2])


def _orthonormal_axes_from_normal(
    nx: float,
    ny: float,
    nz: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    normal = np.asarray([nx, ny, nz], dtype=float)
    normal /= np.linalg.norm(normal)

    # Build two orthonormal vectors for GPS angular rotation.
    # GPS interprets the directed axis with rot2 x rot1, so we return vectors
    # that satisfy rot2 x rot1 = normal.
    ref = np.asarray([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(normal, ref))) > 0.95:
        ref = np.asarray([1.0, 0.0, 0.0], dtype=float)

    rot1 = np.cross(ref, normal)
    rot1_norm = float(np.linalg.norm(rot1))
    if rot1_norm <= 0.0:
        raise ValueError("Failed to build angular basis from source direction.")
    rot1 /= rot1_norm
    rot2 = np.cross(normal, rot1)
    rot2 /= np.linalg.norm(rot2)

    # rot1_old x rot2_old = normal. Return swapped so rot2 x rot1 = normal.
    return (
        (float(rot2[0]), float(rot2[1]), float(rot2[2])),
        (float(rot1[0]), float(rot1[1]), float(rot1[2])),
    )


def _build_macro_header(
    *,
    n_threads: int,
    x_half_mm: float,
    y_half_mm: float,
    z_thickness_mm: float,
    source_x_mm: float,
    source_y_mm: float,
    source_z_mm: float,
    source_dir_x: float,
    source_dir_y: float,
    source_dir_z: float,
    angular_dist: str,
    maxtheta: bool,
    log_mode: str,
    manual_density_gcm3: float | None,
) -> list[str]:
    lines: list[str] = []
    lines.append("/control/verbose 0")
    lines.append("/run/verbose 0")
    lines.append("/event/verbose 0")
    lines.append("/tracking/verbose 0")
    lines.append("")
    lines.append(f"/run/numberOfThreads {int(n_threads)}")
    lines.append(f"/dna/test/setLogMode {log_mode}")
    lines.append("")

    x_size = 2.0 * x_half_mm
    y_size = 2.0 * y_half_mm
    lines.append(f"/dna/test/setIceSize {x_size:.9g} {y_size:.9g} {z_thickness_mm:.9g} mm")
    if manual_density_gcm3 is None:
        # Density is controlled by DNA_PHYSICS in DetectorConstruction when
        # the material is G4_WATER (e.g., ice_hex / ice_am).
        lines.append("/dna/test/setMat G4_WATER")
    else:
        lines.append(
            f"/dna/test/setMatDens G4_WATER_ICE_USER {manual_density_gcm3:.9g} g/cm3"
        )
        lines.append("/dna/test/setMat G4_WATER_ICE_USER")
    lines.append("")
    lines.append("/run/initialize")
    lines.append("")

    lines.append("/gps/particle e-")
    lines.append("/gps/number 1")
    lines.append("/gps/pos/type Point")
    lines.append(
        f"/gps/pos/centre {source_x_mm:.9g} {source_y_mm:.9g} {source_z_mm:.9g} mm"
    )
    dir_x, dir_y, dir_z = _normalize_direction(source_dir_x, source_dir_y, source_dir_z)
    if angular_dist == "none":
        lines.append(f"/gps/direction {dir_x:.9g} {dir_y:.9g} {dir_z:.9g}")
    elif angular_dist == "cos":
        rot1, rot2 = _orthonormal_axes_from_normal(dir_x, dir_y, dir_z)
        lines.append("/gps/ang/type cos")
        lines.append("/gps/ang/mintheta 0 deg")
        lines.append("/gps/ang/minphi 0 deg")
        lines.append("/gps/ang/maxphi 360 deg")
        lines.append(f"/gps/ang/rot1 {rot1[0]:.9g} {rot1[1]:.9g} {rot1[2]:.9g}")
        lines.append(f"/gps/ang/rot2 {rot2[0]:.9g} {rot2[1]:.9g} {rot2[2]:.9g}")
        lines.append(f"/dna/test/setMaxTheta {'true' if maxtheta else 'false'}")
    else:
        raise ValueError(f"Unsupported angular distribution: {angular_dist}")
    lines.append("/gps/ene/type Mono")
    lines.append("")
    return lines


def _write_library_macro(
    out_path: Path,
    *,
    energies_mev: np.ndarray,
    n_per_energy_values: np.ndarray,
    n_threads: int,
    x_half_mm: float,
    y_half_mm: float,
    z_thickness_mm: float,
    source_x_mm: float,
    source_y_mm: float,
    source_z_mm: float,
    source_dir_x: float,
    source_dir_y: float,
    source_dir_z: float,
    angular_dist: str,
    maxtheta: bool,
    log_mode: str,
    manual_density_gcm3: float | None,
) -> None:
    if energies_mev.shape != n_per_energy_values.shape:
        raise ValueError(
            "energies_mev and n_per_energy_values must have identical shapes."
        )
    lines: list[str] = []
    lines.append("# Global Europa energy-library macro (auto-generated)")
    lines.append("# One monoenergetic run per global energy bin center.")
    lines.append("# Reweight per-cell offline using the generated scaling table.")
    lines.append(f"# Energies: {len(energies_mev)} bins")
    unique_counts = np.unique(n_per_energy_values)
    if unique_counts.size == 1:
        lines.append(f"# Particles per energy: {int(unique_counts[0])}")
    else:
        lines.append(
            f"# Particles per energy: piecewise ({int(unique_counts.min())}..{int(unique_counts.max())})"
        )
    lines.append(f"# Angular distribution: {angular_dist}")
    lines.append(f"# maxtheta={str(maxtheta).lower()}")
    lines.append("")
    lines.extend(
        _build_macro_header(
            n_threads=n_threads,
            x_half_mm=x_half_mm,
            y_half_mm=y_half_mm,
            z_thickness_mm=z_thickness_mm,
            source_x_mm=source_x_mm,
            source_y_mm=source_y_mm,
            source_z_mm=source_z_mm,
            source_dir_x=source_dir_x,
            source_dir_y=source_dir_y,
            source_dir_z=source_dir_z,
            angular_dist=angular_dist,
            maxtheta=maxtheta,
            log_mode=log_mode,
            manual_density_gcm3=manual_density_gcm3,
        )
    )

    for e_mev, n_particles in zip(energies_mev, n_per_energy_values):
        lines.append(f"/gps/ene/mono {e_mev:.9g} MeV")
        lines.append(f"/run/beamOn {int(n_particles)}")

    out_path.write_text("\n".join(lines) + "\n")


def _safe_token(value: str) -> str:
    token = value.strip()
    token = token.replace("+", "")
    token = token.replace("-", "m")
    token = token.replace(".", "p")
    token = token.replace("/", "_")
    token = token.replace(" ", "_")
    return token


def _energy_tag_mev(e_mev: float) -> str:
    return _safe_token(f"{e_mev:.9g}MeV")


def _resolve_density_gcm3(dna_physics: str, manual_density_gcm3: float | None) -> float:
    if manual_density_gcm3 is not None:
        return float(manual_density_gcm3)
    if dna_physics == "ice_am":
        return 0.94
    if dna_physics == "ice_hex":
        return 0.917
    # Geant4 water nominal
    return 1.0


def _density_tag(density_gcm3: float) -> str:
    return _safe_token(f"rho{density_gcm3:.6g}")


def _root_basename_for_energy(
    energy_index: int,
    e_mev: float,
    dna_physics: str,
    density_gcm3: float,
) -> str:
    return (
        f"dna_{_safe_token(dna_physics)}_{_density_tag(density_gcm3)}_"
        f"E{energy_index:05d}_{_energy_tag_mev(e_mev)}"
    )


def _resolve_n_per_energy_schedule(
    *,
    energies_mev: np.ndarray,
    default_n_per_energy: int,
    thresholds_mev: list[float] | None,
    values: list[int] | None,
    global_e_max: float,
) -> np.ndarray:
    if thresholds_mev is None and values is None:
        return np.full(energies_mev.shape, int(default_n_per_energy), dtype=int)

    if thresholds_mev is None or values is None:
        raise ValueError(
            "Both --n-per-energy-thresholds and --n-per-energy-values must be provided together."
        )

    if len(thresholds_mev) != len(values):
        raise ValueError(
            "Length mismatch: --n-per-energy-thresholds and --n-per-energy-values must have the same length."
        )

    if len(thresholds_mev) == 0:
        raise ValueError("At least one threshold/value pair is required.")

    thresholds_arr = np.asarray(thresholds_mev, dtype=float)
    values_arr = np.asarray(values, dtype=int)

    if np.any(~np.isfinite(thresholds_arr)) or np.any(thresholds_arr <= 0.0):
        raise ValueError("All thresholds must be finite and > 0 MeV.")
    if np.any(np.diff(thresholds_arr) <= 0.0):
        raise ValueError("Thresholds must be strictly increasing.")
    if np.any(values_arr <= 0):
        raise ValueError("All n-per-energy values must be positive integers.")

    if thresholds_arr[-1] < global_e_max:
        raise ValueError(
            f"Highest threshold ({thresholds_arr[-1]:.9g} MeV) must be >= global E_max "
            f"({global_e_max:.9g} MeV)."
        )

    # Interval convention: E in (threshold[i-1], threshold[i]] gets values[i].
    idx = np.searchsorted(thresholds_arr, energies_mev, side="left")
    if np.any(idx >= len(thresholds_arr)):
        raise ValueError(
            "Some energy bins are above the highest provided threshold; expand threshold list."
        )
    return values_arr[idx].astype(int, copy=False)


def _write_per_energy_macros(
    out_dir: Path,
    *,
    energies_mev: np.ndarray,
    n_per_energy_values: np.ndarray,
    dna_physics: str,
    density_gcm3: float,
    n_threads: int,
    x_half_mm: float,
    y_half_mm: float,
    z_thickness_mm: float,
    source_x_mm: float,
    source_y_mm: float,
    source_z_mm: float,
    source_dir_x: float,
    source_dir_y: float,
    source_dir_z: float,
    angular_dist: str,
    maxtheta: bool,
    log_mode: str,
    manual_density_gcm3: float | None,
) -> list[dict[str, str]]:
    if energies_mev.shape != n_per_energy_values.shape:
        raise ValueError(
            "energies_mev and n_per_energy_values must have identical shapes."
        )
    macro_dir = out_dir / "macros"
    macro_dir.mkdir(parents=True, exist_ok=True)
    # Avoid stale per-energy macros from previous generations.
    for old_macro in macro_dir.glob("europa_E*.mac"):
        old_macro.unlink()

    common_lines = _build_macro_header(
        n_threads=n_threads,
        x_half_mm=x_half_mm,
        y_half_mm=y_half_mm,
        z_thickness_mm=z_thickness_mm,
        source_x_mm=source_x_mm,
        source_y_mm=source_y_mm,
        source_z_mm=source_z_mm,
        source_dir_x=source_dir_x,
        source_dir_y=source_dir_y,
        source_dir_z=source_dir_z,
        angular_dist=angular_dist,
        maxtheta=maxtheta,
        log_mode=log_mode,
        manual_density_gcm3=manual_density_gcm3,
    )

    run_rows: list[dict[str, str]] = []
    for idx, (e_mev, n_particles) in enumerate(zip(energies_mev, n_per_energy_values)):
        tag = _energy_tag_mev(float(e_mev))
        macro_name = f"europa_E{idx:05d}_{tag}.mac"
        macro_path = macro_dir / macro_name

        lines: list[str] = []
        lines.append("# Europa per-energy macro (auto-generated)")
        lines.append(f"# energy_index={idx}")
        lines.append(f"# energy_center={float(e_mev):.9g} MeV")
        lines.append(f"# sim_particles={int(n_particles)}")
        lines.append(f"# dna_physics={dna_physics}")
        lines.append(f"# density_gcm3={density_gcm3:.9g}")
        lines.append(f"# angular_dist={angular_dist}")
        lines.append(f"# maxtheta={str(maxtheta).lower()}")
        lines.append("")
        lines.extend(common_lines)
        lines.append(f"/gps/ene/mono {float(e_mev):.9g} MeV")
        lines.append(f"/run/beamOn {int(n_particles)}")
        macro_path.write_text("\n".join(lines) + "\n")

        root_basename = _root_basename_for_energy(
            idx, float(e_mev), dna_physics, density_gcm3
        )
        run_rows.append(
            {
                "energy_index": str(idx),
                "E_center_MeV": f"{float(e_mev):.9g}",
                "macro_relpath": f"macros/{macro_name}",
                "root_basename": root_basename,
                "root_relpath": f"root/{root_basename}.root",
                "sim_particles": str(int(n_particles)),
                "dna_physics": dna_physics,
                "density_gcm3": f"{density_gcm3:.9g}",
            }
        )

    return run_rows


def _write_runner_script(out_dir: Path, default_threads: int, default_physics: str) -> Path:
    script_path = out_dir / "run_per_energy.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'RUN_TABLE="${SCRIPT_DIR}/per_energy_runs.csv"',
        'ROOT_DIR="${SCRIPT_DIR}/root"',
        'CACHE_FILE="${SCRIPT_DIR}/run_per_energy.cache.csv"',
        'mkdir -p "${ROOT_DIR}"',
        "",
        'DNAPHYSICS_BIN_INPUT="${1:-${SCRIPT_DIR}/../build/dnaphysics}"',
        f'THREADS="${{2:-{int(default_threads)}}}"',
        f'PHYSICS="${{3:-${{DNA_PHYSICS:-{default_physics}}}}}"',
        "",
        'if [[ "${DNAPHYSICS_BIN_INPUT}" = /* ]]; then',
        '  DNAPHYSICS_BIN="${DNAPHYSICS_BIN_INPUT}"',
        "else",
        '  DNAPHYSICS_BIN="${PWD}/${DNAPHYSICS_BIN_INPUT}"',
        "fi",
        "",
        'if [[ ! -f "${RUN_TABLE}" ]]; then',
        '  echo "Missing run table: ${RUN_TABLE}" >&2',
        "  exit 1",
        "fi",
        "",
        'if [[ ! -x "${DNAPHYSICS_BIN}" ]]; then',
        '  echo "dnaphysics binary not executable: ${DNAPHYSICS_BIN}" >&2',
        "  exit 1",
        "fi",
        "",
        'if [[ ! -f "${CACHE_FILE}" ]]; then',
        '  echo "root_basename,energy_index,e_center_mev,sim_particles,threads,physics,utc_timestamp" > "${CACHE_FILE}"',
        "fi",
        "",
        "have_outputs() {",
        '  local base="$1"',
        '  compgen -G "${ROOT_DIR}/${base}*.root" > /dev/null',
        "}",
        "",
        "is_cached() {",
        '  local base="$1"',
        '  awk -F, -v b="${base}" \'NR>1 && $1==b {found=1; exit} END {exit !found}\' "${CACHE_FILE}"',
        "}",
        "",
        "mark_cached() {",
        '  local base="$1"',
        '  local idx="$2"',
        '  local ecenter="$3"',
        '  local np="$4"',
        '  local stamp',
        '  stamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"',
        '  printf "%s,%s,%s,%s,%s,%s,%s\\n" "${base}" "${idx}" "${ecenter}" "${np}" "${THREADS}" "${PHYSICS}" "${stamp}" >> "${CACHE_FILE}"',
        "}",
        "",
        'echo "Running per-energy Europa library with DNA_PHYSICS=${PHYSICS}, threads=${THREADS}"',
        'echo "Cache file: ${CACHE_FILE}"',
        "",
        'tail -n +2 "${RUN_TABLE}" | while IFS=, read -r energy_index e_low_mev e_high_mev e_center_mev dE_mev sim_particles macro_relpath root_basename root_relpath csv_physics csv_density; do',
        '  macro_path="${SCRIPT_DIR}/${macro_relpath}"',
        '  if [[ ! -f "${macro_path}" ]]; then',
        '    echo "Missing macro: ${macro_path}" >&2',
        "    exit 1",
        "  fi",
        '  if have_outputs "${root_basename}"; then',
        '    if ! is_cached "${root_basename}"; then',
        '      mark_cached "${root_basename}" "${energy_index}" "${e_center_mev}" "${sim_particles}"',
        "    fi",
        '    echo "[energy ${energy_index}] SKIP cached/output ${root_basename}*.root"',
        "    continue",
        "  fi",
        '  if is_cached "${root_basename}"; then',
        '    echo "[energy ${energy_index}] cache entry exists but outputs are missing; rerunning ${root_basename}"',
        "  fi",
        '  echo "[energy ${energy_index}] E=${e_center_mev} MeV -> ${root_basename}*.root"',
        '  (',
        '    cd "${ROOT_DIR}"',
        '    DNA_PHYSICS="${PHYSICS}" DNA_NTUPLE_FILES=0 DNA_ROOT_BASENAME="${root_basename}" "${DNAPHYSICS_BIN}" "${macro_path}" "${THREADS}"',
        "  )",
        '  if have_outputs "${root_basename}"; then',
        '    if ! is_cached "${root_basename}"; then',
        '      mark_cached "${root_basename}" "${energy_index}" "${e_center_mev}" "${sim_particles}"',
        "    fi",
        "  else",
        '    echo "[energy ${energy_index}] WARNING: run finished but no ${root_basename}*.root was found." >&2',
        "  fi",
        "done",
    ]
    script_path.write_text("\n".join(lines) + "\n")
    current_mode = script_path.stat().st_mode
    script_path.chmod(current_mode | 0o111)
    return script_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate one global monoenergetic macro + per-cell (lat/lon) "
            "energy assignment/scaling tables for Europa."
        )
    )
    ap.add_argument("--n-lat", type=int, default=90, help="Number of latitude bins.")
    ap.add_argument("--n-lon", type=int, default=90, help="Number of longitude bins.")
    ap.add_argument(
        "--tol",
        type=float,
        default=1.0e-4,
        help=(
            "Tolerance for auto log-bin search, enforced on each valid lat/lon "
            "cell range using the Riemann/integral ratio."
        ),
    )
    ap.add_argument(
        "--initial-bins",
        type=int,
        default=100,
        help="Initial bin count for auto log-bin search.",
    )
    ap.add_argument(
        "--global-e-max",
        type=float,
        default=100.0,
        help="Global maximum energy in MeV.",
    )
    ap.add_argument(
        "--global-e-min",
        type=float,
        default=None,
        help="Optional forced global minimum energy in MeV. Default: min valid cell E_min.",
    )
    ap.add_argument(
        "--leading-cap",
        type=float,
        default=100.0,
        help="Leading-hemisphere upper energy cap in MeV.",
    )
    ap.add_argument(
        "--trailing-min",
        type=float,
        default=0.01,
        help="Fallback trailing minimum energy in MeV when map has no positive entries.",
    )
    ap.add_argument(
        "--n-per-energy",
        type=int,
        default=100,
        help="Simulated primary electrons per global energy bin.",
    )
    ap.add_argument(
        "--n-per-energy-thresholds",
        "--n_per_energy_thresholds",
        dest="n_per_energy_thresholds",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional piecewise upper thresholds in MeV for per-bin sim_particles. "
            "Example: 0.1 1 10 100."
        ),
    )
    ap.add_argument(
        "--n-per-energy-values",
        "--n_per_energy_values",
        dest="n_per_energy_values",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional piecewise sim_particles values matching thresholds. "
            "Example: 10000 1000 100 10."
        ),
    )
    ap.add_argument(
        "--dna-physics",
        choices=("water", "ice_hex", "ice_am"),
        default="ice_hex",
        help="Default DNA_PHYSICS to use in the generated runner script.",
    )
    ap.add_argument("--threads", type=int, default=12, help="Macro thread count.")
    ap.add_argument(
        "--x-half-mm",
        type=float,
        default=500.0,
        help="Ice half-width in X (mm). Ice is centered at x=0.",
    )
    ap.add_argument(
        "--y-half-mm",
        type=float,
        default=500.0,
        help="Ice half-width in Y (mm). Ice is centered at y=0.",
    )
    ap.add_argument(
        "--z-thickness-mm",
        type=float,
        default=1000.0,
        help="Ice thickness in Z (mm). Slab spans z=[0, z_thickness].",
    )
    ap.add_argument(
        "--source-x-mm",
        type=float,
        default=0.0,
        help="Source x position in mm.",
    )
    ap.add_argument(
        "--source-y-mm",
        type=float,
        default=0.0,
        help="Source y position in mm.",
    )
    ap.add_argument(
        "--source-z-mm",
        type=float,
        default=-0.01,
        help="Source z position in mm. Default is just above slab start at z=0.",
    )
    ap.add_argument("--source-dir-x", type=float, default=0.0, help="Source direction x.")
    ap.add_argument("--source-dir-y", type=float, default=0.0, help="Source direction y.")
    ap.add_argument("--source-dir-z", type=float, default=1.0, help="Source direction z.")
    ap.add_argument(
        "--angular-dist",
        "--angular_dist",
        dest="angular_dist",
        choices=("none", "cos"),
        default="none",
        help=(
            "Primary angular mode: 'none' keeps a fixed /gps/direction; "
            "'cos' uses /gps/ang/type cos with theta in [0,90] deg and phi in [0,360] deg "
            "around the source direction axis."
        ),
    )
    ap.add_argument(
        "--manual-density-gcm3",
        type=float,
        default=None,
        help=(
            "If set, macro uses /dna/test/setMatDens with this density (g/cm3). "
            "If omitted, use DNA_PHYSICS-driven default density."
        ),
    )
    ap.add_argument(
        "--log-mode",
        choices=("minimal", "full"),
        default="minimal",
        help="Simulation logging mode for the generated macro.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "europa_energy_library",
        help="Output directory.",
    )
    ap.add_argument(
        "--macro-name",
        default="europa_energy_library.mac",
        help="Generated Geant4 macro filename.",
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    leading_map = _load_map(TOP_ROOT / "dnaphysics-ice/python_scripts/europa/e_bombardment_leading")
    trailing_map = _load_map(TOP_ROOT / "dnaphysics-ice/python_scripts/europa/e_bombardment_trailing")

    if leading_map.ndim != 2:
        raise ValueError(f"Expected 2D leading map, got shape {leading_map.shape}")
    if trailing_map.ndim != 3:
        raise ValueError(f"Expected 3D trailing map, got shape {trailing_map.shape}")

    lat_centers, lon_centers = _lat_lon_centers(args.n_lat, args.n_lon)

    # Build per-cell allowed ranges.
    cell_rows: list[dict[str, float | int | str]] = []
    valid_mins: list[float] = []
    cell_id = 0
    for lat in lat_centers:
        for lon in lon_centers:
            lon_w = _normalize_lon(float(lon))
            hemisphere, e_min, e_max = _resolve_cell_range(
                float(lat),
                lon_w,
                leading_map,
                trailing_map,
                float(args.leading_cap),
                float(args.trailing_min),
            )
            # Clamp to requested global ceiling.
            e_max = min(e_max, float(args.global_e_max))
            is_valid = int(e_min > 0.0 and e_max > e_min)
            if is_valid:
                valid_mins.append(e_min)
            cell_rows.append(
                {
                    "cell_id": cell_id,
                    "lat_deg": float(lat),
                    "lon_w_deg": lon_w,
                    "hemisphere": hemisphere,
                    "e_min_mev": float(e_min),
                    "e_max_mev": float(e_max),
                    "is_valid": is_valid,
                }
            )
            cell_id += 1

    if not valid_mins:
        raise ValueError("No valid lat/lon cells with positive energy range.")

    global_e_min = float(args.global_e_min) if args.global_e_min is not None else float(min(valid_mins))
    global_e_max = float(args.global_e_max)
    if global_e_min <= 0.0 or global_e_max <= global_e_min:
        raise ValueError(
            f"Invalid global energy range: E_min={global_e_min}, E_max={global_e_max}."
        )

    valid_cell_ids: list[int] = []
    tol_e_min: list[float] = []
    tol_e_max: list[float] = []
    for row in cell_rows:
        if int(row["is_valid"]) != 1:
            continue
        e_min_local = max(float(row["e_min_mev"]), global_e_min)
        e_max_local = min(float(row["e_max_mev"]), global_e_max)
        if e_max_local > e_min_local:
            valid_cell_ids.append(int(row["cell_id"]))
            tol_e_min.append(e_min_local)
            tol_e_max.append(e_max_local)

    if not valid_cell_ids:
        raise ValueError("No valid cell ranges remain within the requested global energy bounds.")

    tol_e_min_arr = np.asarray(tol_e_min, dtype=float)
    tol_e_max_arr = np.asarray(tol_e_max, dtype=float)

    (
        nbins,
        ratio,
        integral_sum,
        max_cell_abs_err,
        worst_cell_rel_index,
        worst_cell_ratio,
    ) = _find_log_nbins(
        global_e_min,
        global_e_max,
        initial_bins=int(args.initial_bins),
        tol=float(args.tol),
        max_iter=30,
        max_bins=1_000_000,
        cell_e_min=tol_e_min_arr,
        cell_e_max=tol_e_max_arr,
    )
    if nbins <= 0:
        raise ValueError("Auto log-bin search did not produce a valid bin count.")

    worst_cell_id = -1
    worst_cell_e_min = float("nan")
    worst_cell_e_max = float("nan")
    if worst_cell_rel_index >= 0:
        worst_cell_id = int(valid_cell_ids[worst_cell_rel_index])
        worst_cell_e_min = float(tol_e_min_arr[worst_cell_rel_index])
        worst_cell_e_max = float(tol_e_max_arr[worst_cell_rel_index])

    edges_mev, centers_mev = _energy_bins(
        global_e_min, global_e_max, nbins=nbins, delta_e=None
    )
    widths_mev = edges_mev[1:] - edges_mev[:-1]
    if centers_mev.size == 0:
        raise ValueError("No global energy bins were generated.")
    n_per_energy_values = _resolve_n_per_energy_schedule(
        energies_mev=centers_mev,
        default_n_per_energy=int(args.n_per_energy),
        thresholds_mev=args.n_per_energy_thresholds,
        values=args.n_per_energy_values,
        global_e_max=global_e_max,
    )
    density_gcm3 = _resolve_density_gcm3(
        str(args.dna_physics),
        args.manual_density_gcm3,
    )

    # Macro for global energy library run (optional convenience).
    combined_macro_path = out_dir / args.macro_name
    _write_library_macro(
        combined_macro_path,
        energies_mev=centers_mev,
        n_per_energy_values=n_per_energy_values,
        n_threads=int(args.threads),
        x_half_mm=float(args.x_half_mm),
        y_half_mm=float(args.y_half_mm),
        z_thickness_mm=float(args.z_thickness_mm),
        source_x_mm=float(args.source_x_mm),
        source_y_mm=float(args.source_y_mm),
        source_z_mm=float(args.source_z_mm),
        source_dir_x=float(args.source_dir_x),
        source_dir_y=float(args.source_dir_y),
        source_dir_z=float(args.source_dir_z),
        angular_dist=str(args.angular_dist),
        maxtheta=True,
        log_mode=str(args.log_mode),
        manual_density_gcm3=args.manual_density_gcm3,
    )

    # Per-energy macros + runner (one ROOT output per energy).
    run_rows = _write_per_energy_macros(
        out_dir,
        energies_mev=centers_mev,
        n_per_energy_values=n_per_energy_values,
        dna_physics=str(args.dna_physics),
        density_gcm3=density_gcm3,
        n_threads=int(args.threads),
        x_half_mm=float(args.x_half_mm),
        y_half_mm=float(args.y_half_mm),
        z_thickness_mm=float(args.z_thickness_mm),
        source_x_mm=float(args.source_x_mm),
        source_y_mm=float(args.source_y_mm),
        source_z_mm=float(args.source_z_mm),
        source_dir_x=float(args.source_dir_x),
        source_dir_y=float(args.source_dir_y),
        source_dir_z=float(args.source_dir_z),
        angular_dist=str(args.angular_dist),
        maxtheta=True,
        log_mode=str(args.log_mode),
        manual_density_gcm3=args.manual_density_gcm3,
    )
    run_rows_by_index = {int(row["energy_index"]): row for row in run_rows}

    # Save global energy bin table.
    energy_csv = out_dir / "global_energy_bins.csv"
    with energy_csv.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "energy_index",
                "E_low_MeV",
                "E_high_MeV",
                "E_center_MeV",
                "dE_MeV",
                "sim_particles",
                "macro_relpath",
                "root_basename",
                "root_relpath",
                "dna_physics",
                "density_gcm3",
            ]
        )
        for i, (elo, ehi, ec, de) in enumerate(zip(edges_mev[:-1], edges_mev[1:], centers_mev, widths_mev)):
            run_row = run_rows_by_index[i]
            wr.writerow(
                [
                    i,
                    f"{elo:.9g}",
                    f"{ehi:.9g}",
                    f"{ec:.9g}",
                    f"{de:.9g}",
                    int(run_row["sim_particles"]),
                    run_row["macro_relpath"],
                    run_row["root_basename"],
                    run_row["root_relpath"],
                    run_row["dna_physics"],
                    run_row["density_gcm3"],
                ]
            )

    per_energy_csv = out_dir / "per_energy_runs.csv"
    with per_energy_csv.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "energy_index",
                "E_low_MeV",
                "E_high_MeV",
                "E_center_MeV",
                "dE_MeV",
                "sim_particles",
                "macro_relpath",
                "root_basename",
                "root_relpath",
                "dna_physics",
                "density_gcm3",
            ]
        )
        for i, (elo, ehi, ec, de) in enumerate(zip(edges_mev[:-1], edges_mev[1:], centers_mev, widths_mev)):
            run_row = run_rows_by_index[i]
            wr.writerow(
                [
                    i,
                    f"{elo:.9g}",
                    f"{ehi:.9g}",
                    f"{ec:.9g}",
                    f"{de:.9g}",
                    int(run_row["sim_particles"]),
                    run_row["macro_relpath"],
                    run_row["root_basename"],
                    run_row["root_relpath"],
                    run_row["dna_physics"],
                    run_row["density_gcm3"],
                ]
            )

    runner_script = _write_runner_script(
        out_dir, int(args.threads), str(args.dna_physics)
    )

    # Save per-cell range summary and count of assigned global bins.
    range_csv = out_dir / "latlon_cell_ranges.csv"

    # Long, compressed scaling table:
    # For each valid cell and assigned energy index:
    #   flux_bin_model := J(E_i)*overlap_dE
    #   weight_norm := flux_bin_model / sum_j flux_bin_model
    #   scale_to_sim := flux_bin_model / n_per_energy(E_i)
    scaling_csv_gz = out_dir / "latlon_energy_scaling.csv.gz"
    flux_all = electron_spectrum_fit(centers_mev)

    with range_csv.open("w", newline="") as ranges_fh, gzip.open(
        scaling_csv_gz, "wt", newline=""
    ) as scales_fh:
        range_wr = csv.writer(ranges_fh)
        scale_wr = csv.writer(scales_fh)

        range_wr.writerow(
            [
                "cell_id",
                "lat_deg",
                "lon_w_deg",
                "hemisphere",
                "e_min_mev",
                "e_max_mev",
                "is_valid",
                "n_assigned_bins",
            ]
        )
        scale_wr.writerow(
            [
                "cell_id",
                "lat_deg",
                "lon_w_deg",
                "hemisphere",
                "energy_index",
                "E_center_MeV",
                "overlap_dE_MeV",
                "flux_model",
                "flux_bin_model",
                "weight_norm",
                "sim_particles",
                "scale_to_sim",
            ]
        )

        for row in cell_rows:
            cid = int(row["cell_id"])
            lat = float(row["lat_deg"])
            lon = float(row["lon_w_deg"])
            hemisphere = str(row["hemisphere"])
            e_min = float(row["e_min_mev"])
            e_max = float(row["e_max_mev"])
            valid = int(row["is_valid"])

            assigned = np.array([], dtype=int)
            overlap = np.zeros_like(centers_mev)
            if valid:
                # Assign bins by overlap with [e_min, e_max], not center inclusion.
                overlap_lo = np.maximum(edges_mev[:-1], e_min)
                overlap_hi = np.minimum(edges_mev[1:], e_max)
                overlap = np.maximum(0.0, overlap_hi - overlap_lo)
                assigned = np.where(overlap > 0.0)[0]

            range_wr.writerow(
                [
                    cid,
                    f"{lat:.9g}",
                    f"{lon:.9g}",
                    hemisphere,
                    f"{e_min:.9g}",
                    f"{e_max:.9g}",
                    valid,
                    int(assigned.size),
                ]
            )

            if assigned.size == 0:
                continue

            weights_raw = flux_all[assigned] * overlap[assigned]
            norm = float(np.sum(weights_raw))
            if norm <= 0.0:
                continue
            weights = weights_raw / norm
            scales = weights_raw / n_per_energy_values[assigned].astype(float)

            for idx, w_raw, w, s in zip(assigned, weights_raw, weights, scales):
                scale_wr.writerow(
                    [
                        cid,
                        f"{lat:.9g}",
                        f"{lon:.9g}",
                        hemisphere,
                        int(idx),
                        f"{centers_mev[idx]:.9g}",
                        f"{overlap[idx]:.9g}",
                        f"{flux_all[idx]:.9g}",
                        f"{w_raw:.9g}",
                        f"{w:.9g}",
                        int(n_per_energy_values[idx]),
                        f"{s:.9g}",
                    ]
                )

    print(f"Wrote combined macro: {combined_macro_path}")
    print(f"Wrote per-energy run table: {per_energy_csv}")
    print(f"Wrote per-energy runner script: {runner_script}")
    print(f"Wrote global energy bins: {energy_csv}")
    print(f"Wrote cell ranges: {range_csv}")
    print(f"Wrote compressed scaling table: {scaling_csv_gz}")
    print(f"Grid: {args.n_lat} x {args.n_lon} = {args.n_lat * args.n_lon} cells")
    print(f"Global energy range (MeV): {global_e_min:.9g} to {global_e_max:.9g}")
    print(f"Global bins: {len(centers_mev)}  spacing: logarithmic")
    print(f"Default DNA_PHYSICS in runner: {args.dna_physics}")
    print(f"Density used in macros: {density_gcm3:.9g} g/cm3")
    print(f"Angular distribution in macros: {args.angular_dist}")
    if args.n_per_energy_thresholds is None:
        print(f"Sim particles per energy bin: constant {int(args.n_per_energy)}")
    else:
        pairs = ", ".join(
            f"E<= {thr:.9g} MeV -> {int(n)}"
            for thr, n in zip(args.n_per_energy_thresholds, args.n_per_energy_values)
        )
        print(f"Sim particles per energy bin: piecewise [{pairs}]")
    if integral_sum > 0.0:
        delta_pct = 100.0 * (ratio - 1.0)
        print(f"Riemann/Integral ratio over global range: {ratio:.9g} ({delta_pct:+.4f}%)")
    if worst_cell_rel_index >= 0:
        worst_delta_pct = 100.0 * (worst_cell_ratio - 1.0)
        print(
            f"Per-cell max |Riemann/Integral-1|: {max_cell_abs_err:.9g} "
            f"({100.0 * max_cell_abs_err:.4f}%)"
        )
        print(
            f"Worst tolerance cell_id={worst_cell_id} "
            f"range={worst_cell_e_min:.9g}..{worst_cell_e_max:.9g} MeV "
            f"ratio={worst_cell_ratio:.9g} ({worst_delta_pct:+.4f}%)"
        )
    if args.manual_density_gcm3 is None:
        print("Density mode: phase-default (set DNA_PHYSICS=ice_hex or ice_am at runtime).")
    else:
        print(f"Density mode: manual {args.manual_density_gcm3:.9g} g/cm3 in macro.")


if __name__ == "__main__":
    main()
