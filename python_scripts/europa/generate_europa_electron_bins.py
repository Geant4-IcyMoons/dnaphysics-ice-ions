#!/usr/bin/env python3
"""
Generate electron energy bins for a Europa latitude/longitude (west).

This mirrors the legacy pipeline logic:
- Leading hemisphere: allowed energies are >= E_min at the coordinate.
- Trailing hemisphere: allowed energies are <= E_max at the coordinate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from physics_ice.constants import TOP_ROOT

def electron_spectrum_fit(E, j0=4.23, E0=3.11, a=-1.58, b=1.86):
    # E in MeV
    E = np.asarray(E, dtype=float)
    return j0 * 1e6 * (E ** a) * ((1.0 + (E / E0)) ** (-b))


def _load_map(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing map file: {path}")
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return np.asarray(data, dtype=float)


def _normalize_lon(lon_w: float) -> float:
    lon = float(lon_w) % 360.0
    if np.isclose(lon, 360.0):
        lon = 0.0
    return lon


def _infer_hemisphere(lon_w: float) -> str:
    return "leading" if lon_w < 180.0 else "trailing"


def _lat_lon_to_indices(
    lat_deg: float,
    lon_w_deg: float,
    hemisphere: str,
    shape: tuple[int, int],
) -> tuple[int, int, float, float]:
    n_lat, n_lon = shape
    lat_grid = np.linspace(90.0, -90.0, n_lat)
    if hemisphere == "leading":
        lon_min, lon_max = 0.0, 180.0
    else:
        lon_min, lon_max = 180.0, 360.0
    lon_grid = np.linspace(lon_min, lon_max, n_lon)

    lat_idx = int(np.argmin(np.abs(lat_grid - lat_deg)))
    lon_idx = int(np.argmin(np.abs(lon_grid - lon_w_deg)))
    return lat_idx, lon_idx, float(lat_grid[lat_idx]), float(lon_grid[lon_idx])


def _lat_lon_grids(shape: tuple[int, int], hemisphere: str) -> tuple[np.ndarray, np.ndarray]:
    n_lat, n_lon = shape
    lat_grid = np.linspace(90.0, -90.0, n_lat)
    if hemisphere == "leading":
        lon_min, lon_max = 0.0, 180.0
    else:
        lon_min, lon_max = 180.0, 360.0
    lon_grid = np.linspace(lon_min, lon_max, n_lon)
    return lat_grid, lon_grid


def _bilinear_indices(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    lat_deg: float,
    lon_deg: float,
) -> tuple[int, int, float, int, int, float]:
    lat = float(np.clip(lat_deg, lat_grid[-1], lat_grid[0]))
    lon = float(np.clip(lon_deg, lon_grid[0], lon_grid[-1]))

    inv_lat = -lat_grid
    i1 = int(np.searchsorted(inv_lat, -lat, side="left"))
    i1 = max(0, min(i1, len(lat_grid) - 1))
    i0 = max(i1 - 1, 0)
    if i0 == i1:
        t = 0.0
    else:
        t = (lat - lat_grid[i0]) / (lat_grid[i1] - lat_grid[i0])

    j1 = int(np.searchsorted(lon_grid, lon, side="left"))
    j1 = max(0, min(j1, len(lon_grid) - 1))
    j0 = max(j1 - 1, 0)
    if j0 == j1:
        u = 0.0
    else:
        u = (lon - lon_grid[j0]) / (lon_grid[j1] - lon_grid[j0])

    return i0, i1, t, j0, j1, u


def _interpolate_map_value(
    data: np.ndarray,
    lat_deg: float,
    lon_w_deg: float,
    hemisphere: str,
) -> np.ndarray:
    lat_grid, lon_grid = _lat_lon_grids(data.shape[:2], hemisphere)
    i0, i1, t, j0, j1, u = _bilinear_indices(lat_grid, lon_grid, lat_deg, lon_w_deg)
    w00 = (1.0 - t) * (1.0 - u)
    w10 = t * (1.0 - u)
    w01 = (1.0 - t) * u
    w11 = t * u
    if data.ndim == 2:
        return (
            w00 * data[i0, j0]
            + w10 * data[i1, j0]
            + w01 * data[i0, j1]
            + w11 * data[i1, j1]
        )
    return (
        w00 * data[i0, j0, :]
        + w10 * data[i1, j0, :]
        + w01 * data[i0, j1, :]
        + w11 * data[i1, j1, :]
    )


def _allowed_energy_range_from_value(
    hemisphere: str,
    value: np.ndarray,
    leading_cap: float,
    trailing_min_default: float,
) -> tuple[float, float]:
    if hemisphere == "leading":
        e_min = float(value)
        e_max = float(leading_cap)
        return e_min, e_max

    vec = np.asarray(value, dtype=float)
    e_max = float(np.max(vec))
    if e_max <= 0.0:
        return 0.0, 0.0
    positive = vec[vec > 0.0]
    if positive.size:
        e_min = float(np.min(positive))
    else:
        e_min = float(trailing_min_default)
    return e_min, e_max


def _allowed_energy_range(
    hemisphere: str,
    data: np.ndarray,
    lat_idx: int,
    lon_idx: int,
    leading_cap: float,
    trailing_min_default: float,
) -> tuple[float, float]:
    if hemisphere == "leading":
        e_min = float(data[lat_idx, lon_idx])
        e_max = float(leading_cap)
        return e_min, e_max

    vec = np.asarray(data[lat_idx, lon_idx], dtype=float)
    e_max = float(np.max(vec))
    if e_max <= 0.0:
        return 0.0, 0.0
    positive = vec[vec > 0.0]
    if positive.size:
        e_min = float(np.min(positive))
    else:
        e_min = float(trailing_min_default)
    return e_min, e_max


def _energy_bins(
    e_min: float,
    e_max: float,
    nbins: int,
    delta_e: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if delta_e is None:
        edges = np.logspace(np.log10(e_min), np.log10(e_max), nbins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
        return edges, centers

    edges = np.arange(e_min, e_max + delta_e, delta_e, dtype=float)
    if edges[-1] < e_max:
        edges = np.append(edges, e_max)
    elif edges[-1] > e_max:
        edges[-1] = e_max
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _weights_from_spectrum(
    edges: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    widths = edges[1:] - edges[:-1]
    spectrum = electron_spectrum_fit(centers)
    weights = spectrum * widths
    total = float(np.sum(weights))
    if total <= 0.0:
        return spectrum, np.zeros_like(centers)

    weights_norm = weights / total
    return spectrum, weights_norm


def _integrate_spectrum(e_min: float, e_max: float, samples: int = 20000) -> float:
    if e_min <= 0.0 or e_max <= e_min:
        return 0.0
    grid = np.logspace(np.log10(e_min), np.log10(e_max), samples)
    vals = electron_spectrum_fit(grid)
    return float(np.trapezoid(vals, grid))


def _find_delta_e(
    e_min: float,
    e_max: float,
    initial_bins: int,
    tol: float,
    max_iter: int,
    max_bins: int,
) -> tuple[float, float, float]:
    integral_sum = _integrate_spectrum(e_min, e_max)
    if integral_sum <= 0.0:
        return 0.0, 0.0, 0.0

    bins = max(1, int(initial_bins))
    delta_e = (e_max - e_min) / bins
    if delta_e <= 0.0:
        return 0.0, 0.0, integral_sum

    ratio = 0.0
    for _ in range(max_iter):
        edges, centers = _energy_bins(e_min, e_max, nbins=1, delta_e=delta_e)
        if centers.size > max_bins:
            raise ValueError(
                f"Auto deltaE exceeded max bins ({max_bins}); "
                f"last deltaE={delta_e:.6g} MeV."
            )
        widths = edges[1:] - edges[:-1]
        spectrum_vals = electron_spectrum_fit(centers)
        discrete_sum = float(np.sum(spectrum_vals * widths))
        ratio = discrete_sum / integral_sum
        if abs(ratio - 1.0) <= tol:
            return delta_e, ratio, integral_sum
        delta_e *= 0.5

    raise ValueError(
        f"Auto deltaE failed to reach tol={tol:.2e} after {max_iter} iterations."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate electron counts per energy bin for a Europa coordinate."
    )
    parser.add_argument("--lat", type=float, required=True, help="Latitude in degrees.")
    parser.add_argument(
        "--lon-west",
        type=float,
        required=True,
        help="West longitude in degrees [0, 360).",
    )
    parser.add_argument(
        "--hemisphere",
        type=str,
        choices=("leading", "trailing"),
        default=None,
        help="Override hemisphere selection.",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=100,
        help="Initial number of bins for auto deltaE (ignored when --delta-e is set).",
    )
    parser.add_argument(
        "--n-per-bin",
        type=int,
        default=1000,
        help="Constant number of electrons per bin (simulation input).",
    )
    parser.add_argument(
        "--delta-e",
        type=float,
        default=None,
        help="Linear bin width in MeV (overrides --nbins when set).",
    )
    parser.add_argument(
        "--snap",
        action="store_true",
        help="Use nearest grid point instead of interpolating.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-4,
        help="Relative tolerance for sum/integral when auto-selecting deltaE.",
    )
    parser.add_argument(
        "--leading-cap",
        type=float,
        default=100.0,
        help="Leading hemisphere max energy (MeV).",
    )
    parser.add_argument(
        "--trailing-min",
        type=float,
        default=0.01,
        help="Trailing hemisphere min energy when no lower bound is present (MeV).",
    )
    args = parser.parse_args()

    lon_w = _normalize_lon(args.lon_west)
    hemisphere = args.hemisphere or _infer_hemisphere(lon_w)

    if hemisphere == "leading" and not (0.0 <= lon_w <= 180.0):
        raise ValueError("Leading hemisphere expects west longitude in [0, 180].")
    if hemisphere == "trailing" and not (180.0 <= lon_w <= 360.0):
        raise ValueError("Trailing hemisphere expects west longitude in [180, 360].")

    if hemisphere == "leading":
        map_path = TOP_ROOT / "dnaphysics-ice/python_scripts/europa/e_bombardment_leading"
        data = _load_map(map_path)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D leading map, got shape {data.shape}")
    else:
        map_path = TOP_ROOT / "dnaphysics-ice/python_scripts/europa/e_bombardment_trailing"
        data = _load_map(map_path)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D trailing map, got shape {data.shape}")

    if args.snap:
        lat_idx, lon_idx, lat_snap, lon_snap = _lat_lon_to_indices(
            args.lat, lon_w, hemisphere, data.shape[:2]
        )
        e_min, e_max = _allowed_energy_range(
            hemisphere,
            data,
            lat_idx,
            lon_idx,
            args.leading_cap,
            args.trailing_min,
        )
    else:
        lat_snap, lon_snap = float(args.lat), float(lon_w)
        interp_val = _interpolate_map_value(data, args.lat, lon_w, hemisphere)
        e_min, e_max = _allowed_energy_range_from_value(
            hemisphere,
            interp_val,
            args.leading_cap,
            args.trailing_min,
        )

    if e_max <= 0.0 or e_min <= 0.0 or e_min >= e_max:
        raise ValueError(
            f"No valid energy range at lat={args.lat}, lon_w={lon_w} (MeV)."
        )

    tol = float(args.tol)
    max_iter = 30
    max_bins = 1000000

    if args.delta_e is None:
        delta_e, ratio, integral_sum = _find_delta_e(
            e_min,
            e_max,
            initial_bins=args.nbins,
            tol=tol,
            max_iter=max_iter,
            max_bins=max_bins,
        )
        if delta_e <= 0.0:
            raise ValueError("Auto deltaE failed to produce a valid bin width.")
    else:
        delta_e = float(args.delta_e)
        integral_sum = _integrate_spectrum(e_min, e_max)
        edges_tmp, centers_tmp = _energy_bins(e_min, e_max, args.nbins, delta_e)
        widths_tmp = edges_tmp[1:] - edges_tmp[:-1]
        spectrum_tmp = electron_spectrum_fit(centers_tmp)
        discrete_sum = float(np.sum(spectrum_tmp * widths_tmp))
        ratio = discrete_sum / integral_sum if integral_sum > 0.0 else 0.0

    edges, centers = _energy_bins(e_min, e_max, args.nbins, delta_e)
    spectrum_vals, weights_norm = _weights_from_spectrum(edges, centers)
    widths = edges[1:] - edges[:-1]

    print("Hemisphere:", hemisphere)
    print("Lat/Lon input (deg):", args.lat, lon_w)
    print("Lat/Lon used (deg):", f"{lat_snap:.3f}", f"{lon_snap:.3f}")
    print("Energy range (MeV):", f"{e_min:.4g}", "to", f"{e_max:.4g}")
    print("Bins:", len(centers), "Electrons per bin:", int(args.n_per_bin))
    print("DeltaE (MeV):", f"{delta_e:.6g}")
    if integral_sum > 0.0:
        print(
            "Sum(J(E)*dE)/Integral:",
            f"{ratio:.6f}",
            f"(delta {100.0 * (ratio - 1.0):+.3f}%)",
        )
    print("First 5 bins (E_min, E_max, E_center, deltaE, weight_norm):")
    for i in range(min(5, len(centers))):
        print(
            f"{edges[i]:.6g}",
            f"{edges[i + 1]:.6g}",
            f"{centers[i]:.6g}",
            f"{widths[i]:.6g}",
            f"{weights_norm[i]:.6g}",
        )


if __name__ == "__main__":
    main()
