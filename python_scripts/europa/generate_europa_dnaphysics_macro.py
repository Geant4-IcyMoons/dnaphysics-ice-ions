#!/usr/bin/env python3
"""
Generate a dnaphysics-ice GPS macro for a Europa lat/lon (west) region.

Uses the same hemisphere logic and energy bounds as generate_europa_electron_bins.py.
Geometry assumptions:
  - ice slab spans z = 0 to z = z_max (from legacy z_range)
  - ice size set via /dna/test/setIceSize X Y Z (full lengths)
  - beam starts just outside the ice surface (default z = -0.001 mm)
  - angular distribution: isotropic in the +z hemisphere (phi uniform, mu=cos(theta) uniform)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from generate_europa_electron_bins import (
    _allowed_energy_range,
    _allowed_energy_range_from_value,
    _energy_bins,
    _find_delta_e,
    _infer_hemisphere,
    _interpolate_map_value,
    _lat_lon_to_indices,
    _load_map,
    _normalize_lon,
)
from constants import TOP_ROOT


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


def _write_macro(
    out_path: Path,
    *,
    lat: float,
    lon_w: float,
    hemisphere: str,
    e_min: float,
    e_max: float,
    delta_e: float,
    centers: np.ndarray,
    n_per_bin: int,
    xy_half_mm: float,
    z_start_mm: float,
    ice_z_mm: float,
    material: str,
) -> None:
    lines: list[str] = []
    lines.append("# Europa dnaphysics-ice macro (auto-generated)")
    lines.append(f"# Lat/Lon W: {lat:.3f}, {lon_w:.3f} ({hemisphere})")
    lines.append(f"# Energy range (MeV): {e_min:.6g} to {e_max:.6g}")
    lines.append(f"# deltaE (MeV): {delta_e:.6g}; bins: {len(centers)}")
    lines.append(f"# Ice size (mm): X=Y={xy_half_mm*2:.3f}, Z={ice_z_mm:.3f}")
    lines.append(f"# Beam start z (mm): {z_start_mm:.3f}")
    lines.append("")

    lines += [
        "/control/verbose 0",
        "/run/verbose 0",
        "/event/verbose 0",
        "/tracking/verbose 0",
        "",
        f"/dna/test/setMat {material}",
        f"/dna/test/setIceSize {xy_half_mm*2:.6g} {xy_half_mm*2:.6g} {ice_z_mm:.6g} mm",
        "",
        "/run/initialize",
        "",
        "/gps/particle e-",
        "/gps/number 1",
        "/gps/pos/type Plane",
        "/gps/pos/shape Square",
        f"/gps/pos/centre 0 0 {z_start_mm:.6g} mm",
        f"/gps/pos/halfx {xy_half_mm:.6g} mm",
        f"/gps/pos/halfy {xy_half_mm:.6g} mm",
        "/gps/ang/type iso",
        "/gps/ang/mintheta 0 deg",
        "/gps/ang/maxtheta 90 deg",
        "/gps/ang/minphi 0 deg",
        "/gps/ang/maxphi 360 deg",
        "",
        "# Energy loop: monoenergetic runs, N electrons per bin",
    ]

    for e in centers:
        lines.append(f"/gps/ene/mono {e:.6g} MeV")
        lines.append(f"/run/beamOn {int(n_per_bin)}")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate dnaphysics-ice GPS macro for Europa lat/lon.")
    ap.add_argument("--lat", type=float, required=True, help="Latitude in degrees.")
    ap.add_argument("--lon-west", type=float, required=True, help="West longitude in degrees [0, 360).")
    ap.add_argument("--hemisphere", choices=("leading", "trailing"), default=None,
                    help="Override hemisphere selection.")
    ap.add_argument("--nbins", type=int, default=100, help="Initial bins for auto deltaE.")
    ap.add_argument("--delta-e", type=float, default=None, help="Fixed deltaE in MeV.")
    ap.add_argument("--tol", type=float, default=1.0e-4,
                    help="Relative tolerance for sum/integral when auto-selecting deltaE.")
    ap.add_argument("--n-per-bin", type=int, default=1000, help="Electrons per energy bin.")
    ap.add_argument("--leading-cap", type=float, default=100.0, help="Leading max energy (MeV).")
    ap.add_argument("--trailing-min", type=float, default=0.01, help="Trailing min energy (MeV).")
    ap.add_argument("--material", default="G4_ICE", help="Material for /dna/test/setMat.")
    ap.add_argument("--xy-size-mm", type=float, default=1.0,
                    help="Full XY size in mm for the ice slab.")
    ap.add_argument("--source-z-mm", type=float, default=-0.001,
                    help="Beam start z (mm), relative to ice surface at z=0.")
    ap.add_argument("--snap", action="store_true",
                    help="Use nearest grid point instead of interpolating.")
    ap.add_argument("--out", type=Path, default=None, help="Output macro path.")
    args = ap.parse_args()

    lon_w = _normalize_lon(args.lon_west)
    hemisphere = args.hemisphere or _infer_hemisphere(lon_w)

    if hemisphere == "leading" and not (0.0 <= lon_w <= 180.0):
        raise ValueError("Leading hemisphere expects west longitude in [0, 180].")
    if hemisphere == "trailing" and not (180.0 <= lon_w <= 360.0):
        raise ValueError("Trailing hemisphere expects west longitude in [180, 360].")

    if hemisphere == "leading":
        map_path = TOP_ROOT / "e_bombardment_leading"
        data = _load_map(map_path)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D leading map, got shape {data.shape}")
    else:
        map_path = TOP_ROOT / "e_bombardment_trailing"
        data = _load_map(map_path)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D trailing map, got shape {data.shape}")

    if args.snap:
        lat_idx, lon_idx, lat_snap, lon_snap = _lat_lon_to_indices(
            args.lat, lon_w, hemisphere, data.shape[:2]
        )
        e_min, e_max = _allowed_energy_range(
            hemisphere, data, lat_idx, lon_idx, args.leading_cap, args.trailing_min
        )
    else:
        lat_snap, lon_snap = float(args.lat), float(lon_w)
        interp_val = _interpolate_map_value(data, args.lat, lon_w, hemisphere)
        e_min, e_max = _allowed_energy_range_from_value(
            hemisphere, interp_val, args.leading_cap, args.trailing_min
        )
    if e_max <= 0.0 or e_min <= 0.0 or e_min >= e_max:
        raise ValueError(
            f"No valid energy range at lat={args.lat}, lon_w={lon_w} (MeV)."
        )

    if args.delta_e is None:
        delta_e, _, _ = _find_delta_e(
            e_min, e_max, initial_bins=args.nbins, tol=float(args.tol), max_iter=30, max_bins=1_000_000
        )
    else:
        delta_e = float(args.delta_e)

    edges, centers = _energy_bins(e_min, e_max, args.nbins, delta_e)

    z_range = _legacy_z_range_mm()
    z_max = z_range[-1] + (z_range[-1] - z_range[-2])
    z_start = float(args.source_z_mm)
    xy_half = args.xy_size_mm / 2.0

    if args.out is None:
        name_hemi = "leading" if (0.0 <= lon_w < 180.0) else "trailing"
        out_name = f"europa_lat{lat_snap:.1f}_lonW{lon_snap:.1f}_{name_hemi}.mac"
        out_path = Path(__file__).resolve().parent.parent / out_name
    else:
        out_path = args.out

    _write_macro(
        out_path,
        lat=args.lat,
        lon_w=lon_w,
        hemisphere=hemisphere,
        e_min=e_min,
        e_max=e_max,
        delta_e=delta_e,
        centers=centers,
        n_per_bin=args.n_per_bin,
        xy_half_mm=xy_half,
        z_start_mm=z_start,
        ice_z_mm=z_max,
        material=args.material,
    )

    print("Wrote macro:", out_path)
    print("Hemisphere:", hemisphere)
    print("Lat/Lon used (deg):", f"{lat_snap:.3f}", f"{lon_snap:.3f}")
    print("Energy range (MeV):", f"{e_min:.6g}", "to", f"{e_max:.6g}")
    print("DeltaE (MeV):", f"{delta_e:.6g}", "Bins:", len(centers))


if __name__ == "__main__":
    main()
