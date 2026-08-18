from __future__ import annotations

from pathlib import Path
import sys

import pytest


NEP_DIR = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "nep_mbpol"
)
if str(NEP_DIR) not in sys.path:
    sys.path.insert(0, str(NEP_DIR))

from soft_dft.density_validation import compare_density_cube_files, read_cube_density


def _cube(path: Path, values: list[float], *, origin: float = 0.0) -> None:
    path.write_text(
        "density\n"
        "test fixture\n"
        f"1 {origin:.1f} 0.0 0.0\n"
        "2 0.5 0.0 0.0\n"
        "1 0.0 0.5 0.0\n"
        "1 0.0 0.0 0.5\n"
        "6 0.0 0.0 0.0 0.0\n"
        + " ".join(str(value) for value in values)
        + "\n",
        encoding="utf-8",
    )


def test_same_grid_density_rms_is_measured_without_rescaling(tmp_path: Path) -> None:
    first = tmp_path / "first.cube"
    second = tmp_path / "second.cube"
    _cube(first, [1.0, 2.0])
    _cube(second, [1.0, 2.2])
    result = compare_density_cube_files(first, second)
    assert result["voxel_count"] == 2
    assert result["absolute_rms"] == pytest.approx((0.04 / 2.0) ** 0.5)
    assert result["maximum_absolute_difference"] == pytest.approx(0.2)


def test_density_comparison_rejects_incompatible_grids_and_truncation(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.cube"
    second = tmp_path / "second.cube"
    _cube(first, [1.0, 2.0])
    _cube(second, [1.0, 2.0], origin=0.1)
    with pytest.raises(ValueError, match="different origins"):
        compare_density_cube_files(first, second)
    first.write_text("too short\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Truncated cube header"):
        read_cube_density(first)
