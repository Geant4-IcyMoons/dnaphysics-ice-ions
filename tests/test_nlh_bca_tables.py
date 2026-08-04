from __future__ import annotations

from pathlib import Path
import sys


NEP_MBPOL = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "nep_mbpol"
)
if str(NEP_MBPOL) not in sys.path:
    sys.path.insert(0, str(NEP_MBPOL))

from bca.tables import KernelTableConfig, generate_kernel_tables  # noqa: E402


def test_parallel_table_output_is_deterministic_and_resumable(tmp_path):
    common = dict(
        projectiles=("C",),
        energy_min_ev=1.0e4,
        energy_max_ev=1.0e5,
        energy_points=2,
        impact_points=3,
        quadrature_order=32,
    )
    serial = KernelTableConfig(**common, workers=1)
    parallel = KernelTableConfig(**common, workers=2)
    serial_csv, _ = generate_kernel_tables(
        serial, tmp_path / "serial", show_progress=False
    )
    parallel_csv, _ = generate_kernel_tables(
        parallel, tmp_path / "parallel", show_progress=False
    )
    assert serial_csv.read_bytes() == parallel_csv.read_bytes()

    before = serial_csv.read_bytes()
    resumed_csv, _ = generate_kernel_tables(
        serial, tmp_path / "serial", resume=True, show_progress=False
    )
    assert resumed_csv.read_bytes() == before
