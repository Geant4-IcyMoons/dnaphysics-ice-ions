from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


PHYSICS_SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "python_scripts" / "physics_ice"
)
BENCHMARK_DIR = (
    PHYSICS_SCRIPT_DIR
    / "process_evidence"
    / "charge_exchange_ctmc"
    / "benchmarking"
)
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(PHYSICS_SCRIPT_DIR))

import benchmark_carbon_charge_exchange_ctmc as benchmark  # noqa: E402
import run_ctmc81_c3_reproduction as reproduction  # noqa: E402


def _uniform_rate_table(points: int = 41) -> dict[str, np.ndarray]:
    energies = np.geomspace(1.0, 1.0e4, points)
    rows = []
    for energy in energies:
        for charge in range(7):
            sc = 0.0 if charge == 0 else 0.6e-16
            ti = 0.0 if charge == 0 else 0.4e-16
            sl = 0.0 if charge == 6 else 0.7e-16
            li = 0.0 if charge == 6 else 0.3e-16
            rows.append(
                (
                    energy,
                    charge,
                    sc,
                    ti,
                    sl,
                    li,
                    sc + ti,
                    sl + li,
                    0.2e-16,
                    0.1e-16,
                )
            )
    matrix = np.asarray(rows)
    return {
        name: matrix[:, index]
        for index, name in enumerate(benchmark.EXPECTED_COLUMNS)
    }


def _write_table(path: Path, table: dict[str, np.ndarray]) -> None:
    matrix = np.column_stack(
        [table[column] for column in benchmark.EXPECTED_COLUMNS]
    )
    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header=",".join(benchmark.EXPECTED_COLUMNS),
        comments="",
    )


def test_equilibrium_solver_satisfies_paper_balance_equation() -> None:
    result = benchmark.equilibrium_charge_fractions(_uniform_rate_table())
    np.testing.assert_allclose(result["fractions"], 1.0 / 7.0, rtol=1e-13)
    np.testing.assert_allclose(result["mean_charge"], 3.0, rtol=1e-13)
    assert np.max(result["relative_stationarity_residual"]) < 1.0e-14


def test_plot_energy_is_total_carbon_kinetic_energy() -> None:
    np.testing.assert_allclose(
        benchmark._carbon_total_energy_ev(np.asarray([1.0, 1.0e4])),
        [1.2e4, 1.2e8],
    )
    assert benchmark.PLOT_ENERGY_MIN_EV == 1.0e4
    assert benchmark.PLOT_ENERGY_MAX_EV == 1.0e8


def test_paper_digitization_has_all_three_comparison_figures() -> None:
    paper = benchmark._read_paper_data()
    assert set(np.unique(paper["figure"])) == {12, 13, 14}
    assert np.all(paper["value"] > 0.0)
    assert np.all(paper["digitization_uncertainty"] > 0.0)
    assert len(np.unique(paper[paper["figure"] == 12]["q"])) == 6
    assert len(np.unique(paper[paper["figure"] == 13]["q"])) == 6
    assert len(np.unique(paper[paper["figure"] == 14]["q"])) == 7


def test_formal_ctmc81_archive_is_exact_and_internally_consistent() -> None:
    reference = benchmark._read_formal_reference_archive(
        benchmark.FORMAL_REFERENCE_ARCHIVE
    )
    assert reference["sha256"] == benchmark.FORMAL_REFERENCE_SHA256
    assert set(reference["datasets"]) == {
        "L1", "L2", "L3", "L4", "L5", "loss"
    }
    checks = []
    benchmark._formal_reference_integrity(reference, checks)
    assert checks == [
        next(
            check
            for check in checks
            if check["name"] == "formal_ctmc81_archive_integrity"
        )
    ]
    assert checks[0]["status"] == "pass"
    assert reference["datasets"]["L1"]["metadata"] == {
        "projectile_nuclear_charge": 6.0,
        "projectile_charge_state": 3.0,
        "projectile_mass_u": 12.0,
        "energy_keV_u": 100.0,
        "projectile_velocity_au": 2.00055347542277,
        "initial_z_au": 1000.0,
        "maximum_impact_au": 10.0,
        "impact_point_count": 100,
        "trajectory_count": 20000,
        "integration_time_au": 1000.0,
        "integration_tolerance": 1.0e-6,
        "limit_distance_au": 1000.0,
        "orbital_index": 1,
    }
    assert reference["datasets"]["loss"]["metadata"]["initial_z_au"] == 10000.0
    assert reference["datasets"]["loss"]["metadata"]["integration_time_au"] == 6000.0


def test_ctmc81_reproduction_matches_every_disclosed_point_parameter() -> None:
    reference = benchmark._read_formal_reference_archive(
        benchmark.FORMAL_REFERENCE_ARCHIVE
    )
    specs = reproduction._build_specs(reference)
    assert len(specs) == 410
    counts = [
        sum(spec["dataset"] == label for spec in specs)
        for label in reproduction.REFERENCE_LABELS
    ]
    assert counts == [
        100,
        100,
        50,
        50,
        10,
        100,
    ]
    target = next(spec for spec in specs if spec["dataset"] == "L5")
    loss = next(spec for spec in specs if spec["dataset"] == "loss")
    target_parameters = (
        target["trajectory_count"],
        target["start_separation_au"],
        target["integration_time_au"],
    )
    assert target_parameters == (
        20000,
        1000.0,
        1000.0,
    )
    assert target["projectile_velocity_au"] == 2.00055347542277
    loss_parameters = (
        loss["trajectory_count"],
        loss["start_separation_au"],
        loss["integration_time_au"],
    )
    assert loss_parameters == (
        10000,
        10000.0,
        6000.0,
    )


def test_ctmc81_reproduction_forces_paper_sampler() -> None:
    config = reproduction._configuration(
        trajectory_chunk_size=100,
        seed=20260816,
        rtol=1.0e-11,
        atol=1.0e-13,
        retry_rtol=1.0e-12,
        retry_atol=1.0e-14,
        maximum_relative_energy_drift=1.0e-4,
        maximum_integration_steps=100_000_000,
    )
    assert config.initial_ensemble == (
        benchmark.ctmc.INITIAL_ENSEMBLE_LIAMSUWAN_OLSON_SALOP
    )
    assert config.projectile_loss_initialization == (
        benchmark.ctmc.PROJECTILE_LOSS_INITIALIZATION_CTMC81_NUCLEUS
    )
    assert benchmark.ctmc.random_draws_per_trajectory(
        config.initial_ensemble
    ) == 4


@pytest.mark.parametrize(
    ("ensemble_mode", "paper_reproduction", "expected_status"),
    (
        (
            benchmark.ctmc.INITIAL_ENSEMBLE_LIAMSUWAN_OLSON_SALOP,
            True,
            "pass",
        ),
        (
            benchmark.ctmc.INITIAL_ENSEMBLE_INDEPENDENT_ISOTROPIC,
            False,
            "fail",
        ),
    ),
)
def test_release_gate_accepts_only_paper_sampler(
    tmp_path: Path,
    ensemble_mode: str,
    paper_reproduction: bool,
    expected_status: str,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "doi": benchmark.PAPER_DOI,
                "projectile": {"nuclear_charge": 6},
                "density_applied": False,
                "cross_section_unit": "cm2 per H2O molecule",
                "initial_ensemble": {
                    "mode": ensemble_mode,
                    "paper_reproduction": paper_reproduction,
                },
                "trajectory_statistics": {
                    "failed": 0,
                    "maximum_relative_total_energy_drift": 0.0,
                },
                "ctmc_config": {"maximum_relative_energy_drift": 1.0e-4},
            }
        ),
        encoding="utf-8",
    )
    checks: list[dict[str, object]] = []
    benchmark._metadata_checks(
        metadata_path,
        tmp_path / "missing_probabilities.npz",
        checks,
        require_adaptive=False,
    )
    ensemble_check = next(
        check for check in checks if check["name"] == "paper_initial_ensemble"
    )
    assert ensemble_check["status"] == expected_status


def test_ctmc81_event_columns_preserve_null_and_scheme_identity() -> None:
    assert reproduction._event_index("L1", "ionized") == 0
    assert reproduction._event_index("L1", "captured_projectile") == 1
    assert reproduction._event_index("L1", "retained_target") == 2
    assert reproduction._event_index("loss", "ionized") == 0
    assert reproduction._event_index("loss", "retained_target") == 1
    assert reproduction._event_index("loss", "captured_projectile") == 2
    assert reproduction._event_index("loss", "ambiguous_bound") == 3


def test_ctmc81_derived_cross_sections_use_converged_common_quadrature() -> None:
    reference = benchmark._read_formal_reference_archive(
        benchmark.FORMAL_REFERENCE_ARCHIVE
    )
    specs = reproduction._build_specs(reference)
    events = np.zeros((1, len(specs), 4), dtype=np.int64)
    completed = np.zeros((1, len(specs)), dtype=np.int64)
    for spec in specs:
        dataset = reference["datasets"][spec["dataset"]]
        count = int(spec["trajectory_count"])
        events[0, spec["point_index"]] = np.rint(
            dataset["probabilities"][spec["impact_index"]] * count
        ).astype(np.int64)
        completed[0, spec["point_index"]] = count
    result = reproduction._cross_section_summary(
        reference, specs, events, completed
    )
    channels = {row["channel"]: row for row in result["rows"]}
    assert channels["SC"]["reference_cm2_per_h2o"] == pytest.approx(
        1.2399290372164355e-16,
        rel=1.0e-12,
    )
    assert result["maximum_relative_quadrature_change"] < 1.0e-6
    assert result["quadrature_0p5pct_gate_passed"]


def test_formal_probability_comparison_covers_all_eleven_curves(
    tmp_path: Path,
) -> None:
    reference = benchmark._read_formal_reference_archive(
        benchmark.FORMAL_REFERENCE_ARCHIVE
    )
    impact = np.linspace(0.0, 1.0, 5)
    pi = np.empty((len(impact), 5))
    pc = np.empty_like(pi)
    for orbital in range(5):
        dataset = reference["datasets"][f"L{orbital + 1}"]
        pi[:, orbital] = np.interp(
            impact, dataset["impact_au"], dataset["probabilities"][:, 0]
        )
        pc[:, orbital] = np.interp(
            impact, dataset["impact_au"], dataset["probabilities"][:, 1]
        )
    loss = reference["datasets"]["loss"]
    pl = np.interp(
        impact, loss["impact_au"], loss["probabilities"][:, 0]
    )
    archive = tmp_path / "carbon_charge_exchange_probabilities.npz"
    np.savez(
        archive,
        adaptive_format=np.asarray(True),
        curve_energy_keV_u=np.asarray([100.0]),
        curve_charge=np.asarray([3]),
        curve_offsets=np.asarray([0, len(impact)]),
        impact_au=impact,
        pi=pi,
        pc=pc,
        pl=pl,
        channel_successes=np.full((len(impact), 6), 10000),
    )
    comparison = benchmark._formal_probability_comparison(reference, archive)
    assert len(comparison["comparisons"]) == 11
    assert all(
        row["maximum_absolute_probability_difference"] == 0.0
        for row in comparison["comparisons"]
    )


def test_paper_curve_comparison_never_extrapolates_sparse_run() -> None:
    table = _uniform_rate_table(points=4)
    for column in table:
        if column == "E_keV_u":
            table[column] = np.repeat(
                np.asarray([10.0, 100.0, 500.0, 1000.0]), 7
            )
    order = np.lexsort((table["q"], table["E_keV_u"]))
    table = {name: values[order] for name, values in table.items()}
    comparison = benchmark._paper_curve_comparison(
        table,
        benchmark.equilibrium_charge_fractions(table),
        benchmark._read_paper_data(),
    )

    for figure in ("12", "13"):
        rows = comparison["figures"][figure]["comparison_points"]
        assert rows
        assert all(10.0 <= row["E_keV_u"] <= 1000.0 for row in rows)
        assert (
            comparison["figures"][figure][
                "excluded_paper_points_outside_resolved_curve_domain"
            ]
            > 0
        )


def test_basic_table_and_paired_boundary_checks_pass(tmp_path: Path) -> None:
    table = _uniform_rate_table()
    checks = []
    benchmark._basic_table_checks(table, checks)
    assert all(check["status"] == "pass" for check in checks)

    reference = tmp_path / "expanded.csv"
    _write_table(reference, table)
    benchmark._boundary_comparison(
        table,
        reference,
        checks,
        benchmark.NUMERICAL_RELATIVE_TOLERANCE,
    )
    assert checks[-1]["status"] == "pass"
    assert checks[-1]["observed"]["maximum_relative_difference"] == 0.0


def test_plot_is_png_only_and_equilibrium_csv_is_normalized(
    tmp_path: Path,
) -> None:
    table = _uniform_rate_table()
    equilibrium = benchmark.equilibrium_charge_fractions(table)
    plot_path = tmp_path / "benchmark.png"
    csv_path = tmp_path / "fractions.csv"
    benchmark.plot_benchmark(table, equilibrium, plot_path)
    benchmark._write_equilibrium_csv(csv_path, equilibrium)

    assert plot_path.read_bytes().startswith(b"\x89PNG")
    assert not list(tmp_path.glob("*.pdf"))
    values = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    np.testing.assert_allclose(np.sum(values[:, 1:8], axis=1), 1.0)


def test_adaptive_probability_archive_reconstructs_table(
    tmp_path: Path,
) -> None:
    energies = np.geomspace(1.0, 1.0e4, 41)
    impact = np.asarray([0.0, 1.0])
    table_rows = []
    curve_energy = []
    curve_charge = []
    offsets = [0]
    all_impact = []
    all_pi = []
    all_pc = []
    all_pl = []
    for energy in energies:
        for charge in range(7):
            pi = np.full((2, 5), 0.01)
            pc = np.full((2, 5), 0.02)
            pl = np.full(2, 0.03 if charge < 6 else 0.0)
            channels = benchmark.ctmc.many_electron_probabilities(
                pi, pc, pl, charge
            )
            cross_sections = {
                name: benchmark.ctmc.integrate_impact_parameter(impact, value)
                for name, value in channels.items()
            }
            table_rows.append(
                (
                    energy,
                    charge,
                    cross_sections["SC"],
                    cross_sections["TI"],
                    cross_sections["SL"],
                    cross_sections["LI"],
                    cross_sections["decrease"],
                    cross_sections["increase"],
                    cross_sections["SI"],
                    cross_sections["DI"],
                )
            )
            curve_energy.append(energy)
            curve_charge.append(charge)
            all_impact.append(impact)
            all_pi.append(pi)
            all_pc.append(pc)
            all_pl.append(pl)
            offsets.append(offsets[-1] + len(impact))
    matrix = np.asarray(table_rows)
    table = {
        name: matrix[:, index]
        for index, name in enumerate(benchmark.EXPECTED_COLUMNS)
    }
    archive_path = tmp_path / "carbon_charge_exchange_probabilities.npz"
    np.savez(
        archive_path,
        adaptive_format=np.asarray(True),
        curve_energy_keV_u=np.asarray(curve_energy),
        curve_charge=np.asarray(curve_charge),
        curve_offsets=np.asarray(offsets),
        impact_au=np.concatenate(all_impact),
        pi=np.concatenate(all_pi),
        pc=np.concatenate(all_pc),
        pl=np.concatenate(all_pl),
        failures=np.zeros(len(curve_energy) * len(impact), dtype=int),
    )
    metadata = {
        "base_ctmc_config": {
            "target_bmax_au": [1.0] * 5,
            "loss_bmax_au": [1.0] * 6,
        }
    }
    checks = []
    benchmark._probability_archive_check(
        table, archive_path, metadata, checks
    )
    consistency = next(
        check
        for check in checks
        if check["name"] == "probability_archive_consistency"
    )
    assert consistency["status"] == "pass"
    assert consistency["observed"]["maximum_table_reconstruction_relative_error"] < 1e-14
