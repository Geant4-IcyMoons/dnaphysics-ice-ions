#!/usr/bin/env python3
"""Benchmark the final carbon--H2O CTMC charge-exchange tables.

The three plotted quantities reproduce the definitions in figures 12--14 of
Liamsuwan and Nikjoo, Phys. Med. Biol. 58 (2013) 641--672:

* pure single electron capture, sigma_SC;
* pure single projectile electron loss, sigma_SL; and
* equilibrium C0--C6+ fractions obtained from all adjacent charge-changing
  channels, sigma_SC + sigma_TI and sigma_SL + sigma_LI (paper equation 23).

The paper supplies curves, not machine-readable numerical tables.  This
routine therefore does not fit or silently digitize those curves.  It checks
the numerical archive, emits the directly comparable plots and reports the
quantitative landmarks stated in the paper.  A separate expanded-domain run
may be supplied to make the impact-boundary convergence gate quantitative.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PHYSICS_ICE_ROOT = HERE.parents[2]
sys.path.insert(0, str(PHYSICS_ICE_ROOT))
DEFAULT_RUN_ROOT = HERE / "runs"

import charge_exchange_ctmc as ctmc  # noqa: E402
from constants import (  # noqa: E402
    AASTEX_FULL_WIDTH_IN,
    CARBON_CHARGE_EXCHANGE_DIR,
    PAPER_FONTSIZE,
    RC_BASE_STANDARD,
    THREE_PANEL_ROW_HEIGHT_IN,
    rcparams_with_fontsize,
)


PAPER_DOI = "10.1088/0031-9155/58/3/641"
PAPER_DATA_PATH = HERE / "paper_data" / "liamsuwan_2013_figures_12_14_digitized.csv"
PAPER_PDF_NAME = "Liamsuwan_2013_Phys._Med._Biol._58_641.pdf"
EXPECTED_COLUMNS = (
    "E_keV_u",
    "q",
    "sigma_SC_cm2",
    "sigma_TI_cm2",
    "sigma_SL_cm2",
    "sigma_LI_cm2",
    "sigma_decrease_cm2",
    "sigma_increase_cm2",
    "sigma_SI_cm2",
    "sigma_DI_cm2",
)
FIGURE12_SCALE = {1: 1.0e-4, 2: 1.0e-3, 3: 1.0e-2, 4: 1.0e-1, 5: 1.0, 6: 1.0e1}
NUMERICAL_RELATIVE_TOLERANCE = 5.0e-3


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_table(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header_line = next((line for line in lines if line.strip()), "")
    if path.suffix.lower() == ".csv":
        names = tuple(part.strip() for part in header_line.split(","))
        values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    else:
        names = tuple(header_line.lstrip("# ").split())
        values = np.loadtxt(path, comments="#", ndmin=2)
    if names != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns in {path}: {names}; expected {EXPECTED_COLUMNS}"
        )
    if values.shape[1] != len(names):
        raise ValueError(f"Malformed CTMC table {path}: shape={values.shape}")
    return {name: values[:, index] for index, name in enumerate(names)}


def _resolve_products(input_path: Path) -> tuple[Path, Path, Path]:
    if input_path.is_dir():
        return (
            input_path / "carbon_charge_exchange_h2o.csv",
            input_path / "carbon_charge_exchange_probabilities.npz",
            input_path / "carbon_charge_exchange_metadata.json",
        )
    stem = input_path.parent
    return (
        input_path,
        stem / "carbon_charge_exchange_probabilities.npz",
        stem / "carbon_charge_exchange_metadata.json",
    )


def _charge_curve(
    table: dict[str, np.ndarray], charge: int, column: str
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.rint(table["q"]).astype(int) == charge
    energy = np.asarray(table["E_keV_u"][mask], dtype=float)
    values = np.asarray(table[column][mask], dtype=float)
    order = np.argsort(energy)
    return energy[order], values[order]


def equilibrium_charge_fractions(
    table: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Solve paper equation (23) at energies calculated for every charge."""
    common_energy: set[float] | None = None
    for charge in range(7):
        charge_energy, _ = _charge_curve(
            table, charge, "sigma_decrease_cm2"
        )
        values = set(float(value) for value in charge_energy)
        common_energy = values if common_energy is None else common_energy & values
    energy = np.asarray(sorted(common_energy or ()), dtype=float)
    if len(energy) < 2:
        raise ValueError(
            "The seven carbon charge grids have fewer than two common "
            "explicitly calculated energies"
        )

    capture = np.zeros((len(energy), 7), dtype=float)
    loss = np.zeros_like(capture)
    for energy_index, value in enumerate(energy):
        for charge in range(1, 7):
            row = _find_table_row(table, float(value), charge)
            capture[energy_index, charge] = table["sigma_decrease_cm2"][row]
        for charge in range(6):
            row = _find_table_row(table, float(value), charge)
            loss[energy_index, charge] = table["sigma_increase_cm2"][row]
    resolved = np.all(capture[:, 1:] > 0.0, axis=1) & np.all(
        loss[:, :6] > 0.0, axis=1
    )
    excluded_energy = energy[~resolved]
    if np.count_nonzero(resolved) < 2:
        raise ValueError(
            "Fewer than two energies have positive cross sections for every "
            "adjacent charge transition; equation (23) is unresolved"
        )
    energy = energy[resolved]
    capture = capture[resolved]
    loss = loss[resolved]

    fractions = np.empty_like(capture)
    relative_residual = np.empty(len(energy), dtype=float)
    absolute_residual = np.empty(len(energy), dtype=float)
    for energy_index in range(len(energy)):
        log_fraction = np.zeros(7, dtype=float)
        for charge in range(1, 7):
            log_fraction[charge] = (
                log_fraction[charge - 1]
                + math.log(loss[energy_index, charge - 1])
                - math.log(capture[energy_index, charge])
            )
        log_fraction -= np.max(log_fraction)
        fraction = np.exp(log_fraction)
        fraction /= np.sum(fraction)
        fractions[energy_index] = fraction

        residual = np.empty(7, dtype=float)
        for charge in range(7):
            incoming = 0.0
            if charge > 0:
                incoming += fraction[charge - 1] * loss[energy_index, charge - 1]
            if charge < 6:
                incoming += fraction[charge + 1] * capture[energy_index, charge + 1]
            outgoing = fraction[charge] * (
                capture[energy_index, charge] + loss[energy_index, charge]
            )
            residual[charge] = incoming - outgoing
        absolute_residual[energy_index] = float(np.max(np.abs(residual)))
        flow_scale = float(
            np.max(fraction * (capture[energy_index] + loss[energy_index]))
        )
        relative_residual[energy_index] = (
            absolute_residual[energy_index] / flow_scale
            if flow_scale > 0.0
            else math.inf
        )

    return {
        "E_keV_u": energy,
        "fractions": fractions,
        "mean_charge": fractions @ np.arange(7, dtype=float),
        "capture_cm2": capture,
        "loss_cm2": loss,
        "absolute_stationarity_residual_cm2": absolute_residual,
        "relative_stationarity_residual": relative_residual,
        "excluded_energy_keV_u": excluded_energy,
    }


def _append_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    status: str,
    required_for_release: bool,
    criterion: str,
    observed: Any,
) -> None:
    if status not in {"pass", "fail", "not_evaluated", "informational"}:
        raise ValueError(f"Invalid check status {status}")
    checks.append(
        {
            "name": name,
            "status": status,
            "required_for_release": required_for_release,
            "criterion": criterion,
            "observed": observed,
        }
    )


def _active_columns(charge: int) -> tuple[str, ...]:
    columns = ["sigma_SC_cm2", "sigma_TI_cm2", "sigma_SI_cm2", "sigma_DI_cm2"]
    if charge == 0:
        columns = [name for name in columns if name not in {"sigma_SC_cm2", "sigma_TI_cm2"}]
    if charge < 6:
        columns.extend(("sigma_SL_cm2", "sigma_LI_cm2"))
    return tuple(columns)


def _basic_table_checks(
    table: dict[str, np.ndarray], checks: list[dict[str, Any]]
) -> None:
    matrix = np.column_stack([table[name] for name in EXPECTED_COLUMNS])
    _append_check(
        checks,
        name="finite_nonnegative_table",
        status=(
            "pass"
            if np.all(np.isfinite(matrix))
            and np.all(matrix[:, 0] > 0.0)
            and np.all(matrix[:, 1:] >= 0.0)
            and np.all(matrix[:, 1] == np.rint(matrix[:, 1]))
            else "fail"
        ),
        required_for_release=True,
        criterion="All energies are positive; charges and cross sections are finite and non-negative.",
        observed={"row_count": int(len(matrix))},
    )
    charges = np.unique(np.rint(table["q"]).astype(int))
    grid_ok = np.array_equal(charges, np.arange(7))
    grid_summary: dict[str, Any] = {}
    for charge in charges:
        energy, _ = _charge_curve(table, int(charge), "sigma_SC_cm2")
        grid_summary[str(int(charge))] = {
            "count": int(len(energy)),
            "minimum_keV_u": float(energy[0]),
            "maximum_keV_u": float(energy[-1]),
            "strictly_increasing": bool(np.all(np.diff(energy) > 0.0)),
        }
        grid_ok &= (
            np.all(np.diff(energy) > 0.0)
            and energy[0] <= 1.0
            and energy[-1] >= 1.0e4
        )
    _append_check(
        checks,
        name="published_carbon_domain",
        status="pass" if grid_ok else "fail",
        required_for_release=True,
        criterion="Every charge q=0,...,6 covers 1--10000 keV/u on a unique increasing grid.",
        observed=grid_summary,
    )

    decrease_error = np.max(
        np.abs(
            table["sigma_decrease_cm2"]
            - table["sigma_SC_cm2"]
            - table["sigma_TI_cm2"]
        )
    )
    increase_error = np.max(
        np.abs(
            table["sigma_increase_cm2"]
            - table["sigma_SL_cm2"]
            - table["sigma_LI_cm2"]
        )
    )
    identity_scale = max(
        float(np.max(table["sigma_decrease_cm2"])),
        float(np.max(table["sigma_increase_cm2"])),
        np.finfo(float).tiny,
    )
    identity_error = max(float(decrease_error), float(increase_error)) / identity_scale
    _append_check(
        checks,
        name="published_channel_identities",
        status="pass" if identity_error <= 5.0e-9 else "fail",
        required_for_release=True,
        criterion="decrease=SC+TI and increase=SL+LI to table-write precision.",
        observed={"maximum_relative_error": identity_error},
    )

    q = np.rint(table["q"]).astype(int)
    forbidden_maximum = (
        max(
            float(np.max(table["sigma_decrease_cm2"][q == 0])),
            float(np.max(table["sigma_increase_cm2"][q == 6])),
        )
        if np.any(q == 0) and np.any(q == 6)
        else math.inf
    )
    _append_check(
        checks,
        name="terminal_charge_states",
        status="pass" if forbidden_maximum == 0.0 else "fail",
        required_for_release=True,
        criterion="C0 has no q->-1 channel and C6+ has no q->7 channel.",
        observed={"maximum_forbidden_cross_section_cm2": forbidden_maximum},
    )


def _metadata_checks(
    metadata_path: Path,
    probabilities_path: Path,
    checks: list[dict[str, Any]],
    *,
    require_adaptive: bool,
) -> dict[str, Any] | None:
    if not metadata_path.is_file():
        _append_check(
            checks,
            name="adaptive_numerical_convergence",
            status="not_evaluated",
            required_for_release=require_adaptive,
            criterion="Final metadata must document the adaptive <=0.5% numerical and statistical gates.",
            observed={"missing": str(metadata_path)},
        )
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    adaptive = metadata.get("adaptive_refinement", {})
    combined = float(adaptive.get("combined_discretization_error_bound", math.inf))
    statistical = float(
        adaptive.get("maximum_statistical_relative_confidence_half_width", math.inf)
    )
    adaptive_ok = (
        adaptive.get("enabled") is True
        and combined <= NUMERICAL_RELATIVE_TOLERANCE
        and statistical <= NUMERICAL_RELATIVE_TOLERANCE
    )
    _append_check(
        checks,
        name="adaptive_numerical_convergence",
        status=(
            "pass"
            if adaptive_ok
            else "fail"
            if require_adaptive
            else "informational"
        ),
        required_for_release=require_adaptive,
        criterion=(
            "Combined impact/energy error <=0.5% and separate reported Monte "
            "Carlo 95% relative half-width <=0.5%. For an explicitly requested "
            "fixed-grid reference this gate is reported but is not applicable."
        ),
        observed={
            "enabled": adaptive.get("enabled"),
            "combined_discretization_error_bound": combined,
            "maximum_statistical_relative_confidence_half_width": statistical,
        },
    )
    trajectory = metadata.get("trajectory_statistics", {})
    failed = int(trajectory.get("failed", -1))
    drift = float(trajectory.get("maximum_relative_total_energy_drift", math.inf))
    config = metadata.get("base_ctmc_config", metadata.get("ctmc_config", {}))
    drift_limit = float(config.get("maximum_relative_energy_drift", -math.inf))
    trajectory_ok = failed == 0 and drift_limit > 0.0 and drift <= drift_limit
    _append_check(
        checks,
        name="trajectory_acceptance",
        status="pass" if trajectory_ok else "fail",
        required_for_release=True,
        criterion="No failed trajectories and maximum energy drift within the configured acceptance limit.",
        observed={
            "failed": failed,
            "maximum_relative_total_energy_drift": drift,
            "configured_limit": drift_limit,
        },
    )
    provenance_ok = (
        metadata.get("doi") == PAPER_DOI
        and metadata.get("projectile", {}).get("nuclear_charge") == 6
        and metadata.get("density_applied") is False
        and metadata.get("cross_section_unit") == "cm2 per H2O molecule"
    )
    _append_check(
        checks,
        name="model_and_units_provenance",
        status="pass" if provenance_ok else "fail",
        required_for_release=True,
        criterion="Carbon model DOI, Z=6, microscopic H2O units, and unapplied phase density are explicit.",
        observed={
            "doi": metadata.get("doi"),
            "projectile": metadata.get("projectile"),
            "density_applied": metadata.get("density_applied"),
            "cross_section_unit": metadata.get("cross_section_unit"),
        },
    )
    if not probabilities_path.is_file():
        _append_check(
            checks,
            name="probability_archive_consistency",
            status="not_evaluated",
            required_for_release=True,
            criterion="The final probability archive must reproduce every reported table row.",
            observed={"missing": str(probabilities_path)},
        )
    return metadata


def _find_table_row(
    table: dict[str, np.ndarray], energy: float, charge: int
) -> int:
    q = np.rint(table["q"]).astype(int)
    candidates = np.flatnonzero(q == charge)
    if not len(candidates):
        raise ValueError(f"Missing q={charge} table rows")
    difference = np.abs(table["E_keV_u"][candidates] - energy)
    index = int(candidates[int(np.argmin(difference))])
    if not math.isclose(
        float(table["E_keV_u"][index]), energy, rel_tol=5.0e-9, abs_tol=0.0
    ):
        raise ValueError(f"No table row matches q={charge}, E={energy:.17g}")
    return index


def _probability_archive_check(
    table: dict[str, np.ndarray],
    probabilities_path: Path,
    metadata: dict[str, Any] | None,
    checks: list[dict[str, Any]],
) -> None:
    if not probabilities_path.is_file():
        return
    maximum_relative_error = 0.0
    maximum_primitive_violation = 0.0
    boundary_probabilities = {"target_ionization": 0.0, "target_capture": 0.0, "projectile_loss": 0.0}
    with np.load(probabilities_path, allow_pickle=False) as archive:
        adaptive_format = bool(np.asarray(archive.get("adaptive_format", False)).item())
        if not adaptive_format:
            rows, _ = ctmc.build_cross_section_rows(
                np.asarray(archive["energies_keV_u"]),
                np.asarray(archive["charges"]),
                np.asarray(archive["impact_au"]),
                np.asarray(archive["pi"]),
                np.asarray(archive["pc"]),
                np.asarray(archive["pl"]),
            )
            for row in rows:
                table_index = _find_table_row(table, float(row[0]), int(row[1]))
                reference = np.asarray([table[name][table_index] for name in EXPECTED_COLUMNS])
                scale = np.maximum(np.abs(reference[2:]), np.finfo(float).tiny)
                maximum_relative_error = max(
                    maximum_relative_error,
                    float(np.max(np.abs(row[2:] - reference[2:]) / scale)),
                )
        else:
            curve_energy = np.asarray(archive["curve_energy_keV_u"], dtype=float)
            curve_charge = np.asarray(archive["curve_charge"], dtype=int)
            offsets = np.asarray(archive["curve_offsets"], dtype=np.int64)
            impact_all = np.asarray(archive["impact_au"], dtype=float)
            pi_all = np.asarray(archive["pi"], dtype=float)
            pc_all = np.asarray(archive["pc"], dtype=float)
            pl_all = np.asarray(archive["pl"], dtype=float)
            if not (
                np.all(np.isfinite(pi_all))
                and np.all(np.isfinite(pc_all))
                and np.all(np.isfinite(pl_all))
            ):
                maximum_primitive_violation = math.inf
            else:
                maximum_primitive_violation = max(
                    float(np.max(np.maximum(-pi_all, 0.0))),
                    float(np.max(np.maximum(-pc_all, 0.0))),
                    float(np.max(np.maximum(pi_all + pc_all - 1.0, 0.0))),
                    float(np.max(np.maximum(-pl_all, 0.0))),
                    float(np.max(np.maximum(pl_all - 1.0, 0.0))),
                )
            config = (metadata or {}).get("base_ctmc_config", {})
            target_cutoffs = tuple(float(value) for value in config.get("target_bmax_au", ()))
            loss_cutoffs = tuple(float(value) for value in config.get("loss_bmax_au", ()))
            curves = zip(curve_energy, curve_charge, strict=True)
            for curve_index, (energy, charge) in tqdm(
                enumerate(curves),
                total=len(curve_energy),
                desc="Reintegrating CTMC archive",
                unit="curve",
                dynamic_ncols=True,
            ):
                start, stop = int(offsets[curve_index]), int(offsets[curve_index + 1])
                impact = impact_all[start:stop]
                pi = pi_all[start:stop]
                pc = pc_all[start:stop]
                pl = pl_all[start:stop]
                channels = ctmc.many_electron_probabilities(pi, pc, pl, int(charge))
                reconstructed = np.asarray(
                    [
                        energy,
                        float(charge),
                        ctmc.integrate_impact_parameter(impact, channels["SC"]),
                        ctmc.integrate_impact_parameter(impact, channels["TI"]),
                        ctmc.integrate_impact_parameter(impact, channels["SL"]),
                        ctmc.integrate_impact_parameter(impact, channels["LI"]),
                        ctmc.integrate_impact_parameter(impact, channels["decrease"]),
                        ctmc.integrate_impact_parameter(impact, channels["increase"]),
                        ctmc.integrate_impact_parameter(impact, channels["SI"]),
                        ctmc.integrate_impact_parameter(impact, channels["DI"]),
                    ]
                )
                table_index = _find_table_row(table, float(energy), int(charge))
                reference = np.asarray([table[name][table_index] for name in EXPECTED_COLUMNS])
                active = [EXPECTED_COLUMNS.index(name) for name in _active_columns(int(charge))]
                scale = np.maximum(np.abs(reference[active]), np.finfo(float).tiny)
                maximum_relative_error = max(
                    maximum_relative_error,
                    float(np.max(np.abs(reconstructed[active] - reference[active]) / scale)),
                )
                if len(target_cutoffs) == pi.shape[1]:
                    for orbital_index, cutoff in enumerate(target_cutoffs):
                        cutoff_index = int(np.argmin(np.abs(impact - cutoff)))
                        if math.isclose(float(impact[cutoff_index]), cutoff, rel_tol=1.0e-12, abs_tol=1.0e-12):
                            boundary_probabilities["target_ionization"] = max(
                                boundary_probabilities["target_ionization"],
                                float(pi[cutoff_index, orbital_index]),
                            )
                            boundary_probabilities["target_capture"] = max(
                                boundary_probabilities["target_capture"],
                                float(pc[cutoff_index, orbital_index]),
                            )
                if int(charge) < 6 and len(loss_cutoffs) == 6:
                    cutoff = loss_cutoffs[int(charge)]
                    cutoff_index = int(np.argmin(np.abs(impact - cutoff)))
                    if math.isclose(float(impact[cutoff_index]), cutoff, rel_tol=1.0e-12, abs_tol=1.0e-12):
                        boundary_probabilities["projectile_loss"] = max(
                            boundary_probabilities["projectile_loss"],
                            float(pl[cutoff_index]),
                        )
        failed = int(np.sum(np.asarray(archive["failures"])))
    archive_ok = (
        maximum_relative_error <= 5.0e-8
        and maximum_primitive_violation <= 1.0e-12
        and failed == 0
    )
    _append_check(
        checks,
        name="probability_archive_consistency",
        status="pass" if archive_ok else "fail",
        required_for_release=True,
        criterion="Primitive probabilities are physical, failures are zero, and reintegration reproduces the rounded table.",
        observed={
            "adaptive_format": adaptive_format,
            "failed_trajectories": failed,
            "maximum_primitive_probability_violation": maximum_primitive_violation,
            "maximum_table_reconstruction_relative_error": maximum_relative_error,
        },
    )
    _append_check(
        checks,
        name="cutoff_probability_diagnostic",
        status="informational",
        required_for_release=False,
        criterion="Probabilities at each explicitly supplied impact cutoff are reported; no unpublished probability threshold is imposed.",
        observed=boundary_probabilities,
    )


def _boundary_comparison(
    table: dict[str, np.ndarray],
    reference_path: Path | None,
    checks: list[dict[str, Any]],
    tolerance: float,
) -> None:
    if reference_path is None:
        _append_check(
            checks,
            name="paired_impact_boundary_convergence",
            status="not_evaluated",
            required_for_release=True,
            criterion=f"An independently calculated expanded-domain table changes every active base-grid cross section by <={tolerance:.3%}.",
            observed="No expanded-domain reference table supplied.",
        )
        return
    reference = _read_table(reference_path)
    differences: list[float] = []
    common_per_charge: dict[str, int] = {}
    for charge in range(7):
        current_energy, _ = _charge_curve(table, charge, "sigma_SC_cm2")
        reference_energy, _ = _charge_curve(reference, charge, "sigma_SC_cm2")
        common = []
        for energy in current_energy:
            if np.any(np.isclose(reference_energy, energy, rtol=5.0e-9, atol=0.0)):
                common.append(float(energy))
        common_per_charge[str(charge)] = len(common)
        for energy in common:
            current_index = _find_table_row(table, energy, charge)
            reference_index = _find_table_row(reference, energy, charge)
            for column in _active_columns(charge):
                current_value = float(table[column][current_index])
                reference_value = float(reference[column][reference_index])
                if reference_value == 0.0:
                    differences.append(0.0 if current_value == 0.0 else math.inf)
                else:
                    differences.append(abs(current_value - reference_value) / abs(reference_value))
    coverage_ok = all(count >= 41 for count in common_per_charge.values())
    maximum = max(differences, default=math.inf)
    _append_check(
        checks,
        name="paired_impact_boundary_convergence",
        status="pass" if coverage_ok and maximum <= tolerance else "fail",
        required_for_release=True,
        criterion=f"Expanded-domain rerun agrees on all 41 published base energies for every charge within {tolerance:.3%}.",
        observed={
            "reference": str(reference_path),
            "common_energy_count_per_charge": common_per_charge,
            "maximum_relative_difference": maximum,
        },
    )


def _paper_landmarks(
    equilibrium: dict[str, np.ndarray], table: dict[str, np.ndarray]
) -> dict[str, Any]:
    fractions = equilibrium["fractions"]
    energy = equilibrium["E_keV_u"]
    maxima = np.max(fractions, axis=0)
    peak_indices = np.argmax(fractions, axis=0)
    f6_at_2mev = float(
        np.interp(math.log(2000.0), np.log(energy), fractions[:, 6])
    )
    threshold_ratios: dict[str, float] = {}
    for charge, threshold in ((2, 10.0), (3, 10.0), (4, 100.0), (5, 100.0)):
        curve_energy, curve = _charge_curve(table, charge, "sigma_SL_cm2")
        below = curve[curve_energy < threshold]
        threshold_ratios[str(charge)] = (
            float(np.max(below) / np.max(curve)) if len(below) and np.max(curve) > 0.0 else 0.0
        )
    return {
        "source": {
            "doi": PAPER_DOI,
            "figures": [12, 13, 14],
            "note": (
                "The article provides plotted curves but no numerical table. "
                "These are comparisons with quantitative statements in the text, "
                "not fitted acceptance tolerances."
            ),
        },
        "figure_14": {
            "maximum_fraction_by_charge": {
                str(charge): float(maxima[charge]) for charge in range(7)
            },
            "peak_energy_keV_u_by_charge": {
                str(charge): float(energy[peak_indices[charge]]) for charge in range(7)
            },
            "paper_reports_C4_maximum_approximately": 0.78,
            "paper_reports_C2_C3_C5_maximum_range": [0.37, 0.48],
            "C2_C3_C5_within_reported_range": {
                str(charge): bool(0.37 <= maxima[charge] <= 0.48)
                for charge in (2, 3, 5)
            },
            "C6_fraction_at_2000_keV_u": f6_at_2mev,
            "paper_qualitative_statement": "Carbon is fully stripped above approximately 2 MeV/u.",
        },
        "figure_13": {
            "maximum_below_reported_threshold_divided_by_curve_maximum": threshold_ratios,
            "paper_qualitative_statement": (
                "C2+/C3+ loss is negligible below 10 keV/u and C4+/C5+ "
                "loss is negligible below 100 keV/u."
            ),
        },
    }


def _write_equilibrium_csv(path: Path, equilibrium: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["E_keV_u"]
            + [f"fraction_q{charge}" for charge in range(7)]
            + ["mean_charge", "relative_stationarity_residual"]
        )
        for index, energy in enumerate(equilibrium["E_keV_u"]):
            writer.writerow(
                [f"{energy:.12e}"]
                + [f"{value:.12e}" for value in equilibrium["fractions"][index]]
                + [
                    f"{equilibrium['mean_charge'][index]:.12e}",
                    f"{equilibrium['relative_stationarity_residual'][index]:.12e}",
                ]
            )
    os.replace(temporary, path)


def _configure_plot_style() -> None:
    # Matplotlib cannot read the installed Type-1 Courier faces through its
    # TrueType font manager.  The PDF backend can use Adobe Courier directly
    # as one of the standard PDF fonts; `_save_courier_png` rasterizes that
    # transient PDF and retains only the requested PNG product.
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.monospace"] = ["Courier"]
    plt.rcParams["pdf.use14corefonts"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update(
        rcparams_with_fontsize(
            RC_BASE_STANDARD,
            PAPER_FONTSIZE,
            overrides={
                "axes.linewidth": 0.8,
                "axes.labelweight": "normal",
                "axes.titleweight": "normal",
                "font.weight": "normal",
                "lines.linewidth": 1.0,
                "savefig.dpi": 300,
                "xtick.major.size": 4.0,
                "xtick.major.width": 0.8,
                "xtick.minor.size": 2.0,
                "xtick.minor.width": 0.6,
                "ytick.major.size": 4.0,
                "ytick.major.width": 0.8,
                "ytick.minor.size": 2.0,
                "ytick.minor.width": 0.6,
            },
        )
    )


CARBON_MASS_NUMBER = 12.0
PLOT_ENERGY_MIN_EV = 1.0e4
PLOT_ENERGY_MAX_EV = 1.0e8
# Keep artist-level requests generic.  With ``pdf.use14corefonts`` and the
# rcParams above, the PDF backend maps this family to the standard Adobe
# Courier face.  Asking Matplotlib's TrueType manager for "Courier" directly
# would emit a misleading fallback warning even though the PDF is correct.
PLOT_FONT = "monospace"


def _carbon_total_energy_ev(energy_keV_u: np.ndarray) -> np.ndarray:
    """Convert energy per nucleon to total C-12 kinetic energy."""
    return np.asarray(energy_keV_u, dtype=float) * CARBON_MASS_NUMBER * 1.0e3


def _read_paper_data(path: Path = PAPER_DATA_PATH) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = data.reshape(1)
    required = {
        "figure",
        "observable",
        "q",
        "E_keV_u",
        "pixel_x",
        "pixel_y",
        "value",
        "digitization_uncertainty",
    }
    if not data.dtype.names or required.difference(data.dtype.names):
        raise ValueError(f"Malformed paper digitization table: {path}")
    return data


def _default_paper_pdf() -> Path | None:
    candidates = (
        PHYSICS_ICE_ROOT.parents[1] / "literature" / PAPER_PDF_NAME,
        Path.home() / "work" / "dnaphysics-ice-ions" / "literature" / PAPER_PDF_NAME,
    )
    return next((path for path in candidates if path.is_file()), None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _paper_subset(data: np.ndarray, figure: int, charge: int) -> np.ndarray:
    subset = data[(data["figure"] == figure) & (data["q"] == charge)]
    return np.sort(subset, order="E_keV_u")


def _log_interpolate(
    energy: np.ndarray, values: np.ndarray, sample_energy: np.ndarray
) -> np.ndarray:
    positive = np.isfinite(values) & (values > 0.0)
    if np.count_nonzero(positive) < 2:
        return np.full_like(sample_energy, np.nan, dtype=float)
    return np.exp(
        np.interp(
            np.log(sample_energy),
            np.log(energy[positive]),
            np.log(values[positive]),
        )
    )


def _inside_positive_curve_domain(
    published: np.ndarray,
    energy: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Retain paper points that can be interpolated without extrapolation."""
    positive = np.isfinite(values) & (values > 0.0)
    if np.count_nonzero(positive) < 2:
        return published[:0]
    minimum = float(np.min(energy[positive]))
    maximum = float(np.max(energy[positive]))
    return published[
        (published["E_keV_u"] >= minimum)
        & (published["E_keV_u"] <= maximum)
    ]


def _paper_curve_comparison(
    table: dict[str, np.ndarray],
    equilibrium: dict[str, np.ndarray],
    paper_data: np.ndarray,
) -> dict[str, Any]:
    figures: dict[str, Any] = {}
    for figure, column, charges, scale_capture in (
        (12, "sigma_SC_cm2", range(1, 7), True),
        (13, "sigma_SL_cm2", range(6), False),
    ):
        rows = []
        for charge in charges:
            published = _paper_subset(paper_data, figure, charge)
            if len(published) == 0:
                continue
            energy, values = _charge_curve(table, charge, column)
            if scale_capture:
                values = values * FIGURE12_SCALE[charge]
            inside = _inside_positive_curve_domain(published, energy, values)
            calculated = _log_interpolate(energy, values, inside["E_keV_u"])
            ratio = calculated / inside["value"]
            for source, predicted, value_ratio in zip(
                inside, calculated, ratio, strict=True
            ):
                rows.append(
                    {
                        "q": int(charge),
                        "E_keV_u": float(source["E_keV_u"]),
                        "paper_value": float(source["value"]),
                        "calculated_value": float(predicted),
                        "calculated_over_paper": float(value_ratio),
                        "paper_digitization_fractional_uncertainty": float(
                            source["digitization_uncertainty"]
                        ),
                    }
                )
        ratios = np.asarray([row["calculated_over_paper"] for row in rows])
        figures[str(figure)] = {
            "comparison_points": rows,
            "excluded_paper_points_outside_resolved_curve_domain": int(
                sum(
                    len(_paper_subset(paper_data, figure, charge))
                    for charge in charges
                )
                - len(rows)
            ),
            "median_absolute_log10_ratio": float(
                np.median(np.abs(np.log10(ratios)))
            ) if len(ratios) else None,
            "maximum_factor_difference": float(
                np.max(np.maximum(ratios, 1.0 / ratios))
            ) if len(ratios) else None,
        }

    rows = []
    energy = np.asarray(equilibrium["E_keV_u"], dtype=float)
    for charge in range(7):
        published = _paper_subset(paper_data, 14, charge)
        inside = published[
            (published["E_keV_u"] >= np.min(energy))
            & (published["E_keV_u"] <= np.max(energy))
        ]
        if len(inside) == 0:
            continue
        calculated = np.interp(
            np.log(inside["E_keV_u"]),
            np.log(energy),
            equilibrium["fractions"][:, charge],
        )
        residual = calculated - inside["value"]
        for source, predicted, difference in zip(
            inside, calculated, residual, strict=True
        ):
            rows.append(
                {
                    "q": int(charge),
                    "E_keV_u": float(source["E_keV_u"]),
                    "paper_fraction": float(source["value"]),
                    "calculated_fraction": float(predicted),
                    "absolute_residual": float(difference),
                    "paper_digitization_absolute_uncertainty": float(
                        source["digitization_uncertainty"]
                    ),
                }
            )
    residuals = np.asarray([row["absolute_residual"] for row in rows])
    figures["14"] = {
        "comparison_points": rows,
        "excluded_paper_points_outside_resolved_equilibrium_domain": int(
            np.count_nonzero(paper_data["figure"] == 14) - len(rows)
        ),
        "median_absolute_fraction_residual": float(np.median(np.abs(residuals))),
        "maximum_absolute_fraction_residual": float(np.max(np.abs(residuals))),
    }
    return {
        "status": "informational",
        "reason": (
            "The paper supplies raster curves rather than numerical tables; "
            "digitization error is separate from CTMC numerical uncertainty."
        ),
        "digitized_data": str(PAPER_DATA_PATH.resolve()),
        "figures": figures,
    }


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.0,
        1.025,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=PAPER_FONTSIZE,
        fontfamily=PLOT_FONT,
        fontweight="normal",
        clip_on=False,
        zorder=30,
    )


def _save_courier_png(figure: plt.Figure, output_path: Path) -> None:
    """Write a PNG whose ordinary text uses the standard Adobe Courier font."""
    ghostscript = shutil.which("gs")
    if ghostscript is None:
        raise RuntimeError("Ghostscript is required to render Courier PNG figures")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".carbon-ctmc-plot-", dir=output_path.parent
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        pdf_path = temporary_root / "figure.pdf"
        png_path = temporary_root / "figure.png"
        figure.savefig(pdf_path, facecolor="white", format="pdf")
        subprocess.run(
            [
                ghostscript,
                "-q",
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=pngalpha",
                "-r300",
                f"-sOutputFile={png_path}",
                str(pdf_path),
            ],
            check=True,
        )
        os.replace(png_path, output_path)


def plot_benchmark(
    table: dict[str, np.ndarray],
    equilibrium: dict[str, np.ndarray],
    output_path: Path,
    paper_data: np.ndarray | None = None,
) -> None:
    _configure_plot_style()
    if paper_data is None:
        paper_data = _read_paper_data()
    colors = plt.get_cmap("plasma")(np.linspace(0.08, 0.92, 7))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(AASTEX_FULL_WIDTH_IN, THREE_PANEL_ROW_HEIGHT_IN),
    )
    ax_capture, ax_loss, ax_fraction = axes

    for charge in range(1, 7):
        energy, cross_section = _charge_curve(table, charge, "sigma_SC_cm2")
        plotted = cross_section * FIGURE12_SCALE[charge]
        ax_capture.loglog(
            _carbon_total_energy_ev(energy),
            np.where(plotted > 0.0, plotted, np.nan),
            color=colors[charge],
        )
        published = _paper_subset(paper_data, 12, charge)
        ax_capture.plot(
            _carbon_total_energy_ev(published["E_keV_u"]),
            published["value"],
            color=colors[charge],
            linestyle="--",
            linewidth=0.65,
            marker="o",
            markersize=2.6,
            markerfacecolor="white",
            markeredgewidth=0.55,
        )
    for charge in range(6):
        energy, cross_section = _charge_curve(table, charge, "sigma_SL_cm2")
        ax_loss.loglog(
            _carbon_total_energy_ev(energy),
            np.where(cross_section > 0.0, cross_section, np.nan),
            color=colors[charge],
        )
        published = _paper_subset(paper_data, 13, charge)
        ax_loss.plot(
            _carbon_total_energy_ev(published["E_keV_u"]),
            published["value"],
            color=colors[charge],
            linestyle="--",
            linewidth=0.65,
            marker="o",
            markersize=2.6,
            markerfacecolor="white",
            markeredgewidth=0.55,
        )

    for charge in range(7):
        ax_fraction.semilogx(
            _carbon_total_energy_ev(equilibrium["E_keV_u"]),
            equilibrium["fractions"][:, charge],
            color=colors[charge],
            label=(
                "C0"
                if charge == 0
                else ("C+" if charge == 1 else f"C{charge}+")
            ),
        )
        published = _paper_subset(paper_data, 14, charge)
        ax_fraction.plot(
            _carbon_total_energy_ev(published["E_keV_u"]),
            published["value"],
            color=colors[charge],
            linestyle="--",
            linewidth=0.65,
            marker="o",
            markersize=2.4,
            markerfacecolor="white",
            markeredgewidth=0.5,
        )

    ax_capture.set_title("Pure capture", pad=4)
    ax_loss.set_title("Pure loss", pad=4)
    ax_fraction.set_title("Equilibrium", pad=4)
    ax_capture.set_ylabel(r"$10^{q-5}\sigma_{\rm SC}$ (cm$^2$)")
    ax_loss.set_ylabel(r"$\sigma_{\rm SL}$ (cm$^2$)")
    ax_fraction.set_ylabel("Charge-state fraction")
    for axis, label in zip(axes, ("(a)", "(b)", "(c)"), strict=True):
        axis.set_xlabel(r"Kinetic energy ($T$; eV)")
        axis.set_xlim(PLOT_ENERGY_MIN_EV, PLOT_ENERGY_MAX_EV)
        axis.set_xticks(
            [1.0e4, 1.0e6, 1.0e8],
            ["10 keV", "1 MeV", "100 MeV"],
        )
        axis.get_xticklabels()[0].set_ha("left")
        axis.get_xticklabels()[-1].set_ha("right")
        axis.set_facecolor("white")
        axis.minorticks_on()
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
        axis.tick_params(
            axis="both",
            which="both",
            direction="in",
            bottom=True,
            left=True,
            top=False,
            right=False,
            labelsize=PAPER_FONTSIZE,
        )
        axis.xaxis.label.set_fontfamily(PLOT_FONT)
        axis.xaxis.label.set_fontsize(PAPER_FONTSIZE)
        axis.yaxis.label.set_fontfamily(PLOT_FONT)
        axis.yaxis.label.set_fontsize(PAPER_FONTSIZE)
        for tick_label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            tick_label.set_fontfamily(PLOT_FONT)
            tick_label.set_fontsize(PAPER_FONTSIZE)
        _panel_label(axis, label)
        axis.grid(False)
    ax_fraction.set_ylim(0.0, 1.0)

    handles, labels = ax_fraction.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.058),
        prop={"family": PLOT_FONT, "size": PAPER_FONTSIZE},
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.3,
    )
    fig.legend(
        [
            Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                marker="o",
                markerfacecolor="white",
                markersize=2.8,
                linewidth=0.65,
            ),
        ],
        ["Liamsuwan & Nikjoo (2013), digitized"],
        loc="lower center",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.5, 0.006),
        prop={"family": PLOT_FONT, "size": PAPER_FONTSIZE},
        handlelength=1.5,
        columnspacing=1.5,
        handletextpad=0.4,
    )
    fig.subplots_adjust(
        left=0.086,
        right=0.992,
        bottom=0.34,
        top=0.91,
        wspace=0.48,
    )
    _save_courier_png(fig, output_path)
    plt.close(fig)


def benchmark(
    *,
    table_path: Path,
    probabilities_path: Path,
    metadata_path: Path,
    output_dir: Path,
    boundary_reference: Path | None = None,
    boundary_relative_tolerance: float = NUMERICAL_RELATIVE_TOLERANCE,
    require_adaptive: bool = True,
    paper_pdf: Path | None = None,
) -> dict[str, Any]:
    ctmc.select_projectile("carbon")
    table = _read_table(table_path)
    checks: list[dict[str, Any]] = []
    _basic_table_checks(table, checks)
    metadata = _metadata_checks(
        metadata_path,
        probabilities_path,
        checks,
        require_adaptive=require_adaptive,
    )
    _probability_archive_check(table, probabilities_path, metadata, checks)
    _boundary_comparison(
        table, boundary_reference, checks, boundary_relative_tolerance
    )

    equilibrium = equilibrium_charge_fractions(table)
    maximum_normalization_error = float(
        np.max(np.abs(np.sum(equilibrium["fractions"], axis=1) - 1.0))
    )
    maximum_stationarity_error = float(
        np.max(equilibrium["relative_stationarity_residual"])
    )
    equilibrium_numerically_valid = (
        np.all(equilibrium["fractions"] >= 0.0)
        and maximum_normalization_error <= 1.0e-12
        and maximum_stationarity_error <= 1.0e-12
    )
    excluded_equilibrium_energies = np.asarray(
        equilibrium["excluded_energy_keV_u"], dtype=float
    )
    equilibrium_complete = excluded_equilibrium_energies.size == 0
    equilibrium_ok = equilibrium_numerically_valid and equilibrium_complete
    _append_check(
        checks,
        name="equilibrium_balance_equation",
        status=(
            "pass"
            if equilibrium_ok
            else "fail"
            if require_adaptive
            else "informational"
        ),
        required_for_release=require_adaptive,
        criterion=(
            "Paper equation (23) is evaluated only where every adjacent "
            "transition is positive; a release table must resolve every "
            "energy without an artificial cross-section floor."
        ),
        observed={
            "maximum_normalization_error": maximum_normalization_error,
            "maximum_relative_stationarity_residual": maximum_stationarity_error,
            "resolved_energy_count": int(len(equilibrium["E_keV_u"])),
            "excluded_energy_keV_u": excluded_equilibrium_energies.tolist(),
        },
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paper_data = _read_paper_data()
    curve_comparison = _paper_curve_comparison(table, equilibrium, paper_data)
    if paper_pdf is None:
        paper_pdf = _default_paper_pdf()
    if paper_pdf is not None:
        paper_pdf = paper_pdf.expanduser().resolve()
        if not paper_pdf.is_file():
            raise FileNotFoundError(f"Paper PDF does not exist: {paper_pdf}")
        curve_comparison["source_pdf"] = str(paper_pdf)
        curve_comparison["source_pdf_sha256"] = _sha256(paper_pdf)
    plot_path = output_dir / "carbon_ctmc_paper_benchmark.png"
    equilibrium_path = output_dir / "carbon_equilibrium_charge_fractions.csv"
    report_path = output_dir / "carbon_ctmc_validation.json"
    plot_benchmark(table, equilibrium, plot_path, paper_data)
    _write_equilibrium_csv(equilibrium_path, equilibrium)

    required = [check for check in checks if check["required_for_release"]]
    report: dict[str, Any] = {
        "schema_version": 2,
        "paper": {
            "citation": (
                "T. Liamsuwan and H. Nikjoo, Physics in Medicine and "
                "Biology 58 (2013) 641--672"
            ),
            "doi": PAPER_DOI,
            "reproduced_definitions": {
                "figure_12": "pure single electron capture, sigma_SC",
                "figure_13": "pure single projectile electron loss, sigma_SL",
                "figure_14": (
                    "equilibrium fractions from sigma_SC+sigma_TI and "
                    "sigma_SL+sigma_LI using equation (23)"
                ),
            },
        },
        "inputs": {
            "validation_scope": (
                "adaptive_release" if require_adaptive else "fixed_grid_reference"
            ),
            "table": str(table_path.resolve()),
            "probabilities": str(probabilities_path.resolve()),
            "metadata": str(metadata_path.resolve()),
            "expanded_boundary_reference": (
                str(boundary_reference.resolve()) if boundary_reference else None
            ),
        },
        "outputs": {
            "plot": str(plot_path.resolve()),
            "equilibrium_table": str(equilibrium_path.resolve()),
            "validation_report": str(report_path.resolve()),
        },
        "checks": checks,
        "paper_landmarks": _paper_landmarks(equilibrium, table),
        "paper_curve_comparison": curve_comparison,
        "overall": {
            "required_checks_passed": sum(check["status"] == "pass" for check in required),
            "required_checks_total": len(required),
            "has_failed_required_check": any(check["status"] == "fail" for check in required),
            "has_unevaluated_required_check": any(
                check["status"] == "not_evaluated" for check in required
            ),
            "production_release_ready": require_adaptive
            and all(check["status"] == "pass" for check in required),
            "accuracy_note": (
                "The 0.5% limits are numerical interpolation/statistical "
                "criteria, not physical accuracy or agreement with experiment."
            ),
        },
    }
    _atomic_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot and validate carbon CTMC outputs against the definitions and "
            "reported landmarks of Liamsuwan and Nikjoo (2013), figures 12--14."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=CARBON_CHARGE_EXCHANGE_DIR,
        help="Final carbon output directory or carbon_charge_exchange_h2o.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Benchmark output directory (default: process evidence runs/INPUT_NAME).",
    )
    parser.add_argument(
        "--paper-pdf",
        type=Path,
        help=(
            "Publisher PDF used to verify the digitized source panels; its "
            "path and SHA-256 are recorded in the report."
        ),
    )
    parser.add_argument(
        "--boundary-reference",
        type=Path,
        help="Independently calculated expanded-bmax carbon CSV or DAT table.",
    )
    parser.add_argument(
        "--boundary-relative-tolerance",
        type=float,
        default=NUMERICAL_RELATIVE_TOLERANCE,
        help="Maximum paired-domain difference (default: 0.005 = 0.5%%).",
    )
    parser.add_argument(
        "--strict-release",
        action="store_true",
        help="Exit nonzero unless every production-release check is passed.",
    )
    parser.add_argument(
        "--allow-unrefined-reference",
        action="store_true",
        help=(
            "Validate and plot an explicitly requested fixed-grid reference. "
            "The missing adaptive <=0.5%% gate is informational and the "
            "result is never labelled production-release ready."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 < args.boundary_relative_tolerance < 1.0:
        raise ValueError("--boundary-relative-tolerance must lie between 0 and 1")
    table_path, probabilities_path, metadata_path = _resolve_products(args.input)
    output_dir = args.output_dir or DEFAULT_RUN_ROOT / table_path.parent.name
    report = benchmark(
        table_path=table_path,
        probabilities_path=probabilities_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        boundary_reference=args.boundary_reference,
        boundary_relative_tolerance=args.boundary_relative_tolerance,
        require_adaptive=not args.allow_unrefined_reference,
        paper_pdf=args.paper_pdf,
    )
    for check in report["checks"]:
        print(f"{check['status'].upper():>13}  {check['name']}")
    print(f"Plot: {report['outputs']['plot']}")
    print(f"Report: {report['outputs']['validation_report']}")
    print(
        "Production release ready: "
        f"{report['overall']['production_release_ready']}"
    )
    if report["overall"]["has_failed_required_check"]:
        return 1
    if args.strict_release and not report["overall"]["production_release_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
