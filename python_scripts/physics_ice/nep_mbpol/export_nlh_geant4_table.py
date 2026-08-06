#!/usr/bin/env python3
"""Export one verified NLH projectile table for the Geant4 hard-elastic model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bca.config import ATOMIC_MASS_UNIT_C2_EV  # noqa: E402
from bca.runtime import AdaptiveKernelTable  # noqa: E402
from ion_ice import get_projectile  # noqa: E402
from nlh import get_coefficients  # noqa: E402


SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _mass_c2_ev(symbol: str) -> float:
    return (
        get_projectile(symbol).default_isotope.neutral_atomic_mass_u
        * ATOMIC_MASS_UNIT_C2_EV
    )


def _pair_record(table: AdaptiveKernelTable, projectile: str, target: str) -> dict[str, Any]:
    coefficients = get_coefficients(projectile, target)
    energy_min, energy_max = table.energy_bounds_ev(projectile, target)
    return {
        "target": target,
        "target_mass_c2_ev": _mass_c2_ev(target),
        "threshold_radius_angstrom": table.turning_threshold_radius_angstrom(
            projectile, target
        ),
        "energy_min_ev": energy_min,
        "energy_max_ev": energy_max,
        "nlh_potential_rms_error_above_30_ev_percent": (
            coefficients.rms_error_above_30_ev_percent
        ),
        "nlh_potential_rms_error_above_10_ev_percent": (
            coefficients.rms_error_above_10_ev_percent
        ),
    }


def export_table(
    source: str | Path,
    output_directory: str | Path,
    projectile: str = "C",
    release_status: str = "atomistic_validation_pending",
) -> tuple[Path, Path]:
    """Write a compact, checksum-linked table and provenance manifest."""

    definition = get_projectile(projectile)
    projectile = definition.symbol
    if definition.component_status("nlh_pair_potential") != "implemented":
        raise ValueError(
            f"{projectile} has no implemented NLH H/O pair potential in the "
            "ion--ice registry."
        )
    if release_status not in {"atomistic_validation_pending", "accepted"}:
        raise ValueError("release_status must be atomistic_validation_pending or accepted.")
    table = AdaptiveKernelTable(source)
    required_pairs = {(projectile, "H"), (projectile, "O")}
    if not required_pairs.issubset(set(table.pairs)):
        raise ValueError(f"The source product lacks both {projectile}-H/O pairs.")

    output_directory = Path(output_directory).expanduser().resolve()
    data_path = output_directory / f"nlh_hard_elastic_{projectile}.dat"
    manifest_path = data_path.with_suffix(".manifest.json")
    pair_records = [_pair_record(table, projectile, target) for target in ("H", "O")]
    pair_by_target = {record["target"]: record for record in pair_records}

    rows: list[tuple[str, str, str, str]] = []
    with table.csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["projectile"] == projectile:
                rows.append(
                    (
                        row["target"],
                        row["projectile_energy_ev"],
                        row["area_quantile"],
                        row["theta_cm_rad"],
                    )
                )
    if not rows or {row[0] for row in rows} != {"H", "O"}:
        raise RuntimeError("The verified source table did not yield both target pairs.")

    lines = [
        f"# nlh_geant4_hard_elastic_table_schema {SCHEMA_VERSION}",
        f"# projectile {projectile}",
        f"# release_status {release_status}",
        "# energy_variable total_projectile_kinetic_energy_eV",
        f"# projectile_mass_c2_eV {_mass_c2_ev(projectile):.17g}",
        f"# minimum_turning_potential_eV {table.minimum_turning_potential_ev:.17g}",
        f"# source_manifest_sha256 {_sha256(table.manifest_path)}",
        f"# source_csv_sha256 {table.csv_sha256}",
        "# hard_cross_section sigma_Pt_hard=pi*r_th_Pt^2*(1-V_min/E_cm_Pt) "
        "for E_cm_Pt>V_min, otherwise 0; equals pi*b_max_Pt^2; evaluate exactly",
        "# pure_water_rate Sigma_P_hard=n_H2O*(2*sigma_PH_hard+sigma_PO_hard)",
        "# interpolation theta_cm is linear in q=(b/b_max)^2 and log-log in total energy",
        "# numerical_tolerance 0.005 combined interpolation bound; not physical accuracy",
    ]
    for target in ("H", "O"):
        record = pair_by_target[target]
        lines.extend(
            (
                f"# target_{target}_mass_c2_eV {record['target_mass_c2_ev']:.17g}",
                f"# target_{target}_threshold_radius_angstrom "
                f"{record['threshold_radius_angstrom']:.17g}",
                f"# target_{target}_energy_min_eV {record['energy_min_ev']:.17g}",
                f"# target_{target}_energy_max_eV {record['energy_max_ev']:.17g}",
            )
        )
    lines.append("# columns target projectile_energy_eV area_quantile theta_cm_rad")
    lines.extend(" ".join(row) for row in rows)
    _atomic_write(data_path, "\n".join(lines) + "\n")

    source_error = {
        record["target"]: {
            "rms_error_above_30_ev_percent": record[
                "nlh_potential_rms_error_above_30_ev_percent"
            ],
            "rms_error_above_10_ev_percent": record[
                "nlh_potential_rms_error_above_10_ev_percent"
            ],
        }
        for record in pair_records
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "projectile": projectile,
        "release_status": release_status,
        "projectile_mass_c2_ev": _mass_c2_ev(projectile),
        "targets": pair_records,
        "minimum_turning_potential_ev": table.minimum_turning_potential_ev,
        "energy_variable": "total projectile kinetic energy",
        "table": data_path.name,
        "table_row_count": len(rows),
        "table_sha256": _sha256(data_path),
        "source_manifest": os.path.relpath(table.manifest_path, output_directory),
        "source_manifest_sha256": _sha256(table.manifest_path),
        "source_csv": os.path.relpath(table.csv_path, output_directory),
        "source_csv_sha256": table.csv_sha256,
        "threshold_defined_cross_section": {
            "expression": (
                "sigma_Pt_hard(T_I)=pi*r_th,Pt^2*(1-V_min/E_cm,Pt) "
                "for E_cm,Pt>V_min; 0 otherwise"
            ),
            "equivalent_expression": "sigma_Pt_hard=pi*b_max,Pt^2",
            "runtime_rule": "evaluate analytically; never interpolate cross section",
        },
        "pure_water_ice_macroscopic_rate": (
            "Sigma_P_hard=n_H2O*(2*sigma_PH_hard+sigma_PO_hard)"
        ),
        "interpolation_contract": {
            "impact_axis": "piecewise-linear theta_cm in q=(b/b_max)^2",
            "energy_axis": "piecewise log-log theta_cm in total projectile energy",
            "nominal_combined_relative_tolerance": 0.005,
            "interpretation": (
                "numerical angular/recoil-table interpolation tolerance only; "
                "not a statement of physical accuracy"
            ),
        },
        "physical_model_uncertainty": {
            "nlh_pair_potential_rms_error_percent": source_error,
            "interpretation": (
                "published pair-fit RMS errors are separate from numerical "
                "interpolation error and do not by themselves bound transport "
                "observable uncertainty"
            ),
        },
        "phase_orientation_decision_gate": (
            "density-only phase scaling is permitted only if atomistic validation "
            "finds negligible amorphous-hexagonal and directional differences; "
            "otherwise a validated phase/orientation correction is required"
        ),
        "htran_overlap_policy": (
            "HTran is a complete H/He elastic treatment. NLH and HTran must not "
            "overlap; H/He require a validated energy handoff or explicit angular/"
            "impact-parameter partition before Geant4 release."
        ),
        "scope": (
            f"threshold-defined NLH {projectile} hard elastic scattering on "
            "independent H and O atoms in pure water ice"
        ),
        "excluded_physics": [
            "soft and attractive elastic scattering outside the retained NLH domain",
            "electronic excitation and ionisation",
            "charge exchange",
            "secondary recoil cascades and lattice relaxation",
            "phase/orientation correction pending the atomistic decision gate",
        ],
    }
    _atomic_write(manifest_path, json.dumps(manifest, indent=2) + "\n")
    return data_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=HERE / "collision_kernels" / "nlh_collision_kernels.manifest.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=HERE / "collision_kernels" / "geant4",
    )
    parser.add_argument("--projectile", default="C")
    parser.add_argument(
        "--release-status",
        choices=("atomistic_validation_pending", "accepted"),
        default="atomistic_validation_pending",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path, manifest_path = export_table(
        args.source, args.output_directory, args.projectile, args.release_status
    )
    print(data_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
