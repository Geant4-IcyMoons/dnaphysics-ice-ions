"""Prepare, run, and reduce restart-safe atomistic full-ZBL campaigns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
from tqdm.auto import tqdm

from .generate_backend import REPOSITORY_ROOT, write_manifests


SCHEMA_VERSION = 1
DEFAULT_PHASES = ("hexagonal_ih_100k", "amorphous_lda_80k")
DEFAULT_CUTOFFS_EV = (1.0, 10.0, 30.0)
ORIENTATION_ARGUMENTS = {
    "c_axis": ("--direction", "0", "0", "1"),
    "basal_a_axis": ("--direction", "1", "0", "0"),
    "isotropic": ("--isotropic-directions",),
}
SIMULATOR = (
    REPOSITORY_ROOT
    / "python_scripts/physics_ice/nep_mbpol/simulate_nlh_hard_collisions.py"
)
IMPLEMENTATION_SOURCES = (
    SIMULATOR,
    REPOSITORY_ROOT
    / "python_scripts/physics_ice/nep_mbpol/bca/trajectory.py",
    REPOSITORY_ROOT
    / "python_scripts/physics_ice/nep_mbpol/bca/scattering.py",
    REPOSITORY_ROOT
    / "python_scripts/physics_ice/nep_mbpol/nlh/coefficients.csv",
    Path(__file__).resolve().parent / "backend.py",
    Path(__file__).resolve().parent / "kernel.py",
    Path(__file__).resolve(),
)
QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _energy_grid(minimum_ev: float, maximum_ev: float, points: int) -> list[float]:
    if not (0.0 < minimum_ev < maximum_ev) or points < 2:
        raise ValueError("Require positive ordered energy bounds and at least 2 points.")
    return np.geomspace(minimum_ev, maximum_ev, points).tolist()


def _case_key(case: dict[str, Any]) -> str:
    energy = f"{float(case['energy_ev']):.12g}eV".replace("+", "")
    cutoff = f"{float(case['minimum_transfer_ev']):g}eV"
    return (
        f"{case['phase_id']}/structure{int(case['structure_index']):02d}/"
        f"{case['orientation']}/{cutoff}/{energy}"
    )


def prepare_campaign(
    output_root: Path,
    *,
    projectile: str,
    phases: Sequence[str],
    cutoffs_ev: Sequence[float],
    energy_min_ev: float,
    energy_max_ev: float,
    energy_points: int,
    tolerance: float,
    confidence: float,
    minimum_trajectories: int,
    maximum_trajectories: int,
    trajectory_batch_size: int,
    path_length_angstrom: float,
    interaction_model: str = "zbl_full",
    nlh_boundary_ev: float = 30.0,
) -> Path:
    if projectile not in {"C", "O", "S"}:
        raise ValueError("The atomistic ZBL campaign is restricted to C, O, and S.")
    if not phases or any(phase not in DEFAULT_PHASES for phase in phases):
        raise ValueError(f"Phases must be selected from {DEFAULT_PHASES}.")
    if not cutoffs_ev or any(
        not math.isfinite(value) or value <= 0.0 for value in cutoffs_ev
    ):
        raise ValueError("Transfer cutoffs must be finite and positive.")
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("Tolerance must lie strictly in (0, 1).")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must lie strictly in (0, 1).")
    if minimum_trajectories < 2 or maximum_trajectories < minimum_trajectories:
        raise ValueError("Require 2 <= minimum trajectories <= maximum trajectories.")
    if trajectory_batch_size < 1 or path_length_angstrom <= 0.0:
        raise ValueError("Batch size and path length must be positive.")
    if interaction_model not in {"zbl_full", "zbl_soft"}:
        raise ValueError("interaction_model must be zbl_full or zbl_soft.")
    if interaction_model == "zbl_soft" and (
        not math.isfinite(nlh_boundary_ev) or nlh_boundary_ev < 30.0
    ):
        raise ValueError("zbl_soft requires nlh_boundary_ev >= 30 eV.")
    output_root = output_root.resolve()
    backend_paths = write_manifests(
        output_root / "backends",
        phases,
        projectiles=(projectile,),
        energy_bounds_ev=(energy_min_ev, energy_max_ev),
        minimum_transfer_cutoffs_ev=cutoffs_ev,
        interaction_model=interaction_model,
        nlh_boundary_ev=nlh_boundary_ev,
    )
    energies = _energy_grid(energy_min_ev, energy_max_ev, energy_points)
    cases: list[dict[str, Any]] = []
    for backend_path in backend_paths:
        backend = json.loads(backend_path.read_text(encoding="utf-8"))
        for structure_index, structure in enumerate(backend["structures"]):
            for orientation in backend["orientations"]:
                for cutoff in sorted(float(value) for value in cutoffs_ev):
                    for energy in energies:
                        case = {
                            "phase_id": backend["phase_id"],
                            "backend_manifest": str(backend_path),
                            "structure_index": structure_index,
                            "structure": structure,
                            "orientation": orientation,
                            "minimum_transfer_ev": cutoff,
                            "energy_ev": energy,
                        }
                        case["case_key"] = _case_key(case)
                        case["output_directory"] = str(
                            output_root / "cases" / case["case_key"]
                        )
                        cases.append(case)
    backend_records = [
        {"path": str(path), "sha256": _sha256(path)} for path in backend_paths
    ]
    configuration = {
        "schema_version": SCHEMA_VERSION,
        "interaction_model": interaction_model,
        "nlh_turning_potential_boundary_ev": (
            nlh_boundary_ev if interaction_model == "zbl_soft" else None
        ),
        "projectile": projectile,
        "phases": list(phases),
        "minimum_transfer_cutoffs_ev": sorted(float(value) for value in cutoffs_ev),
        "energy_grid_ev": energies,
        "statistical_relative_tolerance": tolerance,
        "statistical_confidence": confidence,
        "trajectory_cdf_absolute_tolerance": tolerance,
        "minimum_trajectories": minimum_trajectories,
        "maximum_trajectories": maximum_trajectories,
        "trajectory_batch_size": trajectory_batch_size,
        "path_length_angstrom": path_length_angstrom,
        "sampling_mode": "collision_tube_mixture",
        "tube_mixture_fraction": 0.5,
        "control_variate": True,
        "output_detail": "summary",
        "physics_status": "validation_pending",
        "runtime_exclusivity": (
            "zbl_soft is impact-area complementary to nlh_hard"
            if interaction_model == "zbl_soft"
            else "zbl_full replaces nlh_hard and HTran"
        ),
        "backend_manifests": backend_records,
        "implementation_sources": [
            {
                "path": str(path.resolve().relative_to(REPOSITORY_ROOT)),
                "sha256": _sha256(path),
            }
            for path in IMPLEMENTATION_SOURCES
        ],
    }
    signature = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "configuration_signature": signature,
        "configuration": configuration,
        "case_count": len(cases),
        "cases": cases,
    }
    path = output_root / f"{interaction_model}_campaign.manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("configuration_signature") != signature:
            raise RuntimeError(f"Campaign configuration mismatch: {path}")
    else:
        _atomic_json(path, manifest)
    return path


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Malformed ZBL campaign: {path}")
    return value


def run_case(
    campaign_path: Path, case_index: int, workers: int
) -> Path:
    campaign = _campaign(campaign_path)
    available_workers = len(os.sched_getaffinity(0))
    if workers < 1 or workers > available_workers:
        raise ValueError(
            f"Workers must lie in [1, {available_workers}] on this allocation."
        )
    try:
        case = campaign["cases"][case_index]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"Invalid case index {case_index}.") from exc
    configuration = campaign["configuration"]
    for record in configuration["implementation_sources"]:
        source = REPOSITORY_ROOT / record["path"]
        if not source.is_file() or _sha256(source) != record["sha256"]:
            raise RuntimeError(f"Campaign implementation checksum mismatch: {source}")
    structure = case["structure"]
    output = Path(case["output_directory"])
    output_stem = f"{configuration['interaction_model']}_collision"
    manifest_path = output / f"{output_stem}_run.manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = existing.get("statistical_convergence")
        if isinstance(report, dict) and report.get("converged") is True:
            return manifest_path
    seed = int.from_bytes(
        hashlib.sha256(case["case_key"].encode()).digest()[:4], "big"
    )
    command = [
        sys.executable,
        "-u",
        str(SIMULATOR),
        str(REPOSITORY_ROOT / structure["path"]),
        "--metadata",
        str(REPOSITORY_ROOT / structure["metadata"]),
        "--interaction-model",
        configuration["interaction_model"],
        "--minimum-transfer-ev",
        f"{float(case['minimum_transfer_ev']):.17g}",
        *(
            (
                "--nlh-boundary-ev",
                f"{float(configuration['nlh_turning_potential_boundary_ev']):.17g}",
            )
            if configuration["interaction_model"] == "zbl_soft"
            else ()
        ),
        "--projectile",
        configuration["projectile"],
        "--energy-ev",
        f"{float(case['energy_ev']):.17g}",
        "--minimum-trajectories",
        str(configuration["minimum_trajectories"]),
        "--maximum-trajectories",
        str(configuration["maximum_trajectories"]),
        "--trajectory-batch-size",
        str(configuration["trajectory_batch_size"]),
        "--statistical-relative-tolerance",
        f"{float(configuration['statistical_relative_tolerance']):.17g}",
        "--statistical-confidence",
        f"{float(configuration['statistical_confidence']):.17g}",
        "--trajectory-cdf-tolerance",
        f"{float(configuration['trajectory_cdf_absolute_tolerance']):.17g}",
        "--path-length-angstrom",
        f"{float(configuration['path_length_angstrom']):.17g}",
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--sampling-mode",
        configuration["sampling_mode"],
        "--tube-mixture-fraction",
        f"{float(configuration['tube_mixture_fraction']):.17g}",
        "--control-variate",
        "--output-detail",
        configuration["output_detail"],
        "--output-directory",
        str(output),
        *ORIENTATION_ARGUMENTS[case["orientation"]],
    ]
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "zbl_full_progress.log"
    with log_path.open("a", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ZBL case {case_index} exited {completed.returncode}; "
            f"resume with the same command. See {log_path}."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["statistical_convergence"].get("converged") is not True:
        raise RuntimeError(f"ZBL case did not satisfy the 0.5% gate: {manifest_path}")
    receipt = output / "case.receipt.json"
    _atomic_json(
        receipt,
        {
            "schema_version": SCHEMA_VERSION,
            "campaign_signature": campaign["configuration_signature"],
            "case_index": case_index,
            "case_key": case["case_key"],
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
        },
    )
    return manifest_path


def _distribution_samples(manifest: dict[str, Any], field: str) -> np.ndarray:
    checkpoint = Path(manifest["outputs"]["checkpoint_directory"])
    arrays = []
    for record_path in sorted(checkpoint.glob("batch_*.manifest.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        sample_path = checkpoint / record["distribution_sample"]
        with np.load(sample_path) as values:
            arrays.append(np.asarray(values[field], dtype=np.float64))
    if not arrays:
        raise RuntimeError(f"No {field} samples below {checkpoint}.")
    return np.concatenate(arrays)


def _atomic_csv(path: Path, fields: Sequence[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def reduce_campaign(campaign_path: Path) -> tuple[Path, Path, Path]:
    campaign = _campaign(campaign_path)
    output_root = campaign_path.resolve().parent
    cross_sections: list[dict[str, Any]] = []
    angular: list[dict[str, Any]] = []
    for case in tqdm(campaign["cases"], desc="Reducing ZBL cases", unit="case"):
        output_stem = f"{campaign['configuration']['interaction_model']}_collision"
        manifest_path = Path(case["output_directory"]) / f"{output_stem}_run.manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Missing completed case: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = manifest["statistical_convergence"]
        if report.get("converged") is not True:
            raise RuntimeError(f"Unconverged case: {manifest_path}")
        rate = report["observables"]["hard_collision_rate_per_angstrom"]
        density = (
            manifest["structure"]["water_molecules"]
            / manifest["structure"]["volume_angstrom3"]
        )
        cdf = report["trajectory_cdf_convergence"]
        cross_sections.append(
            {
                "phase_id": case["phase_id"],
                "structure_index": case["structure_index"],
                "orientation": case["orientation"],
                "projectile": campaign["configuration"]["projectile"],
                "total_kinetic_energy_ev": case["energy_ev"],
                "minimum_recoil_transfer_ev": case["minimum_transfer_ev"],
                "trajectories": manifest["configuration"]["completed_trajectories"],
                "sigma_per_h2o_angstrom2": rate["estimate"] / density,
                "sigma_half_width_angstrom2": rate["confidence_half_width"] / density,
                "relative_half_width": rate["relative_confidence_half_width"],
                "cdf_absolute_half_width": cdf["absolute_confidence_half_width"],
                "nuclear_stopping_ev_per_angstrom": report["observables"][
                    "hard_nuclear_stopping_ev_per_angstrom"
                ]["estimate"],
                "transport_rate_per_angstrom": report["observables"][
                    "hard_transport_rate_per_angstrom"
                ]["estimate"],
            }
        )
        samples = _distribution_samples(manifest, "final_deflection_rad")
        values = np.quantile(samples, QUANTILES)
        for quantile, value in zip(QUANTILES, values, strict=True):
            angular.append(
                {
                    "phase_id": case["phase_id"],
                    "structure_index": case["structure_index"],
                    "orientation": case["orientation"],
                    "projectile": campaign["configuration"]["projectile"],
                    "total_kinetic_energy_ev": case["energy_ev"],
                    "minimum_recoil_transfer_ev": case["minimum_transfer_ev"],
                    "quantile": quantile,
                    "final_deflection_rad": value,
                    "cdf_absolute_half_width": cdf["absolute_confidence_half_width"],
                }
            )
    model = campaign["configuration"]["interaction_model"]
    cross_path = output_root / f"{model}_cross_sections.csv"
    angular_path = output_root / f"{model}_angular_quantiles.csv"
    _atomic_csv(cross_path, tuple(cross_sections[0]), cross_sections)
    _atomic_csv(angular_path, tuple(angular[0]), angular)
    final_path = output_root / f"{model}_campaign.results.json"
    _atomic_json(
        final_path,
        {
            "schema_version": SCHEMA_VERSION,
            "configuration_signature": campaign["configuration_signature"],
            "numerical_status": "all_cases_satisfy_0.5_percent_gates",
            "physics_status": "validation_pending",
            "case_count": len(cross_sections),
            "cross_sections_csv": cross_path.name,
            "cross_sections_sha256": _sha256(cross_path),
            "angular_quantiles_csv": angular_path.name,
            "angular_quantiles_sha256": _sha256(angular_path),
            "full_angular_samples": "retained in each restart-safe checkpoint directory",
        },
    )
    return cross_path, angular_path, final_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--projectile", choices=("C", "O", "S"), default="C")
    prepare.add_argument(
        "--phases", nargs="+", choices=DEFAULT_PHASES, default=DEFAULT_PHASES
    )
    prepare.add_argument(
        "--transfer-cutoffs-ev",
        nargs="+",
        type=float,
        default=DEFAULT_CUTOFFS_EV,
    )
    prepare.add_argument("--energy-min-ev", type=float, default=1.0e4)
    prepare.add_argument("--energy-max-ev", type=float, default=1.0e8)
    prepare.add_argument("--energy-points", type=int, default=41)
    prepare.add_argument("--tolerance", type=float, default=0.005)
    prepare.add_argument("--confidence", type=float, default=0.95)
    prepare.add_argument("--minimum-trajectories", type=int, default=200_000)
    prepare.add_argument("--maximum-trajectories", type=int, default=64_000_000)
    prepare.add_argument("--trajectory-batch-size", type=int, default=100_000)
    prepare.add_argument("--path-length-angstrom", type=float, default=100.0)
    prepare.add_argument(
        "--interaction-model",
        choices=("zbl_full", "zbl_soft"),
        default="zbl_full",
    )
    prepare.add_argument("--nlh-boundary-ev", type=float, default=30.0)
    run = commands.add_parser("run")
    run.add_argument("campaign", type=Path)
    run.add_argument("--case-index", type=int, required=True)
    run.add_argument("--workers", type=int, default=0)
    reduce_parser = commands.add_parser("reduce")
    reduce_parser.add_argument("campaign", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        path = prepare_campaign(
            args.output_root,
            projectile=args.projectile,
            phases=tuple(args.phases),
            cutoffs_ev=tuple(args.transfer_cutoffs_ev),
            energy_min_ev=args.energy_min_ev,
            energy_max_ev=args.energy_max_ev,
            energy_points=args.energy_points,
            tolerance=args.tolerance,
            confidence=args.confidence,
            minimum_trajectories=args.minimum_trajectories,
            maximum_trajectories=args.maximum_trajectories,
            trajectory_batch_size=args.trajectory_batch_size,
            path_length_angstrom=args.path_length_angstrom,
            interaction_model=args.interaction_model,
            nlh_boundary_ev=args.nlh_boundary_ev,
        )
        print(f"Campaign manifest: {path}")
        print(f"Case count: {_campaign(path)['case_count']}")
        return 0
    if args.command == "run":
        workers = len(os.sched_getaffinity(0)) if args.workers == 0 else args.workers
        print(f"Completed: {run_case(args.campaign, args.case_index, workers)}")
        return 0
    for path in reduce_campaign(args.campaign):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
