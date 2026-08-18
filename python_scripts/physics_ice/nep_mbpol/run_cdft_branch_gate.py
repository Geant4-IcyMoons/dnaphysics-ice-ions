#!/usr/bin/env python3
"""Calibrate and validate one charge-localized ion--H2O CDFT branch."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

from ion_ice import PROCESS_EVIDENCE_ROOT
from soft_dft.branch_execution import CP2KBranchExecutor
from soft_dft.branch_solver import BranchValidationSettings
from soft_dft.cdft_branch import (
    CalibrationSettings,
    build_state_identity,
    load_validated_state,
    publish_validated_state,
    sha256_file,
)
from soft_dft.config import (
    CP2K_PROVENANCE,
    DEFAULT_CP2K_SETTINGS,
    load_builtin_projectile,
)
from soft_dft.cp2k import COMPLEX_ROLE
from soft_dft.geometry import ORIENTATIONS, build_scan_geometry


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
WORKFLOW_SOURCE_RELATIVE_PATHS = (
    "python_scripts/physics_ice/nep_mbpol/run_cdft_branch_gate.py",
    "python_scripts/physics_ice/nep_mbpol/soft_dft/branch_execution.py",
    "python_scripts/physics_ice/nep_mbpol/soft_dft/branch_solver.py",
    "python_scripts/physics_ice/nep_mbpol/soft_dft/cdft_branch.py",
    "python_scripts/physics_ice/nep_mbpol/soft_dft/config.py",
    "python_scripts/physics_ice/nep_mbpol/soft_dft/cp2k.py",
    "python_scripts/physics_ice/nep_mbpol/soft_dft/density_validation.py",
    "python_scripts/physics_ice/nep_mbpol/soft_dft/geometry.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projectile", default="C")
    parser.add_argument("--charge", type=int, required=True)
    parser.add_argument(
        "--orientation",
        choices=[item.name for item in ORIENTATIONS],
        default="oxygen_back",
    )
    parser.add_argument("--separation-angstrom", type=float, default=12.0)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--cp2k-command",
        default=str(REPO_ROOT / "pbs" / "cp2k_container_wrapper.sh"),
    )
    parser.add_argument(
        "--cp2k-image",
        type=Path,
        default=REPO_ROOT
        / "software"
        / "cp2k"
        / "cp2k-2025.2-mpich-x86_64-psmp.sif",
    )
    parser.add_argument("--parent-state", type=Path)
    parser.add_argument("--calibration-initial-step-hartree", type=float, default=0.25)
    parser.add_argument("--calibration-expansion-factor", type=float, default=2.0)
    parser.add_argument("--calibration-max-strength-hartree", type=float, default=8.0)
    parser.add_argument("--calibration-max-probes", type=int, default=12)
    parser.add_argument(
        "--fixed-lambda-diagnostic-strength-hartree",
        type=float,
        help=(
            "Run exactly one active-CDFT fixed-multiplier inner-SCF diagnostic "
            "at this strength, record its immutable evidence, and stop before "
            "branch calibration."
        ),
    )
    parser.add_argument(
        "--unconstrained-scf-diagnostic",
        action="store_true",
        help="Run one fresh complex inner SCF with CDFT completely disabled.",
    )
    parser.add_argument(
        "--complex-mixing-alpha",
        type=float,
        default=DEFAULT_CP2K_SETTINGS.complex_mixing_alpha,
        help=(
            "Pulay fraction of new complex density. This numerical solver "
            "setting is included in the immutable configuration signature."
        ),
    )
    parser.add_argument(
        "--complex-scf-solver",
        choices=("DIAGONALIZATION", "OT"),
        default=DEFAULT_CP2K_SETTINGS.complex_scf_solver,
        help="Inner-SCF solver; OT uses the reviewed CG/3PNT/FULL_ALL route.",
    )
    parser.add_argument(
        "--ot-linesearch",
        choices=("2PNT", "3PNT", "GOLD", "ADAPT"),
        default=DEFAULT_CP2K_SETTINGS.ot_linesearch,
    )
    parser.add_argument(
        "--ot-preconditioner",
        choices=("FULL_ALL", "FULL_KINETIC"),
        default=DEFAULT_CP2K_SETTINGS.ot_preconditioner,
    )
    parser.add_argument(
        "--ot-energy-gap-hartree",
        type=float,
        default=DEFAULT_CP2K_SETTINGS.ot_energy_gap_hartree,
    )
    parser.add_argument(
        "--complex-mixing-method",
        choices=("DIRECT_P_MIXING", "PULAY_MIXING", "BROYDEN_MIXING"),
        default=DEFAULT_CP2K_SETTINGS.complex_mixing_method,
    )
    parser.add_argument("--population-tolerance-electrons", type=float, default=1.0e-5)
    parser.add_argument("--endpoint-strength-tolerance-hartree", type=float, default=1.0e-8)
    parser.add_argument("--endpoint-residual-tolerance-electrons", type=float, default=1.0e-5)
    parser.add_argument("--endpoint-population-tolerance-electrons", type=float, default=1.0e-5)
    parser.add_argument("--endpoint-lagrangian-tolerance-hartree", type=float, default=1.0e-7)
    parser.add_argument("--reciprocal-strength-tolerance-hartree", type=float, default=1.0e-5)
    parser.add_argument("--reciprocal-energy-tolerance-hartree", type=float, default=1.0e-7)
    parser.add_argument("--reciprocal-population-tolerance-electrons", type=float, default=1.0e-5)
    parser.add_argument("--reciprocal-spin-squared-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--reciprocal-density-relative-rms-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--curvature-delta-hartree", type=float, default=0.05)
    parser.add_argument("--curvature-negative-margin-hartree", type=float, default=0.0)
    parser.add_argument("--density-cube-stride", type=int, default=4)
    parser.add_argument(
        "--tolerance-provenance",
        default=(
            "Numerical pilot gates; repeatability and tolerance sensitivity "
            "must be demonstrated before the state enters a physical table."
        ),
    )
    parser.add_argument(
        "--publish-state",
        action="store_true",
        help=(
            "Publish the validated WFN--multiplier pair. Omit for the first "
            "repeatability/calibration pilot."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _signature(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_environment_integer(name: str, default: int = 1) -> int:
    raw = os.environ.get(name, str(default))
    if not raw.isdigit() or int(raw) < 1:
        raise RuntimeError(f"{name} must be a positive integer, got {raw!r}.")
    return int(raw)


def _file_provenance(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"Required CDFT executable artifact is absent: {path}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _executable_provenance(command: str, image: Path) -> dict[str, object]:
    tokens = shlex.split(command)
    if not tokens:
        raise ValueError("The CP2K command cannot be empty.")
    launcher_name = shutil.which(tokens[0]) or tokens[0]
    launcher = Path(launcher_name).resolve()
    command_artifacts: dict[str, dict[str, object]] = {}
    for index, token in enumerate(tokens[1:], start=1):
        candidate = Path(token)
        if candidate.is_file():
            command_artifacts[str(index)] = _file_provenance(candidate)
    source_files = {
        relative: _file_provenance(REPO_ROOT / relative)
        for relative in WORKFLOW_SOURCE_RELATIVE_PATHS
    }
    expected_series = str(CP2K_PROVENANCE["validated_series"])
    expected_revision = str(CP2K_PROVENANCE["validated_source_revision"])
    return {
        "schema_version": 1,
        "command": tokens,
        "launcher": _file_provenance(launcher),
        "command_artifacts": command_artifacts,
        "image": _file_provenance(image),
        "workflow_source_files": source_files,
        "python": {
            **_file_provenance(Path(sys.executable)),
            "version": sys.version,
        },
        "expected_cp2k": {
            "version": f"CP2K version {expected_series}",
            "source_revision": expected_revision,
        },
        "runtime": {
            "cp2k_container_binary": os.environ.get(
                "CP2K_CONTAINER_BINARY", "/opt/cp2k/bin/cp2k"
            ),
            "cp2k_mpi_ranks": _positive_environment_integer("CP2K_MPI_RANKS"),
            "omp_num_threads": _positive_environment_integer("OMP_NUM_THREADS"),
            "openblas_num_threads": _positive_environment_integer(
                "OPENBLAS_NUM_THREADS"
            ),
            "mkl_num_threads": _positive_environment_integer("MKL_NUM_THREADS"),
        },
    }


def main() -> None:
    args = parse_args()
    if (
        args.unconstrained_scf_diagnostic
        and args.fixed_lambda_diagnostic_strength_hartree is not None
    ):
        raise ValueError(
            "Unconstrained and fixed-lambda diagnostic modes are mutually exclusive."
        )
    projectile = load_builtin_projectile(args.projectile)
    state = projectile.state(args.charge)
    orientation = next(
        item for item in ORIENTATIONS if item.name == args.orientation
    )
    geometry = build_scan_geometry(
        orientation, args.separation_angstrom, projectile.symbol
    )
    settings = replace(
        DEFAULT_CP2K_SETTINGS,
        projectile_basis_set=projectile.projectile_basis_set,
        basis_file=projectile.basis_file,
        complex_scf_solver=args.complex_scf_solver,
        complex_mixing_alpha=args.complex_mixing_alpha,
        complex_mixing_method=args.complex_mixing_method,
        ot_linesearch=args.ot_linesearch,
        ot_preconditioner=args.ot_preconditioner,
        ot_energy_gap_hartree=args.ot_energy_gap_hartree,
        cdft_optimizer="BISECT",
        cdft_eps=args.population_tolerance_electrons,
    )
    task = {
        "task_id": (
            f"{projectile.symbol}_q{state.charge}_{orientation.name}_"
            f"r{args.separation_angstrom:.8g}_complex"
        ),
        "role": COMPLEX_ROLE,
        "projectile": projectile.symbol,
        "projectile_atomic_number": projectile.atomic_number,
        "charge": state.charge,
        "electrons_on_projectile": state.electrons_on_projectile,
        "multiplicity": state.multiplicity,
        "scf_spin_mode": state.scf_spin_mode,
        "configuration": state.configuration,
        "term": state.term,
        "cp2k_atomic_guess": state.atomic_guess_dict(),
        "orientation": geometry.orientation,
        "anchor_index": geometry.anchor_index,
        "anchor_element": geometry.anchor_element,
        "separation_angstrom": geometry.separation_angstrom,
        "minimum_pair_distance_angstrom": geometry.minimum_pair_distance_angstrom,
        "coordinates_angstrom": [list(row) for row in geometry.coordinates_angstrom],
    }
    validation = BranchValidationSettings(
        calibration=CalibrationSettings(
            initial_step_hartree=args.calibration_initial_step_hartree,
            expansion_factor=args.calibration_expansion_factor,
            max_abs_strength_hartree=args.calibration_max_strength_hartree,
            max_probe_count=args.calibration_max_probes,
        ),
        population_tolerance_electrons=args.population_tolerance_electrons,
        endpoint_strength_tolerance_hartree=(
            args.endpoint_strength_tolerance_hartree
        ),
        endpoint_residual_tolerance_electrons=(
            args.endpoint_residual_tolerance_electrons
        ),
        endpoint_population_tolerance_electrons=(
            args.endpoint_population_tolerance_electrons
        ),
        endpoint_lagrangian_tolerance_hartree=(
            args.endpoint_lagrangian_tolerance_hartree
        ),
        reciprocal_strength_tolerance_hartree=(
            args.reciprocal_strength_tolerance_hartree
        ),
        reciprocal_energy_tolerance_hartree=(
            args.reciprocal_energy_tolerance_hartree
        ),
        reciprocal_population_tolerance_electrons=(
            args.reciprocal_population_tolerance_electrons
        ),
        reciprocal_spin_squared_tolerance=(
            args.reciprocal_spin_squared_tolerance
        ),
        reciprocal_density_relative_rms_tolerance=(
            args.reciprocal_density_relative_rms_tolerance
        ),
        curvature_delta_hartree=args.curvature_delta_hartree,
        curvature_negative_margin_hartree=(
            args.curvature_negative_margin_hartree
        ),
    )
    image_path = args.cp2k_image.resolve()
    configured_image = os.environ.get("CP2K_IMAGE")
    if configured_image is not None and Path(configured_image).resolve() != image_path:
        raise RuntimeError(
            "CP2K_IMAGE differs from the image bound to this branch calculation."
        )
    os.environ["CP2K_IMAGE"] = str(image_path)
    executable = _executable_provenance(args.cp2k_command, image_path)
    family_configuration = {
        "schema_version": 1,
        "projectile": projectile.symbol,
        "charge": state.charge,
        "electrons_on_projectile": state.electrons_on_projectile,
        "multiplicity": state.multiplicity,
        "scf_spin_mode": state.scf_spin_mode,
        "configuration": state.configuration,
        "term": state.term,
        "cp2k_atomic_guess": state.atomic_guess_dict(),
        "orientation": orientation.name,
        "atom_order": [row[0] for row in task["coordinates_angstrom"]],
        "cp2k": settings.as_dict(),
        "validation": validation.as_dict(),
        "density_cube_stride": args.density_cube_stride,
        "executable": executable,
    }
    family_configuration_signature = _signature(family_configuration)
    successor_identity = build_state_identity(
        task,
        configuration_signature=family_configuration_signature,
        cp2k_settings=settings.as_dict(),
    )
    parent_state = (
        load_validated_state(
            args.parent_state,
            expected_identity=successor_identity,
            same_geometry=False,
        )
        if args.parent_state is not None
        else None
    )
    configuration = {
        "schema_version": 1,
        "task": task,
        "family_configuration_signature": family_configuration_signature,
        "cp2k": settings.as_dict(),
        "validation": validation.as_dict(),
        "density_cube_stride": args.density_cube_stride,
        "tolerance_provenance": args.tolerance_provenance,
        "executable": executable,
        "parent_state_id": parent_state.get("state_id") if parent_state else None,
        "scientific_status": "numerical_branch_validation_pending",
        "execution_mode": (
            "unconstrained_scf_diagnostic"
            if args.unconstrained_scf_diagnostic
            else (
                "single_fixed_lambda_diagnostic"
                if args.fixed_lambda_diagnostic_strength_hartree is not None
                else "reciprocal_branch_gate"
            )
        ),
        "fixed_lambda_diagnostic_strength_hartree": (
            args.fixed_lambda_diagnostic_strength_hartree
        ),
    }
    configuration_signature = _signature(configuration)
    base = args.output_root or (
        PROCESS_EVIDENCE_ROOT
        / "soft_nuclear_collisions"
        / "validation"
        / "runs"
        / f"{projectile.symbol.lower()}_q{state.charge}_cdft_branch_gate"
    )
    run_root = base.resolve() / configuration_signature
    run_root.mkdir(parents=True, exist_ok=True)
    configuration_path = run_root / "configuration.json"
    if configuration_path.exists():
        existing = json.loads(configuration_path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise RuntimeError("Existing CDFT gate configuration is incompatible.")
    else:
        temporary = run_root / ".configuration.json.tmp"
        temporary.write_text(
            json.dumps(configuration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, configuration_path)

    executor = CP2KBranchExecutor(
        task=task,
        settings=settings,
        validation_settings=validation,
        output_directory=run_root,
        cp2k_command=args.cp2k_command,
        run_configuration_signature=configuration_signature,
        state_family_configuration_signature=family_configuration_signature,
        density_cube_stride=args.density_cube_stride,
        executable_provenance=executable,
        parent_state=parent_state,
        progress=not args.no_progress,
    )
    if args.unconstrained_scf_diagnostic:
        result = executor.unconstrained_diagnostic()
        print(f"Run: {run_root}")
        print(f"Configuration: {configuration_signature}")
        print("Execution mode: unconstrained_scf_diagnostic")
        print(f"Inner SCF converged: {result['scf_converged']}")
        print("CDFT enabled: false")
        print("State: not published (diagnostic)")
        print("Physical soft-potential status: validation_pending")
        return
    if args.fixed_lambda_diagnostic_strength_hartree is not None:
        probe = executor.probe(
            args.fixed_lambda_diagnostic_strength_hartree,
            "fixed_lambda_diagnostic",
        )
        print(f"Run: {run_root}")
        print(f"Configuration: {configuration_signature}")
        print("Execution mode: single_fixed_lambda_diagnostic")
        print(f"Strength [hartree]: {probe.strength_hartree:.16g}")
        print(f"Inner SCF converged: {probe.inner_scf_converged}")
        print(f"Population residual [electron]: {probe.residual_electrons:.16g}")
        print("State: not published (diagnostic)")
        print("Physical soft-potential status: validation_pending")
        return
    result = executor.solve()
    state_path: Path | None = None
    if args.publish_state:
        lower = result["roots"]["lower"]
        state_path = run_root / "accepted_state" / "state.json"
        publish_validated_state(
            state_path,
            Path(lower["wavefunction_path"]),
            identity=successor_identity,
            multiplier_hartree=float(lower["cdft_strength"]),
            target_electrons=float(lower["cdft_target_electrons"]),
            current_electrons=float(lower["cdft_current_electrons"]),
            residual_electrons=float(lower["cdft_deviation_electrons"]),
            energy_hartree=float(lower["energy_hartree"]),
            electron_count_alpha=lower.get("electron_count_alpha"),
            electron_count_beta=lower.get("electron_count_beta"),
            spin_squared=lower.get("spin_squared_single_determinant"),
            cdft_trace=lower["cdft_trace"],
            branch_validation=result,
            cp2k_identity={
                "version": lower.get("cp2k_version"),
                "source_revision": lower.get("cp2k_source_revision"),
                **executable,
            },
            input_path=Path(lower["input_path"]),
            output_path=Path(lower["output_path"]),
            parent_state_id=parent_state.get("state_id") if parent_state else None,
        )
    print(f"Run: {run_root}")
    print(f"Configuration: {configuration_signature}")
    print(f"Branch status: {result['status']}")
    print(f"State: {state_path if state_path is not None else 'not published (pilot)'}")
    print("Physical soft-potential status: validation_pending")


if __name__ == "__main__":
    main()
