from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import stat
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

from soft_dft.branch_execution import CP2KBranchExecutor
from soft_dft.branch_solver import BranchValidationSettings
from soft_dft.cdft_branch import (
    CalibrationSettings,
    PHYSICAL_STATE_STATUS,
    RECIPROCAL_VALIDATION_LIMITATION,
    RECIPROCAL_VALIDATION_SCOPE,
    build_state_identity,
    load_validated_state,
    publish_validated_state,
    sha256_file,
)
from soft_dft.config import DEFAULT_CP2K_SETTINGS
from soft_dft.cp2k import COMPLEX_ROLE
import run_cdft_branch_gate as branch_gate_runner


FAKE_CP2K = r'''#!/usr/bin/env python3
from pathlib import Path
import re
import sys

input_path = Path(sys.argv[sys.argv.index("-i") + 1])
text = input_path.read_text()
project = re.search(r"^\s*PROJECT\s+(\S+)", text, re.MULTILINE).group(1)
strength = float(
    re.search(r"^\s*STRENGTH\s+(\S+)", text, re.MULTILINE).group(1)
)
target = float(re.search(r"^\s*TARGET\s+(\S+)", text, re.MULTILINE).group(1))
fixed = bool(
    re.search(
        r"TYPE CDFT_CONSTRAINT\s+EPS_SCF\s+\S+\s+MAX_SCF 0",
        text,
        re.DOTALL,
    )
)
step_match = re.search(r"^\s*STEP_SIZE\s+(\S+)", text, re.MULTILINE)

def point(iteration, lam):
    residual = 2.0 - lam
    current = target + residual
    energy = 10.0 - 0.5 * (lam - 2.0) ** 2
    print(" SCF WAVEFUNCTION OPTIMIZATION")
    print(" *** SCF run converged in 4 steps ***")
    print(
        f" CDFT SCF iter = {iteration:5d} RMS gradient = 1.0E-08 "
        f"energy = {energy:.12f}"
    )
    print(f" Target value of constraint  : {target:.12f}")
    print(f" Current value of constraint : {current:.12f}")
    print(f" Deviation from target       : {residual:.12E}")
    print(f" Strength of constraint      : {lam:.12f}")

print(" CP2K| version string: CP2K version 2025.2")
print(" CP2K| source code revision number: fake-revision")
print(" Number of electrons: 6")
print(" Number of electrons: 6")
print(" Ideal and single determinant S**2 : 0.000000 0.000000")
if fixed:
    point(1, strength)
else:
    step = float(step_match.group(1))
    first_residual = 2.0 - strength
    opposite = strength - step * first_residual
    point(1, strength)
    point(2, opposite)
    point(3, 2.0)
    print(" CDFT SCF loop converged in 3 iterations or 12 steps")
# Deliberately not W(lambda): the solver must use the CDFT-trace Lagrangian.
final_energy = -100.0 + 0.01 * strength
print(f" ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] {final_energy:.12f}")
print(" PROGRAM ENDED AT 2026-08-09 18:00:00")
Path(f"{project}-RESTART.wfn").write_bytes(f"wfn:{strength}".encode())
if "E_DENSITY_CUBE" in text:
    Path("density-ELECTRON_DENSITY-1_0.cube").write_text(
        "density\nfake\n1 0 0 0\n2 0.5 0 0\n1 0 0.5 0\n1 0 0 0.5\n"
        "6 0 0 0 0\n1.0 2.0\n"
    )
'''


def test_branch_runner_exposes_signature_bound_complex_mixing_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cdft_branch_gate.py",
            "--charge",
            "1",
            "--complex-mixing-alpha",
            "0.2",
        ],
    )
    args = branch_gate_runner.parse_args()
    assert args.complex_mixing_alpha == pytest.approx(0.2)

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_cdft_branch_gate.py", "--charge", "2", "--complex-scf-solver", "OT"],
    )
    assert branch_gate_runner.parse_args().complex_scf_solver == "OT"

    pbs = (NEP_DIR.parents[2] / "pbs" / "run_cdft_branch_gate.pbs").read_text(
        encoding="utf-8"
    )
    assert 'COMPLEX_MIXING_ALPHA="${COMPLEX_MIXING_ALPHA:-0.5}"' in pbs
    assert '--complex-mixing-alpha "${COMPLEX_MIXING_ALPHA}"' in pbs
    assert 'COMPLEX_SCF_SOLVER="${COMPLEX_SCF_SOLVER:-DIAGONALIZATION}"' in pbs
    assert '--complex-scf-solver "${COMPLEX_SCF_SOLVER}"' in pbs
    assert '--complex-mixing-method "${COMPLEX_MIXING_METHOD}"' in pbs
    assert '--ot-linesearch "${OT_LINESEARCH}"' in pbs
    assert '--ot-preconditioner "${OT_PRECONDITIONER}"' in pbs


def test_q1_mixing_control_is_preregistered_and_one_parameter_only() -> None:
    contract_path = (
        NEP_DIR
        / "soft_dft"
        / "benchmarking"
        / "carbon_state_initialization"
        / "q1_mixing_control_contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["status"] == "preregistered_before_submission"
    controls = contract["independent_controls"]
    assert [item["complex_mixing_alpha"] for item in controls] == [0.2, 0.1]
    assert all(item["reuse_wavefunction"] is False for item in controls)
    assert contract["system"]["mixing_method"] == "PULAY_MIXING"
    assert contract["system"]["npulay"] == 5
    assert set(contract["agreement_tolerances"]) == {
        "carbon_population_absolute_electrons",
        "spin_squared_absolute",
        "total_density_relative_rms",
        "spin_density_relative_rms",
        "constrained_lagrangian_absolute_hartree",
        "corrected_electronic_energy_absolute_hartree",
    }


def _task() -> dict[str, object]:
    return {
        "task_id": "C_q4_oxygen_back_r000_complex",
        "role": COMPLEX_ROLE,
        "projectile": "C",
        "charge": 4,
        "multiplicity": 1,
        "electrons_on_projectile": 2,
        "scf_spin_mode": "RESTRICTED",
        "cp2k_atomic_guess": {
            "alpha": [
                {"n": 2, "l": 0, "nel": -2},
                {"n": 2, "l": 1, "nel": -2},
            ],
            "beta": [
                {"n": 2, "l": 0, "nel": -2},
                {"n": 2, "l": 1, "nel": -2},
            ],
        },
        "orientation": "oxygen_back",
        "separation_angstrom": 6.0,
        "coordinates_angstrom": [
            ["C", 0.0, 0.0, -6.0],
            ["O", 0.0, 0.0, 0.0],
            ["H", 0.75, 0.0, 0.58],
            ["H", -0.75, 0.0, 0.58],
        ],
    }


def _validation() -> BranchValidationSettings:
    return BranchValidationSettings(
        calibration=CalibrationSettings(1.0, 2.0, 4.0, 6),
        population_tolerance_electrons=1.0e-5,
        endpoint_strength_tolerance_hartree=1.0e-8,
        endpoint_residual_tolerance_electrons=1.0e-8,
        endpoint_population_tolerance_electrons=1.0e-8,
        endpoint_lagrangian_tolerance_hartree=1.0e-8,
        reciprocal_strength_tolerance_hartree=1.0e-8,
        reciprocal_energy_tolerance_hartree=1.0e-8,
        reciprocal_population_tolerance_electrons=1.0e-8,
        reciprocal_spin_squared_tolerance=1.0e-8,
        reciprocal_density_relative_rms_tolerance=1.0e-8,
        curvature_delta_hartree=0.1,
        curvature_negative_margin_hartree=0.0,
    )


def _artifact(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _provenance(
    command: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    image = tmp_path / "fake-cp2k.sif"
    image.write_bytes(b"fake immutable image")
    monkeypatch.setenv("CP2K_IMAGE", str(image.resolve()))
    for name in (
        "CP2K_MPI_RANKS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    return {
        "schema_version": 1,
        "command": command,
        "launcher": _artifact(Path(command[0])),
        "image": _artifact(image),
        "python": _artifact(Path(sys.executable)),
        "command_artifacts": {"1": _artifact(Path(command[1]))},
        "workflow_source_files": {"fake_cp2k.py": _artifact(Path(command[1]))},
        "expected_cp2k": {
            "version": "CP2K version 2025.2",
            "source_revision": "fake-revision",
        },
        "runtime": {
            "cp2k_container_binary": "/opt/cp2k/bin/cp2k",
            "cp2k_mpi_ranks": 1,
            "omp_num_threads": 1,
            "openblas_num_threads": 1,
            "mkl_num_threads": 1,
        },
    }


def test_executor_runs_and_resumes_a_fully_validated_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "fake_cp2k.py"
    fake.write_text(FAKE_CP2K, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    command = [sys.executable, str(fake)]
    executor = CP2KBranchExecutor(
        task=_task(),
        settings=replace(
            DEFAULT_CP2K_SETTINGS,
            cdft_optimizer="BISECT",
            complex_scf_solver="DIAGONALIZATION",
        ),
        validation_settings=_validation(),
        output_directory=tmp_path / "run",
        cp2k_command=command,
        run_configuration_signature="test-run-configuration",
        state_family_configuration_signature="test-family-configuration",
        density_cube_stride=4,
        executable_provenance=_provenance(command, tmp_path, monkeypatch),
        progress=False,
    )
    result = executor.solve()
    assert result["status"] == "validated"
    assert result["reciprocal_validation"]["density"]["relative_rms"] == 0.0
    assert result["reciprocal_validation"][
        "mean_constrained_lagrangian_hartree"
    ] == 10.0
    run_files = sorted((tmp_path / "run" / "runs").glob("*/run.json"))
    assert len(run_files) == 9
    calibration_runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in run_files
        if path.parent.name.startswith("probe_calibration_")
    ]
    assert calibration_runs
    assert all(run["restart_sha256"] is None for run in calibration_runs)
    assert all(
        "SCF_GUESS ATOMIC"
        in Path(run["input_path"]).read_text(encoding="utf-8")
        for run in calibration_runs
    )
    assert executor.solve() == result
    assert sorted((tmp_path / "run" / "runs").glob("*/run.json")) == run_files


def test_executor_rechecks_immutable_run_evidence_on_validated_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = tmp_path / "fake_cp2k.py"
    fake.write_text(FAKE_CP2K, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    command = [sys.executable, str(fake)]
    executor = CP2KBranchExecutor(
        task=_task(),
        settings=replace(
            DEFAULT_CP2K_SETTINGS,
            cdft_optimizer="BISECT",
            complex_scf_solver="DIAGONALIZATION",
        ),
        validation_settings=_validation(),
        output_directory=tmp_path / "run",
        cp2k_command=command,
        run_configuration_signature="test-run-configuration",
        state_family_configuration_signature="test-family-configuration",
        density_cube_stride=4,
        executable_provenance=_provenance(command, tmp_path, monkeypatch),
        progress=False,
    )
    result = executor.solve()
    root_wfn = Path(result["roots"]["lower"]["wavefunction_path"])
    root_wfn.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="wavefunction checksum"):
        executor.solve()


def test_validated_resume_reconstructs_checkpoint_and_reruns_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "fake_cp2k.py"
    fake.write_text(FAKE_CP2K, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    command = [sys.executable, str(fake)]
    executor = CP2KBranchExecutor(
        task=_task(),
        settings=replace(
            DEFAULT_CP2K_SETTINGS,
            cdft_optimizer="BISECT",
            complex_scf_solver="DIAGONALIZATION",
        ),
        validation_settings=_validation(),
        output_directory=tmp_path / "run",
        cp2k_command=command,
        run_configuration_signature="test-run-configuration",
        state_family_configuration_signature="test-family-configuration",
        density_cube_stride=4,
        executable_provenance=_provenance(command, tmp_path, monkeypatch),
        progress=False,
    )
    accepted = executor.solve()
    checkpoint_path = executor.checkpoint_path
    corrupted = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    first_probe = next(iter(corrupted["calibration_probes"].values()))
    first_probe["current_electrons"] = 999.0
    corrupted["roots"]["lower"]["cdft_strength"] = -999.0
    corrupted["reciprocal_validation"]["mean_strength_hartree"] = -999.0
    checkpoint_path.write_text(json.dumps(corrupted), encoding="utf-8")

    resumed = executor.solve()
    assert resumed == accepted
    assert resumed["roots"]["lower"]["cdft_strength"] == 2.0


def test_executor_rejects_wrong_cp2k_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "wrong_cp2k.py"
    fake.write_text(
        FAKE_CP2K.replace("fake-revision", "wrong-revision"), encoding="utf-8"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    command = [sys.executable, str(fake)]
    executor = CP2KBranchExecutor(
        task=_task(),
        settings=replace(DEFAULT_CP2K_SETTINGS, cdft_optimizer="BISECT"),
        validation_settings=_validation(),
        output_directory=tmp_path / "run",
        cp2k_command=command,
        run_configuration_signature="test-run-configuration",
        state_family_configuration_signature="test-family-configuration",
        density_cube_stride=4,
        executable_provenance=_provenance(command, tmp_path, monkeypatch),
        progress=False,
    )
    with pytest.raises(RuntimeError, match="CP2K CDFT execution failed"):
        executor.solve()
    failure = json.loads(
        (
            tmp_path
            / "run"
            / "runs"
            / "probe_calibration_000"
            / "run.json"
        ).read_text(encoding="utf-8")
    )
    assert "source revision differs" in failure["identity_error"]


def test_executor_rejects_changed_signed_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "fake"
    fake.write_bytes(b"fake")
    command = [sys.executable, str(fake)]
    provenance = _provenance(command, tmp_path, monkeypatch)
    Path(str(provenance["image"]["path"])).write_bytes(b"changed image")
    with pytest.raises(RuntimeError, match="container image provenance checksum"):
        CP2KBranchExecutor(
            task=_task(),
            settings=replace(DEFAULT_CP2K_SETTINGS, cdft_optimizer="BISECT"),
            validation_settings=_validation(),
            output_directory=tmp_path / "run",
            cp2k_command=command,
            run_configuration_signature="test-run-configuration",
            state_family_configuration_signature="test-family-configuration",
            density_cube_stride=4,
            executable_provenance=provenance,
            progress=False,
        )


def test_executor_rejects_unverifiable_parent_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "fake"
    fake.write_bytes(b"fake")
    command = [sys.executable, str(fake)]
    with pytest.raises(RuntimeError, match="state-record path"):
        CP2KBranchExecutor(
            task=_task(),
            settings=replace(DEFAULT_CP2K_SETTINGS, cdft_optimizer="BISECT"),
            validation_settings=_validation(),
            output_directory=tmp_path / "run",
            cp2k_command=command,
            run_configuration_signature="test-run-configuration",
            state_family_configuration_signature="test-family-configuration",
            density_cube_stride=4,
            executable_provenance=_provenance(command, tmp_path, monkeypatch),
            parent_state={"state_id": "unverified"},
            progress=False,
        )


def test_executor_separates_run_and_state_family_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "fake"
    fake.write_bytes(b"fake")
    command = [sys.executable, str(fake)]
    common = {
        "task": _task(),
        "settings": replace(DEFAULT_CP2K_SETTINGS, cdft_optimizer="BISECT"),
        "validation_settings": _validation(),
        "cp2k_command": command,
        "density_cube_stride": 4,
        "executable_provenance": _provenance(command, tmp_path, monkeypatch),
        "progress": False,
    }
    first = CP2KBranchExecutor(
        **common,
        output_directory=tmp_path / "first",
        run_configuration_signature="run-one",
        state_family_configuration_signature="family",
    )
    second = CP2KBranchExecutor(
        **common,
        output_directory=tmp_path / "second",
        run_configuration_signature="run-two",
        state_family_configuration_signature="family",
    )
    changed_family = CP2KBranchExecutor(
        **common,
        output_directory=tmp_path / "third",
        run_configuration_signature="run-three",
        state_family_configuration_signature="other-family",
    )
    assert first.state_identity == second.state_identity
    assert first.solver_signature != second.solver_signature
    assert first.state_identity["family_signature"] != changed_family.state_identity[
        "family_signature"
    ]


def test_parent_continuation_probes_center_first_and_chains_by_centered_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = tmp_path / "fake_cp2k.py"
    fake.write_text(FAKE_CP2K, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    settings = replace(
        DEFAULT_CP2K_SETTINGS,
        cdft_optimizer="BISECT",
        complex_scf_solver="DIAGONALIZATION",
    )
    family_signature = "test-family-configuration"
    identity = build_state_identity(
        _task(),
        configuration_signature=family_signature,
        cp2k_settings=settings.as_dict(),
    )
    parent_directory = tmp_path / "parent"
    parent_directory.mkdir()
    source_wfn = parent_directory / "source.wfn"
    source_input = parent_directory / "source.inp"
    source_output = parent_directory / "source.out"
    source_wfn.write_bytes(b"validated-parent-wavefunction")
    source_input.write_text("parent input", encoding="utf-8")
    source_output.write_text("parent output", encoding="utf-8")
    state_path = parent_directory / "state.json"
    publish_validated_state(
        state_path,
        source_wfn,
        identity=identity,
        multiplier_hartree=2.0,
        target_electrons=2.0,
        current_electrons=2.0,
        residual_electrons=0.0,
        energy_hartree=10.0,
        electron_count_alpha=6,
        electron_count_beta=6,
        spin_squared=0.0,
        cdft_trace=[{"strength": 2.0, "residual": 0.0, "energy": 10.0}],
        branch_validation={
            "status": "validated",
            "directions": ["lower", "upper"],
            "validation_scope": RECIPROCAL_VALIDATION_SCOPE,
            "physical_state_status": PHYSICAL_STATE_STATUS,
            "validation_limitation": RECIPROCAL_VALIDATION_LIMITATION,
        },
        cp2k_identity={"version": "test"},
        input_path=source_input,
        output_path=source_output,
        parent_state_id=None,
    )
    parent_state = load_validated_state(state_path)
    parent_wfn = state_path.parent / str(parent_state["wavefunction"]["path"])

    command = [sys.executable, str(fake)]
    executor = CP2KBranchExecutor(
        task=_task(),
        settings=settings,
        validation_settings=_validation(),
        output_directory=tmp_path / "continued",
        cp2k_command=command,
        run_configuration_signature="continued-run-configuration",
        state_family_configuration_signature=family_signature,
        density_cube_stride=4,
        executable_provenance=_provenance(command, tmp_path, monkeypatch),
        parent_state=parent_state,
        progress=False,
    )
    executor.probe(2.0, "calibration_center")
    executor.probe(3.0, "calibration_000")
    executor.probe(1.0, "calibration_001")
    executor.probe(4.0, "calibration_002")
    executor.probe(0.0, "calibration_003")

    records = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "continued" / "runs").glob("*/run.json")
    }
    parent_sha = sha256_file(parent_wfn)
    assert records["probe_calibration_center"]["restart_sha256"] == parent_sha
    assert records["probe_calibration_000"]["restart_sha256"] == parent_sha
    assert records["probe_calibration_001"]["restart_sha256"] == parent_sha
    assert records["probe_calibration_002"]["restart_sha256"] == records[
        "probe_calibration_000"
    ]["wavefunction_sha256"]
    assert records["probe_calibration_003"]["restart_sha256"] == records[
        "probe_calibration_001"
    ]["wavefunction_sha256"]


def test_executor_cannot_accept_stale_partial_wavefunction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "fake_cp2k_without_wfn.py"
    fake.write_text(
        FAKE_CP2K.replace(
            'Path(f"{project}-RESTART.wfn").write_bytes(f"wfn:{strength}".encode())',
            "# Deliberately produce no wavefunction.",
        ),
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    output_directory = tmp_path / "run"
    stale_directory = output_directory / "runs" / "probe_calibration_000"
    stale_directory.mkdir(parents=True)
    stale_wfn = stale_directory / "stale-RESTART.wfn"
    stale_cube = stale_directory / "stale-density.cube"
    stale_output = stale_directory / ".cp2k.out.tmp"
    stale_wfn.write_bytes(b"stale")
    stale_cube.write_text("stale", encoding="utf-8")
    stale_output.write_text("stale", encoding="utf-8")

    command = [sys.executable, str(fake)]
    executor = CP2KBranchExecutor(
        task=_task(),
        settings=replace(
            DEFAULT_CP2K_SETTINGS,
            cdft_optimizer="BISECT",
            complex_scf_solver="DIAGONALIZATION",
        ),
        validation_settings=_validation(),
        output_directory=output_directory,
        cp2k_command=command,
        run_configuration_signature="test-run-configuration",
        state_family_configuration_signature="test-family-configuration",
        density_cube_stride=4,
        executable_provenance=_provenance(command, tmp_path, monkeypatch),
        progress=False,
    )
    with pytest.raises(RuntimeError, match="did not produce"):
        executor.solve()
    assert not stale_wfn.exists()
    assert not stale_cube.exists()
    assert not stale_output.exists()
