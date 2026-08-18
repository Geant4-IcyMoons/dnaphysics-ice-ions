"""Render and parse fixed-charge all-electron CP2K calculations."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from .config import CP2KSettings


COMPLEX_ROLE = "complex_cdft"
PROJECTILE_COUNTERPOISE_ROLE = "projectile_counterpoise"
WATER_COUNTERPOISE_ROLE = "water_counterpoise"
ROLES = (COMPLEX_ROLE, PROJECTILE_COUNTERPOISE_ROLE, WATER_COUNTERPOISE_ROLE)
CDFT_MODE_OPTIMIZED = "optimized"
CDFT_MODE_FIXED_LAMBDA = "fixed_lambda"
CDFT_MODES = (CDFT_MODE_OPTIMIZED, CDFT_MODE_FIXED_LAMBDA)
SCF_MODE_UNCONSTRAINED = "unconstrained"


def _coordinate_kinds(role: str) -> tuple[str, ...]:
    if role == COMPLEX_ROLE:
        return ("P", "O_W", "H_W", "H_W")
    if role == PROJECTILE_COUNTERPOISE_ROLE:
        return ("P", "O_G", "H_G", "H_G")
    if role == WATER_COUNTERPOISE_ROLE:
        return ("P_G", "O_W", "H_W", "H_W")
    raise ValueError(f"Unsupported CP2K task role {role!r}.")


def render_kind_section(
    label: str,
    element: str,
    settings: CP2KSettings,
    *,
    atomic_guess: dict[str, list[dict[str, int]]] | None = None,
) -> str:
    ghost = label.endswith("_G")
    ghost_line = "\n      GHOST TRUE" if ghost else ""
    basis_set = (
        settings.projectile_basis_set
        if label in {"P", "P_G"}
        else settings.water_basis_set
    )
    broken_symmetry = ""
    if atomic_guess and any(atomic_guess.get(spin) for spin in ("alpha", "beta")):
        # CP2K 2025.2 requires ALPHA and BETA N/L/NEL arrays to have the
        # same length (qs_kind_types.F). Store only nonzero changes in the
        # registry, then pad the shared orbital list explicitly here.
        orbitals = list(
            dict.fromkeys(
                (value["n"], value["l"])
                for spin in ("alpha", "beta")
                for value in atomic_guess.get(spin, [])
            )
        )
        spin_sections = []
        for spin in ("alpha", "beta"):
            changes = {
                (value["n"], value["l"]): value["nel"]
                for value in atomic_guess.get(spin, [])
            }
            spin_sections.append(
                f"""        &{spin.upper()}
          N {' '.join(str(n) for n, _ in orbitals)}
          L {' '.join(str(angular_momentum) for _, angular_momentum in orbitals)}
          NEL {' '.join(str(changes.get(orbital, 0)) for orbital in orbitals)}
        &END {spin.upper()}"""
            )
        broken_symmetry = (
            "\n      &BS ON\n" + "\n".join(spin_sections) + "\n      &END BS"
        )
    return f"""    &KIND {label}
      ELEMENT {element}
      BASIS_SET {basis_set}
      POTENTIAL {settings.potential}
      LEBEDEV_GRID 110
      RADIAL_GRID 80{ghost_line}{broken_symmetry}
    &END KIND"""


def render_cdft_section(
    target_electrons: int,
    settings: CP2KSettings,
    *,
    strength: float | None,
    mode: str = CDFT_MODE_OPTIMIZED,
    step_size: float | None = None,
    output_prefix: str = "./cdft",
) -> str:
    if strength is None:
        raise ValueError("The CDFT constraint strength must be explicit.")
    constraint_strength = float(strength)
    if not math.isfinite(constraint_strength):
        raise ValueError("The initial CDFT constraint strength must be finite.")
    if not output_prefix.strip():
        raise ValueError("The CDFT output prefix cannot be empty.")
    if mode not in CDFT_MODES:
        raise ValueError(f"Unsupported CDFT render mode {mode!r}.")

    if mode == CDFT_MODE_FIXED_LAMBDA:
        if step_size is not None:
            raise ValueError("A fixed-lambda diagnostic does not use STEP_SIZE.")
        # CP2K 2025.2's CDFT driver evaluates the source-defined STRENGTH
        # without optimizing it when the CDFT OUTER_SCF MAX_SCF is zero.  The
        # CDFT section must remain active so the reported population is the
        # constrained population at that lambda, not an unconstrained SCF.
        outer_scf = f"""
        &OUTER_SCF ON
          TYPE CDFT_CONSTRAINT
          EPS_SCF {settings.cdft_eps:.12g}
          MAX_SCF 0
          OPTIMIZER {settings.cdft_optimizer}
        &END OUTER_SCF"""
    else:
        if step_size is None:
            raise ValueError("Optimized CDFT requires an explicit STEP_SIZE.")
        constraint_step_size = float(step_size)
        if not math.isfinite(constraint_step_size) or constraint_step_size == 0.0:
            raise ValueError("CDFT STEP_SIZE must be finite and nonzero.")
        optimizer_method = settings.cdft_optimizer
        optimizer_options = (
            """
          &CDFT_OPT ON
            MAX_LS 5
            CONTINUE_LS
            FACTOR_LS 0.5
            JACOBIAN_STEP 1.0E-2
            JACOBIAN_FREQ 1 1
            JACOBIAN_TYPE FD1
            JACOBIAN_RESTART FALSE
          &END CDFT_OPT"""
            if optimizer_method == "NEWTON_LS"
            else "\n          BISECT_TRUST_COUNT 10"
        )
        outer_scf = f"""
        &OUTER_SCF ON
          TYPE CDFT_CONSTRAINT
          EPS_SCF {settings.cdft_eps:.12g}
          MAX_SCF {settings.cdft_max:d}
          EXTRAPOLATION_ORDER 2
          OPTIMIZER {optimizer_method}
          STEP_SIZE {constraint_step_size:.16g}
{optimizer_options}
        &END OUTER_SCF"""
    if settings.cdft_constraint_type == "HIRSHFELD":
        weight_function = """        &HIRSHFELD_CONSTRAINT
          SHAPE_FUNCTION DENSITY
        &END HIRSHFELD_CONSTRAINT"""
    else:
        weight_function = """        &BECKE_CONSTRAINT
          ADJUST_SIZE FALSE
          IN_MEMORY TRUE
          CAVITY_CONFINE FALSE
        &END BECKE_CONSTRAINT"""
    return f"""
      &CDFT
        TYPE_OF_CONSTRAINT {settings.cdft_constraint_type}
        ATOMIC_CHARGES TRUE
        STRENGTH {constraint_strength:.16g}
        TARGET {target_electrons:d}
        &ATOM_GROUP
          ATOMS 1
          COEFF 1
          CONSTRAINT_TYPE CHARGE
        &END ATOM_GROUP
        &DUMMY_ATOMS
          ATOMS 2..4
        &END DUMMY_ATOMS
{outer_scf}
{weight_function}
        &PROGRAM_RUN_INFO ON
          &EACH
            QS_SCF 1
          &END EACH
          COMMON_ITERATION_LEVELS 2
          ADD_LAST NUMERIC
          FILENAME {output_prefix}
        &END PROGRAM_RUN_INFO
      &END CDFT"""


def render_molecular_subsys(
    coordinates: list[list[object]] | tuple[tuple[object, ...], ...],
    settings: CP2KSettings,
    *,
    role: str = COMPLEX_ROLE,
    projectile_atomic_guess: dict[str, list[dict[str, int]]] | None = None,
) -> str:
    """Render the centered projectile--H2O subsystem shared by QS and MIXED."""

    if role not in ROLES:
        raise ValueError(f"Unsupported CP2K task role {role!r}.")
    if len(coordinates) != 4:
        raise ValueError("The molecular pilot requires projectile + O + H + H.")

    raw = np.asarray([[row[1], row[2], row[3]] for row in coordinates], dtype=float)
    if raw.shape != (4, 3) or not np.all(np.isfinite(raw)):
        raise ValueError("Invalid scan coordinates.")
    centered = raw - 0.5 * (raw.min(axis=0) + raw.max(axis=0))
    centered += 0.5 * settings.cell_angstrom
    margin = np.minimum(
        centered.min(axis=0), settings.cell_angstrom - centered.max(axis=0)
    )
    if np.any(margin <= 5.0):
        raise ValueError(
            "The molecular density would have less than 5 A of cell margin; "
            "increase the nonperiodic cell."
        )

    labels = _coordinate_kinds(role)
    coordinate_lines = []
    kinds: dict[str, str] = {}
    if not (len(labels) == len(coordinates) == len(centered)):
        raise RuntimeError("CP2K coordinate-kind arrays have inconsistent lengths.")
    for label, source, xyz in zip(labels, coordinates, centered):
        element = str(source[0])
        kinds[label] = element
        coordinate_lines.append(
            f"      {label:<4s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}"
        )
    kind_sections = "\n".join(
        render_kind_section(
            label,
            element,
            settings,
            atomic_guess=(
                projectile_atomic_guess
                if role == COMPLEX_ROLE and label == "P"
                else None
            ),
        )
        for label, element in kinds.items()
    )
    cell = " ".join(3 * (f"{settings.cell_angstrom:.12g}",))
    return f"""  &SUBSYS
    &CELL
      ABC {cell}
      PERIODIC NONE
    &END CELL
    &COORD
{chr(10).join(coordinate_lines)}
    &END COORD
{kind_sections}
  &END SUBSYS"""


def render_qs_force_eval(
    task: dict[str, Any],
    settings: CP2KSettings,
    *,
    cdft_strength: float | None = None,
    cdft_mode: str = CDFT_MODE_OPTIMIZED,
    cdft_step_size: float | None = None,
    cdft_output_prefix: str = "./cdft",
    wavefunction_restart: str | None = None,
    force_output_prefix: str = "./forces",
    density_cube_stride: int | None = None,
    enable_cdft: bool = True,
) -> str:
    """Render one QS force evaluation, optionally restarting a CDFT state."""

    role = str(task["role"])
    if role not in ROLES:
        raise ValueError(f"Unsupported CP2K task role {role!r}.")
    coordinates = task["coordinates_angstrom"]
    if role == WATER_COUNTERPOISE_ROLE:
        charge = 0
        multiplicity = 1
    else:
        charge = int(task["charge"])
        multiplicity = int(task["multiplicity"])
    if multiplicity < 1:
        raise ValueError("The total spin multiplicity must be positive.")
    scf_spin_mode = str(task.get("scf_spin_mode", "UNRESTRICTED")).upper()
    if scf_spin_mode not in {"RESTRICTED", "UNRESTRICTED"}:
        raise ValueError(
            "scf_spin_mode must be either 'RESTRICTED' or 'UNRESTRICTED'."
        )
    if scf_spin_mode == "RESTRICTED" and multiplicity != 1:
        raise ValueError("Restricted Kohn-Sham mode requires multiplicity 1.")
    uks = scf_spin_mode == "UNRESTRICTED"
    if not force_output_prefix.strip():
        raise ValueError("The force output prefix cannot be empty.")
    if density_cube_stride is not None and density_cube_stride < 1:
        raise ValueError("Density-cube stride must be positive when requested.")

    cdft = (
        render_cdft_section(
            int(task["electrons_on_projectile"]),
            settings,
            strength=cdft_strength,
            mode=cdft_mode,
            step_size=cdft_step_size,
            output_prefix=cdft_output_prefix,
        )
        if role == COMPLEX_ROLE and enable_cdft
        else ""
    )
    if role == COMPLEX_ROLE:
        ot_minimizer = settings.ot_minimizer
        ot_linesearch = settings.ot_linesearch
        ot_algorithm = settings.ot_algorithm
    else:
        ot_minimizer = settings.counterpoise_ot_minimizer
        ot_linesearch = settings.counterpoise_ot_linesearch
        ot_algorithm = settings.counterpoise_ot_algorithm
    scf_solver_name = {
        COMPLEX_ROLE: settings.complex_scf_solver,
        WATER_COUNTERPOISE_ROLE: settings.water_counterpoise_scf_solver,
        PROJECTILE_COUNTERPOISE_ROLE: (
            settings.projectile_counterpoise_scf_solver
        ),
    }[role]
    inner_scf_max = (
        settings.complex_ot_inner_scf_max
        if role == COMPLEX_ROLE and scf_solver_name == "OT"
        else settings.scf_max
    )
    if scf_solver_name == "DIAGONALIZATION":
        if role == COMPLEX_ROLE:
            mixing_alpha = settings.complex_mixing_alpha
            mixing_npulay = settings.complex_mixing_npulay
            mixing_method = settings.complex_mixing_method
        else:
            mixing_alpha = settings.counterpoise_mixing_alpha
            mixing_npulay = settings.counterpoise_mixing_npulay
            mixing_method = settings.counterpoise_mixing_method
        mixing_history = (
            f"\n        NPULAY {mixing_npulay:d}"
            if mixing_method != "DIRECT_P_MIXING"
            else ""
        )
        scf_solver = f"""&DIAGONALIZATION ON
        ALGORITHM STANDARD
      &END DIAGONALIZATION
      &MIXING ON
        METHOD {mixing_method}
        ALPHA {mixing_alpha:.12g}{mixing_history}
      &END MIXING"""
        if role == COMPLEX_ROLE:
            scf_solver += f"""
      &OUTER_SCF ON
        EPS_SCF {settings.scf_eps:.12g}
        MAX_SCF 10
      &END OUTER_SCF"""
    else:
        energy_gap = (
            f"\n        ENERGY_GAP {settings.ot_energy_gap_hartree:.12g}"
            if role == COMPLEX_ROLE and settings.ot_preconditioner == "FULL_ALL"
            else ""
        )
        preconditioner = (
            settings.ot_preconditioner
            if role == COMPLEX_ROLE
            else "FULL_ALL"
        )
        scf_solver = f"""&OT ON
        MINIMIZER {ot_minimizer}
        LINESEARCH {ot_linesearch}
        PRECONDITIONER {preconditioner}{energy_gap}
        ALGORITHM {ot_algorithm}
      &END OT
      &OUTER_SCF ON
        EPS_SCF {settings.scf_eps:.12g}
        MAX_SCF 10
      &END OUTER_SCF"""
    restart = ""
    scf_guess = "ATOMIC"
    if wavefunction_restart is not None:
        if not str(wavefunction_restart).strip():
            raise ValueError("Wavefunction restart path cannot be empty.")
        scf_guess = "RESTART"
        restart = f"\n      WFN_RESTART_FILE_NAME {wavefunction_restart}"
    subsys = render_molecular_subsys(
        coordinates,
        settings,
        role=role,
        projectile_atomic_guess=task.get("cp2k_atomic_guess"),
    )
    density_cube = ""
    if density_cube_stride is not None:
        density_cube = f"""
      &E_DENSITY_CUBE ON
        DENSITY_INCLUDE TOTAL_HARD_APPROX
        STRIDE {density_cube_stride:d} {density_cube_stride:d} {density_cube_stride:d}
        ADD_LAST NUMERIC
        FILENAME ./density
      &END E_DENSITY_CUBE"""
    return f"""&FORCE_EVAL
  METHOD QUICKSTEP
  &DFT
    BASIS_SET_FILE_NAME {settings.basis_file}
    CHARGE {charge:d}
    MULTIPLICITY {multiplicity:d}
    UKS {'TRUE' if uks else 'FALSE'}{restart}
    &QS
      METHOD {settings.method}
      EPS_DEFAULT 1.0E-12{cdft}
    &END QS
    &MGRID
      CUTOFF {settings.mgrid_cutoff_ry:.12g}
      REL_CUTOFF {settings.mgrid_rel_cutoff_ry:.12g}
      NGRIDS 5
    &END MGRID
    &POISSON
      PERIODIC NONE
      POISSON_SOLVER WAVELET
    &END POISSON
    &SCF
      SCF_GUESS {scf_guess}
      EPS_SCF {settings.scf_eps:.12g}
      MAX_SCF {inner_scf_max:d}
      {scf_solver}
    &END SCF
    &XC
      &XC_FUNCTIONAL {settings.xc_functional}
      &END XC_FUNCTIONAL
    &END XC
    &PRINT
      &MULLIKEN ON
      &END MULLIKEN
      &LOWDIN ON
      &END LOWDIN
{density_cube}
    &END PRINT
  &END DFT
{subsys}
  &PRINT
    &FORCES ON
      FILENAME {force_output_prefix}
    &END FORCES
  &END PRINT
&END FORCE_EVAL"""


def render_cp2k_input(
    task: dict[str, Any],
    settings: CP2KSettings,
    *,
    cdft_strength: float | None = None,
    cdft_mode: str = CDFT_MODE_OPTIMIZED,
    cdft_step_size: float | None = None,
    wavefunction_restart: str | None = None,
    density_cube_stride: int | None = None,
    enable_cdft: bool = True,
) -> str:
    """Render one complete, nonperiodic GAPW input without hidden defaults."""

    role = str(task["role"])
    if role not in ROLES:
        raise ValueError(f"Unsupported CP2K task role {role!r}.")
    project = re.sub(r"[^A-Za-z0-9_-]", "_", str(task["task_id"]))[:100]
    if not project:
        raise ValueError("Task ID produced an empty CP2K project name.")
    force_eval = render_qs_force_eval(
        task,
        settings,
        cdft_strength=cdft_strength,
        cdft_mode=cdft_mode,
        cdft_step_size=cdft_step_size,
        wavefunction_restart=wavefunction_restart,
        density_cube_stride=density_cube_stride,
        enable_cdft=enable_cdft,
    )
    return f"""# Generated by soft_dft; pilot data remain validation_pending.
&GLOBAL
  PROJECT {project}
  RUN_TYPE ENERGY_FORCE
  PRINT_LEVEL MEDIUM
&END GLOBAL

{force_eval}
"""


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_ENERGY_RE = re.compile(
    rf"^[ \t]*ENERGY\|[^\r\n]*?\benergy\b[^\r\n]*?({_FLOAT})[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_VERSION_RE = re.compile(
    r"CP2K\|\s+version string:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
_SOURCE_REVISION_RE = re.compile(
    r"CP2K\|\s+source code revision number:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ELECTRON_COUNT_RE = re.compile(
    r"^[ \t]*Number of electrons:\s*(\d+)\s*$", re.MULTILINE
)
_SPIN_SQUARED_RE = re.compile(
    rf"Ideal and single determinant S\*\*2\s*:\s*({_FLOAT})\s+({_FLOAT})",
    re.IGNORECASE,
)
_CDFT_ITERATION_RE = re.compile(
    rf"CDFT\s+SCF\s+iter\s*=\s*(\d+)[^\r\n]*?"
    rf"\benergy\s*=\s*({_FLOAT})",
    re.IGNORECASE,
)
_SCF_STATUS_RE = re.compile(
    r"SCF run\s+(NOT\s+)?converged", re.IGNORECASE
)
_SCF_START_RE = re.compile(
    r"SCF WAVEFUNCTION OPTIMIZATION", re.IGNORECASE
)
_CDFT_OUTER_CONVERGED_RE = re.compile(
    r"CDFT SCF loop converged", re.IGNORECASE
)
_PROGRAM_ENDED_RE = re.compile(r"PROGRAM ENDED AT", re.IGNORECASE)


def _last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None
    return float(str(matches[-1]).replace("D", "E").replace("d", "e"))


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _last_inner_scf_status(text: str) -> bool | None:
    """Return the final completed inner-SCF status, not any earlier success."""

    status_matches = list(_SCF_STATUS_RE.finditer(text))
    if not status_matches:
        return None
    start_matches = list(_SCF_START_RE.finditer(text))
    if start_matches and start_matches[-1].start() > status_matches[-1].start():
        # A later inner solve started but has not printed a terminal status.
        return None
    return status_matches[-1].group(1) is None


def _parse_cdft_trace(text: str) -> list[dict[str, Any]]:
    """Parse CP2K's CDFT outer iterations in their emitted order."""

    markers = list(_CDFT_ITERATION_RE.finditer(text))
    scf_events = list(_SCF_STATUS_RE.finditer(text))
    trace: list[dict[str, Any]] = []
    previous_marker_end = 0
    for index, marker in enumerate(markers):
        block_end = (
            markers[index + 1].start() if index + 1 < len(markers) else len(text)
        )
        block = text[marker.end() : block_end]
        preceding_statuses = [
            event
            for event in scf_events
            if previous_marker_end <= event.start() < marker.start()
        ]
        inner_converged = (
            preceding_statuses[-1].group(1) is None if preceding_statuses else None
        )
        trace.append(
            {
                "iteration": int(marker.group(1)),
                "energy": _float(marker.group(2)),
                "target": _last_float(
                    rf"Target value of constraint\s*:\s*({_FLOAT})", block
                ),
                "current": _last_float(
                    rf"Current value of constraint\s*:\s*({_FLOAT})", block
                ),
                "residual": _last_float(
                    rf"Deviation from target\s*:\s*({_FLOAT})", block
                ),
                "strength": _last_float(
                    rf"Strength of constraint\s*:\s*({_FLOAT})", block
                ),
                "inner_scf_converged": inner_converged,
            }
        )
        previous_marker_end = marker.end()
    return trace


def parse_cp2k_output(
    text: str,
    *,
    require_cdft: bool,
    cdft_mode: str = CDFT_MODE_OPTIMIZED,
    cdft_tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    """Parse CP2K completion gates and the ordered CDFT branch trace.

    Optimized CDFT is accepted only when the final inner SCF, the CDFT outer
    loop, the requested population tolerance, and CP2K's normal termination
    all pass.  A fixed-lambda calculation is a diagnostic probe: it has a
    separate acceptance flag and can never be a valid constrained result.
    """

    if require_cdft and cdft_mode not in CDFT_MODES:
        raise ValueError(f"Unsupported CDFT parse mode {cdft_mode!r}.")
    if require_cdft and (
        not math.isfinite(cdft_tolerance) or cdft_tolerance <= 0.0
    ):
        raise ValueError("CDFT target tolerance must be finite and positive.")

    energy_matches = _ENERGY_RE.findall(text)
    energy = (
        _float(energy_matches[-1])
        if energy_matches
        else None
    )
    version_matches = _VERSION_RE.findall(text)
    revision_matches = _SOURCE_REVISION_RE.findall(text)
    electron_matches = _ELECTRON_COUNT_RE.findall(text)
    spin_squared_matches = _SPIN_SQUARED_RE.findall(text)
    electron_alpha = int(electron_matches[-2]) if len(electron_matches) >= 2 else None
    electron_beta = int(electron_matches[-1]) if len(electron_matches) >= 2 else None
    if spin_squared_matches:
        ideal_s2_text, determinant_s2_text = spin_squared_matches[-1]
        ideal_s2 = _float(ideal_s2_text)
        determinant_s2 = _float(determinant_s2_text)
    else:
        ideal_s2 = determinant_s2 = None
    last_inner_scf_converged = _last_inner_scf_status(text)
    normal_end = _PROGRAM_ENDED_RE.search(text) is not None
    trace = _parse_cdft_trace(text) if require_cdft else []
    last_cdft = trace[-1] if trace else None
    outer_matches = list(_CDFT_OUTER_CONVERGED_RE.finditer(text))
    cdft_outer_converged = bool(
        require_cdft
        and trace
        and outer_matches
        and outer_matches[-1].start()
        >= list(_CDFT_ITERATION_RE.finditer(text))[-1].start()
    )
    residual = last_cdft["residual"] if last_cdft is not None else None
    cdft_target_met = bool(
        require_cdft
        and residual is not None
        and math.isfinite(residual)
        and abs(residual) <= cdft_tolerance
    )
    finite_energy = energy is not None and math.isfinite(energy)
    complete_probe_record = bool(
        last_cdft is not None
        and all(
            value is not None and math.isfinite(value)
            for value in (
                last_cdft["energy"],
                last_cdft["target"],
                last_cdft["current"],
                last_cdft["residual"],
                last_cdft["strength"],
            )
        )
    )
    fixed_lambda_probe_accepted = bool(
        require_cdft
        and cdft_mode == CDFT_MODE_FIXED_LAMBDA
        and finite_energy
        and last_inner_scf_converged is True
        and complete_probe_record
        and normal_end
    )
    result: dict[str, Any] = {
        "energy_hartree": energy,
        "cp2k_version": (
            version_matches[-1].strip() if version_matches else None
        ),
        "cp2k_source_revision": (
            revision_matches[-1].strip() if revision_matches else None
        ),
        "last_inner_scf_converged": last_inner_scf_converged,
        # Compatibility aliases retained for existing collectors.
        "scf_converged": last_inner_scf_converged is True,
        "normal_end": normal_end,
        "program_ended": normal_end,
        "electron_count_alpha": electron_alpha,
        "electron_count_beta": electron_beta,
        "spin_squared_ideal": ideal_s2,
        "spin_squared_single_determinant": determinant_s2,
        "cdft_mode": cdft_mode if require_cdft else None,
        "cdft_trace": trace,
        "cdft_outer_converged": cdft_outer_converged if require_cdft else None,
        "cdft_target_met": cdft_target_met if require_cdft else None,
        "fixed_lambda_probe_accepted": fixed_lambda_probe_accepted,
    }
    if require_cdft:
        result.update(
            {
                "cdft_converged": cdft_outer_converged,
                "cdft_target_electrons": (
                    last_cdft["target"] if last_cdft is not None else None
                ),
                "cdft_current_electrons": (
                    last_cdft["current"] if last_cdft is not None else None
                ),
                "cdft_deviation_electrons": residual,
                "cdft_strength": (
                    last_cdft["strength"] if last_cdft is not None else None
                ),
            }
        )
    else:
        result["cdft_converged"] = None
    result["valid_completion"] = bool(
        finite_energy
        and last_inner_scf_converged is True
        and normal_end
        and (
            not require_cdft
            or (
                cdft_mode == CDFT_MODE_OPTIMIZED
                and cdft_outer_converged
                and cdft_target_met
            )
        )
    )
    return result
