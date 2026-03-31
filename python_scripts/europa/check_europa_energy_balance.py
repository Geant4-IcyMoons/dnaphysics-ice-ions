#!/usr/bin/env python3
"""
Per-macro energy budget check for Europa per-energy runs.

Two modes are supported automatically:

1) Exact budget (preferred): uses ROOT "event" tree columns written by
   dnaphysics-ice when available.
2) Deposition-only fallback: uses only step.totalEnergyDeposit for older files.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import uproot


EVENT_FIELDS = [
    "eventID",
    "primaryEnergy",
    "depositedEnergy",
    "escapedEnergy",
    "escapedBackEnergy",
    "escapedForwardEnergy",
    "escapedLateralEnergy",
    "closureEnergy",
    "nEscapedTracks",
    "nEscapedElectrons",
]


def _candidate_root_files(root_file: Path) -> list[Path]:
    parent = root_file.parent
    stem = root_file.stem
    return sorted(p for p in parent.glob(f"{stem}*.root") if p.is_file())


def _sum_edep_and_events(
    root_files: Iterable[Path],
    step_size: str,
) -> tuple[float, int | None, int]:
    total_edep_eV = 0.0
    event_ids: set[int] = set()
    total_entries = 0
    event_id_seen = False

    for path in root_files:
        with uproot.open(path) as rootf:
            if "step" not in rootf:
                raise KeyError(f"'step' tree missing in {path}")
            tree = rootf["step"]
            needed = ["totalEnergyDeposit"]
            if "eventID" in tree.keys():
                needed.append("eventID")
                event_id_seen = True
            for chunk in tree.iterate(needed, library="np", step_size=step_size):
                edep = np.asarray(chunk["totalEnergyDeposit"], dtype=np.float64)
                total_edep_eV += float(np.sum(edep))
                total_entries += int(edep.size)
                if event_id_seen and "eventID" in chunk:
                    event_ids.update(np.asarray(chunk["eventID"], dtype=np.int64).tolist())

    return total_edep_eV, (len(event_ids) if event_id_seen else None), total_entries


def _sum_event_budget(
    root_files: Iterable[Path],
    step_size: str,
) -> dict[str, float] | None:
    n_events = 0
    primary_eV = 0.0
    deposited_eV = 0.0
    escaped_eV = 0.0
    escaped_back_eV = 0.0
    escaped_forward_eV = 0.0
    escaped_lateral_eV = 0.0
    closure_eV = 0.0
    escaped_tracks = 0
    escaped_electrons = 0
    saw_event_tree = False
    saw_required_cols = False

    for path in root_files:
        with uproot.open(path) as rootf:
            if "event" not in rootf:
                continue
            saw_event_tree = True
            tree = rootf["event"]
            cols = list(tree.keys())
            if not all(name in cols for name in EVENT_FIELDS):
                continue
            saw_required_cols = True
            for chunk in tree.iterate(EVENT_FIELDS, library="np", step_size=step_size):
                n_events += int(np.asarray(chunk["eventID"]).size)
                primary_eV += float(np.sum(np.asarray(chunk["primaryEnergy"], dtype=np.float64)))
                deposited_eV += float(np.sum(np.asarray(chunk["depositedEnergy"], dtype=np.float64)))
                escaped_eV += float(np.sum(np.asarray(chunk["escapedEnergy"], dtype=np.float64)))
                escaped_back_eV += float(np.sum(np.asarray(chunk["escapedBackEnergy"], dtype=np.float64)))
                escaped_forward_eV += float(np.sum(np.asarray(chunk["escapedForwardEnergy"], dtype=np.float64)))
                escaped_lateral_eV += float(np.sum(np.asarray(chunk["escapedLateralEnergy"], dtype=np.float64)))
                closure_eV += float(np.sum(np.asarray(chunk["closureEnergy"], dtype=np.float64)))
                escaped_tracks += int(np.sum(np.asarray(chunk["nEscapedTracks"], dtype=np.int64)))
                escaped_electrons += int(np.sum(np.asarray(chunk["nEscapedElectrons"], dtype=np.int64)))

    if not saw_event_tree or not saw_required_cols or n_events == 0:
        return None

    return {
        "n_events": float(n_events),
        "primary_eV": primary_eV,
        "deposited_eV": deposited_eV,
        "escaped_eV": escaped_eV,
        "escaped_back_eV": escaped_back_eV,
        "escaped_forward_eV": escaped_forward_eV,
        "escaped_lateral_eV": escaped_lateral_eV,
        "closure_eV": closure_eV,
        "escaped_tracks": float(escaped_tracks),
        "escaped_electrons": float(escaped_electrons),
    }


def _float_or_nan(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _fmt_mev(e_ev: float) -> str:
    return f"{(e_ev / 1e6):.9g}" if math.isfinite(e_ev) else "nan"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Check per-energy energy budget from Europa ROOT files."
    )
    ap.add_argument(
        "--run-table",
        type=Path,
        default=Path("europa_energy_library/per_energy_runs.csv"),
        help="Path to per_energy_runs.csv",
    )
    ap.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help=(
            "Optional root directory override. "
            "If omitted, paths from root_relpath are used."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path. Default: <run-table-dir>/energy_balance_summary.csv",
    )
    ap.add_argument(
        "--step-size",
        default="50 MB",
        help="Chunk size passed to uproot.iterate (default: 50 MB).",
    )
    ap.add_argument(
        "--overshoot-tol",
        type=float,
        default=1e-6,
        help="Relative tolerance for flagging E_deposited > E_in_expected.",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional max number of rows/files to process.",
    )
    args = ap.parse_args()

    run_table = args.run_table.resolve()
    if not run_table.exists():
        raise FileNotFoundError(f"Run table not found: {run_table}")

    out_csv = args.out
    if out_csv is None:
        out_csv = run_table.parent / "energy_balance_summary.csv"
    out_csv = out_csv.resolve()

    rows_out: list[dict[str, str]] = []
    n_ok = 0
    n_missing = 0
    n_missing_tree = 0
    n_over_budget = 0
    n_other_errors = 0
    n_exact_mode = 0
    n_fallback_mode = 0
    sum_input_eV = 0.0
    sum_deposited_eV = 0.0
    sum_escaped_eV = 0.0
    sum_closure_eV = 0.0

    with run_table.open("r", newline="") as fh:
        rd = csv.DictReader(fh)
        for idx, row in enumerate(rd):
            if args.max_files is not None and idx >= args.max_files:
                break

            energy_index = row.get("energy_index", "")
            e_center_mev = _float_or_nan(row.get("E_center_MeV", "nan"))
            sim_particles = int(float(row.get("sim_particles", "0")))
            e_in_expected_eV = e_center_mev * 1.0e6 * sim_particles

            if args.root_dir is not None:
                base = row.get("root_basename", "")
                root_file = args.root_dir / f"{base}.root"
            else:
                rel = row.get("root_relpath", "")
                root_file = run_table.parent / rel

            candidates = _candidate_root_files(root_file)
            status = "ok"
            budget_mode = ""
            edep_eV = float("nan")
            escaped_eV = float("nan")
            escaped_back_eV = float("nan")
            escaped_forward_eV = float("nan")
            escaped_lateral_eV = float("nan")
            closure_eV = float("nan")
            n_escaped_tracks = None
            n_escaped_electrons = None
            n_events_in_step = None
            n_step_entries = 0
            error_msg = ""

            if not candidates:
                status = "missing_root"
                n_missing += 1
            else:
                try:
                    event_budget = _sum_event_budget(candidates, args.step_size)
                    if event_budget is not None:
                        budget_mode = "exact_event_tree"
                        n_exact_mode += 1
                        edep_eV = float(event_budget["deposited_eV"])
                        escaped_eV = float(event_budget["escaped_eV"])
                        escaped_back_eV = float(event_budget["escaped_back_eV"])
                        escaped_forward_eV = float(event_budget["escaped_forward_eV"])
                        escaped_lateral_eV = float(event_budget["escaped_lateral_eV"])
                        closure_eV = float(event_budget["closure_eV"])
                        n_events_in_step = int(event_budget["n_events"])
                        n_escaped_tracks = int(event_budget["escaped_tracks"])
                        n_escaped_electrons = int(event_budget["escaped_electrons"])
                    else:
                        budget_mode = "deposition_only"
                        n_fallback_mode += 1
                        edep_eV, n_events_in_step, n_step_entries = _sum_edep_and_events(
                            candidates, args.step_size
                        )

                    sum_input_eV += e_in_expected_eV
                    sum_deposited_eV += edep_eV
                    if math.isfinite(escaped_eV):
                        sum_escaped_eV += escaped_eV
                    if math.isfinite(closure_eV):
                        sum_closure_eV += closure_eV

                    if (
                        math.isfinite(e_in_expected_eV)
                        and e_in_expected_eV > 0.0
                        and edep_eV > e_in_expected_eV * (1.0 + args.overshoot_tol)
                    ):
                        status = "over_budget"
                        n_over_budget += 1
                    else:
                        n_ok += 1
                except KeyError as exc:
                    status = "missing_step_tree"
                    error_msg = str(exc)
                    n_missing_tree += 1
                except Exception as exc:  # pragma: no cover - defensive
                    status = "read_error"
                    error_msg = str(exc)
                    n_other_errors += 1

            dep_fraction = (
                edep_eV / e_in_expected_eV
                if math.isfinite(edep_eV) and math.isfinite(e_in_expected_eV) and e_in_expected_eV > 0.0
                else float("nan")
            )
            not_dep_eV = (
                e_in_expected_eV - edep_eV
                if math.isfinite(edep_eV) and math.isfinite(e_in_expected_eV)
                else float("nan")
            )
            dep_plus_escape_fraction = (
                (edep_eV + escaped_eV) / e_in_expected_eV
                if (
                    math.isfinite(edep_eV)
                    and math.isfinite(escaped_eV)
                    and math.isfinite(e_in_expected_eV)
                    and e_in_expected_eV > 0.0
                )
                else float("nan")
            )
            expected_closure_eV = (
                e_in_expected_eV - edep_eV - escaped_eV
                if (
                    math.isfinite(edep_eV)
                    and math.isfinite(escaped_eV)
                    and math.isfinite(e_in_expected_eV)
                )
                else float("nan")
            )
            event_coverage = (
                (float(n_events_in_step) / float(sim_particles))
                if (n_events_in_step is not None and sim_particles > 0)
                else float("nan")
            )

            rows_out.append(
                {
                    "energy_index": str(energy_index),
                    "E_center_MeV": f"{e_center_mev:.9g}",
                    "sim_particles": str(sim_particles),
                    "root_file_pattern": str(root_file),
                    "n_root_files": str(len(candidates)),
                    "budget_mode": budget_mode,
                    "E_in_expected_MeV": f"{(e_in_expected_eV / 1e6):.9g}",
                    "E_deposited_MeV": _fmt_mev(edep_eV),
                    "E_not_deposited_MeV": _fmt_mev(not_dep_eV),
                    "E_escaped_MeV": _fmt_mev(escaped_eV),
                    "E_escaped_back_MeV": _fmt_mev(escaped_back_eV),
                    "E_escaped_forward_MeV": _fmt_mev(escaped_forward_eV),
                    "E_escaped_lateral_MeV": _fmt_mev(escaped_lateral_eV),
                    "closure_from_event_tree_MeV": _fmt_mev(closure_eV),
                    "expected_closure_MeV": _fmt_mev(expected_closure_eV),
                    "deposited_fraction": (
                        f"{dep_fraction:.9g}" if math.isfinite(dep_fraction) else "nan"
                    ),
                    "deposited_plus_escaped_fraction": (
                        f"{dep_plus_escape_fraction:.9g}"
                        if math.isfinite(dep_plus_escape_fraction)
                        else "nan"
                    ),
                    "n_events": str(n_events_in_step) if n_events_in_step is not None else "",
                    "event_coverage_vs_sim_particles": (
                        f"{event_coverage:.9g}" if math.isfinite(event_coverage) else ""
                    ),
                    "n_escaped_tracks": str(n_escaped_tracks) if n_escaped_tracks is not None else "",
                    "n_escaped_electrons": str(n_escaped_electrons) if n_escaped_electrons is not None else "",
                    "n_step_entries": str(n_step_entries),
                    "status": status,
                    "error": error_msg,
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        wr = csv.DictWriter(
            fh,
            fieldnames=[
                "energy_index",
                "E_center_MeV",
                "sim_particles",
                "root_file_pattern",
                "n_root_files",
                "budget_mode",
                "E_in_expected_MeV",
                "E_deposited_MeV",
                "E_not_deposited_MeV",
                "E_escaped_MeV",
                "E_escaped_back_MeV",
                "E_escaped_forward_MeV",
                "E_escaped_lateral_MeV",
                "closure_from_event_tree_MeV",
                "expected_closure_MeV",
                "deposited_fraction",
                "deposited_plus_escaped_fraction",
                "n_events",
                "event_coverage_vs_sim_particles",
                "n_escaped_tracks",
                "n_escaped_electrons",
                "n_step_entries",
                "status",
                "error",
            ],
        )
        wr.writeheader()
        wr.writerows(rows_out)

    global_dep_fraction = (
        (sum_deposited_eV / sum_input_eV) if sum_input_eV > 0.0 else float("nan")
    )
    global_dep_plus_escape_fraction = (
        ((sum_deposited_eV + sum_escaped_eV) / sum_input_eV)
        if sum_input_eV > 0.0 and sum_escaped_eV > 0.0
        else float("nan")
    )
    global_closure_fraction = (
        (sum_closure_eV / sum_input_eV)
        if sum_input_eV > 0.0 and abs(sum_closure_eV) > 0.0
        else float("nan")
    )

    print(f"Wrote: {out_csv}")
    print(f"Processed rows: {len(rows_out)}")
    print(
        f"ok={n_ok}  missing_root={n_missing}  missing_step_tree={n_missing_tree}  "
        f"over_budget={n_over_budget}  read_error={n_other_errors}"
    )
    print(f"Budget modes: exact_event_tree={n_exact_mode}  deposition_only={n_fallback_mode}")
    if math.isfinite(global_dep_fraction):
        print(
            f"Global deposited/input fraction (rows successfully read): "
            f"{global_dep_fraction:.9g}"
        )
    if math.isfinite(global_dep_plus_escape_fraction):
        print(
            f"Global (deposited+escaped)/input fraction (exact-event rows): "
            f"{global_dep_plus_escape_fraction:.9g}"
        )
    if math.isfinite(global_closure_fraction):
        print(
            f"Global closure/input fraction from event tree sum: "
            f"{global_closure_fraction:.9g}"
        )


if __name__ == "__main__":
    main()
