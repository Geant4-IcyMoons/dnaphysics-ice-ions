#!/usr/bin/env python3
"""Extend one audited v3 case to its next absolute-width sampling target."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("wave_manifest", type=Path)
    parser.add_argument("legacy_nep_root", type=Path)
    parser.add_argument("--failed-case-index", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def main() -> int:
    args = parse_args()
    audit_path = args.audit.expanduser().resolve()
    wave_path = args.wave_manifest.expanduser().resolve()
    legacy_root = args.legacy_nep_root.expanduser().resolve()
    audit = _load_json(audit_path)
    wave = _load_json(wave_path)
    if Path(str(audit["campaign_root"])).resolve() != Path(
        str(wave["particle_root"])
    ).resolve():
        raise RuntimeError("Audit and wave refer to different campaigns.")
    failed = [case for case in audit["cases"] if not bool(case["converged"])]
    if not 0 <= args.failed_case_index < len(failed):
        raise ValueError(
            f"--failed-case-index must lie in 0-{len(failed) - 1}."
        )
    selected = failed[args.failed_case_index]
    target = selected.get("next_scheduled_look")
    if target is None:
        raise RuntimeError("The audit did not produce a finite scheduled target.")
    target = int(target)
    current = int(selected["checkpointed_trajectories"])
    case_key = str(selected["case_key"])
    if current >= target:
        print(f"{case_key} already has {current:,} trajectories; target {target:,}.")
        return 0
    wave_cases = {
        str(item["case_key"]): item for item in wave.get("cases", [])
    }
    if case_key not in wave_cases:
        raise RuntimeError(f"Audited case is absent from the wave: {case_key}")
    if not (legacy_root / "adaptive_nlh_particle_shards.py").is_file():
        raise RuntimeError(f"Legacy controller is missing below {legacy_root}")
    sys.path.insert(0, str(legacy_root))
    legacy_shards = importlib.import_module("adaptive_nlh_particle_shards")
    legacy_serial = legacy_shards.serial
    legacy_args = legacy_shards._args_from_wave(wave)
    workers = legacy_serial._worker_count(args.workers)
    case = legacy_shards._case_from_dict(wave_cases[case_key])
    root = Path(str(wave["particle_root"]))
    calibration = legacy_serial._calibrate_case(
        legacy_args, workers, root, case
    )
    print(
        f"Extending {case_key}: {current:,} -> {target:,} trajectories; "
        f"workers={workers}; estimator={calibration['selected']}"
    )
    if args.dry_run:
        return 0
    return_code, manifest, _ = legacy_serial._run_simulator(
        legacy_args,
        workers,
        root,
        case,
        stage="production",
        fixed_trajectories=target,
        control_variate=bool(calibration.get("use_control_variate", False)),
    )
    if return_code != 0:
        raise RuntimeError(
            f"Legacy fixed-target continuation failed with status {return_code}."
        )
    print(f"Completed fixed target: {manifest}")
    print(f"Re-run the absolute fixed-width audit before any further sampling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
