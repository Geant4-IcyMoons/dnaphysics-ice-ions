#!/usr/bin/env python3
"""Replay identified saved histories with the original git-pinned NLH runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
from tqdm import tqdm

from audit_variance import atomic_json, digest, signed_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("--legacy-commit", required=True)
    parser.add_argument("--case", action="append", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    snapshot = json.loads((args.diagnostic / "snapshot.json").read_text())
    root = Path(snapshot["request"]["root"])
    commit = subprocess.check_output(["git", "rev-parse", "--verify", args.legacy_commit + "^{commit}"], cwd=repo, text=True).strip()
    request = {"legacy_commit": commit, "replay_code_sha256": digest(Path(__file__).read_bytes()),
               "snapshot_sha256": digest((args.diagnostic / "snapshot.json").read_bytes())}
    prefix = "python_scripts/physics_ice/nep_mbpol"
    archive = subprocess.check_output(["git", "archive", commit, prefix + "/bca", prefix + "/nlh",
                                       prefix + "/ion_ice", prefix + "/simulate_nlh_hard_collisions.py"], cwd=repo)
    # Temporary copies are exclusively tracked source at the requested commit.
    with tempfile.TemporaryDirectory(prefix="nlh-tail-replay-") as temporary:
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(temporary, filter="data")
        runtime_root = Path(temporary) / prefix
        spec = importlib.util.spec_from_file_location("_nlh_replay_runtime", runtime_root / "simulate_nlh_hard_collisions.py")
        runtime = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runtime
        spec.loader.exec_module(runtime)
        from bca.scattering import NLHCollisionKernel

        for case in tqdm(args.case, desc="Replayed saved tail histories", unit="history"):
            frozen = next(c for c in snapshot["cases"] if c["case"] == case)
            report = json.loads((args.diagnostic / "cases" / (case.replace("/", "__") + ".json")).read_text())
            cfg, configuration_hash = signed_configuration(root, frozen)
            if cfg["trajectory_implementation_version"] != runtime.TRAJECTORY_IMPLEMENTATION_VERSION:
                raise ValueError("Replay runtime version does not match checkpoint")
            sample_record = max(report["uniform_tail_diagnostics"], key=lambda s: s["recoil_ev"]["top"][0]["value"])
            selected = sample_record["recoil_ev"]["top"][0]
            index = selected["trajectory"]
            output = args.diagnostic / "replays" / f"{case.replace('/', '__')}__{index}.json"
            identity = {**request, "case": case, "trajectory": index,
                        "configuration_sha256": configuration_hash, "distribution_sha256": sample_record["sha256"]}
            if output.exists():
                if json.loads(output.read_text())["identity"] != identity:
                    raise ValueError(f"Incompatible replay resume: {output}")
                continue
            checkpoint = (root / frozen["batches"][0][0]).parent
            data = (checkpoint / sample_record["sample"]).read_bytes()
            if digest(data) != sample_record["sha256"]:
                raise ValueError("Saved sample checksum mismatch")
            with np.load(io.BytesIO(data), allow_pickle=False) as saved:
                position = int(np.flatnonzero(saved["trajectory"] == index)[0])
                expected_recoil = float(saved["total_recoil_energy_ev"][position])
                expected_angle = float(saved["final_deflection_rad"][position])
            calibration = json.loads((root / "estimator_calibration" / case / "hard_collision_run.manifest.json").read_text())
            runtime._initialize_worker(calibration["structure"]["path"], None, calibration["kernel_manifest"], False, cfg["search_window_angstrom"])
            transport = runtime._WORKER_TRANSPORT
            if (runtime._WORKER_STRUCTURE.source_sha256 != cfg["structure_sha256"]
                    or runtime._WORKER_STRUCTURE.frame_index != cfg["structure_frame_index"]
                    or transport.kernels.csv_sha256 != cfg["kernel_csv_sha256"]):
                raise ValueError("Replay physical input mismatch")
            _, result = runtime._run_one((index, cfg["seed"], cfg["projectile"], cfg["projectile_energy_ev"],
                                         cfg["path_length_angstrom"], cfg["fixed_direction"], cfg["max_collisions"],
                                         cfg["control_variate"], cfg["initial_condition_sampling"], cfg["tube_mixture_fraction"]))
            angle = math.acos(min(1., max(-1., float(np.dot(result.initial_direction, result.final_direction)))))
            reference = result.control_variate or transport.straight_line_control_variate(
                cfg["projectile"], cfg["projectile_energy_ev"], result.initial_position_angstrom,
                result.initial_direction, cfg["path_length_angstrom"])
            dominant = max(result.events, key=lambda event: event.recoil_energy_ev)
            displacement = np.array(dominant.target_position_angstrom) - result.initial_position_angstrom
            initial_direction = np.array(result.initial_direction)
            initial_line_impact = float(np.linalg.norm(displacement - np.dot(displacement, initial_direction) * initial_direction))
            quadrature = []
            for order in (96, 192, 384):
                direct = NLHCollisionKernel(cfg["projectile"], dominant.target, dominant.projectile_energy_in_ev,
                                            minimum_turning_potential_ev=transport.kernels.minimum_turning_potential_ev,
                                            quadrature_order=order).solve(dominant.impact_parameter_angstrom)
                quadrature.append({"order": order, **asdict(direct)})
            payload = {"identity": identity, "diagnostic_only": True, "new_independent_samples": 0,
                       "numpy_version": np.__version__, "python_version": sys.version,
                       "saved_recoil_energy_ev": expected_recoil, "replayed_recoil_energy_ev": result.recoil_energy_ev,
                       "saved_deflection_rad": expected_angle, "replayed_deflection_rad": angle,
                       "recoil_relative_replay_difference": (result.recoil_energy_ev - expected_recoil) / expected_recoil,
                       "deflection_absolute_replay_difference": angle - expected_angle,
                       "importance_sampling": asdict(result.importance_sampling),
                       "straight_line_reference": asdict(reference),
                       "dominant_event_initial_line_impact_angstrom": initial_line_impact,
                       "dominant_event": asdict(dominant), "direct_quadrature": quadrature,
                       "trajectory": asdict(result)}
            atomic_json(output, payload)
            print(f"{case}: history {index}; recoil {result.recoil_energy_ev:g} eV; replay relative difference {payload['recoil_relative_replay_difference']:.3g}")


if __name__ == "__main__":
    main()
