#!/usr/bin/env python3
"""Diagnose retained NLH checkpoint variance without changing acceptance or jobs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
from tqdm import tqdm


BATCH = re.compile(r"batch_(\d+)_(\d+)\.manifest\.json$")
FIELDS = ("count", "numerator_sum", "denominator_sum", "numerator_square_sum",
          "denominator_square_sum", "numerator_denominator_sum")
TAIL_OBSERVABLES = ("hard_nuclear_stopping_ev_per_angstrom",
                    "hard_transport_rate_per_angstrom")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def bounds(path):
    match = BATCH.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid batch name: {path}")
    return tuple(map(int, match.groups()))


def descriptor(path, root):
    stat = path.stat()
    return [str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns]


def freeze_case(root, case):
    directories = list((root / "production" / case).glob(".trajectory_checkpoints/*"))
    if len(directories) != 1:
        raise ValueError(f"Expected one checkpoint configuration: {case}")
    paths = sorted(directories[0].glob("batch_*.manifest.json"), key=bounds)
    stop = 0
    for path in paths:
        start, next_stop = bounds(path)
        if start != stop or next_stop <= start:
            raise ValueError(f"Non-contiguous checkpoint prefix: {path}")
        stop = next_stop
    if not stop:
        raise ValueError(f"No committed batches: {case}")
    return {"case": case, "signature": directories[0].name, "trajectories": stop,
            "batches": [descriptor(path, root) for path in paths]}


def verify_snapshot(root, frozen):
    # Appended batches are intentionally ignored; modified frozen batches fail.
    for item in frozen["batches"]:
        if descriptor(root / item[0], root) != item:
            raise ValueError(f"Frozen checkpoint changed: {item[0]}")


def signed_configuration(root, frozen):
    path = (root / frozen["batches"][0][0]).parent / "configuration.json"
    data = path.read_bytes()
    manifest = json.loads(data)
    cfg = manifest["configuration"]
    signature = digest(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode())
    if signature != frozen["signature"] or manifest["configuration_signature"] != signature:
        raise ValueError(f"Signed configuration mismatch: {path}")
    if cfg["initial_condition_sampling"] != "collision_tube_mixture" or cfg["tube_mixture_fraction"] != .5:
        raise ValueError(f"Unsupported proposal: {path}")
    return cfg, digest(data)


def residuals(array, estimate):
    terms = np.stack((array[:, 3], -2 * estimate * array[:, 5],
                      estimate * estimate * array[:, 4]))
    value = terms.sum(axis=0)
    # Bound cancellation by input float64 precision, not by an arbitrary unit floor.
    allowance = 64 * np.finfo(float).eps * np.abs(terms).sum(axis=0)
    if np.any(value < -allowance):
        raise ValueError("Materially negative ratio residual variance")
    return np.maximum(value, 0)


def ratio_summary(array):
    total = array.sum(axis=0, dtype=np.longdouble)
    n, a, b = total[:3]
    if n < 2 or b <= 0 or not np.all(np.isfinite(array)):
        raise ValueError("Invalid ratio sufficient statistics")
    estimate = a / b
    contribution = residuals(array, estimate)
    total_residual = contribution.sum()
    se = np.sqrt(n * total_residual / (n - 1)) / b
    order = np.argsort(-contribution, kind="stable")
    top_count = max(1, math.ceil(len(array) / 100))
    fractions = contribution / total_residual if total_residual > 0 else contribution
    return {"estimate": float(estimate), "standard_error": float(se),
            "residual_square_sum": float(total_residual),
            "largest_batch_variance_fraction": float(fractions[order[0]]),
            "top_one_percent_batches": top_count,
            "top_one_percent_batch_variance_fraction": float(fractions[order[:top_count]].sum()),
            "top_one_percent_trajectory_fraction": float(array[order[:top_count], 0].sum() / n),
            "effective_variance_bearing_batches": (
                float(1 / np.square(fractions).sum()) if total_residual > 0 else None)}, order, fractions


def inspect_distribution(path, record, energy):
    sample = path.parent / record["distribution_sample"]
    if sample.parent != path.parent:
        raise ValueError("Distribution sample must be beside its checkpoint")
    data = sample.read_bytes()
    if digest(data) != record["distribution_sample_sha256"]:
        raise ValueError(f"Distribution checksum mismatch: {sample}")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        indices = archive["trajectory"]
        recoil = archive["total_recoil_energy_ev"]
        angle = archive["final_deflection_rad"]
    n = record["summary"]["target_distribution_sample_count"]
    if any(x.ndim != 1 or len(x) != n for x in (indices, recoil, angle)):
        raise ValueError(f"Distribution count mismatch: {sample}")
    start, stop = record["trajectory_start"], record["trajectory_stop"]
    if not np.issubdtype(indices.dtype, np.integer) or np.any(np.diff(indices) <= 0):
        raise ValueError(f"Invalid trajectory indices: {sample}")
    if n and (indices[0] < start or indices[-1] >= stop):
        raise ValueError(f"Trajectory outside batch: {sample}")
    if (not np.all(np.isfinite(recoil)) or not np.all(np.isfinite(angle))
            or np.any(recoil < 0) or np.any(recoil > energy * (1 + 1e-12))
            or np.any(angle < 0) or np.any(angle > math.pi)):
        raise ValueError(f"Nonphysical saved distribution value: {sample}")
    result = {"sample": sample.name, "sha256": digest(data), "uniform_trajectories": n}
    for name, values in (("recoil_ev", recoil), ("final_deflection_rad", angle)):
        order = np.argsort(-values, kind="stable")[:5]
        squares = np.square(values)
        result[name] = {
            "quantiles_50_99_999_9999_percent": np.quantile(values, [.5, .99, .999, .9999]).tolist() if n else [],
            "top": [{"trajectory": int(indices[i]), "value": float(values[i])} for i in order],
            "largest_value_square_fraction": float(squares.max() / squares.sum()) if n and squares.sum() else 0.0}
    return result


def audit_case(root, frozen, tail_batches, progress):
    rows = {"statistics": {}, "raw_statistics": {}}
    summaries, records, hashes = [], [], []
    verify_snapshot(root, frozen)
    for relative, size, mtime in frozen["batches"]:
        path = root / relative
        data = path.read_bytes()
        record = json.loads(data)
        start, stop = bounds(path)
        if (record["configuration_signature"] != frozen["signature"]
                or record["trajectory_start"] != start or record["trajectory_stop"] != stop
                or record["trajectory_count"] != stop - start):
            raise ValueError(f"Checkpoint identity mismatch: {path}")
        if set(record["statistics"]) != set(record["raw_statistics"]):
            raise ValueError(f"Estimator key mismatch: {path}")
        for estimator, target in rows.items():
            if target and set(target) != set(record[estimator]):
                raise ValueError(f"Observable mismatch: {path}")
            for name, stats in record[estimator].items():
                if stats["count"] != stop - start:
                    raise ValueError(f"Statistic count mismatch: {path}")
                target.setdefault(name, []).append([stats[field] for field in FIELDS])
        s = record["summary"]
        if (sum(s["termination_counts"].values()) != stop - start
                or sum(s["proposal_component_counts"].values()) != stop - start
                or s["proposal_component_counts"].get("uniform", 0) != s["target_distribution_sample_count"]):
            raise ValueError(f"Summary count mismatch: {path}")
        summaries.append(s)
        records.append(record)
        hashes.append(digest(data))
        progress.update(1)
    verify_snapshot(root, frozen)
    arrays = {k: {name: np.array(values, dtype=np.longdouble) for name, values in v.items()}
              for k, v in rows.items()}
    n = frozen["trajectories"]
    weight_sum = math.fsum(s["importance_weight_sum"] for s in summaries)
    weight_squares = math.fsum(s["importance_weight_square_sum"] for s in summaries)
    weight_se = math.sqrt(max(0, weight_squares - weight_sum * weight_sum / n) / (n * (n - 1)))
    components = {key: sum(s["proposal_component_counts"].get(key, 0) for s in summaries)
                  for key in ("uniform", "collision_tube")}
    max_weight = max(s["maximum_importance_weight"] for s in summaries)
    # Active v3 campaign uses a defensive half-uniform mixture: weight <= 2.
    # Use the signed checkpoint configuration, which exists for running cases
    # even when a final run manifest has not yet been written.
    cfg, configuration_hash = signed_configuration(root, frozen)
    if not 0 < max_weight <= 2 * (1 + 1e-12):
        raise ValueError(f"Importance weight outside defensive-mixture bound: {frozen['case']}")
    result = {"case": frozen["case"], "trajectories": n, "batch_count": len(records),
              "configuration_signature": frozen["signature"],
              "ordered_batch_sha256": digest("\n".join(hashes).encode()),
              "checkpoint_configuration_sha256": configuration_hash,
              "control_variate_enabled": cfg["control_variate"],
              "weight_checks": {"mean": weight_sum / n, "mean_standard_error": weight_se,
                                "mean_minus_one_in_standard_errors": (weight_sum / n - 1) / weight_se if weight_se else None,
                                "maximum": max_weight, "effective_sample_fraction": weight_sum**2 / (weight_squares * n),
                                "tube_fraction": components["collision_tube"] / n},
              "termination_counts": {}, "weighted_ambiguous_fractions": {}, "observables": {}}
    for s in summaries:
        for key, value in s["termination_counts"].items():
            result["termination_counts"][key] = result["termination_counts"].get(key, 0) + value
    for name in ("collision_count", "recoil_energy_ev", "transport_moment"):
        actual = math.fsum(s[f"importance_weighted_actual_{name}"] for s in summaries)
        ambiguous = math.fsum(s[f"importance_weighted_ambiguous_{name}"] for s in summaries)
        result["weighted_ambiguous_fractions"][name] = ambiguous / actual if actual else None
    inspect_indices = {0, len(records) // 2, len(records) - 1}
    for name, selected in arrays["statistics"].items():
        current, order, fractions = ratio_summary(selected)
        raw, _, _ = ratio_summary(arrays["raw_statistics"][name])
        current["raw_estimate"] = raw["estimate"]
        current["raw_standard_error"] = raw["standard_error"]
        current["selected_over_raw_standard_error"] = current["standard_error"] / raw["standard_error"] if raw["standard_error"] else None
        current["largest_variance_batches"] = [
            {"manifest": frozen["batches"][i][0], "sha256": hashes[i], "variance_fraction": float(fractions[i])}
            for i in order[:tail_batches]]
        # Fixed descriptive prefixes, not additional acceptance looks.
        current["prefix_diagnostics"] = []
        for denominator in (4, 2, 1):
            stop = max(1, len(selected) // denominator)
            prefix, _, _ = ratio_summary(selected[:stop])
            current["prefix_diagnostics"].append({"trajectories": int(selected[:stop, 0].sum()),
                                                   "estimate": prefix["estimate"], "standard_error": prefix["standard_error"]})
        result["observables"][name] = current
        if name in TAIL_OBSERVABLES:
            inspect_indices.update(map(int, order[:tail_batches]))
    result["uniform_tail_diagnostics"] = []
    energy = float(cfg["projectile_energy_ev"])
    for i in sorted(inspect_indices):
        inspected = inspect_distribution(root / frozen["batches"][i][0], records[i], energy)
        inspected["selection"] = "fixed first/middle/last batch or largest stopping/transport variance; diagnostic only"
        result["uniform_tail_diagnostics"].append(inspected)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--energies-ev", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True, help="Concurrent bounded-memory case readers")
    parser.add_argument("--tail-batches", type=int, required=True, help="Largest variance batches per scalar to retain")
    parser.add_argument("--refresh-analysis", action="store_true", help="Recompute diagnostic products after an audit-code change, retaining exactly the frozen input prefix")
    args = parser.parse_args()
    if args.workers < 1 or args.tail_batches < 1:
        parser.error("workers and tail-batches must be positive")
    root, output = args.campaign_root.resolve(), args.output.resolve()
    if output == root or root / "production" in output.parents or root / "estimator_calibration" in output.parents:
        parser.error("Diagnostic output must not overwrite production or calibration inputs")
    output.mkdir(parents=True, exist_ok=True)
    request = {"root": str(root), "energies_ev": sorted(set(args.energies_ev)),
               "tail_batches": args.tail_batches, "audit_code_sha256": digest(Path(__file__).read_bytes())}
    snapshot_path = output / "snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        if snapshot["request"] != request:
            old = {k: v for k, v in snapshot["request"].items() if k != "audit_code_sha256"}
            new = {k: v for k, v in request.items() if k != "audit_code_sha256"}
            if not args.refresh_analysis or old != new:
                raise ValueError("Incompatible audit resume; preserve this frozen diagnostic")
            snapshot["request"] = request
            atomic_json(snapshot_path, snapshot)
    else:
        cases = sorted(str(p.relative_to(root / "production"))
                       for p in (root / "production").glob("*/*/*eV")
                       if int(p.name[:-2]) in request["energies_ev"])
        if not cases:
            raise ValueError("No matching cases")
        snapshot = {"request": request, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "cases": [freeze_case(root, case) for case in tqdm(cases, desc="Freeze case prefixes")]}
        atomic_json(snapshot_path, snapshot)
    reports = []
    with tqdm(total=sum(len(c["batches"]) for c in snapshot["cases"]), desc="Audited checkpoint batches", unit="batch") as progress:
        def run(frozen):
            path = output / "cases" / (frozen["case"].replace("/", "__") + ".json")
            if path.exists() and not args.refresh_analysis:
                verify_snapshot(root, frozen)
                cached = json.loads(path.read_text())
                if cached.get("audit_code_sha256") == request["audit_code_sha256"]:
                    progress.update(len(frozen["batches"]))
                    return cached
            result = audit_case(root, frozen, args.tail_batches, progress)
            result["audit_code_sha256"] = request["audit_code_sha256"]
            atomic_json(path, result)
            return result
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            reports = list(executor.map(run, snapshot["cases"]))
    atomic_json(output / "report.json", {"status": "diagnostic-only; no acceptance decision or runtime change",
                "snapshot_sha256": digest(snapshot_path.read_bytes()), "cases": reports,
                "limitations": ["Scalar standard errors are the existing trajectory-level ratio delta estimates, not finite-sample guarantees.",
                                "Descriptive prefix checks and ranked tail batches are not additional stopping looks.",
                                "NPZ samples contain only uniform-component trajectories; tube trajectories cannot be reconstructed from these arrays.",
                                "Maximum saved deflection is a net trajectory angle, not the sum of event transport moments.",
                                "Mean weight near one and bounded weights do not prove the proposal density or rare-tail moments are correct.",
                                "Frozen running prefixes need not coincide with predeclared acceptance looks."]})
    print(f"Completed {len(reports)} case diagnostics: {output / 'report.json'}")


if __name__ == "__main__":
    main()
