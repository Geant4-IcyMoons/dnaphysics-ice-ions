"""Prepare Ic and run bounded, restartable classical NEP-MB-pol NVT blocks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from physics.elastic.bca.structure import file_sha256, iter_xyz_frames
from physics.constants import AVOGADRO, H2O_MOLAR_MASS_G_MOL
from models.ice.structure_checks import chill_plus, bernal_fowler_topology

RUNTIME = ROOT / "python_scripts/physics_ice/nep_mbpol"
CONFIG = json.loads((HERE / "config.json").read_text())
RUNS = HERE / "runs"


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def cell_length() -> float:
    rep = CONFIG["replication"]
    if len(set(rep)) != 1:
        raise ValueError("This Ic protocol requires equal replication along all axes")
    return rep[0] * (2 * CONFIG["ih_unit_cell_volume_angstrom3"]) ** (1 / 3)


def input_text(seed: int, first: bool) -> str:
    c = CONFIG
    lines = ["potential nep.txt"]
    if first:
        lines += [f"minimize fire {c['minimization_force_tolerance_ev_angstrom']} {c['minimization_max_steps']}",
                  f"velocity {c['temperature_k']} seed {seed}"]
    lines += [f"time_step {c['timestep_fs']}",
              f"ensemble {c['thermostat']} {c['temperature_k']} {c['temperature_k']} {c['thermostat_coupling_steps']}",
              f"dump_thermo {c['thermo_interval_steps']}",
              f"dump_exyz {c['block_steps']} 1 0",
              f"run {c['block_steps']}"]
    return "\n".join(lines) + "\n"


def prepare(seed: int) -> Path:
    if seed not in CONFIG["seeds"]:
        raise ValueError("Seed is not in the configured campaign")
    run = RUNS / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    potential = RUNTIME / "model/nep-mbpol.nep.txt"
    if file_sha256(potential) != CONFIG["potential_sha256"]:
        raise ValueError("Potential checksum does not match the pinned published artifact")
    if version("genice2") != CONFIG["genice_version"]:
        raise ValueError("GenIce2 version does not match the pinned generator")
    # Pin actual source files; node-local compilation must not modify this tree.
    source = RUNTIME / "upstream/GPUMD-v3.9.3/src"
    source_hashes = {str(p.relative_to(source)): file_sha256(p)
                     for p in sorted(source.rglob("*"))
                     if p.is_file() and (p.suffix in {".cu", ".cuh", ".cpp", ".h"} or p.name == "makefile")}
    contract = {"config": CONFIG, "run_code_sha256": file_sha256(Path(__file__)),
                "gpumd_source_sha256": source_hashes,
                "dependencies": {str(p.relative_to(ROOT)): file_sha256(p) for p in (
                    HERE / "run.pbs", HERE.parent / "structure_checks.py",
                    ROOT / "physics/constants.py", ROOT / "physics/elastic/bca/structure.py",
                    ROOT / "physics/elastic/bca/config.py")}}
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    receipt = run / "preparation.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["signature"] != signature or file_sha256(run / "initial.xyz") != old["initial_sha256"]:
            raise ValueError("Incompatible resume: settings, runtime, or initial coordinates changed")
        return run
    if any(run.glob("block*")):
        raise ValueError("Existing dynamics without a preparation receipt")
    genice = RUNTIME / ".venv/bin/genice2"
    command = [str(genice), "--rep", *map(str, CONFIG["replication"]),
               "--seed", str(seed), "--depol", "strict", "--water", "physical_water",
               "--format", "exyz", "--quiet", "1c"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"GenIce2 failed: {result.stderr}")
    lines = result.stdout.splitlines()
    marker = lines.index("%PBC")
    n = int(lines[marker - 1])
    atoms = [line.split() for line in lines[marker + 1:marker + 1 + n]]
    species = np.asarray([a[0] for a in atoms])
    pos = np.asarray([[float(x) for x in a[1:4]] for a in atoms])
    vectors = {a[0]: list(map(float, a[1:])) for line in lines[marker + 1 + n:]
               if (a := line.split()) and a[0] in {"Vector1", "Vector2", "Vector3"}}
    lattice = np.asarray([vectors[f"Vector{i}"] for i in (1, 2, 3)])
    if n != 24 * np.prod(CONFIG["replication"]) or not np.all(species.reshape(-1, 3) == ["O", "H", "H"]):
        raise ValueError("Unexpected GenIce Ic composition or atom ordering")
    if not np.allclose(lattice, np.eye(3) * lattice[0, 0]):
        raise ValueError("GenIce did not produce the expected cubic cell")
    old_length = lattice[0, 0]
    length = cell_length()
    molecules = pos.reshape(-1, 3, 3)
    offsets = molecules - molecules[:, :1]
    offsets -= np.rint(offsets / old_length) * old_length
    # Set oxygen molecular origins; retain the generator's internal water geometry.
    pos = (molecules[:, :1] * (length / old_length) + offsets).reshape(-1, 3) % length
    lattice = np.eye(3) * length
    header = ('Lattice="' + ' '.join(f'{v:.12f}' for v in lattice.ravel())
              + '" Properties=species:S:1:pos:R:3 pbc="T T T"')
    xyz = [str(n), header] + [f"{a} {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}" for a, v in zip(species, pos)]
    atomic_bytes(run / "initial.xyz", ("\n".join(xyz) + "\n").encode())
    chill, neighbors, _, tree = chill_plus(pos[species == "O"], np.full(3, length))
    topology, _ = bernal_fowler_topology(pos[species == "O"], pos[species == "H"], neighbors, tree)
    if chill["cubic_fraction"] != 1 or not topology["passes_bernal_fowler_rules"]:
        raise ValueError("Initial cubic structure failed topology checks")
    write_json(receipt, {"signature": signature, "contract": contract, "seed": seed,
                        "created_utc": timestamp(), "command": command,
                        "genice_stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                        "initial_sha256": file_sha256(run / "initial.xyz"),
                        "molecules": n // 3, "length_angstrom": length,
                        "density_g_cm3": (n // 3) * H2O_MOLAR_MASS_G_MOL / AVOGADRO / (length ** 3 * 1e-24),
                        "chill": chill, "bernal_fowler": topology,
                        "state": "initial; not thermally equilibrated"})
    return run


def completed_block(block: Path, signature: str, source_sha: str) -> dict | None:
    receipt = block / "completed.json"
    if not receipt.exists():
        return None
    data = json.loads(receipt.read_text())
    if data["signature"] != signature or data["source_sha256"] != source_sha:
        raise ValueError(f"Incompatible block lineage: {block}")
    for name, expected in data["products"].items():
        if file_sha256(block / name) != expected:
            raise ValueError(f"Corrupted checkpoint: {block / name}")
    return data


def simulate(seed: int, executable: Path) -> None:
    run = prepare(seed)
    with (run / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        simulate_locked(run, seed, executable)


def simulate_locked(run: Path, seed: int, executable: Path) -> None:
    preparation = json.loads((run / "preparation.json").read_text())
    signature = preparation["signature"]
    count = CONFIG["equilibration_blocks"] + CONFIG["sampling_blocks"]
    source = run / "initial.xyz"
    with tqdm(total=count, desc=f"Ic seed {seed}: completed 100 ps blocks", unit="block") as bar:
        for index in range(count):
            block = run / f"block{index:03d}"
            source_sha = file_sha256(source)
            if completed_block(block, signature, source_sha) is None:
                # Only this locked seed's uncommitted block may be removed.
                if block.exists():
                    shutil.rmtree(block)
                block.mkdir()
                shutil.copyfile(source, block / "model.xyz")
                (block / "nep.txt").symlink_to(RUNTIME / "model/nep-mbpol.nep.txt")
                (block / "run.in").write_text(input_text(seed, index == 0))
                started = timestamp()
                with (block / "gpumd.log").open("w") as log:
                    process = subprocess.Popen([str(executable)], cwd=block, stdout=log, stderr=subprocess.STDOUT)
                    with tqdm(total=CONFIG["block_steps"], desc=f"Block {index + 1}/{count}",
                              unit="step", leave=False) as inner:
                        try:
                            while process.poll() is None:
                                path = block / "thermo.out"
                                done = 0
                                if path.exists():
                                    with path.open() as handle:
                                        done = min(sum(1 for _ in handle) * CONFIG["thermo_interval_steps"], inner.total)
                                inner.update(max(done - inner.n, 0))
                                time.sleep(5)
                        finally:
                            if process.poll() is None:
                                process.terminate()
                                try:
                                    process.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                    process.wait()
                        if process.returncode:
                            raise RuntimeError(f"GPUMD failed ({process.returncode}); see {block / 'gpumd.log'}")
                        inner.update(inner.total - inner.n)
                frames = list(iter_xyz_frames(block / "dump.xyz"))
                thermo = np.loadtxt(block / "thermo.out", ndmin=2)
                expected = CONFIG["block_steps"] // CONFIG["thermo_interval_steps"]
                if len(frames) != 1 or thermo.shape != (expected, 12) or not np.isfinite(thermo).all():
                    raise ValueError(f"Incomplete or invalid GPUMD output: {block}")
                frame = frames[0]
                if frame.species.size != preparation["molecules"] * 3 or not all(frame.pbc):
                    raise ValueError("Checkpoint composition or periodicity changed")
                if not np.allclose(frame.lattice_angstrom, np.eye(3) * cell_length(), rtol=0, atol=1e-7):
                    raise ValueError("Fixed cell changed")
                if "vel:R:3" not in (block / "dump.xyz").read_text().splitlines()[1]:
                    raise ValueError("Checkpoint is missing velocities")
                if not np.isfinite(np.loadtxt(block / "dump.xyz", skiprows=2, usecols=range(1, 7))).all():
                    raise ValueError("Non-finite checkpoint positions or velocities")
                write_json(block / "completed.json", {
                    "signature": signature, "source_sha256": source_sha,
                    "index": index, "started_utc": started, "completed_utc": timestamp(),
                    "executable_sha256": file_sha256(executable), "job_id": os.environ.get("PBS_JOBID"),
                    "products": {name: file_sha256(block / name) for name in ["dump.xyz", "thermo.out", "run.in", "gpumd.log"]}})
            source = block / "dump.xyz"
            bar.update(1)
    write_json(run / "dynamics_completed.json", {"signature": signature, "blocks": count,
               "final_sha256": file_sha256(source), "completed_utc": timestamp()})


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise InterruptedError(f"Received signal {signum}; completed blocks are preserved")
    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "simulate"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpumd", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        for seed in tqdm([args.seed], desc="Generate and check Ic", unit="configuration"):
            print(prepare(seed))
    else:
        if args.gpumd is None:
            parser.error("simulate requires --gpumd")
        simulate(args.seed, args.gpumd.resolve())
