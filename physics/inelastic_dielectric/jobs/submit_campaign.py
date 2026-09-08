"""Submit a fixed charge-state campaign from a modular source snapshot."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from physics.inelastic_dielectric.checkpoints import atomic_text, source_digest
from physics.constants import PROJECTILE_LIBRARY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projectile", choices=("proton", "alpha"), required=True)
    parser.add_argument("--charge-state", type=int)
    parser.add_argument("--polarization", choices=("both", "off", "on"), default="both")
    parser.add_argument("--queue", choices=("idle", "p72"), default="p72",
                        help="p72 routes to publicx; idle routes to idlex")
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()
    projectile = args.projectile
    nuclear_charge = int(PROJECTILE_LIBRARY[projectile]["charge"])
    charge = nuclear_charge if args.charge_state is None else args.charge_state
    if not 0 <= charge <= nuclear_charge:
        parser.error("Charge state must lie between zero and nuclear Z")
    if args.diagnostic_only and (charge == nuclear_charge or args.polarization != "on"):
        parser.error("Diagnostic-only campaigns require an electron-bearing state and --polarization on")
    if charge < nuclear_charge and args.polarization != "off":
        print("Experimental screened polarization: convergence and physical rejection checks remain enabled.")
    polarization_modes = (False, True) if args.polarization == "both" else (args.polarization == "on",)
    repo = Path(__file__).resolve().parents[3]
    output = repo / "physics" / "inelastic_dielectric" / "output"
    runs = output / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    campaign = Path(tempfile.mkdtemp(prefix=f"{projectile}_q{charge}_{stamp}_", dir=runs))
    snapshot = campaign / "source"
    snapshot.mkdir()
    shutil.copytree(repo / "physics", snapshot / "physics", ignore=shutil.ignore_patterns(
        "output", "plots", "runs", "__pycache__", "*.pyc"))
    shutil.copy2(repo / "requirements.txt", snapshot / "requirements.txt")
    tables = output / "tables" / campaign.name
    tables.mkdir(parents=True)
    logs = campaign / "logs"
    logs.mkdir()
    python = str(repo / ".venv" / "bin" / "python")
    digest = source_digest()
    snapshot_digest = subprocess.check_output(
        [python, "-c", "from physics.inelastic_dielectric.checkpoints import source_digest; print(source_digest())"],
        cwd=snapshot, text=True).strip()
    if snapshot_digest != digest:
        raise RuntimeError("Source changed while making the production snapshot")
    manifest = dict(
        projectile=projectile, charge_state=charge, energy_unit="total", polarization=args.polarization,
        diagnostic_only=args.diagnostic_only,
        source_sha256=digest, source_snapshot=str(snapshot),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        python=python, python_version=sys.version,
        queue=args.queue, ncpus=256, mem="512gb", walltime="72:00:00",
        energy_min_MeV=0.1, energy_max_MeV=100, energy_points=1000, dE=1000, dq=1000,
        tables=str(tables), jobs=[])
    with atomic_text(campaign / "working_tree.patch") as stream:
        subprocess.run(["git", "diff", "--binary", "--", "physics", "README.md", "requirements.txt"],
                       cwd=repo, stdout=stream, check=True)
    with atomic_text(campaign / "python_packages.txt") as stream:
        subprocess.run([python, "-m", "pip", "freeze"], stdout=stream, check=True)
    launcher = snapshot / "physics" / "inelastic_dielectric" / "jobs" / "generate_cross_sections.pbs"
    for phase in ("amorphous", "hexagonal"):
        for kernel in ("pwba", "rpwba"):
            for barkas in polarization_modes:
                case = f"{projectile}_q{charge}_{phase}_{kernel}_{'barkas' if barkas else 'nobarkas'}"
                if args.diagnostic_only:
                    case += "_diagnostic"
                cache = output / "caches" / campaign.name / case
                variables = dict(REPO_ROOT=str(snapshot), PYTHON_BIN=python,
                                 EXPECTED_SOURCE_SHA256=digest, ICE_TYPE=phase,
                                 PROJECTILE=projectile, CHARGE_STATE=str(charge), NCPUS="256",
                                 DIAGNOSTIC_ONLY=str(args.diagnostic_only).lower(),
                                 RPWBA=str(kernel == "rpwba").lower(), POLARIZATION=str(barkas).lower(),
                                 EMIN_MEV="0.1", EMAX_MEV="100", ENERGY_POINTS="1000",
                                 ENERGY_UNIT="total", DE="1000", DQ="1000",
                                 OUTPUT_TABLE_DIR=str(tables), CACHE_DIR=str(cache),
                                 LIVE_LOG_PATH=str(logs / f"{case}.live.out"),
                                 MPLCONFIGDIR=str(repo / ".venv" / "matplotlib"))
                command = ["qsub", "-N", case, "-q", args.queue, "-r", "y", "-l",
                           "select=1:ncpus=256:mem=512gb,walltime=72:00:00", "-j", "oe",
                           "-o", str(logs / f"{case}.pbs.out"), "-v",
                           ",".join(f"{key}={value}" for key, value in variables.items()), str(launcher)]
                job = dict(case=case, command=command, cache=str(cache), job_id=None)
                manifest["jobs"].append(job)
                with atomic_text(campaign / "submission.json") as stream:
                    json.dump(manifest, stream, indent=2)
                job["job_id"] = subprocess.check_output(command, cwd=repo, text=True).strip()
                with atomic_text(campaign / "submission.json") as stream:
                    json.dump(manifest, stream, indent=2)
                print(f"{job['job_id']} {case}", flush=True)
    print(f"Tables: {tables}\nSubmission record: {campaign / 'submission.json'}")


if __name__ == "__main__":
    main()
