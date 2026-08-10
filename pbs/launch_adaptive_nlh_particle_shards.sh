#!/usr/bin/env bash
# Launch a restart-safe, case-sharded adaptive NLH campaign for one projectile.

set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^(H|He|C|O|S)$ ]]; then
    echo "Usage: $0 H|He|C|O|S" >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
qsub -v "PROJECTILE=$1,REPO_ROOT=${REPO_ROOT},MAX_GLOBAL_SHARDS=${MAX_GLOBAL_SHARDS:-256},UNLIMITED_TRAJECTORIES=${UNLIMITED_TRAJECTORIES:-1}" \
    "${SCRIPT_DIR}/supervise_adaptive_nlh_particle_shards.pbs"
