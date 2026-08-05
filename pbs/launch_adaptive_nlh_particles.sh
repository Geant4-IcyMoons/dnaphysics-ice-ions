#!/usr/bin/env bash
# Submit one independent adaptive NLH job per supported projectile.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="${SCRIPT_DIR}/run_adaptive_nlh_particle.pbs"

for projectile in H He C O S; do
    job_id="$(qsub -N "nlh_${projectile}" -v "PROJECTILE=${projectile}" "${PBS_SCRIPT}")"
    printf '%s %s\n' "${projectile}" "${job_id}"
done
