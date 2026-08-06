#!/usr/bin/env bash
# Submit the recursive, single-node adaptive ion CDFT pilot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CP2K_EXE="${CP2K_EXE:-${SCRIPT_DIR}/cp2k_container_wrapper.sh}"
if [[ ! -x "${CP2K_EXE}" ]]; then
    echo "CP2K executable/wrapper is unavailable: ${CP2K_EXE}" >&2
    exit 2
fi
if [[ "$#" -gt 1 ]]; then
    echo "Usage: $0 [ELEMENT_OR_PROJECTILE_DEFINITION.json]" >&2
    exit 2
fi

if [[ "$#" -eq 0 ]]; then
    qsub -v "PROJECTILE=C,CP2K_EXE=${CP2K_EXE}" \
        "${SCRIPT_DIR}/run_adaptive_charge_resolved_dft.pbs"
elif [[ -f "$1" ]]; then
    DEFINITION="$(realpath "$1")"
    qsub -v "PROJECTILE_DEFINITION=${DEFINITION},CP2K_EXE=${CP2K_EXE}" \
        "${SCRIPT_DIR}/run_adaptive_charge_resolved_dft.pbs"
else
    qsub -v "PROJECTILE=$1,CP2K_EXE=${CP2K_EXE}" \
        "${SCRIPT_DIR}/run_adaptive_charge_resolved_dft.pbs"
fi
