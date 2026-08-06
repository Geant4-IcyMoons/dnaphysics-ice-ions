#!/usr/bin/env bash
# Submit a prepared CDFT workflow as restart-safe PBS shards.

set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
    echo "Usage: $0 WORKFLOW_MANIFEST [SHARD_COUNT]" >&2
    exit 2
fi

MANIFEST="$(realpath "$1")"
SHARD_COUNT="${2:-16}"
if [[ ! -f "${MANIFEST}" || ! "${SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Manifest must exist and SHARD_COUNT must be positive." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CP2K_EXE="${CP2K_EXE:-${SCRIPT_DIR}/cp2k_container_wrapper.sh}"
if [[ ! -x "${CP2K_EXE}" ]]; then
    echo "CP2K executable/wrapper is unavailable: ${CP2K_EXE}" >&2
    exit 2
fi
LAST_INDEX=$((SHARD_COUNT - 1))
qsub -J "0-${LAST_INDEX}" \
    -v "MANIFEST=${MANIFEST},SHARD_COUNT=${SHARD_COUNT},CP2K_EXE=${CP2K_EXE}" \
    "${SCRIPT_DIR}/run_charge_resolved_dft.pbs"
