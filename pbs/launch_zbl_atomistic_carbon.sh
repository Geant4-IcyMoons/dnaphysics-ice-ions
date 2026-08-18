#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${REPO_ROOT}/python_scripts/physics_ice/nep_mbpol/.venv/bin/python"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "No usable Python environment found; set PYTHON_BIN." >&2
    exit 2
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/python_scripts/physics_ice/process_evidence/soft_nuclear_collisions/validation/runs/zbl_soft_carbon}"
MAX_ARRAY_SIZE="${MAX_ARRAY_SIZE:-50}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m \
  python_scripts.physics_ice.process_evidence.soft_nuclear_collisions.zbl.run_campaign \
  prepare --output-root "${OUTPUT_ROOT}" --projectile C \
  --interaction-model zbl_soft --nlh-boundary-ev 30 \
  --phases hexagonal_ih_100k \
  --energy-min-ev 1e4 --energy-max-ev 1e8 --energy-points 41 \
  --transfer-cutoffs-ev 1e-4 1e-5 1e-6 --tolerance 0.005 --confidence 0.95

CAMPAIGN_MANIFEST="${OUTPUT_ROOT}/zbl_soft_campaign.manifest.json"
CASE_COUNT="$(jq -r '.case_count' "${CAMPAIGN_MANIFEST}")"
if [[ ! "${CASE_COUNT}" =~ ^[1-9][0-9]*$ || ! "${MAX_ARRAY_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid case count or MAX_ARRAY_SIZE." >&2
    exit 2
fi

pending=()
while IFS=$'\t' read -r case_index output_directory; do
    if [[ ! -f "${output_directory}/case.receipt.json" ]]; then
        pending+=("${case_index}")
    fi
done < <(jq -r '.cases | to_entries[] | [.key, .value.output_directory] | @tsv' "${CAMPAIGN_MANIFEST}")

LOG_DIRECTORY="${OUTPUT_ROOT}/pbs_logs/resume_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${LOG_DIRECTORY}"
echo "Resuming ${#pending[@]} incomplete cases; $((CASE_COUNT - ${#pending[@]})) are complete."

jobs=()
position=0
while (( position < ${#pending[@]} )); do
    remaining=$((${#pending[@]} - position))
    count="${MAX_ARRAY_SIZE}"
    if (( remaining < count )); then count="${remaining}"; fi
    group="$(printf '%03d' "${#jobs[@]}")"
    case_index_file="${LOG_DIRECTORY}/case_indices_group_${group}.txt"
    printf '%s\n' "${pending[@]:position:count}" > "${case_index_file}.tmp"
    mv "${case_index_file}.tmp" "${case_index_file}"
    last=$((count - 1))
    job="$(qsub -J "0-${last}" -o "${LOG_DIRECTORY}/" -v "REPO_ROOT=${REPO_ROOT},CAMPAIGN_MANIFEST=${CAMPAIGN_MANIFEST},CASE_INDEX_FILE=${case_index_file},PYTHON_BIN=${PYTHON_BIN}" "${SCRIPT_DIR}/run_zbl_atomistic_case.pbs")"
    jobs+=("${job}")
    echo "Submitted ${job}: ${count} pending cases"
    position=$((position + count))
done

if (( ${#jobs[@]} > 0 )); then
    dependency="$(IFS=:; echo "${jobs[*]}")"
    reducer="$(qsub -W "depend=afterok:${dependency}" -o "${LOG_DIRECTORY}/" -v "REPO_ROOT=${REPO_ROOT},CAMPAIGN_MANIFEST=${CAMPAIGN_MANIFEST},PYTHON_BIN=${PYTHON_BIN}" "${SCRIPT_DIR}/reduce_zbl_atomistic_campaign.pbs")"
    echo "Reducer ${reducer} waits for ${#jobs[@]} array groups."
else
    reducer="$(qsub -o "${LOG_DIRECTORY}/" -v "REPO_ROOT=${REPO_ROOT},CAMPAIGN_MANIFEST=${CAMPAIGN_MANIFEST},PYTHON_BIN=${PYTHON_BIN}" "${SCRIPT_DIR}/reduce_zbl_atomistic_campaign.pbs")"
    echo "All cases are complete; submitted reducer ${reducer}."
fi
