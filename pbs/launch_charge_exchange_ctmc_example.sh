#!/usr/bin/env bash
#
# Example multi-node launcher for the element-specific CTMC PBS scripts.
#
# Usage:
#   export START_SEPARATION_AU='convergence-derived values'
#   export TARGET_BMAX_AU='five convergence-derived values'
#   export LOSS_BMAX_AU='element-specific convergence-derived values'
#
#   bash pbs/launch_charge_exchange_ctmc_example.sh plan carbon 84
#   bash pbs/launch_charge_exchange_ctmc_example.sh submit carbon 84
#   # After all compute shards finish successfully:
#   bash pbs/launch_charge_exchange_ctmc_example.sh merge carbon 84
#
# This launcher intentionally contains no default boundary/cutoff values.

set -euo pipefail

usage() {
    printf '%s\n' \
        "Usage: $0 {plan|submit|merge} {carbon|lithium|oxygen|sulfur} SHARD_COUNT" \
        "" \
        "Required exported variables:" \
        "  START_SEPARATION_AU  one convergence-derived value per energy" \
        "  TARGET_BMAX_AU       five convergence-derived H2O-orbital values" \
        "  LOSS_BMAX_AU         element-specific projectile-loss values" \
        "" \
        "PBS_ARRAY_MAX defaults to 50 and controls array splitting."
}

if (( $# != 3 )); then
    usage >&2
    exit 2
fi

action="$1"
atom="$2"
shard_count="$3"

case "${action}" in
    plan|submit|merge) ;;
    *)
        printf 'Unknown action: %s\n' "${action}" >&2
        usage >&2
        exit 2
        ;;
esac

case "${atom}" in
    carbon|lithium|oxygen|sulfur) ;;
    *)
        printf 'Unknown element: %s\n' "${atom}" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ ! "${shard_count}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'SHARD_COUNT must be a positive integer: %s\n' "${shard_count}" >&2
    exit 2
fi

array_max="${PBS_ARRAY_MAX:-50}"
if [[ ! "${array_max}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'PBS_ARRAY_MAX must be a positive integer: %s\n' "${array_max}" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
pbs_script="${repo_root}/pbs/generate_${atom}_charge_exchange_ctmc.pbs"
if [[ ! -f "${pbs_script}" ]]; then
    printf 'Missing element PBS script: %s\n' "${pbs_script}" >&2
    exit 1
fi

required_variables=(
    START_SEPARATION_AU
    TARGET_BMAX_AU
    LOSS_BMAX_AU
)
for variable_name in "${required_variables[@]}"; do
    if [[ -z "${!variable_name:-}" ]]; then
        printf 'Required variable is empty or unset: %s\n' \
            "${variable_name}" >&2
        exit 2
    fi
    export "${variable_name}"
done

optional_variables=(
    MINIMUM_INTEGRATION_TIME_AU
    CONDA_SH
    CONDA_ENV
    PYTHON_BIN
    TRAJECTORY_CHUNK_SIZE
    PENDING_FACTOR
    CHECKPOINT_EVERY
    CHECKPOINT_SECONDS
    RTOL
    ATOL
    RETRY_RTOL
    RETRY_ATOL
    MAXIMUM_RELATIVE_ENERGY_DRIFT
    BOUNDARY_EXTENSION_FACTOR
    EXTRA_ARGS
    REPO_ROOT
)

forwarded_variables=("${required_variables[@]}")
for variable_name in "${optional_variables[@]}"; do
    if [[ -n "${!variable_name+x}" ]]; then
        export "${variable_name}"
        forwarded_variables+=("${variable_name}")
    fi
done

variable_list="$(IFS=,; printf '%s' "${forwarded_variables[*]}")"

if [[ "${action}" != "plan" ]] && ! command -v qsub >/dev/null 2>&1; then
    printf 'qsub is not available in PATH\n' >&2
    exit 127
fi

cd "${repo_root}"

if [[ "${action}" == "merge" ]]; then
    if (( shard_count < 2 )); then
        printf '%s\n' \
            "A one-shard run writes final outputs directly; no merge is needed." >&2
        exit 2
    fi
    export_spec="${variable_list},SHARD_COUNT=${shard_count},MERGE_ONLY=1"
    printf 'Submitting one-core merge for %s (%s shards)\n' \
        "${atom}" "${shard_count}"
    qsub \
        -l select=1:ncpus=1:mem=4gb \
        -v "${export_spec}" \
        "${pbs_script}"
    exit 0
fi

printf '%s\n' \
    "Element: ${atom}" \
    "Total shards: ${shard_count}" \
    "Cores per running shard: 64" \
    "Maximum requested cores: $((shard_count * 64))" \
    "PBS array batch limit: ${array_max}" \
    "PBS script: ${pbs_script}"

offset=0
while (( offset < shard_count )); do
    remaining=$((shard_count - offset))
    batch_size="${array_max}"
    if (( remaining < batch_size )); then
        batch_size="${remaining}"
    fi
    array_last=$((batch_size - 1))
    export_spec="${variable_list},SHARD_COUNT=${shard_count},SHARD_OFFSET=${offset}"

    if [[ "${action}" == "plan" ]]; then
        printf 'Would submit shard indices %d--%d as array 0-%d\n' \
            "${offset}" "$((offset + batch_size - 1))" "${array_last}"
    else
        printf 'Submitting shard indices %d--%d as array 0-%d: ' \
            "${offset}" "$((offset + batch_size - 1))" "${array_last}"
        qsub \
            -J "0-${array_last}" \
            -v "${export_spec}" \
            "${pbs_script}"
    fi
    offset=$((offset + batch_size))
done

if [[ "${action}" == "submit" ]]; then
    printf '%s\n' \
        "Compute shards submitted. Do not submit the merge until every array" \
        "element has completed successfully."
fi
