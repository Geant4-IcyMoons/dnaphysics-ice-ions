#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${CP2K_IMAGE:-${REPO_ROOT}/software/cp2k/cp2k-2025.2-mpich-x86_64-psmp.sif}"
CONTAINER_CP2K="${CP2K_CONTAINER_BINARY:-/opt/cp2k/bin/cp2k}"
CONTAINER_ENTRYPOINT="/opt/cp2k/bin/entrypoint.sh"
MPI_RANKS="${CP2K_MPI_RANKS:-1}"

if [[ ! -s "${IMAGE}" ]]; then
    echo "Validated CP2K image is unavailable: ${IMAGE}" >&2
    exit 2
fi
if [[ ! "${MPI_RANKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CP2K_MPI_RANKS must be a positive integer." >&2
    exit 2
fi

container_command=("${CONTAINER_CP2K}")
if (( MPI_RANKS > 1 )); then
    container_command=(/opt/spack/bin/mpiexec -n "${MPI_RANKS}" "${CONTAINER_CP2K}")
fi

exec apptainer exec --cleanenv \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --env "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}" \
    --env "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}" \
    --env "MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}" \
    "${IMAGE}" "${CONTAINER_ENTRYPOINT}" "${container_command[@]}" "$@"
