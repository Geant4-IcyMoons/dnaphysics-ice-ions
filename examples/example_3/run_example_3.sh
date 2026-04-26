#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR="${SCRIPT_DIR}"
PROJECT_DIR=$(cd "${EXAMPLE_DIR}/../.." && pwd)

BINARY="${PROJECT_DIR}/build/dnaphysics"
OUTPUT_DIR="${EXAMPLE_DIR}/outputs"
PHYSICS="ice_am"
THREADS="10"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --binary PATH       Path to dnaphysics binary
  --output-dir PATH   Directory for example_3.root
  --physics MODE      DNA_PHYSICS value: ice_am | ice_hex | water
  --help              Show this message
EOF
}

while (($#)); do
  case "$1" in
    --binary)
      BINARY="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --physics)
      PHYSICS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "${BINARY}" ]]; then
  echo "dnaphysics binary not found or not executable: ${BINARY}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Running example_3 -> ${OUTPUT_DIR}/example_3.root"
(
  cd "${OUTPUT_DIR}"
  DNA_PHYSICS="${PHYSICS}" \
  DNA_ROOT_BASENAME="example_3" \
  DNA_NTUPLE_FILES=0 \
  DNA_NTUPLE_STRINGS=1 \
  "${BINARY}" "${EXAMPLE_DIR}/example_3.mac" "${THREADS}"
)
