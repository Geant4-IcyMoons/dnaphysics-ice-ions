#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR="${SCRIPT_DIR}"
PROJECT_DIR=$(cd "${EXAMPLE_DIR}/../.." && pwd)

BINARY="${PROJECT_DIR}/build/dnaphysics"
OUTPUT_DIR="${EXAMPLE_DIR}/outputs"
WVALUE_FILE="example_5_wvalue.txt"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --binary PATH       Path to dnaphysics binary
  --output-dir PATH   Directory for example_5.root and W-value text output
  --wvalue-file NAME  Output text file for appended W-value rows
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
    --wvalue-file)
      WVALUE_FILE="$2"
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

echo "Running example_5 -> ${OUTPUT_DIR}/example_5.root"
echo "W-value rows will be appended to ${OUTPUT_DIR}/${WVALUE_FILE}"
(
  cd "${OUTPUT_DIR}"
  DNA_PHYSICS="ice_hex" \
  DNA_ROOT_BASENAME="example_5" \
  DNA_WVALUE_FILE="${WVALUE_FILE}" \
  DNA_NTUPLE_FILES=0 \
  DNA_NTUPLE_STRINGS=1 \
  "${BINARY}" "${EXAMPLE_DIR}/example_5.mac" 12
)
