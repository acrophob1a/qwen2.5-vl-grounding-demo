#!/bin/bash
# Train grounding SFT; shutdown only on success (leave instance up on failure).
# Usage: screen -S sft -dm bash finetuning/scripts/run_sft_and_shutdown.sh

set -uo pipefail

PROJECT_ROOT="/root/autodl-tmp/demo1"
RUN_NAME="grounding-sft-0"
WORK_DIR="${PROJECT_ROOT}/work_dirs/${RUN_NAME}"
LOG="${WORK_DIR}/run_all.log"
FAILURE_LOG="${WORK_DIR}/FAILURE.log"

cd "${PROJECT_ROOT}"
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "${WORK_DIR}"

exec > >(tee -a "${LOG}") 2>&1

STATUS=0
STEP="init"

record_failure() {
    local exit_code=$1
    {
        echo "=== FAILURE RECORD $(date -Iseconds) ==="
        echo "Failed step: ${STEP}"
        echo "Exit code: ${exit_code}"
        echo ""
        echo "--- tail of run_all.log ---"
        tail -n 80 "${LOG}" 2>/dev/null || true
        echo ""
        echo "--- tail of output.log ---"
        tail -n 120 "${WORK_DIR}/output.log" 2>/dev/null || true
    } > "${FAILURE_LOG}"
    echo "Failure summary written to ${FAILURE_LOG}"
}

echo "=============================================="
echo " Pipeline start: $(date -Iseconds)"
echo " Project: ${PROJECT_ROOT}"
echo "=============================================="

STEP="sft_training"
echo ""
echo "=== SFT training: $(date -Iseconds) ==="
if bash finetuning/scripts/sft_gnd.sh; then
    echo "SFT training finished successfully."
else
    STATUS=$?
    echo "SFT training FAILED with exit code ${STATUS}."
    record_failure "${STATUS}"
fi

echo ""
echo "=============================================="
if [[ "${STATUS}" -eq 0 ]]; then
    echo " Pipeline completed OK: $(date -Iseconds)"
    echo " Shutting down in 60 seconds to stop billing..."
    echo " (Cancel with: shutdown -c)"
    echo "=============================================="
    sleep 60
    shutdown -h now
else
    echo " Pipeline completed WITH ERRORS (exit ${STATUS}): $(date -Iseconds)"
    echo " Instance left running for debugging."
    echo " See ${FAILURE_LOG}"
    echo "=============================================="
    exit "${STATUS}"
fi
