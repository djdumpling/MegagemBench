#!/usr/bin/env bash
# Qwen3-4B baseline/post-SFT eval driver. Brings up local vLLM, runs
# scripts/eval/eval_qwen_baseline.py, tears vLLM down on exit. Default 190 games.
#
# Env: PRIME_API_KEY (required unless DRY_RUN=1).
# Overrides: MODEL_PATH, SEEDS (default: the locked 'val' split), OUTPUT_DIR,
# MAX_PARALLEL=4, DRY_RUN, YES, SKIP_VLLM, EXTRA_ARGS.
#
#   bash scripts/eval/eval_qwen_baseline.sh
#   MODEL_PATH=checkpoints/...-merged bash scripts/eval/eval_qwen_baseline.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${MODEL_PATH:=Qwen/Qwen3-4B-Instruct-2507}"
: "${SERVED_NAME:=qwen/qwen3-4b-instruct}"
: "${VLLM_HOST:=0.0.0.0}"
: "${VLLM_PORT:=8000}"
: "${MAX_MODEL_LEN:=32768}"
: "${VLLM_READY_TIMEOUT_S:=300}"

: "${SEEDS:=val}"   # split name, range (1151-1160), list, or a seed file
: "${RUN_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${OUTPUT_DIR:=results/eval_qwen_baseline_${RUN_TAG}}"
: "${MAX_PARALLEL:=4}"

: "${SKIP_VLLM:=0}"
: "${DRY_RUN:=0}"
: "${YES:=0}"
: "${EXTRA_ARGS:=}"

: "${PY:=.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "ERROR: python interpreter not found at $PY. Set PY=/path/to/python." >&2
    exit 1
fi
export EMPTY="${EMPTY:-EMPTY}"

if [[ -z "${PRIME_API_KEY:-}" && "${DRY_RUN}" != "1" ]]; then
    echo "ERROR: PRIME_API_KEY not set. export PRIME_API_KEY=... before running." >&2
    echo "       (Or set DRY_RUN=1 to preview the schedule without API calls.)" >&2
    exit 1
fi

VLLM_LOG="${OUTPUT_DIR}/vllm_server.log"
VLLM_URL="http://localhost:${VLLM_PORT}/v1"
VLLM_PID=""

log() { printf '[eval] %s\n' "$*"; }

stop_vllm() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        log "stopping vLLM (pid=${VLLM_PID})"
        kill "${VLLM_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${VLLM_PID}" 2>/dev/null; then
            log "vLLM still alive after SIGTERM; sending SIGKILL"
            kill -9 "${VLLM_PID}" 2>/dev/null || true
        fi
    fi
    VLLM_PID=""
}

trap stop_vllm EXIT INT TERM

start_vllm() {
    mkdir -p "$OUTPUT_DIR"
    log "starting vLLM: model=${MODEL_PATH} served=${SERVED_NAME} port=${VLLM_PORT}"
    log "vLLM stdout/stderr -> ${VLLM_LOG}"
    vllm serve "${MODEL_PATH}" \
        --served-model-name "${SERVED_NAME}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --host "${VLLM_HOST}" \
        --port "${VLLM_PORT}" \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    log "vLLM pid=${VLLM_PID}; waiting up to ${VLLM_READY_TIMEOUT_S}s for /v1/models"

    local elapsed=0
    while (( elapsed < VLLM_READY_TIMEOUT_S )); do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            log "ERROR: vLLM process died during startup. Last 40 log lines:"
            tail -n 40 "${VLLM_LOG}" >&2 || true
            return 1
        fi
        if curl -sf "${VLLM_URL}/models" >/dev/null 2>&1; then
            log "vLLM ready after ${elapsed}s"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done

    log "ERROR: vLLM not ready after ${VLLM_READY_TIMEOUT_S}s. Last 40 log lines:"
    tail -n 40 "${VLLM_LOG}" >&2 || true
    return 1
}

if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN=1 — skipping vLLM startup"
elif [[ "${SKIP_VLLM}" == "1" ]]; then
    log "SKIP_VLLM=1 — assuming vLLM is already serving ${SERVED_NAME} at ${VLLM_URL}"
else
    start_vllm
fi

driver_args=(
    --seeds "${SEEDS}"
    --output-dir "${OUTPUT_DIR}"
    --max-parallel "${MAX_PARALLEL}"
)
[[ "${DRY_RUN}" == "1" ]] && driver_args+=(--dry-run)
[[ "${YES}" == "1" ]] && driver_args+=(--yes)
# Word-split EXTRA_ARGS so callers can pass multiple flags as one env var.
extra=()
if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra=( ${EXTRA_ARGS} )
fi

log "running eval driver -> ${OUTPUT_DIR}"
if (( ${#extra[@]} )); then
    "$PY" scripts/eval/eval_qwen_baseline.py "${driver_args[@]}" "${extra[@]}"
else
    "$PY" scripts/eval/eval_qwen_baseline.py "${driver_args[@]}"
fi
rc=$?

log "driver exited rc=${rc}"
log "outputs: ${OUTPUT_DIR}"
exit $rc
