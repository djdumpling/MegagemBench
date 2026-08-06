#!/usr/bin/env bash
# Phase-2.2 + 2.3 runner (the RL plan §2): brings up vLLM, runs the toy
# ~50-step loop (2.2) then the MegaGem 10–20-step smoke (2.3), tears vLLM
# down. Phase select via SKIP_TOY=1 / SKIP_MEGAGEM=1.
#
#   DRY_RUN=1 ./scripts/training/run_phase2.sh                               # CPU self-test
#   PYBIN=prime-rl/.venv/bin/python MODEL_PATH=<sft-merged-ckpt> \           # GPU run
#       ./scripts/training/run_phase2.sh

set -euo pipefail

# --- knobs ---------------------------------------------------------------- #
: "${PYBIN:=python}"
: "${MODEL_PATH:=Qwen/Qwen3-4B-Instruct-2507}"
: "${SERVED_NAME:=qwen/qwen3-4b-instruct}"
: "${VLLM_HOST:=0.0.0.0}"
: "${VLLM_PORT:=8000}"
: "${MAX_MODEL_LEN:=32768}"
: "${VLLM_READY_TIMEOUT_S:=300}"
: "${VLLM_GPU_MEM_UTIL:=0.3}"              # co-located with trainer; bare default 0.9 OOMs

: "${STEPS:=50}"                           # 2.2 update steps
: "${K:=4}"                                # rollouts/seed (B = SEEDS·K·8)
: "${NUM_SEEDS:=2}"
: "${EVAL_SEEDS:=40}"                      # P1.3 free power lever (eval-only)
: "${KL_MAX:=0.5}"
# Fix a KL-overshoot via LR/KL_BETA — never raise KL_MAX (v5.6 mistake).
: "${LR:=1e-5}"
: "${KL_BETA:=0.05}"
: "${ROLLOUT_VLLM:=1}"                     # 0 ⇒ deterministic CPU stub
: "${ON_POLICY:=0}"                        # 1 ⇒ roll from trainer weights (overrides ROLLOUT_VLLM)
: "${GATE_MODE:=auto}"                     # auto|plumbing|learnability (both verdicts always emitted)
: "${ACCEPT_STALE_ONPOLICY:=0}"            # 1 ⇒ record-and-proceed when spg>1

: "${MG_STEPS:=15}"                        # 2.3 steps (gate window 10–20)
: "${MG_K:=2}"
: "${MG_NUM_SEEDS:=2}"

: "${RUN_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${RESULTS_DIR:=results/phase2_${RUN_TAG}}"

: "${SKIP_TOY:=0}"
: "${SKIP_MEGAGEM:=0}"
: "${DRY_RUN:=0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The megagem package lives under src/; the prime-rl venv PYBIN has no
# editable install of it, so put src/ on PYTHONPATH for every subprocess.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
mkdir -p "$RESULTS_DIR"

VLLM_LOG="${RESULTS_DIR}/vllm_server.log"
TOY_OUT="${RESULTS_DIR}/toy_loop.json"
MG_OUT="${RESULTS_DIR}/megagem_steps.json"
VLLM_URL="http://localhost:${VLLM_PORT}/v1"
VLLM_PID=""

# Match the vllm binary to PYBIN's venv (bare `vllm` may resolve elsewhere — finding 4).
VLLM_BIN="$(dirname "${PYBIN}")/vllm"
[[ -x "${VLLM_BIN}" ]] || VLLM_BIN="vllm"

log() { printf '[phase2] %s\n' "$*"; }

stop_vllm() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        log "stopping vLLM (pid=${VLLM_PID})"
        kill "${VLLM_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -0 "${VLLM_PID}" 2>/dev/null && kill -9 "${VLLM_PID}" 2>/dev/null || true
    fi
    VLLM_PID=""
}
trap stop_vllm EXIT INT TERM

start_vllm() {
    log "starting vLLM (${VLLM_BIN}): model=${MODEL_PATH} served=${SERVED_NAME} port=${VLLM_PORT}"
    log "vLLM stdout/stderr → ${VLLM_LOG}"
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
        --served-model-name "${SERVED_NAME}" \
        --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --host "${VLLM_HOST}" --port "${VLLM_PORT}" \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    log "vLLM pid=${VLLM_PID}; waiting up to ${VLLM_READY_TIMEOUT_S}s for /v1/models"
    local elapsed=0
    while (( elapsed < VLLM_READY_TIMEOUT_S )); do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            log "ERROR: vLLM died during startup. Last 40 log lines:"
            tail -n 40 "${VLLM_LOG}" >&2 || true
            return 1
        fi
        if curl -sf "${VLLM_URL}/models" >/dev/null 2>&1; then
            log "vLLM ready after ${elapsed}s"; return 0
        fi
        sleep 5; elapsed=$((elapsed + 5))
    done
    log "ERROR: vLLM not ready after ${VLLM_READY_TIMEOUT_S}s. Last 40 log lines:"
    tail -n 40 "${VLLM_LOG}" >&2 || true
    return 1
}

DRY_FLAG=""
if [[ "${DRY_RUN}" == "1" ]]; then
    DRY_FLAG="--dry-run"
    log "DRY_RUN=1 — CPU self-test of both drivers; no vLLM/GPU."
fi

# vLLM is needed if 2.2 rolls out via the external server, or 2.3 runs at all.
NEED_VLLM=0
if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ "${SKIP_TOY}" != "1" && "${ROLLOUT_VLLM}" == "1" && "${ON_POLICY}" != "1" ]]; then NEED_VLLM=1; fi
    if [[ "${SKIP_MEGAGEM}" != "1" ]]; then NEED_VLLM=1; fi
fi
(( NEED_VLLM )) && start_vllm

toy_rc=0
mg_rc=0

run_toy() {
    log "=== Phase 2.2 — toy ${STEPS}-step loop (K=${K}) ==="
    local cmd=( "${PYBIN}" scripts/training/toy_loop.py )
    [[ -n "${DRY_FLAG}" ]] && cmd+=( "${DRY_FLAG}" )
    cmd+=( --model "${MODEL_PATH}" --steps "${STEPS}" --k "${K}"
           --num-seeds "${NUM_SEEDS}" --eval-seeds "${EVAL_SEEDS}"
           --kl-max "${KL_MAX}" --gate-mode "${GATE_MODE}"
           --learning-rate "${LR}" --kl-beta "${KL_BETA}"
           --output "${TOY_OUT}" )
    if [[ "${ON_POLICY}" == "1" ]]; then
        cmd+=( --on-policy )
    elif [[ "${DRY_RUN}" != "1" && "${ROLLOUT_VLLM}" == "1" ]]; then
        cmd+=( --rollout-vllm-url "${VLLM_URL}"
               --served-model-name "${SERVED_NAME}" )
    fi
    [[ "${ACCEPT_STALE_ONPOLICY}" == "1" ]] && cmd+=( --accept-stale-onpolicy )
    "${cmd[@]}"
}

run_megagem() {
    log "=== Phase 2.3 — MegaGem ${MG_STEPS}-step robustness smoke ==="
    local cmd=( "${PYBIN}" scripts/training/megagem_steps.py )
    [[ -n "${DRY_FLAG}" ]] && cmd+=( "${DRY_FLAG}" )
    cmd+=( --model "${MODEL_PATH}" --served-model-name "${SERVED_NAME}"
           --steps "${MG_STEPS}" --k "${MG_K}" --num-seeds "${MG_NUM_SEEDS}"
           --kl-max "${KL_MAX}" --output "${MG_OUT}" )
    [[ "${DRY_RUN}" != "1" ]] && cmd+=( --vllm-url "${VLLM_URL}" )
    "${cmd[@]}"
}

if [[ "${SKIP_TOY}" != "1" ]]; then
    set +e; run_toy; toy_rc=$?; set -e
else
    log "SKIP_TOY=1 — skipping Phase 2.2"
fi

if [[ "${SKIP_MEGAGEM}" != "1" ]]; then
    set +e; run_megagem; mg_rc=$?; set -e
else
    log "SKIP_MEGAGEM=1 — skipping Phase 2.3"
fi

(( NEED_VLLM )) && stop_vllm

log "=== done ==="
log "results: ${RESULTS_DIR}"
[[ -f "${TOY_OUT}" ]] && log "  - 2.2 toy:     ${TOY_OUT}"
[[ -f "${MG_OUT}"  ]] && log "  - 2.3 megagem: ${MG_OUT}"
[[ -f "${VLLM_LOG}" ]] && log "  - vllm log:    ${VLLM_LOG}"
(( toy_rc != 0 )) && log "Phase 2.2 rc=${toy_rc} (gate fail or error — see ${TOY_OUT})"
(( mg_rc  != 0 )) && log "Phase 2.3 rc=${mg_rc} (gate fail or error — see ${MG_OUT})"
(( toy_rc != 0 || mg_rc != 0 )) && exit 1
exit 0
