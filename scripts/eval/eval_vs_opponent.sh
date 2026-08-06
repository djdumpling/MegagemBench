#!/usr/bin/env bash
# Serve a policy on local vLLM and run a head-to-head vs an API opponent
# (scripts/eval/eval_vs_gemini.py). Brings vLLM up, optionally runs a tiny
# smoke first to confirm the opponent id resolves on Prime, runs the full
# sweep, and tears vLLM down on exit. The serve idiom mirrors
# scripts/eval/eval_qwen_baseline.sh (single source of the vLLM lifecycle
# pattern); only the eval driver differs.
#
# Env: PRIME_API_KEY (required — the opponent routes through Prime Inference).
# Overrides: MODEL_PATH, SERVED_NAME, OPPONENT, NUM_GAMES, MAX_PARALLEL,
# SEED_START, NUM_PLAYERS, VALUE_CHART, OUTPUT, SMOKE_FIRST, SMOKE_SEED_START,
# VLLM_*, SKIP_VLLM, PY.
#
#   MODEL_PATH=djdumpling/...sft-step1200-v2 OPPONENT=openai/gpt-5.4-nano \
#     NUM_GAMES=20 OUTPUT=results/strength_probe/out.json \
#     bash scripts/eval/eval_vs_opponent.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${MODEL_PATH:=djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2}"
: "${SERVED_NAME:=qwen/qwen3-4b-instruct}"
: "${VLLM_HOST:=0.0.0.0}"
: "${VLLM_PORT:=8000}"
: "${MAX_MODEL_LEN:=32768}"
: "${VLLM_READY_TIMEOUT_S:=600}"

: "${OPPONENT:=openai/gpt-5.4-nano}"
: "${NUM_GAMES:=20}"
: "${MAX_PARALLEL:=4}"
: "${SEED_START:=30000}"
: "${NUM_PLAYERS:=3}"
: "${VALUE_CHART:=A}"
: "${RUN_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${OUTPUT:=results/strength_probe_${RUN_TAG}/${OPPONENT//\//__}_n${NUM_GAMES}.json}"

: "${SMOKE_FIRST:=1}"
: "${SMOKE_GAMES:=2}"
: "${SMOKE_SEED_START:=29000}"   # disjoint from the full-run seed range

: "${SKIP_VLLM:=0}"
: "${PY:=.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "ERROR: python interpreter not found at $PY. Set PY=/path/to/python." >&2
    exit 1
fi
export EMPTY="${EMPTY:-EMPTY}"

if [[ -z "${PRIME_API_KEY:-}" ]]; then
    echo "ERROR: PRIME_API_KEY not set (the opponent routes through Prime Inference)." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
VLLM_LOG="$(dirname "$OUTPUT")/vllm_server.log"
VLLM_URL="http://localhost:${VLLM_PORT}/v1"
VLLM_PID=""

log() { printf '[probe] %s\n' "$*"; }

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
            log "ERROR: vLLM died during startup. Last 40 log lines:"
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

# eval_vs_gemini.py invocation; args after the function name override defaults.
run_eval() {  # $1=num_games $2=max_parallel $3=seed_start $4=output
    "$PY" scripts/eval/eval_vs_gemini.py \
        --model "${SERVED_NAME}" \
        --vllm-url "${VLLM_URL}" \
        --served-model-name "${SERVED_NAME}" \
        --opponent "${OPPONENT}" \
        --num-games "$1" \
        --max-parallel "$2" \
        --seed-start "$3" \
        --num-players "${NUM_PLAYERS}" \
        --value-chart "${VALUE_CHART}" \
        --output "$4"
}

if [[ "${SKIP_VLLM}" == "1" ]]; then
    log "SKIP_VLLM=1 — assuming vLLM already serving ${SERVED_NAME} at ${VLLM_URL}"
else
    start_vllm
fi

# Smoke: a couple of games to confirm the opponent id actually resolves on
# Prime (a bad id surfaces as per-game 404s) BEFORE burning the full sweep.
if [[ "${SMOKE_FIRST}" == "1" ]]; then
    SMOKE_OUT="${OUTPUT%.json}.smoke.json"
    log "smoke: ${SMOKE_GAMES} game(s) vs ${OPPONENT} to validate the opponent id"
    set +e
    run_eval "${SMOKE_GAMES}" 2 "${SMOKE_SEED_START}" "${SMOKE_OUT}"
    smoke_rc=$?
    set -e
    scored=$("$PY" -c "import json; print(json.load(open('${SMOKE_OUT}')).get('n_scored',0))" 2>/dev/null || echo 0)
    if [[ "${smoke_rc}" != "0" || "${scored}" -lt 1 ]]; then
        log "SMOKE FAILED (rc=${smoke_rc}, n_scored=${scored}) — aborting before the full run."
        log "  inspect ${SMOKE_OUT} (per_game[].error) and ${VLLM_LOG}."
        exit 3
    fi
    log "smoke OK (n_scored=${scored}); proceeding to full N=${NUM_GAMES}"
fi

log "full run: ${NUM_GAMES} games, policy=${SERVED_NAME} vs ${OPPONENT}"
set +e
run_eval "${NUM_GAMES}" "${MAX_PARALLEL}" "${SEED_START}" "${OUTPUT}"
rc=$?
set -e
log "eval exited rc=${rc}; output: ${OUTPUT}"
exit "${rc}"
