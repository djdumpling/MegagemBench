#!/usr/bin/env bash
# Phase-3 GRPO self-play + §3.6 paired-bootstrap eval (the RL plan §3.1-3.6).
# Starts a LoRA-enabled rollout vLLM, runs scripts/training/phase3_grpo.py, then
# phase3_eval.py. The default curriculum is no-heuristic: 80% current self-play
# opponent seats + 20% checkpoint opponent seats.
#
#   DRY_RUN=1 ./scripts/training/run_phase3.sh                          # CPU self-test
#   PYBIN=prime-rl/.venv/bin/python HF_TOKEN=hf_xxx \                   # GPU run
#       ./scripts/training/run_phase3.sh

set -euo pipefail

# --- knobs ---------------------------------------------------------------- #
: "${PYBIN:=python}"
: "${MODEL_PATH:=djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2}"
: "${SERVED_NAME:=qwen/qwen3-4b-instruct}"
: "${VLLM_HOST:=0.0.0.0}"                   # bind address passed to vLLM serve --host
: "${VLLM_CLIENT_HOST:=localhost}"           # address used by trainer/poller to TALK to vLLM (NOT the bind address)
: "${VLLM_PORT:=8000}"
: "${HEURISTIC_PORT:=8100}"
: "${MAX_MODEL_LEN:=32768}"
: "${VLLM_READY_TIMEOUT_S:=300}"
: "${VLLM_GPU_MEM_UTIL:=0.3}"                # co-located with trainer
: "${MAX_LORA_RANK:=32}"                     # must match megagem_lora_config r=32
# VLLM_TOKENIZER: override when MODEL_PATH's tokenizer lacks a chat_template.
: "${VLLM_TOKENIZER:=}"
: "${PHASE3_SPLIT_GPUS:=0}"                  # 1 ⇒ trainer and vLLM on separate visible GPUs
: "${PHASE3_TRAIN_CUDA_VISIBLE_DEVICES:=0}"
: "${PHASE3_VLLM_CUDA_VISIBLE_DEVICES:=1}"   # single id (legacy) OR comma-sep list (e.g. "1,2,3,4,5,6,7") for TP>1 / DP>1
: "${VLLM_TENSOR_PARALLEL_SIZE:=1}"          # vLLM tensor parallelism; Qwen3-4B has 8 KV heads ⇒ TP ∈ {1,2,4,8}
: "${PHASE3_N_VLLM:=1}"                      # >1 ⇒ data-parallel: N independent vLLM workers, one per GPU
: "${PHASE3_VLLM_BASE_PORT:=8000}"           # worker i listens on BASE_PORT+i (i=0..N-1)
: "${VLLM_PREFIX_CACHING:=0}"                # 1 ⇒ --enable-prefix-caching (default off ⇒ real runs byte-identical)

# PROFILE picks defaults: `seam` = cheap wiring test (not a spend run); `evidence`
# = §3.6 sizing (N=60). Explicit env vars still win.
: "${PROFILE:=seam}"

# --- wandb resume parity with the Modal path ------------------------------ #
# Modal pre-populates WANDB_* before invoking us; for direct invocations we
# pre-generate an id so the eval subprocess attaches to the trainer's run.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    : "${WANDB_PROJECT:=megagem-phase3}"
    : "${WANDB_RUN_GROUP:=${PROFILE}}"
    _wb_ts=$(date +%s)
    : "${WANDB_NAME:=phase3_${PROFILE}_${_wb_ts}}"
    : "${WANDB_RUN_ID:=phase3_${PROFILE}_${_wb_ts}}"
    export WANDB_PROJECT WANDB_RUN_GROUP WANDB_NAME WANDB_RUN_ID
    [[ "${WANDB_MODE:-disabled}" == "disabled" ]] && export WANDB_MODE=online
fi

case "${PROFILE}" in
    seam)
        # Tuned to stress §3.3 in a tiny run (snapshots created+drawn+evicted in 70 steps).
        _DEF_NUM_SEEDS=2;  _DEF_EVAL_SEEDS=24;  _DEF_EVAL_K=1; _DEF_MICRO_CAP=8
        _DEF_ROWS_PER_GEN=96; _DEF_K=8; _DEF_MAX_PARALLEL=32; _DEF_ON_POLICY=0
        _DEF_MAX_LORAS=8; _DEF_MAX_CPU_LORAS=8
        _DEF_STEPS=70;     _DEF_SNAPSHOT_EVERY=10;  _DEF_MAX_SNAPSHOTS=2
        _DEF_OPP_ANNEAL_START=10;  _DEF_OPP_ANNEAL_END=30 ;;
    evidence)
        _DEF_NUM_SEEDS=32; _DEF_EVAL_SEEDS=60;  _DEF_EVAL_K=8; _DEF_MICRO_CAP=64
        _DEF_ROWS_PER_GEN=4096; _DEF_K=16; _DEF_MAX_PARALLEL=64; _DEF_ON_POLICY=1
        _DEF_MAX_LORAS=10; _DEF_MAX_CPU_LORAS=11
        _DEF_STEPS=200;    _DEF_SNAPSHOT_EVERY=10;  _DEF_MAX_SNAPSHOTS=8
        _DEF_OPP_ANNEAL_START=50;  _DEF_OPP_ANNEAL_END=0 ;;     # 0 ⇒ STEPS
    *) echo "[phase3] unknown PROFILE=${PROFILE} (seam|evidence)"; exit 2 ;;
esac

: "${STEPS:=${_DEF_STEPS}}"
: "${ROWS_PER_GEN:=${_DEF_ROWS_PER_GEN}}"
# Batch shape (phase3-rl-resize-8xh200). Defaults reproduce the legacy off-policy
# ga=1 geometry exactly; opt in to the large on-policy batch:
: "${ON_POLICY:=${_DEF_ON_POLICY}}"          # 1 ⇒ --on-policy (ga=spg, one optimizer step/generation)
: "${GRAD_ACCUM:=}"                          # explicit ga; empty ⇒ legacy ga=1 (ignored if ON_POLICY=1)
: "${NUM_PROCESSES:=1}"                       # DDP world size (>1 ⇒ torchrun; needs PHASE3_ALLOW_DDP=1)
: "${CHECKPOINT_EVERY:=25}"
: "${K:=${_DEF_K}}"                          # §A.7 K rollouts/seed
: "${NUM_SEEDS:=${_DEF_NUM_SEEDS}}"
: "${SEED_START:=9000}"
: "${SEED:=0}"                              # phase3_grpo.py --seed (TRL master RNG)
: "${PHASE2_MICRO_CAP:=${_DEF_MICRO_CAP}}"   # read by phase3_grpo._spg_shape
export PHASE2_MICRO_CAP
: "${EVAL_SEEDS:=${_DEF_EVAL_SEEDS}}"
: "${EVAL_SEED_START:=20000}"                # disjoint from training (no leak)
: "${KL_MAX:=0.5}"
: "${LR:=2e-5}"
: "${KL_BETA:=0.01}"
: "${VALUE_CHART:=A}"                        # chart A only (rl-chart-a-only)
: "${TRAINABLE_SEAT:=0}"
: "${MAX_PARALLEL:=${_DEF_MAX_PARALLEL}}"    # set 1 for sequential fallback
# Lever A — K-sample averaging. K=1 is a noise trap (the phase-3 eval findings F2).
: "${EVAL_SAMPLES_PER_SEED:=${_DEF_EVAL_K}}"
# Lever C — symmetric T override for both arms. Empty ⇒ vLLM default.
: "${EVAL_TEMPERATURE:=}"
# DUMP_ROLLOUTS: "1" ⇒ ${RESULTS_DIR}/rollout_dumps; a path ⇒ verbatim; empty ⇒ off.
: "${DUMP_ROLLOUTS:=}"
: "${FIXED_TRAIN_SEEDS:=0}"                  # 1 ⇒ legacy six-seed reuse mode
: "${ALLOW_LOW_ROWS_PER_GEN:=0}"
: "${EVAL_TRAIN_SEEDS:=0}"                   # diagnostic: also eval train seeds
: "${TRAIN_SEED_EVAL_SEEDS:=${EVAL_SEEDS}}"
: "${EVAL_INTERMEDIATE:=0}"
: "${REQUIRE_ONPOLICY:=0}"                   # 1 ⇒ dry-run rejects spg>1

# --- §3.3 lagged-self opponent pool -------------------------------------- #
: "${OPPONENT_POOL:=1}"                      # 0 ⇒ legacy single heuristic
: "${SNAPSHOT_EVERY:=${_DEF_SNAPSHOT_EVERY}}"
: "${MAX_SNAPSHOTS:=${_DEF_MAX_SNAPSHOTS}}"
: "${OPP_ANNEAL_START:=${_DEF_OPP_ANNEAL_START}}"
: "${OPP_ANNEAL_END:=${_DEF_OPP_ANNEAL_END}}"
[[ "${OPP_ANNEAL_END}" == "0" ]] && OPP_ANNEAL_END="${STEPS}"
: "${OPP_ANNEAL_PMAX:=0.7}"
: "${OPP_API_MODELS:=}"                      # comma-sep API opponents (empty ⇒ lagged-self only)
: "${OPP_API_PROB:=0.0}"
: "${OPP_API_WEIGHTS:=}"                     # comma-sep ints aligned with OPP_API_MODELS; empty ⇒ uniform
: "${OPP_POOL_SEED:=0}"
# repl_08: minimum draw probability for the pinned step_0 anchor among
# snapshot draws. 0.15 default ensures the anchor keeps anchoring even
# after the trainee crushes it (PFSP floor is a WEIGHT floor, not a
# probability floor). 0.0 reproduces pre-Codex behaviour.
: "${OPP_ANCHOR_FLOOR:=0.15}"
: "${P_CURRENT_SELF:=0.80}"                  # current live adapter opponent-seat share
# repl_08 v3: P(draw the scripted heuristic) of TOTAL draws (absolute share,
# not conditional on other gates). Internal decay:
#   p_heuristic_effective = p_heuristic · max(0.10, (1-WR_heur)²)
# Single-tier mix-gate in OpponentPool.draw():
#   heuristic gets p_heuristic_effective; API gets p_api; snapshot gets the
#   rest. Construction-time check: p_heuristic + p_api ≤ 1.
# 0.0 default ⇒ heuristic OFF in training (seam_smoke_02 / pure self-play).
# Set to 0.20-0.30 to reintroduce the easy-win signal (AlphaStar main-
# exploiter role). Diagnosis: seam_smoke_02 (heuristic OFF) gave ≈0 reward
# / ≈33% WR → null learning signal in 70 steps.
: "${P_HEURISTIC:=0.0}"
# repl_08 v3.1: stochastic-but-deterministic-per-state heuristic. 0.0 ⇒ pure
# v1 (bid = floor(0.5 · max_bid)). N>0 ⇒ ε ∈ [-N, +N] derived from a hash of
# the auction descriptor, so fraction spans [0.5-N, 0.5+N] across distinct
# auction states while remaining byte-identical for the same state across
# K-samples (preserves K-group credit assignment). Recommend 0.15 to give
# the policy a 35-65% bid-range to learn against (vs the brittle "bid 41
# always wins" optimum against the pure-0.5 heuristic).
: "${P_HEURISTIC_BID_NOISE:=0.0}"
export PHASE3_HEURISTIC_BID_NOISE="${P_HEURISTIC_BID_NOISE}"
# repl_08 v3.2: step at which the heuristic draw probability anneals to EXACTLY
# 0 (linear from step 0), decoupled from the (1-WR)² decay floor. 0 ⇒ no step
# anneal (floor-only — heuristic persists at ≈0.10·P_HEURISTIC of draws
# forever; the pre-v3.2 behaviour). Set to ~2×SNAPSHOT_EVERY so the heuristic
# bootstraps cold-start symmetry-breaking and then graduates to self-play
# (AlphaStar/OpenAI-Five pattern). Only meaningful when P_HEURISTIC > 0.
: "${HEURISTIC_ANNEAL_END:=0}"
# Rotate the trainable seat round-robin per roll (constant within each K-group,
# so §A.7 is unaffected). 1 ⇒ train all 3 seats — required because the
# TrueSkill/panel eval rates the policy across seat0/1/2; 0 ⇒ pin TRAINABLE_SEAT
# (legacy single-seat training).
: "${ROTATE_SEATS:=1}"
: "${HETERO_OPPONENTS:=1}"                   # 1 ⇒ per-seat 80/20 draws; mixed tables allowed
: "${MAX_LORAS:=${_DEF_MAX_LORAS}}"          # vLLM GPU-resident LoRA slots
: "${MAX_CPU_LORAS:=${_DEF_MAX_CPU_LORAS}}"  # vLLM CPU LoRA cache

# --- GRPO clip-higher (DAPO) — opt-in; empty ⇒ symmetric clip ------------- #
: "${EPSILON:=}"
: "${EPSILON_HIGH:=}"

# --- §3.6 eval opponent --------------------------------------------------- #
: "${EVAL_OPPONENT:=${SERVED_NAME}}"         # default no-heuristic eval vs SFT-served opponents
: "${THRESHOLD:=0.0}"                        # non-heuristic default: require positive CI-low

# Guard LoRA slot capacity: live set = 1 trainable + 1 pinned step_0 anchor
# (repl_08) + MAX_SNAPSHOTS unpinned, so the CPU cache needs +3 (or referenced
# snapshots are LRU-evicted), and one rollout batch can need +2 GPU slots.
if [[ "${OPPONENT_POOL}" == "1" ]]; then
    if (( MAX_CPU_LORAS < MAX_SNAPSHOTS + 3 )); then
        echo "[phase3] ABORT: MAX_CPU_LORAS=${MAX_CPU_LORAS} < MAX_SNAPSHOTS+3=$(( MAX_SNAPSHOTS + 3 )) (trainable + step_0 anchor + every snapshot must fit the CPU cache)" >&2
        exit 2
    fi
    if (( MAX_LORAS < MAX_SNAPSHOTS + 2 )); then
        echo "[phase3] ABORT: MAX_LORAS=${MAX_LORAS} < MAX_SNAPSHOTS+2=$(( MAX_SNAPSHOTS + 2 )) (trainable + step_0 anchor + every snapshot can land in one rollout batch)" >&2
        exit 2
    fi
fi

if [[ "${PROFILE}" == "evidence" && "${ALLOW_LOW_ROWS_PER_GEN}" != "1" ]]; then
    if (( ROWS_PER_GEN < 4096 )); then
        echo "[phase3] ABORT: evidence ROWS_PER_GEN=${ROWS_PER_GEN} < 4096." >&2
        echo "[phase3] no-heuristic self-play evidence runs should keep enough selected rows for k=16 × 32 seeds; set ALLOW_LOW_ROWS_PER_GEN=1 only for forensic reproductions." >&2
        exit 2
    fi
fi
# repl_08 strict-spg=2 path: when ALLOW_LOW_ROWS_PER_GEN=1 is set with a
# below-floor ROWS_PER_GEN under PROFILE=evidence, log a loud banner so we
# notice the bypass after the fact. The guard above already executed; this
# is purely visibility.
if [[ "${PROFILE}" == "evidence" && "${ALLOW_LOW_ROWS_PER_GEN}" == "1" \
      && ${ROWS_PER_GEN} -lt 4096 ]]; then
    echo "[phase3] ==================================================================" >&2
    echo "[phase3] BANNER: ALLOW_LOW_ROWS_PER_GEN=1 — bypassing the 4096-floor guard." >&2
    echo "[phase3]   ROWS_PER_GEN=${ROWS_PER_GEN}  (default no-heuristic evidence floor: 4096)" >&2
    echo "[phase3]   This is useful only for forensic reproductions or seam runs." >&2
    echo "[phase3] ==================================================================" >&2
fi

: "${RUN_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${RESULTS_DIR:=results/phase3_grpo_${RUN_TAG}}"
: "${ADAPTER_ROOT:=${RESULTS_DIR}/adapters}"
[[ "${DUMP_ROLLOUTS}" == "1" ]] && DUMP_ROLLOUTS="${RESULTS_DIR}/rollout_dumps"
: "${DRY_RUN:=0}"
: "${SKIP_PREP_CHECK:=0}"                    # 1 ⇒ skip GPU-prep gate (unsafe)
: "${EVAL_ON_GRPO_FAIL:=0}"                  # 1 ⇒ run §3.6 eval even if GRPO failed
: "${SKIP_GRPO:=0}"                          # 1 ⇒ eval-only mode (needs EXT_*_DIR)
: "${EXT_STEP0_DIR:=}"
: "${EXT_FINAL_DIR:=}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The megagem package lives under src/; the prime-rl venv PYBIN has no
# editable install of it, so put src/ on PYTHONPATH for every subprocess.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
mkdir -p "$RESULTS_DIR"

VLLM_LOG="${RESULTS_DIR}/vllm_server.log"
HEUR_LOG="${RESULTS_DIR}/heuristic_endpoint.log"
GRPO_OUT="${RESULTS_DIR}/phase3_grpo.json"
VLLM_URL="http://${VLLM_CLIENT_HOST}:${VLLM_PORT}/v1"
HEUR_URL="http://${VLLM_CLIENT_HOST}:${HEURISTIC_PORT}/v1"
VLLM_PID="" ; HEUR_PID=""
declare -a VLLM_PIDS=()                       # DP: one PID per worker (legacy single-worker still populates this)
VLLM_URLS=""                                  # DP: comma-sep URL list; legacy = single URL

VLLM_BIN="$(dirname "${PYBIN}")/vllm"
[[ -x "${VLLM_BIN}" ]] || VLLM_BIN="vllm"

log() { printf '[phase3] %s\n' "$*"; }

heuristic_prob_enabled() {
    [[ "${P_HEURISTIC}" != "0" && "${P_HEURISTIC}" != "0.0" && "${P_HEURISTIC}" != "0.00" && "${P_HEURISTIC}" != "0.000" ]]
}

heuristic_training_needed() {
    [[ "${OPPONENT_POOL}" != "1" ]] || heuristic_prob_enabled
}

heuristic_eval_needed() {
    [[ "${EVAL_OPPONENT}" == "heuristic" || "${EVAL_OPPONENT}" == "megagem/heuristic-v1" ]]
}

heuristic_needed() {
    heuristic_training_needed || heuristic_eval_needed
}

if [[ "${PHASE3_SPLIT_GPUS}" == "1" ]]; then
    log "split-GPU mode: trainer CUDA_VISIBLE_DEVICES=${PHASE3_TRAIN_CUDA_VISIBLE_DEVICES}; vLLM CUDA_VISIBLE_DEVICES=${PHASE3_VLLM_CUDA_VISIBLE_DEVICES}"
fi

# De-biased Phase-3 reward cut (the phase-3 eval findings F11-12): tanh terminal squash
# (fixes clip rank-collapse in the ~28% decisive-game tail), reveal proxy off
# (measured wrong-signed, Caveat 2), rec 1a on, λ=0.01 (unset). Env-only; no
# src/megagem/rl edits. Explicit unset prevents a stale container env from leaking in.
unset PHASE3_REWARD_WIN_BONUS PHASE3_SHAPING_LAMBDA PHASE3_ILLEGAL_PENALTY \
      PHASE3_CORRECTION_SCALE 2>/dev/null || true
export PHASE3_TERMINAL_CORRECTION=1
export PHASE3_TERMINAL_SHAPE=tanh
export PHASE3_REWARD_SCALE=19.6            # tanh calib ≈ 1.82×median (clip=21.5)
export PHASE3_REVEAL_SHAPING_WEIGHT=0.0

stop_procs() {
    # Stop all vLLM workers (DP: VLLM_PIDS has N entries; legacy: 1 entry).
    for pid in "${VLLM_PIDS[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            log "stopping vLLM worker (pid=${pid})"
            kill "${pid}" 2>/dev/null || true
            for _ in $(seq 1 30); do kill -0 "${pid}" 2>/dev/null || break; sleep 1; done
            kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null || true
        fi
    done
    if [[ -n "${HEUR_PID}" ]] && kill -0 "${HEUR_PID}" 2>/dev/null; then
        log "stopping HEUR_PID (pid=${HEUR_PID})"
        kill "${HEUR_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "${HEUR_PID}" 2>/dev/null || break; sleep 1; done
        kill -0 "${HEUR_PID}" 2>/dev/null && kill -9 "${HEUR_PID}" 2>/dev/null || true
    fi
    VLLM_PID="" ; HEUR_PID="" ; VLLM_PIDS=()
}
trap stop_procs EXIT INT TERM

poll_ready() {  # $1=url $2=pid $3=name $4=logfile
    local elapsed=0
    while (( elapsed < VLLM_READY_TIMEOUT_S )); do
        if ! kill -0 "$2" 2>/dev/null; then
            log "ERROR: $3 died during startup. Last 40 log lines:"
            tail -n 40 "$4" >&2 || true; return 1
        fi
        curl -sf "$1/models" >/dev/null 2>&1 && { log "$3 ready after ${elapsed}s"; return 0; }
        sleep 5; elapsed=$((elapsed + 5))
    done
    log "ERROR: $3 not ready after ${VLLM_READY_TIMEOUT_S}s. Last 40 log lines:"
    tail -n 40 "$4" >&2 || true; return 1
}

start_heuristic() {
    log "starting heuristic shim on :${HEURISTIC_PORT} → ${HEUR_LOG}"
    "${PYBIN}" -m megagem.training.heuristic_endpoint \
        --host "${VLLM_HOST}" --port "${HEURISTIC_PORT}" \
        > "${HEUR_LOG}" 2>&1 &
    HEUR_PID=$!
    poll_ready "${HEUR_URL}" "${HEUR_PID}" "heuristic-shim" "${HEUR_LOG}"
}

start_vllm() {
    if (( PHASE3_N_VLLM <= 1 )); then
        # Legacy single-worker path (byte-identical to prior runs).
        log "starting LoRA-enabled vLLM: base=${MODEL_PATH} served=${SERVED_NAME} port=${VLLM_PORT} tp=${VLLM_TENSOR_PARALLEL_SIZE}"
        local cmd=( "${VLLM_BIN}" serve "${MODEL_PATH}"
            --served-model-name "${SERVED_NAME}"
            --enable-lora --max-lora-rank "${MAX_LORA_RANK}"
            --max-loras "${MAX_LORAS}" --max-cpu-loras "${MAX_CPU_LORAS}"
            --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
            --max-model-len "${MAX_MODEL_LEN}"
            --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
            --host "${VLLM_HOST}" --port "${VLLM_PORT}" )
        [[ "${VLLM_PREFIX_CACHING}" == "1" ]] && cmd+=( --enable-prefix-caching )
        if [[ -n "${VLLM_TOKENIZER}" ]]; then
            log "  tokenizer override: --tokenizer ${VLLM_TOKENIZER}"
            cmd+=( --tokenizer "${VLLM_TOKENIZER}" )
        fi
        if [[ "${PHASE3_SPLIT_GPUS}" == "1" ]]; then
            CUDA_VISIBLE_DEVICES="${PHASE3_VLLM_CUDA_VISIBLE_DEVICES}" \
                VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 "${cmd[@]}" > "${VLLM_LOG}" 2>&1 &
        else
            VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 "${cmd[@]}" > "${VLLM_LOG}" 2>&1 &
        fi
        VLLM_PID=$!
        VLLM_PIDS=("${VLLM_PID}")
        VLLM_URLS="${VLLM_URL}"
        poll_ready "${VLLM_URL}" "${VLLM_PID}" "rollout-vLLM" "${VLLM_LOG}"
        return
    fi

    # DP fan-out: N independent vLLM workers, each on its own GPU slice.
    # SPLIT_GPUS is REQUIRED for N>1: without explicit per-worker
    # CUDA_VISIBLE_DEVICES, every vLLM process defaults to GPU 0 → 7-way
    # collision + OOM. Hard-abort rather than silently misconfigure.
    if [[ "${PHASE3_SPLIT_GPUS}" != "1" ]]; then
        log "ABORT: PHASE3_N_VLLM=${PHASE3_N_VLLM} requires PHASE3_SPLIT_GPUS=1 (DP needs explicit per-worker GPU pinning; without it all workers collide on GPU 0)"
        return 1
    fi
    log "starting ${PHASE3_N_VLLM} vLLM workers (DP): base=${MODEL_PATH} served=${SERVED_NAME} tp=${VLLM_TENSOR_PARALLEL_SIZE} base_port=${PHASE3_VLLM_BASE_PORT}"
    local -a gpus=()
    IFS=',' read -r -a gpus <<< "${PHASE3_VLLM_CUDA_VISIBLE_DEVICES}"
    local need=$(( PHASE3_N_VLLM * VLLM_TENSOR_PARALLEL_SIZE ))
    if (( ${#gpus[@]} < need )); then
        log "ABORT: PHASE3_N_VLLM=${PHASE3_N_VLLM} × tp=${VLLM_TENSOR_PARALLEL_SIZE} needs ${need} GPUs; PHASE3_VLLM_CUDA_VISIBLE_DEVICES has ${#gpus[@]} (${PHASE3_VLLM_CUDA_VISIBLE_DEVICES})"
        return 1
    fi

    local urls=()
    for ((i=0; i<PHASE3_N_VLLM; i++)); do
        local port=$((PHASE3_VLLM_BASE_PORT + i))
        local worker_log="${RESULTS_DIR}/vllm_worker_${i}.log"
        local cmd=( "${VLLM_BIN}" serve "${MODEL_PATH}"
            --served-model-name "${SERVED_NAME}"
            --enable-lora --max-lora-rank "${MAX_LORA_RANK}"
            --max-loras "${MAX_LORAS}" --max-cpu-loras "${MAX_CPU_LORAS}"
            --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
            --max-model-len "${MAX_MODEL_LEN}"
            --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
            --host "${VLLM_HOST}" --port "${port}" )
        [[ "${VLLM_PREFIX_CACHING}" == "1" ]] && cmd+=( --enable-prefix-caching )
        [[ -n "${VLLM_TOKENIZER}" ]] && cmd+=( --tokenizer "${VLLM_TOKENIZER}" )

        # Slice the GPU list: worker i takes gpus[i*tp .. i*tp+tp-1].
        local start=$(( i * VLLM_TENSOR_PARALLEL_SIZE ))
        local end=$(( start + VLLM_TENSOR_PARALLEL_SIZE - 1 ))
        local slice=()
        for ((j=start; j<=end; j++)); do slice+=("${gpus[j]}"); done
        local worker_gpus
        worker_gpus=$(IFS=,; echo "${slice[*]}")
        # Each worker is its own rank-0 vLLM process. If all DP workers share
        # the same torch.compile cache root, concurrent cold compile can race
        # while saving the same rank_0_0 graph directory. Keep the persistent
        # cache benefit, but shard it per worker.
        local base_cache="${VLLM_CACHE_ROOT:-/root/.cache/vllm}"
        local worker_cache_root="${base_cache%/}/worker_${i}"
        local worker_rpc_root="/tmp/vllm_rpc_worker_${i}"
        mkdir -p "${worker_cache_root}" "${worker_rpc_root}"
        log "  worker[${i}]: gpus=${worker_gpus} port=${port} log=${worker_log} cache=${worker_cache_root}"
        CUDA_VISIBLE_DEVICES="${worker_gpus}" \
            VLLM_CACHE_ROOT="${worker_cache_root}" \
            VLLM_RPC_BASE_PATH="${worker_rpc_root}" \
            VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 "${cmd[@]}" > "${worker_log}" 2>&1 &
        VLLM_PIDS+=($!)
        # URL uses CLIENT_HOST (e.g. localhost) — NOT VLLM_HOST (the bind addr,
        # typically 0.0.0.0). 0.0.0.0 is not a routable client target.
        urls+=("http://${VLLM_CLIENT_HOST}:${port}/v1")
    done

    # Poll each worker for readiness in parallel-friendly order (sequential
    # poll is fine: a worker that starts faster is just immediately ready
    # when we check it; the bottleneck is the slowest worker either way).
    for ((i=0; i<PHASE3_N_VLLM; i++)); do
        local port=$((PHASE3_VLLM_BASE_PORT + i))
        local worker_log="${RESULTS_DIR}/vllm_worker_${i}.log"
        poll_ready "http://${VLLM_CLIENT_HOST}:${port}/v1" "${VLLM_PIDS[i]}" "rollout-vLLM[${i}]" "${worker_log}" || return 1
    done

    VLLM_URLS=$(IFS=,; echo "${urls[*]}")
    # Re-point VLLM_URL at worker 0 so downstream eval (run_eval) hits the
    # right port — under DP, PHASE3_VLLM_BASE_PORT may differ from VLLM_PORT.
    VLLM_URL="${urls[0]}"
    VLLM_PID="${VLLM_PIDS[0]}"               # legacy alias for logs / single-URL callers
    log "all ${PHASE3_N_VLLM} vLLM workers ready; eval will use ${VLLM_URL}"
}

# Partial automated GPU-prep gate. Requires explicit SEAM_VERIFIED=1 ack since
# the authoritative seam tests must be run on-box by a human.
check_gpu_prep() {
    [[ "${SKIP_PREP_CHECK}" == "1" ]] && { log "SKIP_PREP_CHECK=1 — UNSAFE, skipping GPU-prep gate"; return 0; }
    log "GPU-prep gate (PARTIAL):"
    local v
    v="$("${PYBIN}" -c 'import trl, trl.import_utils as u; print(trl.__version__, u.is_vllm_available())' 2>/dev/null || true)"
    log "  trl: ${v:-<import failed>}"
    if [[ "${v}" != *"False"* ]]; then
        log "ABORT: pinned TRL fork not active (need is_vllm_available()→False)."
        log "       Run scripts/training/setup_trl_fork.sh, then the seam tests."
        return 1
    fi
    local pins
    pins="$("${PYBIN}" -c 'import vllm,transformers as t; print(vllm.__version__, t.__version__)' 2>/dev/null || true)"
    log "  vllm/transformers: ${pins:-<import failed>}"
    if [[ "${pins}" != "0.10.2 "* || "${pins}" != *" 4.55.4" ]]; then
        log "ABORT: venv pins wrong (need vllm==0.10.2 transformers==4.55.4; got '${pins}')."
        return 1
    fi
    if [[ "${SEAM_VERIFIED:-0}" != "1" ]]; then
        log "ABORT: seam tests not acknowledged. Run them on-box FIRST:"
        log "       ${PYBIN%/*}/python -m pytest tests/test_trl_seam.py tests/test_megagem_grpo.py -q"
        log "  then re-run with SEAM_VERIFIED=1."
        return 1
    fi
    log "  seam tests: acknowledged (SEAM_VERIFIED=1)"
}

DRY_FLAG=""
if [[ "${DRY_RUN}" == "1" ]]; then
    DRY_FLAG="--dry-run"
    log "DRY_RUN=1 — CPU self-test of driver+eval; no vLLM/GPU/prep gate."
fi

if [[ "${DRY_RUN}" != "1" ]]; then
    check_gpu_prep || exit 2
    if heuristic_needed; then
        start_heuristic
    else
        log "heuristic shim not started (default no-heuristic training/eval path)"
    fi
    start_vllm
fi

grpo_rc=0 ; eval_rc=0

run_grpo() {
    log "=== Phase 3 GRPO — ${STEPS} steps (1 continuous trainer), K=${K}, rows/gen=${ROWS_PER_GEN} (1a ON, λ=0.01, chart ${VALUE_CHART}) ==="
    local cmd=( "${PYBIN}" scripts/training/phase3_grpo.py )
    [[ -n "${DRY_FLAG}" ]] && cmd+=( "${DRY_FLAG}" )
    [[ -n "${DRY_FLAG}" && "${PROFILE}" == "evidence" && "${REQUIRE_ONPOLICY}" == "1" ]] && \
        cmd+=( --require-onpolicy )
    cmd+=( --model "${MODEL_PATH}" --served-model-name "${SERVED_NAME}"
           --steps "${STEPS}" --rows-per-gen "${ROWS_PER_GEN}"
           --checkpoint-every "${CHECKPOINT_EVERY}" --k "${K}"
           --num-seeds "${NUM_SEEDS}" --seed-start "${SEED_START}"
           --value-chart "${VALUE_CHART}" --trainable-seat "${TRAINABLE_SEAT}"
           --kl-max "${KL_MAX}" --learning-rate "${LR}" --kl-beta "${KL_BETA}"
           --max-parallel "${MAX_PARALLEL}"
           --seed "${SEED}"
           --adapter-root "${ADAPTER_ROOT}" --output "${GRPO_OUT}" )
    cmd+=( --num-processes "${NUM_PROCESSES}" )
    [[ "${ON_POLICY}" == "1" ]] && cmd+=( --on-policy )
    [[ -n "${GRAD_ACCUM}" ]] && cmd+=( --gradient-accumulation-steps "${GRAD_ACCUM}" )
    [[ "${FIXED_TRAIN_SEEDS}" == "1" ]] && cmd+=( --fixed-train-seeds )
    [[ "${ROTATE_SEATS}" == "1" ]] && cmd+=( --rotate-seats )
    [[ "${HETERO_OPPONENTS}" == "1" ]] && cmd+=( --hetero-opponents )
    if [[ "${OPPONENT_POOL}" == "1" ]]; then
        cmd+=( --opponent-pool
               --snapshot-every "${SNAPSHOT_EVERY}"
               --max-snapshots "${MAX_SNAPSHOTS}"
               --opp-anneal-start "${OPP_ANNEAL_START}"
               --opp-anneal-end "${OPP_ANNEAL_END}"
               --opp-anneal-pmax "${OPP_ANNEAL_PMAX}"
               --opp-api-prob "${OPP_API_PROB}"
               --opp-pool-seed "${OPP_POOL_SEED}"
               --opp-anchor-floor "${OPP_ANCHOR_FLOOR}"
               --p-current-self "${P_CURRENT_SELF}"
               --p-heuristic "${P_HEURISTIC}"
               --heuristic-anneal-end "${HEURISTIC_ANNEAL_END}" )
        [[ -n "${OPP_API_MODELS}" ]] && \
            cmd+=( --opp-api-models "${OPP_API_MODELS}" )
        [[ -n "${OPP_API_WEIGHTS}" ]] && \
            cmd+=( --opp-api-weights "${OPP_API_WEIGHTS}" )
    else
        cmd+=( --no-opponent-pool )
    fi
    [[ -n "${EPSILON}" ]] && cmd+=( --epsilon "${EPSILON}" )
    [[ -n "${EPSILON_HIGH}" ]] && cmd+=( --epsilon-high "${EPSILON_HIGH}" )
    [[ -n "${DUMP_ROLLOUTS}" ]] && cmd+=( --dump-rollouts "${DUMP_ROLLOUTS}" )
    if [[ "${DRY_RUN}" != "1" ]]; then
        # Pass the URL LIST under --vllm-urls (DP-aware). For legacy single-
        # worker, VLLM_URLS == VLLM_URL so the trainer sees a 1-element list
        # and is byte-identical to the prior --vllm-url path.
        cmd+=( --vllm-urls "${VLLM_URLS}" )
        if heuristic_training_needed; then
            cmd+=( --heuristic-url "${HEUR_URL}" )
        fi
    fi
    # DDP launcher: NUM_PROCESSES>1 ⇒ torchrun (the Python guard still requires
    # PHASE3_ALLOW_DDP=1 + a rank-sharded rollout). Default 1 ⇒ plain python,
    # byte-identical to before. cmd[@]:2 = the args (drop PYBIN + script path).
    local -a launch=( "${cmd[@]}" )
    if (( NUM_PROCESSES > 1 )); then
        launch=( torchrun --nproc_per_node="${NUM_PROCESSES}" --standalone
                 scripts/training/phase3_grpo.py "${cmd[@]:2}" )
    fi
    if [[ "${PHASE3_SPLIT_GPUS}" == "1" ]]; then
        CUDA_VISIBLE_DEVICES="${PHASE3_TRAIN_CUDA_VISIBLE_DEVICES}" "${launch[@]}"
    else
        "${launch[@]}"
    fi
}

run_eval() {  # $1=label  $2=adapter dir  $3=extra flags (e.g. --informational)
    local out="${RESULTS_DIR}/eval_$1.json"
    local seed_start="${4:-${EVAL_SEED_START}}"
    local eval_seeds="${5:-${EVAL_SEEDS}}"
    log "=== Phase 3.6 eval ($1) — paired-bootstrap RL vs SFT1200-v2 ==="
    local cmd=( "${PYBIN}" scripts/training/phase3_eval.py --label "$1" )
    [[ -n "${DRY_FLAG}" ]] && cmd+=( "${DRY_FLAG}" )
    [[ -n "${3:-}" ]] && cmd+=( "$3" )
    cmd+=( --rl-served phase3-trainable --sft-served "${SERVED_NAME}"
           --eval-seeds "${eval_seeds}" --seed-start "${seed_start}"
           --value-chart "${VALUE_CHART}" --trainable-seat "${TRAINABLE_SEAT}"
           --max-parallel "${MAX_PARALLEL}"
           --eval-samples-per-seed "${EVAL_SAMPLES_PER_SEED}"
           --eval-opponent "${EVAL_OPPONENT}"
           --total-steps "${STEPS}"
           --output "${out}" )
    [[ -n "${EVAL_TEMPERATURE}" ]] && cmd+=( --temperature "${EVAL_TEMPERATURE}" )
    [[ -n "${THRESHOLD}" ]] && cmd+=( --threshold "${THRESHOLD}" )
    if [[ "${DRY_RUN}" != "1" ]]; then
        cmd+=( --vllm-url "${VLLM_URL}" )
        if heuristic_eval_needed; then
            cmd+=( --heuristic-url "${HEUR_URL}" )
        fi
        [[ -n "$2" ]] && cmd+=( --rl-adapter-path "$2" )
    fi
    "${cmd[@]}"
}

# Either train→eval, OR eval-only against pre-existing adapter paths (SKIP_GRPO=1
# — Lever C / K-sample re-eval of a saved checkpoint).
if [[ "${SKIP_GRPO}" == "1" ]]; then
    log "SKIP_GRPO=1 — eval-only mode. Will eval adapters supplied via env:"
    log "  EXT_STEP0_DIR=${EXT_STEP0_DIR:-<unset>}"
    log "  EXT_FINAL_DIR=${EXT_FINAL_DIR:-<unset>}"
    if [[ -z "${EXT_FINAL_DIR}" ]]; then
        log "ABORT: SKIP_GRPO=1 requires EXT_FINAL_DIR (the adapter to eval)."
        exit 2
    fi
    grpo_rc=0
    STEP0_DIR="${EXT_STEP0_DIR}"
    FINAL_DIR="${EXT_FINAL_DIR}"
else
    set +e; run_grpo; grpo_rc=$?; set -e

    # §3.6 eval. step_0 = pre-train policy (saved BEFORE training), evaluated
    # --informational so its ≈0 delta cannot poison rc. The default final eval
    # is no-heuristic (SFT-served opponents); heuristic eval requires explicit
    # EVAL_OPPONENT=heuristic opt-in.
    # Read paths from the driver's authoritative JSON, not a lexical glob
    # (step_100 < step_25 sort issue).
    STEP0_DIR="${ADAPTER_ROOT}/step_0"
    FINAL_DIR=""
    if [[ "${DRY_RUN}" != "1" && -f "${GRPO_OUT}" ]]; then
        STEP0_DIR="$("${PYBIN}" -c 'import json,sys;c=json.load(open(sys.argv[1])).get("checkpoints") or {};print(c.get("step_0_pretrain") or "")' "${GRPO_OUT}" 2>/dev/null || echo "${STEP0_DIR}")"
        FINAL_DIR="$("${PYBIN}" -c 'import json,sys;c=json.load(open(sys.argv[1])).get("checkpoints") or {};print(c.get("final") or "")' "${GRPO_OUT}" 2>/dev/null || true)"
    fi
fi

# Honesty banner: if GRPO ran a seam-shape (spg>1 / few refreshes), the §3.6
# number is NOT final spend evidence — say so loudly. SKIP_GRPO mode skips
# the banner (can't introspect externally-supplied adapter's training shape).
SEAM_SHAPE=""
if [[ "${SKIP_GRPO}" != "1" ]]; then
    SEAM_SHAPE="$("${PYBIN}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("is_seam_shape"))' "${GRPO_OUT}" 2>/dev/null || echo "")"
fi
if [[ "${SKIP_GRPO}" != "1" && ( "${PROFILE}" == "seam" || "${SEAM_SHAPE}" == "True" ) ]]; then
    log "############################################################"
    log "# SEAM-TEST SHAPE (PROFILE=${PROFILE}, is_seam_shape=${SEAM_SHAPE:-?})."
    log "# This validates wiring on-box. It is NOT a final spend/no-spend"
    log "# evidence run — use PROFILE=evidence (N=60, strong on-policy)."
    log "############################################################"
fi

# Skip the §3.6 eval entirely if GRPO failed (eval number meaningless,
# games cost $). SKIP_GRPO=1 mode always runs the eval.
eval_rc=0
if [[ "${SKIP_GRPO}" != "1" ]] && (( grpo_rc != 0 )) && [[ "${EVAL_ON_GRPO_FAIL}" != "1" ]]; then
    log "GRPO rc=${grpo_rc} (not PASS) — SKIPPING §3.6 eval to save GPU \$ "
    log "  (set EVAL_ON_GRPO_FAIL=1 to eval anyway)."
else
    set +e
    if [[ -n "${STEP0_DIR}" ]]; then
        run_eval step0 "${STEP0_DIR}" --informational
        e0=$?
    else
        e0=0
        log "  step0 not supplied (eval-only mode) — SKIPPING step0 eval"
    fi
    run_eval final "${FINAL_DIR}"
    ef=$?
    if [[ "${EVAL_TRAIN_SEEDS}" == "1" ]]; then
        if [[ -n "${STEP0_DIR}" ]]; then
            run_eval step0_trainseeds "${STEP0_DIR}" --informational \
                "${SEED_START}" "${TRAIN_SEED_EVAL_SEEDS}" || true
        fi
        run_eval final_trainseeds "${FINAL_DIR}" --informational \
            "${SEED_START}" "${TRAIN_SEED_EVAL_SEEDS}" || true
    fi
    if [[ "${EVAL_INTERMEDIATE}" == "1" && "${DRY_RUN}" != "1" && -f "${GRPO_OUT}" ]]; then
        "${PYBIN}" -c 'import json,sys
c=json.load(open(sys.argv[1])).get("checkpoints") or {}
for rec in c.get("intermediate") or []:
    print(rec.get("step"), rec.get("path"))' "${GRPO_OUT}" |
        while read -r step path; do
            [[ -n "${step}" && -n "${path}" && "${path}" != "None" ]] || continue
            run_eval "step${step}" "${path}" --informational || true
        done
    fi
    set -e
    # Only the final eval gate folds into the spend/exit decision.
    eval_rc=$(( ef != 0 ? 1 : 0 ))
    [[ -n "${STEP0_DIR}" ]] && log "  step0 (informational) rc=${e0} — NOT folded into spend decision"
fi

[[ "${DRY_RUN}" != "1" ]] && stop_procs

log "=== done ===  results: ${RESULTS_DIR}"
[[ -f "${GRPO_OUT}" ]] && log "  - grpo:        ${GRPO_OUT}"
[[ -f "${RESULTS_DIR}/eval_step0.json" ]] && log "  - eval step0:  ${RESULTS_DIR}/eval_step0.json"
[[ -f "${RESULTS_DIR}/eval_final.json" ]] && log "  - eval final:  ${RESULTS_DIR}/eval_final.json (opponent=${EVAL_OPPONENT})"
[[ -f "${VLLM_LOG}" ]] && log "  - vllm log:    ${VLLM_LOG}"
[[ -n "${DUMP_ROLLOUTS}" && -d "${DUMP_ROLLOUTS}" ]] && \
    log "  - rollouts:    ${DUMP_ROLLOUTS}/ ($(find "${DUMP_ROLLOUTS}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ') rolls — reward_score_correlation.py input)"
(( grpo_rc != 0 )) && log "GRPO rc=${grpo_rc} (health gate fail or error — see ${GRPO_OUT})"
(( eval_rc != 0 )) && log "Eval rc=${eval_rc} (§3.6 gate not met or error — see eval_final.json)"
(( grpo_rc != 0 || eval_rc != 0 )) && exit 1
exit 0
