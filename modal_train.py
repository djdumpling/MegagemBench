"""Modal app: Phase-3 GRPO training and its paired-bootstrap eval gates.

PHASE-3 WORKFLOW:
  modal run modal_train.py::verify_env                    # pinned-stack sanity
  modal run modal_train.py::seam_tests_main               # CPU prep gate
  modal run modal_train.py::phase3_main --profile seam    # cheap GPU smoke
  modal run modal_train.py::phase3_main --profile evidence  # real spend
  phase3_main inlines the seam gate by default (--no-run-seam-first to skip).

OTHER ENTRYPOINTS:
  modal run modal_train.py::phase3_eval_only_main --adapter-path ...
  (eval panels live in modal_eval.py; release tooling in modal_release.py;
   interactive play in the standalone modal_play.py)

Shares one `modal.App` with modal_eval.py / modal_release.py via modal_common.
The dual-gate SPEND decision logic lives in megagem.training.dual_gate.
"""

from __future__ import annotations

import os
import pathlib
import time

from modal_common import (
    GPU,
    HF_CACHE,
    RESULTS_DIR,
    VLLM_CACHE_DIR,
    app,
    hf_cache,
    hf_secret,
    prime_secret,
    results_vol,
    vllm_cache,
    wandb_secret,
)
from modal_eval import panel_eval  # registers eval fns on the shared app


# Import probes only — no HF downloads, so no cache volume is needed.
@app.function(gpu=GPU, timeout=86400)
def verify_env() -> str:
    """Cheap pre-flight: prove the pinned stack is intact BEFORE spending on a
    real run."""
    import subprocess

    checks = [
        ("torch.cuda", "import torch; print(torch.cuda.is_available())"),
        ("transformers", "import transformers; print(transformers.__version__)"),
        ("flash_attn", "import torch, flash_attn_2_cuda; print('flash OK')"),
        (
            "trl-fork",
            "import trl, trl.import_utils as iu; from trl import GRPOTrainer; "
            "print('vllm_available=', iu.is_vllm_available())",
        ),
        ("vllm", "import vllm; print(vllm.__version__)"),
    ]
    out = []
    for name, code in checks:
        r = subprocess.run(["python", "-c", code], capture_output=True, text=True)
        out.append(f"[{'OK ' if r.returncode == 0 else 'FAIL'}] {name}: "
                   f"{(r.stdout or r.stderr).strip()}")
    report = "\n".join(out)
    print(report)
    # trl-fork line MUST say vllm_available= False (else upstream TRL leaked in).
    return report

# --- Phase 3 — GRPO + §3.6 paired-bootstrap eval ---
# Writes land on the Volume (the /repo mount is read-only); RESULTS_DIR /
# ADAPTER_ROOT are forced under /results.
SEAM_FILES = "tests/test_trl_seam.py tests/test_megagem_grpo.py"


@app.function(timeout=3600, volumes={HF_CACHE: hf_cache})  # no GPU
def seam_tests() -> dict:
    """Phase-3 HARD prerequisite: the line-pinned seam suite must be green on
    the installed TRL fork. CPU-only.
    """
    import subprocess

    r = subprocess.run(
        ["python", "-m", "pytest", *SEAM_FILES.split(), "-q"],
        cwd="/repo", capture_output=True, text=True,
    )
    # Phase-3 wiring sanity — additive run_game seam + heuristic shim + driver.
    glue = subprocess.run(
        ["python", "-c",
         "import inspect, sys; sys.path[:0]=['scripts/training'];"
         "from megagem import rollout; import phase3_grpo, phase3_eval;"
         "from megagem.training import heuristic_endpoint;"
         "assert 'caller_api_params' in inspect.signature(rollout.run_game)"
         ".parameters, 'run_game missing caller_api_params (#4)';"
         "assert hasattr(phase3_grpo,'make_onpolicy_rollout_func');"
         "print('phase3-glue OK')"],
        cwd="/repo", capture_output=True, text=True,
    )
    seam_ok = r.returncode == 0
    glue_ok = glue.returncode == 0
    tail = "\n".join((r.stdout + r.stderr).splitlines()[-25:])
    print(tail)
    print((glue.stdout + glue.stderr).strip())
    return {
        "seam_ok": seam_ok,
        "glue_ok": glue_ok,
        "passed": seam_ok and glue_ok,
        "seam_tail": tail,
        "glue_out": (glue.stdout + glue.stderr).strip(),
    }



@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={HF_CACHE: hf_cache, RESULTS_DIR: results_vol,
             VLLM_CACHE_DIR: vllm_cache},  # vllm_cache persists Dynamo/cudagraph
    secrets=[hf_secret, prime_secret, wandb_secret],
)
def phase3(
    *,
    profile: str = "seam",               # seam (cheap smoke) | evidence (§3.6)
    model: str = "djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2",
    served_model_name: str = "qwen/qwen3-4b-instruct",
    vllm_tokenizer: str = "",            # override tokenizer (e.g. base id) when
                                         # `model`'s chat_template is missing.
    steps: int = 0,                      # 0 ⇒ PROFILE default (seam 70/ev 200)
    k: int = 8,
    num_seeds: int = 0,                  # 0 ⇒ PROFILE default
    rows_per_gen: int = 96,
    eval_seeds: int = 0,                 # 0 ⇒ PROFILE default (seam 24/ev 60)
    checkpoint_every: int = 25,
    micro_cap: int = 0,                  # 0 ⇒ PROFILE default; raise to push spg fresher
    lr: str = "2e-5",
    kl_beta: str = "0.01",               # KL penalty β·KL(π‖π_ref); raise to 0.05
                                         # if KL crosses 0.1 mid-run
    vllm_gpu_mem_util: str = "0.3",      # co-host cap (phase2-vllm-cohosting-oom)
    vllm_ready_timeout_s: int = 1200,    # cold V1 compile ~5-10 min; cached after
    split_gpus: bool = False,            # H200:2 mode: trainer on GPU0, vLLM on GPU1
    train_cuda_visible_devices: str = "0",
    vllm_cuda_visible_devices: str = "1",  # comma-sep (e.g. "1,2,3,4,5,6,7") for N>1 / TP>1
    vllm_tensor_parallel_size: int = 1,  # vLLM TP; Qwen3-4B has 8 KV heads ⇒ TP ∈ {1,2,4,8}
    n_vllm: int = 1,                     # >1 ⇒ data-parallel: N workers; needs N×TP GPUs in vllm_cuda_visible_devices
    vllm_base_port: int = 8000,          # worker i listens on base_port+i
    pytorch_cuda_alloc_conf: str = "expandable_segments:True",
    # --- 8×H200 on-policy batch shape (phase3-rl-resize-8xh200) --- #
    on_policy: bool = False,             # ga=spg ⇒ ONE optimizer step per
                                         # generation: large on-policy batch at
                                         # ZERO extra activation memory. Legacy
                                         # ga=1 ⇒ spg off-policy refresh-starved.
    gradient_accumulation_steps: int = 0,  # explicit ga; 0 ⇒ legacy ga=1
                                         # (ignored when on_policy=True).
    num_processes: int = 1,              # DDP world size (>1 ⇒ torchrun; needs
                                         # PHASE3_ALLOW_DDP=1; halves ga).
    vllm_prefix_caching: bool = True,    # VLLM_PREFIX_CACHING (resize_smoke
                                         # measured ~56.5% hit @K=8).
    eval_on_grpo_fail: bool = False,
    run_seam_first: bool = True,         # inline HARD gate before any $
    max_parallel: int = 32,              # concurrent games per roll & eval; vLLM
                                         # batches → ~5-8× wall speedup
    eval_samples_per_seed: int = 0,      # Lever A K-sample avg; 0 ⇒ PROFILE default
                                         # (seam K=1, evidence K=8)
    eval_temperature: float | None = None,  # Lever C symmetric T; 0.0 = greedy
    dump_rollouts: bool = False,         # persist actor-tagged schema-v3 games
                                         # under rollout_dumps/roll_NNN/ (~80MB
                                         # for evidence; auto-on for evidence)
    fixed_train_seeds: bool = False,
    allow_low_rows_per_gen: bool = False,
    eval_train_seeds: bool = False,
    train_seed_eval_seeds: int = 0,
    eval_intermediate: bool = False,
    # --- §3.3 lagged-self opponent pool --- #
    opponent_pool: bool = True,          # False ⇒ legacy single heuristic
    snapshot_every: int = 0,             # 0 ⇒ PROFILE default (seam 10/ev 25)
    max_snapshots: int = 0,              # 0 ⇒ PROFILE default (seam 2/ev 5)
    opp_anneal_start: int = -1,          # -1 ⇒ PROFILE default (seam 10/ev 50);
                                         # 0 IS A VALID OVERRIDE (means "anneal
                                         # starts at step 0" — used by v3 to
                                         # make the legacy anneal path inert).
                                         # Codex round-1 fix: pre-fix sentinel
                                         # was 0, which collided with this
                                         # legitimate value so `--opp-anneal-
                                         # start 0` was silently swallowed.
    opp_anneal_end: int = -1,            # -1 ⇒ run_phase3.sh defaults to STEPS;
                                         # 0 means "anneal ends immediately"
                                         # (no anneal phase — v3 anneal-zero
                                         # mode). Same sentinel-collision fix.
    opp_anneal_pmax: str = "0.7",
    opp_anchor_floor: str = "0.15",      # repl_08: min draw P for the
                                         # pinned step_0 anchor (so PFSP
                                         # can't drown it once trainee
                                         # crushes the anchor). 0.0 ⇒
                                         # pre-Codex behaviour.
    p_heuristic: str = "0.0",            # repl_08 v3: P(draw heuristic) of
                                         # TOTAL draws (absolute share —
                                         # single-tier mix-gate with p_api).
                                         # Internal decay:
                                         # p_heuristic·max(0.10, (1-WR)²).
                                         # 0.0 ⇒ heuristic OFF in training
                                         # (seam_smoke_02 / pure self-play).
                                         # Set to 0.20-0.30 to reintroduce
                                         # the easy-win signal. Construction
                                         # guard: p_heuristic + p_api ≤ 1.
    p_heuristic_bid_noise: str = "0.0",  # repl_08 v3.1: width of det-per-
                                         # state ε around BID_FRACTION=0.5.
                                         # 0.0 ⇒ v1 stationary heuristic
                                         # (bid = floor(0.5 · max_bid)).
                                         # 0.15 ⇒ fraction ∈ [0.35, 0.65]
                                         # across distinct auction states
                                         # (broader bid range, harder to
                                         # exploit by memorising a constant
                                         # bid). Det-per-state via SHA-256
                                         # hash → preserves K-group credit
                                         # assignment.
    dapo_heuristic_exempt: bool = True,  # repl_08 v3.1: skip the relative
                                         # DAPO threshold for heuristic
                                         # K-groups (absolute floor still
                                         # applies). Low std vs a stationary
                                         # opponent means "policy learned to
                                         # exploit" not "degenerate group".
    heuristic_anneal_end: int = 0,       # repl_08 v3.2: step to anneal the
                                         # heuristic draw probability to 0
                                         # (linear, decoupled from the (1-WR)²
                                         # floor). 0 ⇒ floor-only (persists at
                                         # ≈0.10·p_heuristic). Set ~2×
                                         # snapshot_every for bootstrap-only.
    hetero_opponents: bool = False,      # PSRO: two DIFFERENT opponents per table
                                         # (vs two clones); pair w/ a diverse pool.
    rotate_seats: bool = True,           # round-robin trainable seat per roll
                                         # (constant within each K-group). True
                                         # trains all 3 seats — the TrueSkill/
                                         # panel eval rates seat0/1/2. False ⇒
                                         # pin --trainable-seat (legacy).
    opp_api_models: str = "",            # comma-sep API ids; "" ⇒ lagged-self
    opp_api_prob: str = "0.0",
    opp_api_weights: str = "",           # comma-sep ints aligned with
                                         # opp_api_models; "" ⇒ uniform
    opp_pool_seed: int = 0,
    seed: int = 0,                       # TRL/sampling master seed for phase3_grpo
    max_loras: int = 10,                 # evidence-compatible vLLM LoRA slots
    max_cpu_loras: int = 11,
    # GRPO clip — epsilon "" ⇒ TRL symmetric 0.2. epsilon_high defaults to
    # 0.28 (DAPO clip-higher) end-to-end; pass "0.2" to restore symmetric clip.
    epsilon: str = "",
    epsilon_high: str = "0.28",
    eval_opponent: str = "heuristic",    # non-heuristic ⇒ re-baseline --threshold
    threshold: str = "",                 # "" ⇒ phase3_eval default (+2)
    # --- DAPO dynamic sampling (Finding 17 mitigation) -------------------- #
    dapo_dynamic_sampling: bool = False,  # drop degenerate K-groups
    dapo_min_group_std: str = "1e-3",     # abs threshold floor
    dapo_opp_rel_threshold: str = "0.0",  # >0 ⇒ opp-aware via per-kind EMA std
    dapo_ema_alpha: str = "0.9",
    run_tag: str = "modal",
) -> dict:
    """run_phase3.sh on a Modal GPU. Inlines the seam suite first
    (run_seam_first=True) so failures abort before vLLM bring-up. PROFILE
    banner + eval `valid_as_final_evidence` / `final_gate_caveats` are the
    script's, intact.
    """
    import glob
    import json
    import subprocess

    if profile not in ("seam", "evidence"):
        raise ValueError(f"profile must be seam|evidence, got {profile!r}")

    # Keep long evidence runs volume-light by default. Pass --dump-rollouts only
    # for forensic runs where every per-step game transcript is needed.

    # HF-token preflight — model is PRIVATE; vLLM `serve` would otherwise
    # 40-line-traceback AFTER GPU bring-up. Distinguish no-token vs no-access.
    tok = (os.environ.get("HF_TOKEN")
           or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    if not tok.strip():
        return {"phase3_rc": 2, "aborted": (
            "HF_TOKEN not in the container env — the `huggingface-secret` "
            "Modal Secret is missing or its KEY isn't HF_TOKEN. Recreate: "
            "modal secret create huggingface-secret HF_TOKEN=hf_xxx  (the "
            "token must have READ access to the private "
            f"{model!r}).")}
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(model, token=tok)  # cheap auth/access probe
    except Exception as e:  # noqa: BLE001
        return {"phase3_rc": 2, "aborted": (
            f"HF_TOKEN is present but cannot access {model!r}: "
            f"{type(e).__name__}: {e}. The token is delivered but lacks "
            f"read access to this PRIVATE repo (or the repo id is wrong). "
            f"Use a token from an account with access to djdumpling/…-v2.")}
    # Stacks vary on which env name they read; mirror so vLLM's subprocess
    # (huggingface_hub) and transformers both see it regardless.
    os.environ["HF_TOKEN"] = tok
    os.environ["HUGGING_FACE_HUB_TOKEN"] = tok

    # (1) inline seam gate — cheap, fail BEFORE vLLM/GRPO spend.
    if run_seam_first:
        sr = subprocess.run(
            ["python", "-m", "pytest", *SEAM_FILES.split(), "-q"],
            cwd="/repo", capture_output=True, text=True,
        )
        if sr.returncode != 0:
            tail = "\n".join((sr.stdout + sr.stderr).splitlines()[-30:])
            print(tail)
            return {"phase3_rc": 2, "aborted": "seam tests FAILED — "
                    "refusing to spend GPU (phase-3 prep checklist)",
                    "seam_tail": tail}

    out_dir = f"{RESULTS_DIR}/phase3_grpo_{profile}_{run_tag}"
    env = {
        **os.environ,
        "PROFILE": profile,
        "MODEL_PATH": model,
        "SERVED_NAME": served_model_name,
        "VLLM_TOKENIZER": vllm_tokenizer,
        "K": str(k),
        "ROWS_PER_GEN": str(rows_per_gen),
        "CHECKPOINT_EVERY": str(checkpoint_every),
        "LR": lr,
        "KL_BETA": kl_beta,
        "VLLM_GPU_MEM_UTIL": str(vllm_gpu_mem_util),
        "VLLM_READY_TIMEOUT_S": str(vllm_ready_timeout_s),
        "PHASE3_SPLIT_GPUS": "1" if split_gpus else "0",
        "PHASE3_TRAIN_CUDA_VISIBLE_DEVICES": train_cuda_visible_devices,
        "PHASE3_VLLM_CUDA_VISIBLE_DEVICES": vllm_cuda_visible_devices,
        "VLLM_TENSOR_PARALLEL_SIZE": str(vllm_tensor_parallel_size),
        "PHASE3_N_VLLM": str(n_vllm),
        "PHASE3_VLLM_BASE_PORT": str(vllm_base_port),
        "VLLM_PREFIX_CACHING": "1" if vllm_prefix_caching else "0",
        "ON_POLICY": "1" if on_policy else "0",
        "NUM_PROCESSES": str(num_processes),
        "PYTORCH_CUDA_ALLOC_CONF": pytorch_cuda_alloc_conf,
        "EVAL_ON_GRPO_FAIL": "1" if eval_on_grpo_fail else "0",
        "MAX_PARALLEL": str(max_parallel),
        "EVAL_TEMPERATURE": ("" if eval_temperature is None
                             else str(eval_temperature)),
        "DUMP_ROLLOUTS": "1" if dump_rollouts else "",
        "FIXED_TRAIN_SEEDS": "1" if fixed_train_seeds else "0",
        "ALLOW_LOW_ROWS_PER_GEN": "1" if allow_low_rows_per_gen else "0",
        "EVAL_TRAIN_SEEDS": "1" if eval_train_seeds else "0",
        "EVAL_INTERMEDIATE": "1" if eval_intermediate else "0",
        "OPPONENT_POOL": "1" if opponent_pool else "0",
        "OPP_ANNEAL_PMAX": opp_anneal_pmax,
        "OPP_ANCHOR_FLOOR": opp_anchor_floor,
        "P_HEURISTIC": p_heuristic,
        "P_HEURISTIC_BID_NOISE": p_heuristic_bid_noise,
        "PHASE3_HEURISTIC_BID_NOISE": p_heuristic_bid_noise,
        "HEURISTIC_ANNEAL_END": str(heuristic_anneal_end),
        "ROTATE_SEATS": "1" if rotate_seats else "0",
        "HETERO_OPPONENTS": "1" if hetero_opponents else "0",
        "PHASE3_DAPO_HEURISTIC_EXEMPT": "1" if dapo_heuristic_exempt else "0",
        "OPP_API_MODELS": opp_api_models,
        "OPP_API_PROB": opp_api_prob,
        "OPP_API_WEIGHTS": opp_api_weights,
        "OPP_POOL_SEED": str(opp_pool_seed),
        "SEED": str(seed),
        "MAX_LORAS": str(max_loras),
        "MAX_CPU_LORAS": str(max_cpu_loras),
        "EPSILON": epsilon,
        "EPSILON_HIGH": epsilon_high,
        "EVAL_OPPONENT": eval_opponent,
        "THRESHOLD": threshold,
        # DAPO dynamic sampling — phase3_grpo.py reads these directly (no
        # run_phase3.sh / argparse plumbing needed).
        "PHASE3_DAPO_DYNAMIC_SAMPLING": "1" if dapo_dynamic_sampling else "0",
        "PHASE3_DAPO_MIN_GROUP_STD": dapo_min_group_std,
        "PHASE3_DAPO_OPP_REL_THRESHOLD": dapo_opp_rel_threshold,
        "PHASE3_DAPO_EMA_ALPHA": dapo_ema_alpha,
        # Force results/adapters onto the WRITABLE Volume (the /repo mount is
        # read-only). run_phase3.sh derives ADAPTER_ROOT=${RESULTS_DIR}/adapters.
        "RESULTS_DIR": out_dir,
        "SEAM_VERIFIED": "1",            # (1) above already ran the suite
        "PYBIN": "python",
        # Quiet per-call tqdm / hub / tokenizer warnings; correctness-neutral.
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # Per-rollout telemetry (vLLM /metrics + nvidia-smi + AsyncOpenAI
        # status counts). Cheap (3 polls/sec total) and prints a one-block
        # summary after each rollout's "games complete in Xs" line. Flip to
        # "0" to disable for production runs that care about log volume.
        "PHASE3_TELEMETRY": os.environ.get("PHASE3_TELEMETRY", "1"),
        "PHASE3_TELEMETRY_INTERVAL": os.environ.get(
            "PHASE3_TELEMETRY_INTERVAL", "1.0"),
    }
    # Wandb — pre-gen WANDB_RUN_ID so the §3.6 eval subprocess resumes the
    # same run (TRL picks it up at wandb.init() time).
    if env.get("WANDB_API_KEY"):
        env["WANDB_MODE"] = "online"
        env.setdefault("WANDB_PROJECT", "megagem-phase3")
        env["WANDB_RUN_GROUP"] = profile         # seam vs evidence groups
        env["WANDB_NAME"] = f"phase3_{profile}_{run_tag}"
        env["WANDB_RUN_ID"] = (
            f"phase3_{profile}_{run_tag}_{int(time.time())}"
        )
        print(f"[phase3] wandb online: project="
              f"{env['WANDB_PROJECT']!r} group={env['WANDB_RUN_GROUP']!r} "
              f"name={env['WANDB_NAME']!r} id={env['WANDB_RUN_ID']!r}")
    else:
        print("[phase3] wandb disabled (no WANDB_API_KEY in env).")
    # 0-sentinel knobs: omitted ⇒ run_phase3.sh's PROFILE default applies.
    if steps:
        env["STEPS"] = str(steps)
    if num_seeds:
        env["NUM_SEEDS"] = str(num_seeds)
    if eval_seeds:
        env["EVAL_SEEDS"] = str(eval_seeds)
    if eval_samples_per_seed:
        env["EVAL_SAMPLES_PER_SEED"] = str(eval_samples_per_seed)
    if train_seed_eval_seeds:
        env["TRAIN_SEED_EVAL_SEEDS"] = str(train_seed_eval_seeds)
    if micro_cap:
        env["PHASE2_MICRO_CAP"] = str(micro_cap)
    # GRAD_ACCUM 0-sentinel: omitted ⇒ run_phase3.sh legacy ga=1 (or ga=spg when
    # ON_POLICY=1). An explicit value amortizes a large gen batch over ga steps.
    if gradient_accumulation_steps:
        env["GRAD_ACCUM"] = str(gradient_accumulation_steps)
    if snapshot_every:
        env["SNAPSHOT_EVERY"] = str(snapshot_every)
    if max_snapshots:
        env["MAX_SNAPSHOTS"] = str(max_snapshots)
    # Use `>= 0` (not truthiness) so an explicit `--opp-anneal-start 0` /
    # `--opp-anneal-end 0` propagates. -1 is the "use shell PROFILE default"
    # sentinel. Pre-fix this used `if opp_anneal_start:` which dropped both
    # the sentinel (0) AND the legitimate "anneal at step 0" value.
    if opp_anneal_start >= 0:
        env["OPP_ANNEAL_START"] = str(opp_anneal_start)
    if opp_anneal_end >= 0:
        env["OPP_ANNEAL_END"] = str(opp_anneal_end)

    rc = subprocess.run(
        ["bash", "scripts/training/run_phase3.sh"], cwd="/repo", env=env,
    ).returncode

    # Harvest the decision artifacts (already on the Volume; load for the
    # return trip + a stable manifest).
    files: dict = {}
    names = ["phase3_grpo.json", "train_log.json"]
    names.extend(p.name for p in sorted(pathlib.Path(out_dir).glob("eval_*.json")))
    for name in names:
        p = pathlib.Path(out_dir) / name
        if p.exists():
            try:
                files[name] = json.loads(p.read_text())
            except Exception as e:  # noqa: BLE001
                files[name] = {"_unreadable": str(e)}
    logs = sorted(glob.glob(f"{out_dir}/*.log"))
    results_vol.commit()
    vllm_cache.commit()                  # persist Dynamo+cudagraph compile so
                                         # the next run polls-ready in ~30 s.
    return {
        "phase3_rc": rc,
        "profile": profile,
        "results_dir": out_dir,
        "files": files,
        "logs": [pathlib.Path(x).name for x in logs],
    }
@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={HF_CACHE: hf_cache, RESULTS_DIR: results_vol,
             VLLM_CACHE_DIR: vllm_cache},
    secrets=[hf_secret, prime_secret, wandb_secret],
)
def phase3_eval_only(
    *,
    adapter_path: str,                       # Volume path to LoRA (e.g.
                                             # /results/.../adapters/step_200).
    step0_adapter_path: str = "",            # optional B≡0 sanity eval; "" skips.
    model: str = "djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2",
    served_model_name: str = "qwen/qwen3-4b-instruct",
    eval_seeds: int = 60,
    seed_start: int = 20000,
    eval_samples_per_seed: int = 1,          # Lever A
    eval_temperature: float | None = None,   # Lever C (0.0 = greedy)
    max_parallel: int = 32,
    vllm_gpu_mem_util: str = "0.3",
    vllm_ready_timeout_s: int = 1200,
    eval_opponent: str = "heuristic",        # non-heuristic ⇒ frontier transfer
                                             # eval; needs PRIME_API_KEY + a
                                             # re-baselined --threshold.
    threshold: str = "",                     # "" ⇒ +2 (heuristic-calibrated);
                                             # "0.0" for frontier transfer.
    run_tag: str = "eval_only",
) -> dict:
    """Eval-only mode — no training. Spins up vLLM + heuristic shim, pushes
    the adapter, runs §3.6 paired-bootstrap eval. Two intended uses:
    Lever C (T=0 greedy corroboration) and K-sample averaging (tighter CI
    on an existing checkpoint without retraining).
    """
    import glob
    import json
    import subprocess

    # HF token preflight (private base model)
    tok = (os.environ.get("HF_TOKEN")
           or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    if not tok.strip():
        return {"phase3_rc": 2, "aborted": (
            "HF_TOKEN not in container env — recreate huggingface-secret.")}
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(model, token=tok)
    except Exception as e:  # noqa: BLE001
        return {"phase3_rc": 2, "aborted": (
            f"HF_TOKEN cannot access {model!r}: {type(e).__name__}: {e}")}
    os.environ["HF_TOKEN"] = tok
    os.environ["HUGGING_FACE_HUB_TOKEN"] = tok

    out_dir = f"{RESULTS_DIR}/phase3_eval_only_{run_tag}"
    env = {
        **os.environ,
        "SKIP_GRPO": "1",
        "EXT_FINAL_DIR": adapter_path,
        "EXT_STEP0_DIR": step0_adapter_path,
        "MODEL_PATH": model,
        "SERVED_NAME": served_model_name,
        "EVAL_SEEDS": str(eval_seeds),
        "EVAL_SEED_START": str(seed_start),
        "MAX_PARALLEL": str(max_parallel),
        "EVAL_SAMPLES_PER_SEED": str(eval_samples_per_seed),
        "EVAL_TEMPERATURE": ("" if eval_temperature is None
                             else str(eval_temperature)),
        "VLLM_GPU_MEM_UTIL": str(vllm_gpu_mem_util),
        "VLLM_READY_TIMEOUT_S": str(vllm_ready_timeout_s),
        "EVAL_OPPONENT": eval_opponent,
        "THRESHOLD": threshold,
        "RESULTS_DIR": out_dir,
        "SEAM_VERIFIED": "1",
        "PYBIN": "python",
        # Quiet HF progress bars (mirror the phase3 entrypoint).
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    # Wandb — standalone eval (no trainer to resume); group="eval_only".
    if env.get("WANDB_API_KEY"):
        env["WANDB_MODE"] = "online"
        env.setdefault("WANDB_PROJECT", "megagem-phase3")
        env["WANDB_RUN_GROUP"] = "eval_only"
        env["WANDB_NAME"] = f"phase3_eval_only_{run_tag}"
        env["WANDB_RUN_ID"] = (
            f"phase3_eval_only_{run_tag}_{int(time.time())}"
        )
        print(f"[phase3_eval_only] wandb online: project="
              f"{env['WANDB_PROJECT']!r} group={env['WANDB_RUN_GROUP']!r} "
              f"name={env['WANDB_NAME']!r} id={env['WANDB_RUN_ID']!r}")
    else:
        print("[phase3_eval_only] wandb disabled (no WANDB_API_KEY in env).")

    rc = subprocess.run(
        ["bash", "scripts/training/run_phase3.sh"], cwd="/repo", env=env,
    ).returncode

    files: dict = {}
    for name in ("eval_step0.json", "eval_final.json"):
        p = pathlib.Path(out_dir) / name
        if p.exists():
            try:
                files[name] = json.loads(p.read_text())
            except Exception as e:  # noqa: BLE001
                files[name] = {"_unreadable": str(e)}
    logs = sorted(glob.glob(f"{out_dir}/*.log"))
    results_vol.commit()
    vllm_cache.commit()
    return {
        "phase3_rc": rc,
        "results_dir": out_dir,
        "adapter_path": adapter_path,
        "eval_temperature": eval_temperature,
        "eval_samples_per_seed": eval_samples_per_seed,
        "files": files,
        "logs": [pathlib.Path(x).name for x in logs],
    }


@app.local_entrypoint()
def seam_tests_main():
    res = seam_tests.remote()
    print(f"\nseam_ok={res['seam_ok']}  glue_ok={res['glue_ok']}  "
          f"=> {'PASS — safe to GPU' if res['passed'] else 'FAIL — DO NOT SPEND'}")
    print(res["glue_out"])
    if not res["passed"]:
        print("\n--- seam tail ---\n" + res["seam_tail"])


@app.local_entrypoint()
def phase3_main(
    profile: str = "seam",
    model: str = "djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2",
    served_model_name: str = "qwen/qwen3-4b-instruct",
    vllm_tokenizer: str = "",
    steps: int = 0,                      # 0 ⇒ PROFILE default (seam 70/ev 200)
    k: int = 8,
    num_seeds: int = 0,
    rows_per_gen: int = 96,
    eval_seeds: int = 0,
    checkpoint_every: int = 25,
    micro_cap: int = 0,
    lr: str = "2e-5",
    kl_beta: str = "0.01",
    vllm_gpu_mem_util: str = "0.3",
    vllm_ready_timeout_s: int = 1200,
    split_gpus: bool = False,
    train_cuda_visible_devices: str = "0",
    vllm_cuda_visible_devices: str = "1",
    vllm_tensor_parallel_size: int = 1,
    n_vllm: int = 1,
    vllm_base_port: int = 8000,
    pytorch_cuda_alloc_conf: str = "expandable_segments:True",
    # --- 8×H200 on-policy batch shape (phase3-rl-resize-8xh200) --- #
    on_policy: bool = False,
    gradient_accumulation_steps: int = 0,
    num_processes: int = 1,
    vllm_prefix_caching: bool = True,
    background: bool = False,
    eval_on_grpo_fail: bool = False,
    run_seam_first: bool = True,
    max_parallel: int = 32,
    eval_samples_per_seed: int = 0,
    eval_temperature: float | None = None,
    dump_rollouts: bool = False,
    fixed_train_seeds: bool = False,
    allow_low_rows_per_gen: bool = False,
    eval_train_seeds: bool = False,
    train_seed_eval_seeds: int = 0,
    eval_intermediate: bool = False,
    opponent_pool: bool = True,
    snapshot_every: int = 0,
    max_snapshots: int = 0,
    opp_anneal_start: int = -1,         # -1 ⇒ PROFILE default; 0 is a
                                        # legitimate override (anneal at
                                        # step 0). Codex round-1 sentinel fix.
    opp_anneal_end: int = -1,           # -1 ⇒ PROFILE default; 0 means
                                        # "anneal ends immediately".
    opp_anneal_pmax: str = "0.7",
    opp_anchor_floor: str = "0.15",     # repl_08 anchor probability floor;
                                        # see phase3() docstring above
    p_heuristic: str = "0.0",           # repl_08 v3 — see phase3() above.
                                        # 0.0 ⇒ pure self-play (seam_smoke_02
                                        # behaviour). Recommended 0.20-0.30
                                        # for evidence runs (heuristic-with-
                                        # decay reintroduces easy-win signal).
    p_heuristic_bid_noise: str = "0.0", # repl_08 v3.1 — see phase3() above.
                                        # 0.15 recommended to soften the
                                        # constant-bid exploit attack surface.
    dapo_heuristic_exempt: bool = True, # repl_08 v3.1 — see phase3() above.
    heuristic_anneal_end: int = 0,      # repl_08 v3.2 — anneal heuristic to 0
                                        # by this step (0 ⇒ floor-only). Set
                                        # ~2×snapshot_every for bootstrap-only.
    hetero_opponents: bool = False,     # PSRO heterogeneous tables (2 diff opponents)
    rotate_seats: bool = True,          # round-robin trainable seat per roll
                                        # (train all 3 seats; panel rates all).
    opp_api_models: str = "",
    opp_api_prob: str = "0.0",
    opp_api_weights: str = "",
    opp_pool_seed: int = 0,
    seed: int = 0,                       # TRL/sampling master seed (phase3_grpo --seed)
    max_loras: int = 10,
    max_cpu_loras: int = 11,
    # epsilon "" ⇒ TRL symmetric 0.2; epsilon_high "0.28" = DAPO clip-higher.
    epsilon: str = "",
    epsilon_high: str = "0.28",
    eval_opponent: str = "heuristic",
    threshold: str = "",
    # --- DAPO dynamic sampling --- #
    dapo_dynamic_sampling: bool = False,
    dapo_min_group_std: str = "1e-3",
    dapo_opp_rel_threshold: str = "0.0",
    dapo_ema_alpha: str = "0.9",
    # --- Dual-gate SPEND (panel-vs-Flash heldout after §3.6 eval) --- #
    # Codex review: n=30 seeds × 3 seats = 90 games gives SE ≈ 5% on win-rate,
    # so a hard threshold of 0.35 has ~16% Type-I error at true WR=0.30 (SFT
    # baseline). Default bumped to 60 (= 180 games, matches the repl_07 panel-
    # vs-Flash heldout n=180 that established the 30%↔34.4% repl_07 result as
    # "within noise of zero"). Threshold stays at 0.35 — operator should treat
    # observed WR ≥ 0.35 at n=180 as "directionally favorable, statistically
    # ~+5pp over SFT baseline", NOT as a strict-confidence proof.
    dual_gate: bool = False,
    # repl_08 review (P0-2): the §3.6 gate measures improvement vs the SAME
    # deterministic heuristic the policy TRAINS against, so it is Goodhart-
    # fooled by opponent-specific exploits (phase3_heuristic_exploit_confirmed:
    # +16.5 vs heuristic / ~noise vs Flash). With dual_gate_flash_primary=True
    # (default) the held-out vs-Flash gate is the BINDING spend criterion and
    # §3.6 is informational — it can no longer VETO a policy that transfers to
    # Flash, nor rubber-stamp one that doesn't. False ⇒ legacy AND (both gates
    # must pass; §3.6 short-circuits the Flash eval to save $ on a sure NO).
    dual_gate_flash_primary: bool = True,
    dual_gate_flash_threshold: float = 0.35,
    dual_gate_flash_seeds: int = 60,
    dual_gate_flash_seed_start: int = 30000,
    # Codex round-2: the dual-gate must do more than `observed_WR > threshold`
    # (a point-estimate is no spend gate). When `dual_gate_sft_baseline_wr > 0`
    # we additionally run a one-sided z-test for "is the observed RL win-rate
    # significantly above the SFT-baseline win-rate?" at the given alpha.
    # PASS requires BOTH the absolute threshold AND the stat test to clear.
    # Set `dual_gate_sft_baseline_wr=0.0` to disable the stat test (legacy
    # threshold-only mode, NOT recommended for spend decisions).
    dual_gate_sft_baseline_wr: float = 0.30,
    dual_gate_significance_alpha: float = 0.05,
    run_tag: str = "modal",
):
    import json

    # Under-fed-worker WARN: at n_vllm>1, max_parallel needs to scale so the
    # DP workers don't sit idle behind the semaphore. Rough rule: ≥16 concurrent
    # games per worker to keep batching saturated.
    if n_vllm > 1 and max_parallel < n_vllm * 16:
        print(
            f"[phase3_main] WARN: n_vllm={n_vllm} but max_parallel={max_parallel} "
            f"(<{n_vllm * 16}). DP workers will be under-fed — most will sit idle "
            f"behind the asyncio semaphore. Suggest --max-parallel {n_vllm * 16} "
            f"(or higher; remember API-prob runs may need rate-limit headroom).",
            flush=True)

    kwargs = dict(
        profile=profile, model=model, served_model_name=served_model_name,
        vllm_tokenizer=vllm_tokenizer,
        steps=steps, k=k, num_seeds=num_seeds, rows_per_gen=rows_per_gen,
        eval_seeds=eval_seeds, checkpoint_every=checkpoint_every,
        micro_cap=micro_cap, lr=lr, kl_beta=kl_beta,
        vllm_gpu_mem_util=vllm_gpu_mem_util,
        vllm_ready_timeout_s=vllm_ready_timeout_s,
        split_gpus=split_gpus,
        train_cuda_visible_devices=train_cuda_visible_devices,
        vllm_cuda_visible_devices=vllm_cuda_visible_devices,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        n_vllm=n_vllm,
        vllm_base_port=vllm_base_port,
        pytorch_cuda_alloc_conf=pytorch_cuda_alloc_conf,
        on_policy=on_policy,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_processes=num_processes,
        vllm_prefix_caching=vllm_prefix_caching,
        eval_on_grpo_fail=eval_on_grpo_fail, run_seam_first=run_seam_first,
        max_parallel=max_parallel,
        eval_samples_per_seed=eval_samples_per_seed,
        eval_temperature=eval_temperature,
        dump_rollouts=dump_rollouts,
        fixed_train_seeds=fixed_train_seeds,
        allow_low_rows_per_gen=allow_low_rows_per_gen,
        eval_train_seeds=eval_train_seeds,
        train_seed_eval_seeds=train_seed_eval_seeds,
        eval_intermediate=eval_intermediate,
        opponent_pool=opponent_pool, snapshot_every=snapshot_every,
        max_snapshots=max_snapshots, opp_anneal_start=opp_anneal_start,
        opp_anneal_end=opp_anneal_end, opp_anneal_pmax=opp_anneal_pmax,
        opp_anchor_floor=opp_anchor_floor,
        p_heuristic=p_heuristic,
        p_heuristic_bid_noise=p_heuristic_bid_noise,
        dapo_heuristic_exempt=dapo_heuristic_exempt,
        heuristic_anneal_end=heuristic_anneal_end,
        rotate_seats=rotate_seats, hetero_opponents=hetero_opponents,
        opp_api_models=opp_api_models, opp_api_prob=opp_api_prob,
        opp_api_weights=opp_api_weights,
        opp_pool_seed=opp_pool_seed, seed=seed, max_loras=max_loras,
        max_cpu_loras=max_cpu_loras, epsilon=epsilon,
        epsilon_high=epsilon_high, eval_opponent=eval_opponent,
        threshold=threshold,
        dapo_dynamic_sampling=dapo_dynamic_sampling,
        dapo_min_group_std=dapo_min_group_std,
        dapo_opp_rel_threshold=dapo_opp_rel_threshold,
        dapo_ema_alpha=dapo_ema_alpha,
        run_tag=run_tag,
    )
    if background:
        call = phase3.spawn(**kwargs)
        print(f"\nspawned phase3 background call: {call.object_id}")
        print(f"dashboard: {call.get_dashboard_url()}")
        print("results volume path will be:")
        print(f"  /results/phase3_grpo_{profile}_{run_tag}")
        print("poll artifacts with:")
        print(f"  modal volume ls megagem-results phase3_grpo_{profile}_{run_tag}")
        return

    res = phase3.remote(**kwargs)
    if res.get("aborted"):
        print(f"\nABORTED: {res['aborted']}")
        return

    local = pathlib.Path(f"results/phase3_{profile}_{run_tag}_modal")
    local.mkdir(parents=True, exist_ok=True)
    for name, blob in res.get("files", {}).items():
        (local / name).write_text(json.dumps(blob, indent=2, default=str))

    g = res.get("files", {}).get("phase3_grpo.json", {})
    ef = res.get("files", {}).get("eval_final.json", {})
    print(f"\nphase3 rc={res['phase3_rc']}  profile={res['profile']}")
    print(f"  Volume={res['results_dir']}  + local copy={local}")
    print(f"  GRPO: status={g.get('status')} "
          f"cadence={g.get('on_policy_cadence')} "
          f"is_seam_shape={g.get('is_seam_shape')} "
          f"refreshes={g.get('n_onpolicy_refreshes')}")
    if g.get("train_log"):
        print(f"  train log: full per-step metrics → {local}/train_log.json")
    op = g.get("opponent_pool") or {}
    if op.get("enabled"):
        print(f"  pool: snapshot_every={op.get('snapshot_every')} "
              f"keep_last={op.get('max_snapshots')} "
              f"final={op.get('final_state')} "
              f"events={len(g.get('snapshot_events') or [])}")
    if ef:
        print(f"  §3.6 eval(final): status={ef.get('status')} "
              f"gate_pass={ef.get('gate_pass')} "
              f"mean_delta={ef.get('mean_delta')} ci_low={ef.get('ci_low')}")
        print(f"     valid_as_final_evidence={ef.get('valid_as_final_evidence')}")
        for c in ef.get("final_gate_caveats", []):
            print(f"     ⚠ {c}")
    if g.get("rollout_dump_dir"):
        base = pathlib.PurePath(res["results_dir"]).name
        # `modal volume get` lands the dump under ./<base>/rollout_dumps/ —
        # not under the `local` dir (which only holds the JSON blobs).
        print(f"  rollout dumps: {g['rollout_dump_dir']}/roll_NNN/ — after "
              f"`modal volume get`, inspect with\n"
              f"     python3 scripts/analysis/inspect_rollouts.py "
              f"{base}/rollout_dumps\n"
              f"     python3 scripts/training/reward_score_correlation.py "
              f"--corpus {base}/rollout_dumps/roll_000")
    # SFT-baseline completion lengths (roll 0, LoRA B≡0) — quick length-drift glance.
    base_len = g.get("sft_baseline_lengths") or {}
    if base_len and "tokens" in base_len:
        tk = base_len["tokens"] or {}
        print(f"  SFT-baseline lengths (roll 0): tokens "
              f"mean={tk.get('mean')} median={tk.get('median')} "
              f"p95={tk.get('p95')} max={tk.get('max')} "
              f"(N={base_len.get('n_completions')})")
    print(f"  raw: modal volume get megagem-results "
          f"$(basename {res['results_dir']})/ ./")

    # ---- Dual-gate SPEND criterion --------------------------------------- #
    # Single-gate §3.6 (ci_low > +2 vs heuristic) is permeable to opponent-
    # overfitting, so spend decisions also require a vs-Flash heldout. The
    # gate composition, cluster-conservative z-test, and verdict rendering
    # live in megagem.training.dual_gate (pure, unit-testable).
    if dual_gate:
        from megagem.training.dual_gate import assess_flash_gate

        gate_36_pass = bool(ef and ef.get("gate_pass")
                            and ef.get("valid_as_final_evidence"))
        # Final adapter path. phase3_grpo.json records it under
        # `checkpoints.final` (absolute path on the Volume).
        final_adapter = (g.get("checkpoints") or {}).get("final") or ""
        if not final_adapter:
            # Fallback: derive from results_dir + global_step.
            final_step = g.get("global_step") or g.get("steps")
            if final_step:
                final_adapter = f"{res['results_dir']}/adapters/step_{final_step}"
        _gate36_role = ("informational" if dual_gate_flash_primary
                        else "binding (AND)")
        print(f"\n[dual-gate] §3.6 gate ({_gate36_role}): "
              f"{'PASS' if gate_36_pass else 'FAIL'} "
              f"(ci_low={ef.get('ci_low') if ef else None}, "
              f"threshold=+{ef.get('threshold') if ef else 2.0})")
        # Run-Flash decision:
        #  * flash-primary (default): Flash is the decision-maker, so ALWAYS
        #    run it (only skip when there is no adapter). The §3.6 heuristic
        #    gate is the Goodhart-fooled one and must NOT veto a policy that
        #    transfers to Flash.
        #  * legacy AND: §3.6 short-circuits the panel_eval container
        #    (~$5-10 GPU + ~$5-10 API) when it already fails the AND.
        if not final_adapter:
            print("[dual-gate] FAIL — no final adapter path; skipping "
                  "panel_eval. OVERALL: NO-SPEND ✗")
        elif (not dual_gate_flash_primary) and (not gate_36_pass):
            print("[dual-gate] §3.6 FAIL (binding AND mode) → short-circuiting "
                  "(Flash heldout cannot recover an AND-relation gate). "
                  "OVERALL: NO-SPEND ✗")
        else:
            print(f"[dual-gate] running panel_eval vs Flash on "
                  f"{final_adapter} ...")
            try:
                pres = panel_eval.remote(
                    adapter_path=final_adapter,
                    base_model=model,
                    panels="vs_flash",
                    num_seeds=dual_gate_flash_seeds,
                    seed_start=dual_gate_flash_seed_start,
                    max_parallel=max_parallel * 2,
                    run_tag=f"{run_tag}_dualgate_flash",
                )
            except Exception as e:  # noqa: BLE001
                pres = {"panel_eval_rc": -1, "aborted": (
                    f"panel_eval call raised: {type(e).__name__}: {e}")}
            panel_rc = pres.get("panel_eval_rc")
            if pres.get("aborted"):
                print(f"[dual-gate] panel_eval ABORTED: {pres['aborted']}")
                print("[dual-gate] OVERALL: NO-SPEND ✗ "
                      "(Flash heldout unavailable)")
            elif panel_rc != 0:
                # Nonzero rc with `win_rates` populated would be stale/partial —
                # an unfinished bash run or a vLLM crash mid-eval. NEVER accept.
                print(f"[dual-gate] panel_eval rc={panel_rc} (nonzero) — "
                      f"results stale/partial, refusing to gate on them.")
                print("[dual-gate] OVERALL: NO-SPEND ✗ "
                      "(Flash heldout rc!=0)")
            else:
                assessment = assess_flash_gate(
                    pres,
                    gate_36_pass=gate_36_pass,
                    flash_primary=dual_gate_flash_primary,
                    flash_seeds=dual_gate_flash_seeds,
                    flash_threshold=dual_gate_flash_threshold,
                    sft_baseline_wr=dual_gate_sft_baseline_wr,
                    significance_alpha=dual_gate_significance_alpha,
                )
                for line in assessment.lines:
                    print(line)


@app.local_entrypoint()
def phase3_eval_only_main(
    adapter_path: str,
    step0_adapter_path: str = "",
    model: str = "djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2",
    served_model_name: str = "qwen/qwen3-4b-instruct",
    eval_seeds: int = 60,
    seed_start: int = 20000,
    eval_samples_per_seed: int = 1,
    eval_temperature: float | None = None,
    max_parallel: int = 32,
    vllm_gpu_mem_util: str = "0.3",
    vllm_ready_timeout_s: int = 1200,
    eval_opponent: str = "heuristic",
    threshold: str = "",
    run_tag: str = "eval_only",
):
    """Eval-only: re-evaluate an existing LoRA adapter at a chosen T / K.

    Usage:
      # Lever C (greedy corroboration):
      modal run modal_train.py::phase3_eval_only_main \\
        --adapter-path /results/<run>/adapters/step_200 \\
        --step0-adapter-path /results/<run>/adapters/step_0 \\
        --eval-temperature 0.0 --run-tag greedy_corroboration

      # Lever A (K=4 averaging, stochastic):
      modal run modal_train.py::phase3_eval_only_main \\
        --adapter-path /results/<run>/adapters/step_200 \\
        --eval-samples-per-seed 4 --run-tag stochastic_k4

      # Frontier transfer (non-heuristic opponent ⇒ explicit --threshold):
      modal run modal_train.py::phase3_eval_only_main \\
        --adapter-path /results/<run>/adapters/step_150 \\
        --eval-opponent google/gemini-3-flash-preview --threshold 0.0 \\
        --eval-seeds 100 --eval-samples-per-seed 8 --run-tag frontier_gate_01
    """
    import json

    res = phase3_eval_only.remote(
        adapter_path=adapter_path,
        step0_adapter_path=step0_adapter_path,
        model=model, served_model_name=served_model_name,
        eval_seeds=eval_seeds, seed_start=seed_start,
        eval_samples_per_seed=eval_samples_per_seed,
        eval_temperature=eval_temperature,
        max_parallel=max_parallel,
        vllm_gpu_mem_util=vllm_gpu_mem_util,
        vllm_ready_timeout_s=vllm_ready_timeout_s,
        eval_opponent=eval_opponent,
        threshold=threshold,
        run_tag=run_tag,
    )
    if res.get("aborted"):
        print(f"\nABORTED: {res['aborted']}")
        return

    local = pathlib.Path(f"results/phase3_eval_only_{run_tag}_modal")
    local.mkdir(parents=True, exist_ok=True)
    for name, blob in res.get("files", {}).items():
        (local / name).write_text(json.dumps(blob, indent=2, default=str))

    ef = res.get("files", {}).get("eval_final.json", {})
    es0 = res.get("files", {}).get("eval_step0.json", {})
    print(f"\nphase3_eval_only rc={res['phase3_rc']}")
    print(f"  Volume={res['results_dir']}  + local copy={local}")
    print(f"  T={res['eval_temperature']}  K={res['eval_samples_per_seed']}")
    if es0:
        print(f"  step0 (informational): mean_delta={es0.get('mean_delta')} "
              f"ci=[{es0.get('ci_low')}, {es0.get('ci_high')}]  n={es0.get('n')}")
    if ef:
        print(f"  §3.6 eval(final): status={ef.get('status')} "
              f"gate_pass={ef.get('gate_pass')} "
              f"mean_delta={ef.get('mean_delta')} "
              f"ci=[{ef.get('ci_low')}, {ef.get('ci_high')}]  n={ef.get('n')}")
        print(f"     valid_as_final_evidence={ef.get('valid_as_final_evidence')}")
        for c in ef.get("final_gate_caveats", []):
            print(f"     ⚠ {c}")
    print(f"  raw: modal volume get megagem-results "
          f"$(basename {res['results_dir']})/ ./")


