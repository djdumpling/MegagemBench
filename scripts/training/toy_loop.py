#!/usr/bin/env python3
"""Phase 2.2 — full custom GRPO loop on the toy (the RL plan §2, line 74).

Gate (line 74): no NaNs; per-token KL bounded; advantage signs sensible;
gradients masked correctly; **policy improves**.
Phase 2 gate (line 77/121): paired-bootstrap CI of
``delta = reward(step50) - reward(step0)`` on the toy clears 0, where
``reward`` is the Phase-2.1-hardened *robust* signal — the trainable seat's
engine-true terminal reward (NOT advantage sign, demoted to a canary in 2.1).

  # GPU box (per gpu_setup_guide: prime-rl venv, never `uv run`):
  prime-rl/.venv/bin/python scripts/training/toy_loop.py \
      --model <sft-merged-ckpt> --steps 50 --k 4 --num-seeds 2 --eval-seeds 20 \
      --rollout-vllm-url http://localhost:8000/v1 \
      --served-model-name qwen/qwen3-4b-instruct \
      --output results/phase2/toy_loop.json

  # CPU box now (no GPU/torch/trl/network) — proves the whole orchestration:
  uv run --no-sync python scripts/training/toy_loop.py --dry-run \
      --output results/phase2/toy_loop_dryrun.json

SCOPE (Caveat 5): same-population heuristic opponents. This validates trainer
mechanics + that the per-turn signal *can* move a policy on the toy. It does
NOT test the self-play→frozen-anchor transfer (first at 3.6). A green run
must not be over-read.

SCOPE (rollout): with --rollout-vllm-url the rollout policy is an EXTERNAL
OpenAI-compatible server, deliberately NOT TRL-internal `use_vllm` synced
generation (see megagem.training.grpo_harness). A live 2.2 PASS therefore proves trainer
mechanics + weighted updates from those external rollouts — it is NOT a proof
of fully on-policy vLLM weight-resynced refresh (that is Phase 3's path).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path

import megagem.training.grpo_harness as H


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                   help="HF id / local path of the RL policy (the trainer model).")
    p.add_argument("--steps", type=int, default=50, help="GRPO update steps.")
    p.add_argument("--k", type=int, default=4,
                   help="K-group size (rollouts/seed). Per-step optimizer "
                        "batch = num_seeds·k·8·grpo_num_generations — keep "
                        "modest; it is GPU-memory bound.")
    p.add_argument("--num-seeds", type=int, default=2,
                   help="distinct training seeds. P1.3: this is GPU-memory "
                        "bound (per-step optimizer batch ∝ num_seeds·k·8·G); "
                        "raise it for a real learnability run only with the "
                        "memory headroom — it is NOT a free power lever, "
                        "unlike --eval-seeds.")
    p.add_argument("--grpo-num-generations", type=int, default=2,
                   help="TRL num_generations (forbidden < 2; per-group std is "
                        "discarded by the subclass — pure ×N overhead).")
    p.add_argument("--seed-start", type=int, default=5000)
    p.add_argument("--eval-seeds", type=int, default=40,
                   help="frozen eval-set size for the paired-bootstrap gate. "
                        "P1.3: this is the FREE power lever (eval-side only, "
                        "no training-memory cost) — CI half-width ∝ 1/√n; the "
                        "old default 20 gave ≈0.10, below the SFT effect. "
                        "Raised to 40; raise further for a learnability run.")
    p.add_argument("--eval-seed-start", type=int, default=9000,
                   help="eval seeds are disjoint from the training seeds.")
    p.add_argument("--kl-max", type=float, default=0.5,
                   help="per-token KL bound; any logged step above → FAIL.")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.05)
    p.add_argument("--rollout-vllm-url", default=None,
                   help="external OpenAI-compatible vLLM server for the "
                        "rollout policy. This is OUR policy backend, NOT "
                        "TRL-internal vLLM (TRL use_vllm stays False). Omit "
                        "→ rollout uses the deterministic CPU stub.")
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--on-policy", action="store_true",
                   help="P0.1: roll out from the trainer's OWN updating "
                        "weights (HF generate, sampling) instead of a frozen "
                        "external vLLM server or the stub — the only genuinely "
                        "on-policy mode. Ignores --rollout-vllm-url. Required "
                        "for a real learnability claim.")
    p.add_argument("--gate-mode", choices=["auto", "plumbing", "learnability"],
                   default="auto",
                   help="P1.1: which verdict is authoritative. 'plumbing' = "
                        "non-inferiority (CI not entirely <0; the v5.6 smoke). "
                        "'learnability' = strict improvement (ci_low>0). "
                        "'auto' = learnability iff --on-policy else plumbing. "
                        "BOTH verdicts are always emitted; this only picks the "
                        "pass/fail criterion (no bar is lowered silently).")
    p.add_argument("--accept-stale-onpolicy", action="store_true",
                   help="P1.1 freshness precondition: a learnability gate is "
                        "only valid if rollouts are actually fresh. With "
                        "steps_per_generation>1 (the default cadence at "
                        "PHASE2_MICRO_CAP=8 ⇒ spg=16) ~50 optimizer steps "
                        "replay a handful of buffered on-policy batches — NOT "
                        "50 fresh on-policy updates. Without this flag, "
                        "learnability + spg>1 FAILs as a precondition error "
                        "(size B·G so cap≥B·G for spg=1, or pass this to "
                        "record-and-proceed with a loud caveat).")
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="CPU harness self-test (no torch/trl/GPU/network).")
    p.add_argument("--bootstrap-resamples", type=int, default=10_000)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    # Enforce the same invariant for dry-run AND live: TRL forbids
    # num_generations < 2 (grpo_config.py:948); validate it at the CLI so
    # --dry-run can't pass a value the live trainer would reject.
    if args.grpo_num_generations < 2:
        p.error("--grpo-num-generations must be >= 2 (TRL forbids < 2; the "
                "per-group std is discarded by the subclass anyway).")
    return args


def _eval_reward_per_seed(policy_fn, eval_seeds: list[int]) -> dict[int, float]:
    """One greedy rollout per eval seed → trainable-seat terminal reward.
    Greedy so the step-0/step-N comparison is an honest before/after of the
    same weights (no sampling noise in the gate measurement)."""
    out: dict[int, float] = {}
    for s in eval_seeds:
        game = H.rollout_group(s, trainable_policy_fn=policy_fn, k=1)[0]
        out[s] = H.trainable_terminal_reward(game)
    return out


def _eval_with_behavior(policy_fn, eval_seeds: list[int]):
    """Like ``_eval_reward_per_seed`` but also returns the rolled games, so the
    step0/stepN behaviour telemetry (realized shade / parse-legal-default) is
    computed from the SAME greedy rollout — no extra (expensive) generation.
    Returns ``(rewards: {seed: float}, games: [game, ...])``."""
    rewards: dict[int, float] = {}
    games = []
    for s in eval_seeds:
        game = H.rollout_group(s, trainable_policy_fn=policy_fn, k=1)[0]
        rewards[s] = H.trainable_terminal_reward(game)
        games.append(game)
    return rewards, games


def _dry_run(args: argparse.Namespace) -> dict:
    """CPU self-test of the entire orchestration — explicitly NOT a training
    result (mirrors update_cost_smoke.py --dry-run / the 2.1 honesty discipline).

    Proves on CPU: (1) fed the exact TRL RepeatSampler generation batch
    (turn-as-prompt dataset, each (seed,k,round) prompt ×G), the rollout_func
    returns exactly len(prompts) rows through the unmodified pipeline, the ×G
    duplicates carry identical advantages, and those advantages equal a single
    true-K-group flatten over all games; (2) the gate metric + paired
    bootstrap separate a known-good from a known-bad policy (the gate cannot
    false-positive on noise / cannot miss a real gap).
    """
    eval_seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.eval_seeds))

    class _FakeTok:
        eos_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [min(255, ord(c)) for c in text[:32]] or [0]}

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return " ".join(m["content"] for m in msgs)

    # (1) rollout_func wiring against the verified TRL path, fed the EXACT
    # RepeatSampler generation batch (utils.py:738; our config → each unique
    # turn-prompt repeated G times consecutively, whole dataset in one chunk).
    G = args.grpo_num_generations
    rf = H.make_toy_rollout_func(_FakeTok(), trainable_policy_fn=lambda _p: "",
                                 expected_k=args.k, stub_strategy="bestresp")
    train_seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    base = H.toy_turn_dataset_prompts(train_seeds, args.k)
    gen_batch = [p for p in base for _ in range(G)]  # RepeatSampler order
    out = rf(prompts=gen_batch)
    n_rows = len(out["precomputed_advantage"])
    expected = len(gen_batch)  # exactly len(prompts) (verified TRL contract)
    keys_ok = {"prompt_ids", "completion_ids", "logprobs",
               "precomputed_reward", "precomputed_advantage"} <= set(out)
    # The verified TRL extra-merge / reward-sizing contract (grpo_trainer.py
    # L2130 / L1198) is on these positionally-aligned fields ONLY; logprobs is
    # NOT in that merge set (cf. test_trl_merge_alignment_simulated, which
    # already excludes it) and post-P0.2 is field-level None by default.
    _aligned = ("prompt_ids", "completion_ids",
                "precomputed_reward", "precomputed_advantage")
    lens_ok = len({len(out[k]) for k in _aligned}) == 1
    _lp = out["logprobs"]
    logprobs_contract_ok = _lp is None or (
        len(_lp) == expected
        and all(len(c) == len(l) for c, l in zip(out["completion_ids"], _lp))
    )
    # P0.2: default mode emits NO fabricated logprobs (field-level None).
    logprobs_not_fabricated = _lp is None
    finite_ok = all(math.isfinite(x) for x in out["precomputed_advantage"]) and \
        all(math.isfinite(x) for x in out["precomputed_reward"])
    # G duplicates of a turn-prompt must carry identical advantages.
    adv_by_prompt: dict[str, set] = {}
    for p, a in zip(gen_batch, out["precomputed_advantage"]):
        adv_by_prompt.setdefault(p, set()).add(round(a, 12))
    dupes_identical = all(len(s) == 1 for s in adv_by_prompt.values())
    # True K-group: the rollout's advantages must equal a SINGLE
    # build_training_rows over ALL games together (the regression the
    # reviewer caught: per-game flatten makes each game its own group-of-1).
    # Reference mirrors the rollout's first-seen (seed,k) order exactly.
    all_games = [
        H.rollout_group(s, trainable_policy_fn=H.make_stub_policy_fn(
            "bestresp", ki), k=1)[0]
        for s in train_seeds for ki in range(args.k)
    ]
    ref = {(r["game_id"], r["round_index"]): r["precomputed_advantage"]
           for r in H.flatten_training_rows(all_games)}
    kgroup_ok = True
    for p, a in zip(gen_batch, out["precomputed_advantage"]):
        s, ki, rnd, _ = H.parse_turn_prompt(p)
        gid = (train_seeds.index(s) * args.k) + ki  # rollout's game_id
        if abs(a - ref[(gid, rnd)]) > 1e-9:
            kgroup_ok = False
            break

    # (2) gate metric: known-good (bestresp ≈ 0.70·v — the corrected, genuinely
    # profitable stub; P0.3) vs known-bad (overbid). Post-P0.3 "known-good" is
    # actually good (positive mean reward), not merely less-bad — checked below.
    good = _eval_reward_per_seed(H.make_stub_policy_fn("bestresp"), eval_seeds)
    bad = _eval_reward_per_seed(H.make_stub_policy_fn("overbid"), eval_seeds)
    good_mean = sum(good.values()) / len(good)
    # CPU sanity-check of the Tier-A shade telemetry: the corrected known-good
    # (bestresp = 0.70·v) must read back as a shade ≈0.70 on the plateau.
    known_good_behavior = H.aggregate_behavior([
        H.rollout_group(s, trainable_policy_fn=H.make_stub_policy_fn("bestresp"),
                        k=1)[0]
        for s in eval_seeds[:min(5, len(eval_seeds))]
    ])
    deltas = [good[s] - bad[s] for s in eval_seeds]
    boot = H.paired_bootstrap_ci(deltas, n_resamples=args.bootstrap_resamples,
                                 seed=args.seed)
    # Negative control: a policy vs itself must NOT clear 0 (no false positive).
    null = H.paired_bootstrap_ci([good[s] - good[s] for s in eval_seeds],
                                 n_resamples=args.bootstrap_resamples, seed=args.seed)

    checks = {
        "rollout_keys_present": keys_ok,
        "rollout_equal_length": lens_ok,
        "rollout_row_count": n_rows,
        "rollout_expected_row_count": expected,
        "rollout_one_to_one_with_prompts": n_rows == expected,
        "rollout_finite_contract_ok": finite_ok,
        "rollout_logprobs_contract_ok": logprobs_contract_ok,
        "rollout_logprobs_not_fabricated": logprobs_not_fabricated,
        "repeatsampler_dupes_identical": dupes_identical,
        "true_kgroup_matches_single_flatten": kgroup_ok,
        "gate_separates_good_from_bad": boot["ci_low"] > 0.0,
        "gate_null_control_straddles_0": null["ci_low"] <= 0.0 <= null["ci_high"],
        # P0.3 acceptance: the corrected known-good is a *positive*-reward
        # policy, not just better than overbid (the old 0.85·v "bestresp" was
        # net-losing, which made "known-good" a misnomer).
        "gate_known_good_is_positive": good_mean > 0.0,
    }
    passed = all(v for k, v in checks.items() if isinstance(v, bool))
    return {
        "step": "2.2",
        "mode": "DRY-RUN (CPU harness self-test — NOT a training result)",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "known_good_mean_reward": good_mean,
        "known_good_behavior": known_good_behavior,  # Tier-A shade self-check
        "known_good_vs_bad_bootstrap": boot,
        "null_control_bootstrap": null,
        # P1.2 — same explicit learning target as the GPU path (CPU-cheap).
        "oracle_shade_sweep": H.oracle_shade_sweep(eval_seeds),
    }


def _gpu_run(args: argparse.Namespace, tmp: str) -> dict:
    H.print_trl_env()
    eval_seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.eval_seeds))
    train_seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    # Rollout-policy precedence (P0.1). Only --on-policy is genuinely
    # on-policy (rollouts reflect the trainer's *updating* weights). The
    # external vLLM server is frozen (never resynced) and the stub is fixed —
    # both are OFF-policy plumbing modes and must NOT back a learnability
    # claim (phase-2 plan P0.1).
    if args.on_policy:
        rollout_mode = "on_policy"
        rollout_policy, stub_strategy, on_policy = None, None, True
    elif args.rollout_vllm_url:
        rollout_mode = "vllm_frozen"
        rollout_policy = H.make_vllm_policy_fn(
            args.rollout_vllm_url, args.served_model_name or args.model
        )
        stub_strategy, on_policy = None, False
    else:
        rollout_mode = "stub"
        rollout_policy, stub_strategy, on_policy = None, "bestresp", False

    t0 = time.perf_counter()
    trainer, tokenizer, info = H.build_toy_trainer(
        model_id=args.model,
        seeds=train_seeds,
        k=args.k,
        trainable_policy_fn=rollout_policy,
        grpo_num_generations=args.grpo_num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        max_steps=args.steps,
        seed=args.seed,
        output_dir=tmp,
        stub_strategy=stub_strategy,
        on_policy=on_policy,
    )
    build_s = round(time.perf_counter() - t0, 2)

    # Gate eval uses the *current trainer weights* greedily, in eval() mode
    # (no LoRA-dropout noise — see make_hf_generate_policy_fn).
    eval_policy = H.make_hf_generate_policy_fn(trainer.model, tokenizer, do_sample=False)

    # Tier-A: eval-determinism check — directly verifies the model.eval() fix
    # (#2). Two greedy passes on a small subset at step 0 MUST be byte-
    # identical; any delta ⇒ LoRA dropout still leaking into the "no sampling
    # noise" gate. Cheap (≤5 extra games), high signal.
    det_seeds = eval_seeds[:min(5, len(eval_seeds))]
    det_a = _eval_reward_per_seed(eval_policy, det_seeds)
    det_b = _eval_reward_per_seed(eval_policy, det_seeds)
    eval_determinism_delta = max(
        (abs(det_a[s] - det_b[s]) for s in det_seeds), default=0.0)

    r0, games0 = _eval_with_behavior(eval_policy, eval_seeds)

    t1 = time.perf_counter()
    trainer.train()
    train_s = round(time.perf_counter() - t1, 2)

    rN, gamesN = _eval_with_behavior(eval_policy, eval_seeds)
    deltas = [rN[s] - r0[s] for s in eval_seeds]
    boot = H.paired_bootstrap_ci(deltas, n_resamples=args.bootstrap_resamples,
                                 seed=args.seed)
    # Tier-A: realized greedy behaviour before/after — turns "reward moved?"
    # into "did the policy move, and WHICH way (toward the oracle plateau? off
    # the JSON contract?)". Same rolled games as r0/rN (no extra generation).
    beh0 = H.aggregate_behavior(games0)
    behN = H.aggregate_behavior(gamesN)

    # Per-step health from the durable log history (TRL clears _metrics on
    # each log(); state.log_history is the persistent record).
    log_hist = list(getattr(trainer.state, "log_history", []))

    def _series(key):
        return [e[key] for e in log_hist if key in e]

    losses = _series("loss")
    kls = _series("kl")
    grad_norms = _series("grad_norm")
    adv_mean = _series("megagem/advantages/mean")
    adv_std = _series("megagem/advantages/std")
    adv_count = _series("megagem/advantages/count")
    # Tier-B: generic completion-length series, if the pinned TRL logs one
    # (key name varies across versions — match defensively, never hard-depend).
    _len_keys = [k for e in log_hist for k in e
                 if ("completion" in k and ("len" in k or "length" in k))]
    comp_len = _series(_len_keys[0]) if _len_keys else []

    nan_free = all(math.isfinite(x) for x in losses + adv_mean + adv_std)
    kl_max_obs = max(kls) if kls else 0.0
    kl_bounded = (not kls) or (math.isfinite(kl_max_obs) and kl_max_obs <= args.kl_max)
    adv_sane = bool(adv_count) and all(c > 0 for c in adv_count) and \
        all(math.isfinite(m) for m in adv_mean)
    # Grad mask: flatten_training_rows already raises if any non-trainable row
    # leaks; here we confirm advantages actually reached the loss at K>1
    # (count>0 every logged step), i.e. the I5 seam fired.
    grad_mask_ok = adv_sane

    # P2.2 — the REAL advantage-signal diagnostic. The two booleans above are
    # a finite+count plumbing contract ("advantages reached the loss"), NOT a
    # sign-correctness test. Here: one fresh SAMPLED rollout over the train
    # seeds with the post-train policy, Spearman(terminal_reward, mean_adv)
    # within each K-group. Reported, NOT gated (a single noisy rollout must not
    # fail a run); ρ≤0/None is the "advantages point the wrong way" signal the
    # booleans never tested. A sampling policy is required (greedy ⇒ degenerate
    # K-group ⇒ ρ undefined).
    try:
        diag_policy = H.make_hf_generate_policy_fn(
            trainer.model, tokenizer, do_sample=True
        )
        rank_corr = H.advantage_reward_rank_corr(
            train_seeds, k=args.k, policy_fn=diag_policy
        )
    except Exception as e:  # noqa: BLE001 — diagnostic must never fail the run
        rank_corr = {"error": f"{type(e).__name__}: {e}"}

    health = {
        "nan_free": nan_free,
        "kl_bounded": kl_bounded,
        "kl_max_observed": kl_max_obs,
        "kl_max_allowed": args.kl_max,
        # NB: a finite+count *plumbing* contract (advantages reached the loss),
        # NOT sign-correctness — that is advantage_reward_rank_corr below.
        "advantage_signs_sensible": adv_sane,
        "gradients_masked_correctly": grad_mask_ok,
        "advantage_reward_rank_corr": rank_corr,  # P2.2 diagnostic, not gated
        "logged_steps": len(losses),
        # Tier-B trajectories — turn raw log_history into the trends a
        # reviewer reads. The v5.6 smoking gun was KL≈0.006≪0.5 ("policy
        # barely moved"); the series makes "no improvement BECAUSE it didn't
        # move" distinguishable from "moved but didn't help". grad_norm≈0 ⇒
        # nothing learning (masking/detached graph); advantage_std→0 ⇒ GRPO
        # signal collapsed (the confident-policy degeneracy).
        "kl_trajectory": H.summarize_series(kls),
        "grad_norm_trajectory": H.summarize_series(grad_norms),
        "loss_trajectory": H.summarize_series(losses),
        "advantage_std_trajectory": H.summarize_series(adv_std),
        "advantage_mean_trajectory": H.summarize_series(adv_mean),
        "completion_length_trajectory": H.summarize_series(comp_len),
        # Tier-A: directly verifies the model.eval() dropout fix (#2).
        "eval_determinism": {
            "max_abs_reward_delta": eval_determinism_delta,
            "subset_n": len(det_seeds),
            "deterministic": eval_determinism_delta == 0.0,
        },
    }

    # P1.3 — make underpowering VISIBLE instead of relaxing the bar to hide it.
    ci_hw = (boot["ci_high"] - boot["ci_low"]) / 2.0
    power = {
        "eval_n": len(eval_seeds),
        "ci_half_width": ci_hw,
        "train_seeds": len(train_seeds),
        "note": "CI half-width ∝ 1/√eval_n. --eval-seeds is the FREE lever "
                "(eval-side only); --num-seeds is GPU-mem bound. A small true "
                "effect below ci_half_width is undetectable regardless of "
                "training — that is a power problem, not a gate threshold.",
    }

    # P1.2 — the explicit learning target (computed once, reused below).
    oracle = H.oracle_shade_sweep(eval_seeds)
    _s0, _sN = beh0.get("shade_mean"), behN.get("shade_mean")
    _tgt = oracle["argmax_shade"]
    # Did greedy behaviour move TOWARD the oracle plateau? (|sN-tgt| < |s0-tgt|)
    shade_moved_toward_target = (
        None if _s0 is None or _sN is None
        else abs(_sN - _tgt) < abs(_s0 - _tgt)
    )
    # Compact realized-shade trajectory from the on-policy rollouts (info has
    # the full per-call series; this is the at-a-glance trend).
    _rser = (info.get("rollout_stats", {}) or {}).get("series", [])
    _rshade = [r.get("shade_mean") for r in _rser if r.get("shade_mean") is not None]

    gate = {
        "rollout_mode": rollout_mode,
        "on_policy": on_policy,
        "reward_step0_mean": sum(r0.values()) / len(r0),
        "reward_stepN_mean": sum(rN.values()) / len(rN),
        # Tier-A: realized greedy behaviour before/after + the explicit target.
        # reward alone can't tell (a) policy never moved, (b) moved wrong way,
        # (c) moved right but reward noisy — shade vs target can.
        "eval_behavior_step0": beh0,
        "eval_behavior_stepN": behN,
        "shade_target_argmax": _tgt,
        "shade_step0": _s0,
        "shade_stepN": _sN,
        "shade_moved_toward_target": shade_moved_toward_target,
        "rollout_shade_trajectory": {
            "first": (_rshade[0] if _rshade else None),
            "last": (_rshade[-1] if _rshade else None),
            "n_calls": (info.get("rollout_stats", {}) or {}).get("calls", 0),
            "series": _rshade,
        },
        # Per-seed deltas: uniform negative shift ⇒ directional objective
        # problem (escalate); a few games collapsing while the rest ≈0 ⇒
        # brittle-toy + off-policy staleness (the cadence hypothesis). Lets the
        # spg sweep be read without re-deriving the bootstrap input.
        "per_seed": [
            {"seed": s, "r0": r0[s], "rN": rN[s], "delta": rN[s] - r0[s]}
            for s in eval_seeds
        ],
        "paired_bootstrap": boot,
        "power": power,
        # P1.1 — BOTH verdicts ALWAYS emitted, honestly labelled. No bar is
        # moved; --gate-mode only selects which is authoritative for pass/fail
        # (it does not silently relax anything — that is the v5.6 mistake this
        # remediation reverses).
        #   strict learnability: 95% paired-bootstrap CI clears 0 (ci_low>0).
        "policy_improves": boot["ci_low"] > 0.0,
        #   non-inferiority smoke: CI not entirely <0 (ci_high>0) — only the
        #   off-policy-stale collapse (CI fully <0) fails. Plumbing, NOT a
        #   learnability claim.
        "policy_not_regressed": boot["ci_high"] > 0.0,
    }

    # P1.1 — resolve which gate is authoritative. 'auto' = learnability iff
    # genuinely on-policy, else the plumbing smoke (preserves the existing
    # off-policy default behaviour — back-compatible — without pretending an
    # off-policy run proved learnability).
    gate_mode = args.gate_mode
    if gate_mode == "auto":
        gate_mode = "learnability" if on_policy else "plumbing"
    gate["evaluation_mode"] = gate_mode

    # P1.1 freshness — ALWAYS recorded so a run is interpretable from JSON
    # alone (the v5.6 sweep could not be). steps_per_generation>1 ⇒ ~50
    # optimizer steps replay a handful of buffered rollout batches; that is
    # NOT "50 fresh on-policy updates" even with --on-policy.
    spg = int(info.get("steps_per_generation", 1) or 1)
    fresh_batches = (-(-args.steps // spg)) if spg > 0 else args.steps  # ceil
    gate["rollout_freshness"] = {
        "steps_per_generation": spg,
        "approx_fresh_rollout_batches": fresh_batches,
        "optimizer_steps": args.steps,
        "fully_fresh": bool(on_policy and spg == 1),
        "accept_stale_onpolicy": bool(args.accept_stale_onpolicy),
    }

    if gate_mode == "learnability":
        gate["pass_criterion"] = "policy_improves (strict: paired CI ci_low > 0)"
        gate_pass = gate["policy_improves"]
        # Precondition: a strict learnability claim is only valid on genuinely
        # fresh on-policy rollouts. Two ways it is NOT (each is a hard
        # precondition failure, not a silent pass — that was the v5.6 mistake):
        precond_fail = None
        if not on_policy:
            precond_fail = (
                f"OFF-policy rollouts (rollout_mode={rollout_mode}): ci_low>0 "
                f"here is not a learnability proof — use --on-policy."
            )
        elif spg > 1 and not args.accept_stale_onpolicy:
            precond_fail = (
                f"stale on-policy cadence: steps_per_generation={spg} ⇒ only "
                f"~{fresh_batches} fresh rollout batch(es) in {args.steps} "
                f"steps (the rest replay a buffer). Set PHASE2_MICRO_CAP ≥ "
                f"B·G (={info.get('generation_batch_size')}) for spg=1 (size "
                f"--num-seeds/-k down to fit memory), or pass "
                f"--accept-stale-onpolicy to record-and-proceed."
            )
        if precond_fail:
            gate["learnability_precondition_ok"] = False
            gate["WARNING"] = (
                f"learnability precondition FAILED — {precond_fail} "
                f"(Phase-2 plan P0.1/P1.1.)"
            )
            gate_pass = False  # cannot legitimately claim learnability
        else:
            gate["learnability_precondition_ok"] = True
            if on_policy and spg > 1 and args.accept_stale_onpolicy:
                gate["CAVEAT"] = (
                    f"--accept-stale-onpolicy: steps_per_generation={spg} "
                    f"(~{fresh_batches} fresh batches / {args.steps} steps) — "
                    f"a WEAKENED on-policy learnability claim, recorded not "
                    f"hidden."
                )
    else:
        gate["pass_criterion"] = (
            "policy_not_regressed (non-inferiority smoke: ci_high > 0)"
        )
        gate_pass = gate["policy_not_regressed"]

    health_ok = all(health[k] for k in
                    ("nan_free", "kl_bounded", "advantage_signs_sensible",
                     "gradients_masked_correctly"))
    passed = health_ok and gate_pass
    return {
        "step": "2.2",
        "mode": "GPU",
        "status": "PASS" if passed else "FAIL",
        "info": info,
        "build_s": build_s,
        "train_s": train_s,
        "health": health,
        "gate": gate,
        # P1.2 — the learning TARGET, explicit (computed once above, reused).
        "oracle_shade_sweep": oracle,
        "last_log": log_hist[-1] if log_hist else None,
    }


def main() -> int:
    args = parse_args()
    try:
        if args.dry_run:
            result = _dry_run(args)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                result = _gpu_run(args, tmp)
    except Exception as e:  # noqa: BLE001 — the integration IS under test
        result = {"step": "2.2", "status": "FAIL",
                  "reason": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))

    print()
    print(f"[{result.get('mode', '?')}] Phase 2.2 — {result['status']}")
    if "gate" in result:
        g = result["gate"]
        boot = g["paired_bootstrap"]
        print(f"  rollout_mode={g.get('rollout_mode')} "
              f"on_policy={g.get('on_policy')} | gate={g.get('evaluation_mode')}"
              f" → {g.get('pass_criterion')}")
        print(f"  reward step0→stepN: {g['reward_step0_mean']:.4f} → "
              f"{g['reward_stepN_mean']:.4f}; paired CI "
              f"[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}] "
              f"(±{g['power']['ci_half_width']:.4f}, n={g['power']['eval_n']})")
        print(f"  verdicts: policy_improves(strict)="
              f"{g.get('policy_improves')}  "
              f"policy_not_regressed(smoke)={g.get('policy_not_regressed')}")
        _fmt = lambda x: ("None" if x is None else f"{x:.3f}")  # noqa: E731
        print(f"  shade: step0={_fmt(g.get('shade_step0'))} → "
              f"stepN={_fmt(g.get('shade_stepN'))} "
              f"(target≈{g.get('shade_target_argmax')}) "
              f"moved_toward_target={g.get('shade_moved_toward_target')}")
        h = result.get("health", {})
        klt = h.get("kl_trajectory", {})
        gnt = h.get("grad_norm_trajectory", {})
        ast = h.get("advantage_std_trajectory", {})
        print(f"  KL {_fmt(klt.get('first'))}→{_fmt(klt.get('last'))} "
              f"(max {_fmt(klt.get('max'))}) | grad_norm "
              f"{_fmt(gnt.get('first'))}→{_fmt(gnt.get('last'))} | adv_std "
              f"{_fmt(ast.get('first'))}→{_fmt(ast.get('last'))}")
        ed = h.get("eval_determinism", {})
        rc = (h.get("advantage_reward_rank_corr") or {})
        print(f"  eval_determinism Δ={_fmt(ed.get('max_abs_reward_delta'))} "
              f"(det={ed.get('deterministic')}) | rank_corr ρ="
              f"{_fmt(rc.get('mean_spearman'))} "
              f"kgroupσ_med={_fmt((rc.get('kgroup_termreward_sigma') or {}).get('median'))}")
        rf = g.get("rollout_freshness", {})
        if rf:
            print(f"  freshness: spg={rf.get('steps_per_generation')} "
                  f"≈{rf.get('approx_fresh_rollout_batches')} fresh batches/"
                  f"{rf.get('optimizer_steps')} steps "
                  f"fully_fresh={rf.get('fully_fresh')}")
        if result.get("oracle_shade_sweep"):
            o = result["oracle_shade_sweep"]
            print(f"  oracle target: shade {o['argmax_shade']} → "
                  f"{o['argmax_mean_reward']:+.4f} (opponents {o['opponent_shade']})")
        if g.get("CAVEAT"):
            print(f"  ⚠ CAVEAT: {g['CAVEAT']}")
        if g.get("WARNING"):
            print(f"  ⚠ {g['WARNING']}")
    if result["status"] != "PASS":
        if "traceback" in result:
            print("\n--- traceback (pinpoints the TRL-pin seam delta) ---")
            print(result["traceback"])
        elif "checks" in result:
            for k, v in result["checks"].items():
                print(f"  {k}: {v}")
    print(f"→ {args.output}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
