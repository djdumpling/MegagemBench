#!/usr/bin/env python3
"""Phase 2.3 — 10–20 GRPO steps on real MegaGem at minimal batch
(the RL plan §2, line 75).

Gate (line 75, deliberately minimal): **trainer doesn't crash; KL stays
bounded**. NO policy-improvement claim — that is Phase 3. 2.3 only proves the
*same* ``MegaGemGRPOTrainer`` seam that 2.2 exercises on the toy also consumes
real-MegaGem-shaped rows (loans/investments/reveals/missions — the full
scorer-replay path, not the toy's pure-coins shortcut) without falling over.

DESIGN — pre-rolled fixed batch (not per-step on-policy)
--------------------------------------------------------
MegaGem games have a *variable* number of trainable turns, so the turn-as-
prompt dataset 2.2 uses (one row per (seed,k,round); B known up front) cannot
be enumerated ahead of the roll. The verified TRL pin requires
``len(rollout_out) == len(prompts)`` (see megagem.training.grpo_harness). So 2.3:

1. rolls a minimal batch of real games **once** (run_game) → post-tags the
   actor mask → the **unmodified** ``build_training_rows`` → a fixed list of
   turn rows;
2. makes those rows the dataset; the rollout_func re-emits them by index
   (static — exactly ``len(prompts)`` rows);
3. runs 10–20 optimizer steps over that fixed batch.

That is a faithful "trainer doesn't crash on real-MegaGem rows + KL bounded"
smoke. Per-step on-policy refresh (and the Phase-3 actor policy that replaces
the post-tag below) is out of 2.3 scope by construction.

  # GPU box (prime-rl venv; never `uv run` — see gpu_setup_guide):
  prime-rl/.venv/bin/python scripts/training/megagem_steps.py \
      --model <sft-merged-ckpt> --vllm-url http://localhost:8000/v1 \
      --served-model-name qwen/qwen3-4b-instruct \
      --steps 15 --num-seeds 2 --k 2 --output results/phase2/megagem_steps.json

  # CPU box now — validates the 2.3-specific glue (actor-mask post-tag incl.
  # nested reveals → unmodified build_training_rows) with no GPU/network:
  uv run --no-sync python scripts/training/megagem_steps.py --dry-run \
      --output results/phase2/megagem_steps_dryrun.json

The real-MegaGem reward/score path is already CPU-validated (the RL plan 1.5 ran
``megagem.rl.export`` over the real 48-game / 3038-row corpus). 2.3's only
residual is live ``run_game`` I/O + the trainer step on those rows.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path

import megagem.training.grpo_harness as H


def _post_tag_actor_mask(
    game: dict, *, trainable_seat: int, opponent_actor_id: str
) -> dict:
    """Stamp Caveat-4 actor_ids onto a real (self-play) MegaGem transcript so
    exactly one seat is trainable — including the **nested reveal sub-record**
    (run_game emits ``players[*].reveal.actor_id`` separately; tagging only
    the top-level bid record would leave reveal actor metadata semantically
    wrong even though mixed-actor player trajectories are dropped). Phase-3
    (3.2/3.3) replaces this with the real actor policy."""
    g = copy.deepcopy(game)
    for rnd in g.get("rounds", []):
        for rec in rnd.get("players", []):
            aid = (H.TRAINABLE_ACTOR_ID
                   if rec.get("player_id") == trainable_seat
                   else opponent_actor_id)
            rec["actor_id"] = aid
            if isinstance(rec.get("reveal"), dict):
                rec["reveal"]["actor_id"] = aid
    return g


def _pre_roll_rows(
    *, model_name, seeds, k, num_players, value_chart,
    trainable_seat, opponent_actor_id, tmp_dir,
) -> list[dict]:
    """Roll seeds×k real games once, post-tag, flatten via the unmodified
    pipeline (seat-parameterized — finding 6). Returns the fixed turn rows."""
    from megagem.rollout import run_game  # lazy: needs endpoints/network

    games = []
    for s in seeds:
        for ki in range(k):
            fname = f"mg_seed{s}_r{ki}.json"
            asyncio.run(run_game(
                models=[model_name] * num_players, value_chart=value_chart,
                seed=s, num_players=num_players, output_file="trajectory",
                json_filename=fname, quiet=True, results_dir=Path(tmp_dir),
            ))
            game = json.loads((Path(tmp_dir) / fname).read_text())
            games.append(_post_tag_actor_mask(
                game, trainable_seat=trainable_seat,
                opponent_actor_id=opponent_actor_id))
    return H.flatten_training_rows(games, trainable_seat=trainable_seat)


def make_static_rollout_func(tokenizer, rows, *, system_prompt=H.SYSTEM_PROMPT):
    """Re-emit the pre-rolled rows by index — exactly ``len(prompts)`` rows,
    row i = rows[i] (the verified TRL alignment contract). Dataset prompts are
    ``[mg-row] i=<n>`` markers, not shown to any model."""
    def _ids(text: str) -> list[int]:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids or [tokenizer.eos_token_id or 0]

    def rollout_func(*args, **kwargs):
        prompts = kwargs.get("prompts") or (args[0] if args else None)
        if prompts is None:
            raise RuntimeError("rollout_func could not find 'prompts'")
        idxs = [int(str(p).split("i=")[1]) for p in prompts]
        pid, cid, pr, pa = [], [], [], []
        for i in idxs:
            row = rows[i]
            msgs = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["prompt"]}]
            ptext = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            cids = _ids(row["completion"])
            pid.append(_ids(ptext)); cid.append(cids)
            pr.append(float(row["precomputed_reward"]))
            pa.append(float(row["precomputed_advantage"]))
        assert len(pid) == len(prompts)
        # P0.2: same non-fabricated logprobs policy as the toy rollout (one
        # shared helper — no toy/megagem divergence footgun).
        return {"prompt_ids": pid, "completion_ids": cid,
                "logprobs": H.rollout_logprobs(cid),
                "precomputed_reward": pr, "precomputed_advantage": pa}

    return rollout_func


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--served-model-name", default="qwen/qwen3-4b-instruct")
    p.add_argument("--vllm-url", default=None,
                   help="register served-model-name as a local vLLM endpoint "
                        "(run_game needs an endpoint; required unless --dry-run).")
    p.add_argument("--steps", type=int, default=15,
                   help="GRPO update steps (gate window is 10–20).")
    p.add_argument("--k", type=int, default=2, help="rollouts per seed (minimal).")
    p.add_argument("--num-seeds", type=int, default=2,
                   help="minimal batch — 2.3 is a robustness smoke.")
    p.add_argument("--seed-start", type=int, default=7000)
    p.add_argument("--num-players", type=int, default=3)
    p.add_argument("--value-chart", default="A")
    p.add_argument("--trainable-seat", type=int, default=0,
                   help="seat tagged 'trainable' (threaded into the seat-"
                        "parameterized flatten — finding 6).")
    p.add_argument("--opponent-actor-id", default="heuristic")
    p.add_argument("--num-generations", type=int, default=2,
                   help="TRL reshape divisor (discarded by the subclass); the "
                        "row batch is truncated to a multiple of this.")
    p.add_argument("--kl-max", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.05)
    p.add_argument("--max-completion-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def _dry_run(args) -> dict:
    """CPU validation of the 2.3-specific glue only: the actor-mask post-tag
    (incl. nested reveal) + the seat-parameterized unmodified export. Uses the
    toy seam to manufacture a structurally-valid schema-v3 carrier and injects
    a synthetic reveal sub-record to exercise the nested tag — a WIRING check,
    not a MegaGem reward/score check (covered by the RL plan 1.5)."""
    games = H.rollout_group(
        args.seed_start, trainable_policy_fn=H.make_stub_policy_fn("bestresp"),
        k=args.k)
    stripped = []
    for g in games:
        gg = copy.deepcopy(g)
        for rnd in gg["rounds"]:
            for rec in rnd["players"]:
                rec.pop("actor_id", None)
        # synthetic reveal sub-record on one non-trainable seat to prove the
        # nested actor_id tag (the toy is sealed-bid → no reveals natively).
        gg["rounds"][0]["players"][1]["reveal"] = {"actor_id": "STALE"}
        stripped.append(gg)

    tagged = [_post_tag_actor_mask(
        g, trainable_seat=args.trainable_seat,
        opponent_actor_id=args.opponent_actor_id) for g in stripped]

    reveal_tag = tagged[0]["rounds"][0]["players"][1]["reveal"]["actor_id"]
    rows = H.flatten_training_rows(tagged, trainable_seat=args.trainable_seat)
    every_row_trainable = all(
        r["actor_id"] == H.TRAINABLE_ACTOR_ID
        and r["player_id"] == args.trainable_seat for r in rows
    )
    checks = {
        "post_tag_top_level": all(
            rec["actor_id"] in (H.TRAINABLE_ACTOR_ID, args.opponent_actor_id)
            for g in tagged for rnd in g["rounds"] for rec in rnd["players"]
        ),
        "post_tag_nested_reveal": reveal_tag == args.opponent_actor_id,
        "export_contract_ok": len(rows) == args.k * H.NUM_ROUNDS,
        "actor_mask_drops_opponents": every_row_trainable,
    }
    passed = all(checks.values())
    return {
        "step": "2.3",
        "mode": "DRY-RUN (CPU — post-tag incl. reveal + export wiring only)",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "rows": len(rows),
    }


def _gpu_run(args, tmp: str) -> dict:
    H.print_trl_env()
    if not args.vllm_url:
        raise SystemExit("non-dry-run needs --vllm-url (run_game endpoint)")

    import os
    from megagem import endpoints
    os.environ.setdefault("EMPTY", "EMPTY")
    endpoints.ENDPOINTS[args.served_model_name] = {
        "model": args.served_model_name, "url": args.vllm_url, "key": "EMPTY",
    }

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer  # noqa: F401
    from datasets import Dataset
    from megagem.training.megagem_grpo import (
        MegaGemGRPOTrainer, precomputed_reward_func,
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16)
    # See megagem.training.grpo_harness: pre-instantiated model + peft_config skips
    # enable_input_require_grads → gradient checkpointing is a no-op. Enable
    # explicitly before TRL wraps with PEFT.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    t0 = time.perf_counter()
    rows = _pre_roll_rows(
        model_name=args.served_model_name, seeds=seeds, k=args.k,
        num_players=args.num_players, value_chart=args.value_chart,
        trainable_seat=args.trainable_seat,
        opponent_actor_id=args.opponent_actor_id, tmp_dir=tmp,
    )
    g = max(2, args.num_generations)
    if not rows:
        raise RuntimeError("pre-roll produced 0 trainable rows; raise "
                           "--num-seeds/--k.")
    roll_s = round(time.perf_counter() - t0, 2)

    # Same one-generation-covers-all-rows sizing as the toy: dataset = the
    # pre-rolled rows; per_device_train_batch_size = N·G so RepeatSampler's
    # one chunk is all N rows ×G. Advantages were already K-grouped correctly
    # by the single flatten in _pre_roll_rows; the static rollout re-emits by
    # index, so order/×G duplication is immaterial here.
    n = len(rows)
    gen_batch = n * g
    # Microbatch the optimizer — the LM-head logits tensor (per_device·seq·
    # vocab) is the binding memory term, immune to LoRA/grad-ckpt. Keep the
    # scoring batch = n·g (TRL: generation_batch_size = micro·spg = n·g) but
    # run forward/backward in `micro` chunks. See megagem.training.grpo_harness.
    _MICRO_CAP = int(os.environ.get("PHASE2_MICRO_CAP", "8"))  # see megagem.training.grpo_harness
    micro = g
    while micro * 2 <= _MICRO_CAP and gen_batch % (micro * 2) == 0:
        micro *= 2
    spg = gen_batch // micro
    assert micro * spg == gen_batch, (micro, spg, gen_batch)
    dataset = Dataset.from_dict({"prompt": [f"[mg-row] i={i}"
                                            for i in range(n)]})
    cfg = GRPOConfig(**H._filter_kwargs(GRPOConfig, dict(
        output_dir=tmp, per_device_train_batch_size=micro,
        num_generations=g,  # generation_batch_size TRL-derived (=micro·spg=n·g);
                             # pinned TRL rejects it set with steps_per_generation
        steps_per_generation=spg, num_iterations=1,
        gradient_accumulation_steps=1,
        max_completion_length=args.max_completion_length,
        max_prompt_length=8192, max_steps=args.steps,
        learning_rate=args.learning_rate, beta=args.kl_beta,
        temperature=1.0, top_p=0.95, logging_steps=1, save_strategy="no",
        report_to=[], use_vllm=False, bf16=True, fp16=False, seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )))
    t1 = time.perf_counter()
    trainer = MegaGemGRPOTrainer(**H._filter_kwargs(GRPOTrainer.__init__, dict(
        model=model, reward_funcs=[precomputed_reward_func], args=cfg,
        train_dataset=dataset, processing_class=tok,
        rollout_func=make_static_rollout_func(tok, rows),
        peft_config=H.megagem_lora_config(),
    )))
    build_s = round(time.perf_counter() - t1, 2)

    crashed, reason = False, None
    t2 = time.perf_counter()
    try:
        trainer.train()
    except Exception as e:  # noqa: BLE001 — "doesn't crash" IS the gate
        crashed, reason = True, f"{type(e).__name__}: {e}"
    train_s = round(time.perf_counter() - t2, 2)

    log_hist = list(getattr(trainer.state, "log_history", []))

    def _series(key):
        return [e[key] for e in log_hist if key in e]

    kls = _series("kl")
    losses = _series("loss")
    grad_norms = _series("grad_norm")
    kl_max_obs = max(kls) if kls else 0.0
    kl_bounded = (not kls) or (math.isfinite(kl_max_obs) and kl_max_obs <= args.kl_max)
    nan_free = all(math.isfinite(x) for x in losses)
    passed = (not crashed) and kl_bounded and nan_free
    # Completion length on the pre-rolled REAL-MegaGem rows (char proxy —
    # build_training_rows guarantees `completion`; the schema-v3 per-player
    # parse/legal fields are NOT verified here, so no fabricated quality
    # metric — honest, per the calibration discipline).
    _clen = [len(r.get("completion") or "") for r in rows]
    return {
        "step": "2.3", "mode": "GPU",
        "status": "PASS" if passed else "FAIL",
        "pre_rolled_rows": len(rows),
        "crashed": crashed, "crash_reason": reason,
        "kl_bounded": kl_bounded, "kl_max_observed": kl_max_obs,
        "kl_max_allowed": args.kl_max, "nan_free": nan_free,
        "logged_steps": len(losses),
        # Tier-B/C telemetry: same trends as 2.2 so a 2.3 no-crash run still
        # surfaces KL/grad/loss drift and completion-length pathologies on the
        # real reward path (where formatting degradation is most likely).
        "kl_trajectory": H.summarize_series(kls),
        "grad_norm_trajectory": H.summarize_series(grad_norms),
        "loss_trajectory": H.summarize_series(losses),
        "rollout_completion_len_chars": H.summarize_series(_clen),
        "rollout_logprobs_mode": os.environ.get(
            "PHASE2_ROLLOUT_LOGPROBS", "none").lower(),
        "roll_s": roll_s, "build_s": build_s, "train_s": train_s,
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
    except Exception as e:  # noqa: BLE001
        result = {"step": "2.3", "status": "FAIL",
                  "reason": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[{result.get('mode','?')}] Phase 2.3 — {result['status']}")
    if result["status"] != "PASS":
        if "traceback" in result:
            print("\n--- traceback (pinpoints the run_game / TRL seam delta) ---")
            print(result["traceback"])
        elif result.get("crash_reason"):
            print(f"  crash: {result['crash_reason']}")
        elif "checks" in result:
            for k, v in result["checks"].items():
                print(f"  {k}: {v}")
    print(f"→ {args.output}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
