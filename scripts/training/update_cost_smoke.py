#!/usr/bin/env python3
"""Phase 0.4 update-cost smoke (the RL plan §0.4, §9): wall-clock per GRPO-style
update step on saved schema-v3 trajectories. Gate: ≤ ~5 min/step on 1×H100.

Not a correctness check — advantages are placeholder (centered final_bid),
loss is a hand-rolled GRPO + k3 KL, not TRL's _compute_loss.

  uv run python scripts/training/update_cost_smoke.py \\
      --base-model Qwen/Qwen3-4B-Instruct-2507 \\
      --lora-checkpoint <sft-v2-checkpoint> \\
      --trajectories "results/throughput_trajectories/*.json" \\
      --num-turns 32 --max-length 8192 --micro-batch-size 1 \\
      --output results/update_cost_smoke.json

Forward/backward is micro-batched with exact gradient accumulation; per-token
logprobs use the logsumexp selective trick (no [B,T,V] log_softmax). So a
full-size 32×8192 step is measurable without OOM — raise --micro-batch-size
to trade memory for speed. Pass --dry-run to validate data prep only (no
torch/peft).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

PHASE_0_4_TARGET_SECONDS_PER_STEP = 300  # 5 minutes


@dataclass
class TurnRecord:
    game_path: str
    round_num: int
    player_id: int
    actor_id: str
    system_prompt: str
    user_prompt: str
    raw_response: str
    parsed_action: int | None
    final_bid: int
    parse_valid: bool
    legal_valid: bool
    default_used: bool


MIN_SCHEMA_VERSION = 3


def load_trajectories(paths: list[str | Path]) -> list[dict]:
    """Load schema-v3 game JSONs. v3 added ``metadata.system_prompt``; earlier
    versions under-count tokens by skipping the system message."""
    games = []
    for path in paths:
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        version = data.get("metadata", {}).get("schema_version", 1)
        if version < MIN_SCHEMA_VERSION:
            raise ValueError(
                f"{path} has schema_version={version}; Phase 0.4 requires "
                f">={MIN_SCHEMA_VERSION}. Re-run with the current megagem.rollout."
            )
        if "system_prompt" not in data.get("metadata", {}):
            raise ValueError(
                f"{path} is schema_version={version} but missing "
                "metadata.system_prompt; the file is malformed."
            )
        data["_path"] = str(path)
        games.append(data)
    return games


def iter_turn_records(games: list[dict]) -> list[TurnRecord]:
    """Flatten v3 games into one TurnRecord per bid turn. Reveals skipped."""
    records = []
    for game in games:
        system_prompt = game["metadata"]["system_prompt"]
        for round_data in game.get("rounds", []):
            for player in round_data.get("players", []):
                if "raw_response" not in player or "prompt" not in player:
                    continue
                records.append(
                    TurnRecord(
                        game_path=game["_path"],
                        round_num=round_data["round_number"],
                        player_id=player["player_id"],
                        actor_id=player.get("actor_id", "trainable"),
                        system_prompt=system_prompt,
                        user_prompt=player["prompt"],
                        raw_response=player["raw_response"],
                        parsed_action=player.get("parsed_action"),
                        final_bid=player.get("bid", 0),
                        parse_valid=player.get("parse_valid", False),
                        legal_valid=player.get("legal_valid", False),
                        default_used=player.get("default_used", False),
                    )
                )
    return records


def build_batch_records(
    records: list[TurnRecord], num_turns: int
) -> list[TurnRecord]:
    """First ``num_turns`` records; no shuffling so wall-clock is reproducible."""
    if len(records) < num_turns:
        raise ValueError(
            f"Need at least {num_turns} turn records; got {len(records)} "
            f"across {len(set(r.game_path for r in records))} games. "
            "Run more games via megagem.evals.game_runner first."
        )
    return records[:num_turns]


def summarize_batch(records: list[TurnRecord]) -> dict:
    """Dry-run batch summary; mean_prompt_chars includes the system message."""
    total_system_chars = sum(len(r.system_prompt) for r in records)
    total_user_chars = sum(len(r.user_prompt) for r in records)
    total_response_chars = sum(len(r.raw_response) for r in records)
    actor_counts = {}
    for r in records:
        actor_counts[r.actor_id] = actor_counts.get(r.actor_id, 0) + 1
    parseable = sum(1 for r in records if r.parse_valid)
    legal = sum(1 for r in records if r.parse_valid and r.legal_valid)
    n = max(len(records), 1)
    return {
        "num_turns": len(records),
        "num_unique_games": len({r.game_path for r in records}),
        "total_system_chars": total_system_chars,
        "total_user_chars": total_user_chars,
        "total_response_chars": total_response_chars,
        "mean_prompt_chars": round((total_system_chars + total_user_chars) / n, 1),
        "mean_response_chars": round(total_response_chars / n, 1),
        "actor_counts": actor_counts,
        "parseable_count": parseable,
        "legal_count": legal,
    }


def run_update_step(
    records: list[TurnRecord],
    base_model: str,
    lora_checkpoint: str,
    learning_rate: float = 1e-5,
    kl_beta: float = 0.05,
    max_length: int = 8192,
    gradient_checkpointing: bool = True,
    micro_batch_size: int = 1,
) -> dict:
    """One GRPO-style update step; returns phase timings.

    Setup (load + tokenize + warmup) is reported under ``setup_s`` and excluded
    from the Phase 0.4 gate. Every GPU phase is CUDA-synced; without that,
    ``perf_counter`` around async kernel launches reports zero.

    The forward/logprob/backward runs in ``micro_batch_size``-row chunks with
    exact gradient accumulation (every chunk divides by the *same* full-batch
    token denominator, so the summed chunk losses and grads equal the
    full-batch ones). Per-token logprobs use the ``logsumexp`` selective trick,
    never a ``[B, T, V]`` ``log_softmax``. Together these mirror TRL's
    ``_compute_loss`` memory profile — the un-chunked full-vocab path OOM'd a
    140 GB H200 at 32×8192 (~80 GB of logits per copy); see the RL plan I2.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def cuda_sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def selective_logprobs(logits, tgt):
        """Per-token logprob of ``tgt`` without the [B, T, V] log_softmax.

        logp = logits.gather(tgt) - logsumexp(logits, -1). Mirrors TRL's
        selective_log_softmax; avoids the ~80 GB full-vocab tensor that OOM'd
        the old log_softmax(...).gather(...) path at 32×8192.
        """
        gathered = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return gathered - torch.logsumexp(logits, dim=-1)

    timings: dict[str, float] = {}

    setup_t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    ).to("cuda")
    policy = PeftModel.from_pretrained(base, lora_checkpoint, is_trainable=True).to("cuda")
    policy.train()
    if gradient_checkpointing:
        # Standard PEFT pattern: re-enable input grads (frozen base would
        # otherwise short-circuit autograd before reaching the LoRA params),
        # disable kv-cache (incompatible with checkpointing), and turn on
        # activation rematerialization. ~3× activation savings, ~15% time cost.
        policy.enable_input_require_grads()
        policy.config.use_cache = False
        policy.gradient_checkpointing_enable()

    # Chat template so prompt tokens match what vLLM sees.
    rendered_prompts: list[str] = []
    full_texts: list[str] = []
    for r in records:
        messages = [
            {"role": "system", "content": r.system_prompt},
            {"role": "user", "content": r.user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        rendered_prompts.append(prompt_text)
        full_texts.append(prompt_text + r.raw_response)

    # Pre-tokenize without truncation to count overflow rows for the output.
    untruncated = tokenizer(full_texts, padding=False, truncation=False)["input_ids"]
    truncated_rows = sum(1 for ids in untruncated if len(ids) > max_length)
    max_observed_len = max((len(ids) for ids in untruncated), default=0)

    encoded = tokenizer(
        full_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    ).to("cuda")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    completion_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    for i, prompt_text in enumerate(rendered_prompts):
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        seq_len = attention_mask[i].sum().item()
        completion_mask[i, prompt_len:seq_len] = 1.0

    actor_mask_per_row = torch.tensor(
        [1.0 if r.actor_id == "trainable" else 0.0 for r in records],
        device="cuda",
        dtype=torch.float32,
    )
    actor_mask = completion_mask * actor_mask_per_row.unsqueeze(1)
    targets = input_ids[:, 1:]
    mask = actor_mask[:, 1:]

    bids = torch.tensor([float(r.final_bid) for r in records], device="cuda")
    advantages = bids - bids.mean()
    if advantages.std() > 0:
        advantages = advantages / advantages.std()

    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad), lr=learning_rate
    )

    mb = max(1, micro_batch_size)
    num_rows = input_ids.shape[0]
    num_micro = (num_rows + mb - 1) // mb

    # Warmup on ONE micro-batch, not the full batch — at 32×8192 a full-batch
    # forward materializes ~80 GB of logits alone and OOMs even a 140 GB H200.
    with torch.no_grad():
        policy(input_ids=input_ids[:mb], attention_mask=attention_mask[:mb])
    cuda_sync()
    timings["setup_s"] = round(time.perf_counter() - setup_t0, 3)

    # Per-token loss is averaged over every trainable-actor completion token in
    # the *full* batch. Chunked backward is exact gradient accumulation iff
    # every chunk divides by this one global denominator (Σ chunk losses ==
    # full-batch loss; Σ chunk grads == full-batch grads).
    global_denom = mask.sum().clamp(min=1.0)

    optimizer.zero_grad()
    acc = {"current_logprobs_s": 0.0, "ref_logprobs_s": 0.0,
           "loss_s": 0.0, "backward_step_s": 0.0}
    total_loss = 0.0
    total_kl_num = torch.zeros((), device="cuda")

    cuda_sync()
    step_t0 = time.perf_counter()
    for c in range(num_micro):
        s = c * mb
        e = min(s + mb, num_rows)
        mb_input = input_ids[s:e]
        mb_attn = attention_mask[s:e]
        mb_targets = targets[s:e]
        mb_mask = mask[s:e]
        mb_adv = advantages[s:e]

        t0 = time.perf_counter()
        out = policy(input_ids=mb_input, attention_mask=mb_attn)
        cur_lp = selective_logprobs(out.logits[:, :-1, :], mb_targets)
        cuda_sync()
        acc["current_logprobs_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad(), policy.disable_adapter():
            ref_out = policy(input_ids=mb_input, attention_mask=mb_attn)
        ref_lp = selective_logprobs(ref_out.logits[:, :-1, :], mb_targets)
        cuda_sync()
        acc["ref_logprobs_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        log_ratio = ref_lp - cur_lp
        per_token_kl = torch.expm1(log_ratio) - log_ratio
        per_token_loss = -(cur_lp * mb_adv.unsqueeze(1)) + kl_beta * per_token_kl
        chunk_loss = (per_token_loss * mb_mask).sum() / global_denom
        cuda_sync()
        acc["loss_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        chunk_loss.backward()
        cuda_sync()
        acc["backward_step_s"] += time.perf_counter() - t0

        total_loss += float(chunk_loss.detach())
        total_kl_num = total_kl_num + (per_token_kl.detach() * mb_mask).sum()

    t0 = time.perf_counter()
    optimizer.step()
    cuda_sync()
    acc["backward_step_s"] += time.perf_counter() - t0

    timings["current_logprobs_s"] = round(acc["current_logprobs_s"], 3)
    timings["ref_logprobs_s"] = round(acc["ref_logprobs_s"], 3)
    timings["loss_s"] = round(acc["loss_s"], 3)
    timings["backward_step_s"] = round(acc["backward_step_s"], 3)
    timings["update_step_s"] = round(time.perf_counter() - step_t0, 3)
    timings["loss_value"] = total_loss
    timings["mean_kl_per_token"] = float(total_kl_num / global_denom)
    timings["micro_batch_size"] = mb
    timings["num_micro_batches"] = num_micro
    timings["max_length"] = max_length
    timings["truncated_rows"] = truncated_rows
    timings["max_observed_token_len"] = max_observed_len
    return timings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--trajectories', nargs='+', required=True)
    p.add_argument("--num-turns", type=int, default=32)
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument(
        "--lora-checkpoint",
        default="djdumpling/qwen3-4b-instruct-megagem-sft-step1200-lora",
    )
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.05)
    p.add_argument('--max-length', type=int, default=8192)
    p.add_argument(
        '--gradient-checkpointing',
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rematerialize activations during backward (~3× activation savings, ~15% time). "
             "Default on so 80 GB H100 fits the default batch.",
    )
    p.add_argument(
        '--micro-batch-size', type=int, default=1,
        help="Rows per forward/backward chunk. Loss is exact gradient "
             "accumulation over the full --num-turns batch, so this only "
             "trades memory for speed (no effect on the optimizer step). "
             "Default 1 fits 32×8192 on 80 GB; raise on bigger cards. "
             "Mirrors TRL per-device micro-batching.",
    )
    p.add_argument('--dry-run', action='store_true')
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    paths: list[str] = []
    for pattern in args.trajectories:
        matched = glob.glob(pattern)
        if not matched:
            print(f"WARNING: no files matched {pattern!r}", file=sys.stderr)
        paths.extend(matched)
    if not paths:
        print("ERROR: no trajectory files found.", file=sys.stderr)
        return 2

    games = load_trajectories(paths)
    records = iter_turn_records(games)
    batch = build_batch_records(records, args.num_turns)
    summary = {
        "config": {
            "num_turns": args.num_turns,
            "base_model": args.base_model,
            "lora_checkpoint": args.lora_checkpoint,
            "learning_rate": args.learning_rate,
            "kl_beta": args.kl_beta,
            "max_length": args.max_length,
            "gradient_checkpointing": args.gradient_checkpointing,
            "micro_batch_size": args.micro_batch_size,
            "dry_run": args.dry_run,
        },
        "batch_summary": summarize_batch(batch),
        "batch_records": [asdict(r) for r in batch],
    }

    if args.dry_run:
        summary["timings_s"] = None
        summary["phase_0_4_gate"] = None
    else:
        timings = run_update_step(
            batch,
            base_model=args.base_model,
            lora_checkpoint=args.lora_checkpoint,
            learning_rate=args.learning_rate,
            kl_beta=args.kl_beta,
            max_length=args.max_length,
            gradient_checkpointing=args.gradient_checkpointing,
            micro_batch_size=args.micro_batch_size,
        )
        summary["timings_s"] = timings
        update_s = timings["update_step_s"]
        summary["phase_0_4_gate"] = {
            "target_seconds_per_step": PHASE_0_4_TARGET_SECONDS_PER_STEP,
            "measured_seconds_per_step": update_s,
            "passed": update_s <= PHASE_0_4_TARGET_SECONDS_PER_STEP,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    bs = summary["batch_summary"]
    print(f"\nBatch: {bs['num_turns']} turns from {bs['num_unique_games']} games "
          f"(actor counts: {bs['actor_counts']})")
    if args.dry_run:
        print("Dry run complete — data prep validated; skipped torch.")
        return 0

    timings = summary["timings_s"]
    print(f"Setup (load + tokenize + warmup): {timings['setup_s']}s")
    print(f"Micro-batching: {timings['num_micro_batches']} chunks × "
          f"{timings['micro_batch_size']} rows (exact grad accumulation)")
    print(f"Update step (gated): {timings['update_step_s']}s")
    for k, v in timings.items():
        if k.endswith("_s") and k not in {"setup_s", "update_step_s"}:
            print(f"  {k}: {v}s")
    print(f"Max-length cap: {timings['max_length']} tokens; "
          f"max observed: {timings['max_observed_token_len']}; "
          f"truncated rows: {timings['truncated_rows']}/{len(batch)}")
    if timings["truncated_rows"] > 0:
        print("WARNING: some rows were truncated. Re-run with a larger "
              "--max-length to measure full update cost.")
    gate = summary["phase_0_4_gate"]
    if gate["passed"]:
        print(f"PASS — clears the {PHASE_0_4_TARGET_SECONDS_PER_STEP}s/step target.")
        return 0
    print(f"FAIL — {timings['update_step_s']}s exceeds the "
          f"{PHASE_0_4_TARGET_SECONDS_PER_STEP}s/step target. "
          "Revisit batch size or trainer choice before Phase 3.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
