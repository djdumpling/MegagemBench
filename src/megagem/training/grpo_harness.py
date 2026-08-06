"""Phase-2 GRPO glue: turns the toy/MegaGem rollout seam into a TRL
``rollout_func`` + ``MegaGemGRPOTrainer`` (the RL plan §2). Per-turn advantage
standardization is done by ``megagem.rl.export.build_training_rows``; this module
is only the TRL-facing wiring (zero ``megagem/rl`` edits). TRL ``use_vllm`` is
always False — generation is via our external vLLM server. The rollout-length
contract (one return row per input prompt) is verified in
``tests/test_grpo_harness.py``. Pure-import safe: torch/trl/openai are lazy.
"""

from __future__ import annotations

import os
import random
import re
import statistics
from collections.abc import Callable, Sequence
from typing import Any

from megagem.training.trl_utils import _filter_kwargs, print_trl_env  # noqa: F401

from megagem.rl.export import build_training_rows, contract_check
from megagem.rl.reward import RewardConfig, terminal_breakdowns
from megagem.rl.scorer import parse_transcript
from megagem.toy.auction_env import NUM_PLAYERS, NUM_ROUNDS, run_one_toy_game
from megagem.toy.prompt import (
    _read as _read_toy_prompt,
    make_heuristic_policy,
    make_stub_policy,
)

PolicyFn = Callable[[str], str]


def reward_config_from_env() -> RewardConfig:
    """RewardConfig with optional per-field env overrides. Empty env ⇒
    byte-identical to ``RewardConfig()``. Float vars: PHASE3_REWARD_SCALE,
    PHASE3_REWARD_WIN_BONUS, PHASE3_SHAPING_LAMBDA, PHASE3_ILLEGAL_PENALTY,
    PHASE3_CORRECTION_SCALE, PHASE3_REVEAL_SHAPING_WEIGHT. Bool:
    PHASE3_TERMINAL_CORRECTION. Enum: PHASE3_TERMINAL_SHAPE (``clip``|``tanh``)."""
    overrides: dict[str, float | bool | str] = {}
    for env_name, field_name in (
        ("PHASE3_REWARD_SCALE", "scale"),
        ("PHASE3_REWARD_WIN_BONUS", "win_bonus"),
        ("PHASE3_SHAPING_LAMBDA", "shaping_lambda"),
        ("PHASE3_ILLEGAL_PENALTY", "illegal_penalty"),
        ("PHASE3_CORRECTION_SCALE", "correction_scale"),
        ("PHASE3_REVEAL_SHAPING_WEIGHT", "reveal_shaping_weight"),
    ):
        raw = os.environ.get(env_name)
        if raw is not None and raw != "":
            overrides[field_name] = float(raw)
    # rec 1a terminal_correction — bool, not float ⇒ parsed separately;
    # unset/empty ⇒ default False, so the no-env path stays byte-identical.
    tc = os.environ.get("PHASE3_TERMINAL_CORRECTION")
    if tc is not None and tc != "":
        overrides["terminal_correction"] = tc.strip().lower() in (
            "1", "true", "yes", "on"
        )
    # rec 2 terminal_shape — string ⇒ parsed separately; validated to the same
    # {clip,tanh} set _terminal_value accepts so a typo fails HERE (at config
    # build, before any GPU spend) rather than deep in the reward path. Unset
    # ⇒ "clip", so the no-env path stays byte-identical.
    ts = os.environ.get("PHASE3_TERMINAL_SHAPE")
    if ts is not None and ts != "":
        ts = ts.strip().lower()
        if ts not in ("clip", "tanh"):
            raise ValueError(
                f"PHASE3_TERMINAL_SHAPE={ts!r} invalid (expected 'clip'|'tanh')"
            )
        overrides["terminal_shape"] = ts
    return RewardConfig(**overrides)

SYSTEM_PROMPT = "Toy sealed-bid auction (Phase 2.1 trainer-mechanics harness)."
TRAINABLE_SEAT = 0
TRAINABLE_ACTOR_ID = "trainable"

# Turn-as-prompt dataset marker (NOT shown to the policy — it sees only the
# per-turn bid_prompt the toy seam builds). Encodes (seed, rollout k, round)
# so rollout_func can roll each (seed,k) game once and re-emit its rows in
# dataset order. Keep stable.
_TURN_PROMPT_RE = re.compile(
    r"\[toy-turn\] seed=(\d+) k=(\d+) round=(\d+) chart=(\S+)"
)


def turn_prompt(seed: int, k: int, round_index: int, value_chart: str = "A") -> str:
    return f"[toy-turn] seed={seed} k={k} round={round_index} chart={value_chart}"


def parse_turn_prompt(text: str) -> tuple[int, int, int, str]:
    m = _TURN_PROMPT_RE.search(text)
    if not m:
        raise ValueError(f"not a toy-turn prompt: {text!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)


def toy_turn_dataset_prompts(
    seeds: Sequence[int], k: int, num_rounds: int = NUM_ROUNDS,
    value_chart: str = "A",
) -> list[str]:
    """One dataset row per (seed, k, round), seed-major then k then round so a
    game's ``num_rounds`` trainable turns are contiguous. Length is exactly
    the per-step trainable-row count B = len(seeds)·k·num_rounds."""
    return [
        turn_prompt(s, ki, r, value_chart)
        for s in seeds for ki in range(k) for r in range(num_rounds)
    ]


# --------------------------------------------------------------------------- #
# Policies — the one ``policy_fn(prompt)->text`` contract, three backings.     #
# --------------------------------------------------------------------------- #

def make_stub_policy_fn(strategy: str = "bestresp", jitter_seed: int | None = None) -> PolicyFn:
    """Deterministic CPU stub (Phase 2.1) — for ``--dry-run`` and the
    step-0 determinism check. No torch/network."""
    return make_stub_policy(strategy, jitter_seed=jitter_seed)


def make_vllm_policy_fn(
    base_url: str,
    model: str,
    *,
    api_key: str = "EMPTY",
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 256,
    timeout: float = 60.0,
    system_prompt: str = SYSTEM_PROMPT,
) -> PolicyFn:
    """Live rollout policy: one chat completion per bid turn against an
    *external* OpenAI-compatible vLLM server (NOT TRL-internal vLLM).

    ``extra_body`` mirrors ``megagem/rollout.py:206`` verbatim — local vLLM
    (``key="EMPTY"``) needs ``enable_thinking=False``. T=1.0/top-p 0.95 are
    the §A.7 rollout sampling so K rollouts of one seed differ → non-degenerate
    within-group advantage std.
    """
    from openai import OpenAI  # lazy: keeps CPU dry-run import-clean

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    def policy(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        return resp.choices[0].message.content or ""

    return policy


def make_hf_generate_policy_fn(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
    system_prompt: str = SYSTEM_PROMPT,
    enable_thinking: bool = False,
) -> PolicyFn:
    """Gate-eval / on-policy-rollout policy: ``model.generate`` on the
    *current* trainer weights. The before/after gate measures exactly the
    policy the loss is updating — independent of any external-server resync.
    Lazy torch.

    Two correctness controls (both matter for an honest before/after):

    * **eval mode during generation.** The LoRA config has
      ``lora_dropout=0.05``; under ``trainer.train()`` the model is in
      ``.train()`` so dropout is LIVE. A greedy "no sampling noise" gate eval
      that left dropout on would inject stochasticity into the supposed-
      deterministic r0/rN measurement (and into on-policy rollouts). We
      ``model.eval()`` for the duration and restore the prior mode in
      ``finally`` (so an in-training on-policy call resumes correctly).
    * **``enable_thinking=False``** forwarded to ``apply_chat_template`` to
      mirror the vLLM rollout surface (``make_vllm_policy_fn`` sends
      ``chat_template_kwargs={'enable_thinking': False}``) and the
      the RL plan Qwen3 native-thinking-off lock — otherwise HF eval and the
      vLLM rollout would decode under different prompt surfaces.
    """
    import torch

    def policy(prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        enc = tokenizer(text, return_tensors="pt").to(model.device)
        # This gate eval is greedy decode under no_grad — no backward — so the
        # training-time gradient checkpointing + use_cache=False are pure
        # overhead: HF forces use_cache=False whenever ckpt is on, making
        # generation cache-less and O(n^2). It dominated wall time (~11 min per
        # eval pass, ×2 per 2.2 run). Greedy decoding with vs without the KV
        # cache is token-identical, so re-enabling the cache here is a LOSSLESS
        # speedup — the gate metric is unchanged. build_toy_trainer/_gpu_run
        # couple use_cache=False ⇔ ckpt-on, so prev_use_cache is False is the
        # reliable "training mode" signal (don't rely on is_gradient_
        # checkpointing alone — it can mis-resolve through the PEFT wrapper).
        # Restore in finally: r0 eval runs BEFORE trainer.train(), so leaving
        # ckpt disabled would OOM training (the original wall, undone). The
        # build-time enable_input_require_grads hook persists across a ckpt
        # disable/enable, so it does not need re-arming.
        prev_use_cache = getattr(getattr(model, "config", None),
                                 "use_cache", None)
        was_ckpt = (prev_use_cache is False) or bool(
            getattr(model, "is_gradient_checkpointing", False))
        # LoRA dropout (0.05) is active in .train(); eval() makes greedy decode
        # genuinely deterministic. Preserve+restore so an on-policy rollout
        # call (which runs INSIDE trainer.train()) resumes in train mode.
        was_training = bool(getattr(model, "training", False))
        try:
            if was_ckpt and hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            if hasattr(model, "eval"):
                model.eval()
            if getattr(model, "config", None) is not None:
                model.config.use_cache = True
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
        finally:
            if was_training and hasattr(model, "train"):
                model.train()  # restore BEFORE re-enabling ckpt below
            if was_ckpt and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            if getattr(model, "config", None) is not None:
                model.config.use_cache = prev_use_cache
        gen = out[0][enc["input_ids"].shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=True)

    return policy


# --------------------------------------------------------------------------- #
# Rollout + the unmodified-pipeline choke point.                              #
# --------------------------------------------------------------------------- #

def rollout_group(
    seed: int,
    *,
    trainable_policy_fn: PolicyFn,
    k: int,
    opponent_actor_id: str = "heuristic",
    value_chart: str = "A",
    num_rounds: int = NUM_ROUNDS,
    num_players: int = NUM_PLAYERS,
    stub_strategy: str | None = None,
) -> list[dict[str, Any]]:
    """K rollouts of one seed → K schema-v3 games sharing ``(seed, chart)``.
    Trainable seat = ``trainable_policy_fn``; opponents fixed heuristic
    (identical across k → a true GRPO group). ``stub_strategy`` rebuilds a
    per-k jittered stub for the dry-run path."""
    heuristic = make_heuristic_policy()
    games: list[dict[str, Any]] = []
    for ki in range(k):
        trainable = (
            make_stub_policy(stub_strategy, jitter_seed=ki)
            if stub_strategy is not None else trainable_policy_fn
        )
        policy_fns = [
            trainable if s == TRAINABLE_SEAT else heuristic
            for s in range(num_players)
        ]
        actor_ids = [
            TRAINABLE_ACTOR_ID if s == TRAINABLE_SEAT else opponent_actor_id
            for s in range(num_players)
        ]
        games.append(run_one_toy_game(
            seed, policy_fns, actor_ids, rollout_index=ki,
            num_players=num_players, num_rounds=num_rounds,
            value_chart=value_chart,
        ))
    return games


def flatten_training_rows(
    games: Sequence[dict[str, Any]],
    reward_cfg: RewardConfig | None = None,
    *,
    trainable_seat: int = TRAINABLE_SEAT,
) -> list[dict[str, Any]]:
    """THE choke point. ``megagem.rl.export.build_training_rows`` verbatim — the
    only path from games to trainer rows.

    Re-asserts the §A.6 actor mask (the *true* invariant is
    ``actor_id == "trainable"``; ``trainable_seat`` is an extra positional
    check the caller knows — parameterized so non-zero seats work) and the
    ``(B,)`` finite-float contract. A regression here, not in ``megagem/rl``, is
    where Phase 2 would surface it."""
    rows = build_training_rows(
        list(games),
        reward_cfg if reward_cfg is not None else reward_config_from_env(),
    )
    for i, r in enumerate(rows):
        if r["actor_id"] != TRAINABLE_ACTOR_ID:
            raise AssertionError(
                f"row {i} leaked a non-trainable turn (actor_id="
                f"{r['actor_id']!r}); build_training_rows mask violated."
            )
        if r["player_id"] != trainable_seat:
            raise AssertionError(
                f"row {i} is seat {r['player_id']}, expected trainable_seat="
                f"{trainable_seat}."
            )
    _guard_degenerate_kgroups(rows)
    contract_check(rows)  # raises unless every row has finite reward+advantage
    return rows


def _guard_degenerate_kgroups(rows: list[dict[str, Any]]) -> None:
    """P1.4 — neutralize outcome-degenerate K-groups (in-place, harness-only).

    A confident policy can produce K near-identical rollouts of a seed → tiny
    within-group RTG spread. ``megagem.rl.advantage`` then divides by a tiny σ
    (``denom = σ if σ>1e-8 else 1.0``): an exactly-equal group already collapses
    to advantage 0, but a *near*-equal one gets nano-noise amplified into large
    ± advantages — pure sampling noise fed to the optimizer. This guard zeroes
    the advantages of any K-group whose per-rollout terminal reward spread is
    ≤ ``PHASE2_DEGENERATE_KGROUP_EPS`` (a no-op signal beats amplified noise).

    **Default OFF** (eps ``0`` ⇒ this returns immediately and the verified
    ``megagem/rl``-unmodified path is byte-identical). Single-rollout groups
    (``<2`` games, e.g. eval / k=1 tests) are never guarded. Kept in the
    harness — NOT ``megagem/rl`` — per the zero-edit choke-point invariant.
    """
    try:
        eps = float(os.environ.get("PHASE2_DEGENERATE_KGROUP_EPS", "0") or 0)
    except ValueError:
        eps = 0.0
    if eps <= 0:
        return
    by_grp: dict[Any, dict[Any, float]] = {}
    for r in rows:
        if r.get("is_terminal_turn"):
            by_grp.setdefault(r["group_key"], {})[r["game_id"]] = float(
                r["precomputed_reward"]
            )
    degenerate = {
        gk for gk, gm in by_grp.items()
        if len(gm) >= 2 and (max(gm.values()) - min(gm.values())) <= eps
    }
    if not degenerate:
        return
    for r in rows:
        if r["group_key"] in degenerate:
            r["precomputed_advantage"] = 0.0


def oracle_shade_sweep(
    eval_seeds: Sequence[int],
    shades: Sequence[float] = (
        0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00,
    ),
    *,
    reward_cfg: RewardConfig | None = None,
    value_chart: str = "A",
    num_rounds: int = NUM_ROUNDS,
    num_players: int = NUM_PLAYERS,
) -> dict[str, Any]:
    """P1.2 — make the learning *target* explicit, not implicit in a stub.

    Mean trainable-seat terminal reward for a fixed ``round(shade·v)`` seat-0
    policy vs the fixed ``0.8·v`` heuristic opponents, over ``eval_seeds``. The
    gate should read "did the policy move toward ``argmax_shade``", not "did it
    imitate any one stub". Pure CPU (one deterministic game per (shade, seed)).
    """
    cfg = reward_cfg or reward_config_from_env()
    heuristic = make_heuristic_policy()
    seeds = list(eval_seeds)

    def _shade_policy(shade: float) -> PolicyFn:
        def policy(prompt: str) -> str:
            d = _read_toy_prompt(prompt)
            b = max(0, min(d["coins"], round(shade * d["private_value"])))
            return f'{{"bid": {b}}}'
        return policy

    table = []
    for sh in shades:
        pol = _shade_policy(sh)
        total = 0.0
        for s in seeds:
            policy_fns = [
                pol if i == TRAINABLE_SEAT else heuristic
                for i in range(num_players)
            ]
            actor_ids = [
                TRAINABLE_ACTOR_ID if i == TRAINABLE_SEAT else "heuristic"
                for i in range(num_players)
            ]
            game = run_one_toy_game(
                s, policy_fns, actor_ids, num_players=num_players,
                num_rounds=num_rounds, value_chart=value_chart,
            )
            total += trainable_terminal_reward(game, reward_cfg=cfg)
        table.append({
            "shade": round(sh, 4),
            "mean_terminal_reward": round(total / len(seeds), 6),
        })
    best = max(table, key=lambda r: r["mean_terminal_reward"])
    return {
        "n_seeds": len(seeds),
        "opponent_shade": 0.8,
        "table": table,
        "argmax_shade": best["shade"],
        "argmax_mean_reward": best["mean_terminal_reward"],
    }


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman ρ = Pearson on average-tie ranks. None if undefined (<2 points
    or zero rank-variance on either side). stdlib only."""
    n = len(xs)
    if n < 2:
        return None

    def _ranks(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-based average rank for ties
            for t in range(i, j + 1):
                rk[order[t]] = avg
            i = j + 1
        return rk

    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 1e-12 or syy <= 1e-12:
        return None
    sxy = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def advantage_reward_rank_corr(
    seeds: Sequence[int],
    *,
    k: int,
    policy_fn: PolicyFn,
    reward_cfg: RewardConfig | None = None,
) -> dict[str, Any]:
    """P2.2 — the *real* advantage-signal health check.

    ``advantage_signs_sensible`` / ``gradients_masked_correctly`` only assert
    finite advantages reached the loss (a plumbing contract), NOT that they
    point the right way. Here: per seed, roll ``k`` rollouts with ``policy_fn``
    (must be stochastic, else the K-group is degenerate), one true-K-group
    flatten, then Spearman-correlate each rollout's mean advantage against its
    terminal reward within the group. A healthy GRPO signal ⇒ ρ > 0 (higher
    reward ⇒ higher advantage). Reported, NOT gated (one noisy rollout must not
    fail a run). Pure given ``policy_fn``."""
    cfg = reward_cfg or reward_config_from_env()
    per_group: list[dict[str, Any]] = []
    for s in seeds:
        games = rollout_group(s, trainable_policy_fn=policy_fn, k=k)
        rows = flatten_training_rows(games, cfg)
        adv_sum: dict[int, float] = {}
        adv_cnt: dict[int, int] = {}
        for r in rows:
            gid = r["game_id"]
            adv_sum[gid] = adv_sum.get(gid, 0.0) + float(r["precomputed_advantage"])
            adv_cnt[gid] = adv_cnt.get(gid, 0) + 1
        gids = sorted(adv_sum)
        mean_adv = [adv_sum[g] / adv_cnt[g] for g in gids]
        term_r = [trainable_terminal_reward(games[g], reward_cfg=cfg) for g in gids]
        rho = _spearman(term_r, mean_adv)
        # Degeneracy proxy = cross-rollout spread of the per-game TERMINAL
        # REWARD (exactly what the P1.4 guard keys on). ~0 ⇒ the K rollouts had
        # identical outcomes (a confident policy produced near-identical
        # trajectories) → GRPO has nothing to rank, regardless of rho. NB: the
        # *advantage* spread is NOT a clean proxy here — the order-dependent
        # EMA baseline makes even identical rollouts differ in advantage.
        term_sigma = (statistics.pstdev(term_r) if len(term_r) > 1 else 0.0)
        adv_sigma = (statistics.pstdev(mean_adv) if len(mean_adv) > 1 else 0.0)
        per_group.append({"seed": s, "k": len(gids), "spearman": rho,
                          "termreward_sigma": term_sigma,
                          "meanadv_sigma": adv_sigma})
    vals = [g["spearman"] for g in per_group if g["spearman"] is not None]
    sigmas = sorted(g["termreward_sigma"] for g in per_group)
    eps = 1e-3  # below this the K rollouts are ~outcome-degenerate
    sigma_summary = {
        "median": (statistics.median(sigmas) if sigmas else None),
        "p10": (sigmas[max(0, int(0.10 * len(sigmas)) - 1)] if sigmas else None),
        "min": (sigmas[0] if sigmas else None),
        "frac_below_eps": (sum(1 for x in sigmas if x < eps) / len(sigmas)
                           if sigmas else None),
        "eps": eps,
    }
    return {
        "metric": "within-K-group Spearman(terminal_reward, mean_advantage)",
        "mean_spearman": (sum(vals) / len(vals)) if vals else None,
        "n_groups_used": len(vals),
        "n_groups_total": len(per_group),
        # P1.4 diagnostic (read-only — does NOT enable the guard): did the K
        # rollouts actually differ in outcome for this policy?
        "kgroup_termreward_sigma": sigma_summary,
        "per_group": per_group,
        "interpretation": "want spearman>0 (reward↑⇒advantage↑) AND "
                          "termreward_sigma≫eps (K rollouts differ); "
                          "sigma→0 ⇒ confident policy, no signal to rank",
    }


def game_behavior_stats(
    game: dict[str, Any],
    *,
    trainable_actor_id: str = TRAINABLE_ACTOR_ID,
) -> dict[str, Any]:
    """Per-game behaviour of the trainable seat (schema-v3, toy auction).

    Turns reward into a *mechanistic* signal: ``shade`` = the policy's chosen
    bid / its private value (only on parse+legal turns with pv>0 — a defaulted
    turn has no intended bid). Lets the gate say WHICH way the policy moved
    (toward the oracle plateau? off the JSON contract?), not just whether
    reward changed. Pure; tolerant of missing keys (returns None fields)."""
    n = pv0 = 0
    pok = lok = dft = 0
    shade_sum = 0.0
    shade_n = 0
    rawlen_sum = 0
    for rnd in game.get("rounds", []):
        for rec in rnd.get("players", []):
            if rec.get("actor_id") != trainable_actor_id:
                continue
            n += 1
            pv = rec.get("private_value")
            parsed = rec.get("parsed_action")
            pvalid = bool(rec.get("parse_valid"))
            lvalid = bool(rec.get("legal_valid"))
            pok += int(pvalid)
            lok += int(lvalid)
            dft += int(bool(rec.get("default_used")))
            rawlen_sum += len(rec.get("raw_response") or "")
            if pv in (0, None):
                pv0 += 1
            if pvalid and lvalid and isinstance(parsed, (int, float)) \
                    and pv not in (0, None):
                shade_sum += float(parsed) / float(pv)
                shade_n += 1
    if n == 0:
        return {"n_turns": 0, "shade_mean": None, "shade_n": 0,
                "parse_valid_frac": None, "legal_valid_frac": None,
                "default_used_frac": None, "raw_len_mean": None,
                "pv_zero_turns": 0}
    return {
        "n_turns": n,
        "shade_mean": (shade_sum / shade_n) if shade_n else None,
        "shade_n": shade_n,
        "parse_valid_frac": pok / n,
        "legal_valid_frac": lok / n,
        "default_used_frac": dft / n,
        "raw_len_mean": rawlen_sum / n,
        "pv_zero_turns": pv0,
    }


def aggregate_behavior(games: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean of ``game_behavior_stats`` across games (shade weighted by the
    number of shade samples, so a game with no usable shade does not skew it)."""
    per = [game_behavior_stats(g) for g in games]
    if not per:
        return {"n_games": 0}

    def _meanf(key):
        xs = [p[key] for p in per if p.get(key) is not None]
        return (sum(xs) / len(xs)) if xs else None

    sw = sum(p["shade_mean"] * p["shade_n"]
             for p in per if p.get("shade_mean") is not None)
    sn = sum(p["shade_n"] for p in per if p.get("shade_mean") is not None)
    return {
        "n_games": len(per),
        "shade_mean": (sw / sn) if sn else None,
        "shade_n": sn,
        "parse_valid_frac": _meanf("parse_valid_frac"),
        "legal_valid_frac": _meanf("legal_valid_frac"),
        "default_used_frac": _meanf("default_used_frac"),
        "raw_len_mean": _meanf("raw_len_mean"),
    }


def summarize_series(values: Sequence[float]) -> dict[str, Any]:
    """first/last/mean/min/max/step_of_max for a per-step log series — turns a
    raw TRL log_history list into the trend a reviewer actually reads."""
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"n": 0, "first": None, "last": None, "mean": None,
                "min": None, "max": None, "step_of_max": None}
    return {
        "n": len(xs),
        "first": xs[0],
        "last": xs[-1],
        "mean": sum(xs) / len(xs),
        "min": min(xs),
        "max": max(xs),
        "step_of_max": xs.index(max(xs)),
    }


# --------------------------------------------------------------------------- #
# Honest gate signal + paired bootstrap (stdlib-only, seeded, deterministic). #
# --------------------------------------------------------------------------- #

def trainable_terminal_reward(
    game: dict[str, Any],
    *,
    seat: int = TRAINABLE_SEAT,
    reward_cfg: RewardConfig | None = None,
) -> float:
    """The Phase-2.1-hardened *robust* signal: the trainable seat's
    engine-true terminal reward (§A.0 ``r_terminal``) — the Phase 2 gate
    metric, NOT advantage sign (a seed-pinned canary in 2.1)."""
    cfg = reward_cfg or reward_config_from_env()
    parsed = parse_transcript(game)
    return float(terminal_breakdowns(parsed, cfg)[seat].r_terminal)


def paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Paired-bootstrap CI on ``mean(delta_i)`` (the RL plan §2 / line 121).
    stdlib ``random`` only (reproducible on any box; dependency-free pytest).
    The gate is ``ci_low > 0``."""
    d = [float(x) for x in deltas]
    n = len(d)
    if n == 0:
        raise ValueError("paired_bootstrap_ci needs at least one delta")
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "n": n,
        "mean_delta": sum(d) / n,
        "ci": ci,
        "ci_low": means[max(0, int(lo_q * n_resamples) - 1)],
        "ci_high": means[min(n_resamples - 1, int(hi_q * n_resamples))],
        "n_resamples": n_resamples,
    }


# --------------------------------------------------------------------------- #
# The TRL rollout_func (turn-as-prompt) + trainer builder.                    #
# --------------------------------------------------------------------------- #

def rollout_logprobs(completion_ids: list[list[int]]):
    """Rollout-provided per-token logprobs — the P0.2 honesty fix.

    On-box read of the pinned TRL (phase-2 plan P0.2): with
    ``use_vllm=False`` and default ``off_policy_mask_threshold=None``, provided
    rollout logprobs become ``sampling_per_token_logps`` but **do not drive the
    PPO ratio** (TRL recomputes ``old_per_token_logps`` from the trainer model
    for ``spg>1``; uses ``per_token_logps.detach()`` for ``spg=1``). The legacy
    ``[0.0]*len`` fabrication was therefore harmless *under the current config*
    but a **latent footgun** (it silently corrupts the importance ratio the
    moment TRL/vLLM importance correction or off-policy masking is enabled —
    Phase 3) and meaningless telemetry. Default now: emit **no** logprobs
    (field-level ``None`` — the standard "rollout did not compute logprobs"
    signal). Env overrides for on-box A/B / the future real path:

    * ``PHASE2_ROLLOUT_LOGPROBS=none``  (default) → ``None``
    * ``PHASE2_ROLLOUT_LOGPROBS=zeros``           → legacy ``[[0.0]*len(c)…]``
      (kept so a v5.x cadence-sweep run is exactly reproducible on-box)
    * ``PHASE2_ROLLOUT_LOGPROBS=real``            → ``NotImplementedError``
      (real capture needs on-policy generation-time scores aligned to the
      re-tokenized completion — on-box only; see P0.1/P0.2)

    ON-BOX: confirm the pinned TRL accepts ``logprobs=None`` from a
    ``rollout_func``; if it hard-requires a list, flip the env to ``zeros``
    (or wire ``real``). ``tests/test_grpo_logprobs_probe.py`` is the vehicle
    and documents the loss-invariance claim that only the GPU box can prove.
    """
    mode = os.environ.get("PHASE2_ROLLOUT_LOGPROBS", "none").lower()
    if mode == "none":
        return None
    if mode == "zeros":
        return [[0.0] * len(c) for c in completion_ids]
    if mode == "real":
        raise NotImplementedError(
            "PHASE2_ROLLOUT_LOGPROBS=real needs on-policy generation-time "
            "per-token logprobs aligned to the re-tokenized completion "
            "(on-box; phase-2 plan P0.1/P0.2)."
        )
    raise ValueError(
        f"unknown PHASE2_ROLLOUT_LOGPROBS={mode!r} (none|zeros|real)"
    )


def make_toy_rollout_func(
    tokenizer: Any,
    *,
    trainable_policy_fn: PolicyFn,
    expected_k: int,
    num_rounds: int = NUM_ROUNDS,
    num_players: int = NUM_PLAYERS,
    system_prompt: str = SYSTEM_PROMPT,
    stub_strategy: str | None = None,
    stats: dict[str, Any] | None = None,
) -> Callable[..., dict[str, list]]:
    """TRL ``rollout_func`` for the turn-as-prompt dataset.

    ``stats`` (optional, mutated in place): per-call rollout telemetry —
    ``{"calls": int, "series": [aggregate_behavior(...), ...]}``. This is the
    only place that sees the policy's *actual training-time* behaviour
    (realized shade / parse-legal-default rates per generation batch), so a
    reviewer can tell on-policy drift toward the oracle plateau from a frozen
    or off-the-rails policy. Accumulation is best-effort: a stats failure must
    never perturb the verified rollout contract.

    TRL's ``RepeatSampler`` (utils.py:738) repeats each unique dataset row
    ``num_generations`` times; ``build_toy_trainer`` sizes the config so one
    generation batch is the **entire** turn dataset (every ``(seed,k,round)``
    once, then ×G consecutive duplicates). This rollout therefore:

    1. rolls each distinct ``(seed,k)`` game **once** (cached in this call);
    2. calls ``flatten_training_rows`` **ONCE over all games together** — so
       ``compute_advantages`` sees all K rollouts of a seed and applies true
       within-K-group standardization on ``(seed, value_chart)`` (calling it
       per game would make each game its own group-of-1 → wrong advantages,
       the regression this revision fixes);
    3. maps rows back by ``(seed, k, round)`` via the row's ``game_id`` +
       ``round_index`` and emits exactly one row per incoming prompt, in
       order — the verified ``len(out)==len(prompts)`` + positional-alignment
       contract. The intrinsic ×G duplicates (TRL requires ``num_generations
       >= 2``) all map to the same row → identical correct precomputed values.

    Enforces the design invariant: the batch must contain *complete*
    K-groups (every seed present with all ``expected_k`` rollouts × all
    ``num_rounds`` rounds). A partial group ⇒ a hard error, not a silently
    miscomputed advantage.
    """

    def _ids(text: str) -> list[int]:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids or [tokenizer.eos_token_id or 0]

    def rollout_func(*args: Any, **kwargs: Any) -> dict[str, list]:
        prompts = kwargs.get("prompts")
        if prompts is None and args:
            prompts = args[0]
        if prompts is None:
            raise RuntimeError(
                f"rollout_func could not find 'prompts' (args={len(args)}, "
                f"kwargs={list(kwargs)})"
            )
        parsed = [parse_turn_prompt(p if isinstance(p, str) else str(p))
                  for p in prompts]

        # Distinct (seed,k,chart) games, in first-seen order → games[idx];
        # build_training_rows stamps row["game_id"] == that index.
        order: list[tuple[int, int, str]] = []
        pos: dict[tuple[int, int, str], int] = {}
        present: dict[int, set[tuple[int, int]]] = {}
        for seed, ki, rnd, chart in parsed:
            key = (seed, ki, chart)
            if key not in pos:
                pos[key] = len(order)
                order.append(key)
            present.setdefault(seed, set()).add((ki, rnd))

        # Invariant: complete K-groups only (else K-group std is wrong).
        for seed, ks in present.items():
            want = {(ki, r) for ki in range(expected_k)
                    for r in range(num_rounds)}
            if ks != want:
                raise RuntimeError(
                    f"incomplete K-group for seed {seed}: have {len(ks)} of "
                    f"{len(want)} (seed,k,round) cells. The generation batch "
                    f"must equal the whole turn dataset — keep "
                    f"per_device_train_batch_size = B·num_generations "
                    f"(build_toy_trainer enforces this; do not override it)."
                )

        games: list[dict[str, Any]] = []
        for seed, ki, chart in order:
            trainable = (
                make_stub_policy(stub_strategy, jitter_seed=ki)
                if stub_strategy is not None else trainable_policy_fn
            )
            games.append(rollout_group(
                seed, trainable_policy_fn=trainable, k=1, value_chart=chart,
                num_rounds=num_rounds, num_players=num_players,
            )[0])

        # Best-effort training-time behaviour telemetry (never perturbs the
        # contract: separate try, no effect on the returned dict).
        if stats is not None:
            try:
                stats["calls"] = stats.get("calls", 0) + 1
                rec = aggregate_behavior(games)
                rec["call"] = stats["calls"]
                stats.setdefault("series", []).append(rec)
            except Exception:  # noqa: BLE001 - telemetry must not break rollout
                stats.setdefault("errors", 0)
                stats["errors"] += 1

        # ONE flatten over ALL games → true K-group advantages.
        rows = flatten_training_rows(games)
        by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
        for r in rows:
            seed, ki, _chart = order[r["game_id"]]
            by_key[(seed, ki, r["round_index"])] = r

        prompt_ids, completion_ids = [], []
        prec_r, prec_a = [], []
        for (seed, ki, rnd, _chart), src in zip(parsed, prompts):
            try:
                row = by_key[(seed, ki, rnd)]
            except KeyError:  # pragma: no cover - guarded by the invariant
                raise RuntimeError(
                    f"no trainable row for prompt {src!r} "
                    f"(seed={seed}, k={ki}, round={rnd})."
                )
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["prompt"]},
            ]
            ptext = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            cids = _ids(row["completion"])
            prompt_ids.append(_ids(ptext))
            completion_ids.append(cids)
            prec_r.append(float(row["precomputed_reward"]))
            prec_a.append(float(row["precomputed_advantage"]))

        assert len(prompt_ids) == len(prompts), (
            f"rollout returned {len(prompt_ids)} rows for {len(prompts)} "
            "prompts — turn-as-prompt invariant broken."
        )
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": rollout_logprobs(completion_ids),  # P0.2: not fabricated
            "precomputed_reward": prec_r,
            "precomputed_advantage": prec_a,
        }

    return rollout_func


def megagem_lora_config():
    """LoRA config mirroring configs/sft/megagem.toml (rank 32 / alpha 64 /
    dropout 0.05 / the 7 proj modules). Phase 2/3 train PEFT/LoRA per
    the trainer-choice notes (item 5; with PEFT the TRL ref model is None and
    KL comes from adapter-peeling). `peft_config=None` = full-FT 4B: fp32
    Adam on all params + a separate ref model → OOM, and it would validate a
    trainer we never ship. Lazy peft import — only the GPU path calls this.
    """
    from peft import LoraConfig

    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )


def build_toy_trainer(
    *,
    model_id: str,
    seeds: Sequence[int],
    k: int,
    trainable_policy_fn: PolicyFn | None,
    num_rounds: int = NUM_ROUNDS,
    max_completion_length: int = 256,
    max_prompt_length: int = 512,
    learning_rate: float = 1e-5,
    kl_beta: float = 0.05,
    temperature: float = 1.0,
    top_p: float = 0.95,
    grpo_num_generations: int = 2,
    max_steps: int = 50,
    seed: int = 0,
    output_dir: str,
    stub_strategy: str | None = None,
    on_policy: bool = False,
):
    """Construct a real ``MegaGemGRPOTrainer`` over the turn-as-prompt dataset.

    Sized so **one generation batch == the entire turn dataset**, which is
    required for correct K-group advantages (the rollout must see all K
    rollouts of every seed together — see ``make_toy_rollout_func``):

    * ``B = len(seeds)·k·num_rounds`` unique turn-prompts;
    * ``G = grpo_num_generations`` (TRL forbids ``< 2``; the resulting
      per-group std is **discarded** by the subclass, which substitutes our
      offline advantage — so G is pure overhead, default 2);
    * ``per_device_train_batch_size = B·G``, ``steps_per_generation=1`` ⇒
      ``generation_batch_size = B·G``; RepeatSampler ``batch_size = B·G//G =
      B`` ⇒ one chunk = all B unique rows, each ×G consecutively, in a
      single rollout call. The ×G duplicates carry identical correct
      precomputed advantages (harmless redundancy, ≈ G× the per-step compute
      — keep B·G modest; it is the optimizer batch).

    ``use_vllm`` is hard-False (external vLLM ≠ TRL-internal). Kwargs are
    ``_filter_kwargs``-guarded for GRPOConfig/GRPOTrainer drift. Returns
    ``(trainer, tokenizer, info)``.

    P0.1: ``on_policy=True`` ⇒ rollouts come from the trainer's OWN updating
    weights (HF sampled generation, bound through a deferred box *after*
    construction, since ``trainer.model`` does not exist until then and
    ``rollout_func`` is a constructor arg). This is the only genuinely
    on-policy mode; ``stub_strategy`` (fixed) and an external vLLM
    ``trainable_policy_fn`` (frozen, never resynced) are off-policy.
    """
    import torch
    from dataclasses import asdict, fields, is_dataclass
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer  # noqa: F401 (import-time §5#6 check)

    from megagem.training.megagem_grpo import (
        MegaGemGRPOTrainer,
        precomputed_reward_func,
    )

    if stub_strategy is None and trainable_policy_fn is None and not on_policy:
        raise ValueError(
            "pass trainable_policy_fn, stub_strategy, or on_policy=True"
        )

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    )
    # LoRA + gradient checkpointing footgun: with a PRE-instantiated model +
    # peft_config, TRL's PEFT path skips enable_input_require_grads, so the
    # checkpoint segments cannot drop/recompute activations (and LoRA grads go
    # None) — the documented grad_checkpoint_none_warning failure. Enable it
    # explicitly here, on the base, before TRL wraps it. Idempotent with
    # GRPOConfig.gradient_checkpointing.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    if grpo_num_generations < 2:
        raise ValueError("TRL forbids num_generations < 2 (grpo_config.py:948)")
    prompts = toy_turn_dataset_prompts(seeds, k, num_rounds)
    dataset = Dataset.from_dict({"prompt": prompts})
    B = len(prompts)                       # unique turn-prompts (seeds·k·rounds)
    G = grpo_num_generations
    gen_batch = B * G                      # one generation == whole dataset ×G

    # The LM-head logits tensor is per_device_batch · seq · vocab. At
    # per_device=B·G · ~768 · 151936 that is ~28 GiB bf16 (~84 GiB once the
    # loss/backward upcast) — un-fittable, and IMMUNE to LoRA / gradient
    # checkpointing (those only touch decoder-layer activations). So
    # microbatch the optimizer: keep the SCORING/generation batch = B·G (TRL
    # derives generation_batch_size = per_device · steps_per_generation =
    # micro · spg = B·G, leaving K-group advantage standardization unchanged)
    # but run forward/backward in `micro`-sized chunks. micro is a multiple of
    # G and a divisor of gen_batch so the RepeatSampler grouping stays exact.
    # Env-overridable so the rollout-refresh cadence (spg) can be swept without
    # a recompile. The 2.2-gate FAIL (2026-05-17, paired CI [-0.65,-0.35]) is
    # hypothesised to be off-policy replay: at cap=8, spg=16 holds the rollout
    # cache static for ~47/50 steps against a never-resynced frozen behaviour
    # policy. Larger micro ⇒ fresher rollouts. For B·G=128: cap 8→spg16 (~3
    # fresh/50, current), 16→spg8 (~6), 32→spg4 (~13), 64→spg2 (~25), 128→spg1
    # (50, but OOM — ~84 GiB fp32-upcast logits). Logits ≈ micro·768·151936·2 B
    # (micro=32 ≈ 7.5 GiB bf16, fits the ~110 GiB free with vLLM at 0.2). See
    # the RL plan v5.5 (toy GRPO must use LoRA).
    _MICRO_CAP = int(os.environ.get("PHASE2_MICRO_CAP", "8"))
    micro = G
    while micro * 2 <= _MICRO_CAP and gen_batch % (micro * 2) == 0:
        micro *= 2
    spg = gen_batch // micro
    assert micro * spg == gen_batch, (micro, spg, gen_batch)

    # Deferred policy box (P0.1). For the stub path make_toy_rollout_func
    # ignores this (it builds the stub itself). For the external-vLLM path the
    # box already holds the policy. For on_policy the box is filled AFTER the
    # trainer exists (trainer.model is needed and rollout_func is a constructor
    # arg) — rollout_func is only ever invoked during trainer.train(), by which
    # point it is bound.
    _policy_box: dict[str, Any] = {"fn": trainable_policy_fn}

    def _deferred_policy(prompt: str) -> str:
        fn = _policy_box["fn"]
        if fn is None:
            raise RuntimeError(
                "on-policy rollout invoked before trainer.model was bound "
                "(build_toy_trainer ordering bug)."
            )
        return fn(prompt)

    # Mutated in place by rollout_func during trainer.train(); the SAME dict
    # object is stored in `info` below, so the result JSON (serialized AFTER
    # train) reflects the populated trajectory.
    roll_stats: dict[str, Any] = {"calls": 0, "series": []}
    rollout_func = make_toy_rollout_func(
        tokenizer,
        trainable_policy_fn=_deferred_policy,
        expected_k=k,
        num_rounds=num_rounds,
        stub_strategy=stub_strategy,
        stats=roll_stats,
    )

    cfg = GRPOConfig(**_filter_kwargs(GRPOConfig, dict(
        output_dir=output_dir,
        per_device_train_batch_size=micro,  # microbatch; gen batch = B·G via spg
        num_generations=G,                 # >=2; TRL per-group std discarded
        # generation_batch_size is TRL-derived (= per_device_train_batch_size
        # × steps_per_generation = B·G). The pinned TRL (grpo_config.py
        # __post_init__) REJECTS passing it together with steps_per_generation
        # ("can not be both configured") — do not re-add it.
        steps_per_generation=spg,          # micro·spg = B·G (K-group scoring
                                           # batch preserved); fewer fresh
                                           # rollouts/50 steps — see the RL plan note
        num_iterations=1,
        gradient_accumulation_steps=1,
        max_completion_length=max_completion_length,
        max_prompt_length=max_prompt_length,
        max_steps=max_steps,
        learning_rate=learning_rate,
        beta=kl_beta,                      # per-token KL coefficient
        temperature=temperature,
        top_p=top_p,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=True,       # activation memory: 128-seq batch
        gradient_checkpointing_kwargs={"use_reentrant": False},  # PEFT-correct
        use_vllm=False,                    # external vLLM ≠ TRL-internal vLLM
        bf16=True,
        fp16=False,
        seed=seed,
    )))

    trainer = MegaGemGRPOTrainer(**_filter_kwargs(GRPOTrainer.__init__, dict(
        model=model,
        reward_funcs=[precomputed_reward_func],
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_func=rollout_func,
        peft_config=megagem_lora_config(),
    )))

    if on_policy:
        # P0.1 — bind the rollout to the trainer's OWN (PEFT-wrapped, updating)
        # weights now that trainer.model exists. Sampling (not greedy) so the K
        # rollouts of a seed differ → non-degenerate within-group advantage.
        # make_hf_generate_policy_fn is the eval path: it toggles grad-ckpt
        # off→on and restores use_cache in `finally`. Inside a training step
        # TRL calls rollout_func (→ this, under no_grad) BEFORE its own grad
        # forward, and the `finally` restores the training config, so TRL's
        # loss still runs checkpointed. Idempotent; ~one toggle per bid turn.
        _policy_box["fn"] = make_hf_generate_policy_fn(
            trainer.model, tokenizer, do_sample=True,
            temperature=temperature, top_p=top_p,
        )

    _cfg_names = {f.name for f in fields(GRPOConfig)} if is_dataclass(GRPOConfig) else set()
    rollout_mode = ("on_policy" if on_policy
                    else "stub" if stub_strategy is not None
                    else "external_vllm_frozen")
    info = {
        "model_id": model_id,
        "B_unique_turn_prompts": B,
        "grpo_num_generations": G,
        # P2.1 — these were ONE mislabeled field
        # (`per_device_train_batch_size` was set to gen_batch=128, i.e. the
        # generation batch, NOT the configured forward microbatch). Report
        # the microbatch / scoring batch / cadence honestly and separately so
        # a cadence sweep is interpretable from the JSON alone.
        "per_device_train_batch_size": micro,    # the ACTUAL forward microbatch
        "generation_batch_size": gen_batch,      # = micro·spg = B·G (was "128")
        "steps_per_generation": spg,
        "phase2_micro_cap": _MICRO_CAP,
        "rollout_mode": rollout_mode,
        "on_policy": on_policy,
        "num_seeds": len(seeds),
        "k": k,
        "num_rounds": num_rounds,
        "trl_use_vllm": False,
        "max_steps": max_steps,
        "kl_beta": kl_beta,
        "beta_lever_present": "beta" in _cfg_names,
        "max_completion_length": max_completion_length,
        # P0.2 — the actually-resolved logprobs mode (so a reader knows which
        # path TRL ran, not just the default). P0.1/#3 — on-policy HF rollouts
        # decode with native thinking OFF (mirrors the vLLM surface + the
        # the RL plan Qwen3 lock).
        "rollout_logprobs_mode": os.environ.get(
            "PHASE2_ROLLOUT_LOGPROBS", "none").lower(),
        "hf_rollout_enable_thinking": False,
        # Phase-3 reward-magnitude A/B telemetry (P2.1 discipline — the run
        # must be interpretable from JSON alone: WHICH reward knobs moved).
        # Byte-identical defaults when no PHASE3_* env is set.
        "reward_config": asdict(reward_config_from_env()),
        "reward_config_env_overridden": sorted(
            e for e in ("PHASE3_REWARD_SCALE", "PHASE3_REWARD_WIN_BONUS",
                        "PHASE3_SHAPING_LAMBDA", "PHASE3_ILLEGAL_PENALTY")
            if os.environ.get(e)
        ),
        # Live ref — mutated by rollout_func during train(); serialized after.
        "rollout_stats": roll_stats,
    }
    return trainer, tokenizer, info
