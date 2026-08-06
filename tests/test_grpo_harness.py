"""GRPO harness proof (the historical phase-2 gate) — the toy/MegaGem GRPO
glue is correct on CPU *before* any GPU spend (prove what is provable now).

Pure CPU: no torch/trl/GPU/network. Critically, ``test_trl_merge_alignment``
*simulates the installed TRL pin's exact extra-field merge + reward sizing*
(grpo_trainer.py L2128-2147 / L1198) so the rollout-length + per-row advantage
alignment contract is a real proof here, not "the first GPU run will tell us".

The harness under test is ``megagem.training.grpo_harness``.
"""

from __future__ import annotations

import copy
import math

import pytest

import megagem.training.grpo_harness as H

_K = 4
_ROWS_PER_GAME = H.NUM_ROUNDS  # 8 trainable bid turns, trainable seat only


class _FakeTok:
    """Deterministic stand-in for an HF tokenizer (no transformers)."""

    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [min(255, ord(c)) for c in text[:24]] or [0]}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in msgs)


def test_turn_prompt_roundtrip_and_dataset_shape():
    assert H.parse_turn_prompt(H.turn_prompt(4242, 3, 5, "A")) == (4242, 3, 5, "A")
    with pytest.raises(ValueError):
        H.parse_turn_prompt("not a toy-turn prompt")
    seeds = [5000, 5001]
    ds = H.toy_turn_dataset_prompts(seeds, _K)
    assert len(ds) == len(seeds) * _K * H.NUM_ROUNDS  # B = seeds·k·rounds
    # seed-major, then k, then round → a game's rounds are contiguous.
    assert H.parse_turn_prompt(ds[0]) == (5000, 0, 0, "A")
    assert H.parse_turn_prompt(ds[H.NUM_ROUNDS - 1]) == (5000, 0, 7, "A")
    assert H.parse_turn_prompt(ds[H.NUM_ROUNDS]) == (5000, 1, 0, "A")


def test_rollout_group_through_unmodified_pipeline():
    games = H.rollout_group(
        5000, trainable_policy_fn=H.make_stub_policy_fn("bestresp"),
        k=_K, stub_strategy="bestresp")
    assert len(games) == _K
    rows = H.flatten_training_rows(games)
    assert len(rows) == _K * _ROWS_PER_GAME
    assert all(r["actor_id"] == "trainable" and r["player_id"] == 0 for r in rows)
    assert all(
        isinstance(r["precomputed_advantage"], float)
        and math.isfinite(r["precomputed_advantage"])
        and math.isfinite(r["precomputed_reward"]) for r in rows
    )


def _repeatsampler_batch(prompts, g):
    """Exact RepeatSampler order for our config (utils.py:738, one chunk =
    whole dataset, shuffle off): each unique prompt repeated g× consecutively."""
    return [p for p in prompts for _ in range(g)]


def test_rollout_func_one_to_one_under_repeatsampler():
    """Fed the EXACT TRL RepeatSampler generation batch (each turn-prompt ×G),
    the rollout returns exactly len(prompts) rows, positionally aligned on the
    TRL-merged fields. logprobs is NOT in the TRL extra-merge set (cf.
    test_trl_merge_alignment_simulated, which already excludes it) and post-P0.2
    defaults to None — no fabricated per-token values."""
    rf = H.make_toy_rollout_func(_FakeTok(), trainable_policy_fn=lambda _p: "",
                                 expected_k=_K, stub_strategy="bestresp")
    base = H.toy_turn_dataset_prompts([5100, 5101], _K)
    aligned = ("prompt_ids", "completion_ids",
               "precomputed_reward", "precomputed_advantage")
    for g in (2, 3):
        gen = _repeatsampler_batch(base, g)
        out = rf(prompts=gen)
        assert {"prompt_ids", "completion_ids", "logprobs",
                "precomputed_reward", "precomputed_advantage"} <= set(out)
        assert {len(out[k]) for k in aligned} == {len(gen)}
        assert out["logprobs"] is None  # default PHASE2_ROLLOUT_LOGPROBS=none


def test_repeatsampler_dupes_identical_and_true_kgroup():
    """The reviewer's regression: rollout advantages must equal a SINGLE
    build_training_rows over ALL K games (true K-group), and the G duplicates
    of a turn-prompt must be identical. Per-game flatten (the old bug) makes
    every game a group-of-1 → different advantages."""
    seeds = [5100, 5101]
    rf = H.make_toy_rollout_func(_FakeTok(), trainable_policy_fn=lambda _p: "",
                                 expected_k=_K, stub_strategy="bestresp")
    gen = _repeatsampler_batch(H.toy_turn_dataset_prompts(seeds, _K), 2)
    out = rf(prompts=gen)

    # G duplicates identical.
    by_p: dict[str, set] = {}
    for p, a in zip(gen, out["precomputed_advantage"]):
        by_p.setdefault(p, set()).add(round(a, 12))
    assert all(len(s) == 1 for s in by_p.values())

    # Reference == ONE flatten over all games in the rollout's first-seen
    # (seed,k) order; game_id = that index.
    all_games = [
        H.rollout_group(s, trainable_policy_fn=H.make_stub_policy_fn("bestresp", ki),
                        k=1)[0]
        for s in seeds for ki in range(_K)
    ]
    ref = {(r["game_id"], r["round_index"]): r["precomputed_advantage"]
           for r in H.flatten_training_rows(all_games)}
    diffs = 0
    for p, a in zip(gen, out["precomputed_advantage"]):
        s, ki, rnd, _ = H.parse_turn_prompt(p)
        gid = seeds.index(s) * _K + ki
        if abs(a - ref[(gid, rnd)]) > 1e-12:
            diffs += 1
    assert diffs == 0  # was 32/32 under the per-game-flatten bug


def test_incomplete_kgroup_raises():
    """Design invariant: a partial K-group (generation batch ≠ whole turn
    dataset) is a hard error, not a silently miscomputed advantage."""
    rf = H.make_toy_rollout_func(_FakeTok(), trainable_policy_fn=lambda _p: "",
                                 expected_k=_K, stub_strategy="bestresp")
    # Only k=0 present (expected_k=_K) → incomplete.
    partial = [H.turn_prompt(5100, 0, r) for r in range(H.NUM_ROUNDS)]
    with pytest.raises(RuntimeError, match="incomplete K-group"):
        rf(prompts=partial)


def test_trl_merge_alignment_simulated():
    """Replay grpo_trainer.py's exact extra-field merge + reward sizing on the
    rollout output to prove the seam works on the installed pin.

    L1198 ``rewards_per_func = zeros(len(prompts), ...)``;
    L2130 ``for i, inp in enumerate(inputs): inp[k]=values[i] if i<len(v)``;
    MegaGemGRPOTrainer ``[row[ADV] for row in inputs]`` → (B,) vs
    ``output['advantages']`` (B,).
    """
    rf = H.make_toy_rollout_func(_FakeTok(), trainable_policy_fn=lambda _p: "",
                                 expected_k=_K, stub_strategy="bestresp")
    prompts = _repeatsampler_batch(H.toy_turn_dataset_prompts([5200], _K), 2)
    out = rf(prompts=prompts)

    # TRL: inputs has exactly len(prompts) rows; extras merge positionally.
    inputs = [dict() for _ in prompts]
    extra = {k: v for k, v in out.items()
             if k not in ("prompt_ids", "completion_ids", "logprobs")}
    for i, inp in enumerate(inputs):
        for key, vals in extra.items():
            if isinstance(vals, list) and i < len(vals):
                inp[key] = vals[i]
    # Every input row received its precomputed pair (no truncation/misalign).
    assert all("precomputed_advantage" in r and "precomputed_reward" in r
               for r in inputs)
    # MegaGemGRPOTrainer._replacement_advantages would build this (B,):
    adv = [r["precomputed_advantage"] for r in inputs]
    assert len(adv) == len(prompts)            # matches output['advantages'] (B,)
    # rewards_per_func = zeros(len(prompts)); aligned to the same B.
    assert len(out["precomputed_reward"]) == len(prompts)


def test_terminal_reward_is_the_robust_signal():
    for s in (5300, 5367, 6004, 8242):
        good = H.rollout_group(s, trainable_policy_fn=H.make_stub_policy_fn("bestresp"), k=1)[0]
        bad = H.rollout_group(s, trainable_policy_fn=H.make_stub_policy_fn("overbid"), k=1)[0]
        assert H.trainable_terminal_reward(good) > H.trainable_terminal_reward(bad)


def test_paired_bootstrap_gate_logic():
    seeds = list(range(9000, 9020))
    good = {s: H.trainable_terminal_reward(
        H.rollout_group(s, trainable_policy_fn=H.make_stub_policy_fn("bestresp"), k=1)[0])
        for s in seeds}
    bad = {s: H.trainable_terminal_reward(
        H.rollout_group(s, trainable_policy_fn=H.make_stub_policy_fn("overbid"), k=1)[0])
        for s in seeds}
    sep = H.paired_bootstrap_ci([good[s] - bad[s] for s in seeds], seed=0)
    assert sep["ci_low"] > 0.0
    null = H.paired_bootstrap_ci([0.0] * len(seeds), seed=0)
    assert null["ci_low"] <= 0.0 <= null["ci_high"]
    a = H.paired_bootstrap_ci([good[s] - bad[s] for s in seeds], seed=7)
    b = H.paired_bootstrap_ci([good[s] - bad[s] for s in seeds], seed=7)
    assert a == b
    with pytest.raises(ValueError):
        H.paired_bootstrap_ci([], seed=0)


def test_flatten_seat_parameterized():
    """Finding 6: a non-zero trainable seat must work end-to-end (the seat
    check is parameterized, not hard-coded to 0)."""
    heuristic = H.make_heuristic_policy()
    stub = H.make_stub_policy_fn("bestresp")
    from megagem.toy.auction_env import run_one_toy_game
    game = run_one_toy_game(
        6100,
        [heuristic, stub, heuristic],
        ["heuristic", "trainable", "heuristic"],
    )
    rows = H.flatten_training_rows([game], trainable_seat=1)
    assert rows and all(r["player_id"] == 1 and r["actor_id"] == "trainable"
                        for r in rows)
    with pytest.raises(AssertionError, match="trainable_seat"):
        H.flatten_training_rows([game], trainable_seat=0)  # wrong seat → caught


def test_bestresp_is_genuinely_profitable_p0_3():
    """P0.3: the corrected ``bestresp`` (0.70·v — a point on the 0.65–0.70
    plateau) is a *positive*-reward policy on this fixed 20-seed set (the old
    0.85·v one was net-LOSING here, making "known-good" a misnomer);
    ``aboveshade`` preserves that old, worse behaviour under an honest name so
    prior cadence-sweep runs stay reproducible. The 20-seed set is pinned
    deliberately (deterministic): the plateau argmax is seed-set dependent
    (0.70 here, 0.65 on 9000–9039) — see test_oracle_shade_sweep_targets."""
    seeds = list(range(9000, 9020))

    def _mean(strategy):
        return sum(
            H.trainable_terminal_reward(H.rollout_group(
                s, trainable_policy_fn=H.make_stub_policy_fn(strategy), k=1)[0])
            for s in seeds
        ) / len(seeds)

    bestresp = _mean("bestresp")
    aboveshade = _mean("aboveshade")
    assert bestresp >= 0.12, f"bestresp not genuinely profitable: {bestresp:.4f}"
    assert aboveshade < bestresp, (aboveshade, bestresp)
    assert aboveshade <= 0.0, (
        f"aboveshade (old 'bestresp', 0.85·v) should be net-losing, "
        f"got {aboveshade:.4f}"
    )


def test_oracle_shade_sweep_targets_070_p1_2():
    """P1.2: the explicit learning target is the 0.65–0.70·v plateau (NOT the
    opponents' 0.8, and emphatically not the old 0.85). argmax is seed-set
    dependent — 0.70 on this pinned 20-seed set, 0.65 on 9000–9039 — so the
    gate consumer must read argmax from the ACTIVE eval set, never hard-code
    0.70. Here we pin 20 seeds (deterministic) to lock the plateau shape."""
    o = H.oracle_shade_sweep(list(range(9000, 9020)))
    by = {r["shade"]: r["mean_terminal_reward"] for r in o["table"]}
    assert o["argmax_shade"] == 0.70                 # seed-set specific
    assert o["argmax_mean_reward"] >= 0.12
    assert by[0.65] > 0 and by[0.70] > 0             # both on the plateau
    assert by[0.70] > by[0.80] > by[1.0]
    assert by[0.85] <= 0.0                           # old mislabeled "bestresp"
    # argmax is derived from the active eval set, not hard-coded.
    o40 = H.oracle_shade_sweep(list(range(9000, 9040)))
    assert o40["argmax_shade"] == 0.65


def test_spearman_p2_2():
    assert H._spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert H._spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert H._spearman([1, 2, 3], [5, 5, 5]) is None      # zero rank-variance
    assert H._spearman([1.0], [2.0]) is None              # <2 points
    # d=[-1,1,-1,1,0] ⇒ Σd²=4 ⇒ ρ = 1 − 6·4/(5·24) = 0.8
    assert H._spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)


def test_advantage_reward_rank_corr_p2_2():
    """A stochastic policy → non-degenerate K-groups → mean advantage tracks
    terminal reward (ρ>0). A deterministic policy → degenerate K-groups → the
    helper honestly reports 'no usable groups' (None) instead of fabricating."""
    import random

    def noisy_policy():
        rng = random.Random(0)

        def p(prompt):
            d = H._read_toy_prompt(prompt)
            b = max(0, min(d["coins"],
                           round(0.70 * d["private_value"]) + rng.randint(-3, 3)))
            return f'{{"bid": {b}}}'
        return p

    good = H.advantage_reward_rank_corr(
        [5500, 5501, 5502], k=4, policy_fn=noisy_policy())
    assert good["n_groups_used"] >= 1
    assert good["mean_spearman"] is not None and good["mean_spearman"] > 0.0

    degen = H.advantage_reward_rank_corr(
        [5600], k=4, policy_fn=H.make_stub_policy_fn("bestresp"))
    assert degen["n_groups_used"] == 0          # all rollouts identical
    assert degen["mean_spearman"] is None        # not fabricated


def test_reward_config_from_env_threads_terminal_correction(monkeypatch):
    """Step 1 of Phase-3 wiring: the decided first-pass cut is
    "1a ON + locked baseline". `reward_config_from_env` must thread
    `PHASE3_TERMINAL_CORRECTION` → `RewardConfig.terminal_correction`, leave
    the no-env path byte-identical, and not disturb the existing float knobs.
    """
    for v in ("PHASE3_TERMINAL_CORRECTION", "PHASE3_SHAPING_LAMBDA"):
        monkeypatch.delenv(v, raising=False)

    # no env ⇒ byte-identical locked baseline (1a OFF)
    base = H.reward_config_from_env()
    assert base == H.RewardConfig()
    assert base.terminal_correction is False

    # the decided first-pass config: 1a ON, everything else at the lock
    monkeypatch.setenv("PHASE3_TERMINAL_CORRECTION", "1")
    on = H.reward_config_from_env()
    assert on.terminal_correction is True
    assert on == H.RewardConfig(terminal_correction=True)  # nothing else moved

    # falsy spellings ⇒ explicitly False (not "any non-empty string is True")
    for falsy in ("0", "false", "no", "off"):
        monkeypatch.setenv("PHASE3_TERMINAL_CORRECTION", falsy)
        assert H.reward_config_from_env().terminal_correction is False

    # composes with an existing float knob; bool field stays bool
    monkeypatch.setenv("PHASE3_TERMINAL_CORRECTION", "true")
    monkeypatch.setenv("PHASE3_SHAPING_LAMBDA", "0.02")
    both = H.reward_config_from_env()
    assert both.terminal_correction is True and both.shaping_lambda == 0.02


def test_degenerate_kgroup_guard_p1_4(monkeypatch):
    """P1.4: default OFF (env unset/0) ⇒ verified path byte-identical;
    eps≥reward-range ⇒ every K-group is zeroed; single-rollout groups (k=1)
    are never guarded (protects the eval/k=1 path)."""
    jittered = [
        H.rollout_group(5400,
                        trainable_policy_fn=H.make_stub_policy_fn("bestresp", ki),
                        k=1)[0]
        for ki in range(4)
    ]

    monkeypatch.delenv("PHASE2_DEGENERATE_KGROUP_EPS", raising=False)
    rows_off = H.flatten_training_rows(jittered)
    assert any(r["precomputed_advantage"] != 0.0 for r in rows_off)

    monkeypatch.setenv("PHASE2_DEGENERATE_KGROUP_EPS", "10.0")
    rows_on = H.flatten_training_rows(jittered)
    assert all(r["precomputed_advantage"] == 0.0 for r in rows_on)

    # k=1 (single-rollout group) must be immune even with a huge eps.
    one = H.rollout_group(5400, trainable_policy_fn=H.make_stub_policy_fn("bestresp"), k=1)
    rows_one = H.flatten_training_rows(one)
    assert len(rows_one) == H.NUM_ROUNDS  # guarded path didn't drop/zero-fail


def test_game_behavior_stats_and_aggregate_logs():
    """Tier-A shade/quality telemetry: a clean stub is parse+legal on every
    turn with a sane shade; `garbage` is all-default with no shade; `overbid`
    is legal but shades ≫1. Discriminative, schema-tolerant."""
    good = H.rollout_group(
        5700, trainable_policy_fn=H.make_stub_policy_fn("bestresp"), k=1)[0]
    g = H.game_behavior_stats(good)
    assert g["n_turns"] == H.NUM_ROUNDS
    assert g["parse_valid_frac"] == 1.0 and g["legal_valid_frac"] == 1.0
    assert g["default_used_frac"] == 0.0
    assert g["shade_n"] == g["n_turns"] - g["pv_zero_turns"]
    assert g["shade_mean"] is not None and 0.4 < g["shade_mean"] < 1.1

    garbage = H.rollout_group(
        5700, trainable_policy_fn=H.make_stub_policy_fn("garbage"), k=1)[0]
    gb = H.game_behavior_stats(garbage)
    assert gb["parse_valid_frac"] == 0.0 and gb["default_used_frac"] == 1.0
    assert gb["shade_mean"] is None and gb["shade_n"] == 0

    # `overbid` is parse+legal every turn (bid=coins) but self-bankrupts, so
    # its realized shade is seed-dependent (→0 once broke) — assert only the
    # robust, deterministic part.
    over = H.rollout_group(
        5700, trainable_policy_fn=H.make_stub_policy_fn("overbid"), k=1)[0]
    ob = H.game_behavior_stats(over)
    assert ob["parse_valid_frac"] == 1.0 and ob["default_used_frac"] == 0.0
    assert ob["shade_mean"] is not None  # defined (legal), value seed-dependent

    agg = H.aggregate_behavior([good, good])
    assert agg["n_games"] == 2 and agg["shade_mean"] == pytest.approx(
        g["shade_mean"])


def test_summarize_series_logs():
    assert H.summarize_series([])["n"] == 0
    s = H.summarize_series([1.0, 2.0, 5.0, 3.0])
    assert (s["first"], s["last"], s["max"], s["min"]) == (1.0, 3.0, 5.0, 1.0)
    assert s["step_of_max"] == 2 and s["mean"] == pytest.approx(2.75)


def test_rollout_stats_accumulates_logs():
    """The stats accumulator is mutated in place, never perturbs the verified
    rollout contract, and records realized behaviour per generation batch."""
    stats: dict = {"calls": 0, "series": []}
    rf = H.make_toy_rollout_func(_FakeTok(), trainable_policy_fn=lambda _p: "",
                                 expected_k=_K, stub_strategy="bestresp",
                                 stats=stats)
    gen = _repeatsampler_batch(H.toy_turn_dataset_prompts([5100, 5101], _K), 2)
    out = rf(prompts=gen)
    assert len(out["precomputed_advantage"]) == len(gen)   # contract intact
    assert stats["calls"] == 1 and len(stats["series"]) == 1
    rec = stats["series"][0]
    assert rec["call"] == 1 and rec["n_games"] == 2 * _K
    assert rec["shade_mean"] is not None and rec["parse_valid_frac"] == 1.0
    rf(prompts=gen)
    assert stats["calls"] == 2 and len(stats["series"]) == 2


def test_rank_corr_kgroup_sigma_logs():
    import random

    def noisy():
        rng = random.Random(0)

        def p(prompt):
            d = H._read_toy_prompt(prompt)
            b = max(0, min(d["coins"],
                           round(0.70 * d["private_value"]) + rng.randint(-3, 3)))
            return f'{{"bid": {b}}}'
        return p

    good = H.advantage_reward_rank_corr([5500, 5501], k=4, policy_fn=noisy())
    sig = good["kgroup_termreward_sigma"]
    assert set(sig) >= {"median", "p10", "min", "frac_below_eps", "eps"}
    assert sig["median"] is not None and sig["median"] > 0.0   # rollouts differ
    assert all("termreward_sigma" in g for g in good["per_group"])

    # A deterministic policy → identical K rollouts → identical terminal
    # reward → outcome-degenerate K-group (the P1.4 hypothesis), regardless of
    # the order-dependent EMA on advantages.
    degen = H.advantage_reward_rank_corr(
        [5600], k=4, policy_fn=H.make_stub_policy_fn("bestresp"))
    assert degen["kgroup_termreward_sigma"]["frac_below_eps"] == 1.0


def test_phase23_post_tag_includes_nested_reveal():
    """Finding 5: the post-tag must stamp players[*].reveal.actor_id too."""
    import megagem_steps as MG

    g = copy.deepcopy(H.rollout_group(
        7000, trainable_policy_fn=H.make_stub_policy_fn("bestresp"), k=1)[0])
    for rnd in g["rounds"]:
        for rec in rnd["players"]:
            rec.pop("actor_id", None)
    g["rounds"][0]["players"][1]["reveal"] = {"actor_id": "STALE"}

    tagged = MG._post_tag_actor_mask(g, trainable_seat=0, opponent_actor_id="heuristic")
    assert tagged["rounds"][0]["players"][1]["reveal"]["actor_id"] == "heuristic"
    assert tagged["rounds"][0]["players"][0]["actor_id"] == "trainable"
    rows = H.flatten_training_rows([tagged], trainable_seat=0)
    assert len(rows) == H.NUM_ROUNDS
    assert all(r["actor_id"] == "trainable" and r["player_id"] == 0 for r in rows)
