"""Phase 1.3 / 1.5 — RTG advantages, RAE buckets, group-norm, determinism, sanity."""

from __future__ import annotations

import json

import pytest

from megagem.rl.advantage import (
    AdvantageConfig,
    broad_phase,
    compute_advantages,
    summarize,
    tiebreak_rank,
)
from megagem.rl.reward import RewardConfig

from _rl_fixtures import fixture_paths

FIXTURES = fixture_paths()


def _make_game(
    seed: int,
    finals: tuple[int, int, int],
    *,
    rounds: int = 4,
    p0_illegal: bool = False,
    actor_ids: tuple[str, str, str] = ("trainable", "trainable", "trainable"),
) -> dict:
    """Minimal valid schema-v3 game: P0 wins every treasure round.

    Empty value display + constant coins ⇒ zero shaping, so every P0 turn's
    return-to-go equals R_terminal — isolating the terminal-driven sign.
    """
    winner = max(range(3), key=lambda p: finals[p])
    game_rounds = []
    for r in range(rounds):
        players = []
        for pid in range(3):
            is_winner = pid == 0
            rec = {
                "player_id": pid,
                "bid": 1,
                "coins_before": 35,
                "coins_after": 35,
                "collection": [],
                "collection_counts": {},
                "is_winner": is_winner,
                "actor_id": actor_ids[pid],
                "parse_valid": not (pid == 0 and p0_illegal),
                "legal_valid": not (pid == 0 and p0_illegal),
                "prompt": f"P{pid} R{r} prompt",
                "raw_response": f"P{pid} R{r} response",
            }
            if is_winner:
                rec["winning_bid"] = 1
                rec["gem_revealed"] = None
                rec["reveal"] = {
                    "actor_id": actor_ids[pid],
                    "parse_valid": not p0_illegal,
                    "legal_valid": not p0_illegal,
                    "prompt": f"P{pid} R{r} reveal prompt",
                    "raw_response": f"P{pid} R{r} reveal response",
                }
            players.append(rec)
        game_rounds.append(
            {
                "round_number": r + 1,
                "auction": {"type": "treasure", "description": "t"},
                "players": players,
                "value_display": {},
                "tiebreak_order": [0, 1, 2],
                "available_missions": [],
                "missions_completed": {},
            }
        )
    return {
        "metadata": {
            "schema_version": 3,
            "seed": seed,
            "num_players": 3,
            "value_chart": "A",
        },
        "rounds": game_rounds,
        "final_results": {
            "winner_id": winner,
            "num_rounds": rounds,
            "available_missions": [],
            "final_scores": [
                {
                    "player_id": pid,
                    "final_score": finals[pid],
                    "loan_payments": 0,
                    "investment_returns": 0,
                }
                for pid in range(3)
            ],
        },
    }


# ---------------------------------------------------------------- helpers ---


def test_phase_and_rank_helpers() -> None:
    assert broad_phase(0, 9) == 0
    assert broad_phase(4, 9) == 1
    assert broad_phase(8, 9) == 2
    assert broad_phase(0, 0) == 0  # degenerate
    assert tiebreak_rank([2, 0, 1], 2) == 0
    assert tiebreak_rank([2, 0, 1], 1) == 2
    assert tiebreak_rank([2, 0, 1], 9) == 3  # absent → len


# --------------------------------------------------------- 1.5 sanity ---


def test_known_good_vs_known_bad_sign() -> None:
    """Within one GRPO group: a big-win P0 trajectory's mean advantage must be
    positive and exceed a big-loss P0 trajectory's (which must be negative)."""
    good = _make_game(seed=999, finals=(120, 10, 10))  # P0 dominates
    bad = _make_game(seed=999, finals=(5, 120, 120))  # P0 crushed
    rows = compute_advantages([good, bad])

    def p0_mean(game_id: int) -> float:
        vs = [r.advantage for r in rows if r.game_id == game_id and r.player_id == 0]
        assert vs
        return sum(vs) / len(vs)

    good_adv, bad_adv = p0_mean(0), p0_mean(1)
    assert good_adv > 0 > bad_adv, (good_adv, bad_adv)
    assert good_adv > bad_adv


def test_illegal_actions_drag_advantage_down() -> None:
    clean = _make_game(seed=1, finals=(60, 50, 40))
    dirty = _make_game(seed=1, finals=(60, 50, 40), p0_illegal=True)
    rows = compute_advantages([clean, dirty])
    clean_adv = [r.advantage for r in rows if r.game_id == 0 and r.player_id == 0]
    dirty_adv = [r.advantage for r in rows if r.game_id == 1 and r.player_id == 0]
    assert sum(clean_adv) / len(clean_adv) > sum(dirty_adv) / len(dirty_adv)


# --------------------------------------------------------- actor masking ---


def test_actor_masking_zeros_non_trainable() -> None:
    g = _make_game(
        seed=7, finals=(90, 10, 10), actor_ids=("trainable", "snapshot_3", "heuristic")
    )
    rows = compute_advantages([g])
    for r in rows:
        if r.player_id == 0:
            assert r.trainable
        else:
            assert not r.trainable
            assert r.advantage == 0.0


# --------------------------------------------------- buckets / collapse ---


def test_bucket_collapse_when_sparse_else_full() -> None:
    g = _make_game(seed=3, finals=(70, 20, 20), rounds=4)
    collapsed = compute_advantages([g], adv_cfg=AdvantageConfig(min_bucket_count=20))
    arity_c = {len(r.ema_bucket) for r in collapsed if r.trainable}
    assert arity_c == {1}, "sparse corpus must collapse to 3 (rank-only) buckets"

    full = compute_advantages(
        [g], adv_cfg=AdvantageConfig(min_bucket_count=0, collapse_when_sparse=True)
    )
    arity_f = {len(r.ema_bucket) for r in full if r.trainable}
    assert arity_f == {2}, "no sparsity ⇒ keep 9 (rank, phase) buckets"
    # the descriptive (rank, phase) bucket is always recorded
    assert all(len(r.bucket) == 2 for r in full)


# ------------------------------------------------------- group-norm ---


@pytest.mark.skipif(not FIXTURES, reason="no schema-v3 fixtures")
def test_group_standardization_zero_mean_unit_std() -> None:
    data = [json.loads(p.read_text()) for p in FIXTURES]
    rows = compute_advantages(data)
    rep = summarize(rows)
    for gk, stats in rep["groups"].items():
        if stats["n"] >= 2 and stats["std"] > 0:
            assert abs(stats["mean"]) < 1e-6, (gk, stats)
            assert abs(stats["std"] - 1.0) < 1e-6, (gk, stats)


# --------------------------------------------------------- determinism ---


def test_deterministic_repeat_runs_byte_identical() -> None:
    corpus = [
        _make_game(seed=42, finals=(80, 30, 20)),
        _make_game(seed=42, finals=(20, 80, 30)),
        _make_game(seed=43, finals=(50, 50, 50)),
    ]
    a = compute_advantages(corpus)
    b = compute_advantages(corpus)
    sig = lambda rows: [
        (
            r.game_id,
            r.player_id,
            r.round_index,
            r.phase,
            round(r.rtg, 9),
            round(r.baseline, 9),
            round(r.advantage, 9),
        )
        for r in rows
    ]
    assert sig(a) == sig(b)


@pytest.mark.skipif(not FIXTURES, reason="no schema-v3 fixtures")
def test_deterministic_on_real_corpus() -> None:
    data = [json.loads(p.read_text()) for p in FIXTURES]
    s1 = summarize(compute_advantages(data))
    s2 = summarize(compute_advantages(data))
    assert json.dumps(s1, sort_keys=True, default=str) == json.dumps(
        s2, sort_keys=True, default=str
    )


def test_group_norm_mode_reports_code_path_vs_k_group() -> None:
    """The report must be honest about 1-rollout vs true K-group corpora."""
    single = compute_advantages([_make_game(seed=77, finals=(80, 20, 20))])
    s1 = summarize(single)["group_norm"]
    assert s1["rollouts_per_group_max"] == 1
    assert "code-path validation only" in s1["mode"]

    # Two rollouts at the SAME seed → a true K=2 group (shared seed+chart+seat).
    k_corpus = [
        _make_game(seed=77, finals=(80, 20, 20)),
        _make_game(seed=77, finals=(20, 80, 20)),
    ]
    s2 = summarize(compute_advantages(k_corpus))["group_norm"]
    assert s2["rollouts_per_group_min"] == 2
    assert s2["rollouts_per_group_max"] == 2
    assert "true K-group" in s2["mode"]

    # Partial: seed 77 has 2 rollouts but seed 88 only 1 (failed/missing) →
    # headline must NOT claim true K-group.
    partial = [
        _make_game(seed=77, finals=(80, 20, 20)),
        _make_game(seed=77, finals=(20, 80, 20)),
        _make_game(seed=88, finals=(70, 30, 20)),
    ]
    s3 = summarize(compute_advantages(partial))["group_norm"]
    assert s3["rollouts_per_group_min"] == 1
    assert s3["rollouts_per_group_max"] == 2
    assert "partial K-group" in s3["mode"]


def test_rtg_equals_terminal_when_no_shaping() -> None:
    """Empty display + constant coins ⇒ every P0 turn's RTG == R_terminal."""
    g = _make_game(seed=5, finals=(100, 20, 20))
    rows = compute_advantages([g], reward_cfg=RewardConfig(shaping_lambda=0.2))
    term = min(1.0, (100 - 20) / 30.0) + 0.1  # margin 80 → clipped to 1.0, +bonus
    for r in rows:
        if r.player_id == 0:
            assert r.rtg == pytest.approx(term, abs=1e-9)


# ----------------------------------------------- Phase-3 reward-design ---


def test_default_rewardconfig_byte_identical_advantages() -> None:
    """The regression guard: a default RewardConfig() must produce per-turn
    (reward, rtg, baseline, advantage) identical to an all-fields-explicit
    reconstruction at the locked constants — proves no default drifted."""
    from dataclasses import replace

    corpus = [_make_game(seed=s, finals=f)
              for s, f in ((1, (90, 20, 10)), (2, (10, 70, 40)),
                           (1, (30, 30, 90)), (2, (60, 60, 5)))]
    explicit = RewardConfig(
        scale=21.5, win_bonus=0.1, shaping_lambda=0.01, illegal_penalty=-0.5,
        terminal_shape="clip", reveal_shaping_weight=1.0,
        terminal_correction=False, correction_scale=None,
    )
    assert RewardConfig() == explicit  # no default constant moved
    a = compute_advantages(corpus, RewardConfig())
    b = compute_advantages(corpus, replace(explicit))
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        for attr in ("reward", "rtg", "baseline", "advantage"):
            assert getattr(ra, attr) == pytest.approx(getattr(rb, attr), abs=1e-12)


def test_terminal_correction_flows_through_rtg() -> None:
    """rec 1a: enabling the correction shifts every one of a player's turns'
    RTG by exactly that player's single terminal correction C_p (it sits on
    the terminal turn; RTG suffix-sum telescopes it to all earlier turns)."""
    from dataclasses import replace

    g = _make_game(seed=11, finals=(90, 20, 10))
    off = compute_advantages([g], RewardConfig())
    on = compute_advantages(
        [g], replace(RewardConfig(), terminal_correction=True)
    )
    by_pid_off: dict[int, list[float]] = {}
    by_pid_on: dict[int, list[float]] = {}
    for r in off:
        by_pid_off.setdefault(r.player_id, []).append(r.rtg)
    for r in on:
        by_pid_on.setdefault(r.player_id, []).append(r.rtg)
    for pid, offs in by_pid_off.items():
        ons = by_pid_on[pid]
        deltas = [b - a for a, b in zip(offs, ons)]
        # constant shift across all of that player's turns == C_p
        assert max(deltas) == pytest.approx(min(deltas), abs=1e-9)
        assert abs(deltas[0]) > 1e-9  # the correction is actually nonzero here
