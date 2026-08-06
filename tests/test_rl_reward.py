"""Phase 1.2 — terminal (§A.0) + legal-action (§A.2) rewards + SCALE calibration."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from megagem.rl.reward import (
    RewardConfig,
    assert_corpus_actor_ids,
    calibrate_scale,
    per_turn_rewards,
    round_shaping,
    round_shaping_components,
    terminal_breakdowns,
)
from megagem.rl.scorer import parse_transcript

from _rl_fixtures import REPO_ROOT, fixture_paths, load as _load

FIXTURES = fixture_paths()
pytestmark = pytest.mark.skipif(not FIXTURES, reason="no schema-v3 fixtures")


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_terminal_reward_matches_hand_computed_A0(path: Path) -> None:
    cfg = RewardConfig(scale=30.0, win_bonus=0.1)
    parsed = parse_transcript(_load(path))
    scores = {s["player_id"]: s["final_score"] for s in parsed.final_scores}
    bds = terminal_breakdowns(parsed, cfg)

    for pid, fs in scores.items():
        others = [v for q, v in scores.items() if q != pid]
        margin = fs - statistics.fmean(others)
        expected = _clip(margin / 30.0, -1.0, 1.0) + (
            0.1 if pid == parsed.winner_id else 0.0
        )
        assert bds[pid].margin == pytest.approx(margin, abs=1e-9)
        assert bds[pid].r_terminal == pytest.approx(expected, abs=1e-9)
        assert -1.0 <= bds[pid].r_terminal <= 1.0 + cfg.win_bonus + 1e-9
    # exactly one rank-1, and it is the recorded winner
    rank1 = [b.player_id for b in bds.values() if b.rank == 1]
    assert rank1 == [parsed.winner_id]


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_terminal_uses_engine_final_not_midgame(path: Path) -> None:
    """§A.0 must read final_results.final_scores (post-reveal), not the scorer.

    The mid-game scorer is biased low by exactly the remaining-hand reveal, so
    margins built from it would differ; this pins that A.0 used the engine
    numbers.
    """
    parsed = parse_transcript(_load(path))
    bds = terminal_breakdowns(parsed, RewardConfig())
    for s in parsed.final_scores:
        assert bds[s["player_id"]].final_score == s["final_score"]


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_legal_reward_zero_or_minus_half_bids_and_reveals(path: Path) -> None:
    cfg = RewardConfig(illegal_penalty=-0.5)
    parsed = parse_transcript(_load(path))
    seen_bid = seen_reveal = False
    for t in per_turn_rewards(parsed, cfg):
        if t.parse_valid and t.legal_valid:
            assert t.legal == 0.0
        else:
            assert t.legal == -0.5
        seen_bid |= t.phase == "bid"
        seen_reveal |= t.phase == "reveal"
    assert seen_bid and seen_reveal, "must score both bid and reveal turns"


@pytest.mark.parametrize("path", FIXTURES[:3], ids=lambda p: p.name)
def test_round_shaping_lambda_scaled_and_zeroed(path: Path) -> None:
    cfg = RewardConfig(shaping_lambda=0.2)
    parsed = parse_transcript(_load(path))
    raw = round_shaping(parsed, cfg)
    raw0 = round_shaping(parsed, RewardConfig(shaping_lambda=0.0))
    assert all(v == 0.0 for v in raw0.values()), "λ=0 ⇒ no shaping"
    # λ scales linearly (the *total* round shaping is the Caveat-1 quantity)
    raw2 = round_shaping(parsed, RewardConfig(shaping_lambda=0.4))
    for k in raw:
        if abs(raw[k]) > 1e-9:
            assert raw2[k] == pytest.approx(2.0 * raw[k], rel=1e-9)


@pytest.mark.parametrize("path", FIXTURES[:3], ids=lambda p: p.name)
def test_bid_reveal_split_preserves_round_total(path: Path) -> None:
    """Caveat 2: the bid/reveal split must re-attribute, never re-weight.

    For every (round, player) the sum of that player's bid + reveal turn
    shaping must equal the old single round-shaping total — this is what
    keeps Caveat 1's locked λ untouched.
    """
    cfg = RewardConfig(shaping_lambda=0.2)
    parsed = parse_transcript(_load(path))
    total = round_shaping(parsed, cfg)
    bid_c, rev_c = round_shaping_components(parsed, cfg)
    for k in total:
        assert bid_c[k] + rev_c[k] == pytest.approx(total[k], abs=1e-9)

    per_player_round: dict[tuple[int, int], float] = {}
    for t in per_turn_rewards(parsed, cfg):
        per_player_round[(t.round_index, t.player_id)] = (
            per_player_round.get((t.round_index, t.player_id), 0.0) + t.shaping
        )
    for k, v in total.items():
        assert per_player_round.get(k, 0.0) == pytest.approx(v, abs=1e-9)


@pytest.mark.parametrize("path", FIXTURES[:3], ids=lambda p: p.name)
def test_reveals_now_carry_dense_shaping(path: Path) -> None:
    """Caveat 2 is not a no-op: real reveals that revalue the display must
    produce a nonzero reveal-turn shaping somewhere in the corpus."""
    cfg = RewardConfig(shaping_lambda=0.2)
    parsed = parse_transcript(_load(path))
    reveal_shapings = [
        t.shaping for t in per_turn_rewards(parsed, cfg) if t.phase == "reveal"
    ]
    assert reveal_shapings, "fixture has no reveal turns"
    assert any(abs(s) > 1e-9 for s in reveal_shapings), (
        "no reveal turn carries shaping — the bid/reveal split is inert"
    )


def test_reveal_subscore_exact_decomposition() -> None:
    """Caveat 2 — hand-computed reveal sub-delta on a controlled 1-round game.

    P0 wins, reveals Red (chart A: count 1 ⇒ value_per_gem 4). Holdings:
    P0=2 Red, P1=1 Red, P2=0 Red; equal coins ⇒ the auction (bid) sub-delta
    is identical for everyone (competitive bid shaping = 0), isolating the
    reveal. Pre-round-1 display is empty, so the reveal sub-delta is exactly
    the gem value the reveal switches on:

        reveal_sub = {P0: +8, P1: +4, P2: 0}
        λ=1 competitive ⇒ P0 +6, P1 0, P2 −6

    P0 (winner) has a reveal turn → its reveal shaping is +6, its bid 0.
    P2/P1 have no reveal turn → their reveal-side shaping folds onto the bid
    turn (nothing lost), so the per-(round,player) totals still telescope.
    """
    rec_common = dict(
        coins_before=35,
        parse_valid=True,
        legal_valid=True,
        reasoning="",
        raw_response="r",
    )
    players = []
    for pid, red in ((0, 2), (1, 1), (2, 0)):
        players.append(
            {
                "player_id": pid,
                "actor_id": "trainable",
                "bid": 1,
                "coins_after": 35,
                "collection": ["Red"] * red,
                "collection_counts": {"Red": red} if red else {},
                "is_winner": pid == 0,
                "prompt": f"P{pid}",
                **rec_common,
            }
        )
    players[0]["winning_bid"] = 1
    players[0]["gem_revealed"] = "Red"
    players[0]["reveal"] = {
        "actor_id": "trainable",
        "parse_valid": True,
        "legal_valid": True,
        "prompt": "P0 reveal",
        "raw_response": "Red",
    }
    game = {
        "metadata": {"schema_version": 3, "seed": 99, "num_players": 3,
                     "value_chart": "A"},
        "rounds": [
            {
                "round_number": 1,
                "auction": {"type": "treasure", "description": "t"},
                "players": players,
                "value_display": {"Red": {"count": 1, "value_per_gem": 4}},
                "tiebreak_order": [0, 1, 2],
                "available_missions": [],
                "missions_completed": {},
            }
        ],
        "final_results": {
            "winner_id": 0,
            "num_rounds": 1,
            "available_missions": [],
            "final_scores": [
                {"player_id": p, "final_score": 35, "loan_payments": 0,
                 "investment_returns": 0}
                for p in range(3)
            ],
        },
    }
    parsed = parse_transcript(game)
    cfg = RewardConfig(shaping_lambda=1.0)
    bid_c, rev_c = round_shaping_components(parsed, cfg)
    assert bid_c[(0, 0)] == pytest.approx(0.0, abs=1e-9)
    assert bid_c[(0, 1)] == pytest.approx(0.0, abs=1e-9)
    assert bid_c[(0, 2)] == pytest.approx(0.0, abs=1e-9)
    assert rev_c[(0, 0)] == pytest.approx(6.0, abs=1e-9)
    assert rev_c[(0, 1)] == pytest.approx(0.0, abs=1e-9)
    assert rev_c[(0, 2)] == pytest.approx(-6.0, abs=1e-9)

    by_turn = {
        (t.round_index, t.player_id, t.phase): t.shaping
        for t in per_turn_rewards(parsed, cfg)
    }
    assert by_turn[(0, 0, "reveal")] == pytest.approx(6.0, abs=1e-9)
    assert by_turn[(0, 0, "bid")] == pytest.approx(0.0, abs=1e-9)
    # P2 has no reveal turn: its reveal-side −6 folds onto the bid turn.
    assert by_turn[(0, 2, "bid")] == pytest.approx(-6.0, abs=1e-9)
    assert (0, 2, "reveal") not in by_turn


def test_calibrate_scale_report() -> None:
    rep = calibrate_scale([_load(p) for p in FIXTURES])
    assert rep["player_margins"] == 3 * len(FIXTURES)
    assert rep["recommended_scale"] > 0
    # median(|margin|) maps to |R_terminal| ≈ 0.5 under the recommendation
    median = rep["abs_margin"]["median"]
    assert median / rep["recommended_scale"] == pytest.approx(0.5, abs=0.05)
    assert 0.0 <= rep["implied_clip_rate_at_recommended"] <= 1.0


# ----------------------------------------- Caveat 4: actor_id fail-closed ---


def test_per_turn_rewards_fails_closed_on_missing_bid_actor_id() -> None:
    from test_rl_advantage import _make_game

    game = _make_game(seed=5, finals=(80, 10, 10))
    del game["rounds"][1]["players"][2]["actor_id"]  # writer "forgot" one
    parsed = parse_transcript(game)
    with pytest.raises(ValueError, match=r"missing/empty actor_id.*bid turn"):
        per_turn_rewards(parsed, RewardConfig())


def test_per_turn_rewards_fails_closed_on_missing_reveal_actor_id() -> None:
    from test_rl_advantage import _make_game

    game = _make_game(seed=6, finals=(80, 10, 10))
    # P0 wins every round → its reveal record exists; drop its actor_id.
    del game["rounds"][0]["players"][0]["reveal"]["actor_id"]
    parsed = parse_transcript(game)
    with pytest.raises(ValueError, match=r"missing/empty actor_id.*reveal turn"):
        per_turn_rewards(parsed, RewardConfig())


def test_per_turn_rewards_rejects_empty_actor_id() -> None:
    from test_rl_advantage import _make_game

    game = _make_game(seed=7, finals=(70, 20, 20))
    game["rounds"][0]["players"][1]["actor_id"] = ""  # present but empty
    with pytest.raises(ValueError, match="missing/empty actor_id"):
        per_turn_rewards(parse_transcript(game), RewardConfig())


def test_assert_corpus_actor_ids_passes_clean_and_pins_locus() -> None:
    from test_rl_advantage import _make_game

    clean = [_make_game(seed=1, finals=(80, 10, 10)),
             _make_game(seed=2, finals=(10, 80, 10))]
    # 3 players × 4 rounds × (bid + winner-reveal) = 4*(3+1) per game
    assert assert_corpus_actor_ids(clean) == 2 * 4 * (3 + 1)

    bad = [_make_game(seed=1, finals=(80, 10, 10)),
           _make_game(seed=2, finals=(10, 80, 10))]
    del bad[1]["rounds"][0]["players"][0]["actor_id"]
    with pytest.raises(ValueError, match=r"corpus transcript #1:.*round 0"):
        assert_corpus_actor_ids(bad)


# ===========================================================================
# Phase-3 reward-design knobs. Every default
# must reproduce the locked behaviour byte-identically; the new behaviours
# are opt-in. (Module-level skipif only affects fixture-parametrised tests;
# the synthetic ones below always run — the tracked fixture is in-tree.)
# ===========================================================================

import math  # noqa: E402
from dataclasses import replace  # noqa: E402

from megagem.rl.reward import _terminal_value  # noqa: E402


def _p3_corpus() -> list[dict]:
    from test_rl_advantage import _make_game
    from test_rl_diagnostics import _reveal_game

    return [
        _reveal_game(seed=1, p0_red=3, finals=(90, 20, 10)),
        _reveal_game(seed=2, p0_red=5, finals=(15, 70, 40)),
        _make_game(seed=3, finals=(120, 10, 10)),
        _make_game(seed=4, finals=(5, 60, 60)),
    ]


def test_terminal_shape_clip_default_is_byte_identical() -> None:
    """Default ``terminal_shape='clip'`` routed through ``_terminal_value``
    must equal the locked hand-written clip expression exactly (re-pins the
    Caveat-1 SCALE lock through the new single squash path)."""
    cfg = RewardConfig()
    assert cfg.terminal_shape == "clip"
    for m in (-200.0, -21.5, -3.3, 0.0, 7.1, 21.5, 999.0):
        assert _terminal_value(m, cfg) == _clip(m / 21.5, -1.0, 1.0)


def test_terminal_shape_tanh_range_and_strict_monotone() -> None:
    """rec 2: tanh is bounded in (-1, 1) and STRICTLY monotone everywhere —
    a win-by-60 still outranks a win-by-22 (the Problem-C property the hard
    clip destroys past ±scale)."""
    cfg = replace(RewardConfig(), terminal_shape="tanh")
    grid = [x * 0.5 for x in range(-400, 401)]
    vals = [_terminal_value(m, cfg) for m in grid]
    assert all(-1.0 < v < 1.0 for v in vals)
    assert all(b > a for a, b in zip(vals, vals[1:]))  # strictly increasing
    # past the clip point the two squashes genuinely differ
    clip_cfg = RewardConfig()
    assert _terminal_value(60.0, cfg) != _terminal_value(60.0, clip_cfg)
    assert _terminal_value(60.0, clip_cfg) == _terminal_value(22.0, clip_cfg)


def test_terminal_shape_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown terminal_shape"):
        _terminal_value(1.0, replace(RewardConfig(), terminal_shape="relu"))


def test_reveal_weight_one_is_byte_identical() -> None:
    """rec 3 knob at the locked weight 1.0 ⇒ byte-identical shaping split."""
    from test_rl_diagnostics import _reveal_game

    g = _reveal_game(seed=7, p0_red=4, finals=(80, 20, 10))
    parsed = parse_transcript(g)
    base_b, base_r = round_shaping_components(parsed, RewardConfig())
    w1_b, w1_r = round_shaping_components(
        parsed, replace(RewardConfig(), reveal_shaping_weight=1.0)
    )
    assert w1_b == base_b and w1_r == base_r


def test_reveal_weight_zero_zeroes_only_reveal_subdelta() -> None:
    """rec 3: weight 0.0 zeroes the reveal sub-delta, leaves the bid
    sub-delta untouched and keeps every (round,player) key present so the
    fold logic is structurally unchanged."""
    from test_rl_diagnostics import _reveal_game

    g = _reveal_game(seed=8, p0_red=5, finals=(80, 20, 10))
    parsed = parse_transcript(g)
    b0, r0 = round_shaping_components(parsed, RewardConfig())
    bz, rz = round_shaping_components(
        parsed, replace(RewardConfig(), reveal_shaping_weight=0.0)
    )
    assert bz == b0  # bid sub-delta untouched
    assert set(rz) == set(r0)  # keys preserved (fold logic intact)
    assert all(v == 0.0 for v in rz.values())  # reveal sub-delta zeroed
    assert any(v != 0.0 for v in r0.values())  # (it was nonzero by construction)
    # round_shaping total == bid-only when reveal is zeroed
    total_z = round_shaping(
        parsed, replace(RewardConfig(), reveal_shaping_weight=0.0)
    )
    assert total_z == {k: bz[k] for k in bz}


def test_locked_path_absolute_golden_on_tracked_fixture() -> None:
    """Legacy ANCHOR (not self-comparison). The other byte-identity tests
    compare ``RewardConfig()`` against an explicit reconstruction of the *same*
    new defaults — they prove the opt-in knobs are inert relative to
    themselves, not that the locked numbers are unchanged. This pins the
    ABSOLUTE locked output on the in-tree tracked fixture: r_terminal =
    clip(margin/21.5,-1,1) + 0.1·is_winner. These constants were cross-checked
    byte-identical against pristine HEAD on the real 48-game corpus; any future
    drift of scale / win_bonus / the clip path fails here against fixed
    numbers, not against a moving reconstruction. (Codex review 2026-05-19.)"""
    from megagem.rl.reward import terminal_breakdowns

    d = json.load(
        open(REPO_ROOT / "tests/fixtures/megagem_schema_v3_sanitized.json")
    )
    parsed = parse_transcript(d)
    turns = per_turn_rewards(parsed, RewardConfig())
    assert len(turns) == 62
    assert sum(t.terminal_correction for t in turns) == 0.0  # 1a off by default
    tb = terminal_breakdowns(parsed, RewardConfig())
    assert sorted(round(v.margin, 12) for v in tb.values()) == [-18.0, 3.0, 15.0]
    assert sorted(round(v.r_terminal, 12) for v in tb.values()) == [
        -0.837209302326,
        0.139534883721,
        0.797674418605,
    ]


def test_terminal_correction_off_is_byte_identical() -> None:
    """Default (terminal_correction=False) ⇒ every TurnReward.terminal_correction
    is 0.0 and per_turn_rewards is byte-identical to pre-Phase-3."""
    for g in _p3_corpus():
        parsed = parse_transcript(g)
        turns = per_turn_rewards(parsed, RewardConfig())
        assert all(t.terminal_correction == 0.0 for t in turns)


def test_telescope_identity_exact_under_both_reveal_settings() -> None:
    """rec 1a — the core correctness property (Rev 7): with the correction on,
    Σ_turns(shaping + terminal_correction) == (λ/Sc)·true_margin exactly, and
    it stays exact when reveal_shaping_weight=0 because s_raw is recomputed
    per-cfg (not bolted on)."""
    for w in (1.0, 0.0):
        cfg = replace(RewardConfig(), terminal_correction=True,
                      reveal_shaping_weight=w)
        for g in _p3_corpus():
            parsed = parse_transcript(g)
            term = terminal_breakdowns(parsed, cfg)
            acc: dict[int, float] = {}
            for t in per_turn_rewards(parsed, cfg):
                acc[t.player_id] = acc.get(t.player_id, 0.0) + (
                    t.shaping + t.terminal_correction
                )
            for pid, bd in term.items():
                expected = cfg.shaping_lambda * (bd.margin / cfg.scale)
                assert acc[pid] == pytest.approx(expected, abs=1e-9)


def test_calibrate_scale_tanh_branch_and_clip_unchanged() -> None:
    """rec 2 calibration: the tanh branch targets tanh(median/SCALE)=0.5
    (SCALE≈1.82·median, NOT the linear 2·median); the default clip call is
    byte-identical to pre-Phase-3 (no new keys)."""
    corpus = _p3_corpus()
    clip = calibrate_scale(corpus)  # default cfg
    assert "implied_clip_rate_at_recommended" in clip
    assert "terminal_shape" not in clip  # clip path unchanged shape
    tanh = calibrate_scale(corpus, replace(RewardConfig(), terminal_shape="tanh"))
    assert tanh["terminal_shape"] == "tanh"
    med = tanh["abs_margin"]["median"]
    assert math.tanh(med / tanh["recommended_scale"]) == pytest.approx(
        0.5, abs=0.05
    )
