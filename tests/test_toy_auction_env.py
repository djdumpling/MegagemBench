"""Phase 2.1 gate proof — the toy sealed-bid auction flows through the
existing, unmodified ``src/megagem/rl`` pipeline; actor mask honored; shaping plumbed.

Pure CPU: no torch/trl/GPU/network. Each test imports only ``megagem.toy.*`` and
``megagem.rl.{scorer,reward,advantage,export}``.
"""

from __future__ import annotations

import copy
import json
import math

import pytest

from megagem.rl.advantage import compute_advantages
from megagem.rl.export import build_training_rows, contract_check, main
from megagem.rl.reward import (
    RewardConfig,
    assert_corpus_actor_ids,
    per_turn_rewards,
    round_shaping_components,
    terminal_breakdowns,
)
from megagem.rl.scorer import parse_transcript, reconstruct_snapshots, score_snapshot
from megagem.toy.auction_env import VALUE_MAX, run_one_toy_game
from megagem.toy.prompt import make_heuristic_policy, make_stub_policy

_STARTING_COINS = 35  # == megagem.rl.scorer._STARTING_COINS[3]

_TRAINABLE_SEAT = 0
_REQUIRED_PLAYER_KEYS = {
    "player_id", "bid", "coins_before", "coins_after", "collection",
    "collection_counts", "is_winner", "actor_id", "prompt", "raw_response",
    "parsed_action", "parse_method", "parse_valid", "legal_valid",
    "default_used", "length_split", "parse_error", "legal_error",
}


def _game(seed: int, strategy: str = "bestresp", jitter: int | None = None) -> dict:
    """One toy game: stub at the trainable seat, heuristic opponents."""
    heuristic = make_heuristic_policy()
    stub = make_stub_policy(strategy, jitter_seed=jitter)
    policy_fns = [stub if s == _TRAINABLE_SEAT else heuristic for s in range(3)]
    actor_ids = ["trainable" if s == _TRAINABLE_SEAT else "heuristic" for s in range(3)]
    return run_one_toy_game(seed, policy_fns, actor_ids)


# ---- 1. schema-v3 valid ----------------------------------------------------

def test_schema_v3_valid():
    g = _game(2000)
    p = parse_transcript(g)  # must not raise
    assert g["metadata"]["schema_version"] == 3
    assert g["metadata"]["num_players"] == 3
    assert g["metadata"]["value_chart"] == "A"
    assert p.num_rounds == 8 and len(g["rounds"]) == 8
    for rnd in g["rounds"]:
        assert "tiebreak_order_before" in rnd
        assert rnd["value_display"] == {}
        assert rnd["available_missions"] == []
        assert rnd["auction"]["type"] == "sealed_bid"
        assert len(rnd["players"]) == 3
        for rec in rnd["players"]:
            assert _REQUIRED_PLAYER_KEYS <= set(rec)
            assert "reveal" not in rec  # sealed-bid: no reveal sub-decision
            assert "hand" not in rec  # leakage gate: never emitted
    fs = {s["player_id"]: s["final_score"] for s in g["final_results"]["final_scores"]}
    assert set(fs) == {0, 1, 2}


# ---- 2. Caveat 4 (fail-closed actor_id) ------------------------------------

def test_caveat4_actor_id_stamped_and_fail_closed():
    g = _game(2001)
    for rnd in g["rounds"]:
        for rec in rnd["players"]:
            assert isinstance(rec["actor_id"], str) and rec["actor_id"]
    assert assert_corpus_actor_ids([g]) == 8 * 3  # 24 bid turns validated

    bad = copy.deepcopy(g)
    del bad["rounds"][3]["players"][1]["actor_id"]
    with pytest.raises(ValueError):
        assert_corpus_actor_ids([bad])


# ---- 3. actor mask honored -------------------------------------------------

def test_actor_mask_honored():
    g = _game(2002)
    rows = build_training_rows([g])
    assert len(rows) == 8  # one bid turn x 8 rounds, trainable seat only
    for row in rows:
        assert row["player_id"] == _TRAINABLE_SEAT
        assert row["actor_id"] == "trainable"

    adv = compute_advantages([g])
    for r in adv:
        if r.actor_id != "trainable":
            assert r.trainable is False
            assert r.advantage == 0.0


# ---- 4. shaping plumbed + invariants ---------------------------------------

def test_shaping_plumbed_and_invariants():
    g = _game(2003)
    parsed = parse_transcript(g)
    cfg = RewardConfig()

    # score == coins for every (round, player)
    snaps = reconstruct_snapshots(parsed)
    for (r_idx, pid), snap in snaps.items():
        assert score_snapshot(snap)["final_score"] == snap.coins

    # Caveat-2 reveal sub-delta is identically 0 (no display mutation)
    bid_shaping, reveal_shaping = round_shaping_components(parsed, cfg)
    assert all(v == 0.0 for v in reveal_shaping.values())

    turns = per_turn_rewards(parsed, cfg)
    assert turns and all(t.phase == "bid" for t in turns)  # no reveal turns
    for t in turns:
        # reveal-0 absorbed onto the bid turn → value unchanged
        assert t.shaping == bid_shaping[(t.round_index, t.player_id)]

    adv = compute_advantages([g])
    assert adv and all(math.isfinite(r.advantage) for r in adv)

    # final_score == final coins
    last = g["rounds"][-1]["players"]
    fs = {s["player_id"]: s["final_score"] for s in g["final_results"]["final_scores"]}
    for rec in last:
        assert fs[rec["player_id"]] == rec["coins_after"]


# ---- 5. known-good vs known-bad: the ROBUST (terminal) invariant ----------

def test_known_good_vs_known_bad_terminal():
    """The honest "known-good positive / known-bad negative" property.

    Review (2026-05-16): standardized *advantage* sign is NOT a general
    invariant — RTG − (per-bucket EMA baseline) then within-group
    standardization can flip it per seed (flips reproduced at seeds 1000,
    1067). The structurally-guaranteed signal lives in the engine-true
    terminal outcome, asserted here across several seeds:

    * ``bestresp`` bids ``round(0.85·v) ≤ v`` so a winning round's Δcoins =
      ``v − bid ≥ 0`` — coins are monotone non-decreasing ⇒ final ≥ starting.
    * ``overbid`` spends all coins every round and can gain at most ``v ≤
      VALUE_MAX`` ⇒ final ≤ VALUE_MAX < starting; opponents only ever gain
      (Δ = +0.2·v ≥ 0) ⇒ mean(others) ≥ starting ⇒ overbid margin < 0.
    """
    cfg = RewardConfig()
    for seed in (1000, 1067, 2004, 4242):
        good = parse_transcript(_game(seed, "bestresp"))
        bad = parse_transcript(_game(seed, "overbid"))
        tg = terminal_breakdowns(good, cfg)[_TRAINABLE_SEAT]
        tb = terminal_breakdowns(bad, cfg)[_TRAINABLE_SEAT]
        assert tg.final_score >= _STARTING_COINS  # monotone non-decreasing
        assert tb.final_score <= VALUE_MAX  # spent it all, gained ≤ v
        assert tg.final_score > tb.final_score
        assert tb.r_terminal < 0.0  # structurally-negative margin

    # illegal-drag: every trainable bid is illegal → legal reward −0.5
    g_ill = _game(2004, "illegal")
    parsed = parse_transcript(g_ill)
    tr = [t for t in per_turn_rewards(parsed, cfg) if t.actor_id == "trainable"]
    assert tr and all(t.legal == -0.5 and not t.legal_valid for t in tr)
    for rec in (p for rnd in g_ill["rounds"] for p in rnd["players"]
                if p["actor_id"] == "trainable"):
        assert rec["legal_valid"] is False and rec["default_used"] is True


def test_advantage_sign_pinned_regression_seed2004():
    """Seed-PINNED wiring canary — NOT a general invariant (see
    test_known_good_vs_known_bad_terminal). At seed 2004 the full
    reward→RTG→EMA-baseline→group-standardize path yields +adv for the
    known-good game and −adv for the known-bad one in the same group; pinned
    so a pipeline regression that silently zeroes/flips advantages is caught.
    """
    adv = compute_advantages([_game(2004, "bestresp"), _game(2004, "overbid")])
    good = next(
        r for r in adv if r.game_id == 0 and r.trainable and r.is_terminal_turn
    )
    bad = next(
        r for r in adv if r.game_id == 1 and r.trainable and r.is_terminal_turn
    )
    assert good.advantage > 0.0 > bad.advantage


# ---- determinism + locked-shape / seat validation -------------------------

def test_deterministic_byte_for_byte():
    a = json.dumps(_game(777, "bestresp"), sort_keys=True)
    b = json.dumps(_game(777, "bestresp"), sort_keys=True)
    assert a == b  # incl. metadata.timestamp (fixed by default)


def test_locked_shape_and_arity_validation():
    h = make_heuristic_policy()
    with pytest.raises(ValueError, match="locked to 3 players"):
        run_one_toy_game(1, [h] * 4, ["heuristic"] * 4, num_players=4)
    with pytest.raises(ValueError, match="num_players entries"):
        run_one_toy_game(1, [h, h], ["trainable", "heuristic"])  # arity mismatch
    with pytest.raises(ValueError, match="num_rounds"):
        run_one_toy_game(1, [h, h, h], ["trainable", "heuristic", "heuristic"],
                         num_rounds=0)


def test_corpus_driver_rejects_bad_seat_and_cleans_stale(tmp_path):
    import toy_corpus

    with pytest.raises(SystemExit):  # out-of-range seat → zero trainable rows
        toy_corpus.main(["--out-dir", str(tmp_path), "--num-seeds", "1",
                         "--trainable-seat", "9"])
    with pytest.raises(SystemExit):  # locked shape
        toy_corpus.main(["--out-dir", str(tmp_path), "--num-seeds", "1",
                         "--num-players", "4"])

    # build_game self-guards too (the seam Phase 2.2's rollout_func calls
    # directly, not via the CLI).
    with pytest.raises(ValueError, match="trainable_seat must be in"):
        toy_corpus.build_game(
            seed=1, rollout_index=0, trainable_seat=9,
            opponent_actor_id="heuristic", strategy="bestresp",
            num_players=3, num_rounds=8, value_chart="A",
        )

    out = tmp_path / "corpus"
    chart = out / "chart_A"
    chart.mkdir(parents=True)
    stale = chart / "toy_seed999_r9.json"
    stale.write_text("{}")  # contaminant from a hypothetical previous run
    assert toy_corpus.main(["--out-dir", str(out), "--num-seeds", "2",
                            "--rollouts-per-seed", "1"]) == 0
    assert not stale.exists()  # cleaned
    assert len(list(chart.glob("toy_seed*_r*.json"))) == 2


# ---- 6. end-to-end export over a K-group corpus ----------------------------

def test_end_to_end_export(tmp_path, capsys):
    chart = tmp_path / "chart_A"
    chart.mkdir()
    seeds, K = 4, 8
    for s in range(3000, 3000 + seeds):
        for k in range(K):
            g = _game(s, "bestresp", jitter=k)  # same metadata.seed across K
            (chart / f"toy_seed{s}_r{k}.json").write_text(json.dumps(g))

    rc = main(["--corpus", str(chart), "--report"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)

    assert report["training_rows_B"] == seeds * K * 8  # trainable rows only
    assert report["corpus_games"] == seeds * K
    gn = report["advantage_summary"]["group_norm"]
    assert gn["groups"] == seeds
    assert gn["rollouts_per_group_min"] == K
    assert "true K-group" in gn["mode"]
    a = report["advantage_summary"]["advantage"]
    assert abs(a["mean"]) < 1e-6 and math.isfinite(a["std"]) and a["std"] > 0.0

    rows = build_training_rows([_game(3000, "bestresp", jitter=0)])
    assert contract_check(rows) == 8
