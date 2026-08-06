"""Phase 1.1 — non-mutating per-state scorer.

Gates (the RL plan §1, made concrete):

* **engine-replay oracle** — drive the real engine deterministically with the
  transcript's recorded actions; at every round the offline transcript-adapter
  score must *exactly* equal the engine's non-mutating ``calculate_final_score``,
  and the replayed terminal scores must reproduce ``final_results``.
* **no mutation / no reveal_remaining_hands** during offline scoring.
* **no opponent-hand leakage** — scrubbing every ``hand`` from the transcript
  must not change any number.
* **loans + investments** included.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from megagem.data import load_auctions, load_gems, load_missions, load_value_charts
from megagem.game.cards import AuctionType, ValueChart
from megagem.game.rules import (
    apply_auction_outcome,
    calculate_final_score,
    complete_mission,
    determine_winner,
    resolve_auction,
    reveal_gem_from_hand,
)
from megagem.game.state import GameState
from megagem.rl.scorer import (
    decision_tiebreak_order,
    parse_transcript,
    reconstruct_snapshots,
    score_at,
    score_snapshot,
)

from _rl_fixtures import fixture_paths, load as _load

FIXTURES = fixture_paths()
pytestmark = pytest.mark.skipif(not FIXTURES, reason="no schema-v3 fixtures")


def _run_mission_phase(game_state: GameState, winner_id: int) -> None:
    """Inline mirror of MegaGemEnv.run_mission_phase (avoids verifiers dep)."""
    if not game_state.available_missions:
        return
    for mission in list(game_state.available_missions):
        complete_mission(game_state, winner_id, mission.id)


def _build_game_state(parsed, chart_id: str) -> GameState:
    charts = load_value_charts()
    return GameState.create_new_game(
        num_players=parsed.num_players,
        gem_cards=load_gems(),
        auction_cards=load_auctions(),
        missions=load_missions(),
        value_chart=ValueChart.from_dict(chart_id, charts[chart_id]),
        seed=parsed.seed,
    )


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_replay_oracle_matches_adapter_every_round(path: Path) -> None:
    """Engine non-mutating score == transcript-adapter score, exactly, each round."""
    data = _load(path)
    chart_id = data.get("metadata", {}).get("value_chart", "A")
    parsed = parse_transcript(data)
    snaps = reconstruct_snapshots(parsed)
    gs = _build_game_state(parsed, chart_id)
    n = parsed.num_players

    for r_idx, rnd in enumerate(parsed.rounds):
        auction = gs.draw_auction_card()
        assert auction is not None, f"engine ran out of auctions at round {r_idx}"
        assert auction.type.value == rnd["auction"]["type"], (
            f"round {r_idx}: replay auction {auction.type.value} != "
            f"recorded {rnd['auction']['type']} (seed/order desync)"
        )

        bids = [
            next(p["bid"] for p in rnd["players"] if p["player_id"] == pid)
            for pid in range(n)
        ]
        outcome = resolve_auction(gs, bids)

        recorded_winner = next(
            p["player_id"] for p in rnd["players"] if p.get("is_winner")
        )
        assert outcome.winner_id == recorded_winner, (
            f"round {r_idx}: replay winner {outcome.winner_id} != "
            f"recorded {recorded_winner}"
        )

        gem_revealed = None
        if auction.type == AuctionType.TREASURE:
            wrec = next(
                p for p in rnd["players"] if p["player_id"] == outcome.winner_id
            )
            gem_revealed = wrec.get("gem_revealed")
        if gem_revealed:
            assert reveal_gem_from_hand(gs, outcome.winner_id, gem_revealed)

        apply_auction_outcome(gs, outcome, bids, gem_revealed)

        if auction.type == AuctionType.TREASURE and gs.available_missions:
            _run_mission_phase(gs, outcome.winner_id)

        # round_callback fires here in the real rollout — compare now.
        for pid in range(n):
            engine = calculate_final_score(gs, pid)
            adapter = score_snapshot(snaps[(r_idx, pid)])
            assert adapter["final_score"] == engine["final_score"], (
                f"round {r_idx} P{pid}: adapter {adapter['final_score']} "
                f"!= engine {engine['final_score']}"
            )
            for key in (
                "coins",
                "gem_value",
                "mission_rewards",
                "loan_payments",
                "investment_returns",
            ):
                assert adapter[key] == engine[key], (
                    f"round {r_idx} P{pid} {key}: {adapter[key]} != {engine[key]}"
                )

    # Terminal: determine_winner reveals remaining hands (mutates) — the
    # replay must reproduce the recorded engine-true final scores exactly.
    winner_id, final_scores = determine_winner(gs)
    assert winner_id == parsed.winner_id
    by_pid = {s["player_id"]: s for s in final_scores}
    for rec in parsed.final_scores:
        assert by_pid[rec["player_id"]]["final_score"] == rec["final_score"]


@pytest.mark.parametrize("path", FIXTURES[:3], ids=lambda p: p.name)
def test_no_reveal_remaining_hands_and_no_mutation(path: Path, monkeypatch) -> None:
    calls = {"n": 0}
    real = GameState.reveal_remaining_hands

    def spy(self):  # pragma: no cover - asserted to never run
        calls["n"] += 1
        return real(self)

    monkeypatch.setattr(GameState, "reveal_remaining_hands", spy)

    data = _load(path)
    before = copy.deepcopy(data)
    parsed = parse_transcript(data)
    snaps = reconstruct_snapshots(parsed)
    first = {k: score_snapshot(v) for k, v in snaps.items()}
    second = {k: score_snapshot(v) for k, v in snaps.items()}

    assert calls["n"] == 0, "offline scorer must never reveal remaining hands"
    assert data == before, "offline scorer mutated the transcript"
    assert first == second, "scorer is not a pure function of the snapshot"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_opponent_hand_leakage_gate(path: Path) -> None:
    """Scrubbing every recorded hand must not change any score (causal proof)."""
    data = _load(path)
    scrubbed = _load(path)
    touched = 0
    for rnd in scrubbed.get("rounds", []):
        for prec in rnd.get("players", []):
            if "hand" in prec:
                prec["hand"] = ["__SCRUBBED__"]
                touched += 1
    assert touched > 0, "fixture has no hands to scrub — gate is vacuous"

    a = {k: score_snapshot(v) for k, v in reconstruct_snapshots(parse_transcript(data)).items()}
    b = {k: score_snapshot(v) for k, v in reconstruct_snapshots(parse_transcript(scrubbed)).items()}
    assert a == b, "scores depend on private hands — leakage!"

    snap = next(iter(reconstruct_snapshots(parse_transcript(data)).values()))
    assert not any("hand" in f for f in snap.__dataclass_fields__), (
        "TurnStateSnapshot exposes a hand field"
    )


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_loans_and_investments_reconstructed(path: Path) -> None:
    """Terminal-round loan/investment totals must equal engine final_results."""
    data = _load(path)
    parsed = parse_transcript(data)
    snaps = reconstruct_snapshots(parsed)
    last = len(parsed.rounds) - 1
    for rec in parsed.final_scores:
        pid = rec["player_id"]
        snap = snaps[(last, pid)]
        assert snap.loan_total == rec["loan_payments"], (
            f"P{pid} loan {snap.loan_total} != {rec['loan_payments']}"
        )
        assert snap.investment_return_total == rec["investment_returns"], (
            f"P{pid} inv {snap.investment_return_total} != {rec['investment_returns']}"
        )


def test_corpus_exercises_loans_and_investments() -> None:
    any_loan = any_inv = False
    for path in FIXTURES:
        parsed = parse_transcript(_load(path))
        for rec in parsed.final_scores:
            any_loan |= rec["loan_payments"] > 0
            any_inv |= rec["investment_returns"] > 0
    assert any_loan and any_inv, "fixtures cover neither loans nor investments"


@pytest.mark.parametrize("path", FIXTURES[:2], ids=lambda p: p.name)
def test_score_at_helper(path: Path) -> None:
    parsed = parse_transcript(_load(path))
    last = len(parsed.rounds) - 1
    for pid in range(parsed.num_players):
        assert isinstance(score_at(parsed, last, pid), int)


def test_decision_tiebreak_order_prefers_before_field_then_r_minus_1() -> None:
    """§A.5 must use decision-time order, not the post-resolution one."""
    data = _load(FIXTURES[0])
    parsed = parse_transcript(data)

    # Field present (post-fix corpora / the tracked fixture) → use it verbatim.
    for r_idx, rnd in enumerate(parsed.rounds):
        if "tiebreak_order_before" in rnd:
            assert decision_tiebreak_order(parsed, r_idx) == rnd[
                "tiebreak_order_before"
            ]

    # Legacy fallback: strip the field; r≥1 must equal r-1's post-update order
    # (update_tiebreak_order fires once per round), r0 falls back to recorded.
    legacy = _load(FIXTURES[0])
    for rnd in legacy["rounds"]:
        rnd.pop("tiebreak_order_before", None)
    lp = parse_transcript(legacy)
    assert decision_tiebreak_order(lp, 0) == lp.rounds[0]["tiebreak_order"]
    for r_idx in range(1, len(lp.rounds)):
        assert (
            decision_tiebreak_order(lp, r_idx)
            == lp.rounds[r_idx - 1]["tiebreak_order"]
        )

    # The post-fix path must agree with the legacy r-1 reconstruction for r≥1
    # (both are the true decision-time order) — proves the field is consistent.
    for r_idx in range(1, len(parsed.rounds)):
        if "tiebreak_order_before" in parsed.rounds[r_idx]:
            assert (
                decision_tiebreak_order(parsed, r_idx)
                == lp.rounds[r_idx - 1]["tiebreak_order"]
            )
