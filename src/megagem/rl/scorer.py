"""Non-mutating per-state scorer over saved schema-v3 transcripts (Phase 1.1).

The engine's terminal scorer (``calculate_all_final_scores``) calls
``GameState.reveal_remaining_hands`` — a side effect that also leaks every
private hand into the value display. The RL per-turn shaping signal (the RL plan
§A.4) needs a *mid-game* score that:

* never mutates anything,
* never calls ``reveal_remaining_hands``,
* never reads any player's ``hand`` (own or opponent — the leakage gate),
* values gems at the value-display count **as of that round** (a biased
  estimate by design — see §A.1a), and
* includes loans and investments.

Gem values, coins, collection, and completed missions are recorded per round
in schema-v3. Loans and investments are *not* snapshotted per round, so they
are reconstructed by replaying the per-round ``auction`` + winner sequence
(deterministic; no model, no engine RNG). Final scoring arithmetic is shared
with the engine via :func:`megagem.game.rules.score_components` (single source of
truth, zero drift).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..game.rules import score_components

# Schema-v3 per-round player records carry the private ``hand``. The Phase 1.1
# leakage gate forbids this scorer from ever consuming it. Parsing drops it;
# tests/test_rl_scorer.py asserts the package never references the key.
_FORBIDDEN_PLAYER_KEYS = ("hand",)

# the RL plan locks 3-player MegaGem; the others mirror GameState.create_new_game.
_STARTING_COINS = {3: 35, 4: 25, 5: 20}


def starting_score(num_players: int) -> int:
    """Score before round 1: just the starting coin pile (no gems/loans/etc.)."""
    return _STARTING_COINS.get(num_players, _STARTING_COINS[3])


@dataclass(frozen=True)
class TurnStateSnapshot:
    """Everything :func:`score_components` needs for one (round, player).

    Deliberately has no ``hand`` field — the leakage gate is structural.
    ``display_value_per_gem`` is the per-color value at this round's display
    count (the biased mid-game valuation, §A.4).
    """

    round_index: int  # 0-based position in transcript["rounds"]
    round_number: int  # 1-based engine round number
    player_id: int
    coins: int
    collection_counts: dict[str, int]
    display_value_per_gem: dict[str, int]
    completed_mission_rewards: int
    loan_total: int
    investment_return_total: int
    tiebreak_order: list[int]


@dataclass
class ParsedTranscript:
    """Schema-v3 transcript reduced to exactly what Phase 1 needs (no hands)."""

    seed: int | None
    num_players: int
    num_rounds: int
    rounds: list[dict[str, Any]]
    final_scores: list[dict[str, Any]]
    winner_id: int
    value_chart: str = "A"
    mission_reward: dict[str, int] = field(default_factory=dict)


def _require_schema_v3(data: dict[str, Any]) -> None:
    version = data.get("metadata", {}).get("schema_version")
    if version != 3:
        raise ValueError(
            f"Offline scorer requires schema_version 3, got {version!r}. "
            "Re-run rollouts on the current megagem/rollout.py."
        )


def _mission_reward_map(
    rounds: list[dict[str, Any]], final_results: dict[str, Any]
) -> dict[str, int]:
    """id -> reward, unioned over every round's available_missions.

    A mission completed in round 1 is removed before that round serializes, so
    we backfill any gaps from the canonical mission data (zero-drift fallback).
    """
    reward: dict[str, int] = {}
    for rnd in rounds:
        for mission in rnd.get("available_missions", []) or []:
            mid = mission.get("id")
            if mid is not None and mid not in reward:
                reward[mid] = int(mission.get("reward", 0))
    for mission in final_results.get("available_missions", []) or []:
        mid = mission.get("id")
        if mid is not None and mid not in reward:
            reward[mid] = int(mission.get("reward", 0))

    # Backfill anything still missing (e.g. completed round 1) from src/megagem/data.
    completed_ids = {
        mid
        for rnd in rounds
        for ids in (rnd.get("missions_completed") or {}).values()
        for mid in ids
    }
    if not completed_ids.issubset(reward):
        from ..data import load_missions

        for m in load_missions():
            reward.setdefault(m["id"], int(m["reward"]))
    return reward


def parse_transcript(data: dict[str, Any]) -> ParsedTranscript:
    """Reduce a schema-v3 dict to a :class:`ParsedTranscript`. Never reads hands."""
    _require_schema_v3(data)
    rounds = data.get("rounds", []) or []
    final_results = data.get("final_results", {}) or {}
    final_scores = final_results.get("final_scores", []) or []
    return ParsedTranscript(
        seed=data.get("metadata", {}).get("seed"),
        num_players=int(data.get("metadata", {}).get("num_players", 3)),
        num_rounds=int(final_results.get("num_rounds", len(rounds))),
        rounds=rounds,
        final_scores=final_scores,
        winner_id=int(final_results.get("winner_id", -1)),
        value_chart=str(data.get("metadata", {}).get("value_chart", "A")),
        mission_reward=_mission_reward_map(rounds, final_results),
    )


def decision_tiebreak_order(
    parsed: ParsedTranscript, round_index: int
) -> list[int]:
    """Tiebreak order players actually bid under in ``round_index`` (§A.5).

    ``megagem/rollout.py`` now logs ``tiebreak_order_before`` (the decision-time
    order). Older transcripts only have post-resolution ``tiebreak_order``;
    for those, round r's decision-time order equals round r-1's recorded
    (post-update) order exactly — ``update_tiebreak_order`` fires once per
    round. Round 0 of a legacy transcript has no recoverable pre-order (it is
    the seed shuffle, never serialized); we fall back to its recorded order
    and accept that one coarse early-phase bucket may be off for *legacy*
    fixtures only — post-fix corpora carry the exact field.
    """
    rnd = parsed.rounds[round_index]
    before = rnd.get("tiebreak_order_before")
    if before is not None:
        return list(before)
    if round_index > 0:
        return list(parsed.rounds[round_index - 1].get("tiebreak_order", []))
    return list(rnd.get("tiebreak_order", []))


def _player_record(rnd: dict[str, Any], player_id: int) -> dict[str, Any]:
    for rec in rnd.get("players", []):
        if rec.get("player_id") == player_id:
            return rec
    raise KeyError(f"player {player_id} not in round {rnd.get('round_number')}")


def _collection_counts(rec: dict[str, Any]) -> dict[str, int]:
    counts = rec.get("collection_counts")
    if isinstance(counts, dict) and counts:
        return {c: int(n) for c, n in counts.items()}
    collection = rec.get("collection", []) or []
    out: dict[str, int] = {}
    for color in collection:
        out[color] = out.get(color, 0) + 1
    return out


def _display_value_per_gem(rnd: dict[str, Any]) -> dict[str, int]:
    # round["value_display"] == {color: {"count": N, "value_per_gem": V}}.
    # Colors absent from the display have value 0 (== ValueChart for count 0).
    vd = rnd.get("value_display", {}) or {}
    return {color: int(info.get("value_per_gem", 0)) for color, info in vd.items()}


def reconstruct_snapshots(
    parsed: ParsedTranscript,
) -> dict[tuple[int, int], TurnStateSnapshot]:
    """All (round_index, player_id) -> snapshot, with cumulative loans/investments.

    Loans/investments are not per-round in schema-v3; rebuild them by walking
    ``auction`` + the winning record forward (see module docstring).
    """
    loan_total: dict[int, int] = {p: 0 for p in range(parsed.num_players)}
    inv_total: dict[int, int] = {p: 0 for p in range(parsed.num_players)}
    completed: dict[int, set[str]] = {p: set() for p in range(parsed.num_players)}

    snapshots: dict[tuple[int, int], TurnStateSnapshot] = {}

    for r_idx, rnd in enumerate(parsed.rounds):
        auction = rnd.get("auction", {}) or {}
        a_type = auction.get("type")

        # --- replay this round's loan / investment effect on the winner ---
        winner_rec: dict[str, Any] | None = None
        for rec in rnd.get("players", []):
            if rec.get("is_winner"):
                winner_rec = rec
                break
        if winner_rec is not None:
            w = winner_rec["player_id"]
            if a_type == "loan":
                loan_total[w] += int(auction.get("loan_amount", 0))
            elif a_type == "investment":
                # Investment.total_return = bid_amount + bonus;
                # bid_amount == winning_bid (rules.apply_auction_outcome).
                inv_total[w] += int(winner_rec.get("winning_bid", 0)) + int(
                    auction.get("bonus", 0)
                )

        # --- accumulate this round's mission completions ---
        for pid_str, mids in (rnd.get("missions_completed") or {}).items():
            completed[int(pid_str)].update(mids)

        display = _display_value_per_gem(rnd)
        tiebreak = list(rnd.get("tiebreak_order", []))

        for player_id in range(parsed.num_players):
            rec = _player_record(rnd, player_id)
            mission_rewards = sum(
                parsed.mission_reward.get(mid, 0) for mid in completed[player_id]
            )
            snapshots[(r_idx, player_id)] = TurnStateSnapshot(
                round_index=r_idx,
                round_number=int(rec.get("round_number", rnd.get("round_number", r_idx + 1))),
                player_id=player_id,
                coins=int(rec.get("coins_after", 0)),
                collection_counts=_collection_counts(rec),
                display_value_per_gem=display,
                completed_mission_rewards=mission_rewards,
                loan_total=loan_total[player_id],
                investment_return_total=inv_total[player_id],
                tiebreak_order=tiebreak,
            )

    return snapshots


def score_snapshot_with_display(
    snap: TurnStateSnapshot, display_value_per_gem: dict[str, int]
) -> dict[str, Any]:
    """Score one snapshot's holdings against a *substituted* value display.

    Identical to :func:`score_snapshot` except the per-gem values come from
    ``display_value_per_gem`` instead of the snapshot's own (post-reveal)
    display. This is the primitive behind the §A.4 bid/reveal shaping split
    (Caveat 2): scoring the round's post-auction holdings at the *pre-reveal*
    display isolates the auction's effect from the reveal's revaluation
    effect. Same structural leakage guarantee as :func:`score_snapshot` — no
    hand is ever read, and the result is a pure function of its inputs.
    """
    return score_components(
        coins=snap.coins,
        collection_counts=snap.collection_counts,
        completed_mission_rewards=snap.completed_mission_rewards,
        loan_total=snap.loan_total,
        investment_return_total=snap.investment_return_total,
        gem_value_fn=lambda color: display_value_per_gem.get(color, 0),
    )


def score_snapshot(snap: TurnStateSnapshot) -> dict[str, Any]:
    """Non-mutating mid-game score for one snapshot (§A.4). No hand access."""
    return score_snapshot_with_display(snap, snap.display_value_per_gem)


def score_at(parsed: ParsedTranscript, round_index: int, player_id: int) -> int:
    """``final_score`` for ``player_id`` at the end of round ``round_index``."""
    snaps = reconstruct_snapshots(parsed)
    return int(score_snapshot(snaps[(round_index, player_id)])["final_score"])
