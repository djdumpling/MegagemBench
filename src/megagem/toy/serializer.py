"""schema-v3 emission for the toy auction.

Field-for-field mirror of ``megagem/rollout.py``'s ``print_round_summary`` round
serializer (megagem/rollout.py:299-417) and the ``final_results`` block
(megagem/rollout.py:704-708), restricted to the pure-coins case:

* ``auction.type == "sealed_bid"`` — a novel type the scorer's loan/investment
  replay (scorer.py:204-221) never matches, so no spurious loan/investment.
* ``value_display == {}`` every round — ``_display_value_per_gem`` returns
  ``{}``; with empty collections ``score_components`` ⇒ ``final_score == coins``
  and the Caveat-2 reveal sub-delta is identically 0.
* ``collection == []`` / ``collection_counts == {}`` every player every round.
* No ``reveal`` key (sealed-bid → ``reward._iter_records`` yields bid-only).
* No ``hand`` key (the Phase-1.1 leakage gate forbids the scorer from ever
  seeing one; we simply never emit it).
* ``actor_id`` stamped on every player record (Caveat 4 — fail-closed).
* ``tiebreak_order_before`` on every round (decision-time order, required for
  correct §A.5 ``(tiebreak_rank, phase)`` buckets — not outcome-conditioned).
"""

from __future__ import annotations

from typing import Any

_SCHEMA_VERSION = 3

# The toy is a deterministic CPU validation harness, so the transcript is
# byte-for-byte reproducible by default — including ``metadata.timestamp``,
# which would otherwise be the only non-deterministic field (``src/megagem/rl`` never
# reads it; the scorer consumes only schema_version/seed/num_players/
# value_chart). Callers that want a wall-clock stamp pass ``timestamp=``.
_DEFAULT_TIMESTAMP = "2026-01-01T00:00:00"

# Keys carried from a toy bid record into the schema-v3 player record. The
# leakage-safe ``private_value`` rides along as inert telemetry (src/megagem/rl reads
# none of: it only consumes the keys enumerated by the scorer/reward audit).
_PLAYER_KEYS = (
    "player_id",
    "actor_id",
    "prompt",
    "raw_response",
    "parsed_action",
    "parse_method",
    "parse_valid",
    "legal_valid",
    "default_used",
    "reasoning",
    "length_split",
    "parse_error",
    "legal_error",
    "private_value",
    "coins_before",
    "coins_after",
    "is_winner",
)


def build_schema_v3(
    *,
    seed: int,
    num_players: int,
    num_rounds: int,
    value_chart: str,
    actor_ids: list[str],
    rounds_records: list[list[dict[str, Any]]],
    tiebreak_before_list: list[list[int]],
    tiebreak_after_list: list[list[int]],
    final_coins: list[int],
    game_winner_id: int,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Assemble the full schema-v3 transcript dict (deterministic by default)."""
    models = [f"toy-{a}" for a in actor_ids]

    rounds: list[dict[str, Any]] = []
    for r_idx, records in enumerate(rounds_records):
        players: list[dict[str, Any]] = []
        for rec in sorted(records, key=lambda x: x["player_id"]):
            pr: dict[str, Any] = {k: rec[k] for k in _PLAYER_KEYS}
            pr["model"] = models[pr["player_id"]]
            pr["bid"] = rec["final_bid"]
            pr["collection"] = []
            pr["collection_counts"] = {}
            if rec["is_winner"]:
                pr["winning_bid"] = rec["winning_bid"]
            players.append(pr)

        rounds.append(
            {
                "round_number": r_idx + 1,
                "auction": {
                    "type": "sealed_bid",
                    "description": (
                        f"Sealed-bid private-value auction, round {r_idx + 1}"
                    ),
                },
                "players": players,
                "value_display": {},  # MUST stay empty (pure-coins invariant)
                "tiebreak_order": list(tiebreak_after_list[r_idx]),
                "tiebreak_order_before": list(tiebreak_before_list[r_idx]),
                "available_missions": [],
                "missions_completed": {},
            }
        )

    final_scores = [
        {"player_id": p, "final_score": int(final_coins[p])}
        for p in range(num_players)
    ]

    return {
        "metadata": {
            "schema_version": _SCHEMA_VERSION,
            "models": models,
            "value_chart": value_chart,
            "seed": seed,
            "num_players": num_players,
            "timestamp": timestamp or _DEFAULT_TIMESTAMP,
            "system_prompt": (
                "Toy sealed-bid auction (Phase 2.1 trainer-mechanics harness)."
            ),
        },
        "rounds": rounds,
        "final_results": {
            "winner_id": game_winner_id,
            "num_rounds": num_rounds,
            "final_scores": final_scores,
            "available_missions": [],
        },
        "statistics": None,
        "telemetry": None,
    }
