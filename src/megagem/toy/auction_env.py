"""Toy game core: a deterministic 3-player first-price private-value sealed-bid
auction over 8 rounds, with *pure-coins* scoring.

``run_one_toy_game`` is the single seam: it takes one ``policy_fn(prompt)->text``
per seat plus per-seat ``actor_ids`` and returns a schema-v3 transcript dict.
The 2.1 corpus driver and tests wrap it with deterministic stubs; Phase 2.2's
``rollout_func`` will wrap it with a live-vLLM policy. Pure aside from the
policy callables — no asyncio, no network, no GPU.

Pure-coins invariant (keeps ``src/megagem/rl`` untouched): no gems are ever held,
``value_display`` is empty, ``auction.type == "sealed_bid"`` (never the
loan/investment types the scorer replays), so ``score_components`` collapses to
``final_score == coins`` and the Caveat-2 reveal sub-delta is identically 0.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from ..game.actions import get_default_bid, length_split, parse_bid
from .serializer import build_schema_v3

NUM_PLAYERS = 3
NUM_ROUNDS = 8
# MUST equal megagem.rl.scorer._STARTING_COINS[3] — it is the §A.4 shaping baseline
# ``base = starting_score(n)``. A mismatch would silently bias round-1 shaping.
STARTING_COINS = 35
VALUE_MIN, VALUE_MAX = 0, 20

# Opponent shade factor: heuristic bids round(0.8 * private_value).
_OPP_SHADE = 0.8


def private_value(seed: int, round_index: int, player_id: int) -> int:
    """Deterministic i.i.d.-uniform integer in ``[VALUE_MIN, VALUE_MAX]``.

    stdlib ``random.Random`` only (no numpy) so the corpus is reproducible on
    any box. Independent of any rollout index — opponents stay fixed per seed
    across a K-group; only the trainable seat's *policy* varies.
    """
    rng = random.Random((seed * 1_000_003) ^ (round_index * 9176) ^ (player_id * 131))
    return rng.randint(VALUE_MIN, VALUE_MAX)


def heuristic_bid(value: int, coins: int) -> int:
    """Constant-shade truthful opponent: ``clamp(round(0.8*v), 0, coins)``."""
    return max(0, min(coins, int(round(_OPP_SHADE * value))))


def resolve_round(
    final_bids: list[int], tiebreak_before: list[int]
) -> tuple[int, int]:
    """First-price winner. Mirrors ``rules.resolve_auction`` tie semantics:
    highest bid wins; among tied max-bidders the one earliest in
    ``tiebreak_before`` wins. Returns ``(winner_id, winning_bid)``."""
    max_bid = max(final_bids)
    tied = [i for i, b in enumerate(final_bids) if b == max_bid]
    winner = min(tied, key=tiebreak_before.index)
    return winner, final_bids[winner]


def update_tiebreak_order(order: list[int], winner_id: int) -> list[int]:
    """Auction winner moves to the back; rest keep relative order. Mirrors
    ``GameState.update_tiebreak_order`` (state.py:204)."""
    return [p for p in order if p != winner_id] + [winner_id]


def _legality(parse_valid: bool, bid: int, coins_before: int) -> tuple[bool, str]:
    """Inline ``0 <= bid <= coins_before`` check (the toy has no GameState, so
    it cannot call ``validate_bid_for_auction``; this mirrors that non-loan
    branch). Unparsed turns are illegal by definition."""
    if not parse_valid:
        return False, "unparseable bid"
    if bid < 0:
        return False, "bid is negative"
    if bid > coins_before:
        return False, f"bid {bid} exceeds current coins {coins_before}"
    return True, ""


def run_one_toy_game(
    seed: int,
    policy_fns: list[Callable[[str], str]],
    actor_ids: list[str],
    *,
    rollout_index: int = 0,
    num_players: int = NUM_PLAYERS,
    num_rounds: int = NUM_ROUNDS,
    value_chart: str = "A",
    timestamp: str | None = None,
) -> dict:
    """Run the full game and return a schema-v3 transcript dict.

    ``policy_fns[seat](prompt) -> completion_text``; ``actor_ids[seat]`` is the
    Caveat-4 actor tag ("trainable" for the one trained seat, e.g. "heuristic"
    for opponents). ``rollout_index`` is informational (the trainable stub
    already varies via its own ``jitter_seed``); kept so the seam signature is
    stable for Phase 2.2.
    """
    from .prompt import bid_prompt  # local: prompt imports heuristic_bid from here

    # Locked shape (the RL plan Phase 2.1 + Caveat-validated invariant). The
    # STARTING_COINS constant equals scorer._STARTING_COINS[3]; for any other
    # num_players the §A.4 shaping baseline `starting_score(n)` would diverge
    # from the toy's actual starting pile and silently bias round-1 shaping.
    if num_players != NUM_PLAYERS:
        raise ValueError(
            f"toy is locked to {NUM_PLAYERS} players (STARTING_COINS == "
            f"scorer._STARTING_COINS[3]); got num_players={num_players}"
        )
    if num_rounds < 1:
        raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")
    if not (len(policy_fns) == len(actor_ids) == num_players):
        raise ValueError("policy_fns / actor_ids must each have num_players entries")

    coins = [STARTING_COINS] * num_players
    tiebreak_before = list(range(num_players))  # round-0 decision order [0,1,2]

    rounds_records: list[list[dict]] = []
    tiebreak_before_list: list[list[int]] = []
    tiebreak_after_list: list[list[int]] = []
    winners: list[int] = []
    winning_bids: list[int] = []

    for r in range(num_rounds):
        coins_before = list(coins)
        records: list[dict] = []
        final_bids: list[int] = []

        for p in range(num_players):
            pv = private_value(seed, r, p)
            prompt = bid_prompt(
                r, num_rounds, p, coins_before[p], pv, list(tiebreak_before)
            )
            raw = policy_fns[p](prompt)

            parsed = parse_bid(raw)
            parse_valid = bool(parsed.valid)
            parsed_action = int(parsed.bid) if parse_valid else None
            legal_valid, legal_error = _legality(
                parse_valid, parsed.bid, coins_before[p]
            )

            if parse_valid and legal_valid:
                final_bid = int(parsed.bid)
                default_used = False
            else:
                final_bid = get_default_bid()  # 0
                default_used = True

            final_bids.append(final_bid)
            records.append(
                {
                    "player_id": p,
                    "actor_id": actor_ids[p],
                    "prompt": prompt,
                    "raw_response": raw,
                    "parsed_action": parsed_action,
                    "parse_method": parsed.parse_method,
                    "parse_valid": parse_valid,
                    "legal_valid": legal_valid,
                    "default_used": default_used,
                    "final_bid": final_bid,
                    "reasoning": parsed.reasoning,
                    "length_split": length_split(raw),
                    "parse_error": parsed.error,
                    "legal_error": legal_error,
                    "private_value": pv,  # inert toy telemetry; src/megagem/rl never reads it
                    "coins_before": coins_before[p],
                }
            )

        winner, winning_bid = resolve_round(final_bids, list(tiebreak_before))
        winner_pv = next(rec["private_value"] for rec in records if rec["player_id"] == winner)

        # Pure-coins payoff: winner pays its (first-price) bid and gains its
        # own private value in coins; losers are unchanged.
        coins[winner] = coins_before[winner] - winning_bid + winner_pv
        for p in range(num_players):
            rec = records[p]
            rec["coins_after"] = coins[p] if p == winner else coins_before[p]
            rec["is_winner"] = p == winner
            if p == winner:
                rec["winning_bid"] = winning_bid

        rounds_records.append(records)
        tiebreak_before_list.append(list(tiebreak_before))
        post = update_tiebreak_order(list(tiebreak_before), winner)
        tiebreak_after_list.append(post)
        tiebreak_before = post
        winners.append(winner)
        winning_bids.append(winning_bid)

    final_coins = list(coins)
    # Game winner: max final coins; ties broken by the final (post-resolution)
    # tiebreak order — mirrors rules.determine_winner.
    max_coins = max(final_coins)
    tied = [p for p in range(num_players) if final_coins[p] == max_coins]
    final_order = tiebreak_before  # already the post-round-(n-1) order
    game_winner_id = min(tied, key=final_order.index)

    return build_schema_v3(
        seed=seed,
        num_players=num_players,
        num_rounds=num_rounds,
        value_chart=value_chart,
        actor_ids=list(actor_ids),
        rounds_records=rounds_records,
        tiebreak_before_list=tiebreak_before_list,
        tiebreak_after_list=tiebreak_after_list,
        final_coins=final_coins,
        game_winner_id=game_winner_id,
        timestamp=timestamp,
    )
