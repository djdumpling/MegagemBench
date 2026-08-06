"""Prompt generation for MegaGem players."""

import json
import statistics
import textwrap
from collections import Counter

from ..game.cards import AuctionType, GemColor
from ..game.state import GameState

GAME_RULES = textwrap.dedent(
    """
    # MegaGem - Strategic Auction Game

    You are playing MegaGem, a strategic auction and collection game involving risk assessment, decision making, and even bluffing.
    You will be playing against the other players at the table (see the `players` list in each turn's state for the exact count and seating).

    ## Response Format
    For each action:
    - Use the current game state provided in the user message.
    - Briefly explain your reasoning.
    - End with exactly one final JSON object matching the requested action format.

    ## Objective
    Earn the most coins by:
    - Winning auctions for gems, loans, and investments
    - Completing Mission cards for coins at the end of the game
    - Building a high-value gem collection based on the Value Chart

    ## Game Length
    - The game ends when all auctionable gems have been won from Treasure auctions, usually in 15-20 rounds (the deck holds 25 auction cards total, so 25 rounds is the hard maximum).
    - Private hand gems are never auctioned; they are revealed to the Value Display during play or at game end.
    - Pace your bids: running out of coins early means you cannot bid above 0 in later non-loan auctions.

    ## Key Concepts

    ### Gem Colors
    There are 5 gem colors (Red, Blue, Green, Purple, Yellow) each with 6 gems total (30 gems in the game).

    ### Value Chart
    The Value Chart determines how much each gem in your collection is worth at the end.
    - Gems revealed to the VALUE DISPLAY determine the per-gem value for each color (see chart).
    - Each gem in your collection scores its color's per-gem value; gems in your hand do not score.
    - Final scoring uses the Value Display after all remaining private hand gems are automatically revealed.

    ### Collection vs Hand
    **Collection** (PUBLIC): Gems you have WON from auctions. These are PUBLIC information visible to all players.
    - Gems in your collection are NOT in your hand
    - Gems in your collection count toward your final score
    - Gems in your collection can be used to complete missions
    - Only gems in your collection count for gem value at scoring; gems in your hand do not score directly (hand gems only affect the Value Display when revealed during the game or at end).

    **Hand** (PRIVATE): Gems you started with that you haven't revealed yet. These are PRIVATE - only you can see them.
    - Gems in your hand are NOT in your collection
    - When you win a Treasure auction, the gem(s) you win go to your collection (not hand)
    - Gems in your hand will eventually be revealed to the Value Display
    - At the end of the game, any remaining gems in your hand are automatically revealed
    - Hand gems are only revealed (during the game or at end); they are never sold or traded

    ### Revealed Gems
    Gems in your hand will eventually be revealed to the Value Display:
    - When you win a Treasure auction, you must reveal one gem from your hand (if you have any)
    - If you have no gems in your hand, you still need to bid the most (or win by tiebreak) to win the auction, but you do not need to reveal any card because you have none
    - The revealed gem does not need to be the same color as the gem you won
    - At the end of the game, any remaining gems in your hand are automatically revealed and counted in the value display for coins
    - Since all private hand gems are eventually revealed, choosing a reveal is about timing, public information, and what you withhold from opponents until later, not about whether that gem ever affects final Value Display counts.

    ### Auction Types
    There will only ever be one auction at a time. The deck has 25 auction cards in total (so at most 25 rounds):
    - 12 × Treasure (1 gem)
    - 5 × Treasure (2 gems)
    - 2 × Loan 10, 2 × Loan 20
    - 2 × Investment 5, 2 × Investment 10

    ### Bidding rules:
    - Bids must be non-negative integers.
    - For Treasure and Investment auctions, your maximum bid is your current coins.
    - Only Loan auctions let you bid more than your current coins.
    1. **Treasure (1 gem)**: Winner pays bid, receives 1 revealed gem
    2. **Treasure (2 gems)**: Winner pays bid, receives both revealed gems
    3. **Loan 10/20**: Winner receives the loan amount now, pays it back at end. You can bid MORE than your current coins (max bid = coins + loan amount). Net now = loan amount - bid.
    4. **Investment 5/10**: Winner pays bid (coins LOCKED until end), then receives bid + bonus at end.

    ### Missions
    Missions are automatically completed when you have the required gems in your collection after winning treasure auctions.
    - First player to qualify for a mission claims it exclusively.
    - Gems used for missions remain in your collection.
    - Consider mission rewards when evaluating the value of winning treasure auctions.

    ### Tiebreak Order
    When bids tie, winner is determined by tiebreak order.
    After winning, the winner moves to the END of tiebreak order.

    ## Scoring
    Final Score = Coins + Gem collection Value + Mission Rewards - Loan Payments + Investment Returns
    """
).strip()


def _json_section(section_name: str, payload: object) -> str:
    """Compact-JSON section. Compact form is ~36% smaller than indent=2 on a
    typical bid turn (measured: 6,002 → 3,813 chars on seed 1000)."""
    return json.dumps({section_name: payload}, separators=(",", ":"))


def format_value_chart_info(game_state: GameState) -> str:
    chart = game_state.value_chart
    return _json_section(
        "value_chart",
        {
            "id": chart.id,
            "description": chart.description,
            "values": [
                {"gems_in_display": count, "value_per_gem": value}
                for count, value in sorted(chart.values.items())
            ],
        },
    )


def format_value_display(game_state: GameState) -> str:
    counts = game_state.get_value_display_counts()
    return _json_section(
        "value_display",
        {
            "revealed_gems": list(game_state.value_display),
            "by_color": [
                {
                    "color": color.value,
                    "count": counts.get(color.value, 0),
                    "current_value_per_gem": game_state.get_gem_value(color.value),
                }
                for color in GemColor
            ],
        },
    )


def format_revealed_gems(game_state: GameState) -> str:
    """Revealed gems available for the current Treasure auction; empty otherwise."""
    auction = game_state.current_auction
    if auction is None or auction.type != AuctionType.TREASURE:
        return ""

    gems_to_show = game_state.revealed_gems[: auction.gems]
    return _json_section(
        "revealed_gems_available_for_auction",
        {
            "gems": gems_to_show,
            "count": len(gems_to_show),
        },
    )


def format_current_auction(game_state: GameState, viewing_player_id: int | None = None) -> str:
    auction = game_state.current_auction
    if auction is None:
        return _json_section("current_auction", None)

    auction_info = {
        "id": auction.id,
        "type": auction.type.value,
        "max_bid_for_you": None,
        "bid_limit_reason": None,
    }

    if auction.type == AuctionType.TREASURE:
        auction_info.update(
            {
                "gems_count": auction.gems,
                "gems_available": game_state.revealed_gems[: auction.gems],
            }
        )
        if viewing_player_id is not None:
            player_coins = game_state.players[viewing_player_id].coins
            auction_info["max_bid_for_you"] = player_coins
            auction_info["bid_limit_reason"] = "Treasure bids cannot exceed your current coins."
    elif auction.type == AuctionType.LOAN:
        auction_info["loan_amount"] = auction.amount
        if viewing_player_id is not None:
            player_coins = game_state.players[viewing_player_id].coins
            auction_info["max_bid_for_you"] = player_coins + auction.amount
            auction_info["bid_limit_reason"] = "Loan bids can be up to current coins plus loan amount."
    else:
        auction_info["investment_bonus"] = auction.bonus
        if viewing_player_id is not None:
            player_coins = game_state.players[viewing_player_id].coins
            auction_info["max_bid_for_you"] = player_coins
            auction_info["bid_limit_reason"] = "Investment bids cannot exceed your current coins."

    return _json_section("current_auction", auction_info)


def format_players_public(game_state: GameState, viewing_player_id: int) -> str:
    return _json_section(
        "players",
        [
            {
                "player_id": player.player_id,
                "is_you": player.player_id == viewing_player_id,
                "coins": player.coins,
                "collection": list(player.collection),
                "collection_counts": player.get_collection_counts(),
                "collection_is_public": True,
                "num_private_hand_cards": len(player.hand),
                "total_loans": sum(player.loans),
                "num_investments": len(player.investments),
                "completed_missions": list(player.completed_missions),
            }
            for player in game_state.players
        ],
    )


def format_missions(game_state: GameState) -> str:
    return _json_section(
        "available_missions",
        [
            {
                "id": mission.id,
                "type": mission.type,
                "description": mission.description,
                "requirement": {
                    "type": mission.requirement.type,
                    "count": mission.requirement.count,
                    "colors": list(mission.requirement.colors),
                    "color": mission.requirement.color,
                },
                "reward": mission.reward,
            }
            for mission in game_state.available_missions
        ],
    )


def format_tiebreak_order(game_state: GameState, viewing_player_id: int) -> str:
    return _json_section(
        "tiebreak_order",
        {
            "first_has_highest_priority": True,
            "order": [
                {
                    "player_id": pid,
                    "is_you": pid == viewing_player_id,
                }
                for pid in game_state.tiebreak_order
            ],
        },
    )


def format_game_progress(game_state: GameState) -> str:
    total_gems_won = sum(len(player.collection) for player in game_state.players)
    auctionable_gems_remaining = len(game_state.gem_deck) + len(game_state.revealed_gems)
    return _json_section(
        "game_progress",
        {
            "round_number": game_state.round_number,
            "typical_total_rounds": "15-20",
            "remaining_auction_cards": len(game_state.auction_deck),
            "gems_won_in_treasure_auctions_so_far": total_gems_won,
            "auctionable_gems_remaining": auctionable_gems_remaining,
            "gems_remaining_in_deck_unrevealed": len(game_state.gem_deck),
            "revealed_gems_waiting_for_treasure_auction": len(game_state.revealed_gems),
        },
    )


def format_auction_history(game_state: GameState, max_auctions: int | None = None) -> str:
    """Auction history; pass max_auctions to keep only the last N entries."""
    auctions_to_show = game_state.auction_history
    if max_auctions is not None and len(auctions_to_show) > max_auctions:
        auctions_to_show = auctions_to_show[-max_auctions:]

    return _json_section(
        "auction_history",
        {
            "total_auctions_so_far": len(game_state.auction_history),
            "showing_last": max_auctions,
            "num_auctions_shown": len(auctions_to_show),
            "auctions": [
                {
                    "round_number": result.round_number,
                    "auction": result.auction_card,
                    "winner_id": result.winner_id,
                    "winning_bid": result.winning_bid,
                    "all_bids_by_player_id": result.bids,
                    "gems_won": list(result.gems_won),
                    "gem_revealed_to_display": result.gem_revealed,
                }
                for result in auctions_to_show
            ],
        },
    )


def format_private_hand(game_state: GameState, player_id: int) -> str:
    hand = game_state.players[player_id].hand
    return _json_section(
        "your_private_hand",
        {
            "cards": list(hand),
            "counts": dict(sorted(Counter(hand).items())),
            "total_cards": len(hand),
            "empty": not hand,
        },
    )


def format_turn_context(game_state: GameState, player_id: int, phase: str) -> str:
    return _json_section(
        "turn_context",
        {
            "round_number": game_state.round_number,
            "phase": phase,
            "your_player_id": player_id,
        },
    )


def format_pending_treasure_win(game_state: GameState) -> str:
    """Treasure gems the reveal-phase winner is about to add to their collection."""
    auction = game_state.current_auction
    won_gems = []
    if auction is not None and auction.type == AuctionType.TREASURE:
        won_gems = game_state.revealed_gems[: auction.gems]

    return _json_section(
        "pending_treasure_win",
        {
            "gems_won": won_gems,
            "note": "These gems will be added to your collection after the reveal decision.",
            "public_collections_include_these_gems": False,
        },
    )


def format_action_request(action: str) -> str:
    if action == "bid":
        payload = {
            "action": "bid",
            "instructions": [
                "Briefly explain your reasoning.",
                "Then output exactly one final JSON object on its own line.",
                "Do not include any other JSON object in your response.",
            ],
            "final_json_schema": {"bid": "integer"},
            "example_final_json": {"bid": 7},
        }
    elif action == "reveal":
        payload = {
            "action": "reveal",
            "instructions": [
                "Briefly explain your reasoning.",
                "Then output exactly one final JSON object on its own line.",
                "Do not include any other JSON object in your response.",
            ],
            "valid_colors": [color.value for color in GemColor],
            "final_json_schema": {"reveal": "color string"},
            "example_final_json": {"reveal": "Blue"},
        }
    else:
        raise ValueError(f"Unsupported action request: {action}")

    return _json_section("action_request", payload)


# E1 history-repair arm: seats in this set get a per-opponent bidding profile
# computed from the FULL public auction history (the default prompt truncates
# the raw history to the last 5 auctions — src/megagem/environment/prompts.py:E1 audit
# H-repair). Empty by default => every legacy prompt is byte-identical. The
# launcher sets {0} for the treatment arm only, so opponents' prompts (and
# their behavior) are held fixed.
HISTORY_REPAIR_SEATS: set[int] = set()


def format_opponent_bid_summaries(game_state: GameState, player_id: int) -> str:
    """Per-opponent bidding profile over the FULL public auction history (all
    bids are public by rule; the raw history section above shows only the last
    5 auctions). Pure summary statistics — no raw transcript bloat."""
    hist = game_state.auction_history
    if not hist:
        return ""

    def _median(xs):
        return statistics.median(xs) if xs else None

    profile = {}
    for seat in range(len(game_state.players)):
        if seat == player_id:
            continue
        bids = [r.bids[seat] for r in hist if seat < len(r.bids)]
        t_bids = [r.bids[seat] for r in hist if r.gems_won and seat < len(r.bids)]
        wins = [r for r in hist if r.winner_id == seat]
        profile[f"player_{seat}"] = {
            "auctions_bid": len(bids),
            "median_bid": _median(bids),
            "max_bid": max(bids) if bids else None,
            "median_treasure_bid": _median(t_bids),
            "last_3_bids": bids[-3:],
            "auctions_won": len(wins),
            "median_winning_bid": _median([r.winning_bid for r in wins]),
            "total_coins_spent": sum(r.winning_bid for r in wins),
        }
    by_size = {}
    for size in (1, 2):
        wbs = [r.winning_bid for r in hist if len(r.gems_won) == size]
        if wbs:
            by_size[f"{size}_gem_treasures"] = {
                "n": len(wbs), "median_winning_bid": _median(wbs), "last_winning_bid": wbs[-1]}
    return _json_section(
        "opponent_bidding_profile",
        {"note": "full-game statistics for each opponent (the auction_history "
                 "section shows only the last 5 auctions)",
         "opponents": profile,
         "market_clearing_prices": by_size},
    )


def generate_bid_prompt(game_state: GameState, player_id: int,
                        value_block: str | None = None) -> str:
    """Bid prompt — all public info plus player_id's private hand.

    ``value_block``: optional in-context value-head estimate (Phase 2), inserted just
    before the action request for the policy seat only. None on every normal/opponent
    path (so opponents and the value-OFF control are byte-identical to before).
    """
    sections = [
        format_turn_context(game_state, player_id, "bidding"),
        "",
        format_value_chart_info(game_state),
        "",
        format_value_display(game_state),
        "",
    ]

    revealed_gems_text = format_revealed_gems(game_state)
    if revealed_gems_text:
        sections.append(revealed_gems_text)
        sections.append("")

    sections.extend(
        [
            format_game_progress(game_state),
            "",
            format_current_auction(game_state, player_id),
            "",
            format_players_public(game_state, player_id),
            "",
            format_missions(game_state),
            "",
            format_tiebreak_order(game_state, player_id),
            "",
            format_auction_history(game_state, max_auctions=5),
            "",
        ]
    )

    if player_id in HISTORY_REPAIR_SEATS:
        profile = format_opponent_bid_summaries(game_state, player_id)
        if profile:
            sections.append(profile)
            sections.append("")

    sections.extend(
        [
            format_private_hand(game_state, player_id),
            "",
        ]
    )

    if value_block:
        sections.append(value_block)
        sections.append("")

    sections.extend(
        [
            "Decide your bid. Loan tip: loans help most when your coins are low (liquidity) and hurt when high (just debt).",
            "",
            format_action_request("bid"),
        ]
    )

    return "\n".join(sections)


def generate_reveal_prompt(game_state: GameState, player_id: int) -> str:
    """Reveal prompt — for the player who just won a Treasure auction."""
    sections = [
        format_turn_context(game_state, player_id, "gem_reveal"),
        "",
        "Reveal ONE gem from your hand to the Value Display.",
        "All private hand gems are eventually revealed at game end, so this choice controls what becomes public now and what stays hidden until later.",
        "",
        format_pending_treasure_win(game_state),
        "",
        format_players_public(game_state, player_id),
        "",
        format_value_chart_info(game_state),
        "",
        format_value_display(game_state),
        "",
        format_private_hand(game_state, player_id),
        "",
        "Choose a reveal based on information timing, opponent inference, future bidding incentives, and the current Value Chart.",
        "",
        format_action_request("reveal"),
    ]

    return "\n".join(sections)

# E2 persona gate: optional strategy-card suffix appended to the system prompt
# for EVERY seat in the process (persona tables are homogeneous by design).
# Empty by default => byte-identical legacy prompts.
PERSONA_SUFFIX: str = ""


def generate_system_prompt() -> str:
    """Static GAME_RULES — kept in the system message so it caches in the
    prefix instead of re-rendering on every user turn."""
    if PERSONA_SUFFIX:
        return GAME_RULES + "\n\n[STRATEGY DIRECTIVE]\n" + PERSONA_SUFFIX
    return GAME_RULES
