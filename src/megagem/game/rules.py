"""Game rules and logic for MegaGem."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .cards import AuctionType
from .state import AuctionResult, GameState, Investment


@dataclass
class AuctionOutcome:
    winner_id: int
    winning_bid: int
    gems_won: list[str]
    coins_received: int      # loans only
    investment_locked: int   # investments only


def resolve_auction(game_state: GameState, bids: list[int]) -> AuctionOutcome:
    auction = game_state.current_auction
    if auction is None:
        raise ValueError("No auction in progress")

    max_bid = max(bids)
    tied_players = [i for i, b in enumerate(bids) if b == max_bid]
    winner_id = min(tied_players, key=lambda p: game_state.tiebreak_order.index(p))
    winning_bid = bids[winner_id]

    gems_won: list[str] = []
    coins_received = 0
    investment_locked = 0

    if auction.type == AuctionType.TREASURE:
        # End-of-game edge case: 2-gem auction with only 1 revealed gem left.
        num_gems = min(auction.gems, len(game_state.revealed_gems))
        gems_won = game_state.revealed_gems[:num_gems]
    elif auction.type == AuctionType.LOAN:
        coins_received = auction.amount
    elif auction.type == AuctionType.INVESTMENT:
        investment_locked = winning_bid

    return AuctionOutcome(
        winner_id=winner_id,
        winning_bid=winning_bid,
        gems_won=gems_won,
        coins_received=coins_received,
        investment_locked=investment_locked,
    )


def apply_auction_outcome(
    game_state: GameState,
    outcome: AuctionOutcome,
    bids: list[int],
    gem_revealed: str | None = None,
) -> AuctionResult:
    auction = game_state.current_auction
    if auction is None:
        raise ValueError("No auction in progress")

    player = game_state.players[outcome.winner_id]

    if auction.type == AuctionType.LOAN:
        # Add loan amount before deducting bid so coins never transiently go
        # negative; validation guarantees bid <= coins + loan_amount.
        player.coins += outcome.coins_received
        player.coins -= outcome.winning_bid
        player.loans.append(auction.amount)

    elif auction.type == AuctionType.TREASURE:
        player.coins -= outcome.winning_bid
        player.collection.extend(outcome.gems_won)
        for gem in outcome.gems_won:
            if gem in game_state.revealed_gems:
                game_state.revealed_gems.remove(gem)
        game_state.replenish_revealed_gems()

    elif auction.type == AuctionType.INVESTMENT:
        player.coins -= outcome.winning_bid
        player.investments.append(Investment(bid_amount=outcome.investment_locked, bonus=auction.bonus))

    result = AuctionResult(
        round_number=game_state.round_number,
        auction_card=auction.to_dict(),
        bids=list(bids),
        winner_id=outcome.winner_id,
        winning_bid=outcome.winning_bid,
        gems_won=list(outcome.gems_won),
        gem_revealed=gem_revealed,
    )
    game_state.auction_history.append(result)
    game_state.update_tiebreak_order(outcome.winner_id)
    game_state.current_auction = None
    return result


def reveal_gem_from_hand(game_state: GameState, player_id: int, gem_color: str) -> bool:
    """Move gem_color from player's hand to the value display; False if not in hand."""
    player = game_state.players[player_id]
    if gem_color not in player.hand:
        return False
    player.hand.remove(gem_color)
    game_state.value_display.append(gem_color)
    return True


def check_mission_completion(game_state: GameState, player_id: int, mission_id: str) -> tuple[bool, str]:
    """(success, error_message)."""
    player = game_state.players[player_id]

    mission = None
    for m in game_state.available_missions:
        if m.id == mission_id:
            mission = m
            break

    if mission is None:
        return False, f"Mission {mission_id} not available"
    if mission_id in player.completed_missions:
        return False, "Already completed this mission"
    if not mission.check_completion(player.collection):
        return False, "Collection does not meet mission requirements"

    return True, ""


def complete_mission(game_state: GameState, player_id: int, mission_id: str) -> bool:
    can_complete, _error = check_mission_completion(game_state, player_id, mission_id)
    if not can_complete:
        return False

    player = game_state.players[player_id]
    for i, m in enumerate(game_state.available_missions):
        if m.id == mission_id:
            player.completed_missions.append(mission_id)
            game_state.available_missions.pop(i)
            return True
    return False


def score_components(
    *,
    coins: int,
    collection_counts: dict[str, int],
    completed_mission_rewards: int,
    loan_total: int,
    investment_return_total: int,
    gem_value_fn: Callable[[str], int],
) -> dict[str, Any]:
    """Pure scoring arithmetic shared by the engine and the offline RL scorer.

    No game state, no side effects, no opponent-hand access. ``gem_value_fn``
    maps a gem color to its per-gem value at the state being scored (mid-game:
    the round's current value-display count; terminal: the post-reveal count).
    This is the single source of truth for ``final_score``; see the RL plan §A.4.
    """
    gem_value = 0
    gem_breakdown: dict[str, dict[str, int]] = {}
    for color, count in collection_counts.items():
        if count <= 0:
            continue
        value_per_gem = gem_value_fn(color)
        total = count * value_per_gem
        gem_value += total
        gem_breakdown[color] = {
            "count": count,
            "value_per_gem": value_per_gem,
            "total": total,
        }

    final_score = (
        coins
        + gem_value
        + completed_mission_rewards
        - loan_total
        + investment_return_total
    )

    return {
        "coins": coins,
        "gem_value": gem_value,
        "gem_breakdown": gem_breakdown,
        "mission_rewards": completed_mission_rewards,
        "loan_payments": loan_total,
        "investment_returns": investment_return_total,
        "final_score": final_score,
    }


def calculate_final_score(game_state: GameState, player_id: int) -> dict[str, Any]:
    """Player score breakdown: coins + gem_value + missions − loans + investments.

    Thin, behavior-preserving wrapper over :func:`score_components`.
    """
    player = game_state.players[player_id]

    collection_counts = {
        color: player.collection.count(color) for color in set(player.collection)
    }

    mission_rewards = 0
    for mission_id in player.completed_missions:
        for m in game_state.all_missions:
            if m.id == mission_id:
                mission_rewards += m.reward
                break

    components = score_components(
        coins=player.coins,
        collection_counts=collection_counts,
        completed_mission_rewards=mission_rewards,
        loan_total=sum(player.loans),
        investment_return_total=sum(inv.total_return for inv in player.investments),
        gem_value_fn=game_state.get_gem_value,
    )

    return {
        "player_id": player_id,
        "coins": components["coins"],
        "gem_value": components["gem_value"],
        "gem_breakdown": components["gem_breakdown"],
        "mission_rewards": components["mission_rewards"],
        "completed_missions": list(player.completed_missions),
        "loan_payments": components["loan_payments"],
        "investment_returns": components["investment_returns"],
        "final_score": components["final_score"],
    }


def calculate_all_final_scores(game_state: GameState) -> list[dict[str, Any]]:
    """Reveals all hands (side-effects the value display), then scores everyone."""
    game_state.reveal_remaining_hands()
    return [calculate_final_score(game_state, pid) for pid in range(game_state.num_players)]


def determine_winner(game_state: GameState) -> tuple[int, list[dict[str, Any]]]:
    """(winner_id, all_scores); ties broken by tiebreak_order."""
    scores = calculate_all_final_scores(game_state)
    max_score = max(s["final_score"] for s in scores)
    tied_players = [s["player_id"] for s in scores if s["final_score"] == max_score]
    winner_id = min(tied_players, key=lambda p: game_state.tiebreak_order.index(p))
    return winner_id, scores
