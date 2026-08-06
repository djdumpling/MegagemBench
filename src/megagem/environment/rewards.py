"""Reward functions for MegaGem."""

from verifiers.types import State


def reward_winner(state: State, player_id: int = 0, **kwargs) -> float:
    """1.0 if player_id won; 0.0 otherwise."""
    return 1.0 if state.get("winner_id", -1) == player_id else 0.0


def reward_final_score(state: State, player_id: int = 0, **kwargs) -> float:
    """Player score as a fraction of the max score in the game; 0 if no scores."""
    final_scores = state.get("final_scores", [])
    if not final_scores:
        return 0.0
    max_score = max(s["final_score"] for s in final_scores)
    if max_score <= 0:
        return 0.0
    player_score = next(
        (s["final_score"] for s in final_scores if s["player_id"] == player_id),
        0,
    )
    return max(0.0, player_score / max_score)


def reward_normalized_rank(state: State, player_id: int = 0, **kwargs) -> float:
    """Rank-based reward in [0, 1] — 1.0 for 1st, 0.0 for last."""
    final_scores = state.get("final_scores", [])
    if not final_scores:
        return 0.0
    sorted_scores = sorted(final_scores, key=lambda s: s["final_score"], reverse=True)
    rank = next(
        (i for i, s in enumerate(sorted_scores) if s["player_id"] == player_id),
        -1,
    )
    if rank < 0:
        return 0.0
    n = len(final_scores)
    if n <= 1:
        return 1.0
    return (n - 1 - rank) / (n - 1)


def reward_missions_completed(state: State, player_id: int = 0, **kwargs) -> float:
    """Fraction of the 4 available missions the player completed."""
    final_scores = state.get("final_scores", [])
    for s in final_scores:
        if s["player_id"] == player_id:
            return min(1.0, len(s.get("completed_missions", [])) / 4)
    return 0.0


def reward_gems_collected(state: State, player_id: int = 0, **kwargs) -> float:
    """Player gem value as a fraction of the max gem value in the game."""
    final_scores = state.get("final_scores", [])
    if not final_scores:
        return 0.0
    gem_values = [s.get("gem_value", 0) for s in final_scores]
    max_gem_value = max(gem_values) if gem_values else 0
    if max_gem_value <= 0:
        return 0.0
    for s in final_scores:
        if s["player_id"] == player_id:
            return s.get("gem_value", 0) / max_gem_value
    return 0.0
