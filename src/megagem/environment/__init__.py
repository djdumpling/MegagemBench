"""MegaGem environment modules."""

from .prompts import generate_bid_prompt, generate_reveal_prompt

__all__ = [
    "MegaGemEnv",
    "generate_bid_prompt",
    "generate_reveal_prompt",
    "reward_final_score",
    "reward_normalized_rank",
    "reward_winner",
]


def __getattr__(name: str):
    if name == "MegaGemEnv":
        from .multi_agent_env import MegaGemEnv

        return MegaGemEnv
    if name in {"reward_final_score", "reward_normalized_rank", "reward_winner"}:
        from . import rewards

        return getattr(rewards, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
