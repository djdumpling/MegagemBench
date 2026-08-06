"""
MegaGem RL self-play environment for the verifiers framework.

Wraps the 3-player MegaGem auction game as a *single-agent* multi-turn
environment (formerly the standalone ``environments/megagem`` package).
Player 0 is the training agent; Players 1 & 2 are opponents served by the
same vLLM instance. The verifiers/prime-rl training path built on this view
was retired in favor of the staged TRL pipeline (see docs/training.md); the
wrapper is kept for evaluation via the verifiers/prime tooling and as a
reference single-agent formulation.

The multi-agent environment used everywhere else in this repo is
``megagem.environment.multi_agent_env.MegaGemEnv`` (via
``megagem.load_environment``).
"""

import asyncio
import logging
import random

import verifiers as vf
from datasets import Dataset

from megagem.data import load_auctions, load_gems, load_missions, load_value_charts
from megagem.environment.llm_response import reconstruct_content
from megagem.environment.prompts import (
    generate_bid_prompt,
    generate_reveal_prompt,
    generate_system_prompt,
)
from megagem.environment.rewards import (
    reward_final_score,
    reward_gems_collected,
    reward_missions_completed,
    reward_winner,
)
from megagem.game.actions import (
    get_default_bid,
    get_default_reveal,
    parse_bid,
    parse_reveal,
    validate_bid_for_auction,
)
from megagem.game.cards import AuctionType, ValueChart
from megagem.game.rules import (
    apply_auction_outcome,
    complete_mission,
    determine_winner,
    resolve_auction,
    reveal_gem_from_hand,
)
from megagem.game.state import GameState

logger = logging.getLogger(__name__)

NUM_PLAYERS = 3
MAX_ROUNDS = 25


# Shared with the multi-agent env and the human-play loop so every loop reads
# thinking-model output identically.
_extract_content = reconstruct_content


def _message_content(message) -> str:
    """Read content from either a verifiers message or a legacy message dict."""
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content or "")


async def _client_completion(state, messages):
    """Generate through the provider-agnostic verifiers client interface."""
    client = state.get("client")
    if client is None:
        return None

    sampling_args = state.get("sampling_args") or {}
    if hasattr(sampling_args, "model_dump"):
        sampling_args = sampling_args.model_dump(exclude_none=True)
    else:
        sampling_args = dict(sampling_args)
    sampling_args.setdefault("max_tokens", 4096)
    sampling_args.setdefault("temperature", 1.0)

    response = await client.get_response(
        prompt=messages,
        model=state.get("model", ""),
        sampling_args=sampling_args,
        state=state,
    )
    return response.message


class MegaGemRLEnv(vf.MultiTurnEnv):
    """
    Single-agent view of MegaGem for RL self-play.

    Each game round is one multi-turn step:
    - Model generates Player 0's bid (reasoning + JSON)
    - Environment collects opponent bids, resolves auction, returns next prompt
    - If Player 0 wins a treasure auction, an extra turn for gem reveal
    """

    def __init__(self, num_seeds=30, value_chart_id="A", **kwargs):
        self.value_chart_id = value_chart_id

        # Load game data
        self.gem_cards = load_gems()
        self.auction_cards = load_auctions()
        self.missions = load_missions()
        self.value_charts = load_value_charts()
        chart_data = self.value_charts[value_chart_id]
        self.value_chart = ValueChart.from_dict(value_chart_id, chart_data)

        # Dataset: each example is a unique game seed
        dataset = Dataset.from_list([
            {
                "prompt": [{"role": "user", "content": f"MegaGem game seed {seed}"}],
                "answer": "",
                "task": "megagem",
                "info": {"seed": seed, "value_chart_id": value_chart_id},
            }
            for seed in range(1, num_seeds + 1)
        ])

        # Rubric: outcome-dominant. final_score is the game objective; missions
        # and gems are kept as small auxiliary terms because they decorrelate
        # from final_score during early-game partial credit (denser signal mid
        # rollout) and target the biggest measured skill gaps (missions ~7x,
        # gems ~2x). winner kept tiny as a tiebreak. score_margin dropped: in
        # symmetric self-play it is a centered version of final_score and adds
        # nothing on top of GRPO's within-group advantage normalization.
        async def combined_reward(state, **kw):
            return (
                0.05 * reward_winner(state, player_id=0)
                + 0.70 * reward_final_score(state, player_id=0)
                + 0.15 * reward_missions_completed(state, player_id=0)
                + 0.10 * reward_gems_collected(state, player_id=0)
            )

        rubric = vf.Rubric(funcs=[combined_reward], weights=[1.0])

        kwargs.setdefault("max_turns", MAX_ROUNDS * 2)
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            system_prompt=generate_system_prompt(),
            **kwargs,
        )

    def _create_game_state(self, seed):
        return GameState.create_new_game(
            num_players=NUM_PLAYERS,
            gem_cards=self.gem_cards,
            auction_cards=self.auction_cards,
            missions=self.missions,
            value_chart=self.value_chart,
            seed=seed,
        )

    async def setup_state(self, state):
        info = state.get("info", {})
        seed = info.get("seed", 42)

        game_state = self._create_game_state(seed)
        state["game_state"] = game_state
        state["current_phase"] = "bid"
        state["game_over"] = False

        auction = game_state.draw_auction_card()
        if auction is None:
            state["game_over"] = True
            state["final_env_response"] = []
            return state

        state["next_prompt"] = generate_bid_prompt(game_state, player_id=0)
        state["prompt"] = [{"role": "user", "content": state["next_prompt"]}]
        return state

    async def env_response(self, messages, state):
        game_state = state["game_state"]
        phase = state["current_phase"]

        if phase == "bid":
            return await self._handle_bid(messages, state, game_state)
        if phase == "reveal":
            return await self._handle_reveal(messages, state, game_state)
        raise ValueError(f"Unknown phase: {phase}")

    async def _handle_bid(self, messages, state, game_state):
        # Parse Player 0's bid
        last_msg = _message_content(messages[-1]) if messages else ""
        parsed = parse_bid(last_msg)
        if parsed.valid:
            valid, _ = validate_bid_for_auction(game_state, 0, parsed.bid)
            p0_bid = parsed.bid if valid else get_default_bid()
        else:
            p0_bid = get_default_bid()

        # Get opponent bids
        opp_bids = await self._get_opponent_bids(state, game_state)
        all_bids = [p0_bid] + opp_bids

        # Resolve auction
        outcome = resolve_auction(game_state, all_bids)
        winner_id = outcome.winner_id
        auction = game_state.current_auction
        gem_revealed = None

        if auction and auction.type == AuctionType.TREASURE:
            if winner_id == 0:
                # Player 0 won — need their reveal action next turn
                state["pending_outcome"] = outcome
                state["pending_bids"] = all_bids
                state["current_phase"] = "reveal"
                prompt = generate_reveal_prompt(game_state, player_id=0)
                return [{"role": "user", "content": prompt}]
            gem_revealed = await self._get_opponent_reveal(state, game_state, winner_id)

        # Apply outcome
        if gem_revealed:
            reveal_gem_from_hand(game_state, winner_id, gem_revealed)
        apply_auction_outcome(game_state, outcome, all_bids, gem_revealed)

        # Missions
        if auction and auction.type == AuctionType.TREASURE and game_state.available_missions:
            for mission in list(game_state.available_missions):
                complete_mission(game_state, winner_id, mission.id)

        return await self._advance(state, game_state)

    async def _handle_reveal(self, messages, state, game_state):
        last_msg = _message_content(messages[-1]) if messages else ""
        player = game_state.players[0]
        parsed = parse_reveal(last_msg)
        if parsed.valid and parsed.gem_color in player.hand:
            gem_revealed = parsed.gem_color
        else:
            gem_revealed = get_default_reveal(player.hand)

        outcome = state.pop("pending_outcome")
        all_bids = state.pop("pending_bids")

        if gem_revealed:
            reveal_gem_from_hand(game_state, 0, gem_revealed)
        apply_auction_outcome(game_state, outcome, all_bids, gem_revealed)

        if game_state.available_missions:
            for mission in list(game_state.available_missions):
                complete_mission(game_state, 0, mission.id)

        state["current_phase"] = "bid"
        return await self._advance(state, game_state)

    async def _advance(self, state, game_state):
        # Check game over (shared with every other game loop).
        if game_state.all_gems_won():
            game_state.game_over = True

        if game_state.is_game_over():
            winner_id, final_scores = determine_winner(game_state)
            state["winner_id"] = winner_id
            state["final_scores"] = final_scores
            state["game_over"] = True
            state["final_env_response"] = []
            return []

        auction = game_state.draw_auction_card()
        if auction is None:
            game_state.game_over = True
            winner_id, final_scores = determine_winner(game_state)
            state["winner_id"] = winner_id
            state["final_scores"] = final_scores
            state["game_over"] = True
            state["final_env_response"] = []
            return []

        prompt = generate_bid_prompt(game_state, player_id=0)
        state["current_phase"] = "bid"
        return [{"role": "user", "content": prompt}]

    async def _get_opponent_bids(self, state, game_state):
        if state.get("client") is None:
            return [random.randint(0, min(5, game_state.players[pid].coins)) for pid in [1, 2]]

        async def get_bid(player_id):
            prompt = generate_bid_prompt(game_state, player_id)
            msgs = [
                {"role": "system", "content": generate_system_prompt()},
                {"role": "user", "content": prompt},
            ]
            try:
                message = await _client_completion(state, msgs)
                parsed = parse_bid(_extract_content(message))
                if parsed.valid:
                    valid, _ = validate_bid_for_auction(game_state, player_id, parsed.bid)
                    return parsed.bid if valid else get_default_bid()
                return get_default_bid()
            except Exception as e:
                logger.warning("Opponent %d bid failed: %s", player_id, e, exc_info=True)
                return get_default_bid()

        bids = await asyncio.gather(get_bid(1), get_bid(2))
        return list(bids)

    async def _get_opponent_reveal(self, state, game_state, winner_id):
        player = game_state.players[winner_id]

        if not player.hand:
            return None
        if state.get("client") is None:
            return get_default_reveal(player.hand)

        prompt = generate_reveal_prompt(game_state, winner_id)
        msgs = [
            {"role": "system", "content": generate_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        try:
            message = await _client_completion(state, msgs)
            parsed = parse_reveal(_extract_content(message))
            if parsed.valid and parsed.gem_color in player.hand:
                return parsed.gem_color
            return get_default_reveal(player.hand)
        except Exception as e:
            logger.warning("Opponent %d reveal failed: %s", winner_id, e, exc_info=True)
            return get_default_reveal(player.hand)

def load_environment(num_seeds=30, value_chart_id="A", **kwargs):
    """Entry point for the verifiers framework."""
    return MegaGemRLEnv(num_seeds=num_seeds, value_chart_id=value_chart_id, **kwargs)
