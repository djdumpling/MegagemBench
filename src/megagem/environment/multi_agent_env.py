"""Multi-agent environment for MegaGem."""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from datasets import Dataset
from openai import AsyncOpenAI

from megagem.environment.llm_response import (
    API_TRANSIENT_ERRORS,
    reconstruct_content,
)
from verifiers.envs.environment import Environment
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Info, Messages, SamplingArgs, State
try:
    from verifiers.types import TextMessage as ChatMessage
except ImportError:
    from verifiers.types import ChatMessage

from ..data import load_auctions, load_gems, load_missions, load_value_charts
from ..game.actions import (
    get_default_bid,
    get_default_reveal,
    length_split,
    parse_bid,
    parse_reveal,
    validate_bid_for_auction,
    validate_reveal_for_hand,
)
from ..game.cards import AuctionType, ValueChart
from ..game.rules import (
    apply_auction_outcome,
    complete_mission,
    determine_winner,
    resolve_auction,
    reveal_gem_from_hand,
)
from ..game.state import GameState
from .prompts import (
    generate_bid_prompt,
    generate_reveal_prompt,
    generate_system_prompt,
)

logger = logging.getLogger(__name__)


DEFAULT_ACTOR_ID = "trainable"

# --- API resilience: transient-failure retry with header-aware backoff ------
# A single 429 from a rate-limited API seat (an opponent in training, or the
# Flash held-out eval gate) must never abort a whole roll — that would burn the
# multi-hour H100/H200 job over one transient hiccup. We retry transient errors
# with a wait that prefers the server's own timing hints, and on exhaustion fall
# back to an empty response: `parse_bid`/`parse_reveal` already turn "" into a
# safe default action, so the game completes instead of crashing. (The local
# vLLM trainable seat does not rate-limit; this protects the external-API seats.)
_API_MAX_RETRIES = 5
_API_BASE_BACKOFF_S = 1.0
_API_MAX_BACKOFF_S = 30.0
_API_TRANSIENT_ERRORS = API_TRANSIENT_ERRORS


def _api_retry_wait_s(exc: Exception, attempt: int) -> float:
    """Seconds to wait before retry `attempt` (0-indexed) of a transient API
    error. Prefers the server's timing headers (`Retry-After` seconds,
    `retry-after-ms`, or this gateway's absolute-unix-ms `X-RateLimit-Reset`),
    each capped at `_API_MAX_BACKOFF_S`; falls back to exponential backoff."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    ra = headers.get("retry-after")
    if ra:
        try:
            return min(max(0.0, float(ra)), _API_MAX_BACKOFF_S)
        except (TypeError, ValueError):
            pass
    ra_ms = headers.get("retry-after-ms")
    if ra_ms:
        try:
            return min(max(0.0, float(ra_ms) / 1000.0), _API_MAX_BACKOFF_S)
        except (TypeError, ValueError):
            pass
    reset = headers.get("x-ratelimit-reset")
    if reset:
        try:
            delta = float(reset) / 1000.0 - time.time()
            if delta > 0:
                return min(delta, _API_MAX_BACKOFF_S)
        except (TypeError, ValueError):
            pass
    return min(_API_BASE_BACKOFF_S * (2 ** attempt), _API_MAX_BACKOFF_S)


@dataclass
class BidTurnRecord:
    """Per-player record for one bid turn — full parse chain plus the env's final bid."""

    player_id: int
    actor_id: str
    prompt: str
    raw_response: str
    parsed_action: int | None
    parse_method: str
    parse_valid: bool
    legal_valid: bool
    default_used: bool
    final_bid: int
    reasoning: str
    length_split: dict[str, int] = field(default_factory=dict)
    parse_error: str = ""
    legal_error: str = ""


@dataclass
class RevealTurnRecord:
    """Per-player record for one gem-reveal turn (treasure-auction winner only)."""

    player_id: int
    actor_id: str
    prompt: str
    raw_response: str
    parsed_action: str | None
    parse_method: str
    parse_valid: bool
    legal_valid: bool
    default_used: bool
    final_reveal: str
    reasoning: str
    length_split: dict[str, int] = field(default_factory=dict)
    parse_error: str = ""
    legal_error: str = ""


class MegaGemEnv(Environment):
    """
    Multi-agent environment for MegaGem.

    This environment handles 3 LLM players competing in an auction game
    with private information (gem hands).
    """

    def __init__(
        self,
        num_players: int = 3,
        value_chart_id: str = "A",
        seed: int = 42,
        max_retries: int = 3,
        extra_api_params: dict | None = None,
        per_seat_api_params: list[dict | None] | None = None,
        fail_on_api_default: bool = False,
        fail_on_default_action: bool = False,
        **kwargs,
    ):
        self.num_players = num_players
        self.value_chart_id = value_chart_id
        self.seed = seed
        self.max_retries = max_retries
        # Extra params forwarded to every chat.completions.create() call.
        # Useful for vLLM-specific options like disabling thinking mode:
        #   extra_api_params={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        self.extra_api_params = extra_api_params or {}
        # Optional PER-SEAT override of extra_api_params (Phase-3 §3.3 opponent
        # pool): a list of length num_players. The trainable seat is sampled
        # (§A.7 T=1.0) while pooled opponents play greedy (T=0) so they stay
        # deterministic within a K-group. A None entry — or the whole list
        # being None — falls back to extra_api_params (unchanged behaviour).
        self.per_seat_api_params = per_seat_api_params
        # Opt-in fail-closed mode for label/eval jobs where a parser/default
        # fallback would silently corrupt downstream Q labels.
        self.fail_on_api_default = fail_on_api_default
        self.fail_on_default_action = fail_on_default_action
        # Optional test-time piKL search (set via set_pikl_searcher /
        # set_pikl_bid_searcher). None on every training/normal path and inside
        # continuations → byte-identical to pre-piKL.
        self._pikl_searcher = None
        self._pikl_bid_searcher = None
        # Optional in-context value-head injection (Phase 2 eval only; never on the
        # training/continuation path). Applies to the configured policy seat(s) only —
        # opponents and the value-OFF control get byte-identical prompts to before.
        self._value_head = None
        self._value_head_seats: set[int] = set()
        # Optional value-coherent (fair-value) opponent seats: these seats bid
        # `fair_value_bid` (priced by the public display + their own hand) and reveal
        # the default colour, in-process and deterministic — no LLM. Empty ⇒ unchanged
        # behaviour. Used by the piKL value-coherent-opponent experiment so the bid
        # likelihood the informed belief inverts is correctly specified.
        self.fv_opponent_seats: set[int] = set()
        self.fv_opp_shade: float = 0.8
        self.opp_model: str = "fair_value"  # "fair_value" | "market" for fv_opponent_seats
        self.market_params: dict | None = None  # overrides MARKET_DEFAULTS when opp_model="market"

        # Load game data
        self.gem_cards = load_gems()
        self.auction_cards = load_auctions()
        self.missions = load_missions()
        self.value_charts = load_value_charts()

        # Get selected value chart
        chart_data = self.value_charts.get(value_chart_id)
        if chart_data is None:
            raise ValueError(f"Invalid value chart ID: {value_chart_id}")
        self.value_chart = ValueChart.from_dict(value_chart_id, chart_data)

        # Create a minimal dataset for compatibility
        # Each "example" represents one game instance
        dataset = Dataset.from_list(
            [
                {
                    "prompt": [{"role": "user", "content": "MegaGem game"}],
                    "answer": "",
                    "task": "megagem",
                    "info": {
                        "num_players": num_players,
                        "value_chart_id": value_chart_id,
                        "seed": seed,
                    },
                }
            ]
        )

        # Initialize rubric (will be set properly)
        rubric = kwargs.pop("rubric", Rubric())

        super().__init__(
            dataset=dataset,
            rubric=rubric,
            system_prompt=generate_system_prompt(),
            **kwargs,
        )

    def create_game_state(self, seed: int | None = None) -> GameState:
        """Create a new game state."""
        return GameState.create_new_game(
            num_players=self.num_players,
            gem_cards=self.gem_cards,
            auction_cards=self.auction_cards,
            missions=self.missions,
            value_chart=self.value_chart,
            seed=seed,
        )

    def _chat_messages(self, prompt: str) -> list[ChatMessage]:
        return [
            {"role": "system", "content": generate_system_prompt()},
            {"role": "user", "content": prompt},
        ]

    def _build_api_kwargs(
        self, sampling_args: SamplingArgs | None, player_id: int | None,
    ) -> dict:
        """Sampling args merged with this seat's extra params. A per-seat
        override (Phase-3 opponent pool) wins over the env-wide default; a None
        entry falls back to ``extra_api_params``. No max_tokens default — the
        backend uses remaining context (the 4B model needs ~9k)."""
        clean: dict = {}
        if sampling_args:
            clean = {k: v for k, v in sampling_args.items() if v is not None}
            if "max_tokens" in clean:
                clean["max_completion_tokens"] = clean.pop("max_tokens")
        extra = self.extra_api_params
        if (
            self.per_seat_api_params is not None
            and player_id is not None
            and 0 <= player_id < len(self.per_seat_api_params)
            and self.per_seat_api_params[player_id] is not None
        ):
            extra = self.per_seat_api_params[player_id]
        return {**clean, **extra}

    _reconstruct_content = staticmethod(reconstruct_content)

    async def get_player_response(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: str,
        sampling_args: SamplingArgs | None = None,
        player_id: int | None = None,
    ) -> str:
        """One LLM move. Transient failures (429/timeout/5xx) retry with
        header-aware backoff so one seat's hiccup can't abort the roll; on
        exhaustion return "" (→ parser default). Non-transient errors (400/401)
        propagate as real misconfiguration."""
        messages = self._chat_messages(prompt)
        api_kwargs = self._build_api_kwargs(sampling_args, player_id)
        last_exc: Exception | None = None
        for attempt in range(_API_MAX_RETRIES + 1):
            try:
                response = await client.chat.completions.create(
                    model=model, messages=messages, **api_kwargs,  # type: ignore
                )
                if response and response.choices and response.choices[0].message:
                    return self._reconstruct_content(response.choices[0].message)
                return ""
            except _API_TRANSIENT_ERRORS as e:
                last_exc = e
                if attempt >= _API_MAX_RETRIES:
                    break
                wait_s = _api_retry_wait_s(e, attempt)
                logger.warning(
                    "Transient API error from %s (attempt %d/%d): %s — retrying in %.1fs",
                    model, attempt + 1, _API_MAX_RETRIES, type(e).__name__, wait_s,
                )
                await asyncio.sleep(wait_s)
            except Exception as e:
                logger.error(f"Error getting player response: {e}")
                raise
        logger.error(
            "API seat %s exhausted %d retries on transient errors (last: %s) — "
            "returning empty response; parser will use a default action.",
            model, _API_MAX_RETRIES, last_exc,
        )
        if self.fail_on_api_default:
            raise RuntimeError(
                f"API seat {model} exhausted retries and fail_on_api_default=True"
            ) from last_exc
        return ""

    async def get_player_samples(
        self, client: AsyncOpenAI, model: str, prompt: str, *,
        n: int, temperature: float, player_id: int,
    ) -> list[str]:
        """N independent completions in one call for piKL τ̂ estimation (local
        vLLM only). Returns reconstructed texts.

        Failure policy (a silently-degraded τ̂ corrupts a whole sweep):
          * TRANSIENT errors (rate-limit / timeout / connection / 5xx) ⇒ warn and
            return [] — the caller falls back (τ̂→uniform / candidates→default),
            acceptable for a rare blip.
          * NON-TRANSIENT errors (auth, bad-request, an `n`/param the backend
            rejects, model-not-found, backend incompatibility) ⇒ RAISE. These
            would otherwise produce a valid-LOOKING but content-free sweep on every
            node; fail fast so the run aborts instead.
          * A SHORT return (fewer than `n` completions) is surfaced — τ̂ is then
            estimated on fewer samples and the var-screen should know."""
        api_kwargs = {**self._build_api_kwargs({"temperature": temperature}, player_id), "n": n}
        try:
            resp = await client.chat.completions.create(
                model=model, messages=self._chat_messages(prompt), **api_kwargs,  # type: ignore
            )
        except _API_TRANSIENT_ERRORS as e:
            if self.fail_on_api_default:
                raise RuntimeError(
                    "piKL sample call hit a transient API error and "
                    "fail_on_api_default=True"
                ) from e
            logger.warning(
                "piKL sample call hit a transient error (%s) — returning no samples; "
                "τ̂ will fall back to uniform / default this node.", type(e).__name__)
            return []
        except Exception as e:
            logger.error(
                "piKL sample call failed non-transiently (%s: %s) — RAISING rather "
                "than silently degrading τ̂ across the sweep.", type(e).__name__, e)
            raise
        texts = [self._reconstruct_content(c.message) for c in (resp.choices or []) if c.message]
        if len(texts) < n:
            logger.warning(
                "piKL sampling returned %d/%d completions (seat %s) — τ̂ estimated on "
                "fewer samples than requested.", len(texts), n, player_id)
        return texts

    async def run_bidding_phase(
        self,
        game_state: GameState,
        clients: list[AsyncOpenAI],
        models: list[str],
        sampling_args: SamplingArgs | None = None,
        model_chat_times: list[float] | None = None,
        actor_ids: list[str] | None = None,
        pikl_bid_searcher=None,
    ) -> tuple[list[int], list[BidTurnRecord]]:
        """Run the bidding phase for all players in parallel.

        Returns ``(bids, turn_records)``. ``bids`` is the post-sanitization
        list the env will use; ``turn_records`` carries each player's raw
        response, parse method, validity flags, and length split.
        ``actor_ids`` labels each player's turn (defaults to all-trainable).
        ``model_chat_times`` accumulates per-player chat-completion seconds.
        """
        if actor_ids is None:
            actor_ids = [DEFAULT_ACTOR_ID] * self.num_players

        def _bid_prompt(pid: int) -> str:
            vb = None
            if self._value_head is not None and pid in self._value_head_seats:
                try:
                    from ..value_head.inject import value_block
                    vb = value_block(game_state, pid, self._value_head)
                except Exception as exc:  # a value-head bug must never abort the game
                    logger.warning("value_head injection failed (seat %s): %s", pid, exc)
                    vb = None
            return generate_bid_prompt(game_state, pid, value_block=vb)

        prompts = [_bid_prompt(player_id) for player_id in range(self.num_players)]

        async def get_bid_with_timing(player_id: int) -> str | None:
            if player_id in self.fv_opponent_seats:
                return None  # value-coherent seat — bid set in-process below, no LLM
            t0 = time.perf_counter()
            response = await self.get_player_response(
                clients[player_id], models[player_id], prompts[player_id],
                sampling_args, player_id=player_id,
            )
            if model_chat_times is not None and player_id < len(model_chat_times):
                model_chat_times[player_id] += time.perf_counter() - t0
            return response

        responses = await asyncio.gather(*(get_bid_with_timing(i) for i in range(self.num_players)))

        bids: list[int] = []
        turn_records: list[BidTurnRecord] = []
        for player_id, response in enumerate(responses):
            if player_id in self.fv_opponent_seats:
                # Value-coherent opponent: deterministic in-process bid, no LLM. opp_model
                # selects the model — "fair_value" (public display + own hand) or "market"
                # (recent winning bids + budget + tiebreak; matches the blueprint's bidding).
                from .pikl_search import opponent_bid
                fv = opponent_bid(game_state, player_id, trainable_seat=player_id,
                                  model=self.opp_model, shade=self.fv_opp_shade,
                                  market_params=self.market_params)
                bids.append(fv)
                turn_records.append(BidTurnRecord(
                    player_id=player_id, actor_id=actor_ids[player_id],
                    prompt=prompts[player_id], raw_response=json.dumps({"bid": fv}),
                    parsed_action=fv,
                    parse_method=("market_anchored" if self.opp_model == "market" else "fair_value"),
                    parse_valid=True,
                    legal_valid=True, default_used=False, final_bid=fv, reasoning="",
                    length_split={}, parse_error="", legal_error=""))
                continue
            parsed = parse_bid(response)
            parse_valid = parsed.valid
            parse_error = parsed.error
            parsed_action = parsed.bid if parsed.valid else None

            legal_valid = False
            legal_error = ""
            default_used = False
            final_bid = get_default_bid()

            if parse_valid:
                legal_valid, legal_error = validate_bid_for_auction(
                    game_state, player_id, parsed.bid
                )
                if legal_valid:
                    final_bid = parsed.bid
                else:
                    default_used = True
                    logger.debug(
                        f"Player {player_id} invalid bid: {legal_error}. Using default."
                    )
            else:
                default_used = True
                logger.debug(
                    f"Player {player_id} parse error: {parse_error}. Using default."
                )
                logger.debug(
                    f"Player {player_id} raw response (first 500 chars): {response[:500]}"
                )

            if default_used and self.fail_on_default_action:
                raise RuntimeError(
                    "default bid action would be used with fail_on_default_action=True "
                    f"(player={player_id}, parse_valid={parse_valid}, "
                    f"legal_valid={legal_valid}, parse_error={parse_error!r}, "
                    f"legal_error={legal_error!r})"
                )

            bids.append(final_bid)
            turn_records.append(
                BidTurnRecord(
                    player_id=player_id,
                    actor_id=actor_ids[player_id],
                    prompt=prompts[player_id],
                    raw_response=response,
                    parsed_action=parsed_action,
                    parse_method=parsed.parse_method,
                    parse_valid=parse_valid,
                    legal_valid=legal_valid,
                    default_used=default_used,
                    final_bid=final_bid,
                    reasoning=parsed.reasoning,
                    length_split=length_split(response),
                    parse_error=parse_error,
                    legal_error=legal_error,
                )
            )

        # Test-time piKL bid search replaces the trainable seat's bid (and, in
        # var-screen mode, records diagnostics while keeping a blueprint-sampled
        # bid). Never set inside continuations → no recursion.
        if pikl_bid_searcher is not None and pikl_bid_searcher.wants_auction(game_state):
            for seat in range(self.num_players):
                if pikl_bid_searcher.is_pikl_seat(seat):
                    if getattr(pikl_bid_searcher, "wants_presampled", False):
                        # ev_dist selector: hand in the normal-path sample so a
                        # gated-off decision returns it untouched (treatment ==
                        # control wherever the gate doesn't fire, 0 extra calls).
                        bids[seat], turn_records[seat] = await pikl_bid_searcher.search(
                            game_state, seat,
                            presampled=(bids[seat], turn_records[seat]))
                    else:
                        bids[seat], turn_records[seat] = await pikl_bid_searcher.search(
                            game_state, seat)

        return bids, turn_records

    async def run_reveal_phase(
        self,
        game_state: GameState,
        winner_id: int,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        actor_id: str = DEFAULT_ACTOR_ID,
        *,
        pikl_searcher=None,
        outcome=None,
        bids: list[int] | None = None,
    ) -> tuple[str | None, RevealTurnRecord | None]:
        """Run the gem reveal phase for the auction winner.

        Returns ``(gem_revealed, turn_record)``. Both are ``None`` when the
        winner has no cards in hand. parse-validity and legal-validity are
        tracked separately on ``turn_record``. When a ``pikl_searcher`` is
        configured for this seat, depth-1 piKL search replaces the single call.
        """
        player = game_state.players[winner_id]

        if not player.hand:
            return None, None

        if (
            pikl_searcher is not None
            and pikl_searcher.is_pikl_seat(winner_id)
            and outcome is not None
            and bids is not None
        ):
            return await pikl_searcher.search(game_state, winner_id, outcome, bids)

        if winner_id in self.fv_opponent_seats:
            # Value-coherent opponent: deterministic default (lexicographic) reveal, no LLM.
            fv_reveal = get_default_reveal(player.hand)
            return fv_reveal, RevealTurnRecord(
                player_id=winner_id, actor_id=actor_id,
                prompt=generate_reveal_prompt(game_state, winner_id),
                raw_response=json.dumps({"reveal": fv_reveal}), parsed_action=fv_reveal,
                parse_method="fair_value", parse_valid=True, legal_valid=True,
                default_used=False, final_reveal=fv_reveal, reasoning="", length_split={},
                parse_error="", legal_error="")

        prompt = generate_reveal_prompt(game_state, winner_id)
        response = await self.get_player_response(
            client, model, prompt, sampling_args, player_id=winner_id)

        parsed = parse_reveal(response)
        parse_valid = parsed.valid
        parse_error = parsed.error
        parsed_action = parsed.gem_color if parsed.valid else None

        legal_valid = False
        legal_error = ""
        default_used = False
        if parse_valid:
            legal_valid, legal_error = validate_reveal_for_hand(
                player.hand, parsed.gem_color
            )
            if legal_valid:
                final_reveal = parsed.gem_color
            else:
                default_used = True
                final_reveal = get_default_reveal(player.hand)
                logger.debug(
                    f"Player {winner_id} reveal illegal: {legal_error}. Using default."
                )
        else:
            default_used = True
            final_reveal = get_default_reveal(player.hand)
            logger.debug(
                f"Player {winner_id} reveal parse error: {parse_error}. Using default."
            )

        if default_used and self.fail_on_default_action:
            raise RuntimeError(
                "default reveal action would be used with fail_on_default_action=True "
                f"(player={winner_id}, parse_valid={parse_valid}, "
                f"legal_valid={legal_valid}, parse_error={parse_error!r}, "
                f"legal_error={legal_error!r})"
            )

        record = RevealTurnRecord(
            player_id=winner_id,
            actor_id=actor_id,
            prompt=prompt,
            raw_response=response,
            parsed_action=parsed_action,
            parse_method=parsed.parse_method,
            parse_valid=parse_valid,
            legal_valid=legal_valid,
            default_used=default_used,
            final_reveal=final_reveal,
            reasoning=parsed.reasoning,
            length_split=length_split(response),
            parse_error=parse_error,
            legal_error=legal_error,
        )

        return final_reveal, record

    def run_mission_phase(
        self,
        game_state: GameState,
        player_id: int,
    ) -> list[str]:
        """
        Automatically verify and complete missions for a player after winning treasure auctions.
        Missions are verified automatically - the LLM does not need to output anything about missions.
        Returns list of completed mission IDs.
        """
        if not game_state.available_missions:
            return []

        # Automatically check and complete all missions the player qualifies for
        completed = []
        
        # Check each available mission (use list() to avoid modification during iteration)
        for mission in list(game_state.available_missions):
            # complete_mission already checks if mission is available and player qualifies
            if complete_mission(game_state, player_id, mission.id):
                completed.append(mission.id)

        return completed

    def _maybe_end_game(self, game_state: GameState) -> None:
        """End the game once every dealt gem has been won (3p: 15, 4p: 16,
        5p: 15) or the deck and revealed pool are both empty.

        The condition itself lives on ``GameState`` so every game loop shares it.
        """
        if game_state.all_gems_won():
            game_state.game_over = True

    async def _run_game_loop(
        self,
        game_state: GameState,
        clients: list[AsyncOpenAI],
        models: list[str],
        sampling_args: SamplingArgs | None,
        actor_ids: list[str],
        *,
        player_messages: list[list[dict]],
        game_events: list[dict],
        model_chat_times: list[float],
        round_num: int = 0,
        round_callback: Callable | None = None,
        pikl_searcher=None,
        pikl_bid_searcher=None,
    ) -> int:
        """One round per iteration to terminal; returns the final round number.
        Shared verbatim by ``rollout`` (full game) and the piKL continuation
        (mid-game resume) so play is byte-identical. ``pikl_searcher`` /
        ``pikl_bid_searcher`` are None on every normal/training path and inside
        continuations (recursion guard)."""
        while not game_state.is_game_over():
            auction = game_state.draw_auction_card()
            if auction is None:
                break

            round_num += 1
            logger.debug(f"Round {round_num}: {auction.get_description()}")

            revealed_gems_before_auction = list(game_state.revealed_gems) if game_state.revealed_gems else []
            # Decision-time tiebreak order — the order players bid under, BEFORE
            # apply_auction_outcome() moves the winner to the back. The §A.5 RL
            # role baseline must key on this, not the post-resolution order.
            tiebreak_order_before = list(game_state.tiebreak_order)

            bids, bid_turn_records = await self.run_bidding_phase(
                game_state, clients, models, sampling_args,
                model_chat_times=model_chat_times, actor_ids=actor_ids,
                pikl_bid_searcher=pikl_bid_searcher,
            )
            for record in bid_turn_records:
                player_messages[record.player_id].append(
                    {"phase": "bid", "round": round_num, **asdict(record)}
                )

            outcome = resolve_auction(game_state, bids)
            winner_id = outcome.winner_id
            logger.debug(f"Auction winner: Player {winner_id} with bid {outcome.winning_bid}")

            gem_revealed: str | None = None
            reveal_record: RevealTurnRecord | None = None
            if auction.type == AuctionType.TREASURE:
                gem_revealed, reveal_record = await self.run_reveal_phase(
                    game_state, winner_id, clients[winner_id], models[winner_id],
                    sampling_args, actor_id=actor_ids[winner_id],
                    pikl_searcher=pikl_searcher, outcome=outcome, bids=bids,
                )

            reveal_response = reveal_record.raw_response if reveal_record is not None else ""
            reveal_reasoning = reveal_record.reasoning if reveal_record is not None else ""
            if reveal_record is not None:
                player_messages[winner_id].append(
                    {"phase": "reveal", "round": round_num, **asdict(reveal_record)}
                )

            coins_before = [player.coins for player in game_state.players]
            if gem_revealed:
                reveal_gem_from_hand(game_state, winner_id, gem_revealed)
            apply_auction_outcome(game_state, outcome, bids, gem_revealed)

            missions_completed_this_round: dict[int, list[str]] = {}
            if auction.type == AuctionType.TREASURE and game_state.available_missions:
                completed = self.run_mission_phase(game_state, winner_id)
                if completed:
                    missions_completed_this_round[winner_id] = completed
                    player_messages[winner_id].append(
                        {"phase": "mission", "round": round_num, "completed_missions": completed}
                    )

            self._maybe_end_game(game_state)

            game_events.append(
                {
                    "round": round_num,
                    "auction": auction.to_dict(),
                    "bids": bids,
                    "winner_id": winner_id,
                    "winning_bid": outcome.winning_bid,
                    "gems_won": outcome.gems_won,
                    "gem_revealed": gem_revealed,
                }
            )

            if round_callback:
                round_callback(
                    round_num=round_num,
                    auction=auction,
                    bids=bids,
                    bid_reasoning=[r.reasoning for r in bid_turn_records],
                    winner_id=winner_id,
                    winning_bid=outcome.winning_bid,
                    gems_won=outcome.gems_won,
                    revealed_gems_available=revealed_gems_before_auction,
                    gem_revealed=gem_revealed,
                    reveal_response=reveal_response if gem_revealed else None,
                    reveal_reasoning=reveal_reasoning if gem_revealed else None,
                    missions_completed=missions_completed_this_round,
                    mission_reasoning={},
                    game_state=game_state,
                    models=models,
                    coins_before=coins_before,
                    tiebreak_order_before=tiebreak_order_before,
                    bid_turn_records=bid_turn_records,
                    reveal_turn_record=reveal_record,
                    actor_ids=actor_ids,
                )

        return round_num

    def set_pikl_searcher(self, searcher) -> None:
        """Install a test-time piKL reveal searcher (eval only; never set on the
        training/continuation path)."""
        self._pikl_searcher = searcher

    def set_pikl_bid_searcher(self, searcher) -> None:
        """Install a test-time piKL bid searcher (eval only; never set on the
        training/continuation path)."""
        self._pikl_bid_searcher = searcher

    def set_value_head(self, estimator, seats) -> None:
        """Install in-context value-head injection for the given policy seat(s)
        (eval only). ``estimator`` is a value_head.ValueEstimator; ``seats`` an
        iterable of seat indices that receive the value block in their bid prompt."""
        self._value_head = estimator
        self._value_head_seats = {int(s) for s in seats}

    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        completion: Messages | None = None,
        answer: str = "",
        state: State | None = None,
        task: str = "default",
        info: Info | None = None,
        example_id: int = 0,
        sampling_args: SamplingArgs | None = None,
        # Multi-agent specific parameters
        clients: list[AsyncOpenAI] | None = None,
        models: list[str] | None = None,
        round_callback: Callable | None = None,  # Callback after each round
        actor_ids: list[str] | None = None,
        **kwargs,
    ) -> tuple[Messages, State]:
        """
        Run a complete MegaGem game with multiple LLM players.

        Args:
            client: Default LLM client (used if clients not provided)
            model: Default model name (used if models not provided)
            clients: List of LLM clients, one per player
            models: List of model names, one per player
            sampling_args: Sampling arguments for generation
            actor_ids: Per-player labels written into each turn record.
                Defaults to ``["trainable"] * num_players``. Caller-defined
                strings are emitted verbatim.

        Returns:
            Tuple of (completion_messages, final_state)
        """
        start_time = time.time()

        # Setup clients and models for each player
        if clients is None:
            clients = [client] * self.num_players
        if models is None:
            models = [model] * self.num_players

        if len(clients) != self.num_players or len(models) != self.num_players:
            raise ValueError(
                f"Expected {self.num_players} clients and models, got {len(clients)} clients and {len(models)} models"
            )

        if actor_ids is None:
            actor_ids = [DEFAULT_ACTOR_ID] * self.num_players
        if len(actor_ids) != self.num_players:
            raise ValueError(
                f"Expected {self.num_players} actor_ids, got {len(actor_ids)}"
            )

        # Initialize game state
        info = info or {}
        seed = info.get("seed", self.seed)
        game_state = self.create_game_state(seed=seed)

        # Track all messages/events for each player
        player_messages: list[list[dict]] = [[] for _ in range(self.num_players)]
        game_events: list[dict] = []

        # Per-model chat completion time (bidding only; exclude reveal phase)
        model_chat_times: list[float] = [0.0] * self.num_players

        round_num = await self._run_game_loop(
            game_state, clients, models, sampling_args, actor_ids,
            player_messages=player_messages, game_events=game_events,
            model_chat_times=model_chat_times, round_callback=round_callback,
            pikl_searcher=self._pikl_searcher,
            pikl_bid_searcher=self._pikl_bid_searcher,
        )

        # Game over - calculate final scores
        winner_id, final_scores = determine_winner(game_state)

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000

        # Build final state
        final_state: State = {
            "prompt": prompt,
            "completion": [],
            "answer": answer,
            "task": task,
            "info": info,
            "example_id": example_id,
            "responses": [],
            "turn": round_num,
            "timing": {
                "generation_ms": elapsed_ms,
                "scoring_ms": 0.0,
                "total_ms": elapsed_ms,
            },
            # MegaGem specific state
            "game_state": game_state.to_public_dict(),
            "player_messages": player_messages,
            "game_events": game_events,
            "final_scores": final_scores,
            "winner_id": winner_id,
            "num_rounds": round_num,
            "models_used": models,
            "actor_ids": list(actor_ids),
            # Per-model chat completion time in seconds (bidding only; exclude reveal)
            "model_chat_times_seconds": list(model_chat_times),
        }

        # Build completion messages (summary of the game)
        completion_messages: Messages = [
            {
                "role": "assistant",
                "content": self._format_game_summary(game_state, final_scores, winner_id, round_num),
            }
        ]

        return completion_messages, final_state

    def _format_game_summary(
        self,
        game_state: GameState,
        final_scores: list[dict[str, Any]],
        winner_id: int,
        num_rounds: int,
    ) -> str:
        """Format a summary of the completed game."""
        lines = [
            "=" * 60,
            "MEGAGEM GAME COMPLETE",
            "=" * 60,
            f"Total Rounds: {num_rounds}",
            f"Winner: Player {winner_id}",
            "",
            "FINAL SCORES:",
        ]

        for score in final_scores:
            player_id = score["player_id"]
            is_winner = player_id == winner_id
            marker = " (WINNER)" if is_winner else ""
            lines.append(f"\nPlayer {player_id}{marker}:")
            lines.append(f"  Coins: {score['coins']}")
            lines.append(f"  Gem Value: {score['gem_value']}")
            lines.append(f"  Mission Rewards: {score['mission_rewards']}")
            lines.append(f"  Loan Payments: -{score['loan_payments']}")
            lines.append(f"  Investment Returns: +{score['investment_returns']}")
            lines.append(f"  FINAL SCORE: {score['final_score']}")

        lines.append("")
        lines.append("Value Display at End:")
        for color, count in game_state.get_value_display_counts().items():
            value = game_state.get_gem_value(color)
            lines.append(f"  {color}: {count} → {value} coins/gem")

        return "\n".join(lines)
