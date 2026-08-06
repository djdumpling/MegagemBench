"""Run a single MegaGem game and emit a schema-v3 JSON trajectory.

  megagem-run
  megagem-run --model anthropic/claude-sonnet-4.5 \\
              --model google/gemini-3-pro-preview \\
              --model openai/gpt-5.2

(or: python -m megagem.rollout)
"""

import argparse
import asyncio
import contextlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table
from rich.text import Text

from megagem import load_environment
from megagem.data import load_value_charts
from megagem.endpoints import ENDPOINTS, pick_url
from megagem.environment.console_format import color_gem_name, format_gem_string_with_colors
from megagem.environment.telemetry import compute_telemetry
from megagem.evals.model_mapping import get_model_for_number, get_model_name, get_number_for_model
from megagem.game.cards import ValueChart


@dataclass
class PlayerStats:
    player_id: int
    bids_by_type: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    wins_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_bids: int = 0
    total_wins: int = 0
    total_bid_amount: int = 0
    total_spent: int = 0
    gems_collected: int = 0
    missions_completed: int = 0
    loans_taken: int = 0
    investments_made: int = 0


async def run_game(
    models: list[str] | None = None,
    value_chart: str = "A",
    seed: int = 42,
    num_players: int = 3,
    output_file: str | None = None,
    json_filename: str | None = None,
    quiet: bool = False,
    silent: bool = False,
    game_label: str | None = None,
    results_dir: Path | None = None,
    caller_api_params: dict | list[dict | None] | None = None,
    pikl_config: dict | None = None,
    value_head_config: dict | None = None,
):
    """Run one MegaGem game. quiet=True prints only "[game_label] Round N" per
    round (JSON output and game logic are unchanged). silent=True suppresses
    even that per-round line — useful when many games run concurrently (Phase 3
    rollouts) and the caller emits its own one-line-per-game progress.

    ``caller_api_params`` may be a single dict (applied to every seat —
    backward-compatible) or a list of length ``num_players`` for a PER-SEAT
    override (Phase-3 §3.3 opponent pool: the trainable seat is sampled while
    pooled opponents play greedy so they stay deterministic within a K-group).
    """
    if models is None:
        default_model = "gpt-4.1" if "gpt-4.1" in ENDPOINTS else list(ENDPOINTS.keys())[0] if ENDPOINTS else "gpt-4.1"
        models = [default_model] * num_players
    elif len(models) != num_players:
        print(f"Warning: {len(models)} models provided but {num_players} players. Repeating last model.")
        while len(models) < num_players:
            models.append(models[-1])

    def find_endpoint_key(model: str) -> str | None:
        """Endpoint key for ``model``: exact match, else strip 'provider/' prefix."""
        if model in ENDPOINTS:
            return model
        if "/" in model:
            model_without_prefix = model.split("/", 1)[1]
            if model_without_prefix in ENDPOINTS:
                return model_without_prefix
        return None

    # Cache AsyncOpenAI instances per (key, url) so the connection pool is shared.
    client_cache: dict[tuple[str, str], AsyncOpenAI] = {}
    clients = []
    # Names sent to the API — distinct from `models` because vLLM's
    # --served-model-name can differ from the on-disk identifier.
    api_models: list[str] = []

    for model in models:
        endpoint_key = find_endpoint_key(model)

        if endpoint_key:
            endpoint = ENDPOINTS[endpoint_key]
            api_key = os.getenv(endpoint["key"])
            if not api_key:
                print(f"Error: {endpoint['key']} not set. Please set {endpoint['key']} environment variable.")
                sys.exit(1)
            # `endpoint["url"]` may be a list (DP layout — multiple vLLM
            # workers serving the same model). Round-robin via pick_url so
            # concurrent rollouts spread across workers; cache_key uses the
            # resolved URL so each URL gets its own connection-pooled client.
            resolved_url = pick_url(endpoint)
            cache_key = (api_key, resolved_url)
            if cache_key not in client_cache:
                # max_retries=0: the env's get_player_response owns retries with
                # header-aware backoff (X-RateLimit-Reset/Retry-After). Letting
                # the SDK ALSO retry would fire extra fast requests during a 429
                # storm, amplifying the rate-limit instead of backing off.
                client_cache[cache_key] = AsyncOpenAI(
                    api_key=api_key, base_url=resolved_url,
                    max_retries=0,
                    timeout=float(endpoint.get("timeout", 120.0)))
                if not quiet:
                    print(f"Created client for endpoint: {resolved_url}")
            client = client_cache[cache_key]
            api_models.append(endpoint.get("model", model))
        else:
            # Fall back to Prime Inference for models not in endpoints config.
            api_key = os.getenv("PRIME_API_KEY")
            if not api_key:
                print(f"Error: Model '{model}' not found in endpoints config and PRIME_API_KEY not set.")
                print(f"Please add '{model}' to megagem/endpoints.py or set PRIME_API_KEY environment variable.")
                sys.exit(1)
            cache_key = (api_key, "https://api.pinference.ai/api/v1")
            if cache_key not in client_cache:
                # See note above: env-level retries own backoff; SDK retries off.
                client_cache[cache_key] = AsyncOpenAI(
                    api_key=api_key, base_url="https://api.pinference.ai/api/v1",
                    max_retries=0, timeout=120.0)
                if not quiet:
                    print("Created client for Prime Inference endpoint")
            client = client_cache[cache_key]
            api_models.append(model)
        clients.append(client)

    # Local vLLM (key="EMPTY") needs enable_thinking=False or <think> tokens
    # eat the entire completion budget and content/reasoning_content come back
    # empty. This is a vLLM-only extra_body — computed PER SEAT (keyed on the
    # endpoint's "EMPTY" key) so a hosted API opponent in a MIXED game is never
    # sent the vLLM-specific chat_template_kwargs, which it would reject.
    def _vllm_base_extra(model: str) -> dict:
        key = ENDPOINTS.get(find_endpoint_key(model) or "", {}).get("key")
        if key == "EMPTY":
            return {"extra_body": {
                "chat_template_kwargs": {"enable_thinking": False}}}
        return {}

    # Optional caller passthrough (e.g. Phase-3 §A.7 rollout sampling
    # temperature/top_p). Backward-compatible: a dict (or None) applies to every
    # seat; a list (length num_players) is a PER-SEAT override — Phase-3's
    # opponent pool samples the trainable seat (T=1.0) while pooled opponents
    # play greedy (T=0) so they stay deterministic within a K-group. Top-level
    # keys (temperature/top_p/...) are set directly; `extra_body` is deep-merged
    # so the enable_thinking kwarg above is preserved.
    if isinstance(caller_api_params, list):
        per_seat_caller: list[dict | None] = list(caller_api_params)
        while len(per_seat_caller) < num_players:
            per_seat_caller.append(None)
        per_seat_caller = per_seat_caller[:num_players]
    else:
        per_seat_caller = [caller_api_params] * num_players

    def _merge_caller(base: dict, caller: dict | None) -> dict:
        merged = {k: (dict(v) if isinstance(v, dict) else v)
                  for k, v in base.items()}
        if caller:
            for k, v in caller.items():
                if k == "extra_body" and isinstance(v, dict):
                    merged.setdefault("extra_body", {})
                    merged["extra_body"] = {**merged["extra_body"], **v}
                else:
                    merged[k] = v
        return merged

    per_seat_api_params = [
        _merge_caller(_vllm_base_extra(models[i]), per_seat_caller[i])
        for i in range(num_players)
    ]

    env = load_environment(
        num_players=num_players,
        value_chart_id=value_chart,
        seed=seed,
        per_seat_api_params=per_seat_api_params,
    )

    # Test-time in-context value-head injection for the policy seat(s) (eval only).
    # The estimator is opponent-independent and uses only public board + the seat's
    # own hand, so it never leaks. Pass a pre-loaded estimator to avoid per-game reload.
    if value_head_config and value_head_config.get("enabled"):
        est = value_head_config.get("estimator")
        if est is None:
            from megagem.value_head.value_estimator import ValueEstimator
            est = ValueEstimator.load(value_head_config["model_path"])
        env.set_value_head(est, value_head_config.get("seats", [0]))

    # Test-time piKL reveal search for seat 0 (eval only). Continuations run on
    # the local blueprint (seat 0 must be a local vLLM model), never the live API
    # opponents. The search RNG (crn_seed) is a CONFIGURED constant independent of
    # the game seed — the game seed encodes the hidden deal, so the policy must not
    # draw from it. λ=∞ ⇒ sampled-τ̂ control (the EXACT blueprint control is piKL off).
    if pikl_config and pikl_config.get("enabled"):
        from megagem.environment.pikl_search import PiklBidSearcher, PiklRevealSearcher
        # "reveal" (Gate A) | "bid" (Gate B) | "ev_dist" (E1 analytic selector)
        target = pikl_config.get("target", "reveal")
        if target == "ev_dist":
            from megagem.environment.ev_dist import (
                DEFAULT_MODEL_PATH, DEFAULT_VALUE_HEAD_PATH, EvDistBidSearcher)
            _seat = int(pikl_config.get("seat", 0))   # default 0 = legacy path
            searcher = EvDistBidSearcher(
                bp_client=clients[_seat], bp_model=api_models[_seat],
                trainable_seat=_seat,
                num_players=num_players, value_chart_id=value_chart,
                bp_extra=_vllm_base_extra(models[_seat]),
                temperature=pikl_config.get("temperature", 1.0),
                max_parallel=pikl_config.get("max_parallel", 4),
                model_path=pikl_config.get("ev_model_path") or DEFAULT_MODEL_PATH,
                value_head_path=pikl_config.get("ev_value_head_path") or DEFAULT_VALUE_HEAD_PATH,
                gate_min_ev=pikl_config.get("ev_gate_min", 1.0),
                use_mission_bonus=pikl_config.get("ev_mission_bonus", True),
                pacing_lam=pikl_config.get("ev_pacing_lam", 0.0),
                vhat_debias=pikl_config.get("ev_vhat_debias", 0.0),
                value_refit_path=pikl_config.get("ev_value_refit_path") or "",
                pacing_schedule=pikl_config.get("ev_pacing_schedule") or "",
            )
            env.set_pikl_bid_searcher(searcher)
        else:
            searcher = (PiklBidSearcher if target == "bid" else PiklRevealSearcher)(
                bp_client=clients[0], bp_model=api_models[0], trainable_seat=0,
                num_players=num_players, value_chart_id=value_chart,
                bp_extra=_vllm_base_extra(models[0]), lam=pikl_config.get("lambda", float("inf")),
                n_tau=pikl_config.get("n", 16), temperature=pikl_config.get("temperature", 1.0),
                alpha=pikl_config.get("alpha", 0.5), m_worlds=pikl_config.get("m", 8),
                crn_seed=pikl_config.get("seed", 0), max_parallel=pikl_config.get("max_parallel", 4),
            )
            searcher.diagnostics = pikl_config.get("diagnostics")  # C2 var-screen sink
            searcher.value_aware = pikl_config.get("value_aware", False)
            searcher.fv_shade = pikl_config.get("fv_shade", 0.8)
            # Behaviour-informed belief over the M determinization worlds (default "uniform").
            searcher.belief = pikl_config.get("belief", "uniform")
            searcher.belief_sigma = pikl_config.get("belief_sigma", 6.0)
            searcher.belief_gamma = pikl_config.get("belief_gamma", 0.5)
            searcher.belief_oversample = pikl_config.get("belief_oversample", 6)
            searcher.belief_window = pikl_config.get("belief_window", 0)
            searcher.belief_coin_cap = pikl_config.get("belief_coin_cap", False)
            # Rollout/live opponent bid model: "fair_value" (default, byte-identical) | "market".
            # market_params override MARKET_DEFAULTS when opp_model="market" (None ⇒ fit defaults).
            opp_model = pikl_config.get("opp_model", "fair_value")
            market_params = pikl_config.get("market_params")
            searcher.opp_model = opp_model
            searcher._market_params = market_params
            env.opp_model = opp_model
            env.market_params = market_params
            # Value-coherent opponent seats: bid in-process (no LLM), via opp_model — makes the
            # bid likelihood the informed belief inverts correctly specified.
            fv_seats = pikl_config.get("fv_opponent_seats") or []
            env.fv_opponent_seats = {int(s) for s in fv_seats}
            env.fv_opp_shade = pikl_config.get("fv_opp_shade", pikl_config.get("fv_shade", 0.8))
            if target == "bid":
                searcher.treasure_only = pikl_config.get("bid_treasure_only", True)
                searcher.candidate_mode = pikl_config.get("candidate_mode", "sampled")
                searcher.max_bid_candidates = pikl_config.get("max_bid_candidates", 0)
                searcher.gate_min_lift = pikl_config.get("gate_min_lift", 0.0)
                searcher.gate_z = pikl_config.get("gate_z", 0.0)
                lambda_mix = pikl_config.get("lambda_mix")
                if lambda_mix:
                    searcher.lambda_mix = tuple(float(x) for x in lambda_mix)
                env.set_pikl_bid_searcher(searcher)
            else:
                env.set_pikl_searcher(searcher)

    if not quiet:
        print(f"Starting MegaGem game with {num_players} players...")
        print(f"Models: {models}")
        print(f"Value Chart: {value_chart}")
        print(f"Seed: {seed}")

    player_stats = [PlayerStats(player_id=i) for i in range(num_players)]

    # User-facing labels (e.g. "megagem-sft") for JSON output and console tables;
    # api_models is what gets sent over the wire.
    user_facing_models = list(models)

    from megagem.environment.prompts import generate_system_prompt

    game_data: dict = {
        "metadata": {
            "schema_version": 3,
            "models": models,
            "value_chart": value_chart,
            "seed": seed,
            "num_players": num_players,
            "timestamp": datetime.now().isoformat(),
            "system_prompt": generate_system_prompt(),
        },
        "rounds": [],
        "final_results": None,
        "statistics": None,
        "telemetry": None,
    }
    pikl_decision_metrics: list[dict] = []

    def capture_pikl_payload(round_num: int, phase: str, player_id: int, record) -> None:
        if record is None or getattr(record, "parse_method", "") != "pikl":
            return
        try:
            payload = json.loads(record.raw_response).get("pikl", {})
        except Exception:  # noqa: BLE001 - payload logging must never affect game play
            return
        entry = {
            "round": round_num,
            "phase": phase,
            "player_id": player_id,
            "chosen": payload.get("chosen"),
            "lambda": payload.get("lambda"),
            "lambda_mix": payload.get("lambda_mix"),
            "candidate_mode": payload.get("candidate_mode"),
            "metrics": payload.get("metrics") or {},
        }
        if "gate" in payload:
            entry["gate"] = payload["gate"]
        pikl_decision_metrics.append(entry)

    def print_round_summary(
        round_num,
        auction,
        bids,
        bid_reasoning=None,
        winner_id=None,
        winning_bid=None,
        gems_won=None,
        revealed_gems_available=None,
        gem_revealed=None,
        reveal_response=None,
        reveal_reasoning=None,
        missions_completed=None,
        mission_reasoning=None,
        game_state=None,
        models=None,
        coins_before=None,
        tiebreak_order_before=None,
        bid_turn_records=None,
        reveal_turn_record=None,
        actor_ids=None,
    ):
        auction_type_key = None
        if auction.type.value == "treasure":
            auction_type_key = f"treasure_{auction.gems}_gem"
        elif auction.type.value == "loan":
            auction_type_key = f"loan_{auction.amount}"
        elif auction.type.value == "investment":
            auction_type_key = f"invest_{auction.bonus}"

        for player_id, bid in enumerate(bids):
            stats = player_stats[player_id]
            stats.total_bids += 1
            stats.total_bid_amount += bid
            if auction_type_key:
                stats.bids_by_type[auction_type_key].append(bid)

        winner_stats = player_stats[winner_id]
        winner_stats.total_wins += 1
        winner_stats.total_spent += winning_bid
        if auction_type_key:
            winner_stats.wins_by_type[auction_type_key] += 1
        if gems_won:
            winner_stats.gems_collected += len(gems_won)
        if auction.type.value == "loan":
            winner_stats.loans_taken += 1
        elif auction.type.value == "investment":
            winner_stats.investments_made += 1

        for player_id, mission_ids in missions_completed.items():
            player_stats[player_id].missions_completed += len(mission_ids)

        round_data = {
            "round_number": round_num,
            "auction": {
                "type": auction.type.value,
                "description": auction.get_description(),
            },
            "players": [],
            "value_display": {},
            # Post-resolution order (winner moved to back). Kept for back-compat.
            "tiebreak_order": list(game_state.tiebreak_order),
            # Decision-time order — what §A.5's RL role baseline must key on.
            "tiebreak_order_before": (
                list(tiebreak_order_before)
                if tiebreak_order_before is not None
                else list(game_state.tiebreak_order)
            ),
            "available_missions": [m.to_dict() for m in game_state.available_missions],
            "missions_completed": {},
        }

        if auction.type.value == "treasure":
            round_data["auction"]["gems_available"] = (
                revealed_gems_available[: auction.gems] if revealed_gems_available else []
            )
        elif auction.type.value == "loan":
            round_data["auction"]["loan_amount"] = auction.amount
        elif auction.type.value == "investment":
            round_data["auction"]["bonus"] = auction.bonus

        for player_id in range(len(bids)):
            player = game_state.players[player_id]
            coins_before_val = (
                coins_before[player_id] if coins_before and player_id < len(coins_before) else player.coins
            )

            # player.coins is already post-auction by the time we get here.
            coins_after_val = player.coins

            player_round_data = {
                "player_id": player_id,
                "model": user_facing_models[player_id] if player_id < len(user_facing_models) else "unknown",
                "bid": bids[player_id],
                "coins_before": coins_before_val,
                "coins_after": coins_after_val,
                "collection": list(player.collection),
                "collection_counts": player.get_collection_counts(),
                "hand": list(player.hand),
                "reasoning": bid_reasoning[player_id] if bid_reasoning and player_id < len(bid_reasoning) else "",
                "is_winner": player_id == winner_id,
            }

            if bid_turn_records and player_id < len(bid_turn_records):
                bid_record = bid_turn_records[player_id]
                capture_pikl_payload(round_num, "bid", player_id, bid_record)
                player_round_data.update(
                    {
                        "actor_id": bid_record.actor_id,
                        "prompt": bid_record.prompt,
                        "raw_response": bid_record.raw_response,
                        "parsed_action": bid_record.parsed_action,
                        "parse_method": bid_record.parse_method,
                        "parse_valid": bid_record.parse_valid,
                        "legal_valid": bid_record.legal_valid,
                        "default_used": bid_record.default_used,
                        "length_split": dict(bid_record.length_split),
                        "parse_error": bid_record.parse_error,
                        "legal_error": bid_record.legal_error,
                    }
                )
            elif actor_ids and player_id < len(actor_ids):
                player_round_data["actor_id"] = actor_ids[player_id]

            # Add winner-specific data
            if player_id == winner_id:
                player_round_data["winning_bid"] = winning_bid
                player_round_data["gems_won"] = gems_won if gems_won else []
                player_round_data["gem_revealed"] = gem_revealed
                player_round_data["reveal_reasoning"] = reveal_reasoning if reveal_reasoning else ""

                if reveal_turn_record is not None and player_id == reveal_turn_record.player_id:
                    capture_pikl_payload(round_num, "reveal", player_id, reveal_turn_record)
                    player_round_data["reveal"] = {
                        "actor_id": reveal_turn_record.actor_id,
                        "prompt": reveal_turn_record.prompt,
                        "raw_response": reveal_turn_record.raw_response,
                        "parsed_action": reveal_turn_record.parsed_action,
                        "parse_method": reveal_turn_record.parse_method,
                        "parse_valid": reveal_turn_record.parse_valid,
                        "legal_valid": reveal_turn_record.legal_valid,
                        "default_used": reveal_turn_record.default_used,
                        "final_reveal": reveal_turn_record.final_reveal,
                        "reasoning": reveal_turn_record.reasoning,
                        "length_split": dict(reveal_turn_record.length_split),
                        "parse_error": reveal_turn_record.parse_error,
                        "legal_error": reveal_turn_record.legal_error,
                    }

            # Add mission completion data
            if missions_completed and player_id in missions_completed:
                player_round_data["missions_completed"] = missions_completed[player_id]
                player_round_data["mission_reasoning"] = (
                    mission_reasoning.get(player_id, "") if mission_reasoning else ""
                )

            round_data["players"].append(player_round_data)

        if missions_completed:
            round_data["missions_completed"] = {
                str(player_id): mission_ids for player_id, mission_ids in missions_completed.items()
            }

        value_counts = game_state.get_value_display_counts()
        round_data["value_display"] = {
            color: {
                "count": count,
                "value_per_gem": game_state.get_gem_value(color),
            }
            for color, count in sorted(value_counts.items())
        }

        game_data["rounds"].append(round_data)

        round_output_lines = []
        round_output_lines.append(f"\n{'=' * 80}")
        round_output_lines.append(f"ROUND {round_num}")
        round_output_lines.append(f"{'=' * 80}\n")

        if auction.type.value == "treasure":
            gems_str = ", ".join(revealed_gems_available[: auction.gems]) if revealed_gems_available else "None"
            auction_desc = f"Treasure: Win {auction.gems} gem(s) - Available: {gems_str}"
        else:
            auction_desc = auction.get_description()
        round_output_lines.append(f"Auction: {auction_desc}\n")

        combined_table_data = []
        for player_id in range(len(bids)):
            model_name = (
                user_facing_models[player_id]
                if player_id < len(user_facing_models)
                else "unknown"
            )
            bid = bids[player_id]

            player = game_state.players[player_id]
            collection_counts = player.get_collection_counts()
            if collection_counts:
                collection_str = ", ".join(f"{color}×{count}" for color, count in sorted(collection_counts.items()))
            else:
                collection_str = "Empty"

            if player.hand:
                hand_counts = Counter(player.hand)
                hand_str = ", ".join(f"{color}×{count}" for color, count in sorted(hand_counts.items()))
            else:
                hand_str = "Empty"

            coins_after = player.coins
            if coins_before and player_id < len(coins_before):
                coins_before_val = coins_before[player_id]
                if coins_before_val == coins_after:
                    coins_str = str(coins_after)
                else:
                    coins_str = f"{coins_before_val} -> {coins_after}"
            else:
                coins_str = str(coins_after)

            winner_marker = "⭐" if player_id == winner_id else ""

            gem_revealed_str = gem_revealed if (player_id == winner_id and gem_revealed) else ""

            model_display = Text()
            if "/" in model_name:
                provider, model = model_name.split("/", 1)
                model_display.append(provider, style="dim")
                model_display.append("/", style="dim")
                model_display.append(model, style="bold bright_cyan")
            else:
                model_display.append(model_name, style="bold bright_cyan")

            collection_display = format_gem_string_with_colors(collection_str)
            hand_display = format_gem_string_with_colors(hand_str)

            gem_revealed_display = Text()
            if gem_revealed_str:
                gem_revealed_display = color_gem_name(gem_revealed_str)

            combined_table_data.append(
                [
                    f"Player {player_id}",
                    model_display,
                    str(bid),
                    coins_str,
                    winner_marker,
                    collection_display,
                    gem_revealed_display,
                    hand_display,
                ]
            )

        table = Table(show_header=True, header_style="bold", box=ROUNDED)
        table.add_column("Player", style="cyan", no_wrap=True)
        table.add_column("Model", max_width=25, no_wrap=True)
        table.add_column("Bid", justify="right", style="cyan", no_wrap=True)
        table.add_column("Coins", justify="right", style="cyan", no_wrap=True)
        table.add_column("Winner", justify="center", style="bold", width=6, no_wrap=True)
        table.add_column("Collection", no_wrap=True)
        table.add_column("Gem revealed", justify="center", width=12, no_wrap=True)
        table.add_column("Hand", no_wrap=True)

        for row in combined_table_data:
            table.add_row(*row)

        reasoning_table = None
        if bid_reasoning and len(bid_reasoning) > 0:
            reasoning_table = Table(show_header=True, header_style="bold", box=ROUNDED)
            reasoning_table.add_column("Player", style="cyan", no_wrap=True, width=10)
            reasoning_table.add_column("Reasoning", style="dim", no_wrap=False, max_width=100)

            for player_id in range(len(bids)):
                reasoning = bid_reasoning[player_id] if player_id < len(bid_reasoning) else ""
                if not reasoning:
                    reasoning = "(No reasoning provided)"
                reasoning_table.add_row(f"Player {player_id}", reasoning)

        missions_table = None
        if game_state.available_missions:
            missions_table = Table(show_header=True, header_style="bold", box=ROUNDED)
            missions_table.add_column("Mission ID", style="cyan", no_wrap=True)
            missions_table.add_column("Description", no_wrap=False, max_width=60)
            missions_table.add_column("Reward", justify="right", style="cyan", no_wrap=True)

            for mission in game_state.available_missions:
                missions_table.add_row(mission.id, mission.description, f"{mission.reward} coins")

        missions_lines = []
        if missions_completed:
            missions_lines.append("\nMissions completed this round:")
            for player_id, mission_ids in missions_completed.items():
                missions_lines.append(f"  Player {player_id}: {', '.join(mission_ids)}")
        missions_text = "\n".join(missions_lines)

        value_counts = game_state.get_value_display_counts()
        if value_counts:
            value_table = Table(show_header=True, header_style="bold", box=ROUNDED)
            value_table.add_column("Color")
            value_table.add_column("In Display", justify="right", style="cyan")
            value_table.add_column("Value/Gem", justify="right", style="cyan")

            for color, count in sorted(value_counts.items()):
                value_per_gem = game_state.get_gem_value(color)
                colored_color = color_gem_name(color)
                value_table.add_row(colored_color, str(count), str(value_per_gem))

        if silent:
            # Truly silent — caller owns per-game progress reporting.
            pass
        elif quiet:
            # One-line progress; tables/reasoning still go into game_data for JSON.
            label = f"[{game_label}] " if game_label else ""
            print(f"{label}Round {round_num}", flush=True)
        else:
            console = Console()
            console.print(f"\n{'=' * 80}")
            console.print(f"ROUND {round_num}")
            console.print(f"{'=' * 80}\n")
            console.print(f"Auction: {auction_desc}\n")
            console.print(table)
            if reasoning_table:
                console.print("\n")
                console.print(reasoning_table)
            if missions_table:
                console.print("\n")
                console.print(missions_table)
            if missions_text:
                console.print(missions_text)
            if value_counts:
                console.print(value_table)

    completion, state = await env.rollout(
        client=clients[0],
        model=api_models[0],
        prompt=[],
        clients=clients,
        models=api_models,
        round_callback=print_round_summary,
    )

    game_state_dict = state.get("game_state", {})
    players_data = game_state_dict.get("players", [])
    for player_id in range(num_players):
        if player_id < len(players_data):
            player_data = players_data[player_id]
            player_stats[player_id].gems_collected = len(player_data.get("collection", []))

    results_lines = []
    results_lines.append("\n" + "=" * 80)
    results_lines.append("GAME RESULTS")
    results_lines.append("=" * 80)
    results_lines.append(f"Winner: Player {state['winner_id']}")
    results_lines.append(f"Total Rounds: {state['num_rounds']}\n")

    scores_table = Table(show_header=True, header_style="bold", box=ROUNDED)
    scores_table.add_column("Player", style="cyan")
    scores_table.add_column("", width=2, justify="center")
    scores_table.add_column("Coins", justify="right", style="cyan")
    scores_table.add_column("Gem Value", justify="right", style="cyan")
    scores_table.add_column("Missions", justify="right", style="cyan")
    scores_table.add_column("Loans", justify="right", style="cyan")
    scores_table.add_column("Investments", justify="right", style="cyan")
    scores_table.add_column("Final Score", justify="right", style="bold cyan")

    for score in state["final_scores"]:
        is_winner = score["player_id"] == state["winner_id"]
        winner_marker = "⭐" if is_winner else ""
        scores_table.add_row(
            f"Player {score['player_id']}",
            winner_marker,
            str(score["coins"]),
            str(score["gem_value"]),
            str(score["mission_rewards"]),
            f"-{score['loan_payments']}",
            f"+{score['investment_returns']}",
            str(score["final_score"]),
        )

    results_lines.append("\n" + "=" * 80)
    results_lines.append("Value Display at End:")
    value_display_counts = state["game_state"]["value_display_counts"]

    value_end_table = Table(show_header=True, header_style="bold", box=ROUNDED)
    value_end_table.add_column("Color")
    value_end_table.add_column("Gems in Display", justify="right", style="cyan")
    value_end_table.add_column("Value per Gem", justify="right", style="cyan")

    charts = load_value_charts()
    chart = ValueChart.from_dict(value_chart, charts[value_chart])

    for color, count in sorted(value_display_counts.items()):
        value = chart.get_gem_value(count)
        colored_color = color_gem_name(color)
        value_end_table.add_row(colored_color, str(count), str(value))

    stats_lines = []
    stats_lines.append("\n" + "=" * 80)
    stats_lines.append("PLAYER STATISTICS")
    stats_lines.append("=" * 80)

    # Define auction type labels
    auction_type_labels = {
        "treasure_1_gem": "Treasure (1 gem)",
        "treasure_2_gem": "Treasure (2 gems)",
        "loan_10": "Loan 10",
        "loan_20": "Loan 20",
        "invest_5": "Investment +5",
        "invest_10": "Investment +10",
    }

    for player_id, stats in enumerate(player_stats):
        stats_lines.append(f"\nPlayer {player_id} ({models[player_id]}):")
        stats_lines.append("-" * 80)

        win_rate = (stats.total_wins / stats.total_bids * 100) if stats.total_bids > 0 else 0
        avg_bid = (stats.total_bid_amount / stats.total_bids) if stats.total_bids > 0 else 0

        stats_lines.append(f"  Total Bids: {stats.total_bids}")
        stats_lines.append(f"  Total Wins: {stats.total_wins}")
        stats_lines.append(f"  Win Rate: {win_rate:.1f}%")
        stats_lines.append(f"  Average Bid: {avg_bid:.2f} coins")
        stats_lines.append(f"  Total Spent: {stats.total_spent} coins")
        stats_lines.append(f"  Collection Size: {stats.gems_collected} gems")
        stats_lines.append(f"  Missions Completed: {stats.missions_completed}")
        stats_lines.append(f"  Loans Taken: {stats.loans_taken}")
        stats_lines.append(f"  Investments Made: {stats.investments_made}")

        stats_lines.append("\n  Average Bids by Auction Type:")
        for auction_type, label in auction_type_labels.items():
            bids = stats.bids_by_type.get(auction_type, [])
            if bids:
                avg = sum(bids) / len(bids)
                wins = stats.wins_by_type.get(auction_type, 0)
                stats_lines.append(f"    {label}: {avg:.2f} coins (bids: {len(bids)}, wins: {wins})")
            else:
                stats_lines.append(f"    {label}: No bids")

    game_state_dict = state.get("game_state", {})
    available_missions_final = game_state_dict.get("available_missions", [])

    game_data["final_results"] = {
        "winner_id": state["winner_id"],
        "num_rounds": state["num_rounds"],
        "final_scores": state["final_scores"],
        "available_missions": available_missions_final,
        "value_display_final": {
            color: {
                "count": count,
                "value_per_gem": chart.get_gem_value(count),
            }
            for color, count in sorted(value_display_counts.items())
        },
    }

    # Missions are constant for a game, so stored once at metadata level.
    game_data["metadata"]["available_missions"] = available_missions_final

    # Per-model chat completion time (bidding only; reveal phase excluded).
    model_chat_times_seconds = state.get("model_chat_times_seconds")
    models_used = state.get("models_used", models)
    if model_chat_times_seconds is not None and models_used:
        game_data["metadata"]["model_chat_times_seconds"] = [
            round(t, 3) for t in model_chat_times_seconds
        ]
        game_data["metadata"]["model_chat_seconds"] = {
            models_used[i]: round(model_chat_times_seconds[i], 3)
            for i in range(min(len(models_used), len(model_chat_times_seconds)))
        }

    stats_data = []
    for player_id, stats in enumerate(player_stats):
        win_rate = (stats.total_wins / stats.total_bids * 100) if stats.total_bids > 0 else 0
        avg_bid = (stats.total_bid_amount / stats.total_bids) if stats.total_bids > 0 else 0

        player_stat_data = {
            "player_id": player_id,
            "model": models[player_id],
            "total_bids": stats.total_bids,
            "total_wins": stats.total_wins,
            "win_rate": round(win_rate, 2),
            "average_bid": round(avg_bid, 2),
            "total_spent": stats.total_spent,
            "gems_collected": stats.gems_collected,
            "missions_completed": stats.missions_completed,
            "loans_taken": stats.loans_taken,
            "investments_made": stats.investments_made,
            "bids_by_type": {
                auction_type: {
                    "count": len(bids),
                    "average": round(sum(bids) / len(bids), 2) if bids else 0,
                    "bids": bids,
                }
                for auction_type, bids in stats.bids_by_type.items()
            },
            "wins_by_type": dict(stats.wins_by_type),
        }
        stats_data.append(player_stat_data)

    game_data["statistics"] = stats_data
    game_data["telemetry"] = compute_telemetry(game_data["rounds"])
    game_data["pikl_decision_metrics"] = list(pikl_decision_metrics)
    # E1 ev_dist selector: per-decision telemetry covers EVERY treasure node
    # (incl. gated-off pass-throughs, which keep the normal turn record).
    _ev_log = getattr(env._pikl_bid_searcher, "decision_log", None)
    if _ev_log:
        game_data["ev_dist_decisions"] = list(_ev_log)

    console = Console()
    if not quiet:
        for line in results_lines:
            console.print(line)
        console.print(scores_table)
        console.print("\n" + "=" * 80)
        console.print("Value Display at End:")
        console.print(value_end_table)
        for line in stats_lines:
            console.print(line)

    try:
        if output_file:
            if json_filename is None:
                model_numbers = []
                if models:
                    for model_id in models:
                        try:
                            num = get_number_for_model(model_id)
                            model_numbers.append(num)
                        except ValueError:
                            pass

                if model_numbers:
                    # Sort so single-game filename ordering is stable across runs.
                    model_numbers.sort()
                    json_filename = f"megagem_{'_'.join(map(str, model_numbers))}.json"
                else:
                    json_filename = output_file.replace(".txt", ".json") if output_file.endswith(".txt") else output_file + ".json"

            if results_dir is None:
                results_dir = Path(__file__).parent / "evals" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            json_file = results_dir / json_filename
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(game_data, f, indent=2, ensure_ascii=False)
            if not quiet:
                console.print(f"\n[green]Game data saved to: {json_file}[/green]")

        state["pikl_decision_metrics"] = list(pikl_decision_metrics)
        if _ev_log:
            state["ev_dist_decisions"] = list(_ev_log)
        return state
    finally:
        # AsyncOpenAI owns an httpx.AsyncClient. If it is left for GC after
        # asyncio.run() closes the loop, httpx may emit noisy "Event loop is
        # closed" task exceptions in long Phase-3 rollout/eval jobs.
        for client in set(client_cache.values()):
            with contextlib.suppress(Exception):
                await client.close()


def main():
    parser = argparse.ArgumentParser(description="Run a MegaGem game")
    parser.add_argument('--model', action='append')
    parser.add_argument('--model-number', type=int, action='append')
    parser.add_argument('--value-chart', default='A', choices=['A', 'B', 'C', 'D', 'E'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-players', type=int, default=3)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    models = args.model or []
    if args.model_number:
        for num in args.model_number:
            try:
                model_id = get_model_for_number(num)
                models.append(model_id)
                print(f"Model number {num} -> {get_model_name(num)} ({model_id})")
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)

    output_file = args.output
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"megagem_game_{timestamp}.txt"

    asyncio.run(
        run_game(
            models=models if models else None,
            value_chart=args.value_chart,
            seed=args.seed,
            num_players=args.num_players,
            output_file=output_file,
        )
    )


if __name__ == "__main__":
    main()
