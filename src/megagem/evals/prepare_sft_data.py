#!/usr/bin/env python3
"""
Prepare SFT training data from top-2-by-final-score teacher traces.

Replays each game deterministically using the same seed to reconstruct
the exact prompts each model saw, then pairs them with the **raw teacher
response** (preserving original formatting) for both bid and reveal turns.
Only players whose model is in V2_TEACHERS contribute examples.

Usage:
    # Full train/val export (default), v2 inputs:
    python src/megagem/evals/prepare_sft_data.py \\
        --input-dir results/sft_v2_data_<ts> \\
        --output-dir data/sft_v2_examples \\
        --seed-split train
    python src/megagem/evals/prepare_sft_data.py \\
        --input-dir results/sft_v2_data_<ts> \\
        --output-dir data/sft_v2_examples \\
        --seed-split val

    # Pilot / ad-hoc directory with seeds outside the configured splits:
    python src/megagem/evals/prepare_sft_data.py \\
        --input-dir results/sft_pilot_flash_20260512_004059 \\
        --no-seed-filter --dry-run

Output format (one JSON object per line in .jsonl):
    {
        "messages": [
            {"role": "system", "content": "<system_prompt>"},
            {"role": "user", "content": "<game_state_prompt>"},
            {"role": "assistant", "content": "<raw_response from teacher>"}
        ],
        "metadata": { ... }
    }
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from megagem.data import load_auctions, load_gems, load_missions, load_value_charts
from megagem.environment.prompts import (
    generate_bid_prompt,
    generate_reveal_prompt,
    generate_system_prompt,
)
from megagem.game.actions import PARSE_METHOD_NONE, PARSE_METHOD_PROSE_FALLBACK
from megagem.game.cards import AuctionType, ValueChart
from megagem.game.rules import (
    AuctionOutcome,
    apply_auction_outcome,
    complete_mission,
    reveal_gem_from_hand,
)
from megagem.game.state import GameState
from megagem.training.seed_splits import ensure_disjoint, load_split_seeds

# Repo root (this file lives at src/megagem/evals/); anchors the default
# --input-dir/--output-dir below.
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "src" / "megagem" / "evals" / "results"
OUTPUT_DIR = REPO_ROOT / "src" / "megagem" / "evals" / "sft_data"

V2_TEACHERS = {
    "google/gemini-3-flash-preview",
    "anthropic/claude-opus-4.6",
}
STRUCTURED_PARSE_FAILURE_METHODS = {PARSE_METHOD_NONE, PARSE_METHOD_PROSE_FALLBACK}

# Cache data files (loaded once, reused for all games)
_data_cache: dict = {}


def _load_data():
    """Load and cache game data files."""
    if not _data_cache:
        _data_cache["gems"] = load_gems()
        _data_cache["auctions"] = load_auctions()
        _data_cache["missions"] = load_missions()
        _data_cache["charts"] = load_value_charts()
    return _data_cache


def init_game_state(seed: int, value_chart_id: str, num_players: int) -> GameState:
    """Initialize a GameState with the same seed used in the original game."""
    data = _load_data()
    chart = ValueChart.from_dict(value_chart_id, data["charts"][value_chart_id])
    return GameState.create_new_game(
        num_players=num_players,
        gem_cards=data["gems"],
        auction_cards=data["auctions"],
        missions=data["missions"],
        value_chart=chart,
        seed=seed,
    )


def advance_state(game_state: GameState, round_data: dict) -> None:
    """Apply a round's recorded outcomes to advance the game state.

    Mirrors the exact sequence in MegaGemEnv.rollout():
      1. gem reveal (if treasure winner has hand)
      2. apply_auction_outcome (coins, collection, tiebreak, history)
      3. mission completion
    """
    # Find round winner
    round_winner_id = None
    winning_bid = 0
    for p in round_data["players"]:
        if p["is_winner"]:
            round_winner_id = p["player_id"]
            winning_bid = p.get("winning_bid", p["bid"])
            break
    if round_winner_id is None:
        raise ValueError(f"No winner in round {round_data['round_number']}")

    auction = game_state.current_auction

    # 1. Gem reveal (before apply_auction_outcome, matching rollout order)
    gem_revealed = None
    if auction.type == AuctionType.TREASURE:
        winner_data = round_data["players"][round_winner_id]
        gem_revealed = winner_data.get("gem_revealed")
        if gem_revealed:
            reveal_gem_from_hand(game_state, round_winner_id, gem_revealed)

    # 2. Construct outcome and apply
    gems_won = []
    coins_received = 0
    investment_locked = 0

    if auction.type == AuctionType.TREASURE:
        gems_won = round_data["players"][round_winner_id].get("gems_won", [])
    elif auction.type == AuctionType.LOAN:
        coins_received = auction.amount
    elif auction.type == AuctionType.INVESTMENT:
        investment_locked = winning_bid

    outcome = AuctionOutcome(
        winner_id=round_winner_id,
        winning_bid=winning_bid,
        gems_won=gems_won,
        coins_received=coins_received,
        investment_locked=investment_locked,
    )
    bids = [p["bid"] for p in round_data["players"]]
    apply_auction_outcome(game_state, outcome, bids, gem_revealed)

    # 3. Mission completions
    missions_completed = round_data.get("missions_completed", {})
    for pid_str, mission_ids in missions_completed.items():
        pid = int(pid_str)
        for mid in mission_ids:
            complete_mission(game_state, pid, mid)


def verify_alignment(
    game_state: GameState,
    round_data: dict,
    game_file: str,
    winner_id: int,
) -> list[str]:
    """Verify that the replayed game state matches the JSON at this round.

    Returns a list of warning messages (empty = perfect alignment).
    """
    warnings = []
    round_num = round_data["round_number"]
    auction = game_state.current_auction

    # Round number
    if game_state.round_number != round_num:
        warnings.append(
            f"{game_file} round {round_num}: round_number mismatch "
            f"(state={game_state.round_number})"
        )

    # Auction type
    json_type = round_data["auction"]["type"]
    if auction.type.value != json_type:
        warnings.append(
            f"{game_file} round {round_num}: auction type mismatch "
            f"(state={auction.type.value} json={json_type})"
        )

    # Winner's coins_before
    json_coins = round_data["players"][winner_id]["coins_before"]
    state_coins = game_state.players[winner_id].coins
    if state_coins != json_coins:
        warnings.append(
            f"{game_file} round {round_num}: coins mismatch for player {winner_id} "
            f"(state={state_coins} json={json_coins})"
        )

    # Winner's hand size
    # JSON records post-round hand (after gem reveal), state is pre-round.
    # If the game winner won this treasure auction and revealed a gem,
    # state will have exactly 1 more gem than JSON — that's expected.
    json_hand = round_data["players"][winner_id]["hand"]
    state_hand = game_state.players[winner_id].hand
    expected_diff = 0
    if (
        auction.type == AuctionType.TREASURE
        and round_data["players"][winner_id]["is_winner"]
        and round_data["players"][winner_id].get("gem_revealed")
    ):
        expected_diff = 1
    actual_diff = len(state_hand) - len(json_hand)
    if actual_diff != expected_diff:
        warnings.append(
            f"{game_file} round {round_num}: hand size mismatch for player {winner_id} "
            f"(state={len(state_hand)} json={len(json_hand)} expected_diff={expected_diff})"
        )

    # Treasure auction: verify revealed gems match
    if auction.type == AuctionType.TREASURE:
        json_gems = round_data["auction"].get("gems_available", [])
        state_gems = game_state.revealed_gems[: auction.gems]
        if state_gems != json_gems:
            warnings.append(
                f"{game_file} round {round_num}: gems_available mismatch "
                f"(state={state_gems} json={json_gems})"
            )

    return warnings


def is_invalid_for_sft(turn_data: dict) -> bool:
    """Return True when a recorded turn should not be used as an SFT target."""
    return (
        not turn_data.get("parse_valid", True)
        or not turn_data.get("legal_valid", True)
        or turn_data.get("default_used", False)
        or turn_data.get("parse_method") in STRUCTURED_PARSE_FAILURE_METHODS
    )


def extract_game_examples(
    game_json_path: Path,
    include_reveals: bool = True,
    validate: bool = False,
    allowed_seeds: set[int] | None = None,
    require_valid: bool = False,
) -> tuple[list[dict], dict]:
    """Extract SFT training examples from a single game JSON.

    Replays the game from scratch using the same seed, generating exact
    prompts for the winning model at each round.

    Returns:
        (examples, stats)
    """
    with open(game_json_path, encoding="utf-8") as f:
        data = json.load(f)

    metadata = data["metadata"]
    models = metadata["models"]
    seed = metadata["seed"]
    value_chart = metadata["value_chart"]
    num_players = metadata["num_players"]

    if allowed_seeds is not None and seed not in allowed_seeds:
        return [], {"skipped": True, "reason": "seed not in selected split", "seed": seed}

    # Top-2 by final score (was: winner-only).
    final_scores = data["final_results"]["final_scores"]
    sorted_players = sorted(
        final_scores, key=lambda s: s["final_score"], reverse=True
    )
    top2_player_ids = [s["player_id"] for s in sorted_players[:2]]
    eligible_player_ids = [
        pid for pid in top2_player_ids if models[pid] in V2_TEACHERS
    ]
    if not eligible_player_ids:
        return [], {"skipped": True, "reason": "no v2-teacher in top-2"}

    # Replay the game from scratch
    game_state = init_game_state(seed, value_chart, num_players)
    system_prompt = generate_system_prompt()

    examples = []
    stats = {
        "game_file": game_json_path.name,
        "top2_player_ids": top2_player_ids,
        "top2_models": [models[pid] for pid in top2_player_ids],
        "eligible_player_ids": eligible_player_ids,
        "seed": seed,
        "value_chart": value_chart,
        "num_rounds": len(data["rounds"]),
        "bid_examples": 0,
        "reveal_examples": 0,
        "skipped_empty_response": 0,
        "skipped_invalid": 0,
        "alignment_warnings": 0,
    }

    for round_data in data["rounds"]:
        # Draw auction card: sets current_auction, increments round_number
        game_state.draw_auction_card()

        # Verify state alignment (once per round — replay is deterministic so
        # any player's coins/hand match; checking eligible_player_ids[0] is
        # enough to surface drift).
        if validate:
            warnings = verify_alignment(
                game_state, round_data, game_json_path.name, eligible_player_ids[0]
            )
            for w in warnings:
                logging.warning(w)
            stats["alignment_warnings"] += len(warnings)

        for pid in eligible_player_ids:
            player_model = models[pid]
            player_round = round_data["players"][pid]
            bid = player_round["bid"]
            raw_bid_response = player_round.get("raw_response", "")

            # --- BID TRAINING EXAMPLE ---
            # Use the raw teacher output verbatim; reconstructing from
            # `reasoning` + `json.dumps({"bid": bid})` would strip formatting
            # (fenced code blocks, original spacing) the student should learn.
            bid_is_invalid = is_invalid_for_sft(player_round)
            if require_valid and bid_is_invalid:
                stats["skipped_invalid"] += 1
            elif raw_bid_response and raw_bid_response.strip():
                bid_prompt = generate_bid_prompt(game_state, pid)

                examples.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": bid_prompt},
                        {"role": "assistant", "content": raw_bid_response},
                    ],
                    "metadata": {
                        "game_file": game_json_path.name,
                        "seed": seed,
                        "value_chart": value_chart,
                        "round": round_data["round_number"],
                        "turn_type": "bid",
                        "model": player_model,
                        "player_id": pid,
                        "auction_type": round_data["auction"]["type"],
                        "bid": bid,
                        "is_round_winner": player_round["is_winner"],
                    },
                })
                stats["bid_examples"] += 1
            else:
                stats["skipped_empty_response"] += 1

            # --- REVEAL TRAINING EXAMPLE ---
            # Only when this player won this round's treasure auction.
            if (
                include_reveals
                and game_state.current_auction.type == AuctionType.TREASURE
                and player_round["is_winner"]
                and player_round.get("gem_revealed")
            ):
                gem_revealed = player_round["gem_revealed"]
                reveal_data = player_round.get("reveal") or {}
                raw_reveal_response = reveal_data.get("raw_response", "")
                reveal_is_invalid = is_invalid_for_sft(reveal_data)

                if require_valid and reveal_is_invalid:
                    stats["skipped_invalid"] += 1
                elif raw_reveal_response and raw_reveal_response.strip():
                    reveal_prompt = generate_reveal_prompt(game_state, pid)

                    examples.append({
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": reveal_prompt},
                            {"role": "assistant", "content": raw_reveal_response},
                        ],
                        "metadata": {
                            "game_file": game_json_path.name,
                            "seed": seed,
                            "value_chart": value_chart,
                            "round": round_data["round_number"],
                            "turn_type": "reveal",
                            "model": player_model,
                            "player_id": pid,
                            "gem_revealed": gem_revealed,
                        },
                    })
                    stats["reveal_examples"] += 1
                else:
                    stats["skipped_empty_response"] += 1

        # Advance game state with this round's outcomes
        advance_state(game_state, round_data)

    return examples, stats


def resolve_seed_filter(seed_split: str) -> tuple[set[int], dict[str, list[int]]]:
    """Return selected seeds and the named splits used to build them.

    ``trainval`` is the default for SFT preprocessing: train and validation
    examples are exported, while test seeds stay out of the artifact.
    """
    if seed_split == "trainval":
        split_to_seeds = {
            "train": load_split_seeds("train"),
            "val": load_split_seeds("val"),
        }
    else:
        split_to_seeds = {
            seed_split: load_split_seeds(seed_split),
        }
    ensure_disjoint(split_to_seeds)
    selected = {seed for seeds in split_to_seeds.values() for seed in seeds}
    return selected, split_to_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT training data from top-2 model traces")
    parser.add_argument("--include-reveals", action="store_true", default=True)
    parser.add_argument("--no-reveals", action="store_false", dest="include_reveals")
    parser.add_argument("--include-selfplay", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed-split", choices=["train", "val", "validation", "trainval", "test"], default="trainval")
    parser.add_argument("--no-seed-filter", action="store_true")
    parser.add_argument("--require-valid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    input_dir = args.input_dir if args.input_dir is not None else RESULTS_DIR
    output_dir = args.output_dir if args.output_dir is not None else OUTPUT_DIR
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output) if args.output
        else output_dir / f"sft_{args.seed_split}.jsonl"
    )
    if args.no_seed_filter:
        allowed_seeds = None
        split_to_seeds = {}
        logging.info("Seed filter disabled (--no-seed-filter); processing every game in input dir")
    else:
        allowed_seeds, split_to_seeds = resolve_seed_filter(args.seed_split)
        logging.info(
            "Seed filter %s: %s",
            args.seed_split,
            {name: len(seeds) for name, seeds in split_to_seeds.items()},
        )

    # Collect game files. The v1 layout used src/megagem/evals/results/megagem_*.json;
    # the v2 layout (cost_probe_match.py / sft_data_gen.py) uses
    # cost_probe_game_seed_*.json under results/sft_v2_data_<ts>/.
    if args.input_dir is not None:
        game_files = sorted(input_dir.glob("cost_probe_game_seed_*.json"))
    else:
        game_files = sorted(input_dir.glob("megagem_*.json"))
        if args.include_selfplay:
            game_files += sorted(input_dir.glob("selfplay_*.json"))
    logging.info("Found %d game files in %s", len(game_files), input_dir)

    all_examples: list[dict] = []
    all_stats: list[dict] = []
    model_slot_counts: Counter = Counter()  # one increment per (game, eligible-teacher-player) pair
    games_processed = 0
    games_skipped = 0
    games_errored = 0
    total_warnings = 0

    for game_file in game_files:
        try:
            examples, stats = extract_game_examples(
                game_file,
                include_reveals=args.include_reveals,
                validate=args.validate,
                allowed_seeds=allowed_seeds,
                require_valid=args.require_valid,
            )
        except Exception as e:
            logging.error("Failed to process %s: %s", game_file.name, e)
            games_errored += 1
            continue

        if stats.get("skipped"):
            games_skipped += 1
            continue

        games_processed += 1
        all_examples.extend(examples)
        all_stats.append(stats)
        # One increment per eligible top-2 player slot. In single-teacher
        # self-play this counts each game twice for that teacher (both top-2
        # players are the same model) — that's the player-slot semantic.
        for pid in stats["eligible_player_ids"]:
            slot_model = stats["top2_models"][stats["top2_player_ids"].index(pid)]
            model_slot_counts[slot_model] += 1
        total_warnings += stats.get("alignment_warnings", 0)

        if games_processed % 50 == 0:
            logging.info(
                "Processed %d games, %d examples so far",
                games_processed,
                len(all_examples),
            )

    # Write training JSONL (skip on dry-run)
    if not args.dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Compute summary
    bid_examples = sum(
        1 for ex in all_examples if ex["metadata"]["turn_type"] == "bid"
    )
    reveal_examples = sum(
        1 for ex in all_examples if ex["metadata"]["turn_type"] == "reveal"
    )
    skipped_empty_response = sum(s.get("skipped_empty_response", 0) for s in all_stats)
    skipped_invalid = sum(s.get("skipped_invalid", 0) for s in all_stats)

    examples_by_model = {}
    for model in V2_TEACHERS:
        examples_by_model[model] = {
            "bid": sum(
                1 for ex in all_examples
                if ex["metadata"]["model"] == model
                and ex["metadata"]["turn_type"] == "bid"
            ),
            "reveal": sum(
                1 for ex in all_examples
                if ex["metadata"]["model"] == model
                and ex["metadata"]["turn_type"] == "reveal"
            ),
        }

    # Auction type breakdown
    auction_type_counts = Counter(
        ex["metadata"]["auction_type"]
        for ex in all_examples
        if ex["metadata"]["turn_type"] == "bid"
    )

    summary = {
        "total_game_files": len(game_files),
        "games_with_v2_teacher": games_processed,
        "games_skipped": games_skipped,
        "games_errored": games_errored,
        "total_examples": len(all_examples),
        "bid_examples": bid_examples,
        "reveal_examples": reveal_examples,
        "skipped_empty_response": skipped_empty_response,
        "skipped_invalid": skipped_invalid,
        "require_valid": args.require_valid,
        "alignment_warnings": total_warnings,
        "seed_split": "no-filter" if args.no_seed_filter else args.seed_split,
        "split_seed_counts": {name: len(seeds) for name, seeds in split_to_seeds.items()},
        "selected_seeds": (
            None if allowed_seeds is None else sorted(allowed_seeds)
        ),
        "examples_by_model": examples_by_model,
        "slots_by_model": dict(model_slot_counts),
        "bid_examples_by_auction_type": dict(auction_type_counts),
        "output_file": str(output_path),
    }

    stats_path = output_path.with_name(f"{output_path.stem}_stats.json")
    if not args.dry_run:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # Print summary
    print(f"\nSFT Data Preparation {'Dry-Run' if args.dry_run else 'Complete'}")
    print(f"{'=' * 50}")
    print(f"Games processed: {games_processed} / {len(game_files)}")
    print(f"Seed filter:     {'(none)' if args.no_seed_filter else args.seed_split}")
    if games_errored:
        print(f"Games errored:   {games_errored}")
    if total_warnings:
        print(f"Alignment warnings: {total_warnings}")
    print(f"\nTotal examples:  {len(all_examples)}")
    print(f"  Bid examples:    {bid_examples}")
    print(f"  Reveal examples: {reveal_examples}")
    print(f"  Skipped (empty raw_response): {skipped_empty_response}")
    if args.require_valid:
        print(f"  Skipped (invalid: parse/legal/default): {skipped_invalid}")
    print("\nBy auction type (bid examples):")
    for atype, count in sorted(auction_type_counts.items()):
        print(f"  {atype}: {count}")
    print("\nBy model:")
    for model in sorted(V2_TEACHERS):
        n_slots = model_slot_counts[model]
        m_stats = examples_by_model[model]
        total = m_stats["bid"] + m_stats["reveal"]
        print(f"  {model}: {n_slots} player slots, {total} examples "
              f"({m_stats['bid']} bids, {m_stats['reveal']} reveals)")
    if args.dry_run:
        print("\n(dry-run — nothing written)")
    else:
        print(f"\nOutput: {output_path}")
        print(f"Stats:  {stats_path}")


if __name__ == "__main__":
    main()
