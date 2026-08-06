#!/usr/bin/env python3
"""Phase 0.3 rollout throughput smoke (the RL plan §0.3, §9): N games against an
OpenAI-compatible endpoint, reports games/hour. Gate ≥30 = plan stands;
10–30 = scale down; <10 = stay on the toy game.

  uv run python -m megagem.evals.game_runner \\
      --model megagem-base --vllm-url http://localhost:8000/v1 \\
      --num-games 10 --max-parallel 4 \\
      --trajectory-dir results/throughput_trajectories \\
      --output results/throughput_smoke.json

``--vllm-url`` registers the model with key=EMPTY so run_game forwards
``enable_thinking=False`` (Thinking-base compat; no-op on Instruct).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

from megagem import endpoints
from megagem.rollout import run_game


PHASE_0_3_TARGET_GAMES_PER_HOUR = 30
PHASE_0_3_FLOOR_GAMES_PER_HOUR = 10


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    rank = max(1, int(round(p / 100.0 * len(s))))
    return float(s[min(rank - 1, len(s) - 1)])


def register_vllm_endpoint(
    model: str,
    served_model_name: str | None,
    vllm_url: str,
    api_key_env: str,
    request_timeout_s: float = 120.0,
) -> None:
    """Register ``model`` as a local vLLM endpoint in megagem.endpoints.

    key="EMPTY" triggers run_game's has_vllm path (forwards enable_thinking=False),
    and the env var must actually exist for os.getenv to read it.
    """
    if api_key_env == "EMPTY" and not os.getenv("EMPTY"):
        os.environ["EMPTY"] = "EMPTY"
    endpoints.ENDPOINTS[model] = {
        "model": served_model_name or model,
        "url": vllm_url,
        "key": api_key_env,
        "timeout": float(request_timeout_s),
    }


async def time_one_game(
    model: str,
    seed: int,
    value_chart: str,
    num_players: int,
    trajectory_dir: Path | None,
) -> dict:
    output_file = "trajectory" if trajectory_dir is not None else None
    json_filename = f"smoke_seed{seed}.json" if trajectory_dir is not None else None

    t0 = time.perf_counter()
    state = await run_game(
        models=[model] * num_players,
        value_chart=value_chart,
        seed=seed,
        num_players=num_players,
        output_file=output_file,
        json_filename=json_filename,
        quiet=True,
        game_label=f"seed={seed}",
        results_dir=trajectory_dir,
    )
    elapsed = time.perf_counter() - t0
    return {
        "seed": seed,
        "elapsed_s": elapsed,
        "num_rounds": state.get("num_rounds"),
        "model_chat_times_s": state.get("model_chat_times_seconds"),
        "winner_id": state.get("winner_id"),
    }


async def run_smoke(
    model: str,
    num_games: int,
    max_parallel: int,
    seed_start: int,
    value_chart: str,
    num_players: int,
    trajectory_dir: Path | None,
) -> dict:
    sem = asyncio.Semaphore(max_parallel)

    async def with_sem(seed: int) -> dict:
        async with sem:
            return await time_one_game(
                model, seed, value_chart, num_players, trajectory_dir
            )

    seeds = list(range(seed_start, seed_start + num_games))

    overall_t0 = time.perf_counter()
    per_game = await asyncio.gather(*(with_sem(s) for s in seeds))
    total_elapsed = time.perf_counter() - overall_t0

    elapsed_list = [r["elapsed_s"] for r in per_game]
    games_per_hour = (num_games / total_elapsed) * 3600

    endpoint = endpoints.ENDPOINTS.get(model, {})

    return {
        "config": {
            "model": model,
            "endpoint_url": endpoint.get("url"),
            "endpoint_key_env": endpoint.get("key"),
            "served_model": endpoint.get("model"),
            "num_games": num_games,
            "max_parallel": max_parallel,
            "value_chart": value_chart,
            "num_players": num_players,
            "seed_range": [seed_start, seed_start + num_games - 1],
            "trajectory_dir": str(trajectory_dir) if trajectory_dir else None,
        },
        "wall_clock_total_s": round(total_elapsed, 3),
        "games_per_hour": round(games_per_hour, 2),
        "per_game_elapsed_s": {
            "mean": round(statistics.mean(elapsed_list), 3),
            "median": round(statistics.median(elapsed_list), 3),
            "p95": round(_percentile(elapsed_list, 95), 3),
            "min": round(min(elapsed_list), 3),
            "max": round(max(elapsed_list), 3),
        },
        "phase_0_3_gate": {
            "target_games_per_hour": PHASE_0_3_TARGET_GAMES_PER_HOUR,
            "floor_games_per_hour": PHASE_0_3_FLOOR_GAMES_PER_HOUR,
            "passed_target": games_per_hour >= PHASE_0_3_TARGET_GAMES_PER_HOUR,
            "above_floor": games_per_hour >= PHASE_0_3_FLOOR_GAMES_PER_HOUR,
        },
        "per_game": per_game,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--model", required=True)
    p.add_argument("--vllm-url", default=None)
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--api-key-env", default="EMPTY")
    p.add_argument("--num-games", type=int, default=10)
    p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--value-chart", default="A")
    p.add_argument("--num-players", type=int, default=3)
    p.add_argument('--trajectory-dir', type=Path, default=None)
    p.add_argument('--output', type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.vllm_url:
        register_vllm_endpoint(
            args.model,
            args.served_model_name,
            args.vllm_url,
            args.api_key_env,
        )
    elif args.model not in endpoints.ENDPOINTS:
        print(
            f"WARNING: model {args.model!r} not in megagem/endpoints.py and "
            "no --vllm-url given; run_game will fall back to PRIME_API_KEY "
            "and enable_thinking=False will NOT be applied.",
            file=sys.stderr,
        )

    if args.trajectory_dir is not None:
        args.trajectory_dir.mkdir(parents=True, exist_ok=True)

    summary = asyncio.run(
        run_smoke(
            model=args.model,
            num_games=args.num_games,
            max_parallel=args.max_parallel,
            seed_start=args.seed_start,
            value_chart=args.value_chart,
            num_players=args.num_players,
            trajectory_dir=args.trajectory_dir,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    gph = summary["games_per_hour"]
    gate = summary["phase_0_3_gate"]
    print(f"\nThroughput: {gph} games/hour over {args.num_games} games "
          f"(parallel={args.max_parallel}, total={summary['wall_clock_total_s']}s)")
    print(f"Per-game elapsed: mean={summary['per_game_elapsed_s']['mean']}s, "
          f"p95={summary['per_game_elapsed_s']['p95']}s")
    if args.trajectory_dir is not None:
        print(f"Trajectories written to: {args.trajectory_dir}")
    if gate["passed_target"]:
        print(f"PASS — clears the {PHASE_0_3_TARGET_GAMES_PER_HOUR} games/hr target.")
        return 0
    if gate["above_floor"]:
        print(f"PARTIAL — above the {PHASE_0_3_FLOOR_GAMES_PER_HOUR} games/hr floor "
              f"but below the {PHASE_0_3_TARGET_GAMES_PER_HOUR} target. "
              "the phase-3 plan should scale down per §9.")
        return 0
    print(f"FAIL — below the {PHASE_0_3_FLOOR_GAMES_PER_HOUR} games/hr floor. "
          "Stay on the toy game longer per the RL plan §9.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
