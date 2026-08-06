#!/usr/bin/env python3
"""One-match API cost probe for the SFT v2 teacher trio.

Runs one MegaGem game between Gemini 3.1 Pro Preview, Gemini 3 Flash Preview,
and Claude Opus 4.6, then writes:
  - <results_dir>/<game_log>.json                  schema-v3 game log via run_game
  - <results_dir>/cost_probe_summary_seed_<seed>.json   per-model usage + $ estimate

Token counts come from each chat completion's ``usage`` field via a
monkey-patch on ``AsyncCompletions.create``. USD numbers are rough — override
``DEFAULT_PRICES_USD_PER_1M`` or pass ``--prices-json`` for a tighter estimate.

    export PRIME_API_KEY=...
    uv run python scripts/eval/cost_probe_match.py --seed 1000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


DEFAULT_MODELS = [
    "anthropic/claude-opus-4.7",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-opus-4.6",
]

DEFAULT_PRICES_USD_PER_1M: dict[str, dict[str, float]] = {
    "google/gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "google/gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "anthropic/claude-opus-4.5":     {"input": 5.00, "output": 25.00},
    "anthropic/claude-opus-4.6":    {"input": 5.00, "output": 25.00},
    "anthropic/claude-opus-4.7":    {"input": 5.00, "output": 25.00},
}


class UsageCapture:
    """Accumulates per-call ``usage`` records. asyncio is single-threaded so no lock."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(self, model: str | None, usage) -> None:
        if usage is None or model is None:
            return
        p = int(getattr(usage, "prompt_tokens", 0) or 0)
        c = int(getattr(usage, "completion_tokens", 0) or 0)
        t = int(getattr(usage, "total_tokens", None) or (p + c))
        self.records.append(
            {"model": model, "prompt_tokens": p, "completion_tokens": c, "total_tokens": t}
        )


def install_usage_hook(capture: UsageCapture) -> None:
    """Wrap ``AsyncCompletions.create`` so every response's usage flows into capture."""
    from openai.resources.chat.completions import AsyncCompletions

    original = AsyncCompletions.create

    async def wrapped(self, *args, **kwargs):
        resp = await original(self, *args, **kwargs)
        capture.record(kwargs.get("model"), getattr(resp, "usage", None))
        return resp

    AsyncCompletions.create = wrapped  # type: ignore[assignment]


def find_price(model: str, prices: dict[str, dict[str, float]]) -> dict[str, float] | None:
    if model in prices:
        return prices[model]
    suffix = model.split("/", 1)[1] if "/" in model else None
    return prices.get(suffix) if suffix else None


def summarize(
    capture: UsageCapture, prices: dict[str, dict[str, float]]
) -> tuple[dict, float, int]:
    totals: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    for r in capture.records:
        t = totals[r["model"]]
        t["calls"] += 1
        t["prompt_tokens"] += r["prompt_tokens"]
        t["completion_tokens"] += r["completion_tokens"]
        t["total_tokens"] += r["total_tokens"]

    grand = 0.0
    per_model: dict[str, dict] = {}
    for m, t in totals.items():
        price = find_price(m, prices)
        cost: float | None = None
        if price is not None:
            cost = (
                t["prompt_tokens"] / 1e6 * price["input"]
                + t["completion_tokens"] / 1e6 * price["output"]
            )
            grand += cost
        per_model[m] = {
            **t,
            "price_used_usd_per_1m": price,
            "estimated_cost_usd": round(cost, 6) if cost is not None else None,
        }
    return per_model, grand, len(capture.records)


async def run_match(
    models: list[str],
    seed: int,
    value_chart: str,
    num_players: int,
    results_dir: Path,
    json_filename: str,
    game_label: str,
    quiet: bool,
) -> None:
    from megagem.rollout import run_game

    results_dir.mkdir(parents=True, exist_ok=True)
    await run_game(
        models=models,
        value_chart=value_chart,
        seed=seed,
        num_players=num_players,
        output_file="dummy.txt",  # presence triggers JSON write inside run_game
        json_filename=json_filename,
        quiet=quiet,
        game_label=game_label,
        results_dir=results_dir,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--value-chart", default="A", choices=["A", "B", "C", "D", "E"])
    p.add_argument("--num-players", type=int, default=3)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "cost_probe")
    p.add_argument("--prices-json", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if len(args.models) != args.num_players:
        print(
            f"ERROR: --models has {len(args.models)} entries but --num-players={args.num_players}",
            file=sys.stderr,
        )
        return 2

    prices = dict(DEFAULT_PRICES_USD_PER_1M)
    if args.prices_json is not None:
        prices.update(json.loads(args.prices_json.read_text()))

    capture = UsageCapture()
    install_usage_hook(capture)

    game_json_filename = f"cost_probe_game_seed_{args.seed}.json"

    args.results_dir.mkdir(parents=True, exist_ok=True)
    game_log_path = args.results_dir / game_json_filename
    cost_summary_path = args.results_dir / f"cost_probe_summary_seed_{args.seed}.json"

    print(f"Cost-probe match: seed={args.seed}, chart={args.value_chart}")
    print(f"Models: {args.models}")
    print(f"Game log     -> {game_log_path}")
    print(f"Cost summary -> {cost_summary_path}")

    t0 = datetime.now()
    asyncio.run(
        run_match(
            models=args.models,
            seed=args.seed,
            value_chart=args.value_chart,
            num_players=args.num_players,
            results_dir=args.results_dir,
            json_filename=game_json_filename,
            game_label=f"cost_probe_seed_{args.seed}",
            quiet=not args.verbose,
        )
    )
    wall_seconds = (datetime.now() - t0).total_seconds()

    per_model, grand_total, total_calls = summarize(capture, prices)
    rel = (
        str(game_log_path.relative_to(REPO_ROOT))
        if REPO_ROOT in game_log_path.parents
        else str(game_log_path)
    )
    summary = {
        "metadata": {
            "seed": args.seed,
            "value_chart": args.value_chart,
            "num_players": args.num_players,
            "models": args.models,
            "timestamp": datetime.now().isoformat(),
            "wall_seconds": round(wall_seconds, 2),
            "game_log_path": rel,
            "price_table_usd_per_1m_tokens": prices,
            "price_note": "Estimates only. Verify against the Prime Inference bill.",
        },
        "totals": {
            "calls": total_calls,
            "estimated_cost_usd": round(grand_total, 6) if total_calls else 0.0,
        },
        "per_model": per_model,
    }
    cost_summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 80)
    print("API USAGE SUMMARY")
    print("=" * 80)
    header = (
        f"{'model':42s}  {'calls':>5s}  {'prompt':>8s}  {'completion':>10s}  "
        f"{'total':>8s}  {'est $':>9s}"
    )
    print(header)
    print("-" * len(header))
    for m, t in per_model.items():
        cost_str = (
            f"{t['estimated_cost_usd']:>9.4f}"
            if t["estimated_cost_usd"] is not None
            else f"{'n/a':>9s}"
        )
        print(
            f"{m:42s}  {t['calls']:>5d}  {t['prompt_tokens']:>8d}  "
            f"{t['completion_tokens']:>10d}  {t['total_tokens']:>8d}  {cost_str}"
        )
    print("-" * len(header))
    print(
        f"Grand total: {total_calls} calls, estimated cost ${grand_total:.4f} "
        f"(wall {wall_seconds:.1f}s)"
    )
    print(f"\nGame log:     {game_log_path}")
    print(f"Cost summary: {cost_summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
