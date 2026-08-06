"""Phase 2.1 toy-corpus driver — the CPU/offline analog of the LLM game runner
(``megagem.evals.game_runner``), but for the toy sealed-bid auction (no vLLM,
no LLM, no GPU, deterministic).

For each (seed, k) it runs one toy game where the trainable seat uses a
deterministic stub (with per-k jitter) and the other two seats are heuristic
opponents, then writes a schema-v3 transcript to
``<out_dir>/chart_<chart>/toy_seed<seed>_r<k>.json``.

True K-group: the K rollouts of a seed share ``metadata.seed`` (so
``rl.advantage.default_group_key == (seed, value_chart)`` groups them) and the
opponents are identical across k — only the trainable stub varies (jitter), so
``export --report`` reports ``group_norm.mode == "true K-group"`` when
``--rollouts-per-seed > 1``.

Validate the output with the existing, unmodified pipeline::

    python -m megagem.rl.export --corpus <out_dir>/chart_<chart> --report
    python -m megagem.rl.export --corpus <out_dir>/chart_<chart> --out rows.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from megagem.toy.auction_env import run_one_toy_game
from megagem.toy.prompt import make_heuristic_policy, make_stub_policy


def build_game(
    *,
    seed: int,
    rollout_index: int,
    trainable_seat: int,
    opponent_actor_id: str,
    strategy: str,
    num_players: int,
    num_rounds: int,
    value_chart: str,
) -> dict:
    """One toy game: stub (jittered by ``rollout_index``) at the trainable
    seat, deterministic heuristic opponents elsewhere.

    Self-guards the seat range so this seam is safe when called directly
    (e.g. Phase 2.2's rollout_func) — not only via the CLI: an out-of-range
    seat would otherwise produce an all-heuristic game with zero trainable
    rows and no error.
    """
    if not (0 <= trainable_seat < num_players):
        raise ValueError(
            f"trainable_seat must be in [0, {num_players}); got "
            f"{trainable_seat} → a game with zero trainable rows."
        )
    heuristic = make_heuristic_policy()
    stub = make_stub_policy(strategy, jitter_seed=rollout_index)
    policy_fns = [
        stub if s == trainable_seat else heuristic for s in range(num_players)
    ]
    actor_ids = [
        "trainable" if s == trainable_seat else opponent_actor_id
        for s in range(num_players)
    ]
    return run_one_toy_game(
        seed,
        policy_fns,
        actor_ids,
        rollout_index=rollout_index,
        num_players=num_players,
        num_rounds=num_rounds,
        value_chart=value_chart,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-seeds", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=3000)
    ap.add_argument("--trainable-seat", type=int, default=0)
    ap.add_argument("--opponent-actor-id", default="heuristic")
    ap.add_argument("--rollouts-per-seed", type=int, default=1)
    ap.add_argument(
        "--strategy",
        default="bestresp",
        choices=["bestresp", "overbid", "illegal", "garbage"],
    )
    ap.add_argument(
        "--num-players", type=int, default=3,
        help="LOCKED to 3 (STARTING_COINS == scorer._STARTING_COINS[3]).",
    )
    ap.add_argument(
        "--num-rounds", type=int, default=8,
        help="design default 8; other values are experimental (src/megagem/rl is "
        "agnostic, but the Phase 2.1 gate was characterized at 8).",
    )
    ap.add_argument("--value-chart", default="A")
    args = ap.parse_args(argv)

    if args.num_players != 3:
        ap.error("--num-players is locked to 3 for the toy (see help).")
    if not (0 <= args.trainable_seat < args.num_players):
        ap.error(
            f"--trainable-seat must be in [0, {args.num_players}); got "
            f"{args.trainable_seat} → a corpus with zero trainable rows."
        )

    chart_dir = Path(args.out_dir) / f"chart_{args.value_chart}"
    chart_dir.mkdir(parents=True, exist_ok=True)
    # Clear our own stale artifacts so a re-run can't contaminate a later
    # `export --report` with games from a previous (different) invocation.
    stale = list(chart_dir.glob("toy_seed*_r*.json"))
    for old in stale:
        old.unlink()
    if stale:
        print(f"[toy_corpus] removed {len(stale)} stale toy_seed*_r*.json")

    written = 0
    for s in range(args.seed_start, args.seed_start + args.num_seeds):
        for k in range(args.rollouts_per_seed):
            game = build_game(
                seed=s,
                rollout_index=k,
                trainable_seat=args.trainable_seat,
                opponent_actor_id=args.opponent_actor_id,
                strategy=args.strategy,
                num_players=args.num_players,
                num_rounds=args.num_rounds,
                value_chart=args.value_chart,
            )
            path = chart_dir / f"toy_seed{s}_r{k}.json"
            path.write_text(json.dumps(game, indent=2))
            written += 1

    print(f"[toy_corpus] wrote {written} schema-v3 games -> {chart_dir}")
    print("[toy_corpus] validate with the existing, unmodified pipeline:")
    print(f"  python -m megagem.rl.export --corpus {chart_dir} --report")
    print(f"  python -m megagem.rl.export --corpus {chart_dir} --out toy_rows.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
