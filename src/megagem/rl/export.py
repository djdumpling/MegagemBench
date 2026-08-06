"""Per-row training-row contract + offline-pass CLI (Phase 1, scope add-on).

Each trainable turn becomes one row carrying the keys the existing TRL seam
consumes. ``MegaGemGRPOTrainer._replacement_advantages`` builds
``torch.as_tensor([row[PRECOMPUTED_ADVANTAGE_KEY] for row in inputs])`` and
validates its shape against ``output["advantages"]`` ``(B,)`` (or
``completion_mask`` ``(B, T)``). A *per-turn scalar per row* satisfies the
``(B,)`` contract directly — locking that here means Phase 2/3 wire the
rollout to these rows unchanged.

Importing the key constants from ``megagem.training.megagem_grpo`` (rather than
re-declaring them) is the contract bond; that module's TRL/torch imports are
guarded, so this works with no RL extra installed and no GPU.

CLI::

    python -m megagem.rl.export --corpus <DIR> --report
    python -m megagem.rl.export --corpus <DIR> --out rows.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any, Sequence

from .advantage import AdvantageConfig, compute_advantages, summarize
from .diagnostics import reveal_policy_diagnostic, shaping_terminal_mass
from .reward import RewardConfig, assert_corpus_actor_ids, calibrate_scale
from ..training.megagem_grpo import (
    PRECOMPUTED_ADVANTAGE_KEY,
    PRECOMPUTED_REWARD_KEY,
)


def build_training_rows(
    transcripts: Sequence[dict[str, Any]],
    reward_cfg: RewardConfig | None = None,
    adv_cfg: AdvantageConfig | None = None,
) -> list[dict[str, Any]]:
    """One dict per trainable turn, in the deterministic processing order."""
    rows = compute_advantages(transcripts, reward_cfg, adv_cfg)
    out: list[dict[str, Any]] = []
    for r in rows:
        if not r.trainable:
            continue  # actor-masked (§A.6): excluded from the training batch
        out.append(
            {
                "prompt": r.prompt,
                "completion": r.completion,
                PRECOMPUTED_REWARD_KEY: float(r.reward),
                PRECOMPUTED_ADVANTAGE_KEY: float(r.advantage),
                "reward_components": {
                    "legal": float(r.legal),
                    "shaping": float(r.shaping),
                    "terminal_correction": float(r.terminal_correction),
                    "terminal": float(r.terminal_reward),
                },
                "actor_id": r.actor_id,
                "bucket": list(r.bucket),
                "ema_bucket": list(r.ema_bucket),
                "group_key": repr(r.group_key),
                "game_id": r.game_id,
                "round_index": r.round_index,
                "player_id": r.player_id,
                "phase": r.phase,
                "is_terminal_turn": r.is_terminal_turn,
            }
        )
    return out


def contract_check(rows: Sequence[dict[str, Any]]) -> int:
    """Emulate ``_replacement_advantages``' expectations without torch.

    Every row must carry both keys as finite floats; returns the ``(B,)``
    batch length the TRL seam would build. Raises on violation.
    """
    for i, row in enumerate(rows):
        for key in (PRECOMPUTED_ADVANTAGE_KEY, PRECOMPUTED_REWARD_KEY):
            if key not in row:
                raise KeyError(f"row {i} missing {key!r}")
            val = row[key]
            if not isinstance(val, float) or not math.isfinite(val):
                raise ValueError(f"row {i} {key!r} not a finite float: {val!r}")
    return len(rows)  # tensor shape (B,), one scalar per row


def load_corpus(
    corpus_dir: str, pattern: str = "*.json", *, assert_schema: bool = True
) -> list[dict[str, Any]]:
    """Load schema-v3 transcripts sorted by path (deterministic order).

    By default this enforces the Caveat 4 corpus schema gate: every bid/reveal
    turn must carry an explicit ``actor_id`` or the whole corpus is rejected
    (before any training-row export / Phase-3 spend). Pass
    ``assert_schema=False`` only for tooling that deliberately inspects a
    raw/legacy corpus.
    """
    out: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "**", pattern), recursive=True)):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and data.get("metadata", {}).get("schema_version") == 3:
            out.append(data)
    if assert_schema and out:
        assert_corpus_actor_ids(out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 1 offline reward/advantage pass")
    ap.add_argument("--corpus", required=True, help="dir of schema-v3 transcripts")
    ap.add_argument("--pattern", default="*.json")
    ap.add_argument("--report", action="store_true", help="print summary + SCALE calibration")
    ap.add_argument("--out", help="write training rows as JSONL")
    args = ap.parse_args(argv)

    corpus = load_corpus(args.corpus, args.pattern)
    if not corpus:
        print(f"no schema-v3 transcripts under {args.corpus!r}")
        return 1

    rows = build_training_rows(corpus)
    b = contract_check(rows)
    adv_rows = compute_advantages(corpus)

    if args.report:
        report = {
            "corpus_games": len(corpus),
            "training_rows_B": b,
            "scale_calibration": calibrate_scale(corpus),
            # Caveat 1 rec #1: the shaping-vs-terminal mass + proxy-vs-true
            # panel the original Phase 1 report was blind to.
            "shaping_terminal_diagnostic": shaping_terminal_mass(corpus),
            # Caveat 2 addendum: reveal-shaping telemetry (observational,
            # CONFOUNDED by player strength — NOT a causal gate). Track
            # mean_signed_reveal_shaping as a trend on a FIXED held-out eval
            # corpus across iterations; the causal test is the within-state
            # counterfactual probe (reads hand; see the addendum).
            "reveal_policy_diagnostic": reveal_policy_diagnostic(corpus),
            "advantage_summary": summarize(adv_rows),
        }
        print(json.dumps(report, indent=2, default=str))

    if args.out:
        with open(args.out, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
