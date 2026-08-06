"""Counterfactual reveal-choice analysis — PRIVATE-HAND, POST-HOC ONLY.

⚠️ **This module READS the winner's private hand.** It exists solely to
answer, *confound-free*, the Caveat-2-addendum question the hand-free
observational telemetry (`megagem.rl.diagnostics.reveal_policy_diagnostic`)
provably cannot: **did the policy steer toward self-favoring reveals among
the legal options it actually had?** Differencing *within a single reveal
state* cancels the player-strength confound (a strong position drives both
larger `reveal_shaping` and a higher terminal margin) that makes the
observational `corr(reveal_shaping, margin)` uninterpretable as a causal gate.

FIREWALL (enforced by ``tests/test_reveal_counterfactual.py``):

* **Not in ``src/megagem/rl/``** — the hand-free leakage gate on the scorer / reward
  path (``tests/test_rl_scorer.py``: scrub-hands-→-identical-scores, no
  ``hand`` field on ``TurnStateSnapshot``) is untouched and stays intact.
* **Not imported by** ``megagem.rl.reward`` / ``advantage`` / ``scorer`` /
  ``diagnostics`` / ``export`` (so not by ``load_corpus`` /
  ``build_training_rows`` / the default ``--report``). Import is
  one-directional: this module pulls hand-free primitives from ``megagem.rl``;
  nothing in ``megagem.rl`` pulls from here.
* **Opt-in CLI only** — refuses to run without an explicit
  ``--i-understand-this-reads-private-hands`` consent flag; never auto-run.

It reuses the *exact* hand-free scoring primitive
(`score_snapshot_with_display`) and snapshots (`reconstruct_snapshots`) from
the production path; the only private input is the choice set (the winner's
hand colours at decision time). For the colour actually revealed the
counterfactual equals the production `reveal_shaping` by construction
(asserted in tests) — this is a re-scoring of legal alternatives, not a
re-derivation of the reward.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Any, Sequence

from ..data import load_value_charts
from ..game.cards import ValueChart
from ..rl.reward import RewardConfig
from ..rl.scorer import (
    parse_transcript,
    reconstruct_snapshots,
    score_snapshot_with_display,
)

_SCRUB_TOKEN = "__SCRUBBED__"

ANALYSIS_LABEL = (
    "PRIVATE-HAND counterfactual (post-hoc; reads winners' private hands — "
    "do not share raw output / treat as sensitive)"
)


def _chart(value_chart_id: str, _cache: dict[str, ValueChart] = {}) -> ValueChart:
    if value_chart_id not in _cache:
        charts = load_value_charts()
        data = charts.get(value_chart_id) or charts["A"]
        _cache[value_chart_id] = ValueChart.from_dict(value_chart_id, data)
    return _cache[value_chart_id]


def _decision_hand_colors(
    player_rec: dict[str, Any], chosen_color: str | None
) -> tuple[set[str], bool]:
    """Colours the winner could legally have revealed at decision time.

    The recorded ``hand`` is the *post-reveal* snapshot (verified against real
    corpora: the revealed card is already gone). Re-adding ``chosen_color``
    reconstructs the decision-time colour set; the union is also correct if a
    writer ever recorded the pre-reveal hand (idempotent on colours).

    Returns ``(colours, scrubbed)`` — ``scrubbed`` True if the hand was
    sanitised away (then the choice set is unrecoverable and the turn must be
    skipped, not silently mis-analysed).
    """
    recorded = player_rec.get("hand")
    recorded = recorded if isinstance(recorded, list) else []
    if any(c == _SCRUB_TOKEN for c in recorded):
        return set(), True
    colours = {c for c in recorded if isinstance(c, str)}
    if chosen_color:
        colours.add(chosen_color)
    return colours, False


def _pre_reveal_display(
    parsed: Any, round_index: int
) -> tuple[dict[str, int], dict[str, int]]:
    """(count, value_per_gem) per colour *before* round ``round_index``'s reveal.

    Only a reveal mutates the display across a round boundary, so round
    ``r-1``'s recorded post-reveal display *is* round ``r``'s pre-reveal
    display; empty before round 1.
    """
    if round_index <= 0:
        return {}, {}
    vd = parsed.rounds[round_index - 1].get("value_display", {}) or {}
    counts = {c: int(i.get("count", 0)) for c, i in vd.items()}
    vpg = {c: int(i.get("value_per_gem", 0)) for c, i in vd.items()}
    return counts, vpg


def _legal_policy_reveal(reveal: dict[str, Any]) -> bool:
    """Mirror ``reward._legal_reward``: a defaulted/illegal reveal is not a
    policy choice and is excluded from the choice signal."""
    return bool(reveal.get("parse_valid")) and bool(reveal.get("legal_valid"))


def reveal_counterfactual_diagnostic(
    transcripts: Sequence[dict[str, Any]],
    reward_cfg: RewardConfig | None = None,
    *,
    min_choice_turns: int = 30,
    selfish_rank_floor: float = 0.65,
    excess_floor: float = 0.0,
) -> dict[str, Any]:
    """Within-state, confound-free reveal-choice analysis (reads private hands).

    For every **policy** reveal turn (treasure winner; reveal parsed *and*
    legal) we re-score, with the production hand-free primitive, the
    `reveal_shaping` the winner *would* have received for **each colour in
    its decision-time hand**, holding the round state fixed. Per turn:

    * ``self_favoring_rank`` — normalised **mid-rank** of the chosen reveal
      among its legal options: ``(midrank − 1)/(k − 1)``. 1.0 ⇒ the strictly
      most self-favouring legal reveal, 0.0 ⇒ the strictly worst, and the
      uniform-random-choice expectation is **exactly 0.5 for every k** (this
      is the calibrated replacement for the inclusive CDF ``#≤chosen/k``,
      whose random null is the k-dependent ``(k+1)/2k`` ≈ 0.75/0.67/…). Ties
      handled by midrank.
    * ``excess`` — chosen shaping − mean over the legal options (λ-units);
      its uniform-random-choice expectation is **exactly 0** by construction
      (``E[chosen] = mean(options)``), k-invariant — the primary co-gate.

    Turns with <2 legal colours are *forced* (no choice) — counted, excluded
    from the choice metrics. Scrubbed/sanitised hands are skipped (unanalysable
    by construction, e.g. the tracked fixture).

    Because differencing is within one state, the player-strength confound
    cancels: a mean rank well above the 0.5 random null (with positive excess)
    is a **causal** statement that the policy selects self-favouring reveals
    among the options it had — though it does not attribute *why* (SFT prior
    vs. the λ·shaping nudge). This is the addendum's escalation trigger.
    """
    reward_cfg = reward_cfg or RewardConfig()
    lam = reward_cfg.shaping_lambda

    ranks: list[float] = []
    excesses: list[float] = []
    n_argmax = 0
    n_reveal = n_default = n_forced = n_scrubbed = 0

    for data in transcripts:
        parsed = parse_transcript(data)
        n = parsed.num_players
        chart = _chart(parsed.value_chart)
        snaps = reconstruct_snapshots(parsed)

        for r_idx, rnd in enumerate(parsed.rounds):
            for prec in rnd.get("players", []):
                reveal = prec.get("reveal")
                if not isinstance(reveal, dict):
                    continue
                n_reveal += 1
                w = prec.get("player_id")
                if not _legal_policy_reveal(reveal):
                    n_default += 1
                    continue
                chosen = (
                    prec.get("gem_revealed")
                    or reveal.get("final_reveal")
                    or reveal.get("parsed_action")
                )
                colours, scrubbed = _decision_hand_colors(prec, chosen)
                if scrubbed:
                    n_scrubbed += 1
                    continue
                if chosen not in colours or len(colours) < 2:
                    n_forced += 1
                    continue
                if any(snaps.get((r_idx, p)) is None for p in range(n)):
                    n_forced += 1
                    continue

                pre_counts, base_map = _pre_reveal_display(parsed, r_idx)

                def cf_shaping(colour: str) -> float:
                    cf_map = dict(base_map)
                    cf_map[colour] = chart.get_gem_value(
                        pre_counts.get(colour, 0) + 1
                    )
                    deltas: dict[int, float] = {}
                    for p in range(n):
                        snap = snaps[(r_idx, p)]
                        post = score_snapshot_with_display(snap, cf_map)[
                            "final_score"
                        ]
                        pre = score_snapshot_with_display(snap, base_map)[
                            "final_score"
                        ]
                        deltas[p] = float(post - pre)
                    others = [deltas[q] for q in range(n) if q != w]
                    return lam * (
                        deltas[w]
                        - (statistics.fmean(others) if others else 0.0)
                    )

                vals = list(cf_shaping(c) for c in colours)
                chosen_v = cf_shaping(chosen)
                k = len(vals)
                # Normalised mid-rank of the chosen reveal among its legal
                # options: 1.0 = strictly best (most self-favoring), 0.0 =
                # strictly worst, and — crucially — the uniform-random-choice
                # expectation is exactly 0.5 for EVERY k (not the inclusive
                # CDF's k-dependent (k+1)/2k). Tie-robust via midrank.
                less = sum(1 for v in vals if v < chosen_v - 1e-12)
                equal = sum(1 for v in vals if abs(v - chosen_v) <= 1e-12)
                midrank = less + (equal + 1) / 2.0  # 1-based average rank
                ranks.append((midrank - 1.0) / (k - 1.0))  # k≥2 guaranteed
                excesses.append(chosen_v - statistics.fmean(vals))
                if chosen_v >= max(vals) - 1e-12:
                    n_argmax += 1

    n_choice = len(ranks)
    mean_rank = round(statistics.fmean(ranks), 4) if ranks else None
    med_rank = round(statistics.median(ranks), 4) if ranks else None
    mean_exc = round(statistics.fmean(excesses), 6) if excesses else None
    frac_argmax = round(n_argmax / n_choice, 4) if n_choice else None

    if n_choice < min_choice_turns:
        observation = (
            f"insufficient data ({n_choice} usable choice turns < "
            f"{min_choice_turns}); need real (un-scrubbed) hands and ≥2 "
            "legal colours per reveal"
        )
    elif (
        mean_rank is not None
        and mean_rank >= selfish_rank_floor
        and mean_exc is not None
        and mean_exc > excess_floor
    ):
        observation = (
            f"SELFISH-REVEAL SELECTION (within-state, confound-free): the "
            f"policy picks high-self-favoring reveals among its legal options "
            f"(mean normalised rank {mean_rank} ≥ {selfish_rank_floor} vs the "
            f"0.5 random-choice null [k-invariant]; mean excess {mean_exc} > "
            f"{excess_floor}, null 0.0). Cause NOT attributed (SFT prior "
            f"and/or the λ·shaping nudge). This IS the Caveat-2-addendum "
            f"escalation trigger — consider dropping the reveal sub-delta + "
            f"re-locking λ, or PBRS."
        )
    else:
        observation = (
            f"no selfish-reveal selection (mean normalised rank {mean_rank} "
            f"vs 0.5 random null, mean excess {mean_exc} vs 0.0); within-state "
            "choice does not skew self-favouring — consistent with "
            "concealment/indifference"
        )

    return {
        "analysis_type": ANALYSIS_LABEL,
        "policy_choice_turns": n_choice,
        "reveal_turns_total": n_reveal,
        "reveal_turns_defaulted_excluded": n_default,
        "reveal_turns_forced_no_choice": n_forced,
        "reveal_turns_scrubbed_hand_skipped": n_scrubbed,
        "mean_self_favoring_rank": mean_rank,
        "median_self_favoring_rank": med_rank,
        "random_choice_rank_null": 0.5,  # k-invariant by construction
        "mean_excess_over_options": mean_exc,
        "excess_null": 0.0,  # exact: E[chosen]=mean(options) under random choice
        "frac_chose_argmax_self_favoring": frac_argmax,
        "selfish_rank_floor": selfish_rank_floor,
        "excess_floor": excess_floor,
        "method": (
            "within-state counterfactual over the winner's legal hand "
            "colours — confound-free (player-strength common cause cancels by "
            "within-state differencing). Self-favoring rank is a normalised "
            "mid-rank: 1.0=best legal option, 0.0=worst, 0.5=uniform-random "
            "null for ANY option count (not the inclusive-CDF (k+1)/2k). "
            "Reads private hands; post-hoc, never on the reward/advantage path."
        ),
        "observation": observation,
    }


_CONSENT_FLAG = "--i-understand-this-reads-private-hands"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "PRIVATE-HAND post-hoc counterfactual reveal-choice analysis "
            "(Caveat 2 addendum). Opt-in; reads winners' private hands."
        )
    )
    ap.add_argument("--corpus", required=True, help="dir of schema-v3 transcripts")
    ap.add_argument("--pattern", default="*.json")
    ap.add_argument("--min-choice-turns", type=int, default=30)
    ap.add_argument("--selfish-rank-floor", type=float, default=0.65,
                    help="floor on mean normalised rank (random null = 0.5)")
    ap.add_argument("--excess-floor", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        _CONSENT_FLAG,
        dest="consent",
        action="store_true",
        help="REQUIRED acknowledgement that this reads private hands",
    )
    args = ap.parse_args(argv)

    if not args.consent:
        print(
            "REFUSING TO RUN: this tool reads players' PRIVATE HANDS for "
            "post-hoc analysis.\nIt is firewalled out of the reward/advantage/"
            "export path on purpose.\nRe-run with the explicit consent flag "
            f"if you understand that:\n  {_CONSENT_FLAG}",
            file=sys.stderr,
        )
        return 2

    # Local import keeps even the corpus loader off this module's import graph
    # until consent is given (and avoids a megagem.rl.export edge at import time).
    from ..rl.export import load_corpus

    corpus = load_corpus(args.corpus, args.pattern)
    if not corpus:
        print(f"no schema-v3 transcripts under {args.corpus!r}")
        return 1

    rep = reveal_counterfactual_diagnostic(
        corpus,
        min_choice_turns=args.min_choice_turns,
        selfish_rank_floor=args.selfish_rank_floor,
        excess_floor=args.excess_floor,
    )

    # Safety label → STDERR so stdout stays clean (pure JSON in --json mode,
    # parseable for automated per-iteration reports). The payload itself also
    # carries `analysis_type`, so the private-hand label survives either way.
    banner = "=" * 72
    print(f"{banner}\n  {ANALYSIS_LABEL}\n{banner}", file=sys.stderr)
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0
    print(f"corpus: {len(corpus)} games")
    print(f"  policy choice turns = {rep['policy_choice_turns']}  "
          f"(reveals: {rep['reveal_turns_total']} total; "
          f"{rep['reveal_turns_defaulted_excluded']} defaulted, "
          f"{rep['reveal_turns_forced_no_choice']} forced, "
          f"{rep['reveal_turns_scrubbed_hand_skipped']} scrubbed)")
    print(f"  mean self-favoring rank = {rep['mean_self_favoring_rank']}  "
          f"(0.5 = uniform-random null for ANY k; →1 best legal, →0 worst)")
    print(f"  mean excess over options = {rep['mean_excess_over_options']}  "
          f"(null 0.0)  frac argmax = {rep['frac_chose_argmax_self_favoring']}")
    print(f"  observation: {rep['observation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
