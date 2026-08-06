#!/usr/bin/env python3
"""Phase-3 reward-signal CPU diagnostic — "is the reward design the bug?"

Answers, on a real schema-v3 game corpus (no GPU / torch / vllm):

  Q1  Does the per-row `precomputed_reward` correlate with the engine-true
      `final_score`? (Sanity: the reward function is wired correctly.)
  Q2  Does the `clip()` squash saturate — flattening big wins and small wins
      into the same +-1 reward, destroying within-outcome ranking?
  Q3  Within each K-group (the GRPO comparison set: K games at one seed/seat
      differing only by sampling), is there enough *outcome spread* and
      *reward spread* for the standardized advantage to carry a real signal?

Q1 is expected to PASS by construction (r_terminal = clip(margin/scale) is
monotone in final_score) — a passing Q1 does NOT exonerate the reward; it only
rules out a miswire. Q2 + Q3 are the ones that can actually indict the design:

  * Q2 fail  -> reward design bug: clip is the culprit; tanh (rec 2) recovers it.
  * Q3 fail on *outcome* spread -> NOT a reward bug; the rollout/opponent
    produces near-identical games, leaving nothing for any reward to grade.
  * Q3 fail on *reward* spread while outcome spread is healthy -> reward design
    bug: the squash is collapsing real outcome differences.

Caveat: real Phase-3 rollout games (SFT-adapter seat 0 vs stationary
heuristics) are never persisted — phase3_grpo.py rolls them into a deleted
TemporaryDirectory and the eval JSONs are summary-only. This script therefore
runs on `results/phase1_corpus/chart_A` (qwen self-play, K=8) as the closest
available proxy. Phase-1 self-play has *more* stochastic seats (3) than a
Phase-3 rollout (1 trainable seat) so its within-group spread is an UPPER
bound on what Phase-3 sees — a Q3 fail here is therefore conservative.

CPU-only: imports just `megagem.rl.{reward,advantage,scorer,diagnostics}` (all
torch-free). Report-only; mutates nothing.

  python3 scripts/training/reward_score_correlation.py
  python3 scripts/training/reward_score_correlation.py --corpus <dir> --out <f>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

from megagem.rl.advantage import AdvantageConfig, compute_advantages
from megagem.rl.diagnostics import _spearman
from megagem.rl.reward import RewardConfig, terminal_breakdowns
from megagem.rl.scorer import parse_transcript


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def _load(corpus_dir, pattern="*.json"):
    out = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "**", pattern),
                                 recursive=True)):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if (isinstance(data, dict)
                and data.get("metadata", {}).get("schema_version") == 3):
            out.append(data)
    return out


def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def run(corpus, cfg, adv_cfg):
    parsed = [parse_transcript(d) for d in corpus]
    rows = compute_advantages(corpus, cfg, adv_cfg)

    # ---- per-(game, player) join: training rows  x  engine terminal facts ----
    # reward sum over a player's turns == that player's total return (rtg @ t0).
    ret_sum = defaultdict(float)        # (gid, pid) -> Σ precomputed_reward
    term_reward = {}                    # (gid, pid) -> terminal-turn reward
    term_adv = {}                       # (gid, pid) -> terminal-turn advantage
    gkey = {}                           # (gid, pid) -> group_key
    n_rows = 0
    n_dead_nonterminal = 0              # non-terminal rows with |reward| < 1e-6
    all_row_rewards = []
    for r in rows:
        if not r.trainable:
            continue
        n_rows += 1
        all_row_rewards.append(r.reward)
        key = (r.game_id, r.player_id)
        ret_sum[key] += r.reward
        gkey[key] = r.group_key
        if r.is_terminal_turn:
            term_reward[key] = r.reward
            term_adv[key] = r.advantage
        elif abs(r.reward) < 1e-6:
            n_dead_nonterminal += 1

    recs = []  # one per (game, player)
    for gid, p in enumerate(parsed):
        for pid, bd in terminal_breakdowns(p, cfg).items():
            key = (gid, pid)
            if key not in term_reward:
                continue
            recs.append({
                "gid": gid, "pid": pid, "group_key": gkey[key],
                "final_score": float(bd.final_score),
                "margin": float(bd.margin),
                "r_terminal_clean": float(bd.r_terminal),  # clip(margin/S)+bonus
                "raw_terminal": float(bd.margin) / cfg.scale,  # un-clipped/un-bonus
                "is_winner": bool(bd.is_winner),
                "term_reward": float(term_reward[key]),
                "total_return": float(ret_sum[key]),
                "term_adv": float(term_adv.get(key, 0.0)),
            })

    # ---- Q1: correlation ---------------------------------------------------
    def corr(xkey, ykey):
        xs = [r[xkey] for r in recs]
        ys = [r[ykey] for r in recs]
        return {"pearson": _pearson(xs, ys), "spearman": _spearman(xs, ys),
                "n": len(xs)}

    q1 = {
        "term_reward_vs_final_score": corr("term_reward", "final_score"),
        "term_reward_vs_margin": corr("term_reward", "margin"),
        "total_return_vs_final_score": corr("total_return", "final_score"),
        "total_return_vs_margin": corr("total_return", "margin"),
    }

    # ---- Q2: clip saturation ----------------------------------------------
    sat = [r for r in recs if abs(r["raw_terminal"]) >= 1.0]
    margins_abs = sorted(abs(r["margin"]) for r in recs)
    q2 = {
        "scale": cfg.scale,
        "n_player_games": len(recs),
        "n_saturated": len(sat),
        "saturation_rate": round(len(sat) / len(recs), 4) if recs else None,
        "abs_margin_median": round(statistics.median(margins_abs), 2),
        "abs_margin_p90": round(margins_abs[int(0.9 * (len(margins_abs) - 1))],
                                2) if margins_abs else None,
        "abs_margin_max": round(margins_abs[-1], 2) if margins_abs else None,
        # how much within-spread the clip removes vs an un-clipped/tanh map:
        # global pstdev of clipped terminal vs raw margin/scale.
        "pstdev_clipped_terminal": round(statistics.pstdev(
            [_clip(r["raw_terminal"]) for r in recs]), 4) if len(recs) > 1
            else None,
        "pstdev_raw_terminal": round(statistics.pstdev(
            [r["raw_terminal"] for r in recs]), 4) if len(recs) > 1 else None,
    }

    # ---- Q3: within-K-group spread ----------------------------------------
    groups = defaultdict(list)
    for r in recs:
        groups[r["group_key"]].append(r)
    grp_stats = []
    for k, g in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(g) < 2:
            continue
        fs = [r["final_score"] for r in g]
        mg = [r["margin"] for r in g]
        rt = [r["r_terminal_clean"] for r in g]
        rawt = [r["raw_terminal"] for r in g]
        ad = [r["term_adv"] for r in g]
        sd_clip = statistics.pstdev(rt)
        sd_raw = statistics.pstdev(rawt)
        grp_stats.append({
            "group_key": str(k), "k": len(g),
            "sd_final_score": round(statistics.pstdev(fs), 3),
            "sd_margin": round(statistics.pstdev(mg), 3),
            "sd_r_terminal_clipped": round(sd_clip, 5),
            "sd_raw_terminal": round(sd_raw, 5),
            # fraction of within-group reward spread the clip removes:
            "clip_spread_retention": round(sd_clip / sd_raw, 4)
                if sd_raw > 1e-12 else None,
            "sd_advantage": round(statistics.pstdev(ad), 4),
            # does the reward rank games the same way the engine margin does?
            "rank_corr_reward_vs_margin": _spearman(rt, mg),
        })

    def _med(key):
        vals = [s[key] for s in grp_stats if s[key] is not None]
        return round(statistics.median(vals), 4) if vals else None

    q3 = {
        "n_groups": len(grp_stats),
        "median_sd_final_score_within_group": _med("sd_final_score"),
        "median_sd_margin_within_group": _med("sd_margin"),
        "median_sd_r_terminal_within_group": _med("sd_r_terminal_clipped"),
        "median_sd_advantage_within_group": _med("sd_advantage"),
        "median_clip_spread_retention": _med("clip_spread_retention"),
        "median_rank_corr_reward_vs_margin": _med("rank_corr_reward_vs_margin"),
        "n_groups_outcome_degenerate_sd_fs_lt_3": sum(
            1 for s in grp_stats if s["sd_final_score"] < 3.0),
        "per_group": grp_stats,
    }

    # ---- per-row reward magnitude -----------------------------------------
    mag = {
        "n_trainable_rows": n_rows,
        "n_dead_nonterminal_rows": n_dead_nonterminal,
        "dead_nonterminal_frac": round(n_dead_nonterminal / n_rows, 4)
            if n_rows else None,
        "row_reward_mean": round(statistics.fmean(all_row_rewards), 5)
            if all_row_rewards else None,
        "row_reward_pstdev": round(statistics.pstdev(all_row_rewards), 5)
            if len(all_row_rewards) > 1 else None,
    }

    return {"q1_correlation": q1, "q2_clip_saturation": q2,
            "q3_within_group_spread": q3, "row_reward_magnitude": mag}


def _verdict(rep):
    out = []
    q1 = rep["q1_correlation"]["term_reward_vs_final_score"]
    p = q1["pearson"]
    out.append(
        f"Q1  reward<->final_score: pearson={p:.3f} "
        f"spearman={rep['q1_correlation']['term_reward_vs_final_score']['spearman']:.3f}"
        f"  -> {'CORRELATED (function wired correctly; not a miswire)' if (p or 0) > 0.7 else 'WEAK/BROKEN — investigate the reward wiring'}")
    q2 = rep["q2_clip_saturation"]
    sr = q2["saturation_rate"]
    out.append(
        f"Q2  clip saturation: {q2['n_saturated']}/{q2['n_player_games']} "
        f"player-games pinned at +-1 ({sr:.1%}); |margin| median="
        f"{q2['abs_margin_median']} p90={q2['abs_margin_p90']} max={q2['abs_margin_max']} "
        f"vs scale={q2['scale']}  -> "
        f"{'clip saturates globally — big/small wins flattened; see Q3 for within-group impact' if (sr or 0) > 0.15 else 'clip rarely binds — not the bug here'}")
    q3 = rep["q3_within_group_spread"]
    deg = q3["n_groups_outcome_degenerate_sd_fs_lt_3"]
    ret = q3["median_clip_spread_retention"]
    out.append(
        f"Q3  within-K-group: median sd(final_score)="
        f"{q3['median_sd_final_score_within_group']}, "
        f"sd(advantage)={q3['median_sd_advantage_within_group']}, "
        f"clip_spread_retention={ret}, "
        f"rank_corr(reward,margin)={q3['median_rank_corr_reward_vs_margin']}; "
        f"outcome-degenerate groups (sd<3)={deg}/{q3['n_groups']}")
    if q3["median_sd_final_score_within_group"] is not None:
        if q3["median_sd_final_score_within_group"] < 3.0:
            out.append("    -> OUTCOME spread is tiny: NOT a reward bug — the "
                       "rollout/opponent produces near-identical games.")
        elif ret is not None and ret < 0.7:
            out.append("    -> outcome spread healthy but clip RETAINS only "
                       f"{ret:.0%} of it: reward design bug (clip squash).")
        else:
            out.append("    -> outcome spread healthy AND reward preserves it: "
                       "reward design is sound; look upstream (opponent / "
                       "eval generalization / steps).")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="results/phase1_corpus/chart_A")
    ap.add_argument("--pattern", default="*.json")
    ap.add_argument("--terminal-correction", default="1",
                    help="1 = locked Phase-3 cut (rec 1a ON); 0 = plain.")
    ap.add_argument("--out", default=None, help="optional report.json path")
    args = ap.parse_args(argv)

    corpus = _load(args.corpus, args.pattern)
    if not corpus:
        print(f"no schema-v3 transcripts under {args.corpus!r}")
        return 1
    tc = args.terminal_correction in ("1", "true", "True", "yes", "on")
    cfg = RewardConfig(terminal_correction=tc)
    seeds = sorted({parse_transcript(d).seed for d in corpus})

    rep = run(corpus, cfg, AdvantageConfig())
    rep["meta"] = {
        "corpus": args.corpus, "n_games": len(corpus), "n_seeds": len(seeds),
        "reward_config": {"scale": cfg.scale, "shaping_lambda": cfg.shaping_lambda,
                          "terminal_shape": cfg.terminal_shape,
                          "terminal_correction": cfg.terminal_correction},
        "proxy_caveat": ("phase-1 self-play corpus used as proxy; real "
                         "Phase-3 rollouts are not persisted. Within-group "
                         "spread here is an UPPER bound on Phase-3."),
    }

    print(f"[reward-corr] corpus={args.corpus} games={len(corpus)} "
          f"seeds={len(seeds)} terminal_correction={cfg.terminal_correction}\n")
    print(json.dumps(rep, indent=2, default=str))
    print("\n=== VERDICT ===")
    print(_verdict(rep))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2, default=str))
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
