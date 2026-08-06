#!/usr/bin/env python3
"""Paired Gate-B outcome analysis ($0/CPU): compare a piKL treatment arm against
a baseline arm on the SAME seeds (the arms replay paired deals), and report the
first-class signals the λ=0 screen needs:

  * paired Δ-gain (treatment policy_delta − baseline policy_delta) and score gain,
    with a seeded bootstrap 95% CI — the underpowered-evidence guard;
  * the sign table (treatment better / worse / tied per seed);
  * the win-flip table (baseline-loss→treatment-win vs baseline-win→treatment-loss)
    and the two-sided McNemar EXACT p (binomial, no χ² approximation — n is small);
  * the treatment arm's confidence-gate telemetry: passed-gate lift (mean/median)
    vs failed-gate lift, and the tie-adjusted chosen-is-best rate.

Why paired: the arms differ only in the piKL policy, so differencing per seed
removes deal variance — the right unit of evidence for "did piKL help?". Win rate
alone (35% vs 22.5% on 40 games) has a wide CI; the paired Δ and the flip table
are the decisive readouts.

Usage
-----
  python scripts/analysis/pikl_paired_outcomes.py \
      --baseline gate_b_gemini-3-flash_market_off.json \
      --treatment gate_b_gemini-3-flash_market_0.json \
      --out results/pikl_paired/flash_lambda0_vs_off.json

Multiple --treatment arms may be passed; each is compared to the one --baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

_TIE_EPS = 1e-9  # |Δ-gain| <= eps ⇒ a tied seed


# --------------------------------------------------------------------------- #
# loading                                                                     #
# --------------------------------------------------------------------------- #
def load_arm(path) -> dict:
    return json.loads(Path(path).read_text())


def _scored_by_seed(arm: dict) -> dict[int, dict]:
    """seed -> per-game record, for games that scored (error games excluded)."""
    out: dict[int, dict] = {}
    for g in arm.get("per_game") or []:
        if "error" in g or "seed" not in g:
            continue
        out[int(g["seed"])] = g
    return out


def _arm_label(arm: dict) -> str:
    pk = (arm.get("config") or {}).get("pikl")
    if not pk:
        return "off"
    lam = pk.get("lambda")
    mix = pk.get("lambda_mix")
    if mix:
        return f"mix{mix}"
    return f"lambda={lam}"


# --------------------------------------------------------------------------- #
# stats primitives                                                            #
# --------------------------------------------------------------------------- #
def mcnemar_exact(n01: int, n10: int) -> float:
    """Two-sided exact McNemar p over the two discordant counts. Under H0 each
    discordant pair is a fair coin, so #flips-one-way ~ Binom(n, 0.5); the p-value
    is 2·P(X <= min(n01, n10)), capped at 1. Symmetric in (n01, n10)."""
    n = int(n01) + int(n10)
    if n == 0:
        return 1.0
    k = min(int(n01), int(n10))
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def bootstrap_ci(diffs: list[float], *, n_boot: int = 10000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float | None, float | None]:
    """Seeded percentile bootstrap CI for the mean of paired differences."""
    if not diffs:
        return (None, None)
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[max(0, int(math.floor((alpha / 2) * n_boot)))]
    hi = means[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return (round(lo, 4), round(hi, 4))


# --------------------------------------------------------------------------- #
# treatment-arm gate / choice telemetry (works on old AND new arm files)      #
# --------------------------------------------------------------------------- #
def gate_choice_stats(decisions: list[dict]) -> dict:
    """Aggregate confidence-gate lift (split by passed/failed) and the
    tie-adjusted chosen-is-best rate over a list of per-decision payloads.

    Robust to legacy files: ``chosen_is_q_best_tieadj`` is used when present,
    otherwise a passed gate (chosen == gate.best by construction) is credited."""
    gates = [d.get("gate") for d in decisions if d.get("gate")]

    def _lift(gs):
        return [float(g["lift"]) for g in gs if isinstance(g.get("lift"), (int, float))]

    passed = [g for g in gates if g.get("passed")]
    failed = [g for g in gates if not g.get("passed")]

    tieadj_hits = tieadj_total = 0
    for d in decisions:
        m = d.get("metrics") or {}
        if "chosen_is_q_best_tieadj" in m:
            tieadj_total += 1
            tieadj_hits += 1 if m["chosen_is_q_best_tieadj"] else 0
        elif d.get("gate") is not None:  # legacy: passed gate ⇒ chosen is gate-best
            tieadj_total += 1
            tieadj_hits += 1 if d["gate"].get("passed") else 0

    def _m(xs):
        return round(statistics.mean(xs), 5) if xs else None

    def _md(xs):
        return round(statistics.median(xs), 5) if xs else None

    return {
        "n_decisions": len(decisions),
        "gate_n": len(gates),
        "gate_passed_n": len(passed),
        "gate_failed_n": len(failed),
        "gate_pass_rate": round(len(passed) / len(gates), 5) if gates else None,
        "gate_pass_lift_mean": _m(_lift(passed)),
        "gate_pass_lift_median": _md(_lift(passed)),
        "gate_fail_lift_mean": _m(_lift(failed)),
        "gate_fail_lift_median": _md(_lift(failed)),
        "chosen_is_q_best_tieadj_rate": round(tieadj_hits / tieadj_total, 5) if tieadj_total else None,
    }


# --------------------------------------------------------------------------- #
# paired report                                                               #
# --------------------------------------------------------------------------- #
def paired_report(baseline: dict, treatment: dict, *, n_boot: int = 10000,
                  seed: int = 0) -> dict:
    base = _scored_by_seed(baseline)
    treat = _scored_by_seed(treatment)
    common = sorted(set(base) & set(treat))

    delta_diffs, score_diffs = [], []
    treat_better = treat_worse = tied = 0
    n01 = n10 = 0  # n01: base-win→treat-loss ; n10: base-loss→treat-win
    per_seed = []
    for s in common:
        b, t = base[s], treat[s]
        dd = float(t["policy_delta"]) - float(b["policy_delta"])
        sg = float(t["policy_score"]) - float(b["policy_score"])
        delta_diffs.append(dd)
        score_diffs.append(sg)
        if dd > _TIE_EPS:
            treat_better += 1
        elif dd < -_TIE_EPS:
            treat_worse += 1
        else:
            tied += 1
        bw, tw = bool(b["win"]), bool(t["win"])
        if bw and not tw:
            n01 += 1
        elif tw and not bw:
            n10 += 1
        per_seed.append({"seed": s, "base_delta": b["policy_delta"],
                         "treat_delta": t["policy_delta"], "delta_gain": round(dd, 3),
                         "base_win": bw, "treat_win": tw})

    treat_decisions = [d for s in common for d in (treat[s].get("pikl_decisions") or [])]

    def _wr(games):
        gs = [games[s] for s in common]
        return round(sum(1 for g in gs if g["win"]) / len(gs), 4) if gs else None

    def _md(games, key):
        gs = [float(games[s][key]) for s in common]
        return round(statistics.mean(gs), 4) if gs else None

    lo, hi = bootstrap_ci(delta_diffs, n_boot=n_boot, seed=seed)
    return {
        "baseline_label": _arm_label(baseline),
        "treatment_label": _arm_label(treatment),
        "n_paired": len(common),
        "baseline_win_rate": _wr(base),
        "treatment_win_rate": _wr(treat),
        "baseline_mean_delta": _md(base, "policy_delta"),
        "treatment_mean_delta": _md(treat, "policy_delta"),
        "paired_delta_gain_mean": round(statistics.mean(delta_diffs), 4) if delta_diffs else None,
        "paired_delta_gain_95ci": [lo, hi],
        "paired_score_gain_mean": round(statistics.mean(score_diffs), 4) if score_diffs else None,
        "delta_sign": {"treat_better": treat_better, "treat_worse": treat_worse, "tied": tied},
        "flips": {"base_loss_treat_win": n10, "base_win_treat_loss": n01},
        "mcnemar_exact_p": round(mcnemar_exact(n01, n10), 4),
        "treatment_gate": gate_choice_stats(treat_decisions),
        "per_seed": per_seed,
    }


def _fmt(v, nd=3):
    return "  --" if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="Baseline arm JSON (e.g. the 'off' control).")
    ap.add_argument("--treatment", required=True, action="append",
                    help="Treatment arm JSON (repeatable).")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = load_arm(args.baseline)
    reports = []
    for tpath in args.treatment:
        rep = paired_report(base, load_arm(tpath), n_boot=args.n_boot, seed=args.seed)
        reports.append(rep)
        g = rep["treatment_gate"]
        print("=" * 78)
        print(f"{rep['treatment_label']}  vs  {rep['baseline_label']}   "
              f"(n_paired={rep['n_paired']})")
        print("-" * 78)
        print(f"  win rate:   {_fmt(rep['treatment_win_rate'])}  vs  "
              f"{_fmt(rep['baseline_win_rate'])}")
        print(f"  mean delta: {_fmt(rep['treatment_mean_delta'],2)}  vs  "
              f"{_fmt(rep['baseline_mean_delta'],2)}")
        print(f"  paired Δ-gain: {_fmt(rep['paired_delta_gain_mean'],3)}  "
              f"95%CI={rep['paired_delta_gain_95ci']}   "
              f"score-gain: {_fmt(rep['paired_score_gain_mean'],2)}")
        sgn = rep["delta_sign"]
        print(f"  sign: better={sgn['treat_better']} worse={sgn['treat_worse']} "
              f"tied={sgn['tied']}")
        fl = rep["flips"]
        print(f"  win-flips: loss→win={fl['base_loss_treat_win']}  "
              f"win→loss={fl['base_win_treat_loss']}  "
              f"McNemar exact p={rep['mcnemar_exact_p']}")
        print(f"  gate: pass={g['gate_passed_n']}/{g['gate_n']} "
              f"({_fmt(g['gate_pass_rate'])})  "
              f"pass-lift(mean/med)={g['gate_pass_lift_mean']}/{g['gate_pass_lift_median']}  "
              f"fail-lift(mean/med)={g['gate_fail_lift_mean']}/{g['gate_fail_lift_median']}")
        print(f"  tie-adjusted chosen-is-best: {g['chosen_is_q_best_tieadj_rate']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"baseline": args.baseline, "reports": reports}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
