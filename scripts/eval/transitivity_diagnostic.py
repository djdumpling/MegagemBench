#!/usr/bin/env python3
"""TRANSITIVITY DIAGNOSTIC — is a scalar rating (TrueSkill/BT/PL) the right ESTIMAND
for this model pool, or is the game intransitive (rock-paper-scissors)?

If the pool is intransitive, no scalar rating's CI is meaningful: more games just
shrink the CI on a quantity that doesn't exist. This is a $0 pre-check that should
gate any leaderboard claim. (Lit: Balduzzi et al. NeurIPS'18 "Re-evaluating
Evaluation"; Jiang-Lim-Yao-Ye 2011 HodgeRank; Hamilton et al. 2024 I(A) ratio;
Czarnecki et al. 2020 "spinning tops" — the TOP of a pool is usually transitive
even when the middle cycles.)

Reads existing game JSONs. Reports:
  - HodgeRank cyclic fraction  (share of pairwise-comparison energy a scalar
    rating CANNOT explain), weighted by #co-occurrences.
  - I(A) intransitivity ratio  (Hamilton: <1 transitive-dominant, >1 intransitive).
  - directed 3-cycle count, raw AND among statistically-RESOLVED pairs (a cycle
    among coin-flip pairs is noise; among resolved pairs it is real intransitivity).
  - schedule-sensitivity: does the scalar ranking churn under cluster resampling?
  - top-tier transitivity (spinning-top): is the leading group cleanly orderable?
  - a reasoned VERDICT.

Usage:
  python3 scripts/eval/transitivity_diagnostic.py            # dry-run on former/
  python3 scripts/eval/transitivity_diagnostic.py --sources <bibd_dir> \
        --restrict 1 2 3 4 5 6 7 8 9 10 11 12 13 --min-co 8
System python3 (numpy + scipy).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from rating_common import (DEFAULT_SOURCES, cluster_key, load_games,
                           model_labels, resolve_restrict, short)

try:
    from scipy.stats import binomtest
    def _binom_p(k, n):
        return binomtest(k, n, 0.5, alternative="two-sided").pvalue if n else 1.0
except Exception:  # noqa: BLE001
    def _binom_p(k, n):  # normal approx fallback
        if not n:
            return 1.0
        z = abs(k - n / 2) / (0.5 * math.sqrt(n))
        return math.erfc(z / math.sqrt(2))


def pairwise(games, restrict=None):
    """Return models list + dicts: above[(i,j)] = #games i scored strictly above j
    (ties split 0.5), n[(i,j)] = #co-occurrences, and per-game within-pair records
    keyed by cluster for the bootstrap."""
    above = defaultdict(float)
    n = defaultdict(float)
    seen = set()
    recs = []  # (cluster, i, j, outcome in {1, .5, 0} for i-vs-j)
    for g in games:
        for a, b in combinations(range(len(g.models)), 2):
            mi, mj = g.models[a], g.models[b]
            if restrict and (mi not in restrict or mj not in restrict):
                continue
            if mi == mj:
                continue
            si, sj = g.scores[a], g.scores[b]
            out = 1.0 if si > sj else (0.0 if si < sj else 0.5)
            above[(mi, mj)] += out
            above[(mj, mi)] += (1.0 - out)
            n[(mi, mj)] += 1
            n[(mj, mi)] += 1
            seen.add(mi)
            seen.add(mj)
            recs.append((cluster_key(g), mi, mj, out))
    return sorted(seen), above, n, recs


def _logit(p):
    p = min(1 - 1e-3, max(1e-3, p))
    return math.log(p / (1 - p))


def solve_potential(edges, M):
    """Weighted graph-Laplacian solve for the transitive potential s minimizing
    Σ w_ij (A_ij-(s_i-s_j))^2. edges = [(i,j,A_ij,w_ij)]. Gauge sum s = 0."""
    L = np.zeros((M, M))
    b = np.zeros(M)
    for i, j, A, w in edges:
        L[i, i] += w
        L[j, j] += w
        L[i, j] -= w
        L[j, i] -= w
        b[i] += w * A
        b[j] -= w * A
    s = np.linalg.pinv(L) @ b
    return s - s.mean()


def hodge_potential(models, above, n, min_co):
    """Returns (s dict, observed edges [(i,j,A,w)], s vector)."""
    idx = {m: k for k, m in enumerate(models)}
    edges = []
    for mi, mj in combinations(models, 2):
        nij = n.get((mi, mj), 0)
        if nij < min_co:
            continue
        edges.append((idx[mi], idx[mj], _logit(above[(mi, mj)] / nij), nij))
    s = solve_potential(edges, len(models))
    return {models[i]: float(s[i]) for i in range(len(models))}, edges, s


def noise_floor(edges, M, s, B, rng):
    """Parametric bootstrap under the TRANSITIVE null: regenerate each edge's wins
    from Binomial(n_ij, sigmoid(s_i-s_j)) — the fitted transitive prediction —
    refit the potential, recompute cyclic fraction & I(A). The distribution of
    cyclic fractions this produces is the noise floor a perfectly transitive game
    yields at THIS sample size, so the observed value is only evidence of real
    intransitivity if it exceeds this floor."""
    cyc_null, I_null = [], []
    for _ in range(B):
        syn = []
        for i, j, _A, w in edges:
            phat = 1.0 / (1.0 + math.exp(-(s[i] - s[j])))
            k = rng.binomial(int(round(w)), phat)
            syn.append((i, j, _logit(k / w), w))
        ss = solve_potential(syn, M)
        c, I = energy_and_I(syn, ss)
        cyc_null.append(c)
        I_null.append(I)
    return np.array(cyc_null), np.array(I_null)


def energy_and_I(edges, s):
    """cyclic fraction (weighted, residual energy / total energy) + Hamilton I(A)."""
    num_w = den_w = 0.0
    res_sq = grad_sq = 0.0  # unweighted Frobenius (each unordered edge once)
    for i, j, A, w in edges:
        grad = s[i] - s[j]
        r = A - grad
        num_w += w * r * r
        den_w += w * A * A
        res_sq += r * r
        grad_sq += grad * grad
    cyc_frac = math.sqrt(num_w / den_w) if den_w > 0 else 0.0
    # I(A) = (1+||A-grad||_F)/(1+||grad||_F); Frobenius over the antisymmetric
    # matrix = sqrt(2 * Σ_{i<j} x_ij^2). The factor 2 cancels in the ratio's spirit
    # but we keep it for fidelity to the definition.
    I = (1 + math.sqrt(2 * res_sq)) / (1 + math.sqrt(2 * grad_sq)) if grad_sq > 0 else float("inf")
    return cyc_frac, I


def three_cycles(models, above, n, min_co, alpha=0.05):
    """Count directed 3-cycles. 'resolved' = only pairs whose direction is
    significant at alpha (binomial vs 0.5)."""
    def direction(mi, mj):
        nij = n.get((mi, mj), 0)
        if nij < min_co:
            return None, None
        p = above[(mi, mj)] / nij
        sig = _binom_p(int(round(above[(mi, mj)])), int(nij)) < alpha
        d = 0 if abs(p - 0.5) < 1e-9 else (1 if p > 0.5 else -1)
        return d, sig

    raw = res = tot = tot_res = 0
    for a, b, c in combinations(models, 3):
        ds = {}
        sigs = {}
        ok = True
        for x, y in ((a, b), (b, c), (a, c)):
            d, sg = direction(x, y)
            if d is None:
                ok = False
                break
            ds[(x, y)] = d
            sigs[(x, y)] = sg
        if not ok:
            continue
        tot += 1
        # cycle iff a>b>c>a or a<b<c<a  (i.e. not a consistent linear order)
        dab, dbc, dac = ds[(a, b)], ds[(b, c)], ds[(a, c)]
        is_cycle = (dab == dbc == -dac) and dab != 0
        if is_cycle:
            raw += 1
        if all(sigs.values()):
            tot_res += 1
            if is_cycle:
                res += 1
    return raw, tot, res, tot_res


def schedule_sensitivity(models, recs, min_co, B, rng):
    """Resample clusters; refit Hodge potential; report Spearman of the rank order
    vs the full-data order, and per-model rank distribution."""
    clusters = defaultdict(list)
    for ck, mi, mj, out in recs:
        clusters[ck].append((mi, mj, out))
    keys = list(clusters)

    def fit(reclist):
        above = defaultdict(float)
        n = defaultdict(float)
        for mi, mj, out in reclist:
            above[(mi, mj)] += out
            above[(mj, mi)] += 1 - out
            n[(mi, mj)] += 1
            n[(mj, mi)] += 1
        s, _, _ = hodge_potential(models, above, n, min_co)
        return s

    full = fit([r for ck in keys for r in clusters[ck]])
    base_order = sorted(models, key=lambda m: -full[m])
    base_rank = {m: i for i, m in enumerate(base_order)}
    spear = []
    rank_samples = defaultdict(list)
    for _ in range(B):
        draw = rng.choice(len(keys), size=len(keys), replace=True)
        rl = [r for di in draw for r in clusters[keys[di]]]
        s = fit(rl)
        order = sorted(models, key=lambda m: -s[m])
        rk = {m: i for i, m in enumerate(order)}
        for m in models:
            rank_samples[m].append(rk[m])
        # Spearman vs base
        d2 = sum((rk[m] - base_rank[m]) ** 2 for m in models)
        M = len(models)
        spear.append(1 - 6 * d2 / (M * (M * M - 1)) if M > 1 else 1.0)
    spear = np.array(spear)
    rank_ci = {m: (int(np.percentile(rank_samples[m], 2.5)) + 1,
                   base_rank[m] + 1,
                   int(np.percentile(rank_samples[m], 97.5)) + 1) for m in models}
    return float(spear.mean()), float(np.percentile(spear, 2.5)), base_order, rank_ci


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    ap.add_argument("--restrict", nargs="+", default=None,
                    help="model_ids / registry numbers / substrings to keep")
    ap.add_argument("--min-co", type=int, default=10, help="min co-occurrences per pair to use it")
    ap.add_argument("--cluster", default="triplet_seed")
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--top", type=int, default=5, help="top-tier size for spinning-top check")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] /
                    "results" / "analysis" / "transitivity_diagnostic.json"))
    a = ap.parse_args()

    restrict = resolve_restrict(a.restrict)
    games, bad = load_games(a.sources, restrict=restrict)
    _, id2disp = model_labels()
    print(f"loaded {len(games)} games ({bad} skipped) from {a.sources}")
    if restrict:
        print(f"restricted to {len(restrict)} models")

    models, above, n, recs = pairwise(games, restrict=restrict)
    if len(models) < 3:
        print("too few models with data"); return
    npairs = sum(1 for mi, mj in combinations(models, 2) if n.get((mi, mj), 0) >= a.min_co)
    tot_pairs = len(models) * (len(models) - 1) // 2
    print(f"{len(models)} models, {npairs}/{tot_pairs} pairs with >= {a.min_co} co-occurrences\n")

    s, edges, svec = hodge_potential(models, above, n, a.min_co)
    cyc, I = energy_and_I(edges, svec)
    raw, tot, res, tot_res = three_cycles(models, above, n, a.min_co)

    # noise floor: what cyclic fraction does a TRANSITIVE game produce at this n?
    rng0 = np.random.default_rng(1)
    cyc_null, I_null = noise_floor(edges, len(models), svec, max(200, a.bootstrap), rng0)
    null_mean, null_p95 = float(cyc_null.mean()), float(np.percentile(cyc_null, 95))
    null_p = float((cyc_null >= cyc).mean())  # P(transitive-null >= observed)

    # top-tier (spinning top): cyclic fraction among the top-T by potential
    order = sorted(models, key=lambda m: -s[m])
    top = set(order[:a.top])
    top_edges = [(i, j, A, w) for (i, j, A, w) in edges
                 if models[i] in top and models[j] in top]
    cyc_top, I_top = energy_and_I(top_edges, svec) if top_edges else (0.0, 1.0)

    rng = np.random.default_rng(0)
    mean_sp, lo_sp, base_order, rank_ci = schedule_sensitivity(models, recs, a.min_co, a.bootstrap, rng)

    print("=" * 78)
    print("HODGE / INTRANSITIVITY")
    print("=" * 78)
    print(f"  cyclic fraction (weighted energy a scalar CANNOT explain) : {cyc:.3f}")
    print(f"    transitive-null floor at this n (mean, 95th pct)        : {null_mean:.3f}, {null_p95:.3f}")
    print(f"    -> P(noise alone >= observed) = {null_p:.3f}   "
          f"({'within noise' if null_p >= 0.05 else 'EXCEEDS noise floor'})")
    print(f"  I(A) intransitivity ratio (Hamilton; <1 transitive)       : {I:.3f}")
    print(f"  directed 3-cycles (raw)                : {raw}/{tot}  ({raw/tot:.1%})" if tot else "  no fully-observed triples")
    print(f"  directed 3-cycles among RESOLVED pairs : {res}/{tot_res}" + (f"  ({res/tot_res:.1%})" if tot_res else "  (none resolved)"))
    print(f"  top-{a.top} cyclic fraction (spinning-top)  : {cyc_top:.3f}   (is the leading group orderable?)")
    print(f"  schedule-sensitivity Spearman (mean, 2.5%): {mean_sp:.3f}, {lo_sp:.3f}   (1.0 = stable order)")
    print()
    print("scalar ranking (Hodge potential) + bootstrap rank CI:")
    print(f"  {'rank':>4}  {'model':28} {'potential':>9}  {'rankCI':>12}")
    for m in base_order:
        lo, mid, hi = rank_ci[m]
        print(f"  {mid:>4}  {short(m, id2disp)[:28]:28} {s[m]:>9.3f}  [{lo:>2},{hi:>2}]")

    # ---- verdict ----
    # Genuine intransitivity requires the cyclic energy to EXCEED the finite-sample
    # noise floor, or a resolved 3-cycle. A high raw cyclic fraction that is within
    # the noise floor = many near-tied pairs at low n = a CI-WIDTH problem, not an
    # estimand problem.
    intransitive = (res > 0) or (null_p < 0.05 and I > 1.0)
    print("\n" + "=" * 78)
    if intransitive:
        verdict = ("INTRANSITIVE / scalar misspecified — cyclic energy exceeds the "
                   "noise floor and/or resolved pairs form cycles. A single "
                   "TrueSkill/BT/PL number is the WRONG estimand. Use alpha-Rank "
                   "(OpenSpiel) or VasE/Iterated-Maximal-Lotteries; report the cyclic "
                   "residual. CI width is moot until the estimand is fixed.")
    else:
        verdict = ("TRANSITIVE (at the resolved level) — the cyclic fraction is within "
                   "the finite-sample noise floor and no resolved pair forms a cycle, "
                   "so a scalar rating IS the right estimand. The cyclic energy you see "
                   "is sampling noise from near-tied pairs at low n: the operative "
                   "concern is CI WIDTH, not intransitivity. Use the deck-aware "
                   "Plackett-Luce + paired same-deck contrasts, and more SEEDS/data to "
                   "resolve the mid-pack.")
    print("VERDICT:", verdict)
    print(f"  (resolved pairs: only {tot_res} of {tot} triples had all 3 directions "
          f"statistically resolved — most mid-pack pairs are coin-flips at this n, "
          f"the real bottleneck.)")
    if cyc_top < cyc:
        print(f"  (Spinning-top: top-{a.top} cyclic fraction {cyc_top:.3f} < pool {cyc:.3f} — "
              f"'who is #1/top-{a.top}' is more defensible than mid-pack order.)")
    print("=" * 78)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "sources": a.sources, "n_games": len(games), "n_models": len(models),
        "min_co": a.min_co, "pairs_used": npairs, "pairs_total": tot_pairs,
        "cyclic_fraction": cyc, "I_A": I,
        "noise_floor": {"mean": null_mean, "p95": null_p95, "null_p_ge_observed": null_p},
        "three_cycles_raw": [raw, tot], "three_cycles_resolved": [res, tot_res],
        "top_tier": {"size": a.top, "cyclic_fraction": cyc_top},
        "schedule_spearman_mean": mean_sp, "schedule_spearman_lo": lo_sp,
        "potential": {short(m, id2disp): s[m] for m in base_order},
        "rank_ci": {short(m, id2disp): rank_ci[m] for m in base_order},
        "verdict": verdict,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
