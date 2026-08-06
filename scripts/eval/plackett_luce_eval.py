#!/usr/bin/env python3
"""DECK-CLUSTERED PLACKETT-LUCE — the honest alternative to TrueSkill for this
setup. Plackett-Luce is the statistically correct model for "3 players ranked
1/2/3" (Bradley-Terry is just PL for k=2; using the full ranking once is more
efficient than breaking each game into 3 dependent pairwise comparisons). CIs come
from a CLUSTER bootstrap that respects the CRN/deck structure — the lever that
actually matters (the rating algorithm is second-order).

Why this over TrueSkill's sigma:
  * TrueSkill's sigma is a mean-field posterior belief width, not a sampling CI,
    and it assumes games are independent (ignores the shared-deck/seat-perm CRN),
    so it understates the true spread (~2x in our sim).
  * PL-MLE + cluster bootstrap reports an actual sampling interval, and the PAIRED
    same-deck contrast (A vs B in the same game) deletes the deck variance — this
    is what made the trio land t=6.5 where marginal CIs overlapped.

Lit: Plackett 1975; Luce 1959; Hunter 2004 (MM for generalized BT / PL);
Chatbot Arena / LMSYS (static BT-MLE + bootstrap, arXiv 2403.04132); GameBench
(arXiv 2406.06613); cluster/block bootstrap for clustered data.

NOTE on decks: the seed fully determines the deck and the BIBD uses only 3 seeds,
so the deck is a controlled fixed factor with few levels. The default cluster
(triplet_seed = matchup-on-deck) gives ~78 resampling units in the 468 BIBD; for
the global table the deck is effectively conditioned-on. Generalization to NEW
decks needs more SEEDS — that is the real argument for more seeds (external
validity), distinct from marginal-CI narrowing.

Usage:
  python3 scripts/eval/plackett_luce_eval.py                 # dry-run on former/
  python3 scripts/eval/plackett_luce_eval.py --sources <bibd_dir> \
        --restrict 1 2 3 4 5 6 7 8 9 10 11 12 13 --bootstrap 500
System python3 (numpy).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

from rating_common import (DEFAULT_SOURCES, cluster_key, load_games,
                           model_labels, ordering, resolve_restrict, short)


def game_to_weighted_orderings(g, restrict=None):
    """Return list of (ordering_of_model_ids_best_to_worst, weight). Ties are
    expanded into the equally-likely strict orderings (rare; keeps PL exact)."""
    mids = [m for m in g.models]
    if restrict and not all(m in restrict for m in mids):
        return []
    order_ids, order_sc = ordering(g)
    # group tied positions
    groups = []
    i = 0
    while i < len(order_ids):
        j = i
        while j + 1 < len(order_sc) and order_sc[j + 1] == order_sc[i]:
            j += 1
        groups.append(order_ids[i:j + 1])
        i = j + 1
    if all(len(gr) == 1 for gr in groups):
        return [(tuple(order_ids), 1.0)]
    # expand the cartesian product of within-group permutations
    perm_lists = [list(permutations(gr)) for gr in groups]
    out = []
    from itertools import product
    combos = list(product(*perm_lists))
    w = 1.0 / len(combos)
    for combo in combos:
        flat = tuple(x for grp in combo for x in grp)
        out.append((flat, w))
    return out


def fit_pl(orderings, models, reg=0.1, max_iter=500, tol=1e-9):
    """Plackett-Luce MLE via Hunter(2004) MM, with symmetric anchor regularization
    (each model gets `reg` virtual wins AND `reg` losses vs a fixed strength-1
    anchor) to guarantee finite, connected estimates on sparse/bootstrapped graphs.
    Returns theta = log-strength (centered to mean 0)."""
    idx = {m: k for k, m in enumerate(models)}
    M = len(models)
    # precompute per-ordering the index sequence
    seqs = [( [idx[m] for m in o], w) for o, w in orderings if all(m in idx for m in o)]
    gamma = np.ones(M)
    # numerator w_i: wins (= appears not-last in an ordering) + reg anchor wins
    w_num = np.full(M, reg)  # reg virtual wins vs anchor
    for seq, w in seqs:
        for r in range(len(seq) - 1):  # last item never "wins" a stage
            w_num[seq[r]] += w
    for _ in range(max_iter):
        denom = np.zeros(M)
        for seq, w in seqs:
            # suffix sums of gamma along the ordering
            k = len(seq)
            suf = 0.0
            Z = [0.0] * k
            for r in range(k - 1, -1, -1):
                suf += gamma[seq[r]]
                Z[r] = suf
            # model at position pos is in stages 0..min(pos, k-2)
            for pos in range(k):
                last_stage = min(pos, k - 2)
                contrib = 0.0
                for r in range(last_stage + 1):
                    contrib += w / Z[r]
                denom[seq[pos]] += contrib
        # anchor losses: each model i compared to anchor (strength 1), reg times,
        # i in the set -> reg / (gamma_i + 1); plus reg anchor-win sets likewise.
        denom += 2.0 * reg / (gamma + 1.0)
        new = w_num / np.maximum(denom, 1e-300)
        new *= math.exp(-np.mean(np.log(new)))  # normalize geomean = 1
        if np.max(np.abs(np.log(new) - np.log(gamma))) < tol:
            gamma = new
            break
        gamma = new
    theta = np.log(gamma)
    theta -= theta.mean()
    return {models[i]: float(theta[i]) for i in range(M)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    ap.add_argument("--restrict", nargs="+", default=None)
    ap.add_argument("--cluster", default="triplet_seed",
                    help="bootstrap unit: triplet_seed|seed|triplet|game")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--reg", type=float, default=0.1)
    ap.add_argument("--elo-scale", type=float, default=400.0 / math.log(10),
                    help="multiply log-strength to get classic-Elo-like points (~173.7)")
    ap.add_argument("--init-rating", type=float, default=1000.0,
                    help="Elo offset (Chatbot Arena uses 1000)")
    ap.add_argument("--anchor", default=None,
                    help="model (registry number/id) pinned to --init-rating to fix the "
                         "additive gauge (Arena anchors one model); else the field is centered")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] /
                    "results" / "analysis" / "plackett_luce_eval.json"))
    a = ap.parse_args()

    restrict = resolve_restrict(a.restrict)
    games, bad = load_games(a.sources, restrict=restrict)
    _, id2disp = model_labels()
    print(f"loaded {len(games)} games ({bad} skipped) from {a.sources}")

    # rebind cluster_key to the chosen mode
    def ck(g):
        return cluster_key(g, a.cluster)
    orderings_all = []
    for g in games:
        for o, w in game_to_weighted_orderings(g, restrict):
            orderings_all.append((ck(g), o, w))
    models = sorted({m for _, o, _ in orderings_all for m in o})
    if len(models) < 2:
        print("too few models"); return
    nclusters = len({c for c, _, _ in orderings_all})
    print(f"{len(models)} models, {len(orderings_all)} ranking-observations, "
          f"{nclusters} bootstrap clusters (cluster={a.cluster})")
    if a.cluster == "seed":
        print("  WARNING: clustering on seed -> very few clusters (deck=seed); "
              "bootstrap will be unstable. Prefer triplet_seed.")

    # anchor (gauge): pin one model to init-rating, else center the field
    anchor_id = None
    if a.anchor:
        cand = resolve_restrict([a.anchor]) or set()
        anchor_id = next((m for m in models if m in cand), None)
        if anchor_id is None:
            print(f"  WARNING: anchor {a.anchor!r} not in pool; centering instead")
        else:
            print(f"  anchor: {short(anchor_id, id2disp)} pinned to {a.init_rating:.0f}")

    def _anchor(thd):
        sh = thd[anchor_id] if anchor_id else 0.0
        return {m: thd[m] - sh for m in thd}

    # point fit
    theta = _anchor(fit_pl([(o, w) for _, o, w in orderings_all], models, reg=a.reg))

    # ---- cluster bootstrap ----
    by_cluster = defaultdict(list)
    for c, o, w in orderings_all:
        by_cluster[c].append((o, w))
    ckeys = list(by_cluster)
    rng = np.random.default_rng(0)
    th_samples = {m: [] for m in models}
    rank_samples = {m: [] for m in models}
    for _ in range(a.bootstrap):
        draw = rng.choice(len(ckeys), size=len(ckeys), replace=True)
        oo = [ow for di in draw for ow in by_cluster[ckeys[di]]]
        th = _anchor(fit_pl(oo, models, reg=a.reg))
        order = sorted(models, key=lambda m: -th[m])
        rk = {m: i + 1 for i, m in enumerate(order)}
        for m in models:
            th_samples[m].append(th[m])
            rank_samples[m].append(rk[m])

    def ci(xs):
        return float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))

    gpm = {m: 0 for m in models}
    for g in games:
        for mm in g.models:
            if mm in gpm:
                gpm[mm] += 1

    order = sorted(models, key=lambda m: -theta[m])
    print("\n" + "=" * 100)
    print("PLACKETT-LUCE RATINGS (log-strength; PL-Elo = x %.1f) + deck-cluster bootstrap 95%% CI" % a.elo_scale)
    print("=" * 100)
    print(f"  {'rank':>4}  {'model':28} {'PL-Elo':>8} {'95% CI':>16} {'CIwidth':>8} {'rankCI':>8} {'games':>6}")
    rows = {}
    for m in order:
        lo, hi = ci(th_samples[m])
        rlo, rhi = int(np.percentile(rank_samples[m], 2.5)), int(np.percentile(rank_samples[m], 97.5))
        e = a.init_rating + theta[m] * a.elo_scale
        elo_lo = a.init_rating + lo * a.elo_scale
        elo_hi = a.init_rating + hi * a.elo_scale
        width = elo_hi - elo_lo
        print(f"  {order.index(m)+1:>4}  {short(m, id2disp)[:28]:28} {e:>8.0f} "
              f"[{elo_lo:>6.0f},{elo_hi:>6.0f}] {width:>8.0f} {f'[{rlo},{rhi}]':>8} {gpm[m]:>6}")
        rows[short(m, id2disp)] = {"pl_logstrength": theta[m], "pl_elo": e,
                                   "ci95_elo": [elo_lo, elo_hi], "ci_width_elo": width,
                                   "rank_ci": [rlo, rhi], "games": gpm[m]}

    # ---- paired same-deck contrasts (the headline A>B claims) ----
    # within-game: among games where BOTH present, A finishes above B; cluster-boot CI
    co = defaultdict(lambda: defaultdict(list))  # (A,B) -> cluster -> list of (a_above_b, score_diff)
    for g in games:
        for ia, ib in combinations(range(len(g.models)), 2):
            A, B = g.models[ia], g.models[ib]
            if restrict and (A not in restrict or B not in restrict):
                continue
            key = (A, B) if A < B else (B, A)
            sa, sb = (g.scores[ia], g.scores[ib]) if key == (A, B) else (g.scores[ib], g.scores[ia])
            out = 1.0 if sa > sb else (0.0 if sa < sb else 0.5)
            co[key][cluster_key(g, a.cluster)].append((out, sa - sb))
    print("\n" + "=" * 92)
    print("PAIRED SAME-DECK/SAME-GAME CONTRASTS (the honest 'A>B' claim; deck-cluster bootstrap)")
    print("=" * 92)
    print(f"  {'pair (A vs B)':44} {'n':>4} {'A>B':>6} {'95% CI':>15} {'Δscore':>8}")
    pair_rows = {}
    for key in sorted(co, key=lambda k: -np.mean([r[0] for c in co[k] for r in co[k][c]])):
        A, B = key
        cl = list(co[key])
        allrecs = [r for c in cl for r in co[key][c]]
        nco = len(allrecs)
        if nco < 5:
            continue
        wr = np.mean([r[0] for r in allrecs])
        dsc = np.mean([r[1] for r in allrecs])
        # cluster bootstrap on the contrast
        boots = []
        for _ in range(a.bootstrap):
            dd = rng.choice(len(cl), size=len(cl), replace=True)
            rr = [r[0] for di in dd for r in co[key][cl[di]]]
            boots.append(np.mean(rr))
        lo, hi = ci(boots)
        sig = "*" if (lo > 0.5 or hi < 0.5) else " "
        print(f"  {(short(A,id2disp)[:20]+' vs '+short(B,id2disp)[:20]):44} {nco:>4} "
              f"{wr:>6.2f} [{lo:>5.2f},{hi:>5.2f}]{sig} {dsc:>8.1f}")
        pair_rows[f"{short(A,id2disp)} vs {short(B,id2disp)}"] = {
            "n": nco, "A_above_B": wr, "ci95": [lo, hi], "mean_score_diff": dsc,
            "resolved": bool(lo > 0.5 or hi < 0.5)}
    print("  (* = CI excludes 0.5 -> direction resolved.  n = #games the pair co-occurred.)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "sources": a.sources, "cluster": a.cluster, "n_games": len(games),
        "n_clusters": nclusters, "bootstrap": a.bootstrap, "reg": a.reg,
        "elo_scale": a.elo_scale, "ratings": rows, "paired_contrasts": pair_rows,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
