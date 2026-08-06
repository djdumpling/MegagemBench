#!/usr/bin/env python3
"""Build the frozen E1 ev_dist opponent-model artifact ($0, CPU, existing logs).

Steps (all reported, all saved):
  1. Fit gbt on BID-TIME features (src/megagem/environment/bid_model.py canonical
     builder — display from round ri-1) over all logged qwen-vs-2xFlash games,
     5-fold CV grouped by SEED. Report OOF MAE/bias/within1.
  2. LEAK CHECK: same fit with the original post-round-display features
     (flash_bid_model_fit convention) — quantifies how much same-round display
     flattered the earlier numbers.
  3. D3 REVALIDATION: recompute the regret ladder (LLM / fixed-shade / EV-point /
     EV-dist / oracle) with the corrected bid-time F-hat. E1's go/no-go
     prediction rests on this number, not the original D3.
  4. LAMBDA-LADDER GATE (pre-registered): held-out per-bid log-loss of
     pi_lambda(b) ∝ F-hat(b) * exp(Q-hat(b)/lambda); adopt the typed model only
     if a finite lambda beats lambda=inf (pure F-hat).
  5. Freeze: deploy gbt fit on ALL rows + pooled OOF residuals + feature spec +
     parity fixtures -> results/flash_bid_model/ev_dist_v1.pkl (+ meta JSON).

Usage: python3 scripts/analysis/build_ev_dist_artifact.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from megagem.environment.bid_model import (
    FEATURES, EvDistModel, MODEL_VERSION, beats_tie, bid_pmf, ev_best,
    feature_vector, iter_nodes_bidtime, win_curve,
)
from megagem.data import load_value_charts
from megagem.game.cards import ValueChart

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("dist_ev_adjudicator", _here / "dist_ev_adjudicator.py")
_adj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adj)

OPP_MARKER = "gemini-3-flash"
SHADES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def charts() -> dict[str, ValueChart]:
    return {cid: ValueChart.from_dict(cid, d) for cid, d in load_value_charts().items()}


def build_rows_bidtime(games: list[dict], ch) -> list[dict]:
    rows = []
    for gi, g in enumerate(games):
        md = g.get("metadata") or {}
        models = md.get("models") or []
        chart = ch.get(md.get("value_chart") or "A", ch["A"])
        flash_seats = [i for i, m in enumerate(models) if OPP_MARKER in str(m)]
        if not flash_seats:
            continue
        for ni, node in enumerate(iter_nodes_bidtime(g)):
            for s in flash_seats:
                if s not in node["bids"]:
                    continue
                rows.append({"game": gi, "group": g["_group"], "seat": s,
                             "node_key": (gi, ni, s), "node": node, "chart": chart,
                             "coins": int(node["coins"].get(s, 0)),
                             "label": int(node["bids"][s])})
    return rows


def fit_oof(rows: list[dict], n_folds=5, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    groups = sorted({r["group"] for r in rows})
    fold_of_group = {g: i % n_folds for i, g in enumerate(groups)}
    X = np.array([feature_vector(r["node"], r["seat"], r["chart"]) for r in rows])
    y = np.array([r["label"] for r in rows], float)
    fold = np.array([fold_of_group[r["group"]] for r in rows])
    preds = np.zeros(len(rows))
    for k in range(n_folds):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        if not len(te):
            continue
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                          max_depth=4, l2_regularization=1.0,
                                          random_state=seed)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    mu = np.array([max(0, min(r["coins"], round(float(p)))) for r, p in zip(rows, preds)])
    err = mu - y
    return {"mu": mu, "fold": fold, "fold_of_group": fold_of_group, "X": X, "y": y,
            "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err)),
            "within1": float(np.mean(np.abs(err) <= 1)), "resid": (y - mu)}


def main() -> int:
    ap = argparse.ArgumentParser()
    # Input corpus = logged games in gitignored results/ scratch (regenerable).
    ap.add_argument("--traj-globs", nargs="+",
                    default=["results/**/*.json", "src/megagem/evals/results/*.json"])
    ap.add_argument("--out", default="results/flash_bid_model/ev_dist_v1.pkl")
    a = ap.parse_args()
    ch = charts()
    games = _adj.load_games(a.traj_globs)
    rows = build_rows_bidtime(games, ch)
    print(f"{len(games)} games -> {len(rows)} Flash treasure-bid rows (bid-time features)")

    # 1. bid-time fit ---------------------------------------------------------
    bt = fit_oof(rows)
    print(f"[1] BID-TIME OOF:   MAE={bt['mae']:.3f} bias={bt['bias']:+.3f} within1={bt['within1']:.0%}")

    # 2. leak check: original post-round-display features ---------------------
    legacy_rows = _adj.build_rows(games)  # flash_bid_model_fit convention
    for r in legacy_rows:
        r["group"] = games[r["game"]]["_group"]
    lg_preds, lg_fold, _ = _adj.fit_oof(legacy_rows, games)
    lg_err = [lg_preds[r["node_key"]] - r["label"] for r in legacy_rows if r["node_key"] in lg_preds]
    print(f"[2] POST-ROUND OOF: MAE={np.mean(np.abs(lg_err)):.3f} bias={np.mean(lg_err):+.3f} "
          f"within1={np.mean(np.abs(np.array(lg_err)) <= 1):.0%}   <- leak delta = "
          f"{bt['mae'] - float(np.mean(np.abs(lg_err))):+.3f} MAE")

    # 3. D3 revalidation with bid-time F-hat ----------------------------------
    mu_by_key = {r["node_key"]: int(m) for r, m in zip(rows, bt["mu"])}
    resid_by_fold = {k: bt["resid"][bt["fold"] != k] for k in range(5)}
    decisions = []
    for gi, g in enumerate(games):
        md = g.get("metadata") or {}
        chart = ch.get(md.get("value_chart") or "A", ch["A"])
        rn = _adj.realized_counts(g)
        k = bt["fold_of_group"].get(g["_group"])
        if k is None:
            continue
        for ni, node in enumerate(iter_nodes_bidtime(g)):
            if 0 not in node["bids"]:
                continue
            opp = [s for s in node["bids"] if s != 0]
            if len(opp) != 2 or any((gi, ni, s) not in mu_by_key for s in opp):
                continue
            V = float(sum(chart.get_gem_value(rn.get(c, 0)) for c in node["gems"]))
            spec = [(mu_by_key[(gi, ni, s)], int(node["coins"].get(s, 0)),
                     beats_tie(node["tiebreak"], 0, s)) for s in opp]
            pwin = win_curve(spec, resid_by_fold[k], int(node["coins"].get(0, 0)))
            decisions.append({
                "group": g["_group"], "V": V, "coins": int(node["coins"].get(0, 0)),
                "bid0": int(node["bids"][0]), "won": node["winner_id"] == 0,
                "opp_bids": {s: int(node["bids"][s]) for s in opp},
                "tb": list(node["tiebreak"]), "pwin": pwin,
                "mu_max": max(m for m, _, _ in spec),
            })
    print(f"[3] D3 revalidation on {len(decisions)} seat-0 decisions (bid-time F-hat):")

    def regret(V, bid, won, price):
        return max(0.0, V - price) - ((V - bid) if won else 0.0)

    def wins(d, b):
        bids = dict(d["opp_bids"]); bids[0] = int(b)
        hi = max(bids.values()); tied = [s for s, x in bids.items() if x == hi]
        if len(tied) == 1:
            return tied[0] == 0
        tb = d["tb"]
        w = min(tied, key=lambda s: tb.index(s) if tb and s in tb else (s if not tb else len(tb)))
        return w == 0

    def score(bid_fn):
        return np.array([regret(d["V"], b, wins(d, b), max(d["opp_bids"].values()))
                         for d, b in ((d, int(max(0, min(d["coins"], bid_fn(d))))) for d in decisions)])

    llm = np.array([regret(d["V"], d["bid0"], d["won"], max(d["opp_bids"].values())) for d in decisions])
    shade_best = min((float(score(lambda d, sh=sh: round(d["V"] * sh)).mean()), sh) for sh in SHADES)
    ev_dist = score(lambda d: ev_best(d["V"], d["pwin"])[0])
    ev_point = score(lambda d: (d["mu_max"] + 1) if d["V"] > d["mu_max"] + 1 else 0)
    oracle = score(lambda d: next((b for b in range(d["coins"] + 1) if wins(d, b) and d["V"] > b), 0))
    d_llm, t_llm, ng = _adj.cluster_t(llm - ev_dist, [d["group"] for d in decisions])
    print(f"    LLM {llm.mean():.3f} | shade {shade_best[0]:.3f}(sh{shade_best[1]}) | "
          f"EV-point {ev_point.mean():.3f} | EV-dist {ev_dist.mean():.3f} | oracle {oracle.mean():.3f}")
    print(f"    D(LLM - EV-dist) = {d_llm:+.3f}  t={t_llm:+.2f}  (clusters={ng})  "
          f"=> {'BUILD holds' if d_llm >= 0.5 else 'REVISIT'}")

    # 4. lambda-ladder gate ----------------------------------------------------
    print("[4] lambda-ladder gate (held-out per-bid log-loss, lower=better):")
    grid = [float("inf"), 8.0, 4.0, 2.0, 1.0, 0.5]
    ll = {lam: [] for lam in grid}
    node_groups = {}
    for gi, g in enumerate(games):
        for ni, node in enumerate(iter_nodes_bidtime(g)):
            node_groups[(gi, ni)] = (node, g)
    by_node: dict[tuple, list] = {}
    for r, m in zip(rows, bt["mu"]):
        by_node.setdefault(r["node_key"][:2], []).append((r, int(m)))
    for (gi, ni), seat_rows in by_node.items():
        node, g = node_groups[(gi, ni)]
        chart = ch.get((g.get("metadata") or {}).get("value_chart") or "A", ch["A"])
        k = bt["fold_of_group"][g["_group"]]
        resid = resid_by_fold[k]
        fv = float(sum(chart.get_gem_value(node["display"].get(c, 0)) for c in node["gems"]))
        for r, mu_s in seat_rows:
            s = r["seat"]
            others = [(q, mq) for q, mq in [(rr["seat"], mm) for rr, mm in seat_rows if rr["seat"] != s]]
            spec = [(mq, int(node["coins"].get(q, 0)), beats_tie(node["tiebreak"], s, q))
                    for q, mq in others]
            if 0 in node["bids"]:  # seat-0 (policy) actual bid as a point opponent
                spec.append((int(node["bids"][0]), int(node["coins"].get(0, 0)),
                             beats_tie(node["tiebreak"], s, 0)))
            coins_s = int(node["coins"].get(s, 0))
            pmf = bid_pmf(mu_s, coins_s, resid)
            pw = win_curve(spec, resid, coins_s)
            b = np.arange(coins_s + 1, dtype=float)
            q = (fv - b) * pw
            for lam in grid:
                w = pmf if np.isinf(lam) else pmf * np.exp((q - q.max()) / lam)
                w = w / w.sum()
                ll[lam].append(-np.log(max(w[min(r["label"], coins_s)], 1e-12)))
    base = float(np.mean(ll[float("inf")]))
    best_lam, best_ll = None, base
    for lam in grid:
        v = float(np.mean(ll[lam]))
        print(f"    lambda={lam:>4}: logloss {v:.4f}" + ("  <- pure F-hat" if np.isinf(lam) else ""))
        if v < best_ll - 1e-4 and not np.isinf(lam):
            best_lam, best_ll = lam, v
    typed = best_lam is not None
    print(f"    => {'ADOPT typed model lambda=' + str(best_lam) if typed else 'KEEP plain F-hat (no finite lambda beats imitation)'}")

    # 5. freeze deploy artifact -------------------------------------------------
    from sklearn.ensemble import HistGradientBoostingRegressor
    import sklearn
    deploy = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                           max_depth=4, l2_regularization=1.0,
                                           random_state=0)
    deploy.fit(bt["X"], bt["y"])
    rng = np.random.default_rng(0)
    fix_idx = rng.choice(len(rows), size=8, replace=False)
    fixtures = [{"x": bt["X"][i].tolist(),
                 "mu_raw": float(deploy.predict(bt["X"][i:i + 1])[0])} for i in fix_idx]
    meta = {"version": MODEL_VERSION, "sklearn": sklearn.__version__,
            "n_games": len(games), "n_rows": len(rows),
            "oof": {"mae": bt["mae"], "bias": bt["bias"], "within1": bt["within1"]},
            "display_semantics": "bidtime (round ri-1)", "typed_lambda": best_lam,
            "d3_revalidation": {"d_llm_minus_evdist": d_llm, "t": t_llm,
                                "ladder": {"llm": float(llm.mean()),
                                           "shade": shade_best[0],
                                           "ev_point": float(ev_point.mean()),
                                           "ev_dist": float(ev_dist.mean()),
                                           "oracle": float(oracle.mean())}},
            "fixtures": fixtures}
    EvDistModel(deploy, bt["resid"], FEATURES, meta).save(a.out)
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[5] froze {a.out} (residuals n={len(bt['resid'])}, sklearn {sklearn.__version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
