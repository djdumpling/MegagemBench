#!/usr/bin/env python3
"""Distributional expected-surplus adjudicator ($0, CPU, existing logs only).

THE QUESTION (pre-registered): the ceiling doc closes point-prediction opponent
models (any point model mis-calls ~20-26% of 1-coin auction resolutions) and
names "distributional opponent modeling" as the sole surviving lever. Before
building anything expensive: on the logged qwen-vs-2xFlash games, how much
ex-post bidding regret does an analytic EXPECTED-SURPLUS bidder recover when it
best-responds to a *distribution* over the max opponent bid, versus

  - the blueprint LLM's actual bids                  (forensics: 4.30)
  - the best fixed-shade bidder on true value        (forensics: 3.83)
  - a point-prediction EV bidder (same info, no dist)
  - the clairvoyant price-aware oracle               (forensics: 2.79)

KILL GATE (pre-registered, from the red-team memo): let D = regret(LLM actual)
- regret(EV-dist on true V). D < 0.3 coins/decision => KILL the distributional
direction (the surviving lever is not worth a build). D >= 0.5 => BUILD.
Between: judgment call, report verbatim.

Also reports D1: a control-variate variance-reduction factor for paired eval
(the "knife-edge luck" term X_g = sum (1{win} - P(win|b)) * (V - b) per game),
measured on the 72 paired value-gate seeds.

Method notes
------------
- F-hat: per opponent seat, gbt point prediction (OOF, GroupKFold) + pooled
  cross-fold OOF residual histogram => integer bid distribution clamped to
  [0, coins_s]. P(win | b) = prod_s [ P(bid_s < b) + tie_s * P(bid_s = b) ]
  with tie_s = 1 iff seat 0 precedes s in the round's tiebreak order
  (independence across the two opponent seats assumed; exact otherwise).
- CV groups = SEED (on/off arms share decks; grouping by game index would leak
  near-duplicate Flash bids across folds).
- All counterfactual bidders resolved against the ACTUAL logged opponent bids
  with the real tiebreak (predicted_winner), so the ladder is internally
  consistent; the LLM row uses the actually-logged win flag.
- V = realized end-game lot value (forensics convention) => the EV bidder here
  isolates the PRICE channel ("perfect value + distributional price"). A
  realistic deployment replaces V with the value head's V-hat; that combo is
  reported as a secondary row where the head's reconstruction exists.
- Myopic caveat: the EV objective ignores future-budget option value; so does
  the price-aware oracle, so the ladder is like-for-like.

Usage:
  python3 scripts/analysis/dist_ev_adjudicator.py \
      --out results/analysis/dist_ev_adjudicator.json
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from megagem.assets import asset_path
from megagem.data import load_value_charts
from megagem.game.cards import ValueChart

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("pikl_transfer_probe", _here / "pikl_transfer_probe.py")
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)
iter_treasure_nodes = _probe.iter_treasure_nodes
predicted_winner = _probe.predicted_winner

_spec2 = importlib.util.spec_from_file_location("flash_bid_model_fit", _here / "flash_bid_model_fit.py")
_fit = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_fit)
build_rows = _fit.build_rows

OPP_MARKER = "gemini-3-flash"
POLICY_MARKER = "qwen"
SHADES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEED_RE = re.compile(r"seed(\d+)")


def load_games(globs: list[str]) -> list[dict]:
    """Load policy-vs-2xFlash trajectory games, tagging each with a stable
    dedup/group key (seed where available) and its arm (on/off/other)."""
    games, seen = [], set()
    for pat in globs:
        for f in sorted(glob.glob(pat, recursive=True)):
            try:
                g = json.load(open(f))
            except Exception:
                continue
            if not isinstance(g, dict):
                continue
            # Result volumes also contain aggregate reports whose ``metadata``
            # fields are scalars/lists.  They are not trajectory records and
            # must not make a broad results/**/*.json scan fail.
            metadata = g.get("metadata") or {}
            if not isinstance(metadata, dict):
                continue
            models = metadata.get("models") or []
            if not (len(models) == 3 and POLICY_MARKER in str(models[0]).lower()
                    and all(OPP_MARKER in str(m).lower() for m in models[1:])):
                continue
            m = SEED_RE.search(os.path.basename(f))
            seed = int(m.group(1)) if m else None
            arm = "on" if "games_on" in f else ("off" if "games_off" in f else "other")
            key = (seed, arm, os.path.basename(f))
            if key in seen:
                continue
            seen.add(key)
            g["_path"], g["_seed"], g["_arm"] = f, seed, arm
            g["_group"] = f"s{seed}" if seed is not None else f"p{os.path.basename(f)}"
            games.append(g)
    return games


def realized_counts(g: dict) -> dict:
    vdf = (g.get("final_results") or {}).get("value_display_final") or {}
    out = {}
    for c, info in vdf.items():
        out[c] = int((info or {}).get("count", 0)) if isinstance(info, dict) else int(info or 0)
    return out


def charts() -> dict[str, ValueChart]:
    return {cid: ValueChart.from_dict(cid, d) for cid, d in load_value_charts().items()}


# --------------------------------------------------------------------------- #
# F-hat: OOF point preds + cross-fold residual pools, grouped by seed          #
# --------------------------------------------------------------------------- #
def fit_oof(rows: list[dict], games: list[dict], n_folds: int = 5, seed: int = 0):
    """Returns (preds, fold_of_group, residuals_by_fold). preds: node_key -> int.
    Residual pool for scoring fold k = OOF residuals of rows in folds != k."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    # remap row group: build_rows used positional game index in node_key[0]
    for r in rows:
        r["group"] = games[r["game"]]["_group"]
    groups = sorted({r["group"] for r in rows})
    fold_of_group = {gr: i % n_folds for i, gr in enumerate(groups)}

    feat_names = sorted(rows[0]["features"])
    X = np.array([[r["features"][f] for f in feat_names] for r in rows], float)
    y = np.array([r["label"] for r in rows], float)
    fold = np.array([fold_of_group[r["group"]] for r in rows])

    preds: dict = {}
    resid_fold = np.zeros(len(rows))
    for k in range(n_folds):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        if not len(te):
            continue
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                              max_depth=4, l2_regularization=1.0,
                                              random_state=seed)
        model.fit(X[tr], y[tr])
        raw = model.predict(X[te])
        for i, p in zip(te, raw):
            ip = max(0, min(rows[i]["coins"], round(float(p))))
            preds[rows[i]["node_key"]] = ip
            resid_fold[i] = rows[i]["label"] - ip  # actual - predicted
    residuals_by_fold = {k: [float(resid_fold[i]) for i in range(len(rows)) if fold[i] != k]
                         for k in range(n_folds)}
    return preds, fold_of_group, residuals_by_fold


def bid_pmf(mu: int, coins: int, resid: np.ndarray) -> np.ndarray:
    """pmf over integer bids 0..coins for one opponent: clamp(mu + eps)."""
    vals = np.clip(np.round(mu + resid).astype(int), 0, coins)
    pmf = np.bincount(vals, minlength=coins + 1).astype(float)
    return pmf / pmf.sum()


def win_prob_curve(opp: list[tuple[int, int, bool]], resid: np.ndarray, bmax: int) -> np.ndarray:
    """P(seat0 wins | b) for b=0..bmax. opp = [(mu, coins, seat0_beats_tie)]."""
    out = np.ones(bmax + 1)
    for mu, coins_s, beats in opp:
        pmf = bid_pmf(mu, coins_s, resid)
        cdf = np.cumsum(pmf)
        for b in range(bmax + 1):
            p_lt = cdf[min(b, coins_s)] - pmf[min(b, coins_s)] if b <= coins_s else 1.0
            p_eq = pmf[b] if b <= coins_s else 0.0
            out[b] *= p_lt + (p_eq if beats else 0.0)
    return out


# --------------------------------------------------------------------------- #
# decisions + regret ladder                                                    #
# --------------------------------------------------------------------------- #
def regret(V: float, bid: float, won: bool, price: float) -> float:
    return max(0.0, V - price) - ((V - bid) if won else 0.0)


def beats_tie(tb: list[int], s: int) -> bool:
    if not tb or 0 not in tb or s not in tb:
        return True  # resolve_auction mirror: no tiebreak -> min id wins
    return tb.index(0) < tb.index(s)


def collect_decisions(games: list[dict], preds: dict, fold_of_group: dict,
                      residuals_by_fold: dict, ch: dict[str, ValueChart]) -> list[dict]:
    out = []
    for gi, g in enumerate(games):
        rn = realized_counts(g)
        chart = ch.get((g.get("metadata") or {}).get("value_chart") or "A", ch["A"])
        k = fold_of_group.get(g["_group"])
        if k is None:  # game contributed no Flash bid rows (e.g. summary JSON)
            continue
        resid = np.array(residuals_by_fold[k])
        for ni, node in enumerate(iter_treasure_nodes(g)):
            if 0 not in node["bids"]:
                continue
            opp_seats = [s for s in node["bids"] if s != 0]
            if len(opp_seats) != 2 or any((gi, ni, s) not in preds for s in opp_seats):
                continue
            V = float(sum(chart.get_gem_value(rn.get(c, 0)) for c in node["gems"]))
            coins0 = int(node["coins"].get(0, 0))
            opp_bids = {s: int(node["bids"][s]) for s in opp_seats}
            price = float(max(opp_bids.values()))
            tb = node["tiebreak"]
            opp_spec = [(preds[(gi, ni, s)], int(node["coins"].get(s, 0)), beats_tie(tb, s))
                        for s in opp_seats]
            pwin = win_prob_curve(opp_spec, resid, coins0)
            out.append({
                "gi": gi, "ni": ni, "group": g["_group"], "seed": g["_seed"],
                "arm": g["_arm"], "V": V, "coins": coins0, "price": price,
                "bid0": int(node["bids"][0]), "won": node["winner_id"] == 0,
                "opp_bids": opp_bids, "tb": list(tb), "pwin": pwin,
                "mu_max": max(p for p, _, _ in opp_spec),
                "gems": list(node["gems"]), "round_number": node["round"],
            })
    return out


def attach_head_values(decisions: list[dict], games: list[dict], model_path: str) -> int:
    """Compute the value head's decision-time lot estimate V-hat for each decision
    (same reconstruction as the retired eval_forensics H1). Returns count attached."""
    try:
        from megagem.value_head.train import bidtime_state  # noqa: PLC0415
        from megagem.value_head.value_estimator import ValueEstimator, value_features  # noqa: PLC0415
        est = ValueEstimator.load(model_path)
    except Exception as e:  # head optional: ladder still valid without it
        print(f"value head unavailable ({e}); skipping V-hat rows")
        return 0
    n = 0
    by_game: dict[int, list[dict]] = defaultdict(list)
    for d in decisions:
        by_game[d["gi"]].append(d)
    for gi, ds in by_game.items():
        g = games[gi]
        rounds = g.get("rounds") or []
        cid = (g.get("metadata") or {}).get("value_chart") or "A"
        rn_to_ri = {r.get("round_number"): i for i, r in enumerate(rounds)}
        for d in ds:
            ri = rn_to_ri.get(d["round_number"])
            if ri is None:
                continue
            try:
                disp, coll_all, own, _ = bidtime_state(rounds, ri)
            except Exception:
                continue
            if not own:
                continue
            rnum = int(d["round_number"] or ri + 1)
            d["Vhat"] = float(sum(
                est.evalue(value_features(color=c, display_counts=disp,
                                          own_hand_counts=own[0],
                                          collection_counts_all=coll_all,
                                          round_number=rnum), cid)
                for c in d["gems"]))
            n += 1
    return n


def wins_vs_actual(d: dict, b: int) -> bool:
    bids = dict(d["opp_bids"]); bids[0] = int(b)
    return predicted_winner(bids, d["tb"]) == 0


def score(decisions: list[dict], bid_fn) -> np.ndarray:
    """Per-decision regret vector for a counterfactual bidder."""
    out = np.zeros(len(decisions))
    for i, d in enumerate(decisions):
        b = int(max(0, min(d["coins"], bid_fn(d))))
        out[i] = regret(d["V"], b, wins_vs_actual(d, b), d["price"])
    return out


def cluster_t(diff: np.ndarray, groups: list) -> tuple[float, float, int]:
    """Mean, t-stat of per-decision diffs clustered by game group."""
    by = defaultdict(list)
    for v, g in zip(diff, groups):
        by[g].append(v)
    means = np.array([np.mean(v) for v in by.values()])
    se = means.std(ddof=1) / np.sqrt(len(means))
    return float(diff.mean()), float(np.mean(means) / se) if se > 0 else 0.0, len(means)


def main() -> int:
    ap = argparse.ArgumentParser()
    # Input corpus = logged games in gitignored results/ scratch (regenerable).
    ap.add_argument("--traj-globs", nargs="+",
                    default=["results/**/*.json", "src/megagem/evals/results/*.json"])
    ap.add_argument("--n-folds", type=int, default=5)
    # Regenerable results/ scratch; D1 is skipped gracefully when absent.
    ap.add_argument("--value-gate-dir", default="results/value_probe/value_gate_flash_v1")
    ap.add_argument("--value-head", default=str(asset_path("value_head.pkl")))
    ap.add_argument("--out", default="results/analysis/dist_ev_adjudicator.json")
    args = ap.parse_args()

    ch = charts()
    games = load_games(args.traj_globs)
    n_on = sum(g["_arm"] == "on" for g in games)
    n_off = sum(g["_arm"] == "off" for g in games)
    print(f"loaded {len(games)} qwen-vs-2xFlash games (on={n_on} off={n_off} other={len(games)-n_on-n_off})")

    rows = build_rows(games)
    print(f"Flash treasure-bid rows: {len(rows)}")
    preds, fold_of_group, residuals_by_fold = fit_oof(rows, games, n_folds=args.n_folds)
    errs = [preds[r["node_key"]] - r["label"] for r in rows if r["node_key"] in preds]
    pooled = np.concatenate([np.array(v) for v in residuals_by_fold.values()])
    print(f"gbt OOF (seed-grouped): MAE={np.mean(np.abs(errs)):.3f} bias={np.mean(errs):+.3f} "
          f"resid sd={pooled.std():.2f} within1={np.mean(np.abs(errs) <= 1):.0%}")

    decisions = collect_decisions(games, preds, fold_of_group, residuals_by_fold, ch)
    print(f"seat-0 treasure decisions scored: {len(decisions)}")
    knife = [abs(d['bid0'] - d['price']) for d in decisions]
    print(f"knife-edge |bid0-price|: <=1 coin {np.mean(np.array(knife)<=1):.0%}, "
          f"median {np.median(knife):.0f}")

    groups = [d["group"] for d in decisions]

    # ---- the ladder ------------------------------------------------------- #
    llm = np.array([regret(d["V"], d["bid0"], d["won"], d["price"]) for d in decisions])

    def shade_regret(sh):
        return np.array([regret(d["V"], b, wins_vs_actual(d, b), d["price"])
                         for d, b in ((d, int(max(0, min(d["coins"], round(d["V"] * sh)))))
                                      for d in decisions)])
    shade_best = min(((float(shade_regret(sh).mean()), sh) for sh in SHADES))
    shade_vec = shade_regret(shade_best[1])

    def ev_dist_bid(d):
        b = np.arange(d["coins"] + 1)
        ev = (d["V"] - b) * d["pwin"]
        return int(b[np.argmax(ev)])

    def ev_point_bid(d):
        # same info, point model: outbid the predicted max (or its tie) iff EV>0
        mu = d["mu_max"]
        need = mu if all(beats_tie(d["tb"], s) for s in d["opp_bids"]) else mu + 1
        return need if (d["V"] > need and need <= d["coins"]) else 0

    def oracle_bid(d):
        for b in range(0, d["coins"] + 1):  # cheapest winning bid vs actual prices
            if wins_vs_actual(d, b):
                return b if d["V"] > b else 0
        return 0

    ev_dist = score(decisions, ev_dist_bid)
    ev_point = score(decisions, ev_point_bid)
    oracle = score(decisions, oracle_bid)

    ladder = {
        "llm_actual": float(llm.mean()),
        f"fixed_shade_best(sh{shade_best[1]})_trueV": float(shade_vec.mean()),
        "ev_point_trueV": float(ev_point.mean()),
        "ev_dist_trueV": float(ev_dist.mean()),
        "price_aware_oracle": float(oracle.mean()),
    }
    print("\nREGRET LADDER (coins/decision, lower=better; vs actual Flash prices)")
    for kname, v in ladder.items():
        print(f"   {kname:<34} {v:.3f}")

    d_llm, t_llm, ngame = cluster_t(llm - ev_dist, groups)
    d_shade, t_shade, _ = cluster_t(shade_vec - ev_dist, groups)
    d_point, t_point, _ = cluster_t(ev_point - ev_dist, groups)
    print(f"\nDELTAS (positive = EV-dist better), cluster-robust by game (n={ngame}):")
    print(f"   vs LLM actual:        D={d_llm:+.3f}  t={t_llm:+.2f}")
    print(f"   vs best fixed shade:  D={d_shade:+.3f}  t={t_shade:+.2f}")
    print(f"   vs EV-point (same F): D={d_point:+.3f}  t={t_point:+.2f}   <- the distributional premium")

    gate = "BUILD" if d_llm >= 0.5 else ("KILL" if d_llm < 0.3 else "JUDGMENT")
    print(f"\nPRE-REGISTERED GATE on D(LLM - EV-dist) = {d_llm:+.3f}  =>  {gate}")

    # ON-only sanity slice (forensics used ON arm, n=853)
    on_idx = [i for i, d in enumerate(decisions) if d["arm"] == "on"]
    if on_idx:
        print(f"\nON-arm slice (n={len(on_idx)}; forensics ladder was 4.30/3.83/2.79):")
        print(f"   llm={llm[on_idx].mean():.3f} shade={shade_vec[on_idx].mean():.3f} "
              f"ev_dist={ev_dist[on_idx].mean():.3f} oracle={oracle[on_idx].mean():.3f}")

    # OFF-only slice: clean blueprint (no injection harm) — the conservative delta
    off_idx = [i for i, d in enumerate(decisions) if d["arm"] == "off"]
    off_stats = None
    if off_idx:
        goff = [decisions[i]["group"] for i in off_idx]
        d_off, t_off, ng_off = cluster_t((llm - ev_dist)[off_idx], goff)
        off_stats = {"n": len(off_idx), "llm": float(llm[off_idx].mean()),
                     "ev_dist": float(ev_dist[off_idx].mean()),
                     "delta": d_off, "t": t_off, "n_groups": ng_off}
        print(f"\nOFF-arm slice (clean blueprint, n={len(off_idx)}):")
        print(f"   llm={llm[off_idx].mean():.3f} ev_dist={ev_dist[off_idx].mean():.3f} "
              f"D={d_off:+.3f} t={t_off:+.2f}")

    # ---- realistic combo: imperfect head V-hat x distributional price ------ #
    n_vh = attach_head_values(decisions, games, args.value_head)
    vh_rows = {}
    if n_vh:
        sub = [i for i, d in enumerate(decisions) if "Vhat" in d]
        dsub = [decisions[i] for i in sub]
        gsub = [d["group"] for d in dsub]

        def shade_vh(sh):
            return np.array([regret(d["V"], b, wins_vs_actual(d, b), d["price"])
                             for d, b in ((d, int(max(0, min(d["coins"], round(d["Vhat"] * sh)))))
                                          for d in dsub)])
        best_vh = min(((float(shade_vh(sh).mean()), sh) for sh in SHADES))

        def ev_dist_vh(d):
            b = np.arange(d["coins"] + 1)
            ev = (d["Vhat"] - b) * d["pwin"]
            return int(b[np.argmax(ev)])
        ev_vh = score(dsub, ev_dist_vh)
        llm_sub = np.array([regret(d["V"], d["bid0"], d["won"], d["price"]) for d in dsub])
        d_vh, t_vh, ng_vh = cluster_t(llm_sub - ev_vh, gsub)
        vh_rows = {
            "n": len(dsub),
            "llm_actual": float(llm_sub.mean()),
            f"fixed_shade_best(sh{best_vh[1]})_headV": float(best_vh[0]),
            "ev_dist_headV": float(ev_vh.mean()),
            "delta_llm_minus_evdist_headV": {"d": d_vh, "t": t_vh, "n_groups": ng_vh},
        }
        print(f"\nREALISTIC COMBO (imperfect head V-hat, n={len(dsub)}):")
        print(f"   llm={llm_sub.mean():.3f} | shade-best(V-hat)={best_vh[0]:.3f}(sh{best_vh[1]}) "
              f"| EV-dist(V-hat)={ev_vh.mean():.3f}")
        print(f"   D(LLM - EV-dist(V-hat)) = {d_vh:+.3f}  t={t_vh:+.2f}  "
              f"<- realizable TODAY with existing components")

    # ---- D1: control-variate variance reduction on paired value-gate seeds - #
    d1 = {}
    try:
        aon = {g["seed"]: g for g in json.load(open(os.path.join(args.value_gate_dir, "value_gate_on.json")))["per_game"]}
        aof = {g["seed"]: g for g in json.load(open(os.path.join(args.value_gate_dir, "value_gate_off.json")))["per_game"]}
        X = defaultdict(float)  # (seed, arm) -> luck term
        for d in decisions:
            if d["seed"] is None or d["arm"] not in ("on", "off"):
                continue
            p_at_actual = d["pwin"][min(d["bid0"], d["coins"])]
            X[(d["seed"], d["arm"])] += (float(d["won"]) - float(p_at_actual)) * (d["V"] - d["bid0"])
        seeds = sorted(set(aon) & set(aof))
        dm = np.array([aon[s]["policy_delta"] - aof[s]["policy_delta"] for s in seeds])
        dx = np.array([X[(s, "on")] - X[(s, "off")] for s in seeds])
        beta = float(np.cov(dm, dx)[0, 1] / np.var(dx, ddof=1))
        adj = dm - beta * dx
        rho = float(np.corrcoef(dm, dx)[0, 1])
        vr = float(np.var(dm, ddof=1) / np.var(adj, ddof=1))
        mde0 = 2.8 * dm.std(ddof=1) / np.sqrt(len(seeds))
        mde1 = 2.8 * adj.std(ddof=1) / np.sqrt(len(seeds))
        d1 = {"n_pairs": len(seeds), "rho": rho, "beta": beta, "var_reduction_factor": vr,
              "mde_before": float(mde0), "mde_after": float(mde1)}
        print(f"\nD1 CONTROL VARIATE (paired value-gate, n={len(seeds)}):")
        print(f"   corr(dMargin, dLuck)={rho:+.3f}  beta={beta:.2f}  "
              f"VR factor={vr:.2f}x  MDE {mde0:.1f} -> {mde1:.1f} margin pts")
    except FileNotFoundError as e:
        print(f"\nD1 skipped: {e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_games": len(games), "n_rows": len(rows), "n_decisions": len(decisions),
        "gbt_oof": {"mae": float(np.mean(np.abs(errs))), "bias": float(np.mean(errs)),
                    "resid_sd": float(pooled.std()),
                    "within1": float(np.mean(np.abs(errs) <= 1))},
        "knife_edge_le1": float(np.mean(np.array(knife) <= 1)),
        "ladder": ladder,
        "deltas_clustered": {
            "llm_minus_evdist": {"d": d_llm, "t": t_llm, "n_groups": ngame},
            "shade_minus_evdist": {"d": d_shade, "t": t_shade},
            "evpoint_minus_evdist": {"d": d_point, "t": t_point},
        },
        "gate": {"D": d_llm, "kill_lt": 0.3, "build_ge": 0.5, "verdict": gate},
        "on_slice": {"n": len(on_idx), "llm": float(llm[on_idx].mean()),
                     "shade": float(shade_vec[on_idx].mean()),
                     "ev_dist": float(ev_dist[on_idx].mean()),
                     "oracle": float(oracle[on_idx].mean())} if on_idx else None,
        "realistic_combo_headV": vh_rows or None,
        "off_slice": off_stats,
        "d1_control_variate": d1,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
