#!/usr/bin/env python3
"""Contextual Gemini-Flash bid model ($0 offline fit): is the piKL bid-search
direction FIXABLE?

The transfer probe (scripts/analysis/pikl_transfer_probe.py) found that the Q
rollout's opponent model is the binding constraint: the market model predicts
Flash's treasure bids at MAE 2.40 while the median auction is decided by 1 coin,
flipping 22.6% of simulated auction resolutions. This script asks whether a
contextual model fit on the saved Flash bids can do MATERIALLY better — using
ONLY public decision-time state, evaluated leak-free.

Method
------
- One row per (treasure auction, Flash seat) from saved trajectories. Features
  are public-only: coins, lot size, lot display fair value, market anchor stats
  (matching/any recent winning bids), tiebreak position, round, display total,
  max other-seat coins, and the market model's own prediction (stacking).
- Ridge + HistGradientBoostingRegressor, predictions rounded and clamped to
  [0, coins]. Baselines: market_pred (current rollout model), fair_value_pred,
  global-median.
- 5-fold GroupKFold BY GAME → out-of-fold predictions for every row → MAE /
  bias / within-1 / within-2, plus the auction-resolution FLIP RATE on the
  gate-B-like games driven by the OOF predictions.

PRE-REGISTERED KILL GATE
------------------------
PASS iff the best learned model reaches OOF MAE <= 1.8 AND flip rate <= 0.15.
Otherwise STOP: point-prediction opponents cannot reach the ~1–2 coin decision
margins, so do not spend more held-out API budget on the current bid-search
operator (prefer distribution-over-bids opponents or the value-head direction).

Usage
-----
  python scripts/analysis/flash_bid_model_fit.py \
      --traj-globs 'results/**/*.json' 'src/megagem/evals/results/*.json' \
      --out results/flash_bid_model/fit.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

from megagem.data import load_value_charts
from megagem.environment.pikl_search import MARKET_DEFAULTS
from megagem.game.cards import ValueChart

# Reuse the tested trajectory-reconstruction layer from the transfer probe.
_spec = importlib.util.spec_from_file_location(
    "pikl_transfer_probe", Path(__file__).resolve().parent / "pikl_transfer_probe.py")
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)

iter_treasure_nodes = _probe.iter_treasure_nodes
market_pred = _probe.market_pred
fair_value_pred = _probe.fair_value_pred
predicted_winner = _probe.predicted_winner
load_traj_games = _probe.load_traj_games

# Pre-registered gate thresholds (see module docstring).
GATE_MAX_MAE = 1.8
GATE_MAX_FLIP = 0.15

OPP_MARKER = "gemini-3-flash"
POLICY_MARKER = "qwen"


# --------------------------------------------------------------------------- #
# featurization                                                                #
# --------------------------------------------------------------------------- #
def _charts() -> dict[str, ValueChart]:
    return {cid: ValueChart.from_dict(cid, d) for cid, d in load_value_charts().items()}


def build_rows(games: list[dict]) -> list[dict]:
    """One row per (treasure auction, Flash seat): public-only features, the
    actual bid as label, and a (game, node, seat) key for OOF flip-rate joins."""
    charts = _charts()
    rows: list[dict] = []
    for gi, g in enumerate(games):
        md = g.get("metadata") or {}
        models = md.get("models") or []
        chart = charts.get(md.get("value_chart") or "A", charts["A"])
        flash_seats = [i for i, m in enumerate(models) if OPP_MARKER in str(m)]
        if not flash_seats:
            continue
        for ni, node in enumerate(iter_treasure_nodes(g)):
            hist = node["history"]
            matching = [wb for (gc, wb) in hist[-5:] if gc == node["gems_count"]]
            any_recent = [wb for (_gc, wb) in hist[-5:]]
            lot_value = sum(chart.get_gem_value(node["display"].get(c, 0))
                            for c in node["gems"])
            lot_value_next = sum(chart.get_gem_value(node["display"].get(c, 0) + 1)
                                 for c in node["gems"])
            for s in flash_seats:
                if s not in node["bids"]:
                    continue
                coins = node["coins"].get(s, 0)
                others = [v for k, v in node["coins"].items() if k != s]
                tb = node["tiebreak"]
                rows.append({
                    "game": gi,
                    "seat": s,
                    "node_key": (gi, ni, s),
                    "coins": coins,
                    "label": int(node["bids"][s]),
                    "features": {
                        "coins": float(coins),
                        "max_other_coins": float(max(others)) if others else 0.0,
                        "gems_count": float(node["gems_count"]),
                        "lot_value": float(lot_value),
                        "lot_value_next": float(lot_value_next),
                        "display_total": float(sum(node["display"].values())),
                        "round": float(node["round"] or 0),
                        "hist_n_total": float(len(hist)),
                        "hist_n_match": float(len(matching)),
                        "match_median": float(statistics.median(matching)) if matching else -1.0,
                        "match_last": float(matching[-1]) if matching else -1.0,
                        "any_median": float(statistics.median(any_recent)) if any_recent else -1.0,
                        "last_winning_bid": float(hist[-1][1]) if hist else -1.0,
                        "is_last_tiebreak": 1.0 if (tb and tb[-1] == s) else 0.0,
                        "market_pred": float(market_pred(node, s, **MARKET_DEFAULTS)),
                        "fair_value_pred": float(fair_value_pred(node, s, chart)),
                    },
                })
    return rows


def clamp_predictions(raw: list[float], coins: list[int]) -> list[int]:
    """Bids are integers in [0, coins]."""
    return [max(0, min(int(c), round(float(r)))) for r, c in zip(raw, coins)]


# --------------------------------------------------------------------------- #
# grouped CV                                                                   #
# --------------------------------------------------------------------------- #
def group_folds(rows: list[dict], n_folds: int = 5) -> list[tuple[list[int], list[int]]]:
    """Deterministic GroupKFold by game id: a game's rows are never split across
    train and test (Flash's bids within one game share context ⇒ leakage)."""
    games = sorted({r["game"] for r in rows})
    folds: list[tuple[list[int], list[int]]] = []
    for k in range(n_folds):
        test_games = {g for i, g in enumerate(games) if i % n_folds == k}
        te = [i for i, r in enumerate(rows) if r["game"] in test_games]
        tr = [i for i, r in enumerate(rows) if r["game"] not in test_games]
        if te:
            folds.append((tr, te))
    return folds


def oof_predict(rows: list[dict], model_factory, n_folds: int = 5) -> dict:
    """Out-of-fold integer predictions keyed by node_key."""
    import numpy as np

    feat_names = sorted(rows[0]["features"])
    X = np.array([[r["features"][f] for f in feat_names] for r in rows], float)
    y = np.array([r["label"] for r in rows], float)
    preds: dict = {}
    for tr, te in group_folds(rows, n_folds):
        model = model_factory()
        model.fit(X[tr], y[tr])
        raw = model.predict(X[te])
        clamped = clamp_predictions(list(raw), [rows[i]["coins"] for i in te])
        for i, p in zip(te, clamped):
            preds[rows[i]["node_key"]] = p
    return preds


def score_preds(rows: list[dict], preds: dict) -> dict:
    errs = [preds[r["node_key"]] - r["label"] for r in rows if r["node_key"] in preds]
    if not errs:
        return {}
    n = len(errs)
    ae = [abs(e) for e in errs]
    return {
        "n": n,
        "mae": round(statistics.fmean(ae), 3),
        "medae": round(statistics.median(ae), 3),
        "bias_mean": round(statistics.fmean(errs), 3),
        "within1": round(sum(1 for a in ae if a <= 1) / n, 4),
        "within2": round(sum(1 for a in ae if a <= 2) / n, 4),
    }


# --------------------------------------------------------------------------- #
# OOF flip rate                                                                #
# --------------------------------------------------------------------------- #
def flip_rate_with_preds(games: list[dict], preds: dict) -> dict:
    """Auction-resolution flip rate on gate-B-like games (policy seat 0 vs
    all-Flash), resolving with the policy's ACTUAL bid and predicted opp bids.
    Auctions with any missing prediction are skipped (OOF coverage is total in
    practice; partial dicts only occur in tests)."""
    n_res = n_flip = 0
    for gi, g in enumerate(games):
        models = (g.get("metadata") or {}).get("models") or []
        if not (models and POLICY_MARKER in str(models[0])
                and all(OPP_MARKER in str(m) for m in models[1:])):
            continue
        for ni, node in enumerate(iter_treasure_nodes(g)):
            if 0 not in node["bids"]:
                continue
            opp_seats = [s for s in node["bids"] if s != 0]
            keys = [(gi, ni, s) for s in opp_seats]
            if any(k not in preds for k in keys):
                continue
            sim = {s: preds[(gi, ni, s)] for s in opp_seats}
            sim[0] = node["bids"][0]
            n_res += 1
            actual_win = node["winner_id"] == 0
            if (predicted_winner(sim, node["tiebreak"]) == 0) != actual_win:
                n_flip += 1
    return {"n_auctions": n_res, "n_flips": n_flip,
            "flip_rate": round(n_flip / n_res, 4) if n_res else None}


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Input corpus = logged games in gitignored results/ scratch (regenerable).
    ap.add_argument("--traj-globs", nargs="+", default=[
        "results/**/*.json", "src/megagem/evals/results/*.json"])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/flash_bid_model/fit.json")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    games = load_traj_games(args.traj_globs)
    rows = build_rows(games)
    print(f"loaded {len(games)} games → {len(rows)} Flash treasure-bid rows")
    if len(rows) < 500:
        print("too few rows — aborting")
        return 1

    # Baselines need no CV (no fitting): use the feature values directly.
    base_preds = {
        "market (current rollout)": {r["node_key"]: int(r["features"]["market_pred"]) for r in rows},
        "fair_value": {r["node_key"]: int(r["features"]["fair_value_pred"]) for r in rows},
    }
    med = int(statistics.median(r["label"] for r in rows))
    base_preds[f"global_median({med})"] = {
        r["node_key"]: min(med, r["coins"]) for r in rows}

    learned = {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "gbt": lambda: HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=args.seed),
    }

    results = {}
    print("\n" + "=" * 78)
    print(f"OOF bid prediction ({args.n_folds}-fold grouped by game) + resolution flip rate")
    print("=" * 78)
    for name, preds in base_preds.items():
        sc = score_preds(rows, preds)
        fr = flip_rate_with_preds(games, preds)
        results[name] = {"score": sc, "flip": fr, "kind": "baseline"}
        print(f"  {name:<26} MAE={sc['mae']:>6}  bias={sc['bias_mean']:>7}  "
              f"w1={sc['within1']:.0%} w2={sc['within2']:.0%}  flip={fr['flip_rate']}")
    for name, factory in learned.items():
        preds = oof_predict(rows, factory, n_folds=args.n_folds)
        sc = score_preds(rows, preds)
        fr = flip_rate_with_preds(games, preds)
        results[name] = {"score": sc, "flip": fr, "kind": "learned"}
        print(f"  {name:<26} MAE={sc['mae']:>6}  bias={sc['bias_mean']:>7}  "
              f"w1={sc['within1']:.0%} w2={sc['within2']:.0%}  flip={fr['flip_rate']}")

    best_name = min((k for k, v in results.items() if v["kind"] == "learned"),
                    key=lambda k: results[k]["score"]["mae"])
    best = results[best_name]
    passed = (best["score"]["mae"] <= GATE_MAX_MAE
              and (best["flip"]["flip_rate"] or 1.0) <= GATE_MAX_FLIP)
    print("\n" + "=" * 78)
    print(f"PRE-REGISTERED GATE (best learned = {best_name}): "
          f"MAE {best['score']['mae']} (need <= {GATE_MAX_MAE})  "
          f"flip {best['flip']['flip_rate']} (need <= {GATE_MAX_FLIP})")
    if passed:
        print("GATE PASS → a contextual point-prediction opponent is sharp enough; "
              "wire it into the Q rollout and rerun a small paired sweep.")
    else:
        print("GATE FAIL → STOP: public-state point prediction cannot reach the "
              "~1-2 coin decision margins. Do not spend more held-out budget on "
              "the current bid-search operator; prefer distribution-over-bids "
              "opponents or the supervised value-head direction.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_games": len(games), "n_rows": len(rows),
        "gate": {"max_mae": GATE_MAX_MAE, "max_flip": GATE_MAX_FLIP,
                 "best_model": best_name, "passed": passed},
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
