#!/usr/bin/env python3
"""Why didn't the piKL bid operator's local Q lift transfer to held-out wins?

Four $0/CPU probes on existing artifacts (no new API spend):

A. ROLLOUT-OPPONENT MISSPECIFICATION (the mechanism test). The Q rollout resolves
   every counterfactual auction with the market-anchored bid model (offline-fit on
   SFT *teacher* bids; treasure MAE 2.4 there). The held-out opponent is Gemini
   Flash. From saved trajectories that contain Flash seats we predict Flash's
   actual treasure bids with BOTH rollout models (market, fair_value) and report
   MAE / signed bias, plus the decisive number: the AUCTION-RESOLUTION FLIP RATE —
   for the policy seat's actual bid, how often would replacing the real opponent
   bids with model-predicted bids flip the policy's win/lose? Every flip is an
   auction the Q rollout scores on the wrong side.

B. LIFT→OUTCOME CALIBRATION. Per seed, the treatment arm's total predicted value
   lift (sum of chosen_lift_vs_tau; tanh-utility units) vs the realized paired
   Δ-gain vs the 'off' control. If the correlation is ~0, the local lift carries
   no held-out outcome information.

C. DEVIATION DIRECTION. chosen − tau_mode over the gated decisions: does piKL
   systematically over/under-bid the blueprint when it deviates?

D. NOISE DOMINANCE. Cross-arm correlation of per-seed paired gains. The arms share
   seeds; if their gains are uncorrelated, seed-level noise dominates any
   systematic treatment effect.

Usage
-----
  python scripts/analysis/pikl_transfer_probe.py \
      --traj-globs 'results/panel_eval_rl_repl03_step200_full_panel/vs_flash/seat*/*.json' \
                   'src/megagem/evals/results/*.json' \
      --baseline /tmp/gate_b_120/.../gate_b_gemini-3-flash_market_off.json \
      --treatment /tmp/gate_b_120/.../gate_b_gemini-3-flash_market_0.json \
      --treatment /tmp/gate_b_120/.../gate_b_gemini-3-flash_market_0.03.json \
      --out results/pikl_transfer_probe/probe.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path

from megagem.data import load_value_charts
from megagem.environment.pikl_search import MARKET_DEFAULTS
from megagem.game.cards import ValueChart

_AGG = {"mean": statistics.fmean, "median": statistics.median,
        "max": max, "last": lambda xs: xs[-1]}


# --------------------------------------------------------------------------- #
# trajectory reconstruction (Probe A)                                          #
# --------------------------------------------------------------------------- #
def iter_treasure_nodes(game: dict):
    """Yield one node per treasure auction with the PUBLIC state the rollout
    bid models consume, reconstructed exactly as the live game would have it:
    history = treasure auctions RESOLVED BEFORE this one (gem count + winning
    bid), tiebreak order in effect during the auction, per-seat coins-before."""
    rounds = game.get("rounds") or []
    history: list[tuple[int, int]] = []  # (gems_count, winning_bid), chronological
    for r in rounds:
        au = r.get("auction") or {}
        a_type = (au.get("type") or "").lower()
        players = r.get("players") or []
        bids, coins, winner_id = {}, {}, None
        for p in players:
            pid = p.get("player_id")
            if pid is None:
                continue
            pid = int(pid)
            bids[pid] = int(p.get("bid", 0) or 0)
            coins[pid] = int(p.get("coins_before", 0) or 0)
            if p.get("is_winner"):
                winner_id = pid
        if a_type == "treasure":
            gems = au.get("gems_available") or []
            disp = {}
            vd = r.get("value_display") or {}
            if isinstance(vd, dict):
                for c, info in vd.items():
                    disp[c] = int((info or {}).get("count", 0) or 0) if isinstance(info, dict) else int(info or 0)
            if gems and len(bids) >= 2:
                yield {
                    "round": r.get("round_number"),
                    "gems": list(gems),
                    "gems_count": len(gems),
                    "bids": bids,
                    "coins": coins,
                    "tiebreak": list(r.get("tiebreak_order_before") or r.get("tiebreak_order") or []),
                    "history": list(history),
                    "display": disp,
                    "winner_id": winner_id,
                }
            if winner_id is not None:
                history.append((len(gems), bids.get(winner_id, 0)))


def market_pred(node: dict, seat: int, *, f=1.0, window=5, agg="median",
                match_gems=True, f_floor=0.25, tiebreak_delta=1) -> int:
    """market_anchored_bid recomputed from a trajectory node (public info only).
    Mirrors src/megagem/environment/pikl_search.py::market_anchored_bid byte-for-byte in
    semantics: anchor to recent same-gem-count winning bids, cold-start to
    f_floor·coins, +tiebreak_delta when last in tiebreak, clamp [0, coins]."""
    coins = node["coins"].get(seat, 0)
    hist = node["history"][-window:]
    A = [wb for (gc, wb) in hist if (not match_gems) or gc == node["gems_count"]]
    pred = (f_floor * coins) if not A else (f * _AGG[agg](A))
    tb = node["tiebreak"]
    if tiebreak_delta and tb and tb[-1] == seat:
        pred += tiebreak_delta
    return max(0, min(coins, round(pred)))


def fair_value_pred(node: dict, seat: int, chart: ValueChart, shade=0.8) -> int:
    """fair_value_bid for an OPPONENT seat (no private-hand peek): shade × sum of
    the auctioned colours' display value, clamped to coins."""
    value = sum(chart.get_gem_value(node["display"].get(c, 0)) for c in node["gems"])
    return max(0, min(node["coins"].get(seat, 0), round(shade * value)))


def predicted_winner(bids: dict[int, int], tiebreak: list[int]) -> int:
    """Mirror resolve_auction: max bid, ties go to the EARLIEST in tiebreak."""
    hi = max(bids.values())
    tied = [s for s, b in bids.items() if b == hi]
    if len(tied) == 1 or not tiebreak:
        return min(tied)
    return min(tied, key=lambda s: tiebreak.index(s) if s in tiebreak else len(tiebreak))


def probe_a(games: list[dict], *, policy_marker="qwen", opp_marker="gemini-3-flash") -> dict:
    """Bid-prediction error of both rollout opponent models on the REAL opponent's
    bids, and the auction-resolution flip rate for the policy seat."""
    charts = {cid: ValueChart.from_dict(cid, d) for cid, d in load_value_charts().items()}
    err_mkt, err_fv = [], []
    n_flip_mkt = n_flip_fv = n_res = 0
    gateb_like = 0
    by_gems_err: dict[int, list[float]] = {1: [], 2: []}
    for g in games:
        models = (g.get("metadata") or {}).get("models") or []
        chart = charts.get((g.get("metadata") or {}).get("value_chart") or "A", charts["A"])
        flash_seats = [i for i, m in enumerate(models) if opp_marker in str(m)]
        policy_seat = 0 if models and policy_marker in str(models[0]) else None
        is_gateb_like = (
            policy_seat == 0
            and all(opp_marker in str(m) for m in models[1:])
        )
        gateb_like += 1 if is_gateb_like else 0
        for node in iter_treasure_nodes(g):
            for s in flash_seats:
                if s not in node["bids"]:
                    continue
                actual = node["bids"][s]
                e_m = market_pred(node, s, **MARKET_DEFAULTS) - actual
                e_f = fair_value_pred(node, s, chart) - actual
                err_mkt.append(e_m)
                err_fv.append(e_f)
                by_gems_err.setdefault(node["gems_count"], []).append(e_m)
            # flip rate: only in the gate-B-like setting (policy seat 0 vs all-Flash)
            if not is_gateb_like or 0 not in node["bids"]:
                continue
            n_res += 1
            actual_win = node["winner_id"] == 0
            for pred_fn, counter in ((market_pred, "mkt"), (fair_value_pred, "fv")):
                if pred_fn is market_pred:
                    sim = {s: market_pred(node, s, **MARKET_DEFAULTS) for s in node["bids"] if s != 0}
                else:
                    sim = {s: fair_value_pred(node, s, chart) for s in node["bids"] if s != 0}
                sim[0] = node["bids"][0]  # the policy's ACTUAL bid, opponents simulated
                pred_win = predicted_winner(sim, node["tiebreak"]) == 0
                if pred_win != actual_win:
                    if counter == "mkt":
                        n_flip_mkt += 1
                    else:
                        n_flip_fv += 1

    def _stats(errs):
        if not errs:
            return {}
        ae = [abs(e) for e in errs]
        return {
            "n": len(errs),
            "mae": round(statistics.fmean(ae), 3),
            "bias_mean": round(statistics.fmean(errs), 3),  # >0 ⇒ model OVER-predicts opp bid
            "bias_median": round(statistics.median(errs), 3),
            "frac_under": round(sum(1 for e in errs if e < 0) / len(errs), 4),
            "frac_over": round(sum(1 for e in errs if e > 0) / len(errs), 4),
        }

    return {
        "n_games": len(games),
        "n_gateb_like_games": gateb_like,
        "market": _stats(err_mkt),
        "fair_value": _stats(err_fv),
        "market_mae_by_gems": {k: round(statistics.fmean(abs(e) for e in v), 3)
                               for k, v in by_gems_err.items() if v},
        "resolution": {
            "n_auctions": n_res,
            "flip_rate_market": round(n_flip_mkt / n_res, 4) if n_res else None,
            "flip_rate_fair_value": round(n_flip_fv / n_res, 4) if n_res else None,
        },
    }


# --------------------------------------------------------------------------- #
# probes B–D on the paired arm files                                            #
# --------------------------------------------------------------------------- #
def _scored_by_seed(arm: dict) -> dict[int, dict]:
    return {int(g["seed"]): g for g in arm.get("per_game") or []
            if "error" not in g and "seed" in g}


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    return pearson(_ranks(xs), _ranks(ys))


def per_seed_exposure(game: dict) -> dict:
    """Predicted-lift exposure of one treatment game: total/chosen lift, passed
    gates, and deviation count (chosen != tau_mode)."""
    lift_all = lift_passed = 0.0
    n_pass = n_gate = n_dev = n_dec = 0
    for d in game.get("pikl_decisions") or []:
        m = d.get("metrics") or {}
        n_dec += 1
        cl = m.get("chosen_lift_vs_tau")
        if isinstance(cl, (int, float)):
            lift_all += float(cl)
        gate = d.get("gate")
        if gate is not None:
            n_gate += 1
            if gate.get("passed"):
                n_pass += 1
                if isinstance(cl, (int, float)):
                    lift_passed += float(cl)
        ch, tm = m.get("chosen"), m.get("tau_mode")
        if ch is not None and tm is not None and str(ch) != str(tm):
            n_dev += 1
    return {"lift_all": lift_all, "lift_passed": lift_passed, "n_pass": n_pass,
            "n_gate": n_gate, "n_dev": n_dev, "n_dec": n_dec}


def probe_b(base: dict, treat: dict) -> dict:
    """Calibration: per-seed predicted lift vs realized paired Δ-gain."""
    b, t = _scored_by_seed(base), _scored_by_seed(treat)
    seeds = sorted(set(b) & set(t))
    xs_lift, xs_pass, ys = [], [], []
    rows = []
    for s in seeds:
        exp = per_seed_exposure(t[s])
        gain = float(t[s]["policy_delta"]) - float(b[s]["policy_delta"])
        xs_lift.append(exp["lift_passed"])
        xs_pass.append(float(exp["n_pass"]))
        ys.append(gain)
        rows.append((exp["lift_passed"], gain))
    # exposure terciles by passed-gate lift
    rows.sort(key=lambda r: r[0])
    k = len(rows) // 3
    terciles = {}
    if k:
        for name, chunk in (("low", rows[:k]), ("mid", rows[k:2 * k]), ("high", rows[2 * k:])):
            terciles[name] = {
                "mean_lift": round(statistics.fmean(r[0] for r in chunk), 3),
                "mean_paired_gain": round(statistics.fmean(r[1] for r in chunk), 3),
                "n": len(chunk),
            }
    return {
        "n_paired": len(seeds),
        "lift_vs_gain_pearson": round(pearson(xs_lift, ys), 4) if pearson(xs_lift, ys) is not None else None,
        "lift_vs_gain_spearman": round(spearman(xs_lift, ys), 4) if spearman(xs_lift, ys) is not None else None,
        "npass_vs_gain_spearman": round(spearman(xs_pass, ys), 4) if spearman(xs_pass, ys) is not None else None,
        "mean_lift_passed_per_game": round(statistics.fmean(xs_lift), 3) if xs_lift else None,
        "terciles_by_passed_lift": terciles,
    }


def probe_c(treat: dict) -> dict:
    """Deviation direction: chosen − tau_mode on gated decisions."""
    diffs_all, diffs_passed = [], []
    for g in treat.get("per_game") or []:
        for d in g.get("pikl_decisions") or []:
            m = d.get("metrics") or {}
            try:
                diff = float(m["chosen"]) - float(m["tau_mode"])
            except (KeyError, TypeError, ValueError):
                continue
            diffs_all.append(diff)
            if (d.get("gate") or {}).get("passed"):
                diffs_passed.append(diff)

    def _dir(ds):
        if not ds:
            return {}
        return {
            "n": len(ds),
            "mean_diff": round(statistics.fmean(ds), 3),
            "median_diff": round(statistics.median(ds), 3),
            "frac_lower": round(sum(1 for x in ds if x < 0) / len(ds), 4),
            "frac_equal": round(sum(1 for x in ds if x == 0) / len(ds), 4),
            "frac_higher": round(sum(1 for x in ds if x > 0) / len(ds), 4),
        }

    return {"all_gated": _dir(diffs_all), "passed_gate": _dir(diffs_passed)}


def probe_d(base: dict, treats: dict[str, dict]) -> dict:
    """Cross-arm per-seed gain correlation (shared seeds ⇒ shared deals)."""
    b = _scored_by_seed(base)
    gains = {}
    for label, t in treats.items():
        tt = _scored_by_seed(t)
        seeds = sorted(set(b) & set(tt))
        gains[label] = {s: float(tt[s]["policy_delta"]) - float(b[s]["policy_delta"])
                        for s in seeds}
    out = {}
    labels = sorted(gains)
    for i, a in enumerate(labels):
        for bl in labels[i + 1:]:
            common = sorted(set(gains[a]) & set(gains[bl]))
            xs = [gains[a][s] for s in common]
            ys = [gains[bl][s] for s in common]
            out[f"{a}_vs_{bl}"] = {
                "n": len(common),
                "pearson": round(pearson(xs, ys), 4) if pearson(xs, ys) is not None else None,
                "spearman": round(spearman(xs, ys), 4) if spearman(xs, ys) is not None else None,
            }
    return out


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def load_traj_games(globs: list[str]) -> list[dict]:
    seen, games = set(), []
    for pat in globs:
        for f in glob.glob(pat, recursive=True):
            if f in seen:
                continue
            seen.add(f)
            try:
                d = json.load(open(f))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(d, dict):
                continue
            md = d.get("metadata") or {}
            if not (d.get("rounds") and isinstance(md.get("models"), list)):
                continue
            if any("gemini-3-flash" in str(m) for m in md["models"]):
                games.append(d)
    return games


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Input corpus = logged games in gitignored results/ scratch (regenerable).
    ap.add_argument("--traj-globs", nargs="+", default=[
        "results/**/*.json", "src/megagem/evals/results/*.json"])
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--treatment", required=True, action="append")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    games = load_traj_games(args.traj_globs)
    print(f"loaded {len(games)} trajectories with Gemini Flash seats")
    a = probe_a(games)

    base = json.loads(Path(args.baseline).read_text())
    treats = {}
    for tp in args.treatment:
        t = json.loads(Path(tp).read_text())
        pk = (t.get("config") or {}).get("pikl") or {}
        label = f"mix{pk.get('lambda_mix')}" if pk.get("lambda_mix") else f"lambda={pk.get('lambda')}"
        treats[label] = t

    print("\n" + "=" * 78)
    print("A. ROLLOUT-OPPONENT MODEL vs ACTUAL GEMINI FLASH BIDS (treasure auctions)")
    print("=" * 78)
    for name in ("market", "fair_value"):
        st = a[name]
        if st:
            print(f"  {name:<11} MAE={st['mae']:>6}  bias(mean/med)={st['bias_mean']:>7}/{st['bias_median']:>5}  "
                  f"under={st['frac_under']:.0%} over={st['frac_over']:.0%}  (n={st['n']})")
    print(f"  market MAE by lot size: {a['market_mae_by_gems']}   "
          f"[reference: MAE 2.4 on the SFT teacher bids it was fit on]")
    res = a["resolution"]
    print(f"  RESOLUTION FLIP RATE (policy's actual bid, simulated opp bids): "
          f"market={res['flip_rate_market']}  fair_value={res['flip_rate_fair_value']}  "
          f"(n={res['n_auctions']} auctions, {a['n_gateb_like_games']} gate-B-like games)")

    print("\n" + "=" * 78)
    print("B. PREDICTED LIFT vs REALIZED PAIRED GAIN (per seed)")
    print("=" * 78)
    b_out = {}
    for label, t in treats.items():
        b = probe_b(base, t)
        b_out[label] = b
        print(f"  {label}: corr(passed-lift, gain) pearson={b['lift_vs_gain_pearson']} "
              f"spearman={b['lift_vs_gain_spearman']}  corr(n_pass, gain)={b['npass_vs_gain_spearman']}")
        for tn, tv in (b.get("terciles_by_passed_lift") or {}).items():
            print(f"    {tn:>4} exposure: mean_lift={tv['mean_lift']:>7}  "
                  f"mean_paired_gain={tv['mean_paired_gain']:>7}  (n={tv['n']})")

    print("\n" + "=" * 78)
    print("C. DEVIATION DIRECTION (chosen − tau_mode)")
    print("=" * 78)
    c_out = {}
    for label, t in treats.items():
        c = probe_c(t)
        c_out[label] = c
        for k in ("passed_gate", "all_gated"):
            st = c[k]
            if st:
                print(f"  {label} [{k}]: mean={st['mean_diff']:>7} med={st['median_diff']:>5}  "
                      f"lower={st['frac_lower']:.0%} equal={st['frac_equal']:.0%} "
                      f"higher={st['frac_higher']:.0%}  (n={st['n']})")

    print("\n" + "=" * 78)
    print("D. CROSS-ARM PER-SEED GAIN CORRELATION (shared deals)")
    print("=" * 78)
    d_out = probe_d(base, treats)
    for k, v in d_out.items():
        print(f"  {k}: pearson={v['pearson']}  spearman={v['spearman']}  (n={v['n']})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"probe_a": a, "probe_b": b_out, "probe_c": c_out, "probe_d": d_out},
            indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
