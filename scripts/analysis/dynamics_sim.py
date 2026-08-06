#!/usr/bin/env python3
"""Dynamics-simulating replay gate ($0, CPU) — the E1 post-mortem's mandated
GO gate: full-GAME simulation on the real engine, so the channels that killed
the live run are priced (budget/financing cascade, market-anchor feedback),
not frozen out as in per-decision replay.

Construction
------------
- REAL engine loop (mirror of pikl_search._heuristic_rollout): draw_auction_card
  -> bids -> resolve_auction -> default reveal -> apply_auction_outcome -> auto
  mission phase -> _maybe_end_game -> determine_winner. Budgets, loans,
  investments, tiebreaks, missions: mechanically exact.
- Flash seats: stochastic fitted law — mu from the Flash artifact
  (ev_dist_v1) + empirical OOF residual draw, clamped legal. Because mu
  conditions on the treasure-history features, the MARKET-ANCHOR FEEDBACK is
  reproduced by construction (win high early => their later bids rise).
- Seat 0 control: the blueprint's fitted law (ev_dist_bp_v1), sampled the same
  way (the live presample's proxy). Loans/investments for every seat: empirical
  samplers from the logged games, keyed (type, amount, coins bucket).
- Seat 0 treatment: presample b_bp from the SAME stream as control (CRN), then
  apply the selector exactly as live, with the fix knobs:
      EV(b) = (V_hat - delta_bias - (1 + lam) * b) * p_hat(win | b)
  lam = budget-pacing shadow price per coin; delta_bias = V-hat de-bias;
  deviate iff EV(b*) - EV(b_bp) >= gate_min and b* != b_bp. V-hat is the REAL
  value head on the sim state (as live).

Obligation 1 — CALIBRATION (run first): with (lam=0, delta=0, gate=1.0) the sim
must retrodict E1's measured signature: paired margin Delta in [-8, -1], ledger
signs (spend +, investment_returns -, loan_payments +), early-market price +.
Only a calibrated sim may adjudicate fixes.

Obligation 2 — FIX SWEEP: grid over (lam, delta, gate); report paired Delta +
ledgers; the winning config becomes the pre-registered stage-2 candidate.

Usage:
  python3 scripts/analysis/dynamics_sim.py --mode calibrate --num-seeds 300
  python3 scripts/analysis/dynamics_sim.py --mode sweep --num-seeds 200
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from megagem.assets import asset_path
from megagem.environment.bid_model import EvDistModel, beats_tie, node_from_live, win_curve
from megagem.environment.multi_agent_env import MegaGemEnv
from megagem.game.actions import get_default_reveal, validate_bid_for_auction
from megagem.game.cards import AuctionType
from megagem.game.rules import apply_auction_outcome, determine_winner, resolve_auction, reveal_gem_from_hand
from megagem.value_head.value_estimator import ValueEstimator

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("dist_ev_adjudicator", _here / "dist_ev_adjudicator.py")
_adj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adj)

FLASH_MODEL = str(asset_path("ev_dist_v1.pkl"))
BP_MODEL = str(asset_path("ev_dist_bp_v1.pkl"))
VALUE_HEAD = str(asset_path("value_head.pkl"))
# Archived E1 game logs live in gitignored results/ scratch (regenerable).
E1_LOG_GLOBS = (
    "results/ev_gate_e1stage1/games_off/**/*.json",
    "results/ev_gate_e1stage1/games_ev/**/*.json",
    "results/ev_gate_e1stage1/games_ev_hist/**/*.json",
    "results/ev_gate_e1stage2/games_off/**/*.json",
    "results/ev_gate_e1stage2/games_ev/**/*.json",
)


# --------------------------------------------------------------------------- #
# empirical loan/investment samplers from the logged games                     #
# --------------------------------------------------------------------------- #
def coins_bucket(c: int) -> int:
    return 0 if c < 10 else (1 if c < 20 else 2)


def build_special_samplers() -> dict:
    """policy ('bp'|'flash') -> {(type, amount, bucket) | (type, amount) | (type,)}
    -> list of logged bids."""
    # Pin the corpus to the archived E1 runs.  A broad results/**/*.json scan
    # is non-reproducible once later experiments append unrelated reports.
    games = _adj.load_games(list(E1_LOG_GLOBS))
    out = {"bp": defaultdict(list), "flash": defaultdict(list)}
    for g in games:
        for r in g.get("rounds") or []:
            au = r.get("auction") or {}
            a_type = (au.get("type") or "").lower()
            if a_type not in ("loan", "investment"):
                continue
            amount = int((au.get("loan_amount") if a_type == "loan" else au.get("bonus")) or 0)
            for p in r.get("players") or []:
                pid = p.get("player_id")
                if pid is None:
                    continue
                pol = "bp" if int(pid) == 0 else "flash"
                b = int(p.get("bid", 0) or 0)
                cb = coins_bucket(int(p.get("coins_before", 0) or 0))
                for key in ((a_type, amount, cb), (a_type, amount), (a_type,)):
                    out[pol][key].append(b)
    return out


def sample_special(samplers, pol, a_type, amount, coins, rng) -> int:
    s = samplers[pol]
    for key in ((a_type, amount, coins_bucket(coins)), (a_type, amount), (a_type,)):
        pool = s.get(key)
        if pool and len(pool) >= 10:
            return int(rng.choice(pool))
    return 0


# --------------------------------------------------------------------------- #
# the simulator                                                                #
# --------------------------------------------------------------------------- #
class Sim:
    def __init__(self):
        self.flash = EvDistModel.load(FLASH_MODEL)
        self.bp = EvDistModel.load(BP_MODEL)
        self.est = ValueEstimator.load(VALUE_HEAD)
        self.special = build_special_samplers()
        self.gem_refit = None     # V2(b): loaded on demand by --mode v2
                                  # (requires an artifact from a retired analysis; not shipped)
        self.p_grab = 1.0 / 3.0   # V2(a) mission-OV acquisition rate

    # ---- V2 value input: gem refit g(x) + mission option value -------------
    def _vhat_v2(self, gs, gems, *, with_ov: bool) -> float:
        from collections import Counter
        from megagem.environment.mission_ov import mission_option_value
        mv = self._marginal(gs, 0, gems)
        disp = gs.get_value_display_counts()
        hand = list(gs.players[0].hand)
        coll = list(gs.players[0].collection)
        tre_seen = sum(1 for r in gs.auction_history
                       if (r.auction_card or {}).get("type") == "treasure") + 1
        treasures_remaining = max(0, 17 - tre_seen)
        lot = list(gems)
        hc, lc = Counter(hand), Counter(lot)
        unseen = {c: max(0, 6 - int(disp.get(c, 0)) - hc.get(c, 0) - lc.get(c, 0))
                  for c in set(lot)}
        feats = [
            float(mv["gem_value"]), float(len(lot)), float(gs.round_number),
            float(sum(hc.get(c, 0) for c in set(lot))),
            float(sum(unseen.values())),
            float(min(unseen.values(), default=0)),
            float(treasures_remaining),
            float(sum(int(disp.get(c, 0)) for c in set(lot))),
            float(sum(coll.count(c) for c in set(lot))),
        ]
        import numpy as _np
        ghat = float(self.gem_refit.predict(_np.array([feats]))[0])
        v = float(mv["gem_value"]) + ghat
        if with_ov:
            v += mission_option_value(
                gs.available_missions, coll, lot, disp, hand,
                treasures_remaining,
                completed_ids=set(gs.players[0].completed_missions),
                p_grab=self.p_grab)
        else:
            v += float(mv["mission_bonus"])
        return v

    def _marginal(self, gs, seat, gems) -> dict:
        from collections import Counter
        coll_all = Counter()
        for p in gs.players:
            for c, k in p.get_collection_counts().items():
                coll_all[c] += k
        return self.est.marginal_value(
            gems=list(gems),
            seat_collection_counts=gs.players[seat].get_collection_counts(),
            available_mission_ids=[m.id for m in gs.available_missions],
            display_counts=gs.get_value_display_counts(),
            own_hand_counts=Counter(gs.players[seat].hand),
            collection_counts_all=coll_all, round_number=gs.round_number,
            chart_id=gs.value_chart.id)

    # fitted-law treasure sample: mu(node, seat) + residual draw, clamped legal
    def _law_sample(self, model: EvDistModel, node, seat, chart, rng) -> int:
        mu = model.mu(node, seat, chart)
        b = int(round(mu + float(rng.choice(model.residuals))))
        return max(0, min(int(node["coins"].get(seat, 0)), b))

    def _vhat(self, gs, seat) -> float:
        from collections import Counter
        auction = gs.current_auction
        display_counts = gs.get_value_display_counts()
        own = Counter(gs.players[seat].hand)
        coll_all = Counter()
        for p in gs.players:
            for c, k in p.get_collection_counts().items():
                coll_all[c] += k
        gems = list(gs.revealed_gems[: auction.gems])
        mv = self.est.marginal_value(
            gems=gems, seat_collection_counts=gs.players[seat].get_collection_counts(),
            available_mission_ids=[m.id for m in gs.available_missions],
            display_counts=display_counts, own_hand_counts=own,
            collection_counts_all=coll_all, round_number=gs.round_number,
            chart_id=gs.value_chart.id)
        return float(mv["gem_value"]) + float(mv["mission_bonus"])

    def play(self, seed: int, *, treatment: bool, lam: float = 0.0,
             delta: float = 0.0, gate_min: float = 1.0,
             value_mode: str = "current", lam_schedule: dict | None = None) -> dict:
        env = MegaGemEnv(num_players=3, value_chart_id="A")
        gs = env.create_game_state(seed=seed)
        # CRN streams: identical across arms (treatment's pass-through == control)
        rng_bp = random.Random((seed * 1_000_003) ^ 0xB1D)
        rng_opp = random.Random((seed * 1_000_003) ^ 0x0FF)
        rng_sp = random.Random((seed * 1_000_003) ^ 0x5BE)
        spend = won = n_dev = 0
        prices_thirds = defaultdict(list)
        node_i = 0
        while not gs.is_game_over():
            auction = gs.draw_auction_card()
            if auction is None:
                break
            chart = gs.value_chart
            if auction.type == AuctionType.TREASURE and gs.revealed_gems:
                node = node_from_live(gs)
                b0 = self._law_sample(self.bp, node, 0, chart, rng_bp)
                mu1 = self.flash.mu(node, 1, chart)
                mu2 = self.flash.mu(node, 2, chart)
                b1 = max(0, min(int(node["coins"][1]), int(round(mu1 + float(rng_opp.choice(self.flash.residuals))))))
                b2 = max(0, min(int(node["coins"][2]), int(round(mu2 + float(rng_opp.choice(self.flash.residuals))))))
                if treatment:
                    # selector with fix knobs; p-hat from the SAME flash mus (sealed)
                    spec = [(mu1, int(node["coins"][1]), beats_tie(node["tiebreak"], 0, 1)),
                            (mu2, int(node["coins"][2]), beats_tie(node["tiebreak"], 0, 2))]
                    pwin = win_curve(spec, self.flash.residuals, int(node["coins"][0]))
                    if value_mode == "current":
                        vhat = self._vhat(gs, 0) - delta
                    else:   # "gem" = refit only; "v2" = refit + mission OV
                        vhat = self._vhat_v2(gs, gs.revealed_gems[: auction.gems],
                                             with_ov=(value_mode == "v2"))
                    lam_eff = lam
                    if lam_schedule is not None:
                        from megagem.environment.pacing import pacing_lambda
                        lam_eff = pacing_lambda(
                            lam_schedule,
                            auctions_resolved=len(gs.auction_history),
                            coins=int(gs.players[0].coins), flat=lam)
                    b_grid = np.arange(len(pwin), dtype=float)
                    ev = (vhat - (1.0 + lam_eff) * b_grid) * pwin
                    b_star = int(np.argmax(ev))
                    if (ev[b_star] - ev[min(b0, len(ev) - 1)]) >= gate_min and b_star != b0:
                        b0 = b_star
                        n_dev += 1
                bids = [b0, b1, b2]
            else:
                amount = int(getattr(auction, "amount", 0) or getattr(auction, "bonus", 0) or 0)
                a_type = "loan" if auction.type == AuctionType.LOAN else "investment"
                bids = []
                for s, (pol, rng) in enumerate((("bp", rng_sp), ("flash", rng_sp), ("flash", rng_sp))):
                    b = sample_special(self.special, pol, a_type, amount, gs.players[s].coins, rng)
                    ok, _ = validate_bid_for_auction(gs, s, b)
                    bids.append(b if ok else 0)
            outcome = resolve_auction(gs, bids)
            if auction.type == AuctionType.TREASURE:
                if outcome.winner_id == 0:
                    spend += bids[0]
                    won += 1
                third = min(2, (3 * node_i) // 17)
                prices_thirds[third].append(bids[outcome.winner_id])
                node_i += 1
                gem = get_default_reveal(gs.players[outcome.winner_id].hand)
                if gem:
                    reveal_gem_from_hand(gs, outcome.winner_id, gem)
                apply_auction_outcome(gs, outcome, bids, gem)
                if gs.available_missions:
                    env.run_mission_phase(gs, outcome.winner_id)
            else:
                apply_auction_outcome(gs, outcome, bids, None)
            env._maybe_end_game(gs)
        _w, scores = determine_winner(gs)
        me = next(s for s in scores if int(s["player_id"]) == 0)
        opp = [float(s["final_score"]) for s in scores if int(s["player_id"]) != 0]
        return {"score": float(me["final_score"]),
                "margin": float(me["final_score"]) - max(opp),
                "coins": float(me.get("coins", 0) or 0),
                "gem_value": float(me.get("gem_value", 0) or 0),
                "missions": float(me.get("mission_rewards", 0) or 0),
                "loans": float(me.get("loan_payments", 0) or 0),
                "invest": float(me.get("investment_returns", 0) or 0),
                "spend": spend, "won": won, "n_dev": n_dev,
                "prices": {t: statistics.fmean(v) for t, v in prices_thirds.items() if v}}


def paired_eval(sim: Sim, seeds, *, lam, delta, gate_min,
                controls: dict[int, dict] | None = None) -> dict:
    """Evaluate a selector arm against CRN controls.

    ``controls`` lets a grid reuse its identical control trajectory for every
    arm.  It is an execution-only optimization: the seed-wise paired values
    are unchanged.
    """
    rows = []
    for sd in seeds:
        c = controls[sd] if controls is not None else sim.play(sd, treatment=False)
        t = sim.play(sd, treatment=True, lam=lam, delta=delta, gate_min=gate_min)
        rows.append((c, t))
    def d(key):
        v = np.array([t[key] - c[key] for c, t in rows])
        return {"mean": round(float(v.mean()), 2),
                "se": round(float(v.std(ddof=1) / np.sqrt(len(v))), 2)}
    dm = np.array([t["margin"] - c["margin"] for c, t in rows])
    price_d = {}
    for third in (0, 1, 2):
        v = [t["prices"].get(third) - c["prices"].get(third) for c, t in rows
             if third in t["prices"] and third in c["prices"]]
        if v:
            price_d[f"third_{third}"] = round(float(np.mean(v)), 2)
    return {"lam": lam, "delta": delta, "gate_min": gate_min, "n": len(rows),
            "paired_margin": {"mean": round(float(dm.mean()), 2),
                              "se": round(float(dm.std(ddof=1) / np.sqrt(len(dm))), 2)},
            "ledger": {k: d(k) for k in ("spend", "won", "coins", "gem_value",
                                          "missions", "loans", "invest", "score")},
            "dev_per_game": round(float(np.mean([t["n_dev"] for _, t in rows])), 2),
            "price_feedback": price_d}


def v2_eval(sim: Sim, seeds) -> dict:
    """V2 adjudication: three SELECTOR arms paired per seed —
    cur (lam=.5, delta=2, today's value), gem (refit replaces delta),
    v2 (refit + mission OV). Diffs reported vs cur."""
    rows = []
    for sd in seeds:
        cur = sim.play(sd, treatment=True, lam=0.5, delta=2.0, value_mode="current")
        gem = sim.play(sd, treatment=True, lam=0.5, delta=0.0, value_mode="gem")
        v2 = sim.play(sd, treatment=True, lam=0.5, delta=0.0, value_mode="v2")
        rows.append((cur, gem, v2))

    def block(idx, name):
        dm = np.array([r[idx]["margin"] - r[0]["margin"] for r in rows])
        led = {k: round(float(np.mean([r[idx][k] - r[0][k] for r in rows])), 2)
               for k in ("spend", "won", "gem_value", "missions", "loans",
                         "invest", "score", "coins")}
        return {"arm": name,
                "margin": {"mean": round(float(dm.mean()), 2),
                           "se": round(float(dm.std(ddof=1) / np.sqrt(len(dm))), 2)},
                "ledger": led,
                "dev_per_game": round(float(np.mean([r[idx]["n_dev"] for r in rows])), 2)}
    return {"n": len(rows), "dev_cur": round(float(np.mean([r[0]["n_dev"] for r in rows])), 2),
            "gem_vs_cur": block(1, "gem"), "v2_vs_cur": block(2, "v2")}


def lam_eval(sim: Sim, seeds, grid: list[tuple[str, str]]) -> dict:
    """V3: paired selector-vs-selector — each schedule arm against the current
    constant lam=0.5/delta=2 selector on CRN seeds."""
    from megagem.environment.pacing import parse_schedule
    base = {sd: sim.play(sd, treatment=True, lam=0.5, delta=2.0) for sd in seeds}
    out = {"n": len(seeds), "arms": []}
    for name, spec_s in grid:
        spec = parse_schedule(spec_s)
        rows = [(base[sd], sim.play(sd, treatment=True, lam=0.5, delta=2.0,
                                    lam_schedule=spec)) for sd in seeds]
        dm = np.array([t["margin"] - c["margin"] for c, t in rows])
        led = {k: round(float(np.mean([t[k] - c[k] for c, t in rows])), 2)
               for k in ("spend", "won", "gem_value", "missions", "loans",
                         "invest", "score", "coins")}
        arm = {"name": name, "spec": spec_s,
               "margin": {"mean": round(float(dm.mean()), 2),
                          "se": round(float(dm.std(ddof=1) / np.sqrt(len(dm))), 2)},
               "ledger": led,
               "dev_per_game": round(float(np.mean([t["n_dev"] for _, t in rows])), 2),
               "dev_base": round(float(np.mean([c["n_dev"] for c, _ in rows])), 2)}
        out["arms"].append(arm)
        print(f"  {name:14s} {spec_s:16s} Δmargin {arm['margin']['mean']:+6.2f}±{arm['margin']['se']:.2f}"
              f"  dev/g {arm['dev_per_game']:>5.2f} (base {arm['dev_base']})"
              f"  invest {led['invest']:+5.2f} spend {led['spend']:+5.2f} "
              f"loans {led['loans']:+5.2f}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("calibrate", "sweep", "v2", "lam"), default="calibrate")
    ap.add_argument("--num-seeds", type=int, default=300)
    ap.add_argument("--base-seed", type=int, default=90000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sim = Sim()
    seeds = [a.base_seed + i for i in range(a.num_seeds)]

    t0 = time.time()
    if a.mode == "lam":
        # pre-registered V3 grid (frontier doc): declining linear + pace
        # families, a flat CRN-sanity control, and a RISING falsification
        # control that must NOT win
        grid = [
            ("lin_08_-08", "linear:0.8,-0.8"),
            ("lin_10_-10", "linear:1.0,-1.0"),
            ("lin_06_-06", "linear:0.6,-0.6"),
            ("lin_075_-05", "linear:0.75,-0.5"),
            ("pace_05_1", "pace:0.5,1.0"),
            ("pace_05_2", "pace:0.5,2.0"),
            ("FLAT_SANITY", "linear:0.5,0.0"),
            ("RISING_CTRL", "linear:0.2,0.6"),
        ]
        import os
        if os.environ.get("LAM_EXT"):   # pre-registered edge extension
            grid = [
                ("lin_12_-12", "linear:1.2,-1.2"),
                ("lin_14_-14", "linear:1.4,-1.4"),
                ("lin_12_-09", "linear:1.2,-0.9"),
            ]
        print(f"V3 LAM SWEEP: {len(grid)} arms x {len(seeds)} paired seeds vs "
              f"current (lam=0.5, delta=2)")
        rep = lam_eval(sim, seeds, grid)
        dest = Path(a.out or "results/analysis/dynamics_sim_lam.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rep, indent=2))
        print(f"wrote {dest}  ({time.time() - t0:.0f}s)")
        return 0
    if a.mode == "v2":
        import pickle
        # gem_refit_g.pkl requires an artifact from a retired analysis; not shipped.
        with open("results/value_probe/gem_refit_g.pkl", "rb") as fh:
            sim.gem_refit = pickle.load(fh)["model"]
        rep = v2_eval(sim, seeds)
        print(f"V2 SIM ADJUDICATION (n={rep['n']}, {time.time() - t0:.0f}s; "
              f"selector-vs-selector, all arms lam=0.5):")
        for k in ("gem_vs_cur", "v2_vs_cur"):
            r = rep[k]
            print(f"  {r['arm']:3s} vs cur: Δmargin {r['margin']['mean']:+.2f} ± {r['margin']['se']:.2f}"
                  f"  dev/g {r['dev_per_game']} (cur {rep['dev_cur']})  ledger {r['ledger']}")
        dest = Path(a.out or "results/analysis/dynamics_sim_v2.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rep, indent=2))
        print(f"wrote {dest}")
        return 0
    if a.mode == "calibrate":
        rep = paired_eval(sim, seeds, lam=0.0, delta=0.0, gate_min=1.0)
        print(f"CALIBRATION (lam=0, delta=0, gate=1.0; n={rep['n']}; "
              f"{time.time() - t0:.0f}s):")
        print(f"  paired margin Δ = {rep['paired_margin']['mean']} ± {rep['paired_margin']['se']}"
              f"   [E1 measured: raw −5.01, CV-adj −1.88 — PASS band −8..−1]")
        print("  ledger Δ: " + "  ".join(f"{k}={v['mean']}±{v['se']}" for k, v in rep["ledger"].items()))
        print("           [E1 measured: spend +6.81, won +0.45, loans +3.73, invest −7.13, score −3.61]")
        print(f"  deviations/game {rep['dev_per_game']}  [E1: 4.27]")
        print(f"  market Δ by third: {rep['price_feedback']}  [E1: +1.48 / +0.27 / −1.04]")
        signs_ok = (rep["ledger"]["spend"]["mean"] > 0 and rep["ledger"]["invest"]["mean"] < 0
                    and rep["ledger"]["loans"]["mean"] > 0)
        band_ok = -8.0 <= rep["paired_margin"]["mean"] <= -1.0
        market_ok = rep["price_feedback"].get("third_0", 0) > 0
        verdict = "CALIBRATED — sim may adjudicate fixes" if (signs_ok and band_ok and market_ok) \
            else "NOT CALIBRATED — do not trust the sweep; fix the sim first"
        print(f"  PRE-REGISTERED CALIBRATION: {verdict}")
        out = {"mode": "calibrate", "report": rep,
               "calibrated": bool(signs_ok and band_ok and market_ok)}
    else:
        grid = []
        for lam in (0.0, 0.5, 1.0, 1.6, 2.5):
            for delta in (0.0, 2.0, 5.0):
                grid.append((lam, delta, 1.0))
        # gate-only and gate+debias variants (does a stricter gate alone kill the cascade?)
        grid += [(0.0, 0.0, 2.0), (0.0, 0.0, 4.0), (0.0, 5.0, 2.0)]
        out = {"mode": "sweep", "reports": []}
        print(f"SWEEP: {len(grid)} configs x {len(seeds)} paired seeds")
        controls = {sd: sim.play(sd, treatment=False) for sd in seeds}
        for lam, delta, gate in grid:
            rep = paired_eval(sim, seeds, lam=lam, delta=delta, gate_min=gate,
                              controls=controls)
            out["reports"].append(rep)
            print(f"  lam={lam:>3} delta={delta:>3} gate={gate}: "
                  f"Δmargin {rep['paired_margin']['mean']:+6.2f}±{rep['paired_margin']['se']:.2f}  "
                  f"dev/g {rep['dev_per_game']:>5.2f}  "
                  f"invest {rep['ledger']['invest']['mean']:+6.2f}  "
                  f"loans {rep['ledger']['loans']['mean']:+5.2f}  "
                  f"spend {rep['ledger']['spend']['mean']:+6.2f}", flush=True)
        best = max(out["reports"], key=lambda r: r["paired_margin"]["mean"])
        print(f"\nBEST: lam={best['lam']} delta={best['delta']} gate={best['gate_min']} "
              f"Δmargin {best['paired_margin']['mean']}±{best['paired_margin']['se']}")
        out["best"] = {k: best[k] for k in ("lam", "delta", "gate_min", "paired_margin")}

    dest = Path(a.out or f"results/analysis/dynamics_sim_{a.mode}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
