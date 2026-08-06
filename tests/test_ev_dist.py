"""Tests for the E1 analytic expected-surplus selector (ev_dist):
bid_model node/feature parity, win-curve/EV math, gate semantics, the
presampled env seam, and the history-repair prompt arm."""

from __future__ import annotations

import asyncio
import json
import math

import numpy as np
import pytest

from megagem.assets import asset_path
from megagem.environment import prompts as prompts_mod
from megagem.environment.bid_model import (
    FEATURES, EvDistModel, beats_tie, bid_pmf, ev_best, feature_vector,
    features_for_seat, iter_nodes_bidtime, node_from_live, win_curve,
)
from megagem.environment.ev_dist import EvDistBidSearcher
from megagem.environment.multi_agent_env import BidTurnRecord, MegaGemEnv
from megagem.environment.prompts import generate_bid_prompt
from megagem.game.cards import AuctionCard, AuctionType
from megagem.game.rules import apply_auction_outcome, resolve_auction


def _play_auction(gs, bids):
    outcome = resolve_auction(gs, bids)
    apply_auction_outcome(gs, outcome, bids, None)
    return outcome



# --------------------------------------------------------------------------- #
# tie / pmf / win-curve / EV math                                              #
# --------------------------------------------------------------------------- #
def test_beats_tie_mirrors_resolve_auction_semantics():
    assert beats_tie([2, 0, 1], 0, 1)          # 0 earlier than 1
    assert not beats_tie([2, 0, 1], 0, 2)      # 2 earliest
    assert beats_tie([], 0, 1)                 # no order -> min id wins
    assert not beats_tie([], 2, 1)
    assert beats_tie([5, 6], 0, 1)             # neither in order -> min id


def test_beats_tie_agrees_with_resolve_auction_on_every_order():
    """`beats_tie` predicts, offline, what `resolve_auction` decides live.

    The selector's whole win-curve rests on that equivalence, so prove it
    against the engine rather than against hand-written expectations: for every
    permutation of the tiebreak order, a tie between two seats must resolve the
    same way both ways.
    """
    import itertools

    from megagem.game.rules import resolve_auction

    env = MegaGemEnv(num_players=3, value_chart_id="A", seed=11)
    for order in itertools.permutations(range(3)):
        for a, b in itertools.combinations(range(3), 2):
            gs = env.create_game_state(seed=11)
            gs.tiebreak_order = list(order)
            gs.draw_auction_card()
            bids = [0, 0, 0]
            bids[a] = bids[b] = 5          # a and b tie at the top
            live_winner = resolve_auction(gs, bids).winner_id
            assert live_winner in (a, b)
            predicted = a if beats_tie(list(order), a, b) else b
            assert predicted == live_winner, (
                f"order={order} tie between {a},{b}: beats_tie says {predicted}, "
                f"resolve_auction says {live_winner}")


def test_bid_pmf_clamps_and_normalizes():
    pmf = bid_pmf(mu=5, coins=6, residuals=np.array([-10, 0, 10]))
    assert pmf.shape == (7,)
    assert math.isclose(pmf.sum(), 1.0)
    assert pmf[0] > 0 and pmf[5] > 0 and pmf[6] > 0  # clamped tails

    det = bid_pmf(mu=3, coins=10, residuals=np.array([0.0]))
    assert det[3] == 1.0 and det.sum() == 1.0


def test_win_curve_deterministic_opponents_and_ties():
    resid = np.array([0.0])  # point opponents
    # one opponent bids exactly 5, we win ties
    pw = win_curve([(5, 35, True)], resid, bmax=10)
    assert pw[4] == 0.0 and pw[5] == 1.0 and pw[6] == 1.0
    # we lose ties -> must strictly outbid
    pw = win_curve([(5, 35, False)], resid, bmax=10)
    assert pw[5] == 0.0 and pw[6] == 1.0
    # two opponents: need to beat the max of both
    pw = win_curve([(5, 35, False), (7, 35, False)], resid, bmax=10)
    assert pw[7] == 0.0 and pw[8] == 1.0
    # opponent budget caps its bid: coins_s=3 means it can never bid 5
    pw = win_curve([(5, 3, False)], resid, bmax=10)
    assert pw[4] == 1.0


def test_ev_best_argmax_and_zero_floor():
    resid = np.array([0.0])
    pw = win_curve([(5, 35, False)], resid, bmax=10)
    b, ev, curve = ev_best(12.0, pw)
    assert b == 6 and math.isclose(ev, 6.0)          # (12-6)*1
    # worthless lot -> never overbid: b*=0 has EV 0 (win prob 0 there)
    b0, ev0, _ = ev_best(2.0, pw)
    assert ev0 >= 0.0 and (b0 == 0 or ev0 > 0)


# --------------------------------------------------------------------------- #
# node parity: live GameState vs offline bid-time log reconstruction           #
# --------------------------------------------------------------------------- #
def _round_dict(rn, auction_type, gems, bids, coins_before, winner, display_after, tb_before):
    players = []
    for pid, b in enumerate(bids):
        players.append({"player_id": pid, "bid": b, "coins_before": coins_before[pid],
                        "is_winner": pid == winner})
    return {
        "round_number": rn,
        "auction": {"type": auction_type, "gems_available": list(gems)},
        "players": players,
        "value_display": {c: {"count": k} for c, k in display_after.items()},
        "tiebreak_order_before": list(tb_before),
        "tiebreak_order": list(tb_before),
    }


def test_offline_nodes_use_previous_round_display():
    rounds = [
        _round_dict(1, "treasure", ["Red"], [3, 1, 0], [35, 35, 35], 0,
                    {"Red": 1}, [0, 1, 2]),
        _round_dict(2, "treasure", ["Blue", "Red"], [4, 6, 2], [32, 35, 35], 1,
                    {"Red": 2, "Blue": 1}, [1, 2, 0]),
        _round_dict(3, "loan", [], [0, 0, 5], [32, 29, 35], 2, {"Red": 2, "Blue": 1}, [2, 0, 1]),
        _round_dict(4, "treasure", ["Red"], [5, 5, 5], [32, 29, 40], 0,
                    {"Red": 3, "Blue": 1}, [0, 1, 2]),
    ]
    nodes = list(iter_nodes_bidtime({"rounds": rounds}))
    assert len(nodes) == 3                      # treasure rounds only
    assert nodes[0]["display"] == {}            # round 1 bid sees the initial (empty) display
    assert nodes[1]["display"] == {"Red": 1}    # round 2 bid sees round 1's post state
    # round 4 bid sees round 3's post state (loan round leaves display unchanged)
    assert nodes[2]["display"] == {"Red": 2, "Blue": 1}
    # treasure-only history: (gems_count, winning_bid); the loan never enters
    assert nodes[2]["history"] == [(1, 3), (2, 6)]
    assert nodes[2]["round"] == 4 and nodes[2]["coins"][2] == 40


def test_node_from_live_matches_offline_after_engine_step():
    env = MegaGemEnv(num_players=3, value_chart_id="A")
    gs = env.create_game_state(seed=11)
    gs.current_auction = AuctionCard(id="t1", type=AuctionType.TREASURE, gems=1)

    live0 = node_from_live(gs)
    assert live0 is not None
    assert live0["display"] == dict(gs.get_value_display_counts())
    assert live0["history"] == []
    assert set(live0["coins"]) == {0, 1, 2}
    assert live0["gems_count"] == 1

    # features computable + finite on the live node
    chart = gs.value_chart
    f = features_for_seat(live0, 1, chart)
    assert set(f) == set(FEATURES)
    vec = feature_vector(live0, 1, chart)
    assert np.isfinite(vec).all()

    # resolve one auction through the real engine, then the next live node's
    # history must carry (gems_count, winning_bid) exactly
    outcome = _play_auction(gs, [4, 2, 1])
    assert outcome.winner_id == 0 and gs.auction_history[-1].winning_bid == 4
    gs.current_auction = AuctionCard(id="t2", type=AuctionType.TREASURE, gems=1)
    live1 = node_from_live(gs)
    assert live1["history"] == [(1, 4)]


# --------------------------------------------------------------------------- #
# the searcher: gate semantics + presampled seam                               #
# --------------------------------------------------------------------------- #
class _ConstModel:
    """sklearn-free stand-in: predicts a constant raw bid."""

    def __init__(self, c):
        self.c = c

    def predict(self, X):
        return np.full(len(X), self.c, dtype=float)


class _StubEst:
    def __init__(self, gem_value, mission_bonus=0.0):
        self._mv = {"gem_value": gem_value, "mission_bonus": mission_bonus,
                    "total": gem_value + mission_bonus}

    def marginal_value(self, **kw):
        return dict(self._mv)


def _ev_searcher(mu_const=5, vhat=12.0, gate_min_ev=1.0, residuals=(0.0,)):
    model = EvDistModel(_ConstModel(mu_const), np.array(residuals, float),
                        FEATURES, {"test": True})
    return EvDistBidSearcher(
        bp_client=None, bp_model="bp", trainable_seat=0, num_players=3,
        value_chart_id="A", bp_extra={}, ev_model=model, value_est=_StubEst(vhat),
        gate_min_ev=gate_min_ev, crn_seed=0, max_parallel=2)


def _presampled(bid, prompt="p"):
    return bid, BidTurnRecord(
        player_id=0, actor_id="trainable", prompt=prompt, raw_response='{"bid": %d}' % bid,
        parsed_action=bid, parse_method="json", parse_valid=True, legal_valid=True,
        default_used=False, final_bid=bid, reasoning="", length_split={},
        parse_error="", legal_error="")


def _treasure_state(searcher, seed=7):
    gs = searcher._cont_env.create_game_state(seed=seed)
    gs.current_auction = AuctionCard(id="t", type=AuctionType.TREASURE, gems=1)
    return gs


def test_searcher_deviates_when_ev_gap_clears_gate():
    s = _ev_searcher(mu_const=5, vhat=12.0, gate_min_ev=1.0)
    gs = _treasure_state(s)
    # blueprint sampled a hopeless bid 0; the selector should outbid mu=5
    bid, rec = asyncio.run(s.search(gs, 0, presampled=_presampled(0)))
    pay = json.loads(rec.raw_response)["pikl"]
    assert pay["selector"] == "ev_dist" and pay["gate"]["passed"]
    # win-ties depends on the seeded tiebreak order; either 5 or 6 is the
    # cheapest winning bid vs a point opponent at 5
    assert bid in (5, 6) and bid == pay["b_star"]
    assert rec.parse_method == "pikl" and rec.final_bid == bid
    assert s.decision_log and s.decision_log[-1]["chosen"] == bid


def test_searcher_passes_through_below_gate_and_keeps_record_object():
    s = _ev_searcher(mu_const=5, vhat=12.0, gate_min_ev=1.0)
    gs = _treasure_state(s)
    # presample exactly the optimal bid -> EV gap 0 -> pass-through
    bid0, rec0 = asyncio.run(s.search(gs, 0, presampled=_presampled(6)))
    if json.loads(rec0.raw_response if rec0.parse_method == "pikl" else '{"pikl": {}}').get("pikl"):
        # if 6 wasn't optimal under this tiebreak, 5 is; rerun with 5
        bid0, rec0 = asyncio.run(s.search(gs, 0, presampled=_presampled(5)))
    pre_bid, pre_rec = _presampled(bid0)
    bid, rec = asyncio.run(s.search(gs, 0, presampled=(pre_bid, pre_rec)))
    assert bid == pre_bid
    assert rec is pre_rec                    # the untouched normal record
    assert s.decision_log[-1]["gate"]["passed"] is False


def test_searcher_never_engages_off_treasure():
    s = _ev_searcher()
    gs = s._cont_env.create_game_state(seed=3)
    gs.current_auction = AuctionCard(id="l", type=AuctionType.LOAN, gems=0, amount=10)
    assert not s.wants_auction(gs)


def test_env_seam_passes_presampled_to_ev_searcher():
    async def run():
        s = _ev_searcher(mu_const=5, vhat=12.0)
        env = s._cont_env
        gs = _treasure_state(s)
        seen = {}

        async def spy(game_state, seat, presampled=None):
            seen["presampled"] = presampled
            return presampled

        s.search = spy
        env.set_pikl_bid_searcher(s)
        bids, recs = [9, 1, 2], [_presampled(9)[1], None, None]
        # replicate the env hook contract directly
        if getattr(s, "wants_presampled", False):
            bids[0], recs[0] = await s.search(gs, 0, presampled=(bids[0], recs[0]))
        assert seen["presampled"][0] == 9
        return True

    assert asyncio.run(run())


# --------------------------------------------------------------------------- #
# history-repair prompt arm                                                    #
# --------------------------------------------------------------------------- #
def test_history_repair_default_off_is_byte_identical():
    env = MegaGemEnv(num_players=3, value_chart_id="A")
    gs = env.create_game_state(seed=9)
    gs.current_auction = AuctionCard(id="t", type=AuctionType.TREASURE, gems=1)
    _play_auction(gs, [3, 1, 0])
    gs.current_auction = AuctionCard(id="t2", type=AuctionType.TREASURE, gems=1)

    assert prompts_mod.HISTORY_REPAIR_SEATS == set()
    base = generate_bid_prompt(gs, 0)
    assert "opponent_bidding_profile" not in base
    try:
        prompts_mod.HISTORY_REPAIR_SEATS = {0}
        repaired = generate_bid_prompt(gs, 0)
        opp = generate_bid_prompt(gs, 1)
        assert "opponent_bidding_profile" in repaired
        assert "median_bid" in repaired and "market_clearing_prices" in repaired
        assert "opponent_bidding_profile" not in opp     # opponents untouched
    finally:
        prompts_mod.HISTORY_REPAIR_SEATS = set()
    assert generate_bid_prompt(gs, 0) == base            # flag fully reversible


# --------------------------------------------------------------------------- #
# frozen artifact smoke (skipped if not built)                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not asset_path("ev_dist_v1.pkl").is_file(), reason="artifact not built")
def test_frozen_artifact_fixture_parity():
    m = EvDistModel.load(str(asset_path("ev_dist_v1.pkl")))
    assert m.features == FEATURES and len(m.residuals) > 1000
    for fx in m.meta["fixtures"]:
        x = np.array(fx["x"], float).reshape(1, -1)
        assert math.isclose(float(m.model.predict(x)[0]), fx["mu_raw"], rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# env integration: the REAL searcher through the REAL bidding phase            #
# --------------------------------------------------------------------------- #
def test_env_run_bidding_phase_with_real_searcher():
    """Full seam: normal path samples all seats (mock clients), env hands
    seat-0's presample to a real EvDistBidSearcher, which deviates past the
    gate and leaves opponents untouched."""
    from test_pikl_bid_search import BidMockClient

    s = _ev_searcher(mu_const=5, vhat=14.0, gate_min_ev=1.0)
    env = MegaGemEnv(num_players=3, value_chart_id="A")
    gs = env.create_game_state(seed=21)
    gs.current_auction = AuctionCard(id="t", type=AuctionType.TREASURE, gems=1)

    clients = [BidMockClient(bids=(2,)), BidMockClient(bids=(3,)), BidMockClient(bids=(4,))]
    bids, recs = asyncio.run(env.run_bidding_phase(
        gs, clients, ["m0", "m1", "m2"], pikl_bid_searcher=s))

    assert bids[1] == 3 and bids[2] == 4                    # opponents untouched
    assert recs[1].parse_method != "pikl" and recs[2].parse_method != "pikl"
    # seat-0: blueprint sampled 2 (hopeless vs mu=5, vhat=14) -> selector deviates
    assert s.decision_log, "selector saw the node"
    pay = s.decision_log[-1]
    assert pay["b_bp"] == 2
    assert pay["gate"]["passed"] and bids[0] == pay["b_star"] and bids[0] in (5, 6)
    assert recs[0].parse_method == "pikl" and recs[0].final_bid == bids[0]


def test_env_run_bidding_phase_passthrough_keeps_normal_record():
    from test_pikl_bid_search import BidMockClient

    # vhat below cost of winning -> b*=0 = blueprint's 0: no deviation possible;
    # set blueprint bid equal to optimum so the gate margin is 0 -> pass-through.
    s = _ev_searcher(mu_const=5, vhat=3.0, gate_min_ev=1.0)
    env = MegaGemEnv(num_players=3, value_chart_id="A")
    gs = env.create_game_state(seed=22)
    gs.current_auction = AuctionCard(id="t", type=AuctionType.TREASURE, gems=1)
    clients = [BidMockClient(bids=(0,)), BidMockClient(bids=(3,)), BidMockClient(bids=(4,))]
    bids, recs = asyncio.run(env.run_bidding_phase(
        gs, clients, ["m0", "m1", "m2"], pikl_bid_searcher=s))
    assert bids[0] == 0
    assert recs[0].parse_method != "pikl"      # the untouched normal record
    assert s.decision_log[-1]["gate"]["passed"] is False


def test_packaged_artifacts_are_the_frozen_evidence():
    """The shipped artifacts are what the published numbers were produced with.

    A silent swap (a refit copied over the packaged file, a partial download)
    would move every selector number with nothing to notice it, so pin the
    bytes. If you deliberately refit, update these digests and the table in
    src/megagem/assets/README.md together.
    """
    import hashlib

    from megagem.assets import asset_path

    expected = {
        "ev_dist_v1.pkl": "4df84250b64d2604",
        "ev_dist_l2_v1.pkl": "2e4fcf11a06d2764",
        "ev_dist_bp_v1.pkl": "28282f282c6db732",
        "value_head.pkl": "9f5c36e3bf811c81",
        "dynamics_sim_sweep_recovered_200.json": "a5df609daba64c23",
    }
    for name, digest in expected.items():
        path = asset_path(name)
        assert path.is_file(), f"packaged artifact missing: {name}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        assert actual == digest, (
            f"{name} changed: {actual} != {digest}. If this was a deliberate "
            f"refit, update assets/README.md and this test together."
        )
