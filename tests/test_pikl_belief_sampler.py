"""Tests for the behaviour-informed M-world belief sampler (pikl_search).

Covers: the uniform default stays byte-identical (regression guard for the running
jobs / training), the bid/reveal likelihood has the correct sign, the SIR resampler is
deterministic and returns M distinct worlds, the informed belief actually shifts the
unseen gems toward the opponent who bid high on a colour, and the new oracle-regret
belief-fidelity diagnostic computes correctly. All CPU-only, no network."""

from __future__ import annotations

import asyncio
import random
import statistics

from megagem.environment.multi_agent_env import MegaGemEnv
from megagem.environment.pikl_search import (
    PiklBidSearcher, _behavior_logweight, _degenerate_diag, _reduce_diag,
    _resample_wor, redeal_consistent,
)
from megagem.game.cards import GemCard, GemColor
from megagem.game.state import AuctionResult


def _gem_deck(colors):
    return [GemCard(id=f"d{i}", color=GemColor(c)) for i, c in enumerate(colors)]


def _state_with_history(*, s1_hand, s2_hand, deck, bids, gems_won, gem_revealed=None):
    """3p state, seat 0 acting; seats 1/2 hands + deck form the controlled unseen pool;
    one prior treasure auction recorded in ``auction_history``."""
    env = MegaGemEnv(num_players=3, value_chart_id="A", seed=1)
    gs = env.create_game_state(seed=1)
    gs.players[0].hand = ["Purple", "Yellow"]   # acting seat — fixed, excluded from the pool
    gs.players[1].hand = list(s1_hand)
    gs.players[2].hand = list(s2_hand)
    gs.gem_deck = _gem_deck(deck)
    gs.value_display = []
    gs.auction_history = [AuctionResult(
        round_number=1, auction_card={"id": "t", "type": "treasure", "gems": 1},
        bids=list(bids), winner_id=int(max(range(len(bids)), key=lambda i: bids[i])),
        winning_bid=max(bids), gems_won=list(gems_won), gem_revealed=gem_revealed)]
    return gs


def _searcher(belief="uniform", *, m_worlds=8, oversample=6, sigma=6.0, gamma=0.5):
    s = PiklBidSearcher(
        bp_client=None, bp_model="bp", trainable_seat=0, num_players=3,
        value_chart_id="A", bp_extra={}, lam=1.0, m_worlds=m_worlds, crn_seed=42)
    s.belief = belief
    s.belief_oversample = oversample
    s.belief_sigma = sigma
    s.belief_gamma = gamma
    return s


# --- 1. uniform default is unchanged (regression guard) ----------------------

def test_uniform_make_worlds_matches_raw_redeal():
    gs = _state_with_history(s1_hand=["Red", "Blue"], s2_hand=["Green", "Blue"],
                             deck=["Red", "Red", "Red", "Green"], bids=[0, 14, 0], gems_won=["Red"])
    worlds = _searcher("uniform")._make_worlds(gs, 0)
    base = (42 * 7919 + gs.round_number) & 0x7FFFFFFF
    expected = [redeal_consistent(gs, 0, random.Random(base * 131 + m)) for m in range(8)]
    got = [[list(p.hand) for p in w.players] for w in worlds]
    exp = [[list(p.hand) for p in w.players] for w in expected]
    assert got == exp


# --- 2/3. behavioural likelihood has the correct sign ------------------------

def test_bid_logweight_favors_more_of_bid_color():
    gs = _state_with_history(s1_hand=["Red", "Red"], s2_hand=["Blue", "Blue"],
                             deck=["Green", "Green"], bids=[0, 12, 0], gems_won=["Red"])

    def lw(s1):
        w = gs.copy()
        w.players[1].hand = list(s1)
        return _behavior_logweight(w, 0, use_bids=True, use_reveals=False,
                                   sigma=6.0, shade=0.8, gamma=0.0)

    # seat 1 bid 12 on a Red treasure ⇒ worlds where seat 1 holds more Red are likelier.
    assert lw(["Red", "Red"]) > lw(["Red", "Blue"]) > lw(["Blue", "Blue"])


def test_reveal_logweight_favors_holding_revealed_color():
    gs = _state_with_history(s1_hand=["Green", "Green"], s2_hand=["Blue", "Blue"],
                             deck=["Green"], bids=[0, 5, 0], gems_won=["Red"], gem_revealed="Green")

    def lw(s1):
        w = gs.copy()
        w.players[1].hand = list(s1)
        return _behavior_logweight(w, 0, use_bids=False, use_reveals=True,
                                   sigma=6.0, shade=0.8, gamma=0.5)

    # seat 1 won and revealed Green ⇒ greedy-reveal makes a Green-holding seat 1 likelier.
    assert lw(["Green", "Green"]) > lw(["Blue", "Blue"])


def test_no_history_or_no_channel_is_zero_weight():
    gs = _state_with_history(s1_hand=["Red", "Red"], s2_hand=["Blue", "Blue"],
                             deck=["Green"], bids=[0, 12, 0], gems_won=["Red"])
    gs.auction_history = []
    assert _behavior_logweight(gs, 0, use_bids=True, use_reveals=True,
                               sigma=6.0, shade=0.8, gamma=0.5) == 0.0


# --- 4. SIR resampler: deterministic, distinct, weight-respecting ------------

def test_resample_wor_deterministic_distinct_and_respects_weight():
    pool = list(range(20))
    flat = [0.0] * 20
    a = _resample_wor(pool, flat, 5, random.Random(7))
    b = _resample_wor(pool, flat, 5, random.Random(7))
    assert a == b and len(set(a)) == 5            # deterministic + distinct

    # items 3 and 9 carry all the weight; the rest underflow to key 0 ⇒ always dropped first.
    peaked = [0.0 if i in (3, 9) else -1000.0 for i in range(20)]
    sel = _resample_wor(pool, peaked, 5, random.Random(7))
    assert 3 in sel and 9 in sel and len(set(sel)) == 5

    # m >= pool ⇒ return the whole pool untouched.
    assert _resample_wor(pool, flat, 50, random.Random(7)) == pool


# --- 5. informed belief shifts the unseen pool toward the high bidder --------

def test_informed_belief_shifts_unseen_gems_toward_bidder():
    # unseen pool = 4 Red, 2 Blue, 2 Green; seats 1 & 2 hold 2 each, 4 sit in the deck.
    gs = _state_with_history(s1_hand=["Red", "Blue"], s2_hand=["Green", "Blue"],
                             deck=["Red", "Red", "Red", "Green"], bids=[0, 14, 0], gems_won=["Red"])

    def mean_red(seat, belief):
        worlds = _searcher(belief, m_worlds=12, oversample=12)._make_worlds(gs, 0)
        return statistics.mean(w.players[seat].hand.count("Red") for w in worlds)

    u1, i1 = mean_red(1, "uniform"), mean_red(1, "bid")
    i2 = mean_red(2, "bid")
    assert i1 > u1     # seat 1 (bid high on Red) holds MORE red than under the uniform belief
    assert i1 > i2     # and more than seat 2, which bid 0 (no red demand revealed)


def test_informed_no_history_returns_full_m_no_crash():
    env = MegaGemEnv(num_players=3, value_chart_id="A", seed=2)
    gs = env.create_game_state(seed=2)            # fresh game → empty auction_history
    worlds = _searcher("both", m_worlds=8, oversample=6)._make_worlds(gs, 0)
    assert len(worlds) == 8                       # all weights 0 ⇒ graceful uniform WOR


# --- 6. oracle-regret belief-fidelity diagnostic -----------------------------

def test_reduce_diag_oracle_regret_and_agreement():
    tau = {"Red": 0.5, "Blue": 0.5}
    # belief says Red is best; the true world says Blue ⇒ positive regret, no agreement.
    d = _reduce_diag(tau, {"Red": [0.4, 0.4], "Blue": [0.1, 0.1]},
                     {"Red": 0.2, "Blue": 0.9}, round_number=3)
    assert abs(d["regret_vs_oracle"] - 0.7) < 1e-9
    assert d["oracle_agree"] == 0.0

    # belief and the true world agree ⇒ zero regret, agreement 1.
    d2 = _reduce_diag(tau, {"Red": [0.9, 0.9], "Blue": [0.1, 0.1]},
                      {"Red": 0.8, "Blue": 0.2}, round_number=3)
    assert d2["regret_vs_oracle"] == 0.0
    assert d2["oracle_agree"] == 1.0


def test_degenerate_diag_has_oracle_fields():
    d = _degenerate_diag(5, round_number=2)
    assert d["regret_vs_oracle"] == 0.0 and d["oracle_agree"] == 1.0 and d["single_candidate"]


# --- 7. corrected-likelihood knobs: coin cap + window -------------------------

def _ar(rnd, bids, winner, wbid, gems_won, revealed=None, atype="treasure"):
    return AuctionResult(round_number=rnd, auction_card={"id": f"a{rnd}", "type": atype},
                         bids=list(bids), winner_id=winner, winning_bid=wbid,
                         gems_won=list(gems_won), gem_revealed=revealed)


def _two_auction_state(s1_hand, history):
    env = MegaGemEnv(num_players=3, value_chart_id="A", seed=1)
    gs = env.create_game_state(seed=1)
    gs.players[0].hand = ["Purple", "Yellow"]
    gs.players[1].hand = list(s1_hand)
    gs.players[2].hand = ["Blue", "Blue"]
    gs.gem_deck = _gem_deck(["Red", "Red", "Green"])
    gs.value_display = []
    gs.auction_history = history
    return gs


def test_coin_cap_lowers_fit_when_opponent_is_broke():
    # seat 1 wins auction-1 paying 30 (coins 35→5), then bids 12 on a Red treasure.
    # In a Red-heavy world its uncapped fair value (~9.6) fits 12 well; the coin cap (5)
    # pulls the prediction down, worsening the fit ⇒ lower (more negative) log-weight.
    hist = [_ar(1, [0, 30, 0], 1, 30, ["Blue"]),
            _ar(2, [0, 12, 0], 1, 12, ["Red"])]
    gs = _two_auction_state(["Red", "Red", "Red"], hist)
    kw = dict(use_bids=True, use_reveals=False, sigma=6.0, shade=0.8, gamma=0.0)
    lw_nocap = _behavior_logweight(gs, 0, **kw, coin_cap=False)
    lw_cap = _behavior_logweight(gs, 0, **kw, coin_cap=True)
    assert lw_cap < lw_nocap            # the cap makes the broke high-bidder less explicable


def test_window_ignores_auctions_outside_the_last_N():
    # auction-1 (early): seat 1 bids high on Red — informative about Red holdings.
    # auction-2 (recent): seat 1 bids 0 on a Blue treasure — uninformative about Red.
    hist = [_ar(1, [0, 12, 0], 1, 12, ["Red"]),
            _ar(2, [0, 0, 0], 0, 0, ["Blue"])]

    def lw(s1, window):
        gs = _two_auction_state(s1, hist)
        return _behavior_logweight(gs, 0, use_bids=True, use_reveals=False, sigma=6.0,
                                   shade=0.8, gamma=0.0, window=window)

    # Worlds differ ONLY in Red (both hold zero Blue) so auction-2 (a Blue bid) cannot
    # tell them apart — the only discriminating signal is auction-1, the early one.
    # full history: the Red-heavy world fits auction-1's high Red bid better → higher weight.
    assert lw(["Red", "Red"], 0) > lw(["Green", "Green"], 0)
    # window=1: only auction-2 counts → Red signal dropped → no discrimination.
    assert lw(["Red", "Red"], 1) == lw(["Green", "Green"], 1)


# --- 8. value-coherent (fair-value) opponent seats in the live game ----------

def test_fv_opponent_seats_bid_fair_value_without_llm():
    import json as _json
    from megagem.environment.pikl_search import fair_value_bid
    from megagem.game.cards import AuctionCard, AuctionType

    class _Msg:
        def __init__(self, c):
            self.content = c
            self.reasoning_content = None

    class _Resp:
        def __init__(self, c):
            self.choices = [type("C", (), {"message": _Msg(c)})()]

    class _Seat0:  # returns a fixed legal bid via an async create()
        def __init__(self):
            async def _create(**kw):
                return _Resp(_json.dumps({"bid": 4}))
            self.chat = type("Ch", (), {"completions": type("Co", (), {
                "create": staticmethod(_create)})()})()

    class _Boom:  # a fair-value seat must never hit its client
        def __init__(self):
            async def _create(**kw):
                raise AssertionError("LLM called for a fair-value seat")
            self.chat = type("Ch", (), {"completions": type("Co", (), {
                "create": staticmethod(_create)})()})()

    env = MegaGemEnv(num_players=3, value_chart_id="A", seed=3)
    gs = env.create_game_state(seed=3)
    gs.current_auction = AuctionCard(id="t", type=AuctionType.TREASURE, gems=1)
    env.fv_opponent_seats = {1, 2}
    env.fv_opp_shade = 0.8
    clients = [_Seat0(), _Boom(), _Boom()]
    bids, recs = asyncio.run(env.run_bidding_phase(
        gs, clients, ["m", "m", "m"], {"temperature": 0.0}))
    for s in (1, 2):
        assert bids[s] == fair_value_bid(gs, s, trainable_seat=s, shade=0.8)
        assert recs[s].parse_method == "fair_value"
    assert bids[0] == 4  # seat-0 LLM bid still flows through normally
