"""Tests for the market-anchored opponent bid model (pikl_search).

The market model bids the way the SFT blueprint actually bids — anchored to recent winning
bids of comparable treasure auctions + own budget + tiebreak — and is used as the opponent
model inside the piKL Q rollout (opp_model="market") and the live fv_opponent_seats injection.

Covers: determinism/purity, cold-start fallback, treasure-only, clamp-to-coins, window +
gem-count match, the byte-identity of the default (fair_value) dispatch path (the critical
regression guard for the locked Gate-B arms), the searcher _opp_bid dispatch, and the
live-env market-opponent injection without an LLM. All CPU-only, no network.
"""
from __future__ import annotations

import asyncio

from megagem.environment.multi_agent_env import MegaGemEnv
from megagem.environment.pikl_search import (
    MARKET_DEFAULTS, PiklBidSearcher, fair_value_bid, market_anchored_bid, opponent_bid,
)
from megagem.game.cards import AuctionCard, AuctionType
from megagem.game.state import AuctionResult


def _ar(gems, winning_bid, *, bids=None, gem_count=None):
    """A treasure (gems nonempty) or non-treasure (gems=[]) auction-history record."""
    n = gem_count if gem_count is not None else (len(gems) or 2)
    atype = "treasure" if gems else "loan"
    card = {"id": "x", "type": atype, "gems": n} if gems else {"id": "x", "type": "loan", "amount": 10}
    return AuctionResult(round_number=1, auction_card=card, bids=bids or [winning_bid, 0, 0],
                         winner_id=0, winning_bid=winning_bid, gems_won=list(gems), gem_revealed=None)


def _state(*, coins=(35, 20, 8), gems=2, history=None, tiebreak=(0, 1, 2), revealed=("Red", "Purple")):
    env = MegaGemEnv(num_players=3, value_chart_id="A", seed=1)
    gs = env.create_game_state(seed=1)
    for i, c in enumerate(coins):
        gs.players[i].coins = c
    gs.players[0].hand = ["Red", "Red", "Blue"]
    gs.current_auction = AuctionCard(id="t", type=AuctionType.TREASURE, gems=gems)
    gs.revealed_gems = list(revealed)
    gs.value_display = []
    gs.tiebreak_order = list(tiebreak)
    gs.auction_history = list(history or [])
    return gs


# --- 1. determinism / purity -------------------------------------------------

def test_market_is_deterministic_and_pure():
    gs = _state(history=[_ar(["Red", "Purple"], 9)])
    before = (len(gs.auction_history), gs.players[1].coins, gs.current_auction.gems)
    r1 = market_anchored_bid(gs, 1, **MARKET_DEFAULTS)
    r2 = market_anchored_bid(gs, 1, **MARKET_DEFAULTS)
    assert r1 == r2
    # pure: no mutation of the game state
    assert (len(gs.auction_history), gs.players[1].coins, gs.current_auction.gems) == before


# --- 2. cold-start fallback --------------------------------------------------

def test_cold_start_uses_budget_fraction():
    gs = _state(coins=(35, 20, 8), history=[])  # no history at all
    assert market_anchored_bid(gs, 1, **MARKET_DEFAULTS) == round(0.25 * 20)
    # also cold when history has no matching-gem-count treasure
    gs = _state(gems=2, history=[_ar(["Blue"], 5, gem_count=1)])  # only a 1-gem treasure
    assert market_anchored_bid(gs, 1, f_floor=0.25, match_gems=True, window=5, agg="median",
                               f=1.0, tiebreak_delta=0) == round(0.25 * 20)


# --- 3. treasure-only --------------------------------------------------------

def test_non_treasure_returns_zero():
    gs = _state(history=[_ar(["Red", "Purple"], 9)])
    for t in (AuctionType.LOAN, AuctionType.INVESTMENT):
        gs.current_auction = AuctionCard(id="x", type=t, amount=10, bonus=5)
        assert market_anchored_bid(gs, 1, **MARKET_DEFAULTS) == 0
    gs.current_auction = None
    assert market_anchored_bid(gs, 1, **MARKET_DEFAULTS) == 0


# --- 4. clamp to coins -------------------------------------------------------

def test_clamp_to_coins():
    gs = _state(coins=(35, 3, 8), history=[_ar(["Red", "Purple"], 20)])  # anchor 20 > seat-1 coins 3
    assert market_anchored_bid(gs, 1, **MARKET_DEFAULTS) == 3
    # never negative
    gs = _state(coins=(35, 0, 8), history=[])
    assert market_anchored_bid(gs, 1, **MARKET_DEFAULTS) == 0


# --- 5. window + gem-count match --------------------------------------------

def test_window_and_gem_count_match():
    # window=2 keeps only the last two records; match_gems keeps only 2-gem treasures
    hist = [_ar(["Red", "Purple"], 99, gem_count=2),  # outside window=2 -> ignored
            _ar(["Blue"], 4, gem_count=1),            # in window but 1-gem -> ignored by match_gems
            _ar(["Green", "Yellow"], 8, gem_count=2)]  # in window, 2-gem -> the only anchor
    gs = _state(gems=2, history=hist)
    # match_gems=True, window=2, f=1.0, median([8]) = 8
    assert market_anchored_bid(gs, 1, f=1.0, window=2, agg="median", match_gems=True,
                               f_floor=0.25, tiebreak_delta=0) == 8
    # match_gems=False, window=2 -> median([4, 8]) = 6
    assert market_anchored_bid(gs, 1, f=1.0, window=2, agg="median", match_gems=False,
                               f_floor=0.25, tiebreak_delta=0) == 6


def test_tiebreak_bump_only_for_last_seat():
    gs = _state(coins=(35, 35, 35), tiebreak=(1, 2, 0), history=[_ar(["Red", "Purple"], 9)])
    # seat 0 is last in tiebreak (order[-1]) -> +1
    assert market_anchored_bid(gs, 0, **MARKET_DEFAULTS) == 10
    # seat 1 is not last -> no bump
    assert market_anchored_bid(gs, 1, **MARKET_DEFAULTS) == 9
    # tiebreak_delta=0 disables it
    assert market_anchored_bid(gs, 0, f=1.0, window=5, agg="median", match_gems=True,
                               f_floor=0.25, tiebreak_delta=0) == 9


# --- 6. byte-identity of the default (fair_value) dispatch path --------------

def test_dispatcher_fair_value_is_byte_identical():
    """opp_model='fair_value' must equal a direct fair_value_bid call (regression guard)."""
    gs = _state(history=[_ar(["Red", "Purple"], 9)])
    for seat in range(3):
        assert opponent_bid(gs, seat, 0, model="fair_value", shade=0.8) == fair_value_bid(gs, seat, 0, 0.8)
        assert opponent_bid(gs, seat, 0, model="fair_value", shade=0.5) == fair_value_bid(gs, seat, 0, 0.5)


def test_searcher_opp_bid_default_matches_fair_value():
    gs = _state(history=[_ar(["Red", "Purple"], 9)])
    s = PiklBidSearcher(bp_client=None, bp_model="bp", trainable_seat=0, num_players=3,
                        value_chart_id="A", bp_extra={}, lam=1.0, m_worlds=8, crn_seed=42)
    assert s.opp_model == "fair_value"  # default
    for seat in range(3):
        assert s._opp_bid(gs, seat) == fair_value_bid(gs, seat, 0, s.fv_shade)
    # switching to market routes to the market model with the fit defaults
    s.opp_model = "market"
    for seat in range(3):
        assert s._opp_bid(gs, seat) == market_anchored_bid(gs, seat, 0, shade=s.fv_shade, **MARKET_DEFAULTS)


# --- 7. live-env market-opponent injection (no LLM) --------------------------

def test_live_env_market_injection_without_llm():
    import json as _json

    class _Msg:
        def __init__(self, c):
            self.content = c
            self.reasoning_content = None

    class _Resp:
        def __init__(self, c):
            self.choices = [type("C", (), {"message": _Msg(c)})()]

    class _Seat0:  # seat-0 (non-market) returns a fixed legal bid via async create()
        def __init__(self):
            async def _create(**kw):
                return _Resp(_json.dumps({"bid": 4}))
            self.chat = type("Ch", (), {"completions": type("Co", (), {
                "create": staticmethod(_create)})()})()

    class _Boom:  # a market seat must never hit its client
        def __init__(self):
            async def _create(**kw):
                raise AssertionError("LLM called for a market opponent seat")
            self.chat = type("Ch", (), {"completions": type("Co", (), {
                "create": staticmethod(_create)})()})()

    env = MegaGemEnv(num_players=3, value_chart_id="A", seed=1)
    gs = env.create_game_state(seed=1)
    for i, c in enumerate((35, 20, 8)):
        gs.players[i].coins = c
    gs.current_auction = AuctionCard(id="t", type=AuctionType.TREASURE, gems=2)
    gs.revealed_gems = ["Red", "Purple"]
    gs.value_display = []
    gs.tiebreak_order = [0, 1, 2]
    gs.auction_history = [_ar(["Red", "Purple"], 9)]
    env.fv_opponent_seats = {1, 2}
    env.opp_model = "market"
    env.fv_opp_shade = 0.8

    bids, recs = asyncio.run(env.run_bidding_phase(
        gs, [_Seat0(), _Boom(), _Boom()], ["m", "m", "m"], {"temperature": 0.0}))
    for s in (1, 2):
        assert recs[s].parse_method == "market_anchored"
        assert bids[s] == market_anchored_bid(gs, s, trainable_seat=s, shade=0.8, **MARKET_DEFAULTS)
    assert bids[0] == 4  # seat-0 LLM bid still flows through normally
