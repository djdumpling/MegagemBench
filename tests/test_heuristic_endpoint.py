"""Tests for the stationary Phase-3 heuristic shim.

Covers v1 stationarity (BID_NOISE=0 → byte-identical to pre-v1.1 behaviour)
and v1.1 deterministic-per-state noise (BID_NOISE>0 → varies across distinct
auctions, but identical for identical auctions — preserves K-group credit
assignment).

We exercise the pure `decide()` / `_heuristic_bid()` functions; no HTTP server
is stood up, so these are fast unit tests with no network or port use.
"""

from __future__ import annotations

import importlib
import json
import sys

import pytest

_MODULE = "megagem.training.heuristic_endpoint"


@pytest.fixture
def heuristic_module(monkeypatch):
    """Re-import megagem.training.heuristic_endpoint with the requested env so the
    process-global BID_FRACTION / BID_FRACTION_NOISE constants are picked up.
    Yields the freshly-imported module; cleans up sys.modules on teardown so
    the next test gets a clean import with its own env.
    """
    def _load(*, bid_fraction: str = "0.5", bid_noise: str = "0.0"):
        monkeypatch.setenv("PHASE3_HEURISTIC_BID_FRACTION", bid_fraction)
        monkeypatch.setenv("PHASE3_HEURISTIC_BID_NOISE", bid_noise)
        sys.modules.pop(_MODULE, None)
        return importlib.import_module(_MODULE)
    yield _load
    sys.modules.pop(_MODULE, None)


def _make_auction(*, max_bid: int = 80, reason: str = "Treasure",
                  auction_id: int = 1, **extra):
    """A minimal current_auction dict shaped like the prompt-embedded one."""
    return {
        "auction_id": auction_id,
        "max_bid_for_you": max_bid,
        "bid_limit_reason": reason,
        **extra,
    }


# ----------------------------------------------------------------------------#
# v1 backward-compat: NOISE=0 must reproduce the original `floor(0.5*max_bid)` #
# ----------------------------------------------------------------------------#

def test_v1_stationary_default_floor(heuristic_module):
    mod = heuristic_module(bid_noise="0.0")
    assert mod.BID_FRACTION == 0.5
    assert mod.BID_FRACTION_NOISE == 0.0
    # 0.5 * 80 = 40
    assert mod._heuristic_bid(_make_auction(max_bid=80)) == 40
    # 0.5 * 79 = 39.5 → floor 39
    assert mod._heuristic_bid(_make_auction(max_bid=79)) == 39


def test_v1_declines_loan_and_investment(heuristic_module):
    mod = heuristic_module(bid_noise="0.0")
    assert mod._heuristic_bid(
        _make_auction(max_bid=80, reason="Loan")) == 0
    assert mod._heuristic_bid(
        _make_auction(max_bid=80, reason="Investment")) == 0


def test_v1_clamps_to_legal_range(heuristic_module):
    mod = heuristic_module(bid_noise="0.0")
    assert mod._heuristic_bid(_make_auction(max_bid=0)) == 0
    assert mod._heuristic_bid(_make_auction(max_bid=-5)) == 0


def test_v1_bid_noise_zero_returns_zero_epsilon(heuristic_module):
    mod = heuristic_module(bid_noise="0.0")
    # Multiple distinct auctions all return ε=0 (short-circuit, no hashing).
    for aid in range(20):
        assert mod._epsilon_for_auction(
            _make_auction(auction_id=aid)) == 0.0


# ----------------------------------------------------------------------------#
# v1.1 deterministic-per-state noise                                          #
# ----------------------------------------------------------------------------#

def test_noise_deterministic_same_auction_same_bid(heuristic_module):
    """K-group credit-assignment invariant: identical auction → identical
    bid, regardless of how many times we call (or what `_random` state
    might exist in the process)."""
    mod = heuristic_module(bid_noise="0.15")
    a = _make_auction(max_bid=80, auction_id=7)
    bids = [mod._heuristic_bid(a) for _ in range(100)]
    assert len(set(bids)) == 1, (
        f"non-deterministic bids for same auction: distinct values={set(bids)}")


def test_noise_varies_across_distinct_auctions(heuristic_module):
    """Distinct auctions should produce distinct bids (not all 40)."""
    mod = heuristic_module(bid_noise="0.15")
    bids = {
        mod._heuristic_bid(_make_auction(max_bid=80, auction_id=i))
        for i in range(50)
    }
    assert len(bids) >= 5, (
        f"only {len(bids)} distinct bids across 50 distinct auctions: {bids}")


def test_noise_bids_in_expected_range(heuristic_module):
    """Fraction ∈ [0.5-0.15, 0.5+0.15] = [0.35, 0.65]; for max_bid=80
    bids should land in [floor(0.35*80), floor(0.65*80)] = [28, 52]."""
    mod = heuristic_module(bid_noise="0.15")
    bids = [
        mod._heuristic_bid(_make_auction(max_bid=80, auction_id=i))
        for i in range(200)
    ]
    assert min(bids) >= 28, f"under-floor: min={min(bids)}"
    assert max(bids) <= 52, f"over-ceiling: max={max(bids)}"


def test_noise_distribution_roughly_uniform(heuristic_module):
    """The hash → uniform[-noise,+noise] mapping should give a distribution
    that *covers* both halves of the [0.5-noise, 0.5+noise] range — i.e.
    we should see both bids below 40 and above 40 across many auctions."""
    mod = heuristic_module(bid_noise="0.15")
    bids = [
        mod._heuristic_bid(_make_auction(max_bid=80, auction_id=i))
        for i in range(200)
    ]
    n_low = sum(1 for b in bids if b < 40)
    n_high = sum(1 for b in bids if b > 40)
    # SHA-256 should give roughly 50/50 split; allow generous slack.
    assert 60 <= n_low <= 140, f"low side underrepresented: n_low={n_low}"
    assert 60 <= n_high <= 140, f"high side underrepresented: n_high={n_high}"


def test_noise_clamps_at_extremes(heuristic_module):
    """fraction = BID_FRACTION + ε clamped to [0,1] — important for
    pathological NOISE values that would otherwise push fraction negative
    or > 1."""
    mod = heuristic_module(bid_fraction="0.95", bid_noise="0.30")
    # fraction could be in [0.65, 1.25] pre-clamp; post-clamp [0.65, 1.0].
    bids = [
        mod._heuristic_bid(_make_auction(max_bid=100, auction_id=i))
        for i in range(100)
    ]
    assert max(bids) <= 100, "bid exceeded max_bid_for_you (clamp broken)"
    assert min(bids) >= 0, "bid went negative"


def test_decide_dispatches_reveal_independent_of_noise(heuristic_module):
    """Reveal phase is unaffected by bid noise — still lex-first gem."""
    mod = heuristic_module(bid_noise="0.15")
    prompt = json.dumps({"action_request": {"action": "reveal"}}) + json.dumps(
        {"your_private_hand": {"cards": ["Ruby", "Emerald", "Sapphire"]}})
    out = mod.decide([{"role": "user", "content": prompt}])
    assert json.loads(out) == {"reveal": "Emerald"}


def test_decide_dispatches_bid_with_noise(heuristic_module):
    """End-to-end through decide() with NOISE=0.15: same prompt → same bid;
    different prompts → different bids (in expected range)."""
    mod = heuristic_module(bid_noise="0.15")

    def _prompt(auction_id):
        ar = json.dumps({"action_request": {"action": "bid"}})
        ca = json.dumps({"current_auction": _make_auction(
            max_bid=80, auction_id=auction_id, reason="Treasure")})
        return ar + ca

    out1a = mod.decide([{"role": "user", "content": _prompt(1)}])
    out1b = mod.decide([{"role": "user", "content": _prompt(1)}])
    assert out1a == out1b, "same prompt → different bid (det-per-state broken)"

    distinct = {mod.decide([{"role": "user", "content": _prompt(i)}])
                for i in range(30)}
    assert len(distinct) >= 3, (
        f"all {len(distinct)} distinct prompts produced same bid: {distinct}")
