"""Canonical opponent bid-distribution model for the E1 `ev_dist` selector.

ONE feature definition shared by the offline artifact fit
(scripts/analysis/build_ev_dist_artifact.py), the live selector
(src/megagem/environment/ev_dist.py), and the parity tests — so live and training
features cannot drift.

Bid-time semantics (load-bearing):
- A logged round's ``value_display`` is a POST-round snapshot (it includes that
  round's reveal); the bid decision sees the display AFTER round ri-1 only.
  ``iter_nodes_bidtime`` therefore takes display from the PREVIOUS round —
  matching megagem/value_head/train.py::bidtime_state, whose live
  parity was verified at corr 0.9991 (H1 of the retired eval_forensics
  probe). The original
  flash_bid_model_fit features used the same-round (post-reveal) display; the
  E1 artifact is refit on bid-time features so the live extractor matches by
  construction.
- History/market convention: TREASURE-only history with the window taken over
  treasure auctions (matches scripts/analysis/pikl_transfer_probe.py::
  market_pred, which the deployed model was fit against). Note the live
  pikl_search.market_anchored_bid windows over ALL auctions then filters —
  a (documented) divergence; the model's canonical convention is the former.

The win-probability curve treats opponent bids as independent given public
context: P(win | b) = prod_s [ P(bid_s < b) + 1{seat wins tie vs s} * P(bid_s = b) ],
with ties resolved by the round's tiebreak order exactly as resolve_auction does
(earliest in order wins; min seat id when absent).
"""

from __future__ import annotations

import pickle
import statistics
from pathlib import Path

import numpy as np

from ..game.cards import AuctionType

MODEL_VERSION = "ev_dist_v1"

FEATURES = [
    "any_median", "coins", "display_total", "fair_value_pred", "gems_count",
    "hist_n_match", "hist_n_total", "is_last_tiebreak", "last_winning_bid",
    "lot_value", "lot_value_next", "market_pred", "match_last", "match_median",
    "max_other_coins", "round",
]

MARKET_DEFAULTS = {"f": 1.0, "window": 5, "agg": "median", "match_gems": True,
                   "f_floor": 0.25, "tiebreak_delta": 1}
_AGG = {"mean": statistics.fmean, "median": statistics.median,
        "max": max, "last": lambda xs: xs[-1]}


# --------------------------------------------------------------------------- #
# node construction (live + offline-bidtime)                                   #
# --------------------------------------------------------------------------- #
def disp_counts(vd) -> dict[str, int]:
    """Tolerant value_display parse: {color: {count: n}} or {color: n}."""
    out: dict[str, int] = {}
    if not isinstance(vd, dict):
        return out
    for c, info in vd.items():
        out[c] = int((info or {}).get("count", 0) or 0) if isinstance(info, dict) else int(info or 0)
    return out


def node_from_live(game_state) -> dict | None:
    """Bid-time node from a live GameState (public info + tiebreak only).

    Field-for-field equal to ``iter_nodes_bidtime`` on the game's own log —
    asserted by tests/test_ev_dist.py parity fixtures.
    """
    auction = game_state.current_auction
    if auction is None or auction.type != AuctionType.TREASURE:
        return None
    gems = [str(g) for g in game_state.revealed_gems[:min(auction.gems, len(game_state.revealed_gems))]]
    if not gems:
        return None
    # treasure ⟺ gems_won nonempty; (gem count, winning bid), chronological
    history = [(len(r.gems_won), int(r.winning_bid))
               for r in game_state.auction_history if r.gems_won]
    return {
        "gems": gems,
        "gems_count": len(gems),
        "coins": {i: int(p.coins) for i, p in enumerate(game_state.players)},
        "tiebreak": list(game_state.tiebreak_order or []),
        "history": history,
        "display": dict(game_state.get_value_display_counts()),
        "round": int(game_state.round_number or 0),
    }


def iter_nodes_bidtime(game: dict):
    """Offline twin of node_from_live over a logged trajectory: one node per
    treasure auction with >=2 recorded bids, display taken from the PREVIOUS
    round (bid-time rule). Adds offline-only fields: bids, winner_id."""
    rounds = game.get("rounds") or []
    history: list[tuple[int, int]] = []
    prev_display: dict[str, int] = {}
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
            if gems and len(bids) >= 2:
                yield {
                    "gems": [str(g) for g in gems],
                    "gems_count": len(gems),
                    "coins": coins,
                    "tiebreak": list(r.get("tiebreak_order_before") or r.get("tiebreak_order") or []),
                    "history": list(history),
                    "display": dict(prev_display),     # bid-time: state after ri-1
                    "round": int(r.get("round_number") or 0),
                    "bids": bids,
                    "winner_id": winner_id,
                }
            if winner_id is not None:
                history.append((len(gems), bids.get(winner_id, 0)))
        prev_display = disp_counts(r.get("value_display"))


# --------------------------------------------------------------------------- #
# features                                                                     #
# --------------------------------------------------------------------------- #
def market_feature(node: dict, seat: int, *, f=1.0, window=5, agg="median",
                   match_gems=True, f_floor=0.25, tiebreak_delta=1) -> int:
    """Market-anchored bid over TREASURE-only history (the model's canonical
    convention; see module docstring)."""
    coins = node["coins"].get(seat, 0)
    hist = node["history"][-window:]
    A = [wb for (gc, wb) in hist if (not match_gems) or gc == node["gems_count"]]
    pred = (f_floor * coins) if not A else (f * _AGG[agg](A))
    tb = node["tiebreak"]
    if tiebreak_delta and tb and tb[-1] == seat:
        pred += tiebreak_delta
    return max(0, min(coins, round(pred)))


def fair_value_feature(node: dict, seat: int, chart, shade=0.8) -> int:
    """Public-display fair value (no hand peek), shade x lot display value."""
    value = sum(chart.get_gem_value(node["display"].get(c, 0)) for c in node["gems"])
    return max(0, min(node["coins"].get(seat, 0), round(shade * value)))


def features_for_seat(node: dict, seat: int, chart) -> dict[str, float]:
    hist = node["history"]
    matching = [wb for (gc, wb) in hist[-5:] if gc == node["gems_count"]]
    any_recent = [wb for (_gc, wb) in hist[-5:]]
    lot_value = sum(chart.get_gem_value(node["display"].get(c, 0)) for c in node["gems"])
    lot_value_next = sum(chart.get_gem_value(node["display"].get(c, 0) + 1) for c in node["gems"])
    coins = node["coins"].get(seat, 0)
    others = [v for k, v in node["coins"].items() if k != seat]
    tb = node["tiebreak"]
    return {
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
        "is_last_tiebreak": 1.0 if (tb and tb[-1] == seat) else 0.0,
        "market_pred": float(market_feature(node, seat, **MARKET_DEFAULTS)),
        "fair_value_pred": float(fair_value_feature(node, seat, chart)),
    }


def feature_vector(node: dict, seat: int, chart) -> np.ndarray:
    f = features_for_seat(node, seat, chart)
    return np.array([f[k] for k in FEATURES], dtype=float)


# --------------------------------------------------------------------------- #
# tie / win-probability / EV                                                   #
# --------------------------------------------------------------------------- #
def beats_tie(tiebreak: list[int], a: int, b: int) -> bool:
    """True iff seat ``a`` wins a bid tie against seat ``b`` — mirror of
    resolve_auction/predicted_winner: earliest in tiebreak order wins; min seat
    id when the order is absent or doesn't contain both."""
    if not tiebreak or a not in tiebreak or b not in tiebreak:
        return a < b
    return tiebreak.index(a) < tiebreak.index(b)


def bid_pmf(mu: int, coins: int, residuals: np.ndarray) -> np.ndarray:
    """PMF over integer bids 0..coins: clamp(round(mu + eps)), eps ~ residuals."""
    vals = np.clip(np.round(mu + residuals).astype(int), 0, max(0, int(coins)))
    pmf = np.bincount(vals, minlength=int(coins) + 1).astype(float)
    return pmf / pmf.sum()


def win_curve(opponents: list[tuple[int, int, bool]], residuals: np.ndarray,
              bmax: int) -> np.ndarray:
    """P(acting seat wins | bid b) for b = 0..bmax.

    opponents: [(mu_s, coins_s, acting_seat_beats_tie_vs_s)], independent seats.
    """
    out = np.ones(bmax + 1)
    for mu, coins_s, beats in opponents:
        pmf = bid_pmf(mu, coins_s, residuals)
        cdf = np.cumsum(pmf)
        col = np.empty(bmax + 1)
        for b in range(bmax + 1):
            if b > coins_s:
                p_lt, p_eq = 1.0, 0.0
            else:
                p_lt = float(cdf[b] - pmf[b])
                p_eq = float(pmf[b])
            col[b] = p_lt + (p_eq if beats else 0.0)
        out *= col
    return out


def ev_best(vhat: float, pwin: np.ndarray) -> tuple[int, float, np.ndarray]:
    """argmax_b (vhat - b) * P(win|b) over b = 0..len(pwin)-1."""
    b = np.arange(len(pwin), dtype=float)
    ev = (vhat - b) * pwin
    i = int(np.argmax(ev))
    return i, float(ev[i]), ev


# --------------------------------------------------------------------------- #
# the frozen artifact                                                          #
# --------------------------------------------------------------------------- #
class EvDistModel:
    """Frozen opponent bid-distribution model: sklearn point regressor +
    pooled out-of-fold residuals + feature spec. Pickled whole; the Modal eval
    image must carry the same scikit-learn version (see meta['sklearn'])."""

    def __init__(self, model, residuals: np.ndarray, features: list[str], meta: dict):
        assert list(features) == FEATURES, "artifact/feature-spec mismatch"
        self.model = model
        self.residuals = np.asarray(residuals, dtype=float)
        self.features = list(features)
        self.meta = dict(meta)

    def mu(self, node: dict, seat: int, chart) -> int:
        x = feature_vector(node, seat, chart).reshape(1, -1)
        raw = float(self.model.predict(x)[0])
        return max(0, min(int(node["coins"].get(seat, 0)), round(raw)))

    def win_curve_for(self, node: dict, acting_seat: int, chart,
                      opp_seats: list[int] | None = None) -> np.ndarray:
        opp = opp_seats if opp_seats is not None else [s for s in node["coins"] if s != acting_seat]
        spec = [(self.mu(node, s, chart), int(node["coins"].get(s, 0)),
                 beats_tie(node["tiebreak"], acting_seat, s)) for s in opp]
        return win_curve(spec, self.residuals, int(node["coins"].get(acting_seat, 0)))

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump({"version": MODEL_VERSION, "model": self.model,
                         "residuals": self.residuals, "features": self.features,
                         "meta": self.meta}, fh)

    @classmethod
    def load(cls, path) -> "EvDistModel":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        assert d.get("version") == MODEL_VERSION, f"artifact version {d.get('version')!r}"
        return cls(d["model"], d["residuals"], d["features"], d.get("meta") or {})
