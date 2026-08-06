"""Mission option value: deck-composition-based P(complete) for every mission
type, replacing the exact-completion-only mission bonus in the EV selector's
value input.

Information set is strictly bid-time-public + own hand: the gem composition is
fixed (6 per color), so the UNSEEN pool per color (6 - public display - own
hand - current lot) prices future acquisition opportunities; the remaining
treasure-card count is public deck-type accounting.

Acquisition model (V1, deliberately simple):
    E[future auctioned gems of color c] = treasures_remaining * avg_lot
                                          * unseen_c / unseen_total
    P(acquire >= k more of c)           = Poisson tail Q(k; p_grab * E[...])
with p_grab the per-lot win probability (default 1/3). Completion probabilities
mirror src/megagem/game/cards.py::Mission.check_completion exactly; "any_different" and
"pairs" use an exact Poisson-binomial DP over colors (independence approx
across colors).

OV(lot) = sum over claimable missions of reward * [P(complete | coll + lot)
                                                   - P(complete | coll)]
The exact-completion case falls out as reward * (1 - P(complete anyway)) —
i.e. this REPLACES (and de-double-counts) the old exact bonus.
"""
from __future__ import annotations

import math
from collections import Counter

COLORS = ("Blue", "Green", "Purple", "Red", "Yellow")
TOTAL_PER_COLOR = 6
N_TREASURE_CARDS = 17
AVG_LOT = 1.28          # empirical mean lot size (1,262 one-gem : 494 two-gem)
P_GRAB_DEFAULT = 1.0 / 3.0


def poisson_tail(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    p_le = sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k))
    return max(0.0, min(1.0, 1.0 - p_le))


def _acquire_probs(unseen: dict[str, int], treasures_remaining: int,
                   p_grab: float) -> dict[str, callable]:
    """color -> (k_needed -> P(acquire >= k more of color))."""
    total_unseen = sum(max(0, unseen.get(c, 0)) for c in COLORS)
    exp_gems = max(0.0, treasures_remaining) * AVG_LOT

    def make(c):
        share = (max(0, unseen.get(c, 0)) / total_unseen) if total_unseen else 0.0
        lam = p_grab * exp_gems * share
        return lambda k: poisson_tail(k, lam)
    return {c: make(c) for c in COLORS}


def _pb_tail(ps: list[float], k: int) -> float:
    """P(sum of independent Bernoulli(ps) >= k) — exact DP."""
    if k <= 0:
        return 1.0
    dp = [1.0] + [0.0] * len(ps)
    for p in ps:
        for i in range(len(dp) - 1, 0, -1):
            dp[i] = dp[i] * (1 - p) + dp[i - 1] * p
        dp[0] *= (1 - p)
    return max(0.0, min(1.0, sum(dp[k:])))


def _req_fields(req) -> dict:
    """Mirror cards.py access on either a Requirement object or a plain dict."""
    if isinstance(req, dict):
        return {"type": req.get("type"), "count": req.get("count"),
                "colors": req.get("colors"), "color": req.get("color")}
    return {"type": getattr(req, "type", None), "count": getattr(req, "count", None),
            "colors": getattr(req, "colors", None), "color": getattr(req, "color", None)}


def complete_prob(req, cc: Counter, acq: dict) -> float:
    """P(eventually complete) given collection counts cc and acquisition fns.
    Branch order mirrors Mission.check_completion exactly."""
    r = _req_fields(req)
    t, count, colors, color = r["type"], r["count"], r["colors"], r["color"]

    if t == "any_different":
        have = sum(1 for v in cc.values() if v >= 1)
        need = (count or 0) - have
        if need <= 0:
            return 1.0
        ps = [acq[c](1) for c in COLORS if cc.get(c, 0) == 0]
        return _pb_tail(ps, need)

    if t == "any_same":
        k = count or 0
        per = [acq[c](max(0, k - cc.get(c, 0))) for c in COLORS]
        miss = 1.0
        for p in per:
            miss *= (1.0 - p)
        return max(0.0, min(1.0, 1.0 - miss))

    if t == "pairs":
        have = sum(1 for v in cc.values() if v >= 2)
        need = (count or 0) - have
        if need <= 0:
            return 1.0
        ps = [acq[c](2 - cc.get(c, 0)) for c in COLORS if cc.get(c, 0) < 2]
        return _pb_tail(ps, need)

    if t == "specific" or colors:
        p = 1.0
        for c in colors or []:
            if cc.get(c, 0) < 1:
                p *= acq[c](1)
        return p

    if t == "same_color" or color:
        k = count or 0
        return acq[color](max(0, k - cc.get(color, 0)))

    return 0.0


def mission_option_value(missions, collection, lot_gems,
                         display_counts: dict[str, int],
                         own_hand, treasures_remaining: int,
                         completed_ids=(), p_grab: float = P_GRAB_DEFAULT) -> float:
    """OV of winning ``lot_gems`` for the seat with ``collection`` (list or
    Counter), given public display counts and own hand. ``missions`` is the
    available-missions list (objects with .requirement/.reward/.id, or dicts)."""
    cc = Counter(collection) if not isinstance(collection, Counter) else collection
    hand = Counter(own_hand) if not isinstance(own_hand, Counter) else own_hand
    lot = Counter(lot_gems)
    unseen = {c: max(0, TOTAL_PER_COLOR - int(display_counts.get(c, 0))
                     - hand.get(c, 0) - lot.get(c, 0)) for c in COLORS}
    acq = _acquire_probs(unseen, treasures_remaining, p_grab)
    cc_with = cc + lot
    ov = 0.0
    for m in missions:
        mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
        if mid in (completed_ids or ()):
            continue
        req = m.get("requirement") if isinstance(m, dict) else getattr(m, "requirement", None)
        reward = float(m.get("reward", 0) if isinstance(m, dict) else getattr(m, "reward", 0))
        if req is None or reward <= 0:
            continue
        ov += reward * max(0.0, complete_prob(req, cc_with, acq)
                           - complete_prob(req, cc, acq))
    return float(ov)
