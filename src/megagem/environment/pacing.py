"""State-dependent pacing shadow price lambda(state) — V3.

One pure function shared verbatim by the live selector (ev_dist) and the
dynamics simulator, so the sim adjudicates exactly what deploys. All inputs are
bid-time PUBLIC information (plus own coins):

  progress    = auctions resolved so far / 25   (deck size is fixed and public)
  spend_frac  = 1 - coins / 35                  (own budget vs starting budget)

Families (2 parameters each, pre-registered in the frontier doc V3 row):
  {"kind": "linear", "a": ..., "b": ...}  ->  lam = a + b * progress
      (declining schedules encode option value -> 0 as opportunities deplete)
  {"kind": "pace", "lam0": ..., "kappa": ...}
      ->  lam = lam0 + kappa * (spend_frac - progress)
      (Balseiro-Gur subgradient shape: ahead of the spend pace -> coins dearer)

Spec strings (CLI): "linear:0.8,-0.8" | "pace:0.5,2.0".
"""
from __future__ import annotations

TOTAL_CARDS = 25
START_COINS = 35
LAM_MIN, LAM_MAX = 0.0, 2.5


def parse_schedule(spec: str | None) -> dict | None:
    """"linear:a,b" / "pace:lam0,kappa" -> spec dict (None/empty -> None)."""
    if not spec:
        return None
    kind, _, rest = spec.partition(":")
    p1, _, p2 = rest.partition(",")
    a, b = float(p1), float(p2)
    if kind == "linear":
        return {"kind": "linear", "a": a, "b": b}
    if kind == "pace":
        return {"kind": "pace", "lam0": a, "kappa": b}
    raise ValueError(f"unknown pacing schedule kind: {kind!r}")


def pacing_lambda(spec: dict | None, *, auctions_resolved: int, coins: int,
                  flat: float = 0.0) -> float:
    """Effective lambda at a decision. ``spec`` None -> the flat (constant)
    value, preserving the legacy selector byte-for-byte."""
    if not spec:
        return float(flat)
    progress = min(1.0, max(0.0, auctions_resolved / TOTAL_CARDS))
    if spec["kind"] == "linear":
        lam = spec["a"] + spec["b"] * progress
    else:  # "pace"
        spend_frac = min(1.0, max(0.0, 1.0 - coins / START_COINS))
        lam = spec["lam0"] + spec["kappa"] * (spend_frac - progress)
    return max(LAM_MIN, min(LAM_MAX, float(lam)))
