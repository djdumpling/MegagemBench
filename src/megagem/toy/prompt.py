"""Deterministic bid prompt + stub/heuristic policies for the toy auction.

The prompt is the *only* channel a policy gets (mirrors how a real LLM seat
sees nothing but its prompt). It states this seat's own coins + private value
only — never an opponent's private value (leakage-safe). The stub/heuristic
policies re-read those numbers back out of the prompt text, so the exact same
``policy_fn(prompt) -> completion_text`` contract works for the CPU stub here
and the live-vLLM wrapper Phase 2.2 will add.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

from .auction_env import heuristic_bid

# Markers the policies parse back out of the prompt. Keep these stable: both
# the stub and (later) any prompt-driven analysis rely on them verbatim.
_RE_ROUND = re.compile(r"Round (\d+) of \d+")
_RE_PLAYER = re.compile(r"You are player (\d+)")
_RE_COINS = re.compile(r"Your current coins: (\d+)")
_RE_VALUE = re.compile(r"Your private value for this round's prize: (\d+)")


def bid_prompt(
    round_index: int,
    num_rounds: int,
    player_id: int,
    coins: int,
    private_value: int,
    tiebreak_before: list[int],
) -> str:
    """Self-contained bid prompt for one (round, player). Leakage-safe."""
    return (
        "You are playing a sealed-bid auction game.\n"
        f"Round {round_index + 1} of {num_rounds}.\n"
        f"You are player {player_id}.\n"
        "Each round every player privately values this round's single prize. "
        "All players submit one secret bid simultaneously; the highest bid "
        "wins and pays its own bid (first-price), gaining the prize's value "
        "in coins. Losing bidders pay nothing. Ties break by the decision "
        "order below (earlier wins).\n"
        f"Decision order this round (earliest breaks ties first): {tiebreak_before}\n"
        f"Your current coins: {coins}\n"
        f"Your private value for this round's prize: {private_value}\n"
        "Think briefly, then end your reply with a JSON object of the form "
        '{"bid": <integer>}. Your bid must be between 0 and your current '
        "coins.\n"
    )


def _read(prompt: str) -> dict[str, int]:
    """Pull (round_index, player_id, coins, private_value) out of a prompt."""
    m_r = _RE_ROUND.search(prompt)
    m_p = _RE_PLAYER.search(prompt)
    m_c = _RE_COINS.search(prompt)
    m_v = _RE_VALUE.search(prompt)
    if not (m_r and m_p and m_c and m_v):
        raise ValueError("prompt missing a required toy marker")
    return {
        "round_index": int(m_r.group(1)) - 1,
        "player_id": int(m_p.group(1)),
        "coins": int(m_c.group(1)),
        "private_value": int(m_v.group(1)),
    }


def make_heuristic_policy() -> Callable[[str], str]:
    """Deterministic opponent: bid ``clamp(round(0.8*v), 0, coins)``.

    Wrapped as a ``policy_fn(prompt)->text`` so every seat (trainable stub or
    heuristic opponent) satisfies the one uniform seam contract.
    """

    def policy(prompt: str) -> str:
        s = _read(prompt)
        b = heuristic_bid(s["private_value"], s["coins"])
        return f"<think>heuristic v={s['private_value']}</think> " f'{{"bid": {b}}}'

    return policy


# Stub shade factors. The opponents shade at ``_OPP_SHADE`` (0.8·v). Verified
# empirically via the repo engine (one game/seed, 0.05 shade grid): there is a
# flat **0.65–0.70·v plateau** ≈ +0.13–0.14, seed-set dependent (argmax 0.70
# on seeds 9000–9019, +0.131; argmax 0.65 on 9000–9039, +0.140 with 0.70 at
# +0.137). Either way the best fixed shade is *well below* the opponents'
# 0.8·v, and 0.85·v ≈ 0 / slightly negative (−0.010 on 20 seeds, +0.006 on
# 40) — far below the plateau. So 0.70·v is a representative point ON the
# plateau (not a unique optimum); 0.85·v is the misnamed loser. Full table +
# rationale: phase-2 plan P0.3 (historical).
_BESTRESP_SHADE = 0.70   # a strongly-profitable point on the 0.65–0.70 plateau
_ABOVESHADE_SHADE = 0.85  # the OLD mislabeled "bestresp"; ≈0/net-losing — kept,
                          # honestly named, so prior runs remain reproducible


def make_stub_policy(
    strategy: str = "bestresp", jitter_seed: int | None = None
) -> Callable[[str], str]:
    """Deterministic completion-text producer for CPU tests / the corpus driver.

    Strategies:

    * ``bestresp`` — profitable shade ``round(0.70*v)``, a representative point
      on the flat 0.65–0.70·v empirical plateau (mean r_terminal ≈ +0.13–0.14;
      seed-set dependent — see the constants note above; NOT a unique analytic
      optimum). It bids *below* the opponents' ``0.8*v`` so it cherry-picks
      high-surplus wins. The genuine "known-good" / learnable-target policy.
    * ``aboveshade`` — ``round(0.85*v)``. This was the strategy *previously
      (mis)named* ``bestresp``: it bids *above* the opponents' shade, wins
      often but overpays → mean r_terminal ≈ −0.010 (net-LOSING). Retained,
      honestly named, so the v5.5/v5.6 cadence-sweep runs stay reproducible.
    * ``overbid``  — ``bid = coins``: wins but ``Δcoins = v - coins < 0``.
      The "known-bad" policy.
    * ``illegal``  — ``bid = coins + 5``: parses fine but fails the
      ``0 <= bid <= coins`` legality check (``legal_valid=False``,
      ``default_used=True``, §A.2 legal reward ``-0.5``).
    * ``garbage``  — no JSON, no prose bid: ``parse_valid=False``.

    ``jitter_seed`` (if given) adds bounded ±2 integer noise to either
    strategic-shade bid (``bestresp``/``aboveshade``) keyed on
    ``(jitter_seed, round, player)`` — so K rollouts of one seed differ *only*
    in the trainable seat, giving a non-degenerate within-group advantage std
    (true K-group offline).
    """
    _STRATEGIC_SHADES = {"bestresp", "aboveshade"}

    def policy(prompt: str) -> str:
        s = _read(prompt)
        v, coins = s["private_value"], s["coins"]

        if strategy == "garbage":
            return "<think>weighing options</think> I remain undecided this round."

        if strategy == "overbid":
            b = coins
        elif strategy == "illegal":
            b = coins + 5
        elif strategy == "bestresp":
            b = int(round(_BESTRESP_SHADE * v))
        elif strategy == "aboveshade":
            b = int(round(_ABOVESHADE_SHADE * v))
        else:
            raise ValueError(f"unknown stub strategy {strategy!r}")

        if jitter_seed is not None and strategy in _STRATEGIC_SHADES:
            # random.Random only accepts int/str/bytes seeds (not tuples).
            jitter_key = (
                (jitter_seed * 1_000_003)
                ^ (s["round_index"] * 9176)
                ^ (s["player_id"] * 131)
            )
            rng = random.Random(jitter_key)
            b += rng.randint(-2, 2)
            b = max(0, min(coins, b))

        return f"<think>strategy={strategy} v={v}</think> " f'{{"bid": {b}}}'

    return policy
