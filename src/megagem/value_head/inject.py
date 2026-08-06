"""Live in-context value injection for the policy seat (Phase 2).

`value_block(game_state, player_id, estimator)` reads the LIVE GameState — public board
plus the acting seat's OWN hand (never an opponent's) — and returns a compact,
clearly-labelled estimate of each color's FINAL per-gem value and the marginal value of
the current Treasure lot, to be inserted into that seat's bid prompt before the action
request. Returns None on non-Treasure auctions (the head values gem lots).

Feature parity with training is exact: GameState's get_value_display_counts /
get_collection_counts return the same string-keyed Counters the offline builder used, and
round_number is the same 1-based field — and every feature is live-knowable (no
final-round-count look-ahead).
"""

from __future__ import annotations

from collections import Counter

from ..game.cards import AuctionType
from .value_estimator import COLORS, value_features


def value_block(game_state, player_id: int, estimator) -> str | None:
    auction = game_state.current_auction
    if auction is None or auction.type != AuctionType.TREASURE:
        return None

    chart_id = game_state.value_chart.id
    display_counts = game_state.get_value_display_counts()        # {color: count}, public
    own = Counter(game_state.players[player_id].hand)             # acting seat's OWN hand
    coll_all: Counter = Counter()                                # public collections (all seats)
    for p in game_state.players:
        for c, k in p.get_collection_counts().items():
            coll_all[c] += k
    rn = game_state.round_number

    per_color = []
    for c in COLORS:
        feat = value_features(color=c, display_counts=display_counts, own_hand_counts=own,
                              collection_counts_all=coll_all, round_number=rn)
        ev = estimator.evalue(feat, chart_id)
        lo, hi = estimator.value_band(feat, chart_id)
        per_color.append(f"{c} ~{ev:.0f} ({lo:.0f}-{hi:.0f})")

    gems = list(game_state.revealed_gems[: auction.gems])
    mv = estimator.marginal_value(
        gems=gems, seat_collection_counts=game_state.players[player_id].get_collection_counts(),
        available_mission_ids=[m.id for m in game_state.available_missions],
        display_counts=display_counts, own_hand_counts=own, collection_counts_all=coll_all,
        round_number=rn, chart_id=chart_id)

    mission = f" +{mv['mission_bonus']} mission" if mv["mission_bonus"] else ""
    return (
        "[Value model — calibrated estimates of each color's FINAL per-gem value]\n"
        + " | ".join(per_color) + "\n"
        + f"This Treasure ({'+'.join(gems) or 'no gem'}): est. end-game value "
        + f"~{mv['gem_value']:.0f}{mission} = ~{mv['total']:.0f} total to your collection.\n"
        + "These are model estimates of FINAL display value (later reveals raise counts); "
        + "weigh them against your own reasoning and the bidding competition."
    )
