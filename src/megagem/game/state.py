"""Game state management for MegaGem."""

import random
from dataclasses import dataclass, field
from typing import Any

from .cards import AuctionCard, GemCard, Mission, ValueChart


@dataclass
class Investment:
    bid_amount: int   # locked until end-game
    bonus: int        # 5 or 10

    @property
    def total_return(self) -> int:
        return self.bid_amount + self.bonus


# Table setup by player count: (starting_coins, cards_per_player). Every gem is
# dealt to a hand at setup, so the deal size also fixes how many gems can ever
# be won — i.e. the game-end threshold. Keep both derived from this one table so
# changing the deal cannot leave an end condition behind (see all_gems_won).
_DEAL: dict[int, tuple[int, int]] = {3: (35, 5), 4: (25, 4)}
_DEAL_FALLBACK = (20, 3)


def deal_for(num_players: int) -> tuple[int, int]:
    """``(starting_coins, cards_per_player)`` for a table of ``num_players``."""
    return _DEAL.get(num_players, _DEAL_FALLBACK)


def total_gems_dealt(num_players: int) -> int:
    """Gems across all hands at setup — the game-end threshold."""
    return num_players * deal_for(num_players)[1]


@dataclass
class PlayerState:
    player_id: int
    coins: int
    hand: list[str] = field(default_factory=list)            # private gem colors
    collection: list[str] = field(default_factory=list)      # publicly visible
    loans: list[int] = field(default_factory=list)
    investments: list[Investment] = field(default_factory=list)
    completed_missions: list[str] = field(default_factory=list)

    def copy(self) -> "PlayerState":
        return PlayerState(
            player_id=self.player_id,
            coins=self.coins,
            hand=list(self.hand),
            collection=list(self.collection),
            loans=list(self.loans),
            investments=[Investment(i.bid_amount, i.bonus) for i in self.investments],
            completed_missions=list(self.completed_missions),
        )

    def get_collection_counts(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(self.collection))

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "coins": self.coins,
            "collection": list(self.collection),
            "collection_counts": self.get_collection_counts(),
            "num_cards_in_hand": len(self.hand),
            "total_loans": sum(self.loans),
            "num_investments": len(self.investments),
            "completed_missions": list(self.completed_missions),
        }

@dataclass
class AuctionResult:
    round_number: int
    auction_card: dict
    bids: list[int]
    winner_id: int
    winning_bid: int
    gems_won: list[str]
    gem_revealed: str | None


@dataclass
class GameState:
    num_players: int
    players: list[PlayerState]
    value_chart: ValueChart
    value_display: list[str] = field(default_factory=list)
    revealed_gems: list[str] = field(default_factory=list)       # always exactly 2 mid-game
    gem_deck: list[GemCard] = field(default_factory=list)
    auction_deck: list[AuctionCard] = field(default_factory=list)
    all_missions: list[Mission] = field(default_factory=list)    # 30 total
    available_missions: list[Mission] = field(default_factory=list)  # 4 revealed
    current_auction: AuctionCard | None = None
    tiebreak_order: list[int] = field(default_factory=list)
    round_number: int = 0
    auction_history: list[AuctionResult] = field(default_factory=list)
    game_over: bool = False

    @classmethod
    def create_new_game(
        cls,
        num_players: int,
        gem_cards: list[dict],
        auction_cards: list[dict],
        missions: list[dict],
        value_chart: ValueChart,
        seed: int | None = None,
    ) -> "GameState":
        # Local Random so concurrent asyncio games don't share global state.
        rng = random.Random(seed)

        gem_deck = [GemCard.from_dict(g) for g in gem_cards]
        rng.shuffle(gem_deck)

        auction_deck = [AuctionCard.from_dict(a) for a in auction_cards]
        rng.shuffle(auction_deck)

        all_missions = [Mission.from_dict(m) for m in missions]
        rng.shuffle(all_missions)
        available_missions = all_missions[:4]

        starting_coins, cards_per_player = deal_for(num_players)

        players = []
        for i in range(num_players):
            hand = [gem_deck.pop().color.value for _ in range(cards_per_player)]
            players.append(PlayerState(player_id=i, coins=starting_coins, hand=hand))

        revealed_gems = [gem_deck.pop().color.value, gem_deck.pop().color.value]

        tiebreak_order = list(range(num_players))
        rng.shuffle(tiebreak_order)

        return cls(
            num_players=num_players,
            players=players,
            value_chart=value_chart,
            value_display=[],
            revealed_gems=revealed_gems,
            gem_deck=gem_deck,
            auction_deck=auction_deck,
            all_missions=all_missions,
            available_missions=available_missions,
            current_auction=None,
            tiebreak_order=tiebreak_order,
            round_number=0,
            auction_history=[],
            game_over=False,
        )

    def copy(self) -> "GameState":
        return GameState(
            num_players=self.num_players,
            players=[p.copy() for p in self.players],
            value_chart=self.value_chart,
            value_display=list(self.value_display),
            revealed_gems=list(self.revealed_gems),
            gem_deck=list(self.gem_deck),
            auction_deck=list(self.auction_deck),
            all_missions=list(self.all_missions),
            available_missions=list(self.available_missions),
            current_auction=self.current_auction,
            tiebreak_order=list(self.tiebreak_order),
            round_number=self.round_number,
            auction_history=list(self.auction_history),
            game_over=self.game_over,
        )

    def draw_auction_card(self) -> AuctionCard | None:
        """Pops next auction card; sets game_over=True if deck is empty."""
        if not self.auction_deck:
            self.game_over = True
            return None
        self.current_auction = self.auction_deck.pop(0)
        self.round_number += 1
        return self.current_auction

    def is_game_over(self) -> bool:
        return self.game_over or (not self.auction_deck and self.current_auction is None)

    def all_gems_won(self) -> bool:
        """The per-round end condition: every dealt gem sits in a collection, or
        the draw pool is exhausted.

        Single source of truth for the four game loops (multi-agent env, piKL
        continuations, interactive play, the self-play wrapper) — they must not
        re-derive it, or a change to the deal would end some games at the wrong
        round with nothing failing.
        """
        won = sum(len(p.collection) for p in self.players)
        return won >= total_gems_dealt(len(self.players)) or (
            not self.gem_deck and not self.revealed_gems)

    def get_gem_value(self, color: str) -> int:
        count = sum(1 for g in self.value_display if g == color)
        return self.value_chart.get_gem_value(count)

    def replenish_revealed_gems(self) -> None:
        """Top up revealed_gems to 2; preserves order so a held-over gem stays in slot 0."""
        while len(self.revealed_gems) < 2 and self.gem_deck:
            self.revealed_gems.append(self.gem_deck.pop(0).color.value)

    def update_tiebreak_order(self, winner_id: int) -> None:
        """Auction winner moves to the end of the tiebreak queue."""
        if winner_id in self.tiebreak_order:
            self.tiebreak_order.remove(winner_id)
            self.tiebreak_order.append(winner_id)

    def get_value_display_counts(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(self.value_display))

    def reveal_remaining_hands(self) -> None:
        """End-of-game: move every remaining hand gem to the value display."""
        for player in self.players:
            self.value_display.extend(player.hand)
            player.hand = []

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "num_players": self.num_players,
            "value_chart_id": self.value_chart.id,
            "value_chart_description": self.value_chart.description,
            "value_display": list(self.value_display),
            "value_display_counts": self.get_value_display_counts(),
            "revealed_gems": list(self.revealed_gems),
            "gems_remaining_in_deck": len(self.gem_deck),
            "auctions_remaining": len(self.auction_deck),
            "current_auction": self.current_auction.to_dict() if self.current_auction else None,
            "available_missions": [m.to_dict() for m in self.available_missions],
            "tiebreak_order": list(self.tiebreak_order),
            "players": [p.to_public_dict() for p in self.players],
            "auction_history": [
                {
                    "round": r.round_number,
                    "auction": r.auction_card,
                    "bids": r.bids,
                    "winner_id": r.winner_id,
                    "winning_bid": r.winning_bid,
                    "gems_won": r.gems_won,
                    "gem_revealed": r.gem_revealed,
                }
                for r in self.auction_history
            ],
            "game_over": self.game_over,
        }

