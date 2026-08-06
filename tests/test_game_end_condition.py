"""The game-end condition and the deal size must have exactly one definition.

Four game loops decide when a game is over (the multi-agent env, piKL
continuations, the interactive human loop, the self-play wrapper). They used to
each re-derive `cards_per_player` and the "all gems won" threshold; a change to
the deal would then have ended some games at the wrong round with nothing
failing. These tests pin the single source and prove the loops agree with it.
"""

from __future__ import annotations

import inspect

import pytest

from megagem.data import load_auctions, load_gems, load_missions, load_value_charts
from megagem.environment.multi_agent_env import MegaGemEnv
from megagem.game.cards import ValueChart
from megagem.game.state import GameState, deal_for, total_gems_dealt


def _state(num_players: int) -> GameState:
    charts = load_value_charts()
    return GameState.create_new_game(
        num_players=num_players,
        gem_cards=load_gems(),
        auction_cards=load_auctions(),
        missions=load_missions(),
        value_chart=ValueChart.from_dict("A", charts["A"]),
        seed=7,
    )


@pytest.mark.parametrize("num_players,coins,per_player", [
    (3, 35, 5), (4, 25, 4), (5, 20, 3),
])
def test_deal_table_is_the_documented_setup(num_players, coins, per_player):
    assert deal_for(num_players) == (coins, per_player)
    assert total_gems_dealt(num_players) == num_players * per_player


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_dealt_state_matches_the_deal_table(num_players):
    """create_new_game must deal exactly what deal_for promises."""
    gs = _state(num_players)
    coins, per_player = deal_for(num_players)
    assert [p.coins for p in gs.players] == [coins] * num_players
    assert [len(p.hand) for p in gs.players] == [per_player] * num_players
    assert sum(len(p.hand) for p in gs.players) == total_gems_dealt(num_players)


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_all_gems_won_fires_exactly_at_the_threshold(num_players):
    gs = _state(num_players)
    threshold = total_gems_dealt(num_players)

    # Keep a non-empty pool so only the won-count branch can trigger.
    gs.revealed_gems = ["Red"]
    gs.players[0].collection = ["Red"] * (threshold - 1)
    assert not gs.all_gems_won(), "fired one gem early"
    gs.players[0].collection = ["Red"] * threshold
    assert gs.all_gems_won(), "did not fire at the threshold"


def test_empty_pool_ends_the_game_even_below_the_threshold():
    gs = _state(3)
    gs.players[0].collection = ["Red"]
    gs.gem_deck = []
    gs.revealed_gems = []
    assert gs.all_gems_won()


def test_env_end_check_delegates_to_the_shared_condition():
    """_maybe_end_game is the documented entry point (docs/rules.md) and has
    five callers; it must not carry its own copy of the rule."""
    src = inspect.getsource(MegaGemEnv._maybe_end_game)
    assert "all_gems_won" in src
    assert "cards_per_player" not in src, "re-derives the deal size"

    gs = _state(3)
    gs.players[0].collection = ["Red"] * total_gems_dealt(3)
    env = MegaGemEnv(num_players=3, value_chart_id="A", seed=7)
    env._maybe_end_game(gs)
    assert gs.game_over


def test_no_game_loop_re_derives_the_deal_size():
    """Guard against a fifth copy appearing."""
    from megagem.environment import rl_selfplay
    from megagem.play import interactive

    for mod in (interactive, rl_selfplay):
        src = inspect.getsource(mod)
        assert "cards_per_player =" not in src, (
            f"{mod.__name__} re-derives cards_per_player; use "
            f"megagem.game.state.deal_for / total_gems_dealt")
        assert "NUM_PLAYERS * 5" not in src, (
            f"{mod.__name__} hard-codes the deal size")
