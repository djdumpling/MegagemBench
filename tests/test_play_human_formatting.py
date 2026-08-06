import io

from rich.console import Console

from megagem.play import interactive as play_interactive
from megagem.game.cards import (
    AuctionCard,
    AuctionType,
    Mission,
    MissionRequirement,
    ValueChart,
)
from megagem.game.state import AuctionResult, GameState, PlayerState


def _sample_state() -> GameState:
    state = GameState(
        num_players=3,
        players=[
            PlayerState(0, 35, hand=["Purple", "Green", "Yellow", "Red", "Purple"]),
            PlayerState(1, 35, hand=["Red"] * 5),
            PlayerState(2, 29, hand=["Blue"] * 4, collection=["Green"]),
        ],
        value_chart=ValueChart(
            "A",
            "More revealed gems increase their value",
            {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 7},
        ),
        revealed_gems=["Blue", "Red"],
        gem_deck=[],
        auction_deck=[],
        available_missions=[
            Mission(
                "specific_rgb",
                "specific_3",
                "Collect Red + Blue + Green",
                MissionRequirement("specific", colors=["Red", "Blue", "Green"]),
                10,
            )
        ],
        current_auction=AuctionCard(
            "invest_5_02", AuctionType.INVESTMENT, bonus=5
        ),
        tiebreak_order=[1, 0, 2],
        round_number=2,
    )
    state.auction_history.append(
        AuctionResult(
            round_number=1,
            auction_card={"id": "treasure_1_10", "type": "treasure", "gems": 1},
            bids=[4, 5, 6],
            winner_id=2,
            winning_bid=6,
            gems_won=["Green"],
            gem_revealed="Green",
        )
    )
    return state


def test_human_game_state_is_readable_and_not_model_json(monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(
        play_interactive,
        "console",
        Console(file=output, width=120, color_system=None),
    )

    play_interactive.display_game_state_for_human(
        _sample_state(),
        model_names=["YOU (Human)", "Distilled A", "Distilled B"],
    )
    rendered = output.getvalue()

    assert '"game_progress"' not in rendered
    assert '"current_auction"' not in rendered
    assert '"your_private_hand"' not in rendered
    assert '"available_missions"' not in rendered
    assert '"auction_history"' not in rendered
    assert "Round 2" in rendered
    assert "INVESTMENT +5" in rendered
    assert "Purple×2" in rendered
    assert "Collect Red + Blue + Green" in rendered
    assert "Tiebreak priority" in rendered
    assert "Recent Auctions" in rendered
