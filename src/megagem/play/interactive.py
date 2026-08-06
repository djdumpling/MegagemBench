#!/usr/bin/env python3
"""
Play MegaGem against two LLM opponents (Gemini 3 Pro and Flash by default).

You (the human) are Player 0; the models take Players 1 and 2. You see the same
information the LLMs see and enter your bids/reveals manually.

Usage:
    megagem-play
    megagem-play --value-chart B --seed 123
    megagem-play --opponent anthropic/claude-opus-4.6 --opponent openai/gpt-5.5
"""

import argparse
import asyncio
import os
import sys
from collections import Counter

from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.box import ROUNDED
from rich.text import Text

from megagem.environment.console_format import color_gem_name, format_gem_string_with_colors
from megagem.environment.llm_response import API_TRANSIENT_ERRORS, response_content
from megagem.environment.prompts import (
    generate_reveal_prompt,
    generate_system_prompt,
)
from megagem.game.cards import AuctionType, GemColor, ValueChart
from megagem.game.actions import (
    get_default_bid,
    get_default_reveal,
    parse_bid,
    parse_reveal,
    validate_bid_for_auction,
)
from megagem.game.rules import (
    apply_auction_outcome,
    complete_mission,
    determine_winner,
    resolve_auction,
    reveal_gem_from_hand,
)
from megagem.game.state import GameState, total_gems_dealt
from megagem.data import load_value_charts

from megagem.endpoints import ENDPOINTS

console = Console()

HUMAN_PLAYER_ID = 0
GEMINI_PRO_MODEL = "google/gemini-3-pro-preview"
GEMINI_FLASH_MODEL = "google/gemini-3-flash-preview"


def _ordered_colors(colors) -> list[str]:
    """Canonical gem order (GemColor), with anything unrecognized appended."""
    ordered = [color.value for color in GemColor]
    ordered.extend(sorted(set(colors) - set(ordered)))
    return ordered


def _gem_counts_str(gems: list[str]) -> str:
    """``"Red×2, Blue×1"`` in canonical order — every gem list the human sees is
    rendered through this, so one screen never shows two different orderings."""
    counts = Counter(gems)
    return ", ".join(
        f"{color}×{counts[color]}" for color in _ordered_colors(counts) if counts[color]
    )


def _format_gem_counts(gems: list[str]) -> Text:
    """Compact, colored gem counts for the human UI."""
    summary = _gem_counts_str(gems)
    if not summary:
        return Text("Empty", style="dim")
    return format_gem_string_with_colors(summary)


def _format_auction_name(auction_data: dict) -> str:
    auction_type = auction_data.get("type", "auction")
    if auction_type == AuctionType.TREASURE.value:
        count = auction_data.get("gems", 0)
        return f"Treasure ({count} gem{'s' if count != 1 else ''})"
    if auction_type == AuctionType.LOAN.value:
        return f"Loan +{auction_data.get('amount', 0)}"
    if auction_type == AuctionType.INVESTMENT.value:
        return f"Investment +{auction_data.get('bonus', 0)}"
    return str(auction_type).title()


def _display_value_chart(game_state: GameState) -> None:
    chart = game_state.value_chart
    table = Table(
        title=f"Value Chart {chart.id} — {chart.description}",
        box=ROUNDED,
        show_lines=False,
    )
    table.add_column("Gems in display", style="bold")
    for count in sorted(chart.values):
        table.add_column(str(count), justify="center", style="cyan")
    table.add_row(
        "Value per gem",
        *(str(chart.values[count]) for count in sorted(chart.values)),
    )
    console.print(table)


def _display_progress(game_state: GameState) -> None:
    total_gems_won = sum(len(player.collection) for player in game_state.players)
    total_gems = total_gems_dealt(game_state.num_players)
    auctionable_remaining = len(game_state.gem_deck) + len(game_state.revealed_gems)

    summary = Text()
    summary.append(f"Round {game_state.round_number}", style="bold cyan")
    summary.append(" of roughly 15–20")
    summary.append("  •  ")
    summary.append(f"{len(game_state.auction_deck)}", style="bold")
    summary.append(" auction cards left")
    summary.append("  •  ")
    summary.append(f"{total_gems_won}/{total_gems}", style="bold green")
    summary.append(" gems won")
    summary.append("\n")
    summary.append(f"{auctionable_remaining}", style="bold")
    summary.append(" auctionable gems remain: ")
    summary.append(f"{len(game_state.gem_deck)} hidden")
    summary.append(f", {len(game_state.revealed_gems)} ready for auction")
    console.print(Panel(summary, title="Game Progress", border_style="dim"))


def _display_current_auction(game_state: GameState, player_id: int) -> None:
    auction = game_state.current_auction
    if auction is None:
        console.print(Panel("No active auction", title="Current Auction"))
        return

    player = game_state.players[player_id]
    details = Text()
    if auction.type == AuctionType.TREASURE:
        details.append(
            f"TREASURE — win {auction.gems} gem{'s' if auction.gems != 1 else ''}",
            style="bold yellow",
        )
        details.append("\nAvailable: ")
        details.append(_format_gem_counts(game_state.revealed_gems[: auction.gems]))
        max_bid = player.coins
        explanation = "Winner pays their bid and adds the shown gems to their collection."
    elif auction.type == AuctionType.LOAN:
        details.append(f"LOAN +{auction.amount}", style="bold yellow")
        details.append(
            f"\nWinner receives {auction.amount} coins now and repays "
            f"{auction.amount} at game end."
        )
        max_bid = player.coins + auction.amount
        explanation = "This is the only auction where you may bid above your current coins."
    else:
        details.append(f"INVESTMENT +{auction.bonus}", style="bold yellow")
        details.append(
            f"\nWinner locks their bid now, then receives the bid plus "
            f"{auction.bonus} at game end."
        )
        max_bid = player.coins
        explanation = "Investment bids cannot exceed your current coins."

    details.append(f"\n{explanation}", style="dim")
    details.append("\nYour maximum bid: ")
    details.append(str(max_bid), style="bold cyan")
    console.print(
        Panel(details, title="Current Auction", border_style="bold yellow")
    )


def _display_missions(game_state: GameState) -> None:
    table = Table(title="Available Missions", box=ROUNDED, expand=True)
    table.add_column("Requirement", ratio=4)
    table.add_column("Reward", justify="right", style="bold green", no_wrap=True)
    for mission in game_state.available_missions:
        table.add_row(color_gem_name(mission.description), f"+{mission.reward} coins")
    if not game_state.available_missions:
        table.add_row("No missions remain", "—")
    console.print(table)


def _display_tiebreak(
    game_state: GameState, player_id: int, model_names: list[str]
) -> None:
    order = Text("Tiebreak priority: ", style="dim")
    for position, pid in enumerate(game_state.tiebreak_order):
        if position:
            order.append("  →  ", style="dim")
        label = "YOU" if pid == player_id else model_names[pid]
        order.append(label, style="bold cyan" if pid == player_id else "bold")
    order.append("  (leftmost wins a tied bid)", style="dim")
    console.print(order)


def _display_history(game_state: GameState, model_names: list[str]) -> None:
    if not game_state.auction_history:
        return

    table = Table(title="Recent Auctions", box=ROUNDED, expand=True)
    table.add_column("Round", justify="right", style="cyan", no_wrap=True)
    table.add_column("Auction", no_wrap=True)
    table.add_column("Winner", no_wrap=True)
    table.add_column("Bids")
    table.add_column("Outcome")
    for result in game_state.auction_history[-5:]:
        bids = " / ".join(
            f"P{pid}: {bid}" for pid, bid in enumerate(result.bids)
        )
        outcome = Text()
        if result.gems_won:
            outcome.append("Won ")
            outcome.append(_format_gem_counts(result.gems_won))
        if result.gem_revealed:
            if result.gems_won:
                outcome.append("; ")
            outcome.append("revealed ")
            outcome.append(color_gem_name(result.gem_revealed))
        if not outcome.plain:
            outcome.append("—", style="dim")
        table.add_row(
            str(result.round_number),
            _format_auction_name(result.auction_card),
            f"P{result.winner_id} · {model_names[result.winner_id]} ({result.winning_bid})",
            bids,
            outcome,
        )
    console.print(table)


def display_game_state_for_human(
    game_state: GameState,
    player_id: int = HUMAN_PLAYER_ID,
    model_names: list[str] | None = None,
):
    """Display the current game state in a human-friendly format."""
    model_names = model_names or ["YOU (Human)", "Gemini 3 Pro", "Gemini 3 Flash"]
    console.print()

    # These are dedicated human views. The JSON serializers in prompts.py stay
    # untouched and continue feeding compact structured state to the models.
    _display_value_chart(game_state)

    # Value Display
    value_counts = game_state.get_value_display_counts()
    if value_counts:
        vd_table = Table(box=ROUNDED, title="Value Display")
        vd_table.add_column("Color")
        vd_table.add_column("In Display", justify="right", style="cyan")
        vd_table.add_column("Value/Gem", justify="right", style="cyan")
        for color, count in sorted(value_counts.items()):
            value_per_gem = game_state.get_gem_value(color)
            vd_table.add_row(color_gem_name(color), str(count), str(value_per_gem))
        console.print(vd_table)

    _display_progress(game_state)
    _display_current_auction(game_state, player_id)

    # Players table
    players_table = Table(title="Players", box=ROUNDED)
    players_table.add_column("Seat", style="cyan", no_wrap=True)
    players_table.add_column("Player", style="bold")
    players_table.add_column("Coins", justify="right", style="cyan")
    players_table.add_column("Collection", ratio=2)
    players_table.add_column("Private", justify="right", style="cyan")
    players_table.add_column("Contracts & Missions", ratio=2)

    for pid in range(game_state.num_players):
        player = game_state.players[pid]
        col_str = _gem_counts_str(player.collection) or "Empty"

        status = []
        if player.loans:
            status.append(f"Loan {sum(player.loans)}")
        if player.investments:
            returns = sum(investment.total_return for investment in player.investments)
            status.append(f"{len(player.investments)} investment(s) → {returns}")
        if player.completed_missions:
            status.append(f"{len(player.completed_missions)} mission(s)")

        players_table.add_row(
            f"P{pid}",
            model_names[pid],
            str(player.coins),
            format_gem_string_with_colors(col_str),
            f"{len(player.hand)} cards",
            " • ".join(status) if status else "—",
        )
    console.print(players_table)

    # Your hand
    console.print(
        Panel(
            _format_gem_counts(game_state.players[player_id].hand),
            title="YOUR HAND (Private)",
            border_style="bold magenta",
        )
    )

    _display_missions(game_state)
    _display_tiebreak(game_state, player_id, model_names)
    _display_history(game_state, model_names)


def get_human_bid(game_state: GameState, player_id: int = HUMAN_PLAYER_ID) -> int:
    """Prompt the human player for their bid."""
    player = game_state.players[player_id]
    auction = game_state.current_auction

    if auction and auction.type == AuctionType.LOAN:
        max_bid = player.coins + auction.amount
        console.print(f"\n[bold yellow]This is a Loan card — you can bid up to {max_bid} (your {player.coins} coins + {auction.amount} loan).[/bold yellow]")
    else:
        max_bid = player.coins
        console.print(f"\n[bold]You have {player.coins} coins. Enter your bid (0 to {max_bid}):[/bold]")

    while True:
        try:
            bid_str = input("Your bid > ").strip()
            if not bid_str:
                console.print("[red]Please enter a number.[/red]")
                continue
            bid = int(bid_str)
            valid, error = validate_bid_for_auction(game_state, player_id, bid)
            if valid:
                return bid
            console.print(f"[red]Invalid bid: {error}[/red]")
        except ValueError:
            console.print("[red]Please enter a valid integer.[/red]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Game aborted.[/yellow]")
            sys.exit(0)


def get_human_reveal(game_state: GameState, player_id: int = HUMAN_PLAYER_ID) -> str:
    """Prompt the human player for which gem to reveal."""
    player = game_state.players[player_id]
    if not player.hand:
        return ""

    hand_str = _gem_counts_str(player.hand)
    console.print("\n[bold magenta]You won the auction! You must reveal one gem from your hand.[/bold magenta]")
    console.print(f"Your hand: {hand_str}")
    valid = [c for c in _ordered_colors(set(player.hand)) if c in set(player.hand)]
    console.print("Valid colors: " + ", ".join(valid))

    while True:
        try:
            color = input("Reveal which gem? > ").strip().title()
            if color in player.hand:
                return color
            console.print(f"[red]You don't have '{color}' in your hand. Choose from: {sorted(set(player.hand))}[/red]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Game aborted.[/yellow]")
            sys.exit(0)


async def play_game(
    value_chart: str = "A",
    seed: int = 42,
    *,
    client: AsyncOpenAI | None = None,
    opponent_models: tuple[str, str] = (GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL),
    opponent_names: tuple[str, str] = ("Gemini 3 Pro", "Gemini 3 Flash"),
    matchup_title: str | None = None,
    request_kwargs: dict | None = None,
    bid_selector=None,
    selector_seats: tuple[int, ...] = (),
):
    """Play a three-player game with a human in seat 0 and two API opponents.

    The defaults preserve the original Gemini matchup. Callers can provide an
    OpenAI-compatible client and two served-model names to reuse the same game
    UI with another endpoint (for example distilled weights served by vLLM).
    ``bid_selector`` may post-process the weights-only bids for the listed
    ``selector_seats``; this is used by the distilled model's deployable
    dynamics-aware policy on treasure auctions.
    """
    num_players = 3
    model_names = ["YOU (Human)", *opponent_names]
    request_kwargs = request_kwargs or {}

    # Create the shared Gemini gateway client unless a caller supplied another
    # OpenAI-compatible endpoint.
    if client is None:
        gemini_endpoint = ENDPOINTS.get(opponent_models[0])
        if not gemini_endpoint:
            console.print(
                f"[red]Error: {opponent_models[0]} not found in endpoints config.[/red]"
            )
            sys.exit(1)

        api_key = os.getenv(gemini_endpoint["key"])
        if not api_key:
            console.print(f"[red]Error: {gemini_endpoint['key']} not set.[/red]")
            sys.exit(1)

        client = AsyncOpenAI(api_key=api_key, base_url=gemini_endpoint["url"])

    # Load environment (reuse the same game setup logic)
    from megagem.data import load_gems, load_auctions, load_missions
    from megagem.game.cards import ValueChart as VC

    gem_cards = load_gems()
    auction_cards = load_auctions()
    missions_data = load_missions()
    value_charts = load_value_charts()

    chart_data = value_charts.get(value_chart)
    if chart_data is None:
        console.print(f"[red]Invalid value chart: {value_chart}[/red]")
        sys.exit(1)
    vc = VC.from_dict(value_chart, chart_data)

    game_state = GameState.create_new_game(
        num_players=num_players,
        gem_cards=gem_cards,
        auction_cards=auction_cards,
        missions=missions_data,
        value_chart=vc,
        seed=seed,
    )

    matchup_title = matchup_title or f"Human vs {opponent_names[0]} vs {opponent_names[1]}"
    policy_line = (
        "\n[bold green]Policy: distilled weights + EV selector "
        "(treasure bids)[/bold green]"
        if bid_selector is not None and selector_seats
        else ""
    )
    console.print(Panel.fit(
        f"[bold cyan]MegaGem: {matchup_title}[/bold cyan]\n\n"
        f"You are [bold]Player 0[/bold]\n"
        f"[bold green]{opponent_names[0]}[/bold green] is Player 1\n"
        f"[bold yellow]{opponent_names[1]}[/bold yellow] is Player 2\n\n"
        f"Value Chart: {value_chart} | Seed: {seed}{policy_line}",
        border_style="bold cyan",
    ))

    round_num = 0

    while not game_state.is_game_over():
        # Draw auction card
        auction = game_state.draw_auction_card()
        if auction is None:
            break

        round_num += 1

        console.print(f"\n{'=' * 80}")
        console.print(f"[bold]ROUND {round_num}[/bold]")
        console.print(f"{'=' * 80}")

        # Show game state to human
        display_game_state_for_human(
            game_state, HUMAN_PLAYER_ID, model_names=model_names
        )

        from megagem.environment.prompts import generate_bid_prompt

        async def get_llm_bid(player_id: int, model: str) -> tuple[int, str, str]:
            prompt = generate_bid_prompt(game_state, player_id)
            messages = [
                {"role": "system", "content": generate_system_prompt()},
                {"role": "user", "content": prompt},
            ]
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **request_kwargs,
                )
                content = response_content(response)
                parsed = parse_bid(content)
                if parsed.valid:
                    valid, error = validate_bid_for_auction(game_state, player_id, parsed.bid)
                    if valid:
                        return parsed.bid, content, parsed.reasoning
                return get_default_bid(), content, parsed.reasoning
            except API_TRANSIENT_ERRORS as e:
                # Transient: fall back to a default bid so one hiccup doesn't
                # end the game, but say so — a silent 0 looks like strategy.
                console.print(
                    f"[red]Transient API error from {model} ({type(e).__name__}) "
                    f"— seat {player_id} bids the default this round.[/red]")
                return get_default_bid(), "", ""

        # Fire off LLM calls immediately (run while human is deciding)
        llm_tasks = asyncio.gather(
            get_llm_bid(1, opponent_models[0]),
            get_llm_bid(2, opponent_models[1]),
        )

        # Get human bid (LLM calls are running in the background)
        human_bid = await asyncio.to_thread(get_human_bid, game_state, HUMAN_PLAYER_ID)

        # Now await the LLM results (likely already done by now)
        console.print("\n[dim]Waiting for the model opponents to bid...[/dim]")
        results = await llm_tasks

        bids = [human_bid, results[0][0], results[1][0]]
        weights_bids = list(bids)
        selector_payloads: dict[int, dict] = {}
        if bid_selector is not None:
            for pid in selector_seats:
                if pid <= HUMAN_PLAYER_ID or pid >= num_players:
                    continue
                try:
                    bids[pid], payload = bid_selector.select(
                        game_state, pid, weights_bids[pid]
                    )
                    if payload is not None:
                        selector_payloads[pid] = payload
                except Exception as exc:  # noqa: BLE001
                    bids[pid] = weights_bids[pid]
                    selector_payloads[pid] = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "b_bp": weights_bids[pid],
                        "chosen": weights_bids[pid],
                    }
                    console.print(
                        f"[yellow]Selector error for Player {pid}; using the "
                        f"weights-only bid: {exc}[/yellow]"
                    )
        reasonings = ["(Human player)", results[0][2], results[1][2]]

        # Show bids
        bid_table = Table(title="Bids", box=ROUNDED)
        bid_table.add_column("Player", style="cyan")
        bid_table.add_column("Role", style="bold")
        bid_table.add_column("Bid", justify="right", style="bold cyan")
        bid_table.add_column("Decision", style="dim")

        for pid in range(num_players):
            if pid == HUMAN_PLAYER_ID:
                decision = "Manual"
            elif pid not in selector_seats or bid_selector is None:
                decision = "Weights"
            else:
                payload = selector_payloads.get(pid)
                if payload is None:
                    decision = "Weights"
                elif payload.get("error"):
                    decision = "Weights (selector error)"
                elif (payload.get("gate") or {}).get("passed"):
                    margin = float((payload.get("gate") or {}).get("margin", 0.0))
                    decision = (
                        f"Selector {weights_bids[pid]}→{bids[pid]} "
                        f"(ΔEV +{margin:.1f})"
                    )
                else:
                    decision = "Weights (selector kept bid)"
            bid_table.add_row(
                f"Player {pid}", model_names[pid], str(bids[pid]), decision
            )
        console.print(bid_table)

        # Resolve auction
        outcome = resolve_auction(game_state, bids)
        winner_id = outcome.winner_id

        winner_label = model_names[winner_id]
        console.print(f"\n[bold green]Winner: Player {winner_id} ({winner_label}) with bid {outcome.winning_bid}![/bold green]")

        if outcome.gems_won:
            gems_str = ", ".join(outcome.gems_won)
            console.print(f"Gems won: {gems_str}")

        # Show LLM reasoning (collapsed by default)
        for pid in [1, 2]:
            if reasonings[pid] and reasonings[pid] != "(Human player)":
                # Truncate long reasoning
                r = reasonings[pid]
                if len(r) > 300:
                    r = r[:300] + "..."
                label = f"{model_names[pid]} weights reasoning"
                if bids[pid] != weights_bids[pid]:
                    label += (
                        f" (proposed {weights_bids[pid]}; selector chose {bids[pid]})"
                    )
                console.print(f"\n[dim]{label}: {r}[/dim]")

        # Gem reveal phase (treasure auctions only)
        gem_revealed = None
        if auction.type == AuctionType.TREASURE:
            if winner_id == HUMAN_PLAYER_ID:
                # Human reveals
                gem_revealed = get_human_reveal(game_state, HUMAN_PLAYER_ID)
            else:
                # LLM reveals
                player = game_state.players[winner_id]
                if player.hand:
                    model = opponent_models[winner_id - 1]
                    prompt = generate_reveal_prompt(game_state, winner_id)
                    messages = [
                        {"role": "system", "content": generate_system_prompt()},
                        {"role": "user", "content": prompt},
                    ]
                    try:
                        response = await client.chat.completions.create(
                            model=model,
                            messages=messages,
                            **request_kwargs,
                        )
                        content = response_content(response)
                        parsed = parse_reveal(content)
                        if parsed.valid and parsed.gem_color in player.hand:
                            gem_revealed = parsed.gem_color
                        else:
                            gem_revealed = get_default_reveal(player.hand)
                    except API_TRANSIENT_ERRORS as e:
                        console.print(
                            f"[red]Transient API error during reveal "
                            f"({type(e).__name__}) — revealing the default.[/red]")
                        gem_revealed = get_default_reveal(player.hand)

                    console.print(f"[dim]{model_names[winner_id]} reveals: {gem_revealed}[/dim]")

        # Track coins before
        coins_before = [p.coins for p in game_state.players]

        # Apply auction outcome
        if gem_revealed:
            reveal_gem_from_hand(game_state, winner_id, gem_revealed)
        apply_auction_outcome(game_state, outcome, bids, gem_revealed)

        # Show coin changes
        for pid in range(num_players):
            old = coins_before[pid]
            new = game_state.players[pid].coins
            if old != new:
                console.print(f"  Player {pid} ({model_names[pid]}): {old} → {new} coins")

        # Mission phase
        if auction.type == AuctionType.TREASURE and game_state.available_missions:
            completed = []
            for mission in list(game_state.available_missions):
                if complete_mission(game_state, winner_id, mission.id):
                    completed.append(mission.id)
            if completed:
                console.print(f"\n[bold green]Player {winner_id} ({model_names[winner_id]}) completed mission(s): {', '.join(completed)}![/bold green]")

        # Check end condition (shared with every other game loop).
        if game_state.all_gems_won():
            game_state.game_over = True

    # Game over
    console.print(f"\n{'=' * 80}")
    console.print("[bold cyan]GAME OVER[/bold cyan]")
    console.print(f"{'=' * 80}")

    winner_id, final_scores = determine_winner(game_state)

    # Final scores table
    scores_table = Table(title="Final Scores", box=ROUNDED)
    scores_table.add_column("Player", style="cyan")
    scores_table.add_column("Role", style="bold")
    scores_table.add_column("", width=2, justify="center")
    scores_table.add_column("Coins", justify="right", style="cyan")
    scores_table.add_column("Gem Value", justify="right", style="cyan")
    scores_table.add_column("Missions", justify="right", style="cyan")
    scores_table.add_column("Loans", justify="right", style="cyan")
    scores_table.add_column("Investments", justify="right", style="cyan")
    scores_table.add_column("Final Score", justify="right", style="bold cyan")

    for score in final_scores:
        pid = score["player_id"]
        is_winner = pid == winner_id
        scores_table.add_row(
            f"Player {pid}",
            model_names[pid],
            "⭐" if is_winner else "",
            str(score["coins"]),
            str(score["gem_value"]),
            str(score["mission_rewards"]),
            f"-{score['loan_payments']}",
            f"+{score['investment_returns']}",
            str(score["final_score"]),
        )
    console.print(scores_table)

    # Final value display
    value_counts = game_state.get_value_display_counts()
    if value_counts:
        charts = load_value_charts()
        chart = ValueChart.from_dict(value_chart, charts[value_chart])
        vd_table = Table(title="Final Value Display", box=ROUNDED)
        vd_table.add_column("Color")
        vd_table.add_column("In Display", justify="right", style="cyan")
        vd_table.add_column("Value/Gem", justify="right", style="cyan")
        for color, count in sorted(value_counts.items()):
            value = chart.get_gem_value(count)
            vd_table.add_row(color_gem_name(color), str(count), str(value))
        console.print(vd_table)

    # Final collections
    coll_table = Table(title="Final Collections", box=ROUNDED)
    coll_table.add_column("Player", style="cyan")
    coll_table.add_column("Role", style="bold")
    coll_table.add_column("Collection")
    for pid in range(num_players):
        player = game_state.players[pid]
        col_str = _gem_counts_str(player.collection) or "Empty"
        coll_table.add_row(f"Player {pid}", model_names[pid], format_gem_string_with_colors(col_str))
    console.print(coll_table)

    winner_label = model_names[winner_id]
    if winner_id == HUMAN_PLAYER_ID:
        console.print("\n[bold green]🎉 Congratulations! You won![/bold green]")
    else:
        console.print(f"\n[bold red]{winner_label} (Player {winner_id}) wins. Better luck next time![/bold red]")


def main():
    parser = argparse.ArgumentParser(
        description="Play MegaGem as the human seat against two LLM opponents.")
    parser.add_argument('--value-chart', default='A', choices=['A', 'B', 'C', 'D', 'E'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--opponent', action='append', metavar='MODEL',
        help="opponent model id, resolved through megagem.endpoints. Repeat for "
             "the two seats; pass once to face the same model twice. "
             f"Default: {GEMINI_PRO_MODEL} and {GEMINI_FLASH_MODEL}.")
    args = parser.parse_args()

    kwargs = {}
    if args.opponent:
        opponents = list(args.opponent)
        if len(opponents) == 1:
            opponents *= 2
        elif len(opponents) > 2:
            parser.error("at most two --opponent models (seats 1 and 2)")
        kwargs["opponent_models"] = tuple(opponents)
        kwargs["opponent_names"] = tuple(opponents)

    asyncio.run(play_game(value_chart=args.value_chart, seed=args.seed, **kwargs))


if __name__ == "__main__":
    main()
