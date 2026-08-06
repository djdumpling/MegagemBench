# MegaGem — Rules as Implemented

This document describes the game **exactly as the engine in this repository implements it**
(`src/megagem/game/` + `src/megagem/environment/multi_agent_env.py`). Every benchmark number in
this repo refers to *this* game. Where the implementation is known to deviate from the original
published MegaGem rules, see [Engine vs. published rules](#8-engine-vs-published-rules--disclosures).

## 1. Overview

MegaGem is a general-sum, simultaneous sealed-bid auction game (benchmark default: 3 players;
the engine supports 3–5). Each player holds a **private hand** of gem cards; everything else —
coins, won collections, bids after each round, the Value Display, missions — is public. Players
bid coins in a sequence of auctions (gems, loans, investments) and win by having the highest
**final score**: coins + gem-collection value (priced by a shared Value Chart over the public
Value Display) + mission rewards − loan paybacks + investment returns.

## 2. Components

**Gems** (`src/megagem/data/gems.json`): 30 gem cards — 6 each of Red, Blue, Green, Purple, Yellow.

**Auction deck** (`src/megagem/data/auctions.json`): 25 cards, shuffled at setup. One card = one round, so a
game lasts at most 25 rounds (typically 15–20; see game end).

| Card | Count | Effect for the winner |
|---|---|---|
| Treasure (1 gem) | 12 | Pay bid; take the face-up gem in slot 0 |
| Treasure (2 gems) | 5 | Pay bid; take both face-up gems |
| Loan 10 | 2 | Receive 10 coins now; 10 is deducted at final scoring |
| Loan 20 | 2 | Receive 20 coins now; 20 is deducted at final scoring |
| Investment +5 | 2 | Pay bid (locked); receive bid + 5 at final scoring |
| Investment +10 | 2 | Pay bid (locked); receive bid + 10 at final scoring |

**Missions** (`src/megagem/data/missions.json`): 30 cards; only 4 are in play per game (see Setup).
Requirements test the player's **collection** (gems won at auction — never the hand). Claimed
gems are *not* spent; they keep scoring and can satisfy other missions.

| Mission(s) | Requirement | Reward |
|---|---|---|
| `flex_3_diff` | ≥ 3 different colors | 5 |
| `flex_2_same` | ≥ 2 of any one color | 5 |
| `flex_3_same` | ≥ 3 of any one color | 10 |
| `flex_4_diff` | ≥ 4 different colors | 10 |
| `flex_2_pairs` | ≥ 2 different colors each with ≥ 2 gems | 15 |
| `specific_3_rbg` … `specific_3_gpy` (10 cards, one per 3-color combination) | ≥ 1 gem of each of the 3 named colors | 10 |
| `specific_2_rb` … `specific_2_py` (10 cards, one per 2-color combination) | ≥ 1 gem of each of the 2 named colors | 5 |
| `same_2_red/blue/green/purple/yellow` (5 cards) | ≥ 2 gems of the named color | 5 |

**Value Charts** (`src/megagem/data/value_charts.json`): one chart is fixed for the whole game (benchmark
default: A). It maps *how many gems of a color sit in the Value Display* to that color's
per-gem value for every player's collection.

The final row is **"5 or more"**, matching the physical card. Six of each color
exist, so a display count of 6 would mean every copy is in the display and no
player holds that color — the per-gem value is then multiplied by zero gems for
everyone and cannot affect scoring. `cards.py::ValueChart.get_gem_value` clamps
any count above the table's maximum to that last row, which implements "5+".

| Display count | A (default) | B | C | D | E |
|---|---|---|---|---|---|
| 0 | 0 | 20 | 0 | 20 | 0 |
| 1 | 4 | 16 | 2 | 18 | 4 |
| 2 | 8 | 12 | 5 | 15 | 10 |
| 3 | 12 | 8 | 9 | 11 | 18 |
| 4 | 16 | 4 | 14 | 6 | 6 |
| 5+ | 20 | 0 | 20 | 0 | 0 |

Shapes: **A** linear rising · **B** inverse linear · **C** convex (accelerating)
· **D** inverse, decaying faster as copies appear · **E** threshold, peaking at 3
copies then collapsing.

## 3. Setup (`state.py::GameState.create_new_game`)

All randomness comes from one game-local `random.Random(seed)`: it shuffles the gem deck, the
auction deck, the mission deck, and the initial tiebreak order. A fixed seed reproduces the
entire setup exactly (hands are dealt from one end of the shuffled gem deck via `pop()`;
mid-game replenishment draws from the other end via `pop(0)` — deterministic either way).

| Players | Starting coins | Hand size | Gems ever auctionable |
|---|---|---|---|
| 3 (benchmark) | 35 | 5 | 30 − 15 = 15 |
| 4 | 25 | 4 | 30 − 16 = 14 |
| 5 | 20 | 3 | 30 − 15 = 15 |

- Each player is dealt their private hand; **hand gems are never auctioned** — they only ever
  move to the Value Display (via reveals or end-of-game auto-reveal).
- Two gems are turned face-up as the **revealed-gems pool** (the gems on offer in Treasure
  auctions). Both are public.
- The 30 missions are shuffled and the first 4 become `available_missions`. **Missions are
  never replenished** — at most 4 mission claims happen per game.
- Tiebreak order starts as a random permutation of seats.

Two public gem zones exist and are distinct: the **revealed-gems pool** (≤ 2 face-up gems,
what Treasure auctions sell) and the **Value Display** (gems revealed from hands, which set
prices). Won gems go to a player's **collection**, never to either zone.

## 4. Round flow (`multi_agent_env.py::_run_game_loop`)

1. **Draw** the next auction card (`state.py::draw_auction_card`). Empty deck ⇒ game over.
2. **Simultaneous sealed bids.** Every player (including the eventual winner of a loan they
   don't want) submits one bid given the public state plus their own hand. Legality
   (`actions.py::validate_bid_for_auction`): integer ≥ 0; for Treasure/Investment, bid ≤ current
   coins; for Loan, bid ≤ current coins + loan amount. An unparsable or illegal bid is replaced
   by the **default bid 0** (`get_default_bid`). Bids are sealed within the round but all three
   bids become public in the auction history afterward.
3. **Resolution** (`rules.py::resolve_auction`): the highest bid wins; among tied high bidders,
   the one **earliest in the current tiebreak order** wins. There is no pass — someone wins
   every auction, even at a unanimous bid of 0, and pays their own bid (first-price; losers pay
   nothing).
4. **Reveal — Treasure auctions only.** If (and only if) the card is a Treasure, the winner —
   knowing which gems they are about to receive — must move **one gem of their choice from
   their hand to the Value Display** (`rules.py::reveal_gem_from_hand`, applied before the
   payment/outcome step). It need not match the won color. A winner with an empty hand skips
   this. On an unparsable or not-in-hand reveal the engine plays the **default reveal**: the
   alphabetically first color in hand (Blue < Green < Purple < Red < Yellow;
   `actions.py::get_default_reveal`). Loan and Investment winners never reveal.
5. **Payment & transfer** (`rules.py::apply_auction_outcome`):
   - *Treasure*: winner pays bid; takes `revealed_gems[:n]` into their collection. A 2-gem card
     with only one gem left in the pool awards just that one gem — at the full winning bid
     (`rules.py::resolve_auction` end-of-game edge case). The pool is then replenished back up
     to 2 from the gem deck (`state.py::replenish_revealed_gems`; a held-over gem keeps slot 0,
     so in a 1-gem auction the *older* face-up gem is the one on sale, and players can see the
     next auction's gem in advance).
   - *Loan*: winner's coins += amount, then −= bid (net = amount − bid, possibly negative in
     effect but coins never go below 0 transiently); the amount is recorded for end-game payback.
   - *Investment*: winner pays bid; the bid is locked away (unusable for later bidding) and
     recorded with its bonus.
   - The winner is moved to the **end of the tiebreak order** (`state.py::update_tiebreak_order`).
6. **Mission auto-claim — Treasure winner only** (`multi_agent_env.py::run_mission_phase`).
   The engine automatically grants the winner *every* still-available mission their collection
   now satisfies. First qualifier claims exclusively (the mission leaves the pool). No player
   action is involved — the `{"complete_missions": [...]}` parser exists
   (`actions.py::parse_mission_claim`) but the game loop never invokes it.
7. **End check** (`multi_agent_env.py::_maybe_end_game`), then next round.

What players see each turn (`environment/prompts.py`): the value chart, Value Display counts,
the face-up gems on offer, remaining deck/auction counts, the current auction with their own
max legal bid, every player's coins / collection / hand *size* / loan total / investment count /
completed missions, the 4 available missions, the tiebreak order, the last 5 auctions with all
bids, and their own hand. Hidden: opponents' hand contents, gem-deck order, upcoming auction cards.

## 5. Economy

- **Coins** never go negative during play (bid validation + loan credit-before-debit ordering).
- **Loans** are interest-free advances repaid only via the −loan term at final scoring; nothing
  is repaid mid-game, and multiple loans stack. The overbid rule (max bid = coins + amount) is
  the only way to bid beyond current coins. A final score can go negative.
- **Investments** lock the winning bid until the end, then return bid + bonus — a guaranteed
  net +bonus for the winner (an uncontested 0-bid investment is free money), at the cost of
  mid-game liquidity. Locked amounts are public only via the bid history (opponents see the
  *count* of your investments, not their sizes, in the player table).
- **Gem prices** move whenever any gem enters the Value Display: each Treasure winner's reveal,
  plus the end-of-game auto-reveal of all remaining hands.

## 6. Scoring & game end

**End conditions** (both checked by the engine):

1. `multi_agent_env.py::_maybe_end_game`, after every round: the game ends when the **total
   gems won across all players' collections** reaches `num_players × hand_size` (3p: 15,
   4p: 16, 5p: 15), **or** when the gem deck and the face-up pool are both empty. With 3
   players the count threshold equals the 15 auctionable gems, so both branches say "every
   auctionable gem has been won". (With 4 players only 14 gems are auctionable, so the coded
   16-threshold is unreachable and the empty-pool branch is what actually fires.)
2. `state.py::draw_auction_card` / `is_game_over`: exhausting the 25-card auction deck ends the
   game — a hard cap of 25 rounds.

**Terminal scoring** (`rules.py::calculate_all_final_scores` → `score_components`):

1. Every remaining hand gem is auto-revealed into the Value Display
   (`state.py::reveal_remaining_hands`) — so *terminal* display counts, hands included, set all
   gem prices.
2. For each player:

   ```
   final_score = coins
               + Σ_color  collection_count(color) × chart_value(terminal_display_count(color))
               + Σ        rewards of missions the player claimed
               − Σ        loan amounts taken
               + Σ        (locked bid + bonus) per investment
   ```

3. **Winner** (`rules.py::determine_winner`): highest final score; ties go to the tied player
   earliest in the *final* tiebreak order. (Rewards in training/eval also use score margins;
   see the repo's metrics docs.)

## 7. Action formats & parse fallbacks (`actions.py`)

Players answer in free text ending with one JSON object:

| Phase | Expected JSON | On failure |
|---|---|---|
| Bid | `{"bid": 7}` | default bid **0** |
| Reveal (Treasure winner) | `{"reveal": "Blue"}` | default: alphabetically first color in hand |
| Missions | — none; auto-claimed by the engine (the documented `{"complete_missions": [...]}` format is parsed by unused code) | — |

JSON extraction tries, in order (`extract_json_with_method`): whole-message JSON
(`strict_json`) → fenced ```` ```json ```` block (`code_block`) → first parseable brace-balanced
substring (`brace_match`); `<think>…</think>` tags are stripped first. Bids additionally get a
**prose fallback** (`prose_fallback`): regexes such as "I'll bid *N*", "bid N coins",
"my bid: N" — taking the *rightmost* match in 0–200. Fractional JSON bids are truncated by
`int()`. A parsed-but-illegal action (negative bid, over-budget bid, color not in hand) falls
back to the same defaults as a parse failure, and the parse method + default usage are logged
per turn (`PARSE_METHOD_*`, `default_used`).

## 8. Engine vs. published rules — disclosures

> **This is an independent reimplementation.** All benchmark and training numbers in this repo
> refer to the engine's game exactly as documented above, which is known to differ from the
> original published MegaGem rules in at least two ways:
>
> 1. **Game end.** The engine stops as soon as all *auctionable* gems have been won — tracked
>    as total gems in all collections reaching 15 in the 3-player benchmark game (a
>    whole-table count, not a per-player one) or the deck+pool emptying — with 25 drawn
>    auction cards as a hard cap (`multi_agent_env.py::_maybe_end_game`,
>    `state.py::draw_auction_card`). Gems dealt to private hands (15 of the 30 in a 3-player
>    game) are **never auctioned**; they only ever enter the Value Display. The published game,
>    as we understand it, plays until all gems have been auctioned.
> 2. **Reveal timing.** In the engine, a from-hand reveal happens **only when a Treasure
>    auction is won** — Loan and Investment winners never reveal
>    (`multi_agent_env.py::_run_game_loop` gates the reveal phase on
>    `auction.type == AuctionType.TREASURE`). The published rules, as we understand them, have
>    the winner reveal after *every* auction.
>
> Where this document and any published rulebook disagree, this document describes what the
> code does.

*MegaGem here is an independent reimplementation inspired by a Jane Street game; this project is
not affiliated with or endorsed by Jane Street or the original game's creators.*
