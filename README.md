# MegaGem

A 3-player, general-sum auction game with private information — inspired by a
Jane Street game — plus the full pipeline used to train a 4B-parameter
specialist that plays it: SFT on frontier-teacher games, GRPO self-play (an
instructive negative result), an analytic paced bid selector, and LoRA
distillation back into the weights.

**Blog post**: [Learning MegaGem, from self-play to price discovery](https://djdumpling.github.io/2026/07/25/my-draft.html)
· **Model**: [`djdumpling/qwen3-4b-instruct-megagem-distill-para`](https://huggingface.co/djdumpling/qwen3-4b-instruct-megagem-distill-para)
· **Dataset**: [`djdumpling/megagem_sft`](https://huggingface.co/datasets/djdumpling/megagem_sft)
· **License**: Apache-2.0

## Game overview

Players compete to earn the most coins by winning gem auctions, completing
mission cards, and building gem collections whose value depends on the shared
Value Display.

- **Private information**: each player holds a hand of gem cards only they can see
- **Simultaneous bidding**: sealed bids each round; highest wins (deterministic tiebreak)
- **Gem reveals**: treasure winners reveal a gem from their hand, shifting values for everyone
- **Dynamic values**: a gem color's value depends on how many of it are in the display
- **Missions**: the first player to qualify claims a mission exclusively
  (auto-claimed by the engine for the treasure winner)

The game ends when all auctionable gems have been won or the decks are
exhausted. Engine-exact rules — including known deviations from the published
game — are documented in [`docs/rules.md`](docs/rules.md).

### Value charts

Value per gem is set by how many of that color sit in the shared Value
Display. The last column is "5 or more" — with six of each color in the game, a
sixth copy in the display would mean no player holds that color at all.

| Chart | 0 | 1 | 2 | 3 | 4 | 5+ | Shape |
| ----- | - | - | - | - | - | -- | ----- |
| A (default) | 0 | 4 | 8 | 12 | 16 | 20 | linear, rising |
| B | 20 | 16 | 12 | 8 | 4 | 0 | inverse linear |
| C | 0 | 2 | 5 | 9 | 14 | 20 | convex, accelerating |
| D | 20 | 18 | 15 | 11 | 6 | 0 | inverse, decaying faster |
| E | 0 | 4 | 10 | 18 | 6 | 0 | threshold, peaks at 3 |

### Auction types

1. **Treasure (1 gem / 2 gems)** — winner pays their bid, receives the revealed gem(s)
2. **Loan 10/20** — winner receives coins now, pays the face value back at game end (overbidding beyond current coins is legal)
3. **Investment +5/+10** — winner's bid is locked until game end, then returned with the bonus

### Action formats

```json
{"bid": 15}          // bidding phase (sealed, simultaneous)
{"reveal": "Red"}    // treasure-winner reveal phase
```

Missions are claimed automatically by the engine when the treasure winner
qualifies — there is no mission action turn (see
[`docs/rules.md`](docs/rules.md)).

## Quickstart

```bash
uv sync
export PRIME_API_KEY="your-api-key"    # Prime Inference gateway (all API models)

# One game, three frontier seats
uv run megagem-run --model anthropic/claude-opus-4.6 \
                   --model google/gemini-3-pro-preview \
                   --model openai/gpt-5.2

# Options
uv run megagem-run --value-chart B --seed 123 --output my_game_log.txt
```

`megagem-run` prints round-by-round tables (bids, collections, coin deltas)
and writes a full schema-v3 trajectory. Model → endpoint routing lives in
`megagem.endpoints`, derived from the model registry in
`megagem.evals.model_mapping`.

### Programmatic use

MegaGem is a multi-agent environment; the rollout takes one client per seat:

```python
import asyncio
from openai import AsyncOpenAI
from megagem import load_environment

client = AsyncOpenAI(api_key="your-prime-key",
                     base_url="https://api.pinference.ai/api/v1")
env = load_environment(num_players=3, value_chart_id="A")

async def run():
    completion, state = await env.rollout(
        client=client, model="anthropic/claude-sonnet-4.5", prompt=[],
        clients=[client] * 3,
        models=["anthropic/claude-sonnet-4.5"] * 3,
    )
    return state

state = asyncio.run(run())
print(f"Winner: Player {state['winner_id']}")
```

To pit a locally served checkpoint against API models, serve it with vLLM and
add the served name to `_LOCAL_MODELS` in `megagem/endpoints.py`. The package
also doubles as a verifiers environment (`prime env install .`).

## Play against the trained model

The deployable policy is the merged distill-para weights **plus** the
dynamics-aware EV selector (certified config: pacing λ=0.5, value de-bias
δ=2.0, EV gate=1.0). The selector post-processes treasure bids analytically —
no extra LLM calls. Its artifacts ship in `src/megagem/assets/`, so play works
from a clean clone with no volume setup:

```bash
uvx modal setup                                   # one-time Modal auth

uvx modal run -i modal_play.py::play_distilled            # play (H100)
uvx modal run -i modal_play.py::play_distilled --once     # single game
uvx modal run -i modal_play.py::play_distilled --loop     # continuous games
uvx modal run -i modal_play.py::play_distilled --seed 123 --value-chart B
uvx modal run -i modal_play.py::play_distilled --weights-only  # no selector
uvx modal run modal_play.py::check_selector       # CPU artifact check, no GPU
```

The `-i` flag attaches your terminal to the in-container `input()` prompts.
`modal_play.py` is a standalone app (own vLLM image, no training secrets);
the model is public, so no HF token is needed. When you exit, the GPU is
released; weight and compile caches persist in Modal volumes.

You can also point the local CLI at any OpenAI-compatible endpoint serving the
model: `megagem-play-distilled --endpoint <url>`. To play against other API
models instead, `megagem-play --opponent <model-id>` (repeat for the second
seat).

## Training pipeline

Full runbook: [`docs/training.md`](docs/training.md). In brief:

1. **SFT** — 150 teacher games (70% Gemini 3 Flash / 30% Claude Opus 4.6) →
   [`djdumpling/megagem_sft`](https://huggingface.co/datasets/djdumpling/megagem_sft),
   trained with prime-rl (`configs/sft/megagem.toml`) →
   `…-sft-step1200-v2` (benchmark rating 551 → 1158).
2. **GRPO self-play** — TRL fork + external vLLM on Modal H100/H200
   (`modal_train.py` → `scripts/training/phase3_grpo.py`). Anchored
   configurations stayed flat: an honest negative result, analyzed in the blog.
3. **Bid model + value head** — gradient-boosted opponent price law F̂ and a
   supervised gem-value estimator V̂ (`scripts/analysis/`, `megagem/assets/`).
4. **Paced selector** — expected-surplus bidding with liquidity pacing;
   certified at λ=0.5/δ=2.0/gate=1.0 by the dynamics simulator (+12.16 points
   where myopic EV maximization *lost* 5).
5. **Distillation** — ~4,700 selector decisions from 400 mixed-policy games,
   LoRA-distilled back into the weights (`scripts/training/distill_*.py`) →
   `…-distill-para` (+7.97 weights-only, +11.99 with selector).

## Evaluation

Methodology: [`docs/eval.md`](docs/eval.md). The rating of record comes from a
Steiner-triple-system tournament (13 models × 624 games, every pair meeting
equally often) fitted with a Plackett–Luce model
(`modal_eval.py::bibd_eval_main`, `scripts/eval/plackett_luce_eval.py`). The
distilled 4B specialist rates 1275 — above Claude Sonnet 4.6, Claude Opus 4.8,
and GPT-5.5 on this benchmark; statistically unresolved vs Gemini 3.1 Pro.

## Environment arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_players` | int | `3` | Number of players (supports 3–5) |
| `value_chart_id` | str | `"A"` | Value chart (A–E) |
| `seed` | int | `42` | Random seed for game setup |
| `player_to_evaluate` | int | `0` | Which player's reward to track |

## Metrics

| Metric | Weight | Description |
| ------ | ------ | ----------- |
| `reward_winner` | 0.5 | 1.0 if the player won, else 0.0 |
| `reward_final_score` | 0.3 | Normalized final score (player/max) |
| `reward_normalized_rank` | 0.2 | Rank-based (1st=1.0, last=0.0) |

## State output

```python
{
    "winner_id": int,
    "final_scores": [
        {"player_id": int, "coins": int, "gem_value": int,
         "mission_rewards": int, "loan_payments": int,
         "investment_returns": int, "final_score": int},
        ...
    ],
    "num_rounds": int,
    "game_events": [...],
    "player_messages": [...],
    "models_used": [...],
}
```

## Project structure

```
MegagemBench/
├── src/megagem/              # The installable package
│   ├── __init__.py           #   load_environment() — public API
│   ├── rollout.py            #   game engine + megagem-run CLI
│   ├── endpoints.py          #   model → API endpoint routing
│   ├── game/                 #   state, actions, rules engine
│   ├── data/                 #   card data: gems, auctions, missions, value charts
│   ├── environment/          #   MegaGemEnv, prompts, rewards, piKL search, EV selector
│   ├── rl/                   #   GRPO reward path, advantages, diagnostics
│   ├── training/             #   GRPO harness, opponent pool, SFT prep, dual gate
│   ├── evals/                #   model registry, BIBD scheduling, game runner
│   ├── value_head/           #   gem-value estimator (+ trainer)
│   ├── play/                 #   interactive play CLIs
│   ├── toy/                  #   phase-2 toy auction environment
│   └── assets/               #   selector artifacts (pkl) + certified sweep
├── modal_common.py           # shared Modal image/app/volumes
├── modal_train.py            # GRPO training + eval gates
├── modal_eval.py             # panels, BIBD frontier benchmark
├── modal_release.py          # HF upload + end-to-end verify
├── modal_play.py             # standalone interactive-play app
├── scripts/
│   ├── training/             # phase-3 GRPO / phase-2 gates / distillation drivers
│   ├── eval/                 # baselines, paired evals, Plackett–Luce ratings
│   └── analysis/             # bid-model fits, dynamics simulator, rollout inspector
├── configs/sft/              # prime-rl SFT config (seed splits live in code)
├── docs/                     # rules.md, training.md, eval.md
└── tests/                    # pytest suite (runs on a clean clone)
```

## Artifacts

| Artifact | What it is |
| -------- | ---------- |
| [`…-megagem-distill-para`](https://huggingface.co/djdumpling/qwen3-4b-instruct-megagem-distill-para) | Final merged weights (SFT + distillation) |
| [`…-megagem-sft-step1200-v2`](https://huggingface.co/djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2) | SFT blueprint checkpoint |
| [`megagem_sft`](https://huggingface.co/datasets/djdumpling/megagem_sft) | Teacher-game SFT dataset |
| `src/megagem/assets/ev_dist*.pkl` | Frozen opponent price laws (F̂) |
| `src/megagem/assets/value_head.pkl` | Gem-value estimator (V̂) |
| `src/megagem/assets/dynamics_sim_sweep_recovered_200.json` | Selector certification sweep |

## License & attribution

Apache-2.0 (see [LICENSE](LICENSE)). MegaGem is an independent reimplementation
inspired by a Jane Street game; this project is not affiliated with or endorsed
by Jane Street. The engine deviates from the published game in documented ways
— see [`docs/rules.md`](docs/rules.md).
