# Evaluation

How MegaGem policies are measured, from a single local game to the
Plackett–Luce rating of record (1275 for the distilled specialist — see the
[blog post](https://djdumpling.github.io/2026/07/25/my-draft.html) for the
full leaderboard and analysis).

## Local quick checks

- **One game**: `megagem-run --model <id> --model <id> --model <id> --seed 42`
  writes a full schema-v3 trajectory. Model numbers from
  `megagem.evals.model_mapping` work too (`--model-number 13`).
- **Reward decomposition**: `uv run python scripts/analysis/inspect_rollouts.py <dump-dir>` re-derives
  every reward term from the same `megagem.rl.{reward,advantage,scorer}`
  modules the trainer uses — a mismatch is a real bug, not a display issue.

## Baselines and paired gates

- **Qwen baseline panels**: `scripts/eval/eval_qwen_baseline.py` (+`.sh`)
  plays the base model against six opponent panels plus self-play;
  `scripts/eval/_aggregate_qwen_eval.py` aggregates.
- **§3.6 paired-bootstrap gate**: `scripts/training/phase3_eval.py` — RL
  checkpoint vs SFT blueprint on paired seeds; pass = bootstrap CI low > +2
  points. Used during training (see docs/training.md §4).
- **Paired outcome analysis**: `scripts/analysis/pikl_paired_outcomes.py`
  (control-variate adjustment for auction-resolution noise);
  `scripts/eval/eval_vs_gemini.py` is the local paired-eval driver vs API
  opponents.

## Rating of record: BIBD + Plackett–Luce

The headline numbers come from a balanced tournament over 13 models:

- **Schedule**: `megagem.evals.generate_bibd_schedule` builds a Steiner
  triple system STS(13) — 26 triplets × 8 seeds × 3 seat rotations = 624
  games, so every model pair meets equally often and seat effects cancel.
- **Runner**: `modal run modal_eval.py::bibd_eval_main` serves the three
  local models (base / SFT / distilled) on one GPU alongside the API panel
  via the PRIME gateway, with resumable game files on the results volume.
  Pre-flight: `modal run modal_eval.py::check_api_models_main` (API ids) and
  `serve_local_smoke_main` (local co-serving).
- **Fit**: `scripts/eval/plackett_luce_eval.py` fits a Plackett–Luce model
  over the 3-way finish orders by majorization–minimization
  (`scripts/eval/rating_common.py` holds the shared loaders); ratings are
  reported on a 1000-anchored scale. `scripts/eval/transitivity_diagnostic.py`
  checks whether a single-scale model is even adequate (cyclic-component
  decomposition).
- **Extensions**: `top3_eval_main` plays extension games among the top
  cluster; `local_trio_eval_main` is the $0 head-to-head of base vs SFT vs
  distilled.

## Deployable vs weights-only

`modal run modal_release.py::verify_hf_model_main` serves the published HF
repo and plays it end-to-end in two configurations:

- **weights-only** — the distilled checkpoint alone;
- **deployable champion** — weights + the certified EV selector
  (F̂ level-2 price law + value head, λ=0.5, δ=2.0, gate=1.0; artifacts in
  `megagem/assets/`).

The model panel ids live in `megagem.evals.model_mapping` (the numbered
registry) and API routing in `megagem.endpoints` — endpoints are derived from
the registry so the two cannot drift.
