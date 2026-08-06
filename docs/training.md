# Training pipeline

How the published MegaGem specialist
([`djdumpling/qwen3-4b-instruct-megagem-distill-para`](https://huggingface.co/djdumpling/qwen3-4b-instruct-megagem-distill-para))
was trained, stage by stage. The narrative version is in the
[blog post](https://djdumpling.github.io/2026/07/25/my-draft.html); this doc
maps each stage to the code that ran it.

```
teacher games ──► SFT ──► GRPO self-play (negative result)
                   │
                   └──► bid-distribution model (F̂) + value head (V̂)
                              │
                              ▼
                        paced EV selector  ──►  distillation  ──►  distill-para
                        (λ=0.5, δ=2.0, gate=1.0)
```

## 1. Supervised fine-tuning (SFT)

- **Data**: teacher games from frontier models, published as the HF dataset
  [`djdumpling/megagem_sft`](https://huggingface.co/datasets/djdumpling/megagem_sft).
  Local preprocessing: `uv run python -m megagem.training.preprocess_sft` (tokenizes and
  splits into `data/sft_processed/`, gitignored scratch). Seed splits are
  locked as contiguous ranges in `megagem.training.seed_splits.SPLIT_RANGES`
  (train 1001–1150, val 1151–1160, test 1161–1170; see `configs/sft/README.md`).
- **Trainer**: [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)'s SFT
  path with `configs/sft/megagem.toml` (`uv run sft @ configs/sft/megagem.toml`
  from a prime-rl checkout). LoRA adapters are saved separately per step.
- **Merge/publish**: `scripts/training/merge_sft_adapter.py` merges the chosen
  adapter into the base model and pushes to HF. The blueprint of record is
  `djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2`.

## 2. The TRL fork (why training needs a patched TRL)

The GRPO/distill stack pins vLLM 0.10.2, but the pinned upstream TRL commit
imports vLLM-0.12+ symbols at module load, so `from trl import GRPOTrainer`
fails. The fork is that exact public commit
(`huggingface/trl@5da60783af85a180f10235e564f37e4da67cc01d`) plus one appended
function (`is_vllm_available() -> False`) so TRL skips its unused internal
vLLM path — rollouts run against our own external vLLM server through the
`rollout_func` seam.

- The Modal training image rebuilds this fork from **public** upstream at
  image-build time (see `modal_common.py`) — no private repository or GitHub
  credential is needed anywhere.
- To reproduce the fork elsewhere: `scripts/training/setup_trl_fork.sh`.
- Contract tests: `tests/test_trl_seam.py` + `tests/test_megagem_grpo.py`
  (skip cleanly when TRL is not installed).

## 3. Phase-2 gates (toy environment)

Before spending on MegaGem GRPO, the trainer glue was validated on a toy
3-player sealed-bid auction (`megagem.toy`): `scripts/training/toy_loop.py`
driven by `scripts/training/run_phase2.sh` proves the TRL seam moves a policy
on a game with a known optimum, with cost/update smoke checks in
`scripts/training/update_cost_smoke.py`. The shared GRPO harness these gates
validated lives at `megagem.training.grpo_harness`.

## 4. Phase-3 GRPO self-play (honest negative result)

Full-game GRPO self-play on Modal H100/H200:

```
modal run modal_train.py::verify_env                    # pinned-stack sanity
modal run modal_train.py::seam_tests_main               # CPU prep gate
modal run modal_train.py::phase3_main --profile seam    # cheap GPU smoke
modal run modal_train.py::phase3_main --profile evidence  # real spend
```

`phase3_main` → `scripts/training/run_phase3.sh` → `scripts/training/phase3_grpo.py`
(rollouts via `megagem.rollout.run_game` against a local vLLM; opponent pool
with lagged snapshots in `megagem.training.opponent_pool`; per-step telemetry in
`megagem.training.run_telemetry`). Evaluation gates:

- **§3.6 paired-bootstrap eval** (`scripts/training/phase3_eval.py`): RL
  checkpoint vs the SFT blueprint, paired seeds, gate = CI low > +2 points.
- **Dual gate** (`megagem.training.dual_gate`): the §3.6 gate alone is
  permeable to opponent-overfitting, so spend decisions also require a
  held-out win-rate vs Gemini 3 Flash with a cluster-conservative one-sided
  z-test (`n_eff = num_seeds`, not games).

**Result**: configurations anchored to the frozen SFT checkpoint stayed flat —
relative outcome rewards did not produce measured transfer. The blog post
discusses why (general-sum rewards, three-player kingmaking, imperfect
information, credit assignment). The pipeline is kept runnable because the
negative result is part of the story.

## 5. Bid-distribution model, value head, and the paced selector

The improvement that *did* transfer came from an analytic expert:

- **F̂ (price law)**: gradient-boosted model of opponent bid distributions /
  win prices, fit by `scripts/analysis/flash_bid_model_fit.py` and frozen via
  `scripts/analysis/build_ev_dist_artifact.py` → packaged as
  `megagem/assets/ev_dist_v1.pkl` (level-2 refit on mixed-policy games:
  `ev_dist_l2_v1.pkl`).
- **V̂ (value head)**: supervised gem-value estimator,
  `uv run python -m megagem.value_head.train --globs '<corpus>/*.json'` →
  `megagem/assets/value_head.pkl`.
- **Paced selector** (`megagem.environment.ev_selector.EvDistSelector`):
  chooses treasure bids by expected surplus with a liquidity pacing term
  λ_coin and a value de-bias δ. Myopic EV maximization *hurt* full-game play
  (premature liquidity depletion); the pacing calibration reversed that.
- **Certification**: `scripts/analysis/dynamics_sim.py` (calibrate/sweep
  modes) certified the deployed config **λ=0.5, δ=2.0, gate=1.0**; the sweep
  of record ships as `megagem/assets/dynamics_sim_sweep_recovered_200.json`
  (plot it with `uv run python scripts/analysis/plot_dynamics_sweep.py`).

## 6. Distillation of record (distill-para)

The shipped model distills the selector's corrections back into the weights:

- **Corpus**: `scripts/training/distill_export.py` (formerly
  `distill1a_export.py`) replays ~400 mixed-policy games and emits SFT rows
  where the selector's bid deviates from the policy's (seat 0 deviations;
  seats 1/2 and financing turns are pass-through anchors) — ~4,700 bid
  decisions.
- **Trainer**: `scripts/training/distill_train.py` (formerly
  `exit_distill_train.py`): TRL `SFTTrainer` LoRA (rank 32, α=64), starting
  from the merged current policy.
- **Provenance note**: the Modal orchestration that ran the original
  `distill1a` round lived in a prior checkout of this project and is not
  in-tree; the exporter and trainer above are the components of record, and
  the resulting weights are published. The blog documents the measured
  recovery (73–78% of selector gains; +7.97 points weights-only vs SFT).
- **Publish/verify**: `modal run modal_release.py::upload_model_main` merges
  and pushes to HF; `modal run modal_release.py::verify_hf_model_main` serves
  the uploaded repo and plays full games end-to-end (weights-only and
  selector-backed).

## Where things run

| Stage | Where | Entry |
|---|---|---|
| SFT | prime-rl (any GPU box) | `uv run sft @ configs/sft/megagem.toml` |
| Phase-2 gates | local/CPU-light | `scripts/training/run_phase2.sh` |
| GRPO | Modal (H100/H200) | `modal run modal_train.py::phase3_main` |
| Selector fit/cert | local CPU | `scripts/analysis/*.py` |
| Distillation | GPU box or Modal | `scripts/training/distill_train.py` |
| Publish/verify | Modal | `modal run modal_release.py::*` |
