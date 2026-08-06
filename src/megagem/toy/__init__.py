"""Phase 2.1 toy environment — a 3-player private-value first-price sealed-bid
auction over 8 rounds with deterministic *pure-coins* scoring.

Purpose (the RL plan Phase 2.1): exercise the **existing, unmodified** ``src/megagem/rl``
reward/advantage pipeline and the ``rollout_func → MegaGemGRPOTrainer`` seam with
a cheap game that reproduces MegaGem's failure modes (3 simultaneous players,
private per-round value, actor masks: one trainable / two heuristic, per-turn
shaping). It does NOT subclass ``MegaGemEnv`` — it mirrors the *interface
contract* via the ``run_one_toy_game`` seam (``policy_fns`` + ``actor_ids``).

The whole design rests on one audit-confirmed invariant: a pure-coins game
(novel ``auction.type == "sealed_bid"``, empty ``collection``/``value_display``,
no missions/loans/investments) makes ``score_components`` collapse to
``final_score == coins`` and the §A.4/Caveat-2 reveal sub-delta identically
``0`` — so the trajectories flow through ``src/megagem/rl`` with zero pipeline changes.

Per Caveat 5, this validates trainer mechanics + that the per-turn signal can
move a policy; it does NOT test the self-play→frozen-anchor transfer (Phase 3.6).

Nothing here imports TRL/torch, calls an LLM, uses asyncio, or touches a GPU.
"""
