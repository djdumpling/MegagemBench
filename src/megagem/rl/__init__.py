"""Offline RL reward / advantage validation (the RL plan Phase 1).

Pure-Python, CPU-only. Operates on saved schema-v3 game transcripts; computes
the non-mutating mid-game scorer (§A.4), terminal + legal-action rewards
(§A.0/§A.2), per-turn return-to-go advantages with a role-conditioned EMA
baseline and within-group normalization (§A.5), and the per-row training-row
contract consumed by ``MegaGemGRPOTrainer`` (Phase 2/3).

Nothing in this package imports TRL or touches a GPU.
"""
