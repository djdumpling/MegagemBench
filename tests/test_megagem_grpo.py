"""Tests for the MegaGem GRPO advantage-replacement seam (the RL plan I5)."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
trl = pytest.importorskip("trl", reason="TRL optional RL stack is not installed.")

from megagem.training.megagem_grpo import (  # noqa: E402
    MegaGemGRPOTrainer,
    precomputed_reward_func,
)


class _ToyModel:
    training = True


class _ToyAccelerator:
    num_processes = 1

    def gather(self, tensor):
        return tensor


def _fresh_toy_trainer() -> MegaGemGRPOTrainer:
    """Create a trainer shell without loading models or tokenizers."""
    trainer = MegaGemGRPOTrainer.__new__(MegaGemGRPOTrainer)
    trainer.model = _ToyModel()
    trainer._metrics = defaultdict(lambda: defaultdict(list))
    trainer.accelerator = _ToyAccelerator()

    # Minimal attributes consumed by TRL's inherited _compute_loss.
    trainer.top_entropy_quantile = 1.0
    trainer.off_policy_mask_threshold = None
    trainer.importance_sampling_level = "token"
    trainer.beta = 0.0
    trainer.loss_type = "bnpo"
    trainer.epsilon_low = 0.2
    trainer.epsilon_high = 0.2
    trainer.args = SimpleNamespace(delta=None)
    trainer.use_vllm = False
    trainer.vllm_importance_sampling_correction = False
    trainer.current_gradient_accumulation_steps = 1

    def fake_logps_and_entropies(
        _model,
        input_ids,
        _attention_mask,
        logits_to_keep,
        compute_entropy,
        **_: Any,
    ):
        assert compute_entropy is True
        shape = (input_ids.size(0), logits_to_keep)
        per_token_logps = torch.zeros(shape, requires_grad=True)
        entropies = torch.zeros(shape)
        return per_token_logps, entropies

    trainer._get_per_token_logps_and_entropies = fake_logps_and_entropies
    return trainer


def test_precomputed_reward_func_returns_rollout_rewards():
    rewards = precomputed_reward_func(
        prompts=["game-0", "game-1", "game-2"],
        completions=["a", "b", "c"],
        precomputed_reward=[1, None, -0.5],
    )
    assert rewards == [1.0, None, -0.5]


def test_precomputed_reward_func_rejects_length_mismatch():
    with pytest.raises(ValueError, match="does not match batch size"):
        precomputed_reward_func(
            prompts=["game-0", "game-1"],
            completions=["a", "b"],
            precomputed_reward=[1.0],
        )


def test_i5_two_game_toy_advantages_flow_into_compute_loss(monkeypatch):
    """The inherited TRL loss sees MegaGem's advantages, not TRL's stock ones."""
    trainer = _fresh_toy_trainer()

    def fake_base_generate_and_score(self, inputs):
        assert self is trainer
        for row, advantage in zip(inputs, [3.0, -1.0], strict=True):
            row["precomputed_advantage"] = advantage
        return {
            "prompt_ids": torch.ones((2, 1), dtype=torch.long),
            "prompt_mask": torch.ones((2, 1), dtype=torch.long),
            "completion_ids": torch.tensor([[2, 3], [4, 5]], dtype=torch.long),
            "completion_mask": torch.ones((2, 2), dtype=torch.long),
            "advantages": torch.zeros(2),  # stock TRL values must be replaced
            "num_items_in_batch": torch.tensor(4),
        }

    monkeypatch.setattr(
        trl.GRPOTrainer,
        "_generate_and_score_completions",
        fake_base_generate_and_score,
    )

    batch = trainer._generate_and_score_completions(
        [{"prompt": "toy-game-0"}, {"prompt": "toy-game-1"}]
    )

    assert torch.equal(batch["advantages"], torch.tensor([3.0, -1.0]))

    # With zero log-ratio and BNPO loss, TRL's inherited _compute_loss reduces
    # to the masked mean of -advantages. The custom advantages produce -1.0;
    # the fake stock zero advantages would produce 0.0.
    loss = trainer._compute_loss(model=None, inputs=batch)
    assert loss.item() == pytest.approx(-1.0)

    metrics = trainer._metrics["train"]
    assert metrics["megagem/advantages/count"] == [2]
    assert metrics["megagem/advantages/mean"] == [1.0]


def test_i5_raises_when_rollout_does_not_supply_advantages(monkeypatch):
    trainer = _fresh_toy_trainer()

    def fake_base_generate_and_score(_self, _inputs):
        return {
            "completion_mask": torch.ones((2, 2), dtype=torch.long),
            "advantages": torch.zeros(2),
        }

    monkeypatch.setattr(
        trl.GRPOTrainer,
        "_generate_and_score_completions",
        fake_base_generate_and_score,
    )

    with pytest.raises(KeyError, match="precomputed_advantage"):
        trainer._generate_and_score_completions(
            [{"prompt": "toy-game-0"}, {"prompt": "toy-game-1"}]
        )
