"""P0.2 probe of ``megagem.training.grpo_harness`` — the rollout no longer
fabricates logprobs, and the genuine TRL-merge contract is independent of
whatever the logprobs field carries.

What this CPU probe proves (and what it deliberately does NOT):

* PROVES (CPU): the default mode emits no fabricated logprobs (field-level
  ``None``); the legacy ``zeros`` mode is still exactly reproducible (so an
  on-box v5.x cadence-sweep A/B is possible); and the positionally-aligned
  fields TRL actually merges (``prompt_ids``/``completion_ids``/
  ``precomputed_reward``/``precomputed_advantage``; grpo_trainer.py L2130/L1198)
  are **byte-identical across logprobs modes** — i.e. the verified rollout
  contract does not depend on the logprobs field at all.
* DOES NOT PROVE: that provided logprobs are loss-invariant under the pinned
  TRL's *current* config. That requires the GPU box (real TRL forward). This
  file is the harness vehicle for that on-box check; docs/training.md records
  the on-box read that motivates the default.
"""

from __future__ import annotations

import pytest

import megagem.training.grpo_harness as H

_K = 4
_ALIGNED = ("prompt_ids", "completion_ids",
            "precomputed_reward", "precomputed_advantage")


class _FakeTok:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [min(255, ord(c)) for c in text[:24]] or [0]}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in msgs)


def _rollout(prompts):
    rf = H.make_toy_rollout_func(_FakeTok(), trainable_policy_fn=lambda _p: "",
                                 expected_k=_K, stub_strategy="bestresp")
    return rf(prompts=prompts)


def _gen():
    base = H.toy_turn_dataset_prompts([5100, 5101], _K)
    return [p for p in base for _ in range(2)]  # RepeatSampler ×G order


def test_default_mode_emits_no_fabricated_logprobs(monkeypatch):
    monkeypatch.delenv("PHASE2_ROLLOUT_LOGPROBS", raising=False)
    out = _rollout(_gen())
    assert "logprobs" in out               # key present (contract)
    assert out["logprobs"] is None         # but no fabricated values


def test_zeros_mode_is_exactly_reproducible_for_on_box_ab(monkeypatch):
    monkeypatch.setenv("PHASE2_ROLLOUT_LOGPROBS", "zeros")
    gen = _gen()
    out = _rollout(gen)
    lp = out["logprobs"]
    assert isinstance(lp, list) and len(lp) == len(gen)
    assert all(len(c) == len(l) for c, l in zip(out["completion_ids"], lp))
    assert all(all(x == 0.0 for x in row) for row in lp)


def test_aligned_contract_is_independent_of_logprobs_mode(monkeypatch):
    """The genuine TRL-merge contract (the only thing that drives training)
    must be byte-identical whether logprobs is None or the legacy zeros."""
    gen = _gen()
    monkeypatch.setenv("PHASE2_ROLLOUT_LOGPROBS", "none")
    a = _rollout(gen)
    monkeypatch.setenv("PHASE2_ROLLOUT_LOGPROBS", "zeros")
    b = _rollout(gen)
    for k in _ALIGNED:
        assert a[k] == b[k], f"{k} differs across logprobs modes"
    assert a["logprobs"] is None
    assert b["logprobs"] is not None


def test_real_mode_is_scaffolded_not_silently_fabricated(monkeypatch):
    monkeypatch.setenv("PHASE2_ROLLOUT_LOGPROBS", "real")
    with pytest.raises(NotImplementedError, match="on-policy generation-time"):
        _rollout(_gen())


def test_unknown_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("PHASE2_ROLLOUT_LOGPROBS", "bogus")
    with pytest.raises(ValueError, match="unknown PHASE2_ROLLOUT_LOGPROBS"):
        _rollout(_gen())
