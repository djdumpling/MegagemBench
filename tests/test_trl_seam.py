"""Static seam tests for ``trl.GRPOTrainer``: rollout_func + peft_config args,
tool_mask plumbing, and per-token k3 KL formula. Skipped only if TRL isn't
installed at all. Re-run before any TRL upgrade — see ``docs/training.md``.

These tests read the TRL source via ``importlib.util.find_spec`` and parse
the file with ``ast``, never importing ``trl`` itself. That matters because
TRL's vLLM-touching modules raise at import time against vLLM < 0.12 (the
pinned fork built by ``scripts/training/setup_trl_fork.sh``; see
``docs/training.md``), which would otherwise force a skip on the H100 even
though the seam properties we want to check are static source-level facts.

TRL is deliberately absent from the local venv (it is built inside the Modal
training image), so every test here skips cleanly off-box.
"""

from __future__ import annotations

import ast
import importlib.util
import textwrap

import pytest


def _trl_grpo_trainer_module_source() -> str:
    """Return the source of ``trl/trainer/grpo_trainer.py`` without importing it."""
    try:
        spec = importlib.util.find_spec("trl.trainer.grpo_trainer")
    except ModuleNotFoundError:
        # `trl` itself is absent (the local venv has no training stack), so
        # find_spec cannot even walk to the submodule.
        spec = None
    if spec is None or spec.origin is None:
        pytest.skip(
            "trl not installed (training runs in the Modal image; see "
            "docs/training.md and scripts/training/setup_trl_fork.sh)"
        )
    with open(spec.origin) as f:
        return f.read()


def _grpo_trainer_class_node() -> tuple[str, ast.ClassDef]:
    """Return (source, ast node) for the GRPOTrainer class definition."""
    src = _trl_grpo_trainer_module_source()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer":
            return src, node
    pytest.skip("class GRPOTrainer not found in trl/trainer/grpo_trainer.py")


def _grpo_method_node(method_name: str) -> tuple[str, ast.FunctionDef]:
    """Return (module source, method node) for a method on GRPOTrainer."""
    src, cls_node = _grpo_trainer_class_node()
    for item in cls_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
            return src, item
    pytest.skip(f"GRPOTrainer.{method_name} not found")


def _grpo_init_param_names() -> set[str]:
    """Parameter names accepted by GRPOTrainer.__init__, positional + kw-only."""
    _, fn = _grpo_method_node("__init__")
    return {arg.arg for arg in fn.args.args} | {arg.arg for arg in fn.args.kwonlyargs}


def _grpo_method_source(method_name: str) -> str:
    src, fn = _grpo_method_node(method_name)
    snippet = ast.get_source_segment(src, fn)
    if snippet is None:
        pytest.skip(f"could not extract source for GRPOTrainer.{method_name}")
    return snippet


def _find_assignment_value(tree: ast.AST, target_name: str) -> ast.AST | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == target_name:
                    return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == target_name
        ):
            return node.value
    return None


def _contains_torch_call(node: ast.AST, attr_name: str) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == attr_name
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
        ):
            return True
    return False


def _contains_numeric_constant(node: ast.AST, value: int | float) -> bool:
    return any(
        isinstance(child, ast.Constant) and child.value == value
        for child in ast.walk(node)
    )


def _contains_subtraction(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.BinOp) and isinstance(child.op, ast.Sub)
        for child in ast.walk(node)
    )


def test_grpo_trainer_accepts_rollout_func():
    assert "rollout_func" in _grpo_init_param_names(), (
        "GRPOTrainer.__init__ no longer takes rollout_func — the entire "
        "external-trajectory (rollout_func) plan is broken until "
        "we either find a new seam or fork."
    )


def test_grpo_trainer_accepts_peft_config():
    assert "peft_config" in _grpo_init_param_names(), (
        "PEFT/LoRA path is required — adapter-peeling for ref logprobs is the "
        "single-H100 memory win from PEFT (ref model is None)."
    )


def test_compute_loss_applies_tool_mask():
    # Actor-mask seam rides on tool_mask × completion_mask
    # (grpo_trainer.py:2444 at the pin). If the multiplication is gone, our
    # env_mask plumbing won't actually mask gradients.
    src = _grpo_method_source("_compute_loss")
    assert "tool_mask" in src, "tool_mask not referenced in _compute_loss"
    assert "completion_mask" in src, "completion_mask not referenced in _compute_loss"


def test_compute_loss_has_k3_kl_term():
    # Per-token k3 estimator (Schulman 2020):
    #     k3 = exp(log_ratio) - log_ratio - 1     (or expm1(log_ratio) - log_ratio)
    # The "- 1" term distinguishes k3 from k1 = log_ratio. If TRL ever quietly
    # switches estimators, our nats-per-token KL telemetry stops being
    # meaningful (the RL plan §A.3) — this check is the canary.
    src = _grpo_method_source("_compute_loss")

    has_per_token_ref = "ref_per_token_logps" in src or "ref_per_token_log_probs" in src
    assert has_per_token_ref, "per-token reference logprobs missing from _compute_loss"

    assert "per_token_kl" in src, "per_token_kl variable missing — KL may be sample-level now"

    tree = ast.parse(textwrap.dedent(src))
    assignment = _find_assignment_value(tree, "per_token_kl")
    assert assignment is not None, "per_token_kl is referenced but never assigned"

    uses_exp_form = (
        _contains_torch_call(assignment, "exp")
        and _contains_subtraction(assignment)
        and _contains_numeric_constant(assignment, 1)
    )
    uses_expm1_form = _contains_torch_call(assignment, "expm1")
    assert uses_exp_form or uses_expm1_form, (
        f"per_token_kl assignment does not look like a k3 estimator "
        "(neither torch.exp(...) - ... - 1 nor torch.expm1 form): "
        f"{ast.unparse(assignment)!r}"
    )
