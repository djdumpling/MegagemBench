"""Static contract tests for modal_play.py (the standalone interactive app).

modal_play.py is never imported here (importing it would require the `modal`
client); the assertions read its source/AST, so they are the cheap guard that
the interactive path keeps its shape: one H100, a local vLLM, a real PTY, the
pinned public checkpoint, and the selector artifacts loaded from the packaged
``src/megagem/assets/`` mount (they used to come from a Modal results volume).
"""

import ast
from pathlib import Path

MODAL_PLAY = Path(__file__).resolve().parents[1] / "modal_play.py"


def test_interactive_modal_player_requests_h100_and_runs_local_vllm() -> None:
    source = MODAL_PLAY.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "play_distilled"
    )
    function_source = ast.get_source_segment(source, function) or ""
    decorators = "\n".join(
        ast.get_source_segment(source, decorator) or ""
        for decorator in function.decorator_list
    )

    assert 'gpu="H100"' in decorators
    assert "modal.interact()" in function_source
    assert '"A PTY shell is already active"' in function_source
    assert "except modal.exception.InternalError" in function_source
    assert "_serve_vllm(" in function_source
    assert '"--enforce-eager"' in source
    # The game is launched as a package module, not a repo-root script path.
    assert '"megagem.play.distilled"' in function_source
    assert '"/repo/play_vs_distilled.py"' not in function_source
    assert 'command.append("--weights-only")' in function_source
    assert "Secret.from_name" not in source


def test_modal_player_uses_public_pinned_merged_model() -> None:
    source = MODAL_PLAY.read_text()

    assert 'MODEL = "djdumpling/qwen3-4b-instruct-megagem-distill-para"' in source
    assert 'MODEL_REVISION = "cd2074eeba56e23edb2625ed58ac4b5778a990c1"' in source
    assert '"vllm==0.21.0"' in source
    assert '"scikit-learn==1.8.0"' in source
    assert "-lora" not in source


def test_modal_player_loads_selector_artifacts_from_the_repo_mount() -> None:
    """The two selector artifacts ship inside the package (megagem/assets) and
    are read from the repo mount — no Modal results volume is involved."""
    source = MODAL_PLAY.read_text()

    assert 'EV_MODEL_PATH = "/repo/src/megagem/assets/ev_dist_l2_v1.pkl"' in source
    assert 'EV_VALUE_HEAD_PATH = "/repo/src/megagem/assets/value_head.pkl"' in source
    # the retired results-volume mount must not come back
    assert 'Volume.from_name("megagem-results"' not in source
    assert "results_vol" not in source
    assert "RESULTS_DIR" not in source
    # the mount must actually carry the package sources
    assert 'PYTHONPATH": "/repo/src"' in source
    assert "gate_min_ev=1.0" in source
    assert "pacing_lam=0.5" in source
    assert "vhat_debias=2.0" in source
    assert "def check_selector()" in source


def test_modal_player_mounts_only_the_two_caches() -> None:
    """volumes = hf_cache + vllm_cache (the results volume is gone), and the
    artifacts are required before a GPU is billed for interactive play."""
    source = MODAL_PLAY.read_text()
    tree = ast.parse(source)

    volume_names = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_name"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert volume_names == {"megagem-hf-cache", "megagem-vllm-cache"}

    assert "_require_selector_artifacts()" in source


def test_modal_play_docstring_has_no_profile_prefix() -> None:
    """The run recipe is a plain `uvx modal run -i` — no MODAL_PROFILE prefix."""
    docstring = ast.get_docstring(ast.parse(MODAL_PLAY.read_text())) or ""
    assert "uvx modal run -i modal_play.py::play_distilled" in docstring
    assert "MODAL_PROFILE" not in docstring
