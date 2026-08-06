import sys

from megagem.play.distilled import (
    DEFAULT_EV_MODEL_PATH,
    DEFAULT_VALUE_HEAD_PATH,
    normalize_openai_base_url,
    parse_args,
)


def test_normalize_openai_base_url_adds_v1() -> None:
    assert (
        normalize_openai_base_url("https://gpu.example.com/")
        == "https://gpu.example.com/v1"
    )


def test_normalize_openai_base_url_preserves_v1() -> None:
    assert (
        normalize_openai_base_url(" https://gpu.example.com/v1/ ")
        == "https://gpu.example.com/v1"
    )


def test_defaults_to_local_endpoint_and_random_seed(monkeypatch) -> None:
    monkeypatch.delenv("MEGAGEM_API_URL", raising=False)
    monkeypatch.delenv("MEGAGEM_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["megagem.play.distilled"])

    args = parse_args()

    assert args.endpoint == "http://127.0.0.1:8000/v1"
    assert args.api_key == ""
    assert args.seed is None
    assert args.weights_only is False
    assert args.ev_model_path == DEFAULT_EV_MODEL_PATH
    assert args.ev_value_head_path == DEFAULT_VALUE_HEAD_PATH


def test_weights_only_is_an_explicit_opt_out(monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["megagem.play.distilled", "--weights-only"]
    )

    args = parse_args()

    assert args.weights_only is True
