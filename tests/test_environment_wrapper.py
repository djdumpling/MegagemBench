"""Regression tests for the verifiers single-agent self-play wrapper
(``megagem.environment.rl_selfplay``, formerly the standalone
``environments/megagem`` package)."""

from types import SimpleNamespace

import pytest

from megagem.environment import rl_selfplay as wrapper


def test_max_turns_can_be_overridden():
    env = wrapper.load_environment(num_seeds=1, max_turns=5)

    assert env.max_turns == 5


def test_max_turns_defaults_to_full_game():
    env = wrapper.load_environment(num_seeds=1)

    assert env.max_turns == wrapper.MAX_ROUNDS * 2


@pytest.mark.asyncio
async def test_setup_uses_game_prompt_and_env_response_returns_messages():
    env = wrapper.load_environment(num_seeds=1)
    state = {"info": {"seed": 1}, "client": None, "model": ""}

    await env.setup_state(state)
    response = await env.env_response(
        [{"role": "assistant", "content": '{"bid": 0}'}],
        state,
    )

    # The wrapper emits plain chat dicts, not verifiers message objects.
    assert isinstance(state["prompt"][0], dict)
    assert state["prompt"][0]["role"] == "user"
    assert '"turn_context"' in state["prompt"][0]["content"]
    assert "MegaGem game seed" not in state["prompt"][0]["content"]
    assert isinstance(response, list)
    assert all(
        isinstance(message, dict) and message["role"] == "user"
        for message in response
    )


@pytest.mark.asyncio
async def test_opponents_use_verifiers_client_interface():
    env = wrapper.load_environment(num_seeds=1)

    class FakeClient:
        def __init__(self):
            self.calls = 0

        async def get_response(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                message=SimpleNamespace(
                    content='{"bid": 0}',
                    reasoning_content=None,
                )
            )

    client = FakeClient()
    state = {
        "info": {"seed": 1},
        "client": client,
        "model": "test-model",
        "sampling_args": {"max_tokens": 32},
    }
    await env.setup_state(state)

    await env.env_response(
        [{"role": "assistant", "content": '{"bid": 0}'}],
        state,
    )

    assert client.calls >= 2
