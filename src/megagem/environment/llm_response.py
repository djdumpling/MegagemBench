"""Reading LLM chat responses, shared by every game loop in the repo.

Deliberately depends on nothing but ``openai`` — no verifiers/datasets — so the
slim interactive-play image (see modal_play.py) can share it with the
training/eval environment. Both the multi-agent environment and the human-play
loop must interpret model output identically, or the same model appears to play
differently depending on which loop is driving it.
"""

from __future__ import annotations

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

# Worth retrying: rate limits, timeouts, connection resets, 5xx. Anything else
# (400 bad request, 401 auth, unknown model) is misconfiguration and should
# surface rather than be absorbed into a default action.
API_TRANSIENT_ERRORS = (
    RateLimitError, APITimeoutError, APIConnectionError, InternalServerError,
)


def reconstruct_content(message) -> str:
    """Full assistant text from a chat-completion message.

    Thinking models served by vLLM put the ``<think>`` block in a separate
    ``reasoning_content`` field, leaving ``content`` empty. Fold it back in, or
    the action parsers see "" and fall through to a default action (bid 0).
    """
    content = getattr(message, "content", None) or ""
    reasoning = getattr(message, "reasoning_content", None)
    return f"<think>{reasoning}</think>{content}" if reasoning else content


def response_content(response) -> str:
    """``reconstruct_content`` of the first choice, or "" if the response is
    empty/malformed (guards the unchecked ``choices[0]`` index)."""
    if response and getattr(response, "choices", None) and response.choices[0].message:
        return reconstruct_content(response.choices[0].message)
    return ""
