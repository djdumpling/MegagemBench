#!/usr/bin/env python3
"""Phase-3 stationary heuristic opponent — OpenAI-compatible shim endpoint.

`megagem.rollout.run_game()` resolves every seat through
`megagem.endpoints.ENDPOINTS` → `AsyncOpenAI(base_url=...)`
(megagem/rollout.py:149-201); there is **no scripted-agent
hook in the real MegaGem engine**. So a genuine, *stationary, deterministic*
opponent is served as a tiny local OpenAI-compatible endpoint: zero
game-engine edits, `megagem.rollout` untouched, and — because it never samples — the
K=8 GRPO group's only stochastic seat is the trainable one (clean credit
assignment; the whole point of choosing heuristic over self-mirror).

The MegaGem engine only ever prompts a player for **two** decisions
(src/megagem/environment/prompts.py): a **bid** (`{"bid": <int>}`, legal range
`0..current_auction.max_bid_for_you`) and a **reveal**
(`{"reveal": "<Color>"}`, must be a color in `your_private_hand.cards`).
Loans/investments are auction *types* you bid on; missions auto-claim — no
extra prompts. The prompt embeds the game state as compact one-object JSON
sections (`_json_section`), so the shim reads `action_request`,
`current_auction` and `your_private_hand` straight out of the user message.

Policy (v1.1 — minimal-but-legal, deterministic-per-state, stationary):
  * reveal  → the lexicographically-first gem color in hand (always legal;
              fully deterministic).
  * bid     → Treasure: ``floor((BID_FRACTION + ε) * max_bid_for_you)`` clamped
              to ``[0, max_bid_for_you]``; Loan / Investment: ``0`` (decline).
  `BID_FRACTION` is a fixed env constant — *chosen once per run*, never
  resampled. `ε` is a per-auction noise term: when
  ``PHASE3_HEURISTIC_BID_NOISE > 0`` we derive ``ε`` deterministically from a
  hash of the auction descriptor so the *same auction-state* always produces
  the *same bid* (preserves the K=8 GRPO group's "only the trainable seat is
  stochastic" credit-assignment property — all 8 K-samples at the same
  decision point see the same heuristic response), while *different* auctions
  across a run see different fractions in ``[BID_FRACTION - noise,
  BID_FRACTION + noise]``. With ``noise=0`` (default) the opponent is
  byte-identical to the v1 stationary heuristic.

The parser (src/megagem/game/actions.py `parse_bid`/`parse_reveal`) accepts a
whole-text JSON object via the strict path, so the completion content is the
bare final JSON object — no prose, hence no stray-brace ambiguity.

Run:  python -m megagem.training.heuristic_endpoint --port 8100
Register (done by the Phase-3 driver, mirrors megagem.evals.game_runner):
  ENDPOINTS["megagem/heuristic-v1"] = {"model": "megagem/heuristic-v1",
      "url": "http://localhost:8100/v1", "key": "EMPTY"}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVED_MODEL_NAME = "megagem/heuristic-v1"
# Stationary policy constants — read ONCE at process start, never per-request.
BID_FRACTION = float(os.environ.get("PHASE3_HEURISTIC_BID_FRACTION", "0.5"))
# Width of the deterministic-per-state ε around BID_FRACTION. 0.0 ⇒ pure-v1
# stationary heuristic (byte-identical to pre-v1.1 behaviour). 0.15 ⇒ fraction
# spans [0.35, 0.65] across distinct auction states, broadening the bid range
# the policy must handle while keeping each K-group internally identical.
BID_FRACTION_NOISE = float(os.environ.get("PHASE3_HEURISTIC_BID_NOISE", "0.0"))


def _epsilon_for_auction(auction: dict) -> float:
    """Deterministic per-auction-state ε ∈ [-BID_FRACTION_NOISE,
    +BID_FRACTION_NOISE]. SHA-256 of the canonical auction descriptor →
    uniform-mapped fractional offset. Identical auction descriptors map to
    identical ε; this is what preserves the K-group credit-assignment
    property (all 8 K-samples at the same decision-point see the same
    heuristic response). With BID_FRACTION_NOISE=0.0 returns 0.0 cheaply
    without hashing.
    """
    if BID_FRACTION_NOISE <= 0.0:
        return 0.0
    try:
        key = json.dumps(auction, sort_keys=True, default=str).encode()
    except (TypeError, ValueError):
        return 0.0
    h = int(hashlib.sha256(key).hexdigest()[:8], 16)
    u = h / 0xFFFFFFFF  # uniform [0, 1]
    return (u - 0.5) * 2.0 * BID_FRACTION_NOISE


def _extract_section(prompt: str, name: str):
    """Return the payload of the compact ``{"<name>":<payload>}`` section
    that ``megagem.environment.prompts._json_section`` emitted, or ``None``.

    Each section is a standalone top-level compact JSON object, so a single
    ``raw_decode`` from the section's opening brace is exact and robust to
    the other sections concatenated around it.
    """
    needle = '{"' + name + '":'
    i = prompt.find(needle)
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(prompt[i:])
    except json.JSONDecodeError:
        return None
    return obj.get(name)


def _user_text(messages) -> str:
    """The game prompt is the last user message (system = static rules)."""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _heuristic_bid(auction: dict | None) -> int:
    if not isinstance(auction, dict):
        return 0
    try:
        max_bid = int(auction.get("max_bid_for_you") or 0)
    except (TypeError, ValueError):
        return 0
    if max_bid <= 0:
        return 0
    reason = str(auction.get("bid_limit_reason") or "")
    # Decline non-Treasure auctions (Loan/Investment) — deterministic, legal.
    if "Loan" in reason or "Investment" in reason:
        return 0
    fraction = BID_FRACTION + _epsilon_for_auction(auction)
    fraction = max(0.0, min(1.0, fraction))
    bid = int(fraction * max_bid)
    return max(0, min(bid, max_bid))


def _heuristic_reveal(hand: dict | None) -> str:
    cards = []
    if isinstance(hand, dict):
        cards = [c for c in (hand.get("cards") or []) if isinstance(c, str)]
    if not cards:
        # No hand ⇒ engine does not require a card; "" parses then the env
        # legal-checks/handles it. Still deterministic.
        return ""
    return sorted(cards)[0]


def decide(messages) -> str:
    """Map an OpenAI chat request to the bare final-JSON completion string.

    Pure function (no I/O) so the legality contract is unit-testable without
    standing up the HTTP server.
    """
    prompt = _user_text(messages)
    req = _extract_section(prompt, "action_request") or {}
    action = req.get("action")
    if action == "reveal":
        return json.dumps({"reveal": _heuristic_reveal(
            _extract_section(prompt, "your_private_hand"))})
    # default to bid (the only other prompted action)
    return json.dumps({"bid": _heuristic_bid(
        _extract_section(prompt, "current_auction"))})


def _chat_completion_envelope(content: str, model: str) -> dict:
    return {
        "id": f"heuristic-{int(time.time() * 1e6)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or SERVED_MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                  "total_tokens": 0},
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence per-request stderr spam
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": SERVED_MODEL_NAME, "object": "model",
                 "owned_by": "megagem-heuristic"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            content = decide(req.get("messages"))
            self._send(200, _chat_completion_envelope(
                content, req.get("model")))
        except Exception as e:  # never 500 a rollout — emit a legal default
            self._send(200, _chat_completion_envelope(
                json.dumps({"bid": 0}), SERVED_MODEL_NAME))
            self.log_error("heuristic shim error -> bid 0 default: %s", e)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[heuristic] {SERVED_MODEL_NAME} on http://{args.host}:{args.port}"
          f"  (BID_FRACTION={BID_FRACTION}, "
          f"BID_NOISE={BID_FRACTION_NOISE}, stationary)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
