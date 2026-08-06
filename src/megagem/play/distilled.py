"""Play MegaGem against two seats backed by a remote distilled-model endpoint.

Set the endpoint URL and, for a protected public server, its bearer token:

    export MEGAGEM_API_URL="https://<gpu-endpoint>/v1"
    export MEGAGEM_API_KEY="<shared-secret>"  # optional for localhost
    megagem-play-distilled --seed 123

(or: python -m megagem.play.distilled)

``MEGAGEM_API_URL`` may include ``/v1`` or just be the endpoint root URL.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
from pathlib import Path

from openai import AsyncOpenAI

from megagem.assets import asset_path
from megagem.play.interactive import play_game


DEFAULT_MODEL = "megagem-distilled"
DEFAULT_EV_MODEL_PATH = str(asset_path("ev_dist_l2_v1.pkl"))
DEFAULT_VALUE_HEAD_PATH = str(asset_path("value_head.pkl"))


def normalize_openai_base_url(url: str) -> str:
    """Accept either an endpoint root URL or an already-normalized /v1 URL."""
    normalized = url.strip().rstrip("/")
    if not normalized:
        return ""
    if not normalized.endswith("/v1"):
        normalized += "/v1"
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play MegaGem against two copies of the distilled model"
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("MEGAGEM_API_URL", "http://127.0.0.1:8000/v1"),
        help="GPU endpoint root or OpenAI base URL (env: MEGAGEM_API_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("MEGAGEM_API_KEY", ""),
        help="Optional endpoint bearer token (env: MEGAGEM_API_KEY)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Disable the EV selector and play against the distilled weights alone",
    )
    parser.add_argument(
        "--ev-model-path",
        default=os.getenv("MEGAGEM_EV_MODEL_PATH", DEFAULT_EV_MODEL_PATH),
        help="Canonical opponent-price artifact used by the EV selector",
    )
    parser.add_argument(
        "--ev-value-head-path",
        default=os.getenv("MEGAGEM_EV_VALUE_HEAD_PATH", DEFAULT_VALUE_HEAD_PATH),
        help="Canonical value-head artifact used by the EV selector",
    )
    parser.add_argument(
        "--value-chart", default="A", choices=["A", "B", "C", "D", "E"]
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="First game seed; default is random"
    )
    replay = parser.add_mutually_exclusive_group()
    replay.add_argument(
        "--once", action="store_true", help="Play one game without a replay prompt"
    )
    replay.add_argument(
        "--loop", action="store_true", help="Start random games continuously"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Per-request timeout in seconds; first request may include a cold start",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    endpoint = normalize_openai_base_url(args.endpoint)
    if not endpoint:
        raise SystemExit(
            "Missing endpoint. Set MEGAGEM_API_URL or pass --endpoint."
        )
    api_key = args.api_key.strip() or "EMPTY"

    bid_selector = None
    if not args.weights_only:
        artifact_paths = [Path(args.ev_model_path), Path(args.ev_value_head_path)]
        missing = [str(path) for path in artifact_paths if not path.is_file()]
        if missing:
            raise SystemExit(
                "The deployable selector is enabled, but its artifact(s) are "
                f"missing: {', '.join(missing)}. The defaults ship with the "
                "megagem package (megagem/assets/); pass the two --ev-*-path "
                "options for custom artifacts, or use --weights-only."
            )
        from megagem.environment.ev_selector import EvDistSelector

        bid_selector = EvDistSelector(
            model_path=args.ev_model_path,
            value_head_path=args.ev_value_head_path,
            gate_min_ev=1.0,
            pacing_lam=0.5,
            vhat_debias=2.0,
        )

    # The two model seats share one connection pool. Their bid requests are
    # issued concurrently, allowing vLLM to batch them on the H100.
    async with AsyncOpenAI(
        api_key=api_key,
        base_url=endpoint,
        timeout=args.timeout,
        max_retries=4,
    ) as client:
        try:
            available = {model.id for model in (await client.models.list()).data}
        except Exception as exc:
            raise SystemExit(f"Could not connect to {endpoint}: {exc}") from exc
        if args.model not in available:
            raise SystemExit(
                f"Endpoint does not serve {args.model!r}; available models: "
                f"{sorted(available)}"
            )
        print(f"Connected to {args.model} at {endpoint}")
        if bid_selector is None:
            print("Opponent policy: distilled weights only")
        else:
            print(
                "Opponent policy: distilled weights + canonical EV selector "
                "(F-hat level 2, pacing=0.5, de-bias=2, gate=1)"
            )

        seed = args.seed
        while True:
            if seed is None:
                seed = secrets.randbelow(2**31)
            await play_game(
                value_chart=args.value_chart,
                seed=seed,
                client=client,
                opponent_models=(args.model, args.model),
                opponent_names=("Distilled A", "Distilled B"),
                matchup_title=(
                    "Human vs 2× MegaGem Distilled + EV Selector"
                    if bid_selector is not None
                    else "Human vs 2× MegaGem Distilled (weights only)"
                ),
                request_kwargs={
                    "extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                },
                bid_selector=bid_selector,
                selector_seats=(1, 2) if bid_selector is not None else (),
            )

            if args.once:
                break
            if not args.loop:
                try:
                    answer = await asyncio.to_thread(
                        input, "\nPlay again with a new random seed? [Y/n] "
                    )
                except (EOFError, KeyboardInterrupt):
                    break
                if answer.strip().lower() in {"n", "no"}:
                    break
            seed = None


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
