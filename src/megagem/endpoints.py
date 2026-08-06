import itertools
import threading

PRIME_URL = "https://api.pinference.ai/api/v1"
PRIME_KEY = "PRIME_API_KEY"


_rr_lock = threading.Lock()
_rr_iters: dict[tuple, "itertools.cycle"] = {}


def pick_url(endpoint: dict) -> str:
    """Resolve `endpoint['url']` which may be a single str OR a list of URLs
    (DP layout — multiple vLLM workers serving the same model). For a list,
    pick round-robin over a process-wide counter keyed on the tuple of URLs
    so balance is preserved across calls. Single-URL endpoints pass through
    unchanged (API opponents, heuristic shim, single-instance vLLM)."""
    u = endpoint.get("url")
    if isinstance(u, str):
        return u
    if isinstance(u, (list, tuple)) and u:
        key = tuple(u)
        with _rr_lock:
            it = _rr_iters.get(key)
            if it is None:
                it = itertools.cycle(key)
                _rr_iters[key] = it
            return next(it)
    raise ValueError(
        f"endpoint missing usable url/list: {endpoint!r}")

# API-served eval models, derived from the canonical table in
# megagem.evals.model_mapping so the two never drift apart. The three locally
# served models (base / SFT / distilled, indices 11-13) are excluded: the SFT
# and distilled checkpoints resolve only when a vLLM endpoint is registered.
from megagem.evals.model_mapping import MODELS as _MODELS

_LOCAL_MODEL_NUMBERS = {11, 12, 13}

ENDPOINTS = {
    model_id: {"model": model_id, "url": PRIME_URL, "key": PRIME_KEY}
    for num, (model_id, _name) in _MODELS.items()
    if num not in _LOCAL_MODEL_NUMBERS
}

# Extra API-served ids not in the numbered table: the base model under its HF
# id, and the legacy cheap panel opponent used by scripts/eval/eval_vs_opponent.sh.
for _extra in ("Qwen/Qwen3-4B-Instruct-2507", "openai/gpt-5.4-nano"):
    ENDPOINTS[_extra] = {"model": _extra, "url": PRIME_URL, "key": PRIME_KEY}

# Local vLLM endpoint for fine-tuned / small models served locally.
# Override the URL when serving Qwen3-4B-Instruct via vLLM on a cloud instance.
VLLM_URL = "http://localhost:8000/v1"
VLLM_KEY = "EMPTY"

_LOCAL_MODELS = [
    "qwen/qwen3-4b-instruct",
]

ENDPOINTS.update({
    model: {"model": model, "url": VLLM_URL, "key": VLLM_KEY}
    for model in _LOCAL_MODELS
})
