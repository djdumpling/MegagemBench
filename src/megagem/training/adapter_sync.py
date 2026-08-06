#!/usr/bin/env python3
"""On-policy LoRA adapter → rollout-vLLM sync (Phase-3 per-generation refresh).

Heuristic opponents mean the **only** model in a rollout is the trainable
seat, so "on-policy" reduces to: before each generation's roll, the rollout
vLLM must serve the *current* LoRA adapter. This is the simple precursor to
3.3's lagged pool (one adapter, no anneal, no pool).

Mechanism (vLLM 0.10.2 dynamic LoRA):
  1. `save_step_adapter(model, step, root)` — peft `save_pretrained` of just
     the adapter (cheap; base weights untouched) to `<root>/step_<n>`.
  2. `push_adapter_to_vllm(base_url, name, path)` — `POST
     {base_url}/load_lora_adapter` (server started with `--enable-lora` and
     `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`); unload the previous same-named
     adapter first so `name` always resolves to the freshest weights.
  3. `megagem.rollout.run_game` then requests `model=<name>` for the
     trainable seat.

**Documented fallback (decided on-box from the seam tests, not assumed):** if
dynamic LoRA load is unreliable on the pinned vLLM 0.10.2, merge adapter→full
weights and replace the vLLM process (`merge_and_replace_hint()` returns the
shape of that path), or amortise by refreshing every N steps instead of every
generation. The driver chooses; this module only provides the fast path + the
explicit hint.

Torch/peft imports are lazy so importing this module on a CPU box (tests,
dry-run) pulls no ML libs — only the pure URL/payload/naming logic, which is
what the CPU tests exercise.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

ADAPTER_NAME = "phase3-trainable"  # the served-model-name the driver requests

# Lagged-self snapshots (§3.3) are served under IMMUTABLE names — unlike
# ADAPTER_NAME, which is re-pushed every refresh — so a drawn snapshot always
# resolves to the exact frozen weights it was registered with.
SNAPSHOT_NAME_PREFIX = "phase3-snap-"


def snapshot_served_name(step: int) -> str:
    """Immutable served-model-name for the lagged-self snapshot frozen at
    `step` (§3.3). Each maps 1:1 to a frozen adapter on the rollout vLLM."""
    return f"{SNAPSHOT_NAME_PREFIX}{int(step)}"


def step_adapter_dir(root: str | Path, step: int) -> Path:
    return Path(root) / f"step_{step}"


def snapshot_adapter_dir(root: str | Path, step: int) -> Path:
    """Snapshot adapters live under ``<root>/snapshots/step_<n>`` — a sibling
    of ``step_adapter_dir``'s ``<root>/step_<n>`` so a §3.3 snapshot never
    collides with the §3.6 eval checkpoints the driver's _Ckpt writes."""
    return Path(root) / "snapshots" / f"step_{int(step)}"


def save_step_adapter(model, step: int, root: str | Path) -> str:
    """Persist ONLY the LoRA adapter for `step` (peft). Returns the dir path.

    `model` is the live (PEFT-wrapped) trainer model. `save_pretrained` on a
    PEFT model writes just the adapter (`adapter_model.safetensors` +
    `adapter_config.json`) — small and fast, no base-weight copy.
    """
    out = step_adapter_dir(root, step)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))  # PEFT ⇒ adapter-only
    return str(out)


def save_snapshot_adapter(model, step: int, root: str | Path) -> str:
    """Freeze the live LoRA adapter as the §3.3 lagged-self snapshot for
    `step` (peft adapter-only `save_pretrained`). Returns the dir path. Writes
    under ``snapshot_adapter_dir`` so it never collides with `save_step_adapter`
    (the §3.6 eval checkpoints)."""
    out = snapshot_adapter_dir(root, step)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))  # PEFT ⇒ adapter-only
    return str(out)


def _post(url: str, payload: dict, timeout: float = 60.0) -> tuple[int, str]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        return 0, str(e)


def _lora_api_base(base_url: str) -> str:
    """vLLM 0.10.2 serves the dynamic-LoRA endpoints UNDER the OpenAI route:
    ``/v1/load_lora_adapter`` and ``/v1/unload_lora_adapter`` (see
    https://docs.vllm.ai/en/v0.10.2/features/lora.html#using-api-endpoints).
    `base_url` is already the ``…/v1`` URL, so do NOT strip it (a prior
    version stripped `/v1` and 404'd on-box)."""
    b = base_url.rstrip("/")
    return b if b.endswith("/v1") else b + "/v1"


def push_adapter_to_vllm(
    base_url: str, name: str, adapter_path: str, *, timeout: float = 60.0
) -> dict:
    """Make `name` resolve to the adapter at `adapter_path` on the vLLM at
    `base_url`. Idempotent: unload-then-load so a re-pushed `name` always
    serves the freshest step. Returns a small status dict (never raises — a
    sync failure is a reported run condition, the driver decides)."""
    api = _lora_api_base(base_url)  # …/v1 (vLLM 0.10.2 mounts these under /v1)
    # best-effort unload of the prior same-named adapter (404/400 is fine)
    un_code, un_body = _post(
        f"{api}/unload_lora_adapter", {"lora_name": name}, timeout=timeout)
    ld_code, ld_body = _post(
        f"{api}/load_lora_adapter",
        {"lora_name": name, "lora_path": adapter_path}, timeout=timeout)
    ok = ld_code == 200
    return {
        "ok": ok,
        "name": name,
        "adapter_path": adapter_path,
        "load_status": ld_code,
        "load_body": ld_body[:500],
        "unload_status": un_code,
        "fallback_hint": None if ok else merge_and_replace_hint(),
    }


def unload_adapter_from_vllm(
    base_url: str, name: str, *, timeout: float = 60.0
) -> dict:
    """Unload `name` from the vLLM at `base_url` — evicts a snapshot aged out
    of the §3.3 ring buffer. Best-effort: never raises; a failed unload only
    leaks one LoRA slot (non-fatal — the pool stays correct)."""
    api = _lora_api_base(base_url)
    code, body = _post(
        f"{api}/unload_lora_adapter", {"lora_name": name}, timeout=timeout)
    return {"ok": code == 200, "name": name,
            "status": code, "body": body[:500]}


def push_adapter_to_vllm_all(
    base_urls: list[str], name: str, adapter_path: str,
    *, timeout: float = 60.0,
) -> dict:
    """Fan out `push_adapter_to_vllm` to every URL in `base_urls` (DP layout).
    All-or-nothing: on partial failure, best-effort unload from URLs where
    push succeeded so round-robin never lands on a half-loaded adapter. The
    aggregate `ok` is `all(per_url[i].ok)`; on failure, `load_status` /
    `load_body` mirror the FIRST failing URL so existing log/abort paths
    that read those fields still surface a real diagnostic."""
    from concurrent.futures import ThreadPoolExecutor

    if not base_urls:
        return {"ok": False, "name": name, "adapter_path": adapter_path,
                "per_url": [], "load_status": 0,
                "load_body": "push_adapter_to_vllm_all: empty base_urls",
                "fallback_hint": merge_and_replace_hint()}

    with ThreadPoolExecutor(max_workers=max(1, len(base_urls))) as ex:
        results = list(ex.map(
            lambda u: push_adapter_to_vllm(
                u, name, adapter_path, timeout=timeout),
            base_urls))
    per_url = [{"url": base_urls[i], **r} for i, r in enumerate(results)]
    all_ok = all(r["ok"] for r in results)
    out: dict = {"ok": all_ok, "name": name, "adapter_path": adapter_path,
                 "per_url": per_url}
    if all_ok:
        # mirror the first per-URL load_status so callers that print this
        # field on success see a real 200.
        out["load_status"] = results[0]["load_status"]
        out["load_body"] = results[0]["load_body"]
        out["fallback_hint"] = None
        return out
    # partial failure → roll back the URLs that succeeded.
    succeeded = [base_urls[i] for i, r in enumerate(results) if r["ok"]]
    if succeeded:
        with ThreadPoolExecutor(max_workers=max(1, len(succeeded))) as ex:
            list(ex.map(
                lambda u: unload_adapter_from_vllm(u, name, timeout=timeout),
                succeeded))
    first_fail = next(r for r in results if not r["ok"])
    out["load_status"] = first_fail["load_status"]
    out["load_body"] = first_fail["load_body"]
    out["fallback_hint"] = merge_and_replace_hint()
    return out


def unload_adapter_from_vllm_all(
    base_urls: list[str], name: str, *, timeout: float = 60.0,
) -> dict:
    """Fan out `unload_adapter_from_vllm` to every URL in `base_urls`. Like
    its single-URL sibling, never raises — a leaked LoRA slot on one worker
    is non-fatal as long as `name` is no longer requested."""
    from concurrent.futures import ThreadPoolExecutor

    if not base_urls:
        return {"ok": True, "name": name, "per_url": []}
    with ThreadPoolExecutor(max_workers=max(1, len(base_urls))) as ex:
        results = list(ex.map(
            lambda u: unload_adapter_from_vllm(u, name, timeout=timeout),
            base_urls))
    per_url = [{"url": base_urls[i], **r} for i, r in enumerate(results)]
    return {"ok": all(r["ok"] for r in results), "name": name,
            "per_url": per_url}


def merge_and_replace_hint() -> str:
    return (
        "dynamic LoRA load failed on pinned vLLM 0.10.2 — fallback: "
        "merge adapter into base (peft merge_and_unload), write a full "
        "checkpoint, and restart the rollout vLLM process pointed at it "
        "(slower, proven), OR refresh every N steps instead of per-generation."
    )
