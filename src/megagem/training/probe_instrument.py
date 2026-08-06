"""Phase-3 probe instrument — per-optimizer-step train time + vLLM prefix-cache.

Attached ONLY when `PHASE3_PROBE=1` (see phase3_grpo._gpu_run). Inert otherwise.

Why a callback rather than `wall_s − Σroll_s`:
  On a 5-step probe run the total wall time is dominated by one-time setup (base
  model load, vLLM readiness, adapter sync) and the §3.6 eval — so deriving
  train time by subtraction is badly contaminated. `on_step_begin/​on_step_end`
  brackets ONLY the optimizer step, giving a clean T_train per step. Generation
  time (T_gen) is already persisted per roll as `rolls_meta[i]["roll_s"]`.

prefix-cache: vLLM exposes prefix-cache counters at `{url}/metrics` (Prometheus).
  Names have drifted across vLLM versions (`vllm:prefix_cache_hits[_total]`,
  `vllm:prefix_cache_queries[_total]`, and/or a `vllm:gpu_prefix_cache_hit_rate`
  gauge), so we scrape ANY `prefix_cache` line, summing counters and maxing
  gauges across DP workers. The report diffs consecutive begin-scrapes → the
  per-roll hit rate (the generation between two steps).
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from transformers import TrainerCallback


def _metrics_url(base: str) -> str:
    root = base.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/metrics"


def scrape_prefix_cache(urls: list[str]) -> dict:
    """Sum counters / max gauges for every `prefix_cache` (+ gpu_cache_usage)
    series across all vLLM workers. Returns {} if all endpoints are unreachable.
    Never raises — telemetry must not break a run."""
    sums: dict[str, float] = {}
    maxes: dict[str, float] = {}
    n_ok = 0
    for u in urls:
        try:
            with urllib.request.urlopen(_metrics_url(u), timeout=2.0) as r:
                body = r.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — endpoint may be transiently down
            continue
        n_ok += 1
        for line in body.splitlines():
            if not line or line.startswith("#"):
                continue
            if "prefix_cache" not in line and "gpu_cache_usage" not in line:
                continue
            name = line.split("{", 1)[0].strip().split(" ")[0]
            try:
                val = float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                continue
            sums[name] = sums.get(name, 0.0) + val
            maxes[name] = max(maxes.get(name, val), val)
    if n_ok == 0:
        return {}
    return {"sums": sums, "maxes": maxes, "n_workers_ok": n_ok}


class ProbeStepTimer(TrainerCallback):
    """Records one JSONL row per optimizer step: clean train wall-time + the
    vLLM prefix-cache counter delta since the previous step (≈ the preceding
    roll's generation cache activity)."""

    def __init__(self, out_dir: str, vllm_urls: list[str]):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "probe_timing.jsonl")
        self.urls = list(vllm_urls or [])
        self._t0: float | None = None
        self._prev_cache: dict | None = None
        # truncate any stale file so the report only sees this run
        with open(self.path, "w") as f:
            f.write("")

    def _append(self, rec: dict) -> None:
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def on_step_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        # Scrape counters BEFORE starting the train clock so /metrics round-trips
        # never inflate train_s. The delta vs the previous step's begin-scrape
        # brackets the roll that just generated.
        self._begin_cache = scrape_prefix_cache(self.urls) if self.urls else {}
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        step = int(getattr(state, "global_step", -1))
        train_s = (time.perf_counter() - self._t0) if self._t0 is not None else None
        rec: dict = {"step": step, "train_s": round(train_s, 4) if train_s else None}

        cache = getattr(self, "_begin_cache", {}) or {}
        if cache:
            rec["kv_cache_usage"] = (cache.get("maxes", {})
                                     .get("vllm:gpu_cache_usage_perc"))
            hit_rate_gauge = None
            for k, v in cache.get("maxes", {}).items():
                if "prefix_cache_hit_rate" in k:
                    hit_rate_gauge = v
            if hit_rate_gauge is not None:
                rec["prefix_cache_hit_rate_gauge"] = round(hit_rate_gauge, 4)
            # counter delta vs previous begin-scrape → per-roll hit rate
            if self._prev_cache:
                ps, cs = self._prev_cache.get("sums", {}), cache.get("sums", {})
                def _delta(substr: str):
                    tot = 0.0
                    for k, v in cs.items():
                        if substr in k and "rate" not in k:
                            tot += v - ps.get(k, 0.0)
                    return tot
                dq = _delta("prefix_cache_queries") or _delta("prefix_cache_query")
                dh = _delta("prefix_cache_hits") or _delta("prefix_cache_hit")
                rec["prefix_cache_queries_delta"] = dq
                rec["prefix_cache_hits_delta"] = dh
                if dq and dq > 0:
                    rec["prefix_cache_hit_rate_window"] = round(dh / dq, 4)
            self._prev_cache = cache
        self._append(rec)
