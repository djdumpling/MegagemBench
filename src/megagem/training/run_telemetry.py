#!/usr/bin/env python3
"""Phase-3 rollout telemetry — vLLM /metrics, nvidia-smi, AsyncOpenAI status.

Purpose: answer "is MP=64 saturating vLLM or are we API/network-bound?" from a
single smoke run. Each sampler is a daemon thread; per-rollout windows are
opened/closed by the rollout driver and summarised in one greppable block.

Three samplers, all opt-in via env (see `install_from_env`):
  * VLLMMetricsSampler   — polls `{vllm_url}/metrics` (Prometheus text) at 1Hz.
                           Extracts gpu_cache_usage_perc, num_requests_running,
                           num_requests_waiting. The KV % is the headroom
                           signal; waiting>0 is the queue-depth-bottleneck
                           signal.
  * NvidiaSmiSampler     — `nvidia-smi --query-gpu=...` at 1Hz, split per GPU
                           index. Trainer+vLLM under --split-gpus map to
                           distinct indices so they are reported separately.
  * APICallTracker       — process-wide monkeypatch of `openai.AsyncOpenAI`.
                           Wraps `chat.completions.create` only on clients
                           whose base_url is NOT the local vLLM (key="EMPTY").
                           Records (model, status, latency) per call so we can
                           count 429s per opponent model.

Telemetry is pure side effect — if a sampler fails to start (no nvidia-smi,
no /metrics endpoint) it logs a one-time warning and the run proceeds.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import urllib.request
from collections import defaultdict


def _p(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


# --------------------------------------------------------------------------- #
# vLLM /metrics scraper                                                       #
# --------------------------------------------------------------------------- #
class VLLMMetricsSampler:
    """1Hz poller of vLLM's Prometheus `/metrics` endpoint.

    `base_url` is the vLLM root (we strip a trailing `/v1` so `/metrics` lands
    at the right place). Each scrape parses three numeric series:
      vllm:gpu_cache_usage_perc      — KV cache fill (0..1)
      vllm:num_requests_running      — concurrent active requests
      vllm:num_requests_waiting      — queue depth (the bottleneck signal)
    """

    KEYS = (
        "vllm:gpu_cache_usage_perc",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
    )

    def __init__(self, base_url: str, interval: float = 1.0):
        # /metrics is served at the server root, not under /v1.
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        self.metrics_url = f"{root}/metrics"
        self.interval = float(interval)
        self._samples: list[tuple[float, dict[str, float]]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned_unreachable = False

    def start(self) -> "VLLMMetricsSampler":
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="vllm-metrics")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _scrape_once(self) -> dict[str, float] | None:
        try:
            with urllib.request.urlopen(self.metrics_url, timeout=2.0) as r:
                body = r.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — endpoint may be transiently down
            if not self._warned_unreachable:
                print(f"  [telemetry] vLLM /metrics unreachable at "
                      f"{self.metrics_url}; will keep retrying silently",
                      flush=True)
                self._warned_unreachable = True
            return None
        out: dict[str, float] = {}
        for line in body.splitlines():
            if not line or line.startswith("#"):
                continue
            for key in self.KEYS:
                if line.startswith(key):
                    parts = line.rsplit(" ", 1)
                    if len(parts) == 2:
                        try:
                            # Multiple series per key (per-model labels) — keep
                            # the MAX so KV across all served LoRAs reads the
                            # busiest. vLLM exposes one series per served name.
                            v = float(parts[1])
                            if key not in out or v > out[key]:
                                out[key] = v
                        except ValueError:
                            pass
                    break
        return out or None

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._scrape_once()
            if sample is not None:
                with self._lock:
                    self._samples.append((time.perf_counter(), sample))
            self._stop.wait(self.interval)

    def window_open(self) -> float:
        return time.perf_counter()

    def window_close(self, t0: float) -> dict:
        with self._lock:
            window = [s for (t, s) in self._samples if t >= t0]
        if not window:
            return {"n_samples": 0}
        out: dict = {"n_samples": len(window)}
        for key in self.KEYS:
            vals = [s[key] for s in window if key in s]
            if not vals:
                continue
            short = key.split(":", 1)[1]
            out[f"{short}_mean"] = sum(vals) / len(vals)
            out[f"{short}_p50"] = _p(vals, 0.50)
            out[f"{short}_p95"] = _p(vals, 0.95)
            out[f"{short}_max"] = max(vals)
        return out


# --------------------------------------------------------------------------- #
# nvidia-smi sampler                                                          #
# --------------------------------------------------------------------------- #
class NvidiaSmiSampler:
    """1Hz `nvidia-smi --query-gpu=...` poller, split per GPU index."""

    def __init__(self, interval: float = 1.0):
        self.interval = float(interval)
        self._samples: list[tuple[float, list[dict]]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._unavailable = False

    def start(self) -> "NvidiaSmiSampler":
        # one-shot availability check before launching the thread
        if self._scrape_once() is None:
            self._unavailable = True
            print("  [telemetry] nvidia-smi unavailable; GPU util will be empty",
                  flush=True)
            return self
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="nvidia-smi")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _scrape_once(self) -> list[dict] | None:
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2.0)
            if r.returncode != 0:
                return None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        out: list[dict] = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                out.append({
                    "gpu": int(parts[0]),
                    "util_pct": float(parts[1]),
                    "mem_used_mib": float(parts[2]),
                    "mem_total_mib": float(parts[3]),
                })
            except ValueError:
                continue
        return out or None

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._scrape_once()
            if sample is not None:
                with self._lock:
                    self._samples.append((time.perf_counter(), sample))
            self._stop.wait(self.interval)

    def window_open(self) -> float:
        return time.perf_counter()

    def window_close(self, t0: float) -> dict:
        if self._unavailable:
            return {"unavailable": True}
        with self._lock:
            window = [(t, s) for (t, s) in self._samples if t >= t0]
        if not window:
            return {"n_samples": 0}
        per_gpu: dict[int, list[dict]] = defaultdict(list)
        for _, sample in window:
            for gpu in sample:
                per_gpu[gpu["gpu"]].append(gpu)
        out: dict = {"n_samples": len(window), "per_gpu": {}}
        for gpu_idx, samples in sorted(per_gpu.items()):
            utils = [s["util_pct"] for s in samples]
            mems = [s["mem_used_mib"] for s in samples]
            mem_total = samples[0]["mem_total_mib"]
            mem_p95 = _p(mems, 0.95) or 0.0
            mem_max = max(mems)
            out["per_gpu"][gpu_idx] = {
                "util_mean": sum(utils) / len(utils),
                "util_p50": _p(utils, 0.50),
                "util_p95": _p(utils, 0.95),
                "util_max": max(utils),
                "mem_used_mib_p95": mem_p95,
                "mem_used_mib_max": mem_max,
                "mem_total_mib": mem_total,
                "mem_used_pct_p95": (mem_p95 / mem_total * 100.0)
                                    if mem_total else 0.0,
                "mem_used_pct_max": (mem_max / mem_total * 100.0)
                                    if mem_total else 0.0,
            }
        return out


# --------------------------------------------------------------------------- #
# AsyncOpenAI status-code tracker (monkeypatch)                               #
# --------------------------------------------------------------------------- #
class APICallTracker:
    """Process-wide tally of OpenAI status codes & latencies per model.

    `install(local_vllm_url)` monkey-patches `openai.AsyncOpenAI.__init__` so
    every subsequently-constructed client gets a wrapped
    `chat.completions.create`. The local vLLM client (whose base_url matches
    `local_vllm_url`) is skipped — we only care about external API opponents.

    Idempotent: install() is a no-op on second call. Recording is lock-guarded
    and append-only so per-rollout `window_close()` can slice the log without
    contention.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[dict] = []
        self._installed = False

    def install(self, local_vllm_url: str = "") -> None:
        if self._installed:
            return
        try:
            from openai import AsyncOpenAI
        except ImportError:
            print("  [telemetry] openai SDK not importable; API tally disabled",
                  flush=True)
            return
        self._installed = True
        tracker = self
        # Extract host:port from the local URL so we can suppress wrapping it.
        local_host = ""
        if local_vllm_url:
            local_host = local_vllm_url.split("//")[-1].split("/")[0]

        orig_init = AsyncOpenAI.__init__

        def patched_init(self_o, *args, **kwargs):  # noqa: ANN001
            orig_init(self_o, *args, **kwargs)
            base = str(getattr(self_o, "base_url", ""))
            # Skip every local endpoint (trainable vLLM AND heuristic shim are
            # both on localhost). Only external API opponents (pinference,
            # openrouter, anthropic-direct, …) are counted toward the API tally.
            is_local = (
                ("localhost" in base) or ("127.0.0.1" in base)
                or (local_host and local_host in base))
            if is_local:
                return
            try:
                orig_create = self_o.chat.completions.create
            except AttributeError:
                return

            async def wrapped(*a, **kw):
                model = kw.get("model", "unknown")
                t0 = time.perf_counter()
                status: int = 200
                err_type: str | None = None
                try:
                    out = await orig_create(*a, **kw)
                    return out
                except Exception as e:  # noqa: BLE001
                    err_type = type(e).__name__
                    sc = getattr(e, "status_code", None)
                    if isinstance(sc, int):
                        status = sc
                    else:
                        # Network/timeout/etc with no HTTP response — bucket as 0.
                        status = 0
                    raise
                finally:
                    dt = time.perf_counter() - t0
                    with tracker._lock:
                        tracker._records.append({
                            "t": time.perf_counter(),
                            "model": model,
                            "status": status,
                            "err_type": err_type,
                            "latency_s": dt,
                        })

            self_o.chat.completions.create = wrapped

        AsyncOpenAI.__init__ = patched_init

    def window_open(self) -> float:
        return time.perf_counter()

    def window_close(self, t0: float) -> dict:
        with self._lock:
            window = [r for r in self._records if r["t"] >= t0]
        if not window:
            return {"n_calls": 0}
        by_model: dict[str, list[dict]] = defaultdict(list)
        for r in window:
            by_model[r["model"]].append(r)
        n_200 = sum(1 for r in window if r["status"] == 200)
        n_429 = sum(1 for r in window if r["status"] == 429)
        n_5xx = sum(1 for r in window if 500 <= r["status"] < 600)
        n_err = sum(1 for r in window if r["err_type"])
        out = {
            "n_calls": len(window),
            "n_200": n_200, "n_429": n_429, "n_5xx": n_5xx, "n_err": n_err,
            "by_model": {},
        }
        for model, records in sorted(by_model.items()):
            lats = [r["latency_s"] for r in records]
            out["by_model"][model] = {
                "n": len(records),
                "n_200": sum(1 for r in records if r["status"] == 200),
                "n_429": sum(1 for r in records if r["status"] == 429),
                "n_5xx": sum(1 for r in records if 500 <= r["status"] < 600),
                "n_err": sum(1 for r in records if r["err_type"]),
                "latency_p50_s": _p(lats, 0.50),
                "latency_p95_s": _p(lats, 0.95),
                "latency_max_s": max(lats) if lats else None,
            }
        return out


# --------------------------------------------------------------------------- #
# Bundle: open three windows together, format one summary block.              #
# --------------------------------------------------------------------------- #
class RolloutTelemetry:
    """Convenience bundle for the rollout driver.

    Owns the three samplers; `window_open()`/`window_close()` returns a single
    stats dict; `format(...)` produces a multi-line block printed by the caller
    immediately after the existing "games complete in Xs" line.
    """

    def __init__(self, vllm: VLLMMetricsSampler | None,
                 gpu: NvidiaSmiSampler | None,
                 api: APICallTracker | None):
        self.vllm = vllm
        self.gpu = gpu
        self.api = api

    def window_open(self) -> dict:
        return {
            "vllm_t0": self.vllm.window_open() if self.vllm else None,
            "gpu_t0": self.gpu.window_open() if self.gpu else None,
            "api_t0": self.api.window_open() if self.api else None,
        }

    def window_close(self, h: dict) -> dict:
        return {
            "vllm": self.vllm.window_close(h["vllm_t0"]) if self.vllm else None,
            "gpu": self.gpu.window_close(h["gpu_t0"]) if self.gpu else None,
            "api": self.api.window_close(h["api_t0"]) if self.api else None,
        }

    @staticmethod
    def format_summary(roll_index: int, wall_s: float,
                       game_durs: list[float], stats: dict) -> str:
        tag = f"roll {roll_index:03d}"
        lines: list[str] = []
        if game_durs:
            sd = sorted(game_durs)
            n = len(sd)
            p50 = _p(sd, 0.50) or 0.0
            p95 = _p(sd, 0.95) or 0.0
            ratio = (p95 / p50) if p50 > 0 else float("nan")
            lines.append(
                f"  [{tag}] TELEMETRY  wall={wall_s:.1f}s  games:n={n} "
                f"p50={p50:.1f}s p95={p95:.1f}s max={max(sd):.1f}s "
                f"p95/p50={ratio:.2f}x")
        vstats = stats.get("vllm") or {}
        if vstats.get("n_samples", 0) > 0:
            kv_p95 = vstats.get("gpu_cache_usage_perc_p95")
            kv_max = vstats.get("gpu_cache_usage_perc_max")
            run_p95 = vstats.get("num_requests_running_p95")
            run_max = vstats.get("num_requests_running_max")
            wait_p95 = vstats.get("num_requests_waiting_p95")
            wait_max = vstats.get("num_requests_waiting_max")
            lines.append(
                f"  [{tag}] TELEMETRY  vllm: KV p95={_fmtpct(kv_p95)} "
                f"max={_fmtpct(kv_max)}  "
                f"running p95={_fmtnum(run_p95)} max={_fmtnum(run_max)}  "
                f"waiting p95={_fmtnum(wait_p95)} max={_fmtnum(wait_max)}  "
                f"({vstats['n_samples']} samples)")
        gstats = stats.get("gpu") or {}
        if gstats.get("per_gpu"):
            # Loud-WARN threshold for headroom on a long-training push. Default
            # 92% leaves ~12 GiB on H200 (143 GiB) for transient allocations
            # the sampler can miss between 1Hz ticks. Override with
            # PHASE3_GPU_MEM_ALERT_PCT (e.g. 95 if you trust the burst headroom).
            try:
                alert_pct = float(os.environ.get("PHASE3_GPU_MEM_ALERT_PCT", "92"))
            except (TypeError, ValueError):
                alert_pct = 92.0
            alerts = []
            for gpu_idx, s in sorted(gstats["per_gpu"].items()):
                lines.append(
                    f"  [{tag}] TELEMETRY  gpu{gpu_idx}: "
                    f"util mean={s['util_mean']:.0f}% p95={s['util_p95']:.0f}% "
                    f"max={s['util_max']:.0f}%  "
                    f"mem p95={s['mem_used_pct_p95']:.0f}% "
                    f"max={s['mem_used_pct_max']:.0f}% "
                    f"({s['mem_used_mib_max']:.0f}/"
                    f"{s['mem_total_mib']:.0f}MiB peak)")
                if s["mem_used_pct_max"] >= alert_pct:
                    alerts.append((gpu_idx, s))
            for gpu_idx, s in alerts:
                lines.append(
                    f"  [{tag}] TELEMETRY  WARN gpu{gpu_idx} MEM-ALERT: "
                    f"peak {s['mem_used_pct_max']:.1f}% "
                    f"({s['mem_used_mib_max']:.0f}/{s['mem_total_mib']:.0f}MiB) "
                    f"≥ alert={alert_pct:.0f}%; "
                    f"OOM risk if config grows. Lower micro-cap, num-seeds, K, "
                    f"or max-parallel; or raise PHASE3_GPU_MEM_ALERT_PCT to "
                    f"silence if intentional.")
        astats = stats.get("api") or {}
        if astats.get("n_calls", 0) > 0:
            lines.append(
                f"  [{tag}] TELEMETRY  api: calls={astats['n_calls']} "
                f"200={astats['n_200']} 429={astats['n_429']} "
                f"5xx={astats['n_5xx']} err={astats['n_err']}")
            for model, s in sorted((astats.get("by_model") or {}).items()):
                lines.append(
                    f"  [{tag}] TELEMETRY  api[{model}]: n={s['n']} "
                    f"200={s['n_200']} 429={s['n_429']} 5xx={s['n_5xx']} "
                    f"err={s['n_err']} "
                    f"lat p50={_fmtsec(s['latency_p50_s'])} "
                    f"p95={_fmtsec(s['latency_p95_s'])} "
                    f"max={_fmtsec(s['latency_max_s'])}")
        return "\n".join(lines)

    @staticmethod
    def format_headroom(roll_index: int, total_games: int, wall_s: float,
                        max_parallel: int, stats: dict) -> str:
        """Interpret the raw counters into an actionable optimization verdict:
        is this roll semaphore-bound (raise max_parallel), GPU-compute-bound, or
        starved? Reads the 7gen+1train split from the PHASE3_*_CUDA_VISIBLE_DEVICES
        env vars so generation-GPU util is separated from the train GPU.

        Decision rules (heuristic, but stated so the operator can override):
          running_max ≪ cap            → not semaphore-bound (too few games in flight)
          waiting_max ≥ 1              → workers SATURATED, max_parallel overfeeds (GPU-bound)
          running≈cap, wait≈0, util<80 → STARVED at the cap, RAISE max_parallel
          running≈cap, wait≈0, util≥80 → BALANCED, near GPU-compute-bound
        """
        tag = f"roll {roll_index:03d}"
        vstats = stats.get("vllm") or {}
        gstats = (stats.get("gpu") or {}).get("per_gpu") or {}
        lines: list[str] = []

        gps = (total_games / wall_s) if wall_s > 0 else float("nan")
        run_max = vstats.get("num_requests_running_max")
        wait_max = vstats.get("num_requests_waiting_max")
        kv_max = vstats.get("gpu_cache_usage_perc_max")

        def _idxs(name: str) -> list[int]:
            return [int(t) for t in os.environ.get(name, "").split(",")
                    if t.strip().isdigit()]

        def _mean_util(idxs) -> float | None:
            vals = [gstats[i]["util_mean"] for i in idxs if i in gstats]
            return (sum(vals) / len(vals)) if vals else None

        gen_idxs = _idxs("PHASE3_VLLM_CUDA_VISIBLE_DEVICES")
        train_idxs = _idxs("PHASE3_TRAIN_CUDA_VISIBLE_DEVICES")
        gen_util = _mean_util(gen_idxs) if gen_idxs else _mean_util(list(gstats))
        train_util = _mean_util(train_idxs) if train_idxs else None

        if run_max is None:
            sem = "vLLM /metrics unavailable — cannot judge max_parallel headroom"
        elif run_max < 0.75 * max_parallel:
            sem = (f"NOT semaphore-bound: running_max={run_max:.0f} « cap={max_parallel} "
                   f"⇒ too few concurrent games to fill the semaphore (limited by "
                   f"total_games={total_games} or per-game call serialization — a small "
                   f"roll can't measure max_parallel headroom; need total_games ≳ cap)")
        elif wait_max is not None and wait_max >= 1.0:
            sem = (f"workers SATURATED: waiting_max={wait_max:.0f}>0 queueing at "
                   f"running≈{run_max:.0f} ⇒ max_parallel already overfeeds vLLM; "
                   f"raising it won't help (GPU-bound)")
        elif gen_util is not None and gen_util < 80.0:
            sem = (f"workers STARVED at cap: running_max={run_max:.0f}≈{max_parallel}, "
                   f"waiting≈0, gen-util={gen_util:.0f}% ⇒ RAISE max_parallel "
                   f"(idle GPU capacity sitting behind the semaphore)")
        else:
            gu = "n/a" if gen_util is None else f"{gen_util:.0f}%"
            sem = (f"BALANCED: running_max={run_max:.0f}≈{max_parallel}, waiting≈0, "
                   f"gen-util={gu} ⇒ near GPU-compute-bound; max_parallel ~right")

        lines.append(f"  [{tag}] HEADROOM  throughput={gps:.2f} games/s  "
                     f"cap=max_parallel={max_parallel}  games={total_games}")
        lines.append(f"  [{tag}] HEADROOM  max_parallel: {sem}")

        extra = []
        if gen_util is not None:
            extra.append(f"gen-GPU compute headroom≈{max(0.0, 100 - gen_util):.0f}% "
                         f"(util {gen_util:.0f}%)")
        if kv_max is not None:
            extra.append(f"KV-cache headroom≈{max(0.0, 100 - kv_max * 100):.0f}% "
                         f"(peak {kv_max * 100:.0f}%)")
        if extra:
            lines.append(f"  [{tag}] HEADROOM  " + "  ".join(extra))
        if (train_util is not None and gen_util is not None
                and train_util < 0.5 * max(gen_util, 1.0)):
            lines.append(
                f"  [{tag}] HEADROOM  train-GPU util={train_util:.0f}% « gen={gen_util:.0f}% "
                f"⇒ generation-bound (expected): throughput scales with MORE gen "
                f"workers / higher max_parallel, NOT a 2nd train GPU (DDP only halves ga)")
        return "\n".join(lines)


def _fmtpct(v) -> str:
    if v is None:
        return "n/a"
    # vLLM reports gpu_cache_usage_perc as a 0..1 fraction.
    return f"{v * 100:.0f}%"


def _fmtnum(v) -> str:
    return "n/a" if v is None else f"{float(v):.1f}"


def _fmtsec(v) -> str:
    return "n/a" if v is None else f"{float(v):.2f}s"


def install_from_env(vllm_url: str | None) -> RolloutTelemetry | None:
    """Set up samplers per env.

    Env knobs (all opt-in; smoke runs should set PHASE3_TELEMETRY=1):
      PHASE3_TELEMETRY=1   — enable all three samplers
      PHASE3_TELEMETRY_INTERVAL=1.0   — sampling interval in seconds

    Returns the bundle, or None if telemetry is off.
    """
    if os.environ.get("PHASE3_TELEMETRY", "0") != "1":
        return None
    interval = float(os.environ.get("PHASE3_TELEMETRY_INTERVAL", "1.0"))
    vllm = VLLMMetricsSampler(vllm_url, interval=interval).start() \
        if vllm_url else None
    gpu = NvidiaSmiSampler(interval=interval).start()
    api = APICallTracker()
    api.install(local_vllm_url=vllm_url or "")
    return RolloutTelemetry(vllm=vllm, gpu=gpu, api=api)


# --------------------------------------------------------------------------- #
# RolloutHealthMonitor — kill-thresholds derived from the apimix_x2_01        #
# post-mortem (phase3_rl_run_findings.md Finding 15).                         #
# --------------------------------------------------------------------------- #
class RolloutHealthMonitor:
    """Track per-rollout health and emit WARN/ABORT alerts on failure patterns.

    The apimix_x2_01 run produced clean training-process metrics (KL flat,
    entropy stable, no clip pressure, group_std rising) while the underlying
    policy was slowly unlearning. The signals that DID show the regression
    were per-rollout `per_game_reward_mean` and the optimizer-step
    `megagem/advantages/mean`. This monitor codifies those into alerts so the
    next run doesn't go 200 steps before we notice.

    Configuration via env (all optional; tuned to apimix's actual failure
    timing — would have alerted by roll ~16 / step 96):
      PHASE3_HEALTH_ADV_NEG_THRESHOLD     default -0.015  (rolling-5 mean)
      PHASE3_HEALTH_REWARD_DROP_PCT       default 0.40   (drop from peak)
      PHASE3_HEALTH_PEAK_DROP_MIN_PEAK    default 0.30   (peak-drop alert
                                                          only fires when
                                                          peak > this; avoids
                                                          spurious "170% drop"
                                                          alerts in self-play
                                                          where rewards center
                                                          near zero — repl_08
                                                          v3 fix)
      PHASE3_HEALTH_NEG_ROLL_FRACTION     default 0.60   (of last 5 rolls)
      PHASE3_HEALTH_MIN_REWARD            default -0.30  (per_game_reward_mean
                                                          floor, after roll 5;
                                                          was 0.30 — repl_08
                                                          v3 lowered to allow
                                                          self-play to dip
                                                          slightly net-negative
                                                          without false alarms.
                                                          Plan-faithful value
                                                          per repl_08 v3 plan.
                                                          Set to 0.30 explicitly
                                                          for repl_07-style
                                                          heuristic-heavy runs.)
      PHASE3_HEALTH_CONSEC_BAD            default 3      (rolls to trigger ABORT
                                                          when PHASE3_HEALTH_ABORT=1)
      PHASE3_HEALTH_ABORT                 default "0"    ("1" ⇒ raise after
                                                          CONSEC_BAD rolls;
                                                          default WARN only)

    Auto-abort is OPT-IN — by default we WARN every rollout but never raise,
    so a stable mid-run dip doesn't kill a long training accidentally. Set
    PHASE3_HEALTH_ABORT=1 for production runs where you want the early-kill
    insurance (saves ~$200/hr of GPU on a doomed run).
    """

    def __init__(
        self,
        *,
        adv_neg_threshold: float | None = None,
        reward_drop_pct: float | None = None,
        neg_roll_fraction: float | None = None,
        min_reward_floor: float | None = None,
        consec_bad_for_abort: int | None = None,
        abort_on_threshold: bool | None = None,
        peak_drop_min_peak: float | None = None,
    ):
        # Env defaults — easier to tune than recompiling.
        env = os.environ
        self.adv_neg_threshold = float(
            adv_neg_threshold
            if adv_neg_threshold is not None
            else env.get("PHASE3_HEALTH_ADV_NEG_THRESHOLD", "-0.015"))
        self.reward_drop_pct = float(
            reward_drop_pct
            if reward_drop_pct is not None
            else env.get("PHASE3_HEALTH_REWARD_DROP_PCT", "0.40"))
        self.peak_drop_min_peak = float(
            peak_drop_min_peak
            if peak_drop_min_peak is not None
            else env.get("PHASE3_HEALTH_PEAK_DROP_MIN_PEAK", "0.30"))
        self.neg_roll_fraction = float(
            neg_roll_fraction
            if neg_roll_fraction is not None
            else env.get("PHASE3_HEALTH_NEG_ROLL_FRACTION", "0.60"))
        # repl_08 v3: default lowered 0.30 → -0.30 (plan-faithful) because
        # seam_smoke_02 showed self-play rewards center at zero with normal
        # roll-to-roll dips; the 0.30 floor false-alarmed every roll. -0.30
        # is loose enough for self-play near-zero noise to not trip, while
        # still catching catastrophic policy degradation. repl_07-style
        # heuristic-heavy runs (where mean reward should be > 0.5) should
        # set PHASE3_HEALTH_MIN_REWARD=0.30 explicitly.
        self.min_reward_floor = float(
            min_reward_floor
            if min_reward_floor is not None
            else env.get("PHASE3_HEALTH_MIN_REWARD", "-0.30"))
        self.consec_bad_for_abort = int(
            consec_bad_for_abort
            if consec_bad_for_abort is not None
            else env.get("PHASE3_HEALTH_CONSEC_BAD", "3"))
        self.abort_on_threshold = (
            abort_on_threshold
            if abort_on_threshold is not None
            else env.get("PHASE3_HEALTH_ABORT", "0") == "1")

        # Rolling state. Each record:
        #   {"roll", "step", "per_game_reward_mean", "advantages_mean",
        #    "train_reward_mean", "alerts": [...]}
        self.history: list[dict] = []
        self._consec_alert_rolls = 0

    # ------------------------------------------------------------------ #
    # API used by the rollout driver                                     #
    # ------------------------------------------------------------------ #
    def record(self, roll_index: int, step: int, *,
               per_game_reward_mean: float | None = None,
               advantages_mean: float | None = None,
               train_reward_mean: float | None = None,
               extras: dict | None = None) -> dict:
        """Record one rollout's health signals; return alert dict.

        Returned dict has keys:
          warn:  list[str]   non-fatal observations (always printed)
          abort: list[str]   reasons consec_bad has been hit (only acted on
                             when self.abort_on_threshold is True)
          consec_alert_rolls: int   how many consecutive rolls hit any alert
          should_abort: bool        true ⇒ caller should raise RuntimeError
        """
        rec = {
            "roll": int(roll_index),
            "step": int(step),
            "per_game_reward_mean": per_game_reward_mean,
            "advantages_mean": advantages_mean,
            "train_reward_mean": train_reward_mean,
            "extras": extras or {},
            "alerts": [],
        }
        self.history.append(rec)

        warn = self._check_warnings(rec)
        rec["alerts"] = warn[:]

        if warn:
            self._consec_alert_rolls += 1
        else:
            self._consec_alert_rolls = 0

        should_abort = (
            self.abort_on_threshold
            and self._consec_alert_rolls >= self.consec_bad_for_abort
        )
        abort_msgs = []
        if should_abort:
            abort_msgs.append(
                f"{self._consec_alert_rolls} consecutive rollouts have hit "
                f"health alerts (threshold={self.consec_bad_for_abort})")
        return {
            "warn": warn,
            "abort": abort_msgs,
            "consec_alert_rolls": self._consec_alert_rolls,
            "should_abort": should_abort,
        }

    # ------------------------------------------------------------------ #
    # Alert logic                                                        #
    # ------------------------------------------------------------------ #
    def _check_warnings(self, rec: dict) -> list[str]:
        """Return list of warning strings for this rollout, [] if all OK."""
        warns: list[str] = []
        last5 = self.history[-5:]
        n5 = len(last5)

        # 1) Rolling-5 advantage mean drifted negative — the apimix signature.
        adv_vals = [r["advantages_mean"] for r in last5
                    if isinstance(r.get("advantages_mean"), (int, float))]
        if len(adv_vals) >= 3:
            avg = sum(adv_vals) / len(adv_vals)
            if avg < self.adv_neg_threshold:
                warns.append(
                    f"advantages_mean rolling-{len(adv_vals)} = {avg:+.4f} "
                    f"< threshold {self.adv_neg_threshold:+.4f} "
                    f"(policy is being trained to AVOID its own completions; "
                    f"slow unlearning signature)")

        # 2) per_game_reward_mean has dropped > X% from its peak.
        # repl_08 v3 fix: only fire when the peak itself was meaningful
        # (> `peak_drop_min_peak`). In self-play the per-game reward centers
        # at zero by construction; a +0.18 peak followed by -0.13 produces a
        # "170% drop" that says nothing — the denominator is too small. The
        # guard requires a peak above the self-play noise floor before the
        # peak-drop signal carries information about regression.
        peak_records = [r["per_game_reward_mean"] for r in self.history
                        if isinstance(r.get("per_game_reward_mean"),
                                      (int, float))]
        cur = rec.get("per_game_reward_mean")
        if (isinstance(cur, (int, float)) and len(peak_records) >= 3
                and max(peak_records) > self.peak_drop_min_peak):
            peak = max(peak_records)
            drop = (peak - cur) / peak
            if drop > self.reward_drop_pct:
                warns.append(
                    f"per_game_reward_mean = {cur:+.3f}, dropped "
                    f"{100*drop:.0f}% from peak {peak:+.3f} "
                    f"(threshold: {100*self.reward_drop_pct:.0f}%, "
                    f"min-peak {self.peak_drop_min_peak:+.2f}) — opponent "
                    f"mix may be dominating the trainable seat")

        # 3) Fraction of last-5 rollouts with negative advantage mean.
        if n5 >= 5:
            neg = sum(1 for r in last5
                      if isinstance(r.get("advantages_mean"), (int, float))
                      and r["advantages_mean"] < 0)
            n_valid = sum(1 for r in last5
                          if isinstance(r.get("advantages_mean"), (int, float)))
            if n_valid >= 3:
                frac = neg / n_valid
                if frac > self.neg_roll_fraction:
                    warns.append(
                        f"{neg}/{n_valid} of last {n5} rolls have negative "
                        f"advantages_mean (frac {frac:.2f} > threshold "
                        f"{self.neg_roll_fraction:.2f}) — gradient direction "
                        f"is inconsistent across rollouts")

        # 4) Absolute per_game_reward_mean floor (after warmup roll 5+).
        if (isinstance(cur, (int, float))
                and rec["roll"] >= 5
                and cur < self.min_reward_floor):
            warns.append(
                f"per_game_reward_mean = {cur:+.3f} < floor "
                f"{self.min_reward_floor:+.3f} after warmup — the policy is "
                f"losing on its training objective; check OPP-BREAKDOWN to "
                f"see which opponent kind is dragging the mean")

        # 5) Training-step reward mean (from TRL log) has gone NEGATIVE.
        train_rwd_last3 = [r["train_reward_mean"] for r in self.history[-3:]
                           if isinstance(r.get("train_reward_mean"),
                                         (int, float))]
        if len(train_rwd_last3) >= 3 and all(v < 0 for v in train_rwd_last3):
            warns.append(
                f"train_reward_mean (TRL log) has been NEGATIVE for 3 "
                f"consecutive rollouts: {train_rwd_last3} — the policy is "
                f"net-losing on its own objective. Apimix went into negative "
                f"territory and never recovered.")

        return warns

    # ------------------------------------------------------------------ #
    # Pretty-print one block (called from rollout driver)                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def format_block(roll_index: int, alerts: dict) -> str:
        if not alerts.get("warn"):
            return ""
        tag = f"roll {roll_index:03d}"
        lines = [f"  [{tag}] HEALTH-WARN ⚠  ({alerts['consec_alert_rolls']} "
                 f"consec alert-rolls):"]
        for w in alerts["warn"]:
            lines.append(f"  [{tag}]    • {w}")
        if alerts.get("should_abort"):
            lines.append(
                f"  [{tag}] HEALTH-ABORT 🛑  raising RuntimeError (env "
                f"PHASE3_HEALTH_ABORT=1)")
        return "\n".join(lines)


def install_health_monitor_from_env() -> RolloutHealthMonitor | None:
    """Construct a monitor if telemetry is enabled, else return None.

    Coupled to PHASE3_TELEMETRY=1 so health alerts ride along with the same
    opt-in switch as the rest of the per-rollout telemetry. Fine-grained
    threshold/abort tuning is via the dedicated PHASE3_HEALTH_* env vars
    documented on the class.
    """
    if os.environ.get("PHASE3_TELEMETRY", "0") != "1":
        return None
    return RolloutHealthMonitor()
