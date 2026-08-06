"""Standalone Modal app for playing against the public distilled checkpoint.

This file intentionally does not import the shared training app
(``modal_common.py``). That app registers the training/evaluation stack and
its private secrets; interactive play needs only vLLM, the public model, the
game code, and the two selector artifacts that ship inside the repo mount
(``src/megagem/assets/``).

Run:

    uvx modal run -i modal_play.py::play_distilled --loop
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shlex
import time

import modal


MODEL = "djdumpling/qwen3-4b-instruct-megagem-distill-para"
MODEL_REVISION = "cd2074eeba56e23edb2625ed58ac4b5778a990c1"
SERVED_NAME = "megagem-distilled"
HF_CACHE_DIR = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"
# Selector artifacts ship inside the repo mount (see src/megagem/assets/).
EV_MODEL_PATH = "/repo/src/megagem/assets/ev_dist_l2_v1.pkl"
EV_VALUE_HEAD_PATH = "/repo/src/megagem/assets/value_head.pkl"
REPO = pathlib.Path(__file__).parent

# This serving image is deliberately separate from the pinned Phase-3 training
# image. It follows Modal's current vLLM image pattern and has no TRL/PEFT or
# private repository dependencies.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.21.0",
        "openai>=1.0.0",
        "rich>=13.0.0",
        "scikit-learn==1.8.0",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/repo/src",  # import megagem from the repo mount
        }
    )
    .workdir("/repo")
    .add_local_dir(
        REPO,
        "/repo",
        ignore=[
            "**/.git",
            "**/__pycache__",
            "**/.venv",
            "prime-rl",
            "results",
            "**/*.pyc",
            "**/.pytest_cache",
            "dist",
        ],
    )
)

app = modal.App("megagem-distilled-play", image=image)
hf_cache = modal.Volume.from_name("megagem-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("megagem-vllm-cache", create_if_missing=True)


def _require_selector_artifacts() -> None:
    missing = [
        path
        for path in (EV_MODEL_PATH, EV_VALUE_HEAD_PATH)
        if not pathlib.Path(path).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Canonical selector artifact(s) missing from the repo mount: "
            + ", ".join(missing)
            + ". They ship in src/megagem/assets/ — check your checkout."
        )


def _log_tail(path: pathlib.Path, lines: int = 120) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except Exception as exc:  # noqa: BLE001
        return f"<could not read vLLM log: {type(exc).__name__}: {exc}>"


@contextlib.contextmanager
def _serve_vllm(*, ready_timeout_s: int = 1800):
    """Start local vLLM, wait for its OpenAI API, and always show useful logs."""
    import subprocess
    import urllib.request

    log_path = pathlib.Path("/tmp/megagem-distilled-vllm.log")
    url = "http://127.0.0.1:8000/v1"
    command = [
        "vllm",
        "serve",
        MODEL,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        SERVED_NAME,
        "--max-model-len",
        "32768",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--enable-prefix-caching",
        # Interactive play is serial and latency-sensitive at startup.  Avoid
        # the long TorchInductor/CUDA-graph compile paid by throughput servers.
        "--enforce-eager",
    ]
    print(f"[serve] starting: {shlex.join(command)}", flush=True)

    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd="/repo",
            env={**os.environ},
        )

        def startup_error(reason: str) -> RuntimeError:
            log_file.flush()
            tail = _log_tail(log_path)
            print(
                "\n========== vLLM startup log (last 120 lines) ==========\n"
                f"{tail or '<log is empty>'}\n"
                "========== end vLLM startup log ==========\n",
                flush=True,
            )
            return RuntimeError(
                f"{reason}; vLLM exit code={proc.poll()}; full log: {log_path}"
            )

        try:
            elapsed = 0
            while elapsed < ready_timeout_s:
                if proc.poll() is not None:
                    raise startup_error("vLLM died during startup")
                try:
                    with urllib.request.urlopen(f"{url}/models", timeout=5):
                        pass
                    print(f"[serve] model ready after {elapsed}s", flush=True)
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(5)
                    elapsed += 5
                    if elapsed % 30 == 0:
                        print(
                            f"[serve] still warming up ({elapsed}s elapsed)",
                            flush=True,
                        )
            else:
                raise startup_error(f"vLLM not ready in {ready_timeout_s}s")
            yield url
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)


def _commit_caches() -> None:
    # Preserve public weights and compiled kernels across separate runs, even
    # when the user exits with Ctrl-C.
    with contextlib.suppress(Exception):
        hf_cache.commit()
    with contextlib.suppress(Exception):
        vllm_cache.commit()


@app.function(
    gpu="H100",
    timeout=21600,
    volumes={
        HF_CACHE_DIR: hf_cache,
        VLLM_CACHE_DIR: vllm_cache,
    },
)
def play_distilled(
    value_chart: str = "A",
    seed: int = -1,
    once: bool = False,
    loop: bool = False,
    weights_only: bool = False,
    vllm_ready_timeout_s: int = 1800,
) -> int:
    """Warm the public model, then launch the selector-backed game by default."""
    import subprocess
    import sys

    if once and loop:
        raise ValueError("once and loop are mutually exclusive")
    value_chart = value_chart.strip().upper()
    if value_chart not in {"A", "B", "C", "D", "E"}:
        raise ValueError("value_chart must be one of A, B, C, D, E")
    if not weights_only:
        _require_selector_artifacts()

    try:
        with _serve_vllm(ready_timeout_s=vllm_ready_timeout_s) as url:
            # vLLM never needs stdin. Attach only after warmup, immediately
            # before the child game begins calling input().
            try:
                modal.interact()
            except modal.exception.InternalError as exc:
                # Modal 1.5.x can report that the CLI's interactive PTY is
                # already active instead of attaching interact() to it.  In
                # that case stdin is already connected, so reuse it.  Keep
                # every other interaction failure fatal.
                if "A PTY shell is already active" not in str(exc):
                    raise
                print("[play] reusing already-active Modal PTY", flush=True)
            command = [
                sys.executable,
                "-m",
                "megagem.play.distilled",
                "--endpoint",
                url,
                "--model",
                SERVED_NAME,
                "--value-chart",
                value_chart,
            ]
            if seed >= 0:
                command.extend(["--seed", str(seed)])
            if weights_only:
                command.append("--weights-only")
            if once:
                command.append("--once")
            elif loop:
                command.append("--loop")

            return subprocess.run(
                command,
                cwd="/repo",
                env={
                    **os.environ,
                    "MEGAGEM_API_URL": url,
                    "MEGAGEM_API_KEY": "EMPTY",
                },
            ).returncode
    finally:
        _commit_caches()


@app.function(cpu=1.0, timeout=600)
def check_selector() -> dict[str, str | int]:
    """Load the exact selector artifacts without allocating a GPU."""
    _require_selector_artifacts()
    from megagem.environment.ev_selector import EvDistSelector

    selector = EvDistSelector(
        model_path=EV_MODEL_PATH,
        value_head_path=EV_VALUE_HEAD_PATH,
        gate_min_ev=1.0,
        pacing_lam=0.5,
        vhat_debias=2.0,
    )
    result: dict[str, str | int] = {
        "status": "ok",
        "price_model": EV_MODEL_PATH,
        "price_residuals": len(selector.ev_model.residuals),
        "value_head": EV_VALUE_HEAD_PATH,
    }
    print(f"[check] selector artifacts: {result}", flush=True)
    return result


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={HF_CACHE_DIR: hf_cache, VLLM_CACHE_DIR: vllm_cache},
)
def check_distilled(vllm_ready_timeout_s: int = 1800) -> dict[str, str]:
    """Non-interactive startup and one-token generation smoke test."""
    import json
    import urllib.request

    try:
        with _serve_vllm(ready_timeout_s=vllm_ready_timeout_s) as url:
            body = json.dumps(
                {
                    "model": SERVED_NAME,
                    "messages": [{"role": "user", "content": "Reply: ok"}],
                    "max_tokens": 8,
                    "temperature": 0,
                }
            ).encode()
            request = urllib.request.Request(
                f"{url}/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
            content = result["choices"][0]["message"]["content"]
            print(f"[check] generation: {content!r}", flush=True)
            return {"status": "ok", "generation": content}
    finally:
        _commit_caches()
