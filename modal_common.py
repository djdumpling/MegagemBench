"""Shared Modal infrastructure for the MegaGem training/eval/release apps.

Defines the pinned GPU image, the shared `modal.App`, Volumes, Secrets, and the
vLLM serving helpers. Consumers: `_serve_vllm` + `SFT_BLUEPRINT` are used by
modal_release.py; `_serve_one` + `LOCAL_VLLM_PATH` by modal_eval.py;
modal_train.py shells out to scripts/training/run_phase3.sh, which brings up
vLLM itself. modal_play.py is deliberately standalone (its own app and image)
so interactive play never has to resolve the training secrets.

NOTE: modal_train.py, modal_eval.py, and modal_release.py all attach their
functions to the ONE app defined here, so each file exposes only its own slice
of it. That is fine for `modal run <file>::<entrypoint>` (the documented
workflow) but means `modal deploy <one file>` would publish a PARTIAL "megagem"
app and drop the other files' functions. Deploy from a module that imports all
three, or not at all.

Image strategy: ABI floor (``_ABI_PINS``) installed in one resolve, frozen
into a pip constraints file, then pure-python extras (``_EXTRAS``) install
WITH deps UNDER that constraint — transitive closure stays consistent while
torch/vLLM/flash-attn/transformers stay immovable. The repo is a runtime
mount, so code edits never rebuild the image.

ONE-TIME SETUP:
  pip install modal && modal token new
  modal secret create huggingface-secret HF_TOKEN=hf_xxx     # private SFT
  modal secret create pi-secret PRIME_API_KEY=...            # API opponents
  modal secret create wandb-secret WANDB_API_KEY=...         # phase-3 logs

NEVER ``uv run`` from these containers — it clobbers the pinned torch.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import time

import modal

CUDA_TAG = "12.8.1-devel-ubuntu24.04"
PY = "3.13"                                # 3.13 + torch from vllm 0.10.2 + cu12
TRL_UPSTREAM_SHA = "5da60783af85a180f10235e564f37e4da67cc01d"
# TRL fork == public huggingface/trl@<sha> + one appended function
# (is_vllm_available→False). Reconstructed in-image from PUBLIC upstream so
# no GitHub credential and no private fork host are needed.
_TRL_PATCH = (
    "\\n\\n# MEGAGEM-VLLM0102-COMPAT\\n"
    "def is_vllm_available(*a, **k):\\n    return False\\n"
)

REPO = pathlib.Path(__file__).parent
HF_CACHE = "/hf-cache"
RESULTS_DIR = "/results"

# ABI floor — installed as ONE resolve so transitive deps can't float. flash-attn
# is installed after, --no-build-isolation, to match the just-pinned torch.
_ABI_PINS = (
    "vllm==0.10.2 transformers==4.55.4 'huggingface-hub>=0.34.0,<1.0' "
    "'numpy<2.3' openai"
)
# Pure-python extras, installed WITH deps but UNDER the constraints file
# generated below. `pandas` explicit (trl.trainer.grpo_trainer imports it at
# module load); `hf_transfer` explicit (env-gated by HF_HUB_ENABLE_HF_TRANSFER,
# nothing's transitive dep); `pytest` for the seam-suite prep gate; `wandb` so
# TRL's `report_to=["wandb"]` import succeeds (inert when WANDB_MODE=disabled).
_EXTRAS = (
    "peft accelerate datasets rich verifiers pandas hf_transfer pytest wandb "
    # E1 ev_dist selector: unpickles the frozen HistGradientBoosting artifact
    # (megagem/assets/ev_dist_v1.pkl). Version-pinned to the fit env —
    # sklearn pickles are not stable across minors (artifact meta records it).
    "scikit-learn==1.8.0 "
    # TrueSkill ratings for the local-trio / BIBD evals (pure-python, tiny dep).
    "trueskill"
)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY)
    .apt_install("git", "build-essential", "g++", "curl", "ca-certificates")
    .run_commands(
        "pip install --upgrade pip wheel setuptools",
        f"pip install {_ABI_PINS}",                          # ABI floor
        "pip install flash-attn --no-build-isolation",       # match pinned torch
        # TRL fork — public upstream + 1-line vLLM-guard patch, --no-deps so the
        # patched fork can't drag an ABI-breaking torch/transformers.
        "git clone --filter=blob:none https://github.com/huggingface/trl.git "
        "/opt/trl-fork",
        f"git -C /opt/trl-fork checkout {TRL_UPSTREAM_SHA}",
        f"python -c \"p='/opt/trl-fork/trl/import_utils.py'; "
        f"open(p,'a').write('{_TRL_PATCH}')\"",
        "pip install --no-deps /opt/trl-fork",
        # Freeze the ABI-critical set as a pip constraints file (only the
        # installed compiled/ABI pkgs are grepped, so absent ones e.g. pandas
        # remain free to resolve against numpy<2.3 below).
        "pip freeze | grep -iE "
        "'^(torch|torchvision|torchaudio|triton|xformers|vllm|transformers|"
        "tokenizers|safetensors|sentencepiece|numpy|flash[-_]attn|"
        "huggingface[-_]hub|nvidia[-_])' > /opt/abi-constraints.txt",
        f"pip install -c /opt/abi-constraints.txt {_EXTRAS}",
    )
    .env(
        {
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
            "WANDB_MODE": "disabled",          # phase3 flips when secret present
            "HF_HOME": HF_CACHE,
            "HF_HUB_CACHE": f"{HF_CACHE}/hub",
            "PYBIN": "python",                  # run_phase*.sh consults this
            # Make the megagem package and the (uninstalled) script drivers
            # importable in every container process and subprocess.
            "PYTHONPATH": "/repo/src:/repo/scripts/training:/repo/scripts/eval:/repo/scripts/analysis",
        }
    )
    .workdir("/repo")
    # Repo as a RUNTIME mount: code edits don't rebuild the image. Modal forbids
    # any build step (incl. .workdir / .env) after `add_local_*`, so this is LAST.
    .add_local_dir(
        REPO,
        "/repo",
        ignore=["**/.git", "**/__pycache__", "**/.venv", "prime-rl",
                "results", "**/*.pyc", "**/.pytest_cache", "dist"],
    )
)

app = modal.App("megagem", image=image)

hf_cache = modal.Volume.from_name("megagem-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("megagem-results", create_if_missing=True)
# Persistent vLLM Dynamo + cudagraph cache (~5-10 min cold compile/4B otherwise).
VLLM_CACHE_DIR = "/root/.cache/vllm"
vllm_cache = modal.Volume.from_name("megagem-vllm-cache", create_if_missing=True)
# Secrets — see module docstring for the `modal secret create` commands.
hf_secret = modal.Secret.from_name("huggingface-secret")  # HF_TOKEN (private SFT)
prime_secret = modal.Secret.from_name("pi-secret")        # PRIME_API_KEY (APIs)
# WANDB_API_KEY: phase3 entrypoints flip WANDB_MODE→online + pre-gen
# WANDB_RUN_ID when present so the §3.6 eval subprocess resumes the trainer's
# run. Absent ⇒ image default WANDB_MODE=disabled keeps wandb quiet.
wandb_secret = modal.Secret.from_name("wandb-secret")

GPU = os.environ.get("MODAL_GPU", "H200")   # H200 default; H100-80GB also fine

SFT_BLUEPRINT = "djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2"

# For the 3 LOCAL BIBD models: registry id (the label in results) -> the real HF
# repo vLLM must download. Only the base needs remapping ("qwen/qwen3-4b-instruct"
# is a served-name alias, not a repo); the djdumpling repos serve directly.
LOCAL_VLLM_PATH = {"qwen/qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507"}
SERVED_NAME = "qwen/qwen3-4b-instruct"


def _hf_token_ok(model: str) -> str | None:
    """Set HF_TOKEN in env and confirm it can read ``model``; return an error
    string or None. The SFT blueprint repo is private."""
    tok = (os.environ.get("HF_TOKEN")
           or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not tok:
        return "HF_TOKEN not in container env — huggingface-secret missing."
    os.environ["HF_TOKEN"] = tok
    os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(model, token=tok)
    except Exception as e:  # noqa: BLE001
        return f"HF_TOKEN cannot access {model!r}: {type(e).__name__}: {e}"
    return None


@contextlib.contextmanager
def _serve_vllm(model_path: str, log_path: str, *, port: int = 8000,
                max_model_len: int = 32768, ready_timeout_s: int = 1200,
                extra_args: list[str] | None = None):
    """Start `vllm serve` (served as SERVED_NAME), wait for /v1/models, yield the
    base URL, tear down on exit. Mirrors scripts/eval/eval_qwen_baseline.sh.
    ``extra_args`` (e.g. --enable-lora) append to the command; default None is
    byte-identical to the legacy invocation."""
    import shlex
    import subprocess
    import urllib.request

    pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    url = f"http://localhost:{port}/v1"
    command = [
        "vllm", "serve", model_path,
        "--served-model-name", SERVED_NAME,
        "--max-model-len", str(max_model_len),
        "--host", "0.0.0.0",
        "--port", str(port),
        *list(extra_args or []),
    ]
    print(f"[serve] starting: {shlex.join(command)}", flush=True)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            command,
            stdout=logf, stderr=subprocess.STDOUT, cwd="/repo", env={**os.environ},
        )

        def startup_error(reason: str) -> RuntimeError:
            """Put the otherwise-container-local vLLM traceback in Modal logs."""
            logf.flush()
            try:
                lines = pathlib.Path(log_path).read_text(
                    errors="replace").splitlines()
                tail = "\n".join(lines[-120:])
            except Exception as exc:  # noqa: BLE001
                tail = f"<could not read vLLM log: {type(exc).__name__}: {exc}>"
            print(
                "\n========== vLLM startup log (last 120 lines) ==========\n"
                f"{tail or '<log is empty>'}\n"
                "========== end vLLM startup log ==========\n",
                flush=True,
            )
            return RuntimeError(
                f"{reason}; vLLM exit code={proc.poll()}; full log: {log_path}")

        try:
            elapsed = 0
            while elapsed < ready_timeout_s:
                if proc.poll() is not None:
                    raise startup_error("vLLM died during startup")
                try:
                    urllib.request.urlopen(f"{url}/models", timeout=5)
                    print(f"[serve] vLLM ready after {elapsed}s ({model_path})",
                          flush=True)
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(5)
                    elapsed += 5
                    if elapsed % 30 == 0:
                        print(f"[serve] still warming up ({elapsed}s elapsed)",
                              flush=True)
            else:
                raise startup_error(
                    f"vLLM not ready in {ready_timeout_s}s")
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                proc.kill()


def _serve_one(model_path, served_name, port, gpu_frac, log_path, ready_timeout_s):
    """Launch one `vllm serve` on its own port/served-name with a GPU-memory
    fraction (so several share one card). Returns (proc, url)."""
    import subprocess
    import urllib.request
    pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    url = f"http://localhost:{port}/v1"
    logf = open(log_path, "w")
    proc = subprocess.Popen(
        ["vllm", "serve", model_path, "--served-model-name", served_name,
         "--max-model-len", "32768", "--host", "0.0.0.0", "--port", str(port),
         "--gpu-memory-utilization", str(gpu_frac)],
        stdout=logf, stderr=subprocess.STDOUT, cwd="/repo", env={**os.environ})
    elapsed = 0
    while elapsed < ready_timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM died serving {model_path} — see {log_path}")
        try:
            urllib.request.urlopen(f"{url}/models", timeout=5)
            print(f"[serve] ready :{port} {served_name} ({model_path}) after {elapsed}s", flush=True)
            return proc, url
        except Exception:  # noqa: BLE001
            time.sleep(5); elapsed += 5
    raise RuntimeError(f"vLLM not ready in {ready_timeout_s}s for {model_path} — see {log_path}")

