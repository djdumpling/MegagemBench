"""Modal app: publish and verify the distilled MegaGem model on Hugging Face.

ENTRYPOINTS:
  modal run modal_release.py::upload_model_main          # merge LoRA -> push to HF
  modal run modal_release.py::verify_hf_model_main       # end-to-end play check

Shares one `modal.App` with modal_train.py / modal_eval.py via modal_common.
"""

from __future__ import annotations

import os
import pathlib

from modal_common import (
    GPU,
    HF_CACHE,
    RESULTS_DIR,
    SERVED_NAME,
    SFT_BLUEPRINT,
    VLLM_CACHE_DIR,
    _hf_token_ok,
    _serve_vllm,
    app,
    hf_cache,
    hf_secret,
    prime_secret,
    results_vol,
    vllm_cache,
)

# Selector artifacts now ship in the repo mount (megagem/assets), not the Volume.
EV_VALUE_HEAD_PATH = "/repo/src/megagem/assets/value_head.pkl"

# ---------------------------------------------------------------------------- #
# Publish the distilled SOTA model to HF: merge the para adapter (artifact of
# record, distill1a_d1a_para/adapter) onto its training base (the SFT blueprint
# SFT_BLUEPRINT — confirmed by the adapter's own adapter_config.json), so the
# merged repo carries SFT + distillation baked in and is servable by plain
# vLLM/transformers (no PEFT needed) exactly like …-step1200-v2. The test-time
# EV selector is CODE, not weights — it is NOT part of this upload; this repo is
# the weights-only champion (≈ +8 vs 2×Flash). Merge is bf16 to MATCH serving
# (bf16 vs fp32 merge delta ~0.3 is inherent to serving in bf16, a
# PASS signal). Private by default — publishing to the user's own account is
# reversible; do not make public without an explicit ask.
# ---------------------------------------------------------------------------- #
@app.function(
    gpu=GPU,
    timeout=7200,
    volumes={HF_CACHE: hf_cache, RESULTS_DIR: results_vol},
    secrets=[hf_secret],
)
def upload_model(
    adapter_path: str = "distill1a_d1a_para/adapter",
    base_model: str = SFT_BLUEPRINT,
    repo_id: str = "djdumpling/qwen3-4b-instruct-megagem-distill-para",
    private: bool = True,
    push_adapter: bool = True,
):
    import os

    tok = (os.environ.get("HF_TOKEN")
           or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if not tok:
        return {"rc": 2, "aborted": "HF_TOKEN missing (huggingface-secret); "
                "base + target repo both live under the djdumpling account."}
    os.environ["HF_TOKEN"] = tok

    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = f"{RESULTS_DIR}/{adapter_path}"
    if not pathlib.Path(f"{adapter_dir}/adapter_config.json").exists():
        return {"rc": 2, "aborted": f"adapter not found on volume: {adapter_dir}"}

    print(f"[upload] base   = {base_model}")
    print(f"[upload] adapter= {adapter_dir}")
    print(f"[upload] target = {repo_id} (private={private})")

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, token=tok)
    peft = PeftModel.from_pretrained(base, adapter_dir)
    merged = peft.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=tok)

    # Sanity: a short generate on the merged weights BEFORE publishing, so a
    # broken merge never reaches HF.
    sanity = None
    try:
        merged.eval()
        msgs = [{"role": "user", "content": "Reply with the single word: ok"}]
        ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(merged.device)
        with torch.no_grad():
            out = merged.generate(ids, max_new_tokens=8, do_sample=False)
        sanity = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"[upload] sanity generate -> {sanity!r}")
    except Exception as e:  # noqa: BLE001
        return {"rc": 3, "aborted": f"merged-model sanity generate FAILED: {e}"}

    out_dir = "/tmp/merged_distill_para"
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    card = (
        "---\nlicense: apache-2.0\nbase_model: " + base_model + "\n"
        "tags:\n- megagem\n- qwen3\n- lora-merged\npipeline_tag: text-generation\n---\n\n"
        "# Qwen3-4B-Instruct · MegaGem distilled (paraphrase) — weights-only champion\n\n"
        "Merged weights: **" + base_model + "** (SFT blueprint) + the round-1 "
        "paraphrase distillation LoRA (`distill1a_d1a_para`). SFT + distillation "
        "are baked in; serve with plain vLLM/transformers.\n\n"
        "Trained via **ExIt with an analytic expert**: self-play games → a fitted "
        "opponent bid-distribution price law (F̂) + a self-play value head (V̂) → "
        "closed-form expected-surplus best-response labels on deviation turns → "
        "self-paraphrased in the blueprint's own voice → SFT LoRA. No RL gradient; "
        "no opponent-API data in any learned component (the price law is fit on the "
        "blueprint's OWN self-play bids).\n\n"
        "Held-out vs 2×Gemini-3-Flash (paired CRN, weights-only, NO test-time "
        "selector): 29.3% → 40.7% win-rate, CV-adjusted +7.4 score/game (t≈3.0). "
        "The deployable champion adds a zero-LLM-cost analytic EV selector at test "
        "time (≈ +11–14 vs Flash/Pro/Sonnet) — that selector is code, not these "
        "weights.\n"
    )
    pathlib.Path(f"{out_dir}/README.md").write_text(card)

    api = HfApi(token=tok)
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=out_dir, repo_id=repo_id, repo_type="model",
                      commit_message="Merged SFT blueprint + paraphrase distill LoRA")
    print(f"[upload] merged model pushed -> https://huggingface.co/{repo_id}")

    adapter_repo = None
    if push_adapter:
        adapter_repo = f"{repo_id}-lora"
        api.create_repo(adapter_repo, private=private, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=adapter_dir, repo_id=adapter_repo,
                          repo_type="model",
                          commit_message="Paraphrase distillation LoRA adapter (PEFT)")
        print(f"[upload] adapter pushed     -> https://huggingface.co/{adapter_repo}")

    return {"rc": 0, "repo_id": repo_id, "adapter_repo": adapter_repo,
            "private": private, "base_model": base_model,
            "sanity_generate": sanity}


@app.function(
    gpu=GPU,
    timeout=14400,
    volumes={HF_CACHE: hf_cache, RESULTS_DIR: results_vol, VLLM_CACHE_DIR: vllm_cache},
    secrets=[hf_secret, prime_secret],   # HF_TOKEN: private merged repo; PRIME: API opp
)
def verify_hf_model(
    *, model: str = "djdumpling/qwen3-4b-instruct-megagem-distill-para",
    opponent: str = "anthropic/claude-sonnet-4.6",
    num_seeds: int = 10, seed_start: int = 61000, max_parallel: int = 16,
    selector: bool = False, ev_model_path: str = "", ev_value_head_path: str = "",
    gate_min_ev: float = 1.0, pacing_lam: float = 0.5, vhat_debias: float = 2.0,
    vllm_ready_timeout_s: int = 1800, run_tag: str = "verify",
) -> dict:
    """Upload-sanity eval: serve the MERGED HF model (loaded from HF, not the
    volume adapter) and play it vs 2x ``opponent`` across a fully-crossed
    num_seeds × 3-seat design (= 3·num_seeds games). ``selector=False`` = the
    plain uploaded weights (weights-only champion); ``selector=True`` adds the
    test-time analytic EV selector (F̂₁ + λ + δ + gate) at the policy's rotating
    seat = the DEPLOYABLE champion. Driver: scripts/eval/verify_hf_eval.py."""
    import json
    import subprocess
    import sys as _sys

    err = _hf_token_ok(model)            # private merged repo needs HF_TOKEN access
    if err:
        return {"rc": 2, "aborted": err}
    if not os.environ.get("PRIME_API_KEY", "").strip():
        return {"rc": 2, "aborted": "PRIME_API_KEY missing (API opponent)"}
    out_dir = f"{RESULTS_DIR}/verify_hf_{run_tag}"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    results: dict = {"results_dir": out_dir, "model": model, "opponent": opponent,
                     "selector": selector,
                     "design": f"{num_seeds} decks x 3 seats = {3 * num_seeds} games"}

    sel_args: list[str] = []
    if selector:
        # Deployable artifact of record: F̂₁ (ev_dist_l2_v1) + value head, λ=.5/δ=2/gate=1.
        evm = ev_model_path or "/repo/src/megagem/assets/ev_dist_l2_v1.pkl"
        evh = ev_value_head_path or EV_VALUE_HEAD_PATH
        for p in (evm, evh):
            if not pathlib.Path(p).exists():
                return {"rc": 2, "aborted": f"selector artifact missing on volume: {p}"}
        sel_args = ["--selector", "--ev-model-path", evm, "--ev-value-head-path", evh,
                    "--gate-min-ev", str(gate_min_ev), "--pacing-lam", str(pacing_lam),
                    "--vhat-debias", str(vhat_debias)]
        results["selector_cfg"] = {"ev_model_path": evm, "ev_value_head_path": evh,
                                   "gate_min_ev": gate_min_ev, "pacing_lam": pacing_lam,
                                   "vhat_debias": vhat_debias}

    # No --enable-lora / --lora-modules: serve the MERGED weights directly under
    # SERVED_NAME, so the driver's policy endpoint IS the uploaded model.
    with _serve_vllm(model, f"{out_dir}/vllm.log",
                     ready_timeout_s=vllm_ready_timeout_s) as url:
        rc = subprocess.run(
            [_sys.executable, "scripts/eval/verify_hf_eval.py",
             "--vllm-url", url, "--opponent", opponent,
             "--num-seeds", str(num_seeds), "--seed-start", str(seed_start),
             "--out-dir", out_dir, "--max-parallel", str(max_parallel),
             "--served-name", SERVED_NAME, "--policy-name", "distill-para"] + sel_args,
            cwd="/repo", env={**os.environ}).returncode
    results_vol.commit()
    vllm_cache.commit()
    try:
        results["summary"] = json.loads(
            pathlib.Path(f"{out_dir}/verify_summary.json").read_text())
    except Exception:  # noqa: BLE001
        results["summary"] = None
    results["rc"] = rc
    return results


@app.local_entrypoint()
def verify_hf_model_main(
    model: str = "djdumpling/qwen3-4b-instruct-megagem-distill-para",
    opponent: str = "anthropic/claude-sonnet-4.6",
    num_seeds: int = 10, seed_start: int = 61000, max_parallel: int = 16,
    selector: bool = False, run_tag: str = "verify",
):
    import json
    res = verify_hf_model.remote(
        model=model, opponent=opponent, num_seeds=num_seeds,
        seed_start=seed_start, max_parallel=max_parallel, selector=selector,
        run_tag=run_tag)
    if res.get("aborted"):
        print(f"\nABORTED: {res['aborted']}")
        return
    print(f"\nverify_hf_model rc={res.get('rc')}  results={res['results_dir']}")
    print(f"  model={res.get('model')}  vs 2x {res.get('opponent')}  "
          f"selector={res.get('selector')}")
    print(f"  design={res.get('design')}")
    s = res.get("summary") or {}
    print(f"  games: {s.get('n_ok')} ok / {s.get('n_error')} error (of {s.get('n_games')})")
    print(f"  WIN-RATE={s.get('win_rate')}  mean-margin={s.get('mean_margin')}  "
          f"mean-score={s.get('mean_policy_score')}")
    print(f"  by-seat: {json.dumps(s.get('by_seat'))}")
    if s.get("errors"):
        print(f"  first errors: {json.dumps(s.get('errors')[:3])}")


@app.local_entrypoint()
def upload_model_main(
    adapter_path: str = "distill1a_d1a_para/adapter",
    base_model: str = SFT_BLUEPRINT,
    repo_id: str = "djdumpling/qwen3-4b-instruct-megagem-distill-para",
    private: bool = True,
    push_adapter: bool = True,
):
    res = upload_model.remote(
        adapter_path=adapter_path, base_model=base_model, repo_id=repo_id,
        private=private, push_adapter=push_adapter,
    )
    if res.get("aborted"):
        print(f"\nABORTED: {res['aborted']}")
        return
    print(f"\nupload_model rc={res.get('rc')}")
    print(f"  merged model : https://huggingface.co/{res.get('repo_id')}")
    if res.get("adapter_repo"):
        print(f"  lora adapter : https://huggingface.co/{res.get('adapter_repo')}")
    print(f"  private={res.get('private')}  base={res.get('base_model')}")


