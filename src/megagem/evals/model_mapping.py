# ACTIVE BIBD tournament set = the 11 models listed in BIBD_MODELS below (2026-06-13
# v2: dropped glm-4.7 #9, gpt-oss-120b #2, intellect-3 #10 — all timed out on the
# PRIME gateway under concurrency — and added moonshotai/kimi-k2.6 #23).
# Indices 11-13 are LOCAL (served via vLLM, registered in the endpoints config);
# the rest route to the PRIME gateway. ALL indices stay defined so older result
# files / analyses still resolve by model id; only BIBD_MODELS controls the
# tournament. The cyclic schedule in generate_bibd_schedule.py is built over
# BIBD_MODELS, so the design is label-agnostic.
#   NOTE: #3 swapped google/gemini-3-pro-preview -> gemini-3.1-pro-preview
#   (the 3-pro preview 404s at the gateway; the Pro panel already used 3.1).
MODELS = {
    1:  ("openai/gpt-5.4", "GPT-5.4"),
    2:  ("openai/gpt-oss-120b", "GPT-OSS-120B"),
    3:  ("google/gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview"),
    4:  ("google/gemini-3-flash-preview", "Gemini 3 Flash Preview"),
    5:  ("x-ai/grok-4.20", "Grok-4.20"),
    6:  ("anthropic/claude-opus-4.8", "Claude Opus 4.8"),
    7:  ("anthropic/claude-sonnet-4.6", "Sonnet 4.6"),
    8:  ("meta-llama/llama-4-maverick", "Llama 4 Maverick"),
    9:  ("z-ai/glm-4.7", "GLM-4.7"),
    10: ("prime-intellect/intellect-3", "Intellect-3"),
    11: ("qwen/qwen3-4b-instruct", "Qwen3-4B Instruct (base)"),
    12: ("djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2", "MegaGem SFT (4B)"),
    13: ("djdumpling/qwen3-4b-instruct-megagem-distill-para", "MegaGem Distilled (4B)"),
    # --- preserved (NOT in the current BIBD; kept so old results still resolve) ---
    14: ("openai/gpt-5.2", "GPT-5.2"),
    15: ("anthropic/claude-opus-4.6", "Claude Opus 4.6"),
    16: ("anthropic/claude-opus-4.5", "Claude Opus 4.5"),
    17: ("anthropic/claude-sonnet-4.5", "Sonnet 4.5"),
    18: ("x-ai/grok-4", "Grok-4"),
    19: ("deepseek/deepseek-r1-0528", "DeepSeek R1-0528"),
    20: ("mistralai/mistral-large-2512", "Mistral Large 2512"),
    21: ("qwen/qwen3-max", "Qwen3 Max"),
    22: ("google/gemini-3-pro-preview", "Gemini 3 Pro Preview (RETIRED 404)"),
    23: ("moonshotai/kimi-k2.6", "Kimi K2.6 (DROPPED — PRIME timeout grind)"),
    24: ("openai/gpt-5.5", "GPT-5.5"),
    25: ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"),
    26: ("google/gemini-2.5-flash", "Gemini 2.5 Flash"),
    27: ("openai/gpt-4.1", "GPT-4.1"),
}

# The ACTIVE BIBD tournament set = 13 models (2026-06-13 v4, STS(13) clean design):
#   6 frontier : gpt-5.5(24) opus-4.8(6) sonnet-4.6(7) gemini-3.1-pro(3) grok-4.20(5) gemini-3-flash(4)
#   4 lower    : gpt-4.1(27) mistral-large-2512(20) gemini-2.5-flash(26) haiku-4.5(25)
#   3 local    : base(11) SFT(12) distilled(13)   [served via vLLM]
# Dropped for PRIME timeouts: gpt-oss-120b(2), glm-4.7(9), intellect-3(10), kimi-k2.6(23),
#   llama-4-maverick(8) (recovered on retry but still timed out — user wants zero).
# gpt-5.4(1) -> gpt-5.5(24); llama(8) -> gpt-4.1(27). mistral-large(20) = open-weight watch-item.
BIBD_MODELS = [24, 6, 7, 3, 5, 4, 27, 20, 26, 25, 11, 12, 13]
NUM_BIBD_MODELS = len(BIBD_MODELS)  # 13

# NUM_MODELS stays len(MODELS) for backward-compat with analyzers that iterate the
# full table (0-game entries are skipped in reports).

MODEL_TO_NUMBER = {model_id: num for num, (model_id, _) in MODELS.items()}

NUM_MODELS = len(MODELS)


def get_model_for_number(num: int) -> str:
    if num not in MODELS:
        raise ValueError(f"Invalid model number: {num}. Must be 1-{NUM_MODELS}.")
    return MODELS[num][0]


def get_number_for_model(model: str) -> int:
    if model not in MODEL_TO_NUMBER:
        raise ValueError(f"Model not found in mapping: {model}")
    return MODEL_TO_NUMBER[model]


def get_model_name(num: int) -> str:
    return MODELS.get(num, (None, f"Model {num}"))[1]
