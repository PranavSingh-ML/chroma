"""All tunables for the pod-side server. Everything can be overridden by env var.

Switching models = change MODEL_* here (or via env). No code changes.
"""
from __future__ import annotations

import os


def _env(name: str, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


# ---- model -----------------------------------------------------------------
# lodestones/Chroma1-HD (Apache 2.0, not gated). ONE download shared by inference and training:
#   - the transformer is the single-file `Chroma1-HD.safetensors` (17.8 GB) - exactly the file
#     ai-toolkit's chroma loader fetches - loaded with ChromaTransformer2DModel.from_single_file
#   - T5-XXL + tokenizer + VAE come from ostris/Flex.1-alpha (text_encoder_2 / tokenizer_2 / vae,
#     ~9.7 GB) - again the repo ai-toolkit's chroma loader uses, so the HF cache is shared.
# Fallback: Chroma1-Base (same file layout in its repo). Set MODEL_REPO=lodestones/Chroma1-Base
# and MODEL_FILE=Chroma1-Base.safetensors.
MODEL_REPO: str = _env("MODEL_REPO", "lodestones/Chroma1-HD")
MODEL_FILE: str = _env("MODEL_FILE", "Chroma1-HD.safetensors")
MODEL_REVISION: str | None = _env("MODEL_REVISION", "0e0c60ece1e82b17cb7f77342d765ba5024c40c0")  # verified 2026-09-21
EXTRAS_REPO: str = _env("EXTRAS_REPO", "ostris/Flex.1-alpha")   # T5 / tokenizer / VAE
EXTRAS_REVISION: str | None = _env("EXTRAS_REVISION", "5dd0b1bc2a9421b891abcf7c9218993f696dd550")  # verified 2026-09-21
EXTRAS_TEXT_ENCODER_SUBFOLDER: str = _env("EXTRAS_TEXT_ENCODER_SUBFOLDER", "text_encoder_2")
EXTRAS_TOKENIZER_SUBFOLDER: str = _env("EXTRAS_TOKENIZER_SUBFOLDER", "tokenizer_2")
EXTRAS_VAE_SUBFOLDER: str = _env("EXTRAS_VAE_SUBFOLDER", "vae")
# Scheduler config (a 300-byte json) comes from the model repo's diffusers layout.
SCHEDULER_REPO: str = _env("SCHEDULER_REPO", MODEL_REPO)

# ---- precision ---------------------------------------------------------------
# bf16 everywhere: 17.8 + 9.5 + 0.2 GB ~= 28 GB resident on a 48 GB card. No quantisation.
# TEXT_ENCODER_FP8=1 casts T5 to fp8 storage (diffusers layerwise casting) if Phase 0 shows
# peak VRAM > 44 GB. Not expected.
TEXT_ENCODER_FP8: bool = _env("TEXT_ENCODER_FP8", False)

# ---- generation defaults ---------------------------------------------------
DEFAULT_STEPS: int = _env("DEFAULT_STEPS", 26)
DEFAULT_GUIDANCE: float = _env("DEFAULT_GUIDANCE", 4.0)    # real CFG - Chroma has no guidance embedding
DEFAULT_NEGATIVE: str = _env("DEFAULT_NEGATIVE", "")
DEFAULT_WIDTH: int = _env("DEFAULT_WIDTH", 1024)
DEFAULT_HEIGHT: int = _env("DEFAULT_HEIGHT", 1024)
MAX_AREA: int = _env("MAX_AREA", 1024 * 1024 * 3 // 2)      # 1.5 MP cap per image
MIN_SIDE: int = 256
MAX_STEPS: int = _env("MAX_STEPS", 60)
MAX_IMAGES_PER_JOB: int = _env("MAX_IMAGES_PER_JOB", 4)
MAX_LORAS_PER_JOB: int = 2
SIZE_MULTIPLE: int = 16                                        # Flux VAE: 8 * patch 2

# ---- LoRA registry -----------------------------------------------------------
LORA_DIR: str = _env("LORA_DIR", "/workspace/loras")

# ---- captioning --------------------------------------------------------------
# Vision-language model used by POST /caption to write training captions ON THE POD, so photos
# never leave the user's own machines. 8.9 GB, ungated (verified 2026-09-22). Loaded on demand
# and unloaded right after, so it never competes with training for VRAM.
CAPTION_MODEL: str = _env("CAPTION_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
CAPTION_MODEL_REVISION: str | None = _env("CAPTION_MODEL_REVISION", "ebb281ec70b0")

# ---- training ----------------------------------------------------------------
AI_TOOLKIT_DIR: str = _env("AI_TOOLKIT_DIR", "/opt/ai-toolkit")
TRAIN_TEMPLATE: str = _env("TRAIN_TEMPLATE", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "train_lora_chroma.template.yaml"))
TRAIN_ROOT: str = _env("TRAIN_ROOT", "/workspace/train")     # datasets + output per job, deleted after
TRAIN_MAX_ZIP_MB: int = _env("TRAIN_MAX_ZIP_MB", 200)
TRAIN_MAX_STEPS: int = _env("TRAIN_MAX_STEPS", 6000)
# defaults for the template; the request's `config` JSON overrides any of these
TRAIN_DEFAULTS: dict = {
    "rank": 16,
    "alpha": 16,
    "lr": 1e-4,
    "steps": None,             # None -> 100 x n_images, clamped to [1000, 3000]
    "resolutions": [512, 768, 1024],
    "batch_size": 1,
    "save_every": 250,
    "sample_every": 250,
    "sample_steps": 20,
    "sample_guidance": 4.0,
    "sample_width": 1024,
    "sample_height": 1024,
    "caption_dropout": 0.05,
    "optimizer": "adamw8bit",
    "ema": True,
    "quantize_base": False,    # 48 GB: train against the bf16 base
    "trigger_word": "",
    "sample_prompts": [],
}

# ---- server ----------------------------------------------------------------
API_TOKEN: str = _env("API_TOKEN", "")
HOST: str = _env("HOST", "0.0.0.0")       # MUST be 0.0.0.0 or the RunPod proxy sees nothing
PORT: int = _env("PORT", 8000)
JOB_TTL_S: int = _env("JOB_TTL_S", 30 * 60)
JOB_CAP: int = _env("JOB_CAP", 50)
WARMUP: bool = _env("WARMUP", True)       # one throwaway 512px generate at startup before model_loaded flips true
MOCK: bool = _env("MOCK", False)          # laptop-only: fake pipeline + fake trainer, no torch needed
LOG_PATH: str = _env("LOG_PATH", "/workspace/jobs.jsonl")
