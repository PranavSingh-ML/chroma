"""Normalise a Chroma LoRA file into a form `ChromaPipeline.load_lora_weights` accepts.

ChromaPipeline uses FluxLoraLoaderMixin (diffusers 0.40), whose `lora_state_dict` accepts:
  (a) diffusers/peft keys      transformer.transformer_blocks.0.attn.to_q.lora_A.weight
  (b) kohya keys               lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight  (+ optional .alpha)
  (c) ComfyUI keys, IF they use lora_down/lora_up:
                               diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight
      (it rewrites `diffusion_model.` -> `lora_unet_` and takes the kohya path, which splits the
       fused qkv / linear1 weights into diffusers' to_q/to_k/to_v etc.)

ai-toolkit's Chroma trainer saves ComfyUI-prefixed keys but with peft names (lora_A / lora_B and
no alpha), which matches none of the three -> the loader raises. This module renames
lora_A->lora_down, lora_B->lora_up for that case (case d). Without an alpha tensor the kohya
converter uses scale 1, which is correct only when alpha == rank - the training template
enforces that.

Phase 0 records which case a freshly trained LoRA falls in. If a new ai-toolkit SHA changes
the format, this is the one file to touch.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

# Modules diffusers' `_convert_kohya_flux_lora_to_diffusers` knows how to map (Flux layout,
# which Chroma shares for the blocks). Everything else is dropped by `normalize`.
_FLUX_MODULE_RE = re.compile(
    r"^diffusion_model\.("
    r"double_blocks\.\d+\.(img_attn\.(qkv|proj)|txt_attn\.(qkv|proj)|img_mlp\.[02]|txt_mlp\.[02]|img_mod\.lin|txt_mod\.lin)"
    r"|single_blocks\.\d+\.(linear1|linear2|modulation\.lin)"
    r"|img_in|txt_in|time_in\.(in_layer|out_layer)|vector_in\.(in_layer|out_layer)|guidance_in\.(in_layer|out_layer)"
    r"|final_layer\.(linear|adaLN_modulation\.1)"
    r")\.(lora_down\.weight|lora_up\.weight|alpha)$"
)


def detect(keys: Iterable[str]) -> str:
    keys = list(keys)
    if not keys:
        return "empty"
    if any(k.startswith("transformer.") for k in keys):
        return "diffusers"
    if any(".lora_down.weight" in k for k in keys):
        return "kohya" if any(k.startswith("lora_unet_") for k in keys) else "comfy_kohya"
    if any(k.startswith("diffusion_model.") and (".lora_A." in k or ".lora_B." in k) for k in keys):
        return "comfy_peft"
    return "unknown"


def normalize(src: str, dst: str | None = None) -> dict:
    """Write a loadable copy of `src` to `dst` (default: alongside, suffix .diffusers-ready.safetensors).

    Returns {"format": detected, "converted": bool, "path": path_to_load, "n_keys": int,
             "dropped": [keys that cannot map to the diffusers transformer]}.
    """
    from safetensors.torch import load_file, save_file

    sd = load_file(src)
    fmt = detect(sd.keys())
    dropped: list[str] = []
    out_path = src
    converted = False
    if fmt in ("comfy_peft", "comfy_kohya"):
        # diffusers' kohya->diffusers Flux converter RAISES on any key it cannot map, so
        # Chroma-only modules (distilled_guidance_layer, ...) must be filtered out here.
        new = {}
        for k, v in sd.items():
            nk = k.replace(".lora_A.weight", ".lora_down.weight").replace(".lora_B.weight", ".lora_up.weight")
            if not _FLUX_MODULE_RE.match(nk):
                dropped.append(k)
                continue
            new[nk] = v
        sd = new
        out_path = dst or (os.path.splitext(src)[0] + ".diffusers-ready.safetensors")
        save_file(sd, out_path)
        converted = True
    elif fmt == "unknown":
        raise ValueError(f"unrecognised LoRA key format; first keys: {list(sd.keys())[:5]}")
    return {"format": fmt, "converted": converted, "path": out_path, "n_keys": len(sd), "dropped": dropped}


def merge_manually(transformer, src: str, scale: float = 1.0) -> dict:
    """Last-resort path if `load_lora_weights` refuses the file even after `normalize`:
    fold W += scale * (B @ A) straight into the diffusers transformer, using diffusers' own
    kohya->diffusers converter to obtain `transformer.<module>.lora_A/B` keys.

    Returns {"applied": n, "missing": [module names not found]}. Keeps a CPU copy of the touched
    weights on `transformer._lora_backup` so `unmerge_manually` can restore them.
    """
    import torch
    from diffusers.loaders.lora_conversion_utils import _convert_kohya_flux_lora_to_diffusers
    from safetensors.torch import load_file

    info = normalize(src)
    sd = load_file(info["path"])
    if info["format"] != "diffusers":
        sd = {k.replace("diffusion_model.", "lora_unet_"): v for k, v in sd.items()}
        sd = _convert_kohya_flux_lora_to_diffusers(sd)
    mods = dict(transformer.named_modules())
    backup = getattr(transformer, "_lora_backup", {})
    applied, missing = 0, []
    for k in [k for k in sd if k.endswith(".lora_A.weight")]:
        base = k[len("transformer."):-len(".lora_A.weight")]
        b_key = k.replace(".lora_A.weight", ".lora_B.weight")
        mod = mods.get(base)
        if mod is None or b_key not in sd:
            missing.append(base)
            continue
        a, b = sd[k].to(torch.float32), sd[b_key].to(torch.float32)
        with torch.no_grad():
            w = mod.weight
            if base not in backup:
                backup[base] = w.detach().to("cpu", copy=True)
            w.add_((b @ a).to(w.device, w.dtype) * scale)
        applied += 1
    transformer._lora_backup = backup
    return {"applied": applied, "missing": missing, **info}


def unmerge_manually(transformer) -> int:
    import torch

    backup = getattr(transformer, "_lora_backup", {})
    mods = dict(transformer.named_modules())
    n = 0
    with torch.no_grad():
        for base, w0 in backup.items():
            mods[base].weight.copy_(w0.to(mods[base].weight.device))
            n += 1
    transformer._lora_backup = {}
    return n
