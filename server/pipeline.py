"""Chroma1-HD load / unload, LoRA registry, and a single generate() function. Nothing about HTTP here.

Weights are shared with the trainer (see config.py): the transformer comes from the single-file
`Chroma1-HD.safetensors`, T5 / tokenizer / VAE from ostris/Flex.1-alpha. Everything bf16, no
quantisation, ~28 GB resident on a 48 GB card (VERIFY in Phase 0).

The engine can be fully unloaded (`unload()`) so a training job can have the GPU, and reloaded
afterwards from the container-disk HF cache (no re-download).
"""
from __future__ import annotations

import gc
import io
import logging
import os
import shutil
import time
from typing import Callable

from PIL import Image, ImageDraw

import config
import lora_convert

log = logging.getLogger("pipeline")

ProgressCb = Callable[[float], None]


# ----------------------------------------------------------------------------
# helpers shared by real and mock engines
# ----------------------------------------------------------------------------
def _round_to(v: int, m: int) -> int:
    return max(m, (int(v) // m) * m)


def clamp_size(width: int | None, height: int | None) -> tuple[int, int]:
    """Multiples of 16, each side >= MIN_SIDE, total area <= MAX_AREA (aspect preserved)."""
    w = int(width or config.DEFAULT_WIDTH)
    h = int(height or config.DEFAULT_HEIGHT)
    w, h = max(w, config.MIN_SIDE), max(h, config.MIN_SIDE)
    if w * h > config.MAX_AREA:
        s = (config.MAX_AREA / (w * h)) ** 0.5
        w, h = int(w * s), int(h * s)
    return _round_to(w, config.SIZE_MULTIPLE), _round_to(h, config.SIZE_MULTIPLE)


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _safe_name(name: str) -> str:
    n = "".join(c if c.isalnum() or c in "-_." else "-" for c in name.strip())
    if not n or n.startswith(".") or n in ("-", "_"):
        raise ValueError("bad lora name")
    return n[:64]


# ----------------------------------------------------------------------------
# Mock engine: lets the whole HTTP surface + web app be tested on a laptop
# ----------------------------------------------------------------------------
class MockEngine:
    def __init__(self) -> None:
        self.info = {"gpu": "MOCK", "capability": [0, 0], "model": "mock", "vram_total_gb": 0}
        self.loras: dict[str, dict] = {}
        self.loaded = False

    def load(self) -> None:
        time.sleep(1.0)
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def vram_used_gb(self) -> float:
        return 0.0

    def add_lora(self, name: str, path: str) -> dict:
        name = _safe_name(name)
        self.loras[name] = {"name": name, "size_mb": round(os.path.getsize(path) / 1e6, 1), "loaded": True,
                            "format": "mock", "path": path}
        return self.loras[name]

    def remove_lora(self, name: str) -> bool:
        return self.loras.pop(name, None) is not None

    def list_loras(self) -> list[dict]:
        return [{k: v for k, v in d.items() if k != "path"} for d in self.loras.values()]

    def generate(self, prompt, negative, steps, guidance, seeds, width, height, loras, progress: ProgressCb):
        import random

        outs = []
        total = steps * len(seeds)
        done = 0
        for seed in seeds:
            rnd = random.Random(seed)
            img = Image.new("RGB", (width, height), (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255)))
            d = ImageDraw.Draw(img)
            for _ in range(40):
                x, y, r = rnd.randint(0, width), rnd.randint(0, height), rnd.randint(10, 120)
                d.ellipse([x - r, y - r, x + r, y + r], fill=(rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255)))
            d.rectangle([0, 0, width, 44], fill=(0, 0, 0))
            d.text((6, 6), f"MOCK seed={seed} steps={steps} cfg={guidance} loras={loras}", fill=(255, 255, 255))
            d.text((6, 24), prompt[:120], fill=(255, 255, 255))
            for _ in range(steps):
                time.sleep(0.05)
                done += 1
                progress(done / total)
            outs.append(img)
        return outs


# ----------------------------------------------------------------------------
# Real engine
# ----------------------------------------------------------------------------
class Engine:
    def __init__(self) -> None:
        self.pipe = None
        self.info: dict = {}
        self._torch = None
        self.loras: dict[str, dict] = {}   # name -> {name, path, size_mb, loaded, format, ...}
        self.loaded = False
        self.model_path: str | None = None  # local single-file path, handed to the trainer

    # -- weights -------------------------------------------------------------
    def ensure_weights(self) -> str:
        """Download (once) the single-file transformer; return its local path."""
        from huggingface_hub import hf_hub_download

        t0 = time.time()
        self.model_path = hf_hub_download(config.MODEL_REPO, config.MODEL_FILE, revision=config.MODEL_REVISION)
        self.info["download_s"] = round(time.time() - t0, 1)
        return self.model_path

    # -- loading -------------------------------------------------------------
    def load(self) -> None:
        import torch
        from diffusers import AutoencoderKL, ChromaPipeline, ChromaTransformer2DModel, FlowMatchEulerDiscreteScheduler
        from transformers import T5EncoderModel, T5TokenizerFast

        self._torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name()
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU=%s capability=%s vram=%.1fGB torch=%s cuda=%s", name, cap, total_gb, torch.__version__, torch.version.cuda)
        self.info.update({"gpu": name, "capability": list(cap), "model": f"{config.MODEL_REPO}/{config.MODEL_FILE}",
                          "torch": torch.__version__, "cuda": torch.version.cuda, "vram_total_gb": round(total_gb, 1)})
        dtype = torch.bfloat16
        t0 = time.time()
        path = self.ensure_weights()
        transformer = ChromaTransformer2DModel.from_single_file(
            path, config=config.MODEL_REPO, subfolder="transformer", revision=config.MODEL_REVISION, torch_dtype=dtype)
        text_encoder = T5EncoderModel.from_pretrained(
            config.EXTRAS_REPO, subfolder=config.EXTRAS_TEXT_ENCODER_SUBFOLDER, revision=config.EXTRAS_REVISION, torch_dtype=dtype)
        tokenizer = T5TokenizerFast.from_pretrained(
            config.EXTRAS_REPO, subfolder=config.EXTRAS_TOKENIZER_SUBFOLDER, revision=config.EXTRAS_REVISION)
        vae = AutoencoderKL.from_pretrained(
            config.EXTRAS_REPO, subfolder=config.EXTRAS_VAE_SUBFOLDER, revision=config.EXTRAS_REVISION, torch_dtype=dtype)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.SCHEDULER_REPO, subfolder="scheduler", revision=config.MODEL_REVISION)
        pipe = ChromaPipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                              transformer=transformer)
        pipe.to("cuda")
        if config.TEXT_ENCODER_FP8:
            from diffusers.hooks import apply_layerwise_casting

            apply_layerwise_casting(pipe.text_encoder, storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype,
                                    skip_modules_pattern=("norm", "embed", "shared"))
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        torch.cuda.empty_cache()
        self.loaded = True
        self.info["load_s"] = round(time.time() - t0, 1)
        self.info["vram_after_load_gb"] = self.vram_used_gb()
        log.info("model loaded in %.0fs, vram used %.1fGB", self.info["load_s"], self.info["vram_after_load_gb"])
        # re-attach LoRAs that were registered before an unload/reload cycle
        for name, meta in list(self.loras.items()):
            try:
                self._attach(name, meta["path"])
            except Exception:  # noqa: BLE001
                log.exception("re-attaching lora %s failed; dropping it from the registry", name)
                self.loras.pop(name, None)

    def unload(self) -> None:
        """Free the GPU completely (a training job needs all of it)."""
        if self.pipe is not None:
            try:
                self.pipe.to("cpu")
            except Exception:  # noqa: BLE001
                pass
            self.pipe = None
        self.loaded = False
        for m in self.loras.values():
            m["loaded"] = False
        gc.collect()
        if self._torch is not None:
            self._torch.cuda.empty_cache()
            self._torch.cuda.synchronize()
        log.info("pipeline unloaded, vram used %.2fGB", self.vram_used_gb())

    def vram_used_gb(self) -> float:
        if self._torch is None or not self._torch.cuda.is_available():
            return 0.0
        return round(self._torch.cuda.memory_allocated() / 1e9, 2)

    # -- LoRA registry -------------------------------------------------------
    def _attach(self, name: str, path: str) -> dict:
        info = lora_convert.normalize(path)
        self.pipe.load_lora_weights(info["path"], adapter_name=name)
        meta = self.loras.get(name, {})
        meta.update({"name": name, "path": path, "size_mb": round(os.path.getsize(path) / 1e6, 1), "loaded": True,
                     "format": info["format"], "converted": info["converted"], "dropped_keys": len(info["dropped"])})
        self.loras[name] = meta
        log.info("lora %s attached (%s, %d keys, %d dropped)", name, info["format"], info["n_keys"], len(info["dropped"]))
        return meta

    def add_lora(self, name: str, path: str) -> dict:
        name = _safe_name(name)
        os.makedirs(config.LORA_DIR, exist_ok=True)
        dst = os.path.join(config.LORA_DIR, f"{name}.safetensors")
        if os.path.abspath(path) != os.path.abspath(dst):
            shutil.copyfile(path, dst)
        if name in self.loras and self.loaded:
            try:
                self.pipe.delete_adapters(name)
            except Exception:  # noqa: BLE001
                pass
        if self.loaded:
            return self._attach(name, dst)
        self.loras[name] = {"name": name, "path": dst, "size_mb": round(os.path.getsize(dst) / 1e6, 1), "loaded": False}
        return self.loras[name]

    def remove_lora(self, name: str) -> bool:
        meta = self.loras.pop(name, None)
        if meta is None:
            return False
        if self.loaded:
            try:
                self.pipe.delete_adapters(name)
            except Exception:  # noqa: BLE001
                log.exception("delete_adapters(%s) failed", name)
        for p in (meta.get("path"), os.path.splitext(meta.get("path", ""))[0] + ".diffusers-ready.safetensors"):
            if p and os.path.exists(p):
                os.remove(p)
        return True

    def list_loras(self) -> list[dict]:
        return [{k: v for k, v in d.items() if k != "path"} for d in self.loras.values()]

    def _activate(self, loras: list[dict]) -> None:
        """loras: [{"name","scale"}]. Exactly these adapters active at these weights; all others off."""
        if not self.loras:
            return
        if not loras:
            self.pipe.disable_lora()
            return
        self.pipe.enable_lora()
        names = [l["name"] for l in loras]
        for n in names:
            if n not in self.loras or not self.loras[n].get("loaded"):
                raise ValueError(f"lora {n!r} is not loaded")
        self.pipe.set_adapters(names, [float(l["scale"]) for l in loras])

    # -- inference -----------------------------------------------------------
    def generate(self, prompt: str, negative: str, steps: int, guidance: float, seeds: list[int],
                 width: int, height: int, loras: list[dict], progress: ProgressCb) -> list[Image.Image]:
        torch = self._torch
        self._activate(loras)
        outs = []
        total = steps * len(seeds)
        done = [0]

        def _cb(pipe, i, t, kw):
            done[0] += 1
            progress(done[0] / total)
            return kw

        with torch.inference_mode():
            for seed in seeds:
                gen = torch.Generator(device="cuda").manual_seed(int(seed))
                out = self.pipe(
                    prompt=prompt, negative_prompt=negative or "", guidance_scale=float(guidance),
                    num_inference_steps=int(steps), width=width, height=height,
                    generator=gen, callback_on_step_end=_cb,
                )
                outs.append(out.images[0])
        return outs


def make_engine():
    return MockEngine() if config.MOCK else Engine()
