"""Auto-caption a dataset with a vision-language model, on the pod.

The instruction below is the whole point: captions must describe pose, framing, clothing,
background and lighting, and must NOT describe the face. A caption that says "green eyes, sharp
jawline" teaches the LoRA that those words produce those features; the identity then fires on the
words instead of on the trigger token, and the LoRA transfers badly. Everything the LoRA should
absorb is exactly what the caption leaves out.

The model is loaded on demand and unloaded right after, so it never competes with training.
"""
from __future__ import annotations

import gc
import logging
import os
import time

import config

log = logging.getLogger("captioner")

INSTRUCTION = (
    "Describe this photograph for an image-generation training caption.\n"
    "Include, in this order and only if visible: the framing (extreme close-up, close-up, "
    "head and shoulders, waist up, three-quarter, or full body), the subject's pose and what they "
    "are doing, their clothing, the background or location, and the lighting.\n"
    "STRICT RULES:\n"
    "- Do NOT describe the person's face, facial features, eyes, nose, mouth, jaw, hair colour, "
    "hair style, skin tone, age, ethnicity, gender, or how attractive they are.\n"
    "- Do NOT name or guess who the person is.\n"
    "- Refer to them only as 'the person'.\n"
    "- One sentence, comma-separated phrases, lower case, no preamble, no full stop at the end.\n"
    "Example of the required style: "
    "waist up, standing with arms crossed, grey hooded sweatshirt, brick wall outdoors, "
    "flat overcast daylight"
)

# Words that must never survive into a caption even if the VLM ignores the instruction.
BANNED = (
    "eyes", "eye ", "nose", "mouth", "lips", "jaw", "chin", "cheek", "eyebrow", "beard",
    "moustache", "mustache", "freckle", "complexion", "skin tone", "blonde", "blond",
    "brunette", "redhead", "ginger", "grey-haired", "gray-haired", "balding", "handsome",
    "beautiful", "pretty", "attractive", "young man", "young woman", "middle-aged", "smiling face",
)


def _strip_identity(text: str) -> str:
    """Drop any comma-separated clause that describes the face. Cheap, deterministic backstop."""
    parts = [p.strip() for p in text.replace("\n", " ").split(",")]
    keep = [p for p in parts if p and not any(b in p.lower() for b in BANNED)]
    return ", ".join(keep)


class Captioner:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self._torch = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        t0 = time.time()
        log.info("loading caption model %s", config.CAPTION_MODEL)
        self.processor = AutoProcessor.from_pretrained(config.CAPTION_MODEL)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.CAPTION_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        log.info("caption model ready in %.0fs", time.time() - t0)

    def unload(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        if self._torch is not None:
            self._torch.cuda.empty_cache()

    def caption_file(self, path: str, trigger: str = "") -> str:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        img.thumbnail((896, 896))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[img], return_tensors="pt").to(self.model.device)
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=120, do_sample=False)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        text = _strip_identity(text.strip().strip('."').lower())
        return f"{trigger} person, {text}".strip(", ") if trigger else text

    def caption_dir(self, d: str, trigger: str = "", overwrite: bool = False) -> dict:
        """Write <stem>.txt next to every image. Returns {filename: caption}."""
        exts = (".jpg", ".jpeg", ".png")
        out = {}
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(exts))
        for i, f in enumerate(files):
            stem = os.path.splitext(f)[0]
            cap_path = os.path.join(d, stem + ".txt")
            if os.path.exists(cap_path) and not overwrite:
                with open(cap_path, encoding="utf-8") as fh:
                    out[f] = fh.read().strip()
                continue
            try:
                cap = self.caption_file(os.path.join(d, f), trigger)
            except Exception as e:  # noqa: BLE001
                log.exception("caption failed for %s", f)
                cap = f"{trigger} person".strip() + f"  # CAPTION FAILED: {e.__class__.__name__}"
            with open(cap_path, "w", encoding="utf-8") as fh:
                fh.write(cap + "\n")
            out[f] = cap
            log.info("caption %d/%d %s: %s", i + 1, len(files), f, cap[:90])
        return out


class MockCaptioner(Captioner):
    def load(self) -> None:
        time.sleep(0.2)

    def unload(self) -> None:
        pass

    def caption_file(self, path: str, trigger: str = "") -> str:
        base = "waist up, standing, plain clothing, neutral indoor background, soft daylight"
        return f"{trigger} person, {base}".strip(", ") if trigger else base


def make_captioner() -> Captioner:
    return MockCaptioner() if config.MOCK else Captioner()
