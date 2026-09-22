"""Training job: dataset unpack + validation, YAML render, ai-toolkit subprocess, progress parsing.

The trainer is ALWAYS a subprocess (`python run.py <cfg>` inside AI_TOOLKIT_DIR): its crash cannot
take the server down and its GPU memory is fully released when it exits. The caller (app.py's
worker) unloads the inference pipeline before `run()` and reloads it after, whatever happened.
"""
from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from typing import Callable

import config
import guard

log = logging.getLogger("trainer")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
# tqdm line from ai-toolkit's TRAINING loop, e.g.
#   "mara-v1:  12%|█▏ | 240/2000 [06:01<44:10,  1.51s/it, lr: 1.0e-04 loss: 3.412e-01]"
# The description prefix is the job name. Matching a bare "N/M [" instead would also match the
# weight-loading and latent-caching bars ("Loading weights: 219/219 [...]"), which would make the
# reported step jump to 219/219 before training even starts (seen in Phase 0, 2026-09-22).
def _step_re(name: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(name)}\s*:.*?(\d+)/(\d+)\s*\[")
_LOSS_RE = re.compile(r"loss:\s*([0-9.]+(?:e[+-]?\d+)?)")
_SIT_RE = re.compile(r"([0-9.]+)s/it")


class TrainError(Exception):
    pass


# ----------------------------------------------------------------------------
# dataset + config
# ----------------------------------------------------------------------------
def unpack_dataset(zip_path: str, dest: str) -> dict:
    """Extract, flatten a single top-level folder, validate. Returns {"images": n, "captions": n, "missing": [...]}."""
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        total = sum(i.file_size for i in z.infolist())
        if total > config.TRAIN_MAX_ZIP_MB * 1024 * 1024 * 3:
            raise TrainError("dataset zip expands too large")
        for i in z.infolist():
            if i.is_dir() or "__MACOSX" in i.filename or os.path.basename(i.filename).startswith("."):
                continue
            name = os.path.basename(i.filename)  # flatten: captions must sit next to images
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTS + (".txt",):
                continue
            with z.open(i) as src, open(os.path.join(dest, name), "wb") as dst:
                shutil.copyfileobj(src, dst)
    images = sorted(f for f in os.listdir(dest) if f.lower().endswith(IMAGE_EXTS))
    if not images:
        raise TrainError("dataset has no .jpg/.jpeg/.png images (ai-toolkit supports only those)")
    missing = [f for f in images if not os.path.exists(os.path.join(dest, os.path.splitext(f)[0] + ".txt"))]
    return {"images": len(images), "captions": len(images) - len(missing), "missing": missing}


def check_captions(dataset_dir: str) -> dict[str, str]:
    texts = {}
    for f in os.listdir(dataset_dir):
        if f.lower().endswith(".txt"):
            with open(os.path.join(dataset_dir, f), encoding="utf-8", errors="replace") as fh:
                texts[f] = fh.read()
    return guard.check_many(texts)


def resolve_steps(requested, n_images: int) -> int:
    if requested:
        return max(20, min(int(requested), config.TRAIN_MAX_STEPS))
    return max(1000, min(3000, 100 * n_images))


def render_config(name: str, dataset_dir: str, output_dir: str, model_path: str, n_images: int,
                  overrides: dict) -> dict:
    """Merge TRAIN_DEFAULTS <- overrides into the YAML template. Returns the config dict."""
    import yaml

    o = {**config.TRAIN_DEFAULTS, **{k: v for k, v in (overrides or {}).items() if k in config.TRAIN_DEFAULTS}}
    with open(config.TRAIN_TEMPLATE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    c = cfg["config"]
    c["name"] = name
    p = c["process"][0]
    p["training_folder"] = output_dir
    p["trigger_word"] = str(o["trigger_word"] or "")
    rank = int(o["rank"])
    p["network"]["linear"] = rank
    p["network"]["linear_alpha"] = rank  # == rank on purpose; see lora_convert.py
    p["save"]["save_every"] = int(o["save_every"])
    ds = p["datasets"][0]
    ds["folder_path"] = dataset_dir
    ds["caption_dropout_rate"] = float(o["caption_dropout"])
    ds["resolution"] = [int(r) for r in o["resolutions"]]
    t = p["train"]
    t["batch_size"] = int(o["batch_size"])
    t["steps"] = resolve_steps(o["steps"], n_images)
    t["optimizer"] = str(o["optimizer"])
    t["lr"] = float(o["lr"])
    t["ema_config"]["use_ema"] = bool(o["ema"])
    p["model"]["name_or_path"] = model_path
    p["model"]["quantize"] = bool(o["quantize_base"])
    s = p["sample"]
    s["sample_every"] = int(o["sample_every"])
    s["sample_start_step"] = int(o["sample_every"])
    s["width"], s["height"] = int(o["sample_width"]), int(o["sample_height"])
    s["guidance_scale"] = float(o["sample_guidance"])
    s["sample_steps"] = int(o["sample_steps"])
    prompts = [str(x) for x in (o["sample_prompts"] or []) if str(x).strip()][:4]
    if not prompts:
        trig = "[trigger]" if o["trigger_word"] else ""
        prompts = [f"photo of {trig} person, medium shot, natural light, city street".strip(),
                   f"close-up portrait of {trig} person, studio lighting, neutral background".strip()]
    s["prompts"] = prompts
    if p["save"]["save_every"] > t["steps"]:
        p["save"]["save_every"] = t["steps"]
    return cfg


# ----------------------------------------------------------------------------
# the subprocess
# ----------------------------------------------------------------------------
class TrainRun:
    """One ai-toolkit run. `run()` blocks; `cancel()` may be called from another thread."""

    def __init__(self, job_dir: str, name: str, cfg: dict) -> None:
        self.job_dir = job_dir
        self.name = name
        self.cfg = cfg
        self.cfg_path = os.path.join(job_dir, "config.yaml")
        self.output_dir = cfg["config"]["process"][0]["training_folder"]
        self.run_dir = os.path.join(self.output_dir, name)
        self.log_path = os.path.join(job_dir, "train.log")
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self.total_steps = int(cfg["config"]["process"][0]["train"]["steps"])
        self.step_re = _step_re(name)

    def write_config(self) -> None:
        import yaml

        with open(self.cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)

    def samples(self) -> list[str]:
        d = os.path.join(self.run_dir, "samples")
        if not os.path.isdir(d):
            return []
        fs = [f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png"))]
        fs.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)))
        return fs

    def sample_path(self, name: str) -> str | None:
        p = os.path.join(self.run_dir, "samples", os.path.basename(name))
        return p if os.path.isfile(p) else None

    def artifact(self) -> str | None:
        final = os.path.join(self.run_dir, f"{self.name}.safetensors")
        if os.path.isfile(final):
            return final
        cands = sorted(glob.glob(os.path.join(self.run_dir, f"{self.name}_*.safetensors")))
        return cands[-1] if cands else None

    def log_tail(self, n: int = 80) -> str:
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return ""

    def run(self, on_progress: Callable[[dict], None]) -> str:
        """Returns the artifact path. Raises TrainError on failure/cancel."""
        self.write_config()
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        cmd = [sys.executable, "run.py", self.cfg_path]
        log.info("trainer: %s (cwd %s)", " ".join(cmd), config.AI_TOOLKIT_DIR)
        t0 = time.time()
        with open(self.log_path, "w", encoding="utf-8") as logf:
            self.proc = subprocess.Popen(cmd, cwd=config.AI_TOOLKIT_DIR, env=env, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         errors="replace", bufsize=1)
            state = {"step": 0, "total_steps": self.total_steps, "loss": None, "eta_s": None, "s_per_it": None}
            last_emit = 0.0
            buf = ""
            while True:
                ch = self.proc.stdout.read(1)
                if ch == "":
                    break
                buf += ch
                if ch in ("\n", "\r"):
                    line = buf.strip()
                    buf = ""
                    if not line:
                        continue
                    logf.write(line + "\n")
                    m = self.step_re.search(line)
                    if m:
                        state["step"], state["total_steps"] = int(m.group(1)), int(m.group(2))
                        ml = _LOSS_RE.search(line)
                        if ml:
                            state["loss"] = float(ml.group(1))
                        ms = _SIT_RE.search(line)
                        if ms:
                            state["s_per_it"] = float(ms.group(1))
                            state["eta_s"] = round((state["total_steps"] - state["step"]) * state["s_per_it"], 1)
                    else:
                        logf.flush()
                    if time.time() - last_emit > 1.0:
                        state["samples"] = self.samples()
                        on_progress(dict(state))
                        last_emit = time.time()
            rc = self.proc.wait()
        state["samples"] = self.samples()
        on_progress(dict(state))
        log.info("trainer exited rc=%s after %.0fs", rc, time.time() - t0)
        if self.cancelled:
            raise TrainError("cancelled")
        if rc != 0:
            raise TrainError(f"ai-toolkit exited with code {rc}\n--- last log lines ---\n{self.log_tail(50)}")
        art = self.artifact()
        if not art:
            raise TrainError(f"trainer finished but no .safetensors in {self.run_dir}\n{self.log_tail(50)}")
        return art

    def cancel(self) -> None:
        self.cancelled = True
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGTERM)
                threading.Timer(20.0, lambda: self.proc.poll() is None and self.proc.kill()).start()
            except Exception:  # noqa: BLE001
                log.exception("cancel failed")

    def cleanup(self) -> None:
        """Delete the dataset + intermediate checkpoints (not the final artifact until the job dies)."""
        shutil.rmtree(os.path.join(self.job_dir, "dataset"), ignore_errors=True)
        for p in glob.glob(os.path.join(self.run_dir, f"{self.name}_0*.safetensors")):
            try:
                os.remove(p)
            except OSError:
                pass


class MockTrainRun(TrainRun):
    """Laptop-only: no ai-toolkit. Sleeps ~0.05 s/step, writes fake samples + a fake artifact."""

    def run(self, on_progress: Callable[[dict], None]) -> str:
        from PIL import Image, ImageDraw

        self.write_config()
        os.makedirs(os.path.join(self.run_dir, "samples"), exist_ok=True)
        every = int(self.cfg["config"]["process"][0]["sample"]["sample_every"])
        with open(self.log_path, "w", encoding="utf-8") as logf:
            for step in range(1, self.total_steps + 1):
                if self.cancelled:
                    raise TrainError("cancelled")
                time.sleep(0.02)
                loss = 0.5 * (0.6 ** (step / self.total_steps)) + 0.01 * ((step * 7919) % 13) / 13
                if step % 10 == 0 or step == self.total_steps:
                    logf.write(f"{self.name}: {step}/{self.total_steps} [00:00<00:00, 0.02s/it, loss: {loss:.3e}]\n")
                if step % every == 0:
                    for i in range(min(2, len(self.cfg["config"]["process"][0]["sample"]["prompts"]))):
                        im = Image.new("RGB", (256, 256), (40 + step % 200, 90, 160))
                        ImageDraw.Draw(im).text((8, 8), f"MOCK sample step {step} #{i}", fill=(255, 255, 255))
                        im.save(os.path.join(self.run_dir, "samples", f"mock__{self.name}_{step:09d}_{i}.jpg"))
                if step % 10 == 0:
                    on_progress({"step": step, "total_steps": self.total_steps, "loss": round(loss, 4),
                                 "eta_s": round((self.total_steps - step) * 0.02, 1), "s_per_it": 0.02,
                                 "samples": self.samples()})
        art = os.path.join(self.run_dir, f"{self.name}.safetensors")
        with open(art, "wb") as f:
            f.write(os.urandom(64 * 1024))
        return art


def make_run(job_dir: str, name: str, cfg: dict) -> TrainRun:
    return MockTrainRun(job_dir, name, cfg) if config.MOCK else TrainRun(job_dir, name, cfg)
