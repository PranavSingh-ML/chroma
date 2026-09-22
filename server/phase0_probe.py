"""Phase 0: run ONCE inside a pod started from chroma-lora:v0 with PHASE0=1 (see scripts/phase0.sh).

Runs in the exact image that ships, using the exact Engine and TrainRun that production uses, so
the pip freeze it writes is the truth and the LoRA-format question is answered by test, not by
reading docs. Produces, in the current directory:

  phase0_report.json        versions, capability, VRAM, timings, lora load path  -> copy into NOTES.md
  phase0_out.png            base-model generation (acceptance criterion)
  phase0_lora_out.png       generation with the throwaway LoRA at scale 1.0 (acceptance criterion)
  requirements.lock.txt     pip freeze minus the packages the base image owns -> server/requirements.lock.txt

Usage:  python phase0_probe.py [--steps 26] [--train-steps 20] [--skip-train]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_IMAGE_OWNED = ("torch==", "torchvision==", "torchaudio==", "triton==", "torchcodec==", "nvidia-", "pytorch-triton")
PROMPT = ("photo of a woman standing in a city street at golden hour, 35mm, shallow depth of field, "
          "natural skin texture, candid expression")


def _synthetic_dataset(dest: str, trigger: str, n: int = 4) -> int:
    """4 distinguishable images + captions. Enough to prove the trainer runs; not a real LoRA."""
    from PIL import Image, ImageDraw

    os.makedirs(dest, exist_ok=True)
    colours = [(180, 70, 60), (60, 120, 180), (90, 160, 90), (170, 140, 60)]
    for i in range(n):
        im = Image.new("RGB", (768, 1024), colours[i % len(colours)])
        d = ImageDraw.Draw(im)
        d.ellipse([234, 150, 534, 450], fill=(240, 210, 180))          # "face"
        d.rectangle([284, 450, 484, 900], fill=(40, 40, 60))            # "body"
        d.text((20, 20), f"phase0 synthetic subject {i}", fill=(255, 255, 255))
        im.save(os.path.join(dest, f"{i:03d}.jpg"), quality=92)
        with open(os.path.join(dest, f"{i:03d}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{trigger} person, full body, plain background, flat lighting, test image {i}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None, help="inference steps (default config.DEFAULT_STEPS)")
    ap.add_argument("--train-steps", type=int, default=20)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    import torch

    import config
    from pipeline import Engine, clamp_size

    report: dict = {"date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "python": platform.python_version(),
                    "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    cap = torch.cuda.get_device_capability()
    report["gpu"] = torch.cuda.get_device_name()
    report["capability"] = list(cap)
    report["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    report["model"] = f"{config.MODEL_REPO}/{config.MODEL_FILE}@{config.MODEL_REVISION}"
    report["extras"] = f"{config.EXTRAS_REPO}@{config.EXTRAS_REVISION}"
    for mod in ("diffusers", "transformers", "accelerate", "peft", "bitsandbytes", "huggingface_hub", "safetensors"):
        try:
            report[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            report[mod] = f"not importable: {e.__class__.__name__}"
    try:
        report["ai_toolkit_sha"] = subprocess.check_output(
            ["git", "-C", config.AI_TOOLKIT_DIR, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        sha_file = os.path.join(config.AI_TOOLKIT_DIR, "COMMIT_SHA")
        if os.path.exists(sha_file):
            with open(sha_file, encoding="utf-8") as fh:
                report["ai_toolkit_sha"] = fh.read().strip()
        else:
            report["ai_toolkit_sha"] = os.environ.get("AI_TOOLKIT_SHA", "unknown")
    try:
        report["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps(report, indent=2), flush=True)

    # ---- 1. load (includes the ~28 GB download on a cold pod) ----------------
    eng = Engine()
    t0 = time.time()
    eng.load()
    report["cold_load_s_incl_download"] = round(time.time() - t0, 1)
    report["download_s"] = eng.info.get("download_s")
    report["vram_idle_gb"] = eng.vram_used_gb()
    try:
        hf = os.environ.get("HF_HOME", "/workspace/hf")
        report["hf_cache_gb"] = round(sum(
            os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(hf) for f in fs
            if os.path.exists(os.path.join(dp, f))) / 1e9, 1)
    except OSError:
        pass
    torch.cuda.reset_peak_memory_stats()
    print(f"loaded in {report['cold_load_s_incl_download']}s, idle VRAM {report['vram_idle_gb']}GB", flush=True)

    # ---- 2. generate: #1 includes kernel warmup, #2 is steady state ----------
    steps = args.steps or config.DEFAULT_STEPS
    w, h = clamp_size(config.DEFAULT_WIDTH, config.DEFAULT_HEIGHT)
    t0 = time.time()
    imgs = eng.generate(args.prompt, "", steps, config.DEFAULT_GUIDANCE, [42], w, h, [],
                        lambda p: print(f"  progress {p:.2f}", flush=True))
    report["gen1_s_incl_warmup"] = round(time.time() - t0, 2)
    imgs[0].save("phase0_out.png")
    t0 = time.time()
    imgs = eng.generate(args.prompt, "", steps, config.DEFAULT_GUIDANCE, [43], w, h, [], lambda p: None)
    report["gen2_s_steady"] = round(time.time() - t0, 2)
    imgs[0].save("phase0_out_2.png")
    report["gen_steps"], report["gen_size"] = steps, f"{w}x{h}"
    report["vram_peak_generate_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(json.dumps({k: report[k] for k in ("gen1_s_incl_warmup", "gen2_s_steady", "vram_peak_generate_gb")}, indent=2), flush=True)

    if not args.skip_train:
        import trainer as trainer_mod

        # ---- 3. train a throwaway LoRA (the real subprocess, the real template) ----
        job_dir = os.path.join(config.TRAIN_ROOT, "phase0")
        shutil.rmtree(job_dir, ignore_errors=True)
        ds_dir = os.path.join(job_dir, "dataset")
        trigger = "ph4se0"
        n_images = _synthetic_dataset(ds_dir, trigger)
        cfg = trainer_mod.render_config(
            "phase0lora", ds_dir, os.path.join(job_dir, "output"), eng.model_path, n_images,
            {"steps": args.train_steps, "save_every": args.train_steps, "sample_every": args.train_steps,
             "trigger_word": trigger, "sample_prompts": [f"photo of {trigger} person, plain background"],
             "sample_steps": 8, "resolutions": [512]})
        report["train_config"] = cfg["config"]["process"][0]["train"] | {"rank": cfg["config"]["process"][0]["network"]["linear"]}
        print("== unloading pipeline before training ==", flush=True)
        eng.unload()
        report["vram_after_unload_gb"] = eng.vram_used_gb()
        run = trainer_mod.make_run(job_dir, "phase0lora", cfg)
        t0 = time.time()
        last = {}
        try:
            artifact = run.run(lambda st: (last.update(st), print(f"  train step {st.get('step')}/{st.get('total_steps')} "
                                                                 f"loss={st.get('loss')} {st.get('s_per_it')}s/it", flush=True))[0])
            report["train_ok"] = True
        except Exception as e:  # noqa: BLE001
            report["train_ok"] = False
            report["train_error"] = str(e)[:2000]
            artifact = None
        report["train_s_total"] = round(time.time() - t0, 1)
        report["train_s_per_step"] = last.get("s_per_it")
        report["train_last_loss"] = last.get("loss")
        report["vram_peak_train_gb"] = "see nvidia-smi during run (separate process)"
        print(json.dumps({k: report.get(k) for k in ("train_ok", "train_s_total", "train_s_per_step")}, indent=2), flush=True)

        # ---- 4. reload the pipeline (must be fast: weights are cached) --------
        t0 = time.time()
        eng.load()
        report["pipeline_reload_s"] = round(time.time() - t0, 1)
        print(f"pipeline reloaded in {report['pipeline_reload_s']}s", flush=True)

        # ---- 5. THE question: can the pipeline load that LoRA? ----------------
        if artifact:
            import lora_convert

            from safetensors.torch import load_file

            keys = list(load_file(artifact).keys())
            report["lora_artifact"] = os.path.basename(artifact)
            report["lora_size_mb"] = round(os.path.getsize(artifact) / 1e6, 1)
            report["lora_n_keys"] = len(keys)
            report["lora_key_samples"] = keys[:6]
            report["lora_detected_format"] = lora_convert.detect(keys)
            try:
                info = lora_convert.normalize(artifact)
                report["lora_normalize"] = {k: v for k, v in info.items() if k != "dropped"}
                report["lora_dropped_keys"] = info["dropped"][:10]
                report["lora_dropped_n"] = len(info["dropped"])
                eng.pipe.load_lora_weights(info["path"], adapter_name="phase0lora")
                eng.pipe.set_adapters(["phase0lora"], [1.0])
                report["lora_load_path"] = "load_lora_weights" + ("+normalize" if info["converted"] else "")
                report["lora_load_ok"] = True
            except Exception as e:  # noqa: BLE001
                report["lora_load_ok"] = False
                report["lora_load_error"] = f"{e.__class__.__name__}: {e}"[:2000]
                print("load_lora_weights FAILED, trying the manual merge fallback:", e, flush=True)
                try:
                    m = lora_convert.merge_manually(eng.pipe.transformer, artifact, 1.0)
                    report["lora_load_path"] = "merge_manually"
                    report["lora_merge"] = {k: v for k, v in m.items() if k != "dropped"}
                    report["lora_load_ok"] = True
                except Exception as e2:  # noqa: BLE001
                    report["lora_load_path"] = None
                    report["lora_merge_error"] = f"{e2.__class__.__name__}: {e2}"[:2000]
            if report.get("lora_load_ok"):
                t0 = time.time()
                out = eng.generate(f"photo of {trigger} person, plain background", "", steps,
                                   config.DEFAULT_GUIDANCE, [42], 768, 768,
                                   [{"name": "phase0lora", "scale": 1.0}] if report["lora_load_path"] != "merge_manually" else [],
                                   lambda p: None)
                out[0].save("phase0_lora_out.png")
                report["gen_with_lora_s"] = round(time.time() - t0, 2)
        shutil.rmtree(job_dir, ignore_errors=True)

    report["vram_peak_overall_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(json.dumps(report, indent=2), flush=True)
    with open("phase0_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ---- 6. lock file -------------------------------------------------------
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    kept = [ln for ln in freeze if ln and not ln.startswith(BASE_IMAGE_OWNED) and " @ file://" not in ln
            and not ln.startswith("-e ")]
    header = [
        f"# Frozen by phase0_probe.py on {report['date']}",
        f"# inside chroma-lora image (python {report['python']}, torch {report['torch']} cuda {report['cuda']}),"
        f" GPU {report['gpu']} cap {cap}",
        f"# ai-toolkit {report['ai_toolkit_sha']}",
        "# torch/torchvision/triton/nvidia-* come from the Dockerfile torch layer and are intentionally omitted.",
    ]
    with open("requirements.lock.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(header + kept) + "\n")
    print("\nWROTE phase0_report.json, phase0_out.png, phase0_lora_out.png, requirements.lock.txt")
    print("Copy requirements.lock.txt -> server/requirements.lock.txt, phase0_report.json -> NOTES.md, "
          "then TERMINATE the pod.")
    return 0 if report.get("lora_load_ok", True) and report.get("train_ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
