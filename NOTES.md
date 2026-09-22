# NOTES — verified values, measurements, spend

Legend: **VERIFIED** = checked against the live source on the date given, from the laptop.
**MEASURED** = recorded from a real pod run. **TODO(pod)** = cannot be known until Phase 0/1
runs on a GPU; fill in from `phase0/phase0_report.json` / smoke-test output.

## Environment pins (VERIFIED 2026-09-21/22)

| Item | Value | Source |
|---|---|---|
| Base image | `nvidia/cuda:12.9.0-base-ubuntu24.04@sha256:48e21b10467354655f5073c05eebdeaac9818c6b40d70f334f7ad2df000463d8` (109 MB) | inherited from `imgedit` (Docker Hub API, 2026-09-17) |
| Python / torch | Ubuntu 24.04 python3 = 3.12; `torch==2.9.1+cu129`, `torchvision==0.24.1+cu129` from download.pytorch.org/whl/cu129 | PyTorch index |
| Build | GitHub Actions → `ghcr.io/pranavsingh-ml/chroma-lora:<tag>` (no local Docker: VT-x disabled in firmware) | `.github/workflows/build-image.yml` |
| Repo | `github.com/PranavSingh-ML/chroma` (image name stays `chroma-lora`) | — |
| diffusers | `0.40.0` — has `ChromaPipeline`, `ChromaTransformer2DModel`, single-file loader (`convert_chroma_transformer_checkpoint_to_diffusers`) | `single_file_model.py` @ v0.40.0 |
| Chroma LoRA loader | `ChromaPipeline` subclasses **`FluxLoraLoaderMixin`** (there is no `ChromaLoraLoaderMixin`) | `pipelines/chroma/pipeline_chroma.py:150-153` @ v0.40.0 |
| transformers / accelerate / peft | 5.17.0 / 1.15.0 / ≥0.17 (same as `imgedit`) | PyPI |
| bitsandbytes | 0.50.2 — **required** here (adamw8bit), unlike `imgedit` where it is optional | PyPI |
| ai-toolkit | `a8dfcf7d7e2b38ccc7b2fb68ece9c6358e61e7a7` (2026-09-20), no releases exist → SHA pin | GitHub API |
| ai-toolkit Chroma example | `config/examples/train_lora_chroma_24gb.yaml` — the basis of `server/train_lora_chroma.template.yaml` | repo @ that SHA |
| Model | `lodestones/Chroma1-HD` @ `0e0c60ece1e82b17cb7f77342d765ba5024c40c0`, Apache 2.0, **not gated** | HF API |
| Model size | repo total 45.3 GB, but we download **only what is needed**: `Chroma1-HD.safetensors` **17.80 GB** (the diffusers `transformer/` folder is the same weights again — skipped) | HF API |
| T5 / tokenizer / VAE | `ostris/Flex.1-alpha` @ `5dd0b1bc2a9421b891abcf7c9218993f696dd550`: `text_encoder_2` 4.99+4.53 GB, `vae` 0.17 GB | HF API |
| **Total download** | **≈ 27.5 GB** (17.8 + 9.5 + 0.2), not 45.3 | computed |
| Why those repos | ai-toolkit's chroma loader fetches exactly these (`hf_hub_download(repo_id="lodestones/Chroma1-HD", filename="Chroma1-HD.safetensors")`, extras from `ostris/Flex.1-alpha`), so the server and the trainer at least agree on *which* weights they use. **The hoped-for "one shared download" did not materialise** — measured cache was 55 GB, see deviation 7. | `extensions_built_in/diffusion_models/chroma/chroma_model.py` @ a8dfcf7 |

### The LoRA key-format problem (designed for, VERIFY in Phase 0)

- `FluxLoraLoaderMixin.lora_state_dict` accepts three shapes: peft/diffusers (`transformer.…lora_A`),
  kohya (`lora_unet_…lora_down` + `.alpha`), and ComfyUI (`diffusion_model.…lora_down`, rewritten to
  the kohya path). ai-toolkit saves **ComfyUI prefixes with peft names** (`diffusion_model.….lora_A`)
  — a fourth shape that matches none of them.
- `server/lora_convert.py` renames `lora_A/B` → `lora_down/up` for that case. With no `.alpha`
  tensor the kohya converter uses scale 1, which is correct **only if alpha == rank** — hence the
  template forces `linear_alpha == linear`.
- The kohya converter **raises** (`ValueError: Incompatible keys detected`) on any key it cannot
  map, so Chroma-only modules (`distilled_guidance_layer`, …) must be dropped, not just warned
  about. `_FLUX_MODULE_RE` in `lora_convert.py` is that whitelist. (`lora_conversion_utils.py:660-664`)
- Fallback if it still refuses: `lora_convert.merge_manually()` folds `W += scale·(B@A)` directly
  into the transformer. `phase0_probe.py` tries `load_lora_weights` first and falls back, recording
  `lora_load_path` in the report. **Whichever wins is the only path kept in `pipeline.py`.**

## Hardware / pricing (VERIFIED 2026-09-17 via imgedit — these move)

| GPU | Secure | Community | Notes |
|---|---|---|---|
| A40 48GB (Ampere, sm_86) | $0.49/hr | $0.35/hr | default; training and serving both fit |
| RTX A6000 48GB (sm_86) | $0.53/hr | $0.33/hr | automatic fallback in `pod.sh` |
| L40S 48GB (Ada, sm_89) | $1.09/hr | $0.79/hr | only if training feels too slow |

Container disk 80 GB ≈ $0.011/hr while running. **No network volume** (user decision) — the
~27.5 GB re-download costs ~$0.03–0.06 per cold boot vs ~$5–7/month for a volume.

## Measurements (MEASURED - Phase 0, 2026-09-22, pod md93x63f6kqmgs)

A40 was out of stock; ran on **RTX A6000** (same Ampere sm_86 / 48 GB class, $0.53/hr Secure).

| Metric | Value |
|---|---|
| GPU / capability | NVIDIA RTX A6000, 47.7 GB, capability **(8, 6)** |
| torch / CUDA / python | 2.9.1+cu129 / 12.9 / 3.12.3 |
| diffusers / transformers / peft | 0.40.0 / 5.17.0 / 0.18.1 |
| image pull (4.7 GB from GHCR) | ~6 min (slower datacenter than imgedit's) |
| cold load incl. ~27.5 GB download | **63.9 s** (~430 MB/s - this DC is fast) |
| warm load (cache hot) | 10.2 s |
| HF cache on disk after everything | **55.0 GB** - see deviation 6: ai-toolkit downloads its own copy |
| VRAM idle after load | **27.5 GB** (spec predicted ~28 - correct) |
| VRAM peak, 1024x1024 generate | **30.07 GB** |
| VRAM after `engine.unload()` | **0.01 GB** - the pre-training unload works |
| generate, 26 steps, 1024x1024, CFG 4 | **61.7 s** (~2.4 s/step) - see deviation 5 |
| generate with LoRA, 26 steps, 768x768 | 39.2 s |
| **training, rank 16, 512 bucket, batch 1** | **2.05 s/step** (20-step test, loss 0.027 -> 0.018) |
| pipeline reload after training | **11.5 s** (cache warm) - retraining in one session is cheap |
| **LoRA format produced by ai-toolkit** | **`comfy_peft`** (`diffusion_model.*.lora_A/B`, no alpha) |
| **LoRA load path that works** | **`load_lora_weights` + `lora_convert.normalize`** |
| LoRA keys converted / dropped | **456 / 0** - the whitelist drops nothing on a real file |
| LoRA size (rank 16) | 112 MB |
| Phase 0 wall time (2 runs incl. the torchaudio fix) | ~50 min |

**The 2.05 s/step above is at `resolution: [512]` only** (the probe used one bucket to stay cheap).
A real run with the recommended `[512, 768, 1024]` buckets measured **~7.3 s/step** on the same
card (4.1 s/it on the 512 bucket, 8.8 s/it on 1024 - a 1024 image is 4x the pixels). So:

| Run | s/step | 1600 steps | cost @ $0.53/hr |
|---|---|---|---|
| probe, 512 only | 2.05 | 55 min | $0.48 |
| **real, [512,768,1024]** | **~7.3** | **~3.3 h** | **~$1.75** |

Never quote the single-bucket number as a run estimate again. Community cloud (same A6000 at
$0.33/hr) takes that to ~$1.10, which is why `pod.sh` now tries Community first.

### Bugs Phase 0 caught (all would have hit mid-training)

1. **`No module named 'torchaudio'`** - the Dockerfile filtered the whole torch family out of
   ai-toolkit's requirements (we pin torch ourselves), but ai-toolkit imports torchaudio at
   runtime. Training died after 9 s. Fixed: `torchaudio==2.9.1` in the torch layer.
2. **Progress parser matched any tqdm bar** - `(\d+)/(\d+)\s*\[` locked onto
   `Loading weights: 219/219`, so the reported step jumped to 219/219 before training started.
   Fixed: anchored to the run-name prefix only the training loop uses.
3. **`pod_ssh.py` emitted CRLF** - python's Windows text mode put a carriage return in the port, so every
   ssh/scp call failed with `Bad port '22079'`. Fixed at the source.

## Spend log

| Date | What | GPU | Hours | $ |
|---|---|---|---|---|
| 2026-09-22 | Phase 0 (probe x2 incl. the torchaudio fix), then captioning + a 143-step partial train | RTX A6000 Secure @ $0.53 | ~1.8 | |
| 2026-09-22 | A40 created by mistake (wrong speed class), killed in seconds | A40 Secure | ~0 | |
| 2026-09-22 | L40S booted for the retry, deleted before training began | L40S Community @ $0.79 | ~0.2 | |
| | | | | **day total $1.35 — balance $9.51 -> $8.16** |

### Where the 2026-09-22 session stopped

Phase 0 PASSED and everything is verified end to end EXCEPT a completed training run.
State to resume from:

- `data/datasets/pranav/` — 16 images + 16 captions, prepped and guard-checked. Ready.
- `v1` image (`ghcr.io/pranavsingh-ml/chroma-lora:v1`) has the torchaudio fix and boots clean.
- The A6000 run reached step 143/1600 and was cancelled to move to a faster card; no
  checkpoint existed yet (first save is at step 250), so nothing was kept.
- Next action: `bash scripts/pod.sh up` (now prefers Ada), wait, then
  `STEPS=1200 bash scripts/train.sh data/datasets/pranav pranav-v1 pr4nv`.
  Expect ~3.6-4 s/step on Ada => ~75 min, ~$1.00. Captions already exist, so
  `caption.sh` will skip them and cost nothing.
- 48 GB stock was thin all session: RTX 6000 Ada unavailable in both clouds, Ampere
  Community gone. L40S Community was the one that landed.

## Laptop-side verification (done 2026-09-22, no GPU)

Against `MOCK=1` (fake engine + fake trainer; queue, polling, guard, zip handling, progress
parsing, artifact download and saving are all the real code):

- `scripts/smoke_test.sh` → PASS: health, 401 without token, **422 on a prompt naming a minor**,
  async generate → poll → PNG (1024×1024).
- `scripts/train_smoke.sh` → PASS: 4-image zip → 20-step train → sample preview pulled →
  `data/loras/<name>.safetensors` + `.json` sidecar → `/loras` lists it → generate with it.
- `scripts/generate.sh` with `SWEEP=1` → 5 jobs at LoRA off / 0.6 / 0.8 / 1.0 / 1.2, same seed.
- Concurrency/robustness: a generate submitted during training stays **queued** (not dropped);
  a second `POST /train` returns **409**; `DELETE /jobs/{id}` on a running train SIGTERMs it, the
  job ends `error: cancelled`, **the pipeline is reloaded anyway**, the queued generate then
  completes and `/health` returns `busy: idle, model_loaded: true`.

Bug found and fixed by these tests: `guard.py`'s age regex matched `"age 0"` inside `"im**age** 0"`,
rejecting every ordinary caption ending in a number. Added a letter boundary; added `teenage` /
`adolescent` to the denylist (they were missing while `teenager` was present).

## Deviations from spec.md (told the user)

1. **Model is loaded from the single-file checkpoint**, not the diffusers folder layout: it is the
   same file ai-toolkit downloads, so inference and training share one 17.8 GB blob instead of
   pulling 35.6 GB (both copies). T5/VAE come from `ostris/Flex.1-alpha` for the same reason.
   Download is ~27.5 GB, not the ~28 GB estimated — close enough, but the *reason* differs.
2. `configs/` was folded into `server/` — the Docker build context is `server/`, so the training
   template has to live there.
3. Added `scripts/train.sh` + `scripts/generate.sh` (not in the spec): they give the complete
   train → LoRA → generate → strength-sweep workflow from the terminal, so the first real LoRA
   does **not** have to wait for the Phase-3 web app.
4. Added `scripts/_py.sh`: Git Bash on Windows has no `python3` (the name resolves to the
   Microsoft Store stub), so the scripts probe for a working interpreter.
5. `spec.md` §4 said `POST /loras` takes the file as `file=` with `name=`; kept, but the server
   also refuses uploads while `busy` is `train`/`reload` (409) since the pipeline is gone then.
6. **Generation is 61.7 s/image, not the 25-35 s the spec estimated.** Chroma has no guidance
   embedding, so `guidance 4.0` runs a real negative pass: 26 steps = **52** forward passes, not
   26. The estimate assumed one pass per step. Levers if it matters: 20 steps (~47 s), or an
   L40S/6000-Ada pod. Training is unaffected (no CFG).
7. **Inference and training do NOT share one download.** The HF cache reached 55 GB, not 27.5:
   ai-toolkit's chroma loader fetches its own copy of the weights rather than reusing ours.
   The 80 GB container disk absorbs it, but spec.md section 1's "ONE download shared by
   inference and training" is wrong, and the first train job on a cold pod pays an extra
   ~27.5 GB download. Worth revisiting if cold-boot cost ever matters.
8. Phase 0 ran twice: the first run died on torchaudio, the fix was applied live on the pod
   (`pip install torchaudio`) and re-run, then baked into the Dockerfile. The frozen lock comes
   from the successful run.
