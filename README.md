# chroma-lora — character LoRA studio for Chroma1-HD on a RunPod GPU

Train a LoRA of **your own** likeness on `lodestones/Chroma1-HD`, then generate with it at any
strength, from your laptop. The laptop keeps **all** state (`data/`); the pod is a disposable,
stateless GPU server built from a pinned Docker image. **No network volume** — datasets go up per
job, the trained LoRA comes straight back down.

- Model: `lodestones/Chroma1-HD` (8.9B, FLUX architecture, T5-XXL only, **Apache 2.0**, not gated,
  not safety-filtered — realistic adult anatomy without prompt gymnastics)
- Trainer: `ostris/ai-toolkit` pinned at `a8dfcf7`, run as a subprocess of the server
- Pod: FastAPI + `diffusers` (no ComfyUI), one worker, two job kinds (`generate` | `train`)
- Sibling project: `../imgedit` — same conventions, same `runpodctl` setup, same auto-stop discipline

```
server/    Dockerfile, app.py, pipeline.py, trainer.py, guard.py, lora_convert.py,
           config.py, phase0_probe.py, train_lora_chroma.template.yaml   (runs on the pod)
scripts/   pod.sh (up|wait|status|down), phase0.sh, smoke_test.sh, train_smoke.sh,
           train.sh (dataset -> LoRA), generate.sh (prompt -> PNG, with LoRA + sweep)
data/      loras/ images/ runs/ datasets/                               (gitignored, yours)
pod.cmd    Windows wrapper: runs scripts/pod.sh under Git Bash (not WSL)
NOTES.md   verified versions, measured latencies, actual spend
spec.md    the design and why each decision was made
```

## Rules, stated once

Train only on **your own likeness**, or on adults who have **explicitly consented** to adult AI
training use. Never on a third party's photos. The server has a deterministic guard that rejects
prompts and captions indicating a minor (`server/guard.py`) — that is a backstop, not a substitute
for the above. `data/` is gitignored and EXIF is stripped from ingested photos.

## Prerequisites

- Laptop: Git, Python 3 + Pillow, `curl`, `ssh`/`scp` (Git for Windows has them), Node ≥ 22
  (only for `pod.sh`). **No Docker** — images build on GitHub Actions.
- RunPod account with ≥ 1 hour of credit, `runpodctl` configured (`tools\runpodctl.exe doctor`),
  and an SSH key registered. **Set a low-balance alert in the RunPod console on day one.**
- This repo pushed to GitHub (it is `PranavSingh-ML/chroma`); the GHCR *package* must be public,
  or RunPod needs registry auth.

## Image builds (GitHub Actions → GHCR)

```
git tag v0 && git push origin main v0      # builds ghcr.io/pranavsingh-ml/chroma-lora:v0
```

`.github/workflows/build-image.yml` builds `server/` and pushes `ghcr.io/<owner>/chroma-lora:<tag>`;
the run summary shows the digest — copy it into `NOTES.md`. Never `latest`.

After the **first** push: GitHub → Packages → `chroma-lora` → Package settings → **Change
visibility → Public**, so RunPod can pull without credentials. (The image contains only code.)

## Build order (do not skip ahead — each phase has an acceptance test)

### Phase 0 — pin the environment (~40 min, ~$0.35, once)

Runs *inside* the `v0` image on a real GPU, so the frozen lock is exactly what ships — and it
answers the one open question (can `ChromaPipeline` load an ai-toolkit LoRA?) by test.

```
git tag v0 && git push origin main v0                            # Actions, ~15 min, free
PHASE0=1 IMAGE=ghcr.io/pranavsingh-ml/chroma-lora:v0 bash scripts/pod.sh up
bash scripts/phase0.sh <POD_ID>                                  # env, load, 2 gens, 20-step train,
                                                                 # LoRA load test, gen with LoRA, freeze
cp phase0/requirements.lock.txt server/requirements.lock.txt
./tools/runpodctl.exe pod delete <POD_ID>                        # TERMINATE
```

*Accept when:* `phase0/phase0_out.png` is a plausible photo, `phase0_lora_out.png` exists, and
`phase0_report.json` records `lora_load_ok: true`. Paste the report into `NOTES.md`, commit.

### Phase 1 — bake and serve

```
git tag v1 && git push origin main v1        # image from the frozen lock
bash scripts/pod.sh up                       # A40 -> A6000 fallback, 80 GB disk
bash scripts/pod.sh wait                     # ~6-12 min cold boot
bash scripts/smoke_test.sh                   # health, auth, guard, generate -> PNG
bash scripts/train_smoke.sh                  # 4 images -> 20-step LoRA -> download -> generate with it
```

*Accept when:* both print `PASS`. Record cold-boot seconds, `server_elapsed`, and training s/step
in `NOTES.md`.

### Phase 2 — your first real LoRA

Put 5–20 photos of yourself in `data/datasets/<name>/` as `.jpg`/`.png`, each with a same-name
`.txt` caption (see **Captions** below), then:

```
bash scripts/pod.sh up && bash scripts/pod.sh wait
bash scripts/train.sh data/datasets/pranav pranav-v1 pr4nv     # ~60-90 min, ~$0.60
bash scripts/generate.sh "photo of pr4nv person, ..." pranav-v1 0.9
SWEEP=1 bash scripts/generate.sh "photo of pr4nv person, ..." pranav-v1   # strength 0.6/0.8/1.0/1.2 + off
bash scripts/pod.sh down                                       # STOPS THE METER
```

The LoRA lands in `data/loras/pranav-v1.safetensors` with a `.json` sidecar; training previews in
`data/runs/<job>/samples/`; images in `data/images/`. **The pod keeps nothing.**

To retrain after fixing captions, run `train.sh` again with `pranav-v2` — no second cold boot
needed, the server unloads the pipeline, trains, and reloads it automatically.

### Phase 3 — the web app (not built yet)

Generate / Datasets / Train / Compare tabs, forked from `imgedit/web`. See `spec.md` §6.

## Captions — this is where LoRA quality comes from

- 5–20 photos, **varied**: close-up, half body, full body; different lighting, backgrounds,
  expressions, clothing. Duplicate framing is worse than fewer images.
- Each caption starts with the trigger word (a rare token + class noun, e.g. `pr4nv person`).
- Describe **everything except identity**: pose, framing, clothing, background, lighting.
  Do **not** describe face shape, eye colour or hair colour — that is what the LoRA must absorb.
- Example: `pr4nv person, half-body shot, grey hoodie, standing on a city street, overcast daylight, looking at camera`
- A LoRA trained only on face crops will not do full-body shots. Include both.

## Pod server contract

Every route except `/health` needs `Authorization: Bearer <API_TOKEN>`.

| Route | |
|---|---|
| `GET /health` | `{status, model_loaded, busy, gpu, capability, vram_used_gb, queue_depth, loras_loaded, defaults}` |
| `POST /generate` (json) | `prompt`, `negative?`, `steps?`, `guidance?`, `width?`, `height?`, `seed?` (-1 random), `n?` (1-4), `loras?` `[{name,scale}]` → `202 {job_id, seeds}` |
| `POST /train` (multipart) | `dataset` (.zip of images+captions), `name`, `config` (JSON overrides) → `202 {job_id, n_images, config}` |
| `GET /jobs/{id}` | generate: `seeds, images, progress`; train: `step, total_steps, loss, eta_s, samples, artifact` |
| `GET /jobs/{id}/image/{i}` · `/samples/{name}` · `/artifact` · `/log` | results, previews, the LoRA, trainer stdout |
| `DELETE /jobs/{id}` | cancels a running train job (SIGTERM), else frees memory |
| `GET/POST/DELETE /loras` | registry: list, upload+load, unload+delete |

Knobs are env vars (`server/config.py`): `MODEL_REPO`, `MODEL_FILE`, `DEFAULT_STEPS`,
`DEFAULT_GUIDANCE`, `MAX_AREA`, `TEXT_ENCODER_FP8`, `TRAIN_MAX_STEPS`. Switching to
`Chroma1-Base` is a config change, not a rewrite.

## Test the whole stack on the laptop without a GPU

```
cd server && MOCK=1 API_TOKEN=test python -m uvicorn app:app --port 8000
# then, in another terminal:
POD_URL=http://127.0.0.1:8000 API_TOKEN=test bash scripts/train_smoke.sh
```

The mock engine paints deterministic placeholder images and the mock trainer emits fake samples
and a fake artifact; queue, polling, guard, zip handling, progress parsing and saving are all real.

## Known traps (all handled, don't undo them)

- RunPod's proxy has a ~100 s Cloudflare timeout → the API is async; never make anything block.
- The server binds `0.0.0.0`; `127.0.0.1` is invisible to the proxy.
- The inference pipeline **must** be fully unloaded before the trainer starts (the server asserts
  < 2 GB allocated) and is reloaded afterwards whatever happened, including on cancel.
- Chroma has no guidance embedding — `guidance` is real CFG. Default 4.0; 1.0 is much worse.
- ai-toolkit saves LoRA keys ComfyUI-style; `server/lora_convert.py` normalises them for
  diffusers' Flux loader and drops Chroma-only modules the converter would reject.
- Stopping a pod without a network volume erases its disk. Nothing you care about lives there.
- RunPod needs ≥ 1 h of credit at the pod's rate before it will deploy.

## Last line

**Terminate the pod at the end of every session** — `.\pod.cmd down` or
`./tools/runpodctl.exe pod delete <POD_ID>`. Only deleting stops the meter; "stop" in the console
does not. A training run left going on a forgotten pod is the most expensive mistake available here.
