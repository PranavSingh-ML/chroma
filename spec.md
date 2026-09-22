# Build spec: character-LoRA studio for Chroma1-HD on a RunPod GPU

You are Claude Code. Build this repo. Read the whole spec before writing any code.
Where the spec says **VERIFY**, do not guess — run the check and record the real value
in `NOTES.md`. This project is the sibling of `../imgedit`; reuse its conventions and,
where noted, its code verbatim.

---

## 0. Goal and hard constraints

Build a **local web app** (laptop) plus a **stateless GPU server** (RunPod pod) that lets
the user:

1. **Train a character LoRA** for `lodestones/Chroma1-HD` from 5–20 photos of a single
   person (initially the user themself), with captions the user writes/edits in the UI.
2. **Generate images** with Chroma1-HD, hot-loading any trained LoRA by name at a chosen
   strength, with seed control, so the character stays consistent across prompts.
3. **Iterate**: train v1 → generate → fix captions / add images → train v2, all in one pod
   session, and compare LoRA versions side by side.

Constraints that drive every decision below:

1. **No persistent RunPod storage.** No network volume. Model weights are re-downloaded
   to container disk on every cold boot (~28 GB, **VERIFY** — same pattern as `imgedit`).
   Datasets go *up* from the laptop at train time; LoRAs come *down* to the laptop the
   moment training finishes. Killing the pod must lose nothing but the warm model.
2. **Pinned, pre-built Docker image. Zero `pip install` on the pod.** Same version-hell
   defence as `imgedit` §2 — one Dockerfile, one frozen lock, tags `v0, v1, …`, never
   `latest`, built on GitHub Actions → GHCR.
3. **No ComfyUI.** `diffusers` for inference, `ai-toolkit` (ostris) for training, both
   inside one image, behind one FastAPI server.
4. **Budget-aware.** A training run must cost ≤ ~$1 on an A40. A generation session
   should feel like `imgedit` (~$0.50/hr, auto-stop when idle).
5. **Adults only, consent only.** See §7. This is enforced in code, not just in docs.

---

## 1. Decisions (do not relitigate these without telling the user)

### Model: `lodestones/Chroma1-HD`

- 8.9B rectified-flow transformer, FLUX.1 architecture, **Apache 2.0**, **not gated**.
  Text encoder is **T5-XXL only** (no CLIP) → long natural-language captions and prompts
  work well, which is what a character LoRA wants.
- Not safety-filtered during training: produces realistic adult anatomy without prompt
  gymnastics. This is why it was chosen over FLUX.1-dev (filtered + non-commercial).
- Diffusers support: `ChromaPipeline` / `ChromaTransformer2DModel` are in the diffusers
  release `imgedit` already pins (0.40.0). **VERIFY** the repo revision and which files to
  download: use `allow_patterns` for `transformer/`, `text_encoder/`, `tokenizer/`, `vae/`,
  `scheduler/`, `*.json` and **exclude** the single-file `Chroma1-HD.safetensors` duplicate.
- No distillation → use real CFG. Defaults: **26 steps, guidance 4.0, 1024×1024**.
  **VERIFY** seconds/image on the A40 and adjust the default steps if >30 s.
- Optional speed-up (Phase 3, not before): `lodestones/Chroma1-Flash` delta weights for
  ~8-step previews. Train against **HD**, never Flash.

**Fallback if Chroma fights diffusers:** `lodestones/Chroma1-Base` (same pipeline class,
same LoRAs). Switching is a config change (`MODEL_ID`), not a rewrite.

### Precision: bf16, no quantisation

Transformer 17.8 GB + T5-XXL 9.5 GB (fp16/bf16) + VAE ≈ **~28 GB resident** on a 48 GB
card, with ~15 GB headroom for 1024² activations and a resident LoRA. **VERIFY** idle and
peak VRAM. Only if peak > 44 GB: cast T5 to fp8 storage with diffusers layerwise casting
(the `imgedit` `fp8_layerwise` path — code exists, copy it). Never CPU-offload on 48 GB.

### Trainer: `ostris/ai-toolkit`, pinned to a commit SHA

- Native Chroma support (`model.arch: "chroma"`), buckets multi-resolution datasets, caches
  latents + T5 embeddings to disk then frees T5, writes periodic samples and checkpoints,
  saves LoRAs in **diffusers/peft key format**. All driven by one YAML.
- Runs as a **subprocess** of the server (`python run.py <config.yaml>`), never in-process:
  a trainer crash must not kill the server, and its memory must be fully released after.
- **Pin the ai-toolkit commit SHA** in the Dockerfile and `NOTES.md`; it has no releases.
  Its own `requirements.txt` is installed at image build and folded into the frozen lock.
- Known: ai-toolkit Chroma training does **not** fit 24 GB (upstream issue #366). 48 GB is
  the floor — which is the card we use anyway.

**Fallback if ai-toolkit's Chroma path is broken at the pinned SHA:** `tdrussell/diffusion-pipe`
(also supports Chroma, more knobs, DeepSpeed-based). Second fallback: OneTrainer. Either
swaps in behind the same `/train` API; the laptop app does not change.

### LoRA loading into the inference pipeline — THE Phase-0 risk

`ChromaPipeline.load_lora_weights(path, adapter_name=…)` is the intended path.
**Known trap:** diffusers' Chroma LoRA converter has a key-mapping bug for *Kohya/OneTrainer*
format files (non-attention sublayers lose their suffix → `KeyError` in
`_maybe_expand_lora_state_dict`). ai-toolkit output should bypass this because it already
uses diffusers keys — **VERIFY in Phase 0 by training a 20-step throwaway LoRA and loading
it**. If it fails, implement `pipeline.merge_lora_manually(path, scale)`: read the
safetensors, map `transformer.<block>.<module>.lora_A/B` → the live module, do
`W += scale · (B @ A)` in bf16, keep the original weights on CPU to allow unloading. ~60
lines, format-agnostic, no dependency on the loader. Decide by test, not by reading docs.

### Hardware: A40 48 GB (Secure), RTX A6000 fallback — same as `imgedit`

- A40 ~$0.49/hr Secure (**VERIFY**, moves). Training and serving on the same card class so
  one image, one lock, one set of measurements covers everything.
- If a training run is annoyingly slow, L40S / RTX 6000 Ada are a `--gpu-id` change with
  the identical image. Do not start there.
- Container disk **80 GB** (28 GB weights + HF cache + dataset latent cache + checkpoints).

### Storage: none (user decision)

Per cold boot: image pull (~2.5 min) + ~28 GB weight download (2–7 min, **VERIFY**). Budget
**~$0.08 per cold boot**. That is the entire cost of skipping a $5–7/month volume.

Where things live:

| Thing | Laptop (`data/`, source of truth) | Pod (ephemeral) |
|---|---|---|
| Datasets (images + `.txt` captions) | `data/datasets/<char>/` | uploaded per train job to `/workspace/datasets/<job>/` |
| Trained LoRAs | `data/loras/<char>-v<N>.safetensors` + `.json` sidecar (config, steps, dataset hash) | `/workspace/output/<job>/`, downloaded by the laptop on `done`, then deleted |
| Generations | `data/images/*.png` + rows in `data/history.sqlite` | job dict, 30-min TTL |
| Training samples (progress previews) | `data/runs/<job>/samples/` | streamed to laptop as they appear |
| Model weights | — | `HF_HOME=/workspace/hf` |

The laptop **must** download the LoRA and mark the run `saved` before it lets the pod be
auto-stopped. A finished-but-unsaved run blocks auto-stop and shows a red banner.

### Serving: one pod, one worker, two job kinds — async API

Same reason as `imgedit`: RunPod's proxy has a ~100 s Cloudflare timeout, so nothing
blocks. **One GPU, one worker thread**, one queue holding jobs of kind `generate` or `train`:

- `generate` — runs on the resident pipeline. Seconds.
- `train` — the worker **unloads the inference pipeline** (`del`, `gc`, `empty_cache`),
  spawns ai-toolkit, streams its progress, and on exit **reloads the pipeline** from
  container disk (weights are cached; ~1–2 min, **VERIFY**) and hot-loads the new LoRA.
  Generate jobs queued behind a train job just wait; the UI shows "training, N generate
  jobs queued".

This gives the whole train → test → retrain loop in **one pod session** with no second
cold boot. It is the main UX win over "one pod per mode".

---

## 2. The version-hell fix

Copy `imgedit` §2 wholesale. Additions specific to this repo:

1. ai-toolkit is `git clone`d at a **pinned SHA** in the Dockerfile
   (`RUN git clone … && git checkout <sha>`), its `requirements.txt` installed **before**
   `pip freeze` in Phase 0 so the lock captures its transitive pins (it pulls in `peft`,
   `oyaml`, `albumentations`, `prodigyopt`, `lycoris-lora`, etc.). Conflicts between its
   pins and diffusers 0.40 are resolved **in Phase 0 on the GPU**, once, then frozen.
2. Base image: reuse `imgedit`'s `nvidia/cuda:12.9.0-base-ubuntu24.04@sha256:…` + torch
   `2.9.1+cu129` unless ai-toolkit at the pinned SHA needs a different torch (**VERIFY**;
   its requirements pin torch loosely — install ours first, then its deps with
   `--no-deps` for torch/torchvision).
3. `bitsandbytes` is **required** here (adamw8bit optimiser), not optional as in `imgedit`.
4. Image tags `ghcr.io/<owner>/chroma-lora:v0, v1, …`. Same Actions workflow file.

---

## 3. Repo layout

```
chroma-lora/
├── server/                    # runs on the pod
│   ├── Dockerfile
│   ├── requirements.in        # top-level pins (provisional until Phase 0)
│   ├── requirements.lock.txt  # frozen in Phase 0, never hand-edited
│   ├── entrypoint.sh          # sshd + /etc/environment fix (copy from imgedit) ; PHASE0=1 -> sshd only
│   ├── app.py                 # FastAPI: auth, routes, single worker, job dict + TTL
│   ├── pipeline.py            # ChromaPipeline load, LoRA registry (load/unload/set_scale), generate()
│   ├── trainer.py             # dataset unpack, YAML render, ai-toolkit subprocess, progress parser
│   ├── guard.py               # prompt/caption guard (§7)
│   ├── config.py              # every tunable, env-overridable (same style as imgedit)
│   └── phase0_probe.py        # env check, gen test, 20-step train test, LoRA load test, freeze
├── configs/
│   └── train_lora_chroma.template.yaml   # ai-toolkit config with {{placeholders}}
├── web/                       # laptop app — FORK of imgedit/web (Vite+React+Express, node:sqlite)
│   ├── server/                # index.ts, db.ts, jobs.ts, pod.ts, runpod.ts, settings.ts (+ datasets.ts, train.ts)
│   └── src/                   # Generate / Datasets / Train / Compare tabs, Pod panel
├── data/                      # laptop-side state, gitignored
│   ├── datasets/<char>/       # 001.jpg 001.txt … + dataset.json (trigger word, notes)
│   ├── loras/                 # <char>-v1.safetensors + <char>-v1.json
│   ├── runs/<job>/samples/    # training preview images
│   ├── images/                # generations
│   └── history.sqlite
├── scripts/
│   ├── pod.sh                 # up|wait|status|down  (copy from imgedit; name prefix "chroma", disk 80)
│   ├── phase0.sh              # drives phase0_probe over ssh, copies lock + report back
│   ├── smoke_test.sh          # /health, POST /generate, poll, PNG  -> PASS
│   └── train_smoke.sh         # uploads a 4-image synthetic dataset, 20-step train, downloads LoRA -> PASS
├── pod.cmd                    # Windows wrapper (copy from imgedit)
├── tools/                     # runpodctl.exe (gitignored)
├── .github/workflows/build-image.yml   # copy from imgedit, image name chroma-lora
├── NOTES.md · README.md · CLAUDE.md · spec.md
```

**Reuse from `imgedit` verbatim** (copy, adjust names only): `Dockerfile` skeleton,
`entrypoint.sh`, `phase0.sh`, `pod.sh`, `pod.cmd`, the Actions workflow, `.gitignore`,
`web/server/{runpod.ts,pod.ts,settings.ts}`, the Pod panel and connection pill components,
the auto-stop logic. Do not re-derive any of it.

---

## 4. Server contract

Auth: every route except `/health` requires `Authorization: Bearer <API_TOKEN>` (pod env
var). Non-negotiable — the proxy URL is public.

```
GET  /health
     -> {"status":"ok","model_loaded":bool,"gpu":str,"capability":[8,6],
         "vram_used_gb":float,"queue_depth":int,"busy":"idle|generate|train",
         "loras_loaded":["mara-v2"],"load_error":str|null}

POST /generate                          (application/json)
     prompt      : str (required)
     negative    : str = ""
     steps       : int = 26          guidance : float = 4.0
     width,height: int = 1024        (multiples of 16, area clamped to config.MAX_AREA)
     seed        : int = -1          (-1 = random; resolved seed is returned)
     n           : int = 1           (1-4; sequential, each its own seed = seed+i)
     loras       : [{"name":"mara-v2","scale":0.9}]   (0-2 entries; must be uploaded first)
     -> 202 {"job_id","seeds":[…]}
     -> 422 if guard.py rejects the prompt (see §7), with the reason

GET  /jobs/{id}
     -> {"kind":"generate|train","status":"queued|running|done|error",
         "progress":0-1,"elapsed_s":float,"error":str|null,
         # generate:
         "seeds":[…],"images":int,
         # train:
         "step":int,"total_steps":int,"loss":float|null,"eta_s":float|null,
         "samples":["s_000250_0.png",…],"artifact":"mara-v2.safetensors"|null}

GET  /jobs/{id}/image/{i}       -> image/png            (generate, done)
GET  /jobs/{id}/samples/{name}  -> image/png            (train, as they appear)
GET  /jobs/{id}/artifact        -> application/octet-stream, the LoRA (train, done)
GET  /jobs/{id}/log             -> text/plain, tail of trainer stdout (train)
DELETE /jobs/{id}               -> cancel if running (SIGTERM the trainer), free memory

POST /loras                      (multipart: file=<.safetensors>, name=<slug>)
     -> {"name","params":int}      # copies into the registry, loads onto the pipeline (unfused, scale 0)
GET  /loras                      -> [{"name","size_mb","loaded":bool}]
DELETE /loras/{name}

POST /train                      (multipart)
     dataset     : file  — a .zip of images + same-stem .txt captions (≤ 200 MB)
     name        : str   — LoRA name, e.g. "mara-v2"
     config      : str   — JSON overrides for the template (§5): trigger_word, steps, rank,
                          lr, resolutions, save_every, sample_every, sample_prompts[]
     -> 202 {"job_id"}
     -> 422 if any caption fails guard.py, listing the offending files
```

Server implementation rules (in addition to `imgedit` §4, which all still apply):

- **LoRA registry:** LoRAs are loaded *unfused* with `adapter_name`, and each generate job
  calls `set_adapters(names, weights)` for exactly the LoRAs it lists (others weight 0).
  Never fuse — the user needs strength sweeps and instant swap.
- **Train job lifecycle:** unpack zip → run guard over every `.txt` → render YAML from the
  template → free the pipeline → `subprocess.Popen([python, run.py, cfg])` with
  `HF_HOME` shared → parse `step/total` and `loss` from stdout (**VERIFY** the line
  format at the pinned SHA; fall back to counting files in the checkpoint dir) → copy
  samples into the job as they appear → on exit code 0 pick the final safetensors →
  reload the pipeline → load the new LoRA into the registry → `done`. Non-zero exit:
  `error` with the last 50 log lines; **pipeline is still reloaded** so the pod remains
  usable.
- **Only one train job at a time**; a second `POST /train` while one runs returns 409.
- Log every job as one JSON line to `/workspace/jobs.jsonl` (prompt, seeds, loras+scales,
  steps, size, elapsed / train config, final loss, elapsed).

---

## 5. Training recipe (the template, and why)

`configs/train_lora_chroma.template.yaml` — start from ai-toolkit's shipped Chroma example
(**VERIFY** its filename under `config/examples/` at the pinned SHA) and parameterise:

| Knob | Default | Why |
|---|---|---|
| `network.type` / `linear` / `linear_alpha` | lora / **16** / 16 | 5–20 photos of one person do not need rank 32; 16 trains faster and overfits less |
| `train.steps` | **100 × n_images, clamped 1000–3000** | rule of thumb for a single identity at batch 1 |
| `train.lr` | **1e-4** | ai-toolkit default for Flux-class LoRAs; drop to 5e-5 if samples look fried |
| `train.optimizer` | adamw8bit | needs bitsandbytes; halves optimiser VRAM |
| `train.batch_size` / `gradient_accumulation` | 1 / 1 | |
| `train.gradient_checkpointing` | true | required for 1024 on 48 GB |
| `train.noise_scheduler` | flowmatch | Chroma is rectified flow |
| `datasets[0].resolution` | **[512, 768, 1024]** | multi-res buckets; user photos are rarely square |
| `datasets[0].caption_ext` / `caption_dropout_rate` | txt / 0.05 | |
| `datasets[0].cache_latents_to_disk` | true | frees VAE + T5 during training |
| `model.quantize` | false | 48 GB — train in bf16, no quantised base |
| `save.save_every` / `sample.sample_every` | 250 / 250 | previews every ~5 min so bad runs die early |
| `sample.prompts` | 4 prompts from the request | must include the trigger word; the UI defaults to 2 SFW + 2 as-the-user-wrote |
| `sample.guidance_scale` / `sample_steps` | 4.0 / 20 | |
| `trigger_word` | user-chosen, e.g. `pr4nv person` | rare token + class noun; injected into captions by the UI if missing |

Captioning rules (shown in the UI as guidance, enforced softly):
- Every caption starts with the trigger word.
- Describe **everything except identity**: pose, framing, clothing, background, lighting,
  expression. Do *not* describe face shape / eye colour / hair colour — that is what the
  LoRA must absorb.
- 5–20 images: vary framing (close-up, half, full body), lighting, backgrounds. Duplicate
  framing is worse than fewer images.
- No auto-captioning in v1. Twenty captions by hand takes ten minutes; the UI provides a
  caption template and "apply trigger word to all".

Expected cost (**VERIFY** in Phase 0 and record): at ~1.5–2.5 s/step on an A40 at
1024-bucket, 2000 steps ≈ 50–85 min ≈ **$0.45–0.70** per run.

---

## 6. Local app

Fork `imgedit/web`. Keep the Pod panel, connection pill, cost meter, auto-stop, settings
persistence to `.env.local`. Replace the edit thread with four tabs:

**Generate** — prompt + negative, LoRA picker (multi, each with a strength slider 0–1.5),
steps / guidance / size / seed / count, a results grid. Each result card: seed, "reuse seed",
"reroll", "same seed, LoRA off" (identity A/B), and "send to Compare". Every PNG lands in
`data/images/` immediately; rows in SQLite `generations` (`id, prompt, negative, seed,
steps, guidance, w, h, loras_json, image_path, created_at`). A quick strength sweep
button: same prompt+seed at LoRA 0.6 / 0.8 / 1.0 / 1.2.

**Datasets** — one folder per character. Drag-drop images (auto-resized so the long side
≤ 1536, EXIF-stripped on ingest — phone photos carry GPS), a caption box per image, the
trigger word at the top, "apply trigger to all", per-image guard status. `dataset.json`
holds trigger word + notes. The SQLite `datasets` table mirrors the folder for search.

**Train** — pick a dataset, LoRA name (auto `<char>-v<N>`), the §5 knobs with defaults,
4 sample prompts, **Start**. Uploads the zip, shows step / total / loss / ETA / cost so
far, sample previews as they arrive (saved to `data/runs/<job>/samples/`), the log tail.
On `done`: download the artifact to `data/loras/`, write the `.json` sidecar, mark the run
`saved`, and only then allow auto-stop again. Row in SQLite `training_runs`
(`id, dataset, name, config_json, status, steps, final_loss, cost_usd, lora_path, started_at, finished_at`).

**Compare** — pick a prompt + seed, pick up to three LoRAs (or versions), render them
side by side at equal strength. This is how the user decides v2 vs v3 without squinting
across tabs.

Behaviour rules:
- Optimistic UI + 1 s polling of `GET /jobs/{id}`, same as `imgedit`.
- "Unsaved LoRA" banner + auto-stop block, per §1 Storage.
- Session cost meter counts training time separately so the user sees "$0.62 of this
  session was training".
- No prompt rewriter in v1.

---

## 7. Consent, age, and what the code enforces

The user has stated the subject is **themself**. The trainer is a general tool, so:

- `README.md` and the Datasets tab state plainly: train only on **your own** likeness or
  on adults who have **explicitly consented** to adult AI training use; never on a third
  party's photos. This is the user's responsibility and the tool will not check it.
- `guard.py` rejects prompts **and captions** that contain terms indicating a minor
  (a short, explicit denylist: age words/numbers below 18, school-age terms, "loli" etc.),
  returning 422 with the matched term. Applied to `POST /generate`, `POST /train` captions,
  and the training `sample.prompts`. This is a cheap, deterministic backstop, not a
  classifier; do not oversell it in the UI.
- The Datasets ingest strips EXIF/GPS. Nothing in `data/` is ever committed (gitignored,
  same as `imgedit`).
- The pod image contains no data; datasets exist on the pod only for the duration of a
  train job and are deleted on job completion.

---

## 8. Build order — each phase has an acceptance test

**Phase 0 — pin the environment (~30–40 min pod time, ~$0.35)**
Build `v0` from provisional top-level pins (diffusers 0.40.0, transformers as `imgedit`,
bitsandbytes, ai-toolkit at a chosen SHA + its requirements). Start a `PHASE0=1` pod
(sshd only). `phase0.sh` runs `phase0_probe.py` inside the image, which:
1. logs GPU / capability / torch / diffusers / ai-toolkit SHA;
2. downloads Chroma1-HD (diffusers folders only) and times it; loads `ChromaPipeline` bf16;
   records idle VRAM;
3. generates two 1024² images (a neutral SFW prompt), records s/image and peak VRAM;
4. writes a 4-image synthetic dataset + captions, renders the template, runs ai-toolkit for
   **20 steps**, records s/step and peak VRAM, confirms a `.safetensors` appears;
5. **loads that LoRA into `ChromaPipeline` via `load_lora_weights`**; on failure, tries the
   manual-merge path and records which one works;
6. generates one image with the LoRA at scale 1.0 (proves the whole loop);
7. `pip freeze` → `requirements.lock.txt` (torch layer stripped), copies everything back.
*Accept when:* `phase0_out.png` and `phase0_lora_out.png` exist, the LoRA-load path is
recorded, and the lock is in the repo. Fill `NOTES.md`. **Terminate the pod.**

**Phase 1 — bake and serve**
`v1` from the frozen lock; `app.py`, `pipeline.py`, `trainer.py`, `guard.py`. Launch from
the custom image, cold, nobody SSHing in.
*Accept when:* `scripts/smoke_test.sh` prints `PASS` (health → generate → PNG) **and**
`scripts/train_smoke.sh` prints `PASS` (upload 4-image zip → 20-step train → artifact
downloaded → `/loras` lists it → a generate job with it succeeds). Record cold-boot seconds
and both elapsed times.

**Phase 2 — minimal app**
Generate tab (no LoRA picker yet) + Datasets tab + Train tab with progress and artifact
download. Verified first against `MOCK=1` on the laptop (fake pipeline; fake trainer that
sleeps and emits samples), then against a real pod with the user's real 5–20 photos.
*Accept when:* the user's first real LoRA is in `data/loras/`, its sidecar is correct, and
three generations with it exist in `data/images/` with SQLite rows.

**Phase 3 — the consistency loop**
LoRA picker with strengths, strength sweep, LoRA-off A/B, Compare tab, unsaved-LoRA
auto-stop block, training cost split in the meter. Then, optionally, Chroma1-Flash preview
mode.

---

## 9. Known traps — check each one explicitly

- Everything in `imgedit` §7 (100 s proxy timeout, bind `0.0.0.0`, warmup before
  `model_loaded`, stopped pod = erased disk, ≥1 h credit, spend alert, terminate at end).
- **Train job and generate job share one GPU.** The pipeline *must* be freed before the
  trainer starts or the trainer OOMs at step 0. Assert `vram_used_gb < 2` before spawning.
- **ai-toolkit and diffusers pin conflicts** surface at image build (`pip install` fails or
  the import check in the Dockerfile fails). Resolve in Phase 0, once; never on a serving pod.
- **T5 tokeniser needs `sentencepiece`** — make sure it is in the lock.
- **ai-toolkit's HF download** uses the same `HF_HOME`; if it re-downloads, the
  `name_or_path` in the YAML does not match the pipeline's repo id/revision — fix the YAML.
- **LoRA-format mismatch** (§1). Decided by test in Phase 0. Whatever path wins is the only
  path in `pipeline.py`; the other is deleted, not left as dead code.
- **Phone photos**: HEIC uploads need conversion on the laptop (use `sharp` in the Express
  server); EXIF orientation must be applied before resize or the trainer sees sideways
  faces.
- **Chroma has no guidance embedding** — `guidance_scale` is real CFG; 1.0 means no
  negative pass and noticeably worse adherence. Default 4.0.
- A LoRA that only ever saw 1024-cropped faces will not do full-body shots. Say so in the
  Datasets tab guidance.
- **A cancelled train job** must SIGTERM the subprocess, wait, and *still* reload the
  pipeline. Test this on the mock.

---

## 10. What to write into NOTES.md as you go

Base image digest · torch/CUDA · diffusers version · ai-toolkit SHA + date · Chroma1-HD
revision + bytes actually downloaded · GPU/capability · VRAM idle / peak generate / peak
train · s/image at 26 steps 1024² · s/step training at [512,768,1024] rank 16 · which
LoRA-load path works · cold-boot seconds · pipeline reload seconds after training · $ per
cold boot / per 2000-step run / per image · running spend total · every deviation from
this spec, told to the user.
