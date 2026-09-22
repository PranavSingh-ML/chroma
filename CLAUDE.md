# chroma-lora — instructions for Claude Code (any machine)

Character-LoRA studio for `lodestones/Chroma1-HD`. The laptop owns all state (`data/`); a
disposable RunPod GPU pod runs `server/` from a pinned Docker image and does both generation and
LoRA training. Design: `spec.md`. Pins, measurements, spend: `NOTES.md`. Human runbook: `README.md`.
Sibling project `../imgedit` — same conventions; much of `scripts/` and the CI workflow came from it.

## Windows note

Type `bash` on Windows may be **WSL** (its own ancient Node) rather than Git Bash. Use
`.\pod.cmd <up|wait|status|down>` on Windows — it runs `scripts/pod.sh` under Git Bash.
There is no `python3` in Git Bash; scripts source `scripts/_py.sh` to find a real interpreter.
`.venv/` here is laptop-only (mock testing), never shipped.

## The one rule

**A running pod costs money every minute ($0.49–0.53/hr). Never end a session with a pod up
unless the user explicitly said to keep it.** Deleting the pod (`bash scripts/pod.sh down`) is the
only thing that stops billing — "stop" in the RunPod console does NOT. Check with
`bash scripts/pod.sh status` whenever unsure. A forgotten pod mid-training is the most expensive
mistake available in this repo.

## Ethics — not negotiable, do not weaken

Training is for the **user's own likeness**, or adults who have **explicitly consented**. Never
help train on a third party's photos. `server/guard.py` rejects prompts/captions indicating a
minor and is applied to `/generate`, every training caption, and sample prompts. Do not remove it,
do not add a bypass flag, do not narrow the denylist. If asked to, decline and say why.

## Daily use

```
bash scripts/pod.sh up && bash scripts/pod.sh wait     # ~6-12 min cold boot
bash scripts/train.sh data/datasets/<name> <lora-v1> <trigger>    # ~60-90 min, ~$0.60
bash scripts/generate.sh "photo of <trigger> person, ..." <lora-v1> 0.9
SWEEP=1 bash scripts/generate.sh "..." <lora-v1>       # strength 0.6/0.8/1.0/1.2 + lora-off
bash scripts/pod.sh down                               # WHEN DONE
```

Retraining (`<lora-v2>`) needs **no second cold boot**: the server unloads the pipeline, runs
ai-toolkit, reloads, and hot-loads the new LoRA. Results: `data/loras/`, `data/images/`,
`data/runs/<job>/samples/` — all gitignored, all the user's, never delete.

## Architecture facts worth not re-deriving

- One GPU, one worker thread, one queue, two job kinds (`generate` | `train`). A train job frees
  the pipeline first (asserts < 2 GB allocated) and **always** reloads it in a `finally`, including
  on cancel or crash. Never make anything synchronous — RunPod's proxy times out at ~100 s.
- The trainer is always a **subprocess** (`python run.py <cfg>` in `/opt/ai-toolkit`). Progress is
  parsed from its tqdm line. Its crash must not kill the server.
- Inference and training **share one HF cache**: the single-file `Chroma1-HD.safetensors` plus
  `ostris/Flex.1-alpha` T5/VAE — exactly what ai-toolkit's chroma loader fetches. Changing
  `MODEL_REPO`/`MODEL_FILE` without keeping that alignment doubles the download to ~45 GB.
- LoRA key formats are a real trap; `server/lora_convert.py` is the single place that deals with
  it and `NOTES.md` explains why each branch exists. Phase 0 decides `lora_load_path` **by test**.
- LoRAs are loaded **unfused** with `adapter_name` and selected per job via `set_adapters`, so
  strength sweeps and instant swaps work. Never `fuse_lora()`.

## Testing without a GPU (do this before every push that touches server/)

```
cd server && MOCK=1 API_TOKEN=test python -m uvicorn app:app --port 8078
POD_URL=http://127.0.0.1:8078 API_TOKEN=test bash scripts/smoke_test.sh
POD_URL=http://127.0.0.1:8078 API_TOKEN=test bash scripts/train_smoke.sh
```

Everything except the model and the trainer is real. `NOTES.md` lists what this has already caught.

## Changing the pod image (rare)

Edit `server/*`, commit, then `git tag v2 && git push origin main v2` → Actions builds
`ghcr.io/pranavsingh-ml/chroma-lora:v2` (~15 min, free). Never tag `latest`. Then
`IMAGE=ghcr.io/pranavsingh-ml/chroma-lora:v2 bash scripts/pod.sh up`. If Python deps change,
re-run Phase 0 to re-freeze `server/requirements.lock.txt` — do not hand-edit the lock.
Bumping `AI_TOOLKIT_SHA` is a Phase-0-level change: the LoRA key format may move with it.

## Where credentials live (never commit, never print)

| Secret | Location | In git? |
|---|---|---|
| RunPod API key | `~/.runpod/config.toml` (written by `runpodctl doctor` — the **user** runs this) | no |
| RunPod SSH key | `~/.runpod/ssh/runpodctl-ssh-key` | no |
| Pod `API_TOKEN` | `web/.env.local` (also passed as pod env at create time) | no (gitignored) |

Claude must not handle the RunPod API key. `data/` is the user's photos — never commit, never
copy outside the project, never include in a bug report.

## State of the build (2026-09-22)

Done: spec, pod server (generate + train + LoRA registry + guard), Dockerfile, ai-toolkit
template, Phase-0 probe, all scripts, mock-verified end to end.
Next: Phase 0 on a GPU (`README.md`) → freeze lock → `v1` → smoke tests → first real LoRA.
Not built: the web app (`web/`, spec.md §6) — Phase 3, and not on the critical path.
