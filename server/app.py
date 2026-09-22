"""FastAPI server: routes + a single-worker queue holding two job kinds (generate | train).

Contract (spec.md section 4):
  GET    /health                        no auth
  POST   /generate                      json -> 202 {"job_id","seeds"}
  POST   /train                         multipart (dataset zip, name, config json) -> 202 {"job_id"}
  GET    /jobs/{id}                     status/progress/... ; train jobs add step/total/loss/samples/artifact
  GET    /jobs/{id}/image/{i}           image/png (generate, done)
  GET    /jobs/{id}/samples/{name}      image (train, as they appear)
  GET    /jobs/{id}/artifact            the LoRA safetensors (train, done)
  GET    /jobs/{id}/log                 trainer stdout tail
  DELETE /jobs/{id}                     cancel if running, free memory
  POST   /loras   GET /loras   DELETE /loras/{name}

One GPU, one worker thread. A train job unloads the inference pipeline first and the worker
reloads it afterwards no matter how training ended. The worker never dies on a job exception.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

import config
import guard
import trainer as trainer_mod
from pipeline import clamp_size, make_engine, to_png_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

if not config.API_TOKEN and not config.MOCK:
    log.error("API_TOKEN env var is empty. Refusing to expose an open GPU endpoint. Set API_TOKEN and restart.")
    sys.exit(2)


# ----------------------------------------------------------------------------
# job store
# ----------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    kind: str  # generate | train
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    elapsed_s: float = 0.0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    # generate
    prompt: str = ""
    negative: str = ""
    steps: int = 26
    guidance: float = 4.0
    width: int = 1024
    height: int = 1024
    seeds: list = field(default_factory=list)
    loras: list = field(default_factory=list)
    pngs: list = field(default_factory=list)
    # train
    name: str = ""
    job_dir: str = ""
    train_cfg: dict = field(default_factory=dict)
    n_images: int = 0
    step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    eta_s: Optional[float] = None
    samples: list = field(default_factory=list)
    artifact: Optional[str] = None
    run: Optional[trainer_mod.TrainRun] = None

    def public(self) -> dict:
        d = {"kind": self.kind, "status": self.status, "progress": round(self.progress, 3),
             "elapsed_s": round(self.elapsed_s, 2), "error": self.error}
        if self.kind == "generate":
            d.update({"seeds": self.seeds, "images": len(self.pngs), "steps": self.steps, "guidance": self.guidance,
                      "width": self.width, "height": self.height, "loras": self.loras})
        else:
            d.update({"name": self.name, "step": self.step, "total_steps": self.total_steps, "loss": self.loss,
                      "eta_s": self.eta_s, "samples": self.samples, "n_images": self.n_images,
                      "artifact": os.path.basename(self.artifact) if self.artifact else None,
                      "config": self.train_cfg})
        return d


class JobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.q: "list[str]" = []
        self.cv = threading.Condition(self.lock)

    def add(self, job: Job) -> None:
        with self.cv:
            self.jobs[job.id] = job
            self.q.append(job.id)
            self._cap_locked()
            self.cv.notify()

    def next(self) -> Job:
        with self.cv:
            while True:
                while self.q:
                    jid = self.q.pop(0)
                    job = self.jobs.get(jid)
                    if job is not None:
                        return job
                self.cv.wait()

    def get(self, job_id: str) -> Optional[Job]:
        with self.lock:
            return self.jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.pop(job_id, None)
            if job_id in self.q:
                self.q.remove(job_id)
        if job is None:
            return False
        if job.kind == "train" and job.job_dir:
            shutil.rmtree(job.job_dir, ignore_errors=True)
        return True

    def depth(self) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if j.status in ("queued", "running"))

    def active_train(self) -> Optional[Job]:
        with self.lock:
            for j in self.jobs.values():
                if j.kind == "train" and j.status in ("queued", "running"):
                    return j
        return None

    def _cap_locked(self) -> None:
        finished = [j for j in self.jobs.values() if j.status in ("done", "error")]
        if len(finished) > config.JOB_CAP:
            finished.sort(key=lambda j: j.finished_at or 0)
            for j in finished[: len(finished) - config.JOB_CAP]:
                self.jobs.pop(j.id, None)

    def sweep(self) -> None:
        now = time.time()
        with self.lock:
            dead = [k for k, j in self.jobs.items()
                    if j.status in ("done", "error") and (now - (j.finished_at or now)) > config.JOB_TTL_S]
            for k in dead:
                j = self.jobs.pop(k)
                if j.kind == "train" and j.job_dir:
                    shutil.rmtree(j.job_dir, ignore_errors=True)
            self._cap_locked()


store = JobStore()
engine = make_engine()
state = {"model_loaded": False, "load_error": None, "started_at": time.time(), "busy": "idle"}


# ----------------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------------
def _log_job(job: Job) -> None:
    rec = {"ts": time.time(), "job_id": job.id, "kind": job.kind, "status": job.status,
           "elapsed_s": round(job.elapsed_s, 2), "error": (job.error or "").splitlines()[-1] if job.error else None}
    if job.kind == "generate":
        rec.update({"prompt": job.prompt, "negative": job.negative, "seeds": job.seeds, "steps": job.steps,
                    "guidance": job.guidance, "width": job.width, "height": job.height, "loras": job.loras})
    else:
        rec.update({"name": job.name, "n_images": job.n_images, "steps": job.total_steps, "final_loss": job.loss,
                    "config": job.train_cfg})
    log.info("JOB %s", json.dumps(rec))
    try:
        os.makedirs(os.path.dirname(config.LOG_PATH) or ".", exist_ok=True)
        with open(config.LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _run_generate(job: Job) -> None:
    t0 = time.time()

    def progress(p: float) -> None:
        job.progress = p
        job.elapsed_s = time.time() - t0

    imgs = engine.generate(job.prompt, job.negative, job.steps, job.guidance, job.seeds, job.width, job.height,
                           job.loras, progress)
    job.pngs = [to_png_bytes(im) for im in imgs]


def _run_train(job: Job) -> None:
    t0 = time.time()
    run = job.run
    assert run is not None
    state["busy"] = "train"
    log.info("train %s: unloading pipeline", job.id)
    engine.unload()
    if not config.MOCK and engine.vram_used_gb() > 2.0:
        raise RuntimeError(f"pipeline did not release the GPU ({engine.vram_used_gb()} GB still allocated)")

    def on_progress(st: dict) -> None:
        job.step, job.total_steps = st.get("step", job.step), st.get("total_steps", job.total_steps)
        job.loss, job.eta_s = st.get("loss", job.loss), st.get("eta_s", job.eta_s)
        job.samples = st.get("samples", job.samples)
        job.progress = job.step / max(1, job.total_steps)
        job.elapsed_s = time.time() - t0

    try:
        job.artifact = run.run(on_progress)
    finally:
        run.cleanup()
        state["busy"] = "reload"
        log.info("train %s: reloading pipeline", job.id)
        t1 = time.time()
        engine.load()
        _warmup()
        log.info("train %s: pipeline back in %.0fs", job.id, time.time() - t1)
    engine.add_lora(job.name, job.artifact)


def _run_job(job: Job) -> None:
    job.status = "running"
    state["busy"] = job.kind
    t0 = time.time()
    try:
        if job.kind == "generate":
            _run_generate(job)
        else:
            _run_train(job)
        job.status = "done"
        job.progress = 1.0
    except Exception as e:  # noqa: BLE001 - keep the worker alive no matter what
        job.error = str(e) if isinstance(e, trainer_mod.TrainError) else traceback.format_exc()
        job.status = "error"
        log.exception("job %s failed", job.id)
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    finally:
        job.elapsed_s = time.time() - t0
        job.finished_at = time.time()
        state["busy"] = "idle"
        _log_job(job)


def _warmup() -> None:
    if not config.WARMUP:
        return
    log.info("warmup generate starting")
    t0 = time.time()
    engine.generate("a grey square", "", 4, 1.0, [0], 512, 512, [], lambda p: None)
    log.info("warmup done in %.1fs", time.time() - t0)


def worker() -> None:
    try:
        engine.load()
        _warmup()
        state["model_loaded"] = True
    except Exception:  # noqa: BLE001
        state["load_error"] = traceback.format_exc()
        log.exception("MODEL LOAD FAILED - server stays up so /health can report it")
        while True:
            job = store.next()
            job.status, job.error, job.finished_at = "error", "model failed to load:\n" + state["load_error"], time.time()
    while True:
        job = store.next()
        _run_job(job)
        state["model_loaded"] = engine.loaded


def sweeper() -> None:
    while True:
        time.sleep(60)
        store.sweep()


threading.Thread(target=worker, name="gpu-worker", daemon=True).start()
threading.Thread(target=sweeper, name="job-sweeper", daemon=True).start()


# ----------------------------------------------------------------------------
# auth
# ----------------------------------------------------------------------------
def require_token(request: Request) -> None:
    if config.MOCK and not config.API_TOKEN:
        return
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    if not hmac.compare_digest(auth[7:].strip(), config.API_TOKEN):
        raise HTTPException(403, "bad token")


app = FastAPI(title="chroma-lora", docs_url=None, redoc_url=None)


# ----------------------------------------------------------------------------
# routes
# ----------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    info = engine.info or {}
    return {
        "status": "ok",
        "model_loaded": state["model_loaded"] and engine.loaded,
        "load_error": state["load_error"],
        "busy": state["busy"],
        "gpu": info.get("gpu"),
        "capability": info.get("capability"),
        "model": info.get("model", f"{config.MODEL_REPO}/{config.MODEL_FILE}"),
        "vram_used_gb": engine.vram_used_gb(),
        "vram_total_gb": info.get("vram_total_gb"),
        "queue_depth": store.depth(),
        "loras_loaded": [l["name"] for l in engine.list_loras() if l.get("loaded")],
        "uptime_s": round(time.time() - state["started_at"], 1),
        "defaults": {"steps": config.DEFAULT_STEPS, "guidance": config.DEFAULT_GUIDANCE,
                     "width": config.DEFAULT_WIDTH, "height": config.DEFAULT_HEIGHT, "train": config.TRAIN_DEFAULTS},
    }


class LoraRef(BaseModel):
    name: str
    scale: float = Field(1.0, ge=0.0, le=2.0)


class GenerateReq(BaseModel):
    prompt: str
    negative: str = ""
    steps: Optional[int] = None
    guidance: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: int = -1
    n: int = Field(1, ge=1, le=config.MAX_IMAGES_PER_JOB)
    loras: list[LoraRef] = Field(default_factory=list)


@app.post("/generate", status_code=202, dependencies=[Depends(require_token)])
def generate(req: GenerateReq) -> dict:
    if not req.prompt.strip():
        raise HTTPException(400, "prompt is empty")
    hit = guard.check(req.prompt) or guard.check(req.negative)
    if hit:
        raise HTTPException(422, f"prompt rejected: {hit!r}")
    if len(req.loras) > config.MAX_LORAS_PER_JOB:
        raise HTTPException(400, f"at most {config.MAX_LORAS_PER_JOB} loras per job")
    known = {l["name"] for l in engine.list_loras()}
    for l in req.loras:
        if l.name not in known:
            raise HTTPException(400, f"unknown lora {l.name!r}; upload it first (POST /loras)")
    steps = int(req.steps) if req.steps is not None else config.DEFAULT_STEPS
    if not (1 <= steps <= config.MAX_STEPS):
        raise HTTPException(400, f"steps must be 1..{config.MAX_STEPS}")
    guidance = float(req.guidance) if req.guidance is not None else config.DEFAULT_GUIDANCE
    w, h = clamp_size(req.width, req.height)
    seed = req.seed if req.seed >= 0 else random.randint(0, 2**31 - 1)
    seeds = [(seed + i) % (2**31) for i in range(req.n)]
    job = Job(id=uuid.uuid4().hex[:12], kind="generate", prompt=req.prompt.strip(),
              negative=req.negative or config.DEFAULT_NEGATIVE, steps=steps, guidance=guidance,
              width=w, height=h, seeds=seeds, loras=[l.model_dump() for l in req.loras])
    store.add(job)
    return {"job_id": job.id, "seeds": seeds, "width": w, "height": h}


@app.post("/train", status_code=202, dependencies=[Depends(require_token)])
async def train(dataset: UploadFile = File(...), name: str = Form(...), config_json: str = Form("{}", alias="config")) -> dict:
    if store.active_train() is not None:
        raise HTTPException(409, "a training job is already queued or running")
    try:
        overrides = json.loads(config_json or "{}")
        if not isinstance(overrides, dict):
            raise ValueError
    except ValueError:
        raise HTTPException(400, "config must be a JSON object") from None
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in name.strip())[:64]
    if not safe:
        raise HTTPException(400, "bad name")
    if safe in {l["name"] for l in engine.list_loras()}:
        raise HTTPException(409, f"lora {safe!r} already exists; pick a new version name")
    hits = guard.check_many({f"sample_prompt[{i}]": str(p) for i, p in enumerate(overrides.get("sample_prompts") or [])})
    hits.update(guard.check_many({"trigger_word": str(overrides.get("trigger_word") or "")}))
    if hits:
        raise HTTPException(422, {"rejected": hits})

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(config.TRAIN_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    zip_path = os.path.join(job_dir, "dataset.zip")
    size = 0
    with open(zip_path, "wb") as f:
        while chunk := await dataset.read(1 << 20):
            size += len(chunk)
            if size > config.TRAIN_MAX_ZIP_MB << 20:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, f"dataset zip > {config.TRAIN_MAX_ZIP_MB} MB")
            f.write(chunk)
    ds_dir = os.path.join(job_dir, "dataset")
    try:
        info = trainer_mod.unpack_dataset(zip_path, ds_dir)
    except (trainer_mod.TrainError, Exception) as e:  # noqa: BLE001
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, f"bad dataset: {e}") from e
    os.remove(zip_path)
    hits = trainer_mod.check_captions(ds_dir)
    if hits:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(422, {"rejected": hits})
    if info["missing"]:
        log.warning("train %s: %d images without captions (trigger word only): %s", job_id, len(info["missing"]), info["missing"][:5])
    model_path = "mock" if config.MOCK else engine.model_path or engine.ensure_weights()
    try:
        cfg = trainer_mod.render_config(safe, ds_dir, os.path.join(job_dir, "output"), model_path, info["images"], overrides)
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, f"bad config: {e}") from e
    run = trainer_mod.make_run(job_dir, safe, cfg)
    p = cfg["config"]["process"][0]
    summary = {"steps": p["train"]["steps"], "rank": p["network"]["linear"], "lr": p["train"]["lr"],
               "resolutions": p["datasets"][0]["resolution"], "trigger_word": p["trigger_word"],
               "sample_prompts": p["sample"]["prompts"], "save_every": p["save"]["save_every"],
               "sample_every": p["sample"]["sample_every"], "optimizer": p["train"]["optimizer"]}
    job = Job(id=job_id, kind="train", name=safe, job_dir=job_dir, train_cfg=summary, n_images=info["images"],
              total_steps=p["train"]["steps"], run=run)
    store.add(job)
    return {"job_id": job.id, "name": safe, "n_images": info["images"], "captions": info["captions"],
            "missing_captions": info["missing"], "config": summary}


def _get(job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job (expired or deleted)")
    return job


@app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
def job_status(job_id: str) -> dict:
    return _get(job_id).public()


@app.get("/jobs/{job_id}/image/{i}", dependencies=[Depends(require_token)])
def job_image(job_id: str, i: int) -> Response:
    job = _get(job_id)
    if job.kind != "generate":
        raise HTTPException(400, "not a generate job")
    if job.status != "done" or not job.pngs:
        return JSONResponse({"status": job.status, "error": job.error}, status_code=409)
    if not (0 <= i < len(job.pngs)):
        raise HTTPException(404, "no such image index")
    return Response(content=job.pngs[i], media_type="image/png",
                    headers={"X-Seed": str(job.seeds[i]), "X-Elapsed-S": f"{job.elapsed_s:.2f}"})


@app.get("/jobs/{job_id}/samples/{name}", dependencies=[Depends(require_token)])
def job_sample(job_id: str, name: str) -> Response:
    job = _get(job_id)
    if job.kind != "train" or job.run is None:
        raise HTTPException(400, "not a train job")
    p = job.run.sample_path(name)
    if not p:
        raise HTTPException(404, "no such sample")
    return FileResponse(p)


@app.get("/jobs/{job_id}/artifact", dependencies=[Depends(require_token)])
def job_artifact(job_id: str) -> Response:
    job = _get(job_id)
    if job.kind != "train":
        raise HTTPException(400, "not a train job")
    if job.status != "done" or not job.artifact or not os.path.isfile(job.artifact):
        return JSONResponse({"status": job.status, "error": job.error}, status_code=409)
    return FileResponse(job.artifact, media_type="application/octet-stream", filename=f"{job.name}.safetensors")


@app.get("/jobs/{job_id}/log", dependencies=[Depends(require_token)])
def job_log(job_id: str, n: int = 80) -> PlainTextResponse:
    job = _get(job_id)
    if job.kind != "train" or job.run is None:
        raise HTTPException(400, "not a train job")
    return PlainTextResponse(job.run.log_tail(max(1, min(n, 2000))))


@app.delete("/jobs/{job_id}", dependencies=[Depends(require_token)])
def job_delete(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        return {"deleted": False}
    if job.kind == "train" and job.status == "running" and job.run is not None:
        job.run.cancel()
        return {"deleted": False, "cancelling": True}
    if job.status == "running":
        raise HTTPException(409, "generate job is running; wait for it")
    return {"deleted": store.delete(job_id)}


# ---- LoRA registry ---------------------------------------------------------
@app.get("/loras", dependencies=[Depends(require_token)])
def loras_list() -> list:
    return engine.list_loras()


@app.post("/loras", dependencies=[Depends(require_token)])
async def loras_upload(file: UploadFile = File(...), name: str = Form(...)) -> dict:
    if state["busy"] in ("train", "reload") or not engine.loaded:
        raise HTTPException(409, "pipeline is not loaded right now (training?) - retry later")
    fd, tmp = tempfile.mkstemp(suffix=".safetensors")
    size = 0
    with os.fdopen(fd, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > 2 << 30:
                os.remove(tmp)
                raise HTTPException(413, "lora too large")
            f.write(chunk)
    try:
        meta = engine.add_lora(name, tmp)
    except Exception as e:  # noqa: BLE001
        log.exception("lora upload failed")
        raise HTTPException(400, f"could not load lora: {e}") from e
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return meta


@app.delete("/loras/{name}", dependencies=[Depends(require_token)])
def loras_delete(name: str) -> dict:
    return {"deleted": engine.remove_lora(name)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
