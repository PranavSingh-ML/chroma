# syntax=docker/dockerfile:1.6
# Same slim base as imgedit: CUDA 12.9 driver shim only (109 MB), torch wheels bring their own CUDA
# libraries. Pinned BY DIGEST (Docker Hub, 2026-09-17). ~6 GB compressed with ai-toolkit.
FROM nvidia/cuda:12.9.0-base-ubuntu24.04@sha256:48e21b10467354655f5073c05eebdeaac9818c6b40d70f334f7ad2df000463d8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/workspace/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_XET_HIGH_PERFORMANCE=1 \
    TOKENIZERS_PARALLELISM=false \
    AI_TOOLKIT_DIR=/opt/ai-toolkit \
    LORA_DIR=/workspace/loras \
    TRAIN_ROOT=/workspace/train \
    PATH=/opt/venv/bin:$PATH

# Python 3.12 (Ubuntu 24.04 default) + git (ai-toolkit checkout) + sshd for Phase 0 / emergencies.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip git openssh-server ca-certificates curl libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd /workspace /app \
 && python3 -m venv /opt/venv \
 && pip install --upgrade pip

# torch layer: exact wheel from the cu129 index. Changes rarely, so it caches well.
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
RUN pip install --index-url https://download.pytorch.org/whl/cu129 \
      "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# ai-toolkit at a PINNED commit (it has no releases). Its requirements are installed with the
# torch family filtered out (ours is above), and BEFORE our lock so our pins win on conflict.
# Record the SHA + date in NOTES.md whenever this changes.
ARG AI_TOOLKIT_SHA=a8dfcf7d7e2b38ccc7b2fb68ece9c6358e61e7a7
RUN git clone --filter=blob:none https://github.com/ostris/ai-toolkit.git /opt/ai-toolkit \
 && cd /opt/ai-toolkit && git checkout "${AI_TOOLKIT_SHA}" \
 && git submodule update --init --recursive \
 && cat requirements.txt requirements_base.txt 2>/dev/null \
      | grep -vE '^\s*(-r|#|$)' \
      | grep -viE '^(torch|torchvision|torchaudio|triton|xformers)\b' \
      > /tmp/ai-toolkit-reqs.txt \
 && echo "== ai-toolkit requirements (filtered):" && cat /tmp/ai-toolkit-reqs.txt \
 && pip install -r /tmp/ai-toolkit-reqs.txt \
 && rm -rf /opt/ai-toolkit/.git

WORKDIR /app

# Zero interactive pip on the pod: everything is baked here, from the frozen lock file.
# The lock never lists torch/torchvision/nvidia-*/triton (phase0_probe strips them) -
# those come from the torch layer above.
COPY requirements.lock.txt /app/requirements.lock.txt
RUN pip install -r /app/requirements.lock.txt \
 && python - <<'PY'
import torch, diffusers, transformers, peft, bitsandbytes, yaml
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("diffusers", diffusers.__version__, "transformers", transformers.__version__, "peft", peft.__version__,
      "bitsandbytes", bitsandbytes.__version__)
from diffusers import ChromaPipeline, ChromaTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler  # fail early
from diffusers.loaders.lora_conversion_utils import _convert_kohya_flux_lora_to_diffusers                       # lora_convert.py
from transformers import T5EncoderModel, T5TokenizerFast
import sys; sys.path.insert(0, "/opt/ai-toolkit")
import importlib; importlib.import_module("toolkit.job")                                                       # ai-toolkit imports
print("ai-toolkit importable")
PY

COPY config.py guard.py lora_convert.py pipeline.py trainer.py captioner.py app.py phase0_probe.py entrypoint.sh /app/
COPY train_lora_chroma.template.yaml /app/train_lora_chroma.template.yaml
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000 22
# RunPod: Expose HTTP Ports 8000, TCP 22; env API_TOKEN=<random>; container disk 80 GB.
# PHASE0=1 -> sshd only (no server), for running phase0_probe.py inside this exact image.
CMD ["/app/entrypoint.sh"]
