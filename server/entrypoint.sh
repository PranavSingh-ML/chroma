#!/usr/bin/env bash
# Pod entrypoint.
#  - RunPod injects PUBLIC_KEY (your account's SSH keys) -> start sshd so `runpodctl ssh info` works.
#  - PHASE0=1 -> sshd only; used once to run phase0_probe.py inside this exact image.
#  - otherwise -> the API server, bound to 0.0.0.0 (127.0.0.1 is invisible to the RunPod proxy).
set -u
# sshd sessions do not inherit Docker ENV; export the container env for PAM/login shells.
env | grep -E '^(PATH|HF_|PYTORCH|API_TOKEN|MODEL_|EXTRAS_|SCHEDULER_|TEXT_ENCODER_FP8|PORT|TOKENIZERS|PHASE0|DEFAULT_|MAX_|AI_TOOLKIT_DIR|LORA_DIR|TRAIN_|LOG_PATH|WARMUP)' > /etc/environment
{ echo 'set -a; . /etc/environment; set +a'; } > /etc/profile.d/10-container-env.sh
if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  ssh-keygen -A >/dev/null 2>&1
  /usr/sbin/sshd -p 22 && echo "[entrypoint] sshd up on :22"
fi
mkdir -p "${LORA_DIR:-/workspace/loras}" "${TRAIN_ROOT:-/workspace/train}" "${HF_HOME:-/workspace/hf}"
cd /app
echo "[entrypoint] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "[entrypoint] HF_HOME=${HF_HOME:-unset}  API_TOKEN set: $([ -n "${API_TOKEN:-}" ] && echo yes || echo NO)"
if [ "${PHASE0:-0}" = "1" ]; then
  echo "[entrypoint] PHASE0=1: not starting the server. ssh in and run: cd /app && python phase0_probe.py"
  exec sleep infinity
fi
exec python -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --log-level info
