#!/usr/bin/env bash
# Phase 0 - runs FROM THE LAPTOP, drives a pod created from chroma-lora:v0 with PHASE0=1.
# Everything happens inside the exact image that will ship, so the pip freeze is the truth and
# the LoRA-format question is answered by test.
#
#   PHASE0=1 IMAGE=ghcr.io/pranavsingh-ml/chroma-lora:v0 bash scripts/pod.sh up
#   bash scripts/phase0.sh <POD_ID> [extra args for phase0_probe.py]
#
# Needs: tools/runpodctl.exe configured (runpodctl doctor - which also registers an SSH key).
set -euo pipefail
POD_ID="${1:?usage: phase0.sh <POD_ID> [--train-steps 20] [--skip-train]}"
cd "$(dirname "$0")/.."
RP="${RUNPODCTL:-./tools/runpodctl.exe}"
KEY="${SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
[ -f "$KEY" ] || KEY="$HOME/.ssh/imgedit_runpod"

echo "== ssh info for $POD_ID =="
read -r HOST PORT < <(python scripts/pod_ssh.py "$POD_ID")
echo "ssh root@$HOST -p $PORT"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$PORT" "root@$HOST")

echo "== environment =="
ENV='set -a; . <(tr "\0" "\n" < /proc/1/environ | grep -E "^(PATH|HF_|PYTORCH|API_TOKEN|MODEL_|EXTRAS_|AI_TOOLKIT_DIR|LORA_DIR|TRAIN_|TOKENIZERS)="); set +a;'
"${SSH[@]}" "$ENV"' nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; df -h / | tail -1; python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"; ls /opt/ai-toolkit/run.py && echo ai-toolkit present'

echo "== probe (downloads ~28GB of weights on first run; then trains a 20-step throwaway LoRA) =="
echo "   watch VRAM in another terminal:  ssh ... 'nvidia-smi --query-gpu=memory.used --format=csv -l 5'"
"${SSH[@]}" "$ENV"' cd /app && python phase0_probe.py' "${@:2}"

echo "== copy results back =="
mkdir -p phase0
for f in requirements.lock.txt phase0_report.json phase0_out.png phase0_out_2.png phase0_lora_out.png; do
  scp -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$PORT" \
    "root@$HOST:/app/$f" phase0/ 2>/dev/null || echo "  (no $f)"
done
ls -la phase0/
echo
echo "NEXT: look at phase0/phase0_out.png and phase0_lora_out.png, then:"
echo "  cp phase0/requirements.lock.txt server/requirements.lock.txt   # freeze it"
echo "  paste phase0/phase0_report.json values into NOTES.md (esp. lora_load_path)"
echo "  $RP pod delete $POD_ID                                         # TERMINATE THE POD"
