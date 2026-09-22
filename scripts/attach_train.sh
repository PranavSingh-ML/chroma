#!/usr/bin/env bash
# Attach to a training job that is ALREADY running on the pod: stream progress, pull sample
# previews as they appear, and download the finished LoRA + sidecar. Use this when the shell
# that started `train.sh` died, or to follow a job from a second terminal.
#
#   scripts/attach_train.sh <job-id> <lora-name> [dataset-dir]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_py.sh"
cd "$(dirname "$0")/.."
JOB="${1:?usage: attach_train.sh <job-id> <lora-name> [dataset-dir]}"
NAME="${2:?usage: attach_train.sh <job-id> <lora-name> [dataset-dir]}"
DS="${3:-}"
POD_URL="${POD_URL:-$(grep -E '^POD_URL=' web/.env.local 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)}"
API_TOKEN="${API_TOKEN:-$(grep -E '^API_TOKEN=' web/.env.local 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)}"
: "${POD_URL:?set POD_URL}"; POD_URL="${POD_URL%/}"
: "${API_TOKEN:?set API_TOKEN}"
AUTH=(-H "Authorization: Bearer ${API_TOKEN}")
py() { "$PY" -c "import sys,json; d=json.load(sys.stdin); print($1)"; }
RUNDIR="data/runs/$JOB"
mkdir -p "$RUNDIR/samples" data/loras

SEEN=""
while true; do
  S="$(curl -sS --max-time 30 "${AUTH[@]}" "$POD_URL/jobs/$JOB" || true)"
  [ -n "$S" ] || { echo "  (no response, retrying)"; sleep 15; continue; }
  ST="$(echo "$S" | py "d['status']")"
  printf "  %s  step %s/%s  loss %s  eta %s min\n" "$ST" "$(echo "$S" | py "d.get('step')")" \
    "$(echo "$S" | py "d.get('total_steps')")" "$(echo "$S" | py "d.get('loss')")" \
    "$("$PY" -c "print(round($(echo "$S" | py "d.get('eta_s') or 0")/60))")"
  for s in $(echo "$S" | py "' '.join(d.get('samples') or [])"); do
    case " $SEEN " in *" $s "*) ;; *)
      curl -sS "${AUTH[@]}" -o "$RUNDIR/samples/$s" "$POD_URL/jobs/$JOB/samples/$s" && echo "    sample -> $RUNDIR/samples/$s"
      SEEN="$SEEN $s" ;;
    esac
  done
  [ "$ST" = "done" ] && break
  [ "$ST" = "error" ] && { echo "FAIL:"; echo "$S" | py "d['error']"; curl -sS "${AUTH[@]}" "$POD_URL/jobs/$JOB/log" | tail -30; exit 1; }
  sleep 20
done

echo "== download the LoRA =="
curl -sS "${AUTH[@]}" -o "data/loras/$NAME.safetensors" "$POD_URL/jobs/$JOB/artifact"
SIZE=$("$PY" -c "import os;print(round(os.path.getsize('data/loras/$NAME.safetensors')/1e6,1))")
"$PY" - "$NAME" "$DS" "$S" <<'PY'
import json, os, sys, time
name, ds, status = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
side = {"name": name, "dataset": os.path.abspath(ds) if ds else None, "server": status,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "file": f"data/loras/{name}.safetensors"}
with open(f"data/loras/{name}.json", "w", encoding="utf-8") as f:
    json.dump(side, f, indent=2)
PY
curl -sS "${AUTH[@]}" "$POD_URL/jobs/$JOB/log" > "$RUNDIR/train.log" || true
echo "PASS: data/loras/$NAME.safetensors ($SIZE MB) + .json sidecar"
echo "      samples: $RUNDIR/samples/    log: $RUNDIR/train.log"
