#!/usr/bin/env bash
# Generate images from the laptop, optionally with a trained LoRA. Saves to data/images/.
#
#   scripts/generate.sh "<prompt>" [lora-name] [scale]
#
# Env: POD_URL API_TOKEN (read from web/.env.local if unset), N STEPS GUIDANCE WIDTH HEIGHT SEED
#      SWEEP=1  -> same prompt+seed at LoRA scale 0.6/0.8/1.0/1.2 (identity strength sweep)
#      UPLOAD=1 -> upload data/loras/<lora-name>.safetensors first (needed after a fresh pod)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_py.sh"
cd "$(dirname "$0")/.."
PROMPT="${1:?usage: generate.sh \"<prompt>\" [lora-name] [scale]}"
LORA="${2:-}"
SCALE="${3:-0.9}"
POD_URL="${POD_URL:-$(grep -E '^POD_URL=' web/.env.local 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)}"
API_TOKEN="${API_TOKEN:-$(grep -E '^API_TOKEN=' web/.env.local 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)}"
: "${POD_URL:?set POD_URL}"; POD_URL="${POD_URL%/}"
: "${API_TOKEN:?set API_TOKEN}"
AUTH=(-H "Authorization: Bearer ${API_TOKEN}")
py() { "$PY" -c "import sys,json; d=json.load(sys.stdin); print($1)"; }
mkdir -p data/images

if [ -n "$LORA" ] && [ "${UPLOAD:-0}" = "1" ]; then
  F="data/loras/$LORA.safetensors"
  [ -f "$F" ] || { echo "no such file: $F"; exit 1; }
  echo "== uploading $F =="
  curl -sS "${AUTH[@]}" -X POST "$POD_URL/loras" -F "file=@$F" -F "name=$LORA"; echo
fi

run_one() {  # $1 = scale (or empty for no lora)
  local sc="$1" tag="$2"
  local body
  body="$("$PY" - "$PROMPT" "$LORA" "$sc" <<'PY'
import json, os, sys
prompt, lora, scale = sys.argv[1], sys.argv[2], sys.argv[3]
req = {"prompt": prompt, "n": int(os.environ.get("N", 1)),
       "steps": int(os.environ.get("STEPS", 26)), "guidance": float(os.environ.get("GUIDANCE", 4.0)),
       "width": int(os.environ.get("WIDTH", 1024)), "height": int(os.environ.get("HEIGHT", 1024)),
       "seed": int(os.environ.get("SEED", -1))}
if os.environ.get("NEGATIVE"): req["negative"] = os.environ["NEGATIVE"]
if lora and scale: req["loras"] = [{"name": lora, "scale": float(scale)}]
print(json.dumps(req))
PY
)"
  local R JOB ST
  R="$(curl -sS "${AUTH[@]}" -X POST "$POD_URL/generate" -H 'Content-Type: application/json' -d "$body")"
  echo "$R" | grep -q '"job_id"' || { echo "FAIL: $R"; return 1; }
  JOB="$(echo "$R" | py "d['job_id']")"
  echo "  job $JOB seeds=$(echo "$R" | py "d['seeds']") $tag"
  while true; do
    S="$(curl -sS "${AUTH[@]}" "$POD_URL/jobs/$JOB")"
    ST="$(echo "$S" | py "d['status']")"
    [ "$ST" = "done" ] && break
    [ "$ST" = "error" ] && { echo "  FAIL:"; echo "$S" | py "d['error']" | tail -5; return 1; }
    printf "\r  %s %.0f%%   " "$ST" "$(echo "$S" | py "d['progress']*100")"
    sleep 2
  done
  local n i out stamp
  n="$(echo "$S" | py "d['images']")"
  stamp="$(date +%Y%m%d-%H%M%S)"
  for i in $(seq 0 $((n - 1))); do
    out="data/images/${stamp}-${JOB}-${i}${tag:+-}${tag}.png"
    curl -sS "${AUTH[@]}" -o "$out" "$POD_URL/jobs/$JOB/image/$i"
    echo -e "\r  -> $out  (seed $(echo "$S" | py "d['seeds'][$i]"), $(echo "$S" | py "d['elapsed_s']")s)"
  done
  curl -sS "${AUTH[@]}" -X DELETE "$POD_URL/jobs/$JOB" >/dev/null
}

if [ "${SWEEP:-0}" = "1" ] && [ -n "$LORA" ]; then
  [ "${SEED:--1}" = "-1" ] && export SEED=$RANDOM
  echo "== strength sweep at seed $SEED =="
  run_one "" "lora-off"
  for s in 0.6 0.8 1.0 1.2; do run_one "$s" "s$s"; done
else
  run_one "${LORA:+$SCALE}" ""
fi
