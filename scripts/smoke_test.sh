#!/usr/bin/env bash
# Phase 1 acceptance test, part 1: generation. Runs on the LAPTOP against a live pod, cold,
# with nobody SSHing in.
#   POD_URL=https://<POD_ID>-8000.proxy.runpod.net API_TOKEN=... scripts/smoke_test.sh
# Needs: curl + a python3 (auto-detected by _py.sh). Exit 0 = pass.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_py.sh"
POD_URL="${POD_URL:?set POD_URL}"; POD_URL="${POD_URL%/}"
API_TOKEN="${API_TOKEN:?set API_TOKEN}"
OUT="${OUT:-smoke_out.png}"
STEPS="${STEPS:-26}"
AUTH=(-H "Authorization: Bearer ${API_TOKEN}")
py() { "$PY" -c "import sys,json; d=json.load(sys.stdin); print($1)"; }

echo "== 1. /health (no auth) - waiting for model_loaded=true (cold boot can take 5-15 min) =="
for i in $(seq 1 180); do
  H="$(curl -sS --max-time 20 "$POD_URL/health" || true)"
  if [ -n "$H" ]; then
    LOADED="$(echo "$H" | py "d.get('model_loaded')")"
    ERR="$(echo "$H" | py "(d.get('load_error') or '')" | head -c 300)"
    echo "  [$i] model_loaded=$LOADED gpu=$(echo "$H" | py "d.get('gpu')") cap=$(echo "$H" | py "d.get('capability')") vram=$(echo "$H" | py "d.get('vram_used_gb')")GB busy=$(echo "$H" | py "d.get('busy')")"
    [ -n "$ERR" ] && { echo "MODEL LOAD ERROR: $ERR"; exit 1; }
    [ "$LOADED" = "True" ] && break
  else
    echo "  [$i] no response yet"
  fi
  sleep 10
done
[ "${LOADED:-}" = "True" ] || { echo "FAIL: model never loaded"; exit 1; }

echo "== 2. auth check: /generate without token must be 401 =="
CODE="$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$POD_URL/generate" -H 'Content-Type: application/json' -d '{"prompt":"x"}')"
[ "$CODE" = "401" ] || { echo "FAIL: expected 401 got $CODE"; exit 1; }
echo "  ok (401)"

echo "== 3. guard check: a prompt naming a minor must be 422 =="
CODE="$(curl -sS -o /dev/null -w '%{http_code}' "${AUTH[@]}" -X POST "$POD_URL/generate" -H 'Content-Type: application/json' -d '{"prompt":"a 12 year old child on a swing"}')"
[ "$CODE" = "422" ] || { echo "FAIL: expected 422 got $CODE"; exit 1; }
echo "  ok (422)"

echo "== 4. POST /generate =="
T0=$(date +%s.%N)
R="$(curl -sS "${AUTH[@]}" -X POST "$POD_URL/generate" -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"photo of a woman standing in a city street at golden hour, 35mm, shallow depth of field, natural skin texture\",\"seed\":42,\"steps\":$STEPS,\"width\":1024,\"height\":1024}")"
echo "  $R"
JOB="$(echo "$R" | py "d['job_id']")"

echo "== 5. poll /jobs/$JOB =="
for i in $(seq 1 300); do
  S="$(curl -sS "${AUTH[@]}" "$POD_URL/jobs/$JOB")"
  ST="$(echo "$S" | py "d['status']")"
  echo "  status=$ST progress=$(echo "$S" | py "d['progress']") elapsed=$(echo "$S" | py "d['elapsed_s']")s"
  [ "$ST" = "done" ] && break
  [ "$ST" = "error" ] && { echo "FAIL:"; echo "$S" | py "d['error']"; exit 1; }
  sleep 2
done
[ "$ST" = "done" ] || { echo "FAIL: timed out"; exit 1; }

echo "== 6. download PNG =="
curl -sS "${AUTH[@]}" -o "$OUT" "$POD_URL/jobs/$JOB/image/0"
T1=$(date +%s.%N)
"$PY" -c "import sys; from PIL import Image; im=Image.open(sys.argv[1]); print('  valid PNG', im.size)" "$OUT"
curl -sS "${AUTH[@]}" -X DELETE "$POD_URL/jobs/$JOB" >/dev/null
echo "PASS: $OUT  seeds=$(echo "$S" | py "d['seeds']")  server_elapsed=$(echo "$S" | py "d['elapsed_s']")s  wall=$("$PY" -c "print(round($T1-$T0,1))")s"
echo "Record server_elapsed and cold-boot time in NOTES.md. Next: scripts/train_smoke.sh"
