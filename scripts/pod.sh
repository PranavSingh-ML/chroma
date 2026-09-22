#!/usr/bin/env bash
# One-command pod lifecycle for the chroma-lora pod. Needs only bash (Git Bash) + node.
#
#   bash scripts/pod.sh up        create a pod from IMAGE (A40 first, A6000 fallback), write POD_URL to web/.env.local
#   bash scripts/pod.sh status    list pods + $/hr + balance
#   bash scripts/pod.sh down      DELETE every pod named chroma* (this is the only thing that stops billing)
#   bash scripts/pod.sh wait      block until /health reports model_loaded=true
#
# PHASE0=1 bash scripts/pod.sh up   creates the Phase-0 pod (sshd only, no server).
# Needs tools/runpodctl.exe configured (tools\runpodctl.exe doctor).
set -euo pipefail
cd "$(dirname "$0")/.."
RP="${RUNPODCTL:-./tools/runpodctl.exe}"
[ -x "$RP" ] || RP="./tools/runpodctl"
[ -x "$RP" ] || { echo "runpodctl not found in tools/ - see README (download it, then: tools\\runpodctl.exe doctor)"; exit 1; }
# Find a Node >= 22 even if an older node is first on bash's PATH (common on Windows).
NODE="${NODE:-}"
PS_NODE=""
if command -v powershell.exe >/dev/null 2>&1; then
  PS_NODE="$(powershell.exe -c '(Get-Command node -ErrorAction SilentlyContinue).Source' 2>/dev/null | tr -d '\r' | head -1 || true)"
  [ -n "$PS_NODE" ] && command -v cygpath >/dev/null 2>&1 && PS_NODE="$(cygpath -u "$PS_NODE")"
fi
for c in ${NODE:+"$NODE"} ${PS_NODE:+"$PS_NODE"} node "/c/Program Files/nodejs/node.exe" "$HOME/AppData/Roaming/fnm/aliases/default/node.exe" "$HOME/AppData/Roaming/nvm/current/node.exe"; do
  if command -v "$c" >/dev/null 2>&1; then
    v="$("$c" -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
    if [ "${v:-0}" -ge 22 ]; then NODE="$c"; break; fi
  fi
done
[ -n "$NODE" ] || { echo "Need Node >= 22 (24 recommended). Install it from nodejs.org and open a new terminal."; exit 1; }
IMAGE="${IMAGE:-ghcr.io/pranavsingh-ml/chroma-lora:v1}"
DISK="${DISK:-80}"
ENVF="web/.env.local"
NAME="${NAME:-chroma}"
REG_AUTH="${REGISTRY_AUTH_ID:-}"   # set if the GHCR package is private (runpodctl registry list)

js() { "$NODE" -e "let input='';process.stdin.on('data',d=>input+=d).on('end',()=>{ $1 })"; }
token() { grep -E '^API_TOKEN=' "$ENVF" 2>/dev/null | cut -d= -f2- | tr -d '\r' || true; }
pods_json() { "$RP" pod list -o json 2>/dev/null || echo "[]"; }
chroma_ids() { pods_json | js 'const ps=JSON.parse(input||"[]"); console.log(ps.filter(p=>(p.name||"").startsWith("chroma")).map(p=>p.id).join(" "))'; }

case "${1:-}" in
  up)
    mkdir -p web
    TOKEN="$(token)"
    if [ -z "$TOKEN" ]; then
      TOKEN="$("$NODE" -e 'console.log(require("crypto").randomBytes(32).toString("hex"))')"
      printf 'POD_URL=\nAPI_TOKEN=%s\nPOD_RATE_USD_HR=0.49\n' "$TOKEN" > "$ENVF"
      echo "generated API_TOKEN -> $ENVF"
    fi
    EXISTING="$(chroma_ids)"
    if [ -n "$EXISTING" ]; then echo "a pod already exists: $EXISTING  (use: pod.sh status / wait / down)"; exit 1; fi
    ENV_JSON="{\"API_TOKEN\":\"$TOKEN\",\"HF_HOME\":\"/workspace/hf\"${PHASE0:+,\"PHASE0\":\"1\"}${EXTRA_ENV:+,$EXTRA_ENV}}"
    try() {
      echo "== trying $1 ($2)"
      "$RP" pod create --name "$NAME" --image "$IMAGE" --gpu-id "$1" --cloud-type "$2" \
        --container-disk-in-gb "$DISK" --ports "8000/http,22/tcp" ${REG_AUTH:+--registry-auth-id "$REG_AUTH"} \
        --env "$ENV_JSON" -o json > .pod_create.json 2>&1 || true
      if grep -q '"error"' .pod_create.json; then js 'console.log("  ", (JSON.parse(input).error||"").slice(0,120))' < .pod_create.json; return 1; fi
      grep -q '"id"' .pod_create.json || { echo "  unexpected response:"; head -c 300 .pod_create.json; echo; return 1; }
    }
    # Community first: the SAME silicon at ~38% less ($0.33-0.35/hr vs $0.49-0.53 Secure).
    # A 1600-step run measured 3.3 h, so that difference is ~$0.65 every single time.
    # Set CLOUD=SECURE to force Secure (marginally better host reliability).
    if [ "${CLOUD:-}" = "SECURE" ]; then
      try "NVIDIA A40" SECURE || try "NVIDIA RTX A6000" SECURE \
        || { echo "no 48GB Ampere card on Secure right now; retry, or unset CLOUD to allow Community"; exit 1; }
    else
      try "NVIDIA A40" COMMUNITY || try "NVIDIA RTX A6000" COMMUNITY \
        || try "NVIDIA A40" SECURE || try "NVIDIA RTX A6000" SECURE \
        || { echo "no 48GB Ampere card available right now; try again in a few minutes"; exit 1; }
    fi
    ID="$(js 'console.log(JSON.parse(input).id)' < .pod_create.json)"
    RATE="$(js 'console.log(JSON.parse(input).costPerHr || "")' < .pod_create.json)"
    rm -f .pod_create.json
    URL="https://${ID}-8000.proxy.runpod.net"
    sed -i "s|^POD_URL=.*|POD_URL=${URL}|; s|^POD_RATE_USD_HR=.*|POD_RATE_USD_HR=${RATE:-0.49}|" "$ENVF"
    echo "created pod $ID at \$${RATE}/hr"
    if [ -n "${PHASE0:-}" ]; then
      echo "PHASE0 pod (sshd only, no server). Next:  bash scripts/phase0.sh $ID"
    else
      echo "POD_URL=$URL  (written to $ENVF)"
      echo "cold boot ~6-12 min; run:  bash scripts/pod.sh wait"
    fi
    echo "WHEN DONE:  bash scripts/pod.sh down"
    ;;
  status)
    pods_json | js '
      const ps=JSON.parse(input||"[]");
      if(!ps.length) console.log("no pods running - not being billed");
      for(const p of ps) console.log(`${p.id}  ${p.name}  ${p.desiredStatus}  $${p.costPerHr}/hr  up ${Math.floor((p.uptimeSeconds||0)/60)} min  https://${p.id}-8000.proxy.runpod.net`);'
    "$RP" user -o json 2>/dev/null | js 'const u=JSON.parse(input||"{}"); if(u.clientBalance!==undefined) console.log(`balance $${u.clientBalance.toFixed(2)}   spend/hr $${u.currentSpendPerHr}`)'
    ;;
  down)
    IDS="$(chroma_ids)"
    [ -n "$IDS" ] || { echo "no chroma pods to delete"; exit 0; }
    for id in $IDS; do "$RP" pod delete "$id" -o json >/dev/null && echo "deleted $id"; done
    LEFT="$(pods_json | js 'console.log(JSON.parse(input||"[]").length)')"
    echo "pods still running (any name): $LEFT"
    ;;
  wait)
    URL="$(grep -E '^POD_URL=' "$ENVF" 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)"
    [ -n "$URL" ] || { echo "POD_URL not set in $ENVF - run: bash scripts/pod.sh up"; exit 1; }
    for i in $(seq 1 120); do
      H="$(curl -s --max-time 15 "$URL/health" || true)"
      if [ -n "$H" ]; then
        L="$(echo "$H" | js 'const d=JSON.parse(input); console.log(d.model_loaded, d.gpu, d.vram_used_gb, (d.load_error||"").slice(0,200))')"
        echo "[$i] $L"
        echo "$L" | grep -q '^true' && { echo "READY: $URL"; exit 0; }
        echo "$L" | grep -q 'Traceback\|Error' && { echo "MODEL LOAD FAILED - run: bash scripts/pod.sh down"; exit 1; }
      else echo "[$i] booting..."; fi
      sleep 10
    done
    echo "gave up after 20 min - check: bash scripts/pod.sh status"; exit 1
    ;;
  *) sed -n 2,10p "$0"; exit 1 ;;
esac
