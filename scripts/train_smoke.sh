#!/usr/bin/env bash
# Phase 1 acceptance test, part 2: the training loop. Makes a 4-image synthetic dataset, trains a
# 20-step throwaway LoRA, downloads it, and generates one image with it. ~5 min of pod time.
#   POD_URL=... API_TOKEN=... scripts/train_smoke.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_py.sh"
cd "$(dirname "$0")/.."
POD_URL="${POD_URL:-$(grep -E '^POD_URL=' web/.env.local 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)}"
API_TOKEN="${API_TOKEN:-$(grep -E '^API_TOKEN=' web/.env.local 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)}"
: "${POD_URL:?set POD_URL}"; export POD_URL API_TOKEN
NAME="${NAME:-smoke-$(date +%H%M%S)}"
DS="$(mktemp -d -t smokeds_XXXX)"

"$PY" - "$DS" <<'PY'
import os, sys
from PIL import Image, ImageDraw
dest = sys.argv[1]
for i, c in enumerate([(180,70,60),(60,120,180),(90,160,90),(170,140,60)]):
    im = Image.new("RGB", (768, 1024), c); d = ImageDraw.Draw(im)
    d.ellipse([234,150,534,450], fill=(240,210,180)); d.rectangle([284,450,484,900], fill=(40,40,60))
    d.text((20,20), f"smoke subject {i}", fill=(255,255,255))
    im.save(os.path.join(dest, f"{i:03d}.jpg"), quality=92)
    open(os.path.join(dest, f"{i:03d}.txt"), "w").write(f"sm0ke person, full body, plain background, test image {i}")
print("  synthetic dataset in", dest)
PY

echo "== train 20 steps =="
STEPS=20 SAVE_EVERY=20 SAMPLE_EVERY=20 RESOLUTIONS=512 \
  SAMPLE_PROMPTS='["photo of sm0ke person, plain background"]' \
  bash scripts/train.sh "$DS" "$NAME" sm0ke

echo "== /loras must list it =="
curl -sS -H "Authorization: Bearer $API_TOKEN" "$POD_URL/loras" | "$PY" -c "
import sys,json; ls=json.load(sys.stdin)
names=[l['name'] for l in ls]; print(' ', ls)
assert '$NAME' in names, 'FAIL: $NAME not in /loras'"

echo "== generate with it =="
OUT_BEFORE=$(ls data/images 2>/dev/null | wc -l)
STEPS=8 WIDTH=768 HEIGHT=768 SEED=42 bash scripts/generate.sh "photo of sm0ke person, plain background" "$NAME" 1.0
OUT_AFTER=$(ls data/images | wc -l)
[ "$OUT_AFTER" -gt "$OUT_BEFORE" ] || { echo "FAIL: no image produced"; exit 1; }

echo "== cleanup =="
curl -sS -H "Authorization: Bearer $API_TOKEN" -X DELETE "$POD_URL/loras/$NAME" ; echo
rm -rf "$DS" "data/loras/$NAME.safetensors" "data/loras/$NAME.json"
echo "PASS: train -> artifact -> upload/registry -> generate-with-lora all work."
echo "Record train seconds and s/step in NOTES.md."
