#!/usr/bin/env bash
# Auto-caption a dataset using a vision model running ON YOUR POD, then write the .txt files
# locally. Your photos go from your laptop to your own pod and nowhere else.
#
#   scripts/caption.sh <dataset-dir> <trigger-word> [--overwrite]
#
# The captions describe framing, pose, clothing, background and lighting, and deliberately never
# describe the face - whatever the caption leaves out is what the LoRA learns from the trigger.
# Read them afterwards and fix anything wrong; a wrong caption is worse than a terse one.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_py.sh"
cd "$(dirname "$0")/.."
DS="${1:?usage: caption.sh <dataset-dir> <trigger-word> [--overwrite]}"
TRIGGER="${2:?need a trigger word, e.g. pr4nv}"
OVERWRITE="${3:-}"
POD_URL="${POD_URL:-$(grep -E '^POD_URL=' web/.env.local 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)}"
API_TOKEN="${API_TOKEN:-$(grep -E '^API_TOKEN=' web/.env.local 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)}"
: "${POD_URL:?set POD_URL (or run scripts/pod.sh up first)}"; POD_URL="${POD_URL%/}"
: "${API_TOKEN:?set API_TOKEN}"
AUTH=(-H "Authorization: Bearer ${API_TOKEN}")
py() { "$PY" -c "import sys,json; d=json.load(sys.stdin); print($1)"; }
[ -d "$DS" ] || { echo "no such dir: $DS"; exit 1; }

ZIP="$(mktemp -t cap_XXXX).zip"
"$PY" - "$DS" "$ZIP" "$OVERWRITE" <<'PY'
import os, sys, zipfile
src, dst, ow = sys.argv[1], sys.argv[2], sys.argv[3]
n = 0
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(src)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")) or f.startswith("."):
            continue
        if ow != "--overwrite" and os.path.exists(os.path.join(src, os.path.splitext(f)[0] + ".txt")):
            continue                      # already captioned; don't pay to redo it
        z.write(os.path.join(src, f), f)
        n += 1
print(f"  {n} image(s) to caption")
if n == 0:
    raise SystemExit("nothing to do - pass --overwrite to redo existing captions")
PY

echo "== POST /caption (first run also downloads the ~9 GB vision model, a few minutes) =="
R="$(curl -sS "${AUTH[@]}" -X POST "$POD_URL/caption" -F "dataset=@$ZIP" -F "trigger=$TRIGGER")"
rm -f "$ZIP"
echo "  $R"
echo "$R" | grep -q '"job_id"' || { echo "FAIL"; exit 1; }
JOB="$(echo "$R" | py "d['job_id']")"
N="$(echo "$R" | py "d['n_images']")"

while true; do
  S="$(curl -sS "${AUTH[@]}" "$POD_URL/jobs/$JOB")"
  ST="$(echo "$S" | py "d['status']")"
  printf "\r  %s  %s/%s captioned   " "$ST" "$(echo "$S" | py "d.get('done',0)")" "$N"
  [ "$ST" = "done" ] && { echo; break; }
  [ "$ST" = "error" ] && { echo; echo "FAIL:"; echo "$S" | py "d['error']" | tail -5; exit 1; }
  sleep 3
done

CAPS="$(mktemp -t caps_XXXX).json"
curl -sS "${AUTH[@]}" -o "$CAPS" "$POD_URL/jobs/$JOB/captions"
"$PY" - "$DS" "$CAPS" <<'PY'
import json, os, sys
ds, capfile = sys.argv[1], sys.argv[2]
caps = json.load(open(capfile, encoding="utf-8"))
for fname, text in sorted(caps.items()):
    out = os.path.join(ds, os.path.splitext(fname)[0] + ".txt")
    open(out, "w", encoding="utf-8").write(text.strip() + "\n")
    print(f"  {os.path.basename(out):12s} {text[:88]}")
print(f"\n  wrote {len(caps)} captions into {ds}")
PY
rm -f "$CAPS"
curl -sS "${AUTH[@]}" -X DELETE "$POD_URL/jobs/$JOB" >/dev/null

cat <<EOF

READ THEM before training - the model gets things wrong, and a wrong caption is worse than a
terse one. Check especially that none of them describe your face:
  grep -inE "eyes|hair|jaw|skin|beard|face" $DS/*.txt   # should print nothing

Then train:  scripts/train.sh $DS <name>-v1 $TRIGGER
EOF
