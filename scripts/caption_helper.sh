#!/usr/bin/env bash
# Write a starter .txt caption next to every image in a dataset folder, so you only have to EDIT
# rather than type from scratch. Never overwrites an existing caption.
#
#   scripts/caption_helper.sh <dataset-dir> <trigger-word>
#
# For a bootstrapped character the prompt that produced each image is in prompts.jsonl, so the
# framing/setting/lighting are recovered from there. For your own photos you get a template.
#
# THEN EDIT THEM. The rules that matter:
#   - start with the trigger word
#   - describe pose, framing, clothing, background, lighting, expression
#   - NEVER describe the face (shape, eyes, hair colour) - that is what the LoRA must absorb
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_py.sh"
DS="${1:?usage: caption_helper.sh <dataset-dir> <trigger-word>}"
TRIGGER="${2:?need a trigger word, e.g. pr4nv}"
[ -d "$DS" ] || { echo "no such dir: $DS"; exit 1; }

"$PY" - "$DS" "$TRIGGER" <<'PY'
import json, os, sys
ds, trigger = sys.argv[1], sys.argv[2]
# Recover the NON-IDENTITY half of each generation prompt, if this came from
# bootstrap_character.sh. `varied` is stored separately on purpose: the character description
# contains commas, so slicing the full prompt would leak face details into the caption.
byidx = {}
pj = os.path.join(ds, "prompts.jsonl")
if os.path.exists(pj):
    for line in open(pj, encoding="utf-8"):
        r = json.loads(line)
        if "varied" in r:
            byidx[f"{r['i']:03d}"] = r["varied"]

made = skipped = 0
for f in sorted(os.listdir(ds)):
    stem, ext = os.path.splitext(f)
    if ext.lower() not in (".jpg", ".jpeg", ".png"):
        continue
    cap = os.path.join(ds, stem + ".txt")
    if os.path.exists(cap):
        skipped += 1
        continue
    if stem in byidx:
        body = byidx[stem]
    else:
        body = "FRAMING, CLOTHING, BACKGROUND, LIGHTING  <- edit me; do not describe the face"
    open(cap, "w", encoding="utf-8").write(f"{trigger} person, {body}\n")
    made += 1
print(f"  wrote {made} captions, left {skipped} existing ones alone")
PY

echo
echo "Now EDIT the .txt files in $DS - the auto text is a starting point, not a caption."
echo "Check them:   grep -H . $DS/*.txt | head -30"
echo "Then train:   scripts/train.sh $DS <name>-v1 $TRIGGER"
