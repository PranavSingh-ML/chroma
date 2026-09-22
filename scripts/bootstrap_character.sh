#!/usr/bin/env bash
# Build a training dataset for a FICTIONAL character by generating it with the base model,
# then curating by hand. This is the honest way to test whether human-identity training works
# without using a real person's photos.
#
#   scripts/bootstrap_character.sh <name> "<character description>" [count]
#
# <name>         dataset folder, e.g. mara
# <description>  ONE paragraph fixing the identity: face shape, eyes, hair, skin, build,
#                distinguishing marks, and explicitly an ADULT age (e.g. "a 28 year old woman").
#                Everything else (pose, clothing, setting, light) is varied by this script.
# [count]        how many candidates to generate (default 40; ~30 s each on an A40 => ~20 min, ~$0.17)
#
# Output: data/datasets/<name>/candidates/*.png  + description.txt + prompts.jsonl
# Then YOU curate: keep only the ones that look like the SAME person, move them up one level,
# write captions, and run scripts/train.sh. See "CURATION" printed at the end.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_py.sh"
cd "$(dirname "$0")/.."
NAME="${1:?usage: bootstrap_character.sh <name> \"<character description>\" [count]}"
DESC="${2:?need a character description - see the header of this script}"
COUNT="${3:-40}"
POD_URL="${POD_URL:-$(grep -E '^POD_URL=' web/.env.local 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)}"
API_TOKEN="${API_TOKEN:-$(grep -E '^API_TOKEN=' web/.env.local 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)}"
: "${POD_URL:?set POD_URL (or run scripts/pod.sh up first)}"; POD_URL="${POD_URL%/}"
: "${API_TOKEN:?set API_TOKEN}"
AUTH=(-H "Authorization: Bearer ${API_TOKEN}")
py() { "$PY" -c "import sys,json; d=json.load(sys.stdin); print($1)"; }

DS="data/datasets/$NAME"
OUT="$DS/candidates"
mkdir -p "$OUT"
printf '%s\n' "$DESC" > "$DS/description.txt"

# Varied framing / setting / light. Deliberately covers close-up through full body, because a LoRA
# trained only on face crops cannot do full-body shots (README, "Captions").
"$PY" - "$DESC" "$COUNT" "$DS/prompts.jsonl" <<'PY'
import itertools, json, random, sys
desc, count, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
framing = ["extreme close-up portrait, head and shoulders", "close-up portrait, head and shoulders",
           "medium shot from the waist up", "medium shot from the waist up, three-quarter angle",
           "full body shot, standing", "full body shot, seated", "half body shot, profile view",
           "medium close-up, looking over the shoulder"]
setting = ["in a plain studio against a neutral grey backdrop", "on a city street", "in a sunlit kitchen",
           "in a park on an overcast day", "against a white wall indoors", "in a cafe by a window",
           "on a balcony at dusk", "in a bedroom with soft daylight"]
light = ["soft diffused studio lighting", "warm golden hour sunlight", "flat overcast daylight",
         "hard directional window light", "cool blue evening light", "bright midday sun"]
expr = ["neutral expression", "a slight smile", "laughing", "looking directly at the camera",
        "looking away from the camera", "a thoughtful expression"]
lens = ["85mm portrait lens, shallow depth of field", "35mm lens", "50mm lens, natural perspective"]
rnd = random.Random(1234)
combos = list(itertools.product(framing, setting, light))
rnd.shuffle(combos)
rows = []
for i in range(count):
    f, s, l = combos[i % len(combos)]
    # `varied` is everything that is NOT identity. caption_helper.sh uses exactly this field, so
    # face details can never leak into a caption (the description itself contains commas, so
    # splitting the prompt on a comma is not good enough).
    varied = f"{f}, {s}, {l}, {rnd.choice(expr)}, {rnd.choice(lens)}"
    p = f"photo of {desc}, {varied}, photorealistic, natural skin texture, sharp focus"
    rows.append({"i": i, "prompt": p, "varied": varied, "seed": 100000 + i * 17})
with open(out, "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(f"  {len(rows)} prompts -> {out}")
PY

echo "== generating $COUNT candidates (Ctrl-C is safe; already-saved files are kept) =="
N_OK=0
while read -r line; do
  I=$(echo "$line" | py "d['i']"); P=$(echo "$line" | py "d['prompt']"); S=$(echo "$line" | py "d['seed']")
  F="$OUT/$(printf '%03d' "$I").png"
  [ -f "$F" ] && { echo "  [$I] exists, skipping"; N_OK=$((N_OK+1)); continue; }
  BODY=$("$PY" - "$P" "$S" <<'PY'
import json, os, sys
print(json.dumps({"prompt": sys.argv[1], "seed": int(sys.argv[2]),
                  "steps": int(os.environ.get("STEPS", 26)), "guidance": float(os.environ.get("GUIDANCE", 4.0)),
                  "width": int(os.environ.get("WIDTH", 832)), "height": int(os.environ.get("HEIGHT", 1216))}))
PY
)
  R=$(curl -sS "${AUTH[@]}" -X POST "$POD_URL/generate" -H 'Content-Type: application/json' -d "$BODY")
  echo "$R" | grep -q '"job_id"' || { echo "  [$I] FAIL: $R"; continue; }
  J=$(echo "$R" | py "d['job_id']")
  while true; do
    ST=$(curl -sS "${AUTH[@]}" "$POD_URL/jobs/$J" | py "d['status']")
    [ "$ST" = "done" ] && break
    [ "$ST" = "error" ] && { echo "  [$I] ERROR"; break; }
    sleep 2
  done
  [ "$ST" = "done" ] || continue
  curl -sS "${AUTH[@]}" -o "$F" "$POD_URL/jobs/$J/image/0"
  curl -sS "${AUTH[@]}" -X DELETE "$POD_URL/jobs/$J" >/dev/null
  N_OK=$((N_OK+1))
  echo "  [$I] -> $F"
done < "$DS/prompts.jsonl"

cat <<EOF

== $N_OK candidates in $OUT

CURATION - this step decides your LoRA quality, nothing else comes close:
  1. Open $OUT and delete every image that is NOT recognisably the same person,
     plus anything with mangled hands/eyes/teeth. Being ruthless here is correct.
  2. Keep 15-25 survivors. They must span close-up / half / full body - if they are all
     head shots, the LoRA will not be able to do full-body shots.
  3. Move the survivors up one level:   mv $OUT/*.png $DS/
  4. Caption them:                      scripts/caption_helper.sh $DS <trigger>
     then edit each .txt: describe pose, clothing, background, lighting - NEVER the face.
  5. Train:                             scripts/train.sh $DS ${NAME}-v1 <trigger>

Then judge quality: generate with the LoRA at 0.8-1.0 and see whether the face holds across
prompts it never saw. That is the test that actually answers "does human identity training work".
EOF
