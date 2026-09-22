#!/usr/bin/env bash
# Train a character LoRA from the laptop. This is the whole Phase-2 workflow without the web app:
# zip the dataset -> POST /train -> poll -> pull samples + the finished LoRA into data/loras/.
#
#   POD_URL=... API_TOKEN=... scripts/train.sh <dataset-dir> <lora-name> [trigger-word]
#
# <dataset-dir>   folder of .jpg/.jpeg/.png + a same-name .txt caption for each
# <lora-name>     e.g. pranav-v1  (must not already exist on the pod)
# [trigger-word]  e.g. pr4nv      (added to captions that lack it; use [trigger] in sample prompts)
#
# Knobs (env): STEPS RANK LR RESOLUTIONS SAMPLE_EVERY SAVE_EVERY SAMPLE_PROMPTS (JSON array)
# The LoRA lands in data/loras/<name>.safetensors with a .json sidecar. The pod keeps NOTHING.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_py.sh"
cd "$(dirname "$0")/.."
DS="${1:?usage: train.sh <dataset-dir> <lora-name> [trigger-word]}"
NAME="${2:?usage: train.sh <dataset-dir> <lora-name> [trigger-word]}"
TRIGGER="${3:-}"
POD_URL="${POD_URL:-$(grep -E '^POD_URL=' web/.env.local 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)}"
API_TOKEN="${API_TOKEN:-$(grep -E '^API_TOKEN=' web/.env.local 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)}"
: "${POD_URL:?set POD_URL (or run scripts/pod.sh up first)}"; POD_URL="${POD_URL%/}"
: "${API_TOKEN:?set API_TOKEN}"
AUTH=(-H "Authorization: Bearer ${API_TOKEN}")
py() { "$PY" -c "import sys,json; d=json.load(sys.stdin); print($1)"; }
[ -d "$DS" ] || { echo "no such dataset dir: $DS"; exit 1; }

N_IMG="$(find "$DS" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' ')"
N_TXT="$(find "$DS" -maxdepth 1 -type f -iname '*.txt' | wc -l | tr -d ' ')"
[ "$N_IMG" -gt 0 ] || { echo "no images in $DS (only .jpg/.jpeg/.png are supported)"; exit 1; }
echo "== dataset: $N_IMG images, $N_TXT captions in $DS"
[ "$N_TXT" -lt "$N_IMG" ] && echo "   WARNING: $((N_IMG - N_TXT)) images have no .txt caption - they will train on the trigger word alone"

CFG="$("$PY" - "$TRIGGER" <<'PY'
import json, os, sys
trigger = sys.argv[1]
cfg = {"trigger_word": trigger}
for env, key, cast in [("STEPS","steps",int), ("RANK","rank",int), ("LR","lr",float),
                       ("SAMPLE_EVERY","sample_every",int), ("SAVE_EVERY","save_every",int)]:
    v = os.environ.get(env)
    if v: cfg[key] = cast(v)
if os.environ.get("RESOLUTIONS"):
    cfg["resolutions"] = [int(x) for x in os.environ["RESOLUTIONS"].replace(","," ").split()]
if os.environ.get("SAMPLE_PROMPTS"):
    cfg["sample_prompts"] = json.loads(os.environ["SAMPLE_PROMPTS"])
elif trigger:
    cfg["sample_prompts"] = [
        f"photo of {trigger} person, medium shot, city street, natural light",
        f"close-up portrait of {trigger} person, studio lighting, neutral background",
        f"photo of {trigger} person, full body, standing in a park, overcast day",
        f"photo of {trigger} person, sitting at a cafe, window light, candid",
    ]
print(json.dumps(cfg))
PY
)"
echo "== config: $CFG"

ZIP="$(mktemp -t ds_XXXX).zip"
"$PY" - "$DS" "$ZIP" <<'PY'
import os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
exts = (".jpg", ".jpeg", ".png", ".txt")
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(src)):
        if f.lower().endswith(exts) and not f.startswith("."):
            z.write(os.path.join(src, f), f)
print("  zipped", os.path.getsize(dst) // 1024, "KB")
PY

echo "== POST /train =="
R="$(curl -sS "${AUTH[@]}" -X POST "$POD_URL/train" -F "dataset=@$ZIP" -F "name=$NAME" -F "config=$CFG")"
rm -f "$ZIP"
echo "  $R"
echo "$R" | grep -q '"job_id"' || { echo "FAIL (see above)"; exit 1; }
JOB="$(echo "$R" | py "d['job_id']")"
TOTAL="$(echo "$R" | py "d['config']['steps']")"
RUNDIR="data/runs/$JOB"
mkdir -p "$RUNDIR/samples" data/loras
echo "$R" > "$RUNDIR/request.json"

echo "== training job $JOB ($TOTAL steps) - Ctrl-C here does NOT stop the pod; it keeps training =="
SEEN=""
while true; do
  S="$(curl -sS "${AUTH[@]}" "$POD_URL/jobs/$JOB" || true)"
  [ -n "$S" ] || { echo "  (no response, retrying)"; sleep 10; continue; }
  ST="$(echo "$S" | py "d['status']")"
  STEP="$(echo "$S" | py "d.get('step')")"; LOSS="$(echo "$S" | py "d.get('loss')")"
  ETA="$(echo "$S" | py "d.get('eta_s') or 0")"
  printf "  %s  step %s/%s  loss %s  eta %s min\n" "$ST" "$STEP" "$TOTAL" "$LOSS" "$("$PY" -c "print(round($ETA/60))")"
  # pull any new sample previews
  for s in $(echo "$S" | py "' '.join(d.get('samples') or [])"); do
    case " $SEEN " in *" $s "*) ;; *)
      curl -sS "${AUTH[@]}" -o "$RUNDIR/samples/$s" "$POD_URL/jobs/$JOB/samples/$s" && echo "    sample -> $RUNDIR/samples/$s"
      SEEN="$SEEN $s" ;;
    esac
  done
  [ "$ST" = "done" ] && break
  [ "$ST" = "error" ] && { echo "FAIL:"; echo "$S" | py "d['error']"; curl -sS "${AUTH[@]}" "$POD_URL/jobs/$JOB/log" | tail -30; exit 1; }
  sleep 15
done

echo "== download the LoRA =="
curl -sS "${AUTH[@]}" -o "data/loras/$NAME.safetensors" "$POD_URL/jobs/$JOB/artifact"
SIZE=$("$PY" -c "import os;print(round(os.path.getsize('data/loras/$NAME.safetensors')/1e6,1))")
"$PY" - "$NAME" "$DS" "$CFG" "$S" <<'PY'
import json, os, sys, time
name, ds, cfg, status = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4])
side = {"name": name, "dataset": os.path.abspath(ds), "config": cfg, "server": status,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "file": f"data/loras/{name}.safetensors"}
with open(f"data/loras/{name}.json", "w", encoding="utf-8") as f:
    json.dump(side, f, indent=2)
PY
curl -sS "${AUTH[@]}" "$POD_URL/jobs/$JOB/log" > "$RUNDIR/train.log" || true
echo "PASS: data/loras/$NAME.safetensors ($SIZE MB) + .json sidecar"
echo "      samples: $RUNDIR/samples/    log: $RUNDIR/train.log"
echo
echo "The LoRA is already loaded on the pod. Generate with it:"
echo "  POD_URL=$POD_URL API_TOKEN=... scripts/generate.sh \"photo of ${TRIGGER:-<trigger>} person, ...\" $NAME"
echo "WHEN DONE:  bash scripts/pod.sh down"
