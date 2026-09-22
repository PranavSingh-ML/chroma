"""Turn a folder of random gallery photos into a training dataset. Runs ENTIRELY on your laptop.

    python scripts/prep_dataset.py <in-dir> <out-dir> [--crop] [--max-side 1536] [--dry-run]

What it does, per photo:
  1. HEIC/HEIF -> RGB (iPhone photos), EXIF orientation APPLIED, then ALL metadata STRIPPED
     (gallery photos carry GPS coordinates and timestamps - none of it reaches the pod).
  2. Local face detection (OpenCV YuNet, a 227 KB model fetched once then offline). It is a
     DETECTOR: it returns a bounding box and has no notion of who anyone is.
  3. Sorts each photo into a framing class from the face-box-to-image ratio:
       close   face  > 16% of frame height   (head and shoulders)
       half    face 6-16%                    (waist up)
       full    face  < 6%                    (full body)
  4. --crop: crops to a subject-centred frame with generous margin, keeping the framing class
     (a close-up stays a close-up). WITHOUT --crop nothing is cropped - ai-toolkit buckets by
     aspect ratio anyway, so cropping is optional and usually unnecessary.
  5. Resizes so the long side is <= --max-side, saves as sequentially-numbered JPEGs.

It REJECTS (copies nothing) photos with no detectable face, and FLAGS photos with more than one
face so you can decide - a second person in frame teaches the LoRA the wrong thing.

Finally it prints the framing distribution, because a dataset that is all close-ups produces a
LoRA that cannot do full-body shots. Nothing is uploaded; nothing is displayed.
"""
from __future__ import annotations

import argparse
import os
import sys

MIN_FACE_PX = 48


def load_heif() -> None:
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass


YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".yunet.onnx")


def _yunet(size):
    """OpenCV's YuNet detector (227 KB ONNX, downloaded once, then fully offline).
    It returns face BOXES. It is a detector, not a recogniser - it has no notion of identity."""
    import cv2

    if not os.path.exists(YUNET_PATH):
        import urllib.request

        print(f"  (one-time) fetching the face detector -> {YUNET_PATH}")
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    return cv2.FaceDetectorYN.create(YUNET_PATH, "", size, score_threshold=0.5, nms_threshold=0.3)


def detect_faces(img):
    """Return a list of (x, y, w, h), biggest first."""
    import cv2
    import numpy as np

    full = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = full.shape[:2]
    # Pass 1 at ~1280px (fast). Pass 2 at full resolution only if pass 1 found nothing - a small
    # face in a full-body shot is precisely the case we cannot afford to drop, since a dataset
    # without full-body shots produces a LoRA that can only draw faces.
    for target in (1280, max(h, w)):
        scale = min(1.0, target / max(h, w))
        bgr = full if scale >= 1.0 else cv2.resize(full, (int(w * scale), int(h * scale)))
        det = _yunet((bgr.shape[1], bgr.shape[0]))
        _, faces = det.detect(bgr)
        out = []
        for f in faces if faces is not None else []:
            x, y, fw, fh = (float(v) / scale for v in f[:4])
            if min(fw, fh) < MIN_FACE_PX:
                continue
            out.append((int(x), int(y), int(fw), int(fh)))
        if out:
            return sorted(out, key=lambda b: -b[2] * b[3])
        if scale >= 1.0:
            break
    return []


def framing_of(face, img_h: int) -> str:
    ratio = face[3] / img_h
    if ratio > 0.16:
        return "close"
    if ratio > 0.06:
        return "half"
    return "full"


def subject_crop(img, face, framing: str):
    """Crop around the subject, keeping the framing class. Generous margins: a LoRA needs context,
    and a tight face crop teaches the model it can only draw faces."""
    W, H = img.size
    fx, fy, fw, fh = face
    cx = fx + fw / 2
    # how much of the frame height the face should occupy after cropping, per class
    target = {"close": 0.28, "half": 0.12, "full": 0.055}[framing]
    crop_h = min(H, fh / target)
    crop_w = min(W, crop_h * 0.75)          # 3:4 portrait, the useful shape for people
    if crop_w > W:
        crop_w = W
        crop_h = min(H, crop_w / 0.75)
    # horizontally centre on the face; vertically leave ~1 face-height of headroom above
    left = max(0, min(W - crop_w, cx - crop_w / 2))
    top = max(0, min(H - crop_h, fy - fh * (1.2 if framing == "close" else 2.2)))
    return img.crop((int(left), int(top), int(left + crop_w), int(top + crop_h)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("outdir")
    ap.add_argument("--crop", action="store_true", help="crop to a subject-centred frame (optional)")
    ap.add_argument("--max-side", type=int, default=1536)
    ap.add_argument("--keep-multi", action="store_true", help="keep photos with >1 face (default: flag and skip)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    try:
        from PIL import Image, ImageOps
    except ImportError:
        print("need Pillow:  pip install pillow opencv-python-headless pillow-heif")
        return 2
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("need OpenCV:  pip install opencv-python-headless")
        return 2
    load_heif()

    exts = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff")
    files = sorted(f for f in os.listdir(args.indir) if f.lower().endswith(exts))
    if not files:
        print(f"no images in {args.indir}")
        return 1
    if not args.dry_run:
        os.makedirs(args.outdir, exist_ok=True)

    counts = {"close": 0, "half": 0, "full": 0}
    kept, skipped_noface, skipped_multi, errors = 0, [], [], []
    print(f"== {len(files)} files in {args.indir}\n")
    for f in files:
        src = os.path.join(args.indir, f)
        try:
            img = Image.open(src)
            img = ImageOps.exif_transpose(img)          # apply rotation...
            img = img.convert("RGB")                     # ...then drop every EXIF/GPS field
        except Exception as e:  # noqa: BLE001
            errors.append((f, str(e)[:60]))
            continue
        faces = detect_faces(img)
        if not faces:
            skipped_noface.append(f)
            print(f"  SKIP  {f:40s} no face detected")
            continue
        if len(faces) > 1 and not args.keep_multi:
            skipped_multi.append(f)
            print(f"  SKIP  {f:40s} {len(faces)} faces - crop to just you, or pass --keep-multi")
            continue
        face = faces[0]
        fr = framing_of(face, img.height)
        if args.crop:
            img = subject_crop(img, face, fr)
        if max(img.size) > args.max_side:
            s = args.max_side / max(img.size)
            img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
        counts[fr] += 1
        kept += 1
        out = os.path.join(args.outdir, f"{kept:03d}.jpg")
        if not args.dry_run:
            img.save(out, "JPEG", quality=95)           # no exif= argument -> metadata gone
        print(f"  ok    {f:40s} -> {os.path.basename(out)}  {img.size[0]}x{img.size[1]}  {fr}")

    print(f"\n== kept {kept}, skipped {len(skipped_noface)} (no face) + {len(skipped_multi)} (multiple faces)"
          + (f", {len(errors)} unreadable" if errors else ""))
    print(f"== framing: close-up {counts['close']}   half body {counts['half']}   full body {counts['full']}")
    warn = []
    if kept < 10:
        warn.append(f"only {kept} images - aim for 15-25")
    if counts["close"] and counts["close"] / max(1, kept) > 0.8:
        warn.append("over 80% close-ups - the LoRA will struggle with full-body shots; add some")
    if counts["full"] == 0:
        warn.append("no full-body shots - add 3-5 or the LoRA will only know your face")
    if counts["half"] == 0:
        warn.append("no half-body shots - the middle range will be weak")
    for w in warn:
        print(f"   !! {w}")
    if not warn and kept:
        print("   framing spread looks good")
    if args.dry_run:
        print("\n(dry run - nothing written)")
    else:
        print(f"\nNEXT:  scripts/caption.sh {args.outdir} <trigger>     # auto-caption on your pod")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
