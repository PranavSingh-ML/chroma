# Sourced by the other scripts. Sets $PY to a working Python 3 with Pillow.
# On Windows, bash's `python3` is usually the Microsoft Store stub, which prints a message and
# exits 9009 - so probe candidates and pick one that actually runs.
if [ -z "${PY:-}" ]; then
  for c in "${PYTHON:-}" "$(dirname "${BASH_SOURCE[0]}")/../.venv/Scripts/python.exe" python3 python py; do
    [ -n "$c" ] || continue
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info[0]==3 else 1)' >/dev/null 2>&1; then
      PY="$c"; break
    fi
  done
fi
[ -n "${PY:-}" ] || { echo "no working python3 found - install Python 3 (with Pillow) or set PYTHON=/path/to/python"; exit 1; }
"$PY" -c 'import PIL' >/dev/null 2>&1 || echo "  note: Pillow missing in $PY - image checks will fail (pip install pillow)"
