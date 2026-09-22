"""Deterministic backstop: reject prompts / captions / sample prompts that indicate a minor.

This is a short explicit denylist with word boundaries, not a classifier. It is applied to
POST /generate prompts, every caption in a training dataset, and the training sample prompts.
Do not oversell it in the UI; do not remove it.
"""
from __future__ import annotations

import re

_TERMS = [
    "child", "children", "kid", "kids", "minor", "minors", "underage", "under-age", "under age",
    "teen", "teens", "teenager", "teenagers", "preteen", "pre-teen", "tween",
    "loli", "lolita", "shota", "shotacon", "lolicon", "jailbait", "pedo", "paedo", "cp",
    "schoolgirl", "schoolboy", "school girl", "school boy", "high school", "highschool",
    "middle school", "junior high", "elementary school", "grade school", "kindergarten",
    "toddler", "infant", "newborn", "little girl", "little boy", "young girl", "young boy",
    "small girl", "small boy", "youthful girl", "childlike", "child-like",
]
_TERM_RE = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in _TERMS) + r")(?![a-z0-9])", re.IGNORECASE)
# "16 year old", "16yo", "16 y/o", "age 15", "aged 17", "17-year-old"
_AGE_RE = re.compile(
    r"(?<!\d)(?:(?:age|aged)\s*:?\s*(\d{1,2})|(\d{1,2})\s*-?\s*(?:years?|yrs?|yo|y/o)\s*-?\s*(?:old)?)(?!\d)",
    re.IGNORECASE,
)


def check(text: str) -> str | None:
    """Return the offending fragment, or None if the text passes."""
    if not text:
        return None
    m = _TERM_RE.search(text)
    if m:
        return m.group(1)
    for m in _AGE_RE.finditer(text):
        n = m.group(1) or m.group(2)
        if n is not None and int(n) < 18:
            return m.group(0).strip()
    return None


def check_many(texts: dict[str, str]) -> dict[str, str]:
    """{label: text} -> {label: offending fragment} for every text that fails."""
    bad = {}
    for label, t in texts.items():
        hit = check(t)
        if hit:
            bad[label] = hit
    return bad
