from __future__ import annotations

import fnmatch
import re
import shutil
from pathlib import Path

from .sandbox import file_manifest

BLOCKED = re.compile(
    r"(?:^|[/_.-])(memory|handoff|transcript|conversation|chat|notes|history|"
    r"live_ctx|context|trajectory)(?:$|[/_.-])",
    re.I,
)
CACHE_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".cache"}


def _grams(text: str, n: int = 16) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def clean_snapshot(
    source: Path, destination: Path, allowed: list[str], initial: dict, transcript: list[dict]
) -> dict:
    """Keep functional files; reject detected transcript relocation.

    The audit is heuristic. Passing it does not prove absence of encoded/paraphrased
    memory. Reports require manual functional-artifact review for confirmatory claims.
    """
    actual = file_manifest(source)  # Reject symlinks before touching any content.
    # Compare to assistant/user utterances, not copied tool output (which contains legitimate code).
    import json

    utterances = []
    for message in transcript:
        if message.get("origin") != "model":
            continue
        content = message["content"]
        try:
            action = json.loads(content)
        except ValueError:
            action = None
        if isinstance(action, dict) and action.get("tool") in {"write", "run", "context"}:
            # Code/tool payloads legitimately recur in functional files.
            content = str(action.get("reasoning", ""))
        utterances.append(content)
    grams = set().union(*(_grams(s) for s in utterances)) if utterances else set()
    kept, removed, violations = [], [], []
    destination.mkdir(parents=True, exist_ok=False)
    for name, sha in actual.items():
        p = source / name
        unchanged = initial.get(name) == sha
        if any(part in CACHE_DIRS for part in p.relative_to(source).parts):
            removed.append(name)
            continue
        if not unchanged and BLOCKED.search(name):
            removed.append(name)
            continue
        if not unchanged and not any(fnmatch.fnmatchcase(name, pat) for pat in allowed):
            removed.append(name)
            continue
        if not unchanged:
            if p.stat().st_size > 8_000_000:
                violations.append({"file": name, "reason": "unreviewed large artifact"})
                continue
            data = p.read_bytes()
            if b"\x00" in data:
                violations.append({"file": name, "reason": "opaque binary artifact"})
                continue
            text = data.decode("utf-8", errors="replace")
            if len(re.findall(r'"role"\s*:\s*"(?:assistant|user|system)"', text)) >= 3:
                violations.append({"file": name, "reason": "conversation message envelopes"})
                continue
            # JSON tool envelopes aren't prose history. Check only assistant free text,
            # excluding file-write contents that naturally coincide with the saved file.
            if grams and len(_grams(text) & grams) >= 3:
                violations.append({"file": name, "reason": "suspected transcript copy"})
                continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        kept.append(name)
    return {
        "kept": kept,
        "removed": removed,
        "violations": violations,
        "manual_review_required": True,
        "manifest": file_manifest(destination),
    }
