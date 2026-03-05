from __future__ import annotations

import json
import re
from pathlib import Path


SRC_BASE = Path("data/repair_ko_direct_v2.jsonl")
SRC_POOL = Path("data/repair_ko_only_clean_v1.jsonl")
OUT_PATH = Path("data/repair_ko_direct_v3.jsonl")
MANIFEST_PATH = Path("data/repair_ko_direct_v3.manifest.json")


META_OUTPUT_HINTS = [
    "\uc548\ub155\ud558\uc138\uc694",
    "\ubc18\uac11\uc2b5\ub2c8\ub2e4",
    "\ud544\uc694\ud558\uba74",
    "\uc6d0\ud558\uba74",
    "\ud575\uc2ec\uc740",
    "\ud55c\uad6d\uc5b4 AI",
    "AI \ub3c4\uc6b0\ubbf8",
    "AI \uc5b4\uc2dc\uc2a4\ud134\ud2b8",
    "\uc548\ub0b4\ud558\ub294 AI",
    "AI\uc785\ub2c8\ub2e4",
]

BLOCK_ANYWHERE = [
    "ACTION ",
    "OBSERVATION ",
    "FINAL ",
    "\ub3c4\uad6c:",
    "\ucd9c\ub825 \ud615\uc2dd:",
    "```",
    "http://",
    "https://",
    "\ufffd",
]

LOGIC_INPUT_HINTS = [
    "\uc218\uc5f4",
    "\ubc18\ub840",
    "\uc99d\uba85",
    "\ubaa8\uc790",
    "\uac70\uc9d3\ub9d0\uc7c1\uc774",
    "\ud56d\uc0c1",
    "\uac00\uc7a5 \uc791\uc740",
    "\uc815\uc218 n",
    "\ud310\ub2e8",
    "\uc124\uba85\ud558\ub77c",
    "\ube44\uc728",
    "\ub17c\ub9ac",
]

DEF_INPUT_HINTS = [
    "\uc774\ub780",
    "\ubb34\uc5c7\uc774\uc57c",
    "\ubb50\uc57c",
    "\uc124\uba85\ud574\uc918",
    "\uc124\uba85\ud558\uc138\uc694",
]


def is_meta_row(inp: str, out: str) -> bool:
    low = f"{inp}\n{out}"
    if any(tok in low for tok in BLOCK_ANYWHERE):
        return True
    if any(tok in out for tok in META_OUTPUT_HINTS):
        return True
    if any(tok in inp for tok in ["\ubcf8\uc778 \uc18c\uac1c", "\uc790\uae30\uc18c\uac1c", "\ub108\uc758 \uc774\ub984"]):
        return True
    return False


def is_clean_direct(inp: str, out: str) -> bool:
    if is_meta_row(inp=inp, out=out):
        return False
    if len(inp) < 3 or len(inp) > 320:
        return False
    if len(out) < 8 or len(out) > 360:
        return False
    return True


def looks_logic_or_math(inp: str) -> bool:
    if any(tok in inp for tok in LOGIC_INPUT_HINTS):
        return True
    if re.search(r"\d", inp) and re.search(r"[+\-*/=]", inp):
        return True
    if re.search(r"[A-Z]\s*(?:,|and|\ubcf4\ub2e4|\ub294)", inp):
        return True
    return False


def looks_definition(inp: str) -> bool:
    return any(tok in inp for tok in DEF_INPUT_HINTS)


def main() -> None:
    stats = {
        "base_rows": 0,
        "base_kept": 0,
        "pool_rows": 0,
        "logic_extra": 0,
        "def_extra": 0,
        "written": 0,
        "duplicates": 0,
    }
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []

    with SRC_BASE.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            stats["base_rows"] += 1
            inp = str(row.get("input", "")).strip()
            out = str(row.get("output", "")).strip()
            if not is_clean_direct(inp=inp, out=out):
                continue
            rows.append(row)
            stats["base_kept"] += 1

    with SRC_POOL.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            stats["pool_rows"] += 1
            inp = str(row.get("input", "")).strip()
            out = str(row.get("output", "")).strip()
            if not is_clean_direct(inp=inp, out=out):
                continue
            if looks_logic_or_math(inp):
                for _ in range(3):
                    rows.append(row)
                stats["logic_extra"] += 3
            elif looks_definition(inp):
                rows.append(row)
                stats["def_extra"] += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as g:
        for row in rows:
            inp = str(row.get("input", "")).strip()
            out = str(row.get("output", "")).strip()
            key = (inp, out)
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
            row["language"] = "ko"
            row["task_type"] = "korean"
            row["segment_tag"] = "ko"
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["written"] += 1

    MANIFEST_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT_PATH), "manifest": str(MANIFEST_PATH), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
