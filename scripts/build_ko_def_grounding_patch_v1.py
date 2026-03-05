from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set


DEF_Q_RE = re.compile(r"(\uc774\ub780\??$|\ub780\??$|\ubb34\uc5c7|\ubb50\uc57c|\uc815\uc758|\uc124\uba85\ud574)")
LOGIC_ANS_RE = re.compile(r"(\ub530\ub77c\uc11c|\uacb0\ub860|\ubcf4\uc7a5|\uac00\uc815|\uad6c\ubd84)")
CODE_LIKE_RE = re.compile(
    r"(^\s*(?:import|from|export|const|let|var|function|class)\b|```|</?\w+[^>]*>|^\s*#include\b|^\s*SELECT\b)",
    re.IGNORECASE | re.MULTILINE,
)
SPACE_RE = re.compile(r"\s+")
REPEAT_RE = re.compile(r"(.)\1{8,}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Korean definition grounding patch dataset (pure-learning).")
    p.add_argument("--in_jsonl", type=Path, default=Path("data/slm_mit_vocabmix_ko_v1.jsonl"))
    p.add_argument("--out_jsonl", type=Path, default=Path("data/ko_def_grounding_patch_v1.jsonl"))
    p.add_argument("--max_rows", type=int, default=60000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def has_final_consonant(ch: str) -> bool:
    c = str(ch or "")
    if not c:
        return False
    cp = ord(c[-1])
    if cp < 0xAC00 or cp > 0xD7A3:
        return False
    return ((cp - 0xAC00) % 28) != 0


def topic_particle(term: str) -> str:
    t = str(term or "").strip()
    if not t:
        return "\ub294"
    if re.search(r"[\uac00-\ud7a3]$", t):
        return "\uc740" if has_final_consonant(t[-1]) else "\ub294"
    return "is"


def extract_term(q: str) -> str:
    s = normalize_space(q)
    s = re.sub(r"(\uc774\ub780\??$|\ub780\??$|\ubb34\uc5c7|\ubb50\uc57c|\uc815\uc758|\uc124\uba85\ud574).*", "", s).strip()
    toks = re.findall(r"[A-Za-z0-9\uac00-\ud7a3]{2,}", s)
    if not toks:
        return ""
    return toks[-1]


def stable_key(inp: str, out: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.hexdigest()


def iter_rows(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))

    seen: Set[str] = set()
    out_rows: List[Dict] = []
    stats = {
        "read": 0,
        "ko_rows": 0,
        "def_q": 0,
        "kept": 0,
        "prefixed": 0,
        "skip_noise": 0,
        "skip_logic": 0,
        "skip_code": 0,
        "skip_short": 0,
        "skip_dup": 0,
    }

    for row in iter_rows(Path(args.in_jsonl)):
        stats["read"] += 1
        lang = str(row.get("language", "")).strip().lower()
        if lang != "ko":
            continue
        stats["ko_rows"] += 1

        inp = normalize_space(str(row.get("input", "")))
        out = normalize_space(str(row.get("output", "")))
        if not inp or not out:
            continue

        if not DEF_Q_RE.search(inp):
            continue
        stats["def_q"] += 1

        if len(out) < 12 or len(out) > 320:
            stats["skip_short"] += 1
            continue
        if "\ufffd" in out or REPEAT_RE.search(out):
            stats["skip_noise"] += 1
            continue
        if LOGIC_ANS_RE.search(out):
            stats["skip_logic"] += 1
            continue
        if CODE_LIKE_RE.search(out):
            stats["skip_code"] += 1
            continue

        term = extract_term(inp)
        if not term or len(term) < 2:
            continue

        low_out = out.lower()
        low_term = term.lower()
        if low_term not in low_out:
            p = topic_particle(term)
            out = normalize_space(f"{term}{p} {out}")
            stats["prefixed"] += 1

        rec = dict(row)
        rec["input"] = inp
        rec["output"] = out
        rec["task_type"] = "korean"
        rec["segment_tag"] = "ko"
        rec["language"] = "ko"
        rec["_meta_quality_tier"] = "high"
        rec["_meta_patch"] = "ko_def_grounding_v1"

        key = stable_key(inp, out)
        if key in seen:
            stats["skip_dup"] += 1
            continue
        seen.add(key)
        out_rows.append(rec)
        stats["kept"] += 1

    rng.shuffle(out_rows)
    lim = max(0, int(args.max_rows))
    if lim > 0 and len(out_rows) > lim:
        out_rows = out_rows[:lim]

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wf:
        for row in out_rows:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "status": "ok",
                "out_jsonl": str(out_path),
                "rows": len(out_rows),
                "stats": stats,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
