from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SPACE_RE = re.compile(r"\s+")
HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
TOKEN_RE = re.compile(r"[A-Za-z0-9\uac00-\ud7a3]{2,}")
REPEAT_RE = re.compile(r"(.)\1{8,}")
DEF_RE = re.compile(r"(\uc774\ub780\??$|\ub780\??$|\ubb34\uc5c7|\ubb50\uc57c|\uc815\uc758|\uc124\uba85\ud574)")
LOGIC_RE = re.compile(
    r"(\ubaa8\ub4e0\s+\S+\s+\ub294\s+\S+|\uc77c\ubd80\s+\S+\s+\ub294\s+\S+|\uac70\uc9d3\ub9d0\uc7c1\uc774|\uc9c4\uc2e4\ub9cc\s+\ub9d0|\ub17c\ub9ac|\ucd94\ub860)"
)
LOGIC_ANSWER_RE = re.compile(r"(\uc774\uc720|\ub530\ub77c\uc11c|\uacb0\ub860|\ubcf4\uc7a5|\uc544\ub2c8|\uad6c\ubd84)")
INTRO_RE = re.compile(
    r"(\uc548\ub155\ud558\uc138\uc694|\ubc18\uac11\uc2b5\ub2c8\ub2e4|\uc800\ub294\s+.+(ai|\uc5b4\uc2dc\uc2a4\ud134\ud2b8|\ub3c4\uc6b0\ubbf8|\ubaa8\ub378))",
    re.IGNORECASE,
)

NORM_SUFFIXES: Tuple[str, ...] = (
    "\uc778\uac00\uc694",
    "\uc778\uac00",
    "\uc774\ub780",
    "\ub780",
    "\uc785\ub2c8\ub2e4",
    "\uc774\uc57c",
    "\uc57c",
    "\uc740",
    "\ub294",
    "\uc774",
    "\uac00",
    "\uc744",
    "\ub97c",
    "\uc5d0",
    "\ub85c",
    "\ub3c4",
    "\ub9cc",
)

Q_KEYS: Tuple[str, ...] = ("input", "instruction", "prompt", "question", "context")
A_KEYS: Tuple[str, ...] = ("output", "response", "answer", "completion", "target")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build term-anchor patch dataset for Korean quality uplift.")
    p.add_argument(
        "--inputs",
        default="data/target_term_logic_clean_v2.jsonl,data/target_term_patch_v1.jsonl,data/ko_quality5x_focus_v1.jsonl",
    )
    p.add_argument("--out", type=Path, default=Path("data/term_anchor_patch_v2.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/term_anchor_patch_v2.manifest.json"))
    p.add_argument("--max_def", type=int, default=12000)
    p.add_argument("--max_logic", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def pick_text(row: Dict, keys: Sequence[str]) -> str:
    for key in keys:
        val = row.get(key)
        if val is None:
            continue
        s = normalize_space(val)
        if s:
            return s
    return ""


def iter_jsonl(path: Path) -> Iterable[Dict]:
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


def normalize_term(term: str) -> str:
    t = normalize_space(term).strip(".,!?\"'`()[]{}")
    low = t.lower()
    for s in NORM_SUFFIXES:
        if len(low) > len(s) + 1 and low.endswith(s):
            return t[: -len(s)]
    return t


def extract_term(question: str) -> str:
    q = normalize_space(question)
    m = re.search(r"(.+?)(\uc774\ub780\??$|\ub780\??$|\ubb34\uc5c7|\ubb50\uc57c|\uc815\uc758|\uc124\uba85\ud574)", q)
    if m:
        q = normalize_space(m.group(1))
    toks = TOKEN_RE.findall(q)
    if not toks:
        return ""
    return normalize_term(toks[-1])


def first_sentence(text: str, max_chars: int = 180) -> str:
    s = normalize_space(text)[: int(max_chars)]
    m = re.search(r"[.!?]\s|[.!?]$", s)
    if m:
        cut = s[: m.end()].strip()
        if len(cut) >= 8:
            return cut
    return s.strip()


def topic_particle(term: str) -> str:
    if not term:
        return "\ub294"
    for ch in reversed(term):
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            jong = (code - 0xAC00) % 28
            return "\uc740" if jong > 0 else "\ub294"
    return "\ub294"


def count_hangul(text: str) -> int:
    return len(HANGUL_RE.findall(str(text or "")))


def is_clean_pair(inp: str, out: str) -> bool:
    q = normalize_space(inp)
    a = normalize_space(out)
    if not q or not a:
        return False
    if len(q) < 2 or len(q) > 900:
        return False
    if len(a) < 12 or len(a) > 1100:
        return False
    if "\ufffd" in q or "\ufffd" in a:
        return False
    if REPEAT_RE.search(a):
        return False
    if INTRO_RE.search(a):
        return False
    if count_hangul(q + "\n" + a) < 8:
        return False
    return True


def anchor_definition_answer(term: str, answer: str) -> str:
    a = first_sentence(answer)
    if not term:
        return a
    if term.lower() in a.lower():
        return a
    part = topic_particle(term)
    merged = f"{term}{part} {a}"
    return normalize_space(merged)


def row_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def make_row(inp: str, out: str, source: str, tag: str) -> Dict:
    return {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": "high",
        "_meta_focus_tag": tag,
        "_meta_source_file": source,
        "input": normalize_space(inp),
        "output": normalize_space(out),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    seen = set()
    stats = Counter()

    defs: List[Dict] = []
    logic: List[Dict] = []
    input_paths = [Path(x.strip()) for x in str(args.inputs).split(",") if x.strip()]

    def add_row(inp: str, out: str, source: str, tag: str) -> bool:
        if not is_clean_pair(inp, out):
            stats["reject_clean"] += 1
            return False
        key = row_key(inp, out)
        if key in seen:
            stats["reject_dup"] += 1
            return False
        seen.add(key)
        if tag == "definition":
            defs.append(make_row(inp, out, source=source, tag=tag))
        else:
            logic.append(make_row(inp, out, source=source, tag=tag))
        stats[f"accept_{tag}"] += 1
        return True

    for path in input_paths:
        if not path.exists():
            stats["missing_input"] += 1
            continue
        src = str(path).replace("\\", "/")
        for row in iter_jsonl(path):
            inp = pick_text(row, Q_KEYS)
            out = pick_text(row, A_KEYS)
            if not inp or not out:
                stats["reject_missing_pair"] += 1
                continue

            if DEF_RE.search(inp):
                term = extract_term(inp)
                if len(term) < 2:
                    stats["reject_no_term"] += 1
                else:
                    anchored = anchor_definition_answer(term=term, answer=out)
                    add_row(inp=inp, out=anchored, source=src, tag="definition")
                    # Keep original when it already contains the exact term.
                    if term.lower() in out.lower():
                        add_row(inp=inp, out=out, source=src, tag="definition")
                continue

            if LOGIC_RE.search(inp):
                if not LOGIC_ANSWER_RE.search(out):
                    stats["reject_logic_weak"] += 1
                    continue
                add_row(inp=inp, out=out, source=src, tag="logic")
                continue

    rng.shuffle(defs)
    rng.shuffle(logic)
    defs = defs[: max(0, int(args.max_def))]
    logic = logic[: max(0, int(args.max_logic))]
    out_rows = defs + logic
    rng.shuffle(out_rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wf:
        for r in out_rows:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "generated_at": now_iso(),
        "out": str(out_path).replace("\\", "/"),
        "rows": len(out_rows),
        "counts": {"definition": len(defs), "logic": len(logic)},
        "inputs": [str(p).replace("\\", "/") for p in input_paths],
        "stats": dict(stats),
        "seed": int(args.seed),
        "limits": {"max_def": int(args.max_def), "max_logic": int(args.max_logic)},
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "rows": len(out_rows), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

