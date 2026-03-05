from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


SPACE_RE = re.compile(r"\s+")
HANGUL_RE = re.compile(r"[가-힣]")
TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]{2,}")
REPEAT_RE = re.compile(r"(.)\1{8,}")
INTRO_RE = re.compile(
    r"(자기소개|소개해|너는\s*누구|모델\s*이름|네?\s*이름|안녕하세요|반갑습니다|저는\s+.+(ai|어시스턴트|도우미|모델))",
    re.IGNORECASE,
)
DEF_RE = re.compile(r"(이란\??$|란\??$|무엇|뭐야|정의|설명해)")
LOGIC_RE = re.compile(r"(모든\s+\S+\s+는\s+\S+|일부\s+\S+\s+는\s+\S+|거짓말쟁이|진실만\s+말|추론|논리)")
MATH_RE = re.compile(r"(\d+\s*[\+\-\*\/]\s*\d+|얼마|계산)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build focus dataset for Korean definition/logic quality uplift.")
    p.add_argument("--in_clean", type=Path, default=Path("data/ko_quality5x_clean_v2.jsonl"))
    p.add_argument("--in_hq", type=Path, default=Path("data/quality/hq_ko_chat_v1.jsonl"))
    p.add_argument("--in_logic", type=Path, default=Path("data/logic_reasoning_v3_combined.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/ko_quality5x_focus_v1.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/ko_quality5x_focus_v1.manifest.json"))
    p.add_argument("--take_def", type=int, default=42000)
    p.add_argument("--take_logic", type=int, default=12000)
    p.add_argument("--take_general", type=int, default=28000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def pick_text(row: Dict, keys: Sequence[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        s = normalize_space(v)
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


def hangul_count(text: str) -> int:
    return len(HANGUL_RE.findall(str(text or "")))


def dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def normalize_term(term: str) -> str:
    t = normalize_space(term).strip(".,!?\"'`()[]{}")
    suffixes = ("이란", "란", "은", "는", "이", "가", "을", "를")
    for s in suffixes:
        if len(t) > len(s) + 1 and t.endswith(s):
            return t[: -len(s)]
    return t


def extract_term(inp: str) -> str:
    s = normalize_space(inp)
    m = re.search(r"(.+?)(이란\??$|란\??$|무엇|뭐야|정의|설명해)", s)
    if m:
        s = normalize_space(m.group(1))
    parts = TOKEN_RE.findall(s)
    if not parts:
        return ""
    return normalize_term(parts[-1])


def is_clean_pair(inp: str, out: str) -> bool:
    q = normalize_space(inp)
    a = normalize_space(out)
    if not q or not a:
        return False
    if len(q) < 2 or len(q) > 900:
        return False
    if len(a) < 10 or len(a) > 1200:
        return False
    merged = f"{q}\n{a}"
    if "\ufffd" in merged or "???" in merged:
        return False
    if REPEAT_RE.search(merged):
        return False
    if INTRO_RE.search(q) or INTRO_RE.search(a):
        return False
    if hangul_count(merged) < 6:
        return False
    return True


def is_def_pair(inp: str, out: str) -> bool:
    q = normalize_space(inp)
    a = normalize_space(out)
    if not DEF_RE.search(q):
        return False
    term = extract_term(q)
    if len(term) < 2:
        return False
    low_a = a.lower()
    low_term = term.lower()
    if low_term in low_a:
        return True
    # Accept common English aliases for acronym terms.
    if low_term == "llm" and ("llm" in low_a or "언어 모델" in a):
        return True
    return False


def is_logic_pair(inp: str, out: str) -> bool:
    q = normalize_space(inp)
    a = normalize_space(out)
    if not (LOGIC_RE.search(q) or MATH_RE.search(q)):
        return False
    if MATH_RE.search(q) and not re.search(r"\d", a):
        return False
    if LOGIC_RE.search(q) and not re.search(r"(이유|따라서|결론|아니|불가|보장|질문)", a):
        return False
    return True


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
    rejects = Counter()

    defs: List[Dict] = []
    logic: List[Dict] = []
    general: List[Dict] = []

    def try_add(inp: str, out: str, source: str) -> None:
        if not is_clean_pair(inp, out):
            rejects["clean_fail"] += 1
            return
        key = dedupe_key(inp, out)
        if key in seen:
            rejects["dup"] += 1
            return
        seen.add(key)

        if is_def_pair(inp, out):
            defs.append(make_row(inp, out, source=source, tag="definition"))
            return
        if is_logic_pair(inp, out):
            logic.append(make_row(inp, out, source=source, tag="logic"))
            return
        general.append(make_row(inp, out, source=source, tag="general"))

    for path in (args.in_clean, args.in_hq, args.in_logic):
        p = Path(path)
        if not p.exists():
            continue
        src = str(p).replace("\\", "/")
        for row in iter_jsonl(p):
            inp = pick_text(row, ("input", "instruction", "prompt", "question", "context"))
            out = pick_text(row, ("output", "response", "answer", "completion", "target"))
            try_add(inp=inp, out=out, source=src)

    rng.shuffle(defs)
    rng.shuffle(logic)
    rng.shuffle(general)

    out_rows: List[Dict] = []
    out_rows.extend(defs[: max(0, int(args.take_def))])
    out_rows.extend(logic[: max(0, int(args.take_logic))])
    out_rows.extend(general[: max(0, int(args.take_general))])
    rng.shuffle(out_rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wf:
        for r in out_rows:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")

    tag_counts = Counter(str(r.get("_meta_focus_tag", "")) for r in out_rows)
    manifest = {
        "generated_at": now_iso(),
        "out": str(out_path).replace("\\", "/"),
        "rows": len(out_rows),
        "seed": int(args.seed),
        "requested": {
            "take_def": int(args.take_def),
            "take_logic": int(args.take_logic),
            "take_general": int(args.take_general),
        },
        "available_pool": {
            "definition": len(defs),
            "logic": len(logic),
            "general": len(general),
        },
        "written_tag_counts": dict(tag_counts),
        "rejects": dict(rejects),
        "inputs": [str(args.in_clean), str(args.in_hq), str(args.in_logic)],
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"out": str(out_path), "rows": len(out_rows), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
