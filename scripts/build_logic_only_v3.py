from __future__ import annotations

import json
import re
from pathlib import Path


SRC_PATHS = [
    Path("data/repair_ko_direct_v3.jsonl"),
    Path("data/final_datasets/한국어유창성_augmented_v1.jsonl"),
    Path("data/final_datasets/한국어_일반대화_augmented_v1.jsonl"),
    Path("data/final_datasets/MIT혼합코퍼스_augmented_v1.jsonl"),
]
OUT_PATH = Path("data/logic_only_v3.jsonl")
MANIFEST_PATH = Path("data/logic_only_v3.manifest.json")


STRONG_LOGIC_PATTERNS = [
    r"거짓말쟁이",
    r"진실만 말하",
    r"항상참",
    r"항상거짓",
    r"무작위",
    r"\bda\b",
    r"\bja\b",
    r"모든 사람은 죽는다",
    r"A는 B보다",
    r"B는 C보다",
    r"수열",
    r"반례",
    r"증명",
    r"모자를 쓰",
    r"두 질문만",
    r"가장 작은",
    r"가장 큰",
    r"정수 n",
    r"n\^2\s*-\s*n\s*\+\s*41",
]

GENERIC_DEF_PATTERNS = [
    r"무엇(?:인가|인가요)?",
    r"뭐야",
    r"정의",
    r"뜻",
    r"란\??$",
    r"무슨 의미",
]

EXPLAIN_PATTERNS = [
    r"전제",
    r"조건",
    r"추론",
    r"결론",
    r"따라서",
    r"그러므로",
    r"반례",
    r"질문\s*1",
    r"\bQ1\b",
    r"1\.",
    r"2\.",
    r"3\.",
]

BAD_OUTPUT_PATTERNS = [
    r"^\s*정답은",
    r"^\s*FINAL\b",
    r"^\s*ACTION\b",
    r"^\s*OBSERVATION\b",
    r"^\s*[-+]?\d+(?:\.\d+)?\s*$",
    r"^\s*[A-Z]\s*$",
]


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def compact(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def is_strong_logic_input(text: str) -> bool:
    src = compact(text)
    if len(src) < 8 or len(src) > 520:
        return False
    if any(re.search(pat, src) for pat in STRONG_LOGIC_PATTERNS):
        return True
    has_digits = bool(re.search(r"\d", src))
    has_ops = bool(re.search(r"[+\-*/=^<>]", src))
    return has_digits and has_ops


def is_generic_definition(text: str) -> bool:
    src = compact(text)
    return any(re.search(pat, src) for pat in GENERIC_DEF_PATTERNS)


def normalize_output(text: str) -> str:
    out = compact(text)
    out = re.sub(r"^\s*정답은\s*", "", out)
    out = re.sub(r"^\s*답\s*[:：]\s*", "", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def has_reasoning_shape(text: str) -> bool:
    if any(re.search(pat, text) for pat in EXPLAIN_PATTERNS):
        return True
    sentence_count = len([s for s in re.split(r"[.!?]\s+|\n+", text) if s.strip()])
    return sentence_count >= 2 and any(tok in text for tok in ["따라서", "그러므로", "왜냐하면"])


def usable(inp: str, out: str) -> bool:
    if len(out) < 16 or len(out) > 420:
        return False
    if any(re.search(pat, out) for pat in BAD_OUTPUT_PATTERNS):
        return False
    if any(tok in out for tok in ["필요하면 예시도", "원하면", "모르겠습니다", "```", "\ufffd"]):
        return False
    return has_reasoning_shape(out)


def row_weight(row: dict, inp: str) -> int:
    source = str(row.get("source", "")).strip()
    source_file = str(row.get("_meta_source_file", "")).strip()
    weight = 1
    if source in {"logic_v1", "cot_logic"}:
        weight += 3
    if "logic_reasoning" in source_file or "clean_mix_v3_logic" in source_file:
        weight += 2
    if any(tok in inp for tok in ["거짓말쟁이", "모든 사람은 죽는다", "A는 B보다"]):
        weight += 2
    return min(weight, 6)


def main() -> None:
    stats = {
        "source_rows": {},
        "kept_unique": 0,
        "written": 0,
        "duplicates": 0,
    }
    seen: set[tuple[str, str]] = set()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as g:
        for path in SRC_PATHS:
            stats["source_rows"][path.name] = 0
            for row in load_rows(path):
                stats["source_rows"][path.name] += 1
                inp = compact(row.get("input", ""))
                out = normalize_output(row.get("output", ""))
                if not is_strong_logic_input(inp):
                    continue
                if is_generic_definition(inp) and not re.search(r"\d", inp):
                    continue
                if not usable(inp, out):
                    continue
                new_row = {
                    "input": inp,
                    "output": out if out.endswith((".", "?", "!")) else f"{out}.",
                    "task_type": "korean",
                    "segment_tag": "ko",
                    "language": "ko",
                    "_meta_source_file": str(row.get("_meta_source_file", path)).strip(),
                    "source": str(row.get("source", "")).strip(),
                }
                key = (new_row["input"], new_row["output"])
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
                stats["kept_unique"] += 1
                for _ in range(row_weight(row, inp)):
                    g.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                    stats["written"] += 1

    MANIFEST_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT_PATH), "manifest": str(MANIFEST_PATH), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
