from __future__ import annotations

import json
import re
from pathlib import Path


SRC_PATHS = [
    Path("data/repair_ko_direct_v3.jsonl"),
    Path("data/final_datasets/한국어유창성_augmented_v1.jsonl"),
    Path("data/final_datasets/한국어_일반대화_augmented_v1.jsonl"),
]
OUT_PATH = Path("data/logic_only_v2.jsonl")
MANIFEST_PATH = Path("data/logic_only_v2.manifest.json")


LOGIC_INPUT_PATTERNS = [
    r"거짓말쟁이",
    r"진실만 말하",
    r"항상참",
    r"항상거짓",
    r"무작위",
    r"\bda\b",
    r"\bja\b",
    r"수열",
    r"반례",
    r"증명",
    r"소수",
    r"정수",
    r"나머지",
    r"비율",
    r"가장 작은",
    r"가장 큰",
    r"모자를 쓰",
    r"두 질문만",
    r"전제",
    r"결론",
    r"추론",
    r"A는 B보다",
    r"B는 C보다",
    r"항상 .*인가",
    r"설명하고 답하라",
]

EXPLAIN_PATTERNS = [
    r"전제",
    r"조건",
    r"추론",
    r"결론",
    r"따라서",
    r"그러므로",
    r"왜냐하면",
    r"반례",
    r"질문\s*1",
    r"\bQ1\b",
    r"먼저",
    r"예를 들어",
    r"1\.",
    r"2\.",
]

BAD_OUTPUT_PATTERNS = [
    r"^\s*정답은\s*[-+]?\d+(?:\.\d+)?입니다\.?\s*$",
    r"^\s*정답은\s*[A-Z]입니다\.?\s*$",
    r"^\s*정답은\s*[-+]?\d+(?:\.\d+)?\s*$",
    r"^\s*[-+]?\d+(?:\.\d+)?\s*$",
    r"^\s*[A-Z]\s*$",
    r"^\s*FINAL\b",
    r"^\s*ACTION\b",
    r"^\s*OBSERVATION\b",
]

BAD_ANYWHERE = [
    "정답은 10.0입니다",
    "정답은 2.0입니다",
    "정답은 240입니다",
    "필요하면 예시도",
    "원하면 더 짧게",
    "ACTION ",
    "FINAL ",
    "OBSERVATION ",
    "```",
    "\ufffd",
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
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_logic_input(text: str) -> bool:
    src = compact(text)
    if len(src) < 6 or len(src) > 520:
        return False
    if any(re.search(pat, src) for pat in LOGIC_INPUT_PATTERNS):
        return True
    has_digits = bool(re.search(r"\d", src))
    has_ops = bool(re.search(r"[+\-*/=^<>]", src))
    return has_digits and has_ops


def normalize_output(text: str) -> str:
    out = compact(text)
    out = re.sub(r"^\s*정답은\s*", "", out)
    out = re.sub(r"^\s*답\s*[:：]\s*", "", out)
    out = re.sub(r"^\s*최종 답\s*[:：]\s*", "", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def has_explanation(text: str) -> bool:
    if any(re.search(pat, text) for pat in EXPLAIN_PATTERNS):
        return True
    sentence_count = len([s for s in re.split(r"[.!?]\s+|\n+", text) if s.strip()])
    return sentence_count >= 2


def usable(inp: str, out: str) -> bool:
    if len(out) < 12 or len(out) > 420:
        return False
    joined = f"{inp}\n{out}"
    if any(tok in joined for tok in BAD_ANYWHERE):
        return False
    if any(re.search(pat, out) for pat in BAD_OUTPUT_PATTERNS):
        return False
    if "모르겠습니다" in out:
        return False
    if out.count("정답은") >= 1:
        return False
    return has_explanation(out)


def main() -> None:
    stats = {
        "source_rows": {},
        "kept_rows": {},
        "duplicates": 0,
        "written": 0,
    }
    seen: set[tuple[str, str]] = set()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as g:
        for path in SRC_PATHS:
            name = path.name
            stats["source_rows"][name] = 0
            stats["kept_rows"][name] = 0
            for row in load_rows(path):
                stats["source_rows"][name] += 1
                inp = compact(row.get("input", ""))
                out = normalize_output(row.get("output", ""))
                if not looks_like_logic_input(inp):
                    continue
                if not usable(inp=inp, out=out):
                    continue
                new_row = {
                    "input": inp,
                    "output": out if out.endswith((".", "?", "!")) else f"{out}.",
                    "task_type": "korean",
                    "segment_tag": "ko",
                    "language": "ko",
                    "_meta_source_file": str(row.get("_meta_source_file", path)).strip(),
                }
                key = (new_row["input"], new_row["output"])
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
                g.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                stats["kept_rows"][name] += 1
                stats["written"] += 1

    MANIFEST_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT_PATH), "manifest": str(MANIFEST_PATH), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
