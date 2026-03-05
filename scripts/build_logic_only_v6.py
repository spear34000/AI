from __future__ import annotations

import json
from pathlib import Path


SRC_PATH = Path("data/final_datasets/논리추론_augmented_v1.jsonl")
OUT_PATH = Path("data/logic_only_v6.jsonl")
MANIFEST_PATH = Path("data/logic_only_v6.manifest.json")

REASONING_HINTS = ["1.", "2.", "전제", "조건", "추론", "결론", "계산", "따라서", "그러므로"]
INPUT_BANNED = [
    "체크리스트",
    "상황에서",
    "습관",
    "자기소개",
    "정체가 뭐야",
    "너의 이름",
    "오늘 날씨",
    "TCP",
    "UDP",
    "API",
    "데이터베이스",
    "REST",
    "HTTP",
    "HTTPS",
    "로드 밸런싱",
    "캐시",
    "마이크로서비스",
    "코드를 작성",
    "프로그램을 작성",
    "파이썬",
    "python",
    "C 프로그램",
    "spear1.0",
]
SOURCE_BANNED = {
    "hf_mit_language_superpack_v1",
    "capability_pack_v1",
    "capability_pack_v2",
    "cot_explain",
    "cot_anti",
}


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def compact(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def valid_input(text: str) -> bool:
    text = compact(text)
    if len(text) < 6 or len(text) > 280:
        return False
    return not any(tok in text for tok in INPUT_BANNED)


def valid_output(text: str) -> bool:
    text = compact(text)
    if len(text) < 14 or len(text) > 420:
        return False
    if text.startswith("정답은"):
        return False
    if "```" in text or "필요하면" in text or "모르겠습니다" in text or "확실하지 않아" in text:
        return False
    return any(tok in text for tok in REASONING_HINTS)


def row_weight(source: str, inp: str, out: str) -> int:
    weight = 1
    if source in {"cot_logic", "logic_v1", "cot_arith", "logic_v2", "logic_v3"}:
        weight += 1
    if any(tok in inp for tok in ["모든 ", "일부 ", "반례", "증명", "거짓말쟁이", "진실화자"]):
        weight += 1
    if any(tok in out for tok in ["추론", "결론", "따라서", "그러므로"]):
        weight += 1
    return min(weight, 4)


def main() -> None:
    stats = {
        "source_rows": 0,
        "kept_unique": 0,
        "written": 0,
        "duplicates": 0,
        "filtered_source": 0,
        "filtered_input": 0,
        "filtered_output": 0,
    }
    seen: set[tuple[str, str]] = set()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as g:
        for row in load_rows(SRC_PATH):
            stats["source_rows"] += 1
            source = str(row.get("source", "")).strip()
            if source in SOURCE_BANNED:
                stats["filtered_source"] += 1
                continue

            inp = compact(row.get("input", ""))
            out = compact(row.get("output", ""))
            if not valid_input(inp):
                stats["filtered_input"] += 1
                continue
            if not valid_output(out):
                stats["filtered_output"] += 1
                continue

            rec = {
                "input": inp,
                "output": out if out.endswith((".", "!", "?")) else f"{out}.",
                "task_type": "korean",
                "segment_tag": "ko",
                "language": "ko",
                "_meta_source_file": "논리추론_augmented_v1.jsonl",
                "source": source,
            }
            key = (rec["input"], rec["output"])
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
            stats["kept_unique"] += 1
            for _ in range(row_weight(source, inp, out)):
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["written"] += 1

    MANIFEST_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT_PATH), "manifest": str(MANIFEST_PATH), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
