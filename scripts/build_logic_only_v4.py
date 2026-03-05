from __future__ import annotations

import json
import re
from pathlib import Path


SRC_PATHS = [
    Path("data/final_datasets/논리추론_augmented_v1.jsonl"),
    Path("data/final_datasets/한국어_일반대화_augmented_v1.jsonl"),
]
OUT_PATH = Path("data/logic_only_v4.jsonl")
MANIFEST_PATH = Path("data/logic_only_v4.manifest.json")

ALLOWED_SOURCES = {"cot_logic", "logic_v1", "cot_arith"}
ALLOWED_META_SOURCES = {
    "logic_reasoning_cot_v1.jsonl",
    "synthetic/deintro_focus_v1_logic",
}

INPUT_REQUIRED_PATTERNS = [
    r"거짓말쟁이|진실만 말하|진실화자|항상참|항상거짓|무작위|da/ja|질문\s*\d|Q1|Q2|Q3",
    r"모든 .+는 .+다|일부 .+는 .+다|반례|증명|항상 소수|참인가|거짓인가|판단하",
    r"수열|다음 수|규칙 기반|누락된 숫자",
    r"모자|색 모자",
    r"[0-9]\s*[,)]\s*.+",
    r"\d.*[+\-*/=^<>].*\d",
    r"A.?는 B.?보다|B.?는 C.?보다|가장 작은|가장 큰|누가 누구",
]

INPUT_BANNED_PATTERNS = [
    r"뭐야\??$|무엇인가요\??$|무엇입니까\??$",
    r"어떻게 생각해|의미가 뭐|어떤 의미|왜 중요한|도움이 되는|매력|유래",
    r"오늘 날씨|자기소개|너의 이름",
    r"TCP|UDP|파이썬|python|C 프로그램|코드를 작성|알고리즘을 구축",
]

OUTPUT_REQUIRED_PATTERNS = [
    r"1\.\s|2\.\s",
    r"전제|조건|추론|결론|계산|따라서|그러므로|왜냐하면",
]

OUTPUT_BANNED_PATTERNS = [
    r"^\s*정답은\b",
    r"^\s*FINAL\b|^\s*ACTION\b|^\s*OBSERVATION\b",
    r"필요하면|예시도|모르겠습니다|확실하지 않아",
    r"```|\ufffd",
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


def allowed_row(row: dict, src_path: Path) -> bool:
    source = str(row.get("source", "")).strip()
    source_file = str(row.get("_meta_source_file", "")).strip()
    if src_path.name == "논리추론_augmented_v1.jsonl":
        if source in {"hf_mit_language_superpack_v1", "cot_explain", "cot_anti", "capability_pack_v1", "capability_pack_v2"}:
            return False
        return source == "" or source.startswith("logic_") or source in {"cot_logic", "cot_arith"}
    if src_path.name == "한국어_일반대화_augmented_v1.jsonl":
        return source_file in ALLOWED_META_SOURCES
    return False


def valid_input(text: str) -> bool:
    text = compact(text)
    if len(text) < 12 or len(text) > 260:
        return False
    if any(re.search(pat, text) for pat in INPUT_BANNED_PATTERNS):
        return False
    return any(re.search(pat, text) for pat in INPUT_REQUIRED_PATTERNS)


def normalize_output(text: str) -> str:
    text = compact(text)
    text = re.sub(r"^\s*답:\s*", "", text)
    return text


def valid_output(text: str) -> bool:
    text = normalize_output(text)
    if len(text) < 24 or len(text) > 420:
        return False
    if any(re.search(pat, text) for pat in OUTPUT_BANNED_PATTERNS):
        return False
    if text.count("\n") > 8:
        return False
    return any(re.search(pat, text) for pat in OUTPUT_REQUIRED_PATTERNS)


def row_weight(row: dict, inp: str) -> int:
    weight = 1
    source = str(row.get("source", "")).strip()
    if source == "cot_logic":
        weight += 2
    elif source == "logic_v1":
        weight += 1
    if re.search(r"거짓말쟁이|진실화자|항상참|항상거짓|무작위|da/ja", inp):
        weight += 2
    if re.search(r"모든 .+는 .+다|일부 .+는 .+다|반례|증명|항상 소수", inp):
        weight += 2
    if re.search(r"수열|모자|가장 작은|가장 큰", inp):
        weight += 1
    return min(weight, 5)


def main() -> None:
    stats = {
        "source_rows": {},
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
        for src_path in SRC_PATHS:
            stats["source_rows"][src_path.name] = 0
            for row in load_rows(src_path):
                stats["source_rows"][src_path.name] += 1
                if not allowed_row(row, src_path):
                    stats["filtered_source"] += 1
                    continue

                inp = compact(row.get("input", ""))
                out = normalize_output(row.get("output", ""))

                if not valid_input(inp):
                    stats["filtered_input"] += 1
                    continue
                if not valid_output(out):
                    stats["filtered_output"] += 1
                    continue

                item = {
                    "input": inp,
                    "output": out if out.endswith((".", "!", "?")) else f"{out}.",
                    "task_type": "korean",
                    "segment_tag": "ko",
                    "language": "ko",
                    "_meta_source_file": str(row.get("_meta_source_file", src_path.name)).strip(),
                    "source": str(row.get("source", "")).strip(),
                }
                key = (item["input"], item["output"])
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
                stats["kept_unique"] += 1

                for _ in range(row_weight(row, inp)):
                    g.write(json.dumps(item, ensure_ascii=False) + "\n")
                    stats["written"] += 1

    MANIFEST_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT_PATH), "manifest": str(MANIFEST_PATH), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
