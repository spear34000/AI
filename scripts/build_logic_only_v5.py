from __future__ import annotations

import json
import re
from pathlib import Path


SRC_PATHS = [
    Path("data/final_datasets/논리추론_augmented_v1.jsonl"),
    Path("data/final_datasets/한국어_일반대화_augmented_v1.jsonl"),
]
OUT_PATH = Path("data/logic_only_v5.jsonl")
MANIFEST_PATH = Path("data/logic_only_v5.manifest.json")

BANNED_SOURCES = {
    "hf_mit_language_superpack_v1",
    "cot_explain",
    "cot_anti",
    "capability_pack_v1",
    "capability_pack_v2",
}

INPUT_ALLOW_PATTERNS = [
    r"모든 .+는 .+다|일부 .+는 .+다|반례|증명|반드시 성립|참인가|거짓인가|판단하",
    r"보다 .+크|보다 .+작|보다 .+빠르|보다 .+무겁|가장 큰|가장 작은|누가",
    r"거짓말쟁이|진실화자|진실만 말하|항상참|항상거짓|무작위|da/ja",
    r"수열|다음 수|누락된 숫자|규칙 기반",
    r"값은|계산|몇 개|몇 명|몇 권|몇 페이지|몇 분|몇 시간|합은|평균은|남은|퍼센트|비율|배수",
    r"\d.*[+\-*/=^<>()].*\d",
]

INPUT_BANNED_PATTERNS = [
    r"체크리스트|상황에서|습관|말투|자기소개|정체가 뭐야|너의 이름|오늘 날씨",
    r"프로젝트 운영|실무 상황|초보자 상황|장기 과제 상황|질문 해석|오류를 줄이는",
    r"TCP|UDP|API|데이터베이스|REST|HTTP|HTTPS|로드 밸런싱|캐시|마이크로서비스",
    r"코드를 작성|프로그램을 작성|파이썬|python|C 프로그램",
    r"뭐야\??$|무엇인가요\??$|무엇입니까\??$",
]

OUTPUT_HINTS = ["1.", "2.", "전제", "조건", "추론", "결론", "계산", "따라서", "그러므로"]
OUTPUT_BANNED_PATTERNS = [
    r"^\s*정답은\b",
    r"^\s*FINAL\b|^\s*ACTION\b|^\s*OBSERVATION\b",
    r"필요하면|모르겠습니다|확실하지 않아|spear1\.0",
    r"```|\ufffd",
]


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
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def allowed_row(row: dict, src_path: Path) -> bool:
    source = str(row.get("source", "")).strip()
    meta = str(row.get("_meta_source_file", "")).strip()
    if src_path.name == "논리추론_augmented_v1.jsonl":
        return source not in BANNED_SOURCES
    if src_path.name == "한국어_일반대화_augmented_v1.jsonl":
        return meta in {"logic_reasoning_cot_v1.jsonl", "synthetic/deintro_focus_v1_logic"}
    return False


def valid_input(text: str) -> bool:
    text = compact(text)
    if len(text) < 8 or len(text) > 260:
        return False
    if any(re.search(pat, text) for pat in INPUT_BANNED_PATTERNS):
        return False
    return any(re.search(pat, text) for pat in INPUT_ALLOW_PATTERNS)


def normalize_output(text: str) -> str:
    text = compact(text)
    text = re.sub(r"^\s*답:\s*", "", text)
    return text


def valid_output(text: str) -> bool:
    text = normalize_output(text)
    if len(text) < 18 or len(text) > 420:
        return False
    if any(re.search(pat, text) for pat in OUTPUT_BANNED_PATTERNS):
        return False
    return any(hint in text for hint in OUTPUT_HINTS)


def row_weight(row: dict, inp: str) -> int:
    source = str(row.get("source", "")).strip()
    meta = str(row.get("_meta_source_file", "")).strip()
    weight = 1
    if source in {"cot_logic", "logic_v1", "cot_arith"}:
        weight += 1
    if meta == "logic_reasoning_cot_v1.jsonl":
        weight += 1
    if meta == "synthetic/deintro_focus_v1_logic":
        weight += 2
    if re.search(r"거짓말쟁이|진실화자|모든 .+는 .+다|일부 .+는 .+다|반례|증명", inp):
        weight += 1
    return min(weight, 4)


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

                rec = {
                    "input": inp,
                    "output": out if out.endswith((".", "!", "?")) else f"{out}.",
                    "task_type": "korean",
                    "segment_tag": "ko",
                    "language": "ko",
                    "_meta_source_file": str(row.get("_meta_source_file", src_path.name)).strip(),
                    "source": str(row.get("source", "")).strip(),
                }
                key = (rec["input"], rec["output"])
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
                stats["kept_unique"] += 1

                for _ in range(row_weight(row, inp)):
                    g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["written"] += 1

    MANIFEST_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT_PATH), "manifest": str(MANIFEST_PATH), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
