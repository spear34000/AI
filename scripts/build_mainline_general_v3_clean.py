from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TRAIN_DEFAULT = Path("data/mainline_general_v3_clean_train.jsonl")
EVAL_DEFAULT = Path("data/mainline_general_v3_clean_eval.jsonl")
MANIFEST_DEFAULT = Path("data/mainline_general_v3_clean.manifest.json")

WORD_RE = re.compile(r"[A-Za-z0-9가-힣]+")
HANGUL_RE = re.compile(r"[가-힣]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
ARITH_EXPR_RE = re.compile(r"\d+\s*[+\-*/=]\s*\d+")

CODE_MARKERS = ("```", "<html", "<script", "http://", "https://", "SELECT ", "INSERT ", "UPDATE ")
ARITH_STYLE_MARKERS = ("최종답", "정답:", "부분곱", "검산", "일의 자리", "십의 자리", "백의 자리", "천의 자리")
BANNED_STYLE_MARKERS = (
    "필요하면 예시도",
    "요청하면 실무 예시",
    "요청하면 관련 개념",
    "추가로 관련 개념",
    "질문 의도를 빠르게 파악해 단계별로",
)


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def digit_ratio(text: str) -> float:
    s = str(text or "")
    if not s:
        return 1.0
    return sum(ch.isdigit() for ch in s) / float(len(s))


def repetition_bad(text: str) -> bool:
    toks = WORD_RE.findall(str(text or "").lower())
    if len(toks) < 8:
        return False
    counts: Dict[str, int] = {}
    for tok in toks:
        counts[tok] = counts.get(tok, 0) + 1
    top = max(counts.values())
    if top >= 4 and top / max(1, len(toks)) >= 0.24:
        return True
    bigrams: Dict[Tuple[str, str], int] = {}
    for i in range(len(toks) - 1):
        key = (toks[i], toks[i + 1])
        bigrams[key] = bigrams.get(key, 0) + 1
    return bool(bigrams) and max(bigrams.values()) >= 3


def bad_prompt(prompt: str) -> bool:
    p = normalize(prompt)
    if len(p) < 2 or len(p) > 140:
        return True
    if "\n" in str(prompt):
        return True
    if ARITH_EXPR_RE.search(p):
        return True
    if any(tok in p for tok in ARITH_STYLE_MARKERS):
        return True
    if any(tok in p for tok in BANNED_STYLE_MARKERS):
        return True
    if any(tok in p.lower() for tok in CODE_MARKERS):
        return True
    return False


def bad_output(output: str) -> bool:
    o = normalize(output)
    if len(o) < 8 or len(o) > 240:
        return True
    if not HANGUL_RE.search(o):
        return True
    if CYRILLIC_RE.search(o):
        return True
    if digit_ratio(o) > 0.22:
        return True
    if repetition_bad(o):
        return True
    if any(tok in o for tok in ARITH_STYLE_MARKERS):
        return True
    if any(tok in o for tok in BANNED_STYLE_MARKERS):
        return True
    if any(tok in o.lower() for tok in CODE_MARKERS):
        return True
    return False


def bad_pair(prompt: str, output: str) -> bool:
    return bad_prompt(prompt) or bad_output(output)


def row_of(inp: str, out: str, source: str) -> Dict[str, str]:
    return {
        "input": normalize(inp),
        "output": normalize(out),
        "task_type": "korean",
        "segment_tag": "ko_mainline_general_v3_clean",
        "language": "ko",
        "source": source,
    }


def primer_rows(repeat: int) -> List[Dict[str, str]]:
    core = [
        ("ㅎㅇ", "안녕하세요. 무엇을 도와드릴까요?"),
        ("한국어로 한 문장 인사", "안녕하세요. 궁금한 내용을 간결하고 정확하게 설명해드리겠습니다."),
        ("짧게 도와줄 수 있는 일을 말해줘", "개념 설명, 비교 정리, 요약, 간단한 예시 제공을 도와드릴 수 있습니다."),
        ("코틀린이란?", "코틀린은 JVM과 안드로이드에서 널리 쓰이는 정적 타입 프로그래밍 언어입니다."),
        ("HTTP란?", "HTTP는 웹에서 클라이언트와 서버가 요청과 응답을 주고받는 통신 프로토콜입니다."),
        ("JWT란?", "JWT는 인증 정보를 안전하게 전달하기 위한 서명 기반 토큰 형식입니다."),
        ("가비지 컬렉션이란?", "가비지 컬렉션은 더 이상 사용하지 않는 메모리를 자동으로 회수하는 기능입니다."),
        ("SLA 99가 뭐야?", "SLA 99는 서비스 가용성을 99% 수준으로 보장한다는 의미입니다."),
        ("비엔나 소시지 생으로 먹어도 돼?", "제품 표시사항을 먼저 확인하고, 가능하면 가열해서 먹는 것이 더 안전합니다."),
        ("모르면 어떻게 답해?", "근거가 부족하면 모른다고 말하고, 확인 가능한 범위를 분명히 안내합니다."),
    ]
    out: List[Dict[str, str]] = []
    for i in range(int(repeat)):
        for q, a in core:
            out.append(row_of(q, a, f"general_v3_primer_{i % 20}"))
    return out


def collect(path: Path, tag: str, limit: int, rng: random.Random) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in iter_jsonl(path):
        inp = str(row.get("input", "")).strip()
        out = str(row.get("output", "")).strip()
        if bad_pair(inp, out):
            continue
        rows.append(row_of(inp, out, tag))
    rng.shuffle(rows)
    return rows[: min(limit, len(rows))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--primer_repeat", type=int, default=1500)
    ap.add_argument("--train_out", default=str(TRAIN_DEFAULT))
    ap.add_argument("--eval_out", default=str(EVAL_DEFAULT))
    ap.add_argument("--manifest_out", default=str(MANIFEST_DEFAULT))
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    root = Path(__file__).resolve().parent.parent
    data = root / "data"
    final_dir = data / "final_datasets"

    sources: List[Tuple[Path, str, int]] = [
        (data / "repair_ko_direct_v3.jsonl", "repair_ko_direct_v3", 70000),
        (data / "repair_ko_only_clean_v1.jsonl", "repair_ko_only_clean_v1", 70000),
        (final_dir / "한국어_일반대화_augmented_v1.jsonl", "final_ko_chat", 120000),
        (final_dir / "한국어유창성_augmented_v1.jsonl", "final_ko_fluency", 90000),
        (final_dir / "코딩기술_augmented_v1.jsonl", "final_ko_coding", 60000),
    ]

    rows: List[Dict[str, str]] = []
    source_counts: Dict[str, int] = {}
    for path, tag, limit in sources:
        if not path.exists():
            continue
        part = collect(path, tag, limit, rng)
        rows.extend(part)
        source_counts[tag] = len(part)

    seed_rows = primer_rows(int(args.primer_repeat))
    rows.extend(seed_rows)
    source_counts["primer_rows"] = len(seed_rows)

    dedup: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        key = (row["input"], row["output"])
        if key not in dedup:
            dedup[key] = row
    rows = list(dedup.values())
    rng.shuffle(rows)

    eval_size = min(4096, max(1024, int(len(rows) * 0.02)))
    eval_rows = rows[:eval_size]
    train_rows = rows[eval_size:]

    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    manifest_out = Path(args.manifest_out)
    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)

    manifest = {
        "train_out": str(train_out),
        "eval_out": str(eval_out),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "seed": int(args.seed),
        "sources": source_counts,
        "filters": {
            "remove_arithmetic_contamination": True,
            "remove_repetitive_suffix_style": True,
            "remove_code_like_samples": True,
            "max_output_len": 240,
        },
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
