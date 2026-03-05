from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TRAIN_DEFAULT = Path("data/mainline_general_v1_train.jsonl")
EVAL_DEFAULT = Path("data/mainline_general_v1_eval.jsonl")
MANIFEST_DEFAULT = Path("data/mainline_general_v1.manifest.json")

WORD_RE = re.compile(r"[A-Za-z0-9\uac00-\ud7a3]+")
HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
ARITH_OP_RE = re.compile(r"\d+\s*[+\-*/=]\s*\d+")
EDGE_PUNCT_RE = re.compile(r"^[^A-Za-z0-9\uac00-\ud7a3]+|[^A-Za-z0-9\uac00-\ud7a3]+$")

CODE_MARKERS = [
    "```",
    "<html",
    "docs_src",
    "giphy",
    "http://",
    "https://",
    "{*",
    "*}",
    "<script",
]
BAD_PROMPT_MARKERS = [
    "다음 문서",
    "핵심 키워드",
    "이어 쓰",
    "선택지",
    "### 질문",
    "### 정답",
    "정답:",
    "증명",
    "반례",
    "검산",
    "최종답",
    "부분곱",
]
BAD_OUTPUT_MARKERS = [
    "요청하면 실무 예시",
    "요청하면 관련 개념",
    "원하면 더 짧게",
    "추가로 관련 개념",
    "질문의 핵심을 빠르게 파악해",
    "### 정답",
    "최종답",
    "부분곱",
    "검산",
]
DEF_CUES = ["이란", "란", "가 뭐야", "는 뭐야", "이 뭐야", "정의"]


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def has_code_like(text: str) -> bool:
    low = str(text or "").lower()
    return any(tok in low for tok in CODE_MARKERS)


def digit_ratio(text: str) -> float:
    s = str(text or "")
    if not s:
        return 1.0
    return sum(ch.isdigit() for ch in s) / float(len(s))


def weird_ratio(text: str) -> float:
    s = str(text or "")
    if not s:
        return 1.0
    weird = sum(1 for ch in s if ord(ch) < 32 and ch not in "\t\n\r")
    return weird / float(len(s))


def repetition_bad(text: str) -> bool:
    toks = WORD_RE.findall(str(text or "").lower())
    if len(toks) < 7:
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


def extract_term_from_prompt(prompt: str) -> str | None:
    q = normalize_space(prompt)
    cut = -1
    for cue in DEF_CUES:
        idx = q.find(cue)
        if idx > 0:
            cut = idx if cut < 0 else min(cut, idx)
    if cut <= 0:
        return None
    left = q[:cut].strip()
    if not left:
        return None
    term = left.split()[-1]
    term = EDGE_PUNCT_RE.sub("", term)
    if 2 <= len(term) <= 24:
        return term
    return None


def definition_mismatch(prompt: str, output: str) -> bool:
    term = extract_term_from_prompt(prompt)
    if not term:
        return False
    out = normalize_space(output)
    low_term = term.lower()
    low_out = out.lower()
    if low_term in low_out:
        return False
    starts = (f"{term}은", f"{term}는", f"{term}이", f"{term}가")
    return not out.startswith(starts)


def is_bad_prompt(prompt: str) -> bool:
    p = normalize_space(prompt)
    if len(p) < 2 or len(p) > 96:
        return True
    if "\n" in prompt:
        return True
    if has_code_like(p):
        return True
    if ARITH_OP_RE.search(p):
        return True
    if any(tok in p for tok in BAD_PROMPT_MARKERS):
        return True
    if digit_ratio(p) > 0.30:
        return True
    return False


def is_bad_output(output: str) -> bool:
    o = normalize_space(output)
    if len(o) < 8 or len(o) > 220:
        return True
    if has_code_like(o):
        return True
    if any(tok in o for tok in BAD_OUTPUT_MARKERS):
        return True
    if CYRILLIC_RE.search(o):
        return True
    if digit_ratio(o) > 0.22:
        return True
    if weird_ratio(o) > 0.03:
        return True
    if not HANGUL_RE.search(o):
        return True
    if repetition_bad(o):
        return True
    return False


def is_bad_pair(prompt: str, output: str) -> bool:
    if is_bad_prompt(prompt):
        return True
    if is_bad_output(output):
        return True
    if definition_mismatch(prompt, output):
        return True
    return False


def row_from_pair(prompt: str, output: str, source: str) -> Dict[str, str]:
    return {
        "input": normalize_space(prompt),
        "output": normalize_space(output),
        "task_type": "korean",
        "segment_tag": "ko_mainline_general_v1",
        "language": "ko",
        "source": source,
    }


def primer_rows(repeat: int = 800) -> List[Dict[str, str]]:
    core = [
        ("ㅎㅇ", "안녕하세요."),
        ("안녕", "안녕하세요."),
        ("한국어로 한 문장 인사", "안녕하세요, 궁금한 점을 짧고 분명하게 도와드리겠습니다."),
        ("짧게 도와줄 수 있는 일을 말해줘", "질문의 핵심을 정리하고 필요한 설명을 간결하게 제공합니다."),
        ("코틀린이란?", "코틀린은 JVM과 안드로이드에서 널리 쓰이는 정적 타입 프로그래밍 언어입니다."),
        ("HTTP란?", "HTTP는 웹에서 클라이언트와 서버가 요청과 응답을 주고받는 통신 프로토콜입니다."),
        ("JWT란?", "JWT는 인증 정보를 안전하게 전달하기 위해 사용하는 서명 기반 토큰 형식입니다."),
        ("가비지 컬렉션이란?", "가비지 컬렉션은 더 이상 사용하지 않는 메모리를 자동으로 회수하는 기능입니다."),
        ("SLA 99가 뭐야?", "SLA 99는 서비스 가용성을 99% 수준으로 보장한다는 의미입니다."),
        ("비엔나 소시지 생으로 먹어도 돼?", "제품 표시사항을 먼저 확인하고, 일반적으로는 가열해서 먹는 것이 안전합니다."),
        ("모르겠으면 어떻게 답해?", "확실하지 않으면 모른다고 말하고 확인 가능한 방법을 함께 안내합니다."),
    ]
    out: List[Dict[str, str]] = []
    for idx in range(repeat):
        for prompt, output in core:
            out.append(row_from_pair(prompt, output, f"seed_primer_{idx % 8}"))
    return out


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def collect_pair_rows(path: Path, source_tag: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in iter_jsonl(path):
        prompt = str(row.get("input", "")).strip()
        output = str(row.get("output", "")).strip()
        if is_bad_pair(prompt, output):
            continue
        rows.append(row_from_pair(prompt, output, source_tag))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_out", default=str(TRAIN_DEFAULT))
    ap.add_argument("--eval_out", default=str(EVAL_DEFAULT))
    ap.add_argument("--manifest_out", default=str(MANIFEST_DEFAULT))
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    final_dir = data_dir / "final_datasets"

    sources: List[Tuple[Path, str, int]] = [
        (data_dir / "repair_ko_direct_v3.jsonl", "repair_ko_direct_v3", 120000),
        (data_dir / "repair_ko_only_clean_v1.jsonl", "repair_ko_only_clean_v1", 120000),
        (final_dir / "한국어유창성_augmented_v1.jsonl", "한국어유창성_augmented_v1", 60000),
    ]

    rows: List[Dict[str, str]] = []
    source_counts: Dict[str, int] = {}

    for path, tag, limit in sources:
        if not path.exists():
            continue
        part = collect_pair_rows(path, tag)
        rng.shuffle(part)
        part = part[:limit]
        rows.extend(part)
        source_counts[tag] = len(part)

    dedup: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        key = (row["input"], row["output"])
        if key not in dedup:
            dedup[key] = row
    rows = list(dedup.values())

    primer = primer_rows(repeat=3000)
    rows.extend(primer)
    source_counts["seed_primer_rows"] = len(primer)

    rng.shuffle(rows)

    eval_size = min(2048, max(512, int(len(rows) * 0.02)))
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
            "remove_template_suffixes": True,
            "remove_long_doc_tasks": True,
            "definition_mismatch_filter": True,
            "reject_cyrillic_outputs": True,
        },
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
