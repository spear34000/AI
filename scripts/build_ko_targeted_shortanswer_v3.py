from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SPACE_RE = re.compile(r"\s+")
HANGUL_RE = re.compile(r"[가-힣]")


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def make_row(inp: str, out: str, tier: str = "high", source: str = "synthetic") -> Dict[str, str]:
    return {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": tier,
        "input": normalize_space(inp),
        "output": normalize_space(out),
        "_meta_source_file": source,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Korean targeted short-answer dataset v3")
    p.add_argument("--hq", default="data/quality/hq_ko_chat_v1.jsonl")
    p.add_argument("--out", default="data/ko_targeted_shortanswer_v3.jsonl")
    p.add_argument("--manifest", default="data/ko_targeted_shortanswer_v3.manifest.json")
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--hq_take", type=int, default=30000)
    p.add_argument("--synthetic_take", type=int, default=26000)
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict]:
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


def is_good_hq_pair(inp: str, out: str) -> bool:
    if not inp or not out:
        return False
    if len(inp) < 2 or len(inp) > 400:
        return False
    if len(out) < 4 or len(out) > 220:
        return False
    if not HANGUL_RE.search(inp + out):
        return False
    if re.search(r"(.)\1{10,}", out):
        return False
    banned = [
        "질문 의도를 파악해 단계별로 정리해드릴게요. 원하면 짧게 3줄 요약으로도 답할 수 있어요.",
        "안녕하세요, 질문을 이해하고 간결하게 답하는 한국어 AI 어시스턴트입니다.",
    ]
    s = normalize_space(out)
    if any(b in s for b in banned):
        return False
    return True


def pick_hq_rows(path: Path, take: int, seed: int) -> List[Dict]:
    rows: List[Dict] = []
    for row in iter_jsonl(path):
        inp = normalize_space(str(row.get("input", "")))
        out = normalize_space(str(row.get("output", "")))
        if not is_good_hq_pair(inp, out):
            continue
        rows.append(make_row(inp, out, tier="high", source=str(path).replace("\\", "/")))
    rnd = random.Random(int(seed))
    rnd.shuffle(rows)
    return rows[: int(take)] if int(take) > 0 else rows


def add_pairs(
    dst: List[Dict],
    seen: set[bytes],
    prompts: Sequence[str],
    outputs: Sequence[str],
    n: int,
    rnd: random.Random,
) -> int:
    if int(n) <= 0:
        return 0
    added = 0
    prefixes = ["", "", "간단히 ", "짧게 ", "한국어로 ", "한 줄로 ", "핵심만 ", "바로 "]
    suffixes = ["", "", " 설명해줘", " 알려줘", " 답해줘", " 부탁해", " 요약해줘", " 짧게 답해줘"]
    output_prefixes = ["", "", "요약하면 ", "한 줄로 말하면 ", "핵심은 ", "간단히 말해 "]
    output_suffixes = [
        "",
        "",
        " 필요하면 예시도 덧붙여 드릴게요.",
        " 원하면 더 짧게 정리해 드릴 수 있어요.",
        " 추가로 관련 개념도 이어서 설명할 수 있어요.",
        " 요청하면 실무 예시로도 풀어드릴게요.",
    ]
    for _ in range(int(n)):
        p = normalize_space(f"{rnd.choice(prefixes)}{rnd.choice(list(prompts))}{rnd.choice(suffixes)}")
        base_out = normalize_space(rnd.choice(list(outputs)))
        o = normalize_space(f"{rnd.choice(output_prefixes)}{base_out}{rnd.choice(output_suffixes)}")
        k = dedupe_key(p, o)
        if k in seen:
            continue
        seen.add(k)
        dst.append(make_row(p, o, tier="high"))
        added += 1
    return added


def add_math(dst: List[Dict], seen: set[bytes], n: int, rnd: random.Random) -> int:
    added = 0
    for _ in range(int(n)):
        a = rnd.randint(1, 999)
        b = rnd.randint(1, 999)
        op = rnd.choice(["+", "-"])
        c = a + b if op == "+" else a - b
        p = rnd.choice(
            [
                f"{a}{op}{b}는?",
                f"{a} {op} {b} = ?",
                f"{a}{op}{b} 계산해줘",
                f"{a} {op} {b} 값만 답해줘",
            ]
        )
        o = rnd.choice(
            [
                f"{c}입니다.",
                f"정답은 {c}입니다.",
                f"{a}{op}{b}의 결과는 {c}입니다.",
            ]
        )
        k = dedupe_key(p, o)
        if k in seen:
            continue
        seen.add(k)
        dst.append(make_row(p, o, tier="high"))
        added += 1
    return added


def main() -> None:
    args = parse_args()
    rnd = random.Random(int(args.seed))

    hq_path = Path(args.hq)
    if not hq_path.exists():
        raise FileNotFoundError(f"hq dataset not found: {hq_path}")

    rows: List[Dict] = []
    seen: set[bytes] = set()

    hq_rows = pick_hq_rows(path=hq_path, take=int(args.hq_take), seed=int(args.seed) + 1)
    for row in hq_rows:
        k = dedupe_key(row["input"], row["output"])
        if k in seen:
            continue
        seen.add(k)
        rows.append(row)

    intro_prompts = [
        "자기소개 한 줄",
        "한국어로 자기소개 한 줄",
        "짧게 자기소개",
        "한 문장으로 본인 소개",
        "너를 한 줄로 소개해줘",
    ]
    intro_outputs = [
        "반갑습니다. 질문의 핵심을 빠르게 파악해 간결하고 정확하게 답변하는 한국어 AI 도우미입니다.",
        "안녕하세요. 필요한 정보를 짧고 분명하게 정리해 드리는 한국어 AI 어시스턴트입니다.",
        "반갑습니다. 요청 의도에 맞춰 핵심만 명확하게 전달하는 한국어 AI입니다.",
        "안녕하세요. 복잡한 내용을 이해하기 쉽게 정리해 답하는 한국어 AI 도우미입니다.",
    ]

    python_prompts = [
        "파이썬이란",
        "파이썬이 뭐야",
        "Python이란",
        "파이썬 설명해줘",
        "파이썬을 한 줄로 설명해줘",
    ]
    python_outputs = [
        "파이썬은 문법이 간결하고 읽기 쉬워 웹, 데이터 분석, 자동화, AI 개발에 널리 쓰이는 프로그래밍 언어입니다.",
        "파이썬은 생산성이 높고 라이브러리가 풍부해 입문과 실무 모두에서 많이 사용하는 범용 프로그래밍 언어입니다.",
        "파이썬은 코드 가독성이 높아 빠르게 개발할 수 있고 다양한 분야에 적용되는 인기 언어입니다.",
    ]

    capital_prompts = [
        "대한민국 수도는?",
        "한국의 수도는?",
        "대한민국 수도 알려줘",
        "한국 수도 정답만",
    ]
    capital_outputs = [
        "대한민국의 수도는 서울입니다.",
        "서울입니다.",
        "정답은 서울입니다.",
    ]

    ai_prompts = ["AI란?", "인공지능이 뭐야", "AI를 간단히 설명해줘", "인공지능 정의"]
    ai_outputs = [
        "인공지능(AI)은 데이터를 바탕으로 학습해 예측, 분류, 생성 같은 지능적 작업을 수행하는 기술입니다.",
        "인공지능은 인간의 인지 능력을 모사해 문제 해결과 의사결정을 돕는 소프트웨어 기술입니다.",
    ]

    db_prompts = ["데이터베이스란?", "DB가 뭐야", "데이터베이스 한 줄 설명"]
    db_outputs = [
        "데이터베이스는 데이터를 구조적으로 저장하고 빠르게 조회·수정할 수 있게 관리하는 시스템입니다.",
        "데이터베이스는 정보를 체계적으로 보관해 필요한 데이터를 효율적으로 처리하는 저장소입니다.",
    ]

    unknown_prompts = [
        "모르면 모른다고 말해줘",
        "확실하지 않으면 어떻게 답해?",
        "추측하지 말고 답해줘",
    ]
    unknown_outputs = [
        "확실하지 않은 정보는 추측하지 않고 모른다고 분명히 말한 뒤, 확인 방법을 함께 안내하겠습니다.",
        "근거가 부족하면 단정하지 않고 불확실성을 먼저 밝힌 다음 필요한 정보를 요청하겠습니다.",
    ]

    counts = {
        "intro": add_pairs(rows, seen, intro_prompts, intro_outputs, 4200, rnd),
        "python": add_pairs(rows, seen, python_prompts, python_outputs, 4200, rnd),
        "capital": add_pairs(rows, seen, capital_prompts, capital_outputs, 1800, rnd),
        "ai": add_pairs(rows, seen, ai_prompts, ai_outputs, 3000, rnd),
        "db": add_pairs(rows, seen, db_prompts, db_outputs, 2600, rnd),
        "unknown": add_pairs(rows, seen, unknown_prompts, unknown_outputs, 2200, rnd),
        "math": add_math(rows, seen, 4200, rnd),
    }

    # Top up synthetic rows if dedupe reduced counts more than expected.
    synthetic_target = int(args.synthetic_take)
    synthetic_now = sum(counts.values())
    if synthetic_now < synthetic_target:
        filler_prompts = [
            "요약이란?",
            "알고리즘이 뭐야",
            "API가 뭐야",
            "클라우드 컴퓨팅이란?",
            "테스트 코드의 목적은?",
        ]
        filler_outputs = [
            "요약은 핵심 정보만 추려 원문보다 짧고 이해하기 쉽게 전달하는 작업입니다.",
            "알고리즘은 문제를 해결하기 위한 절차를 규칙화한 단계적 방법입니다.",
            "API는 서로 다른 소프트웨어가 정해진 방식으로 기능과 데이터를 주고받게 하는 인터페이스입니다.",
            "클라우드 컴퓨팅은 인터넷을 통해 서버·저장소·연산 자원을 필요할 때 제공받는 방식입니다.",
            "테스트 코드는 기능 회귀를 빠르게 탐지해 안정적인 배포를 가능하게 합니다.",
        ]
        _ = add_pairs(rows, seen, filler_prompts, filler_outputs, synthetic_target - synthetic_now, rnd)

    rnd.shuffle(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "name": "ko_targeted_shortanswer_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "inputs": {
            "hq": str(hq_path).replace("\\", "/"),
        },
        "rows": {
            "hq": len(hq_rows),
            "final": len(rows),
        },
        "synthetic": counts,
        "config": {
            "hq_take": int(args.hq_take),
            "synthetic_take": int(args.synthetic_take),
        },
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] rows={len(rows)} -> {out_path}")
    print(f"[done] manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
