from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def make_row(inp: str, out: str, tier: str = "high") -> Dict[str, str]:
    return {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": str(tier).strip().lower() or "high",
        "input": normalize_space(inp),
        "output": normalize_space(out),
    }


def add_pairs(
    rows: List[Dict[str, str]],
    seen: set[Tuple[str, str]],
    rnd: random.Random,
    prompts: Sequence[str],
    outputs: Sequence[str],
    n: int,
    tier: str = "high",
) -> None:
    if not prompts or not outputs or int(n) <= 0:
        return
    prompt_prefixes = [
        "",
        "",
        "",
        "한국어로 ",
        "간단히 ",
        "핵심만 ",
        "짧게 ",
    ]
    prompt_suffixes = [
        "",
        "",
        "",
        " 한 줄로 답해줘",
        " 간단히 답해줘",
        " 짧게 알려줘",
        " 핵심만 말해줘",
    ]
    output_tails = [
        "",
        "",
        "",
        "",
        " 필요하면 예시도 덧붙여 드릴 수 있습니다.",
        " 원하면 더 짧게도 답할 수 있습니다.",
    ]
    for _ in range(int(n)):
        base_in = normalize_space(rnd.choice(list(prompts)))
        inp = normalize_space(f"{rnd.choice(prompt_prefixes)}{base_in}{rnd.choice(prompt_suffixes)}")
        base_out = normalize_space(rnd.choice(list(outputs)))
        out = normalize_space(f"{base_out}{rnd.choice(output_tails)}")
        key = (inp, out)
        if key in seen:
            continue
        seen.add(key)
        rows.append(make_row(inp=inp, out=out, tier=tier))


def build_math_pairs(
    rows: List[Dict[str, str]],
    seen: set[Tuple[str, str]],
    rnd: random.Random,
    n: int,
) -> None:
    templates = [
        "{a}{op}{b}는?",
        "{a} {op} {b} = ?",
        "간단히 계산해줘: {a}{op}{b}",
        "{a}와 {b}를 {verb}하면?",
        "{a} {verb} {b}는 얼마야?",
    ]
    op_map = [
        ("+", "더"),
        ("-", "빼"),
    ]
    out_templates = [
        "{a}{op}{b}는 {c}입니다.",
        "결과는 {c}입니다.",
        "{c}입니다.",
        "정답은 {c}입니다.",
    ]
    for _ in range(int(n)):
        a = rnd.randint(0, 999)
        b = rnd.randint(0, 999)
        op, verb = rnd.choice(op_map)
        c = a + b if op == "+" else a - b
        inp = rnd.choice(templates).format(a=a, b=b, op=op, verb=verb)
        out = rnd.choice(out_templates).format(a=a, b=b, c=c, op=op)
        key = (normalize_space(inp), normalize_space(out))
        if key in seen:
            continue
        seen.add(key)
        rows.append(make_row(inp=inp, out=out, tier="high"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build targeted Korean fluency dataset v1")
    ap.add_argument("--out", default="data/ko_fluency_targeted_v1.jsonl")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--n_intro", type=int, default=5000)
    ap.add_argument("--n_python", type=int, default=7000)
    ap.add_argument("--n_capital", type=int, default=2500)
    ap.add_argument("--n_listcomp", type=int, default=4500)
    ap.add_argument("--n_unknown", type=int, default=2500)
    ap.add_argument("--n_math", type=int, default=8000)
    ap.add_argument("--n_ai", type=int, default=2500)
    args = ap.parse_args()

    rnd = random.Random(int(args.seed))
    rows: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    intro_prompts = [
        "한국어로 자기소개 한 줄",
        "자기소개 한 문장으로 해줘",
        "너 누구야?",
        "짧게 자기소개해줘",
        "본인 소개 부탁해",
        "한 줄 자기소개 부탁해",
        "간단히 본인 소개해줘",
        "자기소개 짧게 답해줘",
        "한국어로 너를 소개해줘",
        "한 문장으로 소개해줘",
        "짧고 정확하게 자기소개해줘",
        "너 자신을 간단히 설명해줘",
        "한 줄로 너를 알려줘",
        "너는 어떤 AI야?",
        "본인 소개를 간결하게 해줘",
    ]
    intro_outputs = [
        "안녕하세요. 질문 의도에 맞춰 간결하고 정확하게 답변하는 한국어 AI입니다.",
        "안녕하세요. 핵심을 빠르게 정리해 드리는 한국어 AI 어시스턴트입니다.",
        "반갑습니다. 필요한 정보를 짧고 명확하게 안내하는 AI입니다.",
        "안녕하세요. 실용적인 답을 중심으로 돕는 한국어 AI입니다.",
        "반갑습니다. 요청에 맞춰 정확한 답변을 제공하는 AI입니다.",
        "안녕하세요. 복잡한 내용을 이해하기 쉽게 설명하는 한국어 AI입니다.",
        "반갑습니다. 핵심 위주로 또렷하게 답하는 AI 어시스턴트입니다.",
        "안녕하세요. 질문의 요점을 파악해 빠르게 답변하는 한국어 AI입니다.",
    ]
    add_pairs(rows, seen, rnd, intro_prompts, intro_outputs, int(args.n_intro), tier="high")

    python_prompts = [
        "파이썬이란",
        "파이썬이 뭐야?",
        "파이썬 언어를 간단히 설명해줘",
        "파이썬 특징 한 줄로 알려줘",
        "파이썬은 어떤 언어야?",
        "파이썬 설명해줘",
        "파이썬 장점이 뭐야?",
        "파이썬을 왜 많이 써?",
        "파이썬에 대해 짧게 알려줘",
        "Python이란?",
        "Python 언어 소개해줘",
        "파이썬이 유명한 이유가 뭐야?",
    ]
    python_outputs = [
        "파이썬은 문법이 간결하고 가독성이 좋아 입문과 실무에 모두 널리 쓰이는 프로그래밍 언어입니다.",
        "파이썬은 배우기 쉬운 문법과 풍부한 라이브러리 덕분에 자동화, 데이터 분석, 웹 개발에 많이 사용됩니다.",
        "파이썬은 코드가 읽기 쉬워 생산성이 높고, 다양한 분야에서 빠르게 결과를 만들 수 있는 언어입니다.",
        "파이썬은 범용 프로그래밍 언어로, AI·데이터 처리·서버 개발·스크립트 자동화에 폭넓게 쓰입니다.",
        "파이썬은 단순한 문법과 큰 생태계를 갖춘 언어로, 초보자와 전문가 모두에게 인기 있는 선택입니다.",
        "파이썬은 짧은 코드로 기능을 구현하기 쉬워 개발 속도가 빠르고 유지보수도 편한 언어입니다.",
    ]
    add_pairs(rows, seen, rnd, python_prompts, python_outputs, int(args.n_python), tier="high")

    capital_prompts = [
        "대한민국 수도는?",
        "한국 수도가 어디야?",
        "대한민국의 수도를 알려줘",
        "한국의 수도는 어디인가요?",
        "수도만 짧게 답해줘: 대한민국",
        "대한민국 수도 한 단어로",
    ]
    capital_outputs = [
        "대한민국의 수도는 서울입니다.",
        "서울입니다.",
        "한국의 수도는 서울입니다.",
        "정답은 서울입니다.",
    ]
    add_pairs(rows, seen, rnd, capital_prompts, capital_outputs, int(args.n_capital), tier="high")

    listcomp_prompts = [
        "리스트 컴프리헨션 예시 하나",
        "파이썬 리스트 컴프리헨션 간단 예시 보여줘",
        "리스트 컴프리헨션이 뭔지 예제로 설명해줘",
        "한 줄로 리스트 컴프리헨션 예시",
        "list comprehension 예시",
        "리스트 컴프리헨션 코드 하나만",
    ]
    listcomp_outputs = [
        "예: `nums = [1,2,3,4]`일 때 `squares = [x*x for x in nums]`처럼 한 줄로 새 리스트를 만들 수 있습니다.",
        "예시: `evens = [x for x in range(10) if x % 2 == 0]`처럼 조건을 넣어 원하는 값만 추출할 수 있습니다.",
        "리스트 컴프리헨션은 `result = [f(x) for x in data if 조건]` 형태로 간결하게 리스트를 만드는 문법입니다.",
        "간단 예시: `words = ['a','bb']; lengths = [len(w) for w in words]`.",
        "예시 코드: `squares = [n**2 for n in range(1,6)]  # [1, 4, 9, 16, 25]`",
    ]
    add_pairs(rows, seen, rnd, listcomp_prompts, listcomp_outputs, int(args.n_listcomp), tier="high")

    unknown_prompts = [
        "모르는 건 모른다고 말해?",
        "확실하지 않으면 어떻게 답해?",
        "근거 없으면 추측하지 말아줄래?",
        "잘 모르면 솔직히 말해줘",
        "불확실한 정보는 어떻게 처리해?",
        "모를 때는 모른다고 해?",
    ]
    unknown_outputs = [
        "네. 확실하지 않은 내용은 추측하지 않고 모른다고 명확히 말씀드립니다.",
        "네. 근거가 부족하면 단정하지 않고 불확실하다고 분명히 안내합니다.",
        "맞습니다. 확인되지 않은 정보는 사실처럼 말하지 않고 제한점을 함께 설명합니다.",
        "네. 모르는 내용은 모른다고 답하고, 필요한 경우 확인 방법을 함께 제시합니다.",
    ]
    add_pairs(rows, seen, rnd, unknown_prompts, unknown_outputs, int(args.n_unknown), tier="high")

    ai_prompts = [
        "인공지능이란?",
        "AI가 뭐야?",
        "인공지능을 짧게 설명해줘",
        "AI 정의 한 문장",
        "인공지능이 어떤 기술인지 알려줘",
    ]
    ai_outputs = [
        "인공지능은 데이터를 바탕으로 학습해 예측, 분류, 생성 같은 작업을 수행하는 기술입니다.",
        "AI는 인간의 인지 과정을 모방해 문제를 해결하도록 설계된 소프트웨어 기술입니다.",
        "인공지능은 패턴을 학습해 의사결정이나 텍스트·이미지 생성 같은 기능을 수행합니다.",
        "AI는 학습된 모델을 통해 주어진 입력에서 유용한 출력을 만들어내는 기술입니다.",
    ]
    add_pairs(rows, seen, rnd, ai_prompts, ai_outputs, int(args.n_ai), tier="high")

    build_math_pairs(rows, seen, rnd, int(args.n_math))

    rnd.shuffle(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "out": str(out_path),
                "rows": int(len(rows)),
                "seed": int(args.seed),
                "config": {
                    "n_intro": int(args.n_intro),
                    "n_python": int(args.n_python),
                    "n_capital": int(args.n_capital),
                    "n_listcomp": int(args.n_listcomp),
                    "n_unknown": int(args.n_unknown),
                    "n_math": int(args.n_math),
                    "n_ai": int(args.n_ai),
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
