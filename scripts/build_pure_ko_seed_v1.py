from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def make_row(user_input: str, assistant_output: str, tier: str = "high") -> dict:
    return {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": tier,
        "input": user_input,
        "output": assistant_output,
    }


BASE_PAIRS = [
    ("한국어로 자기소개 한 줄", "안녕하세요. 한국어로 정확하고 짧게 답변하는 AI 어시스턴트입니다."),
    ("자기소개 해줘", "안녕하세요. 필요한 정보를 빠르고 정확하게 정리해 드리는 AI 어시스턴트입니다."),
    ("파이썬이란", "파이썬은 문법이 읽기 쉬워서 자동화, 데이터 분석, 웹 개발에 널리 쓰이는 범용 프로그래밍 언어입니다."),
    ("파이썬이 뭐야", "파이썬은 배우기 쉬운 문법과 풍부한 라이브러리를 가진 프로그래밍 언어입니다."),
    ("AI란 무엇인가", "AI는 컴퓨터가 학습과 추론을 통해 문제를 해결하도록 만드는 기술 분야입니다."),
    ("머신러닝이 뭐야", "머신러닝은 데이터에서 패턴을 학습해 예측이나 분류를 수행하는 AI의 한 분야입니다."),
    ("딥러닝이 뭐야", "딥러닝은 여러 층의 신경망으로 복잡한 패턴을 학습하는 머신러닝 기법입니다."),
    ("리스트와 튜플 차이", "리스트는 수정 가능하고, 튜플은 생성 후 내용을 바꿀 수 없는 시퀀스입니다."),
    ("파이썬 리스트 컴프리헨션 예시", "예시: nums=[1,2,3,4], squares=[x*x for x in nums]처럼 한 줄로 새 리스트를 만들 수 있습니다."),
    ("함수와 메서드 차이", "함수는 독립적으로 호출되고, 메서드는 객체에 속해 객체 상태와 함께 동작합니다."),
    ("변수명이 중요한 이유", "좋은 변수명은 코드 의도를 빠르게 이해하게 해 유지보수 비용을 줄여줍니다."),
    ("에러 로그 읽는 법", "에러 로그는 마지막 예외 타입과 메시지, 스택트레이스의 첫 실패 지점을 먼저 확인하는 것이 핵심입니다."),
    ("디버깅 순서 알려줘", "재현 조건 고정, 입력 확인, 로그 추가, 가설 검증, 최소 수정, 회귀 테스트 순서로 진행하면 효율적입니다."),
    ("코드 리뷰 포인트", "정확성, 예외 처리, 경계값, 가독성, 테스트 누락 여부를 우선 점검하는 것이 좋습니다."),
    ("REST API 한 줄 설명", "REST API는 자원을 URL로 표현하고 HTTP 메서드로 조작하는 인터페이스 설계 방식입니다."),
    ("SQL 인덱스가 뭐야", "인덱스는 조회 속도를 높이기 위한 자료구조로, 읽기 성능을 높이는 대신 쓰기 비용이 늘 수 있습니다."),
    ("정렬 알고리즘 왜 배우나", "정렬 알고리즘은 시간복잡도와 자료구조 선택 감각을 키워 실전 문제 해결 능력을 높여줍니다."),
    ("빅오 표기법 설명", "빅오 표기법은 입력 크기가 커질 때 알고리즘 실행 시간이나 메모리 증가 경향을 나타냅니다."),
    ("재귀 함수 주의점", "재귀는 종료 조건이 명확해야 하며, 호출 깊이가 깊으면 스택 오버플로를 주의해야 합니다."),
    ("테스트 코드 왜 필요해", "테스트 코드는 버그를 조기에 발견하고 수정 후 회귀를 막아 개발 속도를 안정적으로 높여줍니다."),
    ("한국어로 답해줘", "알겠습니다. 앞으로 한국어로 명확하고 간결하게 답변하겠습니다."),
    ("짧게 답해줘", "알겠습니다. 핵심만 짧게 답변하겠습니다."),
    ("친절하게 설명해줘", "좋습니다. 쉬운 표현으로 단계별로 친절하게 설명해 드리겠습니다."),
    ("고마워", "천만에요. 더 필요한 내용이 있으면 편하게 말씀해 주세요."),
    ("안녕", "안녕하세요. 어떤 도움이 필요하신가요?"),
    ("학습 계획 짜줘", "목표를 작게 나누고 매일 고정 시간에 복습과 실습을 반복하는 계획이 가장 효과적입니다."),
    ("집중이 안 돼", "작업을 25분 단위로 나누고 방해 요소를 끄는 것부터 시작하면 집중 회복에 도움이 됩니다."),
    ("문서 요약해줘", "원문을 보내주시면 핵심, 근거, 결론 순서로 짧고 정확하게 요약해 드리겠습니다."),
    ("체크리스트 만들어줘", "목표와 마감일을 알려주시면 실행 가능한 체크리스트로 정리해 드리겠습니다."),
    ("면접 준비 방법", "자기소개, 핵심 프로젝트 2개, 실패 경험 1개를 STAR 구조로 정리하면 준비 효율이 높습니다."),
]

OUTPUT_VARIANTS: Dict[str, List[str]] = {
    "한국어로 자기소개 한 줄": [
        "안녕하세요. 한국어로 정확하고 짧게 답변하는 AI 어시스턴트입니다.",
        "안녕하세요. 핵심 위주로 명확하게 안내하는 AI 어시스턴트입니다.",
        "반갑습니다. 요청 의도에 맞춰 간단명료하게 답변하는 AI입니다.",
        "안녕하세요. 필요한 정보를 빠르게 정리해 드리는 한국어 AI입니다.",
        "반갑습니다. 질문에 맞춰 실용적으로 답하는 AI 도우미입니다.",
    ],
    "자기소개 해줘": [
        "안녕하세요. 한국어로 정확하고 짧게 답변하는 AI 어시스턴트입니다.",
        "안녕하세요. 핵심 위주로 명확하게 안내하는 AI 어시스턴트입니다.",
        "반갑습니다. 요청 의도에 맞춰 간단명료하게 답변하는 AI입니다.",
    ],
}


STYLE_SUFFIXES = [
    "",
    " 한 문장으로 답해줘.",
    " 짧게 설명해줘.",
    " 쉽게 설명해줘.",
    " 핵심만 말해줘.",
]


def build_rows(seed: int, repeats: int) -> list[dict]:
    rnd = random.Random(seed)
    rows: list[dict] = []

    for inp, out in BASE_PAIRS:
        outputs = OUTPUT_VARIANTS.get(inp, [out])
        for ans in outputs:
            rows.append(make_row(inp, ans, tier="high"))
            rows.append(make_row(inp + "?", ans, tier="high"))
            rows.append(make_row(inp.replace("?", ""), ans, tier="high"))
        for suffix in STYLE_SUFFIXES:
            if not suffix:
                continue
            for ans in outputs:
                rows.append(make_row(inp + suffix, ans, tier="mid"))

    # Deterministic expansion to make the dataset dense enough for quick tuning.
    expanded = list(rows)
    for _ in range(max(1, int(repeats))):
        for inp, out in BASE_PAIRS:
            outputs = OUTPUT_VARIANTS.get(inp, [out])
            ans = rnd.choice(outputs)
            suffix = rnd.choice(STYLE_SUFFIXES)
            tier = "high" if rnd.random() < 0.6 else "mid"
            expanded.append(make_row(inp + suffix, ans, tier=tier))

    rnd.shuffle(expanded)
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/pure_ko_seed_v1.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()

    rows = build_rows(seed=int(args.seed), repeats=int(args.repeats))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "out": str(out_path),
                "rows": len(rows),
                "seed": int(args.seed),
                "repeats": int(args.repeats),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
