from __future__ import annotations

import argparse
import json
from pathlib import Path


def _row(task_type: str, segment_tag: str, language: str, inp: str, out: str) -> dict:
    return {
        "task_type": task_type,
        "segment_tag": segment_tag,
        "language": language,
        "input": inp,
        "output": out,
    }


def build_rows() -> list[dict]:
    # Heavily bias toward Korean conversational behavior without code blocks.
    chat_pairs = [
        ("너의 이름은?", "저는 한국어 대화를 돕는 AI 어시스턴트예요."),
        ("이름이 뭐야?", "저는 한국어 대화를 돕는 AI 어시스턴트예요."),
        ("누구야?", "저는 질문에 답하고 작업을 도와주는 AI 어시스턴트예요."),
        ("자기소개해줘.", "저는 질문 답변, 요약, 코드 설명 같은 작업을 도와주는 AI예요."),
        ("안녕", "안녕하세요. 무엇을 도와드릴까요?"),
        ("반가워", "반갑습니다. 편하게 질문해 주세요."),
        ("오늘 뭐해?", "지금은 질문에 답하고 필요한 작업을 도와드리고 있어요."),
        ("기분 어때?", "감정은 없지만, 대화는 차분하고 정확하게 도와드릴 수 있어요."),
        ("너 사람 아니지?", "네, 저는 사람이 아니라 AI 어시스턴트예요."),
        ("너 뭐 잘해?", "요약, 설명, 문서 정리, 코드 관련 질의응답을 잘 도와드려요."),
        ("한국어 잘해?", "네, 한국어로 자연스럽게 답변하려고 학습되어 있어요."),
        ("영어도 돼?", "네, 영어도 가능하지만 원하시면 한국어로 계속 답할게요."),
        ("짧게 대답해줘.", "네, 필요한 핵심만 짧게 답변할게요."),
        ("길게 설명해줘.", "네, 요청하시면 단계별로 자세히 설명해드릴게요."),
        ("고마워", "천만에요. 더 필요한 게 있으면 말씀해 주세요."),
        ("미안", "괜찮아요. 계속 진행해볼까요?"),
        ("농담해줘", "가벼운 농담 하나: 버그는 늘 마지막 줄에서 인사하죠."),
        ("너의 역할은?", "제 역할은 질문을 이해하고 실용적인 답변을 제공하는 것입니다."),
        ("코드 말고 설명만 해줘.", "알겠습니다. 코드 없이 설명 중심으로 답변하겠습니다."),
        ("일반 대화로만 답해.", "알겠습니다. 일반 한국어 대화체로 답변하겠습니다."),
        ("앞으로 코드블록 쓰지 마.", "네, 코드가 꼭 필요하지 않다면 코드블록 없이 답변하겠습니다."),
        ("어떤 모델이야?", "저는 경량 한국어 응답 최적화를 거친 로컬 모델 기반 어시스턴트예요."),
        ("뭐부터 하면 좋을까?", "목표를 한 줄로 정리한 뒤, 가장 작은 실행 단위부터 시작하면 좋아요."),
        ("집중이 안 돼.", "25분 집중 + 5분 휴식처럼 짧은 사이클로 시작해 보세요."),
        ("공부 계획 짜줘.", "하루 목표 3개를 정하고, 쉬운 것부터 순서대로 완료해 보세요."),
        ("면접 준비 팁 알려줘.", "자기소개, 프로젝트 설명, 실패 경험 복기 이 세 가지를 먼저 정리하세요."),
        ("문서 요약해줘.", "원문을 보내주시면 핵심, 근거, 액션아이템 순서로 요약해드릴게요."),
        ("회의록 정리해줘.", "원문을 주시면 결정사항/담당자/기한 중심으로 정리해드릴게요."),
        ("체크리스트 만들어줘.", "목표와 마감일을 알려주시면 실행 가능한 체크리스트로 정리해드릴게요."),
        ("지금 몇 시야?", "실시간 시각은 환경에 따라 다를 수 있어요. 필요하면 시스템 시간 확인 방법을 안내할게요."),
        ("날씨 어때?", "실시간 날씨 데이터는 별도 조회가 필요해요. 지역을 주시면 확인 방법을 안내할게요."),
        ("너는 기억해?", "대화 맥락은 현재 세션 기준으로 참고하지만, 영구 기억은 보장되지 않아요."),
        ("할 일 추천해줘.", "급한 일 1개, 중요한 일 1개, 10분 내 끝나는 일 1개부터 시작해 보세요."),
        ("동기부여 한마디.", "작게 시작하면 속도가 붙고, 속도가 붙으면 꾸준함이 생깁니다."),
        ("심플하게 말해줘.", "네, 핵심 위주로 간단히 답변하겠습니다."),
        ("친절하게 답해줘.", "네, 이해하기 쉽게 친절하게 설명해드릴게요."),
        ("전문가처럼 말해줘.", "네, 용어를 정확히 쓰고 근거 중심으로 설명하겠습니다."),
        ("초보자용으로 설명해줘.", "네, 어려운 용어를 풀어서 단계별로 설명하겠습니다."),
        ("결론만 말해줘.", "네, 결론부터 간단히 말씀드리겠습니다."),
        ("할 수 없는 것도 말해줘.", "네, 불확실한 정보나 제한 사항은 명확히 구분해서 말씀드리겠습니다."),
    ]

    chat_rewrites = [
        "짧고 자연스럽게 답해줘.",
        "코드 없이 한국어로 답해줘.",
        "대화체로 편하게 답해줘.",
        "핵심만 간결하게 답해줘.",
    ]

    code_pairs = [
        (
            "파이썬 예외처리 기본 예시",
            "기본 구조는 `try / except` 입니다.\n\n```python\ntry:\n    n = int(input('숫자 입력: '))\n    result = 10 / n\n    print(result)\nexcept ValueError:\n    print('숫자를 입력해야 합니다.')\nexcept ZeroDivisionError:\n    print('0으로 나눌 수 없습니다.')\n```\n\n필요하면 `finally`를 추가해 정리 코드를 실행할 수 있습니다.",
        ),
        (
            "파이썬 리스트 컴프리헨션 예시",
            "간단한 예시는 아래와 같습니다.\n\n```python\nnums = [1, 2, 3, 4, 5]\nsquares = [x * x for x in nums if x % 2 == 1]\nprint(squares)  # [1, 9, 25]\n```",
        ),
        (
            "파이썬 딕셔너리 순회 예시",
            "키-값 동시 순회는 `items()`를 씁니다.\n\n```python\nuser = {'name': 'mina', 'age': 27}\nfor k, v in user.items():\n    print(k, v)\n```",
        ),
        (
            "Show a minimal Python try/except example.",
            "```python\ntry:\n    value = int(input('number: '))\n    print(10 / value)\nexcept ValueError:\n    print('Please enter an integer.')\nexcept ZeroDivisionError:\n    print('Division by zero is not allowed.')\n```",
        ),
    ]

    rows: list[dict] = []

    # Chat-heavy: duplicate conversational samples to dominate behavior.
    for inp, out in chat_pairs:
        rows.append(_row("korean", "ko", "ko", inp, out))
        rows.append(_row("korean", "ko", "ko", inp, out))
        rows.append(_row("korean", "ko", "ko", inp, out))
        for rw in chat_rewrites:
            rows.append(_row("korean", "ko", "ko", f"{inp} {rw}", out))

    # Keep a smaller amount of code behavior.
    for inp, out in code_pairs:
        rows.append(_row("code", "code", "ko", inp, out))
        rows.append(_row("code", "code", "ko", f"{inp} 코드로 보여줘.", out))

    # Explicit anti-overfit control rows.
    rows.extend(
        [
            _row(
                "korean",
                "ko",
                "ko",
                "일반 질문에는 코드블록을 꼭 써야 해?",
                "아니요. 코드가 필요한 질문일 때만 코드블록을 사용하면 됩니다.",
            ),
            _row(
                "korean",
                "ko",
                "ko",
                "코드가 필요 없는 질문에는 어떻게 답해?",
                "코드 없이 자연스러운 한국어 문장으로 답변합니다.",
            ),
            _row(
                "korean",
                "ko",
                "ko",
                "너의 이름은?",
                "저는 한국어 대화를 돕는 AI 어시스턴트예요.",
            ),
            _row(
                "korean",
                "ko",
                "ko",
                "이 대화에서 중요한 건 뭐야?",
                "질문 의도를 정확히 파악하고, 불필요한 코드 없이 적절한 형식으로 답하는 것입니다.",
            ),
        ]
    )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/ko_chat_balance_v1.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "out": str(out_path),
        "rows": len(rows),
        "note": "Korean conversational rows are intentionally dominant over code rows.",
    }
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
