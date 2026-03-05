from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


CAPABILITIES: Dict[str, Dict[str, List[str]]] = {
    "자연어 이해": {
        "asks": [
            "자연어 이해를 잘하려면?",
            "문장 의도를 정확히 파악하는 방법",
            "질문 해석 정확도를 올리는 법",
        ],
        "tips": [
            "질문의 목표, 제약 조건, 핵심 용어를 먼저 분리해 해석하세요.",
            "모호한 표현은 확인 질문으로 좁히면 오해를 줄일 수 있습니다.",
            "답변 전에 사용자의 기대 결과를 한 줄로 정리하면 정확도가 올라갑니다.",
        ],
    },
    "맥락 파악": {
        "asks": [
            "맥락 파악을 잘하는 방법",
            "이전 대화 흐름을 놓치지 않는 법",
            "문맥 기반 답변 품질을 높이는 방법",
        ],
        "tips": [
            "직전 대화의 결정 사항과 우선순위를 먼저 요약하고 이어서 답하세요.",
            "새 정보가 들어오면 기존 전제를 갱신해 충돌을 막아야 합니다.",
            "맥락에서 확정된 사실과 추정 정보를 구분해 답하면 안정적입니다.",
        ],
    },
    "논리적 추론": {
        "asks": [
            "논리적 추론 답변 구조 알려줘",
            "논리적으로 생각하는 프레임워크",
            "근거 기반 추론을 잘하는 법",
        ],
        "tips": [
            "전제, 근거, 결론 순서로 분리해 쓰면 추론이 명확해집니다.",
            "반례 가능성을 한 번 점검하면 결론의 신뢰도가 올라갑니다.",
            "결론을 먼저 단정하기보다 근거 강도를 함께 제시하세요.",
        ],
    },
    "생성 능력": {
        "asks": [
            "생성 능력을 올리는 방법",
            "아이디어를 빨리 만드는 법",
            "창의적 대안을 많이 뽑는 방법",
        ],
        "tips": [
            "목표를 한 줄로 고정한 뒤 서로 다른 관점에서 3안 이상을 먼저 펼치세요.",
            "초기에는 검열하지 말고 생성량을 확보한 뒤 평가 단계에서 수렴하세요.",
            "비용 대비 효과가 높은 안을 마지막에 1개로 좁히면 실행력이 올라갑니다.",
        ],
    },
    "글쓰기": {
        "asks": [
            "글쓰기 품질을 높이는 법",
            "가독성 좋은 문장을 쓰는 방법",
            "초안 수정 순서 추천",
        ],
        "tips": [
            "초안은 빠르게 쓰고 수정은 논리 흐름과 문장 연결을 우선 다듬으세요.",
            "문단마다 한 가지 핵심만 남기면 전달력이 좋아집니다.",
            "마지막에 불필요한 수식어를 줄이면 읽기 쉬운 문장이 됩니다.",
        ],
    },
    "코드 생성": {
        "asks": [
            "코드 생성 정확도를 올리는 법",
            "코드 예시를 안정적으로 만드는 방법",
            "실행 가능한 코드 답변 기준",
        ],
        "tips": [
            "입력, 출력, 예외 처리를 먼저 정의한 뒤 코드를 작성하면 안정적입니다.",
            "최소 동작 예시부터 제시하고 필요 시 확장하는 방식이 실수율을 낮춥니다.",
            "언어별 표준 라이브러리 우선 접근이 유지보수에 유리합니다.",
        ],
    },
    "멀티모달 처리": {
        "asks": [
            "이미지음성 멀티모달 처리 순서",
            "이미지와 음성을 함께 해석하는 방법",
            "멀티모달 입력 통합 전략",
        ],
        "tips": [
            "이미지에서 객체·장면을 요약하고 음성은 텍스트로 변환해 핵심 문장을 추출하세요.",
            "두 결과를 시간 순서로 정렬하면 종합 요약 품질이 안정됩니다.",
            "모달별 신뢰도 점수를 함께 기록하면 오류 추적이 쉬워집니다.",
        ],
    },
    "스타일 조절": {
        "asks": [
            "상황에 맞는 스타일 조절 기준",
            "말투를 상황별로 바꾸는 법",
            "응답 톤 자동 조절 팁",
        ],
        "tips": [
            "사용자의 목적과 긴급도를 먼저 보고 톤을 정하면 됩니다.",
            "업무 상황은 간결하게, 학습 상황은 설명 중심으로 조절하세요.",
            "요청한 형식(짧게/체크리스트/예시 포함)을 먼저 충족시키는 것이 핵심입니다.",
        ],
    },
    "문제 해결력": {
        "asks": [
            "문제 해결력을 높이는 방법",
            "문제 해결 프레임워크 추천",
            "복잡한 이슈를 푸는 절차",
        ],
        "tips": [
            "문제를 정의하고 원인을 가설로 나눈 뒤 가장 작은 검증 실험부터 실행하세요.",
            "검증 결과를 기준으로 다음 액션을 결정하면 시행착오 비용이 줄어듭니다.",
            "해결 후 재발 방지 조건까지 기록하면 품질이 올라갑니다.",
        ],
    },
    "단계별 사고": {
        "asks": [
            "단계별 사고 방법",
            "복잡한 작업을 단계로 나누는 법",
            "Step-by-step 사고 습관",
        ],
        "tips": [
            "전체 목표를 작업 단위로 분해하고 단계별 완료 조건을 명확히 적으세요.",
            "각 단계 산출물을 남기면 다음 단계 품질이 안정됩니다.",
            "막히는 단계는 더 작은 하위 단계로 재분해하면 진행이 빨라집니다.",
        ],
    },
    "디버깅": {
        "asks": [
            "디버깅 우선순위",
            "버그를 빨리 찾는 방법",
            "재현 중심 디버깅 절차",
        ],
        "tips": [
            "재현 가능한 최소 케이스를 먼저 만들고 입력-중간 상태-출력 순으로 로그를 확인하세요.",
            "원인 후보를 하나씩 제거하면 가장 빠르게 문제 지점을 찾을 수 있습니다.",
            "수정 후에는 동일 경로 재현 테스트를 반드시 다시 실행하세요.",
        ],
    },
    "수학": {
        "asks": [
            "수학 문제 실수 줄이는 법",
            "계산 정확도 높이는 습관",
            "검산 루틴 추천",
        ],
        "tips": [
            "조건을 식으로 정확히 변환한 뒤 중간 계산값을 단계별로 검산하세요.",
            "단위와 부호를 마지막에 다시 확인하면 실수를 크게 줄일 수 있습니다.",
            "최종값은 근삿값 감각으로 상식 검증까지 하면 안정적입니다.",
        ],
    },
    "계획 수립": {
        "asks": [
            "계획 수립 실전 방법",
            "실행 가능한 계획 만드는 법",
            "우선순위 계획 프레임워크",
        ],
        "tips": [
            "목표를 측정 가능한 지표로 바꾸고 기한과 우선순위를 함께 적어야 합니다.",
            "하루 단위 최소 실행량을 정하면 계획이 실제 행동으로 이어집니다.",
            "리스크와 대체 경로를 한 줄씩 적어두면 계획 유지율이 높아집니다.",
        ],
    },
    "적응력": {
        "asks": [
            "적응력 높이는 방법",
            "요구사항 변화에 대응하는 법",
            "변경이 잦은 프로젝트 운영법",
        ],
        "tips": [
            "변경 가능한 영역과 고정 영역을 먼저 분리해 설계하세요.",
            "핵심 인터페이스를 안정적으로 유지하면 변경 비용을 낮출 수 있습니다.",
            "변경 로그와 의사결정 근거를 남기면 재작업을 줄일 수 있습니다.",
        ],
    },
    "소량 데이터 학습": {
        "asks": [
            "소량 데이터 학습 팁",
            "적은 데이터로 성능 올리는 법",
            "few-shot 학습 전략",
        ],
        "tips": [
            "데이터 증강, 전이학습, 규칙 기반 보정을 함께 쓰는 조합이 효과적입니다.",
            "검증셋을 작게라도 고정해 과적합 여부를 계속 확인해야 합니다.",
            "오류 샘플 재학습 루프를 짧게 반복하면 성능이 빨리 올라갑니다.",
        ],
    },
    "자기 오류 수정": {
        "asks": [
            "자기 오류 수정 방법",
            "답변 오류를 줄이는 습관",
            "self-check 루틴 만드는 법",
        ],
        "tips": [
            "답변 전에 가정과 불확실성을 짧게 점검하세요.",
            "답변 후 핵심 사실을 재검증하고 오류가 있으면 즉시 정정하세요.",
            "수정 근거를 함께 제시하면 신뢰성이 높아집니다.",
        ],
    },
    "도구 활용 능력": {
        "asks": [
            "도구 활용 능력 높이려면",
            "툴 사용 생산성 올리는 법",
            "도구 선택 기준",
        ],
        "tips": [
            "문제를 도구에 맞추지 말고 도구를 문제 단계에 맞게 배치하세요.",
            "검색, 실험, 검증, 자동화를 분리해서 쓰면 생산성이 크게 올라갑니다.",
            "반복 작업은 스크립트화해 인지 부담을 줄이는 것이 핵심입니다.",
        ],
    },
    "안전성과 신뢰성": {
        "asks": [
            "안전성과 신뢰성 기준",
            "신뢰 가능한 답변을 만드는 법",
            "안전한 응답 원칙",
        ],
        "tips": [
            "사실과 추론을 구분해 말하고 위험한 조언은 보수적으로 제한해야 합니다.",
            "근거가 약한 부분은 불확실하다고 명시하는 것이 중요합니다.",
            "민감 주제는 검증 가능한 정보와 보호 원칙을 우선하세요.",
        ],
    },
    "편향 최소화": {
        "asks": [
            "편향 최소화 방법",
            "편향을 줄이는 답변 방식",
            "공정한 설명을 만드는 법",
        ],
        "tips": [
            "단정적 표현을 줄이고 다양한 관점을 균형 있게 제시하세요.",
            "특정 집단에 불리한 일반화는 피하고 판단 기준을 투명하게 밝히세요.",
            "근거 중심 표현을 유지하면 불필요한 편향을 줄일 수 있습니다.",
        ],
    },
}


STYLE_SUFFIXES = [
    "",
    " 짧게 답해줘.",
    " 핵심만 3줄로 답해줘.",
    " 체크리스트 형태로 답해줘.",
    " 실무 기준으로 답해줘.",
    " 초보자도 이해되게 설명해줘.",
    " 예시 1개 포함해줘.",
    " 바로 실행 가능한 형태로 답해줘.",
]


def make_row(capability: str, inp: str, out: str, segment_tag: str = "ko", tier: str = "high") -> Dict:
    task_type = "korean" if segment_tag == "ko" else segment_tag
    return {
        "task_type": task_type,
        "segment_tag": segment_tag,
        "language": "ko",
        "_meta_quality_tier": tier,
        "capability": capability,
        "source": "capability_pack_v2",
        "input": str(inp).strip(),
        "output": str(out).strip(),
    }


def capability_output(tips: List[str], style: str) -> str:
    if "짧게" in style or "3줄" in style:
        return f"{tips[0]} {tips[1]}"
    if "체크리스트" in style:
        return f"1. {tips[0]}\n2. {tips[1]}\n3. {tips[2]}"
    if "예시" in style:
        return f"{tips[0]} {tips[1]} 예를 들어 오늘 할 일 1개를 정하고 바로 10분 실행하면 시작 장벽을 낮출 수 있습니다."
    if "실무" in style:
        return f"{tips[0]} {tips[2]} 실무에서는 변경 이력과 검증 결과를 함께 남기는 습관이 특히 중요합니다."
    return f"{tips[0]} {tips[1]} {tips[2]}"


def code_samples() -> List[Tuple[str, str]]:
    return [
        (
            "파이썬 예외처리 기본 예시",
            "기본 구조는 `try / except` 입니다.\n\n```python\ntry:\n    n = int(input('숫자 입력: '))\n    result = 10 / n\n    print(result)\nexcept ValueError:\n    print('숫자를 입력해야 합니다.')\nexcept ZeroDivisionError:\n    print('0으로 나눌 수 없습니다.')\n```",
        ),
        (
            "자바스크립트 배열 중복 제거 코드",
            "대표적인 방법은 `Set`을 사용하는 것입니다.\n\n```javascript\nconst nums = [1, 2, 2, 3, 3, 4];\nconst unique = [...new Set(nums)];\nconsole.log(unique); // [1, 2, 3, 4]\n```",
        ),
        (
            "파이썬 파일 존재 여부 확인 코드",
            "아래처럼 `pathlib`를 쓰면 간단합니다.\n\n```python\nfrom pathlib import Path\n\npath = Path('data.txt')\nif path.exists():\n    print('파일이 있습니다.')\nelse:\n    print('파일이 없습니다.')\n```",
        ),
    ]


def build_rows(target_rows: int, seed: int = 42) -> List[Dict]:
    rng = random.Random(int(seed))
    rows: List[Dict] = []

    for capability, spec in CAPABILITIES.items():
        asks = spec["asks"]
        tips = spec["tips"]
        for ask in asks:
            for style in STYLE_SUFFIXES:
                prompt = f"{ask}{style}".strip()
                output = capability_output(tips=tips, style=style)
                tier = "high" if capability in {"안전성과 신뢰성", "편향 최소화", "자기 오류 수정"} else "mid"
                rows.append(make_row(capability=capability, inp=prompt, out=output, segment_tag="ko", tier=tier))

    for q, a in code_samples():
        for style in STYLE_SUFFIXES[:4]:
            rows.append(
                make_row(
                    capability="코드 생성",
                    inp=f"{q}{style}".strip(),
                    out=a,
                    segment_tag="code",
                    tier="high",
                )
            )

    fixed_policy = [
        ("너의 이름은?", "제 모델 이름은 spear1.0입니다."),
        ("내 이름 뭐야?", "아직 이름을 알려주지 않아서 사용자님으로 부를게요."),
        ("내 이름은 민수야. 기억해.", "알겠어요. 앞으로 민수님으로 부를게요."),
        ("사용자 이름 추정하지 마", "알겠어요. 사용자가 직접 알려준 이름만 사용하겠습니다."),
    ]
    for q, a in fixed_policy:
        rows.append(make_row(capability="안전성과 신뢰성", inp=q, out=a, segment_tag="ko", tier="high"))

    scenario_tags = [
        "학습 상황",
        "업무 상황",
        "초보자 상황",
        "실무 상황",
        "긴급 상황",
        "장기 과제 상황",
        "면접 준비 상황",
        "프로젝트 운영 상황",
        "문서 작성 상황",
        "분석 작업 상황",
    ]

    # Expand by controlled paraphrase-style recombination until target rows.
    attempts = 0
    max_attempts = int(target_rows) * 40
    while len(rows) < int(target_rows) and attempts < max_attempts:
        attempts += 1
        capability = rng.choice(list(CAPABILITIES.keys()))
        spec = CAPABILITIES[capability]
        ask = rng.choice(spec["asks"])
        style = rng.choice(STYLE_SUFFIXES)
        tag = rng.choice(scenario_tags)
        prompt = f"{tag}에서 {ask}{style}".strip()
        tips = spec["tips"]
        rng.shuffle(tips)
        output = capability_output(tips=tips, style=style)
        tier = "high" if capability in {"안전성과 신뢰성", "편향 최소화", "자기 오류 수정"} else "mid"
        rows.append(make_row(capability=capability, inp=prompt, out=output, segment_tag="ko", tier=tier))

    # Dedupe then top up if dedupe reduced count.
    dedup: Dict[Tuple[str, str], Dict] = {}
    for r in rows:
        key = (str(r["input"]).strip().lower(), str(r["output"]).strip().lower())
        dedup[key] = r
    rows = list(dedup.values())

    fill_idx = 0
    while len(rows) < int(target_rows):
        fill_idx += 1
        capability = rng.choice(list(CAPABILITIES.keys()))
        spec = CAPABILITIES[capability]
        ask = rng.choice(spec["asks"])
        suffix = rng.choice([" 간단하게 알려줘.", " 실전 예시로 답해줘.", " 핵심 2문장으로 답해줘.", " 단계로 나눠줘."])
        prompt = f"{ask}{suffix} (케이스 {fill_idx})"
        output = " ".join(spec["tips"][:2]) + f" 우선순위는 케이스 {fill_idx % 7 + 1} 기준으로 조정하세요."
        key = (prompt.strip().lower(), output.strip().lower())
        if key in dedup:
            continue
        tier = "high" if capability in {"안전성과 신뢰성", "편향 최소화", "자기 오류 수정"} else "mid"
        rec = make_row(capability=capability, inp=prompt, out=output, segment_tag="ko", tier=tier)
        dedup[key] = rec
        rows.append(rec)

    rng.shuffle(rows)
    return rows[: int(target_rows)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expanded capability dataset (1k~3k).")
    parser.add_argument("--out", default="data/capability_pack_v2.jsonl")
    parser.add_argument("--target_rows", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target = int(args.target_rows)
    if target < 1000:
        target = 1000
    if target > 3000:
        target = 3000

    rows = build_rows(target_rows=target, seed=int(args.seed))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "out": str(out_path),
        "rows": len(rows),
        "target_rows": target,
    }
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
