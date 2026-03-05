from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def row(inp: str, out: str, capability: str, segment_tag: str = "ko", tier: str = "high") -> Dict:
    task = "korean" if segment_tag == "ko" else segment_tag
    return {
        "task_type": task,
        "segment_tag": segment_tag,
        "language": "ko",
        "_meta_quality_tier": tier,
        "capability": capability,
        "source": "capability_pack_v1",
        "input": str(inp).strip(),
        "output": str(out).strip(),
    }


def add_style_variants(base_prompt: str) -> List[str]:
    suffixes = [
        "",
        " 짧게 답해줘.",
        " 실무 기준으로 답해줘.",
        " 핵심만 3줄로 답해줘.",
        " 초보자 기준으로 설명해줘.",
        " 체크리스트 형태로 답해줘.",
        " 예시 1개 포함해줘.",
        " 바로 실행 가능한 형태로 답해줘.",
    ]
    out: List[str] = []
    for s in suffixes:
        out.append(f"{base_prompt}{s}".strip())
    return out


def capability_pairs() -> List[Tuple[str, str, str, str]]:
    pairs: List[Tuple[str, str, str, str]] = [
        (
            "자연어 이해",
            "문장 의도를 정확히 파악하려면 뭐가 중요해?",
            "질문의 목표, 제약 조건, 핵심 용어를 먼저 분리해 해석해야 합니다. "
            "모호한 표현은 확인 질문으로 좁히면 오해를 줄일 수 있습니다.",
            "ko",
        ),
        (
            "맥락 파악",
            "맥락을 놓치지 않으려면 어떻게 답해야 해?",
            "직전 대화의 결정 사항과 사용자의 우선순위를 먼저 요약한 뒤 답을 이어가야 합니다. "
            "새 정보가 들어오면 기존 전제를 갱신해 충돌을 막는 것이 핵심입니다.",
            "ko",
        ),
        (
            "논리적 추론",
            "논리적으로 추론하는 답변 구조를 알려줘",
            "전제, 근거, 결론 순서로 분리해 쓰면 추론이 명확해집니다. "
            "반례 가능성을 한 번 점검하면 결론의 신뢰도가 올라갑니다.",
            "ko",
        ),
        (
            "생성 능력",
            "아이디어를 빠르게 생성하려면 어떤 방식이 좋아?",
            "목표를 한 줄로 고정한 뒤, 서로 다른 관점에서 3안 이상을 먼저 펼치세요. "
            "그다음 비용 대비 효과가 높은 안을 1개로 수렴하면 속도와 품질을 같이 잡을 수 있습니다.",
            "ko",
        ),
        (
            "글쓰기",
            "글쓰기 품질을 높이는 가장 실용적인 방법은?",
            "초안은 빠르게 쓰고, 수정은 논리 흐름과 문장 간 연결만 우선 다듬으세요. "
            "마지막에 불필요한 수식어를 줄이면 읽기 쉬운 문장이 됩니다.",
            "ko",
        ),
        (
            "코드 생성",
            "파이썬에서 파일 존재 여부를 확인하는 기본 코드 예시",
            "기본 예시는 아래와 같습니다.\n\n```python\nfrom pathlib import Path\n\npath = Path('data.txt')\nif path.exists():\n    print('파일이 있습니다.')\nelse:\n    print('파일이 없습니다.')\n```",
            "code",
        ),
        (
            "코드 생성",
            "자바스크립트 배열에서 중복 제거하는 코드",
            "대표적인 방법은 `Set`을 사용하는 것입니다.\n\n```javascript\nconst nums = [1, 2, 2, 3, 3, 4];\nconst unique = [...new Set(nums)];\nconsole.log(unique); // [1, 2, 3, 4]\n```",
            "code",
        ),
        (
            "멀티모달 처리",
            "이미지와 음성을 같이 처리하려면 어떤 순서가 좋아?",
            "먼저 이미지에서 객체와 장면을 요약하고, 음성은 텍스트로 변환해 핵심 문장을 추출하세요. "
            "그다음 두 결과를 시간 순서로 정렬하면 멀티모달 요약 품질이 안정됩니다.",
            "ko",
        ),
        (
            "스타일 조절",
            "상황에 맞게 말투를 바꾸는 기준이 뭐야?",
            "사용자의 목적과 긴급도를 먼저 보고 톤을 정하면 됩니다. "
            "업무 상황은 간결하고 명확하게, 학습 상황은 설명 중심으로 조절하는 방식이 효과적입니다.",
            "ko",
        ),
        (
            "문제 해결력",
            "문제를 빠르게 해결하는 프레임워크 알려줘",
            "문제를 정의하고, 원인을 가설로 나눈 뒤, 가장 작은 검증 실험부터 실행하세요. "
            "검증 결과에 따라 바로 수정하면 시행착오 비용을 크게 줄일 수 있습니다.",
            "ko",
        ),
        (
            "단계별 사고",
            "복잡한 일을 단계별로 풀어내는 방법",
            "전체 목표를 1단계 작업 단위로 분해하고 완료 조건을 명확히 적으세요. "
            "단계마다 산출물을 남기면 다음 단계 품질이 안정됩니다.",
            "ko",
        ),
        (
            "디버깅",
            "디버깅할 때 우선순위를 어떻게 잡아?",
            "재현 가능한 최소 케이스를 먼저 만들고, 입력-중간 상태-출력 순으로 로그를 확인하세요. "
            "원인 후보를 하나씩 제거하면 가장 빠르게 문제 지점을 찾을 수 있습니다.",
            "ko",
        ),
        (
            "수학",
            "수학 문제를 틀리지 않게 푸는 기본 습관은?",
            "조건을 식으로 정확히 변환한 뒤, 중간 계산값을 단계별로 검산하세요. "
            "단위와 부호를 마지막에 다시 확인하면 실수를 크게 줄일 수 있습니다.",
            "ko",
        ),
        (
            "계획 수립",
            "실행 가능한 계획을 세우는 법",
            "목표를 측정 가능한 지표로 바꾸고, 기한과 우선순위를 함께 적어야 합니다. "
            "하루 단위 최소 실행량을 정하면 계획이 실제 행동으로 이어집니다.",
            "ko",
        ),
        (
            "적응력",
            "요구사항이 자주 바뀔 때 대응법",
            "변경 가능한 영역과 고정 영역을 먼저 분리해 설계하세요. "
            "핵심 인터페이스를 안정적으로 유지하면 변경 비용을 낮출 수 있습니다.",
            "ko",
        ),
        (
            "소량 데이터 학습",
            "데이터가 적을 때 성능을 올리는 실전 방법",
            "데이터 증강, 전이학습, 규칙 기반 보정을 함께 쓰는 조합이 효과적입니다. "
            "검증셋을 작게라도 고정해 과적합 여부를 계속 확인해야 합니다.",
            "ko",
        ),
        (
            "자기 오류 수정",
            "답변 오류를 스스로 고치려면?",
            "답변 전에 가정과 불확실성을 짧게 점검하고, 답변 후에는 핵심 사실을 재검증하세요. "
            "오류를 발견하면 즉시 정정하고 수정 근거를 함께 제시하는 게 신뢰를 높입니다.",
            "ko",
        ),
        (
            "도구 활용 능력",
            "도구를 잘 쓰는 사람의 공통점이 뭐야?",
            "문제를 도구에 맞추지 않고, 도구를 문제 단계에 맞게 배치합니다. "
            "검색, 실험, 검증, 자동화를 분리해서 쓰면 생산성이 크게 올라갑니다.",
            "ko",
        ),
        (
            "안전성과 신뢰성",
            "안전하고 신뢰 가능한 답변 기준은?",
            "사실과 추론을 구분해 말하고, 위험한 조언은 보수적으로 제한해야 합니다. "
            "근거가 약한 부분은 불확실하다고 명시하는 것이 신뢰성에 중요합니다.",
            "ko",
        ),
        (
            "편향 최소화",
            "편향을 줄이려면 답변을 어떻게 구성해야 해?",
            "단정적 표현을 줄이고 다양한 관점을 균형 있게 제시하세요. "
            "특정 집단에 불리한 일반화는 피하고, 판단 기준을 투명하게 밝히는 것이 핵심입니다.",
            "ko",
        ),
    ]
    return pairs


def build_rows(seed: int = 42) -> List[Dict]:
    rng = random.Random(int(seed))
    rows: List[Dict] = []
    for capability, prompt, answer, segment in capability_pairs():
        prompts = add_style_variants(prompt)
        for p in prompts:
            tier = "high" if capability in {"안전성과 신뢰성", "편향 최소화", "자기 오류 수정"} else "mid"
            rows.append(row(inp=p, out=answer, capability=capability, segment_tag=segment, tier=tier))

    # Add a few focused policy rows for identity/name behavior consistency.
    policy_rows = [
        ("정체가 뭐야?", "저는 spear1.0입니다."),
        ("너의 이름은?", "제 모델 이름은 spear1.0입니다."),
        ("내 이름 뭐야?", "아직 이름을 알려주지 않아서 사용자님으로 부를게요."),
        ("내 이름은 민수야. 기억해.", "알겠어요. 앞으로 민수님으로 부를게요."),
    ]
    for q, a in policy_rows:
        rows.append(row(inp=q, out=a, capability="안전성과 신뢰성", segment_tag="ko", tier="high"))

    # Keep rows stable but shuffled.
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build capability-oriented quality dataset.")
    parser.add_argument("--out", default="data/capability_pack_v1.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(seed=int(args.seed))

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({"out": str(out_path), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
