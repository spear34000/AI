from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


SPACE_RE = re.compile(r"\s+")


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def make_row(inp: str, out: str) -> Dict[str, str]:
    return {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": "high",
        "input": normalize_space(inp),
        "output": normalize_space(out),
        "_meta_source_file": "synthetic/prompt_focus_v1",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build prompt-focused Korean dataset v1")
    p.add_argument("--out", default="data/prompt_focus_v1.jsonl")
    p.add_argument("--manifest", default="data/prompt_focus_v1.manifest.json")
    p.add_argument("--seed", type=int, default=79)
    p.add_argument("--n_intro", type=int, default=7000)
    p.add_argument("--n_python", type=int, default=9000)
    p.add_argument("--n_capital", type=int, default=3500)
    p.add_argument("--n_unknown", type=int, default=3000)
    p.add_argument("--n_general", type=int, default=4500)
    return p.parse_args()


def synth_pairs(
    rows: List[Dict[str, str]],
    seen: set[bytes],
    prompts: List[str],
    outputs: List[str],
    n: int,
    rnd: random.Random,
) -> int:
    added = 0
    q_prefix = ["", "", "간단히 ", "짧게 ", "한국어로 ", "핵심만 ", "한 줄로 "]
    q_suffix = ["", "", " 설명해줘", " 알려줘", " 답해줘", " 부탁해", " 요약해줘"]
    a_prefix = ["", "", "요약하면 ", "한 줄로 말하면 ", "핵심은 ", "간단히 말해 "]
    a_suffix = [
        "",
        "",
        " 필요하면 예시도 덧붙여 드릴게요.",
        " 원하면 더 짧게 정리해 드릴 수 있어요.",
        " 요청하면 관련 개념도 함께 설명할게요.",
    ]
    for _ in range(int(n)):
        q = normalize_space(f"{rnd.choice(q_prefix)}{rnd.choice(prompts)}{rnd.choice(q_suffix)}")
        a = normalize_space(f"{rnd.choice(a_prefix)}{rnd.choice(outputs)}{rnd.choice(a_suffix)}")
        k = dedupe_key(q, a)
        if k in seen:
            continue
        seen.add(k)
        rows.append(make_row(q, a))
        added += 1
    return added


def main() -> None:
    args = parse_args()
    rnd = random.Random(int(args.seed))

    rows: List[Dict[str, str]] = []
    seen: set[bytes] = set()

    intro_prompts = [
        "한국어로 자기소개 한 줄",
        "자기소개 한 줄",
        "짧게 자기소개",
        "한 문장으로 본인 소개",
        "너를 한 줄로 소개해줘",
    ]
    intro_outputs = [
        "반갑습니다. 질문 의도를 빠르게 파악해 간결하고 정확하게 답변하는 한국어 AI 도우미입니다.",
        "안녕하세요. 필요한 정보를 짧고 분명하게 정리해 드리는 한국어 AI 어시스턴트입니다.",
        "반갑습니다. 요청 형식에 맞춰 핵심만 명확하게 전달하는 한국어 AI입니다.",
        "안녕하세요. 복잡한 내용을 이해하기 쉽게 정리해 답변하는 한국어 AI 도우미입니다.",
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

    unknown_prompts = [
        "모르면 모른다고 말해줘",
        "확실하지 않으면 어떻게 답해?",
        "추측하지 말고 답해줘",
        "모르는 건 솔직히 말해줘",
    ]
    unknown_outputs = [
        "확실하지 않은 정보는 추측하지 않고 모른다고 분명히 말한 뒤 확인 방법을 안내하겠습니다.",
        "근거가 부족하면 단정하지 않고 불확실성을 먼저 밝힌 다음 필요한 정보를 요청하겠습니다.",
        "검증되지 않은 내용은 사실처럼 말하지 않고 확인 가능한 근거를 우선 제시하겠습니다.",
    ]

    general_prompts = [
        "AI란?",
        "데이터베이스란?",
        "API가 뭐야",
        "클라우드 컴퓨팅이란?",
        "테스트 코드가 왜 필요해?",
    ]
    general_outputs = [
        "인공지능은 데이터를 학습해 예측, 분류, 생성 같은 작업을 수행하는 기술입니다.",
        "데이터베이스는 데이터를 구조적으로 저장하고 빠르게 조회·수정할 수 있게 관리하는 시스템입니다.",
        "API는 서로 다른 소프트웨어가 정해진 방식으로 기능과 데이터를 주고받게 하는 인터페이스입니다.",
        "클라우드 컴퓨팅은 인터넷을 통해 필요한 서버와 저장소를 유연하게 사용하는 방식입니다.",
        "테스트 코드는 회귀 버그를 빠르게 찾아 안정적인 배포를 가능하게 합니다.",
    ]

    counts = {
        "intro": synth_pairs(rows, seen, intro_prompts, intro_outputs, int(args.n_intro), rnd),
        "python": synth_pairs(rows, seen, python_prompts, python_outputs, int(args.n_python), rnd),
        "capital": synth_pairs(rows, seen, capital_prompts, capital_outputs, int(args.n_capital), rnd),
        "unknown": synth_pairs(rows, seen, unknown_prompts, unknown_outputs, int(args.n_unknown), rnd),
        "general": synth_pairs(rows, seen, general_prompts, general_outputs, int(args.n_general), rnd),
    }

    rnd.shuffle(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "name": "prompt_focus_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "rows": {"final": len(rows)},
        "counts": counts,
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] rows={len(rows)} -> {out_path}")
    print(f"[done] manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
