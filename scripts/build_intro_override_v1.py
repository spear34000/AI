from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


INTRO_PROMPTS = [
    "한국어로 자기소개 한 줄",
    "한국어로 자기소개 한 줄 해줘",
    "한국어로 자기소개 한 문장",
    "자기소개 해줘",
    "너 소개해줘",
    "자기소개 부탁해",
    "한 줄로 소개해줘",
    "짧게 자기소개",
    "한 문장으로 본인 소개",
    "너 누구야 한 줄로",
]

STYLE_SUFFIXES = [
    "",
    "?",
    " 짧게",
    " 간단히",
    " 핵심만",
    " 한 문장으로",
    " 짧게 답해줘",
    " 간결하게 답해줘",
]

INTRO_OUTPUTS = [
    "안녕하세요. 한국어로 핵심을 간단히 정리해 드리는 AI 어시스턴트입니다.",
    "반갑습니다. 요청 의도에 맞춰 짧고 정확하게 답변하는 AI입니다.",
    "안녕하세요. 필요한 정보를 빠르게 정리해 드리는 한국어 AI 도우미입니다.",
    "반갑습니다. 질문에 맞는 실용적인 답변을 간결하게 제공하는 AI입니다.",
    "안녕하세요. 핵심 위주로 명확하게 안내하는 한국어 AI 어시스턴트입니다.",
]


def make_row(inp: str, out: str, tier: str) -> dict:
    return {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": tier,
        "input": inp,
        "output": out,
    }


def build_rows(seed: int, repeats: int) -> list[dict]:
    rnd = random.Random(seed)
    rows: list[dict] = []
    for _ in range(max(1, int(repeats))):
        for p in INTRO_PROMPTS:
            for sfx in STYLE_SUFFIXES:
                inp = f"{p}{sfx}".strip()
                out = rnd.choice(INTRO_OUTPUTS)
                tier = "high" if rnd.random() < 0.8 else "mid"
                rows.append(make_row(inp, out, tier=tier))
    rnd.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/intro_override_v1.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=30)
    args = ap.parse_args()

    rows = build_rows(seed=int(args.seed), repeats=int(args.repeats))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "rows": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

