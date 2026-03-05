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
INTRO_INPUT_RE = re.compile(
    r"(?:자기소개|self[\s-]?intro|너의\s*이름|네\s*이름|이름이\s*뭐|한\s*줄\s*소개|소개해\s*줘|소개해줘)",
    flags=re.IGNORECASE,
)


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def make_row(inp: str, out: str, tier: str = "high") -> Dict[str, str]:
    return {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": str(tier).strip().lower() or "high",
        "input": normalize_space(inp),
        "output": normalize_space(out),
    }


def dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def has_any(text: str, needles: Sequence[str]) -> bool:
    s = normalize_space(text).lower()
    return any(normalize_space(n).lower() in s for n in needles if normalize_space(n))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build de-hardcode patch dataset v2")
    p.add_argument("--pool", default="data/quality/hq_ko_chat_v1.jsonl")
    p.add_argument("--out", default="data/dehardcode_patch_v2.jsonl")
    p.add_argument("--manifest", default="data/dehardcode_patch_v2.manifest.json")
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--pool_rows", type=int, default=22000)
    p.add_argument("--n_intro", type=int, default=4500)
    p.add_argument("--n_python", type=int, default=4500)
    p.add_argument("--n_capital", type=int, default=1800)
    p.add_argument("--n_unknown", type=int, default=1200)
    p.add_argument("--n_math", type=int, default=1600)
    return p.parse_args()


def add_pairs(
    rows: List[Dict[str, str]],
    seen: set[bytes],
    rnd: random.Random,
    prompts: Sequence[str],
    outputs: Sequence[str],
    n: int,
) -> int:
    if not prompts or not outputs or int(n) <= 0:
        return 0
    prefixes = ["", "", "", "간단히 ", "한국어로 ", "핵심만 "]
    suffixes = ["", "", "", " 한 줄로", " 짧게", " 간단히 답해줘", " 핵심만 말해줘"]
    tails = ["", "", "", " 필요하면 더 짧게도 답할 수 있습니다.", " 요청 형식에 맞춰 답하겠습니다."]
    added = 0
    for _ in range(int(n)):
        base_in = normalize_space(rnd.choice(list(prompts)))
        inp = normalize_space(f"{rnd.choice(prefixes)}{base_in}{rnd.choice(suffixes)}")
        base_out = normalize_space(rnd.choice(list(outputs)))
        out = normalize_space(f"{base_out}{rnd.choice(tails)}")
        k = dedupe_key(inp, out)
        if k in seen:
            continue
        seen.add(k)
        rows.append(make_row(inp, out, tier="high"))
        added += 1
    return added


def add_math(
    rows: List[Dict[str, str]],
    seen: set[bytes],
    rnd: random.Random,
    n: int,
) -> int:
    added = 0
    for _ in range(int(n)):
        a = rnd.randint(0, 999)
        b = rnd.randint(0, 999)
        op = rnd.choice(["+", "-"])
        c = a + b if op == "+" else a - b
        inp = rnd.choice(
            [
                f"{a}{op}{b}는?",
                f"{a} {op} {b} = ?",
                f"간단히 계산해줘: {a}{op}{b}",
                f"{a}와 {b}를 {'더하면' if op == '+' else '빼면'}?",
            ]
        )
        out = rnd.choice(
            [
                f"{c}입니다.",
                f"정답은 {c}입니다.",
                f"{a}{op}{b}는 {c}입니다.",
            ]
        )
        k = dedupe_key(inp, out)
        if k in seen:
            continue
        seen.add(k)
        rows.append(make_row(inp, out, tier="high"))
        added += 1
    return added


def main() -> None:
    args = parse_args()
    rnd = random.Random(int(args.seed))

    banned_output_substrings = [
        "질문 의도를 파악해 단계별로 정리해드릴게요",
        "원하면 짧게 3줄 요약으로도 답할 수 있어요",
        "간결하고 정확하게 답변하는 한국어 ai",
        "한국어 ai 어시스턴트입니다",
        "ai 어시스턴트입니다",
    ]

    intro_prompts = [
        "한국어로 자기소개 한 줄",
        "자기소개 한 문장으로 해줘",
        "짧게 자기소개해줘",
        "한 줄로 본인 소개 부탁해",
        "간단히 자기소개",
        "너를 한 줄로 소개해줘",
        "핵심만 자기소개",
        "한 문장 소개",
    ]
    intro_outputs = [
        "반갑습니다. 질문에 맞는 답을 짧고 정확하게 전하는 한국어 도우미입니다.",
        "안녕하세요. 요청 의도를 파악해 핵심부터 간결하게 설명하는 도우미입니다.",
        "반갑습니다. 필요한 정보를 짧고 또렷하게 안내하는 한국어 질의응답 도우미입니다.",
        "안녕하세요. 복잡한 내용을 이해하기 쉽게 정리해 드리는 한국어 도우미입니다.",
        "반갑습니다. 요청한 형식에 맞춰 정확하고 실용적인 답을 제공하는 도우미입니다.",
    ]

    python_prompts = [
        "파이썬이란",
        "파이썬이 뭐야?",
        "파이썬을 한 줄로 설명해줘",
        "Python이란?",
        "파이썬 언어 설명",
        "파이썬 특징 알려줘",
        "파이썬은 어떤 언어야?",
    ]
    python_outputs = [
        "파이썬은 문법이 간결하고 가독성이 좋아 입문과 실무에 모두 널리 쓰이는 프로그래밍 언어입니다.",
        "파이썬은 데이터 분석, 자동화, 웹 개발, AI 등 다양한 분야에서 활용되는 범용 프로그래밍 언어입니다.",
        "파이썬은 짧고 읽기 쉬운 코드로 빠르게 기능을 구현하기 좋은 고수준 프로그래밍 언어입니다.",
        "파이썬은 풍부한 라이브러리 생태계를 기반으로 개발 생산성이 높은 언어입니다.",
    ]

    capital_prompts = [
        "대한민국 수도는?",
        "한국의 수도는 어디야?",
        "대한민국 수도 알려줘",
        "정답만: 한국 수도",
    ]
    capital_outputs = [
        "대한민국의 수도는 서울입니다.",
        "서울입니다.",
        "정답은 서울입니다.",
    ]

    unknown_prompts = [
        "모르면 모른다고 말해줄래?",
        "확실하지 않으면 어떻게 답해?",
        "근거 없으면 추측하지 마",
        "모르는 건 모른다고 해",
    ]
    unknown_outputs = [
        "확실하지 않은 내용은 추측하지 않고, 모르는 부분은 모른다고 분명히 말하겠습니다.",
        "근거가 부족하면 단정하지 않고 확인이 필요한 점을 명확히 안내하겠습니다.",
        "불확실한 정보는 사실처럼 말하지 않고, 확인이 필요한 항목으로 구분해 답하겠습니다.",
    ]

    pool_path = Path(args.pool)
    if not pool_path.exists():
        raise FileNotFoundError(f"pool not found: {pool_path}")

    rows: List[Dict[str, str]] = []
    seen: set[bytes] = set()
    dropped = {
        "pool_intro_input": 0,
        "pool_banned_output": 0,
        "pool_decode": 0,
        "pool_missing_pair": 0,
    }

    pool_rows: List[Dict[str, str]] = []
    with pool_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                dropped["pool_decode"] += 1
                continue
            if not isinstance(row, dict):
                dropped["pool_decode"] += 1
                continue

            inp = normalize_space(str(row.get("input", "")))
            out = normalize_space(str(row.get("output", "")))
            if not inp or not out:
                dropped["pool_missing_pair"] += 1
                continue
            if INTRO_INPUT_RE.search(inp):
                dropped["pool_intro_input"] += 1
                continue
            if has_any(out, banned_output_substrings):
                dropped["pool_banned_output"] += 1
                continue
            if len(inp) < 2 or len(inp) > 1800 or len(out) < 8 or len(out) > 2400:
                continue
            if not re.search(r"[가-힣]", inp + out):
                continue
            if re.search(r"(.)\1{10,}", out):
                continue

            pool_rows.append(
                {
                    "input": inp,
                    "output": out,
                    "_meta_quality_tier": str(row.get("_meta_quality_tier", "high")).strip().lower() or "high",
                    "_meta_source_file": str(pool_path).replace("\\", "/"),
                }
            )

    rnd.shuffle(pool_rows)
    for row in pool_rows[: int(args.pool_rows)]:
        k = dedupe_key(row["input"], row["output"])
        if k in seen:
            continue
        seen.add(k)
        rows.append(
            {
                "task_type": "korean",
                "segment_tag": "ko",
                "language": "ko",
                "_meta_quality_tier": row["_meta_quality_tier"] if row["_meta_quality_tier"] in {"high", "mid"} else "high",
                "input": row["input"],
                "output": row["output"],
                "_meta_source_file": row["_meta_source_file"],
            }
        )

    syn_added = {
        "intro": add_pairs(rows, seen, rnd, intro_prompts, intro_outputs, int(args.n_intro)),
        "python": add_pairs(rows, seen, rnd, python_prompts, python_outputs, int(args.n_python)),
        "capital": add_pairs(rows, seen, rnd, capital_prompts, capital_outputs, int(args.n_capital)),
        "unknown": add_pairs(rows, seen, rnd, unknown_prompts, unknown_outputs, int(args.n_unknown)),
        "math": add_math(rows, seen, rnd, int(args.n_math)),
    }

    rnd.shuffle(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fw:
        for row in rows:
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "out": str(out_path).replace("\\", "/"),
        "rows": len(rows),
        "seed": int(args.seed),
        "pool": str(pool_path).replace("\\", "/"),
        "pool_selected": int(min(len(pool_rows), int(args.pool_rows))),
        "pool_available_filtered": int(len(pool_rows)),
        "synthetic_added": syn_added,
        "dropped": dropped,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"out": str(out_path), "rows": len(rows), "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
