from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def row(inp: str, out: str, tier: str = "high") -> Dict:
    return {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": tier,
        "input": inp.strip(),
        "output": out.strip(),
    }


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def hangul_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    h = sum(1 for c in chars if 0xAC00 <= ord(c) <= 0xD7A3)
    return float(h) / float(len(chars))


CODE_HINT_RE = re.compile(
    r"\b("
    r"python|javascript|typescript|java|kotlin|swift|go|rust|c\+\+|c#|sql|regex|json|yaml|"
    r"class|function|import|return|try|except|def|const|let|var"
    r")\b",
    flags=re.IGNORECASE,
)


def is_noisy(inp: str, out: str) -> bool:
    txt = f"{inp}\n{out}"
    if len(txt) < 20 or len(txt) > 2200:
        return True
    if txt.count("?") >= 4 or "??" in txt:
        return True
    if "\ufffd" in txt:
        return True
    if hangul_ratio(txt) < 0.12:
        return True
    return False


def is_code_like(inp: str, out: str) -> bool:
    txt = f"{inp}\n{out}"
    lower = txt.lower()
    if "```" in lower:
        return True
    if CODE_HINT_RE.search(lower):
        return True
    if re.search(r"[{}();<>\[\]_]=?", txt):
        return True
    return False


def dedupe(rows: Iterable[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for r in rows:
        k = (str(r.get("input", "")).strip(), str(r.get("output", "")).strip())
        if not k[0] or not k[1]:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def policy_rows() -> List[Dict]:
    # These rows teach stable user addressing + short context carry-over.
    return [
        row(
            "\ub0b4 \uc774\ub984\uc740 \ubb50\uc57c?",
            "\uc544\uc9c1 \uc774\ub984\uc744 \uc54c\ub824\uc8fc\uc9c0 \uc54a\uc544\uc11c \uc0ac\uc6a9\uc790\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.",
            tier="high",
        ),
        row(
            "\uc774\ub984 \uc548 \uc54c\ub824\uc8fc\uba74 \uc5b4\ub5bb\uac8c \ubd80\ub97c \uac70\uc57c?",
            "\uc0ac\uc6a9\uc790\uac00 \uc774\ub984\uc744 \uc54c\ub824\uc8fc\uae30 \uc804\uc5d0\ub294 \uc0ac\uc6a9\uc790\ub2d8\uc73c\ub85c \ud638\uce6d\ud574\uc694.",
            tier="high",
        ),
        row(
            "\ub0b4 \uc774\ub984\uc740 \ubbfc\uc218\uc57c. \uae30\uc5b5\ud574.",
            "\uc54c\uaca0\uc5b4\uc694. \uc55e\uc73c\ub85c \ubbfc\uc218\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.",
            tier="high",
        ),
        row(
            "\ub0b4 \uc774\ub984\uc740 Alice\uc57c. \uae30\uc5b5\ud574.",
            "\uc54c\uaca0\uc5b4\uc694. \uc55e\uc73c\ub85c Alice\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.",
            tier="high",
        ),
        row(
            "[User]\n\ub0b4 \uc774\ub984\uc740 \ubbfc\uc218\uc57c.\n\n[Assistant]\n\uc54c\uaca0\uc5b4\uc694. \ubbfc\uc218\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.\n\n[User]\n\ub0b4 \uc774\ub984 \ubb50\uc57c?",
            "\ubbfc\uc218\ub2d8\uc73c\ub85c \uae30\uc5b5\ud558\uace0 \uc788\uc5b4\uc694.",
            tier="high",
        ),
        row(
            "[User]\n\ub124 \uc774\ub984\uc740?\n\n[Assistant]\n\uc800\ub294 AI \uc5b4\uc2dc\uc2a4\ud134\ud2b8\uc608\uc694.\n\n[User]\n\ub0b4 \uc774\ub984 \ubb50\uc57c?",
            "\uc544\uc9c1 \uc774\ub984\uc744 \ubc1b\uc9c0 \uc54a\uc544\uc11c \uc0ac\uc6a9\uc790\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.",
            tier="high",
        ),
        row(
            "\uc774\ub984\uc740 \uc9d3\uc5b4\ub0b4\uc9c0 \ub9c8.",
            "\uc54c\uaca0\uc5b4\uc694. \uc0ac\uc6a9\uc790\uac00 \uc9c1\uc811 \ub9d0\ud558\uc9c0 \uc54a\uc740 \uc774\ub984\uc740 \ucd94\uc815\ud558\uc9c0 \uc54a\uaca0\uc2b5\ub2c8\ub2e4.",
            tier="high",
        ),
        row(
            "\ub9e5\ub77d \uc774\uc5b4\uc11c \ub2f5\ud574\uc918.",
            "\uc54c\uaca0\uc5b4\uc694. \ubc14\ub85c \uc774\uc804 \ub300\ud654 \ud750\ub984\uc744 \uae30\uc900\uc73c\ub85c \uc774\uc5b4\uc11c \ub2f5\ud560\uac8c\uc694.",
            tier="high",
        ),
        row(
            "\ucf54\ub4dc \uc694\uccad\uc774 \uc544\ub2c8\uba74 \uc77c\ubc18 \ub300\ud654\ub85c \ub2f5\ud574.",
            "\uc54c\uaca0\uc5b4\uc694. \ucf54\ub4dc\uac00 \ud544\uc694\ud55c \uc9c8\ubb38\uc5d0\ub9cc \ucf54\ub4dc\ub97c \uc81c\uacf5\ud558\uace0, \uadf8 \uc678\uc5d0\ub294 \uc77c\ubc18 \ub300\ud654\ub85c \ub2f5\ud560\uac8c\uc694.",
            tier="high",
        ),
    ]


def build_dataset(
    chat_base_path: Path,
    extra_ko_path: Path,
    chat_repeat: int,
    policy_repeat: int,
    extra_limit: int,
    seed: int,
) -> Tuple[List[Dict], Dict]:
    rng = random.Random(int(seed))

    base_raw = load_jsonl(chat_base_path)
    base: List[Dict] = []
    for r in base_raw:
        inp = str(r.get("input", "")).strip()
        out = str(r.get("output", "")).strip()
        if not inp or not out:
            continue
        if is_noisy(inp, out):
            continue
        base.append(row(inp, out, tier="high"))
    base = dedupe(base)

    extra_raw = load_jsonl(extra_ko_path)
    extra_pool: List[Dict] = []
    for r in extra_raw:
        inp = str(r.get("input", "")).strip()
        out = str(r.get("output", "")).strip()
        if not inp or not out:
            continue
        if is_noisy(inp, out):
            continue
        if is_code_like(inp, out):
            continue
        # Keep extra rows concise enough for chat adaptation.
        if len(inp) > 300 or len(out) > 420:
            continue
        extra_pool.append(row(inp, out, tier="mid"))
    extra_pool = dedupe(extra_pool)
    rng.shuffle(extra_pool)
    extra = extra_pool[: max(0, int(extra_limit))]

    policy = dedupe(policy_rows())

    rows: List[Dict] = []
    for _ in range(max(1, int(chat_repeat))):
        rows.extend(base)
    for _ in range(max(1, int(policy_repeat))):
        rows.extend(policy)
    rows.extend(extra)
    rows = dedupe(rows)

    # Re-expand after dedupe so high-priority rows dominate sampler frequencies.
    final_rows: List[Dict] = []
    for _ in range(max(1, int(chat_repeat))):
        final_rows.extend(base)
    for _ in range(max(1, int(policy_repeat))):
        final_rows.extend(policy)
    final_rows.extend(extra)

    meta = {
        "base_rows_unique": len(base),
        "policy_rows_unique": len(policy),
        "extra_rows_unique": len(extra),
        "chat_repeat": int(chat_repeat),
        "policy_repeat": int(policy_repeat),
        "final_rows": len(final_rows),
    }
    return final_rows, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_base", default="data/ko_chat_balance_v1.jsonl")
    parser.add_argument("--extra_ko", default="data/ko_ratio_boost_v2.jsonl")
    parser.add_argument("--out", default="data/chat_router_ko_v3_clean.jsonl")
    parser.add_argument("--chat_repeat", type=int, default=7)
    parser.add_argument("--policy_repeat", type=int, default=2)
    parser.add_argument("--extra_limit", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, meta = build_dataset(
        chat_base_path=Path(args.chat_base),
        extra_ko_path=Path(args.extra_ko),
        chat_repeat=int(args.chat_repeat),
        policy_repeat=int(args.policy_repeat),
        extra_limit=int(args.extra_limit),
        seed=int(args.seed),
    )

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    payload = {
        "out": str(out_path),
        **meta,
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
