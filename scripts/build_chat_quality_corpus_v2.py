from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


MODEL_NAME = "spear1.0"


def row(inp: str, out: str, tier: str = "high") -> Dict:
    return {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": tier,
        "input": inp.strip(),
        "output": out.strip(),
    }


def dedupe(rows: Iterable[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for r in rows:
        inp = str(r.get("input", "")).strip()
        out_txt = str(r.get("output", "")).strip()
        if not inp or not out_txt:
            continue
        key = (inp, out_txt)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def hangul_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    h = sum(1 for c in chars if 0xAC00 <= ord(c) <= 0xD7A3)
    return float(h) / float(len(chars))


def is_clean_ko_pair(inp: str, out: str) -> bool:
    txt = f"{inp}\n{out}"
    if len(txt) < 24 or len(txt) > 2200:
        return False
    if txt.count("?") >= 4 or "??" in txt or "\ufffd" in txt:
        return False
    if hangul_ratio(txt) < 0.12:
        return False
    return True


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
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


def identity_pairs() -> List[Tuple[str, str]]:
    return [
        ("\ub108\uc758 \uc774\ub984\uc740?", f"\uc81c \ubaa8\ub378 \uc774\ub984\uc740 {MODEL_NAME}\uc785\ub2c8\ub2e4."),
        ("\ub108 \ub204\uad6c\uc57c?", f"\uc800\ub294 {MODEL_NAME}\uc785\ub2c8\ub2e4."),
        ("\uc815\uccb4\uac00 \ubb50\uc57c?", f"\uc800\ub294 {MODEL_NAME}\uc785\ub2c8\ub2e4."),
        ("\ubaa8\ub378 \uc774\ub984\uc774 \ubb50\uc57c?", f"\ubaa8\ub378 \uc774\ub984\uc740 {MODEL_NAME}\uc785\ub2c8\ub2e4."),
        ("\ubaa8\ub378\uba85 \ub9d0\ud574\uc918.", f"\ubaa8\ub378\uba85\uc740 {MODEL_NAME}\uc785\ub2c8\ub2e4."),
    ]


def user_name_pairs() -> List[Tuple[str, str]]:
    return [
        (
            "\ub0b4 \uc774\ub984 \ubb50\uc57c?",
            "\uc544\uc9c1 \uc774\ub984\uc744 \uc54c\ub824\uc8fc\uc9c0 \uc54a\uc544\uc11c \uc0ac\uc6a9\uc790\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.",
        ),
        (
            "\ub0b4 \uc774\ub984\uc740 \ubbfc\uc218\uc57c. \uae30\uc5b5\ud574.",
            "\uc54c\uaca0\uc5b4\uc694. \uc55e\uc73c\ub85c \ubbfc\uc218\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.",
        ),
        (
            "\ub0b4 \uc774\ub984\uc740 Alice\uc57c. \uae30\uc5b5\ud574.",
            "\uc54c\uaca0\uc5b4\uc694. \uc55e\uc73c\ub85c Alice\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.",
        ),
        (
            "\uc774\ub984 \uc548 \uc54c\ub824\uc8fc\uba74 \uc5b4\ub5bb\uac8c \ubd80\ub97c \uac70\uc57c?",
            "\uc774\ub984\uc744 \ubc1b\uae30 \uc804\uc5d0\ub294 \uc0ac\uc6a9\uc790\ub2d8\uc73c\ub85c \ud638\uce6d\ud560\uac8c\uc694.",
        ),
    ]


def smalltalk_pairs() -> List[Tuple[str, str]]:
    return [
        ("\uc548\ub155", "\uc548\ub155\ud558\uc138\uc694. \ubb34\uc5c7\uc744 \ub3c4\uc640\ub4dc\ub9b4\uae4c\uc694?"),
        ("\uace0\ub9c8\uc6cc", "\ucc9c\ub9cc\uc5d0\uc694. \ud544\uc694\ud55c \uac8c \uc788\uc73c\uba74 \uc774\uc5b4\uc11c \ub3c4\uc640\ub4dc\ub9b4\uac8c\uc694."),
        ("\uc624\ub298 \ub108\ubb34 \uc9c0\uce5c\ub370 \uacf5\ubd80 \uacc4\ud68d \uc9dc\uc918", "\uc9c0\uae08\uc740 10\ubd84 \uc9d1\uc911\uc73c\ub85c \uc2dc\uc791\ud558\uace0, \ud750\ub984\uc774 \ubd99\uc73c\uba74 20\ubd84\uae4c\uc9c0 \ub298\ub824\ubcf4\uc138\uc694."),
        ("\ubb38\uc11c \uc694\uc57d \uc5b4\ub5bb\uac8c \ud558\uba74 \uc798\ud574?", "\uba3c\uc800 \ud575\uc2ec \uc8fc\uc81c 1\uc904, \uadfc\uac70 2~3\uac1c, \uc2e4\ud589 \ud3ec\uc778\ud2b8 \uc21c\uc73c\ub85c \uc815\ub9ac\ud558\uba74 \ud488\uc9c8\uc774 \uc548\uc815\ub429\ub2c8\ub2e4."),
        ("\ub9e5\ub77d \uc774\uc5b4\uc11c \ub2f5\ud574\uc918", "\uc54c\uaca0\uc5b4\uc694. \uc774\uc804 \ub300\ud654 \ud750\ub984\uc744 \uae30\uc900\uc73c\ub85c \uc774\uc5b4\uc11c \ub2f5\ud560\uac8c\uc694."),
        ("\ucf54\ub4dc \ub9d0\uace0 \uc124\uba85\ub9cc \ud574\uc918", "\uc88b\uc2b5\ub2c8\ub2e4. \ucf54\ub4dc \uc5c6\uc774 \uac1c\ub150\uacfc \uc2e4\ud589 \uc21c\uc11c\ub85c \uac04\ub2e8\ud788 \uc124\uba85\ud560\uac8c\uc694."),
    ]


def collect_external_rows(ko_ratio_path: Path, balanced_path: Path, limit_ko_ratio: int, limit_balanced: int, seed: int) -> List[Dict]:
    rng = random.Random(int(seed))
    out: List[Dict] = []

    ko_ratio_raw = load_jsonl(ko_ratio_path)
    ko_ratio: List[Dict] = []
    for r in ko_ratio_raw:
        inp = str(r.get("input", "")).strip()
        out_txt = str(r.get("output", "")).strip()
        if not is_clean_ko_pair(inp, out_txt):
            continue
        # Keep mostly non-code style rows for chat fluency.
        if "```" in inp or "```" in out_txt:
            continue
        ko_ratio.append(row(inp, out_txt, tier="mid"))
    rng.shuffle(ko_ratio)
    out.extend(ko_ratio[: max(0, int(limit_ko_ratio))])

    balanced_raw = load_jsonl(balanced_path)
    balanced: List[Dict] = []
    for r in balanced_raw:
        inp = str(r.get("input", "")).strip()
        out_txt = str(r.get("output", "")).strip()
        if not is_clean_ko_pair(inp, out_txt):
            continue
        if "```" in inp or "```" in out_txt:
            continue
        balanced.append(row(inp, out_txt, tier="mid"))
    rng.shuffle(balanced)
    out.extend(balanced[: max(0, int(limit_balanced))])

    return out


def build_rows(seed: int, ko_ratio_path: Path, balanced_path: Path, limit_ko_ratio: int, limit_balanced: int) -> List[Dict]:
    rng = random.Random(int(seed))
    rows: List[Dict] = []

    # Curated anchors.
    for _ in range(16):
        for inp, out_txt in identity_pairs():
            rows.append(row(inp, out_txt, tier="high"))
            rows.append(row(f"{inp} \ud55c \uc904\ub85c \ub2f5\ud574\uc918.", out_txt, tier="high"))
    for _ in range(12):
        for inp, out_txt in user_name_pairs():
            rows.append(row(inp, out_txt, tier="high"))
    for _ in range(18):
        for inp, out_txt in smalltalk_pairs():
            rows.append(row(inp, out_txt, tier="high"))

    # External clean corpus.
    rows.extend(
        collect_external_rows(
            ko_ratio_path=ko_ratio_path,
            balanced_path=balanced_path,
            limit_ko_ratio=int(limit_ko_ratio),
            limit_balanced=int(limit_balanced),
            seed=int(seed),
        )
    )

    rows = dedupe(rows)
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/chat_quality_corpus_v2.jsonl")
    parser.add_argument("--ko_ratio", default="data/ko_ratio_boost_v2.jsonl")
    parser.add_argument("--balanced", default="data/ko_chat_code_balanced_v2.jsonl")
    parser.add_argument("--limit_ko_ratio", type=int, default=2200)
    parser.add_argument("--limit_balanced", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows(
        seed=int(args.seed),
        ko_ratio_path=Path(args.ko_ratio),
        balanced_path=Path(args.balanced),
        limit_ko_ratio=int(args.limit_ko_ratio),
        limit_balanced=int(args.limit_balanced),
    )

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "out": str(out_path),
                "rows": len(rows),
                "model_name": MODEL_NAME,
                "limit_ko_ratio": int(args.limit_ko_ratio),
                "limit_balanced": int(args.limit_balanced),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
