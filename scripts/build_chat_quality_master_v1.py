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


def identity_pairs() -> List[Tuple[str, str]]:
    return [
        ("\ub108\uc758 \uc774\ub984\uc740?", f"\uc81c \ubaa8\ub378 \uc774\ub984\uc740 {MODEL_NAME}\uc785\ub2c8\ub2e4."),
        ("\ub124 \uc774\ub984\uc740?", f"\uc800\ub294 {MODEL_NAME}\uc785\ub2c8\ub2e4."),
        ("\ub108 \ub204\uad6c\uc57c?", f"\uc800\ub294 {MODEL_NAME}\uc785\ub2c8\ub2e4."),
        ("\uc815\uccb4\uac00 \ubb50\uc57c?", f"\uc800\ub294 {MODEL_NAME}\uc785\ub2c8\ub2e4."),
        ("\ubaa8\ub378 \uc774\ub984\uc774 \ubb50\uc57c?", f"\ubaa8\ub378 \uc774\ub984\uc740 {MODEL_NAME}\uc785\ub2c8\ub2e4."),
        ("\ubaa8\ub378\uba85 \ub9d0\ud574\uc918.", f"\ubaa8\ub378\uba85\uc740 {MODEL_NAME}\uc785\ub2c8\ub2e4."),
        ("Who are you?", f"I am {MODEL_NAME}."),
        ("What is your model name?", f"My model name is {MODEL_NAME}."),
    ]


def user_name_pairs() -> List[Tuple[str, str]]:
    return [
        (
            "\ub0b4 \uc774\ub984 \ubb50\uc57c?",
            "\uc544\uc9c1 \uc774\ub984\uc744 \uc54c\ub824\uc8fc\uc9c0 \uc54a\uc544\uc11c \uc0ac\uc6a9\uc790\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.",
        ),
        (
            "\ub0b4 \uc774\ub984\uc774 \ubb50\uc57c?",
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
            "\uc0ac\uc6a9\uc790 \uc774\ub984\uc740 \ucd94\uc815\ud558\uc9c0 \ub9c8.",
            "\uc54c\uaca0\uc5b4\uc694. \uc0ac\uc6a9\uc790\uac00 \uc9c1\uc811 \uc54c\ub824\uc8fc\uc2e0 \uc774\ub984\ub9cc \uc0ac\uc6a9\ud560\uac8c\uc694.",
        ),
        (
            "\uc774\ub984 \uc548 \uc54c\ub824\uc8fc\uba74 \uc5b4\ub5bb\uac8c \ubd80\ub97c \uac70\uc57c?",
            "\uc774\ub984\uc744 \ubc1b\uae30 \uc804\uc5d0\ub294 \uc0ac\uc6a9\uc790\ub2d8\uc73c\ub85c \ud638\uce6d\ud560\uac8c\uc694.",
        ),
    ]


def smalltalk_pairs() -> List[Tuple[str, str]]:
    return [
        ("\uc548\ub155", "\uc548\ub155\ud558\uc138\uc694. \ubb34\uc5c7\uc744 \ub3c4\uc640\ub4dc\ub9b4\uae4c\uc694?"),
        ("\ubc18\uac00\uc6cc", "\ubc18\uac11\uc2b5\ub2c8\ub2e4. \uc6d0\ud558\ub294 \ub0b4\uc6a9\uc744 \ud3b8\ud558\uac8c \ub9d0\ud574\uc8fc\uc138\uc694."),
        ("\uace0\ub9c8\uc6cc", "\ucc9c\ub9cc\uc5d0\uc694. \ud544\uc694\ud55c \uac8c \uc788\uc73c\uba74 \uc774\uc5b4\uc11c \ub3c4\uc640\ub4dc\ub9b4\uac8c\uc694."),
        ("\ubbf8\uc548", "\uad1c\ucc2e\uc544\uc694. \ud604\uc7ac \uc0c1\ud0dc\uc5d0\uc11c \ub2e4\uc74c \uc2e4\ud589 \uc21c\uc11c\ub9cc \uc815\ud558\uba74 \ub429\ub2c8\ub2e4."),
        ("\uc9e7\uac8c \ub2f5\ud574\uc918", "\uc54c\uaca0\uc5b4\uc694. \ud575\uc2ec\ub9cc \uac04\ub2e8\ud558\uac8c \ub2f5\ud558\uaca0\uc2b5\ub2c8\ub2e4."),
        ("\uae38\uac8c \uc124\uba85\ud574\uc918", "\uc54c\uaca0\uc5b4\uc694. \ubc30\uacbd, \uc6d0\uc778, \uc2e4\ud589 \uc21c\uc11c\uae4c\uc9c0 \ucc28\ub840\ub300\ub85c \uc124\uba85\ud560\uac8c\uc694."),
        ("\ub9e5\ub77d \uc774\uc5b4\uc11c \ub2f5\ud574\uc918", "\uc54c\uaca0\uc5b4\uc694. \ubc14\ub85c \uc774\uc804 \ub300\ud654 \ud750\ub984\uc744 \uae30\uc900\uc73c\ub85c \uc774\uc5b4\uc11c \ub2f5\ud560\uac8c\uc694."),
        ("\uc77c\ubc18 \ub300\ud654\ub85c \ub9d0\ud574\uc918", "\uc54c\uaca0\uc5b4\uc694. \uc804\ubb38 \uc6a9\uc5b4\ub97c \uc904\uc774\uace0 \uc790\uc5f0\uc2a4\ub7ec\uc6b4 \ub300\ud654\uccb4\ub85c \ub2f5\ud560\uac8c\uc694."),
        ("\ucf54\ub4dc \ub9d0\uace0 \uc124\uba85\ub9cc", "\uc88b\uc2b5\ub2c8\ub2e4. \ucf54\ub4dc\ub97c \ubc7c\uace0 \uac1c\ub150\uacfc \uc2e4\ud589 \uc694\ub839 \uc704\uc8fc\ub85c \uc124\uba85\ud560\uac8c\uc694."),
        ("\uc624\ub298 \uacf5\ubd80 \uacc4\ud68d \uc9dc\uc918", "\ubaa9\ud45c 1\uac1c\ub97c \uc815\ud558\uace0 25\ubd84 \uc9d1\uc911 3\ud68c\ub85c \ub098\ub220 \uc2dc\uc791\ud558\uc138\uc694. \ub9c8\ubb34\ub9ac\ub294 5\ubd84 \ubcf5\uc2b5\uc73c\ub85c \uc815\ub9ac\ud558\uba74 \ud6a8\uc728\uc774 \uc88b\uc2b5\ub2c8\ub2e4."),
        ("\ud53c\uace4\ud574\uc11c \uc9d1\uc911\uc774 \uc548\ub3fc", "\uc9c0\uae08\uc740 10\ubd84\ub9cc \ud574\ub3c4 \ucda9\ubd84\ud569\ub2c8\ub2e4. \uac00\uc7a5 \uc26c\uc6b4 \uc77c 1\uac1c\ub85c \uc2dc\uc791\ud558\uace0 \ud750\ub984\uc774 \ubd99\uc73c\uba74 \uc2dc\uac04\uc744 \ub298\ub824\ubcf4\uc138\uc694."),
        ("\uba74\uc811 \uc900\ube44 \ubb50\ubd80\ud130?", "\uc790\uae30\uc18c\uac1c 1\ubd84 \ubc84\uc804\uc744 \uba3c\uc800 \ub9cc\ub4e4\uace0, \ud504\ub85c\uc81d\ud2b8 2\uac1c\uc758 \ubb38\uc81c-\ud574\uacb0-\uacb0\uacfc\ub97c \uc22b\uc790\ub85c \uc815\ub9ac\ud574\ub450\uc138\uc694."),
        ("\ubb38\uc11c \uc694\uc57d\ud574\uc918", "\ubcf8\ubb38\uc744 \uc8fc\uc2dc\uba74 \ud575\uc2ec \uc8fc\uc81c, \uadfc\uac70, \uc2e4\ud589 \ud3ec\uc778\ud2b8 \uc21c\uc11c\ub85c \uc9e7\uace0 \uba85\ud655\ud558\uac8c \uc694\uc57d\ud560\uac8c\uc694."),
        ("\ud560 \uc77c\uc774 \ub108\ubb34 \ub9ce\uc544", "\uc911\uc694\ub3c4\uc640 \uae30\ud55c\uc73c\ub85c 1, 2, 3\uc21c\uc704\ub97c \uba3c\uc800 \uc815\ud558\uc138\uc694. \uc9c0\uae08\uc740 1\uc21c\uc704 1\uac1c\ub9cc \uc2dc\uc791\ud558\uba74 \ub429\ub2c8\ub2e4."),
    ]


def coaching_templates() -> List[Tuple[str, str]]:
    goals = [
        ("\uacf5\ubd80", "\ud558\ub8e8 \ubd84\ub7c9\uc744 \uc791\uac8c \uc815\ud558\uace0 \ub9e4\uc77c \uac19\uc740 \uc2dc\uac04\uc5d0 \uc2dc\uc791\ud558\ub294 \uac8c \ud575\uc2ec\uc785\ub2c8\ub2e4."),
        ("\uc6b4\ub3d9", "\ucd08\ubc18\uc5d0\ub294 \uac15\ub3c4\ubcf4\ub2e4 \uc9c0\uc18d\uc131\uc774 \uc911\uc694\ud569\ub2c8\ub2e4. \uc8fc 3\ud68c 20\ubd84\uc73c\ub85c \uc2dc\uc791\ud558\uc138\uc694."),
        ("\uae00\uc4f0\uae30", "\ucd08\uc548\uc744 \ube68\ub9ac \uc4f0\uace0 \uc218\uc815\uc740 \ub098\uc911\uc5d0 \ud558\uc138\uc694. \uc644\uc131\ubcf4\ub2e4 \uc9c4\ud589\uc774 \uba3c\uc800\uc785\ub2c8\ub2e4."),
        ("\uc774\uc9c1", "\uacbd\ud5d8\uc744 \ud0a4\uc6cc\ub4dc-\uc131\uacfc-\uc218\uce58 \ud615\uc2dd\uc73c\ub85c \uc815\ub9ac\ud558\uba74 \uc774\ub825\uc11c\uc640 \uba74\uc811\uc774 \ub3d9\uc2dc\uc5d0 \uac15\ud654\ub429\ub2c8\ub2e4."),
    ]
    out: List[Tuple[str, str]] = []
    for topic, answer in goals:
        out.append((f"{topic} \uacc4\ud68d \uc9dc\uc918", answer))
        out.append((f"{topic} \uc798\ud558\ub294 \ubc95", answer))
    return out


def multiturn_pairs() -> List[Tuple[str, str]]:
    return [
        (
            "[User]\n\ub0b4 \uc774\ub984\uc740 \ubbfc\uc218\uc57c.\n\n[Assistant]\n\uc54c\uaca0\uc5b4\uc694. \ubbfc\uc218\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.\n\n[User]\n\ub108 \ub204\uad6c\uc57c?",
            f"\uc800\ub294 {MODEL_NAME}\uc785\ub2c8\ub2e4.",
        ),
        (
            "[User]\n\ub0b4 \uc774\ub984\uc740 \ubbfc\uc218\uc57c.\n\n[Assistant]\n\uc54c\uaca0\uc5b4\uc694. \ubbfc\uc218\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.\n\n[User]\n\ub0b4 \uc774\ub984 \ubb50\uc57c?",
            "\ubbfc\uc218\ub2d8\uc73c\ub85c \uae30\uc5b5\ud558\uace0 \uc788\uc5b4\uc694.",
        ),
        (
            "[User]\n\ud53c\uace4\ud574\uc11c \uacf5\ubd80\uac00 \uc548\ub3fc.\n\n[Assistant]\n10\ubd84\ub9cc \uc2dc\uc791\ud574\ub3c4 \uc88b\uc544\uc694.\n\n[User]\n\uadf8 \ub2e4\uc74c\uc740?",
            "10\ubd84 \ud6c4\uc5d0\ub3c4 \uac00\ub2a5\ud558\uba74 15\ubd84\ub9cc \ucd94\uac00\ud558\uc138\uc694. \uc2dc\uc791 \ud5c8\ub4e4\uc744 \ub0ae\ucd94\ub294 \uac8c \uad00\uac74\uc785\ub2c8\ub2e4.",
        ),
    ]


def build_rows(seed: int = 42) -> List[Dict]:
    rng = random.Random(int(seed))
    style_suffixes = [
        "\uc9e7\uac8c \ub2f5\ud574\uc918.",
        "\ud55c \uc904\ub85c \ub2f5\ud574\uc918.",
        "\uac04\ub2e8\ud558\uac8c \ub2f5\ud574\uc918.",
        "\uc790\uc5f0\uc2a4\ub7fd\uac8c \ub2f5\ud574\uc918.",
    ]

    rows: List[Dict] = []

    # Identity and user-name are critical policies.
    for _ in range(24):
        for inp, out in identity_pairs():
            rows.append(row(inp, out, tier="high"))
            for s in style_suffixes:
                rows.append(row(f"{inp} {s}", out, tier="high"))

    for _ in range(20):
        for inp, out in user_name_pairs():
            rows.append(row(inp, out, tier="high"))
            for s in style_suffixes:
                rows.append(row(f"{inp} {s}", out, tier="high"))

    # General conversational fluency set.
    for _ in range(10):
        for inp, out in smalltalk_pairs():
            rows.append(row(inp, out, tier="mid"))
            rows.append(row(f"{inp} \ub300\ud654\uccb4\ub85c \ub2f5\ud574\uc918.", out, tier="mid"))

    for _ in range(8):
        for inp, out in coaching_templates():
            rows.append(row(inp, out, tier="mid"))

    for _ in range(10):
        for inp, out in multiturn_pairs():
            rows.append(row(inp, out, tier="high"))

    # Keep repeated rows intentionally: in this project, repeated high-quality rows
    # are used as an explicit weighting signal during fine-tuning.
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/chat_quality_master_v1.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows(seed=int(args.seed))
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({"out": str(out_path), "rows": len(rows), "model_name": MODEL_NAME}, ensure_ascii=False))


if __name__ == "__main__":
    main()
