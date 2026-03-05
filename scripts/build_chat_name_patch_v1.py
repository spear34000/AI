from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def row(inp: str, out: str, tier: str = "high") -> Dict:
    return {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": tier,
        "input": inp.strip(),
        "output": out.strip(),
    }


def build_rows() -> List[Dict]:
    base_pairs: List[Tuple[str, str]] = [
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
        (
            "\uc774\ub984\uc740 \uc9d3\uc5b4\ub0b4\uc9c0 \ub9c8.",
            "\uc54c\uaca0\uc5b4\uc694. \uc0ac\uc6a9\uc790\uac00 \uc9c1\uc811 \ub9d0\ud55c \uc774\ub984\ub9cc \uc0ac\uc6a9\ud560\uac8c\uc694.",
        ),
        (
            "[User]\n\ub0b4 \uc774\ub984\uc740 \ubbfc\uc218\uc57c.\n\n[Assistant]\n\uc54c\uaca0\uc5b4\uc694. \ubbfc\uc218\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.\n\n[User]\n\ub0b4 \uc774\ub984 \ubb50\uc57c?",
            "\ubbfc\uc218\ub2d8\uc73c\ub85c \uae30\uc5b5\ud558\uace0 \uc788\uc5b4\uc694.",
        ),
        (
            "[User]\n\ub124 \uc774\ub984\uc740?\n\n[Assistant]\n\uc800\ub294 AI \uc5b4\uc2dc\uc2a4\ud134\ud2b8\uc608\uc694.\n\n[User]\n\ub0b4 \uc774\ub984 \ubb50\uc57c?",
            "\uc544\uc9c1 \uc774\ub984\uc744 \ubc1b\uc9c0 \uc54a\uc544\uc11c \uc0ac\uc6a9\uc790\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.",
        ),
        (
            "\ub108\uc758 \uc774\ub984\uc740?",
            "\uc800\ub294 \ud55c\uad6d\uc5b4 \ub300\ud654\ub97c \ub3d5\ub294 AI \uc5b4\uc2dc\uc2a4\ud134\ud2b8\uc608\uc694.",
        ),
        (
            "\uc548\ub155",
            "\uc548\ub155\ud558\uc138\uc694. \ubb34\uc5c7\uc744 \ub3c4\uc640\ub4dc\ub9b4\uae4c\uc694?",
        ),
        (
            "\ub9e5\ub77d \uc774\uc5b4\uc11c \ub2f5\ud574\uc918.",
            "\uc54c\uaca0\uc5b4\uc694. \uc774\uc804 \ub300\ud654 \ud750\ub984\uc744 \ub530\ub77c\uc11c \uc774\uc5b4\uc11c \ub2f5\ud560\uac8c\uc694.",
        ),
    ]

    suffixes = [
        "\uc9e7\uac8c \ub2f5\ud574\uc918.",
        "\ud55c \uc904\ub85c \ub2f5\ud574\uc918.",
        "\uac04\ub2e8\ud558\uac8c \ub9d0\ud574\uc918.",
        "\uc790\uc5f0\uc2a4\ub7fd\uac8c \ub2f5\ud574\uc918.",
    ]

    rows: List[Dict] = []
    for inp, out in base_pairs:
        rows.append(row(inp, out, tier="high"))
        rows.append(row(inp, out, tier="high"))
        rows.append(row(inp, out, tier="high"))
        for s in suffixes:
            rows.append(row(f"{inp} {s}", out, tier="high"))

    # Extra reinforcement for name memory behavior.
    hot_pairs = [
        (
            "\ub0b4 \uc774\ub984 \ubb50\uc57c?",
            "\uc544\uc9c1 \uc774\ub984\uc744 \uc54c\ub824\uc8fc\uc9c0 \uc54a\uc544\uc11c \uc0ac\uc6a9\uc790\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.",
        ),
        (
            "\ub0b4 \uc774\ub984\uc740 \ubbfc\uc218\uc57c. \uae30\uc5b5\ud574.",
            "\uc54c\uaca0\uc5b4\uc694. \uc55e\uc73c\ub85c \ubbfc\uc218\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.",
        ),
    ]
    for _ in range(24):
        for inp, out in hot_pairs:
            rows.append(row(inp, out, tier="high"))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/chat_name_patch_v1.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({"out": str(out_path), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
