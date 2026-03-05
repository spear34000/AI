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
    spear_pairs: List[Tuple[str, str]] = [
        ("\ub108\uc758 \uc774\ub984\uc740?", "\uc800\ub294 spear1.0\uc785\ub2c8\ub2e4."),
        ("\ub124 \uc774\ub984\uc740?", "\uc800\ub294 spear1.0\uc785\ub2c8\ub2e4."),
        ("\uc774\ub984\uc774 \ubb50\uc57c?", "\uc81c \ubaa8\ub378 \uc774\ub984\uc740 spear1.0\uc785\ub2c8\ub2e4."),
        ("\ubaa8\ub378 \uc774\ub984\uc774 \ubb50\uc57c?", "\ubaa8\ub378 \uc774\ub984\uc740 spear1.0\uc785\ub2c8\ub2e4."),
    ]

    user_pairs: List[Tuple[str, str]] = [
        (
            "\ub0b4 \uc774\ub984 \ubb50\uc57c?",
            "\uc544\uc9c1 \uc774\ub984\uc744 \uc54c\ub824\uc8fc\uc9c0 \uc54a\uc544\uc11c \uc0ac\uc6a9\uc790\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.",
        ),
        (
            "\ub0b4 \uc774\ub984\uc740 \ubbfc\uc218\uc57c. \uae30\uc5b5\ud574.",
            "\uc54c\uaca0\uc5b4\uc694. \uc55e\uc73c\ub85c \ubbfc\uc218\ub2d8\uc73c\ub85c \ubd80\ub97c\uac8c\uc694.",
        ),
    ]

    rows: List[Dict] = []

    for _ in range(90):
        for inp, out in spear_pairs:
            rows.append(row(inp, out, tier="high"))
            rows.append(row(f"{inp} \ud55c \uc904\ub85c \ub2f5\ud574\uc918.", out, tier="high"))
            rows.append(row(f"{inp} \uc9e7\uac8c \ub2f5\ud574\uc918.", out, tier="high"))

    for _ in range(40):
        for inp, out in user_pairs:
            rows.append(row(inp, out, tier="high"))
            rows.append(row(f"{inp} \ud55c \uc904\ub85c \ub2f5\ud574\uc918.", out, tier="high"))

    rows.append(row("\uc548\ub155", "\uc548\ub155\ud558\uc138\uc694. \ubb34\uc5c7\uc744 \ub3c4\uc640\ub4dc\ub9b4\uae4c\uc694?", tier="mid"))
    rows.append(row("\uace0\ub9c8\uc6cc", "\ucc9c\ub9cc\uc5d0\uc694. \ud544\uc694\ud55c \uac8c \uc788\uc73c\uba74 \ub9d0\ud574\uc8fc\uc138\uc694.", tier="mid"))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/spear_identity_hotfix_v1.jsonl")
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
