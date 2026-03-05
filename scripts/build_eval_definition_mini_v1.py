from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


PATTERNS = [
    re.compile(r"이란\??$"),
    re.compile(r"무엇(인가|이야|이냐)\??$"),
    re.compile(r"설명해"),
    re.compile(r"정의"),
]


def match_prompt(text: str) -> bool:
    s = str(text).strip()
    return any(p.search(s) for p in PATTERNS)


def is_ok(row: Dict) -> bool:
    inp = str(row.get("input", "")).strip()
    out = str(row.get("output", "")).strip()
    if len(inp) < 3 or len(out) < 8:
        return False
    if "???" in inp or "???" in out:
        return False
    return match_prompt(inp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_rows", type=int, default=256)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    seen = set()
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict) or not is_ok(row):
                continue
            key = (str(row.get("input", "")).strip(), str(row.get("output", "")).strip())
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(rows) >= int(args.max_rows):
                break

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"written": len(rows), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
