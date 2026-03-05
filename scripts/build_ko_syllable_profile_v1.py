from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Sequence


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def pick_first_non_empty(row: Dict, keys: Sequence[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Korean syllable frequency profile")
    ap.add_argument(
        "--sources",
        default="data/slm_mit_unified_v4_ko_only.jsonl,data/pure_ko_seed_v1.jsonl,data/intro_override_v1.jsonl",
    )
    ap.add_argument("--top_k", type=int, default=1800)
    ap.add_argument("--out", default="data/ko_syllable_profile_v1.json")
    args = ap.parse_args()

    counter: Counter[str] = Counter()
    total_rows = 0
    total_syl = 0
    for token in str(args.sources).split(","):
        p = Path(token.strip())
        if not p.exists():
            continue
        for row in iter_jsonl(p):
            total_rows += 1
            text = pick_first_non_empty(row, ["output", "response", "answer", "completion", "target"])
            for ch in re.findall(r"[가-힣]", str(text)):
                counter[ch] += 1
                total_syl += 1

    top = counter.most_common(max(1, int(args.top_k)))
    payload = {
        "format": "ko_syllable_profile_v1",
        "rows_seen": int(total_rows),
        "total_syllables": int(total_syl),
        "top_k": int(args.top_k),
        "syllables": [{"ch": ch, "count": int(cnt)} for ch, cnt in top],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"out": str(out), "rows_seen": int(total_rows), "total_syllables": int(total_syl), "kept": len(top)},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

