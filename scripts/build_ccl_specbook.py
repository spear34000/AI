from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List


HANGUL_RE = re.compile(r"[가-힣]")
LATIN_RE = re.compile(r"[A-Za-z]")


def pick_first_non_empty(row: Dict, keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def infer_language_tag(text: str) -> str:
    hangul_n = len(HANGUL_RE.findall(text))
    latin_n = len(LATIN_RE.findall(text))
    if hangul_n >= 4 and hangul_n >= int(0.2 * max(1, latin_n)):
        return "ko"
    if latin_n > 0:
        return "en"
    return "other"


def tier_from_row(row: Dict) -> str:
    q = str(row.get("_meta_quality_tier", "base")).strip().lower()
    if q == "high":
        return "hard"
    if q == "mid":
        return "soft"
    return "observational"


def nll_threshold_by_tier(tier: str) -> float:
    t = str(tier).strip().lower()
    if t == "hard":
        return 2.6
    if t == "soft":
        return 3.1
    return 3.8


def build_specs(source_jsonl: Path, max_specs: int, seed: int, min_chars: int) -> List[Dict]:
    rnd = random.Random(seed)
    rows: List[Dict] = []

    with source_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue

            prompt = pick_first_non_empty(row, ["input", "instruction", "prompt", "question", "context"])
            target = pick_first_non_empty(row, ["output", "response", "answer", "completion", "target"])
            if len(prompt) < int(min_chars) or len(target) < int(min_chars):
                continue

            tier = tier_from_row(row)
            lang = infer_language_tag(f"{prompt}\n{target}")
            spec = {
                "id": f"spec_{len(rows):07d}",
                "tier": tier,
                "prompt": prompt,
                "target": target,
                "verifier": {
                    "type": "nll",
                    "max_token_nll": nll_threshold_by_tier(tier),
                },
                "meta": {
                    "task_type": str(row.get("task_type", "")),
                    "segment_tag": str(row.get("segment_tag", "")),
                    "language": lang,
                },
            }
            rows.append(spec)

    if not rows:
        raise RuntimeError(f"no valid rows from {source_jsonl}")

    rnd.shuffle(rows)
    if max_specs > 0:
        rows = rows[: int(max_specs)]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_jsonl", default="data/slm_mit_unified_v4.jsonl")
    parser.add_argument("--output_jsonl", default="data/ccl_specbook_v1.jsonl")
    parser.add_argument("--manifest_json", default="data/ccl_specbook_v1.manifest.json")
    parser.add_argument("--max_specs", type=int, default=2400, help="0 means all")
    parser.add_argument("--min_chars", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source_jsonl)
    if not source.exists():
        raise RuntimeError(f"source_jsonl not found: {source}")

    specs = build_specs(
        source_jsonl=source,
        max_specs=int(args.max_specs),
        seed=int(args.seed),
        min_chars=int(args.min_chars),
    )

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in specs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tier_counts = Counter(str(x.get("tier", "soft")) for x in specs)
    lang_counts = Counter(str((x.get("meta") or {}).get("language", "other")) for x in specs)
    manifest = {
        "source_jsonl": str(source),
        "output_jsonl": str(out_path),
        "rows": len(specs),
        "tier_counts": dict(tier_counts),
        "language_counts": dict(lang_counts),
        "seed": int(args.seed),
        "min_chars": int(args.min_chars),
    }

    mpath = Path(args.manifest_json)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

