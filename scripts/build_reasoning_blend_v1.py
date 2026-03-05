from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blend public reasoning with existing Korean high-quality corpora.")
    p.add_argument("--reason_path", type=Path, default=Path("data/public_reasoning_cached_v1.jsonl"))
    p.add_argument("--corpus_path", type=Path, default=Path("data/final_datasets_corpus_v1.jsonl"))
    p.add_argument("--defs_path", type=Path, default=Path("data/final_datasets_defs_boost_v1.jsonl"))
    p.add_argument("--out_jsonl", type=Path, default=Path("data/reasoning_blend_v1.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/reasoning_blend_v1.manifest.json"))
    p.add_argument("--reason_limit", type=int, default=21583)
    p.add_argument("--corpus_limit", type=int, default=24000)
    p.add_argument("--defs_limit", type=int, default=10000)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def normalize_space(text: str) -> str:
    return " ".join(str(text or "").replace("\r", "\n").replace("\n", " ").split()).strip()


def normalize_row(row: Dict) -> Dict | None:
    inp = normalize_space(row.get("input", row.get("instruction", row.get("prompt", row.get("question", "")))))
    out = normalize_space(row.get("output", row.get("response", row.get("answer", ""))))
    if len(inp) < 4 or len(out) < 2:
        return None
    return {
        "input": inp,
        "output": out,
        "task_type": str(row.get("task_type", "korean") or "korean"),
        "segment_tag": str(row.get("segment_tag", "ko") or "ko"),
        "language": str(row.get("language", "ko") or "ko"),
        "domain": str(row.get("domain", "ko") or "ko"),
        "source": str(row.get("source", "")),
        "task": str(row.get("task", "")),
    }


def sample_jsonl(path: Path, limit: int, seed: int) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    rng = random.Random(int(seed))
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = normalize_row(raw)
            if row is None:
                continue
            if len(rows) < int(limit):
                rows.append(row)
                continue
            j = rng.randint(0, idx)
            if j < int(limit):
                rows[j] = row
    rng.shuffle(rows)
    return rows[: int(limit)]


def dedupe(rows: Iterable[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    out: List[Dict] = []
    seen: set[Tuple[str, str]] = set()
    dropped = {"duplicate": 0, "short": 0}
    for row in rows:
        inp = normalize_space(row.get("input", ""))
        out_text = normalize_space(row.get("output", ""))
        if len(inp) < 4 or len(out_text) < 2:
            dropped["short"] += 1
            continue
        key = (inp, out_text)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        out.append(row)
    return out, dropped


def main() -> None:
    args = parse_args()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    reason_rows = sample_jsonl(args.reason_path, limit=int(args.reason_limit), seed=int(args.seed) + 11)
    corpus_rows = sample_jsonl(args.corpus_path, limit=int(args.corpus_limit), seed=int(args.seed) + 23)
    defs_rows = sample_jsonl(args.defs_path, limit=int(args.defs_limit), seed=int(args.seed) + 37)

    merged = reason_rows + corpus_rows + defs_rows
    random.Random(int(args.seed)).shuffle(merged)
    final_rows, dropped = dedupe(merged)

    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "out_jsonl": str(args.out_jsonl),
        "rows_total": int(len(final_rows)),
        "rows_by_source_group": {
            "reason": int(len(reason_rows)),
            "corpus": int(len(corpus_rows)),
            "defs": int(len(defs_rows)),
        },
        "dropped": {k: int(v) for k, v in dropped.items()},
        "inputs": {
            "reason_path": str(args.reason_path),
            "corpus_path": str(args.corpus_path),
            "defs_path": str(args.defs_path),
        },
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
