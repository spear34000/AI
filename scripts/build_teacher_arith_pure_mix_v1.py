from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List


TRAIN_OUT = Path("data/teacher_arith_pure_mix_v1_train.jsonl")
EVAL_OUT = Path("data/teacher_arith_pure_mix_v1_eval.jsonl")
MANIFEST_OUT = Path("data/teacher_arith_pure_mix_v1.manifest.json")

TRAIN_SOURCES = [
    Path("data/teacher_exact_arith_curriculum_v3_train.jsonl"),
    Path("data/teacher_symbolic_arith_mastery_v1_train.jsonl"),
    Path("data/teacher_arith_bridge_v2_train.jsonl"),
]

EVAL_SOURCES = [
    Path("data/teacher_exact_arith_curriculum_v3_eval.jsonl"),
    Path("data/teacher_symbolic_arith_mastery_v1_eval.jsonl"),
    Path("data/teacher_arith_bridge_v2_eval.jsonl"),
]


def read_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_row(row: Dict[str, str], source_name: str, split: str) -> Dict[str, str]:
    out = dict(row)
    out["split"] = split
    out["segment_tag"] = "ko_teacher_arith_pure_mix_v1"
    out["_meta_mix_source"] = source_name
    return out


def shuffled_sample(rows: List[Dict[str, str]], rng: random.Random, limit: int) -> List[Dict[str, str]]:
    pool = list(rows)
    rng.shuffle(pool)
    if limit > 0:
        pool = pool[:limit]
    return pool


def main() -> None:
    rng = random.Random(20260316)

    train_rows: List[Dict[str, str]] = []
    train_limits = {
        "teacher_exact_arith_curriculum_v3_train.jsonl": 176000,
        "teacher_symbolic_arith_mastery_v1_train.jsonl": 180000,
        "teacher_arith_bridge_v2_train.jsonl": 180000,
    }
    source_counts: Dict[str, int] = {}
    for src in TRAIN_SOURCES:
        rows = read_rows(src)
        sampled = shuffled_sample(rows, rng, train_limits[src.name])
        source_counts[src.name] = len(sampled)
        train_rows.extend(normalize_row(row, src.name, "train") for row in sampled)
    rng.shuffle(train_rows)

    eval_rows: List[Dict[str, str]] = []
    eval_counts: Dict[str, int] = {}
    for src in EVAL_SOURCES:
        rows = read_rows(src)
        eval_counts[src.name] = len(rows)
        eval_rows.extend(normalize_row(row, src.name, "eval") for row in rows)
    rng.shuffle(eval_rows)

    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(EVAL_OUT, eval_rows)
    manifest = {
        "seed": 20260316,
        "train_written": len(train_rows),
        "eval_written": len(eval_rows),
        "train_sources": source_counts,
        "eval_sources": eval_counts,
        "purpose": "pure arithmetic from-scratch mix",
        "sources": [str(p) for p in TRAIN_SOURCES + EVAL_SOURCES],
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
