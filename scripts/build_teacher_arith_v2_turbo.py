"""Curate the v2 dataset down to ~80k rows for 20x faster training.

Strategy:
  - Keep higher ratio of hard examples (carry, borrow, mul_2x2, div_hard)
  - Subsample easy/mid examples heavily
  - Total: ~80k rows  (vs 580k in full v2)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

SRC = Path("data/teacher_arith_pure_mix_v2_train.jsonl")
TRAIN_OUT = Path("data/teacher_arith_pure_mix_v2_turbo_train.jsonl")
EVAL_OUT = Path("data/teacher_arith_pure_mix_v2_turbo_eval.jsonl")
MANIFEST_OUT = Path("data/teacher_arith_pure_mix_v2_turbo.manifest.json")

# How many to keep per category
KEEP = {
    # Hard (where model fails) — keep many
    "add_carry_d2": 8000,
    "add_carry_d3": 8000,
    "sub_borrow_d2": 8000,
    "sub_borrow_d3": 8000,
    "mul_2x2": 12000,
    "div_3by1": 5000,
    "div_4by2": 5000,
    # Mid
    "addsub_nocarry_d2": 3000,
    "addsub_nocarry_d3": 3000,
    "mul_2x1": 4000,
    "div_2by1": 3000,
    "linear_x30": 2000,
    "linear_x60": 2000,
    # Easy (model already good — minimal)
    "easy_addsub_d1": 2000,
    "easy_mul_1x1": 2000,
    "easy_div_1by1": 1500,
}


def main() -> None:
    # Load and bucket by category
    buckets: Dict[str, List[str]] = {}
    with SRC.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cat = row.get("category", "unknown")
            buckets.setdefault(cat, []).append(line)

    rng = random.Random(20260320)
    train_rows: List[str] = []
    cat_counts: Dict[str, int] = {}

    for cat, lines in buckets.items():
        limit = KEEP.get(cat, 2000)
        rng.shuffle(lines)
        selected = lines[:limit]
        train_rows.extend(selected)
        cat_counts[cat] = len(selected)

    rng.shuffle(train_rows)

    # Write
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_OUT.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(train_rows)

    # Copy eval from v2 (small, already curated)
    src_eval = Path("data/teacher_arith_pure_mix_v2_eval.jsonl")
    if src_eval.exists():
        import shutil
        shutil.copy2(src_eval, EVAL_OUT)
        eval_count = sum(1 for _ in EVAL_OUT.open("r", encoding="utf-8"))
    else:
        eval_count = 0

    manifest = {
        "source": str(SRC),
        "train_written": len(train_rows),
        "eval_written": eval_count,
        "train_by_category": cat_counts,
        "purpose": "curated subset for 20x faster training, biased toward hard examples",
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
