from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
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
                rows.append(row)
    return rows


def row_key(row: Dict) -> Tuple[str, str]:
    return (
        str(row.get("input", "")).strip(),
        str(row.get("output", "")).strip(),
    )


def is_usable(row: Dict) -> bool:
    inp = str(row.get("input", "")).strip()
    out = str(row.get("output", "")).strip()
    if len(inp) < 3 or len(out) < 3:
        return False
    if "???" in inp or "???" in out:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--general", default="data/repair_ko_direct_v3.jsonl")
    ap.add_argument("--teacher", default="data/teacher_reasoning_long_v1_train.jsonl")
    ap.add_argument("--out", default="data/teacher_reasoning_replay_mix_v1.jsonl")
    ap.add_argument("--manifest", default="data/teacher_reasoning_replay_mix_v1.manifest.json")
    ap.add_argument("--general_ratio", type=float, default=0.70)
    ap.add_argument("--teacher_ratio", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=20260302)
    args = ap.parse_args()

    general_path = Path(args.general)
    teacher_path = Path(args.teacher)
    out_path = Path(args.out)
    manifest_path = Path(args.manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    general_rows = [r for r in read_jsonl(general_path) if is_usable(r)]
    teacher_rows_raw = [r for r in read_jsonl(teacher_path) if is_usable(r)]

    teacher_rows: List[Dict] = []
    seen = set()
    dupes = 0
    for row in teacher_rows_raw:
        key = row_key(row)
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        teacher_rows.append(row)

    if not general_rows or not teacher_rows:
        raise RuntimeError("general_rows or teacher_rows empty after filtering")

    g_ratio = float(args.general_ratio)
    t_ratio = float(args.teacher_ratio)
    if g_ratio <= 0 or t_ratio <= 0:
        raise RuntimeError("ratios must be positive")

    target_teacher = int(round(len(general_rows) * t_ratio / g_ratio))
    mixed: List[Dict] = []
    for row in general_rows:
        row2 = dict(row)
        row2["_mix_source"] = "general"
        mixed.append(row2)

    added = 0
    teacher_pool = list(teacher_rows)
    while added < target_teacher:
        rng.shuffle(teacher_pool)
        for row in teacher_pool:
            if added >= target_teacher:
                break
            row2 = dict(row)
            row2["_mix_source"] = "teacher_reasoning_long"
            mixed.append(row2)
            added += 1

    rng.shuffle(mixed)

    with out_path.open("w", encoding="utf-8") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "general_path": str(general_path),
        "teacher_path": str(teacher_path),
        "general_rows": len(general_rows),
        "teacher_rows_raw": len(teacher_rows_raw),
        "teacher_rows_unique": len(teacher_rows),
        "teacher_duplicates_removed": dupes,
        "general_ratio": g_ratio,
        "teacher_ratio": t_ratio,
        "target_teacher_rows": target_teacher,
        "written_rows": len(mixed),
        "seed": int(args.seed),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
