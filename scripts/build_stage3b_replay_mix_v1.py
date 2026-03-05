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
    return (str(row.get("input", "")).strip(), str(row.get("output", "")).strip())


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
    ap.add_argument("--general", required=True)
    ap.add_argument("--logic", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--general_ratio", type=float, default=0.75)
    ap.add_argument("--logic_ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    general_path = Path(args.general)
    logic_path = Path(args.logic)
    out_path = Path(args.out)
    manifest_path = Path(args.manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rnd = random.Random(int(args.seed))
    general_rows = [r for r in read_jsonl(general_path) if is_usable(r)]
    logic_rows_raw = [r for r in read_jsonl(logic_path) if is_usable(r)]

    logic_seen = set()
    logic_rows: List[Dict] = []
    logic_dupes = 0
    for row in logic_rows_raw:
        k = row_key(row)
        if k in logic_seen:
            logic_dupes += 1
            continue
        logic_seen.add(k)
        logic_rows.append(row)

    if not general_rows or not logic_rows:
        raise RuntimeError("general or logic rows are empty after filtering")

    general_ratio = float(args.general_ratio)
    logic_ratio = float(args.logic_ratio)
    if general_ratio <= 0 or logic_ratio <= 0:
        raise RuntimeError("ratios must be positive")

    target_logic = int(round(len(general_rows) * logic_ratio / general_ratio))
    mixed: List[Dict] = []
    mixed.extend(general_rows)

    logic_added = 0
    while logic_added < target_logic:
        chunk = list(logic_rows)
        rnd.shuffle(chunk)
        for row in chunk:
            if logic_added >= target_logic:
                break
            row2 = dict(row)
            row2["_stage3b_mix"] = "logic_replay"
            mixed.append(row2)
            logic_added += 1

    rnd.shuffle(mixed)

    with out_path.open("w", encoding="utf-8") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "general_path": str(general_path),
        "logic_path": str(logic_path),
        "general_rows": len(general_rows),
        "logic_rows_raw": len(logic_rows_raw),
        "logic_rows_unique": len(logic_rows),
        "logic_duplicates_removed": logic_dupes,
        "target_logic_rows": target_logic,
        "written": len(mixed),
        "general_ratio": general_ratio,
        "logic_ratio": logic_ratio,
        "seed": int(args.seed),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
