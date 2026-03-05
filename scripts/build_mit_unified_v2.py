from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_INPUTS = [
    "data/slm_mit_all_local_v1.jsonl",
    "data_archive_20260219_160805/slm_mit_only_expanded.jsonl",
    "data_archive_20260219_160805/slm_mit_only_expanded_more.jsonl",
    "data_archive_20260219_160805/slm_multilingual_permissive_merged.jsonl",
    "data_archive_20260219_160805/slm_multilingual_mit_pd_only.jsonl",
]


def normalize_license(value: str) -> str:
    return " ".join(str(value).strip().split()).lower()


def is_mit_license(value: str) -> bool:
    lic = normalize_license(value)
    return lic == "mit" or lic == "mit license"


def stable_row_key(row: Dict) -> str:
    task = str(row.get("task_type", ""))
    inp = str(row.get("input", ""))
    out = str(row.get("output", ""))
    raw = f"{task}\n{inp}\n{out}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out_jsonl", default="data/slm_mit_unified_v2.jsonl")
    parser.add_argument("--manifest", default="data/slm_mit_unified_v2.manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--forbidden_terms", nargs="+", default=["\uc784\ucc3d\ube48"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))

    inputs = [Path(p) for p in args.inputs]
    out_jsonl = Path(args.out_jsonl)
    manifest_path = Path(args.manifest)

    records: List[Dict] = []
    seen = set()
    excluded = Counter()
    file_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"read_rows": 0, "kept_rows": 0})

    forbidden_terms = [str(t) for t in args.forbidden_terms if str(t).strip()]

    for path in inputs:
        key = str(path)
        if not path.exists():
            excluded["missing_input_file"] += 1
            continue

        for row in iter_jsonl(path):
            file_stats[key]["read_rows"] += 1

            license_value = str(row.get("license", ""))
            if not is_mit_license(license_value):
                excluded["non_mit_license"] += 1
                continue

            source_dataset = str(row.get("source_dataset", "")).lower()
            if source_dataset.startswith("synthetic_") or "generated in-project" in normalize_license(license_value):
                excluded["synthetic_row"] += 1
                continue

            inp = str(row.get("input", "")).strip()
            out = str(row.get("output", "")).strip()
            if not inp or not out:
                excluded["empty_io"] += 1
                continue

            text = f"{inp}\n{out}"
            if any(term in text for term in forbidden_terms):
                excluded["forbidden_term"] += 1
                continue

            row_key = stable_row_key(row)
            if row_key in seen:
                excluded["duplicate"] += 1
                continue
            seen.add(row_key)
            records.append(row)
            file_stats[key]["kept_rows"] += 1

    if args.shuffle:
        random.shuffle(records)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    task_counts = Counter()
    language_counts = Counter()
    source_counts = Counter()
    for row in records:
        task_counts[str(row.get("task_type", ""))] += 1
        language_counts[str(row.get("language", ""))] += 1
        source_counts[str(row.get("source_dataset", ""))] += 1

    manifest = {
        "inputs": [str(p) for p in inputs],
        "output_jsonl": str(out_jsonl),
        "rows_total": len(records),
        "task_counts": dict(task_counts),
        "language_counts": dict(language_counts),
        "source_counts_top30": dict(source_counts.most_common(30)),
        "forbidden_terms": forbidden_terms,
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "excluded_counts": dict(excluded),
        "file_stats": dict(file_stats),
        "license_policy": "MIT only",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(out_jsonl),
                "rows_total": len(records),
                "task_counts": dict(task_counts),
                "language_counts": dict(language_counts),
                "excluded_counts": dict(excluded),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
