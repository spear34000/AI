from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple


DEFAULT_INPUT = Path("data/hf_mit_language_superpack_v1.jsonl")
DEFAULT_OUTPUT = Path("data/hf_mit_language_superpack_hq_v2.jsonl")
DEFAULT_MANIFEST = Path("data/hf_mit_language_superpack_hq_v2.manifest.json")


DEFAULT_EXCLUDE_PREFIXES = (
    "openai/MMMLU",
    "HAERAE-HUB/HRM8K",
)

LABEL_ONLY_RE = re.compile(r"^\s*(?:[A-D]|[0-9]+(?:\.[0-9]+)?)\s*$")
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter MIT language pack to high-quality conversational rows.")
    p.add_argument("--input_jsonl", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out_jsonl", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--min_input_chars", type=int, default=8)
    p.add_argument("--min_output_chars", type=int, default=48)
    p.add_argument(
        "--exclude_prefix",
        action="append",
        default=[],
        help="Dataset prefix to exclude; can be provided multiple times",
    )
    return p.parse_args()


def iter_rows(path: Path) -> Iterable[Dict]:
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


def norm_text(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "").strip().lower())


def dedup_key(row: Dict) -> Tuple[str, str]:
    return (norm_text(row.get("input", "")), norm_text(row.get("output", "")))


def should_keep(
    row: Dict,
    min_input_chars: int,
    min_output_chars: int,
    exclude_prefixes: Tuple[str, ...],
) -> Tuple[bool, str]:
    if str(row.get("license", "")).strip().lower() != "mit":
        return False, "non_mit"

    task_type = str(row.get("task_type", "")).strip().lower()
    if task_type not in {"english", "korean"}:
        return False, "task_type"

    source_dataset = str(row.get("source_dataset", "")).strip()
    if source_dataset and any(source_dataset.startswith(pref) for pref in exclude_prefixes):
        return False, "excluded_dataset"

    inp = str(row.get("input", "")).strip()
    out = str(row.get("output", "")).strip()
    if len(inp) < int(min_input_chars):
        return False, "short_input"
    if len(out) < int(min_output_chars):
        return False, "short_output"
    if LABEL_ONLY_RE.fullmatch(out):
        return False, "label_only"

    # Skip rows where output is mostly separators/noise.
    alpha_num = sum(ch.isalnum() for ch in out)
    if alpha_num < max(6, len(out) // 12):
        return False, "noisy_output"

    return True, ""


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    out_path = Path(args.out_jsonl)
    manifest_path = Path(args.manifest)

    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    exclude_prefixes = tuple(dict.fromkeys([*DEFAULT_EXCLUDE_PREFIXES, *args.exclude_prefix]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    seen = set()
    reason_counts = Counter()
    source_counts = Counter()
    lang_counts = Counter()

    with out_path.open("w", encoding="utf-8") as wf:
        for row in iter_rows(input_path):
            ok, reason = should_keep(
                row=row,
                min_input_chars=int(args.min_input_chars),
                min_output_chars=int(args.min_output_chars),
                exclude_prefixes=exclude_prefixes,
            )
            if not ok:
                reason_counts[reason] += 1
                continue

            key = dedup_key(row)
            if key in seen:
                reason_counts["duplicate"] += 1
                continue
            seen.add(key)

            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            source_counts[str(row.get("source_dataset", "unknown"))] += 1
            lang_counts[str(row.get("language", "unknown"))] += 1

    manifest = {
        "input_jsonl": str(input_path),
        "out_jsonl": str(out_path),
        "rows_kept": int(kept),
        "exclude_prefixes": list(exclude_prefixes),
        "filters": {
            "license": "MIT only",
            "task_type": ["english", "korean"],
            "min_input_chars": int(args.min_input_chars),
            "min_output_chars": int(args.min_output_chars),
            "drop_label_only": True,
        },
        "excluded_counts": dict(reason_counts),
        "kept_source_dataset_counts": dict(source_counts),
        "kept_language_counts": dict(lang_counts),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows_kept": int(kept), "out_jsonl": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

