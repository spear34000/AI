from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SPACE_RE = re.compile(r"\s+")
HANGUL_RE = re.compile(r"[가-힣]")
LETTER_RE = re.compile(r"[A-Za-z가-힣]")
LABEL_ONLY_RE = re.compile(r"^\s*(?:[A-D]|[0-9]+(?:\.[0-9]+)?)\s*$")
REPEAT_RE = re.compile(r"(.)\1{10,}")
TEMPLATE_ARTIFACT_RE = re.compile(r"\{\*[^{}]{0,400}\*\}")


DEFAULT_EXCLUDE_TOKENS = (
    "quality/",
    "sources_",
    "smoke",
    "resumecheck",
    "bg_test",
    "_test",
    "/test/",
    "stages_",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curate high-quality Korean chat dataset from data/*.jsonl")
    p.add_argument("--data_root", type=Path, default=Path("data"))
    p.add_argument("--out_jsonl", type=Path, default=Path("data/quality/hq_ko_chat_v1.jsonl"))
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/quality/hq_ko_chat_v1.manifest.json"),
    )
    p.add_argument("--probe_lines", type=int, default=180)
    p.add_argument("--min_probe_chat_ratio", type=float, default=0.70)
    p.add_argument("--min_probe_ko_ratio", type=float, default=0.30)
    p.add_argument("--min_input_chars", type=int, default=2)
    p.add_argument("--max_input_chars", type=int, default=1800)
    p.add_argument("--min_output_chars", type=int, default=12)
    p.add_argument("--max_output_chars", type=int, default=2400)
    p.add_argument("--min_hangul_ratio", type=float, default=0.14)
    p.add_argument("--min_hangul_chars", type=int, default=6)
    p.add_argument("--max_rows_per_file", type=int, default=120000)
    p.add_argument("--max_total_rows", type=int, default=0, help="0 means unlimited")
    p.add_argument(
        "--exclude_token",
        action="append",
        default=[],
        help="path token to exclude; can be provided multiple times",
    )
    p.add_argument(
        "--allow_missing_tier",
        action="store_true",
        help="keep rows even if _meta_quality_tier is missing",
    )
    p.add_argument(
        "--allow_mid_tier",
        action="store_true",
        help="keep mid tier rows in addition to high tier",
    )
    return p.parse_args()


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def pick_first_non_empty(row: Dict, keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = normalize_space(str(value))
        if text:
            return text
    return ""


def hangul_ratio(text: str) -> float:
    s = str(text or "")
    letters = LETTER_RE.findall(s)
    if not letters:
        return 0.0
    ko = HANGUL_RE.findall(s)
    return float(len(ko)) / float(max(1, len(letters)))


def hangul_count(text: str) -> int:
    return len(HANGUL_RE.findall(str(text or "")))


def row_is_ko_hint(row: Dict) -> bool:
    for key in ("language", "segment_tag", "task_type"):
        value = str(row.get(key, "")).strip().lower()
        if value in {"ko", "korean"}:
            return True
    return False


def is_noise_text(text: str) -> bool:
    s = str(text or "")
    if "\ufffd" in s:
        return True
    if "??" in s:
        return True
    if REPEAT_RE.search(s):
        return True
    if TEMPLATE_ARTIFACT_RE.search(s):
        return True
    return False


def dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def should_exclude_path(rel_posix: str, exclude_tokens: Tuple[str, ...]) -> bool:
    path = rel_posix.lower()
    return any(token in path for token in exclude_tokens)


def probe_file(path: Path, probe_lines: int) -> Dict[str, float]:
    parsed = 0
    chat_like = 0
    tiered = 0
    ko_like = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if i > int(probe_lines):
                break
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue

            parsed += 1
            inp = pick_first_non_empty(row, ("input", "instruction", "prompt", "question", "context"))
            out = pick_first_non_empty(row, ("output", "response", "answer", "completion", "target"))
            if inp and out:
                chat_like += 1
            if str(row.get("_meta_quality_tier", "")).strip():
                tiered += 1
            text = f"{inp}\n{out}"
            if row_is_ko_hint(row) or hangul_ratio(text) >= 0.16:
                ko_like += 1

    if parsed <= 0:
        return {"parsed": 0.0, "chat_ratio": 0.0, "tier_ratio": 0.0, "ko_ratio": 0.0}

    return {
        "parsed": float(parsed),
        "chat_ratio": float(chat_like) / float(parsed),
        "tier_ratio": float(tiered) / float(parsed),
        "ko_ratio": float(ko_like) / float(parsed),
    }


def clean_row(
    row: Dict,
    source_file: str,
    min_input_chars: int,
    max_input_chars: int,
    min_output_chars: int,
    max_output_chars: int,
    min_hangul_ratio: float,
    min_hangul_chars: int,
    allow_missing_tier: bool,
    allow_mid_tier: bool,
) -> Tuple[bool, str, Dict | None]:
    inp = pick_first_non_empty(row, ("input", "instruction", "prompt", "question", "context"))
    out = pick_first_non_empty(row, ("output", "response", "answer", "completion", "target"))
    if not inp or not out:
        return False, "missing_pair", None

    if len(inp) < int(min_input_chars) or len(inp) > int(max_input_chars):
        return False, "input_length", None
    if len(out) < int(min_output_chars) or len(out) > int(max_output_chars):
        return False, "output_length", None

    text = f"{inp}\n{out}"
    if is_noise_text(text):
        return False, "noise", None

    if LABEL_ONLY_RE.fullmatch(out):
        return False, "label_only", None

    if normalize_space(inp).lower() == normalize_space(out).lower():
        return False, "echo", None

    alnum = sum(ch.isalnum() for ch in out)
    if alnum < max(8, len(out) // 14):
        return False, "noisy_output", None

    tier = str(row.get("_meta_quality_tier", "")).strip().lower()
    if tier:
        if tier != "high" and not (allow_mid_tier and tier == "mid"):
            return False, "tier", None
    elif not allow_missing_tier:
        return False, "missing_tier", None

    is_ko = row_is_ko_hint(row)
    h_count = hangul_count(text)
    h_ratio = hangul_ratio(text)
    if is_ko:
        if h_count < 2:
            return False, "ko_signal_weak", None
    else:
        if h_count < int(min_hangul_chars) or h_ratio < float(min_hangul_ratio):
            return False, "non_ko", None

    cleaned: Dict[str, str] = {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": "high" if tier != "mid" else "mid",
        "input": inp,
        "output": out,
        "_meta_source_file": source_file,
    }
    source = str(row.get("source", "")).strip()
    source_dataset = str(row.get("source_dataset", "")).strip()
    if source:
        cleaned["source"] = source
    if source_dataset:
        cleaned["source_dataset"] = source_dataset
    return True, "ok", cleaned


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.out_jsonl)
    manifest_path = Path(args.manifest)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    exclude_tokens = tuple(
        dict.fromkeys([*(t.lower() for t in DEFAULT_EXCLUDE_TOKENS), *(t.lower() for t in args.exclude_token)])
    )

    all_files = sorted(data_root.rglob("*.jsonl"))
    path_excluded = Counter()
    probe_rejected = Counter()
    selected_files: List[Path] = []
    probe_stats: Dict[str, Dict[str, float]] = {}

    for path in all_files:
        rel_posix = path.relative_to(data_root).as_posix()
        if should_exclude_path(rel_posix, exclude_tokens):
            path_excluded["excluded_by_path_token"] += 1
            continue
        if path.resolve() == out_path.resolve():
            path_excluded["out_path"] += 1
            continue
        if path.stat().st_size <= 0:
            path_excluded["empty_file"] += 1
            continue

        stats = probe_file(path=path, probe_lines=int(args.probe_lines))
        probe_stats[rel_posix] = stats
        if stats["parsed"] <= 0:
            probe_rejected["probe_unparsed"] += 1
            continue
        if stats["chat_ratio"] < float(args.min_probe_chat_ratio):
            probe_rejected["probe_not_chat"] += 1
            continue
        if stats["ko_ratio"] < float(args.min_probe_ko_ratio):
            probe_rejected["probe_not_ko"] += 1
            continue
        if not args.allow_missing_tier and stats["tier_ratio"] <= 0.0:
            probe_rejected["probe_no_tier"] += 1
            continue
        selected_files.append(path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    row_rejected = Counter()
    total_written = 0
    per_file_stats: List[Dict[str, int | str]] = []
    max_total_rows = int(args.max_total_rows)

    with out_path.open("w", encoding="utf-8") as wf:
        for path in selected_files:
            rel = path.relative_to(data_root).as_posix()
            file_loaded = 0
            file_kept = 0
            file_rejected = Counter()

            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    file_loaded += 1

                    try:
                        row = json.loads(s)
                    except json.JSONDecodeError:
                        file_rejected["json_decode"] += 1
                        row_rejected["json_decode"] += 1
                        continue
                    if not isinstance(row, dict):
                        file_rejected["not_dict"] += 1
                        row_rejected["not_dict"] += 1
                        continue

                    ok, reason, cleaned = clean_row(
                        row=row,
                        source_file=rel,
                        min_input_chars=int(args.min_input_chars),
                        max_input_chars=int(args.max_input_chars),
                        min_output_chars=int(args.min_output_chars),
                        max_output_chars=int(args.max_output_chars),
                        min_hangul_ratio=float(args.min_hangul_ratio),
                        min_hangul_chars=int(args.min_hangul_chars),
                        allow_missing_tier=bool(args.allow_missing_tier),
                        allow_mid_tier=bool(args.allow_mid_tier),
                    )
                    if not ok or cleaned is None:
                        file_rejected[reason] += 1
                        row_rejected[reason] += 1
                        continue

                    key = dedupe_key(cleaned["input"], cleaned["output"])
                    if key in seen:
                        file_rejected["duplicate"] += 1
                        row_rejected["duplicate"] += 1
                        continue
                    seen.add(key)

                    wf.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                    file_kept += 1
                    total_written += 1

                    if int(args.max_rows_per_file) > 0 and file_kept >= int(args.max_rows_per_file):
                        break
                    if max_total_rows > 0 and total_written >= max_total_rows:
                        break

                if max_total_rows > 0 and total_written >= max_total_rows:
                    per_file_stats.append(
                        {
                            "file": rel,
                            "loaded_rows": int(file_loaded),
                            "kept_rows": int(file_kept),
                            "rejected_rows": int(sum(file_rejected.values())),
                        }
                    )
                    break

            per_file_stats.append(
                {
                    "file": rel,
                    "loaded_rows": int(file_loaded),
                    "kept_rows": int(file_kept),
                    "rejected_rows": int(sum(file_rejected.values())),
                }
            )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "out_jsonl": str(out_path),
        "rows_written": int(total_written),
        "files_total": int(len(all_files)),
        "files_selected": int(len(selected_files)),
        "files_excluded_by_path": dict(path_excluded),
        "files_rejected_by_probe": dict(probe_rejected),
        "row_rejected_counts": dict(row_rejected),
        "per_file_stats": per_file_stats,
        "filter_config": {
            "probe_lines": int(args.probe_lines),
            "min_probe_chat_ratio": float(args.min_probe_chat_ratio),
            "min_probe_ko_ratio": float(args.min_probe_ko_ratio),
            "min_input_chars": int(args.min_input_chars),
            "max_input_chars": int(args.max_input_chars),
            "min_output_chars": int(args.min_output_chars),
            "max_output_chars": int(args.max_output_chars),
            "min_hangul_ratio": float(args.min_hangul_ratio),
            "min_hangul_chars": int(args.min_hangul_chars),
            "max_rows_per_file": int(args.max_rows_per_file),
            "max_total_rows": int(args.max_total_rows),
            "allow_missing_tier": bool(args.allow_missing_tier),
            "allow_mid_tier": bool(args.allow_mid_tier),
            "exclude_tokens": list(exclude_tokens),
        },
        "probe_stats": probe_stats,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_jsonl": str(out_path),
                "manifest": str(manifest_path),
                "rows_written": int(total_written),
                "files_selected": int(len(selected_files)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
