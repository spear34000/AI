from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SPACE_RE = re.compile(r"\s+")
HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
LETTER_RE = re.compile(r"[A-Za-z\uac00-\ud7a3]")
LABEL_ONLY_RE = re.compile(r"^\s*(?:[A-D]|[0-9]+(?:\.[0-9]+)?)\s*$")
REPEAT_RE = re.compile(r"(.)\1{12,}")
INTRO_RE = re.compile(r"(?:\uc790\uae30\uc18c\uac1c|self[\s-]?intro)", flags=re.IGNORECASE)


DEFAULT_INCLUDE_FIELDS = (
    "ko_chat_general",
    "ko_fluency_targeted",
    "routing_intent",
    "reasoning_logic",
    "translation_multilingual",
    "memory_continual",
)

DEFAULT_EXCLUDE_FIELDS = (
    "stage_curriculum",
    "foundation_mit_mix",
    "profile_meta",
    "ccl_specbook",
    "identity_persona",
    "coding_tech",
)

DEFAULT_EXCLUDE_PATH_TOKENS = (
    "intro_override",
    "ko_ultra_targeted",
    "ko_clean_plus_targeted",
    "ko_fluency_targeted_v1",
    "pure_ko_seed",
    "pure_ko_mix_nofixed",
    "quality/hq_ko_chat_v1",
    "continual_buffer_v2_test",
    "session_memory_v1_test",
)

DEFAULT_BANNED_OUTPUT_SUBSTRINGS = (
    "\uc9c8\ubb38 \uc758\ub3c4\ub97c \ud30c\uc545\ud574 \ub2e8\uacc4\ubcc4\ub85c \uc815\ub9ac\ud574\ub4dc\ub9b4\uac8c\uc694",
    "\uc9c8\ubb38\uc744 \uc774\ud574\ud558\uace0 \uac04\uacb0\ud558\uac8c \ub2f5\ud558\ub294 \ud55c\uad6d\uc5b4 ai",
    "\ud55c\uad6d\uc5b4 ai \uc5b4\uc2dc\uc2a4\ud134\ud2b8\uc785\ub2c8\ub2e4",
    "\uc6d0\ud558\uba74 \uc9e7\uac8c 3\uc904 \uc694\uc57d\uc73c\ub85c\ub3c4 \ub2f5\ud560 \uc218 \uc788\uc5b4\uc694",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Korean pure-learning dataset from data/by_field FILES.tsv")
    p.add_argument("--data_root", type=Path, default=Path("data"))
    p.add_argument("--by_field_root", type=Path, default=Path("data/by_field"))
    p.add_argument("--out_jsonl", type=Path, default=Path("data/by_field_ko_pure_v1.jsonl"))
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/by_field_ko_pure_v1.manifest.json"),
    )
    p.add_argument(
        "--sources_tsv",
        type=Path,
        default=Path("data/by_field_ko_pure_v1.sources.tsv"),
    )
    p.add_argument("--include_field", action="append", default=None, help="field to include; repeatable")
    p.add_argument("--exclude_field", action="append", default=None, help="field to exclude; repeatable")
    p.add_argument("--exclude_path_token", action="append", default=None, help="substring in source path to exclude")
    p.add_argument(
        "--ban_output_substring",
        action="append",
        default=None,
        help="substring in output text to drop; repeatable",
    )
    p.add_argument(
        "--ban_input_substring",
        action="append",
        default=None,
        help="substring in input text to drop; repeatable",
    )
    p.add_argument("--max_rows_per_field", type=int, default=28000)
    p.add_argument("--max_scan_rows_per_file", type=int, default=18000)
    p.add_argument("--max_total_rows", type=int, default=130000, help="0 means unlimited")
    p.add_argument("--min_input_chars", type=int, default=2)
    p.add_argument("--max_input_chars", type=int, default=1800)
    p.add_argument("--min_output_chars", type=int, default=8)
    p.add_argument("--max_output_chars", type=int, default=2400)
    p.add_argument("--min_hangul_ratio", type=float, default=0.12)
    p.add_argument("--min_hangul_chars", type=int, default=6)
    p.add_argument("--max_repeat_output", type=int, default=30)
    p.add_argument("--max_repeat_output_short", type=int, default=10)
    p.add_argument("--short_output_chars", type=int, default=72)
    p.add_argument("--max_intro_rows", type=int, default=260)
    p.add_argument("--max_intro_repeat_output", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
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


def hangul_count(text: str) -> int:
    return len(HANGUL_RE.findall(str(text or "")))


def hangul_ratio(text: str) -> float:
    letters = LETTER_RE.findall(str(text or ""))
    if not letters:
        return 0.0
    ko = HANGUL_RE.findall(str(text or ""))
    return float(len(ko)) / float(max(1, len(letters)))


def row_is_ko_hint(row: Dict) -> bool:
    for key in ("language", "segment_tag", "task_type"):
        value = str(row.get(key, "")).strip().lower()
        if value in {"ko", "korean"}:
            return True
    return False


def stable_dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def is_intro_prompt(inp: str) -> bool:
    return bool(INTRO_RE.search(str(inp or "")))


def contains_any_substring(text: str, needles: Sequence[str]) -> bool:
    s = normalize_space(text).lower()
    return any(normalize_space(n).lower() in s for n in needles if normalize_space(n))


def parse_field_file_list(path: Path) -> List[str]:
    out: List[str] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cols = s.split("\t")
            if not cols:
                continue
            rel = cols[0].strip()
            if not rel or rel.lower() == "path":
                continue
            out.append(rel.replace("\\", "/"))
    return out


def iter_records(path: Path, max_rows: int) -> Iterable[Dict]:
    ext = path.suffix.lower()
    limit = int(max_rows)
    yielded = 0

    if ext == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if limit > 0 and yielded >= limit:
                    break
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                yielded += 1
                yield obj
        return

    if ext == ".json":
        try:
            raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(raw, dict):
            yield raw
            return
        if isinstance(raw, list):
            for obj in raw:
                if limit > 0 and yielded >= limit:
                    break
                if not isinstance(obj, dict):
                    continue
                yielded += 1
                yield obj


def clean_row(
    row: Dict,
    source_path: str,
    field: str,
    min_input_chars: int,
    max_input_chars: int,
    min_output_chars: int,
    max_output_chars: int,
    min_hangul_ratio: float,
    min_hangul_chars: int,
    banned_input_substrings: Sequence[str],
    banned_output_substrings: Sequence[str],
) -> Tuple[bool, str, Dict | None, str]:
    inp = pick_first_non_empty(row, ("input", "instruction", "prompt", "question", "context"))
    out = pick_first_non_empty(row, ("output", "response", "answer", "completion", "target"))
    if not inp or not out:
        return False, "missing_pair", None, ""

    if len(inp) < int(min_input_chars) or len(inp) > int(max_input_chars):
        return False, "input_length", None, ""
    if len(out) < int(min_output_chars) or len(out) > int(max_output_chars):
        return False, "output_length", None, ""

    if contains_any_substring(inp, banned_input_substrings):
        return False, "banned_input_substring", None, ""
    if contains_any_substring(out, banned_output_substrings):
        return False, "banned_output_substring", None, ""

    if "\ufffd" in inp or "\ufffd" in out:
        return False, "replacement_char", None, ""

    if REPEAT_RE.search(out):
        return False, "repeat_noise", None, ""
    if LABEL_ONLY_RE.fullmatch(out):
        return False, "label_only", None, ""

    norm_in = normalize_space(inp)
    norm_out = normalize_space(out)
    if not norm_in or not norm_out:
        return False, "empty_after_norm", None, ""
    if norm_in.lower() == norm_out.lower():
        return False, "echo", None, ""

    if sum(ch.isalnum() for ch in norm_out) < max(8, len(norm_out) // 16):
        return False, "noisy_output", None, ""

    merged = f"{norm_in}\n{norm_out}"
    h_count = hangul_count(merged)
    h_ratio = hangul_ratio(merged)
    if not row_is_ko_hint(row):
        if h_count < int(min_hangul_chars) or h_ratio < float(min_hangul_ratio):
            return False, "non_ko", None, ""

    tier = str(row.get("_meta_quality_tier", "")).strip().lower()
    if tier not in {"high", "mid"}:
        tier = "high"

    cleaned: Dict[str, str] = {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": tier,
        "input": norm_in,
        "output": norm_out,
        "_meta_source_file": source_path,
        "_meta_source_field": field,
    }
    source = str(row.get("source", "")).strip()
    source_dataset = str(row.get("source_dataset", "")).strip()
    if source:
        cleaned["source"] = source
    if source_dataset:
        cleaned["source_dataset"] = source_dataset

    return True, "ok", cleaned, norm_out


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))

    data_root = Path(args.data_root)
    by_field_root = Path(args.by_field_root)
    out_path = Path(args.out_jsonl)
    manifest_path = Path(args.manifest)
    sources_tsv_path = Path(args.sources_tsv)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if not by_field_root.exists():
        raise FileNotFoundError(f"by_field_root not found: {by_field_root}")

    include_fields = list(args.include_field) if args.include_field else list(DEFAULT_INCLUDE_FIELDS)
    include_fields = list(dict.fromkeys([str(f).strip() for f in include_fields if str(f).strip()]))

    exclude_fields = set(DEFAULT_EXCLUDE_FIELDS)
    if args.exclude_field:
        exclude_fields.update(str(f).strip() for f in args.exclude_field if str(f).strip())

    exclude_path_tokens = list(DEFAULT_EXCLUDE_PATH_TOKENS)
    if args.exclude_path_token:
        exclude_path_tokens.extend(str(t).strip() for t in args.exclude_path_token if str(t).strip())
    exclude_path_tokens = [t.lower() for t in dict.fromkeys(exclude_path_tokens)]

    banned_output_substrings = list(DEFAULT_BANNED_OUTPUT_SUBSTRINGS)
    if args.ban_output_substring:
        banned_output_substrings.extend(str(s) for s in args.ban_output_substring)
    banned_output_substrings = [normalize_space(s) for s in dict.fromkeys(banned_output_substrings) if normalize_space(s)]

    banned_input_substrings: List[str] = []
    if args.ban_input_substring:
        banned_input_substrings.extend(str(s) for s in args.ban_input_substring)
    banned_input_substrings = [normalize_space(s) for s in dict.fromkeys(banned_input_substrings) if normalize_space(s)]

    field_to_paths: Dict[str, List[str]] = {}
    for field in include_fields:
        if field in exclude_fields:
            continue
        tsv_path = by_field_root / field / "FILES.tsv"
        rel_paths = parse_field_file_list(tsv_path)
        if rel_paths:
            field_to_paths[field] = rel_paths

    if not field_to_paths:
        raise RuntimeError("No source paths found from by_field FILES.tsv")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sources_tsv_path.parent.mkdir(parents=True, exist_ok=True)

    dropped_reasons = Counter()
    per_field_kept = Counter()
    per_field_seen = Counter()
    per_field_dropped = defaultdict(Counter)
    per_file_kept = Counter()
    per_file_seen = Counter()
    per_file_status: Dict[str, str] = {}

    dedupe_keys = set()
    output_counter = Counter()
    intro_total = 0
    intro_output_counter = Counter()
    written_rows = 0
    output_digest = hashlib.blake2b(digest_size=16)

    with out_path.open("w", encoding="utf-8") as fw:
        for field in include_fields:
            if field not in field_to_paths:
                continue
            if field in exclude_fields:
                continue

            field_cap = int(args.max_rows_per_field)
            for rel_path in field_to_paths[field]:
                if int(args.max_total_rows) > 0 and written_rows >= int(args.max_total_rows):
                    break
                if field_cap > 0 and per_field_kept[field] >= field_cap:
                    break

                rel_norm = rel_path.replace("\\", "/")
                rel_low = rel_norm.lower()
                if any(tok in rel_low for tok in exclude_path_tokens):
                    per_file_status[rel_norm] = "excluded_path_token"
                    dropped_reasons["excluded_path_token"] += 1
                    continue

                src_path = data_root / rel_norm
                if not src_path.exists():
                    per_file_status[rel_norm] = "missing_source_file"
                    dropped_reasons["missing_source_file"] += 1
                    continue

                if src_path.suffix.lower() not in {".jsonl", ".json"}:
                    per_file_status[rel_norm] = "unsupported_extension"
                    dropped_reasons["unsupported_extension"] += 1
                    continue

                per_file_status[rel_norm] = "used"

                for row in iter_records(path=src_path, max_rows=int(args.max_scan_rows_per_file)):
                    if int(args.max_total_rows) > 0 and written_rows >= int(args.max_total_rows):
                        break
                    if field_cap > 0 and per_field_kept[field] >= field_cap:
                        break

                    per_field_seen[field] += 1
                    per_file_seen[rel_norm] += 1

                    ok, reason, cleaned, norm_out = clean_row(
                        row=row,
                        source_path=rel_norm,
                        field=field,
                        min_input_chars=int(args.min_input_chars),
                        max_input_chars=int(args.max_input_chars),
                        min_output_chars=int(args.min_output_chars),
                        max_output_chars=int(args.max_output_chars),
                        min_hangul_ratio=float(args.min_hangul_ratio),
                        min_hangul_chars=int(args.min_hangul_chars),
                        banned_input_substrings=banned_input_substrings,
                        banned_output_substrings=banned_output_substrings,
                    )
                    if not ok or cleaned is None:
                        dropped_reasons[reason] += 1
                        per_field_dropped[field][reason] += 1
                        continue

                    dedupe = stable_dedupe_key(cleaned["input"], cleaned["output"])
                    if dedupe in dedupe_keys:
                        dropped_reasons["duplicate_pair"] += 1
                        per_field_dropped[field]["duplicate_pair"] += 1
                        continue

                    repeat_cap = int(args.max_repeat_output_short) if len(norm_out) <= int(args.short_output_chars) else int(args.max_repeat_output)
                    if output_counter[norm_out] >= repeat_cap:
                        dropped_reasons["repeat_output_cap"] += 1
                        per_field_dropped[field]["repeat_output_cap"] += 1
                        continue

                    if is_intro_prompt(cleaned["input"]):
                        if intro_total >= int(args.max_intro_rows):
                            dropped_reasons["intro_total_cap"] += 1
                            per_field_dropped[field]["intro_total_cap"] += 1
                            continue
                        if intro_output_counter[norm_out] >= int(args.max_intro_repeat_output):
                            dropped_reasons["intro_repeat_cap"] += 1
                            per_field_dropped[field]["intro_repeat_cap"] += 1
                            continue
                        intro_total += 1
                        intro_output_counter[norm_out] += 1

                    dedupe_keys.add(dedupe)
                    output_counter[norm_out] += 1

                    fw.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                    output_digest.update(norm_out.encode("utf-8", errors="ignore"))
                    output_digest.update(b"\n")

                    written_rows += 1
                    per_field_kept[field] += 1
                    per_file_kept[rel_norm] += 1

    source_rows: List[Tuple[str, str, str, int, int]] = []
    for field in include_fields:
        for rel_path in field_to_paths.get(field, []):
            rel_norm = rel_path.replace("\\", "/")
            status = per_file_status.get(rel_norm, "not_used")
            source_rows.append(
                (
                    field,
                    rel_norm,
                    status,
                    int(per_file_seen.get(rel_norm, 0)),
                    int(per_file_kept.get(rel_norm, 0)),
                )
            )

    with sources_tsv_path.open("w", encoding="utf-8") as fsrc:
        fsrc.write("field\tpath\tstatus\tscanned\tkept\n")
        for field, path, status, scanned, kept in source_rows:
            fsrc.write(f"{field}\t{path}\t{status}\t{scanned}\t{kept}\n")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "data_root": data_root.as_posix(),
        "by_field_root": by_field_root.as_posix(),
        "out_jsonl": out_path.as_posix(),
        "sources_tsv": sources_tsv_path.as_posix(),
        "rows_written": int(written_rows),
        "fields_requested": include_fields,
        "fields_used": list(field_to_paths.keys()),
        "fields_excluded": sorted(exclude_fields),
        "per_field_scanned": {k: int(v) for k, v in per_field_seen.items()},
        "per_field_kept": {k: int(v) for k, v in per_field_kept.items()},
        "drop_reasons": {k: int(v) for k, v in dropped_reasons.most_common()},
        "top_field_drop_reasons": {
            field: {k: int(v) for k, v in cnt.most_common(10)}
            for field, cnt in per_field_dropped.items()
        },
        "distinct_outputs": int(len(output_counter)),
        "intro_rows_kept": int(intro_total),
        "output_digest_blake2b16": output_digest.hexdigest(),
        "filters": {
            "exclude_path_tokens": exclude_path_tokens,
            "banned_output_substrings": banned_output_substrings,
            "banned_input_substrings": banned_input_substrings,
            "max_rows_per_field": int(args.max_rows_per_field),
            "max_scan_rows_per_file": int(args.max_scan_rows_per_file),
            "max_total_rows": int(args.max_total_rows),
            "min_input_chars": int(args.min_input_chars),
            "max_input_chars": int(args.max_input_chars),
            "min_output_chars": int(args.min_output_chars),
            "max_output_chars": int(args.max_output_chars),
            "min_hangul_ratio": float(args.min_hangul_ratio),
            "min_hangul_chars": int(args.min_hangul_chars),
            "max_repeat_output": int(args.max_repeat_output),
            "max_repeat_output_short": int(args.max_repeat_output_short),
            "short_output_chars": int(args.short_output_chars),
            "max_intro_rows": int(args.max_intro_rows),
            "max_intro_repeat_output": int(args.max_intro_repeat_output),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"out_jsonl": out_path.as_posix(), "rows": int(written_rows), "manifest": manifest_path.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
