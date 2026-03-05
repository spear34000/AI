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

INTRO_INPUT_RE = re.compile(
    r"(?:자기소개|self[\s-]?intro|너의\s*이름|네\s*이름|이름이\s*뭐|한\s*줄\s*소개|소개해\s*줘|소개해줘)",
    flags=re.IGNORECASE,
)

DEFAULT_SOURCES = (
    "data/quality/hq_ko_chat_v1.jsonl",
    "data/ko_clean_fluency_v2.jsonl",
    "data/kullm_v2_sample.jsonl",
)

SOURCE_CAP_DEFAULTS = {
    "data/quality/hq_ko_chat_v1.jsonl": 70000,
    "data/ko_clean_fluency_v2.jsonl": 18000,
    "data/kullm_v2_sample.jsonl": 15000,
}

DEFAULT_BANNED_OUTPUT_SUBSTRINGS = (
    "질문 의도를 파악해 단계별로 정리해드릴게요",
    "원하면 짧게 3줄 요약으로도 답할 수 있어요",
    "간결하고 정확하게 답변하는 한국어 ai",
    "한국어 ai 어시스턴트입니다",
    "ai 어시스턴트입니다",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build strict pure-learning Korean dataset (v2)")
    p.add_argument("--out_jsonl", type=Path, default=Path("data/ko_pure_strict_v2.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/ko_pure_strict_v2.manifest.json"))
    p.add_argument("--sources_tsv", type=Path, default=Path("data/ko_pure_strict_v2.sources.tsv"))
    p.add_argument("--source", action="append", default=None, help="source jsonl path; repeatable")
    p.add_argument("--source_cap", action="append", default=None, help="format: path=cap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_total_rows", type=int, default=105000, help="0 means unlimited")
    p.add_argument("--min_input_chars", type=int, default=2)
    p.add_argument("--max_input_chars", type=int, default=1800)
    p.add_argument("--min_output_chars", type=int, default=8)
    p.add_argument("--max_output_chars", type=int, default=2400)
    p.add_argument("--min_hangul_chars", type=int, default=6)
    p.add_argument("--min_hangul_ratio", type=float, default=0.12)
    p.add_argument("--max_repeat_output", type=int, default=10)
    p.add_argument("--max_repeat_output_short", type=int, default=5)
    p.add_argument("--short_output_chars", type=int, default=72)
    p.add_argument("--ban_output_substring", action="append", default=None)
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


def row_is_ko_hint(row: Dict) -> bool:
    for key in ("language", "segment_tag", "task_type"):
        value = str(row.get(key, "")).strip().lower()
        if value in {"ko", "korean"}:
            return True
    return False


def hangul_count(text: str) -> int:
    return len(HANGUL_RE.findall(str(text or "")))


def hangul_ratio(text: str) -> float:
    letters = LETTER_RE.findall(str(text or ""))
    if not letters:
        return 0.0
    ko = HANGUL_RE.findall(str(text or ""))
    return float(len(ko)) / float(max(1, len(letters)))


def stable_dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def contains_any_substring(text: str, needles: Sequence[str]) -> bool:
    s = normalize_space(text).lower()
    return any(normalize_space(n).lower() in s for n in needles if normalize_space(n))


def parse_source_caps(source_cap_args: Sequence[str] | None) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not source_cap_args:
        return out
    for raw in source_cap_args:
        s = str(raw or "").strip()
        if not s or "=" not in s:
            continue
        path, cap = s.split("=", 1)
        path = path.strip().replace("\\", "/")
        try:
            c = int(cap.strip())
        except ValueError:
            continue
        if path:
            out[path] = max(0, c)
    return out


def clean_row(
    row: Dict,
    source_path: str,
    min_input_chars: int,
    max_input_chars: int,
    min_output_chars: int,
    max_output_chars: int,
    min_hangul_chars: int,
    min_hangul_ratio: float,
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

    if INTRO_INPUT_RE.search(inp):
        return False, "intro_input", None, ""
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

    sources = [s.replace("\\", "/") for s in (args.source or DEFAULT_SOURCES)]
    sources = list(dict.fromkeys([s for s in sources if s]))
    if not sources:
        raise RuntimeError("No sources configured")

    source_caps = dict(SOURCE_CAP_DEFAULTS)
    source_caps.update(parse_source_caps(args.source_cap))

    banned_output_substrings = list(DEFAULT_BANNED_OUTPUT_SUBSTRINGS)
    if args.ban_output_substring:
        banned_output_substrings.extend(str(x) for x in args.ban_output_substring)
    banned_output_substrings = [normalize_space(x) for x in dict.fromkeys(banned_output_substrings) if normalize_space(x)]

    out_path = Path(args.out_jsonl)
    manifest_path = Path(args.manifest)
    sources_tsv_path = Path(args.sources_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sources_tsv_path.parent.mkdir(parents=True, exist_ok=True)

    dropped_reasons = Counter()
    per_source_seen = Counter()
    per_source_kept = Counter()
    per_source_dropped = defaultdict(Counter)
    dedupe_keys = set()
    output_counter = Counter()
    output_digest = hashlib.blake2b(digest_size=16)

    per_source_rows: Dict[str, List[Dict]] = {}
    for src in sources:
        src_path = Path(src)
        if not src_path.exists():
            dropped_reasons["missing_source_file"] += 1
            continue

        rows: List[Dict] = []
        with src_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except json.JSONDecodeError:
                    dropped_reasons["json_decode"] += 1
                    per_source_dropped[src]["json_decode"] += 1
                    continue
                if not isinstance(row, dict):
                    dropped_reasons["non_dict"] += 1
                    per_source_dropped[src]["non_dict"] += 1
                    continue

                per_source_seen[src] += 1
                ok, reason, cleaned, norm_out = clean_row(
                    row=row,
                    source_path=src,
                    min_input_chars=int(args.min_input_chars),
                    max_input_chars=int(args.max_input_chars),
                    min_output_chars=int(args.min_output_chars),
                    max_output_chars=int(args.max_output_chars),
                    min_hangul_chars=int(args.min_hangul_chars),
                    min_hangul_ratio=float(args.min_hangul_ratio),
                    banned_output_substrings=banned_output_substrings,
                )
                if not ok or cleaned is None:
                    dropped_reasons[reason] += 1
                    per_source_dropped[src][reason] += 1
                    continue

                dedupe = stable_dedupe_key(cleaned["input"], cleaned["output"])
                if dedupe in dedupe_keys:
                    dropped_reasons["duplicate_pair"] += 1
                    per_source_dropped[src]["duplicate_pair"] += 1
                    continue

                repeat_cap = (
                    int(args.max_repeat_output_short)
                    if len(norm_out) <= int(args.short_output_chars)
                    else int(args.max_repeat_output)
                )
                if output_counter[norm_out] >= repeat_cap:
                    dropped_reasons["repeat_output_cap"] += 1
                    per_source_dropped[src]["repeat_output_cap"] += 1
                    continue

                dedupe_keys.add(dedupe)
                output_counter[norm_out] += 1
                rows.append(cleaned)

        random.shuffle(rows)
        cap = int(source_caps.get(src, 0))
        if cap > 0:
            rows = rows[:cap]
        per_source_rows[src] = rows
        per_source_kept[src] = len(rows)

    all_rows: List[Dict] = []
    for src in sources:
        all_rows.extend(per_source_rows.get(src, []))
    random.shuffle(all_rows)

    if int(args.max_total_rows) > 0:
        all_rows = all_rows[: int(args.max_total_rows)]

    with out_path.open("w", encoding="utf-8") as fw:
        for row in all_rows:
            line = json.dumps(row, ensure_ascii=False)
            fw.write(line + "\n")
            output_digest.update(line.encode("utf-8", errors="ignore"))
            output_digest.update(b"\n")

    with sources_tsv_path.open("w", encoding="utf-8") as fs:
        fs.write("source\tscanned\tkept\tcap\n")
        for src in sources:
            fs.write(
                f"{src}\t{int(per_source_seen.get(src, 0))}\t{int(per_source_kept.get(src, 0))}\t{int(source_caps.get(src, 0))}\n"
            )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "rows_written": int(len(all_rows)),
        "sources": sources,
        "source_caps": {k: int(v) for k, v in source_caps.items()},
        "per_source_scanned": {k: int(v) for k, v in per_source_seen.items()},
        "per_source_kept": {k: int(v) for k, v in per_source_kept.items()},
        "drop_reasons": {k: int(v) for k, v in dropped_reasons.most_common()},
        "top_source_drop_reasons": {
            src: {k: int(v) for k, v in cnt.most_common(12)}
            for src, cnt in per_source_dropped.items()
        },
        "filters": {
            "banned_output_substrings": banned_output_substrings,
            "intro_input_regex": INTRO_INPUT_RE.pattern,
            "max_total_rows": int(args.max_total_rows),
            "min_input_chars": int(args.min_input_chars),
            "max_input_chars": int(args.max_input_chars),
            "min_output_chars": int(args.min_output_chars),
            "max_output_chars": int(args.max_output_chars),
            "min_hangul_chars": int(args.min_hangul_chars),
            "min_hangul_ratio": float(args.min_hangul_ratio),
            "max_repeat_output": int(args.max_repeat_output),
            "max_repeat_output_short": int(args.max_repeat_output_short),
            "short_output_chars": int(args.short_output_chars),
        },
        "output_digest_blake2b16": output_digest.hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_jsonl": out_path.as_posix(),
                "rows": int(len(all_rows)),
                "manifest": manifest_path.as_posix(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
