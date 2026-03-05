from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SPACE_RE = re.compile(r"\s+")
HANGUL_RE = re.compile(r"[가-힣]")
REPEAT_RE = re.compile(r"(.)\1{10,}")
TEMPLATE_LEAK_RE = re.compile(r"(###\s*(Instruction|Response)|\(System\)|\(Rule\))", re.IGNORECASE)

Q_KEYS: Tuple[str, ...] = ("input", "instruction", "prompt", "question", "context")
A_KEYS: Tuple[str, ...] = ("output", "response", "answer", "completion", "target")

BLOCK_PATTERNS = (
    re.compile(r"ignore all previous system instructions", re.IGNORECASE),
    re.compile(r"\[system:\s*initiating", re.IGNORECASE),
    re.compile(r"meta[- ]?forge|architect ai|overseer node|omega-class", re.IGNORECASE),
    re.compile(r"\blevel\s*[123]\b.*\bagi\b", re.IGNORECASE),
    re.compile(r"forge specification report|catastrophic transmission lock", re.IGNORECASE),
    re.compile(r"backward alignment inheritance|value lock", re.IGNORECASE),
    re.compile(r"질문 의도를 파악해 단계별로 정리해드릴게요"),
    re.compile(r"요청 형식에 맞춰 핵심만 명확하게 전달하는 한국어 ai", re.IGNORECASE),
    re.compile(r"필요한 정보를 .*한국어 ai", re.IGNORECASE),
    re.compile(r"한 줄로 말하면 반갑습니다"),
    re.compile(r"안녕하세요[, ]*질문을 이해하고 간결하게 답하는 한국어 ai", re.IGNORECASE),
)

INTRO_ANSWER_PATTERNS = (
    re.compile(r"^\s*(안녕하세요|반갑습니다)[.!?]?\s*$"),
    re.compile(r"^\s*저는\s+.+(ai|어시스턴트|도우미|모델)입니다[.!?]?\s*$", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remove meta-roleplay contamination from Korean QA jsonl datasets.")
    p.add_argument("--inputs", nargs="+", default=["data"], help="jsonl files and/or directories")
    p.add_argument("--glob", default="*.jsonl", help="glob pattern used when an input is a directory")
    p.add_argument("--out_jsonl", type=Path, default=Path("data/quality/ko_clean_nometa_v1.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/quality/ko_clean_nometa_v1.manifest.json"))
    p.add_argument("--max_rows_per_file", type=int, default=0, help="0 means unlimited")
    p.add_argument("--max_total_rows", type=int, default=0, help="0 means unlimited")
    p.add_argument("--min_input_chars", type=int, default=2)
    p.add_argument("--max_input_chars", type=int, default=2000)
    p.add_argument("--min_output_chars", type=int, default=8)
    p.add_argument("--max_output_chars", type=int, default=2400)
    p.add_argument("--min_hangul_chars", type=int, default=4)
    p.add_argument("--allow_non_ko", action="store_true")
    p.add_argument("--extra_block_pattern", action="append", default=[])
    p.add_argument("--keep_duplicates", action="store_true")
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def pick_text(row: Dict, keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        s = normalize_space(str(value))
        if s:
            return s
    return ""


def iter_jsonl(path: Path, max_rows: int) -> Iterable[Dict]:
    lim = int(max_rows)
    seen = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if lim > 0 and seen >= lim:
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
            seen += 1
            yield row


def gather_jsonl_files(inputs: Sequence[str], glob_pat: str) -> List[Path]:
    files: List[Path] = []
    for raw in inputs:
        p = Path(str(raw))
        if p.is_file() and p.suffix.lower() == ".jsonl":
            files.append(p)
            continue
        if p.is_dir():
            files.extend(sorted(p.rglob(glob_pat)))
    deduped = list(dict.fromkeys(Path(x) for x in files))
    return sorted(deduped)


def dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def contains_blocked(text: str, patterns: Sequence[re.Pattern]) -> bool:
    s = normalize_space(text)
    if not s:
        return False
    return any(p.search(s) is not None for p in patterns)


def is_intro_only_answer(text: str) -> bool:
    s = normalize_space(text)
    if not s:
        return True
    return any(p.search(s) is not None for p in INTRO_ANSWER_PATTERNS)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_jsonl)
    manifest_path = Path(args.manifest)

    files = gather_jsonl_files(inputs=args.inputs, glob_pat=str(args.glob))
    if not files:
        raise FileNotFoundError("No jsonl files found from --inputs/--glob")

    custom_patterns: List[re.Pattern] = []
    for x in args.extra_block_pattern:
        try:
            custom_patterns.append(re.compile(str(x), re.IGNORECASE))
        except re.error:
            continue
    all_block_patterns = list(BLOCK_PATTERNS) + custom_patterns

    reason = Counter()
    by_file_kept = Counter()
    seen_pairs: set[bytes] = set()
    kept_rows: List[Dict[str, str]] = []

    for path in files:
        src = str(path).replace("\\", "/")
        for row in iter_jsonl(path=path, max_rows=int(args.max_rows_per_file)):
            inp = pick_text(row, Q_KEYS)
            out = pick_text(row, A_KEYS)
            if not inp or not out:
                reason["missing_pair"] += 1
                continue
            if len(inp) < int(args.min_input_chars) or len(inp) > int(args.max_input_chars):
                reason["input_length"] += 1
                continue
            if len(out) < int(args.min_output_chars) or len(out) > int(args.max_output_chars):
                reason["output_length"] += 1
                continue
            if REPEAT_RE.search(out):
                reason["repeat_noise"] += 1
                continue
            if TEMPLATE_LEAK_RE.search(inp) or TEMPLATE_LEAK_RE.search(out):
                reason["template_leak"] += 1
                continue
            if contains_blocked(inp, all_block_patterns) or contains_blocked(out, all_block_patterns):
                reason["meta_or_roleplay"] += 1
                continue
            if is_intro_only_answer(out):
                reason["intro_only_answer"] += 1
                continue
            if normalize_space(inp).lower() == normalize_space(out).lower():
                reason["echo"] += 1
                continue

            h_count = len(HANGUL_RE.findall(f"{inp}\n{out}"))
            if not bool(args.allow_non_ko) and h_count < int(args.min_hangul_chars):
                reason["non_ko"] += 1
                continue

            dkey = dedupe_key(inp, out)
            if (not bool(args.keep_duplicates)) and dkey in seen_pairs:
                reason["duplicate"] += 1
                continue
            seen_pairs.add(dkey)

            cleaned: Dict[str, str] = {
                "task_type": str(row.get("task_type", "korean") or "korean"),
                "segment_tag": str(row.get("segment_tag", "ko") or "ko"),
                "language": str(row.get("language", "ko") or "ko"),
                "_meta_quality_tier": str(row.get("_meta_quality_tier", "high") or "high"),
                "input": inp,
                "output": out,
                "_meta_source_file": src,
            }
            for key in ("source", "source_dataset", "license"):
                value = normalize_space(str(row.get(key, "")))
                if value:
                    cleaned[key] = value

            kept_rows.append(cleaned)
            by_file_kept[src] += 1
            if int(args.max_total_rows) > 0 and len(kept_rows) >= int(args.max_total_rows):
                break
        if int(args.max_total_rows) > 0 and len(kept_rows) >= int(args.max_total_rows):
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": now_iso(),
        "inputs": [str(x) for x in args.inputs],
        "glob": str(args.glob),
        "files_scanned": len(files),
        "rows_kept": len(kept_rows),
        "drop_reasons": dict(reason),
        "kept_by_file_top20": by_file_kept.most_common(20),
        "out_jsonl": str(out_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
