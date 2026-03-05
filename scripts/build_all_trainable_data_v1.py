from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


INPUT_KEYS = ("input", "instruction", "prompt", "question", "context")
OUTPUT_KEYS = ("output", "response", "answer", "completion", "target")
QUESTION_RUN_RE = re.compile(r"\?{3,}")
HANGUL_SYLLABLE_RE = re.compile(r"[\uac00-\ud7a3]")
HANGUL_JAMO_RE = re.compile(r"[\u3131-\u314e\u314f-\u3163]")
CJK_IDEO_RE = re.compile(r"[\u4e00-\u9fff]")
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build one clean trainable JSONL from everything usable under data/.")
    p.add_argument("--data_root", type=Path, default=Path("data"))
    p.add_argument("--out_jsonl", type=Path, default=Path("data/all_trainable_clean_v1.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/all_trainable_clean_v1.manifest.json"))
    p.add_argument("--max_rows_per_file", type=int, default=0, help="0 means unlimited")
    p.add_argument("--min_input_chars", type=int, default=1)
    p.add_argument("--min_output_chars", type=int, default=1)
    p.add_argument("--dedupe", action="store_true", default=True)
    p.add_argument("--no_dedupe", dest="dedupe", action="store_false")
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


def infer_language(row: Dict, text: str) -> str:
    for key in ("language", "segment_tag", "task_type"):
        value = str(row.get(key, "")).strip().lower()
        if value in {"ko", "korean"}:
            return "ko"
        if value in {"en", "english"}:
            return "en"
        if value == "code":
            return "code"
    if HANGUL_SYLLABLE_RE.search(text):
        return "ko"
    if "```" in text or "def " in text or "class " in text or "import " in text:
        return "code"
    return "other"


def infer_segment(language: str) -> str:
    if language == "ko":
        return "ko"
    if language == "code":
        return "code"
    if language == "en":
        return "english"
    return "mixed"


def text_is_corrupt(text: str, language: str) -> bool:
    s = str(text or "")
    if not s:
        return True
    if "\ufffd" in s:
        return True

    q_count = s.count("?")
    if QUESTION_RUN_RE.search(s) and q_count >= max(8, len(s) // 6):
        return True
    if q_count >= 12 and q_count >= len(s) // 5:
        return True

    syll = len(HANGUL_SYLLABLE_RE.findall(s))
    jamo = len(HANGUL_JAMO_RE.findall(s))
    cjk = len(CJK_IDEO_RE.findall(s))
    non_ascii = sum(ord(ch) > 127 for ch in s)

    if language == "ko":
        if syll == 0 and non_ascii >= 10 and (jamo + cjk) >= 8:
            return True
        if cjk >= 6 and cjk > max(2, syll):
            return True
        if jamo >= 8 and jamo > max(4, syll // 2):
            return True

    return False


def dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def extract_pair(row: Dict) -> Tuple[str, str]:
    inp = pick_first_non_empty(row, INPUT_KEYS)
    out = pick_first_non_empty(row, OUTPUT_KEYS)
    if inp and out:
        return inp, out

    term = normalize_space(str(row.get("term", "")))
    answer = normalize_space(str(row.get("answer", "")))
    if term and answer:
        return f"{term}\ub780?", answer
    return "", ""


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                yield {"__json_error__": True}
                continue
            if isinstance(row, dict):
                yield row


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_path = Path(args.out_jsonl)
    manifest_path = Path(args.manifest)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(p for p in data_root.rglob("*.jsonl") if p.resolve() != out_path.resolve())
    seen = set()
    global_counts: Counter[str] = Counter()
    file_rows: List[Dict[str, object]] = []
    delete_candidates: List[str] = []

    with out_path.open("w", encoding="utf-8") as out_f:
        for path in all_files:
            rel = path.as_posix()
            counts: Counter[str] = Counter()
            counts["size_bytes"] = int(path.stat().st_size)

            if counts["size_bytes"] == 0:
                delete_candidates.append(rel)
                file_rows.append({"path": rel, **dict(counts), "status": "empty"})
                global_counts["empty_files"] += 1
                continue

            for row in iter_jsonl(path):
                if bool(row.get("__json_error__")):
                    counts["json_error"] += 1
                    continue

                counts["parsed"] += 1
                inp, out = extract_pair(row)
                if not inp or not out:
                    counts["no_pair"] += 1
                    continue

                if len(inp) < int(args.min_input_chars):
                    counts["short_input"] += 1
                    continue
                if len(out) < int(args.min_output_chars):
                    counts["short_output"] += 1
                    continue

                text = f"{inp}\n{out}"
                language = infer_language(row=row, text=text)
                if text_is_corrupt(inp, language=language) or text_is_corrupt(out, language=language):
                    counts["corrupt"] += 1
                    continue

                if args.dedupe:
                    key = dedupe_key(inp, out)
                    if key in seen:
                        counts["duplicate"] += 1
                        continue
                    seen.add(key)

                cleaned = {
                    "task_type": str(row.get("task_type", infer_segment(language))),
                    "segment_tag": str(row.get("segment_tag", infer_segment(language))),
                    "language": str(row.get("language", language)),
                    "input": inp,
                    "output": out,
                    "_meta_source_file": rel,
                }
                for key in (
                    "_meta_quality_tier",
                    "source",
                    "source_dataset",
                    "license",
                    "id",
                    "_augment_type",
                ):
                    value = row.get(key)
                    if value is None:
                        continue
                    text_value = normalize_space(str(value))
                    if text_value:
                        cleaned[key] = text_value

                out_f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                counts["kept"] += 1

            status = "kept"
            if counts["kept"] == 0:
                if counts["corrupt"] > 0 or counts["parsed"] > 0:
                    status = "excluded"
                    delete_candidates.append(rel)
                else:
                    status = "empty"
                    delete_candidates.append(rel)

            file_rows.append({"path": rel, **dict(counts), "status": status})
            global_counts.update(counts)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "out_jsonl": str(out_path),
        "scanned_jsonl_files": int(len(all_files)),
        "dedupe_enabled": bool(args.dedupe),
        "summary": {
            "parsed_rows": int(global_counts["parsed"]),
            "kept_rows": int(global_counts["kept"]),
            "corrupt_rows": int(global_counts["corrupt"]),
            "duplicate_rows": int(global_counts["duplicate"]),
            "no_pair_rows": int(global_counts["no_pair"]),
            "json_error_rows": int(global_counts["json_error"]),
            "empty_files": int(global_counts["empty_files"]),
        },
        "delete_candidates": sorted(dict.fromkeys(delete_candidates)),
        "files": file_rows,
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_jsonl": str(out_path), "manifest": str(manifest_path), **manifest["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
