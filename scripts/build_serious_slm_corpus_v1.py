from __future__ import annotations

import argparse
import hashlib
import json
import random
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

DEFAULT_SOURCES: List[Tuple[str, int]] = [
    ("data/hf_mit_language_superpack_hq_v2.jsonl", 150000),
    ("data/ko_agent_tool_dataset_v1.jsonl", 111436),
    ("data/mit_ko_quality_v1.jsonl", 90000),
    ("data/hq_dehard_mix_v1.jsonl", 60000),
    ("data/by_field_ko_pure_v1.jsonl", 34000),
    ("data/slm_mit_all_local_v1.jsonl", 30000),
    ("data/mit_english_boost_v1.jsonl", 30000),
    ("data/ko_targeted_shortanswer_v3.jsonl", 28000),
    ("data/dehardcode_patch_v2.jsonl", 24000),
    ("data/clean_mix_v3.jsonl", 20000),
    ("data/deintro_focus_v1.jsonl", 20000),
    ("data/term_focus_clean_v1.jsonl", 16000),
    ("data/term_anchor_patch_v2.jsonl", 14000),
    ("data/prompt_focus_v1.jsonl", 18000),
    ("data/ko_def_grounding_patch_v1.jsonl", 15000),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a serious curated SLM corpus from selected local datasets.")
    p.add_argument("--out_jsonl", type=Path, default=Path("data/serious_slm_corpus_v1.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/serious_slm_corpus_v1.manifest.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--max_total_rows", type=int, default=0, help="0 means unlimited")
    p.add_argument(
        "--source_spec",
        action="append",
        default=[],
        help="repeatable path:max_rows entries; if empty uses built-in curated defaults",
    )
    return p.parse_args()


def parse_source_specs(raw_specs: List[str]) -> List[Tuple[Path, int]]:
    specs: List[Tuple[Path, int]] = []
    for raw in raw_specs:
        s = str(raw or "").strip()
        if not s:
            continue
        if ":" not in s:
            specs.append((Path(s), 0))
            continue
        path_raw, limit_raw = s.rsplit(":", 1)
        try:
            limit = int(limit_raw)
        except Exception:
            limit = 0
        specs.append((Path(path_raw), limit))
    return specs


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
                continue
            if isinstance(row, dict):
                yield row


def source_priority(path: Path) -> int:
    name = path.name
    for idx, (src, _) in enumerate(DEFAULT_SOURCES):
        if Path(src).name == name:
            return idx
    return 10_000


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))

    specs = parse_source_specs(list(args.source_spec))
    if not specs:
        specs = [(Path(p), n) for p, n in DEFAULT_SOURCES]

    out_path = Path(args.out_jsonl)
    manifest_path = Path(args.manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: List[Dict] = []
    seen = set()
    summary = Counter()
    per_source: List[Dict[str, object]] = []

    ordered_specs = sorted(specs, key=lambda x: (source_priority(x[0]), x[0].as_posix()))
    for path, limit in ordered_specs:
        if not path.exists():
            raise FileNotFoundError(f"source not found: {path}")
        counts = Counter()
        local_rows: List[Dict] = []
        for row in iter_jsonl(path):
            counts["parsed"] += 1
            inp, out = extract_pair(row)
            if not inp or not out:
                counts["no_pair"] += 1
                continue
            text = f"{inp}\n{out}"
            language = infer_language(row=row, text=text)
            if text_is_corrupt(inp, language) or text_is_corrupt(out, language):
                counts["corrupt"] += 1
                continue
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
                "_meta_source_file": path.as_posix(),
            }
            for key_name in (
                "_meta_quality_tier",
                "source",
                "source_dataset",
                "license",
                "id",
                "_augment_type",
            ):
                value = row.get(key_name)
                if value is None:
                    continue
                txt = normalize_space(str(value))
                if txt:
                    cleaned[key_name] = txt
            local_rows.append(cleaned)

        if int(limit) > 0 and len(local_rows) > int(limit):
            rng.shuffle(local_rows)
            local_rows = local_rows[: int(limit)]

        rows_out.extend(local_rows)
        counts["kept"] = len(local_rows)
        summary.update(counts)
        per_source.append({"path": path.as_posix(), "limit": int(limit), **dict(counts)})

    if bool(args.shuffle):
        rng.shuffle(rows_out)

    if int(args.max_total_rows) > 0 and len(rows_out) > int(args.max_total_rows):
        rng.shuffle(rows_out)
        rows_out = rows_out[: int(args.max_total_rows)]

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lang_counts = Counter(str(r.get("language", "")) for r in rows_out)
    task_counts = Counter(str(r.get("task_type", "")) for r in rows_out)
    source_counts = Counter(str(r.get("_meta_source_file", "")) for r in rows_out)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out_jsonl": str(out_path),
        "rows_total": int(len(rows_out)),
        "summary": {
            "parsed": int(summary["parsed"]),
            "kept": int(len(rows_out)),
            "corrupt": int(summary["corrupt"]),
            "duplicate": int(summary["duplicate"]),
            "no_pair": int(summary["no_pair"]),
        },
        "language_counts": dict(lang_counts),
        "task_counts": dict(task_counts),
        "top_sources": dict(source_counts.most_common(20)),
        "sources": per_source,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_jsonl": str(out_path), "manifest": str(manifest_path), "rows_total": int(len(rows_out))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
