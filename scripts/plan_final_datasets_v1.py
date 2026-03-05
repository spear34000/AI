from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


INPUT_KEYS = ("input", "instruction", "prompt", "question", "context")
OUTPUT_KEYS = ("output", "response", "answer", "completion", "target")
HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
CODE_HINT_RE = re.compile(r"(```|def |class |import |function |SELECT |<html|</|public void |const |let )", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plan source specs for data/final_datasets.")
    p.add_argument("--root", type=Path, default=Path("data/final_datasets"))
    p.add_argument("--sample_rows", type=int, default=4000)
    p.add_argument("--include_persona", action="store_true")
    p.add_argument("--out_json", type=Path, default=None)
    return p.parse_args()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def pick_text(row: Dict, keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = normalize_space(str(value))
        if text:
            return text
    return ""


def infer_lang(row: Dict, text: str) -> str:
    for key in ("language", "segment_tag", "task_type"):
        value = str(row.get(key, "")).strip().lower()
        if value in {"ko", "korean"}:
            return "ko"
        if value in {"en", "english"}:
            return "en"
        if value == "code":
            return "code"
        if len(value) == 2 and value.isalpha():
            return "other_lang"
    if CODE_HINT_RE.search(text):
        return "code"
    if HANGUL_RE.search(text):
        return "ko"
    if re.search(r"[A-Za-z]{4,}", text):
        return "en"
    return "other"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = str(line).strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def classify_file(path: Path, sample_rows: int) -> Dict[str, object]:
    counts = Counter()
    seen_rows = 0
    for row in iter_jsonl(path):
        inp = pick_text(row, INPUT_KEYS)
        out = pick_text(row, OUTPUT_KEYS)
        text = f"{inp}\n{out}"
        lang = infer_lang(row=row, text=text)
        counts[lang] += 1
        seen_rows += 1
        if seen_rows >= int(sample_rows):
            break

    total = max(1, seen_rows)
    ko_ratio = float(counts["ko"]) / float(total)
    en_ratio = float(counts["en"]) / float(total)
    code_ratio = float(counts["code"]) / float(total)
    other_lang_ratio = float(counts["other_lang"]) / float(total)
    size_mb = float(path.stat().st_size) / (1024.0 * 1024.0)

    category = "mixed"
    if size_mb < 1.0 and total <= 500:
        category = "persona_small"
    elif en_ratio >= 0.85 and ko_ratio < 0.05 and other_lang_ratio < 0.05:
        category = "english_mit"
    elif other_lang_ratio >= 0.20:
        category = "multilingual"
    elif code_ratio >= 0.15:
        category = "code"
    elif ko_ratio >= 0.55:
        category = "korean"

    return {
        "path": path.as_posix(),
        "name": path.name,
        "size_mb": round(size_mb, 2),
        "sample_rows": int(seen_rows),
        "lang_counts": dict(counts),
        "ko_ratio": ko_ratio,
        "en_ratio": en_ratio,
        "code_ratio": code_ratio,
        "other_lang_ratio": other_lang_ratio,
        "category": category,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no jsonl files under: {root}")

    rows: List[Dict[str, object]] = [classify_file(path=p, sample_rows=int(args.sample_rows)) for p in files]

    base_specs: List[str] = []
    defs_specs: List[str] = []
    persona_specs: List[str] = []
    for row in rows:
        rel = Path(str(row["path"])).as_posix()
        spec = f"{rel}:0"
        category = str(row["category"])
        if category == "persona_small":
            persona_specs.append(spec)
            if bool(args.include_persona):
                base_specs.append(spec)
                defs_specs.append(spec)
            continue
        base_specs.append(spec)
        if category in {"korean", "code", "mixed"}:
            defs_specs.append(spec)

    out = {
        "root": root.as_posix(),
        "include_persona": bool(args.include_persona),
        "files": rows,
        "base_specs": base_specs,
        "defs_specs": defs_specs,
        "persona_specs": persona_specs,
    }
    payload = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
