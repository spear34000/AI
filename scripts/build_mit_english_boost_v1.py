from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def normalize_license(value: str) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def is_mit_license(value: str) -> bool:
    lic = normalize_license(value)
    return lic == "mit" or lic == "mit license"


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


def is_code_row(row: Dict) -> bool:
    task = str(row.get("task_type", "")).strip().lower()
    seg = str(row.get("segment_tag", "")).strip().lower()
    if task == "code" or seg == "code":
        return True
    text = f"{str(row.get('input', ''))}\n{str(row.get('output', ''))}"
    return bool(re.search(r"\b(def|class|import|return|const|let|function|SELECT|FROM|try:|except)\b", text))


def extract_code_snippet(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    block = re.search(r"```[a-zA-Z0-9_+-]*\n([\s\S]{8,1800}?)```", src)
    if block:
        src = block.group(1).strip()
    lines = [ln.rstrip() for ln in src.splitlines() if ln.strip()]
    if not lines:
        return ""
    lines = lines[:28]
    out = "\n".join(lines).strip()
    if len(out) > 1200:
        out = out[:1200].rstrip()
    return out


def detect_language(snippet: str) -> Tuple[str, List[str]]:
    s = str(snippet or "")
    sl = s.lower()
    checks = [
        ("Python", [r"\bdef\s+\w+\(", r"\bimport\s+\w+", r"if __name__ == ['\"]__main__['\"]", r"\bexcept\b"]),
        ("JavaScript", [r"\bfunction\s+\w+\(", r"\bconsole\.log\(", r"\bconst\s+\w+", r"\blet\s+\w+"]),
        ("TypeScript", [r"\binterface\s+\w+", r"\btype\s+\w+\s*=", r":\s*[A-Z][A-Za-z0-9_<>,\[\]]*", r"\breadonly\b"]),
        ("Java", [r"\bpublic\s+class\s+\w+", r"\bpublic\s+static\s+void\s+main", r"\bSystem\.out\.println\(", r"\bimport\s+java\."]),
        ("C#", [r"\bnamespace\s+\w+", r"\busing\s+System", r"\bConsole\.WriteLine\(", r"\bpublic\s+class\s+\w+"]),
        ("C++", [r"#include\s*<", r"\bstd::", r"\bint\s+main\s*\(", r"\bcout\s*<<"]),
        ("Go", [r"\bpackage\s+main\b", r"\bfunc\s+\w+\(", r"\bfmt\.", r"\bdefer\b"]),
        ("Rust", [r"\bfn\s+main\s*\(", r"\blet\s+mut\s+", r"\bprintln!\(", r"\bimpl\s+\w+"]),
        ("SQL", [r"\bselect\b", r"\bfrom\b", r"\bwhere\b", r"\bjoin\b"]),
        ("HTML", [r"<!doctype html>", r"<html", r"<div", r"<script"]),
        ("CSS", [r"{\s*[a-z-]+\s*:", r"\.[A-Za-z0-9_-]+\s*{", r"#[A-Za-z0-9_-]+\s*{", r"@media"]),
        ("PHP", [r"<\?php", r"\becho\s+", r"\$\w+", r"->"]),
        ("Ruby", [r"\bdef\s+\w+", r"\bputs\s+", r"\bend\b", r"\bclass\s+\w+"]),
        ("Shell", [r"#!/bin/(ba)?sh", r"\becho\s+", r"\bexport\s+\w+=", r"\bgrep\b"]),
    ]
    for lang, pats in checks:
        hits = []
        for pat in pats:
            m = re.search(pat, sl, flags=re.IGNORECASE)
            if m:
                hits.append(m.group(0)[:30])
        if len(hits) >= 2:
            return lang, hits[:2]
    # Fallbacks
    if re.search(r"[{};]", s) and "function" in sl:
        return "JavaScript", ["function", "{ ... }"]
    if re.search(r"[{};]", s) and "#include" in sl:
        return "C++", ["#include", "{ ... }"]
    return "Unknown", ["syntax markers", "identifier patterns"]


def infer_action(snippet: str) -> str:
    s = str(snippet or "").lower()
    mapping = [
        (["http", "request", "response", "fetch", "axios"], "API request handling"),
        (["json", "parse", "serialize", "dump", "loads"], "data parsing/serialization"),
        (["open(", "read(", "write(", "file", "path"], "file I/O operations"),
        (["select ", "insert ", "update ", "delete ", "from "], "database query/update logic"),
        (["test", "assert", "expect(", "unittest", "pytest"], "automated test logic"),
        (["try", "except", "catch", "finally"], "error handling flow"),
        (["async", "await", "promise", "then("], "asynchronous workflow"),
        (["react", "component", "render(", "<div", "useState"], "UI rendering/state logic"),
        (["sort", "filter", "map(", "reduce("], "collection transformation"),
    ]
    for keys, label in mapping:
        if any(k in s for k in keys):
            return label
    return "general utility logic"


def infer_risk(action: str) -> str:
    risks = {
        "API request handling": "timeouts or retry strategy may be missing",
        "data parsing/serialization": "invalid or unexpected input schema may break parsing",
        "file I/O operations": "missing file checks can cause runtime failures",
        "database query/update logic": "unsanitized input can create injection risks",
        "automated test logic": "assertions may not cover edge conditions",
        "error handling flow": "exceptions may be swallowed without actionable logs",
        "asynchronous workflow": "race conditions can appear under concurrency",
        "UI rendering/state logic": "state updates may trigger unintended re-renders",
        "collection transformation": "empty or null inputs may be unhandled",
        "general utility logic": "edge cases and input validation may be insufficient",
    }
    return risks.get(action, "edge cases and input validation may be insufficient")


def make_rows(snippet: str, lang: str, cues: List[str], action: str, source_dataset: str) -> List[Dict]:
    cue1 = cues[0] if len(cues) >= 1 else "token patterns"
    cue2 = cues[1] if len(cues) >= 2 else "syntax structure"
    risk = infer_risk(action)
    sig = hashlib.sha1((snippet + "\n" + source_dataset).encode("utf-8")).hexdigest()[:16]

    row1 = {
        "id": f"mit_en_boost_lang_{sig}",
        "task_type": "english",
        "segment_tag": "english",
        "language": "en",
        "_meta_quality_tier": "high",
        "license": "MIT",
        "source_dataset": f"{source_dataset}::mit_english_boost_v1",
        "source": "mit_english_boost_v1",
        "input": (
            "Identify the most likely programming language of the snippet and explain your reasoning in two short points.\n\n"
            f"Snippet:\n```text\n{snippet}\n```"
        ),
        "output": f"The most likely language is {lang}. Evidence: 1) {cue1}. 2) {cue2}.",
    }
    row2 = {
        "id": f"mit_en_boost_desc_{sig}",
        "task_type": "english",
        "segment_tag": "english",
        "language": "en",
        "_meta_quality_tier": "mid",
        "license": "MIT",
        "source_dataset": f"{source_dataset}::mit_english_boost_v1",
        "source": "mit_english_boost_v1",
        "input": (
            "Give a concise English description of what this code is mainly doing. Also mention one potential risk or edge case.\n\n"
            f"Snippet:\n```text\n{snippet}\n```"
        ),
        "output": f"This code mainly handles {action}. One potential risk is that {risk}.",
    }
    if str(lang).strip().lower() == "unknown":
        return [row2]
    return [row1, row2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MIT-only English boost rows from code datasets.")
    parser.add_argument("--source_jsonl", default="data/slm_mit_unified_v4.jsonl")
    parser.add_argument("--out_jsonl", default="data/mit_english_boost_v1.jsonl")
    parser.add_argument("--manifest", default="data/mit_english_boost_v1.manifest.json")
    parser.add_argument("--base_limit", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_path = Path(args.source_jsonl)
    if not src_path.exists():
        raise RuntimeError(f"source_jsonl not found: {src_path}")

    rng = random.Random(int(args.seed))
    base_limit = int(args.base_limit)

    reservoir: List[Dict] = []
    seen_base = 0
    counts = Counter()

    for row in iter_jsonl(src_path):
        counts["read_rows"] += 1
        if not is_mit_license(str(row.get("license", ""))):
            counts["non_mit"] += 1
            continue
        if not is_code_row(row):
            counts["non_code"] += 1
            continue

        text_out = str(row.get("output", ""))
        text_in = str(row.get("input", ""))
        snippet = extract_code_snippet(text_out) or extract_code_snippet(text_in)
        if len(snippet) < 24:
            counts["short_snippet"] += 1
            continue

        source_dataset = str(row.get("source_dataset", row.get("source", "unknown"))).strip() or "unknown"
        lang, cues = detect_language(snippet)
        action = infer_action(snippet)
        item = {
            "snippet": snippet,
            "lang": lang,
            "cues": cues,
            "action": action,
            "source_dataset": source_dataset,
        }
        seen_base += 1
        if len(reservoir) < base_limit:
            reservoir.append(item)
        else:
            j = rng.randint(0, seen_base - 1)
            if j < base_limit:
                reservoir[j] = item

    out_rows: List[Dict] = []
    dedupe = set()
    lang_counts = Counter()
    for item in reservoir:
        rows = make_rows(
            snippet=str(item["snippet"]),
            lang=str(item["lang"]),
            cues=[str(x) for x in item.get("cues", [])],
            action=str(item["action"]),
            source_dataset=str(item["source_dataset"]),
        )
        for r in rows:
            key = hashlib.sha1((r["input"] + "\n" + r["output"]).encode("utf-8")).hexdigest()
            if key in dedupe:
                continue
            dedupe.add(key)
            out_rows.append(r)
            lang_counts[str(item["lang"])] += 1

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "source_jsonl": str(src_path),
        "output_jsonl": str(out_path),
        "rows_total": len(out_rows),
        "base_rows_sampled": len(reservoir),
        "language_guess_counts": dict(lang_counts),
        "counters": dict(counts),
        "seed": int(args.seed),
        "base_limit": base_limit,
        "license_policy": "MIT only",
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(out_path),
                "rows_total": len(out_rows),
                "base_rows_sampled": len(reservoir),
                "language_guess_top10": dict(lang_counts.most_common(10)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
