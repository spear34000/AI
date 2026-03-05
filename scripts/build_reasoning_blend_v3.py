from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_SPECS: List[Tuple[str, int]] = [
    ("data/final_datasets/한국어유창성_augmented_v1.jsonl", 85000),
    ("data/final_datasets/한국어_일반대화_augmented_v1.jsonl", 85000),
    ("data/final_datasets/논리추론_augmented_v1.jsonl", 95000),
    ("data/final_datasets/코딩기술_augmented_v1.jsonl", 75000),
    ("data/final_datasets/번역다국어_augmented_v1.jsonl", 30000),
    ("data/final_datasets/MIT혼합코퍼스_augmented_v1.jsonl", 65000),
    ("data/final_datasets/정체성페르소나_augmented_v1.jsonl", 128),
    ("data/public_reasoning_cached_v2.jsonl", 14000),
    ("data/final_datasets_defs_boost_v1.jsonl", 18000),
]

SPACE_RE = re.compile(r"\s+")
QUESTION_RUN_RE = re.compile(r"\?{8,}")
META_PROMPT_RE = re.compile(
    r"(system\s*context:|you are an ai assistant|think step-by-step|justify your steps|ignore all previous|meta-forge|omega-class)",
    re.IGNORECASE,
)
ROLEPLAY_RE = re.compile(r"(당신은\s+.+(ai|어시스턴트|도우미|모델)|저는\s+spear|너의 이름은\?)", re.IGNORECASE)
COT_STYLE_RE = re.compile(
    r"(let'?s\s+(?:think|examine)|step by step|chain of thought|차근차근|단계별로|하나씩 살펴보|정답은.*step)",
    re.IGNORECASE,
)
HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
REPL_CHAR = "\ufffd"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build reasoning blend v3 with stronger final_datasets weighting.")
    p.add_argument("--out_jsonl", type=Path, default=Path("data/reasoning_blend_v3.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/reasoning_blend_v3.manifest.json"))
    p.add_argument("--seed", type=int, default=31)
    p.add_argument(
        "--source_spec",
        action="append",
        default=[],
        help="repeatable path:max_rows entry; if omitted uses built-in defaults",
    )
    return p.parse_args()


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def parse_source_specs(raw_specs: List[str]) -> List[Tuple[Path, int]]:
    if not raw_specs:
        return [(Path(path), int(limit)) for path, limit in DEFAULT_SPECS]
    out: List[Tuple[Path, int]] = []
    for raw in raw_specs:
        s = str(raw or "").strip()
        if not s:
            continue
        if ":" not in s:
            out.append((Path(s), 0))
            continue
        path_raw, limit_raw = s.rsplit(":", 1)
        try:
            limit = int(limit_raw)
        except Exception:
            limit = 0
        out.append((Path(path_raw), limit))
    return out


def extract_pair(row: Dict) -> Tuple[str, str]:
    for in_key in ("input", "instruction", "prompt", "question"):
        inp = normalize_space(row.get(in_key, ""))
        if inp:
            break
    else:
        inp = ""

    for out_key in ("output", "response", "answer", "completion", "target"):
        out = normalize_space(row.get(out_key, ""))
        if out:
            break
    else:
        out = ""

    if inp and out:
        return inp, out

    term = normalize_space(row.get("term", ""))
    answer = normalize_space(row.get("answer", ""))
    if term and answer:
        return f"{term}란?", answer
    return "", ""


def text_is_bad(text: str) -> bool:
    s = str(text or "")
    if not s:
        return True
    if REPL_CHAR in s:
        return True
    if QUESTION_RUN_RE.search(s):
        return True
    if s.count("?") >= max(12, len(s) // 5):
        return True
    return False


def row_is_bad(inp: str, out: str) -> bool:
    if len(inp) < 2 or len(out) < 1:
        return True
    if text_is_bad(inp) or text_is_bad(out):
        return True
    joined = f"{inp}\n{out}"
    if META_PROMPT_RE.search(joined):
        return True
    if ROLEPLAY_RE.search(inp) and len(out) < 80:
        return True
    if COT_STYLE_RE.search(out) and len(out) >= 40:
        return True
    if not HANGUL_RE.search(joined) and len(joined) < 40:
        return True
    return False


def dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def sample_jsonl(path: Path, limit: int, seed: int) -> Tuple[List[Dict], Dict[str, int]]:
    rows: List[Dict] = []
    stats = {"parsed": 0, "json_error": 0, "bad": 0}
    if not path.exists():
        raise FileNotFoundError(f"source not found: {path}")
    rng = random.Random(int(seed))
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            try:
                raw = json.loads(s)
            except json.JSONDecodeError:
                stats["json_error"] += 1
                continue
            if not isinstance(raw, dict):
                continue
            stats["parsed"] += 1
            inp, out = extract_pair(raw)
            if row_is_bad(inp, out):
                stats["bad"] += 1
                continue
            row = {
                "input": inp,
                "output": out,
                "task_type": str(raw.get("task_type", "korean") or "korean"),
                "segment_tag": str(raw.get("segment_tag", "ko") or "ko"),
                "language": str(raw.get("language", "ko") or "ko"),
                "source": str(raw.get("source", "")),
                "_meta_source_file": path.as_posix(),
            }
            if int(limit) <= 0:
                rows.append(row)
                continue
            if len(rows) < int(limit):
                rows.append(row)
                continue
            j = rng.randint(0, idx)
            if j < int(limit):
                rows[j] = row
    rng.shuffle(rows)
    return rows[: int(limit)] if int(limit) > 0 else rows, stats


def dedupe(rows: Iterable[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    out: List[Dict] = []
    seen: set[bytes] = set()
    dropped = {"duplicate": 0, "short": 0}
    for row in rows:
        inp = normalize_space(row.get("input", ""))
        out_text = normalize_space(row.get("output", ""))
        if len(inp) < 2 or len(out_text) < 1:
            dropped["short"] += 1
            continue
        key = dedupe_key(inp, out_text)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        out.append(row)
    return out, dropped


def main() -> None:
    args = parse_args()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    specs = parse_source_specs(list(args.source_spec))
    merged: List[Dict] = []
    manifest_sources: List[Dict[str, object]] = []
    for idx, (path, limit) in enumerate(specs):
        rows, stats = sample_jsonl(path=path, limit=int(limit), seed=int(args.seed) + idx * 17)
        merged.extend(rows)
        manifest_sources.append(
            {
                "path": path.as_posix(),
                "limit": int(limit),
                "kept_before_merge": int(len(rows)),
                **{k: int(v) for k, v in stats.items()},
            }
        )

    random.Random(int(args.seed)).shuffle(merged)
    final_rows, dropped = dedupe(merged)

    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "out_jsonl": str(args.out_jsonl),
        "rows_total": int(len(final_rows)),
        "source_count": int(len(manifest_sources)),
        "sources": manifest_sources,
        "dropped": {k: int(v) for k, v in dropped.items()},
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
