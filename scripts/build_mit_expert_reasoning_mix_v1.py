from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_OUT_JSONL = Path("data/slm_mit_expert_reasoning_mix_v1.jsonl")
DEFAULT_MANIFEST = Path("data/slm_mit_expert_reasoning_mix_v1.manifest.json")

DEFAULT_HQ_SUPERPACK = Path("data/hf_mit_language_superpack_hq_v2.jsonl")
DEFAULT_RAW_SUPERPACK = Path("data/hf_mit_language_superpack_v1.jsonl")
DEFAULT_MIT_UNIFIED = Path("data/slm_mit_unified_v4.jsonl")
DEFAULT_MIT_ALL_LOCAL = Path("data/slm_mit_all_local_v1.jsonl")

KEEP_CODE_AUG_TYPES = {"", "continuation_rephrase", "continuation_rephrase_v2"}
KEEP_KO_DOC_AUG_TYPES = {"", "ko_summary_extract", "ko_summary_extract_v2"}

MMMLU_SOURCE_PREFIX = "openai/MMMLU::"
GSM8K_SOURCE_PREFIX = "openai/gsm8k::"

MMMLU_ANSWER_RE = re.compile(r"\b([ABCD])\b")
MMMLU_SUBJECT_RE = re.compile(r"^Question\s*\((.*?)\)\s*:", flags=re.IGNORECASE)
MMMLU_OPTION_RE = re.compile(r"^([ABCD])\)\s*(.+)$")
WS_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MIT-only expert/reasoning/domain mix v1.")
    p.add_argument("--out_jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--hq_superpack", type=Path, default=DEFAULT_HQ_SUPERPACK)
    p.add_argument("--raw_superpack", type=Path, default=DEFAULT_RAW_SUPERPACK)
    p.add_argument("--mit_unified", type=Path, default=DEFAULT_MIT_UNIFIED)
    p.add_argument("--mit_all_local", type=Path, default=DEFAULT_MIT_ALL_LOCAL)

    # Output quotas (defaults tuned for balanced expertise + reasoning).
    p.add_argument("--code_limit", type=int, default=75000)
    p.add_argument("--english_general_limit", type=int, default=60000)
    p.add_argument("--math_reasoning_limit", type=int, default=18000)
    p.add_argument("--multilingual_reasoning_limit", type=int, default=12000)
    p.add_argument("--korean_doc_limit", type=int, default=3000)

    p.add_argument("--upsample_math_reasoning", action="store_true")
    p.add_argument("--upsample_korean_doc", action="store_true")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def normalize_space(text: str) -> str:
    return WS_RE.sub(" ", str(text or "").strip())


def clean_text(text: str) -> str:
    src = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    src = re.sub(r"[ \t]+", " ", src)
    src = re.sub(r"\n{3,}", "\n\n", src)
    return src.strip()


def normalize_license(value: str) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def is_mit_license(value: str) -> bool:
    lic = normalize_license(value)
    return lic in {"mit", "mit license"}


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
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


def validate_basic_row(row: Dict) -> bool:
    if not is_mit_license(str(row.get("license", ""))):
        return False
    inp = clean_text(row.get("input", ""))
    out = clean_text(row.get("output", ""))
    if len(inp) < 6 or len(out) < 6:
        return False
    if len(inp) > 2600 or len(out) > 3600:
        return False
    if re.search(r"(.)\1{14,}", out):
        return False
    return True


def clone_row(row: Dict) -> Dict:
    rec = dict(row)
    rec["input"] = clean_text(rec.get("input", ""))
    rec["output"] = clean_text(rec.get("output", ""))
    rec["license"] = "MIT"
    return rec


def row_key(row: Dict) -> str:
    raw = f"{normalize_space(str(row.get('input', '')).lower())}\n{normalize_space(str(row.get('output', '')).lower())}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def sample_rows(pool: List[Dict], limit: int, rng: random.Random, upsample: bool = False) -> Tuple[List[Dict], int]:
    lim = int(max(0, limit))
    if lim == 0 or not pool:
        return [], 0

    if len(pool) >= lim:
        idxs = list(range(len(pool)))
        rng.shuffle(idxs)
        return [pool[i] for i in idxs[:lim]], 0

    out = list(pool)
    rng.shuffle(out)
    up_n = 0
    if upsample:
        while len(out) < lim:
            row = dict(pool[rng.randrange(len(pool))])
            row["_meta_upsampled"] = True
            out.append(row)
            up_n += 1
    return out[:lim], up_n


def parse_mmmlu_option_map(inp: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ln in str(inp or "").splitlines():
        m = MMMLU_OPTION_RE.match(ln.strip())
        if not m:
            continue
        k = str(m.group(1)).strip().upper()
        v = clean_text(m.group(2))
        if k and v:
            out[k] = v
    return out


def parse_mmmlu_answer_letter(out: str) -> str:
    m = MMMLU_ANSWER_RE.search(str(out or "").upper())
    if not m:
        return ""
    return str(m.group(1)).strip().upper()


def parse_mmmlu_subject(inp: str) -> str:
    m = MMMLU_SUBJECT_RE.search(str(inp or ""))
    if not m:
        return ""
    return clean_text(m.group(1))


def transform_mmmlu_row(row: Dict) -> Dict | None:
    inp = clean_text(row.get("input", ""))
    out = clean_text(row.get("output", ""))
    lang = str(row.get("language", "en")).strip().lower() or "en"
    source_dataset = str(row.get("source_dataset", ""))

    ans = parse_mmmlu_answer_letter(out)
    opts = parse_mmmlu_option_map(inp)
    if ans not in opts:
        return None
    subject = parse_mmmlu_subject(inp)
    chosen = opts[ans]

    # Replace low-entropy label targets with option content.
    if subject:
        new_out = f"[{subject}] {ans}) {chosen}"
    else:
        new_out = f"{ans}) {chosen}"

    rec = clone_row(row)
    rec["output"] = new_out
    rec["task_type"] = "korean" if lang == "ko" else "english"
    rec["segment_tag"] = "ko" if lang == "ko" else "english"
    rec["source_dataset"] = source_dataset
    rec["_meta_quality_tier"] = "high"
    rec["_meta_transform"] = "mmmlu_option_content_v1"
    return rec


def balanced_multilingual_sample(rows: List[Dict], limit: int, rng: random.Random) -> List[Dict]:
    lim = int(max(0, limit))
    if lim == 0 or not rows:
        return []

    by_lang: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_lang[str(r.get("language", "en")).strip().lower() or "en"].append(r)

    langs = sorted(by_lang.keys())
    if not langs:
        return []

    per_lang = max(1, lim // len(langs))
    out: List[Dict] = []
    used_idx: Dict[str, List[int]] = {}
    for lg in langs:
        pool = by_lang[lg]
        idxs = list(range(len(pool)))
        rng.shuffle(idxs)
        used_idx[lg] = idxs
        take = min(per_lang, len(pool))
        out.extend(pool[i] for i in idxs[:take])

    if len(out) < lim:
        remain: List[Dict] = []
        for lg in langs:
            pool = by_lang[lg]
            idxs = used_idx[lg]
            take_start = min(per_lang, len(pool))
            remain.extend(pool[i] for i in idxs[take_start:])
        rng.shuffle(remain)
        out.extend(remain[: max(0, lim - len(out))])

    if len(out) > lim:
        rng.shuffle(out)
        out = out[:lim]
    return out


def collect_code_pool(mit_all_local: Path, mit_unified: Path) -> Tuple[List[Dict], Counter]:
    pool: List[Dict] = []
    stats = Counter()

    for row in iter_jsonl(mit_all_local):
        stats["all_local_read"] += 1
        if not validate_basic_row(row):
            stats["all_local_filtered"] += 1
            continue
        if str(row.get("task_type", "")).strip().lower() != "code":
            stats["all_local_non_code"] += 1
            continue
        rec = clone_row(row)
        rec["task_type"] = "code"
        rec["segment_tag"] = "code"
        pool.append(rec)
        stats["all_local_kept"] += 1

    for row in iter_jsonl(mit_unified):
        stats["unified_read"] += 1
        if not validate_basic_row(row):
            stats["unified_filtered"] += 1
            continue
        if str(row.get("task_type", "")).strip().lower() != "code":
            continue
        aug = str(row.get("_augment_type", ""))
        if aug not in KEEP_CODE_AUG_TYPES:
            stats["unified_drop_augment"] += 1
            continue
        rec = clone_row(row)
        rec["task_type"] = "code"
        rec["segment_tag"] = "code"
        pool.append(rec)
        stats["unified_kept"] += 1

    return pool, stats


def collect_hq_language_pools(hq_superpack: Path) -> Tuple[List[Dict], List[Dict], Counter]:
    english_general: List[Dict] = []
    math_reasoning: List[Dict] = []
    stats = Counter()
    for row in iter_jsonl(hq_superpack):
        stats["hq_read"] += 1
        if not validate_basic_row(row):
            stats["hq_filtered"] += 1
            continue
        src = str(row.get("source_dataset", "")).strip()
        rec = clone_row(row)
        rec["_meta_quality_tier"] = "high"
        if src.startswith(GSM8K_SOURCE_PREFIX):
            math_reasoning.append(rec)
            stats["hq_math_kept"] += 1
        else:
            english_general.append(rec)
            stats["hq_general_kept"] += 1
    return english_general, math_reasoning, stats


def collect_mmmlu_pool(raw_superpack: Path) -> Tuple[List[Dict], Counter]:
    pool: List[Dict] = []
    stats = Counter()
    for row in iter_jsonl(raw_superpack):
        stats["raw_superpack_read"] += 1
        if not validate_basic_row(row):
            continue
        src = str(row.get("source_dataset", "")).strip()
        if not src.startswith(MMMLU_SOURCE_PREFIX):
            continue
        t = transform_mmmlu_row(row)
        if t is None:
            stats["mmmlu_transform_fail"] += 1
            continue
        pool.append(t)
        stats["mmmlu_transform_ok"] += 1
    return pool, stats


def collect_korean_doc_pool(mit_unified: Path) -> Tuple[List[Dict], Counter]:
    pool: List[Dict] = []
    stats = Counter()
    for row in iter_jsonl(mit_unified):
        stats["ko_doc_read"] += 1
        if not validate_basic_row(row):
            continue
        if str(row.get("task_type", "")).strip().lower() != "korean":
            continue
        aug = str(row.get("_augment_type", ""))
        if aug not in KEEP_KO_DOC_AUG_TYPES:
            stats["ko_doc_drop_augment"] += 1
            continue
        rec = clone_row(row)
        rec["task_type"] = "korean"
        rec["segment_tag"] = "ko"
        rec["language"] = "ko"
        pool.append(rec)
        stats["ko_doc_kept"] += 1
    return pool, stats


def dedupe_rows(rows: List[Dict]) -> Tuple[List[Dict], int]:
    seen = set()
    out: List[Dict] = []
    dup = 0
    for r in rows:
        k = row_key(r)
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        out.append(r)
    return out, dup


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))

    for p in [args.hq_superpack, args.raw_superpack, args.mit_unified, args.mit_all_local]:
        if not Path(p).exists():
            raise FileNotFoundError(f"input not found: {p}")

    code_pool, code_stats = collect_code_pool(Path(args.mit_all_local), Path(args.mit_unified))
    eng_pool, math_pool, hq_stats = collect_hq_language_pools(Path(args.hq_superpack))
    mmmlu_pool, mmmlu_stats = collect_mmmlu_pool(Path(args.raw_superpack))
    ko_doc_pool, ko_doc_stats = collect_korean_doc_pool(Path(args.mit_unified))

    code_rows, code_up = sample_rows(code_pool, int(args.code_limit), rng, upsample=False)
    eng_rows, eng_up = sample_rows(eng_pool, int(args.english_general_limit), rng, upsample=False)
    math_rows, math_up = sample_rows(
        math_pool,
        int(args.math_reasoning_limit),
        rng,
        upsample=bool(args.upsample_math_reasoning),
    )
    mmmlu_rows = balanced_multilingual_sample(mmmlu_pool, int(args.multilingual_reasoning_limit), rng)
    ko_doc_rows, ko_doc_up = sample_rows(
        ko_doc_pool,
        int(args.korean_doc_limit),
        rng,
        upsample=bool(args.upsample_korean_doc),
    )

    merged = []
    merged.extend(code_rows)
    merged.extend(eng_rows)
    merged.extend(math_rows)
    merged.extend(mmmlu_rows)
    merged.extend(ko_doc_rows)

    merged, dedupe_drop = dedupe_rows(merged)
    if bool(args.shuffle):
        rng.shuffle(merged)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    task_counts = Counter(str(r.get("task_type", "")).strip().lower() for r in merged)
    lang_counts = Counter(str(r.get("language", "")).strip().lower() for r in merged)
    source_counts = Counter(str(r.get("source_dataset", "")) for r in merged)

    manifest = {
        "output_jsonl": str(out_path),
        "rows_total": int(len(merged)),
        "task_counts": dict(task_counts),
        "language_counts_top20": dict(lang_counts.most_common(20)),
        "source_counts_top30": dict(source_counts.most_common(30)),
        "limits": {
            "code_limit": int(args.code_limit),
            "english_general_limit": int(args.english_general_limit),
            "math_reasoning_limit": int(args.math_reasoning_limit),
            "multilingual_reasoning_limit": int(args.multilingual_reasoning_limit),
            "korean_doc_limit": int(args.korean_doc_limit),
        },
        "upsample_counts": {
            "code": int(code_up),
            "english_general": int(eng_up),
            "math_reasoning": int(math_up),
            "multilingual_reasoning": 0,
            "korean_doc": int(ko_doc_up),
        },
        "dedupe_dropped": int(dedupe_drop),
        "pool_sizes": {
            "code_pool": int(len(code_pool)),
            "english_general_pool": int(len(eng_pool)),
            "math_reasoning_pool": int(len(math_pool)),
            "multilingual_reasoning_pool": int(len(mmmlu_pool)),
            "korean_doc_pool": int(len(ko_doc_pool)),
        },
        "filters": {
            "license_policy": "MIT only",
            "code_augment_keep": sorted(KEEP_CODE_AUG_TYPES),
            "korean_doc_augment_keep": sorted(KEEP_KO_DOC_AUG_TYPES),
            "mmmlu_transform": "output -> selected option text",
        },
        "collector_stats": {
            "code": dict(code_stats),
            "hq": dict(hq_stats),
            "mmmlu": dict(mmmlu_stats),
            "korean_doc": dict(ko_doc_stats),
        },
        "seed": int(args.seed),
        "shuffle": bool(args.shuffle),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "output_jsonl": str(out_path),
                "rows_total": int(len(merged)),
                "task_counts": dict(task_counts),
                "language_counts_top10": dict(lang_counts.most_common(10)),
                "pool_sizes": manifest["pool_sizes"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

