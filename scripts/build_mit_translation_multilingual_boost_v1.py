from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_OUT_JSONL = Path("data/mit_translation_multilingual_boost_v1.jsonl")
DEFAULT_MANIFEST = Path("data/mit_translation_multilingual_boost_v1.manifest.json")

DEFAULT_HQ_SUPERPACK = Path("data/hf_mit_language_superpack_hq_v2.jsonl")
DEFAULT_RAW_SUPERPACK = Path("data/hf_mit_language_superpack_v1.jsonl")
DEFAULT_MIT_UNIFIED = Path("data/slm_mit_unified_v4.jsonl")

TRANSLATE_HINTS = [
    "translate",
    "translation",
    "번역",
    "통역",
    "traduce",
    "traducción",
    "traduction",
    "traduire",
    "übersetze",
    "übersetzen",
    "переведи",
    "翻译",
    "翻譯",
]

META_NOISE_HINTS = [
    "task:",
    "guidelines:",
    "definition:",
    "confidence:",
    "explanation:",
    "i'm glad you're here",
]

MMMLU_SOURCE_PREFIX = "openai/MMMLU::"
MMMLU_ANSWER_RE = re.compile(r"\b([ABCD])\b")
MMMLU_SUBJECT_RE = re.compile(r"^Question\s*\((.*?)\)\s*:", flags=re.IGNORECASE)
MMMLU_OPTION_RE = re.compile(r"^([ABCD])\)\s*(.+)$")
WS_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MIT translation + multilingual boost dataset.")
    p.add_argument("--out_jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--hq_superpack", type=Path, default=DEFAULT_HQ_SUPERPACK)
    p.add_argument("--raw_superpack", type=Path, default=DEFAULT_RAW_SUPERPACK)
    p.add_argument("--mit_unified", type=Path, default=DEFAULT_MIT_UNIFIED)

    p.add_argument("--translation_limit", type=int, default=9000)
    p.add_argument("--multilingual_reasoning_limit", type=int, default=12000)
    p.add_argument("--english_anchor_limit", type=int, default=15000)
    p.add_argument("--korean_support_limit", type=int, default=3000)

    p.add_argument("--upsample_translation", action="store_true")
    p.add_argument("--upsample_korean_support", action="store_true")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def clean_text(text: str) -> str:
    src = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    src = re.sub(r"[ \t]+", " ", src)
    src = re.sub(r"\n{3,}", "\n\n", src)
    return src.strip()


def normalize_space(text: str) -> str:
    return WS_RE.sub(" ", str(text or "").strip())


def normalize_license(value: str) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def is_mit_license(value: str) -> bool:
    lic = normalize_license(value)
    return lic in {"mit", "mit license"}


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


def clone_row(row: Dict) -> Dict:
    rec = dict(row)
    rec["input"] = clean_text(rec.get("input", ""))
    rec["output"] = clean_text(rec.get("output", ""))
    rec["license"] = "MIT"
    return rec


def row_key(row: Dict) -> str:
    raw = f"{normalize_space(str(row.get('input', '')).lower())}\n{normalize_space(str(row.get('output', '')).lower())}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def validate_io(inp: str, out: str) -> bool:
    i = clean_text(inp)
    o = clean_text(out)
    if len(i) < 8 or len(o) < 8:
        return False
    if len(i) > 2600 or len(o) > 1200:
        return False
    if re.search(r"(.)\1{14,}", o):
        return False
    return True


def is_translation_prompt(text: str) -> bool:
    t = str(text or "").lower()
    return any(k in t for k in TRANSLATE_HINTS)


def clean_translation_output(text: str) -> str:
    out = clean_text(text)
    # Remove confidence tails frequently attached by preference datasets.
    out = re.sub(r"\s*Confidence\s*:\s*\d+%?\s*$", "", out, flags=re.IGNORECASE)
    out = clean_text(out)
    return out


def is_low_quality_translation_pair(inp: str, out: str) -> bool:
    i = str(inp or "").lower()
    o = str(out or "").lower()
    if any(h in o for h in META_NOISE_HINTS):
        return True
    # Avoid long procedural/meta instruction echoes.
    if "given the task definition" in i and len(o) > 420:
        return True
    if o.count("\n") >= 8:
        return True
    return False


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
    chosen = opts[ans]
    subject = parse_mmmlu_subject(inp)
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
    up = 0
    if upsample:
        while len(out) < lim:
            row = dict(pool[rng.randrange(len(pool))])
            row["_meta_upsampled"] = True
            out.append(row)
            up += 1
    return out[:lim], up


def balanced_multilingual_sample(rows: List[Dict], limit: int, rng: random.Random) -> List[Dict]:
    lim = int(max(0, limit))
    if lim == 0 or not rows:
        return []

    by_lang: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        lg = str(r.get("language", "en")).strip().lower() or "en"
        by_lang[lg].append(r)

    langs = sorted(by_lang.keys())
    if not langs:
        return []
    per_lang = max(1, lim // len(langs))

    out: List[Dict] = []
    order: Dict[str, List[int]] = {}
    for lg in langs:
        idxs = list(range(len(by_lang[lg])))
        rng.shuffle(idxs)
        order[lg] = idxs
        take = min(per_lang, len(idxs))
        out.extend(by_lang[lg][i] for i in idxs[:take])

    if len(out) < lim:
        remain: List[Dict] = []
        for lg in langs:
            idxs = order[lg]
            take_start = min(per_lang, len(idxs))
            remain.extend(by_lang[lg][i] for i in idxs[take_start:])
        rng.shuffle(remain)
        out.extend(remain[: max(0, lim - len(out))])

    if len(out) > lim:
        rng.shuffle(out)
        out = out[:lim]
    return out


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

    for p in [args.hq_superpack, args.raw_superpack, args.mit_unified]:
        if not Path(p).exists():
            raise FileNotFoundError(f"input not found: {p}")

    stats = Counter()
    translation_pool: List[Dict] = []
    english_anchor_pool: List[Dict] = []
    mmmlu_pool: List[Dict] = []
    korean_support_pool: List[Dict] = []

    for row in iter_jsonl(Path(args.hq_superpack)):
        stats["hq_read"] += 1
        if not is_mit_license(str(row.get("license", ""))):
            continue
        inp = clean_text(row.get("input", ""))
        out = clean_text(row.get("output", ""))
        if not validate_io(inp, out):
            continue
        rec = clone_row(row)
        rec["_meta_quality_tier"] = "high"

        if is_translation_prompt(inp):
            rec["output"] = clean_translation_output(out)
            if not validate_io(rec["input"], rec["output"]):
                stats["translation_drop_short"] += 1
                continue
            if is_low_quality_translation_pair(rec["input"], rec["output"]):
                stats["translation_drop_noise"] += 1
                continue
            translation_pool.append(rec)
            stats["translation_from_hq"] += 1
        else:
            english_anchor_pool.append(rec)
            stats["anchor_from_hq"] += 1

    for row in iter_jsonl(Path(args.raw_superpack)):
        stats["raw_read"] += 1
        if not is_mit_license(str(row.get("license", ""))):
            continue
        inp = clean_text(row.get("input", ""))
        out = clean_text(row.get("output", ""))
        if not validate_io(inp, out):
            continue

        src = str(row.get("source_dataset", "")).strip()
        if src.startswith(MMMLU_SOURCE_PREFIX):
            t = transform_mmmlu_row(row)
            if t is None:
                stats["mmmlu_transform_fail"] += 1
                continue
            mmmlu_pool.append(t)
            stats["mmmlu_transform_ok"] += 1
            continue

        if is_translation_prompt(inp):
            rec = clone_row(row)
            rec["output"] = clean_translation_output(out)
            if not validate_io(rec["input"], rec["output"]):
                continue
            if is_low_quality_translation_pair(rec["input"], rec["output"]):
                continue
            translation_pool.append(rec)
            stats["translation_from_raw"] += 1

    for row in iter_jsonl(Path(args.mit_unified)):
        stats["unified_read"] += 1
        if not is_mit_license(str(row.get("license", ""))):
            continue
        if str(row.get("task_type", "")).strip().lower() != "korean":
            continue
        aug = str(row.get("_augment_type", ""))
        if aug not in {"", "ko_summary_extract", "ko_summary_extract_v2"}:
            continue
        inp = clean_text(row.get("input", ""))
        out = clean_text(row.get("output", ""))
        if not validate_io(inp, out):
            continue
        rec = clone_row(row)
        rec["task_type"] = "korean"
        rec["segment_tag"] = "ko"
        rec["language"] = "ko"
        korean_support_pool.append(rec)
        stats["korean_support_kept"] += 1

    t_rows, t_up = sample_rows(
        translation_pool,
        int(args.translation_limit),
        rng,
        upsample=bool(args.upsample_translation),
    )
    mm_rows = balanced_multilingual_sample(mmmlu_pool, int(args.multilingual_reasoning_limit), rng)
    en_rows, en_up = sample_rows(english_anchor_pool, int(args.english_anchor_limit), rng, upsample=False)
    ko_rows, ko_up = sample_rows(
        korean_support_pool,
        int(args.korean_support_limit),
        rng,
        upsample=bool(args.upsample_korean_support),
    )

    merged: List[Dict] = []
    merged.extend(t_rows)
    merged.extend(mm_rows)
    merged.extend(en_rows)
    merged.extend(ko_rows)

    merged, dedupe_drop = dedupe_rows(merged)
    if bool(args.shuffle):
        rng.shuffle(merged)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

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
            "translation_limit": int(args.translation_limit),
            "multilingual_reasoning_limit": int(args.multilingual_reasoning_limit),
            "english_anchor_limit": int(args.english_anchor_limit),
            "korean_support_limit": int(args.korean_support_limit),
        },
        "upsample_counts": {
            "translation": int(t_up),
            "multilingual_reasoning": 0,
            "english_anchor": int(en_up),
            "korean_support": int(ko_up),
        },
        "pool_sizes": {
            "translation_pool": int(len(translation_pool)),
            "multilingual_reasoning_pool": int(len(mmmlu_pool)),
            "english_anchor_pool": int(len(english_anchor_pool)),
            "korean_support_pool": int(len(korean_support_pool)),
        },
        "dedupe_dropped": int(dedupe_drop),
        "filters": {
            "license_policy": "MIT only",
            "translation_prompt_hints": TRANSLATE_HINTS,
            "drop_translation_meta_noise": META_NOISE_HINTS,
            "mmmlu_transform": "output -> selected option text",
        },
        "collector_stats": dict(stats),
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

