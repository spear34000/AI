from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_INPUTS = [
    "data/slm_mit_unified_v4.jsonl",
    "data/slm_mit_unified_v3.jsonl",
    "data/slm_mit_unified_v2.jsonl",
    "data/slm_mit_all_local_v1.jsonl",
    "data_archive_20260219_160805/slm_mit_only_expanded_more.jsonl",
    "data_archive_20260219_160805/slm_mit_only_expanded.jsonl",
]


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


def pick_text(row: Dict, keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        t = str(v).strip()
        if t:
            return t
    return ""


def infer_domain(row: Dict) -> str:
    task = str(row.get("task_type", "")).strip().lower()
    seg = str(row.get("segment_tag", "")).strip().lower()
    lang = str(row.get("language", "")).strip().lower()

    if task == "code" or seg == "code":
        return "code"
    if task in {"korean", "ko"} or seg == "ko" or lang == "ko":
        return "korean"
    if task in {"english", "en"} or seg in {"english", "en"} or lang == "en":
        return "english"
    if task == "doc" or seg == "doc":
        return "doc"
    return "other"


def normalize_row(row: Dict) -> Dict | None:
    inp = pick_text(row, ["input", "instruction", "prompt", "question", "context"])
    out = pick_text(row, ["output", "response", "answer", "completion", "target"])
    if not inp or not out:
        return None

    domain = infer_domain(row)
    rec = dict(row)
    rec["input"] = inp
    rec["output"] = out
    rec["license"] = "MIT"

    if domain == "code":
        rec["task_type"] = "code"
        rec["segment_tag"] = "code"
        if not str(rec.get("language", "")).strip():
            rec["language"] = "en"
    elif domain == "korean":
        rec["task_type"] = "korean"
        rec["segment_tag"] = "ko"
        rec["language"] = "ko"
    elif domain == "english":
        rec["task_type"] = "english"
        rec["segment_tag"] = "english"
        rec["language"] = "en"
    elif domain == "doc":
        rec["task_type"] = "doc"
        rec["segment_tag"] = "doc"
        if not str(rec.get("language", "")).strip():
            rec["language"] = "en"
    else:
        if not str(rec.get("task_type", "")).strip():
            rec["task_type"] = "other"
        if not str(rec.get("segment_tag", "")).strip():
            rec["segment_tag"] = "other"

    if not str(rec.get("source_dataset", "")).strip():
        src = str(rec.get("source", "")).strip()
        if src:
            rec["source_dataset"] = src

    return rec


def row_key(row: Dict, domain: str) -> str:
    raw = f"{domain}\n{str(row.get('input', ''))}\n{str(row.get('output', ''))}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def sample_domain_rows(
    pool: List[Dict],
    limit: int,
    rng: random.Random,
    upsample: bool = False,
) -> tuple[List[Dict], int]:
    if int(limit) <= 0 or not pool:
        return [], 0
    if len(pool) >= int(limit):
        idxs = list(range(len(pool)))
        rng.shuffle(idxs)
        return [pool[i] for i in idxs[: int(limit)]], 0

    out = list(pool)
    rng.shuffle(out)
    upsample_n = 0
    if bool(upsample):
        while len(out) < int(limit):
            chosen = dict(pool[rng.randrange(len(pool))])
            chosen["_meta_upsampled"] = True
            out.append(chosen)
            upsample_n += 1
    return out[: int(limit)], int(upsample_n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MIT-only domain-mixed training data.")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out_jsonl", default="data/slm_mit_domain_mix_v1.jsonl")
    parser.add_argument("--manifest", default="data/slm_mit_domain_mix_v1.manifest.json")
    parser.add_argument("--code_limit", type=int, default=120000)
    parser.add_argument("--korean_limit", type=int, default=30000)
    parser.add_argument("--english_limit", type=int, default=0)
    parser.add_argument("--doc_limit", type=int, default=0)
    parser.add_argument("--other_limit", type=int, default=0)
    parser.add_argument("--upsample_korean", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))

    pools: Dict[str, List[Dict]] = defaultdict(list)
    seen = set()
    excluded = Counter()
    file_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"read_rows": 0, "mit_rows": 0, "kept_rows": 0})

    for raw_path in args.inputs:
        path = Path(raw_path)
        pkey = str(path)
        if not path.exists():
            excluded["missing_input_file"] += 1
            continue

        for row in iter_jsonl(path):
            file_stats[pkey]["read_rows"] += 1
            if not is_mit_license(str(row.get("license", ""))):
                excluded["non_mit_license"] += 1
                continue
            file_stats[pkey]["mit_rows"] += 1

            rec = normalize_row(row)
            if rec is None:
                excluded["empty_or_invalid_io"] += 1
                continue
            domain = infer_domain(rec)
            key = row_key(rec, domain)
            if key in seen:
                excluded["duplicate"] += 1
                continue
            seen.add(key)
            pools[domain].append(rec)
            file_stats[pkey]["kept_rows"] += 1

    unique_counts = {k: len(v) for k, v in pools.items()}

    limits = {
        "code": int(args.code_limit),
        "korean": int(args.korean_limit),
        "english": int(args.english_limit),
        "doc": int(args.doc_limit),
        "other": int(args.other_limit),
    }

    sampled: List[Dict] = []
    upsample_counts: Dict[str, int] = {}
    for domain in ["code", "korean", "english", "doc", "other"]:
        rows, up_n = sample_domain_rows(
            pool=pools.get(domain, []),
            limit=int(limits[domain]),
            rng=rng,
            upsample=(domain == "korean" and bool(args.upsample_korean)),
        )
        sampled.extend(rows)
        upsample_counts[domain] = int(up_n)

    if bool(args.shuffle):
        rng.shuffle(sampled)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    final_counts = Counter(str(r.get("task_type", "")).strip().lower() for r in sampled)
    manifest = {
        "inputs": [str(p) for p in args.inputs],
        "output_jsonl": str(out_path),
        "rows_total": len(sampled),
        "unique_domain_counts": unique_counts,
        "final_task_counts": dict(final_counts),
        "limits": limits,
        "upsample_counts": upsample_counts,
        "excluded_counts": dict(excluded),
        "file_stats": dict(file_stats),
        "seed": int(args.seed),
        "shuffle": bool(args.shuffle),
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
                "rows_total": len(sampled),
                "unique_domain_counts": unique_counts,
                "final_task_counts": dict(final_counts),
                "upsample_counts": upsample_counts,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
