from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SPACE_RE = re.compile(r"\s+")
HANGUL_RE = re.compile(r"[가-힣]")


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Korean fluency + dehardcode mix dataset v1")
    p.add_argument("--strict", default="data/ko_pure_strict_v2.jsonl")
    p.add_argument("--dehard", default="data/dehardcode_patch_v2.jsonl")
    p.add_argument("--out", default="data/ko_fluency_dehard_mix_v1.jsonl")
    p.add_argument("--manifest", default="data/ko_fluency_dehard_mix_v1.manifest.json")
    p.add_argument("--seed", type=int, default=57)
    p.add_argument("--strict_take", type=int, default=60000)
    p.add_argument("--dehard_take", type=int, default=27548)
    p.add_argument("--min_input", type=int, default=2)
    p.add_argument("--max_input", type=int, default=1800)
    p.add_argument("--min_output", type=int, default=4)
    p.add_argument("--max_output", type=int, default=2400)
    return p.parse_args()


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


def clean_pair(row: Dict) -> Tuple[str, str, str]:
    inp = normalize_space(str(row.get("input", "")))
    out = normalize_space(str(row.get("output", "")))
    tier = normalize_space(str(row.get("_meta_quality_tier", "high")).lower()) or "high"
    return inp, out, tier


def is_valid_pair(inp: str, out: str, args: argparse.Namespace) -> bool:
    if not inp or not out:
        return False
    if len(inp) < int(args.min_input) or len(inp) > int(args.max_input):
        return False
    if len(out) < int(args.min_output) or len(out) > int(args.max_output):
        return False
    if re.search(r"(.)\1{12,}", out):
        return False
    if not (HANGUL_RE.search(inp + out) or re.search(r"[A-Za-z]", inp + out)):
        return False
    return True


def sample_rows(path: Path, take: int, seed: int, args: argparse.Namespace) -> Tuple[List[Dict], Dict[str, int]]:
    rows: List[Dict] = []
    stats = {"decode": 0, "invalid": 0}
    all_rows: List[Dict] = []
    for row in iter_jsonl(path):
        inp, out, tier = clean_pair(row)
        if not is_valid_pair(inp, out, args):
            stats["invalid"] += 1
            continue
        all_rows.append(
            {
                "task_type": "korean",
                "segment_tag": "ko",
                "language": "ko",
                "_meta_quality_tier": tier,
                "input": inp,
                "output": out,
                "_meta_source_file": str(path).replace("\\", "/"),
            }
        )
    rnd = random.Random(int(seed))
    rnd.shuffle(all_rows)
    if int(take) > 0:
        all_rows = all_rows[: int(take)]
    rows.extend(all_rows)
    return rows, stats


def main() -> None:
    args = parse_args()
    strict_path = Path(args.strict)
    dehard_path = Path(args.dehard)
    if not strict_path.exists():
        raise FileNotFoundError(f"strict dataset not found: {strict_path}")
    if not dehard_path.exists():
        raise FileNotFoundError(f"dehard dataset not found: {dehard_path}")

    strict_rows, strict_stats = sample_rows(
        path=strict_path,
        take=int(args.strict_take),
        seed=int(args.seed) + 1,
        args=args,
    )
    dehard_rows, dehard_stats = sample_rows(
        path=dehard_path,
        take=int(args.dehard_take),
        seed=int(args.seed) + 2,
        args=args,
    )

    merged = strict_rows + dehard_rows
    rnd = random.Random(int(args.seed))
    rnd.shuffle(merged)

    seen: set[bytes] = set()
    final_rows: List[Dict] = []
    dup_dropped = 0
    for row in merged:
        inp = str(row.get("input", ""))
        out = str(row.get("output", ""))
        k = dedupe_key(inp, out)
        if k in seen:
            dup_dropped += 1
            continue
        seen.add(k)
        final_rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "name": "ko_fluency_dehard_mix_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "inputs": {
            "strict": str(strict_path).replace("\\", "/"),
            "dehard": str(dehard_path).replace("\\", "/"),
        },
        "rows": {
            "strict_sampled": len(strict_rows),
            "dehard_sampled": len(dehard_rows),
            "duplicates_dropped": int(dup_dropped),
            "final": len(final_rows),
        },
        "drops": {
            "strict_invalid": int(strict_stats["invalid"]),
            "dehard_invalid": int(dehard_stats["invalid"]),
        },
        "config": {
            "strict_take": int(args.strict_take),
            "dehard_take": int(args.dehard_take),
            "min_input": int(args.min_input),
            "max_input": int(args.max_input),
            "min_output": int(args.min_output),
            "max_output": int(args.max_output),
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] wrote {len(final_rows)} rows -> {out_path}")
    print(f"[done] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
