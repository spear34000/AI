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
    p = argparse.ArgumentParser(description="Build hq_ko_chat + dehardcode mix v1")
    p.add_argument("--hq", default="data/quality/hq_ko_chat_v1.jsonl")
    p.add_argument("--dehard", default="data/dehardcode_patch_v2.jsonl")
    p.add_argument("--out", default="data/hq_dehard_mix_v1.jsonl")
    p.add_argument("--manifest", default="data/hq_dehard_mix_v1.manifest.json")
    p.add_argument("--seed", type=int, default=61)
    p.add_argument("--hq_take", type=int, default=90000)
    p.add_argument("--dehard_take", type=int, default=27548)
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


def normalize_row(row: Dict, source: Path) -> Tuple[str, str, str]:
    inp = normalize_space(str(row.get("input", "")))
    out = normalize_space(str(row.get("output", "")))
    tier = normalize_space(str(row.get("_meta_quality_tier", "high")).lower()) or "high"
    return inp, out, tier


def is_valid_pair(inp: str, out: str) -> bool:
    if not inp or not out:
        return False
    if len(inp) < 2 or len(inp) > 1800:
        return False
    if len(out) < 4 or len(out) > 2400:
        return False
    if re.search(r"(.)\1{12,}", out):
        return False
    if not (HANGUL_RE.search(inp + out) or re.search(r"[A-Za-z]", inp + out)):
        return False
    return True


def contains_banned_output(out: str) -> bool:
    banned = [
        "질문 의도를 파악해 단계별로 정리해드릴게요. 원하면 짧게 3줄 요약으로도 답할 수 있어요.",
        "안녕하세요, 질문을 이해하고 간결하게 답하는 한국어 AI 어시스턴트입니다.",
    ]
    s = normalize_space(out)
    return any(b in s for b in banned)


def sample_rows(path: Path, take: int, seed: int) -> Tuple[List[Dict], Dict[str, int]]:
    stats = {"invalid": 0, "banned": 0}
    rows: List[Dict] = []
    for row in iter_jsonl(path):
        inp, out, tier = normalize_row(row, path)
        if not is_valid_pair(inp, out):
            stats["invalid"] += 1
            continue
        if contains_banned_output(out):
            stats["banned"] += 1
            continue
        rows.append(
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
    rnd.shuffle(rows)
    if int(take) > 0:
        rows = rows[: int(take)]
    return rows, stats


def main() -> None:
    args = parse_args()
    hq_path = Path(args.hq)
    dehard_path = Path(args.dehard)
    if not hq_path.exists():
        raise FileNotFoundError(f"hq dataset not found: {hq_path}")
    if not dehard_path.exists():
        raise FileNotFoundError(f"dehard dataset not found: {dehard_path}")

    hq_rows, hq_stats = sample_rows(hq_path, take=int(args.hq_take), seed=int(args.seed) + 1)
    dehard_rows, dehard_stats = sample_rows(dehard_path, take=int(args.dehard_take), seed=int(args.seed) + 2)

    merged = hq_rows + dehard_rows
    rnd = random.Random(int(args.seed))
    rnd.shuffle(merged)

    seen: set[bytes] = set()
    final_rows: List[Dict] = []
    duplicates_dropped = 0
    for row in merged:
        k = dedupe_key(str(row.get("input", "")), str(row.get("output", "")))
        if k in seen:
            duplicates_dropped += 1
            continue
        seen.add(k)
        final_rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "name": "hq_dehard_mix_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "inputs": {
            "hq": str(hq_path).replace("\\", "/"),
            "dehard": str(dehard_path).replace("\\", "/"),
        },
        "rows": {
            "hq_sampled": len(hq_rows),
            "dehard_sampled": len(dehard_rows),
            "duplicates_dropped": int(duplicates_dropped),
            "final": len(final_rows),
        },
        "drops": {
            "hq_invalid": int(hq_stats["invalid"]),
            "hq_banned": int(hq_stats["banned"]),
            "dehard_invalid": int(dehard_stats["invalid"]),
            "dehard_banned": int(dehard_stats["banned"]),
        },
        "config": {
            "hq_take": int(args.hq_take),
            "dehard_take": int(args.dehard_take),
        },
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] wrote {len(final_rows)} rows -> {out_path}")
    print(f"[done] manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
