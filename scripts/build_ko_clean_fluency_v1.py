from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def pick_first_non_empty(row: Dict, keys: Sequence[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
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


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_key(inp: str, out: str) -> str:
    return f"{normalize_space(inp).lower()}\t{normalize_space(out).lower()}"


def ratio_hangul(text: str) -> float:
    s = str(text or "")
    letters = re.findall(r"[A-Za-z가-힣]", s)
    if not letters:
        return 0.0
    ko = re.findall(r"[가-힣]", s)
    return float(len(ko)) / float(max(1, len(letters)))


def looks_noisy(text: str) -> bool:
    s = str(text or "")
    if "\ufffd" in s:
        return True
    if re.search(r"(.)\1{10,}", s):
        return True
    if len(re.findall(r"[^\w\s가-힣.,!?():;\"'\-\[\]{}+/=*`~@#$%^&|\\]", s)) >= 8:
        return True
    return False


def clean_pair(
    inp: str,
    out: str,
    banned_outputs: Sequence[str],
    banned_patterns: Sequence[str],
    min_out_len: int,
    max_out_len: int,
) -> Tuple[bool, str]:
    q = normalize_space(inp)
    a = normalize_space(out)
    if len(q) < 2 or len(a) < int(min_out_len) or len(a) > int(max_out_len):
        return False, "length"
    for b in banned_outputs:
        if a == normalize_space(b):
            return False, "banned_output"
    for pat in banned_patterns:
        if re.search(pat, a, flags=re.IGNORECASE):
            return False, "banned_pattern"
    if looks_noisy(q) or looks_noisy(a):
        return False, "noisy"

    out_ko_ratio = ratio_hangul(a)
    if out_ko_ratio < 0.30:
        return False, "low_hangul"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Korean clean fluency dataset v1")
    ap.add_argument("--out", default="data/ko_clean_fluency_v1.jsonl")
    ap.add_argument(
        "--sources",
        default=(
            "data/slm_mit_unified_v4_ko_only.jsonl:6000,"
            "data/intro_override_v1.jsonl:2600,"
            "data/pure_ko_seed_v1.jsonl:1800,"
            "data/ko_chat_code_balanced_v2.jsonl:1200,"
            "data/ko_chat_balance_v1.jsonl:800"
        ),
        help="comma separated path:target_count entries",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_out_len", type=int, default=6)
    ap.add_argument("--max_out_len", type=int, default=280)
    args = ap.parse_args()

    rnd = random.Random(int(args.seed))
    banned_outputs = [
        "안녕하세요, 질문을 이해하고 간결하게 답하는 한국어 AI 어시스턴트입니다.",
    ]
    banned_patterns = [
        r"질문 의도를 파악해",
        r"단계별로 정리해드릴게요",
        r"원하면 짧게 3줄 요약",
    ]

    rows: List[Dict] = []
    seen: set[str] = set()
    source_stats: List[Dict] = []
    reject_stats: Dict[str, int] = {}

    for token in str(args.sources).split(","):
        item = token.strip()
        if not item or ":" not in item:
            continue
        p_raw, n_raw = item.rsplit(":", 1)
        path = Path(p_raw.strip())
        try:
            target = max(0, int(n_raw.strip()))
        except ValueError:
            continue
        if (not path.exists()) or target <= 0:
            continue

        src_rows = list(iter_jsonl(path))
        rnd.shuffle(src_rows)
        loaded = 0
        kept = 0

        for row in src_rows:
            inp = pick_first_non_empty(row, ["input", "instruction", "prompt", "question", "context"])
            out = pick_first_non_empty(row, ["output", "response", "answer", "completion", "target"])
            ok, reason = clean_pair(
                inp=inp,
                out=out,
                banned_outputs=banned_outputs,
                banned_patterns=banned_patterns,
                min_out_len=int(args.min_out_len),
                max_out_len=int(args.max_out_len),
            )
            loaded += 1
            if not ok:
                reject_stats[reason] = int(reject_stats.get(reason, 0) + 1)
                continue

            key = normalize_key(inp, out)
            if key in seen:
                reject_stats["dedupe"] = int(reject_stats.get("dedupe", 0) + 1)
                continue
            seen.add(key)

            rows.append(
                {
                    "task_type": str(row.get("task_type", "korean")).strip().lower() or "korean",
                    "segment_tag": str(row.get("segment_tag", "ko")).strip().lower() or "ko",
                    "language": str(row.get("language", "ko")).strip().lower() or "ko",
                    "_meta_quality_tier": str(row.get("_meta_quality_tier", "high")).strip().lower() or "high",
                    "input": normalize_space(inp),
                    "output": normalize_space(out),
                }
            )
            kept += 1
            if kept >= target:
                break

        source_stats.append(
            {
                "path": str(path),
                "target": int(target),
                "loaded": int(loaded),
                "kept": int(kept),
            }
        )

    rnd.shuffle(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "out": str(out_path),
                "rows": int(len(rows)),
                "seed": int(args.seed),
                "sources": source_stats,
                "reject_stats": reject_stats,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

