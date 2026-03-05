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


def is_bad_pair(inp: str, out: str, banned_outputs: Sequence[str], banned_regex: Sequence[str]) -> bool:
    src_out = normalize_space(out)
    src_in = normalize_space(inp)
    if len(src_in) < 2 or len(src_out) < 2:
        return True
    for b in banned_outputs:
        if normalize_space(b) == src_out:
            return True
    for pat in banned_regex:
        if re.search(pat, src_out, flags=re.IGNORECASE):
            return True
    return False


def convert_row(row: Dict) -> Tuple[str, str, str]:
    inp = pick_first_non_empty(row, ["input", "instruction", "prompt", "question", "context"])
    out = pick_first_non_empty(row, ["output", "response", "answer", "completion", "target"])
    tier = str(row.get("_meta_quality_tier", "mid")).strip().lower() or "mid"
    return inp, out, tier


def load_source(
    path: Path,
    target: int,
    rnd: random.Random,
    banned_outputs: Sequence[str],
    banned_regex: Sequence[str],
) -> List[Dict]:
    rows = list(iter_jsonl(path))
    rnd.shuffle(rows)
    picked: List[Dict] = []
    seen: set[str] = set()

    for row in rows:
        inp, out, tier = convert_row(row)
        if is_bad_pair(inp, out, banned_outputs=banned_outputs, banned_regex=banned_regex):
            continue
        key = normalize_key(inp, out)
        if key in seen:
            continue
        seen.add(key)
        picked.append(
            {
                "task_type": str(row.get("task_type", "korean")).strip().lower() or "korean",
                "segment_tag": str(row.get("segment_tag", "ko")).strip().lower() or "ko",
                "language": str(row.get("language", "ko")).strip().lower() or "ko",
                "_meta_quality_tier": tier if tier in {"high", "mid", "low"} else "mid",
                "input": inp,
                "output": out,
            }
        )
        if len(picked) >= int(target):
            break

    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description="Build mixed Korean dataset without fixed intro phrase")
    ap.add_argument("--out", default="data/pure_ko_mix_nofixed_v1.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--sources",
        default="data/intro_override_v1.jsonl:4000,data/pure_ko_seed_v1.jsonl:2400,data/chat_quality_corpus_v2.jsonl:1600",
        help="comma separated path:count list",
    )
    args = ap.parse_args()

    rnd = random.Random(int(args.seed))
    banned_outputs = [
        "안녕하세요, 질문을 이해하고 간결하게 답하는 한국어 AI 어시스턴트입니다.",
    ]
    banned_regex = [
        r"질문 의도를 파악해",
        r"단계별로 정리해드릴게요",
    ]

    buckets: List[Dict] = []
    source_stats: List[Dict] = []
    global_seen: set[str] = set()

    for token in str(args.sources).split(","):
        item = token.strip()
        if not item or ":" not in item:
            continue
        p_raw, c_raw = item.rsplit(":", 1)
        p = Path(p_raw.strip())
        try:
            n_target = max(0, int(c_raw.strip()))
        except ValueError:
            continue
        if not p.exists() or n_target <= 0:
            continue

        part = load_source(
            path=p,
            target=n_target,
            rnd=rnd,
            banned_outputs=banned_outputs,
            banned_regex=banned_regex,
        )
        kept = 0
        for row in part:
            key = normalize_key(str(row.get("input", "")), str(row.get("output", "")))
            if key in global_seen:
                continue
            global_seen.add(key)
            buckets.append(row)
            kept += 1

        source_stats.append(
            {
                "path": str(p),
                "target": int(n_target),
                "loaded": int(len(part)),
                "kept_after_global_dedupe": int(kept),
            }
        )

    rnd.shuffle(buckets)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in buckets:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "out": str(out_path),
                "rows": int(len(buckets)),
                "seed": int(args.seed),
                "sources": source_stats,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

