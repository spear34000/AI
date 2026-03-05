from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SPACE_RE = re.compile(r"\s+")
HANGUL_RE = re.compile(r"[가-힣]")
INTRO_IN_RE = re.compile(r"(자기소개|본인 소개|너를 .*소개|한 줄 소개|who are you|self[ -]?intro)", re.IGNORECASE)
INTRO_OUT_RE = re.compile(r"(AI 어시스턴트|AI 도우미|질문 의도|한 줄로 말하면|간단히 말해|요약하면)", re.IGNORECASE)
BAD_REPEAT_RE = re.compile(r"(.)\1{12,}")


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def make_row(inp: str, out: str, source: str, tier: str = "high") -> Dict[str, str]:
    return {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": str(tier or "high"),
        "input": normalize_space(inp),
        "output": normalize_space(out),
        "_meta_source_file": source,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build balanced Korean general+reasoning dataset")
    p.add_argument("--hq", default="data/quality/hq_ko_chat_v1.jsonl")
    p.add_argument("--kullm", default="data/kullm_v2_sample.jsonl")
    p.add_argument(
        "--logic_files",
        default="data/logic_reasoning_v3_combined.jsonl,data/logic_reasoning_cot_v1.jsonl,data/capability_pack_v2.jsonl",
    )
    p.add_argument("--term", default="data/term_focus_clean_v2.jsonl")
    p.add_argument("--out", default="data/ko_general_balance_v1.jsonl")
    p.add_argument("--manifest", default="data/ko_general_balance_v1.manifest.json")
    p.add_argument("--seed", type=int, default=97)
    p.add_argument("--take_hq", type=int, default=50000)
    p.add_argument("--take_kullm", type=int, default=7000)
    p.add_argument("--take_logic", type=int, default=9000)
    p.add_argument("--take_term", type=int, default=12000)
    p.add_argument("--take_intro", type=int, default=220)
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


def pick_text(row: Dict, keys: Sequence[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        s = normalize_space(v)
        if s:
            return s
    return ""


def is_intro_input(text: str) -> bool:
    return bool(INTRO_IN_RE.search(normalize_space(text)))


def is_intro_output(text: str) -> bool:
    return bool(INTRO_OUT_RE.search(normalize_space(text)))


def valid_pair(inp: str, out: str) -> bool:
    if not inp or not out:
        return False
    if len(inp) < 2 or len(inp) > 320:
        return False
    if len(out) < 8 or len(out) > 640:
        return False
    if BAD_REPEAT_RE.search(out):
        return False
    if "\ufffd" in inp or "\ufffd" in out:
        return False
    text = inp + "\n" + out
    hangul_n = len(HANGUL_RE.findall(text))
    if hangul_n < 2:
        return False
    return True


def gather_rows(path: Path, source_label: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    general_rows: List[Dict[str, str]] = []
    intro_rows: List[Dict[str, str]] = []
    if not path.exists():
        return general_rows, intro_rows

    for row in iter_jsonl(path):
        inp = pick_text(row, ["input", "instruction", "prompt", "question"])
        out = pick_text(row, ["output", "response", "answer", "completion", "target"])
        if not valid_pair(inp, out):
            continue

        intro_in = is_intro_input(inp)
        intro_out = is_intro_output(out)
        if intro_in:
            if intro_out:
                intro_rows.append(make_row(inp, out, source=source_label, tier=str(row.get("_meta_quality_tier", "high"))))
            continue
        if intro_out:
            # Avoid generic template collapse for non-intro prompts.
            continue

        general_rows.append(make_row(inp, out, source=source_label, tier=str(row.get("_meta_quality_tier", "high"))))

    return general_rows, intro_rows


def take_unique(rows: Sequence[Dict[str, str]], n: int, seen: set[bytes], dst: List[Dict[str, str]]) -> int:
    added = 0
    for row in rows:
        if added >= int(n):
            break
        k = dedupe_key(row["input"], row["output"])
        if k in seen:
            continue
        seen.add(k)
        dst.append(row)
        added += 1
    return added


def main() -> None:
    args = parse_args()
    rnd = random.Random(int(args.seed))
    dataset_name = Path(args.out).stem

    hq_path = Path(args.hq)
    kullm_path = Path(args.kullm)
    logic_paths = [Path(x.strip()) for x in str(args.logic_files).split(",") if x.strip()]
    term_path = Path(args.term)

    if not hq_path.exists():
        raise FileNotFoundError(f"hq dataset not found: {hq_path}")
    if not term_path.exists():
        raise FileNotFoundError(f"term dataset not found: {term_path}")

    hq_general, hq_intro = gather_rows(hq_path, source_label=str(hq_path).replace("\\", "/"))
    kullm_general, _ = gather_rows(kullm_path, source_label=str(kullm_path).replace("\\", "/"))

    logic_general: List[Dict[str, str]] = []
    for lp in logic_paths:
        lg, _ = gather_rows(lp, source_label=str(lp).replace("\\", "/"))
        logic_general.extend(lg)

    term_general, _ = gather_rows(term_path, source_label=str(term_path).replace("\\", "/"))

    rnd.shuffle(hq_general)
    rnd.shuffle(hq_intro)
    rnd.shuffle(kullm_general)
    rnd.shuffle(logic_general)
    rnd.shuffle(term_general)

    rows: List[Dict[str, str]] = []
    seen: set[bytes] = set()

    add_hq = take_unique(hq_general, int(args.take_hq), seen, rows)
    add_kullm = take_unique(kullm_general, int(args.take_kullm), seen, rows)
    add_logic = take_unique(logic_general, int(args.take_logic), seen, rows)
    add_term = take_unique(term_general, int(args.take_term), seen, rows)
    add_intro = take_unique(hq_intro, int(args.take_intro), seen, rows)

    rnd.shuffle(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "name": str(dataset_name),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "inputs": {
            "hq": str(hq_path).replace("\\", "/"),
            "kullm": str(kullm_path).replace("\\", "/"),
            "logic_files": [str(p).replace("\\", "/") for p in logic_paths],
            "term": str(term_path).replace("\\", "/"),
        },
        "rows": {
            "hq_added": int(add_hq),
            "kullm_added": int(add_kullm),
            "logic_added": int(add_logic),
            "term_added": int(add_term),
            "intro_added": int(add_intro),
            "final": len(rows),
        },
        "pool": {
            "hq_general_pool": len(hq_general),
            "hq_intro_pool": len(hq_intro),
            "kullm_pool": len(kullm_general),
            "logic_pool": len(logic_general),
            "term_pool": len(term_general),
        },
        "config": {
            "take_hq": int(args.take_hq),
            "take_kullm": int(args.take_kullm),
            "take_logic": int(args.take_logic),
            "take_term": int(args.take_term),
            "take_intro": int(args.take_intro),
        },
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] rows={len(rows)} -> {out_path}")
    print(f"[done] manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
