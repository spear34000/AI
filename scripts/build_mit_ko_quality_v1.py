from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset


def clean_text(text: str) -> str:
    t = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def normalize_key(text: str) -> str:
    t = clean_text(text).lower()
    t = re.sub(r"\s+", " ", t)
    return t


def hangul_count(text: str) -> int:
    return sum(1 for ch in text if 0xAC00 <= ord(ch) <= 0xD7A3)


def looks_noisy(text: str) -> bool:
    t = clean_text(text)
    if not t:
        return True

    q_n = t.count("?")
    if "???" in t:
        return True
    if q_n >= 12 and (q_n / float(max(1, len(t)))) > 0.03:
        return True
    if re.search(r"(.)\1{10,}", t):
        return True

    tokens = re.findall(r"[A-Za-z0-9가-힣]{2,}", t.lower())
    if len(tokens) >= 24:
        top_count = Counter(tokens).most_common(1)[0][1]
        if (top_count / float(len(tokens))) > 0.30:
            return True
    return False


def valid_pair(inp: str, out: str, require_hangul: bool = True) -> bool:
    i = clean_text(inp)
    o = clean_text(out)
    if len(i) < 2 or len(o) < 8:
        return False
    if len(i) > 2200 or len(o) > 2600:
        return False

    merged = f"{i}\n{o}"
    if looks_noisy(merged):
        return False
    if require_hangul and hangul_count(merged) < 8:
        return False
    return True


def make_row(inp: str, out: str, tier: str, source: str, segment_tag: str = "ko") -> Dict:
    return {
        "task_type": "korean" if segment_tag == "ko" else segment_tag,
        "segment_tag": segment_tag,
        "language": "ko",
        "_meta_quality_tier": tier,
        "source": source,
        "input": clean_text(inp),
        "output": clean_text(out),
    }


def dedupe_rows(rows: Iterable[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for r in rows:
        inp = str(r.get("input", "")).strip()
        out_txt = str(r.get("output", "")).strip()
        if not inp or not out_txt:
            continue
        key = (normalize_key(inp), normalize_key(out_txt))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def collect_kovast(limit_pairs: int, seed: int) -> List[Dict]:
    ds = load_dataset("maywell/koVast", split="train")
    ds = ds.shuffle(seed=int(seed))

    rows: List[Dict] = []
    for rec in ds:
        conv = rec.get("conversations")
        if not isinstance(conv, list) or not conv:
            continue

        pending_user = ""
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", "")).strip().lower()
            text = clean_text(turn.get("value", ""))
            if not text:
                continue
            if role in {"human", "user"}:
                pending_user = text
                continue
            if role in {"gpt", "assistant", "bot"}:
                if not pending_user:
                    continue
                if valid_pair(pending_user, text, require_hangul=True):
                    rows.append(make_row(pending_user, text, tier="high", source="maywell/koVast", segment_tag="ko"))
                    if len(rows) >= int(limit_pairs):
                        return rows
                pending_user = ""
    return rows


def collect_ko_commercial(limit_pairs: int, seed: int) -> List[Dict]:
    ds = load_dataset("MarkrAI/KoCommercial-Dataset", split="train")
    ds = ds.shuffle(seed=int(seed))

    rows: List[Dict] = []
    for rec in ds:
        instruction = clean_text(rec.get("instruction", ""))
        extra_input = clean_text(rec.get("input", ""))
        output = clean_text(rec.get("output", ""))
        if not instruction:
            continue
        inp = instruction if not extra_input else f"{instruction}\n\n추가 정보:\n{extra_input}"
        if not valid_pair(inp, output, require_hangul=True):
            continue
        rows.append(make_row(inp, output, tier="mid", source="MarkrAI/KoCommercial-Dataset", segment_tag="ko"))
        if len(rows) >= int(limit_pairs):
            break
    return rows


def pick_best_won_response(rec: Dict) -> str:
    keys = [
        "original_response",
        "Qwen/Qwen2.5-7B-Instruct_response",
        "google/gemma-2-9b-it_response",
        "Qwen/Qwen2.5-1.5B-Instruct_response",
        "google/gemma-2-2b-it_response",
    ]
    for k in keys:
        v = clean_text(rec.get(k, ""))
        if v:
            return v
    return ""


def collect_won_instruct(limit_pairs: int, seed: int) -> List[Dict]:
    ds = load_dataset("KRX-Data/Won-Instruct", split="train")
    ds = ds.shuffle(seed=int(seed))

    rows: List[Dict] = []
    for rec in ds:
        prompt = clean_text(rec.get("prompt", ""))
        answer = pick_best_won_response(rec)
        if not valid_pair(prompt, answer, require_hangul=True):
            continue
        rows.append(make_row(prompt, answer, tier="mid", source="KRX-Data/Won-Instruct", segment_tag="ko"))
        if len(rows) >= int(limit_pairs):
            break
    return rows


def load_local_rows(path: Path, limit: int, seed: int) -> List[Dict]:
    if not path.exists() or int(limit) <= 0:
        return []

    rows: List[Dict] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            inp = clean_text(obj.get("input", ""))
            out = clean_text(obj.get("output", ""))
            if not valid_pair(inp, out, require_hangul=False):
                continue
            segment = str(obj.get("segment_tag", "ko")).strip().lower()
            if segment not in {"ko", "code", "doc", "english", "other"}:
                segment = "ko"
            tier = str(obj.get("_meta_quality_tier", "mid")).strip().lower()
            if tier not in {"high", "mid", "base"}:
                tier = "mid"
            rows.append(
                {
                    "task_type": "korean" if segment == "ko" else segment,
                    "segment_tag": segment,
                    "language": "ko",
                    "_meta_quality_tier": tier,
                    "source": str(path),
                    "input": inp,
                    "output": out,
                }
            )

    rng = random.Random(int(seed))
    rng.shuffle(rows)
    return rows[: int(limit)]


def filter_local_code_rows(rows: List[Dict], limit: int) -> List[Dict]:
    out: List[Dict] = []
    for r in rows:
        inp = str(r.get("input", ""))
        out_txt = str(r.get("output", ""))
        text = f"{inp}\n{out_txt}"
        if "```" not in text and not re.search(r"\b(def|class|import|try:|except|return)\b", text):
            continue
        r2 = dict(r)
        r2["segment_tag"] = "code"
        r2["task_type"] = "code"
        out.append(r2)
        if len(out) >= int(limit):
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean MIT Korean quality dataset.")
    parser.add_argument("--out", default="data/mit_ko_quality_v1.jsonl")
    parser.add_argument("--kovast_limit", type=int, default=90000)
    parser.add_argument("--kocommercial_limit", type=int, default=60000)
    parser.add_argument("--won_limit", type=int, default=45000)
    parser.add_argument("--anchor_path", default="data/chat_quality_master_v1.jsonl")
    parser.add_argument("--anchor_limit", type=int, default=2200)
    parser.add_argument("--capability_path", default="data/capability_pack_v2.jsonl")
    parser.add_argument("--capability_limit", type=int, default=2200)
    parser.add_argument("--local_code_path", default="data/ko_chat_code_balanced_v2.jsonl")
    parser.add_argument("--local_code_limit", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = int(args.seed)
    rows: List[Dict] = []

    kovast = collect_kovast(limit_pairs=int(args.kovast_limit), seed=seed + 11)
    kocom = collect_ko_commercial(limit_pairs=int(args.kocommercial_limit), seed=seed + 23)
    won = collect_won_instruct(limit_pairs=int(args.won_limit), seed=seed + 37)

    anchors = load_local_rows(path=Path(args.anchor_path), limit=int(args.anchor_limit), seed=seed + 41)
    capabilities = load_local_rows(path=Path(args.capability_path), limit=int(args.capability_limit), seed=seed + 47)
    local_code_raw = load_local_rows(path=Path(args.local_code_path), limit=int(args.local_code_limit) * 3, seed=seed + 59)
    local_code = filter_local_code_rows(local_code_raw, limit=int(args.local_code_limit))

    rows.extend(kovast)
    rows.extend(kocom)
    rows.extend(won)
    rows.extend(anchors)
    rows.extend(capabilities)
    rows.extend(local_code)

    rows = dedupe_rows(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "out": str(out_path),
        "rows": len(rows),
        "source_counts": {
            "maywell/koVast": len(kovast),
            "MarkrAI/KoCommercial-Dataset": len(kocom),
            "KRX-Data/Won-Instruct": len(won),
            "anchors": len(anchors),
            "capabilities": len(capabilities),
            "local_code": len(local_code),
        },
    }
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
