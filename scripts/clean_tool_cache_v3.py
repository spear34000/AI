from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple


SPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
LATIN_RE = re.compile(r"[A-Za-z]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean noisy local definition cache into a stricter usable cache.")
    p.add_argument("--input", type=Path, default=Path("data/tool_knowledge_cache_final_v1.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/tool_knowledge_cache_v3_clean.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/tool_knowledge_cache_v3_clean.manifest.json"))
    return p.parse_args()


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def normalize_ko_token(token: str) -> str:
    t = str(token or "").strip().lower()
    if not t:
        return ""
    suffixes = [
        "인가요",
        "인가",
        "이란",
        "란",
        "입니다",
        "이야",
        "야",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "에",
        "로",
        "도",
        "만",
    ]
    for s in suffixes:
        if len(t) > len(s) + 1 and t.endswith(s):
            t = t[: -len(s)]
            break
    return t


def trim_to_first_sentence(text: str, max_chars: int = 220) -> str:
    s = normalize_space(text)
    if not s:
        return s

    def _first_sentence(src: str) -> str:
        end_local = re.search(r"[.!?]\s|[.!?]$", src)
        if not end_local:
            return src.strip()
        return src[: end_local.end()].strip()

    if len(s) <= int(max_chars):
        cut = _first_sentence(s)
        return cut if len(cut) >= 4 else s
    head = s[: int(max_chars)]
    end = re.search(r"[.!?]\s|[.!?]$", head)
    if end:
        cut = head[: end.end()].strip()
        if len(cut) >= 4:
            return cut
    return head.strip()


def normalize_term_key(term: str) -> str:
    t = normalize_ko_token(term)
    return t if t else normalize_space(term).lower()


def term_is_usable(term: str) -> bool:
    t = normalize_space(term)
    if len(t) < 2 or len(t) > 48:
        return False
    if re.fullmatch(r"0x[0-9a-f]+", t.lower()):
        return False
    if re.fullmatch(r"[\W_]+", t):
        return False
    if re.fullmatch(r"[0-9.\-_/]+", t):
        return False
    if len(HANGUL_RE.findall(t)) == 0 and len(LATIN_RE.findall(t)) == 0:
        return False
    if len(LATIN_RE.findall(t)) > 0 and len(HANGUL_RE.findall(t)) == 0 and len(t) < 3:
        return False
    return True


def clean_answer(term: str, answer: str) -> str:
    s = normalize_space(HTML_TAG_RE.sub(" ", str(answer or "")))
    s = s.replace(" ,", ",").replace(" .", ".")
    if not s:
        return ""
    if "\ufffd" in s:
        return ""
    if s.count("?") >= max(3, len(s) // 18):
        return ""
    s = trim_to_first_sentence(s, max_chars=220)
    if len(s) < 12:
        return ""
    if len(HANGUL_RE.findall(s)) < 2 and len(LATIN_RE.findall(s)) < 8:
        return ""
    return s


def answer_score(term_key: str, answer: str) -> Tuple[int, int, int]:
    low = str(answer or "").lower()
    contains = 1 if term_key and term_key in normalize_term_key(low) else 0
    hangul_n = len(HANGUL_RE.findall(answer))
    length_score = -abs(len(answer) - 80)
    return contains, hangul_n, length_score


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    kept: Dict[str, Dict] = {}
    stats = {"parsed": 0, "json_error": 0, "bad_term": 0, "bad_answer": 0, "replaced": 0}

    with args.input.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                stats["json_error"] += 1
                continue
            if not isinstance(row, dict):
                continue
            stats["parsed"] += 1
            term = normalize_space(row.get("term", ""))
            if not term_is_usable(term):
                stats["bad_term"] += 1
                continue
            answer = clean_answer(term=term, answer=str(row.get("answer", "")))
            if not answer:
                stats["bad_answer"] += 1
                continue
            key = normalize_term_key(term)
            cand = {
                "term": term,
                "answer": answer,
                "lang": str(row.get("lang", "ko") or "ko"),
                "source": str(row.get("source", "cache_clean_v3") or "cache_clean_v3"),
                "source_url": str(row.get("source_url", "") or ""),
            }
            prev = kept.get(key)
            if prev is None or answer_score(key, cand["answer"]) > answer_score(key, str(prev.get("answer", ""))):
                if prev is not None:
                    stats["replaced"] += 1
                kept[key] = cand

    rows = sorted(kept.values(), key=lambda r: normalize_term_key(str(r.get("term", ""))))
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "rows_total": int(len(rows)),
        "stats": {k: int(v) for k, v in stats.items()},
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
