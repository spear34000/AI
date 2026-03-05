from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SPACE_RE = re.compile(r"\s+")
DEF_QUERY_RE = re.compile(
    r"(\uc774\ub780\??$|\ub780\??$|\ubb34\uc5c7|\ubb50\uc57c|\uc815\uc758|\uc124\uba85\ud574|\uc54c\ub824\uc918|\uc18c\uac1c\ud574|\ud575\uc2ec\ub9cc|\uac04\ub2e8\ud788|\uc27d\uac8c|\uc9e7\uac8c|\ud55c\s*\uc904\ub85c)"
)
TOKEN_RE = re.compile(r"[A-Za-z0-9\uac00-\ud7a3][A-Za-z0-9\uac00-\ud7a3+#._-]{1,31}")
VALID_TERM_RE = re.compile(r"^[A-Za-z0-9\uac00-\ud7a3][A-Za-z0-9\uac00-\ud7a3+#._ -]{1,31}$")
Q_KEYS: Tuple[str, ...] = ("input", "instruction", "prompt", "question", "context")
A_KEYS: Tuple[str, ...] = ("output", "response", "answer", "completion", "target")
SOURCE_PRIORITY = {
    "deintro_focus_v1.jsonl": 100,
    "term_focus_clean_v1.jsonl": 95,
    "term_anchor_patch_v2.jsonl": 90,
    "ko_def_grounding_patch_v1.jsonl": 86,
    "ko_targeted_shortanswer_v3.jsonl": 84,
    "clean_mix_v3.jsonl": 80,
    "serious_defs_boost_v1.jsonl": 76,
}
LEADING_FILLERS: Tuple[str, ...] = (
    "\ud55c\uad6d\uc5b4\ub85c",
    "\uac04\ub2e8\ud788",
    "\uc27d\uac8c",
    "\uc9e7\uac8c",
    "\ud575\uc2ec\ub9cc",
    "\uae30\ubcf8",
    "\uac1c\ub150",
    "\uc815\uc758\ub9cc",
    "\ud55c \uc904\ub85c",
    "\ud55c\uc904\ub85c",
)
TERM_STOPWORDS = {
    "\ud55c\uad6d\uc5b4\ub85c",
    "\uac04\ub2e8\ud788",
    "\uc27d\uac8c",
    "\uc9e7\uac8c",
    "\ud575\uc2ec\ub9cc",
    "\uae30\ubcf8",
    "\uac1c\ub150",
    "\uc815\uc758",
    "\uc124\uba85",
    "\uc124\uba85\ud574",
    "\uc54c\ub824\uc918",
    "\uc18c\uac1c\ud574",
    "\ud55c",
    "\uc904\ub85c",
    "\ud55c\uc904\ub85c",
    "\ubb34\uc5c7",
    "\ubb50\uc57c",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a clean local definition tool cache from JSONL datasets.")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out_jsonl", type=Path, default=Path("data/tool_knowledge_cache_v3_clean.jsonl"))
    p.add_argument("--min_answer_chars", type=int, default=18)
    p.add_argument("--max_answer_chars", type=int, default=260)
    return p.parse_args()


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def clean_char_ratio(text: str) -> float:
    src = str(text or "")
    if not src:
        return 0.0
    good = len(re.findall(r"[A-Za-z0-9\uac00-\ud7a3\s\.,:;()/%+\-\"'`]", src))
    return float(good) / float(max(1, len(src)))


def source_priority(source: str) -> int:
    name = Path(str(source or "")).name.lower()
    return int(SOURCE_PRIORITY.get(name, 0))


def pick_text(row: Dict, keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        s = normalize_space(str(value))
        if s:
            return s
    return ""


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = str(line).strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def normalize_term(term: str) -> str:
    t = normalize_space(term)
    if not t:
        return ""
    t = t.strip("\"'`“”‘’")
    suffixes = (
        "\uc774\ub780?",
        "\ub780?",
        "\uc774\ub780",
        "\ub780",
        "\uc815\uc758",
        "\uc124\uba85\ud574",
        "\uc124\uba85\ud574\uc918",
        "\ubb34\uc5c7\uc778\uc9c0 \uc54c\ub824\uc918",
        "\ubb34\uc5c7\uc778\uc9c0",
        "\ubb50\uc57c",
        "\uac00 \ubb50\uc57c?",
        "\uc774 \ubb50\uc57c?",
        "\uac00 \ubb50\uc57c",
        "\uc774 \ubb50\uc57c",
    )
    for suffix in suffixes:
        if len(t) > len(suffix) + 1 and t.endswith(suffix):
            t = t[: -len(suffix)].strip()
            break
    t = t.rstrip(" ?!.")
    return t


def strip_fillers(text: str) -> str:
    s = normalize_space(text)
    if not s:
        return ""
    changed = True
    while changed:
        changed = False
        for prefix in LEADING_FILLERS:
            if s.startswith(prefix + " "):
                s = s[len(prefix) :].strip()
                changed = True
    return s


def term_is_usable(term: str) -> bool:
    t = normalize_space(term)
    if len(t) < 2 or len(t) > 32:
        return False
    if t.lower() in TERM_STOPWORDS:
        return False
    if t.count(" ") > 2:
        return False
    if "?" in t or "\"" in t or "'" in t:
        return False
    if not VALID_TERM_RE.match(t):
        return False
    if clean_char_ratio(t) < 0.96:
        return False
    return True


def extract_term(question: str) -> str:
    q = strip_fillers(question)
    if not DEF_QUERY_RE.search(q):
        return ""
    q = q.rstrip(" ?!.")
    q = normalize_term(q)
    toks = [tok for tok in TOKEN_RE.findall(q) if tok.lower() not in TERM_STOPWORDS]
    if not toks:
        return ""
    term = normalize_term(toks[-1])
    if term_is_usable(term):
        return term
    return ""


def answer_is_usable(answer: str, min_chars: int, max_chars: int) -> bool:
    s = normalize_space(answer)
    if len(s) < int(min_chars) or len(s) > int(max_chars):
        return False
    if "\ufffd" in s:
        return False
    if s.count("?") >= max(3, len(s) // 18):
        return False
    if clean_char_ratio(s) < 0.86:
        return False
    return True


def row_score(term: str, answer: str, source: str) -> Tuple[int, int, int]:
    return (
        int(source_priority(source)),
        -abs(len(term) - 6),
        -len(answer),
    )


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best: Dict[str, Tuple[Tuple[int, int, int], Dict]] = {}
    for raw in args.inputs:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"input not found: {path}")
        for row in iter_jsonl(path):
            q = pick_text(row, Q_KEYS)
            a = pick_text(row, A_KEYS)
            if not q or not a:
                continue
            term = extract_term(q)
            if not term_is_usable(term):
                continue
            if not answer_is_usable(a, min_chars=int(args.min_answer_chars), max_chars=int(args.max_answer_chars)):
                continue
            source = normalize_space(str(row.get("_meta_source_file", path.as_posix())))
            cur = {
                "term": term,
                "answer": normalize_space(a),
                "lang": "ko",
                "source": source,
                "source_url": "",
            }
            score = row_score(term=term, answer=cur["answer"], source=source)
            prev = best.get(term)
            if prev is None or score > prev[0]:
                best[term] = (score, cur)

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for term in sorted(best.keys()):
            f.write(json.dumps(best[term][1], ensure_ascii=False) + "\n")

    print(json.dumps({"out_jsonl": str(out_path), "terms": len(best)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
