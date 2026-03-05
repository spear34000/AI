from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]{2,}")
DEF_QUERY_RE = re.compile(r"(이란\??$|란\??$|무엇|뭐야|정의|설명해)")
Q_KEYS: Tuple[str, ...] = ("input", "instruction", "prompt", "question", "context")
A_KEYS: Tuple[str, ...] = ("output", "response", "answer", "completion", "target")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk build tool cache/training rows from definition terms via Wikipedia.")
    p.add_argument("--source_jsonl", type=Path, default=Path("data/quality/ko_clean_nometa_v1.jsonl"))
    p.add_argument("--cache_jsonl", type=Path, default=Path("data/tool_knowledge_cache_v1.jsonl"))
    p.add_argument("--train_jsonl", type=Path, default=Path("data/ko_agent_tool_wiki_v1.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/ko_agent_tool_wiki_v1.manifest.json"))
    p.add_argument("--max_terms", type=int, default=5000)
    p.add_argument("--sleep_ms", type=int, default=120)
    p.add_argument("--timeout_sec", type=float, default=5.0)
    return p.parse_args()


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


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
    if not path.exists():
        return
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


def normalize_ko_token(token: str) -> str:
    t = normalize_space(token).lower()
    if not t:
        return ""
    suffixes = (
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
    )
    for s in suffixes:
        if len(t) > len(s) + 1 and t.endswith(s):
            return t[: -len(s)]
    return t


def extract_definition_term(text: str) -> str:
    s = normalize_space(text)
    m = re.search(r"(.+?)(이란\??$|란\??$|무엇|뭐야|정의|설명해)", s)
    if m:
        s = normalize_space(m.group(1))
    toks = TOKEN_RE.findall(s)
    if not toks:
        return ""
    return normalize_ko_token(toks[-1])


def collect_terms(path: Path, max_terms: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for row in iter_jsonl(path):
        q = pick_text(row, Q_KEYS)
        if not DEF_QUERY_RE.search(q):
            continue
        term = extract_definition_term(q)
        if len(term) < 2:
            continue
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
        if len(out) >= int(max_terms):
            break
    return out


def http_json(url: str, timeout_sec: float) -> Dict | List | None:
    req = Request(url, headers={"User-Agent": "spear1.0-bulk-tool-cache/1.0"})
    try:
        with urlopen(req, timeout=max(1.0, float(timeout_sec))) as resp:
            raw = resp.read()
    except (HTTPError, URLError, TimeoutError, OSError):
        return None
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None


def wiki_search_title(term: str, lang: str, timeout_sec: float) -> str:
    url = (
        f"https://{lang}.wikipedia.org/w/api.php?action=opensearch&search={quote(term)}"
        "&limit=1&namespace=0&format=json"
    )
    payload = http_json(url, timeout_sec=float(timeout_sec))
    if not isinstance(payload, list) or len(payload) < 2:
        return ""
    titles = payload[1]
    if not isinstance(titles, list) or not titles:
        return ""
    return normalize_space(str(titles[0]))


def wiki_summary(title: str, lang: str, timeout_sec: float) -> Tuple[str, str]:
    if not title:
        return "", ""
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    payload = http_json(url, timeout_sec=float(timeout_sec))
    if not isinstance(payload, dict):
        return "", ""
    extract = normalize_space(str(payload.get("extract", "")))
    source_url = ""
    content_urls = payload.get("content_urls", {})
    if isinstance(content_urls, dict):
        desktop = content_urls.get("desktop", {})
        if isinstance(desktop, dict):
            source_url = normalize_space(str(desktop.get("page", "")))
    return extract, source_url


def trim_to_first_sentence(text: str, max_chars: int = 220) -> str:
    s = normalize_space(text)
    if not s:
        return ""
    if len(s) <= int(max_chars):
        return s
    m = re.search(r"[.!?]\s|[.!?]$", s[: int(max_chars)])
    if m:
        return s[: m.end()].strip()
    return s[: int(max_chars)].strip()


def has_final_consonant(ch: str) -> bool:
    cp = ord(ch)
    return 0xAC00 <= cp <= 0xD7A3 and ((cp - 0xAC00) % 28) != 0


def compose_answer(term: str, summary: str) -> str:
    t = normalize_space(term)
    s = trim_to_first_sentence(summary, max_chars=220)
    if not t or not s:
        return ""
    if t.lower() in s.lower():
        return s
    particle = "은" if has_final_consonant(t[-1]) else "는"
    return f"{t}{particle} {s}"


def format_obs_prompt(term: str, answer: str) -> str:
    return (
        "도구:\n"
        "- search(term): 개념, 서비스, 제품, 인물, 장소 정보를 찾는다.\n\n"
        f"질문: {term}이란?\n"
        f"ACTION search {term}\n"
        f"OBSERVATION {answer}\n"
        "출력 형식: FINAL <답>\n"
        "한 줄만 출력하라."
    )


def main() -> None:
    args = parse_args()
    terms = collect_terms(path=Path(args.source_jsonl), max_terms=int(args.max_terms))
    cache_rows: List[Dict[str, str]] = []
    train_rows: List[Dict[str, str]] = []
    stats = {"terms": len(terms), "resolved": 0, "failed": 0, "ko_hits": 0, "en_hits": 0}

    for idx, term in enumerate(terms, start=1):
        found = False
        for lang in ("ko", "en"):
            title = wiki_search_title(term=term, lang=lang, timeout_sec=float(args.timeout_sec))
            if not title:
                continue
            summary, source_url = wiki_summary(title=title, lang=lang, timeout_sec=float(args.timeout_sec))
            answer = compose_answer(term=term, summary=summary)
            if len(answer) < 20:
                continue
            cache_rows.append(
                {
                    "term": term,
                    "answer": answer,
                    "lang": lang,
                    "source": "wikipedia_summary",
                    "source_url": source_url,
                }
            )
            train_rows.append(
                {
                    "task_type": "korean",
                    "segment_tag": "ko",
                    "language": "ko",
                    "_meta_quality_tier": "high",
                    "source": "build_wiki_tool_cache_from_definitions_v1",
                    "source_dataset": "wiki_tool_final",
                    "input": format_obs_prompt(term=term, answer=answer),
                    "output": f"FINAL {answer}",
                }
            )
            stats["resolved"] += 1
            stats["ko_hits" if lang == "ko" else "en_hits"] += 1
            found = True
            break
        if not found:
            stats["failed"] += 1
        time.sleep(max(0, int(args.sleep_ms)) / 1000.0)

    cache_path = Path(args.cache_jsonl)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in cache_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_path = Path(args.train_jsonl)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "source_jsonl": str(args.source_jsonl),
        "cache_jsonl": str(cache_path),
        "train_jsonl": str(train_path),
        "stats": stats,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
