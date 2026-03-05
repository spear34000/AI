from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SPACE_RE = re.compile(r"\s+")
HANGUL_RE = re.compile(r"[가-힣]")
TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]{2,}")
REPEAT_RE = re.compile(r"(.)\1{8,}")
DEF_QUERY_RE = re.compile(r"(이란\??$|란\??$|무엇|뭐야|정의|설명해)")

Q_KEYS: Tuple[str, ...] = ("input", "instruction", "prompt", "question", "context")
A_KEYS: Tuple[str, ...] = ("output", "response", "answer", "completion", "target")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Korean tool-use curriculum dataset from downloaded/public datasets.")
    p.add_argument("--ko_source", type=Path, default=Path("data/quality/ko_clean_nometa_v1.jsonl"))
    p.add_argument("--hf_source", type=Path, default=Path("data/hf_mit_language_superpack_v1.jsonl"))
    p.add_argument("--out_jsonl", type=Path, default=Path("data/ko_agent_tool_dataset_v1.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/ko_agent_tool_dataset_v1.manifest.json"))
    p.add_argument("--max_ko_direct_rows", type=int, default=80000)
    p.add_argument("--max_hf_ko_direct_rows", type=int, default=12000)
    p.add_argument("--max_def_rows", type=int, default=50000)
    p.add_argument("--max_calc_rows", type=int, default=20000)
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def hangul_count(text: str) -> int:
    return len(HANGUL_RE.findall(str(text or "")))


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


def best_term_similarity(term: str, text: str) -> float:
    t = str(term or "").strip().lower()
    toks = TOKEN_RE.findall(str(text or "").lower())
    if not t or not toks:
        return 0.0
    best = 0.0
    for tok in toks:
        a = t
        b = tok
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i]
            for j, cb in enumerate(b, start=1):
                cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
            prev = cur
        d = prev[-1]
        sim = 1.0 - float(d) / float(max(len(a), len(b), 1))
        if sim > best:
            best = sim
    return max(0.0, min(1.0, best))


def is_definition_pair(inp: str, out: str) -> Tuple[bool, str]:
    q = normalize_space(inp)
    a = normalize_space(out)
    if not DEF_QUERY_RE.search(q):
        return False, ""
    term = extract_definition_term(q)
    if len(term) < 2:
        return False, ""
    low_a = a.lower()
    if term in low_a:
        return True, term
    if best_term_similarity(term, low_a) >= 0.86:
        return True, term
    return False, term


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


def format_agent_prompt(question: str, observation: str = "") -> str:
    q = normalize_space(question)
    obs = normalize_space(observation)
    head = (
        "도구:\n"
        "- search(term): 개념, 서비스, 제품, 인물, 장소 정보를 찾는다.\n"
        "- calculator(expr): 수식 계산을 한다.\n\n"
        f"질문: {q}\n"
    )
    if obs:
        return (
            head
            + f"{obs}\n"
            + "출력 형식: FINAL <답>\n"
            + "한 줄만 출력하라."
        )
    return (
        head
        + "출력 형식:\n"
        + "- ACTION search <검색어>\n"
        + "- ACTION calculator <식>\n"
        + "- FINAL <답>\n"
        + "한 줄만 출력하라."
    )


def make_row(inp: str, out: str, source_tag: str) -> Dict[str, str]:
    return {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": "high",
        "source": "build_ko_agent_tool_dataset_v1",
        "source_dataset": source_tag,
        "input": normalize_space(inp),
        "output": normalize_space(out),
    }


def dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def _safe_eval_numeric_expr(expr: str) -> float | None:
    if not re.fullmatch(r"[0-9\.\+\-\*/\(\)\s]+", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def _fmt_num(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def build_calc_examples(question: str) -> List[Tuple[str, str]]:
    q = normalize_space(question)
    out: List[Tuple[str, str]] = []

    expr_m = re.search(r"([0-9\.\+\-\*/\(\)\s]{3,})", q)
    if expr_m and re.search(r"\d+\s*[\+\-\*/]\s*\d+", q):
        expr = expr_m.group(1).strip()
        val = _safe_eval_numeric_expr(expr)
        if val is not None:
            out.append((format_agent_prompt(q), f"ACTION calculator {expr}"))
            obs = f"ACTION calculator {expr}\nOBSERVATION {_fmt_num(val)}"
            out.append((format_agent_prompt(q, observation=obs), f"FINAL {_fmt_num(val)}입니다."))
            return out

    ratio_m = re.search(
        r"([A-Za-z가-힣]+)\s*과\s*([A-Za-z가-힣]+)\s*(?:의\s*)?비율(?:이|은|는)?\s*([0-9]+(?:\.[0-9]+)?)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        q,
    )
    given_m = re.search(r"([A-Za-z가-힣]+)\s*가\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z가-힣%]*)\s*이면", q)
    ask_m = re.search(r"(?:이면|일때|일 때)\s*([A-Za-z가-힣]+)\s*(?:은|는)\s*\??\s*$", q)
    if ratio_m and given_m and ask_m:
        a = ratio_m.group(1)
        b = ratio_m.group(2)
        ra = float(ratio_m.group(3))
        rb = float(ratio_m.group(4))
        given_name = given_m.group(1)
        given_val = float(given_m.group(2))
        unit = given_m.group(3)
        ask_name = ask_m.group(1)
        coeff = {a: ra, b: rb}
        if given_name in coeff and ask_name in coeff and coeff[given_name] > 0:
            expr = f"{_fmt_num(given_val)}*{_fmt_num(coeff[ask_name])}/{_fmt_num(coeff[given_name])}"
            ans = given_val * coeff[ask_name] / coeff[given_name]
            out.append((format_agent_prompt(q), f"ACTION calculator {expr}"))
            obs = f"ACTION calculator {expr}\nOBSERVATION {_fmt_num(ans)}"
            out.append((format_agent_prompt(q, observation=obs), f"FINAL {ask_name}은 {_fmt_num(ans)}{unit}입니다."))
            return out

    return out


def is_good_row(inp: str, out: str) -> bool:
    q = normalize_space(inp)
    a = normalize_space(out)
    if not q or not a:
        return False
    if len(q) < 2 or len(q) > 1200:
        return False
    if len(a) < 4 or len(a) > 1800:
        return False
    if hangul_count(f"{q}\n{a}") < 4:
        return False
    if REPEAT_RE.search(a):
        return False
    return True


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, str]] = []
    reasons = Counter()
    seen: set[bytes] = set()

    def _append(inp: str, out: str, source_tag: str) -> None:
        key = dedupe_key(inp, out)
        if key in seen:
            reasons["duplicate"] += 1
            return
        seen.add(key)
        rows.append(make_row(inp, out, source_tag=source_tag))

    def_count = 0
    calc_count = 0
    ko_direct = 0
    hf_direct = 0

    for row in iter_jsonl(Path(args.ko_source)):
        inp = pick_text(row, Q_KEYS)
        out = pick_text(row, A_KEYS)
        if not is_good_row(inp, out):
            reasons["bad_ko_row"] += 1
            continue

        is_def, term = is_definition_pair(inp, out)
        if is_def and def_count < int(args.max_def_rows):
            obs_text = trim_to_first_sentence(out, max_chars=220)
            _append(format_agent_prompt(inp), f"ACTION search {term}", "ko_tool_action_def")
            obs = f"ACTION search {term}\nOBSERVATION {obs_text}"
            _append(format_agent_prompt(inp, observation=obs), f"FINAL {normalize_space(out)}", "ko_tool_final_def")
            def_count += 1
            continue

        calc_examples = build_calc_examples(inp)
        if calc_examples and calc_count < int(args.max_calc_rows):
            for ex_inp, ex_out in calc_examples:
                _append(ex_inp, ex_out, "ko_tool_calc")
            calc_count += 1
            continue

        if ko_direct < int(args.max_ko_direct_rows):
            _append(format_agent_prompt(inp), f"FINAL {normalize_space(out)}", "ko_direct_final")
            ko_direct += 1

    for row in iter_jsonl(Path(args.hf_source)):
        inp = pick_text(row, Q_KEYS)
        out = pick_text(row, A_KEYS)
        if str(row.get("language", "")).strip().lower() != "ko":
            continue
        if not is_good_row(inp, out):
            reasons["bad_hf_row"] += 1
            continue
        if hf_direct >= int(args.max_hf_ko_direct_rows):
            break
        _append(format_agent_prompt(inp), f"FINAL {normalize_space(out)}", "hf_ko_direct_final")
        hf_direct += 1

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": now_iso(),
        "out_jsonl": str(out_path),
        "rows_total": len(rows),
        "counts": {
            "definition_action_pairs": def_count,
            "calculator_pairs": calc_count,
            "ko_direct_final": ko_direct,
            "hf_ko_direct_final": hf_direct,
        },
        "drop_reasons": dict(reasons),
        "sources": {
            "ko_source": str(args.ko_source),
            "hf_source": str(args.hf_source),
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
