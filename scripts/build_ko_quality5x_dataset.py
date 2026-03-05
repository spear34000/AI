from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


SPACE_RE = re.compile(r"\s+")
HANGUL_RE = re.compile(r"[가-힣]")
LETTER_RE = re.compile(r"[A-Za-z가-힣]")
TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]{2,}")
LONG_REPEAT_RE = re.compile(r"(.)\1{8,}")
LEAK_RE = re.compile(r"(###\s*Instruction|###\s*Response|\[INST\]|\[/INST\]|<\|assistant\|>|<\|user\|>)", re.IGNORECASE)
CODE_RE = re.compile(
    r"(```|^\s*def\s|^\s*class\s|^\s*import\s|public\s+static\s+void|SELECT\s+.+\s+FROM|<html|console\.log\(|function\s+\w+\()",
    re.IGNORECASE | re.MULTILINE,
)
INTRO_QUERY_RE = re.compile(
    r"(자기소개|소개해|너는\s*누구|네?\s*이름|모델\s*이름|본인\s*소개|정체가\s*뭐|who\s+are\s+you|model\s+name)",
    re.IGNORECASE,
)
INTRO_ANSWER_RE = re.compile(
    r"(반갑습니다|안녕하세요|질문\s*의도|저는\s+.+(ai|어시스턴트|도우미|모델)|i\s+am\s+.+(ai|assistant|model))",
    re.IGNORECASE,
)
GENERIC_META_RE = re.compile(
    r"(원하면\s+더\s+짧게|원하면\s+3줄|필요하면\s+더\s+자세히|요청\s+형식에\s+맞춰|핵심만\s+전달)",
    re.IGNORECASE,
)
DEF_QUERY_RE = re.compile(r"(이란|란\??$|무엇|뭐야|정의|설명해)")
LOGIC_HINT_RE = re.compile(r"(논리|추론|이유|참|거짓|모순|정답)")
MATH_HINT_RE = re.compile(r"(\d+\s*[\+\-\*\/]\s*\d+|얼마|계산|수식)")
SYLLOGISM_HINT_RE = re.compile(r"(모든\s+\S+\s+는\s+\S+|일부\s+\S+\s+는\s+\S+|거짓말쟁이|진실만\s+말)")


DEFAULT_INCLUDE_FIELDS = ("ko_chat_general", "ko_fluency_targeted", "reasoning_logic")
DEFAULT_EXCLUDE_PATH_TOKENS = (
    "intro_override",
    "chat_quality_master",
    "identity",
    "persona",
    "routing",
    "continual",
    "memory",
    "specbook",
    "ccl_",
    "translation",
    "turbo_boost_ko_code",
    "code_balanced",
    "ko_ultra_targeted",
)

STOPWORDS = {
    "한국어",
    "한국어로",
    "간단히",
    "짧게",
    "핵심",
    "설명",
    "설명해줘",
    "말해줘",
    "알려줘",
    "답해줘",
    "부탁해",
    "부탁",
    "요약",
    "한줄",
    "한",
    "줄",
    "해주세요",
    "해줘",
    "이란",
    "무엇",
    "뭐야",
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a strict Korean clean dataset for 5x sentence-quality target.")
    p.add_argument("--index_json", type=Path, default=Path("data/by_field/dataset_field_index_v1.json"))
    p.add_argument("--data_root", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("data/ko_quality5x_clean_v1.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/ko_quality5x_clean_v1.manifest.json"))
    p.add_argument("--include_fields", default=",".join(DEFAULT_INCLUDE_FIELDS))
    p.add_argument("--exclude_path_tokens", default=",".join(DEFAULT_EXCLUDE_PATH_TOKENS))
    p.add_argument("--min_input_chars", type=int, default=2)
    p.add_argument("--max_input_chars", type=int, default=900)
    p.add_argument("--min_output_chars", type=int, default=12)
    p.add_argument("--max_output_chars", type=int, default=1100)
    p.add_argument("--min_hangul_chars", type=int, default=6)
    p.add_argument("--min_hangul_ratio", type=float, default=0.30)
    p.add_argument("--min_quality_score", type=float, default=1.10)
    p.add_argument("--max_rows_per_file", type=int, default=45000)
    p.add_argument("--max_total_rows", type=int, default=260000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def pick_text(row: Dict, keys: Sequence[str]) -> str:
    for key in keys:
        v = row.get(key)
        if v is None:
            continue
        s = normalize_space(v)
        if s:
            return s
    return ""


def hangul_count(text: str) -> int:
    return len(HANGUL_RE.findall(str(text or "")))


def hangul_ratio(text: str) -> float:
    src = str(text or "")
    letters = LETTER_RE.findall(src)
    if not letters:
        return 0.0
    ko = HANGUL_RE.findall(src)
    return float(len(ko)) / float(max(1, len(letters)))


def is_code_like(text: str) -> bool:
    return bool(CODE_RE.search(str(text or "")))


def normalize_token(token: str) -> str:
    t = str(token or "").strip().lower()
    suffixes = (
        "인가요",
        "인가",
        "이란",
        "란",
        "입니다",
        "이에요",
        "예요",
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


def keyword_terms(text: str) -> Set[str]:
    out: Set[str] = set()
    for tok in TOKEN_RE.findall(str(text or "").lower()):
        n = normalize_token(tok)
        if len(n) < 2:
            continue
        if n in STOPWORDS:
            continue
        out.add(n)
    return out


def dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


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


def path_is_excluded(path_text: str, tokens: Sequence[str]) -> bool:
    low = str(path_text or "").replace("\\", "/").lower()
    return any(t in low for t in tokens)


def parse_csv_set(text: str) -> Set[str]:
    return {x.strip() for x in str(text or "").split(",") if x.strip()}


def tier_bonus(tier: str) -> float:
    t = str(tier or "").strip().lower()
    if t == "high":
        return 0.45
    if t == "mid":
        return 0.20
    return 0.0


def keep_row(
    inp: str,
    out: str,
    tier: str,
    min_input_chars: int,
    max_input_chars: int,
    min_output_chars: int,
    max_output_chars: int,
    min_hangul_chars: int,
    min_hangul_ratio: float,
    min_quality_score: float,
) -> Tuple[bool, str, float]:
    q = normalize_space(inp)
    a = normalize_space(out)
    if not q or not a:
        return False, "missing_pair", 0.0
    if len(q) < int(min_input_chars) or len(q) > int(max_input_chars):
        return False, "input_length", 0.0
    if len(a) < int(min_output_chars) or len(a) > int(max_output_chars):
        return False, "output_length", 0.0

    merged = f"{q}\n{a}"
    if "\ufffd" in merged or "???" in merged:
        return False, "encoding_noise", 0.0
    if LEAK_RE.search(merged):
        return False, "template_leak", 0.0
    if LONG_REPEAT_RE.search(merged):
        return False, "char_repeat", 0.0
    if is_code_like(merged):
        return False, "code_like", 0.0

    if INTRO_QUERY_RE.search(q):
        return False, "intro_query", 0.0
    if INTRO_ANSWER_RE.search(a):
        return False, "intro_answer", 0.0
    if GENERIC_META_RE.search(a):
        return False, "meta_style", 0.0

    h_count = hangul_count(merged)
    h_ratio = hangul_ratio(merged)
    if h_count < int(min_hangul_chars) or h_ratio < float(min_hangul_ratio):
        return False, "ko_weak", 0.0

    q_tokens = keyword_terms(q)
    a_tokens = keyword_terms(a)
    inter = q_tokens.intersection(a_tokens)
    q_is_logic = bool(LOGIC_HINT_RE.search(q))
    q_is_math = bool(MATH_HINT_RE.search(q))
    q_is_syllogism = bool(SYLLOGISM_HINT_RE.search(q))
    q_is_def = bool(DEF_QUERY_RE.search(q))

    if q_tokens and len(inter) == 0 and not (q_is_logic or q_is_math or q_is_syllogism):
        # Definition prompts should preserve at least one key term.
        if q_is_def or len(q_tokens) <= 4:
            return False, "keyword_miss", 0.0

    if q_is_math and not re.search(r"\d", a):
        return False, "math_without_digits", 0.0
    if q_is_syllogism and not re.search(r"(아니|불가|보장|결론|따라서|참|거짓)", a):
        return False, "syllogism_weak", 0.0

    out_tokens = TOKEN_RE.findall(a.lower())
    if len(out_tokens) >= 14:
        div = len(set(out_tokens)) / float(max(1, len(out_tokens)))
        if div < 0.32:
            return False, "low_diversity", 0.0

    score = 0.0
    score += min(len(a), 260) / 260.0
    score += 0.9 * h_ratio
    if q_tokens:
        score += 1.1 * (len(inter) / float(max(1, len(q_tokens))))
    score += tier_bonus(tier)

    if q_is_logic and re.search(r"(왜냐|따라서|그러므로|즉|결론)", a):
        score += 0.30
    if q_is_math and re.search(r"\d", a):
        score += 0.22
    if q_is_syllogism and re.search(r"(아니|보장|결론|따라서|즉)", a):
        score += 0.22

    if score < float(min_quality_score):
        return False, "quality_score", score
    return True, "ok", score


def load_candidates(index_json: Path, include_fields: Set[str], exclude_path_tokens: Sequence[str]) -> List[Dict]:
    raw = json.loads(index_json.read_text(encoding="utf-8"))
    datasets = raw.get("datasets", [])
    if not isinstance(datasets, list):
        return []
    out: List[Dict] = []
    for row in datasets:
        if not isinstance(row, dict):
            continue
        field = str(row.get("field", "")).strip()
        rel_path = str(row.get("path", "")).strip().replace("\\", "/")
        if not field or not rel_path:
            continue
        if field not in include_fields:
            continue
        if path_is_excluded(rel_path, exclude_path_tokens):
            continue
        out.append(
            {
                "field": field,
                "path": rel_path,
                "confidence": float(row.get("confidence", 0.0)),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))

    index_json = Path(args.index_json)
    data_root = Path(args.data_root)
    out_path = Path(args.out)
    manifest_path = Path(args.manifest)
    if not index_json.exists():
        raise FileNotFoundError(f"index not found: {index_json}")
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    include_fields = parse_csv_set(args.include_fields)
    exclude_path_tokens = sorted(parse_csv_set(args.exclude_path_tokens))
    candidates = load_candidates(index_json=index_json, include_fields=include_fields, exclude_path_tokens=exclude_path_tokens)

    if not candidates:
        raise RuntimeError("no candidate datasets found after field/path filtering")

    all_rows: List[Dict] = []
    rejects = Counter()
    seen = set()
    per_file_kept = Counter()
    per_field_kept = Counter()
    per_file_scores: Dict[str, List[float]] = defaultdict(list)

    for item in candidates:
        rel_path = str(item["path"])
        field = str(item["field"])
        source_path = data_root / Path(rel_path)
        if not source_path.exists() or not source_path.is_file():
            rejects["missing_file"] += 1
            continue

        rows_local: List[Dict] = []
        for row in iter_jsonl(source_path):
            inp = pick_text(row, ("input", "instruction", "prompt", "question", "context"))
            out = pick_text(row, ("output", "response", "answer", "completion", "target"))
            tier = str(row.get("_meta_quality_tier", "mid")).strip().lower()
            ok, reason, score = keep_row(
                inp=inp,
                out=out,
                tier=tier,
                min_input_chars=int(args.min_input_chars),
                max_input_chars=int(args.max_input_chars),
                min_output_chars=int(args.min_output_chars),
                max_output_chars=int(args.max_output_chars),
                min_hangul_chars=int(args.min_hangul_chars),
                min_hangul_ratio=float(args.min_hangul_ratio),
                min_quality_score=float(args.min_quality_score),
            )
            if not ok:
                rejects[reason] += 1
                continue

            key = dedupe_key(inp, out)
            if key in seen:
                rejects["duplicate"] += 1
                continue
            seen.add(key)

            cleaned = {
                "task_type": "korean",
                "segment_tag": "ko",
                "language": "ko",
                "_meta_quality_tier": "high" if tier == "high" else "mid",
                "input": normalize_space(inp),
                "output": normalize_space(out),
                "_meta_source_file": rel_path,
                "_meta_field": field,
                "_meta_quality_score": round(float(score), 6),
            }
            rows_local.append(cleaned)
            per_file_scores[rel_path].append(float(score))

        rng.shuffle(rows_local)
        capped = rows_local[: max(0, int(args.max_rows_per_file))]
        all_rows.extend(capped)
        per_file_kept[rel_path] += len(capped)
        per_field_kept[field] += len(capped)

    rng.shuffle(all_rows)
    max_total = max(0, int(args.max_total_rows))
    if max_total > 0:
        all_rows = all_rows[:max_total]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wf:
        for r in all_rows:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg_score_by_file = {
        k: (sum(v) / float(len(v)) if v else 0.0) for k, v in per_file_scores.items()
    }
    avg_score_by_file = dict(sorted(avg_score_by_file.items(), key=lambda kv: kv[1], reverse=True))

    manifest = {
        "generated_at": now_iso(),
        "out": str(out_path).replace("\\", "/"),
        "rows": len(all_rows),
        "seed": int(args.seed),
        "include_fields": sorted(include_fields),
        "exclude_path_tokens": exclude_path_tokens,
        "limits": {
            "max_rows_per_file": int(args.max_rows_per_file),
            "max_total_rows": int(args.max_total_rows),
            "min_input_chars": int(args.min_input_chars),
            "max_input_chars": int(args.max_input_chars),
            "min_output_chars": int(args.min_output_chars),
            "max_output_chars": int(args.max_output_chars),
            "min_hangul_chars": int(args.min_hangul_chars),
            "min_hangul_ratio": float(args.min_hangul_ratio),
            "min_quality_score": float(args.min_quality_score),
        },
        "candidate_files": [str(c["path"]) for c in candidates],
        "kept_per_field": dict(sorted(per_field_kept.items(), key=lambda kv: kv[0])),
        "kept_per_file": dict(sorted(per_file_kept.items(), key=lambda kv: kv[1], reverse=True)),
        "avg_quality_score_per_file": avg_score_by_file,
        "rejects": dict(rejects),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"out": str(out_path), "rows": len(all_rows), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
