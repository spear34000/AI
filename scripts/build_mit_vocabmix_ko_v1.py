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
HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
LETTER_RE = re.compile(r"[A-Za-z\uac00-\ud7a3]")
TOKEN_RE = re.compile(r"[A-Za-z0-9\uac00-\ud7a3]{2,}")
REPEAT_RE = re.compile(r"(.)\1{8,}")
ASCII_LEAD_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_\-+.#]{2,}")
CODE_TOPIC_RE = re.compile(
    r"\b(?:javascript|typescript|python|java|react|node(?:\.js)?|sql|html|css|docker|kubernetes|git)\b",
    re.IGNORECASE,
)
CODE_SYMBOL_RE = re.compile(r"[{}\[\]<>;=()#`]")
CODE_LIKE_RE = re.compile(
    r"(^\s*(?:import|from|export|const|let|var|function|class)\b|```|</?\w+[^>]*>|^\s*#include\b|^\s*SELECT\b)",
    re.IGNORECASE | re.MULTILINE,
)

MIT_FILE_HINT_RE = re.compile(r"(?:^|[_\-/])(mit)(?:[_\-/]|$)", re.IGNORECASE)
INTRO_QUERY_RE = re.compile(r"(?:\uc790\uae30\s*\uc18c\uac1c|\uc18c\uac1c\ud574|\ub108\ub294\s*\ub204\uad6c|\ub124?\s*\uc774\ub984|\ubaa8\ub378\s*\uc774\ub984)", re.IGNORECASE)
DEF_QUERY_RE = re.compile(r"(?:\uc774\ub780\??$|\ub780\??$|\ubb34\uc5c7|\ubb50\uc57c|\uc815\uc758|\uc124\uba85\ud574)")
LOGIC_QUERY_RE = re.compile(r"(?:\ubaa8\ub4e0\s+\S+\s+\ub294\s+\S+|\uc77c\ubd80\s+\S+\s+\ub294\s+\S+|\uac70\uc9d3\ub9d0\uc7c1\uc774|\uc9c4\uc2e4\ub9cc\s+\ub9d0|\ub17c\ub9ac|\ucd94\ub860)")
LOGIC_ANSWER_RE = re.compile(r"(?:\uc774\uc720|\ub530\ub77c\uc11c|\uacb0\ub860|\ubcf4\uc7a5|\uc544\ub2c8|\uad6c\ubd84)")
INTRO_ANSWER_RE = re.compile(
    r"(?:\uc548\ub155\ud558\uc138\uc694|\ubc18\uac11\uc2b5\ub2c8\ub2e4|\uc800\ub294\s+.+(?:ai|\uc5b4\uc2dc\uc2a4\ud134\ud2b8|\ub3c4\uc6b0\ubbf8|\ubaa8\ub378))",
    re.IGNORECASE,
)

Q_KEYS: Tuple[str, ...] = ("input", "instruction", "prompt", "question", "context")
A_KEYS: Tuple[str, ...] = ("output", "response", "answer", "completion", "target")

DEFAULT_INCLUDE_FIELDS = ("foundation_mit_mix",)
DEFAULT_EXCLUDE_PATH_TOKENS = (
    "intro_override",
    "chat_name",
    "identity",
    "persona",
    "session_memory",
    "continual_buffer",
    "capability_pack",
    "router",
    "memory",
    "specbook",
    "ccl_",
    "stages_",
)
DEFAULT_BANNED_OUTPUT_SUBSTRINGS = (
    "\uc9c8\ubb38 \uc758\ub3c4\ub97c \ud30c\uc545\ud574 \ub2e8\uacc4\ubcc4\ub85c \uc815\ub9ac\ud574\ub4dc\ub9b4\uac8c\uc694",
    "\uc694\uccad \ud615\uc2dd\uc5d0 \ub9de\ucdb0 \ud575\uc2ec\ub9cc \uba85\ud655\ud558\uac8c \uc804\ub2ec\ud558\ub294 \ud55c\uad6d\uc5b4 ai",
    "\ud544\uc694\ud55c \uc815\ubcf4\ub97c \ube60\ub974\uac8c \uc815\ub9ac\ud574 \ub4dc\ub9ac\ub294 \ud55c\uad6d\uc5b4 ai",
    "\ud55c\uad6d\uc5b4 ai \uc5b4\uc2dc\uc2a4\ud134\ud2b8\uc785\ub2c8\ub2e4",
    "\ud55c \uc904\ub85c \ub9d0\ud558\uba74 \ubc18\uac11\uc2b5\ub2c8\ub2e4",
)

GRADE_WEIGHT = {"A": 1.0, "B": 0.7, "C": 0.45}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MIT-all + Korean vocab-focused pure-learning dataset.")
    p.add_argument("--data_root", type=Path, default=Path("data"))
    p.add_argument("--index_json", type=Path, default=Path("data/by_field/dataset_field_index_v1.json"))
    p.add_argument("--archive_root", type=Path, default=Path("data_archive_20260219_160805"))
    p.add_argument("--archive_glob", default="*mit*.jsonl")
    p.add_argument("--vocab_path", type=Path, default=Path("data/\ud55c\uad6d\uc5b4 \ud559\uc2b5\uc6a9 \uc5b4\ud718 \ubaa9\ub85d.txt"))
    p.add_argument("--out_jsonl", type=Path, default=Path("data/slm_mit_vocabmix_ko_v1.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/slm_mit_vocabmix_ko_v1.manifest.json"))
    p.add_argument("--sources_tsv", type=Path, default=Path("data/slm_mit_vocabmix_ko_v1.sources.tsv"))
    p.add_argument("--include_fields", default=",".join(DEFAULT_INCLUDE_FIELDS))
    p.add_argument("--exclude_path_tokens", default=",".join(DEFAULT_EXCLUDE_PATH_TOKENS))
    p.add_argument("--banned_output_substrings", default="||".join(DEFAULT_BANNED_OUTPUT_SUBSTRINGS))
    p.add_argument("--extra_inputs", default="", help="comma-separated file paths")
    p.add_argument("--max_rows_per_file", type=int, default=0, help="0 means scan all rows per file")
    p.add_argument("--max_kept_rows_per_file", type=int, default=70000, help="0 means no keep-cap per source file")
    p.add_argument("--max_rows_per_source_dataset", type=int, default=0, help="0 means no cap per source_dataset")
    p.add_argument("--max_total_rows", type=int, default=180000, help="0 means keep all accepted rows")
    p.add_argument("--target_ko_ratio", type=float, default=0.84)
    p.add_argument(
        "--strict_row_license",
        action="store_true",
        help="when set, keep only rows with explicit MIT license field",
    )
    p.add_argument("--add_vocab_rows", action="store_true")
    p.add_argument("--max_vocab_rows", type=int, default=5000)
    p.add_argument("--vocab_rank_max", type=int, default=6000, help="0 means no rank cap")
    p.add_argument("--min_input_chars", type=int, default=2)
    p.add_argument("--max_input_chars", type=int, default=1400)
    p.add_argument("--min_output_chars", type=int, default=10)
    p.add_argument("--max_output_chars", type=int, default=2200)
    p.add_argument("--min_hangul_chars", type=int, default=6)
    p.add_argument("--min_hangul_ratio", type=float, default=0.12)
    p.add_argument("--ko_min_output_hangul_ratio", type=float, default=0.28)
    p.add_argument("--ko_max_output_latin_ratio", type=float, default=0.60)
    p.add_argument("--ko_ascii_lead_max_hangul", type=int, default=4)
    p.add_argument("--max_ko_prefix_repeat", type=int, default=260)
    p.add_argument("--ko_vocab_boost", type=float, default=0.55)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def normalize_license(value: str) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def is_mit_license(value: str) -> bool:
    s = normalize_license(value)
    return s in {"mit", "mit license"}


def parse_csv_set(text: str) -> List[str]:
    out = [x.strip() for x in str(text or "").split(",") if x.strip()]
    return list(dict.fromkeys(out))


def parse_double_bar_list(text: str) -> List[str]:
    out = [x.strip() for x in str(text or "").split("||") if x.strip()]
    return list(dict.fromkeys(out))


def pick_text(row: Dict, keys: Sequence[str]) -> str:
    for key in keys:
        val = row.get(key)
        if val is None:
            continue
        s = normalize_space(str(val))
        if s:
            return s
    return ""


def iter_rows(path: Path, max_rows: int) -> Iterable[Dict]:
    if not path.exists():
        return
    lim = int(max_rows)
    seen_n = 0
    ext = path.suffix.lower()
    if ext == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if lim > 0 and seen_n >= lim:
                    break
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                seen_n += 1
                yield row
        return

    if ext == ".json":
        try:
            raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(raw, dict):
            yield raw
            return
        if isinstance(raw, list):
            for row in raw:
                if lim > 0 and seen_n >= lim:
                    break
                if not isinstance(row, dict):
                    continue
                seen_n += 1
                yield row


def hangul_ratio(text: str) -> float:
    letters = LETTER_RE.findall(str(text or ""))
    if not letters:
        return 0.0
    ko = HANGUL_RE.findall(str(text or ""))
    return float(len(ko)) / float(max(1, len(letters)))


def hangul_count(text: str) -> int:
    return len(HANGUL_RE.findall(str(text or "")))


def latin_ratio(text: str) -> float:
    src = str(text or "")
    letters = LETTER_RE.findall(src)
    if not letters:
        return 0.0
    latin = re.findall(r"[A-Za-z]", src)
    return float(len(latin)) / float(max(1, len(letters)))


def code_symbol_ratio(text: str) -> float:
    src = str(text or "")
    if not src:
        return 0.0
    symbols = CODE_SYMBOL_RE.findall(src)
    return float(len(symbols)) / float(max(1, len(src)))


def answer_prefix_key(text: str, max_tokens: int = 4) -> str:
    toks = re.findall(r"[A-Za-z0-9\uac00-\ud7a3]+", normalize_space(text).lower())
    if not toks:
        return ""
    return " ".join(toks[: max(1, int(max_tokens))])


def normalize_ko_token(token: str) -> str:
    t = normalize_space(token).strip(".,!?\"'`()[]{}")
    t = re.sub(r"\d+$", "", t)
    low = t.lower()
    suffixes = (
        "\uc778\uac00\uc694",
        "\uc778\uac00",
        "\uc774\ub780",
        "\ub780",
        "\uc785\ub2c8\ub2e4",
        "\uc774\uc57c",
        "\uc57c",
        "\uc740",
        "\ub294",
        "\uc774",
        "\uac00",
        "\uc744",
        "\ub97c",
        "\uc5d0",
        "\ub85c",
        "\ub3c4",
        "\ub9cc",
    )
    for s in suffixes:
        if len(low) > len(s) + 1 and low.endswith(s):
            low = low[: -len(s)]
            break
    return low


def keyword_terms(text: str) -> Set[str]:
    out: Set[str] = set()
    for tok in TOKEN_RE.findall(str(text or "")):
        n = normalize_ko_token(tok)
        if len(n) < 2:
            continue
        out.add(n)
    return out


def stable_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def tier_rank(tier: str) -> int:
    t = str(tier or "").strip().lower()
    if t == "high":
        return 2
    if t == "mid":
        return 1
    return 0


def rank_to_tier(rank: int) -> str:
    if int(rank) >= 2:
        return "high"
    if int(rank) == 1:
        return "mid"
    return "base"


def infer_domain(row: Dict, inp: str, out: str) -> str:
    task = str(row.get("task_type", "")).strip().lower()
    seg = str(row.get("segment_tag", "")).strip().lower()
    lang = str(row.get("language", "")).strip().lower()
    text = f"{inp}\n{out}"
    h_n = hangul_count(text)
    l_n = len(re.findall(r"[A-Za-z]", text))

    code_hint = (
        ("```" in text)
        or ("def " in text)
        or ("class " in text)
        or ("public static void" in text)
        or ("import " in text)
    )
    if task == "code" or seg == "code":
        return "code"
    if code_hint and h_n < 24 and l_n > int(1.25 * max(1, h_n)):
        return "code"
    if task in {"korean", "ko"} or seg == "ko" or lang == "ko":
        return "korean"
    if task in {"english", "en"} or seg in {"english", "en"} or lang == "en":
        return "english"
    if hangul_ratio(text) >= 0.15 and hangul_count(text) >= 6:
        return "korean"
    return "other"


def is_code_like_text(text: str) -> bool:
    return bool(CODE_LIKE_RE.search(str(text or "")))


def path_is_excluded(rel_or_abs: str, exclude_tokens: Sequence[str]) -> bool:
    low = str(rel_or_abs or "").replace("\\", "/").lower()
    return any(tok in low for tok in exclude_tokens)


def load_candidate_files(
    data_root: Path,
    index_json: Path,
    archive_root: Path,
    archive_glob: str,
    include_fields: Sequence[str],
    exclude_path_tokens: Sequence[str],
    extra_inputs: Sequence[str],
) -> Tuple[List[Path], Dict[str, str]]:
    files: List[Path] = []
    source_reason: Dict[str, str] = {}

    include_fields_set = {str(x).strip() for x in include_fields if str(x).strip()}

    if index_json.exists():
        raw = json.loads(index_json.read_text(encoding="utf-8", errors="replace"))
        rows = raw.get("datasets", []) if isinstance(raw, dict) else []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                rel = str(row.get("path", "")).strip().replace("\\", "/")
                field = str(row.get("field", "")).strip()
                if not rel:
                    continue
                if path_is_excluded(rel, exclude_path_tokens):
                    continue
                low = rel.lower()
                mit_hint = bool(MIT_FILE_HINT_RE.search(low)) or ("slm_mit" in low) or ("hf_mit" in low)
                if (field not in include_fields_set) and (not mit_hint):
                    continue
                abs_path = data_root / rel
                if not abs_path.exists():
                    continue
                key = str(abs_path.resolve())
                if key in source_reason:
                    continue
                files.append(abs_path)
                source_reason[key] = f"index:{field or 'unknown'}"

    if archive_root.exists():
        for p in sorted(archive_root.glob(str(archive_glob))):
            if not p.is_file():
                continue
            low = p.name.lower()
            if not MIT_FILE_HINT_RE.search(low) and ("slm_mit" not in low):
                continue
            key = str(p.resolve())
            if key in source_reason:
                continue
            files.append(p)
            source_reason[key] = "archive_glob"

    for raw_in in extra_inputs:
        s = str(raw_in).strip()
        if not s:
            continue
        p = Path(s)
        if not p.is_absolute():
            if p.exists():
                pass
            elif (data_root / p).exists():
                p = data_root / p
        if not p.exists() or not p.is_file():
            continue
        key = str(p.resolve())
        if key in source_reason:
            continue
        files.append(p)
        source_reason[key] = "extra_input"

    files = sorted(files, key=lambda x: str(x).lower())
    return files, source_reason


def load_vocab(vocab_path: Path, vocab_rank_max: int) -> Tuple[Set[str], Dict[str, str], Dict[str, int]]:
    vocab_terms: Set[str] = set()
    vocab_grade: Dict[str, str] = {}
    stats = Counter()

    if not vocab_path.exists():
        return vocab_terms, vocab_grade, dict(stats)

    rank_cap = int(vocab_rank_max)
    with vocab_path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            cols = line.rstrip("\n").split("\t")
            if i == 0:
                stats["header_rows"] += 1
                continue
            if len(cols) < 5:
                stats["bad_columns"] += 1
                continue
            rank_raw = str(cols[0]).strip()
            term_raw = str(cols[1]).strip()
            grade = str(cols[4]).strip().upper()
            if not term_raw:
                stats["empty_term"] += 1
                continue
            try:
                rank = int(rank_raw)
            except Exception:
                rank = 0
            if rank_cap > 0 and rank > rank_cap:
                stats["rank_filtered"] += 1
                continue

            norm = normalize_ko_token(term_raw)
            if len(norm) < 2:
                stats["short_norm_term"] += 1
                continue
            vocab_terms.add(norm)
            old = vocab_grade.get(norm, "")
            if GRADE_WEIGHT.get(grade, 0.5) >= GRADE_WEIGHT.get(old, 0.0):
                vocab_grade[norm] = grade if grade in {"A", "B", "C"} else ""
            stats["kept_terms"] += 1
    stats["unique_terms"] = len(vocab_terms)
    return vocab_terms, vocab_grade, dict(stats)


def pick_vocab_glossary_rows(vocab_path: Path, max_vocab_rows: int) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    if not vocab_path.exists():
        return rows
    lim = int(max_vocab_rows)
    with vocab_path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if lim > 0 and len(rows) >= lim:
                break
            if i == 0:
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            term = normalize_space(cols[1])
            gloss = normalize_space(cols[3])
            grade = normalize_space(cols[4]).upper()
            if not term or not gloss:
                continue
            if len(term) < 2 or len(gloss) < 6:
                continue
            if REPEAT_RE.search(gloss):
                continue
            rows.append((term, gloss, grade if grade in {"A", "B", "C"} else ""))
    return rows


def weighted_sample_without_replacement(rows: Sequence[Dict], weights: Sequence[float], k: int, rng: random.Random) -> List[Dict]:
    n = len(rows)
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(rows)
    if len(weights) != n:
        raise ValueError("weights length mismatch")

    keyed: List[Tuple[float, int]] = []
    for i in range(n):
        w = max(1e-6, float(weights[i]))
        u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
        key = u ** (1.0 / w)
        keyed.append((key, i))
    keyed.sort(key=lambda x: x[0], reverse=True)
    return [rows[i] for _, i in keyed[:k]]


def contains_any(text: str, needles: Sequence[str]) -> bool:
    low = normalize_space(text).lower()
    return any(normalize_space(n).lower() in low for n in needles if normalize_space(n))


def is_hardcode_style_leak(question: str, answer: str, banned_output_substrings: Sequence[str]) -> bool:
    q = normalize_space(question)
    a = normalize_space(answer)
    if not a:
        return True
    if contains_any(a, banned_output_substrings):
        return True
    intro_query = bool(INTRO_QUERY_RE.search(q))
    if not intro_query and INTRO_ANSWER_RE.search(a):
        return True
    if (not intro_query) and ("spear1.0" in a.lower()):
        return True
    return False


def normalize_record(
    row: Dict,
    source_path: str,
    inp: str,
    out: str,
    mit_source: str,
    domain: str,
    tier: str,
    vocab_hit_count: int,
    vocab_score: float,
) -> Dict:
    rec = dict(row)
    rec["input"] = normalize_space(inp)
    rec["output"] = normalize_space(out)
    rec["license"] = "MIT"
    rec["_meta_mit_source"] = str(mit_source)
    rec["_meta_source_file"] = str(source_path).replace("\\", "/")
    rec["_meta_quality_tier"] = str(tier)
    rec["_meta_vocab_hit_count"] = int(vocab_hit_count)
    rec["_meta_vocab_score"] = round(float(vocab_score), 4)

    if domain == "korean":
        rec["task_type"] = "korean"
        rec["segment_tag"] = "ko"
        rec["language"] = "ko"
    elif domain == "code":
        rec["task_type"] = "code"
        rec["segment_tag"] = "code"
        if not str(rec.get("language", "")).strip():
            rec["language"] = "en"
    elif domain == "english":
        rec["task_type"] = "english"
        rec["segment_tag"] = "english"
        rec["language"] = "en"
    else:
        rec["task_type"] = str(rec.get("task_type", "")).strip() or "other"
        rec["segment_tag"] = str(rec.get("segment_tag", "")).strip() or "other"
    return rec


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))

    include_fields = parse_csv_set(args.include_fields)
    exclude_path_tokens = [x.lower() for x in parse_csv_set(args.exclude_path_tokens)]
    banned_output_substrings = parse_double_bar_list(args.banned_output_substrings)
    extra_inputs = parse_csv_set(args.extra_inputs)

    data_root = Path(args.data_root)
    index_json = Path(args.index_json)
    archive_root = Path(args.archive_root)
    vocab_path = Path(args.vocab_path)

    candidate_files, source_reason = load_candidate_files(
        data_root=data_root,
        index_json=index_json,
        archive_root=archive_root,
        archive_glob=str(args.archive_glob),
        include_fields=include_fields,
        exclude_path_tokens=exclude_path_tokens,
        extra_inputs=extra_inputs,
    )
    if not candidate_files:
        raise RuntimeError("no candidate MIT files found from index/archive filters")

    trusted_file_set = {str(p.resolve()) for p in candidate_files}
    allow_file_level_mit = not bool(args.strict_row_license)
    max_rows_per_file = int(args.max_rows_per_file)

    vocab_terms, vocab_grade, vocab_stats = load_vocab(vocab_path=vocab_path, vocab_rank_max=int(args.vocab_rank_max))

    seen = set()
    rejects = Counter()
    file_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "read_rows": 0,
            "kept_rows": 0,
            "mit_row_license": 0,
            "mit_file_allowlist": 0,
        }
    )
    source_counts = Counter()
    source_kept_counts = Counter()
    ko_prefix_counts = Counter()
    task_counts = Counter()
    lang_counts = Counter()

    ko_pool: List[Dict] = []
    ko_weights: List[float] = []
    other_pool: List[Dict] = []

    file_keep_cap = max(0, int(args.max_kept_rows_per_file))
    source_keep_cap = max(0, int(args.max_rows_per_source_dataset))
    max_ko_prefix_repeat = max(0, int(args.max_ko_prefix_repeat))

    for path in candidate_files:
        pkey = str(path).replace("\\", "/")
        file_reason = source_reason.get(str(path.resolve()), "")
        low_name = path.name.lower()
        file_has_mit_hint = bool(MIT_FILE_HINT_RE.search(low_name)) or ("slm_mit" in low_name) or ("hf_mit" in low_name)

        for row in iter_rows(path, max_rows=max_rows_per_file):
            if file_keep_cap > 0 and int(file_stats[pkey]["kept_rows"]) >= file_keep_cap:
                rejects["file_cap_reached"] += 1
                break

            file_stats[pkey]["read_rows"] += 1

            inp = pick_text(row, Q_KEYS)
            out = pick_text(row, A_KEYS)
            if not inp or not out:
                rejects["missing_io"] += 1
                continue

            source_dataset = normalize_space(str(row.get("source_dataset", "")))
            source_bucket = source_dataset if source_dataset else pkey
            if source_keep_cap > 0 and int(source_kept_counts[source_bucket]) >= source_keep_cap:
                rejects["source_dataset_cap"] += 1
                continue

            if len(inp) < int(args.min_input_chars) or len(inp) > int(args.max_input_chars):
                rejects["input_length"] += 1
                continue
            if len(out) < int(args.min_output_chars) or len(out) > int(args.max_output_chars):
                rejects["output_length"] += 1
                continue

            merged = f"{inp}\n{out}"
            if "\ufffd" in merged or REPEAT_RE.search(merged):
                rejects["noise"] += 1
                continue
            if is_hardcode_style_leak(inp, out, banned_output_substrings=banned_output_substrings):
                rejects["hardcode_style_leak"] += 1
                continue

            lic = str(row.get("license", ""))
            row_mit = is_mit_license(lic)
            file_mit = allow_file_level_mit and (str(path.resolve()) in trusted_file_set) and file_has_mit_hint
            if row_mit:
                mit_source = "row_license"
                file_stats[pkey]["mit_row_license"] += 1
            elif file_mit:
                mit_source = "file_allowlist"
                file_stats[pkey]["mit_file_allowlist"] += 1
            else:
                rejects["non_mit"] += 1
                continue

            key = stable_key(inp, out)
            if key in seen:
                rejects["duplicate"] += 1
                continue
            seen.add(key)

            domain = infer_domain(row, inp=inp, out=out)
            is_ko = domain == "korean"
            if is_ko:
                if hangul_count(merged) < int(args.min_hangul_chars) or hangul_ratio(merged) < float(args.min_hangul_ratio):
                    rejects["ko_weak"] += 1
                    continue
                if is_code_like_text(inp) or is_code_like_text(out):
                    rejects["ko_code_like"] += 1
                    continue
                out_hangul = hangul_count(out)
                out_hangul_ratio = hangul_ratio(out)
                out_latin = len(re.findall(r"[A-Za-z]", out))
                out_latin_ratio = latin_ratio(out)
                if out_hangul < 4 and out_latin >= 8:
                    rejects["ko_latin_heavy"] += 1
                    continue
                if len(out) <= 40 and hangul_count(out) <= 1 and re.search(r"[A-Za-z]{3,}", out):
                    rejects["ko_short_ascii_answer"] += 1
                    continue
                if len(out) >= 24 and out_hangul_ratio < float(args.ko_min_output_hangul_ratio):
                    rejects["ko_output_low_hangul_ratio"] += 1
                    continue
                if len(out) >= 24 and out_latin_ratio > float(args.ko_max_output_latin_ratio):
                    rejects["ko_output_latin_dominant"] += 1
                    continue
                if ASCII_LEAD_RE.search(out) and out_hangul <= int(args.ko_ascii_lead_max_hangul):
                    rejects["ko_ascii_lead"] += 1
                    continue
                if CODE_TOPIC_RE.search(out) and out_hangul_ratio < 0.52:
                    rejects["ko_code_topic_dominant"] += 1
                    continue
                if code_symbol_ratio(out) > 0.11 and out_hangul_ratio < 0.55:
                    rejects["ko_code_symbol_dominant"] += 1
                    continue

            q_terms = keyword_terms(inp)
            a_terms = keyword_terms(out)
            qa_terms = q_terms.union(a_terms)
            hit_terms = qa_terms.intersection(vocab_terms) if vocab_terms else set()
            vocab_hit_count = int(len(hit_terms))
            vocab_score = 0.0
            for term in hit_terms:
                grade = vocab_grade.get(term, "")
                vocab_score += float(GRADE_WEIGHT.get(grade, 0.5))
            overlap = len(q_terms.intersection(a_terms))

            if is_ko:
                q_is_def = bool(DEF_QUERY_RE.search(inp))
                q_is_logic = bool(LOGIC_QUERY_RE.search(inp))
                q_has_math = bool(re.search(r"\d+\s*[\+\-\*\/]\s*\d+", inp))
                out_has_digits = bool(re.search(r"\d", out))
                out_starts_answer = bool(re.match(r"^\s*(?:\uc815\ub2f5\uc740|answer\s+is)\b", out, re.IGNORECASE))

                if len(out) < 14 and not out_has_digits:
                    rejects["ko_too_short"] += 1
                    continue
                if q_is_def and overlap == 0:
                    rejects["ko_def_no_overlap"] += 1
                    continue
                if q_is_logic and not LOGIC_ANSWER_RE.search(out):
                    rejects["ko_logic_weak"] += 1
                    continue
                if overlap == 0 and not (q_is_logic or (q_has_math and out_has_digits)):
                    rejects["ko_no_keyword_overlap"] += 1
                    continue
                if out_starts_answer and not (q_is_logic or q_has_math):
                    rejects["ko_answer_only_style"] += 1
                    continue

            ko_prefix_key = ""
            if is_ko:
                ko_prefix_key = answer_prefix_key(out)
                if max_ko_prefix_repeat > 0 and ko_prefix_key:
                    if int(ko_prefix_counts[ko_prefix_key]) >= max_ko_prefix_repeat:
                        rejects["ko_prefix_overrepresented"] += 1
                        continue

            base_tier = str(row.get("_meta_quality_tier", "base")).strip().lower()
            tier_n = tier_rank(base_tier)
            if is_ko and vocab_hit_count >= 2:
                tier_n = max(tier_n, 2)
            elif is_ko and vocab_hit_count >= 1:
                tier_n = max(tier_n, 1)
            if is_ko and overlap > 0 and vocab_hit_count >= 1:
                tier_n = max(tier_n, 2)
            tier = rank_to_tier(tier_n)

            rec = normalize_record(
                row=row,
                source_path=pkey,
                inp=inp,
                out=out,
                mit_source=mit_source,
                domain=domain,
                tier=tier,
                vocab_hit_count=vocab_hit_count,
                vocab_score=vocab_score,
            )
            file_stats[pkey]["kept_rows"] += 1
            source_counts[str(rec.get("source_dataset", ""))] += 1
            source_kept_counts[source_bucket] += 1
            task_counts[str(rec.get("task_type", ""))] += 1
            lang_counts[str(rec.get("language", ""))] += 1

            if is_ko:
                if ko_prefix_key:
                    ko_prefix_counts[ko_prefix_key] += 1
                sample_w = 1.0 + float(args.ko_vocab_boost) * float(vocab_score)
                sample_w += 0.18 * float(overlap > 0)
                if tier == "high":
                    sample_w += 0.22
                elif tier == "mid":
                    sample_w += 0.10
                ko_pool.append(rec)
                ko_weights.append(float(max(0.05, sample_w)))
            else:
                other_pool.append(rec)

            if mit_source == "row_license":
                rejects["accepted_mit_row_license"] += 1
            else:
                rejects["accepted_mit_file_allowlist"] += 1
            if file_reason:
                rejects[f"source_reason::{file_reason}"] += 1

    vocab_rows_added = 0
    if bool(args.add_vocab_rows):
        for term, gloss, grade in pick_vocab_glossary_rows(vocab_path=vocab_path, max_vocab_rows=int(args.max_vocab_rows)):
            key = stable_key(term, gloss)
            if key in seen:
                continue
            seen.add(key)
            rec = {
                "task_type": "korean",
                "segment_tag": "ko",
                "language": "ko",
                "license": "MIT",
                "_meta_mit_source": "vocab_glossary",
                "_meta_source_file": str(vocab_path).replace("\\", "/"),
                "_meta_quality_tier": "high",
                "_meta_vocab_hit_count": 1,
                "_meta_vocab_score": round(float(GRADE_WEIGHT.get(grade, 0.5)), 4),
                "input": term,
                "output": gloss,
            }
            ko_pool.append(rec)
            ko_weights.append(1.5 + float(GRADE_WEIGHT.get(grade, 0.5)))
            vocab_rows_added += 1
            source_counts["vocab_glossary"] += 1
            task_counts["korean"] += 1
            lang_counts["ko"] += 1

    max_total_rows = int(args.max_total_rows)
    target_ko_ratio = min(max(float(args.target_ko_ratio), 0.0), 1.0)

    if max_total_rows <= 0:
        selected_ko = list(ko_pool)
        selected_other = list(other_pool)
    else:
        total_cap = int(max_total_rows)
        target_ko = int(round(total_cap * target_ko_ratio))
        target_ko = min(target_ko, len(ko_pool))
        target_other = min(total_cap - target_ko, len(other_pool))

        if (target_ko + target_other) < total_cap:
            remain = total_cap - (target_ko + target_other)
            spare_ko = max(0, len(ko_pool) - target_ko)
            add_ko = min(spare_ko, remain)
            target_ko += add_ko
            remain -= add_ko
            if remain > 0:
                spare_other = max(0, len(other_pool) - target_other)
                target_other += min(spare_other, remain)

        selected_ko = weighted_sample_without_replacement(ko_pool, ko_weights, k=target_ko, rng=rng)
        if target_other >= len(other_pool):
            selected_other = list(other_pool)
        else:
            idxs = list(range(len(other_pool)))
            rng.shuffle(idxs)
            selected_other = [other_pool[i] for i in idxs[:target_other]]

    out_rows = selected_ko + selected_other
    if bool(args.shuffle):
        rng.shuffle(out_rows)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wf:
        for row in out_rows:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    sources_path = Path(args.sources_tsv)
    sources_path.parent.mkdir(parents=True, exist_ok=True)
    src_lines = ["path\tread_rows\tkept_rows\tmit_row_license\tmit_file_allowlist"]
    for p in sorted(file_stats.keys()):
        st = file_stats[p]
        src_lines.append(
            f"{p}\t{int(st.get('read_rows', 0))}\t{int(st.get('kept_rows', 0))}\t{int(st.get('mit_row_license', 0))}\t{int(st.get('mit_file_allowlist', 0))}"
        )
    sources_path.write_text("\n".join(src_lines) + "\n", encoding="utf-8")

    out_ko = sum(1 for r in out_rows if str(r.get("language", "")).strip().lower() == "ko")
    manifest = {
        "generated_at": now_iso(),
        "out_jsonl": str(out_path).replace("\\", "/"),
        "sources_tsv": str(sources_path).replace("\\", "/"),
        "rows_total": len(out_rows),
        "rows_ko": int(out_ko),
        "rows_other": int(len(out_rows) - out_ko),
        "selected_ko_ratio": round(float(out_ko / max(1, len(out_rows))), 6),
        "pool_sizes": {
            "ko_pool": len(ko_pool),
            "other_pool": len(other_pool),
        },
        "vocab_rows_added": int(vocab_rows_added),
        "vocab_stats": vocab_stats,
        "inputs": [str(p).replace("\\", "/") for p in candidate_files],
        "include_fields": include_fields,
        "exclude_path_tokens": exclude_path_tokens,
        "banned_output_substrings": banned_output_substrings,
        "allow_file_level_mit": bool(allow_file_level_mit),
        "strict_row_license": bool(args.strict_row_license),
        "max_rows_per_file": int(max_rows_per_file),
        "max_kept_rows_per_file": int(file_keep_cap),
        "max_rows_per_source_dataset": int(source_keep_cap),
        "max_total_rows": int(max_total_rows),
        "target_ko_ratio": float(target_ko_ratio),
        "ko_min_output_hangul_ratio": float(args.ko_min_output_hangul_ratio),
        "ko_max_output_latin_ratio": float(args.ko_max_output_latin_ratio),
        "ko_ascii_lead_max_hangul": int(args.ko_ascii_lead_max_hangul),
        "max_ko_prefix_repeat": int(max_ko_prefix_repeat),
        "ko_vocab_boost": float(args.ko_vocab_boost),
        "seed": int(args.seed),
        "shuffle": bool(args.shuffle),
        "task_counts": dict(task_counts),
        "language_counts": dict(lang_counts),
        "source_dataset_top30": dict(source_counts.most_common(30)),
        "ko_prefix_top30": dict(ko_prefix_counts.most_common(30)),
        "reject_or_event_counts": dict(rejects),
        "file_stats": dict(file_stats),
        "mit_policy": {
            "row_license_required": not bool(allow_file_level_mit),
            "file_allowlist_uses_index_and_mit_filename_hint": bool(allow_file_level_mit),
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "out_jsonl": str(out_path),
                "rows_total": len(out_rows),
                "rows_ko": int(out_ko),
                "rows_other": int(len(out_rows) - out_ko),
                "ko_ratio": round(float(out_ko / max(1, len(out_rows))), 4),
                "vocab_rows_added": int(vocab_rows_added),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
