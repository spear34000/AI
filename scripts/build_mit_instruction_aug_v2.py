from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CODE_PROMPTS_EN = (
    "Continue the source file naturally. Preserve style and behavior.",
    "Continue the following source code in the same style and behavior.",
)
CODE_PROMPTS_KO = (
    "다음 소스 코드를 자연스럽게 이어서 완성하세요. 스타일과 동작을 유지하세요.",
    "아래 코드 파일을 기존 스타일과 동작을 유지하면서 이어서 완성해줘.",
)
DOC_PROMPT_KO = "다음 한국어 기술 문서를 자연스럽게 이어 쓰세요:"


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "you",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "can",
    "will",
    "its",
    "about",
    "http",
    "https",
    "www",
    "com",
    "있습니다",
    "합니다",
    "그리고",
    "또한",
    "으로",
    "에서",
    "하는",
    "대한",
    "위한",
    "있는",
    "없는",
}

CODE_KEYWORDS = {
    "import",
    "from",
    "const",
    "let",
    "var",
    "return",
    "class",
    "def",
    "function",
    "public",
    "private",
    "protected",
    "internal",
    "async",
    "await",
    "new",
    "this",
    "true",
    "false",
    "null",
    "undefined",
}


LANG_BY_EXT = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TSX",
    ".jsx": "JSX",
    ".java": "Java",
    ".go": "Go",
    ".rs": "Rust",
    ".c": "C",
    ".cpp": "C++",
    ".cc": "C++",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".cs": "C#",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".php": "PHP",
    ".rb": "Ruby",
    ".sh": "Shell",
}


IMPORT_PATTERNS = [
    re.compile(r"^\s*import\s+.+"),
    re.compile(r"^\s*from\s+.+\s+import\s+.+"),
    re.compile(r"^\s*use\s+.+"),
    re.compile(r"^\s*#include\s+.+"),
    re.compile(r"^\s*(const|let|var)\s+\w+\s*=\s*require\(.+\)"),
    re.compile(r"^\s*require\(.+\)"),
    re.compile(r"^\s*using\s+.+"),
]


SIGN_PATTERNS = [
    re.compile(r"^\s*(def|class)\s+\w+"),
    re.compile(r"^\s*(export\s+)?(async\s+)?function\s+\w+"),
    re.compile(r"^\s*(export\s+)?class\s+\w+"),
    re.compile(r"^\s*(pub\s+)?(async\s+)?fn\s+\w+"),
    re.compile(r"^\s*func\s+\w+"),
    re.compile(r"^\s*(public|private|protected|internal)\s+.+\("),
    re.compile(r"^\s*(const|let|var)\s+\w+\s*=\s*(async\s*)?\(.+\)\s*=>"),
]


CONTROL_PREFIXES = ("if ", "for ", "while ", "switch ", "catch ", "return ")
TOKEN_RE = re.compile(r"[A-Za-z가-힣_][A-Za-z0-9_+#\.-]{1,40}")
SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+|\n+")


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def strip_known_prompt(text: str) -> str:
    s = str(text).strip()
    if "\n\n" in s:
        first, rest = s.split("\n\n", 1)
        first_l = first.lower()
        if (
            first in CODE_PROMPTS_EN
            or first in CODE_PROMPTS_KO
            or first == DOC_PROMPT_KO
            or "continue the source file naturally" in first_l
            or "continue the following source code" in first_l
            or "다음 소스 코드" in first
            or "아래 코드 파일" in first
            or "다음 한국어 기술 문서" in first
        ):
            return rest.strip()
    if s.startswith(DOC_PROMPT_KO):
        return s[len(DOC_PROMPT_KO) :].strip()
    return s


def row_key(row: Dict) -> str:
    raw = f"{row.get('task_type','')}\n{row.get('language','')}\n{row.get('input','')}\n{row.get('output','')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def clean_text_for_prompt(s: str, max_chars: int) -> str:
    t = str(s).replace("\r\n", "\n").replace("\r", "\n").strip()
    return t[:max_chars].strip()


def build_record(
    *,
    rid: str,
    task_type: str,
    segment_tag: str,
    language: str,
    inp: str,
    out: str,
    src_row: Dict,
    aug_type: str,
) -> Dict:
    return {
        "id": rid,
        "task_type": task_type,
        "segment_tag": segment_tag,
        "language": language,
        "input": inp.strip(),
        "output": out.strip(),
        "_meta_quality_tier": "high",
        "source_dataset": str(src_row.get("source_dataset", "")),
        "source_path": str(src_row.get("source_path", "")),
        "source_url": str(src_row.get("source_url", "")),
        "license": "MIT",
        "_augment_type": aug_type,
    }


def extract_import_lines(code: str, max_lines: int = 14) -> List[str]:
    out: List[str] = []
    seen = set()
    for line in str(code).splitlines():
        t = line.strip()
        if not t or len(t) > 220:
            continue
        if any(p.match(t) for p in IMPORT_PATTERNS):
            if t not in seen:
                seen.add(t)
                out.append(t)
                if len(out) >= max_lines:
                    break
    return out


def extract_signature_lines(code: str, max_lines: int = 14) -> List[str]:
    out: List[str] = []
    seen = set()
    for line in str(code).splitlines():
        t = line.strip()
        if not t or len(t) > 240:
            continue
        low = t.lower()
        if low.startswith(CONTROL_PREFIXES):
            continue
        ok = any(p.match(t) for p in SIGN_PATTERNS)
        if not ok and "(" in t and ")" in t and t.endswith("{"):
            ok = True
        if ok and t not in seen:
            seen.add(t)
            out.append(t)
            if len(out) >= max_lines:
                break
    return out


def extract_identifier_tokens(code: str, k: int = 12) -> List[str]:
    c = Counter()
    for t in TOKEN_RE.findall(code):
        tok = t.strip()
        tl = tok.lower()
        if len(tok) < 3:
            continue
        if tl in STOPWORDS:
            continue
        if tl in CODE_KEYWORDS:
            continue
        if re.fullmatch(r"[0-9_]+", tok):
            continue
        c[tok] += 1
    return [w for w, _ in c.most_common(k)]


def detect_language_from_path(path: str) -> str:
    p = Path(str(path).strip().lower())
    return LANG_BY_EXT.get(p.suffix, "Unknown")


def classify_code_role(path: str, code_text: str) -> str:
    p = str(path).lower()
    c = str(code_text).lower()
    if any(k in p for k in ["/test", "\\test", "/tests", "\\tests"]) or "pytest" in c or "assert " in c:
        return "test"
    if any(k in p for k in [".tsx", ".jsx", "/components", "\\components"]) or "react" in c:
        return "frontend"
    if any(k in p for k in ["/cli", "\\cli", "command"]) or "argparse" in c or "typer" in c or "click." in c:
        return "cli"
    if any(k in p for k in ["/config", "\\config"]) or "eslint" in c or "prettier" in c or "tsconfig" in c:
        return "config"
    if any(k in p for k in ["/algorithms", "\\algorithms"]) or "binary search" in c:
        return "algorithm"
    if any(k in p for k in ["/docs", "\\docs", "readme"]):
        return "docs"
    return "library"


def split_sentences(text: str) -> List[str]:
    return [p.strip() for p in SENT_SPLIT_RE.split(str(text)) if p and p.strip()]


def pick_summary(text: str, max_sentences: int = 2, max_chars: int = 340) -> str:
    parts = split_sentences(text)
    chosen: List[str] = []
    cur = 0
    for p in parts:
        if len(p) < 25:
            continue
        if cur + len(p) + (1 if chosen else 0) > max_chars:
            break
        chosen.append(p)
        cur += len(p) + (1 if chosen else 0)
        if len(chosen) >= max_sentences:
            break
    return " ".join(chosen).strip()


def extract_keywords(text: str, k: int = 6) -> List[str]:
    c = Counter()
    for m in TOKEN_RE.findall(text):
        tok = m.strip().lower()
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        c[tok] += 1
    return [w for w, _ in c.most_common(k)]


def pick_sentence_with_keyword(text: str, keyword: str) -> str:
    for s in split_sentences(text):
        if keyword.lower() in s.lower() and len(s) >= 20:
            return s
    return ""


def first_markdown_heading(text: str) -> str:
    for line in str(text).splitlines():
        t = line.strip()
        if t.startswith("#"):
            h = t.lstrip("#").strip()
            if len(h) < 3 or len(h) > 120:
                continue
            if h.count(".") > 1 and len(h) > 60:
                continue
            return h
    return ""


def maybe_code_augments(row: Dict, rnd: random.Random) -> List[Tuple[Dict, str]]:
    out: List[Tuple[Dict, str]] = []
    lang = str(row.get("language", "")).strip().lower()
    left = strip_known_prompt(str(row.get("input", "")))
    right = str(row.get("output", "")).strip()
    if not left or not right:
        return out

    code_full = f"{left}\n{right}"
    snippet = clean_text_for_prompt(code_full, max_chars=2800)
    src_path = str(row.get("source_path", ""))
    base_hash = hashlib.sha1(f"{row.get('id','')}|v2".encode("utf-8")).hexdigest()[:14]

    if rnd.random() < 0.25:
        if lang == "ko":
            inp = CODE_PROMPTS_KO[1] + "\n\n" + clean_text_for_prompt(left, max_chars=1800)
            lg = "ko"
        else:
            inp = CODE_PROMPTS_EN[1] + "\n\n" + clean_text_for_prompt(left, max_chars=1800)
            lg = "en"
        out.append(
            (
                build_record(
                    rid=f"mit_v4_cont_{base_hash}",
                    task_type="code",
                    segment_tag="code",
                    language=lg,
                    inp=inp,
                    out=right,
                    src_row=row,
                    aug_type="continuation_rephrase_v2",
                ),
                "continuation_rephrase_v2",
            )
        )

    imports = extract_import_lines(code_full)
    if imports and rnd.random() < 0.55:
        if lang == "ko":
            inp = "다음 코드에서 import/의존성 선언 라인만 추려서 줄바꿈으로 출력해줘.\n\n" + snippet
            lg = "ko"
        else:
            inp = (
                "Extract only the import/dependency declaration lines from the code below. "
                "Return one line per declaration.\n\n" + snippet
            )
            lg = "en"
        out.append(
            (
                build_record(
                    rid=f"mit_v4_imp_{base_hash}",
                    task_type="code",
                    segment_tag="code",
                    language=lg,
                    inp=inp,
                    out="\n".join(imports),
                    src_row=row,
                    aug_type="extract_imports_v2",
                ),
                "extract_imports_v2",
            )
        )

    sigs = extract_signature_lines(code_full)
    if sigs and rnd.random() < 0.55:
        if lang == "ko":
            inp = "다음 코드에서 함수/클래스 선언 라인만 추려서 줄바꿈으로 출력해줘.\n\n" + snippet
            lg = "ko"
        else:
            inp = (
                "Extract only function/class declaration lines from the code below. "
                "Return one line per declaration.\n\n" + snippet
            )
            lg = "en"
        out.append(
            (
                build_record(
                    rid=f"mit_v4_sig_{base_hash}",
                    task_type="code",
                    segment_tag="code",
                    language=lg,
                    inp=inp,
                    out="\n".join(sigs),
                    src_row=row,
                    aug_type="extract_signatures_v2",
                ),
                "extract_signatures_v2",
            )
        )

    detected_lang = detect_language_from_path(src_path)
    if detected_lang != "Unknown" and rnd.random() < 0.50:
        if lang == "ko":
            inp = "다음 코드 조각의 주 언어를 한 단어로 답해줘.\n\n" + snippet
            lg = "ko"
        else:
            inp = "Identify the primary programming language of this code snippet in one word.\n\n" + snippet
            lg = "en"
        out.append(
            (
                build_record(
                    rid=f"mit_v4_lang_{base_hash}",
                    task_type="code",
                    segment_tag="code",
                    language=lg,
                    inp=inp,
                    out=detected_lang,
                    src_row=row,
                    aug_type="identify_language_v1",
                ),
                "identify_language_v1",
            )
        )

    role = classify_code_role(src_path, code_full)
    if rnd.random() < 0.50:
        if lang == "ko":
            inp = (
                "다음 코드의 역할을 한 단어로 분류해줘. "
                "가능한 라벨: test, frontend, cli, algorithm, config, docs, library.\n\n" + snippet
            )
            lg = "ko"
        else:
            inp = (
                "Classify the role of this code into one label from: "
                "test, frontend, cli, algorithm, config, docs, library.\n\n" + snippet
            )
            lg = "en"
        out.append(
            (
                build_record(
                    rid=f"mit_v4_role_{base_hash}",
                    task_type="code",
                    segment_tag="code",
                    language=lg,
                    inp=inp,
                    out=role,
                    src_row=row,
                    aug_type="classify_role_v1",
                ),
                "classify_role_v1",
            )
        )

    ids = extract_identifier_tokens(code_full, k=12)
    if len(ids) >= 5 and rnd.random() < 0.45:
        if lang == "ko":
            inp = "다음 코드에서 중요한 식별자 이름 10개를 쉼표로 출력해줘.\n\n" + snippet
            lg = "ko"
        else:
            inp = "List 10 important identifier names from this code, separated by commas.\n\n" + snippet
            lg = "en"
        out.append(
            (
                build_record(
                    rid=f"mit_v4_ids_{base_hash}",
                    task_type="code",
                    segment_tag="code",
                    language=lg,
                    inp=inp,
                    out=", ".join(ids[:10]),
                    src_row=row,
                    aug_type="extract_identifiers_v1",
                ),
                "extract_identifiers_v1",
            )
        )

    return out


def maybe_korean_augments(row: Dict, rnd: random.Random) -> List[Tuple[Dict, str]]:
    out: List[Tuple[Dict, str]] = []
    left = strip_known_prompt(str(row.get("input", "")))
    right = str(row.get("output", "")).strip()
    if not left or not right:
        return out
    full = clean_text_for_prompt(f"{left}\n\n{right}", max_chars=2600)
    if len(full) < 180:
        return out

    base_hash = hashlib.sha1(f"{row.get('id','')}|ko_v2".encode("utf-8")).hexdigest()[:14]

    summary = pick_summary(full, max_sentences=2, max_chars=340)
    if len(summary) >= 40:
        out.append(
            (
                build_record(
                    rid=f"mit_v4_ko_sum_{base_hash}",
                    task_type="korean",
                    segment_tag="ko",
                    language="ko",
                    inp="다음 문서를 2문장 이내로 요약해줘.\n\n" + full,
                    out=summary,
                    src_row=row,
                    aug_type="ko_summary_extract_v2",
                ),
                "ko_summary_extract_v2",
            )
        )

    kws = extract_keywords(full, k=6)
    if len(kws) >= 4:
        out.append(
            (
                build_record(
                    rid=f"mit_v4_ko_kw_{base_hash}",
                    task_type="korean",
                    segment_tag="ko",
                    language="ko",
                    inp="다음 문서의 핵심 키워드 6개를 쉼표로 출력해줘.\n\n" + full,
                    out=", ".join(kws[:6]),
                    src_row=row,
                    aug_type="ko_keywords_extract_v2",
                ),
                "ko_keywords_extract_v2",
            )
        )

    if kws and rnd.random() < 0.65:
        s = pick_sentence_with_keyword(full, kws[0])
        if len(s) >= 20:
            out.append(
                (
                    build_record(
                        rid=f"mit_v4_ko_sent_{base_hash}",
                        task_type="korean",
                        segment_tag="ko",
                        language="ko",
                        inp=f"다음 문서에서 '{kws[0]}'와 직접 관련된 문장 하나를 그대로 찾아줘.\n\n{full}",
                        out=s,
                        src_row=row,
                        aug_type="ko_sentence_extract_v1",
                    ),
                    "ko_sentence_extract_v1",
                )
            )

    heading = first_markdown_heading(full)
    if heading and rnd.random() < 0.7:
        out.append(
            (
                build_record(
                    rid=f"mit_v4_ko_title_{base_hash}",
                    task_type="korean",
                    segment_tag="ko",
                    language="ko",
                    inp="다음 문서의 제목을 한 줄로 답해줘.\n\n" + full,
                    out=heading,
                    src_row=row,
                    aug_type="ko_title_extract_v1",
                ),
                "ko_title_extract_v1",
            )
        )

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_jsonl", default="data/slm_mit_unified_v3.jsonl")
    parser.add_argument("--out_jsonl", default="data/slm_mit_unified_v4.jsonl")
    parser.add_argument("--manifest", default="data/slm_mit_unified_v4.manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rnd = random.Random(int(args.seed))
    base_path = Path(args.base_jsonl)
    if not base_path.exists():
        raise RuntimeError(f"base_jsonl not found: {base_path}")

    rows = list(iter_jsonl(base_path))
    merged: List[Dict] = []
    seen = set()
    aug_counts = Counter()
    task_counts = Counter()
    language_counts = Counter()

    base_rows = 0
    existing_aug_rows = 0
    new_aug_rows = 0

    for row in rows:
        if str(row.get("license", "")).strip().upper() != "MIT":
            continue

        k0 = row_key(row)
        if k0 in seen:
            continue
        seen.add(k0)
        merged.append(row)

        existing_aug = str(row.get("_augment_type", "")).strip()
        if existing_aug:
            existing_aug_rows += 1
            continue

        base_rows += 1
        task = str(row.get("task_type", "")).strip().lower()
        if task == "code":
            created = maybe_code_augments(row, rnd)
        elif task == "korean":
            created = maybe_korean_augments(row, rnd)
        else:
            created = []

        for rec, name in created:
            kr = row_key(rec)
            if kr in seen:
                continue
            seen.add(kr)
            merged.append(rec)
            aug_counts[name] += 1
            new_aug_rows += 1

    rnd.shuffle(merged)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            task_counts[str(row.get("task_type", ""))] += 1
            language_counts[str(row.get("language", ""))] += 1

    manifest = {
        "base_jsonl": str(base_path),
        "output_jsonl": str(out_path),
        "rows_in_base_file": len(rows),
        "rows_total": len(merged),
        "rows_base_no_augment_type": base_rows,
        "rows_existing_augment": existing_aug_rows,
        "rows_new_augmented": new_aug_rows,
        "augment_counts": dict(aug_counts),
        "task_counts": dict(task_counts),
        "language_counts": dict(language_counts),
        "license_policy": "MIT only",
        "seed": int(args.seed),
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
