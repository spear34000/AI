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
    "합니다",
    "그리고",
    "또한",
    "에서",
    "으로",
    "하는",
    "대한",
    "위한",
    "할",
    "수",
    "있는",
    "없는",
}


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
        if first in CODE_PROMPTS_EN or first in CODE_PROMPTS_KO or first == DOC_PROMPT_KO:
            return rest.strip()
    if s.startswith(DOC_PROMPT_KO):
        return s[len(DOC_PROMPT_KO) :].strip()
    return s


def row_key(row: Dict) -> str:
    raw = f"{row.get('task_type','')}\n{row.get('language','')}\n{row.get('input','')}\n{row.get('output','')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_base_record(
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


IMPORT_PATTERNS = [
    re.compile(r"^\s*import\s+.+"),
    re.compile(r"^\s*from\s+.+\s+import\s+.+"),
    re.compile(r"^\s*use\s+.+"),
    re.compile(r"^\s*#include\s+.+"),
    re.compile(r"^\s*(const|let|var)\s+\w+\s*=\s*require\(.+\)"),
    re.compile(r"^\s*require\(.+\)"),
    re.compile(r"^\s*using\s+.+"),
]


def extract_import_lines(code: str, max_lines: int = 12) -> List[str]:
    out: List[str] = []
    seen = set()
    for line in str(code).splitlines():
        t = line.strip()
        if not t or len(t) > 200:
            continue
        if any(p.match(t) for p in IMPORT_PATTERNS):
            if t not in seen:
                seen.add(t)
                out.append(t)
                if len(out) >= max_lines:
                    break
    return out


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


def extract_signature_lines(code: str, max_lines: int = 12) -> List[str]:
    out: List[str] = []
    seen = set()
    for line in str(code).splitlines():
        t = line.strip()
        if not t or len(t) > 220:
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


def clean_text_for_prompt(s: str, max_chars: int) -> str:
    t = str(s).replace("\r\n", "\n").replace("\r", "\n").strip()
    return t[:max_chars].strip()


SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+|\n+")


def pick_summary(text: str, max_sentences: int = 2, max_chars: int = 320) -> str:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p and p.strip()]
    chosen: List[str] = []
    cur = 0
    for p in parts:
        if len(p) < 20:
            continue
        if cur + len(p) + (1 if chosen else 0) > max_chars:
            break
        chosen.append(p)
        cur += len(p) + (1 if chosen else 0)
        if len(chosen) >= max_sentences:
            break
    return " ".join(chosen).strip()


TOKEN_RE = re.compile(r"[A-Za-z가-힣][A-Za-z0-9_+#\.-]{1,30}")


def extract_keywords(text: str, k: int = 6) -> List[str]:
    c = Counter()
    for m in TOKEN_RE.findall(text):
        tok = m.strip().lower()
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        c[tok] += 1
    return [w for w, _ in c.most_common(k)]


def maybe_code_augments(row: Dict, rnd: random.Random) -> List[Tuple[Dict, str]]:
    out: List[Tuple[Dict, str]] = []
    lang = str(row.get("language", "")).strip().lower()
    left = strip_known_prompt(str(row.get("input", "")))
    right = str(row.get("output", "")).strip()
    if not left or not right:
        return out

    code_full = f"{left}\n{right}"
    code_head = clean_text_for_prompt(code_full, max_chars=2400)

    base_hash = hashlib.sha1(f"{row.get('id','')}|code_aug".encode("utf-8")).hexdigest()[:12]

    if rnd.random() < 0.35:
        if lang == "ko":
            inp = CODE_PROMPTS_KO[1] + "\n\n" + clean_text_for_prompt(left, max_chars=1800)
            lg = "ko"
        else:
            inp = CODE_PROMPTS_EN[1] + "\n\n" + clean_text_for_prompt(left, max_chars=1800)
            lg = "en"
        rec = build_base_record(
            rid=f"mit_aug_cont_{base_hash}",
            task_type="code",
            segment_tag="code",
            language=lg,
            inp=inp,
            out=right,
            src_row=row,
            aug_type="continuation_rephrase",
        )
        out.append((rec, "continuation_rephrase"))

    imports = extract_import_lines(code_full)
    if imports and rnd.random() < 0.65:
        if lang == "ko":
            inp = "다음 코드에서 import/의존성 선언 라인만 추려서 줄바꿈으로 출력해줘.\n\n" + code_head
            lg = "ko"
        else:
            inp = (
                "Extract only the import/dependency declaration lines from the code below. "
                "Return one line per declaration.\n\n" + code_head
            )
            lg = "en"
        rec = build_base_record(
            rid=f"mit_aug_import_{base_hash}",
            task_type="code",
            segment_tag="code",
            language=lg,
            inp=inp,
            out="\n".join(imports),
            src_row=row,
            aug_type="extract_imports",
        )
        out.append((rec, "extract_imports"))

    sigs = extract_signature_lines(code_full)
    if sigs and rnd.random() < 0.6:
        if lang == "ko":
            inp = "다음 코드에서 함수/클래스 선언 라인만 추려서 줄바꿈으로 출력해줘.\n\n" + code_head
            lg = "ko"
        else:
            inp = (
                "Extract only function/class declaration lines from the code below. "
                "Return one line per declaration.\n\n" + code_head
            )
            lg = "en"
        rec = build_base_record(
            rid=f"mit_aug_sign_{base_hash}",
            task_type="code",
            segment_tag="code",
            language=lg,
            inp=inp,
            out="\n".join(sigs),
            src_row=row,
            aug_type="extract_signatures",
        )
        out.append((rec, "extract_signatures"))

    return out


def maybe_korean_augments(row: Dict) -> List[Tuple[Dict, str]]:
    out: List[Tuple[Dict, str]] = []
    left = strip_known_prompt(str(row.get("input", "")))
    right = str(row.get("output", "")).strip()
    if not left or not right:
        return out
    full = clean_text_for_prompt(f"{left}\n\n{right}", max_chars=2200)
    if len(full) < 180:
        return out

    base_hash = hashlib.sha1(f"{row.get('id','')}|ko_aug".encode("utf-8")).hexdigest()[:12]

    summary = pick_summary(full, max_sentences=2, max_chars=320)
    if len(summary) >= 40:
        rec = build_base_record(
            rid=f"mit_aug_ko_sum_{base_hash}",
            task_type="korean",
            segment_tag="ko",
            language="ko",
            inp="다음 문서를 2문장 이내로 요약해줘.\n\n" + full,
            out=summary,
            src_row=row,
            aug_type="ko_summary_extract",
        )
        out.append((rec, "ko_summary_extract"))

    kws = extract_keywords(full, k=6)
    if len(kws) >= 4:
        rec = build_base_record(
            rid=f"mit_aug_ko_kw_{base_hash}",
            task_type="korean",
            segment_tag="ko",
            language="ko",
            inp="다음 문서의 핵심 키워드 6개를 쉼표로 출력해줘.\n\n" + full,
            out=", ".join(kws[:6]),
            src_row=row,
            aug_type="ko_keywords_extract",
        )
        out.append((rec, "ko_keywords_extract"))

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_jsonl", default="data/slm_mit_unified_v2.jsonl")
    parser.add_argument("--out_jsonl", default="data/slm_mit_unified_v3.jsonl")
    parser.add_argument("--manifest", default="data/slm_mit_unified_v3.manifest.json")
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
    lang_counts = Counter()

    for row in rows:
        if str(row.get("license", "")).strip().upper() != "MIT":
            continue
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)

        task = str(row.get("task_type", "")).strip().lower()
        if task == "code":
            for rec, kind in maybe_code_augments(row, rnd):
                k = row_key(rec)
                if k in seen:
                    continue
                seen.add(k)
                merged.append(rec)
                aug_counts[kind] += 1
        elif task == "korean":
            for rec, kind in maybe_korean_augments(row):
                k = row_key(rec)
                if k in seen:
                    continue
                seen.add(k)
                merged.append(rec)
                aug_counts[kind] += 1

    rnd.shuffle(merged)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            task_counts[str(row.get("task_type", ""))] += 1
            lang_counts[str(row.get("language", ""))] += 1

    manifest = {
        "base_jsonl": str(base_path),
        "output_jsonl": str(out_path),
        "rows_base": len(rows),
        "rows_total": len(merged),
        "rows_augmented": int(len(merged) - len(rows)),
        "augment_counts": dict(aug_counts),
        "task_counts": dict(task_counts),
        "language_counts": dict(lang_counts),
        "license_policy": "MIT only",
        "seed": int(args.seed),
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
