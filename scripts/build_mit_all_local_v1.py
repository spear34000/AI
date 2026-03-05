from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


MAX_RAW_FILE_BYTES = 2_000_000
KO_CODE_PROMPT = "다음 소스 코드를 자연스럽게 이어서 완성하세요. 스타일과 동작을 유지하세요.\n\n"
KO_DOC_PROMPT = "다음 한국어 기술 문서를 자연스럽게 이어 쓰세요:\n\n"

CODE_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".cc",
    ".h",
    ".hpp",
    ".cs",
    ".swift",
    ".kt",
    ".php",
    ".rb",
    ".sh",
}
DOC_EXTS = {".md", ".txt", ".rst"}


def normalize_text(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def decode_text_auto(raw: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp949", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def hangul_ratio(text: str) -> float:
    if not text:
        return 0.0
    k = 0
    for ch in text:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            k += 1
    return k / max(1, len(text))


def split_paragraphs(text: str, min_len: int = 70) -> List[str]:
    out: List[str] = []
    for p in re.split(r"\n\s*\n+", text):
        s = " ".join(p.split())
        if len(s) >= min_len:
            out.append(s)
    return out


def chunk_paragraphs(paragraphs: List[str], min_chars: int, target_chars: int) -> Iterator[str]:
    buf: List[str] = []
    cur = 0
    for p in paragraphs:
        add = len(p) + (2 if buf else 0)
        if cur >= min_chars and cur + add > target_chars:
            yield "\n\n".join(buf)
            buf = [p]
            cur = len(p)
            continue
        buf.append(p)
        cur += add
    if cur >= min_chars:
        yield "\n\n".join(buf)


def is_likely_binary(text: str) -> bool:
    if not text:
        return True
    if "\x00" in text:
        return True
    ctrl = sum(1 for ch in text if ord(ch) < 9 or (13 < ord(ch) < 32))
    return (ctrl / max(1, len(text))) > 0.02


def is_code_file(name: str) -> bool:
    return Path(name.lower()).suffix in CODE_EXTS


def is_doc_file(name: str) -> bool:
    p = Path(name.lower())
    return p.suffix in DOC_EXTS or p.name.lower().startswith("readme")


def parse_repo_from_zip_name(name: str) -> str:
    # owner__repo__branch.zip
    stem = Path(name).stem
    parts = stem.split("__")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return stem


def code_record_pair(repo: str, file_name: str, text: str, idx: int) -> tuple[Dict, Dict] | None:
    s = normalize_text(text).strip()
    if len(s) < 260 or len(s) > 60000:
        return None
    split_idx = int(len(s) * 0.55)
    split_idx = max(140, split_idx)
    split_idx = min(split_idx, len(s) - 140)
    if split_idx <= 0 or split_idx >= len(s):
        return None
    left = s[:split_idx].strip()
    right = s[split_idx:].strip()
    if len(left) < 120 or len(right) < 120:
        return None
    base_id = hashlib.sha1(f"{repo}|{file_name}|{idx}".encode("utf-8")).hexdigest()[:16]
    en = {
        "id": f"mit_all_code_en_{base_id}",
        "task_type": "code",
        "segment_tag": "code",
        "language": "en",
        "input": f"Continue the source file naturally. Preserve style and behavior.\n\n{left}",
        "output": right,
        "source_dataset": repo,
        "source_path": file_name,
        "license": "MIT",
        "source_url": f"https://github.com/{repo}",
    }
    ko = {
        "id": f"mit_all_code_ko_{base_id}",
        "task_type": "code",
        "segment_tag": "code",
        "language": "ko",
        "input": KO_CODE_PROMPT + left,
        "output": right,
        "source_dataset": repo,
        "source_path": file_name,
        "license": "MIT",
        "source_url": f"https://github.com/{repo}",
    }
    return en, ko


def korean_doc_records(repo: str, file_name: str, text: str, max_per_file: int = 24) -> List[Dict]:
    s = normalize_text(text).strip()
    if len(s) < 120 or len(s) > 200000:
        return []
    if hangul_ratio(s) < 0.12:
        return []
    paras = split_paragraphs(s, min_len=60)
    if not paras:
        return []

    out: List[Dict] = []
    j = 0
    for chunk in chunk_paragraphs(paras, min_chars=500, target_chars=1500):
        if hangul_ratio(chunk) < 0.18:
            continue
        split_idx = int(len(chunk) * 0.55)
        split_idx = max(120, split_idx)
        split_idx = min(split_idx, len(chunk) - 120)
        if split_idx <= 0 or split_idx >= len(chunk):
            continue
        left = chunk[:split_idx].strip()
        right = chunk[split_idx:].strip()
        if len(left) < 100 or len(right) < 100:
            continue
        rid = hashlib.sha1(f"{repo}|{file_name}|doc|{j}".encode("utf-8")).hexdigest()[:16]
        out.append(
            {
                "id": f"mit_all_ko_doc_{rid}",
                "task_type": "korean",
                "segment_tag": "ko",
                "language": "ko",
                "input": KO_DOC_PROMPT + left,
                "output": right,
                "source_dataset": repo,
                "source_path": file_name,
                "license": "MIT",
                "source_url": f"https://github.com/{repo}",
            }
        )
        j += 1
        if j >= max_per_file:
            break
    return out


def dedupe(records: Iterable[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for r in records:
        key = hashlib.sha1((r.get("task_type", "") + "\n" + r.get("input", "") + "\n" + r.get("output", "")).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_dir", default="data_archive_20260219_160805/sources_mit")
    parser.add_argument("--out_jsonl", default="data/slm_mit_all_local_v1.jsonl")
    parser.add_argument("--manifest", default="data/slm_mit_all_local_v1.manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_code_per_repo", type=int, default=3200)
    parser.add_argument("--max_korean_doc_per_repo", type=int, default=1400)
    args = parser.parse_args()

    random.seed(int(args.seed))
    zip_dir = Path(args.zip_dir)
    if not zip_dir.exists():
        raise RuntimeError(f"zip_dir not found: {zip_dir}")

    zips = sorted(zip_dir.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"no zip files in: {zip_dir}")

    records: List[Dict] = []
    repo_summary: Dict[str, Dict[str, int]] = {}

    for z in zips:
        repo = parse_repo_from_zip_name(z.name)
        counts = {"code_en": 0, "code_ko_prompt": 0, "korean_doc": 0}

        try:
            with zipfile.ZipFile(z, "r") as zf:
                idx = 0
                for name in sorted(zf.namelist()):
                    if name.endswith("/") or "/.git/" in name:
                        continue
                    low = name.lower()
                    if any(k in low for k in ["/node_modules/", "/dist/", "/build/", "/vendor/", "/coverage/", "/.next/", "/third_party/", "/deps/"]):
                        continue
                    try:
                        info = zf.getinfo(name)
                    except KeyError:
                        continue
                    if info.file_size <= 0 or info.file_size > MAX_RAW_FILE_BYTES:
                        continue
                    try:
                        raw = zf.read(name)
                    except Exception:
                        continue
                    text = decode_text_auto(raw)
                    if is_likely_binary(text):
                        continue

                    if is_code_file(name):
                        pair = code_record_pair(repo, name, text, idx)
                        if pair is not None:
                            en, ko = pair
                            if counts["code_en"] < int(args.max_code_per_repo):
                                records.append(en)
                                counts["code_en"] += 1
                            if counts["code_ko_prompt"] < int(args.max_code_per_repo):
                                records.append(ko)
                                counts["code_ko_prompt"] += 1
                            idx += 1

                    if is_doc_file(name):
                        docs = korean_doc_records(repo, name, text, max_per_file=24)
                        for d in docs:
                            if counts["korean_doc"] >= int(args.max_korean_doc_per_repo):
                                break
                            records.append(d)
                            counts["korean_doc"] += 1

                    if (
                        counts["code_en"] >= int(args.max_code_per_repo)
                        and counts["code_ko_prompt"] >= int(args.max_code_per_repo)
                        and counts["korean_doc"] >= int(args.max_korean_doc_per_repo)
                    ):
                        break
        except Exception:
            continue

        repo_summary[repo] = counts

    records = dedupe(records)
    random.shuffle(records)

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_task: Dict[str, int] = {}
    by_lang: Dict[str, int] = {}
    for r in records:
        by_task[r["task_type"]] = by_task.get(r["task_type"], 0) + 1
        by_lang[r["language"]] = by_lang.get(r["language"], 0) + 1

    manifest = {
        "zip_dir": str(zip_dir),
        "zip_count": len(zips),
        "output_jsonl": str(out_jsonl),
        "rows_total": len(records),
        "task_counts": by_task,
        "language_counts": by_lang,
        "license_policy": "MIT local sources only",
        "repo_summary": repo_summary,
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
