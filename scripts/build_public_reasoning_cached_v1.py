from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import pyarrow as pa
import pyarrow.ipc as ipc


CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "datasets"
KOBEST_BOOLQ_DIR = CACHE_ROOT / "skt___kobest_v1" / "boolq" / "0.0.0" / "a5ea15e3ac77ed694b79f6204eb31889a2ba989f"
KMMLU_ROOT = CACHE_ROOT / "HAERAE-HUB___kmmlu"
MMMLU_KO_DIR = CACHE_ROOT / "openai___mmmlu" / "KO_KR" / "0.0.0" / "325a01dc3e173cac1578df94120499aaca2e2504"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Korean public reasoning dataset from local HF arrow caches.")
    p.add_argument("--out_jsonl", type=Path, default=Path("data/public_reasoning_cached_v1.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/public_reasoning_cached_v1.manifest.json"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--kobest_limit", type=int, default=3600)
    p.add_argument("--kmmlu_limit", type=int, default=6000)
    p.add_argument("--mmmlu_limit", type=int, default=12000)
    return p.parse_args()


def normalize_space(text: str) -> str:
    s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def open_arrow_table(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as src:
        try:
            reader = ipc.open_file(src)
        except Exception:
            reader = ipc.open_stream(src)
        return reader.read_all()


def iter_rows(path: Path) -> Iterator[Dict]:
    table = open_arrow_table(path)
    data = table.to_pylist()
    for row in data:
        if isinstance(row, dict):
            yield row


def maybe_shuffle_limit(rows: List[Dict], limit: int, seed: int) -> List[Dict]:
    if len(rows) <= int(limit):
        return rows
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    return rows[: int(limit)]


def make_row(inp: str, out: str, source: str, task: str) -> Dict:
    return {
        "input": normalize_space(inp),
        "output": normalize_space(out),
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "domain": "ko",
        "source": str(source),
        "task": str(task),
    }


def build_kobest_boolq(limit: int, seed: int) -> List[Dict]:
    rows: List[Dict] = []
    for split_name in ("kobest_v1-train.arrow", "kobest_v1-validation.arrow"):
        path = KOBEST_BOOLQ_DIR / split_name
        if not path.exists():
            continue
        for row in iter_rows(path):
            paragraph = normalize_space(row.get("paragraph", ""))
            question = normalize_space(row.get("question", ""))
            label = row.get("label", None)
            if not paragraph or not question:
                continue
            if label not in {0, 1}:
                continue
            answer = "예." if int(label) == 1 else "아니오."
            inp = (
                "다음 문단을 읽고 질문에 예 또는 아니오로 답하라.\n"
                f"문단: {paragraph}\n"
                f"질문: {question}"
            )
            rows.append(make_row(inp=inp, out=answer, source="hf_cache/skt_kobest_v1_boolq", task="boolq"))
    return maybe_shuffle_limit(rows, limit=int(limit), seed=int(seed))


def kmmlu_answer_letter(value) -> str:
    try:
        idx = int(value)
    except Exception:
        return ""
    letters = ("A", "B", "C", "D")
    if 0 <= idx < len(letters):
        return letters[idx]
    return ""


def kmmlu_answer_text(row: Dict, letter: str) -> str:
    return normalize_space(row.get(letter, "")) if letter in {"A", "B", "C", "D"} else ""


def build_kmmlu(limit: int, seed: int) -> List[Dict]:
    rows: List[Dict] = []
    if not KMMLU_ROOT.exists():
        return rows
    for cfg_dir in sorted(p for p in KMMLU_ROOT.iterdir() if p.is_dir()):
        pattern = list(cfg_dir.glob("0.0.0/*/kmmlu-train.arrow"))
        if not pattern:
            pattern = list(cfg_dir.glob("0.0.0/*/kmmlu-dev.arrow"))
        if not pattern:
            continue
        for path in pattern[:1]:
            for row in iter_rows(path):
                q = normalize_space(row.get("question", ""))
                a = normalize_space(row.get("A", ""))
                b = normalize_space(row.get("B", ""))
                c = normalize_space(row.get("C", ""))
                d = normalize_space(row.get("D", ""))
                letter = kmmlu_answer_letter(row.get("answer"))
                answer_text = kmmlu_answer_text(row, letter)
                if not q or not a or not b or not c or not d or not letter or not answer_text:
                    continue
                inp = (
                    "다음 객관식 문제를 풀어라. 정답 하나를 고르라.\n"
                    f"문제: {q}\n"
                    f"A. {a}\n"
                    f"B. {b}\n"
                    f"C. {c}\n"
                    f"D. {d}"
                )
                out = f"정답은 {letter}입니다. {answer_text}"
                src = f"hf_cache/kmmlu/{cfg_dir.name}"
                rows.append(make_row(inp=inp, out=out, source=src, task="multiple_choice"))
    return maybe_shuffle_limit(rows, limit=int(limit), seed=int(seed))


def build_mmmlu_ko(limit: int, seed: int) -> List[Dict]:
    rows: List[Dict] = []
    for split_name in ("mmmlu-test.arrow",):
        path = MMMLU_KO_DIR / split_name
        if not path.exists():
            continue
        for row in iter_rows(path):
            q = normalize_space(row.get("Question", ""))
            a = normalize_space(row.get("A", ""))
            b = normalize_space(row.get("B", ""))
            c = normalize_space(row.get("C", ""))
            d = normalize_space(row.get("D", ""))
            letter = normalize_space(row.get("Answer", "")).upper()
            if letter not in {"A", "B", "C", "D"}:
                continue
            ans_text = normalize_space(row.get(letter, ""))
            if not q or not a or not b or not c or not d or not ans_text:
                continue
            subject = normalize_space(row.get("Subject", ""))
            prefix = "다음 객관식 문제를 풀어라. 정답 하나를 고르라."
            if subject:
                prefix += f" 분야: {subject}."
            inp = (
                f"{prefix}\n"
                f"문제: {q}\n"
                f"A. {a}\n"
                f"B. {b}\n"
                f"C. {c}\n"
                f"D. {d}"
            )
            out = f"정답은 {letter}입니다. {ans_text}"
            rows.append(make_row(inp=inp, out=out, source="hf_cache/mmmlu_ko_kr", task="multiple_choice"))
    return maybe_shuffle_limit(rows, limit=int(limit), seed=int(seed))


def dedupe_rows(rows: Iterable[Dict]) -> Tuple[List[Dict], Counter]:
    kept: List[Dict] = []
    seen: set[Tuple[str, str]] = set()
    stats: Counter = Counter()
    for row in rows:
        inp = normalize_space(row.get("input", ""))
        out = normalize_space(row.get("output", ""))
        if len(inp) < 8 or len(out) < 2:
            stats["short"] += 1
            continue
        key = (inp, out)
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept, stats


def main() -> None:
    args = parse_args()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    source_rows: Dict[str, List[Dict]] = {
        "kobest_boolq": build_kobest_boolq(limit=int(args.kobest_limit), seed=int(args.seed) + 11),
        "kmmlu": build_kmmlu(limit=int(args.kmmlu_limit), seed=int(args.seed) + 23),
        "mmmlu_ko": build_mmmlu_ko(limit=int(args.mmmlu_limit), seed=int(args.seed) + 37),
    }

    merged: List[Dict] = []
    for key in ("kobest_boolq", "kmmlu", "mmmlu_ko"):
        merged.extend(source_rows[key])

    deduped, dropped = dedupe_rows(merged)
    random.Random(int(args.seed)).shuffle(deduped)

    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "out_jsonl": str(args.out_jsonl),
        "rows_total": int(len(deduped)),
        "rows_by_source": {k: int(len(v)) for k, v in source_rows.items()},
        "dropped": {k: int(v) for k, v in dropped.items()},
        "limits": {
            "kobest_limit": int(args.kobest_limit),
            "kmmlu_limit": int(args.kmmlu_limit),
            "mmmlu_limit": int(args.mmmlu_limit),
        },
        "cache_roots": {
            "kobest_boolq": str(KOBEST_BOOLQ_DIR),
            "kmmlu_root": str(KMMLU_ROOT),
            "mmmlu_ko": str(MMMLU_KO_DIR),
        },
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
