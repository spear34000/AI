from __future__ import annotations

import argparse
import hashlib
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
PAWSX_KO_DIR = CACHE_ROOT / "google-research-datasets___paws-x" / "ko" / "0.0.0" / "4cd8187c404bda33cb1f62b49b001115862acf37"
KMMLU_ROOT = CACHE_ROOT / "HAERAE-HUB___kmmlu"
MMMLU_KO_DIR = CACHE_ROOT / "openai___mmmlu" / "KO_KR" / "0.0.0" / "325a01dc3e173cac1578df94120499aaca2e2504"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build diversified Korean reasoning dataset from cached HF arrow files.")
    p.add_argument("--out_jsonl", type=Path, default=Path("data/public_reasoning_cached_v2.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/public_reasoning_cached_v2.manifest.json"))
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--kobest_limit", type=int, default=5000)
    p.add_argument("--pawsx_limit", type=int, default=7000)
    p.add_argument("--kmmlu_limit", type=int, default=12000)
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
    for row in table.to_pylist():
        if isinstance(row, dict):
            yield row


def stable_mod(text: str, mod: int) -> int:
    h = hashlib.sha1(str(text).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % int(max(1, mod))


def maybe_trim(text: str, max_chars: int) -> str:
    s = normalize_space(text)
    if len(s) <= int(max_chars):
        return s
    return s[: int(max_chars)].rstrip()


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


def reservoir_extend(rows: List[Dict], buf: List[Dict], limit: int, seed: int) -> List[Dict]:
    rng = random.Random(int(seed))
    for idx, row in enumerate(buf):
        if len(rows) < int(limit):
            rows.append(row)
            continue
        j = rng.randint(0, idx)
        if j < int(limit):
            rows[j] = row
    rng.shuffle(rows)
    return rows[: int(limit)]


def bool_answer(label: int) -> str:
    return "예" if int(label) == 1 else "아니오"


def bool_explained(label: int) -> str:
    if int(label) == 1:
        return "예. 문단의 내용과 일치합니다."
    return "아니오. 문단의 내용과 일치하지 않습니다."


def build_kobest_boolq(limit: int, seed: int) -> List[Dict]:
    rows: List[Dict] = []
    buf: List[Dict] = []
    for split_name in ("kobest_v1-train.arrow", "kobest_v1-validation.arrow"):
        path = KOBEST_BOOLQ_DIR / split_name
        if not path.exists():
            continue
        for row in iter_rows(path):
            paragraph = maybe_trim(row.get("paragraph", ""), 700)
            question = normalize_space(row.get("question", ""))
            label = row.get("label", None)
            if not paragraph or not question or label not in {0, 1}:
                continue
            yn = bool_answer(int(label))
            explained = bool_explained(int(label))
            buf.append(
                make_row(
                    inp=(
                        "다음 문단을 읽고 질문에 예 또는 아니오만 답하라.\n"
                        f"문단: {paragraph}\n"
                        f"질문: {question}"
                    ),
                    out=yn,
                    source="hf_cache/skt_kobest_v1_boolq",
                    task="boolq",
                )
            )
            buf.append(
                make_row(
                    inp=(
                        "다음 문단을 읽고 질문에 한 문장으로 답하라.\n"
                        f"문단: {paragraph}\n"
                        f"질문: {question}"
                    ),
                    out=explained,
                    source="hf_cache/skt_kobest_v1_boolq",
                    task="boolq",
                )
            )
    return reservoir_extend(rows, buf, limit=int(limit), seed=int(seed))


def build_pawsx_ko(limit: int, seed: int) -> List[Dict]:
    rows: List[Dict] = []
    buf: List[Dict] = []
    for split_name in ("paws-x-train.arrow", "paws-x-validation.arrow"):
        path = PAWSX_KO_DIR / split_name
        if not path.exists():
            continue
        for row in iter_rows(path):
            s1 = maybe_trim(row.get("sentence1", ""), 280)
            s2 = maybe_trim(row.get("sentence2", ""), 280)
            label = row.get("label", None)
            if not s1 or not s2 or label not in {0, 1}:
                continue
            yn = bool_answer(int(label))
            explained = "예. 두 문장은 같은 의미입니다." if int(label) == 1 else "아니오. 두 문장의 의미가 다릅니다."
            buf.append(
                make_row(
                    inp=(
                        "다음 두 문장이 같은 의미인지 판단하라. 예 또는 아니오만 답하라.\n"
                        f"문장1: {s1}\n"
                        f"문장2: {s2}"
                    ),
                    out=yn,
                    source="hf_cache/paws_x_ko",
                    task="sentence_relation",
                )
            )
            buf.append(
                make_row(
                    inp=(
                        "다음 두 문장의 의미 관계를 한 문장으로 답하라.\n"
                        f"문장1: {s1}\n"
                        f"문장2: {s2}"
                    ),
                    out=explained,
                    source="hf_cache/paws_x_ko",
                    task="sentence_relation",
                )
            )
    return reservoir_extend(rows, buf, limit=int(limit), seed=int(seed))


def answer_letter_from_index(value) -> str:
    try:
        idx = int(value)
    except Exception:
        return ""
    letters = ("A", "B", "C", "D")
    if 0 <= idx < len(letters):
        return letters[idx]
    return ""


def mcq_templates(question: str, options: Dict[str, str], answer_letter: str, answer_text: str, subject: str, source: str) -> List[Dict]:
    subject_txt = f" 분야: {subject}." if subject else ""
    option_block = "\n".join(f"{k}. {v}" for k, v in options.items())
    rows: List[Dict] = []
    rows.append(
        make_row(
            inp=(
                f"다음 객관식 문제를 풀고 선택지 문자만 답하라.{subject_txt}\n"
                f"문제: {question}\n{option_block}"
            ),
            out=answer_letter,
            source=source,
            task="multiple_choice",
        )
    )
    rows.append(
        make_row(
            inp=(
                f"다음 객관식 문제를 풀고 정답 내용을 짧게 답하라.{subject_txt}\n"
                f"문제: {question}\n{option_block}"
            ),
            out=answer_text,
            source=source,
            task="multiple_choice",
        )
    )
    rows.append(
        make_row(
            inp=(
                f"다음 객관식 문제를 풀어 한 줄로 답하라.{subject_txt}\n"
                f"문제: {question}\n{option_block}"
            ),
            out=f"{answer_letter}. {answer_text}",
            source=source,
            task="multiple_choice",
        )
    )
    return rows


def build_kmmlu(limit: int, seed: int) -> List[Dict]:
    rows: List[Dict] = []
    buf: List[Dict] = []
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
                q = maybe_trim(row.get("question", ""), 420)
                options = {k: maybe_trim(row.get(k, ""), 180) for k in ("A", "B", "C", "D")}
                answer_letter = answer_letter_from_index(row.get("answer"))
                answer_text = normalize_space(options.get(answer_letter, ""))
                if not q or not answer_letter or not answer_text or not all(options.values()):
                    continue
                buf.extend(
                    mcq_templates(
                        question=q,
                        options=options,
                        answer_letter=answer_letter,
                        answer_text=answer_text,
                        subject=str(cfg_dir.name),
                        source=f"hf_cache/kmmlu/{cfg_dir.name}",
                    )
                )
    return reservoir_extend(rows, buf, limit=int(limit), seed=int(seed))


def build_mmmlu_ko(limit: int, seed: int) -> List[Dict]:
    rows: List[Dict] = []
    buf: List[Dict] = []
    path = MMMLU_KO_DIR / "mmmlu-test.arrow"
    if not path.exists():
        return rows
    for row in iter_rows(path):
        q = maybe_trim(row.get("Question", ""), 420)
        options = {k: maybe_trim(row.get(k, ""), 180) for k in ("A", "B", "C", "D")}
        answer_letter = normalize_space(row.get("Answer", "")).upper()
        answer_text = normalize_space(options.get(answer_letter, ""))
        if not q or answer_letter not in {"A", "B", "C", "D"} or not answer_text or not all(options.values()):
            continue
        buf.extend(
            mcq_templates(
                question=q,
                options=options,
                answer_letter=answer_letter,
                answer_text=answer_text,
                subject=normalize_space(row.get("Subject", "")),
                source="hf_cache/mmmlu_ko_kr",
            )
        )
    return reservoir_extend(rows, buf, limit=int(limit), seed=int(seed))


def dedupe_rows(rows: Iterable[Dict]) -> Tuple[List[Dict], Counter]:
    kept: List[Dict] = []
    seen: set[Tuple[str, str]] = set()
    stats: Counter = Counter()
    for row in rows:
        inp = normalize_space(row.get("input", ""))
        out = normalize_space(row.get("output", ""))
        if len(inp) < 8 or len(out) < 1:
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
        "pawsx_ko": build_pawsx_ko(limit=int(args.pawsx_limit), seed=int(args.seed) + 17),
        "kmmlu": build_kmmlu(limit=int(args.kmmlu_limit), seed=int(args.seed) + 23),
        "mmmlu_ko": build_mmmlu_ko(limit=int(args.mmmlu_limit), seed=int(args.seed) + 31),
    }

    merged: List[Dict] = []
    for key in ("kobest_boolq", "pawsx_ko", "kmmlu", "mmmlu_ko"):
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
            "pawsx_limit": int(args.pawsx_limit),
            "kmmlu_limit": int(args.kmmlu_limit),
            "mmmlu_limit": int(args.mmmlu_limit),
        },
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
