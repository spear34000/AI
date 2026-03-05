from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


TRAIN_OUT = Path("data/teacher_arith_focus_v4_train.jsonl")
EVAL_OUT = Path("data/teacher_arith_focus_v4_eval.jsonl")
MANIFEST_OUT = Path("data/teacher_arith_focus_v4.manifest.json")

LAST_LINE_ONLY = "최종답만 써라."
LAST_LINE_ONLY_LONG = "마지막 줄에 최종답만 써라."
FINAL_PREFIX = "최종답: "
OUTPUT_FORMAT = "최종답: ..."
PROMPT_CALC = "다음을 계산하라."


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_row(prompt: str, final_answer: str, category: str, split: str, style: str) -> Dict[str, str]:
    answer = str(final_answer).strip()
    return {
        "input": prompt.strip(),
        "output": f"{FINAL_PREFIX}{answer}",
        "task_type": "korean",
        "segment_tag": "ko_teacher_arith_focus_v4",
        "language": "ko",
        "category": category,
        "split": split,
        "prompt_style": style,
        "final_answer": answer,
        "answer_format": OUTPUT_FORMAT,
        "_meta_source_file": "teacher_arith_focus_v4",
    }


def prompt_variants(expr: str, a: int, op: str, b: int) -> List[Tuple[str, str]]:
    op_word = {"+": "더하기", "-": "빼기", "*": "곱하기"}[op]
    return [
        (f"{expr}=\n{LAST_LINE_ONLY}", "symbolic"),
        (f"{PROMPT_CALC} {a} {op} {b}\n{LAST_LINE_ONLY_LONG}", "korean_calc"),
        (f"{a} {op_word} {b}의 값은?\n{LAST_LINE_ONLY}", "korean_short"),
    ]


def add_rows(rows: List[Dict[str, str]], prompts: List[Tuple[str, str]], answer: int, category: str, split: str) -> None:
    for prompt, style in prompts:
        rows.append(make_row(prompt, str(answer), category, split, style))


def gen_add_carry(rng: random.Random, split: str, digits: int, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    lo = 10 ** (digits - 1)
    hi = (10**digits) - 1
    while len(rows) < n * 3:
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        if (a % 10) + (b % 10) < 10:
            continue
        add_rows(rows, prompt_variants(f"{a}+{b}", a, "+", b), a + b, f"add_carry_d{digits}", split)
    return rows


def gen_sub_borrow(rng: random.Random, split: str, digits: int, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    lo = 10 ** (digits - 1)
    hi = (10**digits) - 1
    while len(rows) < n * 3:
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        if a < b:
            a, b = b, a
        if (a % 10) >= (b % 10):
            continue
        add_rows(rows, prompt_variants(f"{a}-{b}", a, "-", b), a - b, f"sub_borrow_d{digits}", split)
    return rows


def gen_mul_2x2(rng: random.Random, split: str, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(n):
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        add_rows(rows, prompt_variants(f"{a}*{b}", a, "*", b), a * b, "mul_2x2_focus", split)
    return rows


def gen_mul_2x2_hard(rng: random.Random, split: str, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    while len(rows) < n * 3:
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        if (a % 10) * (b % 10) < 10:
            continue
        add_rows(rows, prompt_variants(f"{a}*{b}", a, "*", b), a * b, "mul_2x2_hard", split)
    return rows


def gen_easy_replay(rng: random.Random, split: str, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(n):
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        add_rows(rows, prompt_variants(f"{a}+{b}", a, "+", b), a + b, "replay_addsub_d1", split)
        add_rows(rows, prompt_variants(f"{a}*{b}", a, "*", b), a * b, "replay_mul_1x1", split)
    return rows


def build_train(seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    rows.extend(gen_add_carry(rng, "train", 2, 16000))
    rows.extend(gen_add_carry(rng, "train", 3, 16000))
    rows.extend(gen_sub_borrow(rng, "train", 2, 16000))
    rows.extend(gen_sub_borrow(rng, "train", 3, 16000))
    rows.extend(gen_mul_2x2(rng, "train", 22000))
    rows.extend(gen_mul_2x2_hard(rng, "train", 18000))
    rows.extend(gen_easy_replay(rng, "train", 6000))
    rng.shuffle(rows)
    return rows


def build_eval(seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    rows.extend(gen_add_carry(rng, "eval", 2, 64))
    rows.extend(gen_add_carry(rng, "eval", 3, 64))
    rows.extend(gen_sub_borrow(rng, "eval", 2, 64))
    rows.extend(gen_sub_borrow(rng, "eval", 3, 64))
    rows.extend(gen_mul_2x2(rng, "eval", 96))
    rows.extend(gen_mul_2x2_hard(rng, "eval", 96))
    rows.extend(gen_easy_replay(rng, "eval", 32))
    rng.shuffle(rows)
    return rows


def main() -> None:
    train_rows = build_train(20260317)
    eval_rows = build_eval(20260318)
    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(EVAL_OUT, eval_rows)
    manifest = {
        "seed_train": 20260317,
        "seed_eval": 20260318,
        "train_written": len(train_rows),
        "eval_written": len(eval_rows),
        "output_format": OUTPUT_FORMAT,
        "focus": {
            "add_carry_d2": 16000,
            "add_carry_d3": 16000,
            "sub_borrow_d2": 16000,
            "sub_borrow_d3": 16000,
            "mul_2x2_focus": 22000,
            "mul_2x2_hard": 18000,
            "easy_replay": 6000,
        },
        "prompt_styles": ["symbolic", "korean_calc", "korean_short"],
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
