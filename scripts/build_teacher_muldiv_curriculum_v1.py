from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List


TRAIN_OUT = Path("data/teacher_muldiv_curriculum_v1_train.jsonl")
EVAL_OUT = Path("data/teacher_muldiv_curriculum_v1_eval.jsonl")
MANIFEST_OUT = Path("data/teacher_muldiv_curriculum_v1.manifest.json")

LAST_LINE_ONLY = "\ub9c8\uc9c0\ub9c9 \uc904\uc5d0 \ucd5c\uc885\ub2f5\ub9cc \uc368\ub77c."
FINAL_PREFIX = "\ucd5c\uc885\ub2f5: "
OUTPUT_FORMAT = "\ucd5c\uc885\ub2f5: ..."
PROMPT_CALC = "\ub2e4\uc74c\uc744 \uacc4\uc0b0\ud558\ub77c."
PROMPT_LINEAR_FMT = (
    "\uc5b4\ub5a4 \uc218\uc5d0 {add_v}\ub97c \ub354\ud558\uace0 {mul_v}\ub97c \uacf1\ud558\uba74 "
    "{out}\uc774 \ub41c\ub2e4. \uadf8 \uc218\ub97c \uad6c\ud558\ub77c."
)


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_row(prompt: str, final_answer: str, category: str, split: str) -> Dict[str, str]:
    answer = str(final_answer).strip()
    return {
        "input": f"{prompt.strip()}\n{LAST_LINE_ONLY}",
        "output": f"{FINAL_PREFIX}{answer}",
        "task_type": "korean",
        "segment_tag": "ko_teacher_muldiv_curriculum_v1",
        "language": "ko",
        "category": category,
        "split": split,
        "final_answer": answer,
        "answer_format": OUTPUT_FORMAT,
        "_meta_source_file": "teacher_muldiv_curriculum_v1",
    }


def gen_addsub(rng: random.Random, split: str, digits: int, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    lo = 1 if digits == 1 else 10 ** (digits - 1)
    hi = (10**digits) - 1
    for _ in range(n):
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        op = rng.choice(["+", "-"])
        if op == "-" and b > a:
            a, b = b, a
        ans = a + b if op == "+" else a - b
        rows.append(make_row(f"{PROMPT_CALC} {a} {op} {b}", str(ans), f"addsub_d{digits}", split))
    return rows


def gen_mul(rng: random.Random, split: str, a_digits: int, b_digits: int, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    a_lo = 1 if a_digits == 1 else 10 ** (a_digits - 1)
    a_hi = (10**a_digits) - 1
    b_lo = 1 if b_digits == 1 else 10 ** (b_digits - 1)
    b_hi = (10**b_digits) - 1
    for _ in range(n):
        a = rng.randint(a_lo, a_hi)
        b = rng.randint(b_lo, b_hi)
        rows.append(make_row(f"{PROMPT_CALC} {a} * {b}", str(a * b), f"mul_{a_digits}x{b_digits}", split))
    return rows


def gen_div(rng: random.Random, split: str, b_digits: int, q_digits: int, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    b_lo = 2 if b_digits == 1 else 10 ** (b_digits - 1)
    b_hi = (10**b_digits) - 1
    q_lo = 1 if q_digits == 1 else 10 ** (q_digits - 1)
    q_hi = (10**q_digits) - 1
    for _ in range(n):
        b = rng.randint(b_lo, b_hi)
        q = rng.randint(q_lo, q_hi)
        a = b * q
        rows.append(make_row(f"{PROMPT_CALC} {a} / {b}", str(q), f"div_{len(str(a))}by{b_digits}", split))
    return rows


def gen_linear(rng: random.Random, split: str, x_hi: int, add_hi: int, mul_hi: int, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(n):
        x = rng.randint(1, x_hi)
        add_v = rng.randint(1, add_hi)
        mul_v = rng.randint(2, mul_hi)
        out = (x + add_v) * mul_v
        prompt = PROMPT_LINEAR_FMT.format(add_v=add_v, mul_v=mul_v, out=out)
        rows.append(make_row(prompt, str(x), f"linear_x{x_hi}", split))
    return rows


def build_train(seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    rows.extend(gen_mul(rng, "train", 1, 1, 36000))
    rows.extend(gen_mul(rng, "train", 2, 1, 28000))
    rows.extend(gen_mul(rng, "train", 2, 2, 22000))
    rows.extend(gen_div(rng, "train", 1, 1, 18000))
    rows.extend(gen_div(rng, "train", 1, 2, 22000))
    rows.extend(gen_div(rng, "train", 2, 2, 18000))
    rows.extend(gen_addsub(rng, "train", 1, 6000))
    rows.extend(gen_addsub(rng, "train", 2, 6000))
    rows.extend(gen_linear(rng, "train", 20, 10, 5, 6000))
    rng.shuffle(rows)
    return rows


def build_eval(seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    rows.extend(gen_mul(rng, "eval", 1, 1, 64))
    rows.extend(gen_mul(rng, "eval", 2, 1, 64))
    rows.extend(gen_mul(rng, "eval", 2, 2, 64))
    rows.extend(gen_div(rng, "eval", 1, 1, 32))
    rows.extend(gen_div(rng, "eval", 1, 2, 64))
    rows.extend(gen_div(rng, "eval", 2, 2, 64))
    rows.extend(gen_addsub(rng, "eval", 1, 32))
    rows.extend(gen_addsub(rng, "eval", 2, 32))
    rows.extend(gen_linear(rng, "eval", 20, 10, 5, 32))
    rng.shuffle(rows)
    return rows


def main() -> None:
    train_rows = build_train(20260308)
    eval_rows = build_eval(20260309)
    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(EVAL_OUT, eval_rows)
    manifest = {
        "seed_train": 20260308,
        "seed_eval": 20260309,
        "train_written": len(train_rows),
        "eval_written": len(eval_rows),
        "output_format": OUTPUT_FORMAT,
        "prompt_style": "muldiv_curriculum_v1",
        "focus": {
            "mul_1x1": 36000,
            "mul_2x1": 28000,
            "mul_2x2": 22000,
            "div_1by1": 18000,
            "div_2by1": 22000,
            "div_4by2": 18000,
            "replay_addsub_d1": 6000,
            "replay_addsub_d2": 6000,
            "replay_linear_x20": 6000,
        },
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
