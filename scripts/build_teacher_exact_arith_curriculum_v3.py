from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List


TRAIN_OUT = Path("data/teacher_exact_arith_curriculum_v3_train.jsonl")
EVAL_OUT = Path("data/teacher_exact_arith_curriculum_v3_eval.jsonl")
MANIFEST_OUT = Path("data/teacher_exact_arith_curriculum_v3.manifest.json")

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
        "segment_tag": "ko_exact_arith_curriculum_v3",
        "language": "ko",
        "category": category,
        "split": split,
        "final_answer": answer,
        "answer_format": OUTPUT_FORMAT,
        "_meta_source_file": "teacher_exact_arith_curriculum_v3",
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
        cat = f"addsub_d{digits}"
        rows.append(make_row(f"{PROMPT_CALC} {a} {op} {b}", str(ans), cat, split))
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
        cat = f"mul_{a_digits}x{b_digits}"
        rows.append(make_row(f"{PROMPT_CALC} {a} * {b}", str(a * b), cat, split))
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
        cat = f"div_{len(str(a))}by{b_digits}"
        rows.append(make_row(f"{PROMPT_CALC} {a} / {b}", str(q), cat, split))
    return rows


def gen_linear(rng: random.Random, split: str, x_hi: int, add_hi: int, mul_hi: int, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(n):
        x = rng.randint(1, x_hi)
        add_v = rng.randint(1, add_hi)
        mul_v = rng.randint(2, mul_hi)
        out = (x + add_v) * mul_v
        cat = f"linear_x{x_hi}"
        prompt = PROMPT_LINEAR_FMT.format(add_v=add_v, mul_v=mul_v, out=out)
        rows.append(make_row(prompt, str(x), cat, split))
    return rows


def build_train(seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    rows.extend(gen_addsub(rng, "train", digits=1, n=20000))
    rows.extend(gen_addsub(rng, "train", digits=2, n=20000))
    rows.extend(gen_addsub(rng, "train", digits=3, n=12000))
    rows.extend(gen_mul(rng, "train", a_digits=1, b_digits=1, n=24000))
    rows.extend(gen_mul(rng, "train", a_digits=2, b_digits=1, n=18000))
    rows.extend(gen_mul(rng, "train", a_digits=2, b_digits=2, n=10000))
    rows.extend(gen_div(rng, "train", b_digits=1, q_digits=1, n=22000))
    rows.extend(gen_div(rng, "train", b_digits=1, q_digits=2, n=16000))
    rows.extend(gen_div(rng, "train", b_digits=2, q_digits=2, n=8000))
    rows.extend(gen_linear(rng, "train", x_hi=20, add_hi=10, mul_hi=5, n=16000))
    rows.extend(gen_linear(rng, "train", x_hi=60, add_hi=15, mul_hi=7, n=10000))
    rng.shuffle(rows)
    return rows


def build_eval(seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    rows.extend(gen_addsub(rng, "eval", digits=1, n=64))
    rows.extend(gen_addsub(rng, "eval", digits=2, n=64))
    rows.extend(gen_addsub(rng, "eval", digits=3, n=64))
    rows.extend(gen_mul(rng, "eval", a_digits=1, b_digits=1, n=64))
    rows.extend(gen_mul(rng, "eval", a_digits=2, b_digits=1, n=64))
    rows.extend(gen_mul(rng, "eval", a_digits=2, b_digits=2, n=64))
    rows.extend(gen_div(rng, "eval", b_digits=1, q_digits=1, n=64))
    rows.extend(gen_div(rng, "eval", b_digits=1, q_digits=2, n=64))
    rows.extend(gen_div(rng, "eval", b_digits=2, q_digits=2, n=64))
    rows.extend(gen_linear(rng, "eval", x_hi=20, add_hi=10, mul_hi=5, n=64))
    rows.extend(gen_linear(rng, "eval", x_hi=60, add_hi=15, mul_hi=7, n=64))
    rng.shuffle(rows)
    return rows


def main() -> None:
    train_rows = build_train(seed=20260305)
    eval_rows = build_eval(seed=20260306)

    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(EVAL_OUT, eval_rows)

    manifest = {
        "seed_train": 20260305,
        "seed_eval": 20260306,
        "train_written": len(train_rows),
        "eval_written": len(eval_rows),
        "output_format": OUTPUT_FORMAT,
        "prompt_style": "arith_curriculum_v3",
        "train_mix": {
            "addsub_d1": 20000,
            "addsub_d2": 20000,
            "addsub_d3": 12000,
            "mul_1x1": 24000,
            "mul_2x1": 18000,
            "mul_2x2": 10000,
            "div_small": 22000,
            "div_mid": 16000,
            "div_2digit": 8000,
            "linear_easy": 16000,
            "linear_mid": 10000,
        },
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
