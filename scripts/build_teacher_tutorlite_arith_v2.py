from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List


TRAIN_OUT = Path("data/teacher_tutorlite_arith_v2_train.jsonl")
EVAL_OUT = Path("data/teacher_tutorlite_arith_v2_eval.jsonl")
MANIFEST_OUT = Path("data/teacher_tutorlite_arith_v2.manifest.json")


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_row(prompt: str, final_answer: str, category: str, split: str) -> Dict[str, str]:
    answer = str(final_answer).strip()
    return {
        "input": prompt.strip(),
        "output": f"최종답: {answer}",
        "task_type": "korean",
        "segment_tag": "ko_teacher_tutorlite_arith_v2",
        "language": "ko",
        "category": category,
        "split": split,
        "final_answer": answer,
        "answer_format": "최종답: ...",
        "_meta_source_file": "teacher_tutorlite_arith_v2",
    }


def tutorlite_prompt(problem: str, hint1: str, hint2: str) -> str:
    return (
        f"문제: {problem}\n"
        f"힌트1: {hint1}\n"
        f"힌트2: {hint2}\n"
        "마지막 줄에 최종답: ... 형식으로만 답하라."
    )


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
        cat = f"tutorlite_addsub_d{digits}"
        if op == "+":
            hint1 = f"{a}와 {b}를 합치면 된다."
            hint2 = f"{a} + {b} = {ans}"
        else:
            hint1 = f"{a}에서 {b}를 빼면 된다."
            hint2 = f"{a} - {b} = {ans}"
        prompt = tutorlite_prompt(f"{a} {op} {b}", hint1, hint2)
        rows.append(make_row(prompt, str(ans), cat, split))
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
        ans = a * b
        cat = f"tutorlite_mul_{a_digits}x{b_digits}"
        hint1 = f"{a}를 {b}번 더한 값과 같다."
        hint2 = f"{a} * {b} = {ans}"
        prompt = tutorlite_prompt(f"{a} * {b}", hint1, hint2)
        rows.append(make_row(prompt, str(ans), cat, split))
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
        cat = f"tutorlite_div_{len(str(a))}by{b_digits}"
        hint1 = f"{b}에 어떤 수를 곱하면 {a}가 되는지 찾으면 된다."
        hint2 = f"{b} * {q} = {a} 이므로 {a} / {b} = {q}"
        prompt = tutorlite_prompt(f"{a} / {b}", hint1, hint2)
        rows.append(make_row(prompt, str(q), cat, split))
    return rows


def gen_linear(rng: random.Random, split: str, x_hi: int, add_hi: int, mul_hi: int, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(n):
        x = rng.randint(1, x_hi)
        add_v = rng.randint(1, add_hi)
        mul_v = rng.randint(2, mul_hi)
        out = (x + add_v) * mul_v
        cat = f"tutorlite_linear_x{x_hi}"
        problem = f"어떤 수에 {add_v}를 더하고 {mul_v}를 곱하면 {out}이다. 그 수는?"
        hint1 = f"먼저 {out}을 {mul_v}로 나눈 뒤 {add_v}를 빼면 된다."
        hint2 = f"{out} / {mul_v} = {x + add_v}, 여기서 {add_v}를 빼면 {x}"
        prompt = tutorlite_prompt(problem, hint1, hint2)
        rows.append(make_row(prompt, str(x), cat, split))
    return rows


def build_train(seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    rows.extend(gen_addsub(rng, "train", 1, 12000))
    rows.extend(gen_addsub(rng, "train", 2, 12000))
    rows.extend(gen_mul(rng, "train", 1, 1, 14000))
    rows.extend(gen_mul(rng, "train", 2, 1, 10000))
    rows.extend(gen_div(rng, "train", 1, 1, 12000))
    rows.extend(gen_div(rng, "train", 1, 2, 10000))
    rows.extend(gen_linear(rng, "train", 20, 10, 5, 10000))
    rng.shuffle(rows)
    return rows


def build_eval(seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    rows.extend(gen_addsub(rng, "eval", 1, 48))
    rows.extend(gen_addsub(rng, "eval", 2, 48))
    rows.extend(gen_mul(rng, "eval", 1, 1, 48))
    rows.extend(gen_mul(rng, "eval", 2, 1, 48))
    rows.extend(gen_div(rng, "eval", 1, 1, 48))
    rows.extend(gen_div(rng, "eval", 1, 2, 48))
    rows.extend(gen_linear(rng, "eval", 20, 10, 5, 48))
    rng.shuffle(rows)
    return rows


def main() -> None:
    train_rows = build_train(20260307)
    eval_rows = build_eval(20260308)
    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(EVAL_OUT, eval_rows)
    manifest = {
        "seed_train": 20260307,
        "seed_eval": 20260308,
        "train_written": len(train_rows),
        "eval_written": len(eval_rows),
        "output_format": "최종답: ...",
        "prompt_style": "tutorlite_v2",
        "focus": ["addsub_d1", "addsub_d2", "mul_1x1", "mul_2x1", "div_small", "div_mid", "linear_easy"],
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
