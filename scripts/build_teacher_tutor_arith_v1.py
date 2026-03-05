from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List


TRAIN_OUT = Path("data/teacher_tutor_arith_v1_train.jsonl")
EVAL_OUT = Path("data/teacher_tutor_arith_v1_eval.jsonl")
MANIFEST_OUT = Path("data/teacher_tutor_arith_v1.manifest.json")


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_row(dialogue: str, final_answer: str, category: str, split: str) -> Dict[str, str]:
    answer = str(final_answer).strip()
    return {
        "input": dialogue.strip(),
        "output": f"학생: 최종답: {answer}",
        "task_type": "korean",
        "segment_tag": "ko_tutor_arith_v1",
        "language": "ko",
        "category": category,
        "split": split,
        "final_answer": answer,
        "answer_format": "학생: 최종답: ...",
        "_meta_source_file": "teacher_tutor_arith_v1",
    }


def tutor_template(problem: str, hint1: str, try1: str, hint2: str, try2: str, answer: str) -> str:
    return (
        f"교사: {problem}\n"
        f"학생: 모르겠습니다.\n"
        f"교사: 힌트 1) {hint1}\n"
        f"학생: {try1}\n"
        f"교사: 힌트 2) {hint2}\n"
        f"학생: {try2}\n"
        f"교사: 마지막 줄에 최종답만 써라."
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
        wrong = (a + b) if op == "-" else max(0, a - b)
        if wrong == ans:
            wrong = ans + 1
        problem = f"다음을 계산하라. {a} {op} {b}"
        hint1 = f"{op} 기호를 보고 두 수를 차례대로 계산해라."
        hint2 = f"정확한 값은 {a} {op} {b} = {ans} 이다."
        rows.append(
            make_row(
                tutor_template(problem, hint1, f"{wrong}", hint2, f"{ans}", str(ans)),
                str(ans),
                f"addsub_d{digits}",
                split,
            )
        )
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
        rough = a * max(1, b - 1)
        if rough == ans:
            rough = ans + a
        problem = f"다음을 계산하라. {a} * {b}"
        hint1 = f"곱셈이다. {a}를 {b}번 더한 값과 같다."
        hint2 = f"정확한 값은 {a} * {b} = {ans} 이다."
        rows.append(
            make_row(
                tutor_template(problem, hint1, str(rough), hint2, str(ans), str(ans)),
                str(ans),
                f"mul_{a_digits}x{b_digits}",
                split,
            )
        )
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
        wrong = max(1, q - 1)
        if wrong == q:
            wrong = q + 1
        problem = f"다음을 계산하라. {a} / {b}"
        hint1 = f"나눗셈이다. 어떤 수에 {b}를 곱하면 {a}가 되는지 찾아라."
        hint2 = f"정확한 값은 {q} 이다. 왜냐하면 {b} * {q} = {a} 이다."
        rows.append(
            make_row(
                tutor_template(problem, hint1, str(wrong), hint2, str(q), str(q)),
                str(q),
                f"div_{len(str(a))}by{b_digits}",
                split,
            )
        )
    return rows


def gen_linear(rng: random.Random, split: str, x_hi: int, add_hi: int, mul_hi: int, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(n):
        x = rng.randint(1, x_hi)
        add_v = rng.randint(1, add_hi)
        mul_v = rng.randint(2, mul_hi)
        out = (x + add_v) * mul_v
        wrong = max(1, x - 1)
        if wrong == x:
            wrong = x + 1
        problem = f"어떤 수에 {add_v}를 더하고 {mul_v}를 곱하면 {out}이 된다. 그 수를 구하라."
        hint1 = f"먼저 {out}을 {mul_v}로 나누고, 그다음 {add_v}를 빼라."
        hint2 = f"{out} / {mul_v} = {x + add_v}, 거기서 {add_v}를 빼면 {x} 이다."
        rows.append(
            make_row(
                tutor_template(problem, hint1, str(wrong), hint2, str(x), str(x)),
                str(x),
                f"linear_x{x_hi}",
                split,
            )
        )
    return rows


def build_train(seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    rows.extend(gen_addsub(rng, "train", 1, 12000))
    rows.extend(gen_addsub(rng, "train", 2, 12000))
    rows.extend(gen_addsub(rng, "train", 3, 8000))
    rows.extend(gen_mul(rng, "train", 1, 1, 14000))
    rows.extend(gen_mul(rng, "train", 2, 1, 10000))
    rows.extend(gen_div(rng, "train", 1, 1, 12000))
    rows.extend(gen_div(rng, "train", 1, 2, 10000))
    rows.extend(gen_linear(rng, "train", 20, 10, 5, 10000))
    rows.extend(gen_linear(rng, "train", 60, 15, 7, 8000))
    rng.shuffle(rows)
    return rows


def build_eval(seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    rows.extend(gen_addsub(rng, "eval", 1, 48))
    rows.extend(gen_addsub(rng, "eval", 2, 48))
    rows.extend(gen_addsub(rng, "eval", 3, 48))
    rows.extend(gen_mul(rng, "eval", 1, 1, 48))
    rows.extend(gen_mul(rng, "eval", 2, 1, 48))
    rows.extend(gen_div(rng, "eval", 1, 1, 48))
    rows.extend(gen_div(rng, "eval", 1, 2, 48))
    rows.extend(gen_linear(rng, "eval", 20, 10, 5, 48))
    rows.extend(gen_linear(rng, "eval", 60, 15, 7, 48))
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
        "output_format": "학생: 최종답: ...",
        "prompt_style": "tutor_multiturn_v1",
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
