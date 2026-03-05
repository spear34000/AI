from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple


TRAIN_OUT = Path("data/teacher_exact_numeric_latex_core_v2_train.jsonl")
EVAL_OUT = Path("data/teacher_exact_numeric_latex_core_v2_eval.jsonl")
MANIFEST_OUT = Path("data/teacher_exact_numeric_latex_core_v2.manifest.json")


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def reduce_fraction(numer: int, denom: int) -> str:
    g = math.gcd(numer, denom)
    numer //= g
    denom //= g
    if denom == 1:
        return str(numer)
    return f"{numer}/{denom}"


def make_row(prompt: str, final_answer: str, category: str, split: str) -> Dict[str, str]:
    answer = str(final_answer).strip()
    return {
        "input": f"{prompt.strip()}\n마지막 줄에 최종답만 써라.",
        "output": f"최종답: {answer}",
        "task_type": "korean",
        "segment_tag": "ko_exact_numeric_latex_core_v2",
        "language": "ko",
        "category": category,
        "split": split,
        "final_answer": answer,
        "answer_format": "최종답: ...",
        "_meta_source_file": "teacher_exact_numeric_latex_core_v2",
    }


def gen_addsub(rng: random.Random, split: str) -> Dict[str, str]:
    a = rng.randint(10, 99999)
    b = rng.randint(1, 99999)
    op = rng.choice(["+", "-"])
    if op == "-" and b > a:
        a, b = b, a
    ans = a + b if op == "+" else a - b
    return make_row(f"다음을 계산하라. {a} {op} {b}", str(ans), "arithmetic_addsub", split)


def gen_mul(rng: random.Random, split: str) -> Dict[str, str]:
    a = rng.randint(10, 999)
    b = rng.randint(2, 99)
    return make_row(f"다음을 계산하라. {a} * {b}", str(a * b), "arithmetic_mul", split)


def gen_div(rng: random.Random, split: str) -> Dict[str, str]:
    b = rng.randint(2, 99)
    q = rng.randint(2, 999)
    a = b * q
    return make_row(f"다음을 계산하라. {a} / {b}", str(q), "arithmetic_div", split)


def gen_linear(rng: random.Random, split: str) -> Dict[str, str]:
    x = rng.randint(1, 100)
    add_v = rng.randint(1, 20)
    mul_v = rng.randint(2, 9)
    out = (x + add_v) * mul_v
    prompt = f"어떤 수에 {add_v}를 더하고 {mul_v}를 곱하면 {out}이 된다. 그 수를 구하라."
    return make_row(prompt, str(x), "linear_equation", split)


def gen_ratio(rng: random.Random, split: str) -> Dict[str, str]:
    a = rng.randint(1, 9)
    b = rng.randint(1, 9)
    scale = rng.randint(2, 40)
    known = b * scale
    ans = a * scale
    prompt = f"A:B = {a}:{b}이다. B가 {known}이면 A는 얼마인가?"
    return make_row(prompt, str(ans), "ratio", split)


def gen_probability(rng: random.Random, split: str) -> Dict[str, str]:
    red = rng.randint(1, 5)
    blue = rng.randint(1, 5)
    green = rng.randint(1, 5)
    total = red + blue + green
    target = rng.choice([("빨간", red), ("파란", blue), ("초록", green)])
    ans = reduce_fraction(target[1], total)
    prompt = (
        f"주머니에 빨간 공 {red}개, 파란 공 {blue}개, 초록 공 {green}개가 있다. "
        f"한 개를 무작위로 꺼낼 때 {target[0]} 공일 확률을 기약분수로 구하라."
    )
    return make_row(prompt, ans, "probability", split)


def gen_latex_fraction(rng: random.Random, split: str) -> Dict[str, str]:
    numer = rng.randint(1, 20)
    denom = rng.randint(2, 20)
    answer = f"\\frac{{{numer}}}{{{denom}}}"
    prompt = f"분수 {numer}/{denom}를 LaTeX 수식으로 써라."
    return make_row(prompt, answer, "latex_fraction", split)


def gen_latex_equation(rng: random.Random, split: str) -> Dict[str, str]:
    mode = rng.choice(["sum", "linear"])
    if mode == "sum":
        upper = rng.randint(3, 12)
        answer = f"\\sum_{{i=1}}^{{{upper}}} i"
        prompt = f"1부터 {upper}까지의 합을 나타내는 시그마 수식을 LaTeX로 써라."
    else:
        a = rng.randint(2, 9)
        b = rng.randint(1, 12)
        c = rng.randint(20, 120)
        answer = f"{a}x+{b}={c}"
        prompt = f"일차방정식 {a}x + {b} = {c}를 LaTeX로 써라."
    return make_row(prompt, answer, "latex_equation", split)


GENERATORS: List[Tuple[str, Callable[[random.Random, str], Dict[str, str]]]] = [
    ("arithmetic_addsub", gen_addsub),
    ("arithmetic_mul", gen_mul),
    ("arithmetic_div", gen_div),
    ("linear_equation", gen_linear),
    ("ratio", gen_ratio),
    ("probability", gen_probability),
    ("latex_fraction", gen_latex_fraction),
    ("latex_equation", gen_latex_equation),
]


def build_rows(count_per_category: int, split: str, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    for _name, fn in GENERATORS:
        for _ in range(count_per_category):
            rows.append(fn(rng, split))
    rng.shuffle(rows)
    return rows


def main() -> None:
    train_rows = build_rows(count_per_category=12000, split="train", seed=20260303)
    eval_rows = build_rows(count_per_category=128, split="eval", seed=20260304)

    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(EVAL_OUT, eval_rows)

    manifest = {
        "seed_train": 20260303,
        "seed_eval": 20260304,
        "train_written": len(train_rows),
        "eval_written": len(eval_rows),
        "categories": {name: 12000 for name, _fn in GENERATORS},
        "eval_categories": {name: 128 for name, _fn in GENERATORS},
        "output_format": "최종답: ...",
        "prompt_style": "short_exact_core_v2",
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
