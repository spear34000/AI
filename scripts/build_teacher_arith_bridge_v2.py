from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


TRAIN_OUT = Path("data/teacher_arith_bridge_v2_train.jsonl")
EVAL_OUT = Path("data/teacher_arith_bridge_v2_eval.jsonl")
MANIFEST_OUT = Path("data/teacher_arith_bridge_v2.manifest.json")

LAST_LINE_ONLY = "\ucd5c\uc885\ub2f5\ub9cc \uc368\ub77c."
LAST_LINE_ONLY_LONG = "\ub9c8\uc9c0\ub9c9 \uc904\uc5d0 \ucd5c\uc885\ub2f5\ub9cc \uc368\ub77c."
FINAL_PREFIX = "\ucd5c\uc885\ub2f5: "
OUTPUT_FORMAT = "\ucd5c\uc885\ub2f5: ..."
PROMPT_CALC = "\ub2e4\uc74c\uc744 \uacc4\uc0b0\ud558\ub77c."
OP_WORD = {
    "+": "\ub354\ud558\uae30",
    "-": "\ube7c\uae30",
    "*": "\uacf1\ud558\uae30",
    "/": "\ub098\ub204\uae30",
}


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
        "segment_tag": "ko_teacher_arith_bridge_v2",
        "language": "ko",
        "category": category,
        "split": split,
        "prompt_style": style,
        "final_answer": answer,
        "answer_format": OUTPUT_FORMAT,
        "_meta_source_file": "teacher_arith_bridge_v2",
    }


def prompt_variants(expr: str, a: int, op: str, b: int) -> List[Tuple[str, str]]:
    op_word = OP_WORD[op]
    return [
        (f"{expr}=\n{LAST_LINE_ONLY}", "symbolic"),
        (f"{PROMPT_CALC} {a} {op} {b}\n{LAST_LINE_ONLY_LONG}", "korean_calc"),
        (f"{a} {op_word} {b}\uc758 \uac12\uc740?\n{LAST_LINE_ONLY}", "korean_short"),
    ]


def add_rows(rows: List[Dict[str, str]], prompts: List[Tuple[str, str]], answer: int, category: str, split: str) -> None:
    for prompt, style in prompts:
        rows.append(make_row(prompt, str(answer), category, split, style))


def gen_addsub(rng: random.Random, split: str, digits: int, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    lo = 10 ** (digits - 1)
    hi = (10**digits) - 1
    for _ in range(n):
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        op = rng.choice(["+", "-"])
        if op == "-" and b > a:
            a, b = b, a
        ans = a + b if op == "+" else a - b
        add_rows(rows, prompt_variants(f"{a}{op}{b}", a, op, b), ans, f"bridge_addsub_d{digits}", split)
    return rows


def gen_mul(rng: random.Random, split: str, a_digits: int, b_digits: int, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    a_lo = 10 ** (a_digits - 1)
    a_hi = (10**a_digits) - 1
    b_lo = 1 if b_digits == 1 else 10 ** (b_digits - 1)
    b_hi = (10**b_digits) - 1
    for _ in range(n):
        a = rng.randint(a_lo, a_hi)
        b = rng.randint(b_lo, b_hi)
        add_rows(rows, prompt_variants(f"{a}*{b}", a, "*", b), a * b, f"bridge_mul_{a_digits}x{b_digits}", split)
    return rows


def gen_div(rng: random.Random, split: str, b_digits: int, q_digits: int, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    b_lo = 2 if b_digits == 1 else 10 ** (b_digits - 1)
    b_hi = (10**b_digits) - 1
    q_lo = 10 ** (q_digits - 1)
    q_hi = (10**q_digits) - 1
    for _ in range(n):
        b = rng.randint(b_lo, b_hi)
        q = rng.randint(q_lo, q_hi)
        a = b * q
        add_rows(rows, prompt_variants(f"{a}/{b}", a, "/", b), q, f"bridge_div_{len(str(a))}by{b_digits}", split)
    return rows


def gen_easy_replay(rng: random.Random, split: str, n: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(n):
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        add_rows(rows, prompt_variants(f"{a}+{b}", a, "+", b), a + b, "bridge_replay_add_d1", split)
        add_rows(rows, prompt_variants(f"{a}*{b}", a, "*", b), a * b, "bridge_replay_mul_1x1", split)
    return rows


def build_train(seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    rows.extend(gen_mul(rng, "train", 2, 1, 18000))
    rows.extend(gen_mul(rng, "train", 2, 2, 18000))
    rows.extend(gen_div(rng, "train", 1, 2, 14000))
    rows.extend(gen_div(rng, "train", 1, 3, 12000))
    rows.extend(gen_div(rng, "train", 2, 2, 12000))
    rows.extend(gen_addsub(rng, "train", 2, 12000))
    rows.extend(gen_addsub(rng, "train", 3, 12000))
    rows.extend(gen_easy_replay(rng, "train", 4000))
    rng.shuffle(rows)
    return rows


def build_eval(seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    rows.extend(gen_mul(rng, "eval", 2, 1, 64))
    rows.extend(gen_mul(rng, "eval", 2, 2, 64))
    rows.extend(gen_div(rng, "eval", 1, 2, 64))
    rows.extend(gen_div(rng, "eval", 1, 3, 64))
    rows.extend(gen_div(rng, "eval", 2, 2, 64))
    rows.extend(gen_addsub(rng, "eval", 2, 64))
    rows.extend(gen_addsub(rng, "eval", 3, 64))
    rng.shuffle(rows)
    return rows


def main() -> None:
    train_rows = build_train(20260314)
    eval_rows = build_eval(20260315)
    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(EVAL_OUT, eval_rows)
    manifest = {
        "seed_train": 20260314,
        "seed_eval": 20260315,
        "train_written": len(train_rows),
        "eval_written": len(eval_rows),
        "output_format": OUTPUT_FORMAT,
        "focus": {
            "mul_2x1": 18000,
            "mul_2x2": 18000,
            "div_2by1": 14000,
            "div_3by1": 12000,
            "div_4by2": 12000,
            "addsub_d2": 12000,
            "addsub_d3": 12000,
            "easy_replay": 4000,
        },
        "prompt_styles": ["symbolic", "korean_calc", "korean_short"],
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
