from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


FINAL_PREFIX = "최종답: "


TRAIN_DEFAULT = Path("data/teacher_arith_pure_mix_v2_clean_train.jsonl")
EVAL_DEFAULT = Path("data/teacher_arith_pure_mix_v2_clean_eval.jsonl")
MANIFEST_DEFAULT = Path("data/teacher_arith_pure_mix_v2_clean.manifest.json")


TRAIN_COUNTS = {
    "addsub_d1": 40000,
    "add_carry_d2": 50000,
    "sub_borrow_d2": 50000,
    "add_carry_d3": 60000,
    "sub_borrow_d3": 60000,
    "mul_1x1": 40000,
    "mul_2x1": 50000,
    "mul_2x2": 70000,
    "div_2by1": 30000,
    "div_3by1": 30000,
    "div_4by2": 30000,
    "linear_add_then_mul": 30000,
    "linear_mul_then_add": 30000,
    "ratio": 30000,
}


EVAL_COUNTS = {
    "addsub_d1": 512,
    "add_carry_d2": 512,
    "sub_borrow_d2": 512,
    "add_carry_d3": 768,
    "sub_borrow_d3": 768,
    "mul_1x1": 512,
    "mul_2x1": 512,
    "mul_2x2": 768,
    "div_2by1": 384,
    "div_3by1": 384,
    "div_4by2": 384,
    "linear_add_then_mul": 384,
    "linear_mul_then_add": 384,
    "ratio": 384,
}


def scaled(count: int, scale: float) -> int:
    return max(1, int(round(float(count) * float(scale))))


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_row(
    prompt: str,
    output_text: str,
    final_answer: str,
    category: str,
    split: str,
    prompt_style: str,
    has_cot: bool,
) -> Dict[str, str]:
    return {
        "input": prompt.strip(),
        "output": output_text.strip(),
        "task_type": "korean",
        "segment_tag": "ko_teacher_arith_pure_mix_v2_clean",
        "language": "ko",
        "category": category,
        "split": split,
        "prompt_style": prompt_style,
        "final_answer": str(final_answer).strip(),
        "answer_format": "최종답: ...",
        "has_cot": "1" if has_cot else "0",
        "_meta_source_file": "teacher_arith_pure_mix_v2_clean",
    }


def final_only(answer: int) -> str:
    return f"{FINAL_PREFIX}{answer}"


def explained(answer: int, steps: List[str]) -> str:
    return "풀이: " + " ".join(s.strip() for s in steps if s.strip()) + "\n" + final_only(answer)


def prompt_variants(expr: str, question: str, with_steps: bool, style_hint: str) -> List[Tuple[str, str]]:
    if with_steps:
        return [
            (f"{question}\n풀이를 짧게 쓰고 마지막 줄에 최종답만 써라.", f"{style_hint}_cot_ko"),
            (f"{expr}\n풀이를 짧게 쓰고 마지막 줄에 최종답만 써라.", f"{style_hint}_cot_symbolic"),
            (f"{question}\n단계를 보인 뒤 마지막 줄에 최종답만 써라.", f"{style_hint}_steps"),
        ]
    return [
        (f"{question}\n마지막 줄에 최종답만 써라.", f"{style_hint}_ko"),
        (f"{expr}\n마지막 줄에 최종답만 써라.", f"{style_hint}_symbolic"),
        (expr, f"{style_hint}_bare"),
    ]


def place_name(index_from_right: int) -> str:
    names = {0: "1의 자리", 1: "10의 자리", 2: "100의 자리", 3: "1000의 자리"}
    return names.get(index_from_right, f"10^{index_from_right} 자리")


def add_steps(a: int, b: int) -> List[str]:
    sa = str(a)[::-1]
    sb = str(b)[::-1]
    carry = 0
    steps: List[str] = []
    for i in range(max(len(sa), len(sb))):
        da = int(sa[i]) if i < len(sa) else 0
        db = int(sb[i]) if i < len(sb) else 0
        total = da + db + carry
        text = f"{place_name(i)}에서 {da}+{db}"
        if carry:
            text += f"+올림 {carry}"
        text += f"={total}"
        if total >= 10:
            text += f", {total % 10}을 쓰고 {total // 10} 올림"
        else:
            text += f", 그대로 {total}를 쓴다"
        steps.append(text)
        carry = total // 10
    if carry:
        steps.append(f"마지막 올림 {carry}를 맨 앞에 둔다")
    return steps


def sub_steps(a: int, b: int) -> List[str]:
    sa = str(a)[::-1]
    sb = str(b)[::-1]
    borrow = 0
    steps: List[str] = []
    for i in range(max(len(sa), len(sb))):
        da = int(sa[i]) if i < len(sa) else 0
        db = int(sb[i]) if i < len(sb) else 0
        current = da - borrow
        text = f"{place_name(i)}에서 "
        if current < db:
            text += f"{da}"
            if borrow:
                text += f"-빌림 {borrow}"
            text += f"가 {db}보다 작아서 10을 빌려 {current + 10}-{db}={current + 10 - db}"
            borrow = 1
        else:
            text += f"{da}"
            if borrow:
                text += f"-빌림 {borrow}"
            text += f"-{db}={current - db}"
            borrow = 0
        steps.append(text)
    return steps


def mul_steps(a: int, b: int) -> List[str]:
    steps: List[str] = []
    sb = str(b)[::-1]
    partials: List[int] = []
    for i, ch in enumerate(sb):
        digit = int(ch)
        place = 10 ** i
        partial = a * digit * place
        if place == 1:
            steps.append(f"{a}×{digit}={a * digit}")
        else:
            steps.append(f"{a}×{digit}×{place}={partial}")
        partials.append(partial)
    if len(partials) > 1:
        steps.append("부분곱을 더한다: " + " + ".join(str(p) for p in partials) + f" = {sum(partials)}")
    return steps


def div_steps(dividend: int, divisor: int, quotient: int) -> List[str]:
    return [f"{dividend}÷{divisor}={quotient}", f"검산: {divisor}×{quotient}={dividend}"]


def linear_add_then_mul_steps(offset: int, mul: int, target: int, answer: int) -> List[str]:
    mid = target // mul
    return [
        f"(x+{offset})×{mul}={target}",
        f"x+{offset}={target}÷{mul}={mid}",
        f"x={mid}-{offset}={answer}",
    ]


def linear_mul_then_add_steps(mul: int, offset: int, target: int, answer: int) -> List[str]:
    mid = target - offset
    return [
        f"{mul}x+{offset}={target}",
        f"{mul}x={target}-{offset}={mid}",
        f"x={mid}÷{mul}={answer}",
    ]


def ratio_steps(left: int, right: int, known_side: str, known_value: int, answer: int) -> List[str]:
    if known_side == "right":
        unit = known_value // right
        return [
            f"비 {left}:{right}에서 {right}칸이 {known_value}이므로 한 칸은 {known_value}÷{right}={unit}",
            f"왼쪽 값은 {unit}×{left}={answer}",
        ]
    unit = known_value // left
    return [
        f"비 {left}:{right}에서 {left}칸이 {known_value}이므로 한 칸은 {known_value}÷{left}={unit}",
        f"오른쪽 값은 {unit}×{right}={answer}",
    ]


def gen_addsub_d1(rng: random.Random, split: str, count: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(count):
        op = rng.choice(["+", "-"])
        a = rng.randint(0, 9)
        b = rng.randint(0, 9)
        if op == "-" and b > a:
            a, b = b, a
        ans = a + b if op == "+" else a - b
        question = f"다음을 계산하라. {a} {op} {b}"
        expr = f"{a}{op}{b}="
        prompt, style = rng.choice(prompt_variants(expr, question, False, "addsub_d1"))
        rows.append(make_row(prompt, final_only(ans), str(ans), "addsub_d1", split, style, False))
    return rows


def gen_add_carry(rng: random.Random, split: str, count: int, digits: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    low = 10 ** (digits - 1)
    high = (10 ** digits) - 1
    while len(rows) < count:
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        if sum(int(x) + int(y) >= 10 for x, y in zip(str(a)[::-1], str(b)[::-1])) == 0:
            continue
        ans = a + b
        question = f"다음을 계산하라. {a} + {b}"
        expr = f"{a}+{b}="
        prompt, style = rng.choice(prompt_variants(expr, question, digits >= 3, f"add_carry_d{digits}"))
        output = explained(ans, add_steps(a, b)) if digits >= 3 or rng.random() < 0.55 else final_only(ans)
        rows.append(make_row(prompt, output, str(ans), f"add_carry_d{digits}", split, style, "풀이:" in output))
    return rows


def gen_sub_borrow(rng: random.Random, split: str, count: int, digits: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    low = 10 ** (digits - 1)
    high = (10 ** digits) - 1
    while len(rows) < count:
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        if a < b:
            a, b = b, a
        if sum(int(x) < int(y) for x, y in zip(str(a)[::-1], str(b)[::-1])) == 0:
            continue
        ans = a - b
        question = f"다음을 계산하라. {a} - {b}"
        expr = f"{a}-{b}="
        prompt, style = rng.choice(prompt_variants(expr, question, digits >= 3, f"sub_borrow_d{digits}"))
        output = explained(ans, sub_steps(a, b)) if digits >= 3 or rng.random() < 0.55 else final_only(ans)
        rows.append(make_row(prompt, output, str(ans), f"sub_borrow_d{digits}", split, style, "풀이:" in output))
    return rows


def gen_mul_1x1(rng: random.Random, split: str, count: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(count):
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        ans = a * b
        question = f"다음을 계산하라. {a} × {b}"
        expr = f"{a}*{b}="
        prompt, style = rng.choice(prompt_variants(expr, question, False, "mul_1x1"))
        rows.append(make_row(prompt, final_only(ans), str(ans), "mul_1x1", split, style, False))
    return rows


def gen_mul_2x1(rng: random.Random, split: str, count: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(count):
        a = rng.randint(10, 99)
        b = rng.randint(2, 9)
        ans = a * b
        question = f"다음을 계산하라. {a} × {b}"
        expr = f"{a}*{b}="
        prompt, style = rng.choice(prompt_variants(expr, question, True, "mul_2x1"))
        rows.append(make_row(prompt, explained(ans, mul_steps(a, b)), str(ans), "mul_2x1", split, style, True))
    return rows


def gen_mul_2x2(rng: random.Random, split: str, count: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(count):
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        ans = a * b
        question = f"다음을 계산하라. {a} × {b}"
        expr = f"{a}*{b}="
        prompt, style = rng.choice(prompt_variants(expr, question, True, "mul_2x2"))
        rows.append(make_row(prompt, explained(ans, mul_steps(a, b)), str(ans), "mul_2x2", split, style, True))
    return rows


def gen_div_exact(rng: random.Random, split: str, count: int, dividend_digits: int, divisor_digits: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    q_low = 10 if dividend_digits >= 3 else 2
    q_high = 99 if dividend_digits >= 4 else 99
    while len(rows) < count:
        divisor = rng.randint(10 ** (divisor_digits - 1), (10 ** divisor_digits) - 1)
        quotient = rng.randint(q_low, q_high)
        dividend = divisor * quotient
        if len(str(dividend)) != dividend_digits:
            continue
        ans = quotient
        question = f"다음을 계산하라. {dividend} ÷ {divisor}"
        expr = f"{dividend}/{divisor}="
        with_steps = dividend_digits >= 3
        prompt, style = rng.choice(prompt_variants(expr, question, with_steps, f"div_{dividend_digits}by{divisor_digits}"))
        output = explained(ans, div_steps(dividend, divisor, quotient)) if with_steps else final_only(ans)
        rows.append(
            make_row(
                prompt,
                output,
                str(ans),
                f"div_{dividend_digits}by{divisor_digits}",
                split,
                style,
                with_steps,
            )
        )
    return rows


def gen_linear_add_then_mul(rng: random.Random, split: str, count: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(count):
        x = rng.randint(2, 60)
        offset = rng.randint(2, 12)
        mul = rng.randint(2, 9)
        target = (x + offset) * mul
        question = f"어떤 수에 {offset}을 더하고 {mul}를 곱하면 {target}가 된다. 그 수를 구하라."
        expr = f"(x+{offset})*{mul}={target}"
        prompt, style = rng.choice(prompt_variants(expr, question, True, "linear_add_then_mul"))
        rows.append(
            make_row(
                prompt,
                explained(x, linear_add_then_mul_steps(offset, mul, target, x)),
                str(x),
                "linear_add_then_mul",
                split,
                style,
                True,
            )
        )
    return rows


def gen_linear_mul_then_add(rng: random.Random, split: str, count: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(count):
        x = rng.randint(2, 60)
        mul = rng.randint(2, 12)
        offset = rng.randint(2, 30)
        target = (mul * x) + offset
        question = f"어떤 수에 {mul}를 곱하고 {offset}을 더하면 {target}가 된다. 그 수를 구하라."
        expr = f"{mul}x+{offset}={target}"
        prompt, style = rng.choice(prompt_variants(expr, question, True, "linear_mul_then_add"))
        rows.append(
            make_row(
                prompt,
                explained(x, linear_mul_then_add_steps(mul, offset, target, x)),
                str(x),
                "linear_mul_then_add",
                split,
                style,
                True,
            )
        )
    return rows


def gen_ratio(rng: random.Random, split: str, count: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(count):
        left = rng.randint(2, 12)
        right = rng.randint(2, 12)
        unit = rng.randint(2, 40)
        if rng.random() < 0.5:
            known_value = right * unit
            answer = left * unit
            question = f"x:y={left}:{right}일 때 y={known_value}이면 x는 얼마인가?"
            expr = f"x:y={left}:{right}, y={known_value}, x=?"
            steps = ratio_steps(left, right, "right", known_value, answer)
        else:
            known_value = left * unit
            answer = right * unit
            question = f"x:y={left}:{right}일 때 x={known_value}이면 y는 얼마인가?"
            expr = f"x:y={left}:{right}, x={known_value}, y=?"
            steps = ratio_steps(left, right, "left", known_value, answer)
        prompt, style = rng.choice(prompt_variants(expr, question, True, "ratio"))
        rows.append(make_row(prompt, explained(answer, steps), str(answer), "ratio", split, style, True))
    return rows


def build_rows(split: str, counts: Dict[str, int], seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    rows.extend(gen_addsub_d1(rng, split, counts["addsub_d1"]))
    rows.extend(gen_add_carry(rng, split, counts["add_carry_d2"], 2))
    rows.extend(gen_sub_borrow(rng, split, counts["sub_borrow_d2"], 2))
    rows.extend(gen_add_carry(rng, split, counts["add_carry_d3"], 3))
    rows.extend(gen_sub_borrow(rng, split, counts["sub_borrow_d3"], 3))
    rows.extend(gen_mul_1x1(rng, split, counts["mul_1x1"]))
    rows.extend(gen_mul_2x1(rng, split, counts["mul_2x1"]))
    rows.extend(gen_mul_2x2(rng, split, counts["mul_2x2"]))
    rows.extend(gen_div_exact(rng, split, counts["div_2by1"], 2, 1))
    rows.extend(gen_div_exact(rng, split, counts["div_3by1"], 3, 1))
    rows.extend(gen_div_exact(rng, split, counts["div_4by2"], 4, 2))
    rows.extend(gen_linear_add_then_mul(rng, split, counts["linear_add_then_mul"]))
    rows.extend(gen_linear_mul_then_add(rng, split, counts["linear_mul_then_add"]))
    rows.extend(gen_ratio(rng, split, counts["ratio"]))
    rng.shuffle(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_out", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--eval_out", type=Path, default=EVAL_DEFAULT)
    parser.add_argument("--manifest_out", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260304)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_counts = {k: scaled(v, args.scale) for k, v in TRAIN_COUNTS.items()}
    eval_counts = {k: scaled(v, args.scale) for k, v in EVAL_COUNTS.items()}
    train_rows = build_rows("train", train_counts, args.seed)
    eval_rows = build_rows("eval", eval_counts, args.seed + 1)

    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, eval_rows)

    manifest = {
        "seed": int(args.seed),
        "scale": float(args.scale),
        "train_written": len(train_rows),
        "eval_written": len(eval_rows),
        "train_counts": train_counts,
        "eval_counts": eval_counts,
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "final_prefix": FINAL_PREFIX,
        "segment_tag": "ko_teacher_arith_pure_mix_v2_clean",
    }
    args.manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
