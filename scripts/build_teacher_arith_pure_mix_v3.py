from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


FINAL_PREFIX = "\uCD5C\uC885\uB2F5: "
LAST_LINE_ONLY = "\uB9C8\uC9C0\uB9C9 \uC904\uC5D0 \uCD5C\uC885\uB2F5\uB9CC \uC368\uB77C."
LAST_LINE_WITH_SHORT_STEPS = "\uC9E7\uAC8C \uD480\uC774\uD558\uACE0 \uB9C8\uC9C0\uB9C9 \uC904\uC5D0 \uCD5C\uC885\uB2F5\uB9CC \uC368\uB77C."
PROMPT_CALC = "\uB2E4\uC74C\uC744 \uACC4\uC0B0\uD558\uB77C."
EXPLAIN_PREFIX = "\uD480\uC774: "

TRAIN_DEFAULT = Path("data/teacher_arith_pure_mix_v3_train.jsonl")
EVAL_DEFAULT = Path("data/teacher_arith_pure_mix_v3_eval.jsonl")
MANIFEST_DEFAULT = Path("data/teacher_arith_pure_mix_v3.manifest.json")
GENERAL_REPLAY_DEFAULT = Path("data/repair_ko_direct_v3.jsonl")


TRAIN_COUNTS = {
    "general_replay": 12000,
    "addsub_d1": 24000,
    "add_carry_d2": 52000,
    "sub_borrow_d2": 52000,
    "add_carry_d3": 72000,
    "sub_borrow_d3": 72000,
    "add_carry_d4": 36000,
    "sub_borrow_d4": 36000,
    "mul_1x1": 24000,
    "mul_2x1": 52000,
    "mul_2x2_easy": 60000,
    "mul_2x2_hard": 90000,
    "div_2by1_easy": 18000,
    "div_3by1_easy": 18000,
}

EVAL_COUNTS = {
    "general_replay": 256,
    "addsub_d1": 256,
    "add_carry_d2": 384,
    "sub_borrow_d2": 384,
    "add_carry_d3": 512,
    "sub_borrow_d3": 512,
    "add_carry_d4": 256,
    "sub_borrow_d4": 256,
    "mul_1x1": 256,
    "mul_2x1": 384,
    "mul_2x2_easy": 512,
    "mul_2x2_hard": 512,
    "div_2by1_easy": 192,
    "div_3by1_easy": 192,
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
    segment_tag: str = "ko_teacher_arith_pure_mix_v3",
    source_file: str = "teacher_arith_pure_mix_v3",
) -> Dict[str, str]:
    return {
        "input": prompt.strip(),
        "output": output_text.strip(),
        "task_type": "korean",
        "segment_tag": segment_tag,
        "language": "ko",
        "category": category,
        "split": split,
        "prompt_style": prompt_style,
        "final_answer": str(final_answer).strip(),
        "answer_format": "\uCD5C\uC885\uB2F5: ...",
        "has_cot": "1" if has_cot else "0",
        "_meta_source_file": source_file,
    }


def final_only(answer: int) -> str:
    return f"{FINAL_PREFIX}{answer}"


def with_steps(answer: int, steps: Sequence[str]) -> str:
    clean = [s.strip() for s in steps if str(s).strip()]
    return EXPLAIN_PREFIX + " ".join(clean) + "\n" + final_only(answer)


def prompt_variants(expr: str, question: str, style_hint: str, with_steps: bool) -> List[Tuple[str, str]]:
    suffix = LAST_LINE_WITH_SHORT_STEPS if with_steps else LAST_LINE_ONLY
    return [
        (f"{PROMPT_CALC} {expr}\n{suffix}", f"{style_hint}_calc"),
        (f"{expr}=\n{suffix}", f"{style_hint}_symbolic"),
        (f"{question}\n{suffix}", f"{style_hint}_word"),
    ]


def place_name(index_from_right: int) -> str:
    names = {
        0: "\uC77C\uC758 \uC790\uB9AC",
        1: "\uC2ED\uC758 \uC790\uB9AC",
        2: "\uBC31\uC758 \uC790\uB9AC",
        3: "\uCC9C\uC758 \uC790\uB9AC",
    }
    return names.get(index_from_right, f"10^{index_from_right} \uC790\uB9AC")


def add_steps(a: int, b: int) -> List[str]:
    sa = str(a)[::-1]
    sb = str(b)[::-1]
    carry = 0
    steps: List[str] = []
    for i in range(max(len(sa), len(sb))):
        da = int(sa[i]) if i < len(sa) else 0
        db = int(sb[i]) if i < len(sb) else 0
        total = da + db + carry
        out_digit = total % 10
        next_carry = total // 10
        text = f"{place_name(i)}: {da}+{db}"
        if carry:
            text += f"+\uC62C\uB9BC {carry}"
        text += f"={total}"
        if next_carry:
            text += f", {out_digit}\uB97C \uC4F0\uACE0 {next_carry} \uC62C\uB9BC"
        else:
            text += f", {out_digit}\uB97C \uC4F4\uB2E4"
        steps.append(text)
        carry = next_carry
    if carry:
        steps.append(f"\uB0A8\uC740 \uC62C\uB9BC {carry}\uC744 \uC55E\uC5D0 \uBD99\uC778\uB2E4")
    return steps


def sub_steps(a: int, b: int) -> List[str]:
    a_digits = [int(ch) for ch in str(a)][::-1]
    b_digits = [int(ch) for ch in str(b)][::-1]
    borrow = 0
    steps: List[str] = []
    for i in range(max(len(a_digits), len(b_digits))):
        da = a_digits[i] if i < len(a_digits) else 0
        db = b_digits[i] if i < len(b_digits) else 0
        current = da - borrow
        text = f"{place_name(i)}: "
        if current < db:
            text += f"{da}"
            if borrow:
                text += f"-\uBE4C\uB9BC {borrow}"
            text += f"\uAC00 {db}\uBCF4\uB2E4 \uC791\uC544 10\uC744 \uBE4C\uB824 {current + 10}-{db}={current + 10 - db}"
            borrow = 1
        else:
            text += f"{da}"
            if borrow:
                text += f"-\uBE4C\uB9BC {borrow}"
            text += f"-{db}={current - db}"
            borrow = 0
        steps.append(text)
    return steps


def mul_steps(a: int, b: int) -> List[str]:
    steps: List[str] = []
    partials: List[int] = []
    for i, ch in enumerate(str(b)[::-1]):
        digit = int(ch)
        place = 10**i
        partial = a * digit * place
        if place == 1:
            steps.append(f"\uBD80\uBD84\uACF11: {a}\u00D7{digit}={a * digit}")
        else:
            steps.append(f"\uBD80\uBD84\uACF1{i+1}: {a}\u00D7{digit}\u00D7{place}={partial}")
        partials.append(partial)
    if len(partials) > 1:
        joined = " + ".join(str(p) for p in partials)
        steps.append(f"\uD569: {joined} = {sum(partials)}")
    return steps


def div_steps(dividend: int, divisor: int, quotient: int) -> List[str]:
    return [
        f"{dividend}\u00F7{divisor}={quotient}",
        f"\uAC80\uC0B0: {divisor}\u00D7{quotient}={dividend}",
    ]


def add_general_replay(
    rows: List[Dict[str, str]],
    *,
    path: Path,
    rng: random.Random,
    count: int,
    split: str,
) -> None:
    candidates: List[Dict[str, str]] = []
    blocked = (
        "최종답",
        "정답:",
        "+",
        "-",
        "*",
        "/",
        "=",
        "곱하기",
        "나누기",
        "더하기",
        "빼기",
        "계산",
    )
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            prompt = str(row.get("input", "")).strip()
            output = str(row.get("output", "")).strip()
            if not prompt or not output:
                continue
            if len(prompt) > 180 or len(output) > 280:
                continue
            joined = prompt + "\n" + output
            if any(tok in joined for tok in blocked):
                continue
            candidates.append(
                {
                    "input": prompt,
                    "output": output,
                    "task_type": "korean",
                    "segment_tag": "ko_teacher_arith_pure_mix_v3_replay",
                    "language": "ko",
                    "category": "general_replay",
                    "split": split,
                    "prompt_style": "general_replay",
                    "final_answer": "",
                    "answer_format": "freeform",
                    "has_cot": "0",
                    "_meta_source_file": str(path.name),
                }
            )
    if len(candidates) < count:
        raise RuntimeError(f"not enough general replay rows in {path}")
    rng.shuffle(candidates)
    rows.extend(candidates[:count])


def gen_addsub_d1(rng: random.Random, split: str, count: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(count):
        op = rng.choice(["+", "-"])
        a = rng.randint(0, 9)
        b = rng.randint(0, 9)
        if op == "-" and b > a:
            a, b = b, a
        ans = a + b if op == "+" else a - b
        question = (
            f"{a}\uC640 {b}\uB97C \uB354\uD558\uBA74?"
            if op == "+"
            else f"{a}\uC5D0\uC11C {b}\uB97C \uBE7C\uBA74?"
        )
        expr = f"{a}{op}{b}"
        prompt, style = rng.choice(prompt_variants(expr, question, "addsub_d1", False))
        rows.append(make_row(prompt, final_only(ans), str(ans), "addsub_d1", split, style, False))
    return rows


def gen_add_carry(rng: random.Random, split: str, count: int, digits: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    low = 10 ** (digits - 1)
    high = (10**digits) - 1
    while len(rows) < count:
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        if not any((int(x) + int(y)) >= 10 for x, y in zip(str(a)[::-1], str(b)[::-1])):
            continue
        ans = a + b
        question = f"{a}\uC5D0 {b}\uB97C \uB354\uD558\uBA74?"
        expr = f"{a}+{b}"
        use_steps = digits >= 3 or rng.random() < 0.45
        prompt, style = rng.choice(prompt_variants(expr, question, f"add_carry_d{digits}", use_steps))
        output = with_steps(ans, add_steps(a, b)) if use_steps else final_only(ans)
        rows.append(make_row(prompt, output, str(ans), f"add_carry_d{digits}", split, style, use_steps))
    return rows


def gen_sub_borrow(rng: random.Random, split: str, count: int, digits: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    low = 10 ** (digits - 1)
    high = (10**digits) - 1
    while len(rows) < count:
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        if a < b:
            a, b = b, a
        if not any(int(x) < int(y) for x, y in zip(str(a)[::-1], str(b)[::-1])):
            continue
        ans = a - b
        question = f"{a}\uC5D0\uC11C {b}\uB97C \uBE7C\uBA74?"
        expr = f"{a}-{b}"
        use_steps = digits >= 3 or rng.random() < 0.45
        prompt, style = rng.choice(prompt_variants(expr, question, f"sub_borrow_d{digits}", use_steps))
        output = with_steps(ans, sub_steps(a, b)) if use_steps else final_only(ans)
        rows.append(make_row(prompt, output, str(ans), f"sub_borrow_d{digits}", split, style, use_steps))
    return rows


def gen_mul_1x1(rng: random.Random, split: str, count: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(count):
        a = rng.randint(2, 9)
        b = rng.randint(2, 9)
        ans = a * b
        question = f"{a} \uACF1\uD558\uAE30 {b}\uC740?"
        expr = f"{a}*{b}"
        prompt, style = rng.choice(prompt_variants(expr, question, "mul_1x1", False))
        rows.append(make_row(prompt, final_only(ans), str(ans), "mul_1x1", split, style, False))
    return rows


def gen_mul_2x1(rng: random.Random, split: str, count: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(count):
        a = rng.randint(12, 99)
        b = rng.randint(2, 9)
        ans = a * b
        question = f"{a}\uC5D0 {b}\uB97C \uACF1\uD558\uBA74?"
        expr = f"{a}*{b}"
        use_steps = True
        prompt, style = rng.choice(prompt_variants(expr, question, "mul_2x1", use_steps))
        rows.append(make_row(prompt, with_steps(ans, mul_steps(a, b)), str(ans), "mul_2x1", split, style, True))
    return rows


def gen_mul_2x2(
    rng: random.Random,
    split: str,
    count: int,
    *,
    hard_only: bool,
    category: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    while len(rows) < count:
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        if hard_only and ((a % 10) * (b % 10) < 10 and (a // 10) * (b % 10) < 10):
            continue
        ans = a * b
        question = f"{a}\uC5D0 {b}\uB97C \uACF1\uD558\uBA74?"
        expr = f"{a}*{b}"
        prompt, style = rng.choice(prompt_variants(expr, question, category, True))
        rows.append(make_row(prompt, with_steps(ans, mul_steps(a, b)), str(ans), category, split, style, True))
    return rows


def gen_div_exact(rng: random.Random, split: str, count: int, digits: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    while len(rows) < count:
        divisor = rng.randint(2, 9)
        quotient = rng.randint(2, 99)
        dividend = divisor * quotient
        if len(str(dividend)) != digits:
            continue
        question = f"{dividend}\uC744 {divisor}\uB85C \uB098\uB204\uBA74?"
        expr = f"{dividend}/{divisor}"
        use_steps = digits >= 3 or rng.random() < 0.4
        prompt, style = rng.choice(prompt_variants(expr, question, f"div_{digits}by1_easy", use_steps))
        output = with_steps(quotient, div_steps(dividend, divisor, quotient)) if use_steps else final_only(quotient)
        rows.append(
            make_row(
                prompt,
                output,
                str(quotient),
                f"div_{digits}by1_easy",
                split,
                style,
                use_steps,
            )
        )
    return rows


def build_rows(
    split: str,
    counts: Dict[str, int],
    seed: int,
    general_replay_path: Path,
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    add_general_replay(
        rows,
        path=general_replay_path,
        rng=rng,
        count=counts["general_replay"],
        split=split,
    )
    rows.extend(gen_addsub_d1(rng, split, counts["addsub_d1"]))
    rows.extend(gen_add_carry(rng, split, counts["add_carry_d2"], 2))
    rows.extend(gen_sub_borrow(rng, split, counts["sub_borrow_d2"], 2))
    rows.extend(gen_add_carry(rng, split, counts["add_carry_d3"], 3))
    rows.extend(gen_sub_borrow(rng, split, counts["sub_borrow_d3"], 3))
    rows.extend(gen_add_carry(rng, split, counts["add_carry_d4"], 4))
    rows.extend(gen_sub_borrow(rng, split, counts["sub_borrow_d4"], 4))
    rows.extend(gen_mul_1x1(rng, split, counts["mul_1x1"]))
    rows.extend(gen_mul_2x1(rng, split, counts["mul_2x1"]))
    rows.extend(gen_mul_2x2(rng, split, counts["mul_2x2_easy"], hard_only=False, category="mul_2x2_easy"))
    rows.extend(gen_mul_2x2(rng, split, counts["mul_2x2_hard"], hard_only=True, category="mul_2x2_hard"))
    rows.extend(gen_div_exact(rng, split, counts["div_2by1_easy"], 2))
    rows.extend(gen_div_exact(rng, split, counts["div_3by1_easy"], 3))
    rng.shuffle(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_out", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--eval_out", type=Path, default=EVAL_DEFAULT)
    parser.add_argument("--manifest_out", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--general_replay_path", type=Path, default=GENERAL_REPLAY_DEFAULT)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260304)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_counts = {k: scaled(v, args.scale) for k, v in TRAIN_COUNTS.items()}
    eval_counts = {k: scaled(v, args.scale) for k, v in EVAL_COUNTS.items()}
    train_rows = build_rows("train", train_counts, args.seed, args.general_replay_path)
    eval_rows = build_rows("eval", eval_counts, args.seed + 1, args.general_replay_path)
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
        "general_replay_path": str(args.general_replay_path),
        "segment_tag": "ko_teacher_arith_pure_mix_v3",
    }
    args.manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
