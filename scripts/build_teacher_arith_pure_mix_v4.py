from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TRAIN_DEFAULT = Path("data/teacher_arith_pure_mix_v4_train.jsonl")
EVAL_DEFAULT = Path("data/teacher_arith_pure_mix_v4_eval.jsonl")
EVAL_ARITH_DEFAULT = Path("data/teacher_arith_pure_mix_v4_eval_arith.jsonl")
MANIFEST_DEFAULT = Path("data/teacher_arith_pure_mix_v4.manifest.json")
GENERAL_REPLAY_DEFAULT = Path("data/repair_ko_direct_v3.jsonl")


TRAIN_COUNTS = {
    "general_replay": 80000,
    "addsub_d1": 16000,
    "add_carry_d2": 36000,
    "sub_borrow_d2": 36000,
    "add_carry_d3": 48000,
    "sub_borrow_d3": 48000,
    "add_carry_d4": 24000,
    "sub_borrow_d4": 24000,
    "mul_1x1": 20000,
    "mul_2x1": 36000,
    "mul_2x2_easy": 42000,
    "mul_2x2_hard": 52000,
    "div_2by1_easy": 12000,
    "div_3by1_easy": 12000,
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


PROMPT_LAST = "마지막 줄에 답만 써라."
PROMPT_LAST_FINAL = "마지막 줄에 최종답만 써라."
PROMPT_LAST_SHORT = "짧게 풀이하고 마지막 줄에 답만 써라."

EXPLAIN_PREFIX = "풀이: "


def scaled(count: int, scale: float) -> int:
    return max(1, int(round(float(count) * float(scale))))


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_answer(answer: int, mode: str) -> str:
    if mode == "bare":
        return str(answer)
    if mode == "final":
        return f"최종답: {answer}"
    if mode == "answer":
        return f"정답: {answer}"
    raise ValueError(mode)


def make_output(answer: int, steps: Sequence[str], *, with_steps: bool, mode: str) -> str:
    tail = format_answer(answer, mode)
    if with_steps:
        return EXPLAIN_PREFIX + " ".join(s.strip() for s in steps if s.strip()) + "\n" + tail
    return tail


def make_row(
    prompt: str,
    output_text: str,
    final_answer: str,
    category: str,
    split: str,
    prompt_style: str,
    has_cot: bool,
    source_file: str = "teacher_arith_pure_mix_v4",
) -> Dict[str, str]:
    return {
        "input": prompt.strip(),
        "output": output_text.strip(),
        "task_type": "korean",
        "segment_tag": "ko_teacher_arith_pure_mix_v4",
        "language": "ko",
        "category": category,
        "split": split,
        "prompt_style": prompt_style,
        "final_answer": str(final_answer).strip(),
        "answer_format": "bare|최종답: ...|정답: ...",
        "has_cot": "1" if has_cot else "0",
        "_meta_source_file": source_file,
    }


def prompt_variants(expr: str, question: str, style_hint: str, with_steps: bool) -> List[Tuple[str, str]]:
    suffix = PROMPT_LAST_SHORT if with_steps else random.choice([PROMPT_LAST, PROMPT_LAST_FINAL])
    return [
        (f"{expr}=\n{suffix}", f"{style_hint}_symbolic"),
        (f"다음을 계산하라. {expr}\n{suffix}", f"{style_hint}_calc"),
        (f"{question}\n{suffix}", f"{style_hint}_word"),
    ]


def place_name(index_from_right: int) -> str:
    names = {0: "일의 자리", 1: "십의 자리", 2: "백의 자리", 3: "천의 자리"}
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
        digit = total % 10
        next_carry = total // 10
        text = f"{place_name(i)}: {da}+{db}"
        if carry:
            text += f"+올림 {carry}"
        text += f"={total}"
        if next_carry:
            text += f", {digit}를 쓰고 {next_carry} 올림"
        else:
            text += f", {digit}를 쓴다"
        steps.append(text)
        carry = next_carry
    if carry:
        steps.append(f"남은 올림 {carry}을 앞에 붙인다")
    return steps


def sub_steps(a: int, b: int) -> List[str]:
    a_digits = [int(ch) for ch in str(a)][::-1]
    b_digits = [int(ch) for ch in str(b)][::-1]
    borrow = 0
    steps: List[str] = []
    for i in range(max(len(a_digits), len(b_digits))):
        da = a_digits[i] if i < len(a_digits) else 0
        db = b_digits[i] if i < len(b_digits) else 0
        cur = da - borrow
        text = f"{place_name(i)}: "
        if cur < db:
            text += f"{da}"
            if borrow:
                text += f"-빌림 {borrow}"
            text += f"가 {db}보다 작아 10을 빌려 {cur + 10}-{db}={cur + 10 - db}"
            borrow = 1
        else:
            text += f"{da}"
            if borrow:
                text += f"-빌림 {borrow}"
            text += f"-{db}={cur - db}"
            borrow = 0
        steps.append(text)
    return steps


def mul_steps(a: int, b: int) -> List[str]:
    parts: List[int] = []
    steps: List[str] = []
    for i, ch in enumerate(str(b)[::-1]):
        d = int(ch)
        place = 10**i
        part = a * d * place
        if place == 1:
            steps.append(f"부분곱1: {a}×{d}={a * d}")
        else:
            steps.append(f"부분곱{i+1}: {a}×{d}×{place}={part}")
        parts.append(part)
    if len(parts) > 1:
        steps.append(f"합: {' + '.join(str(x) for x in parts)} = {sum(parts)}")
    return steps


def div_steps(dividend: int, divisor: int, quotient: int) -> List[str]:
    return [f"{dividend}÷{divisor}={quotient}", f"검산: {divisor}×{quotient}={dividend}"]


def choose_answer_mode(rng: random.Random, *, with_steps: bool) -> str:
    if with_steps:
        return rng.choices(["final", "answer", "bare"], weights=[0.45, 0.20, 0.35], k=1)[0]
    return rng.choices(["bare", "final", "answer"], weights=[0.55, 0.30, 0.15], k=1)[0]


def add_general_replay(
    rows: List[Dict[str, str]],
    *,
    path: Path,
    rng: random.Random,
    count: int,
    split: str,
) -> None:
    candidates: List[Dict[str, str]] = []
    blocked = ("최종답", "정답:", "곱하기", "나누면", "더하면", "빼면", "계산하라", "*", "/", "=")
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
            if len(prompt) > 120 or len(output) > 220:
                continue
            joined = prompt + "\n" + output
            if any(tok in joined for tok in blocked):
                continue
            candidates.append(
                {
                    "input": prompt,
                    "output": output,
                    "task_type": "korean",
                    "segment_tag": "ko_teacher_arith_pure_mix_v4_replay",
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
        question = f"{a}와 {b}를 더하면?" if op == "+" else f"{a}에서 {b}를 빼면?"
        expr = f"{a}{op}{b}"
        prompt, style = rng.choice(prompt_variants(expr, question, "addsub_d1", False))
        mode = choose_answer_mode(rng, with_steps=False)
        rows.append(make_row(prompt, make_output(ans, [], with_steps=False, mode=mode), str(ans), "addsub_d1", split, style, False))
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
        question = f"{a}에 {b}를 더하면?"
        expr = f"{a}+{b}"
        use_steps = digits >= 4 or rng.random() < 0.35
        prompt, style = rng.choice(prompt_variants(expr, question, f"add_carry_d{digits}", use_steps))
        mode = choose_answer_mode(rng, with_steps=use_steps)
        rows.append(make_row(prompt, make_output(ans, add_steps(a, b), with_steps=use_steps, mode=mode), str(ans), f"add_carry_d{digits}", split, style, use_steps))
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
        question = f"{a}에서 {b}를 빼면?"
        expr = f"{a}-{b}"
        use_steps = digits >= 4 or rng.random() < 0.35
        prompt, style = rng.choice(prompt_variants(expr, question, f"sub_borrow_d{digits}", use_steps))
        mode = choose_answer_mode(rng, with_steps=use_steps)
        rows.append(make_row(prompt, make_output(ans, sub_steps(a, b), with_steps=use_steps, mode=mode), str(ans), f"sub_borrow_d{digits}", split, style, use_steps))
    return rows


def gen_mul_1x1(rng: random.Random, split: str, count: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(count):
        a = rng.randint(2, 9)
        b = rng.randint(2, 9)
        ans = a * b
        question = f"{a} 곱하기 {b}는?"
        expr = f"{a}*{b}"
        prompt, style = rng.choice(prompt_variants(expr, question, "mul_1x1", False))
        mode = choose_answer_mode(rng, with_steps=False)
        rows.append(make_row(prompt, make_output(ans, [], with_steps=False, mode=mode), str(ans), "mul_1x1", split, style, False))
    return rows


def gen_mul_2x1(rng: random.Random, split: str, count: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(count):
        a = rng.randint(12, 99)
        b = rng.randint(2, 9)
        ans = a * b
        question = f"{a}에 {b}를 곱하면?"
        expr = f"{a}*{b}"
        use_steps = rng.random() < 0.75
        prompt, style = rng.choice(prompt_variants(expr, question, "mul_2x1", use_steps))
        mode = choose_answer_mode(rng, with_steps=use_steps)
        rows.append(make_row(prompt, make_output(ans, mul_steps(a, b), with_steps=use_steps, mode=mode), str(ans), "mul_2x1", split, style, use_steps))
    return rows


def gen_mul_2x2(rng: random.Random, split: str, count: int, *, hard_only: bool, category: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    while len(rows) < count:
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        if hard_only and ((a % 10) * (b % 10) < 10 and (a // 10) * (b % 10) < 10):
            continue
        ans = a * b
        question = f"{a}에 {b}를 곱하면?"
        expr = f"{a}*{b}"
        use_steps = rng.random() < 0.60
        prompt, style = rng.choice(prompt_variants(expr, question, category, use_steps))
        mode = choose_answer_mode(rng, with_steps=use_steps)
        rows.append(make_row(prompt, make_output(ans, mul_steps(a, b), with_steps=use_steps, mode=mode), str(ans), category, split, style, use_steps))
    return rows


def gen_div_exact(rng: random.Random, split: str, count: int, digits: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    while len(rows) < count:
        divisor = rng.randint(2, 9)
        quotient = rng.randint(2, 99)
        dividend = divisor * quotient
        if len(str(dividend)) != digits:
            continue
        question = f"{dividend}을 {divisor}로 나누면?"
        expr = f"{dividend}/{divisor}"
        use_steps = rng.random() < 0.50
        prompt, style = rng.choice(prompt_variants(expr, question, f"div_{digits}by1_easy", use_steps))
        mode = choose_answer_mode(rng, with_steps=use_steps)
        rows.append(make_row(prompt, make_output(quotient, div_steps(dividend, divisor, quotient), with_steps=use_steps, mode=mode), str(quotient), f"div_{digits}by1_easy", split, style, use_steps))
    return rows


def build_rows(split: str, counts: Dict[str, int], seed: int, general_replay_path: Path) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    add_general_replay(rows, path=general_replay_path, rng=rng, count=counts["general_replay"], split=split)
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
    parser.add_argument("--eval_arith_out", type=Path, default=EVAL_ARITH_DEFAULT)
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
    eval_arith_rows = [row for row in eval_rows if str(row.get("category", "")) != "general_replay"]
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, eval_rows)
    write_jsonl(args.eval_arith_out, eval_arith_rows)
    manifest = {
        "seed": int(args.seed),
        "scale": float(args.scale),
        "train_written": len(train_rows),
        "eval_written": len(eval_rows),
        "eval_arith_written": len(eval_arith_rows),
        "train_counts": train_counts,
        "eval_counts": eval_counts,
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "eval_arith_out": str(args.eval_arith_out),
        "general_replay_path": str(args.general_replay_path),
        "segment_tag": "ko_teacher_arith_pure_mix_v4",
    }
    args.manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
