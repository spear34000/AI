from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple


GENERAL_PATH = Path("data/repair_ko_direct_v3.jsonl")
TRAIN_OUT = Path("data/teacher_exact_short_latex_mix_v1_train.jsonl")
EVAL_OUT = Path("data/teacher_exact_short_latex_mix_v1_eval.jsonl")
MANIFEST_OUT = Path("data/teacher_exact_short_latex_mix_v1.manifest.json")


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def simplify_fraction(numer: int, denom: int) -> str:
    g = math.gcd(int(numer), int(denom))
    numer //= g
    denom //= g
    if denom == 1:
        return str(numer)
    return f"{numer}/{denom}"


def make_exact_record(*, prompt: str, final_answer: str, category: str, split: str) -> Dict[str, str]:
    final_answer = str(final_answer).strip()
    return {
        "input": str(prompt).strip(),
        "output": f"최종답: {final_answer}",
        "task_type": "korean",
        "segment_tag": "ko_exact_short",
        "language": "ko",
        "category": str(category),
        "split": str(split),
        "final_answer": final_answer,
        "answer_format": "최종답: ...",
        "_meta_source_file": "teacher_exact_short_latex_mix_v1",
    }


def gen_addsub(rng: random.Random, split: str) -> Dict[str, str]:
    a = rng.randint(100, 999999)
    b = rng.randint(10, 999999)
    op = rng.choice(["+", "-"])
    if op == "-" and b > a:
        a, b = b, a
    ans = a + b if op == "+" else a - b
    return make_exact_record(
        prompt=f"다음을 계산하라. {a} {op} {b}\n마지막 줄에 최종답만 써라.",
        final_answer=str(ans),
        category="arithmetic_addsub",
        split=split,
    )


def gen_mul(rng: random.Random, split: str) -> Dict[str, str]:
    a = rng.randint(12, 9999)
    b = rng.randint(2, 999)
    ans = a * b
    return make_exact_record(
        prompt=f"다음을 계산하라. {a} × {b}\n마지막 줄에 최종답만 써라.",
        final_answer=str(ans),
        category="arithmetic_mul",
        split=split,
    )


def gen_div(rng: random.Random, split: str) -> Dict[str, str]:
    b = rng.randint(2, 999)
    ans = rng.randint(2, 9999)
    a = b * ans
    return make_exact_record(
        prompt=f"다음을 계산하라. {a} ÷ {b}\n마지막 줄에 최종답만 써라.",
        final_answer=str(ans),
        category="arithmetic_div",
        split=split,
    )


def gen_ratio(rng: random.Random, split: str) -> Dict[str, str]:
    left, right = rng.choice(
        [
            ("설탕", "밀가루"),
            ("사과", "배"),
            ("빨간 구슬", "파란 구슬"),
            ("남학생", "여학생"),
        ]
    )
    unit = rng.choice(["g", "개", "명"])
    a = rng.randint(1, 9)
    b = rng.randint(1, 9)
    scale = rng.randint(4, 300)
    known = b * scale
    ans = a * scale
    return make_exact_record(
        prompt=(
            f"{left}과 {right}의 비율이 {a}:{b}이다. "
            f"{right}가 {known}{unit}이면 {left}은 얼마인가?\n"
            "마지막 줄에 최종답만 써라."
        ),
        final_answer=f"{ans}{unit}",
        category="ratio",
        split=split,
    )


def gen_probability(rng: random.Random, split: str) -> Dict[str, str]:
    colors = ["빨간", "파란", "초록", "노란"]
    target = rng.randrange(len(colors))
    counts = [rng.randint(1, 8) for _ in colors]
    total = sum(counts)
    numer = counts[target]
    ans = simplify_fraction(numer, total)
    joined = ", ".join(f"{colors[i]} 공 {counts[i]}개" for i in range(len(colors)))
    return make_exact_record(
        prompt=(
            f"주머니에 {joined}가 들어 있다. 한 개를 무작위로 꺼낼 때 "
            f"{colors[target]} 공일 확률을 기약분수로 구하라.\n"
            "마지막 줄에 최종답만 써라."
        ),
        final_answer=ans,
        category="probability",
        split=split,
    )


def gen_compare(rng: random.Random, split: str) -> Dict[str, str]:
    labels = ["A", "B", "C", "D"]
    picked = rng.sample(labels, k=3)
    vals = rng.sample(range(10, 99999), k=3)
    ask = rng.choice(["가장 작은", "가장 큰"])
    pairs = ", ".join(f"{picked[i]}={vals[i]}" for i in range(3))
    if ask == "가장 작은":
        ans = picked[min(range(3), key=lambda i: vals[i])]
        cat = "compare_smallest"
    else:
        ans = picked[max(range(3), key=lambda i: vals[i])]
        cat = "compare_largest"
    return make_exact_record(
        prompt=f"{pairs}일 때 {ask} 값을 가진 항목을 답하라.\n마지막 줄에 최종답만 써라.",
        final_answer=ans,
        category=cat,
        split=split,
    )


def gen_linear(rng: random.Random, split: str) -> Dict[str, str]:
    x = rng.randint(2, 120)
    add_v = rng.randint(1, 40)
    mul_v = rng.randint(2, 12)
    out = (x + add_v) * mul_v
    return make_exact_record(
        prompt=(
            f"어떤 수에 {add_v}를 더하고 {mul_v}를 곱하면 {out}이 된다. "
            "그 수를 구하라.\n마지막 줄에 최종답만 써라."
        ),
        final_answer=str(x),
        category="linear_equation",
        split=split,
    )


def gen_clock(rng: random.Random, split: str) -> Dict[str, str]:
    per_day = rng.randint(1, 18)
    days = rng.randint(2, 30)
    ans = per_day * days
    return make_exact_record(
        prompt=(
            f"시계가 하루에 {per_day}분 느려진다. 정확한 시간에 맞춘 뒤 "
            f"{days}일이 지나면 몇 분 차이인가?\n마지막 줄에 최종답만 써라."
        ),
        final_answer=f"{ans}분",
        category="clock_drift",
        split=split,
    )


def gen_sequence(rng: random.Random, split: str) -> Dict[str, str]:
    mode = rng.choice(["ap", "gp", "alt"])
    if mode == "ap":
        start = rng.randint(2, 50)
        diff = rng.randint(2, 14)
        seq = [start + diff * i for i in range(6)]
        ans = seq[-1] + diff
    elif mode == "gp":
        start = rng.randint(2, 8)
        ratio = rng.randint(2, 4)
        seq = [start]
        for _ in range(5):
            seq.append(seq[-1] * ratio)
        ans = seq[-1] * ratio
    else:
        start = rng.randint(2, 12)
        mul = rng.randint(2, 4)
        add = rng.randint(1, 8)
        seq = [start]
        for i in range(5):
            prev = seq[-1]
            seq.append(prev * mul if i % 2 == 0 else prev + add)
        ans = seq[-1] * mul
    return make_exact_record(
        prompt=f"수열 {', '.join(str(x) for x in seq)}, ? 의 다음 수를 구하라.\n마지막 줄에 최종답만 써라.",
        final_answer=str(ans),
        category="sequence",
        split=split,
    )


def gen_syllogism(rng: random.Random, split: str) -> Dict[str, str]:
    mode = rng.choice(["yes", "no", "unknown"])
    if mode == "yes":
        a, b, c = rng.sample(["A", "B", "C", "D", "E"], k=3)
        prompt = (
            f"모든 {a}는 {b}다. 모든 {b}는 {c}다. "
            f"따라서 모든 {a}는 {c}인가?\n마지막 줄에 최종답만 써라."
        )
        ans = "예"
    elif mode == "no":
        a, b = rng.sample(["A", "B", "C", "D", "E"], k=2)
        x = rng.choice(["x", "y", "z"])
        prompt = (
            f"어떤 {a}도 {b}가 아니다. {x}는 {a}다. "
            f"따라서 {x}는 {b}인가?\n마지막 줄에 최종답만 써라."
        )
        ans = "아니오"
    else:
        a, b, c = rng.sample(["A", "B", "C", "D", "E"], k=3)
        prompt = (
            f"모든 {a}는 {b}다. 일부 {b}는 {c}다. "
            f"따라서 일부 {a}는 {c}인가?\n마지막 줄에 최종답만 써라."
        )
        ans = "알 수 없음"
    return make_exact_record(
        prompt=prompt,
        final_answer=ans,
        category="syllogism",
        split=split,
    )


def gen_latex_fraction(rng: random.Random, split: str) -> Dict[str, str]:
    numer = rng.randint(1, 40)
    denom = rng.randint(2, 40)
    return make_exact_record(
        prompt=f"분수 {numer}/{denom}를 LaTeX 수식으로 써라.\n마지막 줄에 최종답만 써라.",
        final_answer=f"\\frac{{{numer}}}{{{denom}}}",
        category="latex_fraction",
        split=split,
    )


def gen_latex_equation(rng: random.Random, split: str) -> Dict[str, str]:
    mode = rng.choice(["square", "linear", "sum"])
    if mode == "square":
        n = rng.randint(2, 14)
        final = f"(x+{n})^2=x^2+{2*n}x+{n*n}"
        prompt = f"(x+{n})^2의 전개식을 LaTeX로 써라.\n마지막 줄에 최종답만 써라."
    elif mode == "linear":
        a = rng.randint(2, 15)
        b = rng.randint(1, 20)
        c = rng.randint(20, 200)
        final = f"{a}x+{b}={c}"
        prompt = f"일차방정식 {a}x + {b} = {c}를 LaTeX로 써라.\n마지막 줄에 최종답만 써라."
    else:
        upper = rng.randint(5, 20)
        final = f"\\sum_{{i=1}}^{{{upper}}}i"
        prompt = f"1부터 {upper}까지의 합을 나타내는 시그마 수식을 LaTeX로 써라.\n마지막 줄에 최종답만 써라."
    return make_exact_record(
        prompt=prompt,
        final_answer=final,
        category="latex_equation",
        split=split,
    )


GENERATORS: List[Tuple[str, float, object]] = [
    ("arithmetic_addsub", 0.16, gen_addsub),
    ("arithmetic_mul", 0.10, gen_mul),
    ("arithmetic_div", 0.10, gen_div),
    ("ratio", 0.11, gen_ratio),
    ("probability", 0.08, gen_probability),
    ("compare", 0.09, gen_compare),
    ("linear_equation", 0.10, gen_linear),
    ("clock_drift", 0.06, gen_clock),
    ("sequence", 0.08, gen_sequence),
    ("syllogism", 0.06, gen_syllogism),
    ("latex_fraction", 0.03, gen_latex_fraction),
    ("latex_equation", 0.03, gen_latex_equation),
]


def sample_generator(rng: random.Random):
    r = rng.random()
    acc = 0.0
    for _name, weight, fn in GENERATORS:
        acc += weight
        if r <= acc:
            return fn
    return GENERATORS[-1][2]


def read_general_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            inp = str(row.get("input", "")).strip()
            out = str(row.get("output", "")).strip()
            if len(inp) < 4 or len(out) < 4:
                continue
            rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_exact_size", type=int, default=180000)
    p.add_argument("--eval_exact_size", type=int, default=2048)
    p.add_argument("--general_replay_size", type=int, default=60000)
    p.add_argument("--seed", type=int, default=20260302)
    p.add_argument("--general_path", type=str, default=str(GENERAL_PATH))
    p.add_argument("--train_out", type=str, default=str(TRAIN_OUT))
    p.add_argument("--eval_out", type=str, default=str(EVAL_OUT))
    p.add_argument("--manifest_out", type=str, default=str(MANIFEST_OUT))
    args = p.parse_args()

    rng = random.Random(int(args.seed))
    train_rows: List[Dict[str, str]] = []
    eval_rows: List[Dict[str, str]] = []
    stats: Dict[str, int] = {}

    for split, size, bucket in [
        ("train", int(args.train_exact_size), train_rows),
        ("eval", int(args.eval_exact_size), eval_rows),
    ]:
        for _ in range(size):
            row = sample_generator(rng)(rng, split)
            bucket.append(row)
            key = f"{split}_{row['category']}"
            stats[key] = int(stats.get(key, 0)) + 1

    general_rows = read_general_rows(Path(args.general_path))
    rng.shuffle(general_rows)
    replay_size = min(int(args.general_replay_size), len(general_rows))
    replay_rows = []
    for row in general_rows[:replay_size]:
        row2 = dict(row)
        row2["split"] = "train"
        row2["_mix_source"] = "general_replay"
        replay_rows.append(row2)
    mixed_train = list(train_rows) + replay_rows
    rng.shuffle(mixed_train)

    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    manifest_out = Path(args.manifest_out)
    write_jsonl(train_out, mixed_train)
    write_jsonl(eval_out, eval_rows)

    manifest = {
        "seed": int(args.seed),
        "general_path": str(args.general_path),
        "train_exact_size": len(train_rows),
        "eval_exact_size": len(eval_rows),
        "general_replay_size": len(replay_rows),
        "train_written": len(mixed_train),
        "eval_written": len(eval_rows),
        "categories": stats,
        "output_format": "최종답 only",
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
