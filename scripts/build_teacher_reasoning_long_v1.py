from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple


DEFAULT_TRAIN_OUT = Path("data/teacher_reasoning_long_v1_train.jsonl")
DEFAULT_EVAL_OUT = Path("data/teacher_reasoning_long_v1_eval.jsonl")
DEFAULT_MANIFEST_OUT = Path("data/teacher_reasoning_long_v1.manifest.json")


def simplify_fraction(numer: int, denom: int) -> str:
    g = math.gcd(int(numer), int(denom))
    numer //= g
    denom //= g
    if denom == 1:
        return str(numer)
    return f"{numer}/{denom}"


def make_output(explanation_lines: List[str], final_answer: str) -> str:
    body = "\n".join(str(x).strip() for x in explanation_lines if str(x).strip())
    return f"{body}\n최종답: {str(final_answer).strip()}"


def make_record(
    *,
    prompt: str,
    explanation_lines: List[str],
    final_answer: str,
    category: str,
    split: str,
) -> Dict[str, str]:
    return {
        "input": str(prompt).strip(),
        "output": make_output(explanation_lines, final_answer),
        "task_type": "korean",
        "segment_tag": "ko_reasoning_long",
        "language": "ko",
        "category": str(category),
        "split": str(split),
        "final_answer": str(final_answer).strip(),
        "answer_format": "최종답: ...",
        "_meta_source_file": "teacher_reasoning_long_v1",
    }


def gen_addsub(rng: random.Random, split: str) -> Dict[str, str]:
    a = rng.randint(100, 99999)
    b = rng.randint(100, 99999)
    op = rng.choice(["+", "-"])
    if op == "-" and b > a:
        a, b = b, a
    ans = a + b if op == "+" else a - b
    prompt = f"다음 계산을 단계별로 풀고 마지막 줄에 최종답을 써라.\n문제: {a} {op} {b}"
    lines = [
        f"1. 식은 {a} {op} {b} 이다.",
        "2. 자릿수 계산 순서를 유지하면서 값을 구한다.",
        f"3. 계산 결과는 {ans} 이다.",
    ]
    return make_record(prompt=prompt, explanation_lines=lines, final_answer=str(ans), category="arithmetic_addsub", split=split)


def gen_muldiv(rng: random.Random, split: str) -> Dict[str, str]:
    if rng.random() < 0.5:
        a = rng.randint(12, 9999)
        b = rng.randint(2, 999)
        ans = a * b
        prompt = f"다음 계산을 단계별로 풀고 마지막 줄에 최종답을 써라.\n문제: {a} × {b}"
        lines = [
            f"1. 곱셈 문제는 {a} × {b} 이다.",
            f"2. 한 수를 다른 수만큼 묶어 계산하면 된다.",
            f"3. 곱한 값은 {ans} 이다.",
        ]
        cat = "arithmetic_mul"
    else:
        d = rng.randint(2, 999)
        ans = rng.randint(2, 9999)
        n = d * ans
        prompt = f"다음 계산을 단계별로 풀고 마지막 줄에 최종답을 써라.\n문제: {n} ÷ {d}"
        lines = [
            f"1. 나눗셈 문제는 {n} ÷ {d} 이다.",
            f"2. {d} 를 몇 번 곱하면 {n} 이 되는지 찾는다.",
            f"3. 몫은 {ans} 이다.",
        ]
        cat = "arithmetic_div"
    return make_record(prompt=prompt, explanation_lines=lines, final_answer=str(ans), category=cat, split=split)


def gen_ratio(rng: random.Random, split: str) -> Dict[str, str]:
    left_name, right_name = rng.choice(
        [
            ("설탕", "밀가루"),
            ("사과", "배"),
            ("빨간 구슬", "파란 구슬"),
            ("남학생", "여학생"),
        ]
    )
    a = rng.randint(1, 9)
    b = rng.randint(1, 9)
    scale = rng.randint(3, 300)
    known = b * scale
    ans = a * scale
    unit = rng.choice(["g", "개", "명"])
    prompt = (
        f"{left_name}과 {right_name}의 비율이 {a}:{b} 이다. "
        f"{right_name}가 {known}{unit} 이면 {left_name}은 얼마인가? 단계별로 설명하고 마지막 줄에 최종답을 써라."
    )
    lines = [
        f"1. 비율 {a}:{b} 에서 {right_name} {b}칸이 실제로는 {known}{unit} 이다.",
        f"2. 따라서 한 칸의 크기는 {known}/{b} = {scale}{unit} 이다.",
        f"3. {left_name}은 {a}칸이므로 {a} × {scale} = {ans}{unit} 이다.",
    ]
    return make_record(prompt=prompt, explanation_lines=lines, final_answer=f"{ans}{unit}", category="ratio", split=split)


def gen_probability(rng: random.Random, split: str) -> Dict[str, str]:
    colors = ["빨간 공", "파란 공", "초록 공", "노란 공"]
    target = rng.randrange(len(colors))
    counts = [rng.randint(1, 8) for _ in colors]
    total = sum(counts)
    numer = counts[target]
    answer = simplify_fraction(numer, total)
    joined = ", ".join(f"{colors[i]} {counts[i]}개" for i in range(len(colors)))
    prompt = (
        f"주머니에 {joined}가 들어 있다. "
        f"한 개를 무작위로 꺼낼 때 {colors[target]}일 확률을 기약분수로 구하고 마지막 줄에 최종답을 써라."
    )
    lines = [
        f"1. 전체 공의 개수는 {total}개이다.",
        f"2. 원하는 경우의 수는 {colors[target]} {numer}개이다.",
        f"3. 확률은 {numer}/{total} 이고 기약분수로 줄이면 {answer} 이다.",
    ]
    return make_record(prompt=prompt, explanation_lines=lines, final_answer=answer, category="probability", split=split)


def gen_compare(rng: random.Random, split: str) -> Dict[str, str]:
    labels = ["A", "B", "C"]
    vals = rng.sample(range(10, 9999), k=3)
    ask_min = rng.random() < 0.5
    pairs = ", ".join(f"{labels[i]}={vals[i]}" for i in range(3))
    if ask_min:
        best = labels[min(range(3), key=lambda i: vals[i])]
        desc = "가장 작은 값"
        ordered = sorted(zip(labels, vals), key=lambda kv: kv[1])
    else:
        best = labels[max(range(3), key=lambda i: vals[i])]
        desc = "가장 큰 값"
        ordered = sorted(zip(labels, vals), key=lambda kv: kv[1], reverse=True)
    prompt = f"{pairs} 일 때 {desc}을 가진 항목은 무엇인가? 비교 과정을 쓰고 마지막 줄에 최종답을 써라."
    lines = [
        f"1. 비교 대상은 {pairs} 이다.",
        f"2. 값의 순서를 정리하면 " + " > ".join(f"{k}({v})" for k, v in ordered) if not ask_min else " < ".join(f"{k}({v})" for k, v in ordered),
        f"3. 따라서 {desc}을 가진 항목은 {best} 이다.",
    ]
    return make_record(prompt=prompt, explanation_lines=lines, final_answer=best, category="compare", split=split)


def gen_linear(rng: random.Random, split: str) -> Dict[str, str]:
    x = rng.randint(2, 80)
    add_v = rng.randint(1, 30)
    mul_v = rng.randint(2, 12)
    out = (x + add_v) * mul_v
    prompt = (
        f"어떤 수에 {add_v}를 더하고 {mul_v}를 곱하면 {out}이 된다. "
        f"그 수를 구하는 과정을 쓰고 마지막 줄에 최종답을 써라."
    )
    lines = [
        f"1. 미지수를 x 라고 두면 식은 (x + {add_v}) × {mul_v} = {out} 이다.",
        f"2. 양변을 {mul_v}로 나누면 x + {add_v} = {out // mul_v} 이다.",
        f"3. 양변에서 {add_v}를 빼면 x = {x} 이다.",
    ]
    return make_record(prompt=prompt, explanation_lines=lines, final_answer=str(x), category="linear_equation", split=split)


def gen_clock(rng: random.Random, split: str) -> Dict[str, str]:
    per_day = rng.randint(1, 15)
    days = rng.randint(2, 20)
    ans = per_day * days
    prompt = (
        f"시계가 하루에 {per_day}분 느려진다. 정확한 시간에 맞춘 뒤 {days}일이 지나면 몇 분 차이인가? "
        f"계산 과정을 쓰고 마지막 줄에 최종답을 써라."
    )
    lines = [
        f"1. 하루에 느려지는 시간은 {per_day}분이다.",
        f"2. {days}일 동안 누적 오차는 {per_day} × {days} 이다.",
        f"3. 따라서 전체 차이는 {ans}분이다.",
    ]
    return make_record(prompt=prompt, explanation_lines=lines, final_answer=f"{ans}분", category="clock_drift", split=split)


def gen_syllogism(rng: random.Random, split: str) -> Dict[str, str]:
    kind = rng.choice(["yes", "no", "unknown"])
    a, b, c = rng.sample(["고양이", "동물", "포유류", "검은색인 것", "학생", "사람"], k=3)
    if kind == "yes":
        prompt = f"모든 {a}는 {b}이다. 모든 {b}는 {c}이다. 따라서 모든 {a}는 {c}인가? 이유를 설명하고 마지막 줄에 최종답을 써라."
        answer = "예"
        lines = [
            f"1. 첫 전제에 따르면 {a}는 모두 {b}에 포함된다.",
            f"2. 둘째 전제에 따르면 {b}는 모두 {c}에 포함된다.",
            f"3. 포함 관계를 이어 붙이면 모든 {a}는 {c}라고 결론낼 수 있다.",
        ]
    elif kind == "no":
        prompt = f"어떤 {a}도 {b}가 아니다. 어떤 x는 {a}이다. 따라서 x는 {b}인가? 이유를 설명하고 마지막 줄에 최종답을 써라."
        answer = "아니오"
        lines = [
            f"1. 첫 전제는 {a}와 {b}가 겹치지 않는다고 말한다.",
            f"2. 둘째 전제에서 x는 {a}에 속한다.",
            f"3. 따라서 x는 {b}일 수 없다.",
        ]
    else:
        prompt = f"모든 {a}는 {b}이다. 어떤 {b}는 {c}이다. 따라서 어떤 {a}는 {c}인가? 이유를 설명하고 마지막 줄에 최종답을 써라."
        answer = "보장할 수 없음"
        lines = [
            f"1. 첫 전제는 {a}가 {b}에 포함된다는 정보만 준다.",
            f"2. 둘째 전제는 {b} 중 일부가 {c}라는 정보만 준다.",
            f"3. 그 일부가 {a}와 겹치는지는 알 수 없으므로 결론을 보장할 수 없다.",
        ]
    return make_record(prompt=prompt, explanation_lines=lines, final_answer=answer, category="syllogism", split=split)


def gen_sequence(rng: random.Random, split: str) -> Dict[str, str]:
    mode = rng.choice(["ap", "gp", "alt"])
    if mode == "ap":
        start = rng.randint(1, 40)
        diff = rng.randint(2, 15)
        seq = [start + diff * i for i in range(6)]
        ans = seq[-1] + diff
        lines = [
            f"1. 이 수열은 앞 항에서 {diff}씩 더해진다.",
            f"2. 마지막 항 {seq[-1]}에 {diff}를 더하면 다음 항을 얻는다.",
            f"3. 따라서 다음 수는 {ans} 이다.",
        ]
    elif mode == "gp":
        start = rng.randint(1, 8)
        ratio = rng.randint(2, 4)
        seq = [start]
        for _ in range(5):
            seq.append(seq[-1] * ratio)
        ans = seq[-1] * ratio
        lines = [
            f"1. 이 수열은 앞 항에 {ratio}를 곱해 다음 항을 만든다.",
            f"2. 마지막 항 {seq[-1]}에 {ratio}를 곱한다.",
            f"3. 따라서 다음 수는 {ans} 이다.",
        ]
    else:
        start = rng.randint(2, 12)
        mul = rng.randint(2, 4)
        add = rng.randint(1, 9)
        seq = [start]
        for i in range(5):
            prev = seq[-1]
            seq.append(prev * mul if i % 2 == 0 else prev + add)
        last = seq[-1]
        ans = last * mul
        lines = [
            f"1. 규칙은 번갈아 가며 ×{mul}, +{add} 이다.",
            f"2. 주어진 마지막 변환은 +{add} 단계까지 적용된 상태이다.",
            f"3. 따라서 다음에는 {last}에 {mul}을 곱해 {ans} 이 된다.",
        ]
    prompt = f"수열 {', '.join(str(x) for x in seq)}, ? 의 다음 수를 구하고 이유를 설명한 뒤 마지막 줄에 최종답을 써라."
    return make_record(prompt=prompt, explanation_lines=lines, final_answer=str(ans), category="sequence", split=split)


GENERATORS: List[Tuple[str, Callable[[random.Random, str], Dict[str, str]], float]] = [
    ("arithmetic_addsub", gen_addsub, 0.18),
    ("arithmetic_muldiv", gen_muldiv, 0.17),
    ("ratio", gen_ratio, 0.14),
    ("probability", gen_probability, 0.12),
    ("compare", gen_compare, 0.12),
    ("linear_equation", gen_linear, 0.10),
    ("clock_drift", gen_clock, 0.06),
    ("syllogism", gen_syllogism, 0.06),
    ("sequence", gen_sequence, 0.05),
]


def sample_generator(rng: random.Random) -> Callable[[random.Random, str], Dict[str, str]]:
    r = rng.random()
    acc = 0.0
    for _, fn, weight in GENERATORS:
        acc += weight
        if r <= acc:
            return fn
    return GENERATORS[-1][1]


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_rows(size: int, split: str, rng: random.Random, stats: Dict[str, int]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _ in range(int(size)):
        fn = sample_generator(rng)
        row = fn(rng, split)
        rows.append(row)
        stats[f"{split}_{row['category']}"] = int(stats.get(f"{split}_{row['category']}", 0)) + 1
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_size", type=int, default=160000)
    p.add_argument("--eval_size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=20260301)
    p.add_argument("--train_out", default=str(DEFAULT_TRAIN_OUT))
    p.add_argument("--eval_out", default=str(DEFAULT_EVAL_OUT))
    p.add_argument("--manifest_out", default=str(DEFAULT_MANIFEST_OUT))
    args = p.parse_args()

    rng = random.Random(int(args.seed))
    stats: Dict[str, int] = {}
    train_rows = build_rows(int(args.train_size), "train", rng, stats)
    eval_rows = build_rows(int(args.eval_size), "eval", rng, stats)

    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    manifest_out = Path(args.manifest_out)
    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)

    manifest = {
        "finished": True,
        "seed": int(args.seed),
        "train_out": str(train_out),
        "eval_out": str(eval_out),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "categories": stats,
        "output_format": "long_explanation_plus_final_answer",
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
