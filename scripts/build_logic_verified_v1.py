from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple


TRAIN_OUT_PATH = Path("data/logic_verified_v1_train.jsonl")
EVAL_OUT_PATH = Path("data/logic_verified_v1_eval.jsonl")
MANIFEST_OUT_PATH = Path("data/logic_verified_v1.manifest.json")


def simplify_fraction(numer: int, denom: int) -> str:
    g = math.gcd(int(numer), int(denom))
    numer //= g
    denom //= g
    if denom == 1:
        return str(numer)
    return f"{numer}/{denom}"


def make_record(
    *,
    prompt: str,
    explanation: str,
    final_answer: str,
    category: str,
    split: str,
) -> Dict[str, str]:
    prompt = str(prompt).strip()
    explanation = str(explanation).strip()
    final_answer = str(final_answer).strip()
    output = f"설명: {explanation}\n최종답: {final_answer}"
    return {
        "input": prompt,
        "output": output,
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "category": category,
        "split": split,
        "final_answer": final_answer,
        "answer_format": "최종답: ...",
        "_meta_source_file": "logic_verified_v1",
    }


def gen_arithmetic_addsub(rng: random.Random, split: str) -> Dict[str, str]:
    a = rng.randint(10, 999)
    b = rng.randint(10, 999)
    op = rng.choice(["+", "-"])
    if op == "-" and b > a:
        a, b = b, a
    ans = a + b if op == "+" else a - b
    prompt = (
        f"다음 계산을 하라: {a} {op} {b}\n"
        "답변 형식:\n설명: 한 줄 설명\n최종답: 숫자 하나"
    )
    explanation = f"{a} {op} {b}를 계산하면 {ans}이다."
    return make_record(
        prompt=prompt,
        explanation=explanation,
        final_answer=str(ans),
        category="arithmetic_addsub",
        split=split,
    )


def gen_arithmetic_muldiv(rng: random.Random, split: str) -> Dict[str, str]:
    if rng.random() < 0.5:
        a = rng.randint(2, 39)
        b = rng.randint(2, 19)
        ans = a * b
        prompt = (
            f"다음 계산을 하라: {a} × {b}\n"
            "답변 형식:\n설명: 한 줄 설명\n최종답: 숫자 하나"
        )
        explanation = f"{a}에 {b}를 곱하면 {ans}이다."
    else:
        b = rng.randint(2, 25)
        ans = rng.randint(2, 80)
        a = b * ans
        prompt = (
            f"다음 계산을 하라: {a} ÷ {b}\n"
            "답변 형식:\n설명: 한 줄 설명\n최종답: 숫자 하나"
        )
        explanation = f"{a}를 {b}로 나누면 {ans}이다."
    return make_record(
        prompt=prompt,
        explanation=explanation,
        final_answer=str(ans),
        category="arithmetic_muldiv",
        split=split,
    )


def gen_probability(rng: random.Random, split: str) -> Dict[str, str]:
    colors = ["빨간", "파란", "초록", "노란"]
    target_idx = rng.randrange(len(colors))
    counts = [rng.randint(1, 7) for _ in colors]
    total = sum(counts)
    numer = counts[target_idx]
    answer = simplify_fraction(numer, total)
    parts = [f"{colors[i]} 공 {counts[i]}개" for i in range(len(colors))]
    prompt = (
        f"주머니에 {', '.join(parts)}가 있다. 한 개를 무작위로 뽑을 때 "
        f"{colors[target_idx]} 공일 확률을 기약분수로 구하라.\n"
        "답변 형식:\n설명: 한 줄 설명\n최종답: 기약분수"
    )
    explanation = f"전체는 {total}개이고 {colors[target_idx]} 공은 {numer}개이므로 확률은 {numer}/{total}이고 기약분수는 {answer}이다."
    return make_record(
        prompt=prompt,
        explanation=explanation,
        final_answer=answer,
        category="probability",
        split=split,
    )


def gen_ratio(rng: random.Random, split: str) -> Dict[str, str]:
    a = rng.randint(1, 9)
    b = rng.randint(1, 9)
    mult = rng.randint(5, 80)
    known = b * mult
    ans = a * mult
    item_left = rng.choice(["설탕", "연필", "빨간 구슬", "남학생"])
    item_right = rng.choice(["밀가루", "공책", "파란 구슬", "여학생"])
    prompt = (
        f"{item_left}과 {item_right}의 비율이 {a}:{b}이다. {item_right}이 {known}개라면 {item_left}은 몇 개인가?\n"
        "답변 형식:\n설명: 한 줄 설명\n최종답: 숫자 하나"
    )
    explanation = f"{item_right} {known}개는 비율 {b}에 해당하므로 한 단위는 {mult}이고, {item_left}은 {a}단위라서 {ans}개이다."
    return make_record(
        prompt=prompt,
        explanation=explanation,
        final_answer=str(ans),
        category="ratio",
        split=split,
    )


def gen_compare(rng: random.Random, split: str) -> Dict[str, str]:
    names = rng.sample(["A", "B", "C", "D"], k=3)
    prompt = (
        f"{names[0]}는 {names[1]}보다 크고, {names[1]}는 {names[2]}보다 크다. "
        "가장 작은 것을 답하라.\n"
        "답변 형식:\n설명: 한 줄 설명\n최종답: 대상 하나"
    )
    explanation = f"{names[0]} > {names[1]} > {names[2]} 이므로 가장 작은 것은 {names[2]}이다."
    return make_record(
        prompt=prompt,
        explanation=explanation,
        final_answer=names[2],
        category="compare_smallest",
        split=split,
    )


def gen_syllogism(rng: random.Random, split: str) -> Dict[str, str]:
    mode = rng.choice(["yes", "no", "unknown"])
    if mode == "yes":
        a, b, c = rng.sample(["A", "B", "C", "D", "E"], k=3)
        prompt = (
            f"모든 {a}는 {b}다. 모든 {b}는 {c}다. 그러면 모든 {a}는 {c}인가?\n"
            "답변 형식:\n설명: 한 줄 설명\n최종답: 예/아니오/알 수 없음"
        )
        explanation = f"{a}는 모두 {b}에 포함되고 {b}는 모두 {c}에 포함되므로 {a}는 모두 {c}이다."
        answer = "예"
    elif mode == "no":
        a, b = rng.sample(["A", "B", "C", "D", "E"], k=2)
        x = rng.choice(["x", "y", "z"])
        prompt = (
            f"어떤 {a}도 {b}가 아니다. {x}는 {a}다. 그러면 {x}는 {b}인가?\n"
            "답변 형식:\n설명: 한 줄 설명\n최종답: 예/아니오/알 수 없음"
        )
        explanation = f"{a}와 {b}는 겹치지 않고 {x}는 {a}이므로 {x}는 {b}가 아니다."
        answer = "아니오"
    else:
        a, b, c = rng.sample(["A", "B", "C", "D", "E"], k=3)
        prompt = (
            f"모든 {a}는 {b}다. 일부 {b}는 {c}다. 그러면 일부 {a}는 {c}인가?\n"
            "답변 형식:\n설명: 한 줄 설명\n최종답: 예/아니오/알 수 없음"
        )
        explanation = f"일부 {b}가 {c}라는 정보만으로 그 일부가 {a}인지 알 수 없으므로 결론을 보장할 수 없다."
        answer = "알 수 없음"
    return make_record(
        prompt=prompt,
        explanation=explanation,
        final_answer=answer,
        category="syllogism",
        split=split,
    )


def gen_sequence(rng: random.Random, split: str) -> Dict[str, str]:
    mode = rng.choice(["ap", "gp", "alt"])
    if mode == "ap":
        start = rng.randint(1, 30)
        diff = rng.randint(2, 15)
        seq = [start + diff * i for i in range(6)]
        ans = seq[-1] + diff
        explanation = f"항이 매번 {diff}씩 증가하므로 다음 수는 {ans}이다."
    elif mode == "gp":
        start = rng.randint(1, 8)
        ratio = rng.randint(2, 4)
        seq = [start]
        for _ in range(5):
            seq.append(seq[-1] * ratio)
        ans = seq[-1] * ratio
        explanation = f"항이 매번 {ratio}배가 되므로 다음 수는 {ans}이다."
    else:
        start = rng.randint(2, 12)
        mul = rng.randint(2, 4)
        add = rng.randint(1, 9)
        seq = [start]
        for i in range(5):
            prev = seq[-1]
            seq.append(prev * mul if i % 2 == 0 else prev + add)
        prev = seq[-1]
        ans = prev * mul if 5 % 2 == 0 else prev + add
        explanation = f"규칙이 번갈아 ×{mul}, +{add} 이므로 다음 수는 {ans}이다."
    prompt = (
        f"수열 {', '.join(str(x) for x in seq)}, ? 의 다음 수를 구하라.\n"
        "답변 형식:\n설명: 한 줄 설명\n최종답: 숫자 하나"
    )
    return make_record(
        prompt=prompt,
        explanation=explanation,
        final_answer=str(ans),
        category="sequence",
        split=split,
    )


GENERATORS: List[Tuple[str, Callable[[random.Random, str], Dict[str, str]], float]] = [
    ("arithmetic_addsub", gen_arithmetic_addsub, 0.22),
    ("arithmetic_muldiv", gen_arithmetic_muldiv, 0.18),
    ("probability", gen_probability, 0.14),
    ("ratio", gen_ratio, 0.14),
    ("compare_smallest", gen_compare, 0.12),
    ("syllogism", gen_syllogism, 0.10),
    ("sequence", gen_sequence, 0.10),
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_size", type=int, default=48000)
    p.add_argument("--eval_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_out", type=str, default=str(TRAIN_OUT_PATH))
    p.add_argument("--eval_out", type=str, default=str(EVAL_OUT_PATH))
    p.add_argument("--manifest_out", type=str, default=str(MANIFEST_OUT_PATH))
    args = p.parse_args()

    rng = random.Random(int(args.seed))
    train_rows: List[Dict[str, str]] = []
    eval_rows: List[Dict[str, str]] = []
    stats: Dict[str, int] = {}

    for split, size, bucket in [("train", int(args.train_size), train_rows), ("eval", int(args.eval_size), eval_rows)]:
        for _ in range(size):
            fn = sample_generator(rng)
            row = fn(rng, split)
            bucket.append(row)
            cat = str(row.get("category", "unknown"))
            stats[f"{split}_{cat}"] = int(stats.get(f"{split}_{cat}", 0)) + 1

    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    manifest_out = Path(args.manifest_out)
    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)

    manifest = {
        "train_out": str(train_out),
        "eval_out": str(eval_out),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "seed": int(args.seed),
        "categories": stats,
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
