from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple


DEFAULT_TRAIN_OUT = Path("data/numeric_symbolic_bootstrap_v1_train.jsonl")
DEFAULT_EVAL_OUT = Path("data/numeric_symbolic_bootstrap_v1_eval.jsonl")
DEFAULT_MANIFEST_OUT = Path("data/numeric_symbolic_bootstrap_v1.manifest.json")


def make_row(prompt: str, answer: str, category: str, split: str) -> Dict[str, str]:
    return {
        "input": str(prompt).strip(),
        "output": str(answer).strip(),
        "final_answer": str(answer).strip(),
        "category": str(category).strip(),
        "split": str(split).strip(),
        "task_type": "symbolic",
        "segment_tag": "numeric_symbolic",
        "language": "symbolic",
        "answer_format": "value_only",
        "_meta_source_file": "numeric_symbolic_bootstrap_v1",
    }


def gen_addsub(rng: random.Random, split: str) -> Dict[str, str]:
    a = rng.randint(0, 9999)
    b = rng.randint(0, 9999)
    op = rng.choice(["+", "-"])
    if op == "-" and b > a:
        a, b = b, a
    ans = a + b if op == "+" else a - b
    return make_row(f"{a}{op}{b}=", str(ans), "addsub", split)


def gen_mul(rng: random.Random, split: str) -> Dict[str, str]:
    a = rng.randint(2, 999)
    b = rng.randint(2, 99)
    ans = a * b
    return make_row(f"{a}*{b}=", str(ans), "mul", split)


def gen_div(rng: random.Random, split: str) -> Dict[str, str]:
    b = rng.randint(2, 99)
    ans = rng.randint(2, 999)
    a = b * ans
    return make_row(f"{a}/{b}=", str(ans), "div", split)


def gen_ratio(rng: random.Random, split: str) -> Dict[str, str]:
    left = rng.randint(1, 9)
    right = rng.randint(1, 9)
    scale = rng.randint(2, 500)
    known = right * scale
    ans = left * scale
    return make_row(f"x:y={left}:{right},y={known},x=?", str(ans), "ratio", split)


def gen_compare_min(rng: random.Random, split: str) -> Dict[str, str]:
    vals = rng.sample(range(1, 5000), k=3)
    labels = ["A", "B", "C"]
    pairs = [f"{labels[i]}={vals[i]}" for i in range(3)]
    ans = labels[min(range(3), key=lambda i: vals[i])]
    return make_row(",".join(pairs) + ",min=?", ans, "compare_min", split)


def gen_compare_max(rng: random.Random, split: str) -> Dict[str, str]:
    vals = rng.sample(range(1, 5000), k=3)
    labels = ["A", "B", "C"]
    pairs = [f"{labels[i]}={vals[i]}" for i in range(3)]
    ans = labels[max(range(3), key=lambda i: vals[i])]
    return make_row(",".join(pairs) + ",max=?", ans, "compare_max", split)


GENERATORS: List[Tuple[str, Callable[[random.Random, str], Dict[str, str]], float]] = [
    ("addsub", gen_addsub, 0.30),
    ("mul", gen_mul, 0.20),
    ("div", gen_div, 0.15),
    ("ratio", gen_ratio, 0.15),
    ("compare_min", gen_compare_min, 0.10),
    ("compare_max", gen_compare_max, 0.10),
]


def sample_generator(rng: random.Random) -> Callable[[random.Random, str], Dict[str, str]]:
    r = rng.random()
    acc = 0.0
    for _name, fn, weight in GENERATORS:
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
        cat = str(row["category"])
        stats[f"{split}_{cat}"] = int(stats.get(f"{split}_{cat}", 0)) + 1
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_size", type=int, default=96000)
    p.add_argument("--eval_size", type=int, default=1024)
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
        "output_format": "value_only",
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
