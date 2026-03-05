from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple


DEFAULT_TRAIN_OUT = Path("data/numeric_arith_bootstrap_v2_train.jsonl")
DEFAULT_EVAL_OUT = Path("data/numeric_arith_bootstrap_v2_eval.jsonl")
DEFAULT_MANIFEST_OUT = Path("data/numeric_arith_bootstrap_v2.manifest.json")


def make_row(prompt: str, answer: str, category: str, split: str) -> Dict[str, str]:
    return {
        "input": str(prompt).strip(),
        "output": str(answer).strip(),
        "final_answer": str(answer).strip(),
        "category": str(category).strip(),
        "split": str(split).strip(),
        "task_type": "symbolic",
        "segment_tag": "numeric_arith",
        "language": "symbolic",
        "answer_format": "value_only",
        "_meta_source_file": "numeric_arith_bootstrap_v2",
    }


def gen_add(rng: random.Random, split: str) -> Dict[str, str]:
    a = rng.randint(0, 99999)
    b = rng.randint(0, 99999)
    return make_row(f"{a}+{b}=", str(a + b), "add", split)


def gen_sub(rng: random.Random, split: str) -> Dict[str, str]:
    a = rng.randint(0, 99999)
    b = rng.randint(0, 99999)
    if b > a:
        a, b = b, a
    return make_row(f"{a}-{b}=", str(a - b), "sub", split)


def gen_mul(rng: random.Random, split: str) -> Dict[str, str]:
    a = rng.randint(2, 9999)
    b = rng.randint(2, 999)
    return make_row(f"{a}*{b}=", str(a * b), "mul", split)


def gen_div(rng: random.Random, split: str) -> Dict[str, str]:
    b = rng.randint(2, 999)
    q = rng.randint(2, 9999)
    a = b * q
    return make_row(f"{a}/{b}=", str(q), "div", split)


GENERATORS: List[Tuple[str, Callable[[random.Random, str], Dict[str, str]], float]] = [
    ("add", gen_add, 0.30),
    ("sub", gen_sub, 0.30),
    ("mul", gen_mul, 0.25),
    ("div", gen_div, 0.15),
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
        key = f"{split}_{row['category']}"
        stats[key] = int(stats.get(key, 0)) + 1
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_size", type=int, default=120000)
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
        "output_format": "value_only",
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
