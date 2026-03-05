from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List


TRAIN_DEFAULT = Path("data/mainline_logic_verified_v2_train.jsonl")
EVAL_DEFAULT = Path("data/mainline_logic_verified_v2_eval.jsonl")
MANIFEST_DEFAULT = Path("data/mainline_logic_verified_v2.manifest.json")


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_row(prompt: str, reason: str, answer: str, category: str, split: str, source: str) -> Dict[str, str]:
    output = f"이유: {reason.strip()}\n답: {str(answer).strip()}"
    return {
        "input": str(prompt).strip(),
        "output": output,
        "task_type": "korean",
        "segment_tag": "ko_mainline_logic_v2",
        "language": "ko",
        "category": category,
        "split": split,
        "final_answer": str(answer).strip(),
        "answer_format": "답: ...",
        "_meta_source_file": source,
    }


def simplify_fraction(numer: int, denom: int) -> str:
    g = math.gcd(numer, denom)
    numer //= g
    denom //= g
    if denom == 1:
        return str(numer)
    return f"{numer}/{denom}"


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def import_public_yesno(path: Path, split: str, limit: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in iter_jsonl(path):
        out = str(row.get("output", "")).strip()
        if out not in {"예", "아니오"}:
            continue
        prompt = str(row.get("input", "")).strip()
        if not prompt or len(prompt) > 260:
            continue
        reason = "문장 관계를 비교하면 그렇게 판단할 수 있습니다." if out == "예" else "문장 관계를 비교하면 같지 않다고 판단할 수 있습니다."
        rows.append(make_row(prompt, reason, out, "sentence_relation_yesno", split, path.name))
        if len(rows) >= limit:
            break
    return rows


def gen_compare(rng: random.Random, split: str) -> Dict[str, str]:
    names = rng.sample(["A", "B", "C", "D"], 3)
    answer = names[2]
    prompt = f"{names[0]}는 {names[1]}보다 크고, {names[1]}는 {names[2]}보다 크다. 가장 작은 것은 누구인가?"
    reason = f"{names[0]} > {names[1]} > {names[2]}이므로 가장 작은 것은 {names[2]}입니다."
    return make_row(prompt, reason, answer, "compare_smallest", split, "synthetic_compare")


def gen_syllogism(rng: random.Random, split: str) -> Dict[str, str]:
    tokens = rng.sample(["A", "B", "C", "D", "E"], 3)
    mode = rng.choice(["yes", "unknown"])
    if mode == "yes":
        prompt = f"모든 {tokens[0]}는 {tokens[1]}다. 모든 {tokens[1]}는 {tokens[2]}다. 그러면 모든 {tokens[0]}는 {tokens[2]}인가?"
        reason = f"{tokens[0]}가 모두 {tokens[1]}에 포함되고, {tokens[1]}가 모두 {tokens[2]}에 포함되므로 예입니다."
        answer = "예"
    else:
        prompt = f"모든 {tokens[0]}는 {tokens[1]}다. 어떤 {tokens[1]}는 {tokens[2]}다. 그러면 어떤 {tokens[0]}는 {tokens[2]}인가?"
        reason = f"어떤 {tokens[1]}가 {tokens[2]}라는 사실만으로 그 대상이 {tokens[0]}인지 알 수 없으므로 보장할 수 없습니다."
        answer = "보장할 수 없습니다"
    return make_row(prompt, reason, answer, "syllogism", split, "synthetic_syllogism")


def gen_probability(rng: random.Random, split: str) -> Dict[str, str]:
    red = rng.randint(1, 4)
    blue = rng.randint(2, 6)
    total = red + blue
    answer = simplify_fraction(blue, total)
    prompt = f"주머니에 빨간 공 {red}개와 파란 공 {blue}개가 있다. 하나를 무작위로 뽑을 때 파란 공일 확률은?"
    reason = f"전체 {total}개 중 파란 공이 {blue}개이므로 확률은 {blue}/{total}이고 기약분수는 {answer}입니다."
    return make_row(prompt, reason, answer, "probability", split, "synthetic_probability")


def gen_sequence(rng: random.Random, split: str) -> Dict[str, str]:
    start = rng.randint(2, 12)
    mul = rng.choice([2, 3])
    add = rng.randint(1, 5)
    seq = [start]
    for idx in range(5):
        prev = seq[-1]
        seq.append(prev * mul if idx % 2 == 0 else prev + add)
    answer = seq[-1] * mul
    prompt = f"수열 {', '.join(str(x) for x in seq)}, ? 다음 수를 규칙 기반으로 설명하고 답하라."
    reason = f"규칙이 번갈아 ×{mul}, +{add}이므로 다음 수는 {answer}입니다."
    return make_row(prompt, reason, str(answer), "sequence", split, "synthetic_sequence")


GENERATORS = [gen_compare, gen_syllogism, gen_probability, gen_sequence]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synthetic_train", type=int, default=64000)
    ap.add_argument("--synthetic_eval", type=int, default=1536)
    ap.add_argument("--train_out", default=str(TRAIN_DEFAULT))
    ap.add_argument("--eval_out", default=str(EVAL_DEFAULT))
    ap.add_argument("--manifest_out", default=str(MANIFEST_DEFAULT))
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"

    train_rows: List[Dict[str, str]] = []
    eval_rows: List[Dict[str, str]] = []
    public_path = data_dir / "public_reasoning_cached_v2.jsonl"
    if public_path.exists():
        imported_train = import_public_yesno(public_path, "train", 12000)
        imported_eval = import_public_yesno(public_path, "eval", 512)
        train_rows.extend(imported_train)
        eval_rows.extend(imported_eval)

    for _ in range(int(args.synthetic_train)):
        train_rows.append(rng.choice(GENERATORS)(rng, "train"))
    for _ in range(int(args.synthetic_eval)):
        eval_rows.append(rng.choice(GENERATORS)(rng, "eval"))

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    manifest_out = Path(args.manifest_out)
    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)

    manifest = {
        "train_out": str(train_out),
        "eval_out": str(eval_out),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "seed": int(args.seed),
        "sources": {
            "public_reasoning_cached_v2_train": 12000 if public_path.exists() else 0,
            "public_reasoning_cached_v2_eval": 512 if public_path.exists() else 0,
            "synthetic_train": int(args.synthetic_train),
            "synthetic_eval": int(args.synthetic_eval)
        },
        "categories": ["compare_smallest", "syllogism", "probability", "sequence", "sentence_relation_yesno"]
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
