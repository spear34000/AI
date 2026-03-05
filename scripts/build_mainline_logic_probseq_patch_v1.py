from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List


TRAIN_DEFAULT = Path("data/mainline_logic_probseq_patch_v1_train.jsonl")
EVAL_DEFAULT = Path("data/mainline_logic_probseq_patch_v1_eval.jsonl")
MANIFEST_DEFAULT = Path("data/mainline_logic_probseq_patch_v1.manifest.json")


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def mk(inp: str, out: str, source: str, segment: str) -> Dict[str, str]:
    return {
        "input": str(inp).strip(),
        "output": str(out).strip(),
        "task_type": "korean",
        "segment_tag": segment,
        "language": "ko",
        "source": source,
    }


def simplify_fraction(n: int, d: int) -> str:
    g = math.gcd(n, d)
    n //= g
    d //= g
    return str(n) if d == 1 else f"{n}/{d}"


def gen_prob(rng: random.Random) -> Dict[str, str]:
    red = rng.randint(1, 6)
    blue = rng.randint(1, 7)
    total = red + blue
    ans = simplify_fraction(blue, total)
    inp = f"주머니에 빨간 공 {red}개와 파란 공 {blue}개가 있다. 하나를 무작위로 뽑을 때 파란 공일 확률은? 이유를 짧게 설명하라."
    out = f"이유: 전체 공 수는 {red}+{blue}={total}개이고 파란 공은 {blue}개이므로 확률은 {blue}/{total}입니다. 기약분수는 {ans}입니다.\n답: {ans}"
    return mk(inp, out, "logic_patch_prob_v1", "ko_mainline_logic_patch_v1")


def gen_seq(rng: random.Random) -> Dict[str, str]:
    start = rng.randint(2, 12)
    mul = rng.choice([2, 3])
    add = rng.randint(1, 5)
    seq = [start]
    for i in range(5):
        prev = seq[-1]
        if i % 2 == 0:
            seq.append(prev * mul)
        else:
            seq.append(prev + add)
    ans = seq[-1] * mul
    inp = f"수열 {', '.join(str(x) for x in seq)}, ? 다음 수를 규칙 기반으로 설명하고 답하라."
    out = f"이유: 규칙이 번갈아 곱하기 {mul}, 더하기 {add}이므로 다음 수는 {ans}입니다.\n답: {ans}"
    return mk(inp, out, "logic_patch_sequence_v1", "ko_mainline_logic_patch_v1")


def gen_compare(rng: random.Random) -> Dict[str, str]:
    names = rng.sample(["A", "B", "C", "D"], 3)
    a, b, c = names
    inp = f"{a}는 {b}보다 크고, {b}는 {c}보다 크다. 가장 작은 것은 누구인가? 이유를 짧게 설명하라."
    out = f"이유: {a}가 {b}보다 크고 {b}가 {c}보다 크므로 가장 작은 것은 {c}입니다.\n답: {c}"
    return mk(inp, out, "logic_patch_compare_v1", "ko_mainline_logic_patch_v1")


def gen_syllogism(rng: random.Random) -> Dict[str, str]:
    t = rng.sample(["A", "B", "C", "D", "E"], 3)
    if rng.random() < 0.5:
        inp = f"모든 {t[0]}는 {t[1]}다. 모든 {t[1]}는 {t[2]}다. 그러면 모든 {t[0]}는 {t[2]}인가?"
        out = f"이유: {t[0]}가 모두 {t[1]}에 포함되고 {t[1]}가 모두 {t[2]}에 포함되므로 예입니다.\n답: 예"
    else:
        inp = f"모든 {t[0]}는 {t[1]}다. 어떤 {t[1]}는 {t[2]}다. 그러면 어떤 {t[0]}는 {t[2]}인가?"
        out = f"이유: 어떤 {t[1]}가 {t[2]}라는 사실만으로 그 대상이 {t[0]}인지 알 수 없으므로 보장할 수 없습니다.\n답: 보장할 수 없습니다"
    return mk(inp, out, "logic_patch_syllogism_v1", "ko_mainline_logic_patch_v1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--general_replay", default="data/mainline_suite_fix_v1_train.jsonl")
    ap.add_argument("--general_rows", type=int, default=62000)
    ap.add_argument("--prob_rows", type=int, default=12000)
    ap.add_argument("--seq_rows", type=int, default=12000)
    ap.add_argument("--compare_rows", type=int, default=4000)
    ap.add_argument("--syllogism_rows", type=int, default=4000)
    ap.add_argument("--eval_rows", type=int, default=2048)
    ap.add_argument("--train_out", default=str(TRAIN_DEFAULT))
    ap.add_argument("--eval_out", default=str(EVAL_DEFAULT))
    ap.add_argument("--manifest_out", default=str(MANIFEST_DEFAULT))
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    root = Path(__file__).resolve().parent.parent

    rows: List[Dict[str, str]] = []
    general_pool = list(iter_jsonl(root / args.general_replay))
    rng.shuffle(general_pool)
    general_take = min(len(general_pool), int(args.general_rows))
    for row in general_pool[:general_take]:
        rows.append(
            mk(
                str(row.get("input", "")),
                str(row.get("output", "")),
                "general_replay_suite_fix_v1",
                "ko_mainline_general_v1",
            )
        )

    for _ in range(int(args.prob_rows)):
        rows.append(gen_prob(rng))
    for _ in range(int(args.seq_rows)):
        rows.append(gen_seq(rng))
    for _ in range(int(args.compare_rows)):
        rows.append(gen_compare(rng))
    for _ in range(int(args.syllogism_rows)):
        rows.append(gen_syllogism(rng))

    rng.shuffle(rows)
    target_eval = int(args.eval_rows)
    eval_rows = rows[:target_eval]
    train_rows = rows[target_eval:]

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
        "components": {
            "general_replay_rows": general_take,
            "prob_rows": int(args.prob_rows),
            "seq_rows": int(args.seq_rows),
            "compare_rows": int(args.compare_rows),
            "syllogism_rows": int(args.syllogism_rows),
        },
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
