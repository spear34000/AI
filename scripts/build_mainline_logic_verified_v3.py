from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List


TRAIN_DEFAULT = Path("data/mainline_logic_verified_v3_train.jsonl")
EVAL_DEFAULT = Path("data/mainline_logic_verified_v3_eval.jsonl")
MANIFEST_DEFAULT = Path("data/mainline_logic_verified_v3.manifest.json")


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_row(prompt: str, reason: str, answer: str, category: str, split: str, source: str) -> Dict[str, str]:
    return {
        "input": str(prompt).strip(),
        "output": f"이유: {str(reason).strip()}\n답: {str(answer).strip()}",
        "task_type": "korean",
        "segment_tag": "ko_mainline_logic_v3",
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


def gen_compare_smallest(rng: random.Random, split: str) -> Dict[str, str]:
    names = rng.sample(["A", "B", "C", "D"], 3)
    prompt = f"{names[0]}는 {names[1]}보다 크고, {names[1]}는 {names[2]}보다 크다. 가장 작은 것은 누구인가?"
    answer = names[2]
    reason = f"{names[0]} > {names[1]} > {names[2]} 이므로 가장 작은 것은 {answer}입니다."
    return make_row(prompt, reason, answer, "compare_smallest", split, "synthetic_compare")


def gen_probability(rng: random.Random, split: str) -> Dict[str, str]:
    red = rng.randint(1, 6)
    blue = rng.randint(1, 6)
    total = red + blue
    answer = simplify_fraction(blue, total)
    prompt = f"주머니에 빨간 공 {red}개와 파란 공 {blue}개가 있다. 하나를 무작위로 뽑을 때 파란 공일 확률은?"
    reason = f"전체 {total}개 중 파란 공이 {blue}개이므로 확률은 {blue}/{total}이고 기약분수는 {answer}입니다."
    return make_row(prompt, reason, answer, "probability", split, "synthetic_probability")


def gen_sequence_deterministic(rng: random.Random, split: str) -> Dict[str, str]:
    start = rng.randint(2, 12)
    mul = rng.choice([2, 3])
    add = rng.randint(1, 5)

    # Pattern: x*mul, +add, x*mul, +add, x*mul, then predict next (+add).
    seq = [start]
    for idx in range(5):
        prev = seq[-1]
        if idx % 2 == 0:
            seq.append(prev * mul)
        else:
            seq.append(prev + add)
    answer = seq[-1] + add

    prompt = f"수열 {', '.join(str(x) for x in seq)}, ? 다음 수를 규칙 기반으로 설명하고 답하라."
    reason = f"규칙이 번갈아 ×{mul}, +{add} 이므로 다음 수는 {answer}입니다."
    return make_row(prompt, reason, str(answer), "sequence", split, "synthetic_sequence")


def gen_sequence_ambiguous_case(split: str) -> Dict[str, str]:
    prompt = "2, 6, 7, 21, 23, 69, ? 다음 수는?"
    reason = "가능한 규칙이 여러 개라 다음 수를 하나로 단정할 수 없습니다."
    return make_row(prompt, reason, "알 수 없음", "sequence", split, "handcrafted_sequence_ambiguous")


def gen_syllogism(rng: random.Random, split: str) -> Dict[str, str]:
    toks = rng.sample(["A", "B", "C", "D", "E"], 3)
    mode = rng.choice(["yes", "unknown"])
    if mode == "yes":
        prompt = f"모든 {toks[0]}는 {toks[1]}다. 모든 {toks[1]}는 {toks[2]}다. 그러면 모든 {toks[0]}는 {toks[2]}인가?"
        reason = f"{toks[0]}가 모두 {toks[1]}에 포함되고 {toks[1]}가 모두 {toks[2]}에 포함되므로 예입니다."
        answer = "예"
    else:
        prompt = f"모든 {toks[0]}는 {toks[1]}다. 어떤 {toks[1]}는 {toks[2]}다. 그러면 어떤 {toks[0]}는 {toks[2]}인가?"
        reason = f"{toks[1]} 일부가 {toks[2]}라는 정보만으로 그 대상이 {toks[0]}인지 알 수 없어 보장할 수 없습니다."
        answer = "알 수 없음"
    return make_row(prompt, reason, answer, "syllogism", split, "synthetic_syllogism")


def gen_sentence_relation_yesno(rng: random.Random, split: str) -> Dict[str, str]:
    item = rng.choice(
        [
            {
                "p1": "모든 고양이는 동물이다.",
                "p2": "나비는 고양이다.",
                "q": "위 두 문장을 함께 보면 '나비는 동물이다'가 참인가? 예/아니오로 답하라.",
                "a": "예",
                "r": "나비가 고양이고 모든 고양이가 동물이므로 나비는 동물입니다.",
            },
            {
                "p1": "모든 포유류는 동물이다.",
                "p2": "고래는 포유류다.",
                "q": "위 두 문장을 함께 보면 '고래는 동물이다'가 참인가? 예/아니오로 답하라.",
                "a": "예",
                "r": "고래가 포유류이고 모든 포유류가 동물이므로 고래는 동물입니다.",
            },
            {
                "p1": "모든 새는 동물이다.",
                "p2": "펭귄은 새다.",
                "q": "위 두 문장을 함께 보면 '펭귄은 동물이다'가 참인가? 예/아니오로 답하라.",
                "a": "예",
                "r": "펭귄이 새이고 모든 새가 동물이므로 펭귄은 동물입니다.",
            },
            {
                "p1": "모든 고양이는 동물이다.",
                "p2": "나비는 동물이다.",
                "q": "위 두 문장을 함께 보면 '나비는 고양이다'가 참인가? 예/아니오로 답하라.",
                "a": "아니오",
                "r": "동물이라는 사실만으로 고양이라고 결론낼 수 없으므로 참이라고 볼 수 없습니다.",
            },
            {
                "p1": "모든 포유류는 동물이다.",
                "p2": "참새는 동물이다.",
                "q": "위 두 문장을 함께 보면 '참새는 포유류다'가 참인가? 예/아니오로 답하라.",
                "a": "아니오",
                "r": "동물이라는 사실만으로 포유류라고 단정할 수 없으므로 참이라고 볼 수 없습니다.",
            },
        ]
    )
    prompt = f"{item['p1']} {item['p2']} {item['q']}"
    return make_row(prompt, item["r"], item["a"], "sentence_relation_yesno", split, "synthetic_sentence_relation")


GENERATORS = (
    gen_compare_smallest,
    gen_probability,
    gen_sequence_deterministic,
    gen_syllogism,
    gen_sentence_relation_yesno,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_rows", type=int, default=72000)
    ap.add_argument("--eval_rows", type=int, default=2048)
    ap.add_argument("--ambiguous_sequence_train_rows", type=int, default=2400)
    ap.add_argument("--ambiguous_sequence_eval_rows", type=int, default=96)
    ap.add_argument("--train_out", default=str(TRAIN_DEFAULT))
    ap.add_argument("--eval_out", default=str(EVAL_DEFAULT))
    ap.add_argument("--manifest_out", default=str(MANIFEST_DEFAULT))
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    train_rows: List[Dict[str, str]] = []
    eval_rows: List[Dict[str, str]] = []

    for _ in range(int(args.train_rows)):
        gen = rng.choice(GENERATORS)
        train_rows.append(gen(rng, "train"))
    for _ in range(int(args.eval_rows)):
        gen = rng.choice(GENERATORS)
        eval_rows.append(gen(rng, "eval"))

    for _ in range(int(args.ambiguous_sequence_train_rows)):
        train_rows.append(gen_sequence_ambiguous_case("train"))
    for _ in range(int(args.ambiguous_sequence_eval_rows)):
        eval_rows.append(gen_sequence_ambiguous_case("eval"))

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
        "categories": [
            "compare_smallest",
            "probability",
            "sequence",
            "syllogism",
            "sentence_relation_yesno",
        ],
        "notes": [
            "sequence labels use a deterministic alternating rule",
            "ambiguous canonical sequence (2,6,7,21,23,69) is labeled as 알 수 없음",
        ],
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
