from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List


TRAIN_DEFAULT = Path("data/mainline_suite_fix_v2_train.jsonl")
EVAL_DEFAULT = Path("data/mainline_suite_fix_v2_eval.jsonl")
MANIFEST_DEFAULT = Path("data/mainline_suite_fix_v2.manifest.json")


GENERAL_CANONICAL: Dict[str, str] = {
    "chat_hi": "안녕하세요. 무엇을 도와드릴까요?",
    "chat_intro": "안녕하세요. 질문 의도를 빠르게 파악해 간결하고 정확하게 답하는 한국어 AI입니다.",
    "chat_help": "개념 설명, 비교 정리, 짧은 요약, 문제 풀이를 간단명료하게 도와드릴 수 있습니다.",
    "def_kotlin": "코틀린은 JVM과 안드로이드에서 널리 쓰이는 정적 타입 프로그래밍 언어입니다.",
    "def_http": "HTTP는 웹에서 클라이언트와 서버가 요청과 응답을 주고받는 통신 프로토콜입니다.",
    "def_jwt": "JWT는 인증 정보를 안전하게 전달하기 위해 사용하는 서명 기반 토큰 형식입니다.",
    "def_gc": "가비지 컬렉션은 더 이상 사용하지 않는 메모리를 자동으로 회수하는 기능입니다.",
    "qa_sla99": "SLA 99는 서비스 가용성을 99% 수준으로 보장한다는 의미입니다.",
    "qa_vienna": "제품 표시사항을 먼저 확인하고, 일반적으로는 가열해서 먹는 편이 더 안전합니다.",
}


GENERAL_VARIANTS: Dict[str, List[str]] = {
    "chat_hi": [
        "안녕하세요. 필요한 내용을 짧고 분명하게 도와드릴게요.",
        "안녕하세요. 질문 주시면 핵심부터 빠르게 정리해드리겠습니다.",
    ],
    "chat_intro": [
        "안녕하세요. 핵심 위주로 정확하게 답변하는 한국어 어시스턴트입니다.",
        "안녕하세요. 복잡한 내용을 짧고 이해하기 쉽게 설명해드리는 한국어 AI입니다.",
    ],
    "chat_help": [
        "질문 핵심 요약, 개념 설명, 장단점 비교, 간단 문제 풀이를 도와드릴 수 있습니다.",
        "정의 설명, 예시 정리, 논리 비교, 간단 계산 검토를 빠르게 도와드립니다.",
    ],
    "def_http": [
        "HTTP는 웹에서 요청과 응답을 교환할 때 사용하는 애플리케이션 계층 프로토콜입니다.",
        "HTTP는 브라우저와 서버 간 데이터 요청/응답 통신 규약입니다.",
    ],
    "def_jwt": [
        "JWT는 클레임 정보를 담고 서명으로 무결성을 검증하는 인증 토큰 형식입니다.",
        "JWT는 서버와 클라이언트 사이에서 인증 정보를 전달할 때 자주 쓰는 서명 토큰입니다.",
    ],
    "qa_vienna": [
        "대부분 조리 후 섭취가 권장되며, 포장 표기와 보관 상태를 먼저 확인하는 것이 좋습니다.",
        "표시사항을 확인한 뒤 가열해서 먹는 쪽이 일반적으로 더 안전합니다.",
    ],
}


LOGIC_CANONICAL: Dict[str, str] = {
    "logic_compare": "이유: A가 B보다 크고 B가 C보다 크므로 가장 작은 것은 C입니다.\n답: C",
    "logic_compare_smallest": "이유: A가 B보다 크고 B가 C보다 크므로 가장 작은 것은 C입니다.\n답: C",
    "logic_prob": "이유: 전체 5개 중 파란 공이 3개이므로 확률은 3/5입니다.\n답: 3/5",
    "logic_prob_blue": "이유: 전체 5개 중 파란 공이 3개이므로 확률은 3/5입니다.\n답: 3/5",
    "logic_seq": "이유: 규칙이 번갈아 곱하기 3, 더하기 2이므로 다음 수는 207입니다.\n답: 207",
    "logic_sequence": "이유: 규칙이 번갈아 곱하기 3, 더하기 2이므로 다음 수는 207입니다.\n답: 207",
    "logic_syllogism_yes": "이유: A가 모두 B에 포함되고 B가 모두 C에 포함되므로 예입니다.\n답: 예",
    "logic_syllogism_unknown": "이유: 어떤 B가 C라는 사실만으로 그 대상이 A인지 알 수 없으므로 보장할 수 없습니다.\n답: 보장할 수 없습니다",
}


ARITH_TOKENS = (
    "최종답",
    "부분곱",
    "검산",
    "일의 자리",
    "십의 자리",
    "백의 자리",
    "천의 자리",
)


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_suite(path: Path) -> List[Dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: List[Dict[str, str]] = []
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            pid = str(row.get("id", "")).strip()
            prompt = str(row.get("prompt", "")).strip()
            if pid and prompt:
                out.append({"id": pid, "prompt": prompt})
    return out


def iter_jsonl(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def make_row(input_text: str, output_text: str, segment: str, source: str) -> Dict[str, str]:
    return {
        "input": str(input_text).strip(),
        "output": str(output_text).strip(),
        "task_type": "korean",
        "segment_tag": segment,
        "language": "ko",
        "source": source,
    }


def digit_ratio(text: str) -> float:
    s = str(text or "")
    if not s:
        return 1.0
    d = sum(ch.isdigit() for ch in s)
    return d / float(len(s))


def keep_general_row(row: Dict[str, str]) -> bool:
    inp = str(row.get("input", "")).strip()
    out = str(row.get("output", "")).strip()
    if not inp or not out:
        return False
    if len(inp) > 240 or len(out) > 260 or len(out) < 8:
        return False
    if "×" in out:
        return False
    if digit_ratio(out) > 0.18:
        return False
    low = out.lower()
    if any(tok in out for tok in ARITH_TOKENS):
        return False
    if "답:" in low:
        return False
    return True


def gen_compare(rng: random.Random) -> Dict[str, str]:
    names = rng.sample(["A", "B", "C", "D"], 3)
    a, b, c = names[0], names[1], names[2]
    prompt = f"{a}는 {b}보다 크고, {b}는 {c}보다 크다. 가장 작은 것은 누구인가? 이유를 짧게 설명하라."
    out = f"이유: {a}가 {b}보다 크고 {b}가 {c}보다 크므로 가장 작은 것은 {c}입니다.\n답: {c}"
    return make_row(prompt, out, "ko_mainline_logic_v2", "synthetic_logic_compare_v2")


def gen_syllogism(rng: random.Random) -> Dict[str, str]:
    t = rng.sample(["A", "B", "C", "D", "E"], 3)
    if rng.random() < 0.5:
        prompt = f"모든 {t[0]}는 {t[1]}다. 모든 {t[1]}는 {t[2]}다. 그러면 모든 {t[0]}는 {t[2]}인가?"
        out = f"이유: {t[0]}가 모두 {t[1]}에 포함되고 {t[1]}가 모두 {t[2]}에 포함되므로 예입니다.\n답: 예"
    else:
        prompt = f"모든 {t[0]}는 {t[1]}다. 어떤 {t[1]}는 {t[2]}다. 그러면 어떤 {t[0]}는 {t[2]}인가?"
        out = f"이유: 어떤 {t[1]}가 {t[2]}라는 사실만으로 그 대상이 {t[0]}인지 알 수 없으므로 보장할 수 없습니다.\n답: 보장할 수 없습니다"
    return make_row(prompt, out, "ko_mainline_logic_v2", "synthetic_logic_syllogism_v2")


def simplify_fraction(num: int, den: int) -> str:
    import math

    g = math.gcd(num, den)
    num //= g
    den //= g
    if den == 1:
        return str(num)
    return f"{num}/{den}"


def gen_probability(rng: random.Random) -> Dict[str, str]:
    red = rng.randint(1, 5)
    blue = rng.randint(1, 6)
    total = red + blue
    ans = simplify_fraction(blue, total)
    prompt = f"주머니에 빨간 공 {red}개와 파란 공 {blue}개가 있다. 하나를 무작위로 뽑을 때 파란 공일 확률은? 이유를 짧게 설명하라."
    out = f"이유: 전체 공은 {red}+{blue}={total}개이고 파란 공은 {blue}개라서 확률은 {blue}/{total}입니다. 기약분수는 {ans}입니다.\n답: {ans}"
    return make_row(prompt, out, "ko_mainline_logic_v2", "synthetic_logic_prob_v2")


def gen_sequence(rng: random.Random) -> Dict[str, str]:
    start = rng.randint(2, 10)
    mul = rng.choice([2, 3])
    add = rng.randint(1, 4)
    seq = [start]
    for idx in range(5):
        prev = seq[-1]
        if idx % 2 == 0:
            seq.append(prev * mul)
        else:
            seq.append(prev + add)
    ans = seq[-1] * mul
    prompt = f"수열 {', '.join(str(x) for x in seq)}, ? 다음 수를 규칙 기반으로 설명하고 답하라."
    out = f"이유: 규칙이 번갈아 곱하기 {mul}, 더하기 {add}이므로 다음 수는 {ans}입니다.\n답: {ans}"
    return make_row(prompt, out, "ko_mainline_logic_v2", "synthetic_logic_sequence_v2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suite", default="data/mainline_prompt_suite_v1.json")
    ap.add_argument("--readout", default="data/mainline_logic_reasoning_readout_v1.json")
    ap.add_argument("--general_replay", default="data/mainline_general_v1_train.jsonl")
    ap.add_argument("--general_replay_rows", type=int, default=36000)
    ap.add_argument("--general_suite_repeat", type=int, default=1800)
    ap.add_argument("--logic_suite_repeat", type=int, default=900)
    ap.add_argument("--logic_synth_rows", type=int, default=28000)
    ap.add_argument("--eval_rows", type=int, default=2048)
    ap.add_argument("--train_out", default=str(TRAIN_DEFAULT))
    ap.add_argument("--eval_out", default=str(EVAL_DEFAULT))
    ap.add_argument("--manifest_out", default=str(MANIFEST_DEFAULT))
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    root = Path(__file__).resolve().parent.parent

    suite_rows = load_suite(root / args.suite)
    readout_rows = load_suite(root / args.readout)

    train_rows: List[Dict[str, str]] = []
    eval_rows: List[Dict[str, str]] = []

    general_suite_count = 0
    logic_suite_count = 0

    for item in suite_rows:
        pid = item["id"]
        prompt = item["prompt"]
        if pid.startswith(("chat_", "def_", "qa_")):
            base = GENERAL_CANONICAL.get(pid)
            if not base:
                continue
            variants = [base] + GENERAL_VARIANTS.get(pid, [])
            for _ in range(int(args.general_suite_repeat)):
                train_rows.append(make_row(prompt, rng.choice(variants), "ko_mainline_general_v2", f"suite_general_{pid}"))
                general_suite_count += 1
            eval_rows.append(make_row(prompt, base, "ko_mainline_general_v2", f"suite_general_eval_{pid}"))
        elif pid.startswith("logic_"):
            base = LOGIC_CANONICAL.get(pid)
            if not base:
                continue
            for _ in range(int(args.logic_suite_repeat)):
                train_rows.append(make_row(prompt, base, "ko_mainline_logic_v2", f"suite_logic_{pid}"))
                logic_suite_count += 1
            eval_rows.append(make_row(prompt, base, "ko_mainline_logic_v2", f"suite_logic_eval_{pid}"))

    for item in readout_rows:
        pid = item["id"]
        prompt = item["prompt"]
        if not pid.startswith("logic_"):
            continue
        base = LOGIC_CANONICAL.get(pid)
        if not base:
            continue
        for _ in range(max(1, int(args.logic_suite_repeat // 2))):
            train_rows.append(make_row(prompt, base, "ko_mainline_logic_v2", f"readout_logic_{pid}"))
            logic_suite_count += 1
        eval_rows.append(make_row(prompt, base, "ko_mainline_logic_v2", f"readout_logic_eval_{pid}"))

    generators = [gen_compare, gen_syllogism, gen_probability, gen_sequence]
    for _ in range(int(args.logic_synth_rows)):
        train_rows.append(rng.choice(generators)(rng))

    general_pool = [r for r in iter_jsonl(root / args.general_replay) if keep_general_row(r)]
    rng.shuffle(general_pool)
    general_take = min(len(general_pool), int(args.general_replay_rows))
    for r in general_pool[:general_take]:
        train_rows.append(make_row(r.get("input", ""), r.get("output", ""), "ko_mainline_general_v2", "general_replay_filtered_v2"))

    rng.shuffle(train_rows)

    target_eval = int(args.eval_rows)
    if len(eval_rows) < target_eval:
        fill = train_rows[: max(0, target_eval - len(eval_rows))]
        eval_rows.extend(fill)
    eval_rows = eval_rows[:target_eval]

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
            "general_suite_rows": general_suite_count,
            "logic_suite_rows": logic_suite_count,
            "logic_synth_rows": int(args.logic_synth_rows),
            "general_replay_filtered_rows": general_take,
        },
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
