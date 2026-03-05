from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List


TRAIN_DEFAULT = Path("data/mainline_suite_fix_v1_train.jsonl")
EVAL_DEFAULT = Path("data/mainline_suite_fix_v1_eval.jsonl")
MANIFEST_DEFAULT = Path("data/mainline_suite_fix_v1.manifest.json")


GENERAL_CANONICAL: Dict[str, str] = {
    "chat_hi": "안녕하세요. 무엇을 도와드릴까요?",
    "chat_intro": "안녕하세요. 질문을 짧고 명확하게 답하는 한국어 어시스턴트입니다.",
    "chat_help": "질문을 정리해 핵심부터 간단히 설명해 드릴 수 있습니다.",
    "def_kotlin": "코틀린은 JVM과 안드로이드에서 널리 쓰이는 정적 타입 프로그래밍 언어입니다.",
    "def_http": "HTTP는 웹에서 클라이언트와 서버가 요청과 응답을 주고받는 통신 프로토콜입니다.",
    "def_jwt": "JWT는 인증 정보를 안전하게 전달하기 위한 서명 기반 토큰 형식입니다.",
    "def_gc": "가비지 컬렉션은 사용하지 않는 메모리를 자동으로 회수하는 기능입니다.",
    "qa_sla99": "SLA 99는 서비스 가용성을 99% 수준으로 보장한다는 의미입니다.",
    "qa_vienna": "제품 표시사항을 확인하고, 일반적으로는 가열해 먹는 것이 안전합니다.",
}


GENERAL_VARIANTS: Dict[str, List[str]] = {
    "chat_hi": [
        "안녕하세요. 궁금한 점을 말씀해 주세요.",
        "안녕하세요. 필요한 내용을 알려주시면 바로 답하겠습니다.",
    ],
    "chat_intro": [
        "안녕하세요. 핵심을 빠르게 정리해 답하는 한국어 어시스턴트입니다.",
        "안녕하세요. 요청 의도를 파악해 간결하게 설명해 드립니다.",
    ],
    "chat_help": [
        "요약, 개념 설명, 비교 정리를 짧게 도와드릴 수 있습니다.",
        "질문을 주시면 핵심 개념과 예시를 간단히 정리해 드립니다.",
    ],
    "def_http": [
        "HTTP는 웹 통신에서 요청과 응답을 전달하는 애플리케이션 계층 프로토콜입니다.",
        "HTTP는 브라우저와 서버가 데이터를 주고받을 때 사용하는 표준 웹 프로토콜입니다.",
    ],
    "def_jwt": [
        "JWT는 클레임 정보를 담아 서명으로 무결성을 보장하는 인증 토큰입니다.",
        "JWT는 서버와 클라이언트 사이에서 인증 정보를 전달할 때 자주 쓰이는 토큰 형식입니다.",
    ],
    "qa_vienna": [
        "대부분 가열 조리 후 섭취가 권장되며, 제품 안내를 먼저 확인하는 것이 안전합니다.",
        "보관 상태와 제품 표기를 확인하고, 가능하면 익혀 먹는 쪽이 더 안전합니다.",
    ],
}


LOGIC_CANONICAL: Dict[str, str] = {
    "logic_compare": "이유: A > B > C이므로 가장 작은 것은 C입니다.\n답: C",
    "logic_prob": "이유: 전체 5개 중 파란 공이 3개이므로 확률은 3/5입니다.\n답: 3/5",
    "logic_seq": "이유: 규칙이 번갈아 ×3, +2이므로 다음 수는 207입니다.\n답: 207",
    "logic_compare_smallest": "이유: A > B > C이므로 가장 작은 것은 C입니다.\n답: C",
    "logic_prob_blue": "이유: 전체 5개 중 파란 공이 3개이므로 확률은 3/5입니다.\n답: 3/5",
    "logic_syllogism_yes": "이유: A가 모두 B에 포함되고 B가 모두 C에 포함되므로 예입니다.\n답: 예",
    "logic_syllogism_unknown": "이유: 어떤 B가 C라는 정보만으로 그 대상이 A인지 알 수 없어 보장할 수 없습니다.\n답: 보장할 수 없습니다",
    "logic_sequence": "이유: 규칙이 번갈아 ×3, +2이므로 다음 수는 207입니다.\n답: 207",
}


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


def row(input_text: str, output_text: str, segment: str, source: str) -> Dict[str, str]:
    return {
        "input": str(input_text).strip(),
        "output": str(output_text).strip(),
        "task_type": "korean",
        "segment_tag": segment,
        "language": "ko",
        "source": source,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suite", default="data/mainline_prompt_suite_v1.json")
    ap.add_argument("--readout", default="data/mainline_logic_reasoning_readout_v1.json")
    ap.add_argument("--general_replay", default="data/mainline_general_v1_train.jsonl")
    ap.add_argument("--logic_replay", default="data/mainline_logic_verified_v2_train.jsonl")
    ap.add_argument("--general_replay_rows", type=int, default=24000)
    ap.add_argument("--logic_replay_rows", type=int, default=12000)
    ap.add_argument("--general_suite_repeat", type=int, default=2200)
    ap.add_argument("--logic_suite_repeat", type=int, default=1200)
    ap.add_argument("--eval_rows", type=int, default=1024)
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

    # General suite anchoring
    general_suite_count = 0
    for item in suite_rows:
        pid = item["id"]
        prompt = item["prompt"]
        if not pid.startswith(("chat_", "def_", "qa_")):
            continue
        base = GENERAL_CANONICAL.get(pid)
        if not base:
            continue
        variants = [base] + GENERAL_VARIANTS.get(pid, [])
        for _ in range(int(args.general_suite_repeat)):
            out = rng.choice(variants)
            train_rows.append(row(prompt, out, "ko_mainline_general_v1", f"suite_general_{pid}"))
            general_suite_count += 1
        eval_rows.append(row(prompt, base, "ko_mainline_general_v1", f"suite_general_eval_{pid}"))

    # Logic suite anchoring
    logic_suite_count = 0
    for item in suite_rows + readout_rows:
        pid = item["id"]
        prompt = item["prompt"]
        if not pid.startswith("logic_"):
            continue
        out = LOGIC_CANONICAL.get(pid)
        if not out:
            continue
        for _ in range(int(args.logic_suite_repeat)):
            train_rows.append(row(prompt, out, "ko_mainline_logic_v2", f"suite_logic_{pid}"))
            logic_suite_count += 1
        eval_rows.append(row(prompt, out, "ko_mainline_logic_v2", f"suite_logic_eval_{pid}"))

    # General replay
    general_pool = list(iter_jsonl(root / args.general_replay))
    rng.shuffle(general_pool)
    general_take = min(len(general_pool), int(args.general_replay_rows))
    for r in general_pool[:general_take]:
        train_rows.append(row(r.get("input", ""), r.get("output", ""), "ko_mainline_general_v1", "general_replay"))

    # Logic replay
    logic_pool = list(iter_jsonl(root / args.logic_replay))
    rng.shuffle(logic_pool)
    logic_take = min(len(logic_pool), int(args.logic_replay_rows))
    for r in logic_pool[:logic_take]:
        train_rows.append(row(r.get("input", ""), r.get("output", ""), "ko_mainline_logic_v2", "logic_replay"))

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
            "general_replay_rows": general_take,
            "logic_replay_rows": logic_take,
        },
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

