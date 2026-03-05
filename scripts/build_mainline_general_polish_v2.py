from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List


TRAIN_DEFAULT = Path("data/mainline_general_polish_v2_train.jsonl")
EVAL_DEFAULT = Path("data/mainline_general_polish_v2_eval.jsonl")
MANIFEST_DEFAULT = Path("data/mainline_general_polish_v2.manifest.json")


GENERAL_BANK: Dict[str, List[str]] = {
    "chat_hi": [
        "안녕하세요. 무엇을 도와드릴까요?",
        "안녕하세요. 질문 주시면 핵심부터 간단히 답해드릴게요.",
        "반갑습니다. 필요한 내용을 짧고 정확하게 정리해드리겠습니다.",
    ],
    "chat_intro": [
        "안녕하세요. 질문 의도를 빠르게 파악해 간결하게 답하는 한국어 AI입니다.",
        "안녕하세요. 핵심 설명과 짧은 예시 중심으로 도와드리는 한국어 어시스턴트입니다.",
        "안녕하세요. 복잡한 내용을 이해하기 쉽게 정리해드리는 한국어 AI입니다.",
    ],
    "chat_help": [
        "개념 설명, 핵심 요약, 비교 정리, 간단 문제 풀이를 도와드릴 수 있습니다.",
        "질문 요약, 정의 설명, 선택지 비교, 짧은 계산 점검을 도와드릴 수 있습니다.",
        "핵심 개념 정리와 실무형 예시 설명을 간단명료하게 도와드립니다.",
    ],
    "def_kotlin": [
        "코틀린은 JVM과 안드로이드에서 널리 쓰이는 정적 타입 프로그래밍 언어입니다.",
        "코틀린은 자바와 상호운용이 가능하며 안드로이드 개발에서 많이 사용하는 언어입니다.",
        "코틀린은 간결한 문법과 널 안전성 지원이 강점인 JVM 기반 언어입니다.",
    ],
    "def_http": [
        "HTTP는 웹에서 클라이언트와 서버가 요청과 응답을 주고받는 통신 프로토콜입니다.",
        "HTTP는 브라우저와 서버가 데이터를 교환할 때 쓰는 요청/응답 기반 프로토콜입니다.",
        "HTTP는 웹 애플리케이션 계층에서 동작하는 대표적인 통신 규약입니다.",
    ],
    "def_jwt": [
        "JWT는 인증 정보를 안전하게 전달하기 위해 사용하는 서명 기반 토큰 형식입니다.",
        "JWT는 클레임 정보를 담고 서명으로 무결성을 검증하는 인증 토큰입니다.",
        "JWT는 서버와 클라이언트 사이 인증/인가 정보 전달에 자주 쓰는 토큰입니다.",
    ],
    "def_gc": [
        "가비지 컬렉션은 더 이상 사용하지 않는 메모리를 자동으로 회수하는 기능입니다.",
        "가비지 컬렉션은 불필요한 객체를 정리해 메모리 누수를 줄이는 메모리 관리 기법입니다.",
        "가비지 컬렉션은 런타임이 도달 불가능 객체를 찾아 메모리를 회수하는 과정입니다.",
    ],
    "qa_sla99": [
        "SLA 99는 서비스 가용성을 99% 수준으로 보장한다는 의미입니다.",
        "SLA 99는 약속된 기간 동안 서비스가 99% 시간 이상 정상 동작함을 뜻합니다.",
        "SLA 99는 허용 가능한 다운타임을 1% 이내로 제한하는 운영 목표입니다.",
    ],
    "qa_vienna": [
        "제품 표시사항을 먼저 확인하고, 일반적으로는 가열해서 먹는 편이 더 안전합니다.",
        "비엔나 소시지는 보관 상태와 유통기한을 확인한 뒤 가열 섭취하는 것을 권장합니다.",
        "생식 가능 표기가 없다면 가열 후 섭취하는 쪽이 안전합니다.",
    ],
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
            rid = str(row.get("id", "")).strip()
            prompt = str(row.get("prompt", "")).strip()
            if rid and prompt and rid.startswith(("chat_", "def_", "qa_")):
                out.append({"id": rid, "prompt": prompt})
    return out


def make_row(inp: str, out: str, source: str) -> Dict[str, str]:
    return {
        "input": inp.strip(),
        "output": out.strip(),
        "task_type": "korean",
        "segment_tag": "ko_mainline_general_polish_v2",
        "language": "ko",
        "source": source,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suite", default="data/mainline_prompt_suite_v1.json")
    ap.add_argument("--repeat_per_prompt", type=int, default=2600)
    ap.add_argument("--eval_rows", type=int, default=1024)
    ap.add_argument("--train_out", default=str(TRAIN_DEFAULT))
    ap.add_argument("--eval_out", default=str(EVAL_DEFAULT))
    ap.add_argument("--manifest_out", default=str(MANIFEST_DEFAULT))
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    root = Path(__file__).resolve().parent.parent
    suite_rows = load_suite(root / args.suite)

    train_rows: List[Dict[str, str]] = []
    eval_rows: List[Dict[str, str]] = []

    for item in suite_rows:
        rid = item["id"]
        prompt = item["prompt"]
        bank = GENERAL_BANK.get(rid, [])
        if not bank:
            continue
        canonical = bank[0]
        eval_rows.append(make_row(prompt, canonical, f"general_polish_eval_{rid}"))
        for _ in range(int(args.repeat_per_prompt)):
            train_rows.append(make_row(prompt, rng.choice(bank), f"general_polish_train_{rid}"))

    rng.shuffle(train_rows)
    target_eval = int(args.eval_rows)
    if len(eval_rows) < target_eval:
        eval_rows.extend(train_rows[: max(0, target_eval - len(eval_rows))])
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
        "repeat_per_prompt": int(args.repeat_per_prompt),
        "prompt_count": len(suite_rows),
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
