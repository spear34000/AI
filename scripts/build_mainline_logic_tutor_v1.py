from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


TRAIN_DEFAULT = Path("data/mainline_logic_tutor_v1_train.jsonl")
EVAL_DEFAULT = Path("data/mainline_logic_tutor_v1_eval.jsonl")
MANIFEST_DEFAULT = Path("data/mainline_logic_tutor_v1.manifest.json")


def iter_jsonl(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tutorize(row: Dict[str, str]) -> Dict[str, str]:
    prompt = str(row.get("input", "")).strip()
    answer = str(row.get("final_answer", "")).strip()
    reason_text = str(row.get("output", "")).strip()
    reason_line = ""
    for line in reason_text.splitlines():
        line = line.strip()
        if line.startswith("이유:"):
            reason_line = line.replace("이유:", "", 1).strip()
            break
    if not reason_line:
        reason_line = "조건을 차례대로 비교하면 답을 정할 수 있습니다."
    tutor_output = (
        f"힌트: 핵심 조건을 한 줄씩 정리하라.\n"
        f"재시도: {reason_line}\n"
        f"정답: {answer}\n"
        f"요약: 핵심 조건을 따라가면 답은 {answer}입니다."
    )
    return {
        "input": prompt,
        "output": tutor_output,
        "task_type": "korean",
        "segment_tag": "ko_mainline_logic_tutor_v1",
        "language": "ko",
        "category": str(row.get("category", "logic_tutor")),
        "split": str(row.get("split", "train")),
        "final_answer": answer,
        "answer_format": "정답: ...",
        "_meta_source_file": str(row.get("_meta_source_file", "mainline_logic_verified_v2")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_train", default="data/mainline_logic_verified_v2_train.jsonl")
    ap.add_argument("--source_eval", default="data/mainline_logic_verified_v2_eval.jsonl")
    ap.add_argument("--train_out", default=str(TRAIN_DEFAULT))
    ap.add_argument("--eval_out", default=str(EVAL_DEFAULT))
    ap.add_argument("--manifest_out", default=str(MANIFEST_DEFAULT))
    args = ap.parse_args()

    source_train = Path(args.source_train)
    source_eval = Path(args.source_eval)
    train_rows = [tutorize(row) for row in iter_jsonl(source_train)]
    eval_rows = [tutorize(row) for row in iter_jsonl(source_eval)]

    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    manifest_out = Path(args.manifest_out)
    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)

    manifest = {
        "source_train": str(source_train),
        "source_eval": str(source_eval),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "answer_format": "정답: ...",
        "style": "hint_retry_answer_summary",
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
