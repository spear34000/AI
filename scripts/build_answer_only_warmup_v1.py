from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_TRAIN_IN = Path("data/logic_verified_v1_train.jsonl")
DEFAULT_EVAL_IN = Path("data/logic_verified_v1_eval.jsonl")
DEFAULT_TRAIN_OUT = Path("data/answer_only_warmup_v1_train.jsonl")
DEFAULT_EVAL_OUT = Path("data/answer_only_warmup_v1_eval.jsonl")
DEFAULT_MANIFEST = Path("data/answer_only_warmup_v1.manifest.json")


PROMPT_SUFFIX = (
    "\n\n추가 규칙:\n"
    "- 설명은 쓰지 마라.\n"
    "- 다른 문장은 쓰지 마라.\n"
    "- 마지막 한 줄만 출력하라.\n"
    "- 형식은 정확히 `최종답: 정답` 으로만 써라."
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def transform_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    final_answer = str(row.get("final_answer", "")).strip()
    prompt = str(row.get("input", "")).strip()
    if not final_answer or not prompt:
        return None
    out: Dict[str, Any] = dict(row)
    out["input"] = prompt + PROMPT_SUFFIX
    out["output"] = f"최종답: {final_answer}"
    out["answer_format"] = "최종답: ..."
    out["_meta_source_file"] = "answer_only_warmup_v1"
    out["warmup_mode"] = "answer_only"
    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_split(src: Path, dst: Path) -> int:
    rows = load_jsonl(src)
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        tr = transform_row(row)
        if tr is not None:
            out_rows.append(tr)
    return write_jsonl(dst, out_rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_in", default=str(DEFAULT_TRAIN_IN))
    p.add_argument("--eval_in", default=str(DEFAULT_EVAL_IN))
    p.add_argument("--train_out", default=str(DEFAULT_TRAIN_OUT))
    p.add_argument("--eval_out", default=str(DEFAULT_EVAL_OUT))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = p.parse_args()

    train_in = Path(args.train_in)
    eval_in = Path(args.eval_in)
    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    manifest_path = Path(args.manifest)

    train_count = build_split(train_in, train_out)
    eval_count = build_split(eval_in, eval_out)

    manifest = {
        "finished": True,
        "train_in": str(train_in),
        "eval_in": str(eval_in),
        "train_out": str(train_out),
        "eval_out": str(eval_out),
        "train_rows": int(train_count),
        "eval_rows": int(eval_count),
        "prompt_suffix": PROMPT_SUFFIX,
        "output_format": "최종답: ...",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
