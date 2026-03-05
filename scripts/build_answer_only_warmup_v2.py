from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_TRAIN_IN = Path("data/logic_verified_v1_train.jsonl")
DEFAULT_EVAL_IN = Path("data/logic_verified_v1_eval.jsonl")
DEFAULT_TRAIN_OUT = Path("data/answer_only_warmup_v2_train.jsonl")
DEFAULT_EVAL_OUT = Path("data/answer_only_warmup_v2_eval.jsonl")
DEFAULT_MANIFEST = Path("data/answer_only_warmup_v2.manifest.json")

KEEP_CATEGORIES = {
    "arithmetic_addsub",
    "arithmetic_muldiv",
    "ratio",
    "compare_smallest",
}

PROMPT_SUFFIX = (
    "\n"
    + "\ucd94\uac00 \uaddc\uce59:\n"
    + "- \uc124\uba85 \uae08\uc9c0\n"
    + "- \ub2e4\ub978 \ubb38\uc7a5 \uae08\uc9c0\n"
    + "- \ud55c \uc904\ub9cc \ucd9c\ub825\n"
    + "- \ud615\uc2dd\uc740 \uc815\ud655\ud788 `\uc815\ub2f5: \uac12`\n"
    + "- \uc608\uc2dc: \uc815\ub2f5: 42"
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


def simplify_prompt(text: str) -> str:
    text = str(text).strip().replace("\r", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    first = lines[0]
    if "\ub2f5\ubcc0 \ud615\uc2dd" in first:
        first = first.split("\ub2f5\ubcc0 \ud615\uc2dd", 1)[0].strip()
    return first


def transform_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    category = str(row.get("category", "")).strip()
    if category not in KEEP_CATEGORIES:
        return None
    final_answer = str(row.get("final_answer", "")).strip()
    prompt = simplify_prompt(str(row.get("input", "")))
    if not final_answer or not prompt:
        return None
    out: Dict[str, Any] = dict(row)
    out["input"] = prompt + PROMPT_SUFFIX
    out["output"] = "\uc815\ub2f5: " + final_answer
    out["answer_format"] = "\uc815\ub2f5: ..."
    out["_meta_source_file"] = "answer_only_warmup_v2"
    out["warmup_mode"] = "answer_only_easy"
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
        "categories": sorted(KEEP_CATEGORIES),
        "prompt_suffix": PROMPT_SUFFIX,
        "output_format": "\uc815\ub2f5: ...",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
