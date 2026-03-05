from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a SentencePiece tokenizer from JSONL datasets.")
    p.add_argument(
        "--data_paths",
        nargs="+",
        required=True,
        help="One or more JSONL dataset files (expects input/output fields or similar).",
    )
    p.add_argument("--out_dir", default="artifacts_tokenizer_spm_ko_v1")
    p.add_argument("--model_prefix", default="ko_spm_v1")
    p.add_argument("--vocab_size", type=int, default=16000)
    p.add_argument("--character_coverage", type=float, default=0.9995)
    p.add_argument("--model_type", default="unigram", choices=["unigram", "bpe"])
    p.add_argument("--normalization", default="nmt_nfkc", help="sentencepiece normalization_rule_name")
    p.add_argument("--max_rows", type=int, default=0, help="0 = all")
    p.add_argument("--max_line_chars", type=int, default=1600)
    return p.parse_args()


def pick_text(row: Dict) -> List[str]:
    inp = str(
        row.get("input")
        or row.get("instruction")
        or row.get("prompt")
        or row.get("question")
        or row.get("context")
        or ""
    ).strip()
    out = str(
        row.get("output")
        or row.get("response")
        or row.get("answer")
        or row.get("completion")
        or row.get("target")
        or ""
    ).strip()

    lines: List[str] = []
    if inp:
        lines.append(inp)
    if out:
        lines.append(out)
    if inp and out:
        lines.append(f"### Instruction {inp} ### Response {out}")
    return lines


def iter_rows(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / f"{args.model_prefix}.train.txt"
    rows = 0
    lines_written = 0
    source_stats: Dict[str, int] = {}

    with txt_path.open("w", encoding="utf-8", newline="\n") as w:
        for p in args.data_paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"data path not found: {path}")
            count = 0
            for row in iter_rows(path):
                rows += 1
                count += 1
                for line in pick_text(row):
                    s = " ".join(str(line).split()).strip()
                    if len(s) < 2:
                        continue
                    if len(s) > int(args.max_line_chars):
                        s = s[: int(args.max_line_chars)].rstrip()
                    w.write(s + "\n")
                    lines_written += 1
                if int(args.max_rows) > 0 and rows >= int(args.max_rows):
                    break
            source_stats[str(path).replace("\\", "/")] = int(count)
            if int(args.max_rows) > 0 and rows >= int(args.max_rows):
                break

    model_prefix = str((out_dir / args.model_prefix).resolve())
    spm.SentencePieceTrainer.train(
        input=str(txt_path),
        model_prefix=model_prefix,
        vocab_size=int(args.vocab_size),
        model_type=str(args.model_type),
        character_coverage=float(args.character_coverage),
        normalization_rule_name=str(args.normalization),
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        hard_vocab_limit=False,
        max_sentence_length=max(256, int(args.max_line_chars)),
    )

    manifest = {
        "name": "spm_tokenizer_training",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir).replace("\\", "/"),
        "model_prefix": str(args.model_prefix),
        "model_file": str((out_dir / f"{args.model_prefix}.model")).replace("\\", "/"),
        "vocab_file": str((out_dir / f"{args.model_prefix}.vocab")).replace("\\", "/"),
        "rows_read": int(rows),
        "lines_written": int(lines_written),
        "source_rows": source_stats,
        "config": {
            "vocab_size": int(args.vocab_size),
            "character_coverage": float(args.character_coverage),
            "model_type": str(args.model_type),
            "normalization": str(args.normalization),
            "max_rows": int(args.max_rows),
            "max_line_chars": int(args.max_line_chars),
        },
    }
    manifest_path = out_dir / f"{args.model_prefix}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "manifest": str(manifest_path).replace("\\", "/")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
