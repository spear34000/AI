from __future__ import annotations

import json
import re
from pathlib import Path


SRC_DIRECT = Path("data/repair_ko_direct_v3.jsonl")
SRC_REASON = next(Path("data/final_datasets").glob("*논리추론*.jsonl"))
OUT_PATH = Path("data/logic_only_v1.jsonl")
MANIFEST_PATH = Path("data/logic_only_v1.manifest.json")


LOGIC_HINTS = [
    "수열",
    "반례",
    "증명",
    "모자",
    "거짓말쟁이",
    "항상",
    "정수 n",
    "판단",
    "가장 작은",
    "비율",
    "전략",
    "판별",
]


def is_logic_prompt(text: str) -> bool:
    src = str(text or "")
    if any(tok in src for tok in LOGIC_HINTS):
        return True
    if re.search(r"\d", src) and re.search(r"[+\-*/=^]", src):
        return True
    return False


def clean_reason_output(text: str) -> str:
    out = str(text or "").strip()
    out = re.sub(r"^\s*정답은\s*", "", out)
    out = re.sub(r"\s*입니다\.\s*$", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    if not out:
        return ""
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", out):
        if out.endswith(".0"):
            out = out[:-2]
        return f"{out}입니다."
    return out if out.endswith(".") else f"{out}."


def usable(inp: str, out: str) -> bool:
    text = f"{inp}\n{out}"
    if len(inp) < 3 or len(inp) > 420:
        return False
    if len(out) < 2 or len(out) > 260:
        return False
    if any(tok in text for tok in ["ACTION ", "OBSERVATION ", "FINAL ", "```", "http://", "https://", "\ufffd"]):
        return False
    if any(tok in out for tok in ["안녕하세요", "반갑습니다", "필요하면", "원하면", "AI"]):
        return False
    return True


def main() -> None:
    stats = {
        "direct_rows": 0,
        "direct_kept": 0,
        "reason_rows": 0,
        "reason_kept": 0,
        "duplicates": 0,
        "written": 0,
    }
    seen: set[tuple[str, str]] = set()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as g:
        with SRC_DIRECT.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                stats["direct_rows"] += 1
                inp = str(row.get("input", "")).strip()
                out = str(row.get("output", "")).strip()
                if not is_logic_prompt(inp):
                    continue
                if not usable(inp=inp, out=out):
                    continue
                key = (inp, out)
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
                g.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["direct_kept"] += 1
                stats["written"] += 1

        with SRC_REASON.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                row = json.loads(line)
                stats["reason_rows"] += 1
                inp = str(row.get("input", "")).strip()
                out = clean_reason_output(row.get("output", ""))
                lang = str(row.get("language", "")).strip().lower()
                if lang != "ko":
                    continue
                if not is_logic_prompt(inp):
                    continue
                if not usable(inp=inp, out=out):
                    continue
                new_row = {
                    "input": inp,
                    "output": out,
                    "language": "ko",
                    "task_type": "korean",
                    "segment_tag": "ko",
                    "source": str(row.get("source", "")).strip() or "logic_reason_pool",
                }
                key = (inp, out)
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
                g.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                stats["reason_kept"] += 1
                stats["written"] += 1

    MANIFEST_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT_PATH), "manifest": str(MANIFEST_PATH), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
