from __future__ import annotations

import json
import re
from pathlib import Path


BAD_SUBSTRINGS = [
    "\ufffd",
    "?쒓",
    "?대",
    "?쎄",
    "?ㅼ",
    "?몃",
    "?덈",
    "?섎",
    "필요하면 예시도",
    "더 짧게도 답할 수 있어요",
    "원하면 더 짧게도 답할 수 있어요",
    "질문 의도를",
    "한 줄만드",
]

BLOCK_TOKENS = [
    "ACTION ",
    "OBSERVATION ",
    "FINAL ",
    "도구:",
    "출력 형식:",
    "```",
    "http://",
    "https://",
]


def hangul_ratio(text: str) -> float:
    src = str(text or "")
    if not src:
        return 0.0
    hangul = len(re.findall(r"[가-힣]", src))
    return float(hangul) / float(max(1, len(src)))


def looks_clean(inp: str, out: str, source: str) -> bool:
    text = f"{inp}\n{out}"
    if any(tok in text for tok in BAD_SUBSTRINGS):
        return False
    if any(tok in text for tok in BLOCK_TOKENS):
        return False
    if "build_ko_agent_tool_dataset_v1" in source:
        return False
    if out.startswith("정답은") or out.startswith("답은"):
        return False
    if " 정답은 " in out or " 답은 " in out:
        return False
    if len(inp) < 3 or len(inp) > 280:
        return False
    if len(out) < 8 or len(out) > 320:
        return False
    if out.count("\n") > 4:
        return False
    if hangul_ratio(out) < 0.18:
        return False
    return True


def main() -> None:
    src = Path("data/repair_ko_only_clean_v1.jsonl")
    out = Path("data/repair_ko_direct_v2.jsonl")
    manifest = Path("data/repair_ko_direct_v2.manifest.json")

    stats = {
        "parsed": 0,
        "kept": 0,
        "bad_clean": 0,
        "duplicate": 0,
    }
    seen: set[tuple[str, str]] = set()

    out.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8", newline="\n") as g:
        for line in f:
            s = str(line).strip()
            if not s:
                continue
            row = json.loads(s)
            stats["parsed"] += 1

            inp = str(row.get("input", "")).strip()
            out_text = str(row.get("output", "")).strip()
            source = str(row.get("source", "")).strip()
            if not looks_clean(inp=inp, out=out_text, source=source):
                stats["bad_clean"] += 1
                continue

            key = (inp, out_text)
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)

            row["language"] = "ko"
            row["task_type"] = "korean"
            row["segment_tag"] = "ko"
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    manifest.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "manifest": str(manifest), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
