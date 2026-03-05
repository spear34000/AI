from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from chat_slm import ModelRouter, resolve_device, run_one_turn


ANSWER_RE = re.compile(
    r"(?:(?:최종답)|(?:정답)|(?:답)|(?:결론))\s*(?::|은)\s*(.+)",
    re.IGNORECASE,
)
FLOAT_INT_RE = re.compile(r"^-?\d+\.0+$")


def load_rows(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def normalize_answer(text: str) -> str:
    s = str(text or "").strip()
    s = s.replace("\r", "\n")
    matches = list(ANSWER_RE.finditer(s))
    if matches:
        s = matches[-1].group(1).strip().splitlines()[0].strip()
    else:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if lines:
            s = lines[-1]
    s = s.strip().strip("`").strip()
    s = re.sub(r"(?:(?:입니다)|(?:이다)|(?:임))$", "", s).strip()
    s = s.strip(".").strip()
    s = s.replace(" ", "")
    s = s.replace(",", "")
    low = s.lower()
    if low in {"예", "네", "맞다", "맞음", "yes"}:
        return "예"
    if low in {"아니오", "아니다", "아님", "no"}:
        return "아니오"
    if low in {"알수없음", "모름", "모르겠다", "보장할수없다", "판단불가", "unknown"}:
        return "알 수 없음"
    if FLOAT_INT_RE.match(low):
        return low.split(".")[0]
    if re.fullmatch(r"[a-d]", low):
        return low.upper()
    return s


def has_explicit_final_answer(text: str) -> bool:
    return bool(ANSWER_RE.search(str(text or "")))


def run_prompt(
    *,
    python_exe: str,
    chat_script: Path,
    checkpoint: str,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    cmd = [
        python_exe,
        str(chat_script),
        "--checkpoint",
        checkpoint,
        "--router",
        "single",
        "--prompt",
        prompt,
        "--device",
        device,
        "--temperature",
        "0",
        "--top_k",
        "1",
        "--top_p",
        "1.0",
        "--max_new_tokens",
        str(int(max_new_tokens)),
        "--disable_quality_rerank",
        "--zero_shot_mode",
        "off",
        "--heuristic_mode",
        "off",
        "--agent_mode",
        "off",
        "--force_raw",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        env=env,
    )
    return {
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout or "").strip(),
        "stderr": str(proc.stderr or "").strip(),
    }


def run_prompt_inprocess(
    *,
    router: ModelRouter,
    checkpoint: str,
    prompt: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    response, _bundle, _meta = run_one_turn(
        router=router,
        user_prompt=str(prompt),
        history=[],
        system_prompt="",
        max_new_tokens=int(max_new_tokens),
        max_input_tokens=39768,
        max_output_tokens=38000,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.08,
        min_new_tokens=0,
        disable_ko_guard=False,
        ko_guard_topk=96,
        ko_guard_rare_penalty=0.9,
        ko_guard_latin_penalty=0.25,
        disable_retrieval=True,
        disable_quality_rerank=True,
        rerank_candidates=1,
        rerank_temp_floor=0.15,
        rerank_top_k=1,
        rerank_top_p=1.0,
        zero_shot_mode="off",
        agent_mode="off",
        tool_cache_path="data/tool_knowledge_cache_v3_clean.jsonl",
        tool_lookup_timeout=0.1,
        disable_web_tool_lookup=True,
        heuristic_mode="off",
        force_raw=True,
    )
    return {
        "returncode": 0,
        "stdout": str(response or "").strip(),
        "stderr": "",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--data_path", default="data/logic_verified_v1_eval.jsonl")
    p.add_argument("--python_exe", default=sys.executable)
    p.add_argument("--chat_script", default="scripts/chat_slm.py")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--mode", default="inprocess", choices=["inprocess", "subprocess"])
    p.add_argument("--out_json", default="")
    args = p.parse_args()

    rows = load_rows(Path(args.data_path), int(args.limit))
    chat_script = Path(args.chat_script)
    results: List[Dict[str, Any]] = []
    for ckpt in args.checkpoints:
        router = None
        if str(args.mode) == "inprocess":
            router = ModelRouter(
                device=resolve_device(str(args.device)),
                checkpoint=str(ckpt),
                use_ema=False,
                router_mode="single",
                code_checkpoint="",
                chat_checkpoint="",
                quantize_int8=False,
            )
        hits = 0
        parsed = 0
        explicit = 0
        items: List[Dict[str, Any]] = []
        for row in rows:
            prompt = str(row.get("input", "")).strip()
            gold = normalize_answer(str(row.get("final_answer", "")))
            if str(args.mode) == "inprocess":
                run = run_prompt_inprocess(
                    router=router,
                    checkpoint=str(ckpt),
                    prompt=prompt,
                    max_new_tokens=int(args.max_new_tokens),
                )
            else:
                run = run_prompt(
                    python_exe=str(args.python_exe),
                    chat_script=chat_script,
                    checkpoint=str(ckpt),
                    prompt=prompt,
                    device=str(args.device),
                    max_new_tokens=int(args.max_new_tokens),
                )
            pred = normalize_answer(str(run["stdout"]))
            explicit += int(has_explicit_final_answer(str(run["stdout"])))
            ok = bool(pred == gold and pred != "")
            parsed += int(pred != "")
            hits += int(ok)
            items.append(
                {
                    "category": str(row.get("category", "")),
                    "gold": gold,
                    "pred": pred,
                    "match": ok,
                    "prompt": prompt,
                    "stdout": run["stdout"],
                }
            )
        total = len(items)
        by_cat: Dict[str, Dict[str, int]] = {}
        for item in items:
            cat = item["category"]
            if cat not in by_cat:
                by_cat[cat] = {"count": 0, "match": 0}
            by_cat[cat]["count"] += 1
            by_cat[cat]["match"] += int(item["match"])
        summary = {
            "checkpoint": str(ckpt),
            "count": total,
            "exact_match": (hits / total) if total else 0.0,
            "parse_rate": (parsed / total) if total else 0.0,
            "explicit_final_answer_rate": (explicit / total) if total else 0.0,
            "by_category": {
                k: {
                    "count": int(v["count"]),
                    "exact_match": (float(v["match"]) / float(v["count"])) if v["count"] else 0.0,
                }
                for k, v in sorted(by_cat.items())
            },
            "samples": items[: min(24, len(items))],
        }
        results.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    payload = {"finished": True, "count": len(results), "results": results}
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
