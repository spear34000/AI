from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Any, Dict, List


def load_suite(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("suite file must be a JSON array")
    out: List[Dict[str, str]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        out.append({"id": str(row.get("id", f"item_{len(out)}")), "prompt": prompt})
    if not out:
        raise RuntimeError("suite file has no usable prompts")
    return out


def run_prompt(
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
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    return {
        "returncode": int(proc.returncode),
        "stdout": stdout,
        "stderr": stderr,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--suite", default="data/eval_prompt_suite_v1.json")
    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--chat_script", default="scripts/chat_slm.py")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    suite = load_suite(Path(args.suite))
    chat_script = Path(args.chat_script)
    results: List[Dict[str, Any]] = []
    for ckpt in args.checkpoints:
        ckpt_row: Dict[str, Any] = {"checkpoint": ckpt, "items": []}
        for item in suite:
            run = run_prompt(
                python_exe=str(args.python_exe),
                chat_script=chat_script,
                checkpoint=str(ckpt),
                prompt=item["prompt"],
                device=str(args.device),
                max_new_tokens=int(args.max_new_tokens),
            )
            entry = {
                "id": item["id"],
                "prompt": item["prompt"],
                "returncode": int(run["returncode"]),
                "stdout": str(run["stdout"]),
                "stderr": str(run["stderr"]),
            }
            ckpt_row["items"].append(entry)
        results.append(ckpt_row)

    summary = {"finished": True, "count": len(results), "results": results}
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = json.dumps(summary, ensure_ascii=False)
    try:
        print(payload)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(payload.encode(enc, errors="replace").decode(enc, errors="replace"))


if __name__ == "__main__":
    main()
