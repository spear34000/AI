from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def sha1_of_file(path: Path, max_bytes: int = 1_048_576) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()[:16]


def file_fingerprint(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "sha1_head": sha1_of_file(path),
    }


def count_jsonl(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def resolve_python(root: Path) -> str:
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",
        Path(r"C:\Program Files\Python311\python.exe"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            proc = subprocess.run(
                [str(candidate), "-c", "import torch, sentencepiece; print(torch.__version__)"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode == 0:
                return str(candidate)
        except Exception:
            continue
    return sys.executable


def run_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Path,
    timeout: int = 3600,
    env: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    t0 = time.time()
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
        check=False,
    )
    return {
        "command": list(cmd),
        "cwd": str(cwd),
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout),
        "stderr": str(proc.stderr),
        "elapsed_sec": float(time.time() - t0),
    }


def pick_first_existing(paths: Sequence[str | Path]) -> str:
    for raw in paths:
        p = Path(str(raw))
        if p.exists():
            return str(p)
    return ""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

