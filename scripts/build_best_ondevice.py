from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_slm import ByteTokenizer, TinyGPT, top_k_top_p_filtering


FILE_PRIORITY = {
    "slm_last.pt": 3,
    "slm_best.pt": 2,
    "slm_infer_fp16.pt": 1,
    "slm_infer_bf16.pt": 1,
}


PROBES: Tuple[Dict[str, str], ...] = (
    {
        "id": "general_explain",
        "instruction": "Explain the difference between on-device AI and cloud AI in two short sentences.",
        "expect": "general",
    },
    {
        "id": "code_python",
        "instruction": "Write a Python function that returns n*n for an integer n.",
        "expect": "code",
    },
    {
        "id": "math_simple",
        "instruction": "What is 12*(7+3)? Respond with number only.",
        "expect": "math",
    },
)


@dataclass
class Candidate:
    path: Path
    parent: Path
    filename: str
    val_loss: float
    mtime: float
    bytes_size: int
    meta_score: float
    probe_score: float = -1.0
    final_score: float = -1.0
    used_ema_for_probe: bool = False
    probes: List[Dict[str, Any]] = field(default_factory=list)
    step: int = -1
    has_ema: bool = False
    seq_len: int = -1
    d_model: int = -1
    n_layers: int = -1
    n_heads: int = -1
    params: int = -1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_candidates(root: Path) -> List[Path]:
    chosen_by_dir: Dict[Path, Path] = {}
    for p in root.rglob("slm_*.pt"):
        if not p.is_file():
            continue
        if p.name not in FILE_PRIORITY:
            continue
        low = str(p).lower()
        if "artifacts_qlora" in low:
            continue
        if "checkpoint-" in low:
            continue
        if "artifacts" not in low:
            continue

        parent = p.parent
        old = chosen_by_dir.get(parent)
        if old is None or FILE_PRIORITY[p.name] > FILE_PRIORITY.get(old.name, 0):
            chosen_by_dir[parent] = p

    out = sorted(chosen_by_dir.values(), key=lambda x: str(x).lower())
    return out


def read_summary_val_loss(dir_path: Path) -> float:
    summary_path = dir_path / "train_summary.json"
    if not summary_path.exists():
        return float("nan")
    try:
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return float("nan")
    if not isinstance(raw, dict):
        return float("nan")

    history = raw.get("history_tail", [])
    if isinstance(history, list):
        for row in reversed(history):
            if isinstance(row, dict):
                v = row.get("val_loss")
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    return float(v)

    v = raw.get("best_val_loss")
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return float("nan")


def normalize_loss_for_score(val_loss: float) -> float:
    if not math.isfinite(val_loss):
        return 0.0
    clipped = min(max(val_loss, 0.0), 3.0)
    # lower loss is better; 0.0 -> 1.0, 3.0 -> 0.0
    return (3.0 - clipped) / 3.0


def score_meta(val_loss: float, mtime: float, newest_mtime: float, oldest_mtime: float) -> float:
    loss_score = normalize_loss_for_score(val_loss)
    if newest_mtime <= oldest_mtime:
        recency = 0.5
    else:
        recency = (mtime - oldest_mtime) / float(newest_mtime - oldest_mtime)
    return 0.88 * loss_score + 0.12 * recency


def parse_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def cast_state_dict(state: Dict[str, Any], dtype: torch.dtype) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            t = v.detach().cpu()
            if t.is_floating_point():
                t = t.to(dtype=dtype)
            out[k] = t
        else:
            out[k] = v
    return out


def count_params(state: Dict[str, Any]) -> int:
    total = 0
    for v in state.values():
        if torch.is_tensor(v):
            total += int(v.numel())
    return int(total)


def extract_response(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt) :].strip()
    marker = "### Response"
    idx = full_text.rfind(marker)
    if idx >= 0:
        return full_text[idx + len(marker) :].lstrip(": \n\t").strip()
    return full_text.strip()


@torch.no_grad()
def generate_text(
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> str:
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    for _ in range(int(max_new_tokens)):
        x = torch.tensor([tokens[-model.seq_len :]], dtype=torch.long, device=device)
        logits, _ = model(x, targets=None)
        next_logits = logits[0, -1]
        if float(temperature) <= 0.0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            next_logits = next_logits / float(temperature)
            next_logits = top_k_top_p_filtering(next_logits, top_k=int(top_k), top_p=float(top_p))
            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        tokens.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    decoded = tokenizer.decode(tokens)
    return extract_response(decoded, prompt=prompt)


def build_model_from_ckpt(ckpt: Dict[str, Any], prefer_ema: bool, device: torch.device) -> Tuple[TinyGPT, ByteTokenizer, bool]:
    args = ckpt.get("args", {})
    if not isinstance(args, dict):
        args = {}

    tokenizer = ByteTokenizer()
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        seq_len=int(args.get("seq_len", 384)),
        d_model=int(args.get("d_model", 384)),
        n_heads=int(args.get("n_heads", 6)),
        n_layers=int(args.get("n_layers", 8)),
        mlp_mult=int(args.get("mlp_mult", 4)),
        dropout=float(args.get("dropout", 0.0)),
        gradient_checkpointing=False,
    )

    use_ema = False
    state = ckpt.get("model_state")
    ema_state = ckpt.get("ema_model_state")
    if bool(prefer_ema) and isinstance(ema_state, dict) and len(ema_state) > 0:
        state = ema_state
        use_ema = True
    if not isinstance(state, dict) or len(state) == 0:
        raise RuntimeError("checkpoint has no valid model_state")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device)
    model.eval()
    return model, tokenizer, use_ema


def bad_text_penalty(text: str) -> float:
    src = str(text or "").strip()
    if not src:
        return 1.0
    if re.search(r"(.)\1{8,}", src):
        return 0.8
    words = re.findall(r"[A-Za-z0-9_]+", src.lower())
    if len(words) >= 12:
        diversity = len(set(words)) / float(len(words))
        if diversity < 0.34:
            return 0.55
    return 0.0


def score_probe_output(expect: str, text: str) -> float:
    out = str(text or "").strip()
    length_score = min(len(out), 120) / 120.0
    penalty = bad_text_penalty(out)

    if expect == "code":
        code_hits = 0
        for token in ("def ", "return ", "```", "lambda ", "function "):
            if token in out:
                code_hits += 1
        code_score = min(1.0, code_hits / 2.0)
        return max(0.0, 0.60 * code_score + 0.40 * length_score - penalty)

    if expect == "math":
        has_120 = 1.0 if re.search(r"\b120\b", out) else 0.0
        numeric = 1.0 if re.search(r"\d", out) else 0.0
        return max(0.0, 0.75 * has_120 + 0.25 * numeric - penalty)

    # general
    sentence_like = 1.0 if (("." in out) or (";" in out) or (len(out.split()) >= 8)) else 0.0
    return max(0.0, 0.55 * sentence_like + 0.45 * length_score - penalty)


def run_probes_for_candidate(candidate: Candidate, prefer_ema: bool, max_new_tokens: int) -> Candidate:
    ckpt = torch.load(candidate.path, map_location="cpu", weights_only=False)
    model, tokenizer, used_ema = build_model_from_ckpt(ckpt=ckpt, prefer_ema=prefer_ema, device=torch.device("cpu"))

    args = ckpt.get("args", {})
    if not isinstance(args, dict):
        args = {}

    candidate.step = int(ckpt.get("step", -1))
    candidate.seq_len = int(args.get("seq_len", -1))
    candidate.d_model = int(args.get("d_model", -1))
    candidate.n_layers = int(args.get("n_layers", -1))
    candidate.n_heads = int(args.get("n_heads", -1))

    model_state = ckpt.get("model_state", {})
    if isinstance(model_state, dict):
        candidate.params = count_params(model_state)
    candidate.has_ema = isinstance(ckpt.get("ema_model_state"), dict) and len(ckpt.get("ema_model_state")) > 0
    candidate.used_ema_for_probe = bool(used_ema)

    results: List[Dict[str, Any]] = []
    scores: List[float] = []
    for probe in PROBES:
        instruction = str(probe["instruction"])
        prompt = f"### Instruction\n{instruction}\n\n### Response\n"
        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
        )
        s = score_probe_output(expect=str(probe["expect"]), text=response)
        scores.append(float(s))
        results.append(
            {
                "id": str(probe["id"]),
                "instruction": instruction,
                "response": response,
                "score": float(s),
            }
        )

    candidate.probes = results
    candidate.probe_score = float(sum(scores) / max(1, len(scores)))

    del model
    del tokenizer
    del ckpt
    gc.collect()
    return candidate


def select_best_candidate(
    candidates: List[Candidate],
    probe_top_n: int,
    prefer_ema: bool,
    max_new_tokens: int,
    skip_probe: bool,
) -> Tuple[Candidate, List[Candidate]]:
    sorted_by_meta = sorted(candidates, key=lambda c: c.meta_score, reverse=True)

    if not skip_probe:
        n = max(1, min(int(probe_top_n), len(sorted_by_meta)))
        for i in range(n):
            sorted_by_meta[i] = run_probes_for_candidate(
                candidate=sorted_by_meta[i],
                prefer_ema=prefer_ema,
                max_new_tokens=max_new_tokens,
            )

    for c in sorted_by_meta:
        if c.probe_score >= 0.0:
            c.final_score = 0.68 * c.meta_score + 0.32 * c.probe_score
        else:
            c.final_score = 0.95 * c.meta_score

    ranked = sorted(sorted_by_meta, key=lambda c: c.final_score, reverse=True)
    return ranked[0], ranked


def export_ondevice_checkpoint(
    source_ckpt_path: Path,
    output_ckpt_path: Path,
    dtype_name: str,
    prefer_ema: bool,
    selection_info: Dict[str, Any],
) -> Dict[str, Any]:
    ckpt = torch.load(source_ckpt_path, map_location="cpu", weights_only=False)
    dtype = parse_dtype(dtype_name)

    use_ema = False
    state = ckpt.get("model_state")
    ema_state = ckpt.get("ema_model_state")
    if bool(prefer_ema) and isinstance(ema_state, dict) and len(ema_state) > 0:
        state = ema_state
        use_ema = True
    if not isinstance(state, dict) or len(state) == 0:
        raise RuntimeError(f"model_state not found in {source_ckpt_path}")

    output_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "slm_infer_only_v2",
        "source_checkpoint": str(source_ckpt_path),
        "source_used_ema": bool(use_ema),
        "built_at_utc": utc_now_iso(),
        "step": int(ckpt.get("step", 0)),
        "best_val_loss": float(ckpt.get("best_val_loss", 0.0)),
        "args": ckpt.get("args", {}),
        "tokenizer": ckpt.get("tokenizer", {"type": "byte", "vocab_size": 259}),
        "selection": selection_info,
        "model_state": cast_state_dict(state=state, dtype=dtype),
    }
    torch.save(payload, output_ckpt_path)

    src_bytes = source_ckpt_path.stat().st_size
    out_bytes = output_ckpt_path.stat().st_size
    return {
        "source_used_ema": bool(use_ema),
        "input_bytes": int(src_bytes),
        "output_bytes": int(out_bytes),
        "size_ratio": float(out_bytes) / float(max(1, src_bytes)),
    }


def to_report_row(c: Candidate) -> Dict[str, Any]:
    return {
        "path": str(c.path),
        "parent": str(c.parent),
        "filename": c.filename,
        "val_loss": None if not math.isfinite(c.val_loss) else float(c.val_loss),
        "mtime": float(c.mtime),
        "bytes_size": int(c.bytes_size),
        "meta_score": float(c.meta_score),
        "probe_score": None if c.probe_score < 0.0 else float(c.probe_score),
        "final_score": None if c.final_score < 0.0 else float(c.final_score),
        "step": int(c.step),
        "has_ema": bool(c.has_ema),
        "used_ema_for_probe": bool(c.used_ema_for_probe),
        "seq_len": int(c.seq_len),
        "d_model": int(c.d_model),
        "n_layers": int(c.n_layers),
        "n_heads": int(c.n_heads),
        "params": int(c.params),
        "probes": c.probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and export the best local on-device TinyGPT checkpoint.")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--output_dir", default="artifacts_ondevice_best")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--probe_top_n", type=int, default=2, help="How many top meta candidates to probe")
    parser.add_argument("--probe_max_new_tokens", type=int, default=72)
    parser.add_argument("--skip_probe", action="store_true")
    parser.set_defaults(prefer_ema=True)
    parser.add_argument("--prefer_ema", dest="prefer_ema", action="store_true")
    parser.add_argument("--no_prefer_ema", dest="prefer_ema", action="store_false")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()

    paths = discover_candidates(root=root)
    if not paths:
        raise RuntimeError("No local TinyGPT checkpoints found under artifacts*/slm_*.pt")

    mtimes = [p.stat().st_mtime for p in paths]
    newest_mtime = max(mtimes)
    oldest_mtime = min(mtimes)

    candidates: List[Candidate] = []
    for p in paths:
        val_loss = read_summary_val_loss(p.parent)
        st = p.stat()
        mscore = score_meta(val_loss=val_loss, mtime=st.st_mtime, newest_mtime=newest_mtime, oldest_mtime=oldest_mtime)
        candidates.append(
            Candidate(
                path=p,
                parent=p.parent,
                filename=p.name,
                val_loss=val_loss,
                mtime=float(st.st_mtime),
                bytes_size=int(st.st_size),
                meta_score=float(mscore),
            )
        )

    best, ranked = select_best_candidate(
        candidates=candidates,
        probe_top_n=int(args.probe_top_n),
        prefer_ema=bool(args.prefer_ema),
        max_new_tokens=int(args.probe_max_new_tokens),
        skip_probe=bool(args.skip_probe),
    )

    output_ckpt = output_dir / f"slm_ondevice_{str(args.dtype).lower()}.pt"
    selection_info = {
        "selected_from": str(best.path),
        "selected_final_score": float(best.final_score),
        "selected_meta_score": float(best.meta_score),
        "selected_probe_score": None if best.probe_score < 0.0 else float(best.probe_score),
        "policy": {
            "external_models_forbidden": True,
            "local_tinygpt_only": True,
            "prefer_ema": bool(args.prefer_ema),
        },
    }

    export_stats = export_ondevice_checkpoint(
        source_ckpt_path=best.path,
        output_ckpt_path=output_ckpt,
        dtype_name=str(args.dtype),
        prefer_ema=bool(args.prefer_ema),
        selection_info=selection_info,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "created_at_utc": utc_now_iso(),
        "output_checkpoint": str(output_ckpt),
        "selected_source_checkpoint": str(best.path),
        "dtype": str(args.dtype).lower(),
        "prefer_ema": bool(args.prefer_ema),
        "skip_probe": bool(args.skip_probe),
        "probe_top_n": int(args.probe_top_n),
        "probe_max_new_tokens": int(args.probe_max_new_tokens),
        "export_stats": export_stats,
        "selected": to_report_row(best),
        "ranked_candidates": [to_report_row(c) for c in ranked],
        "usage": {
            "build_command": "python scripts/build_best_ondevice.py",
            "chat_command": "powershell -ExecutionPolicy Bypass -File scripts/chat_ondevice_best.ps1",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_checkpoint": str(output_ckpt),
                "manifest": str(manifest_path),
                "selected_source_checkpoint": str(best.path),
                "final_score": float(best.final_score),
                "meta_score": float(best.meta_score),
                "probe_score": None if best.probe_score < 0.0 else float(best.probe_score),
                "size_ratio": export_stats["size_ratio"],
                "source_used_ema": export_stats["source_used_ema"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
