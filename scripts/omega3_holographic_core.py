from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from omega2_agentic_core import (
    extract_keywords,
    hard_verify_response,
    load_specs,
    soft_judge_score,
)
from train_slm import top_k_top_p_filtering


def build_response_prompt(prompt: str) -> str:
    src = str(prompt or "").strip()
    return f"### Instruction\n{src}\n\n### Response\n"


def extract_response(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt) :].strip()
    marker = "### Response"
    idx = full_text.rfind(marker)
    if idx >= 0:
        return full_text[idx + len(marker) :].lstrip(": \n\t").strip()
    return full_text.strip()


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def simple_tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_\\uac00-\\ud7a3]{2,}", str(text or "").lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    u = len(a.union(b))
    if u <= 0:
        return 0.0
    return float(len(a.intersection(b))) / float(u)


def choose_default_layers(model: torch.nn.Module, max_layers: int = 4) -> List[int]:
    n_layers = int(len(getattr(model, "blocks", [])))
    if n_layers <= 0:
        return [0]
    k = max(1, min(int(max_layers), n_layers))
    start = max(0, n_layers - k)
    return list(range(start, n_layers))


def parse_layer_list(raw: str, model: torch.nn.Module, default_max_layers: int = 4) -> List[int]:
    vals: List[int] = []
    for tok in str(raw or "").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            vals.append(int(t))
        except Exception:
            continue
    n_layers = int(len(getattr(model, "blocks", [])))
    if not vals:
        vals = choose_default_layers(model, max_layers=default_max_layers)
    out = sorted({x for x in vals if 0 <= int(x) < max(1, n_layers)})
    if not out:
        out = [max(0, n_layers - 1)]
    return out


def _normalize_vector(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = v.detach().to(dtype=torch.float32, device="cpu")
    n = float(torch.norm(x).item())
    if n <= eps:
        return x
    return x / float(n)


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    aa = _normalize_vector(a, eps=eps)
    bb = _normalize_vector(b, eps=eps)
    denom = float(torch.norm(aa).item() * torch.norm(bb).item())
    if denom <= eps:
        return 0.0
    return float(torch.dot(aa.view(-1), bb.view(-1)).item())


@dataclass
class VectorRecord:
    vector_id: str
    name: str
    kind: str
    layer: int
    keywords: List[str]
    score: float
    uses: int
    created_step: int
    status: str
    meta: Dict[str, Any]

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "VectorRecord":
        return cls(
            vector_id=str(row.get("vector_id", "")).strip(),
            name=str(row.get("name", "")).strip(),
            kind=str(row.get("kind", "skill")).strip().lower() or "skill",
            layer=int(row.get("layer", 0)),
            keywords=[str(x) for x in row.get("keywords", []) if str(x).strip()],
            score=float(row.get("score", 0.0)),
            uses=int(row.get("uses", 0)),
            created_step=int(row.get("created_step", 0)),
            status=str(row.get("status", "active")).strip().lower() or "active",
            meta=row.get("meta", {}) if isinstance(row.get("meta", {}), dict) else {},
        )

    def to_row(self) -> Dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "name": self.name,
            "kind": self.kind,
            "layer": int(self.layer),
            "keywords": list(self.keywords),
            "score": float(self.score),
            "uses": int(self.uses),
            "created_step": int(self.created_step),
            "status": self.status,
            "meta": dict(self.meta),
        }


class HolographicVectorMemory:
    def __init__(self, path: Path, merge_cosine: float = 0.985) -> None:
        self.path = Path(path)
        self.merge_cosine = float(merge_cosine)
        self.records: List[VectorRecord] = []
        self.vectors: Dict[str, torch.Tensor] = {}
        self.next_id = 1

    def load(self) -> None:
        self.records = []
        self.vectors = {}
        self.next_id = 1
        if not self.path.exists():
            return
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            return
        rows = payload.get("records", [])
        vecs = payload.get("vectors", {})
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    rec = VectorRecord.from_row(row)
                    if rec.vector_id:
                        self.records.append(rec)
        if isinstance(vecs, dict):
            for k, v in vecs.items():
                if torch.is_tensor(v):
                    self.vectors[str(k)] = v.detach().to(dtype=torch.float32, device="cpu")
        self.next_id = int(payload.get("next_id", 1))
        if self.next_id <= 0:
            self.next_id = 1
        if self.next_id <= len(self.records):
            self.next_id = len(self.records) + 1

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "omega3_holo_v1",
            "next_id": int(self.next_id),
            "records": [r.to_row() for r in self.records],
            "vectors": {k: v.detach().to(dtype=torch.float16, device="cpu") for k, v in self.vectors.items()},
        }
        torch.save(payload, self.path)

    def active_records(self, kind: str | None = None) -> List[VectorRecord]:
        out = [r for r in self.records if r.status == "active" and r.vector_id in self.vectors]
        if kind is not None:
            out = [r for r in out if r.kind == str(kind).strip().lower()]
        return out

    def count(self, kind: str | None = None) -> int:
        return len(self.active_records(kind=kind))

    def _find_merge_target(self, kind: str, layer: int, vector: torch.Tensor) -> Tuple[VectorRecord | None, float]:
        best_rec: VectorRecord | None = None
        best_cos = -1.0
        for rec in self.active_records(kind=kind):
            if int(rec.layer) != int(layer):
                continue
            old = self.vectors.get(rec.vector_id)
            if old is None:
                continue
            c = _cosine(old, vector)
            if c > best_cos:
                best_cos = c
                best_rec = rec
        return best_rec, float(best_cos)

    def add_or_merge(
        self,
        kind: str,
        name: str,
        layer: int,
        vector: torch.Tensor,
        keywords: Sequence[str],
        score: float,
        created_step: int,
        meta: Dict[str, Any],
    ) -> Tuple[bool, VectorRecord]:
        k = str(kind).strip().lower() or "skill"
        lname = str(name).strip() or f"{k}_vector"
        vec = _normalize_vector(vector)
        kws = sorted({str(x).strip().lower() for x in keywords if str(x).strip()})
        if not kws:
            kws = ["vector"]

        target, cosv = self._find_merge_target(k, int(layer), vec)
        if target is not None and cosv >= float(self.merge_cosine):
            old = self.vectors[target.vector_id]
            merged = _normalize_vector(0.7 * old + 0.3 * vec)
            self.vectors[target.vector_id] = merged
            target.score = float(max(target.score, float(score)))
            target.keywords = sorted(set(target.keywords).union(kws))
            target.meta = dict(target.meta)
            target.meta["last_merge_step"] = int(created_step)
            return False, target

        vid = f"vec_{self.next_id:07d}"
        self.next_id += 1
        rec = VectorRecord(
            vector_id=vid,
            name=lname,
            kind=k,
            layer=int(layer),
            keywords=kws,
            score=float(score),
            uses=0,
            created_step=int(created_step),
            status="active",
            meta=dict(meta or {}),
        )
        self.records.append(rec)
        self.vectors[vid] = vec
        return True, rec

    def retrieve(self, prompt: str, top_k: int, kind_allow: Sequence[str] | None = None) -> List[Tuple[float, VectorRecord]]:
        q = simple_tokens(prompt)
        allow = None
        if kind_allow is not None:
            allow = {str(x).strip().lower() for x in kind_allow}
        scored: List[Tuple[float, VectorRecord]] = []
        for rec in self.active_records():
            if allow is not None and rec.kind not in allow:
                continue
            kw = set(rec.keywords)
            overlap = jaccard(q, kw)
            stable = min(1.0, max(0.0, float(rec.score)))
            usage = min(0.15, 0.02 * math.log1p(max(0, int(rec.uses))))
            score = 0.60 * overlap + 0.30 * stable + usage
            if score <= 0.0:
                continue
            scored.append((float(score), rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[: max(0, int(top_k))]

    def mark_used(self, ids: Sequence[str]) -> None:
        s = {str(x) for x in ids}
        if not s:
            return
        for rec in self.records:
            if rec.vector_id in s:
                rec.uses = int(rec.uses) + 1

    def compose_layer_vectors(
        self,
        retrieved: Sequence[Tuple[float, VectorRecord]],
        kind_weights: Dict[str, float],
        max_norm: float = 2.0,
    ) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        for rel, rec in retrieved:
            base = self.vectors.get(rec.vector_id)
            if base is None:
                continue
            kw = float(kind_weights.get(rec.kind, 1.0))
            contrib = base.to(dtype=torch.float32, device="cpu") * (float(rel) * kw)
            layer = int(rec.layer)
            if layer in out:
                out[layer] = out[layer] + contrib
            else:
                out[layer] = contrib
        for k in list(out.keys()):
            v = out[k]
            n = float(torch.norm(v).item())
            if n > float(max_norm) and n > 1e-8:
                v = v * (float(max_norm) / n)
            out[k] = v
        return out


class SteeringContext:
    def __init__(
        self,
        model: torch.nn.Module,
        layer_vectors: Dict[int, torch.Tensor],
        alpha_by_layer: Dict[int, float],
    ) -> None:
        self.model = model
        self.layer_vectors = {int(k): v.detach().to(dtype=torch.float32, device="cpu") for k, v in layer_vectors.items()}
        self.alpha_by_layer = {int(k): float(v) for k, v in alpha_by_layer.items()}
        self._hooks: List[Any] = []

    def __enter__(self) -> "SteeringContext":
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            return self

        for layer_idx, vec in self.layer_vectors.items():
            if layer_idx < 0 or layer_idx >= len(blocks):
                continue
            alpha = float(self.alpha_by_layer.get(int(layer_idx), 1.0))
            if abs(alpha) <= 1e-9:
                continue
            layer_module = blocks[layer_idx]

            def _hook(_module, _inputs, output, vec_cpu=vec, a=alpha):
                if not torch.is_tensor(output):
                    return output
                v = vec_cpu.to(device=output.device, dtype=output.dtype).view(1, 1, -1)
                return output + (float(a) * v)

            h = layer_module.register_forward_hook(_hook)
            self._hooks.append(h)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []


@torch.no_grad()
def generate_with_steering(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    layer_vectors: Dict[int, torch.Tensor],
    alpha_by_layer: Dict[int, float],
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    full_prompt = build_response_prompt(prompt)
    tokens = tokenizer.encode(full_prompt, add_bos=True, add_eos=False)
    seq_len = int(getattr(model, "seq_len", 384))

    with SteeringContext(model=model, layer_vectors=layer_vectors, alpha_by_layer=alpha_by_layer):
        for _ in range(max(1, int(max_new_tokens))):
            x = torch.tensor([tokens[-seq_len:]], dtype=torch.long, device=device)
            logits, _ = model(x, targets=None)
            next_logits = logits[0, -1]
            if float(temperature) <= 0.0:
                next_id = int(torch.argmax(next_logits).item())
            else:
                next_logits = next_logits / max(1e-5, float(temperature))
                next_logits = top_k_top_p_filtering(next_logits, top_k=int(top_k), top_p=float(top_p))
                probs = torch.softmax(next_logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            tokens.append(next_id)
            if next_id == tokenizer.eos_id:
                break

    decoded = tokenizer.decode(tokens)
    return extract_response(decoded, prompt=full_prompt)


@torch.no_grad()
def capture_layer_signature(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    response: str,
    layers: Sequence[int],
    device: torch.device,
    tail_tokens: int = 64,
) -> Dict[int, torch.Tensor]:
    full_prompt = build_response_prompt(prompt) + str(response or "").strip()
    ids = tokenizer.encode(full_prompt, add_bos=True, add_eos=True)
    seq_len = int(getattr(model, "seq_len", 384))
    if len(ids) > seq_len:
        ids = ids[-seq_len:]

    x = torch.tensor([ids], dtype=torch.long, device=device)
    captures: Dict[int, torch.Tensor] = {}
    hooks = []
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return captures

    wanted = sorted({int(l) for l in layers if 0 <= int(l) < len(blocks)})
    if not wanted:
        return captures

    for layer_idx in wanted:
        module = blocks[layer_idx]

        def _hook(_module, _inputs, output, idx=layer_idx):
            if torch.is_tensor(output):
                captures[idx] = output.detach()

        hooks.append(module.register_forward_hook(_hook))

    try:
        model(x, targets=None)
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    out: Dict[int, torch.Tensor] = {}
    for layer_idx, t in captures.items():
        if t.dim() != 3:
            continue
        # t: [1, seq, d_model]
        tail = int(max(1, min(int(tail_tokens), int(t.shape[1]))))
        part = t[0, -tail:, :]
        v = part.mean(dim=0)
        out[int(layer_idx)] = _normalize_vector(v)
    return out


@torch.no_grad()
def capture_layer_signature_from_token_ids(
    model: torch.nn.Module,
    token_ids: Sequence[int],
    layers: Sequence[int],
    device: torch.device,
    tail_tokens: int = 64,
) -> Dict[int, torch.Tensor]:
    ids = [int(x) for x in token_ids]
    if not ids:
        return {}
    seq_len = int(getattr(model, "seq_len", 384))
    if len(ids) > seq_len:
        ids = ids[-seq_len:]

    x = torch.tensor([ids], dtype=torch.long, device=device)
    captures: Dict[int, torch.Tensor] = {}
    hooks = []
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return captures

    wanted = sorted({int(l) for l in layers if 0 <= int(l) < len(blocks)})
    if not wanted:
        return captures

    for layer_idx in wanted:
        module = blocks[layer_idx]

        def _hook(_module, _inputs, output, idx=layer_idx):
            if torch.is_tensor(output):
                captures[idx] = output.detach()

        hooks.append(module.register_forward_hook(_hook))

    try:
        model(x, targets=None)
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    out: Dict[int, torch.Tensor] = {}
    for layer_idx, t in captures.items():
        if t.dim() != 3:
            continue
        tail = int(max(1, min(int(tail_tokens), int(t.shape[1]))))
        part = t[0, -tail:, :]
        v = part.mean(dim=0)
        out[int(layer_idx)] = _normalize_vector(v)
    return out


@torch.no_grad()
def compress_prompt_to_latent_state(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    layers: Sequence[int],
    device: torch.device,
    max_input_tokens: int = 39768,
    chunk_tokens: int = 256,
    overlap_tokens: int = 32,
    tail_tokens: int = 48,
) -> Tuple[str, Dict[int, torch.Tensor], Dict[str, Any]]:
    raw_prompt = str(prompt or "").strip()
    ids = tokenizer.encode(raw_prompt, add_bos=True, add_eos=False)
    original_n = int(len(ids))
    max_in = max(64, int(max_input_tokens))
    truncated = False
    if len(ids) > max_in:
        ids = ids[-max_in:]
        truncated = True

    seq_len = int(getattr(model, "seq_len", 384))
    chunk = max(8, min(int(chunk_tokens), seq_len))
    overlap = max(0, min(int(overlap_tokens), chunk - 1))
    step = max(1, chunk - overlap)

    state: Dict[int, torch.Tensor] = {}
    chunk_count = 0
    if len(ids) > seq_len:
        acc: Dict[int, torch.Tensor] = {}
        cnt: Dict[int, int] = {}
        for start in range(0, len(ids), step):
            piece = ids[start : start + chunk]
            if not piece:
                break
            sig = capture_layer_signature_from_token_ids(
                model=model,
                token_ids=piece,
                layers=layers,
                device=device,
                tail_tokens=tail_tokens,
            )
            chunk_count += 1
            for layer_idx, vec in sig.items():
                if layer_idx in acc:
                    acc[layer_idx] = acc[layer_idx] + vec
                    cnt[layer_idx] = int(cnt[layer_idx]) + 1
                else:
                    acc[layer_idx] = vec.clone()
                    cnt[layer_idx] = 1
            if start + chunk >= len(ids):
                break
        for layer_idx, vec_sum in acc.items():
            c = max(1, int(cnt.get(layer_idx, 1)))
            state[layer_idx] = _normalize_vector(vec_sum / float(c))
    else:
        chunk_count = 1

    gen_ids = ids[-seq_len:]
    prompt_for_generation = tokenizer.decode(gen_ids)
    meta = {
        "prompt_tokens_original": int(original_n),
        "prompt_tokens_capped": int(len(ids)),
        "prompt_tokens_generation_window": int(len(gen_ids)),
        "input_truncated": bool(truncated),
        "compressed_chunks": int(chunk_count),
        "compressed_state_layers": sorted(int(k) for k in state.keys()),
        "max_input_tokens": int(max_in),
    }
    return prompt_for_generation, state, meta


def objective_score(
    hard_ok: bool,
    soft_score: float,
) -> float:
    return (2.0 if bool(hard_ok) else 0.0) + float(soft_score)


def evaluate_with_scalars(
    model: torch.nn.Module,
    tokenizer: Any,
    spec: Any,
    device: torch.device,
    max_new_tokens: int,
    base_layer_vectors: Dict[int, torch.Tensor],
    scalars: Dict[int, float],
    temperature: float,
    top_k: int,
    top_p: float,
    seq_len: int,
) -> Tuple[float, str, bool, float, str]:
    resp = generate_with_steering(
        model=model,
        tokenizer=tokenizer,
        prompt=spec.prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        layer_vectors=base_layer_vectors,
        alpha_by_layer=scalars,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    hard_ok, reason, _nll = hard_verify_response(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        prompt_for_eval=spec.prompt,
        response=resp,
        seq_len=seq_len,
        device=device,
    )
    soft = soft_judge_score(target=spec.target, response=resp)
    obj = objective_score(hard_ok=hard_ok, soft_score=soft)
    return float(obj), resp, bool(hard_ok), float(soft), str(reason)


def mezo_tune_scalars(
    model: torch.nn.Module,
    tokenizer: Any,
    spec: Any,
    device: torch.device,
    max_new_tokens: int,
    base_layer_vectors: Dict[int, torch.Tensor],
    init_scalars: Dict[int, float],
    seq_len: int,
    iters: int,
    sigma: float,
    lr: float,
    temperature: float,
    top_k: int,
    top_p: float,
    scalar_clip: float = 2.5,
    rnd: random.Random | None = None,
) -> Tuple[Dict[int, float], str, Dict[str, Any]]:
    if rnd is None:
        rnd = random.Random(42)
    keys = sorted(base_layer_vectors.keys())
    scalars = {int(k): float(init_scalars.get(int(k), 1.0)) for k in keys}
    best_scalars = dict(scalars)
    best_obj, best_resp, best_hard, best_soft, best_reason = evaluate_with_scalars(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        device=device,
        max_new_tokens=max_new_tokens,
        base_layer_vectors=base_layer_vectors,
        scalars=scalars,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seq_len=seq_len,
    )

    trace: List[Dict[str, Any]] = [
        {"iter": 0, "obj": float(best_obj), "hard_ok": bool(best_hard), "soft": float(best_soft), "reason": best_reason}
    ]

    for it in range(1, max(1, int(iters)) + 1):
        z: Dict[int, float] = {k: float(rnd.gauss(0.0, 1.0)) for k in keys}
        plus = {k: float(scalars[k] + float(sigma) * z[k]) for k in keys}
        minus = {k: float(scalars[k] - float(sigma) * z[k]) for k in keys}

        obj_p, _resp_p, _h_p, _s_p, _r_p = evaluate_with_scalars(
            model=model,
            tokenizer=tokenizer,
            spec=spec,
            device=device,
            max_new_tokens=max_new_tokens,
            base_layer_vectors=base_layer_vectors,
            scalars=plus,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seq_len=seq_len,
        )
        obj_m, _resp_m, _h_m, _s_m, _r_m = evaluate_with_scalars(
            model=model,
            tokenizer=tokenizer,
            spec=spec,
            device=device,
            max_new_tokens=max_new_tokens,
            base_layer_vectors=base_layer_vectors,
            scalars=minus,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seq_len=seq_len,
        )

        grad_scale = float(obj_p - obj_m) / float(max(1e-8, 2.0 * float(sigma)))
        for k in keys:
            scalars[k] = float(scalars[k] + float(lr) * grad_scale * z[k])
            scalars[k] = float(max(-abs(float(scalar_clip)), min(abs(float(scalar_clip)), scalars[k])))

        cur_obj, cur_resp, cur_hard, cur_soft, cur_reason = evaluate_with_scalars(
            model=model,
            tokenizer=tokenizer,
            spec=spec,
            device=device,
            max_new_tokens=max_new_tokens,
            base_layer_vectors=base_layer_vectors,
            scalars=scalars,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seq_len=seq_len,
        )
        trace.append(
            {
                "iter": int(it),
                "obj": float(cur_obj),
                "hard_ok": bool(cur_hard),
                "soft": float(cur_soft),
                "reason": cur_reason,
                "obj_plus": float(obj_p),
                "obj_minus": float(obj_m),
            }
        )
        if cur_obj > best_obj:
            best_obj = float(cur_obj)
            best_resp = str(cur_resp)
            best_scalars = dict(scalars)
            best_hard = bool(cur_hard)
            best_soft = float(cur_soft)
            best_reason = str(cur_reason)

    return best_scalars, best_resp, {
        "best_obj": float(best_obj),
        "best_hard_ok": bool(best_hard),
        "best_soft": float(best_soft),
        "best_reason": best_reason,
        "trace": trace,
    }


def build_patch_vectors_from_failure(
    model: torch.nn.Module,
    tokenizer: Any,
    spec: Any,
    response: str,
    layers: Sequence[int],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    sig_bad = capture_layer_signature(
        model=model,
        tokenizer=tokenizer,
        prompt=spec.prompt,
        response=response,
        layers=layers,
        device=device,
    )
    sig_good = capture_layer_signature(
        model=model,
        tokenizer=tokenizer,
        prompt=spec.prompt,
        response=spec.target,
        layers=layers,
        device=device,
    )
    out: Dict[int, torch.Tensor] = {}
    for l in layers:
        li = int(l)
        if li not in sig_bad or li not in sig_good:
            continue
        v = sig_good[li] - sig_bad[li]
        out[li] = _normalize_vector(v)
    return out


def build_skill_vectors_from_success(
    model: torch.nn.Module,
    tokenizer: Any,
    spec: Any,
    layers: Sequence[int],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    sig = capture_layer_signature(
        model=model,
        tokenizer=tokenizer,
        prompt=spec.prompt,
        response=spec.target,
        layers=layers,
        device=device,
    )
    return {int(k): _normalize_vector(v) for k, v in sig.items()}
