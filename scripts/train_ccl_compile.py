
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_slm import ByteTokenizer, TinyGPT


TIER_WEIGHT = {
    "hard": 1.0,
    "soft": 0.5,
    "observational": 0.2,
}


def second_order_attention_context(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()

    attn_mod = getattr(torch.nn, "attention", None)
    if attn_mod is not None and hasattr(attn_mod, "sdpa_kernel") and hasattr(attn_mod, "SDPBackend"):
        try:
            return attn_mod.sdpa_kernel(backends=[attn_mod.SDPBackend.MATH])
        except Exception:
            pass

    if hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    return contextlib.nullcontext()


def safe_json_print(payload: Dict[str, Any]) -> None:
    s = json.dumps(payload, ensure_ascii=False)
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        s_safe = s.encode(enc, errors="replace").decode(enc, errors="replace")
        print(s_safe, flush=True)


def pick_first_non_empty(row: Dict[str, Any], keys: Sequence[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def normalize_tier(tier: str) -> str:
    t = str(tier).strip().lower()
    if t in ("hard", "high"):
        return "hard"
    if t in ("soft", "mid", "medium"):
        return "soft"
    return "observational"


def default_nll_threshold(tier: str) -> float:
    t = normalize_tier(tier)
    if t == "hard":
        return 2.6
    if t == "soft":
        return 3.1
    return 3.8


def build_response_prompt(prompt: str) -> str:
    src = str(prompt or "").strip()
    return f"### Instruction\n{src}\n\n### Response\n"


def normalize_text_for_match(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def extract_response(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt) :].strip()
    marker = "### Response"
    idx = full_text.rfind(marker)
    if idx >= 0:
        return full_text[idx + len(marker) :].lstrip(": \n\t").strip()
    return full_text.strip()


def simple_word_set(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9_]+", str(text).lower())
    return set(words)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


@dataclass
class SpecCase:
    spec_id: str
    tier: str
    prompt: str
    target: str
    verifier: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass
class Counterexample:
    spec_id: str
    tier: str
    prompt: str
    target: str
    mutator: str
    failure_score: float
    failure_reason: str
    verifier: Dict[str, Any]
    meta: Dict[str, Any]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be nn.Linear")
        if int(rank) <= 0:
            raise ValueError("rank must be > 0")

        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                self.rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))

        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        z = F.linear(self.drop(x), self.lora_A)
        z = F.linear(z, self.lora_B) * self.scale
        return out + z


def apply_lora_to_tinygpt(
    model: TinyGPT,
    top_layers: int,
    rank: int,
    alpha: float,
    dropout: float,
    targets: set[str],
) -> List[str]:
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = len(model.blocks)
    start_idx = max(0, n_layers - max(1, int(top_layers)))
    patched: List[str] = []

    for idx, blk in enumerate(model.blocks):
        if idx < start_idx:
            continue

        if "qkv" in targets:
            blk.attn.qkv = LoRALinear(blk.attn.qkv, rank=rank, alpha=alpha, dropout=dropout)
            patched.append(f"blocks.{idx}.attn.qkv")
        if "proj" in targets:
            blk.attn.proj = LoRALinear(blk.attn.proj, rank=rank, alpha=alpha, dropout=dropout)
            patched.append(f"blocks.{idx}.attn.proj")
        if "mlp_in" in targets:
            blk.mlp[0] = LoRALinear(blk.mlp[0], rank=rank, alpha=alpha, dropout=dropout)
            patched.append(f"blocks.{idx}.mlp.0")
        if "mlp_out" in targets:
            blk.mlp[2] = LoRALinear(blk.mlp[2], rank=rank, alpha=alpha, dropout=dropout)
            patched.append(f"blocks.{idx}.mlp.2")

    if not patched:
        raise RuntimeError("no modules patched; check --patch_targets")
    return patched


def trainable_patch_params(model: nn.Module) -> List[nn.Parameter]:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable patch parameters found")
    return params


def parse_patch_targets(raw: str) -> set[str]:
    out = set()
    for token in str(raw).split(","):
        t = token.strip().lower()
        if t:
            out.add(t)
    if not out:
        out = {"qkv", "proj", "mlp_in", "mlp_out"}
    return out


def parse_mutators(raw: str) -> List[str]:
    out = []
    for token in str(raw).split(","):
        t = token.strip().lower()
        if t:
            out.append(t)
    if "identity" not in out:
        out.insert(0, "identity")
    return out


def mutate_prompt(prompt: str, mutator: str, rnd: random.Random) -> str:
    src = str(prompt or "").strip()
    m = str(mutator).strip().lower()
    if m == "identity":
        return src
    if m == "boundary":
        return src + "\n\nExtra constraint: include boundary values 0, 1, -1 and one very large number case."
    if m == "constraint_clash":
        return src + "\n\nExtra constraint: be concise, but include one explicit exception case."
    if m == "reorder":
        parts = re.split(r"(?<=[.!?])\\s+", src)
        parts = [p for p in parts if p.strip()]
        if len(parts) >= 2:
            return " ".join(parts[::-1])
        return src
    if m == "ambiguity":
        return src + "\n\nIf any ambiguity exists, state assumptions first."
    if m == "whitespace":
        words = src.split()
        if len(words) <= 3:
            return src
        keep = []
        for w in words:
            keep.append(w)
            if rnd.random() < 0.18:
                keep.append("\n")
        return " ".join(keep).replace(" \n ", "\n")
    if m == "length_cap":
        return src + "\n\nConstraint: answer within 6 sentences."
    return src


def mutate_prompt_batch(prompt: str, mutators: Sequence[str], rnd: random.Random, n: int) -> List[Tuple[str, str]]:
    names = list(mutators) if mutators else ["identity"]
    out: List[Tuple[str, str]] = []
    used = set()

    base = mutate_prompt(prompt, "identity", rnd)
    out.append(("identity", base))
    used.add(("identity", base))

    while len(out) < max(1, int(n)):
        m = rnd.choice(names)
        p = mutate_prompt(prompt, m, rnd)
        key = (m, p)
        if key in used:
            if len(used) >= len(names):
                break
            continue
        used.add(key)
        out.append(key)
    return out


def load_specbook_from_jsonl(path: Path, max_specs: int, seed: int) -> List[SpecCase]:
    rows: List[SpecCase] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue

            prompt = pick_first_non_empty(row, ["prompt", "input", "instruction", "question", "context"])
            target = pick_first_non_empty(row, ["target", "output", "response", "answer", "completion"])
            if not prompt or not target:
                continue

            spec_id = str(row.get("id", f"spec_{idx:07d}")).strip() or f"spec_{idx:07d}"
            tier = normalize_tier(str(row.get("tier", row.get("severity", "soft"))))
            verifier = row.get("verifier", {})
            if not isinstance(verifier, dict):
                verifier = {}
            if "type" not in verifier:
                verifier["type"] = "nll"
            if str(verifier.get("type")).strip().lower() == "nll" and "max_token_nll" not in verifier:
                verifier["max_token_nll"] = default_nll_threshold(tier)
            meta = row.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}

            rows.append(
                SpecCase(
                    spec_id=spec_id,
                    tier=tier,
                    prompt=prompt,
                    target=target,
                    verifier=verifier,
                    meta=meta,
                )
            )

    rnd = random.Random(seed)
    rnd.shuffle(rows)
    if max_specs > 0:
        rows = rows[: int(max_specs)]
    if not rows:
        raise RuntimeError(f"no valid specs from {path}")
    return rows


def build_specbook_from_data(path: Path, max_specs: int, seed: int) -> List[SpecCase]:
    rows: List[SpecCase] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue

            prompt = pick_first_non_empty(row, ["input", "instruction", "prompt", "question", "context"])
            target = pick_first_non_empty(row, ["output", "response", "answer", "completion", "target"])
            if len(prompt) < 8 or len(target) < 8:
                continue

            q = str(row.get("_meta_quality_tier", "base")).strip().lower()
            if q == "high":
                tier = "hard"
            elif q == "mid":
                tier = "soft"
            else:
                tier = "observational"

            rows.append(
                SpecCase(
                    spec_id=f"auto_spec_{idx:07d}",
                    tier=tier,
                    prompt=prompt,
                    target=target,
                    verifier={"type": "nll", "max_token_nll": default_nll_threshold(tier)},
                    meta={
                        "task_type": str(row.get("task_type", "")),
                        "segment_tag": str(row.get("segment_tag", "")),
                    },
                )
            )

    rnd = random.Random(seed)
    rnd.shuffle(rows)
    if max_specs > 0:
        rows = rows[: int(max_specs)]
    if not rows:
        raise RuntimeError(f"no valid rows from {path}")
    return rows

def encode_case_xy(tokenizer: ByteTokenizer, prompt: str, target: str, seq_len: int) -> Tuple[List[int], List[int]]:
    prefix = build_response_prompt(prompt)
    full = prefix + str(target).strip()
    full_ids = tokenizer.encode(full, add_bos=True, add_eos=True)
    prefix_ids = tokenizer.encode(prefix, add_bos=True, add_eos=False)

    if len(full_ids) < 2:
        full_ids = [tokenizer.bos_id, tokenizer.eos_id]

    x = list(full_ids[:-1])
    y = list(full_ids[1:])
    prefix_pred_len = max(0, len(prefix_ids) - 1)
    for i in range(min(prefix_pred_len, len(y))):
        y[i] = -100

    if len(x) > int(seq_len):
        x = x[-int(seq_len) :]
        y = y[-int(seq_len) :]

    if all(int(v) == -100 for v in y):
        y[-1] = tokenizer.eos_id
    return x, y


def collate_xy(
    tokenizer: ByteTokenizer,
    samples: Sequence[Tuple[List[int], List[int], float]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x, _, _ in samples)
    x_out: List[List[int]] = []
    y_out: List[List[int]] = []
    w_out: List[float] = []
    for x, y, w in samples:
        pad_n = max_len - len(x)
        x_out.append(list(x) + [tokenizer.pad_id] * pad_n)
        y_out.append(list(y) + [-100] * pad_n)
        w_out.append(float(w))

    x_t = torch.tensor(x_out, dtype=torch.long, device=device)
    y_t = torch.tensor(y_out, dtype=torch.long, device=device)
    w_t = torch.tensor(w_out, dtype=torch.float32, device=device)
    return x_t, y_t, w_t


def masked_ce_seq(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, seqlen, vocab = logits.shape
    tok_loss = F.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(bsz, seqlen)
    mask = (targets != -100).float()
    seq_loss = (tok_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return seq_loss, mask.sum(dim=1)


def compute_cases_loss(
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    cases: Sequence[Counterexample | SpecCase],
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    samples: List[Tuple[List[int], List[int], float]] = []
    for c in cases:
        tier = normalize_tier(getattr(c, "tier"))
        w = float(TIER_WEIGHT.get(tier, 0.2))
        x, y = encode_case_xy(tokenizer, prompt=getattr(c, "prompt"), target=getattr(c, "target"), seq_len=seq_len)
        samples.append((x, y, w))

    x_t, y_t, w_t = collate_xy(tokenizer=tokenizer, samples=samples, device=device)
    logits, _ = model(x_t, targets=None)
    seq_loss, _ = masked_ce_seq(logits, y_t)
    loss = (seq_loss * w_t).sum() / w_t.sum().clamp_min(1e-6)
    return loss, seq_loss.detach()


@torch.no_grad()
def case_target_nll(
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    prompt: str,
    target: str,
    seq_len: int,
    device: torch.device,
) -> float:
    x, y = encode_case_xy(tokenizer=tokenizer, prompt=prompt, target=target, seq_len=seq_len)
    x_t = torch.tensor([x], dtype=torch.long, device=device)
    y_t = torch.tensor([y], dtype=torch.long, device=device)
    logits, _ = model(x_t, targets=None)
    seq_loss, _ = masked_ce_seq(logits, y_t)
    return float(seq_loss[0].detach().cpu().item())


@torch.no_grad()
def generate_response(
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    prefix = build_response_prompt(prompt)
    tokens = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    for _ in range(max(1, int(max_new_tokens))):
        x = torch.tensor([tokens[-model.seq_len :]], dtype=torch.long, device=device)
        logits, _ = model(x, targets=None)
        next_id = int(torch.argmax(logits[0, -1]).item())
        tokens.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    decoded = tokenizer.decode(tokens)
    return extract_response(decoded, prompt=prefix)


def verifier_passes(
    case: Counterexample | SpecCase,
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    seq_len: int,
    device: torch.device,
    max_new_tokens: int,
) -> Tuple[bool, str, float]:
    verifier = getattr(case, "verifier", {})
    if not isinstance(verifier, dict):
        verifier = {}
    vtype = str(verifier.get("type", "nll")).strip().lower()

    nll = case_target_nll(
        model=model,
        tokenizer=tokenizer,
        prompt=getattr(case, "prompt"),
        target=getattr(case, "target"),
        seq_len=seq_len,
        device=device,
    )

    if vtype == "nll":
        thr = float(verifier.get("max_token_nll", default_nll_threshold(getattr(case, "tier"))))
        passed = nll <= thr
        reason = "" if passed else f"token_nll>{thr:.3f}"
        return passed, reason, nll

    generated = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=getattr(case, "prompt"),
        device=device,
        max_new_tokens=max_new_tokens,
    )
    g_norm = normalize_text_for_match(generated)
    t_norm = normalize_text_for_match(getattr(case, "target"))

    if vtype == "exact":
        passed = g_norm == t_norm
        return passed, "" if passed else "exact_mismatch", nll

    if vtype == "contains":
        required = verifier.get("required", [])
        if not isinstance(required, list):
            required = []
        missing = []
        for tok in required:
            s = str(tok).strip().lower()
            if s and s not in g_norm:
                missing.append(s)
        passed = len(missing) == 0
        return passed, "" if passed else f"missing:{','.join(missing[:4])}", nll

    if vtype == "regex":
        pattern = str(verifier.get("pattern", "")).strip()
        if not pattern:
            return False, "empty_regex", nll
        try:
            ok = re.search(pattern, generated) is not None
        except re.error:
            return False, "bad_regex", nll
        return ok, "" if ok else "regex_mismatch", nll

    thr = float(verifier.get("max_token_nll", default_nll_threshold(getattr(case, "tier"))))
    passed = nll <= thr
    return passed, "" if passed else f"token_nll>{thr:.3f}", nll


def flatten_grads(
    grads: Sequence[torch.Tensor | None],
    params: Sequence[nn.Parameter],
) -> torch.Tensor:
    chunks = []
    for g, p in zip(grads, params):
        if g is None:
            chunks.append(torch.zeros_like(p, dtype=torch.float32).reshape(-1))
        else:
            chunks.append(g.reshape(-1).to(dtype=torch.float32))
    return torch.cat(chunks, dim=0)


def apply_flat_update(params: Sequence[nn.Parameter], delta: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            piece = delta[offset : offset + n].view_as(p).to(dtype=p.dtype, device=p.device)
            p.add_(piece)
            offset += n


def conjugate_gradient(matvec, b: torch.Tensor, iters: int, tol: float) -> Tuple[torch.Tensor, int, float]:
    x = torch.zeros_like(b)
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    resid = float(torch.sqrt(rs_old).item())
    if resid < float(tol):
        return x, 0, resid

    n_iter = 0
    for i in range(max(1, int(iters))):
        ap = matvec(p)
        denom = torch.dot(p, ap).clamp_min(1e-12)
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.dot(r, r)
        resid = float(torch.sqrt(rs_new).item())
        n_iter = i + 1
        if resid < float(tol):
            break
        beta = rs_new / rs_old.clamp_min(1e-12)
        p = r + beta * p
        rs_old = rs_new
    return x, n_iter, resid


class ProtectedSubspaceLedger:
    def __init__(self, dim: int, max_basis: int, device: torch.device) -> None:
        self.dim = int(dim)
        self.max_basis = int(max_basis)
        self.device = device
        self._basis: List[torch.Tensor] = []

    def size(self) -> int:
        return len(self._basis)

    def project(self, v: torch.Tensor) -> torch.Tensor:
        if not self._basis:
            return v
        b = torch.stack(self._basis, dim=1)
        coeff = torch.matmul(b.transpose(0, 1), v)
        return v - torch.matmul(b, coeff)

    def add(self, vec: torch.Tensor, eps: float = 1e-8) -> bool:
        v = vec.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        if v.numel() != self.dim:
            return False
        if float(v.norm().item()) <= float(eps):
            return False
        if self._basis:
            b = torch.stack(self._basis, dim=1)
            v = v - torch.matmul(b, torch.matmul(b.transpose(0, 1), v))
        n1 = float(v.norm().item())
        if n1 <= float(eps):
            return False
        v = v / max(n1, float(eps))
        self._basis.append(v)
        if len(self._basis) > self.max_basis:
            self._basis = self._basis[-self.max_basis :]
        return True

    def state_dict(self) -> Dict[str, Any]:
        if self._basis:
            mat = torch.stack(self._basis, dim=0).detach().cpu()
        else:
            mat = torch.empty((0, self.dim), dtype=torch.float32)
        return {"dim": int(self.dim), "max_basis": int(self.max_basis), "basis": mat}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._basis = []
        if not isinstance(state, dict):
            return
        raw = state.get("basis")
        if not torch.is_tensor(raw):
            return
        raw = raw.to(dtype=torch.float32, device=self.device)
        if raw.ndim != 2:
            return
        for i in range(raw.size(0)):
            self.add(raw[i])

def sample_specs_weighted(specs: Sequence[SpecCase], k: int, rnd: random.Random) -> List[SpecCase]:
    if not specs:
        return []
    weights = []
    for s in specs:
        tier = normalize_tier(s.tier)
        if tier == "hard":
            weights.append(2.0)
        elif tier == "soft":
            weights.append(1.2)
        else:
            weights.append(0.7)
    return list(rnd.choices(specs, weights=weights, k=max(1, int(k))))


def mine_counterexamples(
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    specs: Sequence[SpecCase],
    device: torch.device,
    seq_len: int,
    rnd: random.Random,
    mine_seeds: int,
    mutators: Sequence[str],
    mutations_per_seed: int,
    max_counterexamples: int,
    novelty_threshold: float,
    max_new_tokens: int,
) -> List[Counterexample]:
    seeds = sample_specs_weighted(specs=specs, k=mine_seeds, rnd=rnd)
    candidates: List[Counterexample] = []
    near_miss_pool: List[Counterexample] = []
    for s in seeds:
        mutated = mutate_prompt_batch(prompt=s.prompt, mutators=mutators, rnd=rnd, n=mutations_per_seed)
        for m_name, m_prompt in mutated:
            probe = Counterexample(
                spec_id=s.spec_id,
                tier=s.tier,
                prompt=m_prompt,
                target=s.target,
                mutator=m_name,
                failure_score=0.0,
                failure_reason="",
                verifier=s.verifier,
                meta=dict(s.meta),
            )
            passed, reason, nll = verifier_passes(
                case=probe,
                model=model,
                tokenizer=tokenizer,
                seq_len=seq_len,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            if passed:
                probe.failure_score = float(nll)
                probe.failure_reason = "near_miss"
                near_miss_pool.append(probe)
            else:
                probe.failure_score = float(nll)
                probe.failure_reason = reason or "verify_failed"
                candidates.append(probe)

    if not candidates and near_miss_pool:
        near_miss_pool.sort(key=lambda x: float(x.failure_score), reverse=True)
        top_n = near_miss_pool[: max(1, int(max_counterexamples) * 3)]
        candidates.extend(top_n)

    candidates.sort(key=lambda x: float(x.failure_score), reverse=True)
    selected: List[Counterexample] = []
    seen_word_sets: List[set[str]] = []
    for c in candidates:
        ws = simple_word_set(c.prompt)
        is_novel = True
        for old in seen_word_sets:
            if jaccard(ws, old) >= float(novelty_threshold):
                is_novel = False
                break
        if not is_novel:
            continue
        selected.append(c)
        seen_word_sets.append(ws)
        if len(selected) >= int(max_counterexamples):
            break
    return selected


def evaluate_specs(
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    specs: Sequence[SpecCase],
    device: torch.device,
    seq_len: int,
    max_specs: int,
    max_new_tokens: int,
    rnd: random.Random,
) -> Dict[str, Any]:
    pool = list(specs)
    rnd.shuffle(pool)
    if max_specs > 0:
        pool = pool[: int(max_specs)]

    tier_total = {"hard": 0, "soft": 0, "observational": 0}
    tier_pass = {"hard": 0, "soft": 0, "observational": 0}
    tier_nll_sum = {"hard": 0.0, "soft": 0.0, "observational": 0.0}
    hard_case_pass: Dict[str, bool] = {}

    for s in pool:
        tier = normalize_tier(s.tier)
        passed, _, nll = verifier_passes(
            case=s,
            model=model,
            tokenizer=tokenizer,
            seq_len=seq_len,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        tier_total[tier] += 1
        tier_nll_sum[tier] += float(nll)
        if passed:
            tier_pass[tier] += 1
        if tier == "hard":
            hard_case_pass[s.spec_id] = bool(passed)

    def ratio(a: int, b: int) -> float:
        if b <= 0:
            return float("nan")
        return float(a) / float(b)

    return {
        "checked": len(pool),
        "tier_total": tier_total,
        "tier_pass": tier_pass,
        "hard_pass_rate": ratio(tier_pass["hard"], tier_total["hard"]),
        "soft_pass_rate": ratio(tier_pass["soft"], tier_total["soft"]),
        "obs_pass_rate": ratio(tier_pass["observational"], tier_total["observational"]),
        "tier_avg_nll": {
            "hard": (tier_nll_sum["hard"] / max(1, tier_total["hard"])),
            "soft": (tier_nll_sum["soft"] / max(1, tier_total["soft"])),
            "observational": (tier_nll_sum["observational"] / max(1, tier_total["observational"])),
        },
        "hard_case_pass": hard_case_pass,
    }


def refresh_ledger(
    ledger: ProtectedSubspaceLedger,
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    protect_specs: Sequence[SpecCase],
    patch_params: Sequence[nn.Parameter],
    device: torch.device,
    seq_len: int,
    max_add: int,
) -> int:
    add_n = 0
    for spec in protect_specs:
        if add_n >= int(max_add):
            break
        loss, _ = compute_cases_loss(
            model=model,
            tokenizer=tokenizer,
            cases=[spec],
            seq_len=seq_len,
            device=device,
        )
        grads = torch.autograd.grad(loss, patch_params, retain_graph=False, create_graph=False, allow_unused=True)
        gvec = flatten_grads(grads=grads, params=patch_params)
        if ledger.add(gvec):
            add_n += 1
    return add_n


def compile_update_step(
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    batch_cases: Sequence[Counterexample],
    patch_params: Sequence[nn.Parameter],
    ledger: ProtectedSubspaceLedger | None,
    device: torch.device,
    seq_len: int,
    cg_iters: int,
    cg_tol: float,
    damping: float,
    step_scale: float,
    max_update_norm: float,
) -> Dict[str, Any]:
    if not batch_cases:
        return {
            "loss_before": float("nan"),
            "loss_after": float("nan"),
            "cg_iters": 0,
            "cg_resid": float("nan"),
            "delta_norm_raw": 0.0,
            "delta_norm_proj": 0.0,
        }

    with second_order_attention_context(device):
        loss, _ = compute_cases_loss(
            model=model,
            tokenizer=tokenizer,
            cases=batch_cases,
            seq_len=seq_len,
            device=device,
        )
        grads = torch.autograd.grad(loss, patch_params, create_graph=True, retain_graph=True, allow_unused=True)
    g = flatten_grads(grads=grads, params=patch_params)
    if not torch.isfinite(g).all():
        return {
            "loss_before": float(loss.detach().cpu().item()),
            "loss_after": float(loss.detach().cpu().item()),
            "cg_iters": 0,
            "cg_resid": float("nan"),
            "delta_norm_raw": 0.0,
            "delta_norm_proj": 0.0,
            "reverted": True,
        }
    rhs = g.detach()

    def matvec(v: torch.Tensor) -> torch.Tensor:
        gv = torch.dot(g, v)
        with second_order_attention_context(device):
            hv_list = torch.autograd.grad(gv, patch_params, create_graph=False, retain_graph=True, allow_unused=True)
        hv = flatten_grads(grads=hv_list, params=patch_params).detach()
        if not torch.isfinite(hv).all():
            hv = torch.zeros_like(v)
        return hv + float(damping) * v

    delta_raw, n_cg, resid = conjugate_gradient(matvec=matvec, b=rhs, iters=cg_iters, tol=cg_tol)
    delta_raw = -float(step_scale) * delta_raw
    if not torch.isfinite(delta_raw).all():
        delta_raw = torch.zeros_like(delta_raw)
    raw_norm = float(delta_raw.norm().detach().cpu().item())
    max_norm = max(1e-8, float(max_update_norm))
    if raw_norm > max_norm:
        delta_raw = delta_raw * (max_norm / raw_norm)
    delta = ledger.project(delta_raw) if ledger is not None else delta_raw
    if not torch.isfinite(delta).all():
        delta = torch.zeros_like(delta)
    proj_norm = float(delta.norm().detach().cpu().item())
    if proj_norm > max_norm:
        delta = delta * (max_norm / proj_norm)
    apply_flat_update(params=patch_params, delta=delta)

    with torch.no_grad():
        loss_after, _ = compute_cases_loss(
            model=model,
            tokenizer=tokenizer,
            cases=batch_cases,
            seq_len=seq_len,
            device=device,
        )
    loss_before_v = float(loss.detach().cpu().item())
    loss_after_v = float(loss_after.detach().cpu().item())
    reverted = False
    if (not math.isfinite(loss_after_v)) or (loss_after_v > max(4.0, loss_before_v * 4.0)):
        apply_flat_update(params=patch_params, delta=-delta)
        loss_after_v = loss_before_v
        reverted = True

    return {
        "loss_before": loss_before_v,
        "loss_after": loss_after_v,
        "cg_iters": int(n_cg),
        "cg_resid": float(resid),
        "delta_norm_raw": float(delta_raw.norm().detach().cpu().item()),
        "delta_norm_proj": float(delta.norm().detach().cpu().item()),
        "reverted": bool(reverted),
    }


def lora_patch_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if "lora_A" in name or "lora_B" in name:
            out[name] = tensor.detach().cpu()
    return out


def load_lora_patch_state(model: nn.Module, state: Dict[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    current = model.state_dict()
    update: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k not in current:
            continue
        if not torch.is_tensor(v):
            continue
        if tuple(v.shape) != tuple(current[k].shape):
            continue
        update[k] = v
    if update:
        current.update(update)
        model.load_state_dict(current, strict=False)


def build_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[TinyGPT, ByteTokenizer, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    state = ckpt.get("ema_model_state")
    if not isinstance(state, dict) or not state:
        state = ckpt.get("model_state")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("checkpoint has no model_state")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    return model, tokenizer, args

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", default="artifacts_ondevice_best/slm_ondevice_fp16.pt")
    parser.add_argument("--spec_path", default="data/ccl_specbook_v1.jsonl")
    parser.add_argument("--data_path", default="data/slm_mit_unified_v4.jsonl")
    parser.add_argument("--output_dir", default="artifacts_ccl2_compile")
    parser.add_argument("--resume_patch", default="")

    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--max_specs", type=int, default=1800)
    parser.add_argument("--mine_seeds", type=int, default=32)
    parser.add_argument("--mutations_per_seed", type=int, default=4)
    parser.add_argument("--counterexamples_per_step", type=int, default=8)
    parser.add_argument("--compile_batch_size", type=int, default=6)
    parser.add_argument("--verify_interval", type=int, default=4)
    parser.add_argument("--verify_specs", type=int, default=300)
    parser.add_argument("--target_hard_pass", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--mutators", type=str, default="identity,boundary,constraint_clash,reorder,ambiguity,whitespace,length_cap")

    parser.add_argument("--patch_top_layers", type=int, default=4)
    parser.add_argument("--patch_targets", type=str, default="qkv,proj,mlp_in,mlp_out")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument("--cg_iters", type=int, default=10)
    parser.add_argument("--cg_tol", type=float, default=1e-4)
    parser.add_argument("--damping", type=float, default=0.08)
    parser.add_argument("--step_scale", type=float, default=0.8)
    parser.add_argument("--max_update_norm", type=float, default=0.08)

    parser.add_argument("--ledger_max_basis", type=int, default=24)
    parser.add_argument("--ledger_refresh_interval", type=int, default=3)
    parser.add_argument("--ledger_refresh_samples", type=int, default=6)
    parser.add_argument("--novelty_threshold", type=float, default=0.92)

    parser.add_argument("--save_interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=0, help="0 means use model seq_len")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    rnd = random.Random(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")

    base_ckpt = Path(args.base_checkpoint)
    if not base_ckpt.exists():
        raise RuntimeError(f"base checkpoint not found: {base_ckpt}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cex_ledger_path = out_dir / "counterexample_ledger.jsonl"

    model, tokenizer, model_args = build_model_from_checkpoint(base_ckpt, device=device)
    seq_len = int(args.seq_len) if int(args.seq_len) > 0 else int(model.seq_len)

    patched_modules = apply_lora_to_tinygpt(
        model=model,
        top_layers=int(args.patch_top_layers),
        rank=int(args.lora_rank),
        alpha=float(args.lora_alpha),
        dropout=float(args.lora_dropout),
        targets=parse_patch_targets(args.patch_targets),
    )
    patch_params = trainable_patch_params(model)
    patch_dim = int(sum(p.numel() for p in patch_params))

    ledger = ProtectedSubspaceLedger(dim=patch_dim, max_basis=int(args.ledger_max_basis), device=device)
    resume_patch = str(args.resume_patch).strip()
    if resume_patch:
        rp = Path(resume_patch)
        if not rp.exists():
            raise RuntimeError(f"resume_patch not found: {rp}")
        payload = torch.load(rp, map_location="cpu", weights_only=False)
        load_lora_patch_state(model, payload.get("patch_state", {}))
        ledger.load_state_dict(payload.get("ledger_state", {}))

    spec_path = Path(args.spec_path)
    if spec_path.exists() and spec_path.stat().st_size > 0:
        specs = load_specbook_from_jsonl(spec_path, max_specs=int(args.max_specs), seed=int(args.seed))
        spec_source = str(spec_path)
    else:
        data_path = Path(args.data_path)
        if not data_path.exists():
            raise RuntimeError(f"spec_path not found and data_path missing: {data_path}")
        specs = build_specbook_from_data(data_path, max_specs=int(args.max_specs), seed=int(args.seed))
        spec_source = str(data_path)

    hard_specs = [s for s in specs if normalize_tier(s.tier) == "hard"]
    if not hard_specs:
        raise RuntimeError("specbook has no hard specs; add high-quality constraints first")

    mutators = parse_mutators(args.mutators)

    safe_json_print(
        {
            "mode": "ccl2_compile",
            "device": str(device),
            "base_checkpoint": str(base_ckpt),
            "spec_source": spec_source,
            "specs_total": len(specs),
            "specs_hard": len(hard_specs),
            "steps": int(args.steps),
            "seq_len": int(seq_len),
            "patch": {
                "top_layers": int(args.patch_top_layers),
                "targets": sorted(parse_patch_targets(args.patch_targets)),
                "lora_rank": int(args.lora_rank),
                "lora_alpha": float(args.lora_alpha),
                "lora_dropout": float(args.lora_dropout),
                "trainable_params": int(patch_dim),
                "patched_modules": len(patched_modules),
            },
            "compile": {
                "cg_iters": int(args.cg_iters),
                "cg_tol": float(args.cg_tol),
                "damping": float(args.damping),
                "step_scale": float(args.step_scale),
                "max_update_norm": float(args.max_update_norm),
            },
            "ledger": {
                "max_basis": int(args.ledger_max_basis),
                "start_basis": int(ledger.size()),
            },
        }
    )

    initial_seed_specs = rnd.sample(hard_specs, k=min(len(hard_specs), max(1, int(args.ledger_refresh_samples))))
    seeded = refresh_ledger(
        ledger=ledger,
        model=model,
        tokenizer=tokenizer,
        protect_specs=initial_seed_specs,
        patch_params=patch_params,
        device=device,
        seq_len=seq_len,
        max_add=max(1, int(args.ledger_refresh_samples)),
    )
    safe_json_print({"ledger_seeded": int(seeded), "ledger_basis": int(ledger.size())})

    history: List[Dict[str, Any]] = []
    hard_prev: Dict[str, bool] = {}
    fixed_count = 0
    regression_count = 0
    target_hit_step = -1
    target_hit_time_sec = float("nan")
    best_hard_pass = -1.0
    best_step = 0
    start = time.time()

    for step in range(1, int(args.steps) + 1):
        model.train()
        counterexamples = mine_counterexamples(
            model=model,
            tokenizer=tokenizer,
            specs=specs,
            device=device,
            seq_len=seq_len,
            rnd=rnd,
            mine_seeds=int(args.mine_seeds),
            mutators=mutators,
            mutations_per_seed=int(args.mutations_per_seed),
            max_counterexamples=int(args.counterexamples_per_step),
            novelty_threshold=float(args.novelty_threshold),
            max_new_tokens=int(args.max_new_tokens),
        )

        if counterexamples:
            batch = counterexamples[: max(1, int(args.compile_batch_size))]
            comp = compile_update_step(
                model=model,
                tokenizer=tokenizer,
                batch_cases=batch,
                patch_params=patch_params,
                ledger=ledger,
                device=device,
                seq_len=seq_len,
                cg_iters=int(args.cg_iters),
                cg_tol=float(args.cg_tol),
                damping=float(args.damping),
                step_scale=float(args.step_scale),
                max_update_norm=float(args.max_update_norm),
            )
        else:
            comp = {
                "loss_before": float("nan"),
                "loss_after": float("nan"),
                "cg_iters": 0,
                "cg_resid": float("nan"),
                "delta_norm_raw": 0.0,
                "delta_norm_proj": 0.0,
            }

        with cex_ledger_path.open("a", encoding="utf-8") as f:
            for c in counterexamples:
                row = {
                    "step": int(step),
                    "spec_id": c.spec_id,
                    "tier": c.tier,
                    "mutator": c.mutator,
                    "failure_score": float(c.failure_score),
                    "failure_reason": c.failure_reason,
                    "prompt": c.prompt,
                    "target": c.target,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        record = {
            "step": int(step),
            "mined_counterexamples": int(len(counterexamples)),
            "compile": comp,
            "ledger_basis": int(ledger.size()),
        }

        need_verify = step == 1 or step % int(args.verify_interval) == 0 or step == int(args.steps)
        if need_verify:
            model.eval()
            ev = evaluate_specs(
                model=model,
                tokenizer=tokenizer,
                specs=specs,
                device=device,
                seq_len=seq_len,
                max_specs=int(args.verify_specs),
                max_new_tokens=int(args.max_new_tokens),
                rnd=rnd,
            )
            hard_pass = float(ev.get("hard_pass_rate", float("nan")))
            if not math.isnan(hard_pass) and hard_pass > best_hard_pass:
                best_hard_pass = hard_pass
                best_step = int(step)

            hard_map = ev.get("hard_case_pass", {})
            if isinstance(hard_map, dict):
                for sid, curr in hard_map.items():
                    curr_b = bool(curr)
                    if sid in hard_prev:
                        if hard_prev[sid] is False and curr_b is True:
                            fixed_count += 1
                        elif hard_prev[sid] is True and curr_b is False:
                            regression_count += 1
                    hard_prev[sid] = curr_b

            if target_hit_step < 0 and (not math.isnan(hard_pass)) and hard_pass >= float(args.target_hard_pass):
                target_hit_step = int(step)
                target_hit_time_sec = float(time.time() - start)

            pass_hard_ids = [sid for sid, ok in hard_map.items() if bool(ok)] if isinstance(hard_map, dict) else []
            protect_pool = [s for s in hard_specs if s.spec_id in set(pass_hard_ids)]
            if not protect_pool:
                protect_pool = hard_specs

            if step % int(args.ledger_refresh_interval) == 0:
                rnd.shuffle(protect_pool)
                added = refresh_ledger(
                    ledger=ledger,
                    model=model,
                    tokenizer=tokenizer,
                    protect_specs=protect_pool[: max(1, int(args.ledger_refresh_samples))],
                    patch_params=patch_params,
                    device=device,
                    seq_len=seq_len,
                    max_add=max(1, int(args.ledger_refresh_samples)),
                )
                record["ledger_refreshed"] = int(added)
                record["ledger_basis"] = int(ledger.size())

            elapsed_h = max(1e-9, (time.time() - start) / 3600.0)
            record["verify"] = {
                "checked": int(ev.get("checked", 0)),
                "hard_pass_rate": hard_pass,
                "soft_pass_rate": float(ev.get("soft_pass_rate", float("nan"))),
                "obs_pass_rate": float(ev.get("obs_pass_rate", float("nan"))),
                "hard_avg_nll": float((ev.get("tier_avg_nll") or {}).get("hard", float("nan"))),
                "fixed_count": int(fixed_count),
                "regression_count": int(regression_count),
                "fix_per_hour": float(fixed_count / elapsed_h),
                "regression_rate": float(regression_count / max(1, fixed_count + regression_count)),
            }

        history.append(record)
        if step % 2 == 0 or need_verify:
            safe_json_print(record)

        if step % int(args.save_interval) == 0 or step == int(args.steps):
            payload = {
                "format": "ccl_patch_v1",
                "base_checkpoint": str(base_ckpt),
                "step": int(step),
                "args": vars(args),
                "model_args": model_args,
                "patch_state": lora_patch_state_dict(model),
                "ledger_state": ledger.state_dict(),
                "best_hard_pass_rate": float(best_hard_pass),
                "best_step": int(best_step),
            }
            torch.save(payload, out_dir / "ccl_patch_last.pt")
            if best_step == step:
                torch.save(payload, out_dir / "ccl_patch_best.pt")

    elapsed = float(time.time() - start)
    summary = {
        "finished": True,
        "elapsed_sec": elapsed,
        "base_checkpoint": str(base_ckpt),
        "spec_source": spec_source,
        "steps": int(args.steps),
        "best_hard_pass_rate": float(best_hard_pass),
        "best_step": int(best_step),
        "target_hard_pass": float(args.target_hard_pass),
        "target_hit_step": int(target_hit_step),
        "target_hit_time_sec": float(target_hit_time_sec),
        "counterexample_ledger": str(cex_ledger_path),
        "patch_last": str(out_dir / "ccl_patch_last.pt"),
        "patch_best": str(out_dir / "ccl_patch_best.pt"),
        "history_tail": history[-12:],
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    safe_json_print(summary)


if __name__ == "__main__":
    main()
