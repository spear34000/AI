from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class ByteTokenizer:
    bos_id = 256
    eos_id = 257
    pad_id = 258
    vocab_size = 259

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        b = list(text.encode("utf-8", errors="replace"))
        out: List[int] = []
        if add_bos:
            out.append(self.bos_id)
        out.extend(b)
        if add_eos:
            out.append(self.eos_id)
        return out

    def decode(self, token_ids: List[int]) -> str:
        b = [int(t) for t in token_ids if 0 <= int(t) <= 255]
        # Drop invalid UTF-8 fragments instead of emitting replacement characters.
        return bytes(b).decode("utf-8", errors="ignore")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "type": "byte",
            "vocab_size": int(self.vocab_size),
            "bos_id": int(self.bos_id),
            "eos_id": int(self.eos_id),
            "pad_id": int(self.pad_id),
        }


class SentencePieceTokenizer:
    def __init__(self, model_proto: bytes) -> None:
        try:
            import sentencepiece as spm  # type: ignore
        except Exception as exc:
            raise RuntimeError("sentencepiece is required for tokenizer_type=spm") from exc

        self._model_proto = bytes(model_proto)
        self._sp = spm.SentencePieceProcessor(model_proto=self._model_proto)

        self.vocab_size = int(self._sp.vocab_size())
        self.unk_id = int(self._sp.unk_id())
        self.bos_id = int(self._sp.bos_id())
        self.eos_id = int(self._sp.eos_id())
        self.pad_id = int(self._sp.pad_id())

        if self.bos_id < 0 or self.eos_id < 0 or self.pad_id < 0:
            raise RuntimeError("SPM model must include bos/eos/pad ids")

    @classmethod
    def from_model_file(cls, model_path: str | Path) -> "SentencePieceTokenizer":
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"SPM model not found: {p}")
        return cls(model_proto=p.read_bytes())

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = list(self._sp.encode(str(text or ""), out_type=int))
        out: List[int] = []
        if bool(add_bos):
            out.append(int(self.bos_id))
        out.extend(int(i) for i in ids)
        if bool(add_eos):
            out.append(int(self.eos_id))
        return out

    def decode(self, token_ids: List[int]) -> str:
        ids = [int(i) for i in token_ids if int(i) >= 0 and int(i) < int(self.vocab_size)]
        banned = {int(self.bos_id), int(self.eos_id), int(self.pad_id)}
        if int(self.unk_id) >= 0:
            banned.add(int(self.unk_id))
        ids = [i for i in ids if i not in banned]
        if not ids:
            return ""
        return str(self._sp.decode(ids))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "type": "spm",
            "vocab_size": int(self.vocab_size),
            "unk_id": int(self.unk_id),
            "bos_id": int(self.bos_id),
            "eos_id": int(self.eos_id),
            "pad_id": int(self.pad_id),
            "model_proto": self._model_proto,
        }


def tokenizer_from_state(tok_state: Dict[str, Any] | None) -> ByteTokenizer | SentencePieceTokenizer:
    state = tok_state if isinstance(tok_state, dict) else {}
    t = str(state.get("type", "byte")).strip().lower()
    if t == "spm":
        proto = state.get("model_proto")
        if not isinstance(proto, (bytes, bytearray)):
            raise RuntimeError("SPM tokenizer state missing model_proto bytes")
        return SentencePieceTokenizer(model_proto=bytes(proto))
    return ByteTokenizer()


def build_tokenizer(tokenizer_type: str, tokenizer_model: str) -> ByteTokenizer | SentencePieceTokenizer:
    t = str(tokenizer_type or "byte").strip().lower()
    if t == "spm":
        if not str(tokenizer_model or "").strip():
            raise RuntimeError("--tokenizer_model is required when --tokenizer_type spm")
        return SentencePieceTokenizer.from_model_file(tokenizer_model)
    return ByteTokenizer()


def safe_json_print(payload: Dict) -> None:
    s = json.dumps(payload, ensure_ascii=False)
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        s_safe = s.encode(enc, errors="replace").decode(enc, errors="replace")
        print(s_safe, flush=True)


def dataset_file_fingerprint(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    st = p.stat()
    return {
        "path": str(p),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def tokenizer_fingerprint(tokenizer: Any) -> Dict[str, Any]:
    state = tokenizer.state_dict() if hasattr(tokenizer, "state_dict") else {"type": "byte"}
    t = str(state.get("type", "byte")).strip().lower()
    if t == "spm":
        proto = state.get("model_proto", b"")
        proto_bytes = bytes(proto) if isinstance(proto, (bytes, bytearray)) else b""
        return {
            "type": "spm",
            "vocab_size": int(state.get("vocab_size", 0)),
            "sha1": hashlib.sha1(proto_bytes).hexdigest()[:16],
        }
    return {
        "type": "byte",
        "vocab_size": int(state.get("vocab_size", 259)),
        "sha1": "byte-tokenizer",
    }


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(sum(values) / len(values))
    if len(values) <= 1:
        return mean, 0.0
    var = float(sum((v - mean) * (v - mean) for v in values) / len(values))
    return mean, math.sqrt(max(0.0, var))


def _pick_first_non_empty(row: Dict, keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def format_example(row: Dict) -> str:
    inp = _pick_first_non_empty(row, ["input", "instruction", "prompt", "question", "context"])
    out = _pick_first_non_empty(row, ["output", "response", "answer", "completion", "target"])

    if not inp and not out:
        # Lossless fallback for any unexpected schema.
        raw = json.dumps(row, ensure_ascii=False)
        inp = "Use the following JSON as context and respond helpfully."
        out = raw

    return f"### Instruction\n{inp}\n\n### Response\n{out}\n"


def format_example_split(row: Dict) -> Tuple[str, int]:
    """Return (full_text, char_offset_where_response_starts).

    This allows masking the instruction portion so loss is only computed
    on the response tokens (answer-only loss masking).
    """
    inp = _pick_first_non_empty(row, ["input", "instruction", "prompt", "question", "context"])
    out = _pick_first_non_empty(row, ["output", "response", "answer", "completion", "target"])

    if not inp and not out:
        raw = json.dumps(row, ensure_ascii=False)
        inp = "Use the following JSON as context and respond helpfully."
        out = raw

    prefix = f"### Instruction\n{inp}\n\n### Response\n"
    full = f"{prefix}{out}\n"
    return full, len(prefix)


def tier_to_id(tier: str) -> int:
    t = str(tier).strip().lower()
    if t == "high":
        return 2
    if t == "mid":
        return 1
    return 0


def domain_to_id(segment_or_task: str) -> int:
    s = str(segment_or_task).strip().lower()
    if s == "code":
        return 0
    if s in ("doc", "english"):
        return 1
    if s in ("ko", "korean"):
        return 2
    return 3


HANGUL_RE = re.compile(r"[가-힣]")
LATIN_RE = re.compile(r"[A-Za-z]")


def infer_domain_id(row: Dict) -> int:
    seg = str(row.get("segment_tag", row.get("task_type", ""))).strip().lower()
    inp = str(row.get("input", ""))
    out = str(row.get("output", ""))
    text = f"{inp}\n{out}"

    hangul_n = len(HANGUL_RE.findall(text))
    latin_n = len(LATIN_RE.findall(text))
    code_hint = (
        ("```" in text)
        or ("def " in text)
        or ("class " in text)
        or ("public static void" in text)
        or ("import " in text)
    )

    if hangul_n >= 16 and hangul_n >= int(0.15 * max(1, latin_n)):
        return 2
    if seg == "code" or str(row.get("task_type", "")).strip().lower() == "code" or code_hint:
        return 0
    if seg in ("doc", "english") or str(row.get("task_type", "")).strip().lower() == "english":
        return 1
    if hangul_n > 0:
        return 2
    return 3


def _load_resume_args(resume_from: str | Path) -> Dict:
    path = Path(str(resume_from))
    if not str(resume_from).strip() or not path.exists():
        return {}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    return args if isinstance(args, dict) else {}


def _maybe_override_model_args_from_resume(args: argparse.Namespace) -> None:
    resume = str(getattr(args, "resume_from", "") or "").strip()
    if not resume:
        return
    ck_args = _load_resume_args(resume)
    if not ck_args:
        return

    for k in ("d_model", "n_heads", "n_layers", "seq_len", "mlp_mult"):
        v = ck_args.get(k)
        if v is None:
            continue
        try:
            setattr(args, k, int(v))
        except Exception:
            continue
    # Resume-time dropout should be conservative; keep CLI dropout if explicitly set by user.
    if getattr(args, "dropout", None) is None:
        v = ck_args.get("dropout")
        if v is not None:
            try:
                setattr(args, "dropout", float(v))
            except Exception:
                pass


@dataclass
class RecordRef:
    offset: int
    tier_id: int
    domain_id: int


class JsonlCausalDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: ByteTokenizer,
        seq_len: int,
        split: str,
        val_ratio: float,
        seed: int,
        max_records: int,
        min_chars: int = 40,
        response_loss_only: bool = False,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.split = split
        self.response_loss_only = bool(response_loss_only)
        self._fp = None

        refs: List[RecordRef] = []
        with self.path.open("rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                try:
                    row = json.loads(line.decode("utf-8", errors="replace"))
                    if not isinstance(row, dict):
                        continue
                except json.JSONDecodeError:
                    continue

                text = format_example(row)
                if len(text) < int(min_chars):
                    continue

                tier_id = tier_to_id(str(row.get("_meta_quality_tier", "base")))
                domain_id = infer_domain_id(row)
                refs.append(RecordRef(offset=pos, tier_id=tier_id, domain_id=domain_id))
                if max_records > 0 and len(refs) >= int(max_records):
                    break

        if not refs:
            raise RuntimeError(f"No valid records from {self.path}")

        indices = list(range(len(refs)))
        rnd = random.Random(seed)
        rnd.shuffle(indices)

        cut = int(len(indices) * (1.0 - float(val_ratio)))
        if cut <= 0 or cut >= len(indices):
            cut = max(1, len(indices) - 1)

        if split == "all":
            sel = indices
        else:
            sel = indices[:cut] if split == "train" else indices[cut:]
        if not sel:
            # Keep loaders valid for tiny smoke datasets.
            sel = [indices[0] if split == "train" else indices[-1]]
        self.refs = [refs[i] for i in sel]

    def _ensure_fp(self) -> None:
        if self._fp is None:
            self._fp = self.path.open("rb")

    def __len__(self) -> int:
        return len(self.refs)

    def _read_row(self, idx: int) -> Dict:
        self._ensure_fp()
        assert self._fp is not None
        ref = self.refs[idx]
        self._fp.seek(ref.offset)
        line = self._fp.readline()
        row = json.loads(line.decode("utf-8", errors="replace"))
        if not isinstance(row, dict):
            row = {}
        return row

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ref = self.refs[idx]
        row = self._read_row(idx)

        if self.response_loss_only:
            text, resp_char_offset = format_example_split(row)
        else:
            text = format_example(row)
            resp_char_offset = 0

        tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)

        # Find the token index where the response starts.
        resp_token_offset = 0
        if self.response_loss_only and resp_char_offset > 0:
            prefix_text = text[:resp_char_offset]
            prefix_tokens = self.tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
            resp_token_offset = len(prefix_tokens)

        target_len = self.seq_len + 1
        if len(tokens) >= target_len:
            start = random.randint(0, len(tokens) - target_len) if self.split == "train" else 0
            chunk = tokens[start : start + target_len]
            x = chunk[:-1]
            y = chunk[1:]
            # Adjust resp offset relative to the chunk start.
            resp_token_offset = max(0, resp_token_offset - start)
        else:
            x = tokens[:-1]
            y = tokens[1:]
            pad_n = target_len - 1 - len(x)
            if pad_n > 0:
                x = x + [self.tokenizer.pad_id] * pad_n
                y = y + [-100] * pad_n

        # Mask instruction tokens: set targets to -100 so loss ignores them.
        if self.response_loss_only and resp_token_offset > 0:
            for i in range(min(resp_token_offset, len(y))):
                y[i] = -100

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(ref.tier_id, dtype=torch.long),
            torch.tensor(ref.domain_id, dtype=torch.long),
        )

    def sample_weights(self) -> List[float]:
        # Initial sampler bias before dynamic curriculum scaling in the loss.
        tier_w = {2: 1.7, 1: 1.25, 0: 1.0}
        domain_w = {0: 1.0, 1: 1.3, 2: 1.6, 3: 1.1}
        out = []
        for r in self.refs:
            w = float(tier_w.get(r.tier_id, 1.0)) * float(domain_w.get(r.domain_id, 1.0))
            out.append(w)
        return out

    def signature(self) -> str:
        h = hashlib.sha1()
        h.update(str(self.path).encode("utf-8", errors="ignore"))
        h.update(f"|{self.split}|{len(self.refs)}|".encode("utf-8"))
        for ref in self.refs:
            h.update(f"{int(ref.offset)}:{int(ref.tier_id)}:{int(ref.domain_id)}|".encode("utf-8"))
        return h.hexdigest()[:16]

    def preview_stats(self, max_items: int = 256) -> Dict[str, float | int]:
        n = min(len(self.refs), max(1, int(max_items)))
        char_lens: List[int] = []
        token_lens: List[int] = []
        for idx in range(n):
            row = self._read_row(idx)
            text = format_example(row)
            char_lens.append(len(text))
            token_lens.append(len(self.tokenizer.encode(text, add_bos=True, add_eos=True)))
        char_mean = float(sum(char_lens) / max(1, len(char_lens)))
        tok_mean = float(sum(token_lens) / max(1, len(token_lens)))
        return {
            "sampled_records": int(n),
            "mean_chars": round(char_mean, 2),
            "mean_tokens": round(tok_mean, 2),
        }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear base")
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(max(1, r))
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        for p in self.base.parameters():
            p.requires_grad_(False)
        if self.r > 0:
            self.lora_a = nn.Parameter(torch.empty(self.r, self.in_features))
            self.lora_b = nn.Parameter(torch.zeros(self.out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5.0))
            nn.init.zeros_(self.lora_b)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r <= 0:
            return out
        lora = F.linear(self.dropout(x), self.lora_a)
        lora = F.linear(lora, self.lora_b)
        return out + (lora * self.scaling)


def _resolve_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = [p for p in str(module_name).split(".") if p]
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    if not parts:
        raise RuntimeError("empty module name")
    return parent, parts[-1]


def apply_lora_to_model(
    model: nn.Module,
    *,
    r: int,
    alpha: float,
    dropout: float,
    target_suffixes: List[str],
) -> List[str]:
    replaced: List[str] = []
    for p in model.parameters():
        p.requires_grad_(False)
    module_items = list(model.named_modules())
    for name, module in module_items:
        if not name or not isinstance(module, nn.Linear):
            continue
        if not any(str(name).endswith(suf) for suf in target_suffixes):
            continue
        parent, child_name = _resolve_parent_module(model, name)
        wrapped = LoRALinear(module, r=int(r), alpha=float(alpha), dropout=float(dropout))
        wrapped = wrapped.to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(parent, child_name, wrapped)
        replaced.append(str(name))
    return replaced


def _checkpoint_has_lora_state(model_state: Dict[str, Any]) -> bool:
    for k in model_state.keys():
        ks = str(k)
        if ".lora_a" in ks or ".lora_b" in ks or ".base.weight" in ks or ".base.bias" in ks:
            return True
    return False


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_mult: int,
        dropout: float,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.freeze_prefix_layers = 0
        self.freeze_embeddings = False
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, mlp_mult, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def set_turbo_freeze(self, freeze_prefix_layers: int, freeze_embeddings: bool) -> None:
        max_layers = len(self.blocks)
        self.freeze_prefix_layers = max(0, min(int(freeze_prefix_layers), int(max_layers)))
        self.freeze_embeddings = bool(freeze_embeddings)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seqlen = input_ids.shape
        if seqlen > self.seq_len:
            raise RuntimeError(f"Input length {seqlen} exceeds model seq_len {self.seq_len}")
        pos = torch.arange(0, seqlen, device=input_ids.device, dtype=torch.long).unsqueeze(0)

        if self.training and bool(self.freeze_embeddings):
            with torch.no_grad():
                tok = self.token_emb(input_ids)
                pos_tok = self.pos_emb(pos)
            x = tok + pos_tok
        else:
            x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        for layer_idx, blk in enumerate(self.blocks):
            if self.training and layer_idx < int(self.freeze_prefix_layers):
                with torch.no_grad():
                    x = blk(x)
            elif self.gradient_checkpointing and self.training and x.requires_grad:
                x = checkpoint_utils.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss


def per_sample_ce(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, seqlen, vocab = logits.shape
    token_loss = F.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(bsz, seqlen)
    mask = (targets != -100).float()
    seq_loss = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return seq_loss, seq_loss.mean()


def curriculum_weights(tier_ids: torch.Tensor, progress: float) -> torch.Tensor:
    # 0=base,1=mid,2=high
    base_w = torch.ones_like(tier_ids, dtype=torch.float32)
    mid_w = torch.full_like(base_w, 1.1)
    high_w = torch.full_like(base_w, 1.2)
    static_w = torch.where(tier_ids == 2, high_w, torch.where(tier_ids == 1, mid_w, base_w))

    # Ramp quality emphasis over training steps.
    boost_base = torch.ones_like(base_w)
    boost_mid = torch.full_like(base_w, 1.0 + 0.35 * float(progress))
    boost_high = torch.full_like(base_w, 1.0 + 0.80 * float(progress))
    boost = torch.where(tier_ids == 2, boost_high, torch.where(tier_ids == 1, boost_mid, boost_base))
    return static_w * boost


def domain_curriculum_weights(domain_ids: torch.Tensor, progress: float, ko_focus: float, doc_focus: float) -> torch.Tensor:
    # 0=code,1=doc/en,2=ko,3=other
    base = torch.ones_like(domain_ids, dtype=torch.float32)
    code_w = base
    doc_w = torch.full_like(base, 1.0 + float(doc_focus) * float(progress))
    ko_w = torch.full_like(base, 1.0 + float(ko_focus) * float(progress))
    other_w = torch.full_like(base, 1.0 + 0.15 * float(progress))
    return torch.where(
        domain_ids == 2,
        ko_w,
        torch.where(domain_ids == 1, doc_w, torch.where(domain_ids == 0, code_w, other_w)),
    )


def turbo_active_seq_len(progress: float, full_seq_len: int, min_seq_len: int, warmup_ratio: float) -> int:
    full_len = max(32, int(full_seq_len))
    min_len = max(32, min(int(min_seq_len), full_len))
    ratio = min(max(float(warmup_ratio), 1e-4), 1.0)
    ramp = min(max(float(progress) / ratio, 0.0), 1.0)
    # Sqrt ramp keeps early steps cheap while avoiding an abrupt jump near the end.
    ramp = math.sqrt(ramp)
    raw = int(round(min_len + (full_len - min_len) * ramp))
    aligned = max(32, (raw // 8) * 8)
    return max(min_len, min(full_len, aligned))


def turbo_frozen_prefix_layers(progress: float, n_layers: int, max_frozen_layers: int, warmup_ratio: float) -> int:
    total_layers = max(0, int(n_layers))
    max_frozen = max(0, min(int(max_frozen_layers), total_layers))
    if max_frozen <= 0:
        return 0
    ratio = min(max(float(warmup_ratio), 1e-4), 1.0)
    ramp = min(max(float(progress) / ratio, 0.0), 1.0)
    remain = int(round(max_frozen * (1.0 - ramp)))
    return max(0, min(max_frozen, remain))


def update_ema(ema_model: TinyGPT, model: TinyGPT, decay: float) -> None:
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, p_ema in ema_params.items():
            p = model_params[name]
            p_ema.mul_(decay).add_(p, alpha=(1.0 - decay))

        ema_buffers = dict(ema_model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, b_ema in ema_buffers.items():
            b = model_buffers[name]
            b_ema.copy_(b)


@torch.no_grad()
def evaluate(model: TinyGPT, loader: DataLoader, device: torch.device, max_batches: int) -> Tuple[float, int]:
    model.eval()
    losses = []
    it = iter(loader)
    for _ in range(max_batches):
        try:
            x, y, _, _ = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x, targets=None)
        _, mean_ce = per_sample_ce(logits, y)
        losses.append(float(mean_ce.detach().cpu().item()))
    model.train()
    if not losses:
        return float("nan"), 0
    return float(sum(losses) / len(losses)), int(len(losses))


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    out = logits

    if top_k > 0:
        v, idx = torch.topk(out, k=min(top_k, out.numel()))
        mask = torch.full_like(out, float("-inf"))
        mask[idx] = v
        out = mask

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(out, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        remove = cumprobs > top_p
        remove[0] = False
        sorted_logits[remove] = float("-inf")
        new_logits = torch.full_like(out, float("-inf"))
        new_logits[sorted_indices] = sorted_logits
        out = new_logits

    return out


@torch.no_grad()
def quick_generate(
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 80,
    top_p: float = 0.95,
) -> str:
    model.eval()
    tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)

    for _ in range(int(max_new_tokens)):
        x = torch.tensor([tokens[-model.seq_len :]], dtype=torch.long, device=device)
        logits, _ = model(x, targets=None)
        next_logits = logits[0, -1]

        if temperature <= 0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            next_logits = next_logits / float(temperature)
            next_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        tokens.append(next_id)
        if next_id == tokenizer.eos_id:
            break

    model.train()
    return tokenizer.decode(tokens)


def build_checkpoint(
    model: TinyGPT,
    ema_model: TinyGPT | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    tokenizer: Any,
    args: argparse.Namespace,
    step: int,
    best_val_loss: float,
    use_amp: bool,
) -> Dict:
    tok_state = tokenizer.state_dict() if hasattr(tokenizer, "state_dict") else {"type": "byte"}
    ckpt = {
        "model_state": model.state_dict(),
        "ema_model_state": ema_model.state_dict() if ema_model is not None else None,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if use_amp else {},
        "tokenizer": tok_state,
        "args": vars(args),
        "step": int(step),
        "best_val_loss": float(best_val_loss),
    }
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/slm_mit_unified_v4.jsonl")
    parser.add_argument("--output_dir", default="artifacts_slm_mit_unified_v4")
    parser.add_argument("--max_records", type=int, default=0, help="0 means use all")
    parser.add_argument("--seq_len", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=18)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--save_step_checkpoints", action="store_true")
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--skip_sample", action="store_true")
    parser.add_argument("--no_best_checkpoint", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--mlp_mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--sample_prompt", type=str, default="### Instruction\nSay hello in Korean.\n\n### Response\n")
    parser.add_argument("--resume_from", type=str, default="", help="optional checkpoint path to resume")
    parser.add_argument("--resume_weights_only", action="store_true")
    parser.add_argument(
        "--reset_step_on_resume",
        action="store_true",
        help="with --resume_weights_only, restart scheduler/global step from 0",
    )
    parser.add_argument(
        "--reset_best_on_resume",
        action="store_true",
        help="reset best validation tracking when resuming (useful for new datasets)",
    )
    parser.add_argument("--tokenizer_type", default="byte", choices=["byte", "spm"])
    parser.add_argument("--tokenizer_model", default="", help="path to sentencepiece model when tokenizer_type=spm")
    parser.add_argument("--lora_r", type=int, default=0, help="enable LoRA adapters when > 0")
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="attn.qkv,attn.proj,mlp.0,mlp.2",
        help="comma-separated module suffixes to wrap with LoRA",
    )

    # Innovative objective: EMA teacher consistency + dynamic quality curriculum.
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--consistency_lambda", type=float, default=0.05)
    parser.add_argument("--consistency_temp", type=float, default=1.5)
    parser.add_argument("--consistency_warmup", type=int, default=40)
    parser.add_argument("--eval_with_ema", action="store_true")
    parser.add_argument("--ko_focus", type=float, default=1.0, help="curriculum boost strength for Korean domain")
    parser.add_argument("--doc_focus", type=float, default=0.4, help="curriculum boost strength for doc/english domain")
    parser.add_argument(
        "--plain_sft",
        action="store_true",
        help="disable weighted sampler/domain curriculum and run plain supervised fine-tuning",
    )
    parser.add_argument(
        "--response_loss_only",
        action="store_true",
        help="only compute loss on response tokens, mask instruction prefix",
    )
    parser.add_argument("--turbo_mode", action="store_true", help="enable Turbo4060 schedule for faster convergence")
    parser.add_argument("--compile_model", action="store_true", help="torch.compile the model for faster training")
    parser.add_argument("--turbo_min_seq_len", type=int, default=128, help="minimum sequence length used at training start")
    parser.add_argument(
        "--turbo_seq_warmup_ratio",
        type=float,
        default=0.55,
        help="fraction of total steps used to ramp sequence length to full context",
    )
    parser.add_argument("--turbo_max_frozen_layers", type=int, default=4, help="max number of lower blocks frozen at start")
    parser.add_argument(
        "--turbo_depth_warmup_ratio",
        type=float,
        default=0.35,
        help="fraction of total steps to unfreeze lower blocks back to full depth",
    )
    parser.add_argument(
        "--turbo_freeze_embeddings",
        action="store_true",
        help="freeze token/position embeddings while lower blocks are frozen",
    )

    args = parser.parse_args()

    _maybe_override_model_args_from_resume(args)

    if args.dropout is None:
        args.dropout = 0.1

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise RuntimeError(f"data_path not found: {data_path}")

    tokenizer = None
    resume_probe = str(getattr(args, "resume_from", "") or "").strip()
    if resume_probe:
        p = Path(resume_probe)
        if p.exists():
            try:
                ck_probe = torch.load(p, map_location="cpu", weights_only=False)
                tok_state = ck_probe.get("tokenizer", {})
                if isinstance(tok_state, dict):
                    tokenizer = tokenizer_from_state(tok_state)
            except Exception:
                tokenizer = None
    if tokenizer is None:
        tokenizer = build_tokenizer(tokenizer_type=str(args.tokenizer_type), tokenizer_model=str(args.tokenizer_model))
    _resp_loss = bool(getattr(args, "response_loss_only", False))
    train_ds = JsonlCausalDataset(
        path=data_path,
        tokenizer=tokenizer,
        seq_len=int(args.seq_len),
        split="train",
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        max_records=int(args.max_records),
        response_loss_only=_resp_loss,
    )
    val_ds = JsonlCausalDataset(
        path=data_path,
        tokenizer=tokenizer,
        seq_len=int(args.seq_len),
        split="val",
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        max_records=int(args.max_records),
    )

    grad_accum_steps = max(1, int(args.grad_accum_steps))
    if bool(args.plain_sft):
        train_loader = DataLoader(
            train_ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=(device.type == "cuda"),
        )
    else:
        train_weights = train_ds.sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(train_weights, dtype=torch.double),
            num_samples=int(args.steps) * int(args.batch_size) * grad_accum_steps,
            replacement=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=int(args.batch_size),
            sampler=train_sampler,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=(device.type == "cuda"),
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
    )
    val_sig = val_ds.signature()
    val_preview = val_ds.preview_stats(max_items=256)
    data_fp = dataset_file_fingerprint(data_path)
    tok_fp = tokenizer_fingerprint(tokenizer)

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        seq_len=int(args.seq_len),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        mlp_mult=int(args.mlp_mult),
        dropout=float(args.dropout),
        gradient_checkpointing=bool(args.activation_checkpointing),
    ).to(device)

    if bool(getattr(args, "compile_model", False)) and hasattr(torch, "compile"):
        print(json.dumps({"torch_compile": True, "backend": "inductor"}))
        model = torch.compile(model, backend="inductor")

    lora_enabled = int(args.lora_r) > 0
    lora_targets = [s.strip() for s in str(args.lora_targets).split(",") if str(s).strip()]
    resume_from = str(args.resume_from).strip()
    resume_ckpt = None
    resume_has_lora = False
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise RuntimeError(f"resume checkpoint not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model_state = resume_ckpt.get("model_state", {})
        if not isinstance(model_state, dict):
            raise RuntimeError("resume checkpoint missing model_state")
        resume_has_lora = _checkpoint_has_lora_state(model_state)
        if not (lora_enabled and resume_has_lora):
            model.load_state_dict(model_state)

    lora_replaced: List[str] = []
    if lora_enabled:
        lora_replaced = apply_lora_to_model(
            model,
            r=int(args.lora_r),
            alpha=float(args.lora_alpha),
            dropout=float(args.lora_dropout),
            target_suffixes=lora_targets,
        )
        if resume_ckpt is not None and resume_has_lora:
            model.load_state_dict(resume_ckpt["model_state"], strict=False)

    ema_model: TinyGPT | None = None
    if float(args.ema_decay) > 0.0 and not lora_enabled:
        ema_model = TinyGPT(
            vocab_size=tokenizer.vocab_size,
            seq_len=int(args.seq_len),
            d_model=int(args.d_model),
            n_heads=int(args.n_heads),
            n_layers=int(args.n_layers),
            mlp_mult=int(args.mlp_mult),
            dropout=float(args.dropout),
            gradient_checkpointing=False,
        ).to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters found")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(args.lr),
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=float(args.weight_decay),
    )

    def lr_lambda(step: int) -> float:
        if step < int(args.warmup_steps):
            return float(step + 1) / float(max(1, int(args.warmup_steps)))
        t = (step - int(args.warmup_steps)) / float(max(1, int(args.steps) - int(args.warmup_steps)))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    train_iter = iter(train_loader)
    best_val = float("inf")
    best_step = 0
    start_step = 0
    history = []
    start_time = time.time()
    train_loss_window: deque[float] = deque(maxlen=100)
    lm_loss_window: deque[float] = deque(maxlen=100)
    raw_ce_window: deque[float] = deque(maxlen=100)

    if resume_ckpt is not None:
        ckpt = resume_ckpt
        if lora_enabled and resume_has_lora:
            model.load_state_dict(ckpt["model_state"], strict=False)
        elif lora_enabled and not resume_has_lora:
            pass
        else:
            model.load_state_dict(ckpt["model_state"])
        if ema_model is not None and ckpt.get("ema_model_state"):
            ema_model.load_state_dict(ckpt["ema_model_state"])

        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val_loss", best_val))

        if bool(args.resume_weights_only):
            if bool(args.reset_step_on_resume):
                start_step = 0
            base_lr = float(args.lr)
            current_lr = base_lr * float(lr_lambda(start_step))
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            scheduler.last_epoch = int(start_step)
        else:
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            if use_amp and "scaler_state" in ckpt:
                scaler_state = ckpt.get("scaler_state")
                if isinstance(scaler_state, dict) and scaler_state:
                    scaler.load_state_dict(scaler_state)

        if bool(args.reset_best_on_resume):
            best_val = float("inf")
            best_step = int(start_step)

    if start_step >= int(args.steps):
        raise RuntimeError(
            f"resume step {start_step} is already >= target steps {int(args.steps)}; increase --steps or remove --resume_from"
        )

    param_count = int(sum(p.numel() for p in model.parameters()))
    trainable_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    effective_tokens_per_step = int(args.batch_size) * grad_accum_steps * int(args.seq_len)

    safe_json_print(
        {
            "device": str(device),
            "train_records": len(train_ds),
            "val_records": len(val_ds),
            "seq_len": int(args.seq_len),
            "batch_size": int(args.batch_size),
            "grad_accum_steps": int(grad_accum_steps),
            "effective_tokens_per_step": int(effective_tokens_per_step),
            "steps": int(args.steps),
            "validation": {
                "fixed_eval": True,
                "shuffle": False,
                "eval_batches": int(args.eval_batches),
                "signature": val_sig,
                "preview": val_preview,
                "seed": int(args.seed),
                "dataset": data_fp,
            },
            "tokenizer": tok_fp,
            "model": {
                "d_model": int(args.d_model),
                "n_heads": int(args.n_heads),
                "n_layers": int(args.n_layers),
                "params_total": int(param_count),
                "params_trainable": int(trainable_count),
                "lora_enabled": bool(lora_enabled),
                "lora_r": int(args.lora_r),
                "lora_replaced": lora_replaced,
            },
            "innovation": {
                "dynamic_curriculum": not bool(args.plain_sft),
                "domain_balance_curriculum": not bool(args.plain_sft),
                "ema_self_distill": bool(ema_model is not None and float(args.consistency_lambda) > 0.0),
                "activation_checkpointing": bool(args.activation_checkpointing),
                "resume_weights_only": bool(args.resume_weights_only),
                "plain_sft": bool(args.plain_sft),
                "turbo_mode": bool(args.turbo_mode),
                "turbo_min_seq_len": int(args.turbo_min_seq_len),
                "turbo_seq_warmup_ratio": float(args.turbo_seq_warmup_ratio),
                "turbo_max_frozen_layers": int(args.turbo_max_frozen_layers),
                "turbo_depth_warmup_ratio": float(args.turbo_depth_warmup_ratio),
                "turbo_freeze_embeddings": bool(args.turbo_freeze_embeddings),
            },
            "start_step": int(start_step),
        }
    )

    for step in range(start_step + 1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        progress = float(step) / float(max(1, int(args.steps)))
        active_seq_len = int(args.seq_len)
        frozen_prefix_layers = 0
        freeze_embeddings = False
        if bool(args.turbo_mode):
            active_seq_len = turbo_active_seq_len(
                progress=progress,
                full_seq_len=int(args.seq_len),
                min_seq_len=int(args.turbo_min_seq_len),
                warmup_ratio=float(args.turbo_seq_warmup_ratio),
            )
            frozen_prefix_layers = turbo_frozen_prefix_layers(
                progress=progress,
                n_layers=int(args.n_layers),
                max_frozen_layers=int(args.turbo_max_frozen_layers),
                warmup_ratio=float(args.turbo_depth_warmup_ratio),
            )
            freeze_embeddings = bool(args.turbo_freeze_embeddings and frozen_prefix_layers > 0)
        model.set_turbo_freeze(
            freeze_prefix_layers=int(frozen_prefix_layers),
            freeze_embeddings=bool(freeze_embeddings),
        )

        train_loss_sum = 0.0
        lm_loss_sum = 0.0
        raw_ce_sum = 0.0
        consistency_sum = 0.0

        for _ in range(grad_accum_steps):
            try:
                x, y, tier_ids, domain_ids = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y, tier_ids, domain_ids = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            tier_ids = tier_ids.to(device, non_blocking=True)
            domain_ids = domain_ids.to(device, non_blocking=True)
            if active_seq_len < x.size(1):
                x = x[:, :active_seq_len]
                y = y[:, :active_seq_len]

            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                logits, _ = model(x, targets=None)
                seq_loss, raw_ce = per_sample_ce(logits, y)
                if bool(args.plain_sft):
                    lm_loss = raw_ce
                else:
                    tier_w = curriculum_weights(tier_ids, progress=progress)
                    domain_w = domain_curriculum_weights(
                        domain_ids,
                        progress=progress,
                        ko_focus=float(args.ko_focus),
                        doc_focus=float(args.doc_focus),
                    )
                    sample_w = tier_w * domain_w
                    lm_loss = (seq_loss * sample_w).mean()

                consistency_loss = torch.zeros((), device=device)
                use_consistency = (
                    ema_model is not None
                    and float(args.consistency_lambda) > 0.0
                    and step >= int(args.consistency_warmup)
                )
                if use_consistency:
                    assert ema_model is not None
                    with torch.no_grad():
                        ema_logits, _ = ema_model(x, targets=None)
                    t = float(args.consistency_temp)
                    student_logp = F.log_softmax(logits / t, dim=-1)
                    teacher_p = F.softmax(ema_logits / t, dim=-1)
                    kl_tok = F.kl_div(student_logp, teacher_p, reduction="none").sum(dim=-1)
                    mask = (y != -100).float()
                    consistency_loss = (kl_tok * mask).sum() / mask.sum().clamp_min(1.0)
                    consistency_loss = consistency_loss * (t * t)

                loss = lm_loss + float(args.consistency_lambda) * consistency_loss

            scaler.scale(loss / float(grad_accum_steps)).backward()
            train_loss_sum += float(loss.detach().cpu().item())
            lm_loss_sum += float(lm_loss.detach().cpu().item())
            raw_ce_sum += float(raw_ce.detach().cpu().item())
            consistency_sum += float(consistency_loss.detach().cpu().item())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if ema_model is not None:
            update_ema(ema_model, model, decay=float(args.ema_decay))

        record = {
            "step": step,
            "train_loss": float(train_loss_sum / float(grad_accum_steps)),
            "lm_loss": float(lm_loss_sum / float(grad_accum_steps)),
            "raw_ce": float(raw_ce_sum / float(grad_accum_steps)),
            "consistency": float(consistency_sum / float(grad_accum_steps)),
            "curriculum_progress": progress,
            "ko_focus": float(args.ko_focus),
            "doc_focus": float(args.doc_focus),
            "active_seq_len": int(active_seq_len),
            "frozen_prefix_layers": int(frozen_prefix_layers),
            "freeze_embeddings": bool(freeze_embeddings),
            "effective_tokens_step": int(int(args.batch_size) * grad_accum_steps * int(active_seq_len)),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        train_loss_window.append(float(record["train_loss"]))
        lm_loss_window.append(float(record["lm_loss"]))
        raw_ce_window.append(float(record["raw_ce"]))
        train_mean, train_std = _mean_std(list(train_loss_window))
        lm_mean, _ = _mean_std(list(lm_loss_window))
        raw_mean, _ = _mean_std(list(raw_ce_window))
        record["train_loss_ma100"] = float(train_mean)
        record["train_loss_std100"] = float(train_std)
        record["lm_loss_ma100"] = float(lm_mean)
        record["raw_ce_ma100"] = float(raw_mean)

        if step == 1 or step % int(args.eval_interval) == 0 or step == int(args.steps):
            eval_model = ema_model if bool(args.eval_with_ema) and ema_model is not None else model
            val_loss, val_batches_used = evaluate(eval_model, val_loader, device=device, max_batches=int(args.eval_batches))
            record["val_loss"] = float(val_loss)
            record["val_batches_used"] = int(val_batches_used)
            record["val_signature"] = val_sig
            record["tokenizer_signature"] = str(tok_fp.get("sha1", ""))

            if not math.isnan(val_loss) and val_loss < best_val:
                best_val = float(val_loss)
                best_step = int(step)
                if not bool(args.no_best_checkpoint):
                    ckpt_best = build_checkpoint(
                        model=model,
                        ema_model=ema_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        tokenizer=tokenizer,
                        args=args,
                        step=step,
                        best_val_loss=best_val,
                        use_amp=use_amp,
                    )
                    torch.save(ckpt_best, out_dir / "slm_best.pt")

            if not bool(args.skip_sample):
                sample_model = ema_model if ema_model is not None else model
                sample = quick_generate(
                    model=sample_model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=str(args.sample_prompt),
                    max_new_tokens=80,
                    temperature=0.7,
                    top_k=80,
                    top_p=0.95,
                )
                record["sample"] = sample[:220]

        history.append(record)
        if step % 10 == 0 or "val_loss" in record:
            safe_json_print(record)

        if step % int(args.save_interval) == 0 or step == int(args.steps):
            ckpt = build_checkpoint(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                tokenizer=tokenizer,
                args=args,
                step=step,
                best_val_loss=best_val,
                use_amp=use_amp,
            )
            if bool(args.save_step_checkpoints):
                torch.save(ckpt, out_dir / f"slm_step_{step}.pt")
            torch.save(ckpt, out_dir / "slm_last.pt")

    elapsed = time.time() - start_time
    summary = {
        "finished": True,
        "elapsed_sec": elapsed,
        "best_val_loss": best_val,
        "best_step": best_step,
        "last_checkpoint": str(out_dir / "slm_last.pt"),
        "best_checkpoint": str(out_dir / "slm_best.pt"),
        "history_tail": history[-8:],
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    safe_json_print(summary)


if __name__ == "__main__":
    main()
