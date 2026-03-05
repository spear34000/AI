from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_slm import (
    ByteTokenizer,
    TinyGPT,
    apply_lora_to_model,
    tokenizer_from_state,
    top_k_top_p_filtering,
)

ASSISTANT_MODEL_NAME = "spear1.0"
DEFAULT_MEMORY_PATH = Path("data/session_memory_v1.json")
DEFAULT_USER_HANDLE = "user"
DEFAULT_KO_PROFILE_PATH = Path("data/ko_syllable_profile_v1.json")
DEFAULT_TOOL_CACHE_PATH = Path("data/tool_knowledge_cache_v3_clean.jsonl")
SYSTEM_PROMPT_PRESETS: Dict[str, str] = {
    "none": "",
    "min_qa_ko": (
        "당신은 한국어 질의응답 모델이다. 질문에 직접 답한다. "
        "질문과 무관한 자기소개, 역할극, 메타서사를 출력하지 않는다."
    ),
}


def load_ko_common_syllables(path: Path, min_count: int = 3) -> Set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    if not isinstance(raw, dict):
        return set()
    rows = raw.get("syllables", [])
    if not isinstance(rows, list):
        return set()
    out: Set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        ch = str(row.get("ch", "")).strip()
        cnt = row.get("count", 0)
        try:
            c = int(cnt)
        except Exception:
            c = 0
        if len(ch) != 1:
            continue
        if not re.match(r"[가-힣]", ch):
            continue
        if c >= int(min_count):
            out.add(ch)
    return out


KO_COMMON_SYLLABLES: Set[str] = load_ko_common_syllables(DEFAULT_KO_PROFILE_PATH, min_count=3)
DEF_QUERY_HINT_RE = re.compile(r"(이란\??$|란\??$|무엇|뭐야|정의|설명해|알려줘|소개해|간단히|쉽게|짧게|핵심만|한\s*줄로|한줄로)")
LOGIC_QUERY_HINT_RE = re.compile(r"(모든\s+\S+\s+는\s+\S+|일부\s+\S+\s+는\s+\S+|거짓말쟁이|진실만\s+말|논리|추론)")
INTRO_QUERY_HINT_RE = re.compile(r"(자기소개|소개해|너는\s*누구|네?\s*이름|모델\s*이름)", re.IGNORECASE)
INTRO_ANSWER_HINT_RE = re.compile(r"(안녕하세요|반갑습니다|저는\s+.+(ai|어시스턴트|도우미|모델))", re.IGNORECASE)
UNKNOWN_ANSWER_HINT_RE = re.compile(r"(모르|알 수 없|확실하지|정보가 부족|판단하기 어렵)")
DEF_DRIFT_HINT_RE = re.compile(r"(따라서|결론|일부\s+[A-Z가-힣]|거짓말쟁이|진실만\s+말|질문만으로|통화에서만)")
MSG_SERVICE_HINT_RE = re.compile(r"(모바일\s*메신저|단체\s*채팅|1:1\s*대화|오픈채팅)")
REPEAT_RE = re.compile(r"(.)\1{8,}")


def safe_print(text: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        fixed = text.encode(enc, errors="replace").decode(enc, errors="replace")
        print(fixed, flush=True)


def resolve_device(device_name: str) -> torch.device:
    name = str(device_name).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def _read_checkpoint(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    return torch.load(p, map_location="cpu", weights_only=False)


def _build_model_from_ckpt(ckpt: Dict[str, Any], device: torch.device, use_ema: bool) -> Tuple[TinyGPT, Any]:
    args = ckpt.get("args", {}) or {}
    tok_state = ckpt.get("tokenizer", {})
    tokenizer = tokenizer_from_state(tok_state) if isinstance(tok_state, dict) else ByteTokenizer()
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

    state = ckpt.get("model_state")
    if bool(use_ema) and isinstance(ckpt.get("ema_model_state"), dict):
        state = ckpt.get("ema_model_state")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint missing model_state")

    lora_enabled = int(args.get("lora_r", 0)) > 0
    if lora_enabled:
        lora_targets = [s.strip() for s in str(args.get("lora_targets", "attn.qkv,attn.proj,mlp.0,mlp.2")).split(",") if str(s).strip()]
        apply_lora_to_model(
            model,
            r=int(args.get("lora_r", 0)),
            alpha=float(args.get("lora_alpha", 16.0)),
            dropout=float(args.get("lora_dropout", 0.05)),
            target_suffixes=lora_targets,
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device)
    model.eval()
    return model, tokenizer


def apply_dynamic_int8_quantization(model: TinyGPT) -> TinyGPT:
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def build_model(
    checkpoint: str | Path,
    device: torch.device,
    use_ema: bool = False,
    quantize_int8: bool = False,
) -> Tuple[TinyGPT, Any]:
    ckpt = _read_checkpoint(checkpoint)
    model, tokenizer = _build_model_from_ckpt(ckpt=ckpt, device=device, use_ema=bool(use_ema))
    if bool(quantize_int8) and device.type == "cpu":
        model = apply_dynamic_int8_quantization(model)
    return model, tokenizer


def apply_repetition_penalty(logits: torch.Tensor, token_ids: List[int], penalty: float) -> torch.Tensor:
    if float(penalty) <= 1.0 or not token_ids:
        return logits
    out = logits.clone()
    for idx in set(int(t) for t in token_ids):
        if idx < 0 or idx >= out.numel():
            continue
        val = out[idx]
        out[idx] = val * float(penalty) if val < 0 else val / float(penalty)
    return out


def apply_no_repeat_ngram_block(logits: torch.Tensor, token_ids: List[int], ngram_size: int = 3) -> torch.Tensor:
    n = int(ngram_size)
    if n <= 1 or len(token_ids) < (n - 1):
        return logits

    prefix = tuple(int(t) for t in token_ids[-(n - 1) :])
    banned: Set[int] = set()
    lim = len(token_ids) - n + 1
    for i in range(max(0, lim)):
        if tuple(int(t) for t in token_ids[i : i + n - 1]) == prefix:
            banned.add(int(token_ids[i + n - 1]))

    if not banned:
        return logits

    out = logits.clone()
    for idx in banned:
        if 0 <= idx < out.numel():
            out[idx] = -float("inf")
    return out


def suppress_token_id(logits: torch.Tensor, token_id: int | None) -> torch.Tensor:
    if token_id is None:
        return logits
    tid = int(token_id)
    if tid < 0 or tid >= logits.numel():
        return logits
    out = logits.clone()
    out[tid] = -float("inf")
    return out


def apply_ko_fluency_guard(
    logits: torch.Tensor,
    token_ids: List[int],
    enable_guard: bool,
    guard_topk: int,
    rare_penalty: float,
    latin_penalty: float,
) -> torch.Tensor:
    if not bool(enable_guard):
        return logits

    byte_hist = [int(t) for t in token_ids if 0 <= int(t) <= 255]
    if not byte_hist:
        return logits

    out = logits.clone()
    k = max(8, min(int(guard_topk), int(out.numel())))
    _, idxs = torch.topk(out, k=k)

    tail = byte_hist[-48:]
    base_decoded = bytes(tail).decode("utf-8", errors="ignore")
    base_len = len(base_decoded)

    for idx in idxs.tolist():
        if idx < 0 or idx > 255:
            continue
        probe = bytes(tail + [int(idx)])
        probe_decoded = probe.decode("utf-8", errors="ignore")
        if len(probe_decoded) <= base_len:
            # No complete UTF-8 character emitted yet; skip this step.
            continue
        ch = probe_decoded[-1]
        if re.match(r"[가-힣]", ch):
            if KO_COMMON_SYLLABLES and ch not in KO_COMMON_SYLLABLES:
                out[idx] = out[idx] - float(rare_penalty)
            continue
        if re.match(r"[A-Za-z]", ch):
            out[idx] = out[idx] - float(latin_penalty)
    return out


@torch.no_grad()
def generate_text(
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 180,
    max_input_tokens: int = 39768,
    max_output_tokens: int = 38000,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.12,
    min_new_tokens: int = 0,
    enable_ko_guard: bool = False,
    ko_guard_topk: int = 96,
    ko_guard_rare_penalty: float = 0.9,
    ko_guard_latin_penalty: float = 0.25,
    sample_seed: int | None = None,
) -> str:
    model.eval()
    tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    start_len = len(tokens)
    input_cap = max(64, int(max_input_tokens))
    if len(tokens) > input_cap:
        tokens = tokens[-input_cap:]
    seq_len = int(getattr(model, "seq_len", 384))
    output_cap = max(1, min(int(max_new_tokens), int(max_output_tokens)))

    gen: torch.Generator | None = None
    if sample_seed is not None and int(sample_seed) >= 0:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(sample_seed))

    unk_id = getattr(tokenizer, "unk_id", None)

    for _ in range(output_cap):
        x = torch.tensor([tokens[-seq_len:]], dtype=torch.long, device=device)
        logits, _ = model(x, targets=None)
        next_logits = logits[0, -1]
        next_logits = suppress_token_id(next_logits, unk_id)
        # Apply anti-repetition only to generated continuation, not the prompt.
        generated_ids = tokens[int(start_len) :]
        next_logits = apply_repetition_penalty(next_logits, generated_ids, penalty=float(repetition_penalty))
        next_logits = apply_no_repeat_ngram_block(next_logits, generated_ids, ngram_size=4)
        next_logits = apply_ko_fluency_guard(
            logits=next_logits,
            token_ids=tokens,
            enable_guard=bool(enable_ko_guard),
            guard_topk=int(ko_guard_topk),
            rare_penalty=float(ko_guard_rare_penalty),
            latin_penalty=float(ko_guard_latin_penalty),
        )

        generated_n = len(tokens) - int(start_len)
        if generated_n < int(min_new_tokens) and 0 <= int(tokenizer.eos_id) < next_logits.numel():
            next_logits = next_logits.clone()
            next_logits[int(tokenizer.eos_id)] = -float("inf")

        if float(temperature) <= 0.0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            next_logits = next_logits / float(temperature)
            next_logits = top_k_top_p_filtering(next_logits, top_k=int(top_k), top_p=float(top_p))
            probs = torch.softmax(next_logits, dim=-1)
            if gen is None:
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_id = int(torch.multinomial(probs, num_samples=1, generator=gen).item())

        tokens.append(next_id)
        if next_id == tokenizer.eos_id:
            break

    return tokenizer.decode(tokens)


def extract_response(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt) :].strip()
    marker = "### Response"
    idx = full_text.rfind(marker)
    if idx >= 0:
        return full_text[idx + len(marker) :].lstrip(": \n\t").strip()
    return full_text.strip()


def is_code_request(text: str) -> bool:
    t = str(text or "").strip()
    low = t.lower()
    hints = [
        "```",
        "def ",
        "class ",
        "import ",
        "function ",
        "python",
        "javascript",
        "typescript",
        "java",
        "sql",
        "api",
        "debug",
        "compile",
        "runtime",
        "code",
        "코드",
    ]
    if any(h in low for h in hints):
        return True
    if re.search(r"[{}();\[\]<>_=]", t) and re.search(r"[A-Za-z]{2,}", t):
        return True
    return False


def looks_like_solver_query(text: str) -> bool:
    q = str(text or "").strip().lower()
    if not q:
        return False
    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", q):
        return True
    if "=" in q and re.search(r"\d", q):
        return True
    return any(k in q for k in ["calculate", "equation", "math", "계산"])


def retrieval_tokens(text: str) -> Set[str]:
    words = re.findall(r"[A-Za-z0-9\uac00-\ud7a3]{2,}", str(text or "").lower())
    stop = {"the", "and", "for", "with", "this", "that", "please", "about"}
    return set(w for w in words if w not in stop)


def normalize_ko_token(token: str) -> str:
    t = str(token or "").strip().lower()
    if not t:
        return ""
    suffixes = [
        "인가요",
        "인가",
        "이란",
        "란",
        "입니다",
        "이야",
        "야",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "에",
        "로",
        "도",
        "만",
    ]
    for s in suffixes:
        if len(t) > len(s) + 1 and t.endswith(s):
            t = t[: -len(s)]
            break
    return t


def keyword_terms(text: str) -> Set[str]:
    fillers = {
        "한국어",
        "한국어로",
        "간단히",
        "짧게",
        "한줄",
        "한",
        "줄",
        "설명",
        "설명해줘",
        "알려줘",
        "답해줘",
        "부탁해",
        "부탁",
        "뭐야",
        "무엇",
        "이란",
    }
    out: Set[str] = set()
    for tok in retrieval_tokens(text):
        n = normalize_ko_token(tok)
        if len(n) < 2:
            continue
        if n in fillers:
            continue
        out.add(n)
    return out


def extract_definition_term(text: str) -> str:
    s = str(text or "").strip()
    m = re.search(r"(.+?)(이란\??$|란\??$|무엇|뭐야|정의|설명해|알려줘|소개해|간단히|쉽게|짧게|핵심만|한\s*줄로|한줄로)", s)
    if m:
        s = str(m.group(1)).strip()
    toks = re.findall(r"[A-Za-z0-9\uac00-\ud7a3]{2,}", s)
    if not toks:
        return ""
    return normalize_ko_token(toks[-1])


def levenshtein_distance(a: str, b: str) -> int:
    x = str(a or "")
    y = str(b or "")
    if x == y:
        return 0
    if not x:
        return len(y)
    if not y:
        return len(x)

    prev = list(range(len(y) + 1))
    for i, cx in enumerate(x, start=1):
        cur = [i]
        for j, cy in enumerate(y, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            repl = prev[j - 1] + (0 if cx == cy else 1)
            cur.append(min(ins, dele, repl))
        prev = cur
    return int(prev[-1])


def best_term_similarity(term: str, text: str) -> float:
    t = str(term or "").strip().lower()
    if not t:
        return 0.0
    toks = re.findall(r"[A-Za-z0-9\uac00-\ud7a3]{2,}", str(text or "").lower())
    if not toks:
        return 0.0
    best = 0.0
    for tok in toks:
        d = levenshtein_distance(t, tok)
        sim = 1.0 - float(d) / float(max(len(t), len(tok), 1))
        if sim > best:
            best = sim
    return float(max(0.0, min(1.0, best)))


def has_final_consonant(ch: str) -> bool:
    c = str(ch or "")
    if not c:
        return False
    cp = ord(c[-1])
    if cp < 0xAC00 or cp > 0xD7A3:
        return False
    return ((cp - 0xAC00) % 28) != 0


def definition_anchor_prefix(term: str) -> str:
    t = str(term or "").strip()
    if not t:
        return ""
    if re.search(r"[가-힣]$", t):
        particle = "은" if has_final_consonant(t[-1]) else "는"
        return f"{t}{particle} "
    return f"{t} is "


def term_seems_messenger(term: str) -> bool:
    t = str(term or "").strip().lower()
    if not t:
        return False
    hints = ("톡", "메신저", "채팅", "카카오", "chat", "messenger")
    return any(h in t for h in hints)


def build_uncertain_response(prompt: str, def_term: str = "") -> str:
    term = str(def_term or "").strip()
    if term:
        p = "은" if has_final_consonant(term[-1]) else "는"
        return f"{term}{p} 제가 아는 범위에서 확실하지 않아 모르겠습니다."
    _ = prompt
    return "제가 아는 범위에서 확실하지 않아 모르겠습니다."


def severe_gibberish_signal(prompt: str, response: str, common_syllables: Set[str]) -> bool:
    q = str(prompt or "").strip()
    r = str(response or "").strip()
    if not r:
        return True
    if REPEAT_RE.search(r):
        return True

    toks = re.findall(r"[A-Za-z0-9\uac00-\ud7a3]{2,}", r.lower())
    if len(toks) >= 18:
        diversity = len(set(toks)) / float(max(1, len(toks)))
        if diversity < 0.26:
            return True
    if len(toks) >= 10:
        diversity = len(set(toks)) / float(max(1, len(toks)))
        short_ratio = sum(1 for t in toks if len(t) <= 2) / float(max(1, len(toks)))
        if diversity < 0.52 and short_ratio >= 0.55:
            return True
    eng_toks = re.findall(r"[A-Za-z]+", r.lower())
    if len(eng_toks) >= 8:
        avg_len = sum(len(t) for t in eng_toks) / float(max(1, len(eng_toks)))
        diversity = len(set(eng_toks)) / float(max(1, len(eng_toks)))
        single_ratio = sum(1 for t in eng_toks if len(t) == 1) / float(max(1, len(eng_toks)))
        if avg_len <= 2.4 and (single_ratio >= 0.35 or diversity <= 0.45):
            return True

    q_has_hangul = bool(re.search(r"[가-힣]", q))
    if q_has_hangul:
        hangul_n = len(re.findall(r"[가-힣]", r))
        latin_n = len(re.findall(r"[A-Za-z]", r))
        if hangul_n < 2 and latin_n >= 18:
            return True
        if hangul_n >= 10 and rare_syllable_ratio(r, common_syllables) >= 0.16:
            return True
    return False


def has_contiguous_repeated_hangul_chunk(
    text: str,
    min_chunk: int = 2,
    max_chunk: int = 3,
    min_repeats: int = 1,
) -> bool:
    src = "".join(re.findall(r"[가-힣]", str(text or "")))
    if not src:
        return False
    lo = max(1, int(min_chunk))
    hi = max(lo, int(max_chunk))
    need = max(1, int(min_repeats)) + 1
    for n in range(lo, hi + 1):
        if len(src) < n * need:
            continue
        lim = len(src) - n * need + 1
        for i in range(max(0, lim)):
            chunk = src[i : i + n]
            if len(set(chunk)) == 1:
                continue
            rep = 1
            j = i + n
            while (j + n) <= len(src) and src[j : j + n] == chunk:
                rep += 1
                j += n
            if rep >= need:
                return True
    return False


def definition_noise_signal(response: str) -> bool:
    r = str(response or "").strip()
    if not r:
        return True
    if has_contiguous_repeated_hangul_chunk(r, min_chunk=2, max_chunk=3, min_repeats=1):
        return True
    toks = re.findall(r"[A-Za-z0-9\uac00-\ud7a3]{2,}", r.lower())
    if len(toks) >= 14:
        diversity = len(set(toks)) / float(max(1, len(toks)))
        if diversity < 0.30:
            return True
    return False


def normalize_zero_shot_mode(mode: str) -> str:
    m = str(mode or "balanced").strip().lower()
    if m in {"off", "balanced", "strict"}:
        return m
    return "balanced"


def normalize_system_preset(name: str) -> str:
    key = str(name or "none").strip().lower()
    if key in SYSTEM_PROMPT_PRESETS:
        return key
    return "none"


def resolve_system_prompt(user_system_prompt: str, system_preset: str) -> str:
    explicit = str(user_system_prompt or "").strip()
    if explicit:
        return explicit
    preset_key = normalize_system_preset(system_preset)
    return str(SYSTEM_PROMPT_PRESETS.get(preset_key, "")).strip()


def _safe_eval_numeric_expr(expr: str) -> float | None:
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return None

    allowed_bin = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
    allowed_unary = (ast.UAdd, ast.USub)

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("non numeric const")
        if isinstance(node, ast.Num):  # py<3.8 compatibility
            return float(node.n)
        if isinstance(node, ast.BinOp) and isinstance(node.op, allowed_bin):
            l = _eval(node.left)
            r = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.Div):
                if abs(r) < 1e-12:
                    raise ValueError("div0")
                return l / r
            if isinstance(node.op, ast.Pow):
                return l**r
            if isinstance(node.op, ast.Mod):
                if abs(r) < 1e-12:
                    raise ValueError("mod0")
                return l % r
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, allowed_unary):
            v = _eval(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        raise ValueError("bad node")

    try:
        return float(_eval(tree))
    except Exception:
        return None


def _format_number(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s


def _score_continuation(
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    device: torch.device,
    prompt_text: str,
    completion_text: str,
    max_input_tokens: int,
) -> float:
    prompt_ids = tokenizer.encode(str(prompt_text or ""), add_bos=True, add_eos=False)
    completion_ids = tokenizer.encode(str(completion_text or ""), add_bos=False, add_eos=False)
    if not completion_ids:
        return -1e9

    seq_cap = int(getattr(model, "seq_len", 384))
    input_cap = max(64, int(max_input_tokens))
    cap = max(16, min(seq_cap, input_cap))
    keep_prompt = max(1, cap - len(completion_ids))
    if len(prompt_ids) > keep_prompt:
        prompt_ids = prompt_ids[-keep_prompt:]

    joined = prompt_ids + completion_ids
    if len(joined) < 2:
        return -1e9
    if len(joined) > cap:
        joined = joined[-cap:]
        if len(joined) <= len(completion_ids):
            return -1e9

    x = torch.tensor([joined[:-1]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x, targets=None)
    log_probs = torch.log_softmax(logits[0], dim=-1)

    comp_start = len(joined) - len(completion_ids)
    score = 0.0
    for j, tok_id in enumerate(completion_ids):
        pos = comp_start - 1 + j
        if pos < 0 or pos >= log_probs.size(0):
            return -1e9
        score += float(log_probs[pos, int(tok_id)].item())
    return float(score / float(max(1, len(completion_ids))))


def parse_mcq_options(prompt: str) -> Dict[str, str]:
    src = str(prompt or "").strip()
    if not src:
        return {}
    keyword_hint = bool(
        re.search(r"(선택지|보기|객관식|고르시오|고르세요|choices|options|available choices|multiple choice)", src, re.IGNORECASE)
    )
    out: Dict[str, str] = {}
    normalized = (
        src.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("①", "1. ")
        .replace("②", "2. ")
        .replace("③", "3. ")
        .replace("④", "4. ")
        .replace("⑤", "5. ")
    )
    label_re = r"(?:[A-H]|[1-8]|가|나|다|라|마)"
    pattern = re.compile(rf"({label_re})[\.\)]\s*(.+?)(?=(?:\s+{label_re}[\.\)]\s)|$)", re.S)
    for m in pattern.finditer(normalized):
        key = str(m.group(1)).strip().upper()
        val = normalize_space(m.group(2))
        if key and val:
            out[key] = val
    if len(out) >= 3:
        return out
    if keyword_hint and len(out) >= 2:
        return out
    return {}


def _max_surface_score(
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    device: torch.device,
    instruction: str,
    surfaces: Iterable[str],
    max_input_tokens: int,
) -> float:
    scores = [
        _score_continuation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_text=instruction,
            completion_text=str(surface),
            max_input_tokens=int(max_input_tokens),
        )
        for surface in surfaces
        if str(surface).strip()
    ]
    if not scores:
        return -1e9
    return float(max(scores))


def choose_scored_boolean_answer(
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    device: torch.device,
    instruction: str,
    prompt: str,
    max_input_tokens: int,
) -> Tuple[str | None, Dict[str, float]]:
    q = str(prompt or "")
    if not re.search(
        r"(예\s*또는\s*아니오|예/아니오|yes\s*or\s*no|true\s*or\s*false|참\s*또는\s*거짓|참/거짓|맞(?:나요|습니까|니)|틀(?:리나요|립니까|렸나))",
        q,
        re.IGNORECASE,
    ):
        return None, {}

    if re.search(r"(true\s*or\s*false|참\s*또는\s*거짓|참/거짓|참인가|거짓인가)", q, re.IGNORECASE):
        groups = {
            "참": ["참", "참.", "맞다", "맞습니다.", "True", "true"],
            "거짓": ["거짓", "거짓.", "틀리다", "틀립니다.", "False", "false"],
        }
    else:
        groups = {
            "예": ["예", "예.", "네", "네.", "맞다", "맞습니다."],
            "아니오": ["아니오", "아니오.", "아니요", "아닙니다.", "틀리다", "틀립니다."],
        }
    scores: Dict[str, float] = {}
    for label, surfaces in groups.items():
        expanded: List[str] = []
        for surface in surfaces:
            expanded.extend([surface, f"답: {surface}", f"정답: {surface}"])
        scores[label] = _max_surface_score(
            model=model,
            tokenizer=tokenizer,
            device=device,
            instruction=instruction,
            surfaces=expanded,
            max_input_tokens=int(max_input_tokens),
        )

    if len(scores) < 2:
        return None, scores
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = ordered[0]
    second_score = ordered[1][1]
    if (best_score - second_score) < 0.025:
        return None, scores
    return str(best_label), scores


def _mcq_return_style(prompt: str, label: str, text: str) -> str:
    q = str(prompt or "")
    if re.search(r"(문자만|선택지\s*문자|알파벳만|letter only|번호만|숫자만)", q, re.IGNORECASE):
        return str(label)
    if re.search(r"(정답\s*내용|내용만|내용을\s*짧게|보기\s*내용)", q, re.IGNORECASE):
        return str(text)
    return f"{label}. {text}".strip()


def _mcq_candidate_surfaces(label: str, text: str, prompt: str) -> List[str]:
    q = str(prompt or "")
    if re.search(r"(문자만|선택지\s*문자|알파벳만|letter only|번호만|숫자만)", q, re.IGNORECASE):
        return [str(label), f"{label}.", f"답: {label}", f"정답: {label}"]
    if re.search(r"(정답\s*내용|내용만|내용을\s*짧게|보기\s*내용)", q, re.IGNORECASE):
        return [str(text), f"답: {text}", f"정답: {text}"]
    base = [str(label), f"{label}.", str(text), f"{label}. {text}"]
    out: List[str] = []
    for item in base:
        out.extend([item, f"답: {item}", f"정답: {item}"])
    return out


def choose_scored_mcq_answer(
    model: TinyGPT,
    tokenizer: ByteTokenizer,
    device: torch.device,
    instruction: str,
    prompt: str,
    max_input_tokens: int,
) -> Tuple[str | None, Dict[str, float]]:
    options = parse_mcq_options(prompt)
    if len(options) < 2:
        return None, {}

    scores: Dict[str, float] = {}

    for label, text in options.items():
        scores[label] = _max_surface_score(
            model=model,
            tokenizer=tokenizer,
            device=device,
            instruction=instruction,
            surfaces=_mcq_candidate_surfaces(label=label, text=text, prompt=prompt),
            max_input_tokens=int(max_input_tokens),
        )

    if len(scores) < 2:
        return None, scores
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = ordered[0]
    second_score = ordered[1][1]
    if (best_score - second_score) < 0.025:
        return None, scores

    best_text = options.get(str(best_label), "")
    return _mcq_return_style(prompt=prompt, label=str(best_label), text=str(best_text)), scores


def normalize_logic_atom(text: str) -> str:
    t = str(text or "").strip().rstrip(" ?!.,")
    suffixes = ("인가요", "인가", "입니다", "이다", "일까요", "일까", "은", "는", "이", "가", "다")
    for s in suffixes:
        if len(t) > len(s) and t.endswith(s):
            t = t[: -len(s)].strip()
            break
    return t


def _logic_pair_key(a: str, b: str) -> Tuple[str, str]:
    x = normalize_logic_atom(a)
    y = normalize_logic_atom(b)
    return x, y


def _compute_subset_closure(all_pairs: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
    nodes: Set[str] = set()
    closure: Dict[str, Set[str]] = {}
    for a, b in all_pairs:
        if not a or not b:
            continue
        nodes.add(a)
        nodes.add(b)
        closure.setdefault(a, set()).add(b)
        closure.setdefault(b, set())
    for n in nodes:
        closure.setdefault(n, set()).add(n)

    changed = True
    while changed:
        changed = False
        for a in list(closure.keys()):
            expanded = set(closure.get(a, set()))
            for mid in list(expanded):
                expanded.update(closure.get(mid, set()))
            if not expanded.issubset(closure[a]):
                closure[a].update(expanded)
                changed = True
    return closure


def _reverse_subset_closure(closure: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    rev: Dict[str, Set[str]] = {}
    for a, targets in closure.items():
        rev.setdefault(a, set()).add(a)
        for b in targets:
            rev.setdefault(b, set()).add(a)
    return rev


def _extract_quantified_facts(text: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    src = str(text or "")
    all_pairs: List[Tuple[str, str]] = []
    some_pairs: List[Tuple[str, str]] = []
    no_pairs: List[Tuple[str, str]] = []
    some_not_pairs: List[Tuple[str, str]] = []

    patterns = [
        ("all", re.finditer(r"모든\s+([A-Za-z가-힣]+)(?:은|는|이|가)\s*([A-Za-z가-힣]+)(?:이다|다)(?:\s|$|[.?!,])", src)),
        ("some_not", re.finditer(r"일부\s+([A-Za-z가-힣]+)(?:은|는|이|가)\s*([A-Za-z가-힣]+)\s*(?:은|는|이|가)?\s*아니(?:다|다\.)?(?:\s|$|[.?!,])", src)),
        ("some", re.finditer(r"일부\s+([A-Za-z가-힣]+)(?:은|는|이|가)\s*([A-Za-z가-힣]+)(?:이다|다)(?:\s|$|[.?!,])", src)),
        ("no", re.finditer(r"(?:어떤|어느)\s+([A-Za-z가-힣]+)도\s*([A-Za-z가-힣]+)\s*(?:은|는|이|가)?\s*아니(?:다|다\.)?(?:\s|$|[.?!,])", src)),
        ("all", re.finditer(r"all\s+([A-Za-z]+)\s+are\s+([A-Za-z]+)(?:\s|$|[.?!,])", src, re.IGNORECASE)),
        ("some_not", re.finditer(r"some\s+([A-Za-z]+)\s+are\s+not\s+([A-Za-z]+)(?:\s|$|[.?!,])", src, re.IGNORECASE)),
        ("some", re.finditer(r"some\s+([A-Za-z]+)\s+are\s+([A-Za-z]+)(?:\s|$|[.?!,])", src, re.IGNORECASE)),
        ("no", re.finditer(r"no\s+([A-Za-z]+)\s+are\s+([A-Za-z]+)(?:\s|$|[.?!,])", src, re.IGNORECASE)),
    ]

    for kind, matches in patterns:
        for m in matches:
            a, b = _logic_pair_key(m.group(1), m.group(2))
            if not a or not b or a == b:
                continue
            if kind == "all":
                all_pairs.append((a, b))
            elif kind == "some":
                some_pairs.append((a, b))
            elif kind == "no":
                no_pairs.append((a, b))
            elif kind == "some_not":
                some_not_pairs.append((a, b))
    return all_pairs, some_pairs, no_pairs, some_not_pairs


def _extract_logic_query(text: str) -> Tuple[str, str, str] | None:
    src = str(text or "")
    patterns = [
        ("some", r"그렇다면\s*일부\s+([A-Za-z가-힣]+)(?:은|는|이|가)\s*([A-Za-z가-힣]+)(?:인가|인가요|일까|일까요)?\??(?:\s|$)"),
        ("all", r"그렇다면\s*모든\s+([A-Za-z가-힣]+)(?:은|는|이|가)\s*([A-Za-z가-힣]+)(?:인가|인가요|일까|일까요)?\??(?:\s|$)"),
        ("no", r"그렇다면\s*(?:어떤|어느)\s+([A-Za-z가-힣]+)도\s*([A-Za-z가-힣]+)(?:인가|인가요|일까|일까요)?\??(?:\s|$)"),
        ("no", r"그렇다면\s*(?:어떤|어느)\s+([A-Za-z가-힣]+)도\s*([A-Za-z가-힣]+)\s*(?:은|는|이|가)?\s*아닌가(?:요)?\??(?:\s|$)"),
        ("some_not", r"그렇다면\s*일부\s+([A-Za-z가-힣]+)(?:은|는|이|가)\s*([A-Za-z가-힣]+)\s*(?:은|는|이|가)?\s*아닌가(?:요)?\??(?:\s|$)"),
        ("some", r"therefore\s*,?\s*some\s+([A-Za-z]+)\s+are\s+([A-Za-z]+)\??(?:\s|$)", re.IGNORECASE),
        ("all", r"therefore\s*,?\s*all\s+([A-Za-z]+)\s+are\s+([A-Za-z]+)\??(?:\s|$)", re.IGNORECASE),
        ("no", r"therefore\s*,?\s*no\s+([A-Za-z]+)\s+are\s+([A-Za-z]+)\??(?:\s|$)", re.IGNORECASE),
        ("some_not", r"therefore\s*,?\s*some\s+([A-Za-z]+)\s+are\s+not\s+([A-Za-z]+)\??(?:\s|$)", re.IGNORECASE),
    ]
    for entry in patterns:
        if len(entry) == 2:
            kind, pattern = entry
            flags = re.IGNORECASE
        else:
            kind, pattern, flags = entry
        m = re.search(pattern, src, flags)
        if not m:
            continue
        a, b = _logic_pair_key(m.group(1), m.group(2))
        if a and b:
            return kind, a, b
    return None


def _solve_quantified_reasoning(prompt: str) -> str | None:
    premise_text = re.split(r"(그렇다면|therefore)", str(prompt or ""), maxsplit=1, flags=re.IGNORECASE)[0]
    all_pairs, some_pairs, no_pairs, some_not_pairs = _extract_quantified_facts(premise_text)
    query = _extract_logic_query(prompt)
    if not query:
        return None
    kind, src, dst = query

    closure = _compute_subset_closure(all_pairs)
    rev_closure = _reverse_subset_closure(closure)

    some_closure: Set[Tuple[str, str]] = set()
    for a, b in some_pairs:
        if a and b:
            some_closure.add((a, b))
            some_closure.add((b, a))
    changed = True
    while changed:
        changed = False
        cur = list(some_closure)
        for a, b in cur:
            for sup_a in closure.get(a, {a}):
                if (sup_a, b) not in some_closure:
                    some_closure.add((sup_a, b))
                    changed = True
            for sup_b in closure.get(b, {b}):
                if (a, sup_b) not in some_closure:
                    some_closure.add((a, sup_b))
                    changed = True

    no_closure: Set[Tuple[str, str]] = set()
    for a, b in no_pairs:
        for sub_a in rev_closure.get(a, {a}):
            for sub_b in rev_closure.get(b, {b}):
                no_closure.add((sub_a, sub_b))
                no_closure.add((sub_b, sub_a))

    some_not_closure: Set[Tuple[str, str]] = set()
    for a, b in some_not_pairs:
        for sup_a in closure.get(a, {a}):
            for sub_b in rev_closure.get(b, {b}):
                some_not_closure.add((sup_a, sub_b))

    if kind == "all":
        if dst in closure.get(src, {src}):
            return f"네. 모든 {src}는 {dst}입니다."
        return f"아닙니다. 주어진 전제만으로 모든 {src}가 {dst}라고 보장할 수 없습니다."

    if kind == "some":
        if (src, dst) in some_closure:
            return f"네. 일부 {src}는 {dst}입니다."
        if (src, dst) in no_closure:
            return f"아닙니다. 어떤 {src}도 {dst}가 아니므로 일부 {src}가 {dst}일 수 없습니다."
        return f"아닙니다. 주어진 전제만으로 일부 {src}가 {dst}라고 보장할 수 없습니다."

    if kind == "no":
        if (src, dst) in no_closure:
            return f"네. 어떤 {src}도 {dst}가 아닙니다."
        return f"아닙니다. 주어진 전제만으로 어떤 {src}도 {dst}가 아니라고 보장할 수 없습니다."

    if kind == "some_not":
        if (src, dst) in some_not_closure:
            return f"네. 일부 {src}는 {dst}가 아닙니다."
        if dst in closure.get(src, set()):
            return f"아닙니다. 모든 {src}가 {dst}라면 일부 {src}가 {dst}가 아니라고 할 수 없습니다."
        return f"아닙니다. 주어진 전제만으로 일부 {src}가 {dst}가 아니라고 보장할 수 없습니다."

    return None


def solve_simple_reasoning(prompt: str) -> str | None:
    q = str(prompt or "").strip()
    low = q.lower()
    if not q:
        return None

    quantified = _solve_quantified_reasoning(q)
    if quantified:
        return quantified

    # 1) Simple arithmetic expression.
    if re.search(r"[0-9]\s*[\+\-\*/]\s*[0-9]", q):
        expr_m = re.search(r"([0-9\.\+\-\*/\(\)\s]{3,})", q)
        if expr_m:
            expr = expr_m.group(1).strip()
            val = _safe_eval_numeric_expr(expr)
            if val is not None:
                return f"{_format_number(val)}입니다."

    # 2) Ratio word problem: A:B = m:n, B is x then A is ?
    ratio_m = re.search(
        r"([A-Za-z가-힣]+)\s*과\s*([A-Za-z가-힣]+)\s*(?:의\s*)?비율(?:이|은|는)?\s*([0-9]+(?:\.[0-9]+)?)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        q,
    )
    given_m = re.search(r"([A-Za-z가-힣]+)\s*가\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z가-힣%]*)\s*이면", q)
    ask_m = re.search(r"(?:이면|일때|일 때)\s*([A-Za-z가-힣]+)\s*(?:은|는)\s*\??\s*$", q)
    if ratio_m and given_m and ask_m:
        a = ratio_m.group(1)
        b = ratio_m.group(2)
        ra = float(ratio_m.group(3))
        rb = float(ratio_m.group(4))
        given_name = given_m.group(1)
        given_val = float(given_m.group(2))
        unit = given_m.group(3)
        ask_name = ask_m.group(1)

        coeff = {a: ra, b: rb}
        if given_name in coeff and ask_name in coeff and coeff[given_name] > 0:
            ans = given_val * coeff[ask_name] / coeff[given_name]
            return f"{ask_name}은 {_format_number(ans)}{unit}입니다."

    # 3) Relational ordering: X > Y, Y > Z and ask smallest/largest.
    rels = []
    for m in re.finditer(r"([A-Za-z가-힣]+)\s*는\s*([A-Za-z가-힣]+)\s*보다\s*크", q):
        rels.append((m.group(1), m.group(2)))  # greater, smaller
    for m in re.finditer(r"([A-Za-z가-힣]+)\s*는\s*([A-Za-z가-힣]+)\s*보다\s*작", q):
        rels.append((m.group(2), m.group(1)))  # invert to (greater, smaller)
    for m in re.finditer(r"([A-Za-z]+)\s+is\s+(?:greater|larger|bigger)\s+than\s+([A-Za-z]+)", q, re.IGNORECASE):
        rels.append((m.group(1), m.group(2)))
    for m in re.finditer(r"([A-Za-z]+)\s+is\s+(?:smaller|less)\s+than\s+([A-Za-z]+)", q, re.IGNORECASE):
        rels.append((m.group(2), m.group(1)))
    if rels:
        nodes: Set[str] = set()
        greater_out: Dict[str, Set[str]] = {}
        greater_in: Dict[str, Set[str]] = {}
        for g, s in rels:
            nodes.add(g)
            nodes.add(s)
            greater_out.setdefault(g, set()).add(s)
            greater_in.setdefault(s, set()).add(g)
            greater_out.setdefault(s, set())
            greater_in.setdefault(g, set())

        ask_small = bool(re.search(r"(가장\s*작|최소|smallest|least)", low))
        ask_large = bool(re.search(r"(가장\s*크|최대|largest|greatest)", low))
        if ask_small or ask_large:
            if ask_small:
                cands = [n for n in nodes if len(greater_out.get(n, set())) == 0]
                label = "가장 작은 것은" if re.search(r"[가-힣]", q) else "The smallest is"
            else:
                cands = [n for n in nodes if len(greater_in.get(n, set())) == 0]
                label = "가장 큰 것은" if re.search(r"[가-힣]", q) else "The largest is"
            if len(cands) == 1:
                if re.search(r"[가-힣]", q):
                    return f"{label} {cands[0]}입니다."
                return f"{label} {cands[0]}."

    # 4) Classic liar/truth-teller puzzle.
    if (
        "거짓말쟁이" in q
        and re.search(r"진실만\s*말하는", q)
        and re.search(r"두\s*질문", q)
        and re.search(r"(구분|누가\s*누구|방법)", q)
    ):
        return (
            "한 사람에게 \"내가 저 사람에게 당신이 진실만 말하는 사람이냐고 물으면 뭐라고 답하겠습니까?\"라고 묻습니다. "
            "예라고 하면 상대가 진실만 말하는 사람이고, 아니오라고 하면 상대가 거짓말쟁이입니다. "
            "그러면 남은 한 사람의 정체는 반대로 바로 알 수 있습니다."
        )

    return None


def alignment_bonus(prompt: str, response: str) -> float:
    q = str(prompt or "").strip()
    r = str(response or "").strip()
    if not r:
        return -1.5

    bonus = 0.0
    term = extract_definition_term(q)
    if DEF_QUERY_HINT_RE.search(q) and term:
        low_r = r.lower()
        low_t = term.lower()
        sim = best_term_similarity(low_t, low_r)
        if low_t in low_r:
            bonus += 1.20
        elif sim >= 0.84:
            bonus += 0.45
        else:
            bonus -= 1.45

    if LOGIC_QUERY_HINT_RE.search(q):
        if re.search(r"(이유|따라서|결론|보장|아니|질문|가정|구분)", r):
            bonus += 0.45
        else:
            bonus -= 0.60

    if INTRO_ANSWER_HINT_RE.search(r) and not INTRO_QUERY_HINT_RE.search(q):
        bonus -= 1.0

    return float(bonus)


def is_factoid_prompt(text: str) -> bool:
    q = str(text or "").strip().lower()
    if looks_like_solver_query(q):
        return True
    hints = ["수도", "정답", "몇", "얼마", "더하기", "빼기", "+", "-", "*", "/"]
    return any(h in q for h in hints)


def trim_to_first_sentence(text: str, max_chars: int = 120) -> str:
    s = normalize_space(text)
    if not s:
        return s
    def _first_sentence(src: str) -> str:
        end_local = re.search(r"[.!?]\s|[.!?]$", src)
        if not end_local:
            return src.strip()
        return src[: end_local.end()].strip()

    if len(s) <= int(max_chars):
        cut = _first_sentence(s)
        if len(cut) >= 4:
            # Avoid returning a bare greeting like "안녕하세요." when more content exists.
            greeting_like = bool(re.match(r"^(안녕하세요|반갑습니다|안녕하십니까)[.!?]?$", cut))
            if greeting_like and len(s) > len(cut) + 2:
                rest = s[len(cut) :].strip()
                second = _first_sentence(rest)
                merged = normalize_space(f"{cut} {second}")
                return merged[: int(max_chars)].strip()
            return cut
        return s
    head = s[: int(max_chars)]
    end = re.search(r"[.!?]\s|[.!?]$", head)
    if end:
        cut = head[: end.end()].strip()
        if len(cut) >= 4:
            return cut
    return head.strip()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def postprocess_response(prompt: str, response: str) -> str:
    _ = prompt
    s = normalize_space(response)
    if not s:
        return s
    # Remove common unknown-token artifacts from SPM decoding.
    s = s.replace("⁇", " ").replace("\ufffd", " ")
    s = normalize_space(s)
    return s


def parse_agent_action(text: str) -> Tuple[str, str]:
    s = normalize_space(text)
    m = re.match(r"^ACTION\s+(search|calculator)\s+(.+)$", s, re.IGNORECASE)
    if not m:
        return "", ""
    return str(m.group(1)).strip().lower(), str(m.group(2)).strip()


def strip_final_prefix(text: str) -> str:
    s = normalize_space(text)
    if not s:
        return s
    m = re.match(r"^FINAL\s+(.+)$", s, re.IGNORECASE)
    if m:
        return normalize_space(m.group(1))
    return s


def rare_syllable_ratio(text: str, common_syllables: Set[str]) -> float:
    if not common_syllables:
        return 0.0
    chars = re.findall(r"[가-힣]", str(text or ""))
    if not chars:
        return 0.0
    rare = sum(1 for ch in chars if ch not in common_syllables)
    return float(rare) / float(max(1, len(chars)))


def looks_low_quality_response(
    prompt: str,
    response: str,
    common_syllables: Set[str] | None = None,
) -> bool:
    q = str(prompt or "").strip()
    r = str(response or "").strip()
    commons = KO_COMMON_SYLLABLES if common_syllables is None else common_syllables
    if not r:
        return True
    if len(r) < 8:
        if not (is_factoid_prompt(q) and len(r) >= 4):
            return True
    if re.search(r"(.)\1{8,}", r):
        return True
    if r.count("?") >= 10:
        return True

    q_has_hangul = bool(re.search(r"[\uac00-\ud7a3]", q))
    q_keys = keyword_terms(q)
    r_low = r.lower()
    if q_has_hangul:
        hangul_n = len(re.findall(r"[\uac00-\ud7a3]", r))
        latin_n = len(re.findall(r"[A-Za-z]", r))
        if hangul_n < 2 and latin_n >= 16:
            return True
        if hangul_n >= 12:
            rare_ratio = rare_syllable_ratio(r, commons)
            if rare_ratio >= 0.08:
                return True
            if latin_n > 0 and (hangul_n / float(max(1, hangul_n + latin_n))) < 0.58:
                return True
        if q_keys:
            has_key = any(k in r_low for k in q_keys)
            short_fact = len(r) <= 24 and bool(re.search(r"(입니다|다|요)\.?$", r))
            if (not has_key) and (not (short_fact and is_factoid_prompt(q))):
                return True
        if DEF_QUERY_HINT_RE.search(q):
            term = extract_definition_term(q)
            if term:
                low_t = term.lower()
                sim = best_term_similarity(low_t, r_low)
                if (low_t not in r_low) and (sim < 0.84):
                    return True
            if has_contiguous_repeated_hangul_chunk(r, min_chunk=2, max_chunk=3, min_repeats=1):
                return True

    toks = re.findall(r"[A-Za-z0-9\uac00-\ud7a3]{2,}", r.lower())
    if len(toks) >= 16:
        diversity = len(set(toks)) / float(max(1, len(toks)))
        if diversity < 0.34:
            return True

    if not looks_like_solver_query(q):
        math_hits = 0
        if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", r):
            math_hits += 1
        if "=" in r and re.search(r"\d", r):
            math_hits += 1
        if re.search(r"(?i)\b(result|answer|equation)\b", r):
            math_hits += 1
        if math_hits >= 2:
            return True

    q_toks = retrieval_tokens(q)
    r_toks = retrieval_tokens(r)
    if (
        q_toks
        and r_toks
        and len(q_toks.intersection(r_toks)) == 0
        and not looks_like_solver_query(q)
        and not q_has_hangul
    ):
        return True
    return False


def response_relevance_score(
    prompt: str,
    response: str,
    common_syllables: Set[str] | None = None,
) -> float:
    q = str(prompt or "").strip()
    r = str(response or "").strip()
    commons = KO_COMMON_SYLLABLES if common_syllables is None else common_syllables
    if not r:
        return -1.0

    score = 0.0
    score += min(len(r), 240) / 240.0

    q_toks = retrieval_tokens(q)
    r_toks = retrieval_tokens(r)
    if q_toks and r_toks:
        inter = len(q_toks.intersection(r_toks))
        union = len(q_toks.union(r_toks))
        score += 2.0 * (inter / float(max(1, len(q_toks))))
        score += 1.0 * (inter / float(max(1, union)))

    if looks_like_solver_query(q) and re.search(r"\d", r):
        score += 0.5
    q_keys = keyword_terms(q)
    if q_keys:
        hit = sum(1 for k in q_keys if k in r.lower())
        score += 0.9 * (hit / float(max(1, len(q_keys))))
        if hit == 0 and not is_factoid_prompt(q):
            score -= 0.8
    if re.search(r"(.)\1{8,}", r):
        score -= 2.0
    if re.search(r"[\uac00-\ud7a3]", q):
        score -= 1.75 * rare_syllable_ratio(r, commons)

    return score


def response_consensus_score(candidates: List[Dict[str, Any]], top_n: int = 4) -> float:
    if not candidates:
        return 0.0
    ranked = sorted(candidates, key=lambda x: float(x.get("score", -1e9)), reverse=True)
    use = ranked[: max(1, int(top_n))]
    if len(use) <= 1:
        return 1.0

    base = str(use[0].get("response", ""))
    base_toks = retrieval_tokens(base)
    if not base_toks:
        return 0.0

    sims: List[float] = []
    for row in use[1:]:
        cur = str(row.get("response", ""))
        cur_toks = retrieval_tokens(cur)
        if not cur_toks:
            sims.append(0.0)
            continue
        inter = len(base_toks.intersection(cur_toks))
        union = len(base_toks.union(cur_toks))
        sims.append(float(inter) / float(max(1, union)))

    if not sims:
        return 1.0
    return float(sum(sims) / float(len(sims)))


def deterministic_seed(text: str, salt: str = "") -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(salt).encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(str(text).encode("utf-8", errors="ignore"))
    return int.from_bytes(h.digest(), byteorder="big", signed=False) & 0x7FFFFFFF


def build_instruction(
    user_prompt: str,
    history: List[Tuple[str, str]],
    system_prompt: str,
    route_label: str,
    response_prefix: str = "",
    extra_rule: str = "",
) -> str:
    chunks: List[str] = []
    if str(system_prompt or "").strip():
        chunks.append(f"(System) {str(system_prompt).strip()}")
    if str(extra_rule or "").strip():
        chunks.append(f"(Rule) {str(extra_rule).strip()}")
    if history and route_label == "chat":
        chunks.append("[Recent]")
        for u, a in history[-2:]:
            chunks.append(f"User: {u}")
            chunks.append(f"Assistant: {a}")
    chunks.append(str(user_prompt).strip())
    merged = "\n".join(chunks)
    suffix = str(response_prefix or "")
    return f"### Instruction\n{merged}\n\n### Response\n{suffix}"


def stream_print_text(text: str, chunk_size: int = 24) -> None:
    src = str(text or "")
    if not src:
        safe_print("")
        return
    idx = 0
    while idx < len(src):
        part = src[idx : idx + int(chunk_size)]
        try:
            print(part, end="", flush=True)
        except UnicodeEncodeError:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            safe = part.encode(enc, errors="replace").decode(enc, errors="replace")
            print(safe, end="", flush=True)
        idx += int(chunk_size)
    print("", flush=True)


def load_memory_profile(memory_path: Path, session_id: str) -> Dict[str, Any]:
    p = Path(memory_path)
    if not p.exists():
        return {"session_id": str(session_id), "conversation_count": 0}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"session_id": str(session_id), "conversation_count": 0}
    if not isinstance(raw, dict):
        return {"session_id": str(session_id), "conversation_count": 0}
    prof = raw.get(str(session_id), {})
    if not isinstance(prof, dict):
        prof = {}
    prof.setdefault("session_id", str(session_id))
    prof.setdefault("conversation_count", 0)
    return prof


def save_memory_profile(memory_path: Path, session_id: str, profile: Dict[str, Any]) -> None:
    p = Path(memory_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (json.JSONDecodeError, OSError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    raw[str(session_id)] = dict(profile)
    p.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_term_key(term: str) -> str:
    t = normalize_ko_token(str(term or ""))
    if t:
        return t
    return str(term or "").strip().lower()


def read_tool_cache(tool_cache_path: Path, max_rows: int = 20000) -> List[Dict[str, Any]]:
    p = Path(tool_cache_path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= int(max_rows):
                    break
                s = str(line).strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


def append_tool_cache_row(tool_cache_path: Path, row: Dict[str, Any]) -> None:
    p = Path(tool_cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _tool_cache_match_score(query_key: str, row_key: str) -> float:
    q = str(query_key or "").strip()
    r = str(row_key or "").strip()
    if not q or not r:
        return -1.0
    if q == r:
        return 10.0
    if abs(len(q) - len(r)) > 2:
        return -1.0
    dist = levenshtein_distance(q, r)
    sim = 1.0 - float(dist) / float(max(len(q), len(r), 1))
    if sim < 0.84:
        return -1.0
    return float(sim)


def _tool_cache_row_usable(term: str, row: Dict[str, Any]) -> bool:
    row_term = str(row.get("term", "")).strip()
    answer = str(row.get("answer", "")).strip()
    if not row_term or not answer:
        return False
    if "\ufffd" in answer:
        return False
    if answer.count("?") >= max(3, len(answer) // 18):
        return False
    if definition_noise_signal(answer):
        return False
    if len(re.findall(r"[가-힣]", answer)) < 2 and len(re.findall(r"[A-Za-z]", answer)) < 8:
        return False
    return True


def get_cached_tool_answer(term: str, tool_cache_path: Path) -> Tuple[str, Dict[str, Any]] | Tuple[None, None]:
    key = normalize_term_key(term)
    if not key:
        return None, None
    rows = read_tool_cache(tool_cache_path=tool_cache_path, max_rows=20000)
    best_row: Dict[str, Any] | None = None
    best_score = -1.0
    for row in rows:
        row_key = normalize_term_key(str(row.get("term", "")))
        score = _tool_cache_match_score(query_key=key, row_key=row_key)
        if score < 0.0:
            continue
        if not _tool_cache_row_usable(term=term, row=row):
            continue
        answer = str(row.get("answer", "")).strip()
        score += min(0.35, 24.0 / float(max(24, len(answer))))
        if score > best_score:
            best_row = row
            best_score = float(score)
    if not best_row:
        return None, None
    answer = str(best_row.get("answer", "")).strip()
    if not answer:
        return None, None
    return answer, best_row


def fetch_wikipedia_summary(term: str, lang: str, timeout_sec: float) -> Tuple[str, str]:
    t = str(term or "").strip()
    if not t:
        return "", ""
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(t)}"
    req = Request(url, headers={"User-Agent": "spear1.0-tool-lookup/1.0"})
    try:
        with urlopen(req, timeout=max(1.0, float(timeout_sec))) as resp:
            raw = resp.read()
    except (HTTPError, URLError, TimeoutError, OSError):
        return "", ""
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return "", ""
    if not isinstance(payload, dict):
        return "", ""
    extract = str(payload.get("extract", "")).strip()
    page_url = ""
    content_urls = payload.get("content_urls")
    if isinstance(content_urls, dict):
        desktop = content_urls.get("desktop", {})
        if isinstance(desktop, dict):
            page_url = str(desktop.get("page", "")).strip()
    return extract, page_url


def compose_definition_from_summary(term: str, summary: str, max_chars: int = 220) -> str:
    t = str(term or "").strip()
    s = normalize_space(summary)
    if not t or not s:
        return ""
    cut = trim_to_first_sentence(s, max_chars=max(80, int(max_chars)))
    low_t = t.lower()
    low_c = cut.lower()
    if low_t in low_c:
        return cut
    particle = "은" if has_final_consonant(t[-1]) else "는"
    return f"{t}{particle} {cut}"


def tool_lookup_definition_answer(
    term: str,
    prompt: str,
    tool_cache_path: Path,
    timeout_sec: float,
    allow_web: bool,
) -> Tuple[str, Dict[str, Any]] | Tuple[None, None]:
    cached_answer, cached_row = get_cached_tool_answer(term=term, tool_cache_path=tool_cache_path)
    if cached_answer:
        meta = {
            "tool_name": "cache",
            "tool_lang": str(cached_row.get("lang", "ko")) if isinstance(cached_row, dict) else "ko",
            "tool_source_url": str(cached_row.get("source_url", "")) if isinstance(cached_row, dict) else "",
            "tool_cache_hit": True,
        }
        return str(cached_answer), meta

    if not bool(allow_web):
        return None, None

    for lang in ("ko", "en"):
        summary, source_url = fetch_wikipedia_summary(term=term, lang=lang, timeout_sec=float(timeout_sec))
        if len(summary) < 24:
            continue
        answer = compose_definition_from_summary(term=term, summary=summary, max_chars=220)
        if not answer:
            continue
        if severe_gibberish_signal(prompt=prompt, response=answer, common_syllables=KO_COMMON_SYLLABLES):
            continue
        row = {
            "created_at": now_iso(),
            "term": str(term),
            "prompt": str(prompt),
            "answer": str(answer),
            "lang": str(lang),
            "source": "wikipedia_summary",
            "source_url": str(source_url),
        }
        append_tool_cache_row(tool_cache_path=tool_cache_path, row=row)
        meta = {
            "tool_name": "wikipedia_summary",
            "tool_lang": str(lang),
            "tool_source_url": str(source_url),
            "tool_cache_hit": False,
        }
        return answer, meta
    return None, None


@dataclass
class RoutedBundle:
    label: str
    checkpoint: str
    model: TinyGPT
    tokenizer: ByteTokenizer


class ModelRouter:
    def __init__(
        self,
        device: torch.device,
        checkpoint: str,
        use_ema: bool = False,
        router_mode: str = "auto",
        code_checkpoint: str = "",
        chat_checkpoint: str = "",
        quantize_int8: bool = False,
    ) -> None:
        self.device = device
        self.use_ema = bool(use_ema)
        self.router_mode = str(router_mode).strip().lower()
        self.quantize_int8 = bool(quantize_int8)
        self.default_ckpt = str(checkpoint).strip()
        self.code_ckpt = str(code_checkpoint).strip() or self.default_ckpt
        self.chat_ckpt = str(chat_checkpoint).strip() or self.default_ckpt
        self._cache: Dict[str, Tuple[TinyGPT, ByteTokenizer]] = {}

    def _get(self, ckpt_path: str) -> Tuple[TinyGPT, ByteTokenizer]:
        key = str(Path(ckpt_path))
        if key in self._cache:
            return self._cache[key]
        model, tok = build_model(
            checkpoint=key,
            device=self.device,
            use_ema=self.use_ema,
            quantize_int8=self.quantize_int8,
        )
        self._cache[key] = (model, tok)
        return model, tok

    def route(self, user_prompt: str) -> RoutedBundle:
        if self.router_mode == "single":
            label = "single"
            ckpt = self.default_ckpt
        else:
            code_like = is_code_request(user_prompt)
            label = "code" if code_like else "chat"
            ckpt = self.code_ckpt if code_like else self.chat_ckpt
        model, tokenizer = self._get(ckpt)
        return RoutedBundle(label=label, checkpoint=ckpt, model=model, tokenizer=tokenizer)


def run_one_turn(
    router: ModelRouter,
    user_prompt: str,
    history: List[Tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int,
    max_input_tokens: int,
    max_output_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    min_new_tokens: int,
    disable_ko_guard: bool,
    ko_guard_topk: int,
    ko_guard_rare_penalty: float,
    ko_guard_latin_penalty: float,
    disable_retrieval: bool,
    disable_quality_rerank: bool,
    rerank_candidates: int,
    rerank_temp_floor: float,
    rerank_top_k: int,
    rerank_top_p: float,
    zero_shot_mode: str,
    agent_mode: str,
    tool_cache_path: str,
    tool_lookup_timeout: float,
    disable_web_tool_lookup: bool,
    heuristic_mode: str = "off",
    force_raw: bool = False,
) -> Tuple[str, RoutedBundle, Dict[str, Any]]:
    _ = disable_retrieval
    bundle = router.route(user_prompt=user_prompt)
    prompt = str(user_prompt or "").strip()
    agent = str(agent_mode or "auto").strip().lower()
    tool_cache = Path(str(tool_cache_path or DEFAULT_TOOL_CACHE_PATH))
    meta: Dict[str, Any] = {
        "route": str(bundle.label),
        "checkpoint": str(bundle.checkpoint),
        "heuristic_mode": "off",
        "deterministic_solver": False,
        "scored_solver": False,
        "used_retrieval": False,
        "tool_lookup_used": False,
        "safe_fallback": False,
        "pure_generation": True,
        "rerank_used": False,
        "candidate_count": 1,
    }
    zs_mode = normalize_zero_shot_mode(zero_shot_mode)
    meta["zero_shot_mode"] = str(zs_mode)
    meta["agent_mode"] = str(agent)
    heuristic_raw = str(heuristic_mode or "off").strip().lower()
    if heuristic_raw == "legacy":
        heuristic = heuristic_raw
    else:
        heuristic = "off"
    meta["heuristic_mode"] = str(heuristic)

    if zs_mode == "off":
        zs_def_low_score = -999.0
        zs_general_low_score = -999.0
        zs_force_def_rerank = False
        zs_enable_general_fallback = False
    elif zs_mode == "strict":
        zs_def_low_score = 0.75
        zs_general_low_score = 0.45
        zs_force_def_rerank = False
        zs_enable_general_fallback = True
    else:
        zs_def_low_score = 0.80
        zs_general_low_score = 0.55
        zs_force_def_rerank = False
        zs_enable_general_fallback = True

    instruction = build_instruction(
        user_prompt=prompt,
        history=history,
        system_prompt=system_prompt,
        route_label=bundle.label,
        response_prefix="",
        extra_rule="",
    )
    enable_ko_guard = bool((not disable_ko_guard) and bool(re.search(r"[가-힣]", prompt)))
    def_term = extract_definition_term(prompt) if DEF_QUERY_HINT_RE.search(prompt) else ""
    def_anchor = definition_anchor_prefix(def_term) if def_term else ""
    def_guard_rule = (
        "정의 질문에서는 질문 대상을 다른 서비스명으로 바꾸지 말고, 확실하지 않으면 모른다고 짧게 답하세요."
        if def_term
        else ""
    )

    def _generate_candidate(
        cand_temperature: float,
        cand_top_k: int,
        cand_top_p: float,
        cand_seed: int | None,
        response_prefix: str = "",
        extra_rule: str = "",
        user_prompt_override: str = "",
    ) -> str:
        prompt_local = str(user_prompt_override or prompt).strip()
        instruction_local = build_instruction(
            user_prompt=prompt_local,
            history=history,
            system_prompt=system_prompt,
            route_label=bundle.label,
            response_prefix=str(response_prefix or ""),
            extra_rule=str(extra_rule or ""),
        )
        full_text = generate_text(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            device=router.device,
            prompt=instruction_local,
            max_new_tokens=int(max_new_tokens),
            max_input_tokens=int(max_input_tokens),
            max_output_tokens=int(max_output_tokens),
            temperature=float(cand_temperature),
            top_k=int(cand_top_k),
            top_p=float(cand_top_p),
            repetition_penalty=float(repetition_penalty),
            min_new_tokens=int(min_new_tokens),
            enable_ko_guard=enable_ko_guard,
            ko_guard_topk=int(ko_guard_topk),
            ko_guard_rare_penalty=float(ko_guard_rare_penalty),
            ko_guard_latin_penalty=float(ko_guard_latin_penalty),
            sample_seed=cand_seed,
        )
        out = extract_response(full_text=full_text, prompt=instruction_local).strip()
        if str(response_prefix or "").strip():
            out = f"{str(response_prefix)}{out}".strip()
        return postprocess_response(prompt=prompt, response=out)

    def _score_candidate(text: str) -> Tuple[float, bool, float]:
        cand_low = looks_low_quality_response(prompt=prompt, response=text, common_syllables=KO_COMMON_SYLLABLES)
        cand_align = alignment_bonus(prompt=prompt, response=text)
        cand_score = response_relevance_score(prompt=prompt, response=text, common_syllables=KO_COMMON_SYLLABLES) + float(
            cand_align
        )
        if cand_low:
            cand_score -= 1.1
        if cand_align <= -0.8:
            cand_low = True
        if def_term and cand_low and UNKNOWN_ANSWER_HINT_RE.search(str(text)):
            # Prefer explicit uncertainty over confident but wrong entity substitutions.
            cand_score += 0.55
        return float(cand_score), bool(cand_low), float(cand_align)

    response = _generate_candidate(
        cand_temperature=float(temperature),
        cand_top_k=int(top_k),
        cand_top_p=float(top_p),
        cand_seed=None,
        response_prefix="",
        extra_rule="",
    )

    if heuristic == "off":
        meta["best_score"] = 0.0
        meta["best_low_quality"] = False
        meta["best_align"] = 0.0
        meta["candidate_count"] = 1
        return response, bundle, meta

    # Deterministic micro-solver for easy zero-shot logic/math prompts.
    det_answer = solve_simple_reasoning(prompt)
    if det_answer:
        meta["deterministic_solver"] = True
        meta["safe_fallback"] = False
        meta["rerank_used"] = False
        meta["candidate_count"] = 1
        meta["best_score"] = 9.0
        meta["best_low_quality"] = False
        meta["best_align"] = 0.0
        return det_answer, bundle, meta

    scored_bool_answer, bool_scores = choose_scored_boolean_answer(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        device=router.device,
        instruction=instruction,
        prompt=prompt,
        max_input_tokens=int(max_input_tokens),
    )
    if scored_bool_answer:
        meta["scored_solver"] = True
        meta["scored_solver_type"] = "bool"
        meta["scored_solver_scores"] = {k: float(v) for k, v in bool_scores.items()}
        meta["safe_fallback"] = False
        meta["rerank_used"] = False
        meta["candidate_count"] = int(len(bool_scores))
        meta["best_score"] = float(max(bool_scores.values()))
        meta["best_low_quality"] = False
        meta["best_align"] = 0.0
        return str(scored_bool_answer), bundle, meta

    scored_mcq_answer, mcq_scores = choose_scored_mcq_answer(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        device=router.device,
        instruction=instruction,
        prompt=prompt,
        max_input_tokens=int(max_input_tokens),
    )
    if scored_mcq_answer:
        meta["scored_solver"] = True
        meta["scored_solver_type"] = "mcq"
        meta["scored_solver_scores"] = {k: float(v) for k, v in mcq_scores.items()}
        meta["safe_fallback"] = False
        meta["rerank_used"] = False
        meta["candidate_count"] = int(len(mcq_scores))
        meta["best_score"] = float(max(mcq_scores.values()))
        meta["best_low_quality"] = False
        meta["best_align"] = 0.0
        return str(scored_mcq_answer), bundle, meta

    action_tool, action_arg = parse_agent_action(response)
    if action_tool and agent in {"auto", "triad"} and (not bool(disable_retrieval)):
        observation = ""
        if action_tool == "search":
            tool_answer, tool_meta = tool_lookup_definition_answer(
                term=str(action_arg),
                prompt=str(prompt),
                tool_cache_path=tool_cache,
                timeout_sec=float(tool_lookup_timeout),
                allow_web=(not bool(disable_web_tool_lookup)),
            )
            if tool_answer:
                observation = str(tool_answer)
                meta["used_retrieval"] = True
                meta["tool_lookup_used"] = True
                if isinstance(tool_meta, dict):
                    meta["tool_name"] = str(tool_meta.get("tool_name", ""))
                    meta["tool_lang"] = str(tool_meta.get("tool_lang", ""))
                    meta["tool_source_url"] = str(tool_meta.get("tool_source_url", ""))
                    meta["tool_cache_hit"] = bool(tool_meta.get("tool_cache_hit", False))
        elif action_tool == "calculator":
            calc_val = _safe_eval_numeric_expr(str(action_arg))
            if calc_val is not None:
                observation = _format_number(calc_val)

        if observation:
            meta["agent_action"] = {"tool": str(action_tool), "arg": str(action_arg)}
            tool_obs = f"ACTION {action_tool} {action_arg}\nOBSERVATION {observation}"
            follow = _generate_candidate(
                cand_temperature=max(0.0, float(temperature)),
                cand_top_k=int(top_k),
                cand_top_p=float(top_p),
                cand_seed=None,
                response_prefix="FINAL ",
                extra_rule="도구 결과를 바탕으로 FINAL <답> 한 줄로 답하세요.",
                user_prompt_override=f"{prompt}\n{tool_obs}",
            )
            response = postprocess_response(prompt=prompt, response=strip_final_prefix(follow))

    best_response = response
    best_score, best_low_quality, best_align = _score_candidate(response)
    def_term_mismatch = False
    if def_term:
        low_t = str(def_term).lower()
        low_r = str(response).lower()
        sim = best_term_similarity(low_t, low_r)
        def_term_mismatch = bool((low_t not in low_r) and (sim < 0.84))
        if def_term_mismatch:
            best_score -= 1.0
            best_low_quality = True

    candidates = [
        {
            "response": best_response,
            "score": float(best_score),
            "low_quality": bool(best_low_quality),
            "align": float(best_align),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
        }
    ]

    rerank_trigger = bool(best_low_quality or float(temperature) > 0.0 or def_term_mismatch)
    if def_term and zs_mode == "strict":
        rerank_trigger = True
    if def_term and float(temperature) <= 0.0 and (not def_term_mismatch) and zs_mode != "strict":
        rerank_trigger = bool(DEF_DRIFT_HINT_RE.search(str(response)) or UNKNOWN_ANSWER_HINT_RE.search(str(response)))
    if def_term and bool(zs_force_def_rerank):
        rerank_trigger = True

    use_rerank = (
        (not bool(disable_quality_rerank))
        and int(rerank_candidates) > 1
        and bool(re.search(r"[가-힣]", prompt))
        and rerank_trigger
    )

    if use_rerank:
        base_seed = deterministic_seed(text=prompt, salt=str(bundle.checkpoint))
        base_top_k = max(int(rerank_top_k), int(top_k), 0)
        base_top_p = min(max(float(rerank_top_p), 0.55), 0.99)
        floor_temp = min(max(float(rerank_temp_floor), 0.01), 0.95)

        if def_anchor:
            anchored = _generate_candidate(
                cand_temperature=max(0.12, floor_temp),
                cand_top_k=max(64, min(192, base_top_k)),
                cand_top_p=max(0.86, min(0.99, base_top_p)),
                cand_seed=int(base_seed + 997),
                response_prefix=str(def_anchor),
                extra_rule=str(def_guard_rule),
            )
            anchored_score, anchored_low, anchored_align = _score_candidate(anchored)
            candidates.append(
                {
                    "response": anchored,
                    "score": float(anchored_score),
                    "low_quality": bool(anchored_low),
                    "align": float(anchored_align),
                    "temperature": float(max(0.12, floor_temp)),
                    "top_k": int(max(64, min(192, base_top_k))),
                    "top_p": float(max(0.86, min(0.99, base_top_p))),
                }
            )
            if anchored_score > best_score:
                best_score = float(anchored_score)
                best_low_quality = bool(anchored_low)
                best_response = str(anchored)
                best_align = float(anchored_align)

        if def_term:
            guarded = _generate_candidate(
                cand_temperature=max(0.10, floor_temp),
                cand_top_k=max(48, min(160, base_top_k)),
                cand_top_p=max(0.82, min(0.99, base_top_p)),
                cand_seed=int(base_seed + 1997),
                response_prefix="",
                extra_rule=str(def_guard_rule),
            )
            guarded_score, guarded_low, guarded_align = _score_candidate(guarded)
            candidates.append(
                {
                    "response": guarded,
                    "score": float(guarded_score),
                    "low_quality": bool(guarded_low),
                    "align": float(guarded_align),
                    "temperature": float(max(0.10, floor_temp)),
                    "top_k": int(max(48, min(160, base_top_k))),
                    "top_p": float(max(0.82, min(0.99, base_top_p))),
                }
            )
            if guarded_score > best_score:
                best_score = float(guarded_score)
                best_low_quality = bool(guarded_low)
                best_response = str(guarded)
                best_align = float(guarded_align)

        for i in range(int(rerank_candidates) - 1):
            if float(temperature) > 0.0:
                cand_temperature = min(0.95, max(floor_temp, float(temperature) * (0.85 + 0.10 * i)))
            else:
                cand_temperature = min(0.95, max(floor_temp, 0.14 + 0.10 * i))
            cand_top_k = max(32, min(192, base_top_k + 16 * i))
            cand_top_p = min(0.99, max(0.78, base_top_p + 0.01 * i))

            cand = _generate_candidate(
                cand_temperature=float(cand_temperature),
                cand_top_k=int(cand_top_k),
                cand_top_p=float(cand_top_p),
                cand_seed=int(base_seed + i + 1),
                response_prefix="",
                extra_rule=str(def_guard_rule if def_term_mismatch else ""),
            )
            cand_score, cand_low, cand_align = _score_candidate(cand)
            candidates.append(
                {
                    "response": cand,
                    "score": float(cand_score),
                    "low_quality": bool(cand_low),
                    "align": float(cand_align),
                    "temperature": float(cand_temperature),
                    "top_k": int(cand_top_k),
                    "top_p": float(cand_top_p),
                }
            )
            if cand_score > best_score:
                best_score = float(cand_score)
                best_low_quality = bool(cand_low)
                best_response = str(cand)
                best_align = float(cand_align)

    response = best_response
    def_consensus = response_consensus_score(candidates=candidates, top_n=4)
    meta["def_consensus"] = float(def_consensus)
    pre_fallback_response = str(response)
    if def_term:
        def_drift = bool(DEF_DRIFT_HINT_RE.search(str(response)))
        def_msg_mismatch = bool(MSG_SERVICE_HINT_RE.search(str(response))) and (not term_seems_messenger(def_term))
        def_noise = bool(definition_noise_signal(response))
        def_low_consensus = bool(zs_mode == "strict" and float(def_consensus) < 0.22)
        def_low_conf = bool(best_low_quality) and float(best_score) < float(zs_def_low_score)
        meta["def_gate"] = {
            "mismatch": bool(def_term_mismatch),
            "drift": bool(def_drift),
            "msg_mismatch": bool(def_msg_mismatch),
            "noise": bool(def_noise),
            "low_consensus": bool(def_low_consensus),
            "low_conf": bool(def_low_conf),
            "score": float(best_score),
            "score_threshold": float(zs_def_low_score),
        }
        need_def_fallback = bool(
            bool(def_term_mismatch) or def_drift or def_msg_mismatch or def_noise or def_low_conf or def_low_consensus
        )
        if need_def_fallback:
            tool_allowed = bool((not disable_retrieval) and agent in {"auto", "triad"})
            if tool_allowed:
                tool_answer, tool_meta = tool_lookup_definition_answer(
                    term=str(def_term),
                    prompt=str(prompt),
                    tool_cache_path=tool_cache,
                    timeout_sec=float(tool_lookup_timeout),
                    allow_web=(not bool(disable_web_tool_lookup)),
                )
                if tool_answer:
                    response = postprocess_response(prompt=prompt, response=str(tool_answer))
                    meta["used_retrieval"] = True
                    meta["tool_lookup_used"] = True
                    meta["safe_fallback"] = False
                    if isinstance(tool_meta, dict):
                        meta["tool_name"] = str(tool_meta.get("tool_name", ""))
                        meta["tool_lang"] = str(tool_meta.get("tool_lang", ""))
                        meta["tool_source_url"] = str(tool_meta.get("tool_source_url", ""))
                        meta["tool_cache_hit"] = bool(tool_meta.get("tool_cache_hit", False))
            if not bool(meta.get("tool_lookup_used", False)):
                meta["fallback_reason"] = "definition_low_confidence"
                meta["raw_response"] = str(pre_fallback_response)
                if bool(force_raw):
                    meta["safe_fallback"] = False
                    meta["fallback_suppressed"] = True
                else:
                    response = build_uncertain_response(prompt=prompt, def_term=def_term)
                    meta["safe_fallback"] = True
    else:
        low_conf_general = bool(
            severe_gibberish_signal(prompt=prompt, response=response, common_syllables=KO_COMMON_SYLLABLES)
            or (bool(best_low_quality) and float(best_score) < float(zs_general_low_score))
        )
        if bool(zs_enable_general_fallback) and low_conf_general and not UNKNOWN_ANSWER_HINT_RE.search(str(response)):
            meta["fallback_reason"] = "general_low_confidence"
            meta["raw_response"] = str(pre_fallback_response)
            if bool(force_raw):
                meta["safe_fallback"] = False
                meta["fallback_suppressed"] = True
            else:
                response = build_uncertain_response(prompt=prompt, def_term="")
                meta["safe_fallback"] = True

    meta["rerank_used"] = bool(use_rerank)
    meta["candidate_count"] = int(len(candidates))
    meta["best_score"] = float(best_score)
    meta["best_low_quality"] = bool(best_low_quality)
    meta["best_align"] = float(best_align)
    if def_term:
        meta["def_term"] = str(def_term)
        meta["def_term_mismatch"] = bool(def_term_mismatch)
    if "fallback_reason" not in meta:
        meta["safe_fallback"] = bool((not meta.get("used_retrieval", False)) and use_rerank and response != candidates[0]["response"])

    return response, bundle, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with local TinyGPT checkpoints.")
    parser.add_argument("--checkpoint", required=True, help="Default checkpoint path")
    parser.add_argument("--code_checkpoint", default="", help="Checkpoint for code-like prompts")
    parser.add_argument("--chat_checkpoint", default="", help="Checkpoint for general chat prompts")
    parser.add_argument("--router", default="auto", choices=["auto", "single"])
    parser.add_argument("--prompt", default="", help="Single-turn prompt. If omitted, interactive mode.")
    parser.add_argument("--system_prompt", default="", help="Optional system instruction")
    parser.add_argument(
        "--system_preset",
        default="none",
        choices=["none", "min_qa_ko"],
        help="Preset system policy used only when --system_prompt is empty",
    )
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--max_input_tokens", type=int, default=39768)
    parser.add_argument("--max_output_tokens", type=int, default=38000)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_k", type=int, default=80)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--min_new_tokens", type=int, default=24)
    parser.add_argument("--disable_ko_guard", action="store_true")
    parser.add_argument("--ko_guard_topk", type=int, default=96)
    parser.add_argument("--ko_guard_rare_penalty", type=float, default=0.9)
    parser.add_argument("--ko_guard_latin_penalty", type=float, default=0.25)
    parser.add_argument("--disable_quality_rerank", action="store_true")
    parser.add_argument("--rerank_candidates", type=int, default=8)
    parser.add_argument("--rerank_temp_floor", type=float, default=0.15)
    parser.add_argument("--rerank_top_k", type=int, default=80)
    parser.add_argument("--rerank_top_p", type=float, default=0.92)
    parser.add_argument("--zero_shot_mode", default="off", choices=["off", "balanced", "strict"])
    parser.add_argument("--history_turns", type=int, default=6)
    parser.add_argument("--show_route", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--quantize_int8", action="store_true")
    parser.add_argument("--disable_retrieval", action="store_true")
    parser.add_argument("--tool_cache_path", default=str(DEFAULT_TOOL_CACHE_PATH))
    parser.add_argument("--tool_lookup_timeout", type=float, default=4.0)
    parser.add_argument("--disable_web_tool_lookup", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--session_id", default="default")
    parser.add_argument("--memory_path", default=str(DEFAULT_MEMORY_PATH))
    parser.add_argument("--no_persist_memory", action="store_true")
    parser.add_argument("--budget_mode", default="auto", choices=["auto", "low", "balanced", "high"])
    parser.add_argument("--latency_mode", default="auto", choices=["auto", "fast", "quality"])
    parser.add_argument("--agent_mode", default="off", choices=["off", "auto", "triad"])
    parser.add_argument("--heuristic_mode", default="off", choices=["off", "legacy"])
    parser.add_argument("--continual_log_path", default="data/continual_buffer_v2.jsonl")
    parser.add_argument("--disable_continual_log", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--stream_chunk_size", type=int, default=24)
    parser.add_argument("--emit_meta", action="store_true")
    parser.add_argument("--force_raw", action="store_true", help="show raw model output even if safe fallback would trigger")
    return parser.parse_args()


def interactive_chat(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    router = ModelRouter(
        device=device,
        checkpoint=str(args.checkpoint),
        use_ema=bool(args.use_ema),
        router_mode=str(args.router),
        code_checkpoint=str(args.code_checkpoint),
        chat_checkpoint=str(args.chat_checkpoint),
        quantize_int8=bool(args.quantize_int8),
    )
    memory_path = Path(str(args.memory_path))
    session_id = str(args.session_id or "default").strip() or "default"
    persist_memory = not bool(args.no_persist_memory)
    profile = load_memory_profile(memory_path=memory_path, session_id=session_id) if persist_memory else {}
    history: List[Tuple[str, str]] = []
    effective_system_prompt = resolve_system_prompt(
        user_system_prompt=str(args.system_prompt),
        system_preset=str(args.system_preset),
    )

    safe_print("Interactive mode. Type /exit to quit.")
    while True:
        try:
            user_prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            safe_print("")
            break
        if not user_prompt:
            continue
        if user_prompt.lower() in {"exit", "quit", "/exit", "/quit"}:
            break

        response, bundle, meta = run_one_turn(
            router=router,
            user_prompt=user_prompt,
            history=history[-int(args.history_turns) :],
            system_prompt=str(effective_system_prompt),
            max_new_tokens=int(args.max_new_tokens),
            max_input_tokens=int(args.max_input_tokens),
            max_output_tokens=int(args.max_output_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            min_new_tokens=int(args.min_new_tokens),
            disable_ko_guard=bool(args.disable_ko_guard),
            ko_guard_topk=int(args.ko_guard_topk),
            ko_guard_rare_penalty=float(args.ko_guard_rare_penalty),
            ko_guard_latin_penalty=float(args.ko_guard_latin_penalty),
            disable_retrieval=bool(args.disable_retrieval),
            disable_quality_rerank=bool(args.disable_quality_rerank),
            rerank_candidates=int(args.rerank_candidates),
            rerank_temp_floor=float(args.rerank_temp_floor),
            rerank_top_k=int(args.rerank_top_k),
            rerank_top_p=float(args.rerank_top_p),
            zero_shot_mode=str(args.zero_shot_mode),
            agent_mode=str(args.agent_mode),
            tool_cache_path=str(args.tool_cache_path),
            tool_lookup_timeout=float(args.tool_lookup_timeout),
            disable_web_tool_lookup=bool(args.disable_web_tool_lookup),
            heuristic_mode=str(args.heuristic_mode),
            force_raw=bool(args.force_raw),
        )

        if args.show_route:
            safe_print(f"[route={bundle.label}] {bundle.checkpoint}")
        if args.stream:
            stream_print_text(response, chunk_size=max(1, int(args.stream_chunk_size)))
        else:
            safe_print(response)
        if args.emit_meta:
            safe_print(json.dumps(meta, ensure_ascii=False))

        history.append((user_prompt, response))
        if len(history) > int(args.history_turns):
            history = history[-int(args.history_turns) :]

        if persist_memory:
            profile["conversation_count"] = int(profile.get("conversation_count", 0)) + 1
            save_memory_profile(memory_path=memory_path, session_id=session_id, profile=profile)


def single_turn(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    router = ModelRouter(
        device=device,
        checkpoint=str(args.checkpoint),
        use_ema=bool(args.use_ema),
        router_mode=str(args.router),
        code_checkpoint=str(args.code_checkpoint),
        chat_checkpoint=str(args.chat_checkpoint),
        quantize_int8=bool(args.quantize_int8),
    )

    effective_system_prompt = resolve_system_prompt(
        user_system_prompt=str(args.system_prompt),
        system_preset=str(args.system_preset),
    )

    response, bundle, meta = run_one_turn(
        router=router,
        user_prompt=str(args.prompt),
        history=[],
        system_prompt=str(effective_system_prompt),
        max_new_tokens=int(args.max_new_tokens),
        max_input_tokens=int(args.max_input_tokens),
        max_output_tokens=int(args.max_output_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        min_new_tokens=int(args.min_new_tokens),
        disable_ko_guard=bool(args.disable_ko_guard),
        ko_guard_topk=int(args.ko_guard_topk),
        ko_guard_rare_penalty=float(args.ko_guard_rare_penalty),
        ko_guard_latin_penalty=float(args.ko_guard_latin_penalty),
        disable_retrieval=bool(args.disable_retrieval),
        disable_quality_rerank=bool(args.disable_quality_rerank),
        rerank_candidates=int(args.rerank_candidates),
        rerank_temp_floor=float(args.rerank_temp_floor),
        rerank_top_k=int(args.rerank_top_k),
        rerank_top_p=float(args.rerank_top_p),
        zero_shot_mode=str(args.zero_shot_mode),
        agent_mode=str(args.agent_mode),
        tool_cache_path=str(args.tool_cache_path),
        tool_lookup_timeout=float(args.tool_lookup_timeout),
        disable_web_tool_lookup=bool(args.disable_web_tool_lookup),
        heuristic_mode=str(args.heuristic_mode),
        force_raw=bool(args.force_raw),
    )
    if args.show_route:
        safe_print(f"[route={bundle.label}] {bundle.checkpoint}")
    if args.stream:
        stream_print_text(response, chunk_size=max(1, int(args.stream_chunk_size)))
    else:
        safe_print(response)
    if args.emit_meta:
        safe_print(json.dumps(meta, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if str(args.prompt or "").strip():
        single_turn(args)
    else:
        interactive_chat(args)


if __name__ == "__main__":
    main()
