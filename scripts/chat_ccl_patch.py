from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_slm import ByteTokenizer, top_k_top_p_filtering
from train_ccl_compile import (
    apply_lora_to_tinygpt,
    build_model_from_checkpoint,
    load_lora_patch_state,
    parse_patch_targets,
)


def set_lora_patch_scale(model: torch.nn.Module, scale: float) -> int:
    s = float(scale)
    count = 0
    for mod in model.modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and hasattr(mod, "scale"):
            try:
                base_scale = float(getattr(mod, "_base_scale"))
            except Exception:
                base_scale = float(getattr(mod, "scale"))
                setattr(mod, "_base_scale", base_scale)
            setattr(mod, "scale", base_scale * s)
            count += 1
    return count


def safe_print(text: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        fixed = text.encode(enc, errors="replace").decode(enc, errors="replace")
        print(fixed, flush=True)


def resolve_device(name: str) -> torch.device:
    key = str(name).strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def build_chat_prompt(
    user_prompt: str,
    history: Sequence[Tuple[str, str]],
    history_turns: int,
    system_prompt: str,
) -> str:
    chunks: List[str] = []
    sys_text = str(system_prompt or "").strip()
    if sys_text:
        chunks.append(f"(System) {sys_text}")
    for u, a in list(history)[-max(0, int(history_turns)) :]:
        chunks.append(f"User: {u}\nAssistant: {a}")
    chunks.append(f"User: {str(user_prompt).strip()}\nAssistant:")
    instruction = "\n\n".join(chunks)
    return f"### Instruction\n{instruction}\n\n### Response\n"


def extract_response(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt) :].strip()
    marker = "### Response"
    idx = full_text.rfind(marker)
    if idx >= 0:
        return full_text[idx + len(marker) :].lstrip(": \n\t").strip()
    return full_text.strip()


def _safe_eval_expr(expr: str) -> float | None:
    src = str(expr or "").replace("^", "**").strip()
    if not src:
        return None
    if re.search(r"[^0-9\+\-\*\/\(\)\.\s]", src):
        return None
    try:
        val = eval(src, {"__builtins__": {}}, {})
    except Exception:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _fmt_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _solve_linear_equation(prompt: str) -> str:
    # Handles forms like 3x + 5 = 2x + 17
    q = str(prompt or "").replace(" ", "")
    m = re.search(r"([\-]?\d*)x([\+\-]\d+)?=([\-]?\d*)x([\+\-]\d+)?", q)
    if not m:
        return ""
    a = m.group(1)
    b = m.group(2)
    c = m.group(3)
    d = m.group(4)

    def _coef(s: str | None, default_if_empty: int = 0, for_x: bool = False) -> int:
        if s is None or s == "":
            return 1 if for_x else default_if_empty
        if s == "+":
            return 1
        if s == "-":
            return -1
        return int(s)

    A = _coef(a, for_x=True)
    B = _coef(b, default_if_empty=0)
    C = _coef(c, for_x=True)
    D = _coef(d, default_if_empty=0)
    denom = A - C
    if denom == 0:
        return "No unique solution."
    x = (D - B) / float(denom)
    if abs(x - round(x)) < 1e-9:
        x = int(round(x))
    return f"x = {x}"


def _solve_ratio_question(prompt: str) -> str:
    q = str(prompt or "").lower()
    if "ratio" not in q:
        return ""

    # ex) 2:3 ratio of sugar to flour
    m = re.search(r"(\d+)\s*:\s*(\d+)", q)
    if not m:
        return ""
    a = int(m.group(1))
    b = int(m.group(2))

    flour_candidates: List[float] = []
    for mm in re.finditer(r"flour\s*(?:is|=)?\s*(\d+(?:\.\d+)?)\s*g", q):
        flour_candidates.append(float(mm.group(1)))
    for mm in re.finditer(r"(\d+(?:\.\d+)?)\s*g(?:rams?)?\s+of\s+flour", q):
        flour_candidates.append(float(mm.group(1)))
    if not flour_candidates:
        return ""

    flour = flour_candidates[0]

    # Handle spill/loss cases like "12 g of flour spills out".
    spill = 0.0
    mspill_after = re.search(
        r"(?:spill|spills|spilled|lost|loss|leak)(?:\s+out)?(?:\s+of)?\s+(\d+(?:\.\d+)?)\s*g",
        q,
    )
    mspill_before = re.search(
        r"(\d+(?:\.\d+)?)\s*g(?:rams?)?(?:\s+of\s+flour)?\s+(?:spill|spills|spilled|lost|loss|leak)",
        q,
    )
    if mspill_after:
        spill = float(mspill_after.group(1))
    elif mspill_before:
        spill = float(mspill_before.group(1))

    flour_effective = flour
    if spill > 0.0 and flour > spill:
        flour_effective = flour - spill

    sugar = flour_effective * (float(a) / float(max(1, b)))
    return f"Sugar needed: {_fmt_number(sugar)}g."


def _solve_logic_question(prompt: str) -> str:
    q = str(prompt or "").lower()
    if "all zibs are wugs" in q and "some wugs are loms" in q:
        return "No. It does not follow that some Zibs are Loms."
    if (
        "either a or b" in q
        and "if a then c" in q
        and ("c is false" in q or "c=false" in q or "c is not true" in q)
    ):
        return "A must be false, and B must be true."
    if (
        "either a or b is true" in q
        and "if a then c" in q
        and ("c is false" in q or "c=false" in q)
    ):
        return "A must be false, and B must be true."
    return ""


def _solve_counterexample_question(prompt: str) -> str:
    q = str(prompt or "").lower()
    if (
        "smallest counterexample" in q
        and "divisible by 6" in q
        and "divisible by 4" in q
    ):
        return "6"
    return ""


def _solve_arithmetic_question(prompt: str) -> str:
    q = str(prompt or "")
    low = q.lower()
    if any(k in low for k in ["either a or b", "counterexample", "does it follow", "claim:"]):
        return ""

    # Heuristic for "1719 without calculator" -> 17*19
    m2 = re.search(r"\b(\d{4})\b", low)
    if m2 and ("without a calculator" in low or "show your steps" in low):
        s = m2.group(1)
        a = int(s[:2])
        b = int(s[2:])
        prod = a * b
        return f"{a} x {b} = {prod}"

    # Standard arithmetic expression in prompt.
    m = re.search(r"([0-9\(\)][0-9\s\+\-\*\/\(\)\.]{2,})", q)
    if m:
        expr = m.group(1).strip()
        val = _safe_eval_expr(expr)
        if val is not None:
            if abs(val - round(val)) < 1e-9:
                return str(int(round(val)))
            return str(val)
    return ""


def is_code_query(text: str) -> bool:
    q = str(text or "").lower()
    if not q:
        return False
    if "code" in q or "snippet" in q:
        return True
    if re.search(r"\b(exception|traceback|stack trace|error|bug|fix|refactor|rewrite|idiomatic)\b", q):
        return True
    if "optimize" in q and "memory" in q:
        return True
    langs = ["python", "javascript", "typescript", "kotlin", "java", "rust", "go"]
    intents = ["rewrite", "idiomatic", "optimize", "difference", "differences", "between", "compare"]
    if any(lang in q for lang in langs) and any(word in q for word in intents):
        return True
    return False


def is_creative_query(text: str) -> bool:
    q = str(text or "").lower()
    if not q:
        return False
    markers = [
        "write",
        "create",
        "compose",
        "poem",
        "acrostic",
        "story",
        "lyrics",
        "creative",
        "haiku",
        "sonnet",
        "시 ",
        "시를",
        "아크로스틱",
    ]
    return any(m in q for m in markers)


def extract_requested_line_count(text: str) -> int | None:
    q = str(text or "").lower()
    if not q:
        return None

    word_to_num = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    for w, n in word_to_num.items():
        if re.search(rf"\b{w}\s*-\s*line\b", q) or re.search(rf"\b{w}\s+lines?\b", q):
            return n

    m = re.search(r"\b(\d+)\s*-\s*line\b", q)
    if not m:
        m = re.search(r"\b(\d+)\s+lines?\b", q)
    if m:
        return max(1, min(20, int(m.group(1))))
    return None


def extract_acrostic_target(text: str) -> str:
    q = str(text or "").strip()
    low = q.lower()
    if "acrostic" not in low:
        return ""

    m = re.search(r"(?:using|for|of)\s+([A-Za-z]{2,16})\b", q)
    if m:
        return m.group(1).upper()
    return ""


def _creative_structure_ok(prompt: str, response: str) -> bool:
    if not is_creative_query(prompt):
        return True
    lines = [ln.strip() for ln in str(response or "").splitlines() if ln.strip()]
    if not lines:
        return False

    target_lines = extract_requested_line_count(prompt)
    if target_lines is not None and len(lines) != target_lines:
        return False

    acrostic_target = extract_acrostic_target(prompt)
    if acrostic_target:
        if target_lines is None and len(lines) != len(acrostic_target):
            return False
        need = min(len(lines), len(acrostic_target))
        for i in range(need):
            first_alpha = re.search(r"[A-Za-z]", lines[i])
            if not first_alpha:
                return False
            if lines[i][first_alpha.start()].upper() != acrostic_target[i]:
                return False
    return True


def _solve_code_question(prompt: str) -> str:
    q = str(prompt or "").lower()
    if "minimal reproducible example" in q or re.search(r"\bmre\b", q):
        return (
            "MRE template:\n"
            "1) Minimal code (10-30 lines) that still fails.\n"
            "2) Exact input and expected vs actual output.\n"
            "3) Runtime/version info.\n"
            "4) Full error/traceback."
        )
    if "race condition" in q or "concurrency issue" in q:
        return "Please paste the snippet and runtime context (threads/async/processes) to confirm the race condition."
    if "exception" in q or "error" in q:
        return (
            "Most likely cause: invalid input/state not handled before the failing line.\n"
            "Minimal fix: add a guard for null/empty/out-of-range input right before that line."
        )
    if "optimize" in q and "memory" in q:
        return (
            "Use streaming/chunk processing and avoid full copies in memory.\n"
            "Trade-off: lower peak RAM, but slightly more code complexity and often more I/O overhead."
        )
    if "rewrite" in q and ("idiomatic" in q or "python" in q or "javascript" in q or "kotlin" in q):
        return "Pick Python: use small pure functions, clear names, list/dict comprehensions, and early returns."
    return ""


def _solve_compare_question(prompt: str) -> str:
    q = str(prompt or "").lower()
    if "rust" in q and "go" in q and ("difference" in q or "differences" in q or "between" in q):
        return (
            "Rust: stronger memory safety guarantees at compile time, typically higher complexity.\n"
            "Go: simpler language/tooling with fast development, but less strict low-level control."
        )
    return ""


def _solve_synonyms_question(prompt: str) -> str:
    q = str(prompt or "").lower()
    if "synonym" not in q:
        return ""
    m = re.search(r"list\s+(\d+)\s+synonyms?\s+for\s+([a-z\-]+)", q)
    if m:
        want = max(1, min(10, int(m.group(1))))
        word = m.group(2)
    else:
        m2 = re.search(r"synonyms?\s+for\s+([a-z\-]+)", q)
        if not m2:
            return ""
        want = 5
        word = m2.group(1)

    thesaurus: Dict[str, List[str]] = {
        "fast": ["quick", "rapid", "swift", "speedy", "brisk", "prompt", "fleet"],
        "slow": ["sluggish", "lethargic", "gradual", "unhurried", "plodding"],
        "smart": ["clever", "bright", "intelligent", "sharp", "astute"],
    }
    cands = thesaurus.get(word, [])
    if len(cands) < want:
        return ""
    pick = cands[:want]
    return "\n".join(f"{i+1}. {w}" for i, w in enumerate(pick))


def _solve_fraction_decimal_question(prompt: str) -> str:
    q = str(prompt or "").lower()
    if "decimal" not in q:
        return ""
    mfrac = re.search(r"(-?\d+)\s*/\s*(-?\d+)", q)
    if not mfrac:
        return ""
    mplaces = re.search(r"to\s+(\d+)\s+places?", q)
    if not mplaces:
        mplaces = re.search(r"(\d+)\s+decimal\s+places?", q)
    if not mplaces:
        return ""
    num = int(mfrac.group(1))
    den = int(mfrac.group(2))
    if den == 0:
        return "Undefined (division by zero)."
    places = max(0, min(12, int(mplaces.group(1))))
    val = num / float(den)
    return f"{val:.{places}f}"


def _solve_discount_original_question(prompt: str) -> str:
    q = str(prompt or "").lower()
    if "discount" not in q:
        return ""
    if "original" not in q and "before discount" not in q:
        return ""
    mp = re.search(r"(\d+(?:\.\d+)?)\s*%\s*discount", q)
    if not mp:
        return ""
    pct = float(mp.group(1))
    if pct >= 100.0:
        return ""
    mprice = re.search(r"(?:price|cost|total)\s*(?:is|=|becomes|become|to|at|makes\s+the\s+price)?\s*(\d+(?:\.\d+)?)", q)
    discounted = float(mprice.group(1)) if mprice else None
    if discounted is None:
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", q)]
        for n in nums[::-1]:
            if abs(n - pct) > 1e-9:
                discounted = n
                break
    if discounted is None:
        return ""
    original = discounted / (1.0 - pct / 100.0)
    return f"Original price: {_fmt_number(original)}."


def _solve_latest_version_question(prompt: str) -> str:
    q = str(prompt or "").strip()
    low = q.lower()
    if "latest version" not in low:
        return ""
    if re.search(r"\bversion of x\b", low) or re.search(r"\bof x\b", low):
        return (
            "Replace 'X' with the exact product/package name. "
            "I cannot verify live latest versions or links in local offline mode."
        )
    if "link" in low:
        return (
            "Give the exact package name (e.g., Python, Node.js, PyTorch). "
            "Then I can provide the official release links to check."
        )
    return ""


def _solve_acrostic_question(prompt: str) -> str:
    q = str(prompt or "").strip()
    low = q.lower()
    if "acrostic" not in low:
        return ""

    target = ""
    if "your name" in low or "using your name" in low:
        target = "SPEAR"
    if not target:
        m = re.search(r"(?:for|of|using)\s+([a-z]{3,12})", low)
        if m:
            target = m.group(1).upper()
    if not target:
        target = "SPEAR"

    templates = [
        "Stays steady through hard problems.",
        "Plans clearly before acting.",
        "Explains decisions with concrete details.",
        "Adapts quickly when requirements change.",
        "Repeats and refines until it works.",
        "Builds reliable results under constraints.",
    ]
    lines: List[str] = []
    for i, ch in enumerate(target):
        phrase = templates[i % len(templates)]
        lines.append(f"{ch} - {phrase}")
    return "\n".join(lines)


def _auxiliary_reasoning_answer(prompt: str) -> str:
    # User requested no hardcoded content answers.
    # Keep this hook disabled; rely on model generation + retry only.
    return ""


def _looks_low_quality_response(prompt: str, response: str) -> bool:
    r = str(response or "").strip()
    if not r:
        return True
    if looks_repetition_spam(r):
        return True
    if not is_math_query(prompt) and looks_math_spam(r):
        return True

    low = r.lower()
    if "i understand your request" in low or "ask once more" in low:
        return True
    if not _creative_structure_ok(prompt, r):
        return True
    return False


def deterministic_answer(prompt: str) -> str:
    # User requested no hardcoded content answers.
    _ = prompt
    return ""


def is_math_query(text: str) -> bool:
    q = str(text or "").strip().lower()
    if not q:
        return False
    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", q):
        return True
    return any(k in q for k in ["calculate", "equation", "math", "solve", "compute"])

def is_name_query(text: str) -> bool:
    q = str(text or "").strip().lower()
    if not q:
        return False
    # Exclude creative tasks that reference the assistant's name as material.
    if is_creative_query(q) and (
        "your name" in q
        or "using your name" in q
        or "with your name" in q
        or "이름으로" in q
    ):
        return False
    if re.search(r"^\s*what(?:'s|\s+is)\s+your\s+name\??\s*$", q):
        return True
    if re.search(r"^\s*your\s+name\??\s*$", q):
        return True
    if any(k == q for k in ["너의 이름", "네 이름", "니 이름", "이름은", "너의 이름은", "네 이름은", "니 이름은"]):
        return True
    if re.search(r"^\s*(너|네|니)\s*이름(은|이)?\s*\??\s*$", q):
        return True
    return False

def is_greeting(text: str) -> bool:
    q = str(text or "").strip().lower()
    if not q:
        return False
    words = set(re.findall(r"[a-z]+", q))
    if words.intersection({"hi", "hello", "hey", "yo"}):
        return True
    return q in {"ㅎㅇ", "안녕", "안녕하세요", "하이"}

def looks_math_spam(response: str) -> bool:
    r = str(response or "").strip()
    if not r:
        return True
    if len(r) < 8:
        return True
    math_hits = 0
    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", r):
        math_hits += 1
    if re.search(r"(result|therefore|calculation)", r.lower()):
        math_hits += 1
    if "=" in r and re.search(r"\d", r):
        math_hits += 1
    if math_hits >= 2:
        return True
    return False

def looks_repetition_spam(response: str) -> bool:
    r = str(response or "").strip().lower()
    if not r:
        return True
    if re.search(r"(in the\s+){6,}", r):
        return True
    toks = re.findall(r"[a-z0-9_]+", r)
    if len(toks) >= 16:
        diversity = len(set(toks)) / float(len(toks))
        if diversity < 0.28:
            return True
    if re.search(r"(.)\1{8,}", r):
        return True
    return False


def safe_fallback_reply(user_prompt: str) -> str:
    # Pure-learning mode: do not return hardcoded fallback text.
    _ = user_prompt
    return ""

def run_one_turn(
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    device: torch.device,
    user_prompt: str,
    history: Sequence[Tuple[str, str]],
    history_turns: int,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    det = deterministic_answer(user_prompt)
    if det:
        return det
    creative = is_creative_query(user_prompt)
    line_target = extract_requested_line_count(user_prompt)
    acrostic_target = extract_acrostic_target(user_prompt)

    effective_prompt = str(user_prompt).strip()
    if creative:
        constraints: List[str] = [
            "Write clean, natural text.",
            "Do not number lines.",
        ]
        if line_target is not None:
            constraints.append(f"Use exactly {line_target} lines.")
        if acrostic_target:
            constraints.append(
                f"For the acrostic, start each line with consecutive letters of '{acrostic_target}'."
            )
        effective_prompt = effective_prompt + "\n\nConstraints:\n- " + "\n- ".join(constraints)

    token_budget = int(max_new_tokens)
    if creative and line_target is not None:
        token_budget = min(token_budget, 96)

    if creative:
        attempt_specs: List[Tuple[float, int, float, str]] = [
            (max(float(temperature), 0.65), max(int(top_k), 50), max(float(top_p), 0.92), ""),
            (max(float(temperature), 0.80), max(int(top_k), 80), max(float(top_p), 0.96), ""),
            (
                max(float(temperature), 0.90),
                max(int(top_k), 120),
                max(float(top_p), 0.98),
                "Be concise and coherent. Follow the constraints exactly.",
            ),
        ]
    else:
        attempt_specs = [
            (float(temperature), int(top_k), float(top_p), ""),
            (
                max(float(temperature), 0.2),
                int(top_k),
                float(top_p),
                "Answer directly. For math/logic, show short reasoning steps. Do not repeat tokens.",
            ),
        ]

    last_response = ""
    for temp_i, topk_i, topp_i, extra_instruction in attempt_specs:
        run_prompt = effective_prompt
        if extra_instruction:
            run_prompt = run_prompt + "\n\n" + extra_instruction
        prompt_text = build_chat_prompt(
            user_prompt=run_prompt,
            history=history,
            history_turns=history_turns,
            system_prompt=system_prompt,
        )
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt_text,
            max_new_tokens=token_budget,
            temperature=temp_i,
            top_k=topk_i,
            top_p=topp_i,
            repetition_penalty=repetition_penalty,
        )
        last_response = str(response or "").strip()
        if not _looks_low_quality_response(user_prompt, response):
            return response
    # Keep output purely model-generated even when quality checks fail.
    return last_response or safe_fallback_reply(user_prompt)


def apply_repetition_penalty(logits: torch.Tensor, token_ids: List[int], penalty: float) -> torch.Tensor:
    if float(penalty) <= 1.0 or not token_ids:
        return logits
    out = logits.clone()
    for idx in set(int(t) for t in token_ids):
        if 0 <= idx < out.numel():
            val = out[idx]
            out[idx] = val * float(penalty) if val < 0 else val / float(penalty)
    return out


def apply_no_repeat_ngram_block(logits: torch.Tensor, token_ids: List[int], ngram_size: int = 3) -> torch.Tensor:
    n = int(ngram_size)
    if n <= 1:
        return logits
    if len(token_ids) < (n - 1):
        return logits

    prefix = tuple(int(t) for t in token_ids[-(n - 1) :])
    banned: set[int] = set()
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


@torch.no_grad()
def generate_response(
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    model.eval()
    tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    seq_len = int(getattr(model, "seq_len", 384))

    for _ in range(max(1, int(max_new_tokens))):
        x = torch.tensor([tokens[-seq_len:]], dtype=torch.long, device=device)
        logits, _ = model(x, targets=None)
        next_logits = logits[0, -1]
        next_logits = apply_repetition_penalty(next_logits, tokens, penalty=float(repetition_penalty))
        next_logits = apply_no_repeat_ngram_block(next_logits, tokens, ngram_size=3)

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


def resolve_base_checkpoint_path(
    patch_payload: Dict[str, Any],
    patch_path: Path,
    override: str,
) -> Path:
    if str(override or "").strip():
        p = Path(str(override).strip())
        if p.exists():
            return p
        raise FileNotFoundError(f"base checkpoint not found: {p}")

    base_raw = str(patch_payload.get("base_checkpoint", "")).strip()
    if not base_raw:
        raise RuntimeError("patch payload missing base_checkpoint")

    cand = Path(base_raw)
    if cand.exists():
        return cand

    repo_root = SCRIPT_DIR.parent
    cand_repo = repo_root / base_raw
    if cand_repo.exists():
        return cand_repo

    cand_patch_rel = patch_path.parent / base_raw
    if cand_patch_rel.exists():
        return cand_patch_rel

    raise FileNotFoundError(f"base checkpoint not found from patch reference: {base_raw}")


def load_ccl_model(
    patch_path: Path,
    device: torch.device,
    base_checkpoint_override: str,
    patch_scale: float,
) -> Tuple[torch.nn.Module, ByteTokenizer, Dict[str, Any], Path]:
    if not patch_path.exists():
        raise FileNotFoundError(f"patch not found: {patch_path}")
    payload = torch.load(patch_path, map_location="cpu", weights_only=False)
    fmt = str(payload.get("format", "")).strip().lower()
    if fmt != "ccl_patch_v1":
        raise RuntimeError(f"unsupported patch format: {fmt}")

    base_ckpt = resolve_base_checkpoint_path(payload, patch_path=patch_path, override=base_checkpoint_override)
    model, tokenizer, _ = build_model_from_checkpoint(base_ckpt, device=device)

    args = payload.get("args", {}) or {}
    top_layers = int(args.get("patch_top_layers", 4))
    targets = parse_patch_targets(str(args.get("patch_targets", "qkv,proj,mlp_in,mlp_out")))
    lora_rank = int(args.get("lora_rank", 8))
    lora_alpha = float(args.get("lora_alpha", 16.0))
    lora_dropout = float(args.get("lora_dropout", 0.0))

    apply_lora_to_tinygpt(
        model=model,
        top_layers=top_layers,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        targets=targets,
    )
    load_lora_patch_state(model=model, state=payload.get("patch_state", {}))
    set_lora_patch_scale(model=model, scale=float(patch_scale))
    model.to(device)
    model.eval()
    return model, tokenizer, payload, base_ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with CCL patch + local base checkpoint")
    parser.add_argument("--patch", default="artifacts_ccl2_compile/ccl_patch_best.pt")
    parser.add_argument("--base_checkpoint", default="", help="optional override for patch base checkpoint")
    parser.add_argument("--patch_scale", type=float, default=0.35, help="LoRA patch strength scale")
    parser.add_argument("--prompt", default="", help="single-turn prompt; if omitted interactive mode")
    parser.add_argument(
        "--system_prompt",
        default="Answer directly and clearly. For math/logic, show brief reasoning steps.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--history_turns", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.12)
    parser.add_argument("--show_meta", action="store_true")
    return parser.parse_args()


def run_single_turn(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    patch_path = Path(str(args.patch))
    model, tokenizer, payload, base_ckpt = load_ccl_model(
        patch_path=patch_path,
        device=device,
        base_checkpoint_override=str(args.base_checkpoint),
        patch_scale=float(args.patch_scale),
    )
    answer = run_one_turn(
        model=model,
        tokenizer=tokenizer,
        device=device,
        user_prompt=str(args.prompt),
        history=[],
        history_turns=int(args.history_turns),
        system_prompt=str(args.system_prompt),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
    )
    safe_print(answer)
    if args.show_meta:
        safe_print(
            str(
                {
                    "patch": str(patch_path),
                    "base_checkpoint": str(base_ckpt),
                    "best_hard_pass_rate": payload.get("best_hard_pass_rate"),
                    "best_step": payload.get("best_step"),
                    "patch_scale": float(args.patch_scale),
                }
            )
        )


def run_interactive(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    patch_path = Path(str(args.patch))
    model, tokenizer, payload, base_ckpt = load_ccl_model(
        patch_path=patch_path,
        device=device,
        base_checkpoint_override=str(args.base_checkpoint),
        patch_scale=float(args.patch_scale),
    )

    safe_print("Interactive CCL mode. Type /exit to quit.")
    if args.show_meta:
        safe_print(
            f"[patch={patch_path}] [base={base_ckpt}] "
            f"[best_hard_pass={payload.get('best_hard_pass_rate')}] "
            f"[patch_scale={float(args.patch_scale):.3f}]"
        )

    history: List[Tuple[str, str]] = []
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

        answer = run_one_turn(
            model=model,
            tokenizer=tokenizer,
            device=device,
            user_prompt=user_prompt,
            history=history,
            history_turns=int(args.history_turns),
            system_prompt=str(args.system_prompt),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
        )
        safe_print(answer)
        history.append((user_prompt, answer))
        if len(history) > int(args.history_turns):
            history = history[-int(args.history_turns) :]


def main() -> None:
    args = parse_args()
    if str(args.prompt or "").strip():
        run_single_turn(args)
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()

