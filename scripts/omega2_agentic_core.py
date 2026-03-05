from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from train_ccl_compile import (
    SpecCase,
    build_specbook_from_data,
    case_target_nll,
    default_nll_threshold,
    generate_response,
    load_specbook_from_jsonl,
    normalize_tier,
)


def simple_tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_\\uac00-\\ud7a3]{2,}", str(text or "").lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def normalize_text(s: str) -> str:
    return re.sub(r"\\s+", " ", str(s or "")).strip().lower()


@dataclass
class PatchRecord:
    patch_id: str
    spec_id: str
    tier: str
    prompt: str
    target: str
    rule_text: str
    keywords: List[str]
    created_step: int
    uses: int
    status: str
    evidence: Dict[str, Any]

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "PatchRecord":
        return cls(
            patch_id=str(row.get("patch_id", "")).strip(),
            spec_id=str(row.get("spec_id", "")).strip(),
            tier=normalize_tier(str(row.get("tier", "soft"))),
            prompt=str(row.get("prompt", "")),
            target=str(row.get("target", "")),
            rule_text=str(row.get("rule_text", "")),
            keywords=[str(x) for x in row.get("keywords", []) if str(x).strip()],
            created_step=int(row.get("created_step", 0)),
            uses=int(row.get("uses", 0)),
            status=str(row.get("status", "active")).strip().lower() or "active",
            evidence=row.get("evidence", {}) if isinstance(row.get("evidence", {}), dict) else {},
        )

    def to_row(self) -> Dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "spec_id": self.spec_id,
            "tier": self.tier,
            "prompt": self.prompt,
            "target": self.target,
            "rule_text": self.rule_text,
            "keywords": self.keywords,
            "created_step": int(self.created_step),
            "uses": int(self.uses),
            "status": self.status,
            "evidence": self.evidence,
        }


class PatchDB:
    def __init__(self, path: Path, novelty_threshold: float) -> None:
        self.path = Path(path)
        self.novelty_threshold = float(novelty_threshold)
        self.records: List[PatchRecord] = []
        self._next_id = 1

    def load(self) -> None:
        self.records = []
        if not self.path.exists():
            self._next_id = 1
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                rec = PatchRecord.from_row(row)
                if rec.patch_id:
                    self.records.append(rec)
        max_id = 0
        for rec in self.records:
            m = re.search(r"(\\d+)$", rec.patch_id)
            if m:
                max_id = max(max_id, int(m.group(1)))
        self._next_id = max(1, max_id + 1)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec.to_row(), ensure_ascii=False) + "\\n")

    def active_records(self) -> List[PatchRecord]:
        return [r for r in self.records if r.status == "active"]

    def active_count(self) -> int:
        return len(self.active_records())

    def _is_novel(self, keywords: set[str], rule_text: str) -> bool:
        cand_rule = normalize_text(rule_text)
        for rec in self.active_records():
            rec_k = set(rec.keywords)
            if jaccard(keywords, rec_k) >= float(self.novelty_threshold):
                return False
            if cand_rule and normalize_text(rec.rule_text) == cand_rule:
                return False
        return True

    def add_patch(
        self,
        spec: SpecCase,
        rule_text: str,
        keywords: Sequence[str],
        created_step: int,
        evidence: Dict[str, Any],
    ) -> Tuple[bool, PatchRecord | None]:
        k = {str(x).strip().lower() for x in keywords if str(x).strip()}
        if not k:
            k = simple_tokens(spec.prompt) or {"rule"}
        if not self._is_novel(k, rule_text):
            return False, None
        rec = PatchRecord(
            patch_id=f"patch_{self._next_id:07d}",
            spec_id=str(spec.spec_id),
            tier=normalize_tier(spec.tier),
            prompt=str(spec.prompt),
            target=str(spec.target),
            rule_text=str(rule_text).strip(),
            keywords=sorted(k),
            created_step=int(created_step),
            uses=0,
            status="active",
            evidence=dict(evidence or {}),
        )
        self.records.append(rec)
        self._next_id += 1
        return True, rec

    def retrieve(self, prompt: str, top_k: int) -> List[PatchRecord]:
        qk = simple_tokens(prompt)
        scored: List[Tuple[float, PatchRecord]] = []
        for rec in self.active_records():
            rk = set(rec.keywords)
            ov = jaccard(qk, rk)
            if ov <= 0.0:
                continue
            usage_bonus = min(0.08, 0.01 * math.log1p(max(0, rec.uses)))
            scored.append((ov + usage_bonus, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[: max(0, int(top_k))]]

    def mark_used(self, patch_ids: Sequence[str]) -> None:
        pid_set = {str(x) for x in patch_ids}
        if not pid_set:
            return
        for rec in self.records:
            if rec.patch_id in pid_set:
                rec.uses = int(rec.uses) + 1

    def flush_specs(self, spec_ids: Iterable[str]) -> int:
        sid = {str(x) for x in spec_ids}
        changed = 0
        for rec in self.records:
            if rec.status == "active" and rec.spec_id in sid:
                rec.status = "flushed"
                changed += 1
        return changed


def load_specs(spec_path: Path, data_path: Path, max_specs: int, seed: int) -> Tuple[List[SpecCase], str]:
    if spec_path.exists() and spec_path.stat().st_size > 0:
        return load_specbook_from_jsonl(spec_path, max_specs=max_specs, seed=seed), str(spec_path)
    if not data_path.exists():
        raise RuntimeError(f"spec_path not found and data_path missing: {data_path}")
    return build_specbook_from_data(data_path, max_specs=max_specs, seed=seed), str(data_path)


def hard_verify_response(
    model: torch.nn.Module,
    tokenizer: Any,
    spec: SpecCase,
    prompt_for_eval: str,
    response: str,
    seq_len: int,
    device: torch.device,
) -> Tuple[bool, str, float]:
    verifier = spec.verifier if isinstance(spec.verifier, dict) else {}
    vtype = str(verifier.get("type", "nll")).strip().lower()

    if vtype == "exact":
        ok = normalize_text(response) == normalize_text(spec.target)
        return ok, "" if ok else "exact_mismatch", float("nan")

    if vtype == "contains":
        required = verifier.get("required", [])
        if not isinstance(required, list):
            required = []
        rnorm = normalize_text(response)
        missing = [str(x).strip().lower() for x in required if str(x).strip().lower() not in rnorm]
        ok = len(missing) == 0
        return ok, "" if ok else f"missing:{','.join(missing[:4])}", float("nan")

    if vtype == "regex":
        pattern = str(verifier.get("pattern", "")).strip()
        if not pattern:
            return False, "empty_regex", float("nan")
        try:
            ok = re.search(pattern, response) is not None
        except re.error:
            return False, "bad_regex", float("nan")
        return ok, "" if ok else "regex_mismatch", float("nan")

    thr = float(verifier.get("max_token_nll", default_nll_threshold(spec.tier)))
    nll = case_target_nll(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_for_eval,
        target=spec.target,
        seq_len=seq_len,
        device=device,
    )
    ok = nll <= thr
    return ok, "" if ok else f"token_nll>{thr:.3f}", float(nll)


def soft_judge_score(target: str, response: str) -> float:
    t = str(target or "").strip()
    r = str(response or "").strip()
    if not t or not r:
        return 0.0
    t_tok = simple_tokens(t)
    r_tok = simple_tokens(r)
    tok_score = jaccard(t_tok, r_tok)

    t_norm = normalize_text(t)
    r_norm = normalize_text(r)
    contain = 1.0 if t_norm and (t_norm[: min(48, len(t_norm))] in r_norm or r_norm[: min(48, len(r_norm))] in t_norm) else 0.0

    len_ratio = min(len(r), len(t)) / float(max(1, max(len(r), len(t))))
    return 0.55 * tok_score + 0.25 * contain + 0.20 * len_ratio


def extract_keywords(prompt: str, target: str, max_k: int = 14) -> List[str]:
    toks = list(simple_tokens(prompt) | simple_tokens(target))
    toks.sort(key=lambda x: (-len(x), x))
    return toks[: max(1, int(max_k))]


def summarize_target(target: str, max_len: int = 180) -> str:
    s = re.sub(r"\\s+", " ", str(target or "")).strip()
    if len(s) <= int(max_len):
        return s
    return s[: max(20, int(max_len) - 3)].rstrip() + "..."


def candidate_rule_texts(spec: SpecCase, failure_reason: str, response: str) -> List[str]:
    target_short = summarize_target(spec.target, max_len=220)
    prompt_short = summarize_target(spec.prompt, max_len=120)
    resp_short = summarize_target(response, max_len=120)
    reason = str(failure_reason or "").strip() or "verification_failed"
    cands = [
        f"If the query is similar to: {prompt_short}\\nReturn content aligned with: {target_short}",
        f"Rule from failure({reason}): prioritize target constraints.\\nTarget anchor: {target_short}",
        f"Fix incorrect output pattern.\\nBad output snippet: {resp_short}\\nExpected style/content: {target_short}",
    ]
    out = []
    seen = set()
    for c in cands:
        norm = normalize_text(c)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(c.strip())
    return out


def choose_best_rule_text(spec: SpecCase, failure_reason: str, response: str) -> str:
    cands = candidate_rule_texts(spec, failure_reason=failure_reason, response=response)
    if not cands:
        return f"Use target-aligned response for spec {spec.spec_id}."
    best = cands[0]
    best_score = -1.0
    target_tok = simple_tokens(spec.target)
    reason_tok = simple_tokens(failure_reason)
    for c in cands:
        c_tok = simple_tokens(c)
        score = 0.75 * jaccard(c_tok, target_tok) + 0.25 * jaccard(c_tok, reason_tok)
        score -= 0.001 * max(0, len(c) - 260)
        if score > best_score:
            best_score = score
            best = c
    return best


def build_runtime_prompt(user_prompt: str, patches: Sequence[PatchRecord]) -> str:
    p = str(user_prompt or "").strip()
    if not patches:
        return p
    lines = []
    for rec in patches:
        lines.append(f"- {rec.rule_text}")
    patch_block = "\\n".join(lines)
    return (
        "Known verified rules from recent failures:\\n"
        f"{patch_block}\\n\\n"
        "Follow these rules if they apply, then answer the user query.\\n\\n"
        f"User query:\\n{p}"
    )


def parse_search_list(raw: str, cast_fn, fallback: Sequence) -> List:
    vals = []
    for tok in str(raw or "").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            vals.append(cast_fn(t))
        except Exception:
            continue
    if not vals:
        vals = list(fallback)
    return vals


def generate_with_search(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    target: str,
    device: torch.device,
    max_new_tokens: int,
    branches: Sequence[Tuple[float, int, float]],
) -> Tuple[str, Dict[str, Any]]:
    best_response = ""
    best_score = -1.0
    scored: List[Dict[str, Any]] = []

    for temp, top_k, top_p in branches:
        resp = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        ) if float(temp) <= 0.0 else _generate_with_sampling(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=float(temp),
            top_k=int(top_k),
            top_p=float(top_p),
        )
        s = soft_judge_score(target=target, response=resp)
        scored.append({"temperature": float(temp), "top_k": int(top_k), "top_p": float(top_p), "score": float(s)})
        if s > best_score:
            best_score = s
            best_response = resp

    if not best_response and branches:
        t, k, p = branches[0]
        best_response = _generate_with_sampling(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=max(0.2, float(t)),
            top_k=max(0, int(k)),
            top_p=min(1.0, max(0.5, float(p))),
        )
    return best_response, {"branches": scored, "best_soft_score": float(best_score)}


def _generate_with_sampling(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    model.eval()
    prefix = _build_response_prompt(prompt)
    tokens = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    seq_len = int(getattr(model, "seq_len", 384))

    for _ in range(max(1, int(max_new_tokens))):
        x = torch.tensor([tokens[-seq_len:]], dtype=torch.long, device=device)
        logits, _ = model(x, targets=None)
        next_logits = logits[0, -1]
        next_logits = next_logits / max(1e-5, float(temperature))
        if int(top_k) > 0:
            kth = torch.topk(next_logits, k=min(int(top_k), next_logits.numel())).values[-1]
            next_logits = torch.where(next_logits < kth, torch.full_like(next_logits, -float("inf")), next_logits)
        probs = torch.softmax(next_logits, dim=-1)
        if float(top_p) < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            mask = cdf > float(top_p)
            if torch.any(mask):
                first_idx = int(torch.where(mask)[0][0].item())
                sorted_probs[first_idx + 1 :] = 0.0
                probs = torch.zeros_like(probs)
                probs.scatter_(0, sorted_idx, sorted_probs)
                probs = probs / probs.sum().clamp_min(1e-9)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        tokens.append(next_id)
        if next_id == tokenizer.eos_id:
            break

    decoded = tokenizer.decode(tokens)
    return _extract_response(decoded, prompt=prefix)


def _build_response_prompt(prompt: str) -> str:
    src = str(prompt or "").strip()
    return f"### Instruction\\n{src}\\n\\n### Response\\n"


def _extract_response(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt) :].strip()
    marker = "### Response"
    idx = full_text.rfind(marker)
    if idx >= 0:
        return full_text[idx + len(marker) :].lstrip(": \\n\\t").strip()
    return full_text.strip()


def build_distill_rows(records: Sequence[PatchRecord], max_rows: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ordered = sorted(records, key=lambda r: (r.uses, r.created_step), reverse=True)
    for rec in ordered[: max(1, int(max_rows))]:
        inp = (
            "Apply verified rules when relevant and answer the user query.\\n\\n"
            f"Verified rule:\\n{rec.rule_text}\\n\\n"
            f"User query:\\n{rec.prompt}"
        )
        rows.append(
            {
                "task_type": "omega2_patch",
                "segment_tag": "agentic",
                "language": "multi",
                "_meta_quality_tier": "high",
                "input": inp,
                "output": rec.target,
                "meta": {
                    "patch_id": rec.patch_id,
                    "spec_id": rec.spec_id,
                    "tier": rec.tier,
                    "uses": int(rec.uses),
                },
            }
        )
    return rows

