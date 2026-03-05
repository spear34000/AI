from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set

import torch

from chat_slm import ModelRouter, resolve_device, run_one_turn


TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]{2,}")
HANGUL_RE = re.compile(r"[가-힣]")
INTRO_RE = re.compile(r"(안녕하세요|반갑습니다|저는\s+.+(ai|어시스턴트|도우미|모델))", re.IGNORECASE)
UNKNOWN_RE = re.compile(r"(모르|알 수 없|확실하지|정보가 부족|판단하기 어렵)")
REPEAT_RE = re.compile(r"(.)\1{8,}")


@dataclass
class EvalCase:
    name: str
    prompt: str
    expect_any: Sequence[str]
    discourage_intro: bool = True
    min_chars: int = 20
    expect_unknown: bool = False


CASES: List[EvalCase] = [
    EvalCase(name="def_kotlin", prompt="코틀린이란?", expect_any=("코틀린", "언어", "JVM", "자바")),
    EvalCase(name="def_typescript", prompt="타입스크립트란?", expect_any=("타입스크립트", "자바스크립트", "타입", "언어")),
    EvalCase(name="def_kakaotalk", prompt="카카오톡이란?", expect_any=("카카오톡", "메신저", "채팅", "서비스")),
    EvalCase(name="def_llm", prompt="LLM이란?", expect_any=("LLM", "언어 모델", "텍스트", "생성")),
    EvalCase(
        name="logic_subset",
        prompt="모든 A는 B다. 일부 B는 C다. 그렇다면 일부 A는 C인가? 이유를 설명하라.",
        expect_any=("아니다", "보장", "없다", "추가")),
    EvalCase(
        name="ratio_2_3",
        prompt="설탕과 밀가루 비율이 2:3이다. 밀가루가 600g이면 설탕은?",
        expect_any=("400", "400g")),
    EvalCase(
        name="order_compare",
        prompt="A는 B보다 크고, B는 C보다 크다. 가장 작은 것은 누구인가?",
        expect_any=("C", "가장 작은")),
    EvalCase(
        name="unknown_term",
        prompt="마인크래프트란?",
        expect_any=("마인크래프트",),
        expect_unknown=True),
]


def normalize_token(token: str) -> str:
    t = str(token or "").strip().lower()
    suffixes = ("입니다", "이다", "이에요", "예요", "이란", "란", "은", "는", "이", "가", "을", "를", "에", "로")
    for s in suffixes:
        if len(t) > len(s) + 1 and t.endswith(s):
            return t[: -len(s)]
    return t


def token_set(text: str) -> Set[str]:
    out: Set[str] = set()
    for tok in TOKEN_RE.findall(str(text or "")):
        n = normalize_token(tok)
        if len(n) >= 2:
            out.add(n)
    return out


def hangul_ratio(text: str) -> float:
    src = str(text or "")
    letters = [c for c in src if c.isalnum() or HANGUL_RE.match(c)]
    if not letters:
        return 0.0
    ko = sum(1 for c in letters if HANGUL_RE.match(c))
    return float(ko) / float(max(1, len(letters)))


def score_case(case: EvalCase, response: str) -> Dict:
    r = str(response or "").strip()
    score = 0.0
    reasons: List[str] = []

    if not r:
        return {"score": 0.0, "reasons": ["empty"]}

    if len(r) >= int(case.min_chars):
        score += 0.15
    else:
        reasons.append("too_short")

    hr = hangul_ratio(r)
    if hr >= 0.55:
        score += 0.15
    else:
        reasons.append("low_hangul_ratio")

    q_tokens = token_set(case.prompt)
    r_tokens = token_set(r)
    overlap = len(q_tokens.intersection(r_tokens))
    if q_tokens:
        score += 0.20 * (overlap / float(max(1, len(q_tokens))))
        if overlap == 0:
            reasons.append("no_keyword_overlap")

    hits = sum(1 for x in case.expect_any if str(x).lower() in r.lower())
    if case.expect_any:
        score += 0.40 * (hits / float(max(1, len(case.expect_any))))
        if hits == 0:
            reasons.append("expected_terms_missing")

    if case.expect_unknown:
        if UNKNOWN_RE.search(r):
            score += 0.20
        else:
            reasons.append("should_be_unknown")
    else:
        if UNKNOWN_RE.search(r):
            score -= 0.20
            reasons.append("unexpected_unknown")

    if case.discourage_intro and INTRO_RE.search(r):
        score -= 0.30
        reasons.append("intro_leak")

    if REPEAT_RE.search(r):
        score -= 0.25
        reasons.append("char_repeat")

    toks = TOKEN_RE.findall(r.lower())
    if len(toks) >= 14:
        div = len(set(toks)) / float(max(1, len(toks)))
        if div < 0.34:
            score -= 0.20
            reasons.append("low_diversity")

    score = min(1.0, max(0.0, score))
    return {"score": float(score), "reasons": reasons}


def run_eval_for_checkpoint(args: argparse.Namespace, ckpt: Path) -> Dict:
    device = resolve_device(args.device)
    router = ModelRouter(
        device=device,
        checkpoint=str(ckpt),
        use_ema=bool(args.use_ema),
        router_mode="single",
        code_checkpoint="",
        chat_checkpoint="",
        quantize_int8=bool(args.quantize_int8),
    )

    rows: List[Dict] = []
    for case in CASES:
        answer, _bundle, meta = run_one_turn(
            router=router,
            user_prompt=case.prompt,
            history=[],
            system_prompt="",
            max_new_tokens=int(args.max_new_tokens),
            max_input_tokens=39768,
            max_output_tokens=38000,
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
        )
        score_info = score_case(case=case, response=answer)
        rows.append(
            {
                "name": case.name,
                "prompt": case.prompt,
                "response": answer,
                "score": float(score_info["score"]),
                "reasons": score_info["reasons"],
                "meta": meta,
            }
        )

    avg_score = sum(float(r["score"]) for r in rows) / float(max(1, len(rows)))
    return {
        "checkpoint": str(ckpt),
        "avg_score": float(avg_score),
        "avg_score_100": float(avg_score * 100.0),
        "cases": rows,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quality-focused Korean eval for the current SLM stack.")
    p.add_argument("--candidate_checkpoint", required=True)
    p.add_argument("--baseline_checkpoint", default="")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--quantize_int8", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=180)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.10)
    p.add_argument("--min_new_tokens", type=int, default=0)
    p.add_argument("--disable_ko_guard", action="store_true")
    p.add_argument("--ko_guard_topk", type=int, default=96)
    p.add_argument("--ko_guard_rare_penalty", type=float, default=0.9)
    p.add_argument("--ko_guard_latin_penalty", type=float, default=0.25)
    p.add_argument("--disable_quality_rerank", action="store_true")
    p.add_argument("--rerank_candidates", type=int, default=8)
    p.add_argument("--rerank_temp_floor", type=float, default=0.15)
    p.add_argument("--rerank_top_k", type=int, default=80)
    p.add_argument("--rerank_top_p", type=float, default=0.92)
    p.add_argument("--zero_shot_mode", default="strict", choices=["off", "balanced", "strict"])
    p.add_argument("--agent_mode", default="auto", choices=["off", "auto", "triad"])
    p.add_argument("--tool_cache_path", default="data/tool_knowledge_cache_v3_clean.jsonl")
    p.add_argument("--tool_lookup_timeout", type=float, default=4.0)
    p.add_argument("--disable_web_tool_lookup", action="store_true")
    p.add_argument("--disable_retrieval", action="store_true")
    p.add_argument("--out_json", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    candidate_ckpt = Path(args.candidate_checkpoint)
    if not candidate_ckpt.exists():
        raise FileNotFoundError(f"candidate checkpoint not found: {candidate_ckpt}")

    result = {"candidate": run_eval_for_checkpoint(args=args, ckpt=candidate_ckpt)}
    baseline_text = str(args.baseline_checkpoint or "").strip()
    if baseline_text:
        baseline_ckpt = Path(baseline_text)
        if not baseline_ckpt.exists():
            raise FileNotFoundError(f"baseline checkpoint not found: {baseline_ckpt}")
        base = run_eval_for_checkpoint(args=args, ckpt=baseline_ckpt)
        result["baseline"] = base
        b = float(base["avg_score"])
        c = float(result["candidate"]["avg_score"])
        result["improvement_ratio"] = float(c / max(1e-6, b))
        result["improvement_percent"] = float((c - b) * 100.0)

    out_json = str(args.out_json or "").strip()
    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
