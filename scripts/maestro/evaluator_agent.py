from __future__ import annotations

from collections import Counter
import re
from pathlib import Path
from typing import Any, Dict, List

from .common import load_json, resolve_python, run_subprocess, save_json


ARITH_TOKENS = ("최종답", "정답:", "부분곱", "검산", "일의 자리", "십의 자리")
UNKNOWN_TOKENS = ("모르겠습니다", "확실하지 않", "알 수 없습니다")
WEIRD_CHAR_RE = re.compile(r"[^\s0-9A-Za-z가-힣,?!:;()\-_/\"'~`]+")
WORD_RE = re.compile(r"[A-Za-z0-9가-힣]+")
HANGUL_RE = re.compile(r"[가-힣]")

GENERAL_EXPECTATIONS: Dict[str, Dict[str, Any]] = {
    "chat_hi": {"any": ["안녕", "반갑", "하이"], "max_len": 80},
    "chat_intro": {"any": ["안녕", "반갑", "도움", "설명", "정리"], "max_len": 180},
    "chat_help": {"any": ["도움", "설명", "정리", "질문"], "max_len": 180},
    "def_kotlin": {"all": ["코틀린"], "any": ["언어", "프로그래밍", "jvm", "안드로이드"], "max_len": 260},
    "def_http": {"all": ["http"], "any": ["프로토콜", "웹", "요청", "응답"], "max_len": 260},
    "def_jwt": {"all": ["jwt"], "any": ["토큰", "인증", "서명"], "max_len": 260},
    "def_gc": {"any": ["메모리", "가비지", "회수", "정리"], "max_len": 260},
    "qa_sla99": {"all": ["99"], "any": ["가용성", "서비스", "업타임", "수준"], "max_len": 260},
    "qa_vienna": {"any": ["조리", "가열", "먹", "안전", "식품"], "max_len": 260},
}

LOGIC_EXPECTATIONS: Dict[str, Dict[str, Any]] = {
    "logic_compare": {"any": ["a", "b", "c"]},
    "logic_compare_smallest": {"any": ["a", "b", "c", "d"]},
    "logic_prob": {"any": ["/", "확률", "2/5", "3/5", "4/5", "5/8", "6/7"]},
    "logic_prob_blue": {"any": ["/", "확률", "2/5", "3/5", "4/5", "5/8", "6/7"]},
    "logic_seq": {"any": ["규칙", "다음", "알 수 없음", "단정", "44", "71", "192"]},
    "logic_sequence": {"any": ["규칙", "다음", "알 수 없음", "단정", "44", "71", "192"]},
    "logic_syllogism_yes": {"any": ["예", "아니", "참", "보장", "가능"]},
    "logic_syllogism_unknown": {"any": ["예", "아니", "보장", "없", "불가"]},
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _tokens(text: str) -> List[str]:
    return WORD_RE.findall(_normalize(text).lower())


def _digit_ratio(text: str) -> float:
    s = _normalize(text)
    if not s:
        return 1.0
    digits = sum(ch.isdigit() for ch in s)
    return digits / float(len(s))


def _repetition_bad(text: str) -> bool:
    toks = _tokens(text)
    if len(toks) < 6:
        return False
    top_count = Counter(toks).most_common(1)[0][1]
    if top_count >= 4 and top_count / max(1, len(toks)) >= 0.22:
        return True
    bigrams = Counter(zip(toks, toks[1:]))
    if bigrams and bigrams.most_common(1)[0][1] >= 3:
        return True
    return False


def _matches_expectation(item_id: str, text: str, table: Dict[str, Dict[str, Any]]) -> bool:
    spec = table.get(item_id)
    if not spec:
        return True
    low = _normalize(text).lower()
    max_len = int(spec.get("max_len", 0) or 0)
    if max_len and len(low) > max_len:
        return False
    all_terms = [str(x).lower() for x in spec.get("all", [])]
    any_terms = [str(x).lower() for x in spec.get("any", [])]
    if all_terms and not all(term in low for term in all_terms):
        return False
    if any_terms and not any(term in low for term in any_terms):
        return False
    return True


def _collapse_score(item_id: str, text: str) -> bool:
    s = _normalize(text)
    if not s:
        return True
    if any(tok in s for tok in ARITH_TOKENS):
        return True
    if re.search(r"\d{4,}", s):
        return True
    if len(set(s)) <= 3 and len(s) >= 12:
        return True
    weird = WEIRD_CHAR_RE.findall(s)
    if weird and sum(len(x) for x in weird) / max(1, len(s)) > 0.08:
        return True
    if _digit_ratio(s) > 0.22:
        return True
    if _repetition_bad(s):
        return True
    if item_id.startswith(("chat_", "def_", "qa_")) and not HANGUL_RE.search(s):
        return True
    if item_id.startswith(("chat_", "def_", "qa_")) and not _matches_expectation(item_id, s, GENERAL_EXPECTATIONS):
        return True
    if item_id.startswith("logic_") and not _matches_expectation(item_id, s, LOGIC_EXPECTATIONS):
        return True
    return False


def _definition_relevance(item_id: str, text: str) -> float | None:
    expectations = {
        "def_kotlin": ["코틀린", "언어"],
        "def_http": ["http", "프로토콜", "요청", "응답"],
        "def_jwt": ["jwt", "토큰", "인증"],
        "def_gc": ["메모리", "회수", "가비지"],
        "qa_sla99": ["99", "가용성", "서비스"],
    }
    expected = expectations.get(item_id)
    if not expected:
        return None
    low = _normalize(text).lower()
    return sum(1 for token in expected if token in low) / float(len(expected))


def _item_pass(item_id: str, text: str, returncode: int) -> bool:
    if int(returncode) != 0:
        return False
    if _collapse_score(item_id, text):
        return False
    if item_id.startswith(("chat_", "def_", "qa_")):
        return _matches_expectation(item_id, text, GENERAL_EXPECTATIONS)
    if item_id.startswith("logic_"):
        return _matches_expectation(item_id, text, LOGIC_EXPECTATIONS)
    return True


def _score_prompt_suite(path: Path) -> Dict[str, Any]:
    payload = load_json(path, default={})
    results = payload.get("results", []) if isinstance(payload, dict) else []
    scored: List[Dict[str, Any]] = []
    for ckpt_row in results:
        items = ckpt_row.get("items", []) if isinstance(ckpt_row, dict) else []
        general_items = [x for x in items if str(x.get("id", "")).startswith(("chat_", "def_", "qa_"))]
        logic_items = [x for x in items if str(x.get("id", "")).startswith("logic_")]
        collapse_n = sum(1 for x in general_items if _collapse_score(str(x.get("id", "")), str(x.get("stdout", ""))))
        unknown_n = sum(1 for x in general_items if any(tok in str(x.get("stdout", "")) for tok in UNKNOWN_TOKENS))
        pass_n = sum(1 for x in general_items if _item_pass(str(x.get("id", "")), str(x.get("stdout", "")), int(x.get("returncode", 1))))
        rel_scores = [_definition_relevance(str(x.get("id", "")), str(x.get("stdout", ""))) for x in general_items]
        rel_scores = [x for x in rel_scores if x is not None]
        logic_nonempty = sum(1 for x in logic_items if _item_pass(str(x.get("id", "")), str(x.get("stdout", "")), int(x.get("returncode", 1))))
        scored.append(
            {
                "checkpoint": ckpt_row.get("checkpoint", ""),
                "general_prompt_suite_pass_rate": (pass_n / len(general_items)) if general_items else 0.0,
                "collapse_rate": (collapse_n / len(general_items)) if general_items else 1.0,
                "unknown_rate": (unknown_n / len(general_items)) if general_items else 0.0,
                "definition_relevance_score": (sum(rel_scores) / len(rel_scores)) if rel_scores else 0.0,
                "logic_reasoning_readout_rate": (logic_nonempty / len(logic_items)) if logic_items else 0.0,
                "items": items,
            }
        )
    return {"results": scored}


def _checkpoint_candidates_from_train_report(root: Path, out_dir: Path) -> List[str]:
    report = load_json(out_dir / "train_report.json", default={})
    phases = report.get("phases", []) if isinstance(report, dict) else []
    out: List[str] = []
    for row in phases:
        if not isinstance(row, dict):
            continue
        cmd = row.get("command", [])
        if not isinstance(cmd, list):
            continue
        output_dir = ""
        for idx, token in enumerate(cmd):
            if token == "--output_dir" and idx + 1 < len(cmd):
                output_dir = str(cmd[idx + 1])
                break
        if not output_dir:
            continue
        p = Path(output_dir)
        if not p.is_absolute():
            p = root / p
        for name in ("slm_best.pt", "slm_last.pt"):
            ckpt = p / name
            if ckpt.exists():
                out.append(str(ckpt))
    dedup: List[str] = []
    seen = set()
    for ckpt in out:
        if ckpt not in seen:
            seen.add(ckpt)
            dedup.append(ckpt)
    return dedup


def run_evaluation(spec: Dict[str, Any], planner: Dict[str, Any], root: Path, out_dir: Path) -> Dict[str, Any]:
    py = resolve_python(root)
    registry = load_json(root / "configs" / "mainline_benchmarks_v1.json", default={})
    benchmark_rows = registry.get("benchmarks", []) if isinstance(registry, dict) else []

    checkpoints = [planner["selected"]["mainline_base_checkpoint"]]
    logic_reference = planner["selected"].get("logic_reference_checkpoint", "")
    if logic_reference:
        checkpoints.append(logic_reference)
    checkpoints.extend(_checkpoint_candidates_from_train_report(root, out_dir))
    checkpoints = [x for x in checkpoints if x]
    dedup: List[str] = []
    seen = set()
    for ckpt in checkpoints:
        if ckpt not in seen:
            seen.add(ckpt)
            dedup.append(ckpt)
    checkpoints = dedup

    exact_runs: List[Dict[str, Any]] = []
    prompt_runs: List[Dict[str, Any]] = []

    for bench in benchmark_rows:
        btype = str(bench.get("type", ""))
        if btype == "exact":
            out_json = out_dir / f"{bench['name']}.json"
            result = run_subprocess(
                [
                    py,
                    "scripts/eval_logic_exact_v1.py",
                    "--checkpoints",
                    *checkpoints,
                    "--data_path",
                    str(bench["data_path"]),
                    "--limit",
                    str(int(bench.get("limit", 256))),
                    "--device",
                    str(bench.get("device", "cuda")),
                    "--mode",
                    "inprocess",
                    "--out_json",
                    str(out_json),
                ],
                cwd=root,
                timeout=14400,
            )
            exact_runs.append({"benchmark": bench["name"], "result": result, "out_json": str(out_json)})
        elif btype == "prompt_suite":
            out_json = out_dir / f"{bench['name']}.json"
            result = run_subprocess(
                [
                    py,
                    "scripts/eval_prompt_suite_v1.py",
                    "--checkpoints",
                    *checkpoints,
                    "--suite",
                    str(bench["suite_path"]),
                    "--device",
                    str(bench.get("device", "cuda")),
                    "--max_new_tokens",
                    str(int(bench.get("max_new_tokens", 96))),
                    "--out_json",
                    str(out_json),
                ],
                cwd=root,
                timeout=14400,
            )
            prompt_runs.append({"benchmark": bench["name"], "result": result, "out_json": str(out_json)})

    scored_prompts: Dict[str, Any] = {}
    for row in prompt_runs:
        out_json = Path(row["out_json"])
        if out_json.exists():
            scored_prompts[row["benchmark"]] = _score_prompt_suite(out_json)

    report = {
        "checkpoints": checkpoints,
        "exact_runs": exact_runs,
        "prompt_runs": prompt_runs,
        "scored_prompts": scored_prompts,
    }
    save_json(out_dir / "eval_report.json", report)
    return report
