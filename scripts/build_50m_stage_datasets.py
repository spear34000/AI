from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CURRIC_AUG = {
    "continuation_rephrase",
    "continuation_rephrase_v2",
    "extract_imports",
    "extract_imports_v2",
    "extract_signatures",
    "extract_signatures_v2",
}

DISTILL_AUG = {
    "identify_language_v1",
    "classify_role_v1",
    "extract_identifiers_v1",
    "ko_summary_extract",
    "ko_summary_extract_v2",
    "ko_keywords_extract",
    "ko_keywords_extract_v2",
    "ko_sentence_extract_v1",
    "ko_title_extract_v1",
}


@dataclass
class ExampleRef:
    key: str
    row: Dict
    tokens: int
    task: str
    language: str
    augment_type: str
    input_chars: int
    output_chars: int
    has_action_trace: bool


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
        raw = json.dumps(row, ensure_ascii=False)
        inp = "Use the following JSON as context and respond helpfully."
        out = raw
    return f"### Instruction\n{inp}\n\n### Response\n{out}\n"


def count_tokens_approx(row: Dict) -> int:
    # ByteTokenizer-compatible rough count (bytes + BOS/EOS).
    txt = format_example(row)
    return len(txt.encode("utf-8", errors="replace")) + 2


def row_key(row: Dict) -> str:
    raw = (
        str(row.get("task_type", ""))
        + "\n"
        + str(row.get("language", ""))
        + "\n"
        + str(row.get("input", ""))
        + "\n"
        + str(row.get("output", ""))
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def load_examples(source_jsonl: Path, max_tokens_per_row: int) -> List[ExampleRef]:
    out: List[ExampleRef] = []
    seen_keys: set[str] = set()
    for row in iter_jsonl(source_jsonl):
        if not _license_allowed(row, LICENSE_POLICY):
            continue
        key = row_key(row)
        if key in seen_keys:
            continue
        t = count_tokens_approx(row)
        if t <= 0 or t > int(max_tokens_per_row):
            continue
        inp = _pick_first_non_empty(row, ["input", "instruction", "prompt", "question", "context"])
        out_txt = _pick_first_non_empty(row, ["output", "response", "answer", "completion", "target"])
        seen_keys.add(key)
        out.append(
            ExampleRef(
                key=key,
                row=row,
                tokens=t,
                task=str(row.get("task_type", "")).strip().lower(),
                language=str(row.get("language", "")).strip().lower(),
                augment_type=str(row.get("_augment_type", "")).strip(),
                input_chars=len(inp),
                output_chars=len(out_txt),
                has_action_trace=("ACTION " in inp or "ACTION " in out_txt or "FINAL " in inp or "FINAL " in out_txt),
            )
        )
    return out


def unique_examples(rows: List[ExampleRef]) -> List[ExampleRef]:
    out: List[ExampleRef] = []
    seen: set[str] = set()
    for row in rows:
        if row.key in seen:
            continue
        seen.add(row.key)
        out.append(row)
    return out


LICENSE_POLICY = "mit_only"


def _license_allowed(row: Dict, policy: str) -> bool:
    p = str(policy or "mit_only").strip().lower()
    lic = str(row.get("license", "")).strip()
    lic_up = lic.upper()
    if p == "all":
        return True
    if p == "allow_missing":
        return (not lic) or lic_up == "MIT"
    return lic_up == "MIT"


def _fill_until_budget(
    *,
    candidates: List[ExampleRef],
    budget_tokens: int,
    used_keys: set[str],
    rng: random.Random,
    initial_selected: List[ExampleRef] | None = None,
    initial_tokens: int = 0,
) -> Tuple[List[ExampleRef], int]:
    selected: List[ExampleRef] = list(initial_selected or [])
    total = int(initial_tokens)

    cands = [c for c in candidates if c.key not in used_keys]
    rng.shuffle(cands)
    for c in cands:
        if total + c.tokens > budget_tokens:
            continue
        selected.append(c)
        used_keys.add(c.key)
        total += c.tokens
        if total >= budget_tokens:
            break

    # Greedy remainder fill with short examples.
    if total < budget_tokens:
        rem = [c for c in cands if c.key not in used_keys]
        rem.sort(key=lambda x: x.tokens)
        for c in rem:
            if total + c.tokens > budget_tokens:
                continue
            selected.append(c)
            used_keys.add(c.key)
            total += c.tokens
            if total >= budget_tokens:
                break

    return selected, total


def dump_stage(path: Path, selected: List[ExampleRef]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in selected:
            f.write(json.dumps(e.row, ensure_ascii=False) + "\n")


def stage_stats(selected: List[ExampleRef], token_budget: int) -> Dict:
    by_task = Counter()
    by_lang = Counter()
    by_aug = Counter()
    token_total = 0
    for e in selected:
        by_task[e.task] += 1
        by_lang[e.language] += 1
        by_aug[e.augment_type or "base"] += 1
        token_total += int(e.tokens)
    return {
        "rows": len(selected),
        "tokens": int(token_total),
        "token_budget": int(token_budget),
        "token_utilization": float(token_total / max(1, token_budget)),
        "task_counts": dict(by_task),
        "language_counts": dict(by_lang),
        "augment_counts_top20": dict(by_aug.most_common(20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_jsonl", default="data/slm_mit_unified_v4.jsonl")
    parser.add_argument("--out_dir", default="data/stages_50m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_total_tokens", type=int, default=50_000_000)
    parser.add_argument("--stage1_tokens", type=int, default=22_000_000)
    parser.add_argument("--stage2_tokens", type=int, default=10_000_000)
    parser.add_argument("--stage3_tokens", type=int, default=14_000_000)
    parser.add_argument("--stage4_tokens", type=int, default=4_000_000)
    parser.add_argument("--max_tokens_per_row", type=int, default=12_000)
    parser.add_argument("--license_policy", default="mit_only", choices=["mit_only", "allow_missing", "all"])
    args = parser.parse_args()

    global LICENSE_POLICY
    LICENSE_POLICY = str(args.license_policy)

    rng = random.Random(int(args.seed))
    source = Path(args.source_jsonl)
    if not source.exists():
        raise RuntimeError(f"source_jsonl not found: {source}")

    stage_sum = int(args.stage1_tokens) + int(args.stage2_tokens) + int(args.stage3_tokens) + int(args.stage4_tokens)
    if stage_sum > int(args.max_total_tokens):
        raise RuntimeError(f"stage token sum {stage_sum} exceeds max_total_tokens {int(args.max_total_tokens)}")

    examples = load_examples(source, max_tokens_per_row=int(args.max_tokens_per_row))
    if not examples:
        raise RuntimeError("no usable examples found")

    base_pool = [e for e in examples if not e.augment_type]
    curriculum_pool = unique_examples(
        [e for e in examples if e.augment_type in CURRIC_AUG or (not e.augment_type and e.tokens >= 900) or e.has_action_trace]
    )
    distill_pool = unique_examples(
        [e for e in examples if e.augment_type in DISTILL_AUG]
        + [
            e
            for e in examples
            if (
                e.task == "korean"
                and e.tokens <= 1400
                and e.input_chars <= 600
                and e.output_chars <= 220
                and (not e.augment_type or e.has_action_trace)
            )
        ]
        + [
            e
            for e in examples
            if (
                e.task == "code"
                and e.tokens <= 900
                and e.output_chars <= 140
            )
        ]
    )
    instruction_pool = unique_examples(
        [
        e
        for e in examples
        if (
            (e.augment_type in DISTILL_AUG and e.tokens <= 1100)
            or (e.task == "korean" and e.tokens <= 1200 and e.output_chars <= 240)
            or (e.has_action_trace and e.tokens <= 1400)
        )
        ]
    )

    used: set[str] = set()
    stage_rows: Dict[str, List[ExampleRef]] = {}
    stage_info: Dict[str, Dict] = {}

    # Stage 1: clean base LM only.
    s1_rows, s1_tok = _fill_until_budget(
        candidates=base_pool,
        budget_tokens=int(args.stage1_tokens),
        used_keys=used,
        rng=rng,
    )
    stage_rows["stage1_base_lm"] = s1_rows
    stage_info["stage1_base_lm"] = stage_stats(s1_rows, int(args.stage1_tokens))
    stage_info["stage1_base_lm"]["selected_tokens"] = s1_tok

    # Stage 2: harder curriculum (long base + structure extraction).
    s2_rows, s2_tok = _fill_until_budget(
        candidates=curriculum_pool,
        budget_tokens=int(args.stage2_tokens),
        used_keys=used,
        rng=rng,
    )
    stage_rows["stage2_curriculum"] = s2_rows
    stage_info["stage2_curriculum"] = stage_stats(s2_rows, int(args.stage2_tokens))
    stage_info["stage2_curriculum"]["selected_tokens"] = s2_tok

    # Stage 3: distillation-heavy synthetic supervision.
    s3_rows, s3_tok = _fill_until_budget(
        candidates=distill_pool,
        budget_tokens=int(args.stage3_tokens),
        used_keys=used,
        rng=rng,
    )
    stage_rows["stage3_distill"] = s3_rows
    stage_info["stage3_distill"] = stage_stats(s3_rows, int(args.stage3_tokens))
    stage_info["stage3_distill"]["selected_tokens"] = s3_tok

    # Stage 4: instruction tuning (short, direct I/O).
    s4_rows, s4_tok = _fill_until_budget(
        candidates=instruction_pool,
        budget_tokens=int(args.stage4_tokens),
        used_keys=used,
        rng=rng,
    )
    stage_rows["stage4_instruction"] = s4_rows
    stage_info["stage4_instruction"] = stage_stats(s4_rows, int(args.stage4_tokens))
    stage_info["stage4_instruction"]["selected_tokens"] = s4_tok

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in stage_rows.items():
        dump_stage(out_dir / f"{name}.jsonl", rows)

    total_tokens = s1_tok + s2_tok + s3_tok + s4_tok
    total_rows = sum(len(v) for v in stage_rows.values())

    manifest = {
        "source_jsonl": str(source),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "max_total_tokens": int(args.max_total_tokens),
        "token_budget_sum": int(stage_sum),
        "selected_total_tokens": int(total_tokens),
        "selected_total_rows": int(total_rows),
        "stage_files": {k: str(out_dir / f"{k}.jsonl") for k in stage_rows.keys()},
        "stages": stage_info,
        "pool_sizes": {
            "all_examples": len(examples),
            "base_pool": len(base_pool),
            "curriculum_pool": len(curriculum_pool),
            "distill_pool": len(distill_pool),
            "instruction_pool": len(instruction_pool),
        },
        "license_policy": str(args.license_policy),
    }

    (out_dir / "stage_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
