from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import HfApi


MIT_DATASETS = [
    "HuggingFaceH4/ultrachat_200k",
    "openbmb/UltraChat",
    "openbmb/UltraFeedback",
    "Open-Orca/OpenOrca",
    "openai/gsm8k",
    "openai/MMMLU",
    "HAERAE-HUB/HRM8K",
]


def normalize_license_value(value) -> str:
    if isinstance(value, list):
        merged = " ".join(str(x).strip() for x in value if str(x).strip())
        return merged.lower()
    return str(value or "").strip().lower()


def has_mit_license_tag(tags: List[str]) -> bool:
    return any(str(t).strip().lower() == "license:mit" for t in (tags or []))


def verify_mit_datasets(dataset_ids: List[str]) -> Dict[str, bool]:
    api = HfApi()
    out: Dict[str, bool] = {}
    for ds in dataset_ids:
        info = api.dataset_info(ds)
        card = info.cardData or {}
        card_license = normalize_license_value(card.get("license", ""))
        by_card = "mit" in card_license
        by_tag = has_mit_license_tag(info.tags or [])
        out[ds] = bool(by_card or by_tag)
    return out


def clean_text(text: str) -> str:
    t = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def normalize_key(text: str) -> str:
    t = clean_text(text).lower()
    t = re.sub(r"\s+", " ", t)
    return t


def valid_pair(inp: str, out: str) -> bool:
    i = clean_text(inp)
    o = clean_text(out)
    if len(i) < 4 or len(o) < 3:
        return False
    if len(i) > 2400 or len(o) > 3200:
        return False
    if i.count("?") > 24 or o.count("?") > 24:
        return False
    if re.search(r"(.)\1{14,}", i) or re.search(r"(.)\1{14,}", o):
        return False
    return True


def parse_lang_from_cfg(cfg_name: str) -> str:
    cfg = str(cfg_name or "").strip().upper()
    if not cfg:
        return "en"
    if "_" in cfg:
        return cfg.split("_", 1)[0].lower()
    return cfg.lower()


def row_task_segment(language: str) -> Tuple[str, str]:
    lang = str(language or "").strip().lower()
    if lang == "ko":
        return "korean", "ko"
    # Train as language/document domain for all non-ko languages.
    return "english", "english"


def make_row(
    source_dataset: str,
    language: str,
    inp: str,
    out: str,
    tier: str = "high",
    source: str = "hf_mit_language_superpack_v1",
) -> Dict:
    task_type, segment_tag = row_task_segment(language=language)
    rid_raw = f"{source_dataset}\n{language}\n{inp}\n{out}"
    rid = hashlib.sha1(rid_raw.encode("utf-8")).hexdigest()[:20]
    return {
        "id": f"hf_mit_lang_{rid}",
        "task_type": task_type,
        "segment_tag": segment_tag,
        "language": str(language).strip().lower() or "en",
        "_meta_quality_tier": str(tier),
        "license": "MIT",
        "source_dataset": str(source_dataset),
        "source": str(source),
        "input": clean_text(inp),
        "output": clean_text(out),
    }


def sampled_indices(n_rows: int, limit: int, seed: int) -> Iterable[int]:
    n = int(max(0, n_rows))
    lim = int(max(0, limit))
    if n == 0 or lim == 0:
        return []
    if lim >= n:
        return range(n)
    step = max(1, n // lim)
    rng = random.Random(int(seed))
    offset = rng.randint(0, max(0, step - 1))
    idxs = []
    for idx in range(offset, n, step):
        idxs.append(idx)
        if len(idxs) >= lim:
            break
    if len(idxs) < lim:
        # Backfill deterministically if needed.
        extra = [i for i in range(n) if i not in set(idxs)]
        rng.shuffle(extra)
        idxs.extend(extra[: max(0, lim - len(idxs))])
    return idxs[:lim]


def extract_dialog_pairs_from_messages(messages: List[Dict]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    pending_user = ""
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = clean_text(item.get("content", ""))
        if not content:
            continue
        if role == "user":
            pending_user = content
            continue
        if role == "assistant" and pending_user:
            out.append((pending_user, content))
            pending_user = ""
    return out


def collect_ultrachat_h4(limit_pairs: int, seed: int) -> List[Dict]:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", "default", split="train_sft")
    rows: List[Dict] = []
    for idx in sampled_indices(len(ds), max(1, int(limit_pairs // 2)), seed=seed + 11):
        rec = ds[int(idx)]
        pairs = extract_dialog_pairs_from_messages(rec.get("messages", []))
        for inp, out in pairs:
            if not valid_pair(inp, out):
                continue
            rows.append(
                make_row(
                    source_dataset="HuggingFaceH4/ultrachat_200k",
                    language="en",
                    inp=inp,
                    out=out,
                    tier="high",
                )
            )
            if len(rows) >= int(limit_pairs):
                return rows
    return rows


def collect_ultrachat_openbmb(limit_pairs: int, seed: int) -> List[Dict]:
    ds = load_dataset("openbmb/UltraChat", "default", split="train")
    rows: List[Dict] = []
    for idx in sampled_indices(len(ds), max(1, int(limit_pairs // 2)), seed=seed + 23):
        rec = ds[int(idx)]
        data = rec.get("data", [])
        if not isinstance(data, list) or len(data) < 2:
            continue
        for i in range(0, len(data) - 1, 2):
            inp = clean_text(data[i])
            out = clean_text(data[i + 1])
            if not valid_pair(inp, out):
                continue
            rows.append(
                make_row(
                    source_dataset="openbmb/UltraChat",
                    language="en",
                    inp=inp,
                    out=out,
                    tier="high",
                )
            )
            if len(rows) >= int(limit_pairs):
                return rows
    return rows


def _to_float(value, default: float = -1e9) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def pick_best_ultrafeedback_response(completions) -> str:
    if not isinstance(completions, list) or not completions:
        return ""
    best = None
    best_score = -1e9
    for c in completions:
        if not isinstance(c, dict):
            continue
        s = _to_float(c.get("overall_score", -1e9), default=-1e9)
        if s > best_score:
            best_score = s
            best = c
    if not isinstance(best, dict):
        return ""
    return clean_text(best.get("response", ""))


def collect_ultrafeedback(limit_pairs: int, seed: int) -> List[Dict]:
    ds = load_dataset("openbmb/UltraFeedback", "default", split="train")
    rows: List[Dict] = []
    for idx in sampled_indices(len(ds), int(limit_pairs), seed=seed + 37):
        rec = ds[int(idx)]
        inp = clean_text(rec.get("instruction", ""))
        out = pick_best_ultrafeedback_response(rec.get("completions", []))
        if not valid_pair(inp, out):
            continue
        rows.append(
            make_row(
                source_dataset="openbmb/UltraFeedback",
                language="en",
                inp=inp,
                out=out,
                tier="high",
            )
        )
        if len(rows) >= int(limit_pairs):
            break
    return rows


def collect_openorca(limit_pairs: int, seed: int) -> List[Dict]:
    ds = load_dataset("Open-Orca/OpenOrca", split="train")
    rows: List[Dict] = []
    for idx in sampled_indices(len(ds), int(limit_pairs), seed=seed + 41):
        rec = ds[int(idx)]
        sys_p = clean_text(rec.get("system_prompt", ""))
        q = clean_text(rec.get("question", ""))
        a = clean_text(rec.get("response", ""))
        if not q:
            continue
        inp = q if not sys_p else f"System context:\n{sys_p}\n\nUser question:\n{q}"
        if not valid_pair(inp, a):
            continue
        rows.append(
            make_row(
                source_dataset="Open-Orca/OpenOrca",
                language="en",
                inp=inp,
                out=a,
                tier="mid",
            )
        )
        if len(rows) >= int(limit_pairs):
            break
    return rows


def collect_gsm8k(limit_train: int, limit_test: int, seed: int) -> List[Dict]:
    rows: List[Dict] = []
    for split, lim, salt in [("train", int(limit_train), 53), ("test", int(limit_test), 59)]:
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for idx in sampled_indices(len(ds), int(lim), seed=seed + salt):
            rec = ds[int(idx)]
            inp = clean_text(rec.get("question", ""))
            out = clean_text(rec.get("answer", ""))
            if not valid_pair(inp, out):
                continue
            rows.append(
                make_row(
                    source_dataset=f"openai/gsm8k::{split}",
                    language="en",
                    inp=inp,
                    out=out,
                    tier="high",
                )
            )
    return rows


def collect_mmmlu(per_lang_limit: int, seed: int, include_default: bool = False) -> List[Dict]:
    rows: List[Dict] = []
    cfgs = get_dataset_config_names("openai/MMMLU")
    for cfg in cfgs:
        if str(cfg) == "default" and not bool(include_default):
            continue
        ds = load_dataset("openai/MMMLU", cfg, split="test")
        lang = parse_lang_from_cfg(cfg)
        for idx in sampled_indices(len(ds), int(per_lang_limit), seed=seed + (abs(hash(cfg)) % 10000)):
            rec = ds[int(idx)]
            q = clean_text(rec.get("Question", ""))
            a = clean_text(rec.get("A", ""))
            b = clean_text(rec.get("B", ""))
            c = clean_text(rec.get("C", ""))
            d = clean_text(rec.get("D", ""))
            ans = clean_text(rec.get("Answer", ""))
            subject = clean_text(rec.get("Subject", ""))
            if not q or not ans:
                continue
            inp = (
                f"Question ({subject}): {q}\n"
                f"A) {a}\nB) {b}\nC) {c}\nD) {d}\n"
                "Choose the best option."
            )
            if lang == "ko":
                out = f"정답은 {ans}입니다."
            else:
                out = f"The correct answer is {ans}."
            if not valid_pair(inp, out):
                continue
            rows.append(
                make_row(
                    source_dataset=f"openai/MMMLU::{cfg}",
                    language=lang,
                    inp=inp,
                    out=out,
                    tier="high",
                )
            )
    return rows


def collect_hrm8k(per_cfg_limit: int, seed: int) -> List[Dict]:
    rows: List[Dict] = []
    cfgs = get_dataset_config_names("HAERAE-HUB/HRM8K")
    for cfg in cfgs:
        ds = load_dataset("HAERAE-HUB/HRM8K", cfg, split="test")
        for idx in sampled_indices(len(ds), int(per_cfg_limit), seed=seed + (abs(hash("hrm_" + cfg)) % 10000)):
            rec = ds[int(idx)]
            q = clean_text(rec.get("question", ""))
            a = clean_text(rec.get("answer", ""))
            if not q or not a:
                continue
            out = f"정답은 {a}입니다."
            if not valid_pair(q, out):
                continue
            rows.append(
                make_row(
                    source_dataset=f"HAERAE-HUB/HRM8K::{cfg}",
                    language="ko",
                    inp=q,
                    out=out,
                    tier="high",
                )
            )
    return rows


def dedupe_rows(rows: List[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for r in rows:
        inp = str(r.get("input", ""))
        out_txt = str(r.get("output", ""))
        key = (normalize_key(inp), normalize_key(out_txt))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a high-quality MIT multilingual language superpack.")
    parser.add_argument("--out_jsonl", default="data/hf_mit_language_superpack_v1.jsonl")
    parser.add_argument("--manifest", default="data/hf_mit_language_superpack_v1.manifest.json")
    parser.add_argument("--h4_ultrachat_limit", type=int, default=80000)
    parser.add_argument("--openbmb_ultrachat_limit", type=int, default=30000)
    parser.add_argument("--ultrafeedback_limit", type=int, default=25000)
    parser.add_argument("--openorca_limit", type=int, default=30000)
    parser.add_argument("--gsm8k_train_limit", type=int, default=6500)
    parser.add_argument("--gsm8k_test_limit", type=int, default=1200)
    parser.add_argument("--mmmlu_per_lang_limit", type=int, default=1200)
    parser.add_argument("--hrm8k_per_cfg_limit", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = int(args.seed)
    rng = random.Random(seed)

    mit_ok = verify_mit_datasets(MIT_DATASETS)
    not_mit = [k for k, ok in mit_ok.items() if not ok]
    if not_mit:
        raise RuntimeError(f"non-MIT datasets detected: {not_mit}")

    rows: List[Dict] = []
    source_rows: Dict[str, int] = {}

    part_h4 = collect_ultrachat_h4(limit_pairs=int(args.h4_ultrachat_limit), seed=seed + 1)
    rows.extend(part_h4)
    source_rows["HuggingFaceH4/ultrachat_200k"] = len(part_h4)

    part_uc = collect_ultrachat_openbmb(limit_pairs=int(args.openbmb_ultrachat_limit), seed=seed + 2)
    rows.extend(part_uc)
    source_rows["openbmb/UltraChat"] = len(part_uc)

    part_uf = collect_ultrafeedback(limit_pairs=int(args.ultrafeedback_limit), seed=seed + 3)
    rows.extend(part_uf)
    source_rows["openbmb/UltraFeedback"] = len(part_uf)

    part_orca = collect_openorca(limit_pairs=int(args.openorca_limit), seed=seed + 4)
    rows.extend(part_orca)
    source_rows["Open-Orca/OpenOrca"] = len(part_orca)

    part_gsm = collect_gsm8k(limit_train=int(args.gsm8k_train_limit), limit_test=int(args.gsm8k_test_limit), seed=seed + 5)
    rows.extend(part_gsm)
    source_rows["openai/gsm8k"] = len(part_gsm)

    part_mmmlu = collect_mmmlu(per_lang_limit=int(args.mmmlu_per_lang_limit), seed=seed + 6, include_default=False)
    rows.extend(part_mmmlu)
    source_rows["openai/MMMLU"] = len(part_mmmlu)

    part_hrm = collect_hrm8k(per_cfg_limit=int(args.hrm8k_per_cfg_limit), seed=seed + 7)
    rows.extend(part_hrm)
    source_rows["HAERAE-HUB/HRM8K"] = len(part_hrm)

    rows = dedupe_rows(rows)
    rng.shuffle(rows)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    task_counts = Counter()
    seg_counts = Counter()
    lang_counts = Counter()
    src_counts = Counter()
    for r in rows:
        task_counts[str(r.get("task_type", ""))] += 1
        seg_counts[str(r.get("segment_tag", ""))] += 1
        lang_counts[str(r.get("language", ""))] += 1
        src_counts[str(r.get("source_dataset", ""))] += 1

    manifest = {
        "output_jsonl": str(out_path),
        "rows_total": len(rows),
        "source_rows_before_dedupe": source_rows,
        "task_counts": dict(task_counts),
        "segment_counts": dict(seg_counts),
        "language_counts_top30": dict(lang_counts.most_common(30)),
        "source_counts_top30": dict(src_counts.most_common(30)),
        "limits": {
            "h4_ultrachat_limit": int(args.h4_ultrachat_limit),
            "openbmb_ultrachat_limit": int(args.openbmb_ultrachat_limit),
            "ultrafeedback_limit": int(args.ultrafeedback_limit),
            "openorca_limit": int(args.openorca_limit),
            "gsm8k_train_limit": int(args.gsm8k_train_limit),
            "gsm8k_test_limit": int(args.gsm8k_test_limit),
            "mmmlu_per_lang_limit": int(args.mmmlu_per_lang_limit),
            "hrm8k_per_cfg_limit": int(args.hrm8k_per_cfg_limit),
        },
        "seed": seed,
        "license_policy": "MIT only (verified by HF card/tag)",
        "verified_datasets": mit_ok,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(out_path),
                "rows_total": len(rows),
                "task_counts": dict(task_counts),
                "segment_counts": dict(seg_counts),
                "language_counts_top10": dict(lang_counts.most_common(10)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()


