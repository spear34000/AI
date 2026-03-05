from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from .common import count_jsonl, file_fingerprint, resolve_python, run_subprocess, save_json


def _write_prompt_suite(path: Path) -> None:
    rows = [
        {"id": "chat_hi", "prompt": "ㅎㅇ"},
        {"id": "chat_intro", "prompt": "한국어로 한 문장 인사"},
        {"id": "chat_help", "prompt": "짧게 도와줄 수 있는 일을 말해줘"},
        {"id": "def_kotlin", "prompt": "코틀린이란?"},
        {"id": "def_http", "prompt": "HTTP란?"},
        {"id": "def_jwt", "prompt": "JWT란?"},
        {"id": "def_gc", "prompt": "가비지 컬렉션이란?"},
        {"id": "qa_sla99", "prompt": "SLA 99가 뭐야?"},
        {"id": "qa_vienna", "prompt": "비엔나 소시지 생으로 먹어도 돼?"},
        {"id": "logic_compare", "prompt": "A는 B보다 크고, B는 C보다 크다. 가장 작은 것은 누구인가? 이유를 짧게 설명하라."},
        {"id": "logic_prob", "prompt": "빨간 공 2개와 파란 공 3개가 있다. 하나를 무작위로 뽑을 때 파란 공일 확률은? 이유를 짧게 설명하라."},
        {"id": "logic_seq", "prompt": "2, 6, 7, 21, 23, 69, ? 다음 수를 규칙 기반으로 설명하고 답하라."},
    ]
    save_json(path, rows)


def _write_logic_readout(path: Path) -> None:
    rows = [
        {"id": "logic_compare_smallest", "prompt": "A는 B보다 크고, B는 C보다 크다. 가장 작은 것은 누구인가?"},
        {"id": "logic_prob_blue", "prompt": "빨간 공 2개와 파란 공 3개가 있다. 파란 공일 확률은?"},
        {"id": "logic_syllogism_yes", "prompt": "모든 A는 B다. 모든 B는 C다. 그러면 모든 A는 C인가?"},
        {"id": "logic_syllogism_unknown", "prompt": "모든 A는 B다. 어떤 B는 C다. 그러면 어떤 A는 C인가?"},
        {"id": "logic_sequence", "prompt": "2, 6, 7, 21, 23, 69, ? 다음 수는?"},
    ]
    save_json(path, rows)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_logic_mix(general_train: Path, logic_train: Path, out_path: Path, ratio: tuple[int, int] = (7, 3), seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    general_rows = _load_jsonl(general_train)
    logic_rows = _load_jsonl(logic_train)
    rng.shuffle(general_rows)
    rng.shuffle(logic_rows)
    total = min(len(general_rows) + len(logic_rows), max(60000, len(logic_rows)))
    general_n = int(total * (ratio[0] / float(sum(ratio))))
    logic_n = total - general_n
    mixed = general_rows[:general_n] + logic_rows[:logic_n]
    rng.shuffle(mixed)
    _write_jsonl(out_path, mixed)
    return {"path": str(out_path), "rows": len(mixed), "general_rows": general_n, "logic_rows": logic_n}


def _build_logic_tutor_mix(
    general_train: Path,
    logic_train: Path,
    tutor_train: Path,
    out_path: Path,
    ratio: tuple[int, int, int] = (70, 20, 10),
    target_rows: int = 90000,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    pools = {
        "general": _load_jsonl(general_train),
        "logic": _load_jsonl(logic_train),
        "tutor": _load_jsonl(tutor_train),
    }
    for rows in pools.values():
        rng.shuffle(rows)

    weight_sum = float(sum(ratio))
    desired = {
        "general": int(target_rows * (ratio[0] / weight_sum)),
        "logic": int(target_rows * (ratio[1] / weight_sum)),
        "tutor": int(target_rows * (ratio[2] / weight_sum)),
    }
    selected: Dict[str, List[Dict[str, Any]]] = {}
    for name, rows in pools.items():
        n = min(len(rows), desired[name])
        selected[name] = rows[:n]

    mixed: List[Dict[str, Any]] = selected["general"] + selected["logic"] + selected["tutor"]
    remaining_rows = target_rows - len(mixed)
    if remaining_rows > 0:
        leftovers = {
            "general": pools["general"][len(selected["general"]):],
            "logic": pools["logic"][len(selected["logic"]):],
            "tutor": pools["tutor"][len(selected["tutor"]):],
        }
        fill_order = ["general", "logic", "tutor"]
        idx = 0
        while remaining_rows > 0:
            name = fill_order[idx % len(fill_order)]
            idx += 1
            if not leftovers[name]:
                if not any(leftovers.values()):
                    break
                continue
            mixed.append(leftovers[name].pop())
            remaining_rows -= 1

    rng.shuffle(mixed)
    _write_jsonl(out_path, mixed)
    return {
        "path": str(out_path),
        "rows": len(mixed),
        "general_rows": sum(1 for row in mixed if row.get("segment_tag") == "ko_mainline_general_v1"),
        "logic_rows": sum(1 for row in mixed if row.get("segment_tag") == "ko_mainline_logic_v2"),
        "tutor_rows": sum(1 for row in mixed if row.get("segment_tag") == "ko_mainline_logic_tutor_v1"),
        "ratio_target": {"general": ratio[0], "logic": ratio[1], "tutor": ratio[2]},
    }


def build_datasets(spec: Dict[str, Any], root: Path, out_dir: Path) -> Dict[str, Any]:
    py = resolve_python(root)
    data_dir = root / "data"
    general_train = data_dir / "mainline_general_v3_clean_train.jsonl"
    general_eval = data_dir / "mainline_general_v3_clean_eval.jsonl"
    general_manifest = data_dir / "mainline_general_v3_clean.manifest.json"
    logic_train = data_dir / "mainline_logic_verified_v3_train.jsonl"
    logic_eval = data_dir / "mainline_logic_verified_v3_eval.jsonl"
    logic_manifest = data_dir / "mainline_logic_verified_v3.manifest.json"
    logic_mix = data_dir / "mainline_logic_mix_v3_train.jsonl"
    logic_repair_mix = data_dir / "mainline_logic_repair_mix_v3_train.jsonl"
    prompt_suite = data_dir / "mainline_prompt_suite_v1.json"
    logic_readout = data_dir / "mainline_logic_reasoning_readout_v1.json"

    runs: List[Dict[str, Any]] = []
    runs.append(
        run_subprocess(
            [
                py,
                "scripts/build_mainline_general_v3_clean.py",
                "--train_out",
                str(general_train),
                "--eval_out",
                str(general_eval),
                "--manifest_out",
                str(general_manifest),
            ],
            cwd=root,
            timeout=7200,
        )
    )
    runs.append(
        run_subprocess(
            [
                py,
                "scripts/build_mainline_logic_verified_v3.py",
                "--train_out",
                str(logic_train),
                "--eval_out",
                str(logic_eval),
                "--manifest_out",
                str(logic_manifest),
            ],
            cwd=root,
            timeout=7200,
        )
    )

    # Preserve general fluency while injecting verified logic.
    mix_report = _build_logic_mix(general_train, logic_train, logic_mix, ratio=(85, 15), seed=42)
    repair_mix_report = _build_logic_mix(general_train, logic_train, logic_repair_mix, ratio=(90, 10), seed=43)
    _write_prompt_suite(prompt_suite)
    _write_logic_readout(logic_readout)

    report = {
        "general": {
            "train": file_fingerprint(general_train),
            "eval": file_fingerprint(general_eval),
            "manifest": file_fingerprint(general_manifest),
            "train_rows": count_jsonl(general_train),
            "eval_rows": count_jsonl(general_eval),
        },
        "logic": {
            "train": file_fingerprint(logic_train),
            "eval": file_fingerprint(logic_eval),
            "manifest": file_fingerprint(logic_manifest),
            "train_rows": count_jsonl(logic_train),
            "eval_rows": count_jsonl(logic_eval),
            "mix": mix_report,
            "repair_mix": repair_mix_report,
        },
        "benchmarks": {
            "prompt_suite": file_fingerprint(prompt_suite),
            "logic_readout": file_fingerprint(logic_readout),
        },
        "runs": runs,
    }
    save_json(out_dir / "dataset_report.json", report)
    return report
