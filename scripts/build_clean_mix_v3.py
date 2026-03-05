from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SPACE_RE = re.compile(r"\s+")
HANGUL_RE = re.compile(r"[가-힣]")
BAD_REPEAT_RE = re.compile(r"(.)\1{10,}")
INTRO_OUT_RE = re.compile(r"(AI 어시스턴트|AI 도우미|질문 의도를|한 줄로 말하면|요약하면|간단히 말해)", re.IGNORECASE)
DEF_Q_RE = re.compile(r"(이란|란|뭐야|무엇이야|정의|설명)\??$", re.IGNORECASE)


TERM_REQUIRED_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "코틀린": ("언어", "JVM", "자바", "안드로이드"),
    "안드로이드": ("운영체제", "플랫폼", "OS", "구글", "모바일"),
    "타입스크립트": ("타입", "자바스크립트", "언어"),
    "LLM": ("언어 모델", "대규모", "텍스트", "생성"),
    "카카오톡": ("메신저", "카카오", "채팅", "서비스"),
    "핸드폰": ("휴대", "전화", "통화", "스마트폰"),
    "휴대폰": ("휴대", "전화", "통화", "스마트폰"),
}


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


def make_row(inp: str, out: str, source: str, tier: str = "high") -> Dict[str, str]:
    return {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": str(tier or "high"),
        "input": normalize_space(inp),
        "output": normalize_space(out),
        "_meta_source_file": source,
    }


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def pick_text(row: Dict, keys: Sequence[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        s = normalize_space(v)
        if s:
            return s
    return ""


def valid_pair(inp: str, out: str, max_input: int, max_output: int) -> bool:
    if not inp or not out:
        return False
    if len(inp) < 2 or len(inp) > int(max_input):
        return False
    if len(out) < 8 or len(out) > int(max_output):
        return False
    if "\ufffd" in inp or "\ufffd" in out:
        return False
    if BAD_REPEAT_RE.search(out):
        return False
    if INTRO_OUT_RE.search(out):
        return False
    if not HANGUL_RE.search(inp + out):
        return False
    latin_n = len(re.findall(r"[A-Za-z]", out))
    hangul_n = len(HANGUL_RE.findall(out))
    if latin_n >= 24 and hangul_n < 4:
        return False
    return True


def has_final_consonant(term: str) -> bool:
    s = normalize_space(term)
    if not s:
        return False
    ch = s[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return ((code - 0xAC00) % 28) != 0
    return False


def josa_subject(term: str) -> str:
    return "이" if has_final_consonant(term) else "가"


def josa_object(term: str) -> str:
    return "을" if has_final_consonant(term) else "를"


def josa_iran(term: str) -> str:
    return "이란" if has_final_consonant(term) else "란"


def looks_definition_query(text: str) -> bool:
    s = normalize_space(text)
    return bool(DEF_Q_RE.search(s)) or ("이란" in s) or ("란" in s) or ("뭐야" in s)


def extract_term(text: str) -> str:
    s = normalize_space(text)
    for k in ("한국어로", "간단히", "짧게", "핵심만", "한 줄로", "설명해줘", "알려줘", "답해줘", "부탁해", "요약해줘"):
        s = s.replace(k, " ")
    s = normalize_space(re.sub(r"[?.,!\"'`()\[\]{}:;/\\]+", " ", s))
    m = re.search(r"(.+?)(?:이란|란|는 뭐야|이 뭐야|가 뭐야|뭐야|정의|설명)$", s)
    if m:
        s = m.group(1).strip()
    if not s:
        return ""
    return s.split(" ")[0].strip()


def semantically_aligned(term: str, out: str) -> bool:
    needed = TERM_REQUIRED_KEYWORDS.get(normalize_space(term))
    if not needed:
        return True
    low = normalize_space(out).lower()
    return any(k.lower() in low for k in needed)


def collect_base_rows(path: Path, max_input: int, max_output: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    for row in iter_jsonl(path):
        inp = pick_text(row, ("input", "instruction", "prompt", "question"))
        out = pick_text(row, ("output", "response", "answer", "completion", "target"))
        if not valid_pair(inp, out, max_input=max_input, max_output=max_output):
            continue
        # Remove symbolic-logic prompts from noisy pool; they are injected cleanly later.
        if re.search(r"(모든\s+[A-Z]\s*는\s+[A-Z]\s*다|일부\s+[A-Z]\s*는\s+[A-Z]\s*다|거짓말쟁이|진실만 말하는)", inp):
            continue
        rows.append(make_row(inp, out, source=str(path).replace("\\", "/"), tier=str(row.get("_meta_quality_tier", "high"))))
    return rows


def collect_term_definition_rows(path: Path, max_input: int, max_output: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    for row in iter_jsonl(path):
        inp = pick_text(row, ("input", "instruction", "prompt", "question"))
        out = pick_text(row, ("output", "response", "answer", "completion", "target"))
        if not valid_pair(inp, out, max_input=max_input, max_output=max_output):
            continue
        if not looks_definition_query(inp):
            continue
        term = extract_term(inp)
        if len(term) < 2:
            continue
        if term.lower() not in out.lower():
            continue
        if not semantically_aligned(term, out):
            continue
        rows.append(make_row(inp, out, source=str(path).replace("\\", "/"), tier=str(row.get("_meta_quality_tier", "high"))))
    return rows


def add_synthetic_term_rows(rows: List[Dict[str, str]], seen: set[bytes], rnd: random.Random, per_term: int) -> int:
    term_defs: Dict[str, Sequence[str]] = {
        "LLM": [
            "LLM은 대규모 텍스트 데이터로 학습해 문장을 이해하고 생성하는 대규모 언어 모델입니다.",
            "LLM은 사람의 언어 패턴을 학습해 요약, 번역, 질의응답 같은 작업을 수행하는 언어 모델입니다.",
        ],
        "카카오톡": [
            "카카오톡은 카카오가 제공하는 메신저 서비스로 채팅, 음성·영상통화, 파일 공유 기능을 지원합니다.",
            "카카오톡은 한국에서 널리 쓰이는 모바일 메신저 앱으로 1:1 대화와 단체 채팅을 제공합니다.",
        ],
        "핸드폰": [
            "핸드폰은 이동 중 통화와 문자, 인터넷 사용이 가능한 휴대용 전화기입니다.",
            "핸드폰은 무선 통신망을 통해 음성 통화와 데이터 통신을 처리하는 휴대 기기입니다.",
        ],
        "휴대폰": [
            "휴대폰은 어디서나 통화와 메시지 전송이 가능한 개인용 이동 통신 기기입니다.",
            "휴대폰은 스마트폰을 포함하는 개념으로, 이동 중 통신과 앱 사용을 지원하는 기기입니다.",
        ],
    }

    added = 0
    for term, defs in term_defs.items():
        subj = josa_subject(term)
        obj = josa_object(term)
        iran = josa_iran(term)
        prompts = [
            f"{term}{iran}",
            f"{term}{subj} 뭐야?",
            f"{term} 정의",
            f"{term} 설명해줘",
            f"한국어로 {term}{iran}",
            f"{term}{obj} 한 줄로 설명해줘",
            f"{term}{subj} 무엇인지 알려줘",
        ]
        for _ in range(int(per_term)):
            q = rnd.choice(prompts)
            a = normalize_space(rnd.choice(list(defs)))
            k = dedupe_key(q, a)
            if k in seen:
                continue
            seen.add(k)
            rows.append(make_row(q, a, source="synthetic/clean_mix_v3_term"))
            added += 1
    return added


def add_synthetic_logic_rows(rows: List[Dict[str, str]], seen: set[bytes], rnd: random.Random, n_syllogism: int, n_liar: int) -> Tuple[int, int]:
    syll_added = 0
    liar_added = 0

    labels = [
        ("A", "B", "C"),
        ("고양이", "포유류", "동물"),
        ("개발자", "직장인", "성인"),
        ("학생", "사람", "운동선수"),
    ]

    syll_q_tpl = [
        "모든 {a}는 {b}다. 일부 {b}는 {c}다. 그렇다면 일부 {a}는 {c}인가? 이유를 설명하라.",
        "전제: 모든 {a}는 {b}다. 일부 {b}는 {c}다. 결론: 일부 {a}는 {c}다. 이 결론은 반드시 성립하나?",
        "모든 {a}는 {b}이고, 일부 {b}는 {c}라고 하자. 그때 일부 {a}가 {c}라고 단정할 수 있나?",
    ]
    syll_a_tpl = [
        "아니요. '모든 {a}는 {b}'와 '일부 {b}는 {c}'만으로는 '일부 {a}는 {c}'를 보장할 수 없습니다. {a} 집합이 {c}와 겹치지 않는 경우도 가능하기 때문입니다.",
        "결론은 일반적으로 성립하지 않습니다. 일부 {b}가 {c}라는 사실만으로 {a}가 {c}에 포함된다고 할 근거는 없습니다.",
    ]

    for _ in range(int(n_syllogism)):
        a, b, c = rnd.choice(labels)
        q = rnd.choice(syll_q_tpl).format(a=a, b=b, c=c)
        ans = rnd.choice(syll_a_tpl).format(a=a, b=b, c=c)
        k = dedupe_key(q, ans)
        if k in seen:
            continue
        seen.add(k)
        rows.append(make_row(q, ans, source="synthetic/clean_mix_v3_logic"))
        syll_added += 1

    liar_q_tpl = [
        "거짓말쟁이와 진실만 말하는 사람이 있다. 두 질문만으로 누가 누구인지 구분하는 방법을 제시하라.",
        "한 명은 항상 거짓말하고 다른 한 명은 항상 진실을 말한다. 질문 두 번으로 구분하는 절차를 설명해줘.",
        "진실만 말하는 사람 1명과 거짓말쟁이 1명이 있다. 두 개의 예/아니오 질문으로 식별하는 법은?",
    ]
    liar_a_tpl = [
        "같은 사람 X에게 두 번 묻습니다. 1) '다른 사람 Y가 진실만 말하나요?' 2) '방금 네 대답은 사실이었나요?' 두 답이 같으면 X는 거짓말쟁이, 다르면 X는 진실화자이며 Y는 반대입니다.",
        "절차는 한 사람 X를 고정해 질문 2개를 던지는 것입니다. 먼저 Y의 정체를 묻고, 이어서 X의 직전 답변이 사실인지 묻습니다. 두 답이 동일하면 X는 거짓말쟁이, 다르면 X는 진실을 말하는 사람입니다.",
    ]

    for _ in range(int(n_liar)):
        q = rnd.choice(liar_q_tpl)
        ans = rnd.choice(liar_a_tpl)
        k = dedupe_key(q, ans)
        if k in seen:
            continue
        seen.add(k)
        rows.append(make_row(q, ans, source="synthetic/clean_mix_v3_logic"))
        liar_added += 1

    return syll_added, liar_added


def take_unique(rows: Sequence[Dict[str, str]], n: int, seen: set[bytes], dst: List[Dict[str, str]]) -> int:
    added = 0
    for row in rows:
        if added >= int(n):
            break
        k = dedupe_key(row["input"], row["output"])
        if k in seen:
            continue
        seen.add(k)
        dst.append(row)
        added += 1
    return added


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build clean mixed Korean dataset v3")
    p.add_argument("--base", default="data/by_field_ko_pure_v2.jsonl")
    p.add_argument("--term", default="data/term_focus_clean_v2.jsonl")
    p.add_argument("--out", default="data/clean_mix_v3.jsonl")
    p.add_argument("--manifest", default="data/clean_mix_v3.manifest.json")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--take_base", type=int, default=18000)
    p.add_argument("--take_term_def", type=int, default=9000)
    p.add_argument("--max_input_chars", type=int, default=240)
    p.add_argument("--max_output_chars", type=int, default=300)
    p.add_argument("--synth_term_per", type=int, default=260)
    p.add_argument("--synth_logic_syllogism", type=int, default=1800)
    p.add_argument("--synth_logic_liar", type=int, default=900)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rnd = random.Random(int(args.seed))
    dataset_name = Path(args.out).stem

    base_rows = collect_base_rows(Path(args.base), max_input=int(args.max_input_chars), max_output=int(args.max_output_chars))
    term_rows = collect_term_definition_rows(Path(args.term), max_input=int(args.max_input_chars), max_output=int(args.max_output_chars))

    rnd.shuffle(base_rows)
    rnd.shuffle(term_rows)

    rows: List[Dict[str, str]] = []
    seen: set[bytes] = set()

    add_base = take_unique(base_rows, int(args.take_base), seen, rows)
    add_term = take_unique(term_rows, int(args.take_term_def), seen, rows)
    add_term_synth = add_synthetic_term_rows(rows, seen, rnd=rnd, per_term=int(args.synth_term_per))
    add_logic_synth, add_liar_synth = add_synthetic_logic_rows(
        rows,
        seen,
        rnd=rnd,
        n_syllogism=int(args.synth_logic_syllogism),
        n_liar=int(args.synth_logic_liar),
    )

    rnd.shuffle(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "name": str(dataset_name),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "inputs": {"base": str(Path(args.base)).replace("\\", "/"), "term": str(Path(args.term)).replace("\\", "/")},
        "rows": {
            "base_added": int(add_base),
            "term_def_added": int(add_term),
            "synth_term_added": int(add_term_synth),
            "synth_logic_syllogism_added": int(add_logic_synth),
            "synth_logic_liar_added": int(add_liar_synth),
            "final": len(rows),
        },
        "pool": {"base_pool": len(base_rows), "term_pool": len(term_rows)},
        "config": {
            "take_base": int(args.take_base),
            "take_term_def": int(args.take_term_def),
            "max_input_chars": int(args.max_input_chars),
            "max_output_chars": int(args.max_output_chars),
            "synth_term_per": int(args.synth_term_per),
            "synth_logic_syllogism": int(args.synth_logic_syllogism),
            "synth_logic_liar": int(args.synth_logic_liar),
        },
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] rows={len(rows)} -> {out_path}")
    print(f"[done] manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
