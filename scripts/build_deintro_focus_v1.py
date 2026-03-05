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
BAD_REPEAT_RE = re.compile(r"(.)\1{8,}")

INTRO_QUERY_RE = re.compile(r"(자기소개|본인 소개|너 소개|한 줄로 소개|안녕|반가워)")
INTRO_ANSWER_RE = re.compile(
    r"(질문 의도|한국어 AI|AI 어시스턴트|AI 도우미|반갑습니다|안녕하세요\.\s*저는|저는\s*spear1\.0)",
    re.IGNORECASE,
)
LEAK_RE = re.compile(r"(###\s*(Instruction|Response|지시|답변)|입력 없음|None)", re.IGNORECASE)
CODE_TRANSLATE_RE = re.compile(
    r"(코드|스크립트|함수|class\s|def\s|import\s|sql|번역|translate|영어로|프랑스어|HTML|Java|Python)",
    re.IGNORECASE,
)

DEF_END_RE = re.compile(r"(이란|란|정의|설명해줘|설명해주세요|설명|무엇인가요|무엇인가|무엇|뭐야|알려줘)\??$")
LOGIC_Q_RE = re.compile(r"(모든\s+\S+는\s+\S+다|거짓말쟁이|진실만 말하는 사람|두 질문|전제:)")

TERM_ALIASES: Dict[str, Tuple[str, ...]] = {
    "llm": ("LLM", "대규모 언어 모델", "언어 모델"),
    "카카오톡": ("카카오톡", "메신저"),
    "핸드폰": ("핸드폰", "휴대폰", "스마트폰"),
    "휴대폰": ("휴대폰", "핸드폰", "스마트폰"),
    "코틀린": ("코틀린", "프로그래밍 언어"),
    "안드로이드": ("안드로이드", "운영체제", "모바일 운영체제"),
    "타입스크립트": ("타입스크립트", "TypeScript", "프로그래밍 언어"),
    "파이썬": ("파이썬", "Python", "프로그래밍 언어"),
    "자바스크립트": ("자바스크립트", "JavaScript", "프로그래밍 언어"),
}


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


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
    if len(HANGUL_RE.findall(inp + out)) < 4:
        return False
    return True


def is_intro_or_leak(inp: str, out: str) -> bool:
    q = normalize_space(inp)
    a = normalize_space(out)
    if INTRO_QUERY_RE.search(q):
        return True
    if INTRO_ANSWER_RE.search(a):
        return True
    if LEAK_RE.search(q) or LEAK_RE.search(a):
        return True
    return False


def is_definition_query(inp: str) -> bool:
    s = normalize_space(inp)
    if DEF_END_RE.search(s):
        return True
    return bool(re.search(r"(이란|란)\??", s))


def has_final_consonant(ch: str) -> bool:
    if len(ch) != 1:
        return False
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return ((code - 0xAC00) % 28) != 0
    return False


def strip_trailing_particle(term: str) -> str:
    t = normalize_space(term)
    if len(t) < 2:
        return t
    suffixes = ("은", "는", "이", "가", "을", "를", "에", "로", "도", "만")
    for s in suffixes:
        if len(t) > len(s) + 1 and t.endswith(s):
            return t[: -len(s)]
    return t


def extract_term(inp: str) -> str:
    s = normalize_space(inp)
    for p in ("한국어로", "간단히", "짧게", "한 줄로", "한줄로", "핵심만", "쉽게", "정확히"):
        s = s.replace(p, " ")
    s = normalize_space(s)
    s = re.sub(r"[\"'`“”‘’(),.!?:;/\\\[\]{}]+", " ", s)
    s = normalize_space(s)
    m = re.search(r"(.+?)(?:이란|란|정의|설명해줘|설명해주세요|설명|무엇인가요|무엇인가|무엇|뭐야|알려줘)$", s)
    if m:
        s = normalize_space(m.group(1))
    if not s:
        return ""
    t = s.split(" ")[0]
    return strip_trailing_particle(t)


def alias_terms(term: str) -> Tuple[str, ...]:
    t = normalize_space(term)
    if not t:
        return tuple()
    key = t.lower()
    if key in TERM_ALIASES:
        return TERM_ALIASES[key]
    if t in TERM_ALIASES:
        return TERM_ALIASES[t]
    return (t,)


def answer_starts_with_any(out: str, terms: Sequence[str]) -> bool:
    a = normalize_space(out)
    for t in terms:
        tt = normalize_space(t)
        if not tt:
            continue
        if a.startswith(tt):
            return True
        if a.startswith(f"{tt}은") or a.startswith(f"{tt}는") or a.startswith(f"{tt}이") or a.startswith(f"{tt}가"):
            return True
    return False


def is_term_answer_aligned(term: str, out: str) -> bool:
    terms = alias_terms(term)
    if not terms:
        return False
    a = normalize_space(out)
    if is_intro_or_leak(term, a):
        return False
    if a.count("?") >= 2:
        return False
    if not answer_starts_with_any(a, terms):
        return False
    return any(t in a for t in terms)


def collect_definition_rows(paths: Sequence[Path], max_input: int, max_output: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        src = str(path).replace("\\", "/")
        for row in iter_jsonl(path):
            inp = pick_text(row, ("input", "instruction", "prompt", "question"))
            out = pick_text(row, ("output", "response", "answer", "completion", "target"))
            if not valid_pair(inp, out, max_input=max_input, max_output=max_output):
                continue
            if is_intro_or_leak(inp, out):
                continue
            if CODE_TRANSLATE_RE.search(inp) or CODE_TRANSLATE_RE.search(out):
                continue
            if not is_definition_query(inp):
                continue
            term = extract_term(inp)
            if len(term) < 2:
                continue
            if not is_term_answer_aligned(term, out):
                continue
            rows.append(make_row(inp, out, source=src, tier=str(row.get("_meta_quality_tier", "high"))))
    return rows


def looks_logic_answer(out: str) -> bool:
    a = normalize_space(out)
    hints = ("아니요", "성립", "보장", "거짓말쟁이", "진실", "같으면", "다르면")
    return any(h in a for h in hints)


def collect_logic_rows(paths: Sequence[Path], max_input: int, max_output: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        src = str(path).replace("\\", "/")
        for row in iter_jsonl(path):
            inp = pick_text(row, ("input", "instruction", "prompt", "question"))
            out = pick_text(row, ("output", "response", "answer", "completion", "target"))
            if not valid_pair(inp, out, max_input=max_input, max_output=max_output):
                continue
            if is_intro_or_leak(inp, out):
                continue
            if not LOGIC_Q_RE.search(inp):
                continue
            if not looks_logic_answer(out):
                continue
            rows.append(make_row(inp, out, source=src, tier=str(row.get("_meta_quality_tier", "high"))))
    return rows


def collect_general_rows(paths: Sequence[Path], max_input: int, max_output: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    questionish = re.compile(r"(왜|어떻게|무엇|뭐|뜻|설명|차이|비교|\?)")
    for path in paths:
        if not path.exists():
            continue
        src = str(path).replace("\\", "/")
        for row in iter_jsonl(path):
            inp = pick_text(row, ("input", "instruction", "prompt", "question"))
            out = pick_text(row, ("output", "response", "answer", "completion", "target"))
            if not valid_pair(inp, out, max_input=max_input, max_output=max_output):
                continue
            if is_intro_or_leak(inp, out):
                continue
            if CODE_TRANSLATE_RE.search(inp) or CODE_TRANSLATE_RE.search(out):
                continue
            if is_definition_query(inp):
                continue
            if LOGIC_Q_RE.search(inp):
                continue
            if not questionish.search(inp):
                continue
            if len(inp) > 120 or len(out) > 220:
                continue
            if len(re.findall(r"\d+\.", out)) >= 3:
                continue
            rows.append(make_row(inp, out, source=src, tier=str(row.get("_meta_quality_tier", "high"))))
    return rows


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


def add_synth_term_rows(rows: List[Dict[str, str]], seen: set[bytes], rnd: random.Random, per_term: int) -> int:
    term_defs: Dict[str, Sequence[str]] = {
        "LLM": [
            "LLM은 대규모 텍스트 데이터로 학습해 문장을 이해하고 생성하는 언어 모델입니다.",
            "LLM은 방대한 말뭉치를 학습해 요약, 질의응답, 생성 작업을 수행하는 대규모 언어 모델입니다.",
        ],
        "카카오톡": [
            "카카오톡은 카카오가 제공하는 모바일 메신저 서비스로 1:1 대화와 단체 채팅을 지원합니다.",
            "카카오톡은 한국에서 널리 쓰이는 메신저 앱으로 채팅, 음성·영상통화, 파일 공유를 제공합니다.",
        ],
        "핸드폰": [
            "핸드폰은 이동 중 통화와 문자, 데이터 통신이 가능한 휴대용 전화기입니다.",
            "핸드폰은 무선 통신망을 이용해 음성 통화와 인터넷 접속을 처리하는 휴대 기기입니다.",
        ],
        "휴대폰": [
            "휴대폰은 이동 중 통신과 앱 사용을 지원하는 개인용 이동 통신 기기입니다.",
            "휴대폰은 통화, 문자, 인터넷 기능을 제공하는 휴대형 통신 장치입니다.",
        ],
        "코틀린": [
            "코틀린은 JVM과 안드로이드에서 널리 쓰이는 정적 타입 프로그래밍 언어입니다.",
            "코틀린은 자바와 상호 운용이 가능한 현대적 프로그래밍 언어입니다.",
        ],
        "안드로이드": [
            "안드로이드는 구글이 개발한 모바일 운영체제로 스마트폰과 태블릿에서 사용됩니다.",
            "안드로이드는 앱 생태계가 큰 모바일 OS로 다양한 제조사의 기기에서 동작합니다.",
        ],
        "타입스크립트": [
            "타입스크립트는 자바스크립트에 타입 시스템을 추가한 프로그래밍 언어입니다.",
            "타입스크립트는 정적 타입 검사를 통해 대규모 자바스크립트 개발을 돕는 언어입니다.",
        ],
        "파이썬": [
            "파이썬은 문법이 간결하고 범용성이 높은 프로그래밍 언어입니다.",
            "파이썬은 데이터 분석, 웹 개발, 자동화에 널리 쓰이는 고수준 언어입니다.",
        ],
    }

    prompts = [
        "{term}이란?",
        "{term}란?",
        "{term} 정의",
        "{term} 설명해줘",
        "{term}가 뭐야?",
        "한국어로 {term}이란",
        "{term}를 한 줄로 설명해줘",
        "{term}이 무엇인지 알려줘",
        "{term} 뜻",
        "{term} 개념 설명",
        "{term}를 간단히 설명해줘",
        "{term}를 짧게 정의해줘",
        "{term}에 대해 알려줘",
        "{term}가 무엇인가요?",
        "{term}는 뭐하는 거야?",
        "{term} 기본 정의",
        "쉽게 {term} 설명",
        "한 문장으로 {term} 설명",
        "{term} 핵심만 설명해줘",
        "{term} 정리해줘",
    ]

    added = 0
    for term, defs in term_defs.items():
        for _ in range(int(per_term)):
            q = rnd.choice(prompts).format(term=term)
            a = normalize_space(rnd.choice(list(defs)))
            k = dedupe_key(q, a)
            if k in seen:
                continue
            seen.add(k)
            rows.append(make_row(q, a, source="synthetic/deintro_focus_v1_term"))
            added += 1
    return added


def add_synth_logic_rows(
    rows: List[Dict[str, str]],
    seen: set[bytes],
    rnd: random.Random,
    n_syllogism: int,
    n_liar: int,
) -> Tuple[int, int]:
    syll_added = 0
    liar_added = 0

    labels = [
        ("A", "B", "C"),
        ("X", "Y", "Z"),
        ("P", "Q", "R"),
        ("개발자", "직장인", "성인"),
        ("고양이", "포유류", "동물"),
        ("사과", "과일", "식품"),
        ("학생", "사람", "시민"),
        ("의사", "전문직", "직장인"),
        ("참새", "새", "동물"),
        ("장미", "꽃", "식물"),
        ("파이썬 개발자", "개발자", "직장인"),
    ]

    q_templates = [
        "모든 {a}는 {b}다. 일부 {b}는 {c}다. 그렇다면 일부 {a}는 {c}인가? 이유를 설명하라.",
        "전제: 모든 {a}는 {b}다. 일부 {b}는 {c}다. 결론: 일부 {a}는 {c}다. 이 결론은 반드시 성립하나?",
        "명제: 모든 {a}는 {b}이고, 일부 {b}는 {c}이다. 따라서 일부 {a}는 {c}라고 할 수 있나?",
        "모든 {a}가 {b}이고 일부 {b}가 {c}일 때, 일부 {a}가 {c}라는 결론이 항상 참인지 판단해줘.",
        "논리 문제: 모든 {a}는 {b}다. 일부 {b}는 {c}다. 일부 {a}는 {c}인가?",
        "집합 추론: A={a}, B={b}, C={c}일 때 '모든 A는 B, 일부 B는 C'에서 '일부 A는 C'가 따라오나?",
    ]
    a_templates = [
        "아니요. '모든 {a}는 {b}'와 '일부 {b}는 {c}'만으로는 '일부 {a}는 {c}'를 보장할 수 없습니다. {a}가 {c}와 겹치지 않는 경우가 가능하기 때문입니다.",
        "결론은 일반적으로 성립하지 않습니다. 일부 {b}가 {c}라는 사실만으로 {a}가 {c}에 포함된다고 단정할 수 없습니다.",
        "반드시 참이 아닙니다. {a} 집합이 {c}와 교집합이 없을 수도 있으므로 '일부 {a}는 {c}'를 도출할 수 없습니다.",
        "성립 보장은 불가능합니다. 주어진 전제는 {b}와 {c}의 부분 겹침만 말할 뿐, {a}와 {c}의 겹침을 직접 보장하지 않습니다.",
        "정답은 '아니요'입니다. 반례로 {a}가 전부 {c} 밖에 있어도 전제는 모두 참이 될 수 있습니다.",
        "일반적으로 결론이 따라오지 않습니다. 따라서 '일부 {a}는 {c}'는 추가 전제 없이는 확정할 수 없습니다.",
    ]

    for _ in range(int(n_syllogism)):
        a, b, c = rnd.choice(labels)
        q = rnd.choice(q_templates).format(a=a, b=b, c=c)
        ans = rnd.choice(a_templates).format(a=a, b=b, c=c)
        k = dedupe_key(q, ans)
        if k in seen:
            continue
        seen.add(k)
        rows.append(make_row(q, ans, source="synthetic/deintro_focus_v1_logic"))
        syll_added += 1

    liar_q = [
        "{x}와 {y} 중 한 명은 거짓말쟁이이고 다른 한 명은 진실만 말한다. 질문 2개로 구분하는 방법은?",
        "두 사람 {x}, {y}가 있다. 한 명은 항상 거짓말하고 한 명은 항상 진실을 말한다. 두 질문 절차를 설명해줘.",
        "{x}와 {y} 중 누가 거짓말쟁이인지 예/아니오 질문 두 번으로 식별하는 방법을 제시하라.",
        "{x}, {y}의 정체(거짓말쟁이/진실화자)를 질문 두 개만으로 판별하는 알고리즘을 알려줘.",
        "거짓말쟁이와 진실화자가 {x}, {y}로 주어졌을 때 두 질문으로 구분하는 방법을 설명하라.",
    ]
    liar_a = [
        "{x}에게 연속으로 두 번 묻습니다. 1) '{y}는 진실만 말하나요?' 2) '방금 네 대답은 사실이었나요?' 두 답이 같으면 {x}는 거짓말쟁이, 다르면 {x}는 진실화자입니다.",
        "{x}를 고정해 첫 질문으로 {y}의 정체를 묻고, 둘째 질문으로 직전 답의 진위를 묻습니다. 두 답이 동일하면 {x}=거짓말쟁이, 다르면 {x}=진실화자입니다.",
        "{x}에게 Q1: '{y}는 진실만 말하나요?' Q2: '방금 답이 사실인가요?'를 묻습니다. 답이 같으면 {x}는 거짓말쟁이, 다르면 진실화자이며 {y}는 반대입니다.",
        "{x}를 대상으로 질문 두 개를 던집니다. 첫 답과 둘째 답이 같으면 {x}를 거짓말쟁이로, 다르면 진실화자로 판정하면 됩니다.",
        "{x}에게 같은 맥락의 질문을 두 번 연속해 답의 일치 여부를 봅니다. 일치면 {x}=거짓말쟁이, 불일치면 {x}=진실화자입니다.",
    ]
    names = [
        ("X", "Y"),
        ("A", "B"),
        ("철수", "영희"),
        ("민수", "지수"),
        ("가", "나"),
        ("사람1", "사람2"),
        ("갑", "을"),
        ("왼쪽 사람", "오른쪽 사람"),
        ("첫 번째 사람", "두 번째 사람"),
    ]

    for _ in range(int(n_liar)):
        x, y = rnd.choice(names)
        q = rnd.choice(liar_q).format(x=x, y=y)
        ans = rnd.choice(liar_a).format(x=x, y=y)
        k = dedupe_key(q, ans)
        if k in seen:
            continue
        seen.add(k)
        rows.append(make_row(q, ans, source="synthetic/deintro_focus_v1_logic"))
        liar_added += 1

    return syll_added, liar_added


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build strict de-intro focused Korean dataset v1")
    p.add_argument("--base", default="data/by_field_ko_pure_v2.jsonl")
    p.add_argument("--term", default="data/term_focus_clean_v2.jsonl")
    p.add_argument("--out", default="data/deintro_focus_v1.jsonl")
    p.add_argument("--manifest", default="data/deintro_focus_v1.manifest.json")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max_input_chars", type=int, default=220)
    p.add_argument("--max_output_chars", type=int, default=260)
    p.add_argument("--take_general", type=int, default=4500)
    p.add_argument("--take_term", type=int, default=9000)
    p.add_argument("--take_logic", type=int, default=1800)
    p.add_argument("--synth_term_per", type=int, default=220)
    p.add_argument("--synth_logic_syllogism", type=int, default=1800)
    p.add_argument("--synth_logic_liar", type=int, default=900)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rnd = random.Random(int(args.seed))
    base_path = Path(args.base)
    term_path = Path(args.term)

    general_pool = collect_general_rows(
        paths=[base_path],
        max_input=int(args.max_input_chars),
        max_output=int(args.max_output_chars),
    )
    term_pool = collect_definition_rows(
        paths=[term_path, base_path],
        max_input=int(args.max_input_chars),
        max_output=int(args.max_output_chars),
    )
    logic_pool = collect_logic_rows(
        paths=[term_path, base_path],
        max_input=int(args.max_input_chars),
        max_output=int(args.max_output_chars),
    )

    rnd.shuffle(general_pool)
    rnd.shuffle(term_pool)
    rnd.shuffle(logic_pool)

    rows: List[Dict[str, str]] = []
    seen: set[bytes] = set()

    add_general = take_unique(general_pool, int(args.take_general), seen, rows)
    add_term = take_unique(term_pool, int(args.take_term), seen, rows)
    add_logic = take_unique(logic_pool, int(args.take_logic), seen, rows)
    add_s_term = add_synth_term_rows(rows, seen, rnd=rnd, per_term=int(args.synth_term_per))
    add_s_syll, add_s_liar = add_synth_logic_rows(
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
        "name": str(out_path.stem),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "inputs": {"base": str(base_path).replace("\\", "/"), "term": str(term_path).replace("\\", "/")},
        "pool": {
            "general_pool": len(general_pool),
            "term_pool": len(term_pool),
            "logic_pool": len(logic_pool),
        },
        "rows": {
            "general_added": int(add_general),
            "term_added": int(add_term),
            "logic_added": int(add_logic),
            "synth_term_added": int(add_s_term),
            "synth_logic_syllogism_added": int(add_s_syll),
            "synth_logic_liar_added": int(add_s_liar),
            "final": len(rows),
        },
        "config": {
            "max_input_chars": int(args.max_input_chars),
            "max_output_chars": int(args.max_output_chars),
            "take_general": int(args.take_general),
            "take_term": int(args.take_term),
            "take_logic": int(args.take_logic),
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
