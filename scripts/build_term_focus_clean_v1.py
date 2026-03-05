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
GENERIC_OUT_PAT = re.compile(
    r"(한 줄로 말하면|요약하면|핵심은|질문 의도를|간결하고 정확|필요한 정보를 짧고 분명|필요한 정보를 빠르게 정리)",
    flags=re.IGNORECASE,
)

TERM_REQUIRED_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "코틀린": ("언어", "JVM", "자바", "안드로이드"),
    "안드로이드": ("운영체제", "플랫폼", "OS", "구글", "모바일"),
    "자바": ("언어", "JVM", "객체지향"),
    "스프링": ("프레임워크", "자바", "백엔드", "DI"),
    "리액트": ("라이브러리", "UI", "자바스크립트", "프론트엔드"),
    "타입스크립트": ("정적 타입", "자바스크립트", "언어"),
    "파이썬": ("언어", "문법", "라이브러리"),
    "데이터베이스": ("데이터", "저장", "조회", "시스템"),
    "API": ("인터페이스", "통신", "요청", "응답"),
    "클라우드": ("인터넷", "컴퓨팅", "자원", "인프라"),
}


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def dedupe_key(inp: str, out: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(normalize_space(inp).lower().encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(normalize_space(out).lower().encode("utf-8", errors="ignore"))
    return h.digest()


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


def contains_term(term: str, text: str) -> bool:
    t = normalize_space(term).lower()
    s = normalize_space(text).lower()
    return bool(t) and (t in s)


def semantically_aligned(term: str, out: str) -> bool:
    needed = TERM_REQUIRED_KEYWORDS.get(normalize_space(term))
    if not needed:
        return True
    low = normalize_space(out).lower()
    return any(k.lower() in low for k in needed)


def make_row(inp: str, out: str, source: str) -> Dict[str, str]:
    return {
        "task_type": "korean",
        "segment_tag": "ko",
        "language": "ko",
        "_meta_quality_tier": "high",
        "input": normalize_space(inp),
        "output": normalize_space(out),
        "_meta_source_file": source,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build clean term-focus dataset v1")
    p.add_argument("--hq", default="data/quality/hq_ko_chat_v1.jsonl")
    p.add_argument("--out", default="data/term_focus_clean_v1.jsonl")
    p.add_argument("--manifest", default="data/term_focus_clean_v1.manifest.json")
    p.add_argument("--seed", type=int, default=83)
    p.add_argument("--take_hq_term", type=int, default=20000)
    p.add_argument("--take_hq_general", type=int, default=8000)
    p.add_argument("--take_intro", type=int, default=320)
    p.add_argument("--synthetic_per_term", type=int, default=240)
    return p.parse_args()


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


def is_intro_input(text: str) -> bool:
    s = normalize_space(text)
    return bool(re.search(r"(자기소개|본인 소개|너를 .*소개|한 줄 소개|self[ -]?intro)", s, flags=re.IGNORECASE))


def is_intro_output(text: str) -> bool:
    s = normalize_space(text)
    needles = [
        "한국어 AI",
        "AI 어시스턴트",
        "AI 도우미",
        "질문 의도를 빠르게 파악",
        "필요한 정보를 짧고 분명하게 정리",
    ]
    return any(n in s for n in needles)


def is_generic_style_output(text: str) -> bool:
    s = normalize_space(text)
    return bool(GENERIC_OUT_PAT.search(s))


def looks_definition_query(text: str) -> bool:
    s = normalize_space(text)
    patterns = [
        r"(이란|란)\??$",
        r"(뭐야|무엇이야)\??$",
        r"(정의|설명)\s*해줘\??$",
        r"알려줘\??$",
    ]
    return any(re.search(p, s) for p in patterns)


def extract_term(text: str) -> str:
    s = normalize_space(text)
    # Remove wrappers/fillers that cause noisy keys.
    fillers = [
        "한국어로",
        "간단히",
        "짧게",
        "핵심만",
        "한 줄로",
        "설명해줘",
        "알려줘",
        "답해줘",
        "부탁해",
        "요약해줘",
    ]
    for f in fillers:
        s = s.replace(f, " ")
    s = normalize_space(s)

    m = re.search(r"(.+?)(?:이란|란|는 뭐야|이 뭐야|가 뭐야|뭐야|정의)$", s)
    if m:
        s = m.group(1).strip()
    s = re.sub(r"[?.,!\"'`()\[\]{}:;/\\]+", " ", s)
    s = normalize_space(s)
    if not s:
        return ""
    # Prefer the first token for phrase-like questions.
    tok = s.split(" ")[0]
    return tok.strip()


def valid_pair(inp: str, out: str) -> bool:
    if not inp or not out:
        return False
    if len(inp) < 2 or len(inp) > 280:
        return False
    if len(out) < 8 or len(out) > 420:
        return False
    if re.search(r"(.)\1{10,}", out):
        return False
    if not HANGUL_RE.search(inp + out):
        return False
    return True


def add_synth_term_rows(
    rows: List[Dict[str, str]],
    seen: set[bytes],
    per_term: int,
    rnd: random.Random,
    source_tag: str,
) -> int:
    terms: Dict[str, Sequence[str]] = {
        "코틀린": [
            "코틀린은 JVM 기반으로 동작하며 간결한 문법과 널 안전성으로 안드로이드 개발에 많이 쓰이는 프로그래밍 언어입니다.",
            "코틀린은 자바와 상호 운용이 가능하고 보일러플레이트 코드를 줄여 생산성을 높이는 현대적 언어입니다.",
            "코틀린은 자바와 100% 상호 운용되며 안드로이드 공식 지원을 받는 현대적인 언어입니다.",
            "코틀린은 null 안전성과 간결한 표현력이 강점인 JVM 기반 언어입니다.",
        ],
        "안드로이드": [
            "안드로이드는 구글이 개발한 모바일 운영체제로 스마트폰·태블릿·웨어러블 등 다양한 기기에서 사용됩니다.",
            "안드로이드는 리눅스 커널 기반 운영체제로 앱 생태계가 크고 기기 선택 폭이 넓은 플랫폼입니다.",
            "안드로이드는 리눅스 커널을 기반으로 한 구글의 모바일 운영체제 플랫폼입니다.",
            "안드로이드는 다양한 제조사 기기에서 동작하는 오픈 생태계의 모바일 OS입니다.",
        ],
        "자바": [
            "자바는 객체지향 언어로 이식성이 높고 서버·안드로이드·기업 시스템에서 널리 사용됩니다.",
            "자바는 JVM 위에서 실행되어 플랫폼 독립성이 높고 안정적인 생태계를 가진 언어입니다.",
            "자바는 JVM을 통해 다양한 환경에서 실행되는 범용 프로그래밍 언어입니다.",
            "자바는 장기적으로 안정적인 라이브러리 생태계를 갖춘 객체지향 언어입니다.",
        ],
        "스프링": [
            "스프링은 자바 기반 애플리케이션 개발을 단순화하는 프레임워크로 DI와 AOP를 핵심으로 합니다.",
            "스프링은 웹·데이터 접근·보안 등을 모듈화해 대규모 백엔드 개발에 많이 사용됩니다.",
            "스프링은 자바 백엔드 개발에서 의존성 주입을 제공해 구조를 깔끔하게 만드는 프레임워크입니다.",
            "스프링은 엔터프라이즈급 백엔드 서비스를 빠르게 구축하도록 돕는 자바 프레임워크입니다.",
        ],
        "리액트": [
            "리액트는 UI를 컴포넌트 단위로 구성하는 자바스크립트 라이브러리로 상태 기반 렌더링에 강합니다.",
            "리액트는 선언형 방식으로 화면을 구성해 재사용성과 유지보수성을 높이기 좋은 프론트엔드 도구입니다.",
            "리액트는 컴포넌트 중심으로 UI를 구성하는 프론트엔드 라이브러리입니다.",
            "리액트는 상태 변화에 따라 화면을 효율적으로 갱신하는 자바스크립트 UI 라이브러리입니다.",
        ],
        "타입스크립트": [
            "타입스크립트는 자바스크립트에 정적 타입을 더해 대규모 코드베이스의 안정성과 생산성을 높이는 언어입니다.",
            "타입스크립트는 컴파일 단계에서 타입 오류를 잡아 협업과 리팩터링에 유리한 개발 경험을 제공합니다.",
            "타입스크립트는 자바스크립트 기반에 타입 시스템을 결합해 안정성을 높인 언어입니다.",
            "타입스크립트는 정적 타입 검사를 통해 런타임 오류를 줄이는 개발 언어입니다.",
        ],
        "파이썬": [
            "파이썬은 문법이 간결하고 읽기 쉬워 자동화, 데이터 분석, 웹 개발, AI에 널리 쓰이는 언어입니다.",
            "파이썬은 라이브러리 생태계가 풍부해 입문과 실무 모두에서 활용도가 높은 범용 언어입니다.",
            "파이썬은 가독성이 높은 문법과 풍부한 라이브러리로 다양한 분야에 쓰이는 언어입니다.",
            "파이썬은 데이터 분석과 AI 개발에서 널리 활용되는 범용 프로그래밍 언어입니다.",
        ],
        "데이터베이스": [
            "데이터베이스는 데이터를 구조적으로 저장하고 조회·수정·삭제를 효율적으로 처리하는 시스템입니다.",
            "데이터베이스는 대량의 정보를 일관성 있게 관리하기 위한 저장·검색 인프라입니다.",
            "데이터베이스는 데이터를 체계적으로 저장하고 빠르게 검색하도록 설계된 시스템입니다.",
            "데이터베이스는 서비스에서 필요한 데이터를 안정적으로 관리하는 핵심 저장소입니다.",
        ],
        "API": [
            "API는 서로 다른 소프트웨어가 정해진 규칙으로 기능과 데이터를 주고받게 하는 인터페이스입니다.",
            "API는 시스템 간 통신 규약을 정의해 서비스 연동과 확장을 쉽게 만드는 접점입니다.",
            "API는 애플리케이션 사이에서 요청과 응답을 주고받는 표준 인터페이스입니다.",
            "API는 외부 서비스 기능을 프로그램에서 호출할 수 있게 만든 연결 규약입니다.",
        ],
        "클라우드": [
            "클라우드는 인터넷을 통해 서버·스토리지·네트워크 같은 컴퓨팅 자원을 유연하게 쓰는 방식입니다.",
            "클라우드는 필요한 만큼 자원을 사용하고 비용을 지불하는 온디맨드 인프라 모델입니다.",
            "클라우드는 인터넷 기반으로 컴퓨팅 자원을 탄력적으로 제공하는 인프라 방식입니다.",
            "클라우드는 서버를 직접 보유하지 않고 필요한 자원을 빌려 쓰는 운영 모델입니다.",
        ],
    }

    added = 0
    for term, defs in terms.items():
        subj = josa_subject(term)
        obj = josa_object(term)
        iran = josa_iran(term)
        prompts = [
            f"{term}{iran}",
            f"{term}{subj} 뭐야",
            f"{term} 정의",
            f"{term} 설명해줘",
            f"한국어로 {term}{iran}",
            f"간단히 {term} 설명해줘",
            f"{term}{obj} 한 줄로 설명해줘",
            f"{term}{subj} 무엇인지 알려줘",
            f"{term} 개념 설명해줘",
            f"{term}{obj} 쉽게 설명해줘",
            f"{term} 핵심만 설명해줘",
            f"{term} 뜻 알려줘",
        ]
        for _ in range(int(per_term)):
            q = rnd.choice(prompts)
            a = normalize_space(rnd.choice(list(defs)))
            k = dedupe_key(q, a)
            if k in seen:
                continue
            seen.add(k)
            rows.append(make_row(q, a, source=f"synthetic/{source_tag}"))
            added += 1
    return added


def main() -> None:
    args = parse_args()
    rnd = random.Random(int(args.seed))
    hq_path = Path(args.hq)
    dataset_name = Path(args.out).stem
    if not hq_path.exists():
        raise FileNotFoundError(f"hq dataset not found: {hq_path}")

    hq_term: List[Dict[str, str]] = []
    hq_general: List[Dict[str, str]] = []
    intro_rows: List[Dict[str, str]] = []

    for row in iter_jsonl(hq_path):
        inp = normalize_space(str(row.get("input", "")))
        out = normalize_space(str(row.get("output", "")))
        if not valid_pair(inp, out):
            continue

        intro_in = is_intro_input(inp)
        intro_out = is_intro_output(out)
        if intro_in:
            if intro_out:
                intro_rows.append(make_row(inp, out, source=str(hq_path).replace("\\", "/")))
            continue
        if intro_out or is_generic_style_output(out):
            # Drop generic self-intro / style-template outputs for non-intro inputs.
            continue

        if looks_definition_query(inp):
            term = extract_term(inp)
            # keep only aligned term-definition pairs
            if len(term) >= 2 and contains_term(term, out) and semantically_aligned(term, out):
                hq_term.append(make_row(inp, out, source=str(hq_path).replace("\\", "/")))
            continue

        hq_general.append(make_row(inp, out, source=str(hq_path).replace("\\", "/")))

    rnd.shuffle(hq_term)
    rnd.shuffle(hq_general)
    rnd.shuffle(intro_rows)

    rows: List[Dict[str, str]] = []
    seen: set[bytes] = set()

    def take(src: Sequence[Dict[str, str]], n: int) -> int:
        added = 0
        for row in src[: int(n)]:
            k = dedupe_key(row["input"], row["output"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(row)
            added += 1
        return added

    take_term = take(hq_term, int(args.take_hq_term))
    take_general = take(hq_general, int(args.take_hq_general))
    take_intro = take(intro_rows, int(args.take_intro))
    synth_added = add_synth_term_rows(
        rows,
        seen,
        per_term=int(args.synthetic_per_term),
        rnd=rnd,
        source_tag=str(dataset_name),
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
        "inputs": {"hq": str(hq_path).replace("\\", "/")},
        "rows": {
            "hq_term_added": int(take_term),
            "hq_general_added": int(take_general),
            "hq_intro_added": int(take_intro),
            "synthetic_term_added": int(synth_added),
            "final": len(rows),
        },
        "pool": {
            "hq_term_pool": len(hq_term),
            "hq_general_pool": len(hq_general),
            "hq_intro_pool": len(intro_rows),
        },
        "config": {
            "take_hq_term": int(args.take_hq_term),
            "take_hq_general": int(args.take_hq_general),
            "take_intro": int(args.take_intro),
            "synthetic_per_term": int(args.synthetic_per_term),
        },
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] rows={len(rows)} -> {out_path}")
    print(f"[done] manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
