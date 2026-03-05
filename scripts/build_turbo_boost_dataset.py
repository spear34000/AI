from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_rows() -> list[dict]:
    seeds_ko = [
        (
            "파이썬 예외처리 기본 예시를 보여줘.",
            "기본 구조는 `try / except` 입니다.\n\n```python\ntry:\n    n = int(input('숫자 입력: '))\n    result = 10 / n\n    print(result)\nexcept ValueError:\n    print('숫자를 입력해야 합니다.')\nexcept ZeroDivisionError:\n    print('0으로 나눌 수 없습니다.')\n```\n\n필요하면 마지막에 `finally`를 추가해 정리 코드를 실행할 수 있습니다.",
        ),
        (
            "파이썬 리스트 컴프리헨션 예시를 간단히 알려줘.",
            "가장 기본적인 예시는 아래와 같습니다.\n\n```python\nnums = [1, 2, 3, 4, 5]\nsquares = [x * x for x in nums if x % 2 == 1]\nprint(squares)  # [1, 9, 25]\n```\n\n`for`와 `if`를 한 줄로 결합해 가독성을 높일 수 있습니다.",
        ),
        (
            "파이썬 딕셔너리 순회 예시를 보여줘.",
            "키-값을 동시에 순회할 때는 `items()`를 사용합니다.\n\n```python\nuser = {'name': 'mina', 'age': 27, 'role': 'admin'}\nfor k, v in user.items():\n    print(k, v)\n```\n\n키만 필요하면 `for k in user:` 형태도 자주 씁니다.",
        ),
        (
            "파이썬에서 파일 읽기 기본 코드를 알려줘.",
            "컨텍스트 매니저(`with`)를 사용하면 자동으로 파일이 닫혀 안전합니다.\n\n```python\nwith open('sample.txt', 'r', encoding='utf-8') as f:\n    text = f.read()\nprint(text)\n```\n\n한 줄씩 처리하려면 `for line in f:` 패턴을 쓰면 됩니다.",
        ),
        (
            "파이썬 함수 기본 예시를 보여줘.",
            "입력과 반환값이 있는 간단한 함수 예시입니다.\n\n```python\ndef add(a: int, b: int) -> int:\n    return a + b\n\nprint(add(2, 3))  # 5\n```\n\n작은 단위 함수로 분리하면 테스트와 재사용이 쉬워집니다.",
        ),
        (
            "파이썬 클래스 기본 예시를 알려줘.",
            "생성자와 메서드를 포함한 최소 클래스 예시입니다.\n\n```python\nclass Counter:\n    def __init__(self):\n        self.value = 0\n\n    def inc(self):\n        self.value += 1\n\nc = Counter()\nc.inc()\nprint(c.value)  # 1\n```\n\n상태를 묶어 다룰 때 클래스를 사용합니다.",
        ),
        (
            "자바스크립트 async/await 예시를 보여줘.",
            "비동기 API 호출의 기본 패턴입니다.\n\n```javascript\nasync function loadUser(id) {\n  try {\n    const res = await fetch(`/api/users/${id}`);\n    if (!res.ok) throw new Error('request failed');\n    const data = await res.json();\n    console.log(data);\n  } catch (err) {\n    console.error(err.message);\n  }\n}\n```\n\n`await`는 `async` 함수 안에서만 사용할 수 있습니다.",
        ),
        (
            "TypeScript interface 예시를 간단히 보여줘.",
            "객체 구조를 명시할 때 `interface`를 사용합니다.\n\n```typescript\ninterface User {\n  id: number;\n  name: string;\n  active: boolean;\n}\n\nconst u: User = { id: 1, name: 'mina', active: true };\n```\n\n컴파일 시점에 속성 누락/타입 오류를 잡을 수 있습니다.",
        ),
        (
            "SQL LEFT JOIN 기본 예시를 알려줘.",
            "왼쪽 테이블 기준으로 매칭 결과를 가져옵니다.\n\n```sql\nSELECT u.id, u.name, o.order_id\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id;\n```\n\n주문이 없는 사용자도 결과에 포함된다는 점이 핵심입니다.",
        ),
        (
            "파이썬에서 JSON 파일 읽고 쓰는 예시를 보여줘.",
            "표준 라이브러리 `json`으로 쉽게 처리할 수 있습니다.\n\n```python\nimport json\n\ndata = {'name': 'mina', 'score': 95}\nwith open('data.json', 'w', encoding='utf-8') as f:\n    json.dump(data, f, ensure_ascii=False, indent=2)\n\nwith open('data.json', 'r', encoding='utf-8') as f:\n    loaded = json.load(f)\nprint(loaded)\n```\n\nUTF-8과 `ensure_ascii=False`를 함께 쓰면 한글이 유지됩니다.",
        ),
        (
            "파이썬에서 requests 에러 처리 예시를 알려줘.",
            "요청 실패를 안전하게 처리하려면 `raise_for_status()`와 예외 클래스를 사용합니다.\n\n```python\nimport requests\n\ntry:\n    r = requests.get('https://example.com/api', timeout=5)\n    r.raise_for_status()\n    print(r.json())\nexcept requests.Timeout:\n    print('요청 시간 초과')\nexcept requests.RequestException as e:\n    print('요청 실패:', e)\n```\n\n네트워크 코드는 항상 타임아웃을 지정하는 것이 좋습니다.",
        ),
        (
            "파이썬에서 로깅 기본 설정 예시를 보여줘.",
            "디버깅/운영 로그는 `logging` 모듈을 쓰는 것이 표준입니다.\n\n```python\nimport logging\n\nlogging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')\nlogging.info('server started')\nlogging.warning('disk usage high')\n```\n\n`print` 대신 로그 레벨을 사용하면 운영 관찰이 훨씬 편합니다.",
        ),
    ]

    seeds_en = [
        (
            "Show a minimal Python try/except example.",
            "Use `try` for risky code and handle expected failures with `except`.\n\n```python\ntry:\n    value = int(input('number: '))\n    print(10 / value)\nexcept ValueError:\n    print('Please enter a valid integer.')\nexcept ZeroDivisionError:\n    print('Division by zero is not allowed.')\n```\n\nAdd `finally` if you need cleanup logic.",
        ),
        (
            "Show a compact Python list comprehension example.",
            "A clear example is filtering and transforming in one line.\n\n```python\nnums = [1, 2, 3, 4, 5]\nodd_squares = [n * n for n in nums if n % 2 == 1]\nprint(odd_squares)  # [1, 9, 25]\n```\n\nUse comprehension when it improves readability.",
        ),
        (
            "Show a basic JavaScript async/await example.",
            "Use `async` functions and wrap awaits with `try/catch`.\n\n```javascript\nasync function loadPost(id) {\n  try {\n    const res = await fetch(`/api/posts/${id}`);\n    if (!res.ok) throw new Error('request failed');\n    const post = await res.json();\n    console.log(post);\n  } catch (err) {\n    console.error(err.message);\n  }\n}\n```\n\nThis pattern is the default starting point for API calls.",
        ),
        (
            "Give a simple TypeScript interface example.",
            "Interfaces define object shapes at compile time.\n\n```typescript\ninterface Product {\n  id: number;\n  title: string;\n  price: number;\n}\n\nconst p: Product = { id: 1, title: 'Book', price: 12.5 };\n```\n\nThis helps catch missing fields early.",
        ),
        (
            "Show a basic SQL JOIN example.",
            "A common starter is `LEFT JOIN`.\n\n```sql\nSELECT c.id, c.name, o.id AS order_id\nFROM customers c\nLEFT JOIN orders o ON c.id = o.customer_id;\n```\n\nRows from `customers` stay in the result even when `orders` is missing.",
        ),
    ]

    prefixes_ko = [
        "핵심만",
        "실무 기준으로",
        "초보자도 이해되게",
        "짧고 정확하게",
    ]
    prefixes_en = [
        "Keep it concise.",
        "Make it beginner-friendly.",
        "Use practical style.",
    ]

    rows: list[dict] = []

    for inp, out in seeds_ko:
        rows.append(
            {
                "task_type": "korean",
                "segment_tag": "ko",
                "language": "ko",
                "input": inp,
                "output": out,
            }
        )
        for p in prefixes_ko:
            rows.append(
                {
                    "task_type": "korean",
                    "segment_tag": "ko",
                    "language": "ko",
                    "input": f"{inp} {p} 설명해줘.",
                    "output": out,
                }
            )

    for inp, out in seeds_en:
        rows.append(
            {
                "task_type": "english",
                "segment_tag": "english",
                "language": "en",
                "input": inp,
                "output": out,
            }
        )
        for p in prefixes_en:
            rows.append(
                {
                    "task_type": "english",
                    "segment_tag": "english",
                    "language": "en",
                    "input": f"{inp} {p}",
                    "output": out,
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/turbo_boost_ko_code.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = make_rows()
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "out": str(out_path),
                "rows": len(rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
