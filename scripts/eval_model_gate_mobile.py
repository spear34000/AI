from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chat_slm import build_model, extract_response, generate_text


def hangul_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    h = sum(1 for c in chars if 0xAC00 <= ord(c) <= 0xD7A3)
    return float(h) / float(len(chars))


def score_code(text: str) -> float:
    t = text.lower()
    hints = [
        "```",
        "def ",
        "class ",
        "print(",
        "function ",
        "return ",
        "for ",
        "while ",
    ]
    hits = sum(1 for h in hints if h in t)
    return min(1.0, float(hits) / 2.0)


def score_ops(text: str) -> float:
    keys = ["체크", "점검", "로그", "학습", "손실", "검증", "데이터", "체크리스트"]
    hits = sum(1 for k in keys if k in text)
    return min(1.0, float(hits) / 2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.12)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise RuntimeError(f"checkpoint not found: {ckpt}")

    model, tokenizer = build_model(ckpt, device=device, use_ema=bool(args.use_ema))

    prompts = {
        "code": "Python으로 hello world 코드 보여줘",
        "ko": "사과라는 단어의 의미를 설명해줘",
        "ops": "학습 로그 점검 체크리스트 알려줘",
    }

    responses = {}
    for name, user_prompt in prompts.items():
        prompt = f"### Instruction\n{user_prompt}\n\n### Response\n"
        full = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
        )
        responses[name] = extract_response(full, prompt)

    g_code = score_code(responses["code"])
    g_ko = min(1.0, hangul_ratio(responses["ko"]) / 0.45)
    g_ops = score_ops(responses["ops"])

    payload = {
        "G_code": float(g_code),
        "G_ko": float(g_ko),
        "G_ops": float(g_ops),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
