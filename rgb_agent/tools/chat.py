from __future__ import annotations

import argparse
import sys
from typing import Any

import requests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the local analyzer model.")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--system", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def _resolve_model(base_url: str, model: str, timeout: float) -> str:
    requested = model.strip()
    if requested and requested.lower() != "auto":
        return requested

    response = requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    models = payload.get("data", [])
    if not models:
        raise requests.RequestException("server returned no models")
    resolved = str(models[0]["id"])
    print(f"using model: {resolved}", flush=True)
    return resolved


def _request_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        },
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text.strip()
        message = f"{exc}"
        if detail:
            message += f" | response: {detail}"
        raise requests.RequestException(message) from exc
    payload: dict[str, Any] = response.json()
    return str(payload["choices"][0]["message"]["content"])


def main() -> int:
    args = _parse_args()
    try:
        model = _resolve_model(args.base_url, args.model, args.timeout)
    except requests.RequestException as exc:
        print(f"failed to resolve model: {exc}", file=sys.stderr)
        return 1

    messages: list[dict[str, str]] = []
    if args.system.strip():
        messages.append({"role": "system", "content": args.system.strip()})

    if args.prompt.strip():
        messages.append({"role": "user", "content": args.prompt.strip()})
        try:
            print(
                _request_chat(
                    base_url=args.base_url,
                    model=model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    timeout=args.timeout,
                )
            )
            return 0
        except requests.RequestException as exc:
            print(f"request failed: {exc}", file=sys.stderr)
            return 1

    print("Interactive chat. Ctrl-D or /exit to quit. /reset clears history.", flush=True)
    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print()
            return 0

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return 0
        if user_text == "/reset":
            messages = [msg for msg in messages if msg["role"] == "system"]
            print("history cleared", flush=True)
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            reply = _request_chat(
                base_url=args.base_url,
                model=model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
            )
        except requests.RequestException as exc:
            print(f"request failed: {exc}", file=sys.stderr)
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"model> {reply}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
