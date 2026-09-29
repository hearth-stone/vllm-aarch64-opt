#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the vLLM endpoints required by lm-eval and save their responses."""

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(url: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=body)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url}: HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{url}: {exc.reason}") from exc


def save_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8004")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    models = request_json(f"{base_url}/v1/models")
    if args.model not in {item["id"] for item in models["data"]}:
        parser.error(f"model '{args.model}' absent from /v1/models")
    tokenizer = request_json(f"{base_url}/tokenizer_info")

    completion = request_json(
        f"{base_url}/v1/completions",
        {
            "model": args.model,
            "prompt": "Question: 1 + 1 = ?\nA. 1\nB. 2\nC. 3\nD. 4\nAnswer: B",
            "max_tokens": 1,
            "temperature": 0,
            "echo": True,
            "logprobs": 1,
        },
    )
    logprobs = completion["choices"][0]["logprobs"]["token_logprobs"]
    if len(logprobs) <= 2 or not any(value is not None for value in logprobs[:-1]):
        parser.error("/v1/completions did not return prompt token logprobs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "models.json", models)
    save_json(args.output_dir / "tokenizer-info.json", tokenizer)
    save_json(args.output_dir / "completion-smoke.json", completion)
    print(f"Serving: {args.model}; prompt logprobs: {len(logprobs)} tokens")
    print(f"Responses saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
