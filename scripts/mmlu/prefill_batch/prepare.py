# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare a small five-shot MMLU subset with the deployed model tokenizer."""

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> None:
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bf16-tokenizer", type=Path, required=True)
    parser.add_argument("--int8-tokenizer", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "encoding", repo / "vllm/tokenizers/deepseek_v4_encoding.py"
    )
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)
    tok = Tokenizer.from_file(str(args.bf16_tokenizer))
    letters = "ABCD"
    allowed = [tok.encode(c, add_special_tokens=False).ids for c in letters]
    assert all(len(ids) == 1 for ids in allowed), allowed

    def format_question(row: dict, demonstration: bool = False) -> str:
        question = (
            row["question"]
            + "\n"
            + "\n".join(
                f"{letter}. {choice}" for letter, choice in zip(letters, row["choices"])
            )
        )
        return (
            question
            + "\nAnswer:"
            + (f" {letters[row['answer']]}\n\n" if demonstration else "")
        )

    dev = json.loads((args.data_dir / "mmlu_geography_dev5.json").read_text())["rows"]
    intro = (
        "The following are multiple choice questions (with answers) about high "
        "school geography. Choose the correct answer. Respond with exactly one "
        "letter: A, B, C, or D.\n\n"
    )
    prefix = intro + "".join(format_question(r["row"], True) for r in dev)
    cases = []
    for r in json.loads((args.data_dir / "mmlu_geography_test40.json").read_text())[
        "rows"
    ]:
        content = prefix + format_question(r["row"])
        prompt = encoding.encode_messages(
            [{"role": "user", "content": content}], thinking_mode="chat"
        )
        ids = tok.encode(prompt, add_special_tokens=False).ids
        cases.append(
            {
                "case_id": f"mmlu-geography-test-{r['row_idx']:03d}",
                "row_idx": r["row_idx"],
                "prompt": prompt,
                "prompt_token_ids": ids,
                "input_tokens": len(ids),
                "gold": letters[r["row"]["answer"]],
            }
        )
    cases = sorted(cases, key=lambda c: (c["input_tokens"], c["row_idx"]))[:32]
    assert max(c["input_tokens"] for c in cases) + 1 <= 2048
    assert sum(c["input_tokens"] for c in cases) > 8192
    int8 = Tokenizer.from_file(str(args.int8_tokenizer))
    assert all(
        int8.encode(c["prompt"], add_special_tokens=False).ids == c["prompt_token_ids"]
        for c in cases
    )
    assert [int8.encode(c, add_special_tokens=False).ids for c in letters] == allowed
    (root / "cases32.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    metadata = {
        "dataset": "cais/mmlu",
        "subject": "high_school_geography",
        "selection": "32 shortest tokenized prompts from test rows 0 through 39",
        "demonstrations": "all five dev rows; no test labels in prompts",
        "shots": 5,
        "allowed_token_ids": [ids[0] for ids in allowed],
        "letters": letters,
        "count": len(cases),
        "min_tokens": min(c["input_tokens"] for c in cases),
        "max_tokens": max(c["input_tokens"] for c in cases),
        "sum_tokens": sum(c["input_tokens"] for c in cases),
        "source_url": "https://huggingface.co/datasets/cais/mmlu/tree/main/high_school_geography",
        "test_rows_total": 198,
        "bf16_int8_tokenizers_match": True,
    }
    (root / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
