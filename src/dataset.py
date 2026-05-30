import json
import os
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


PROMPT_TEMPLATE = "<s>[INST] {instruction}\n\n{input} [/INST] {output}</s>"
PROMPT_TEMPLATE_NO_INPUT = "<s>[INST] {instruction} [/INST] {output}</s>"


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_sample(sample: dict) -> dict:
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "").strip()
    output = sample.get("output", "")

    if input_text:
        text = PROMPT_TEMPLATE.format(
            instruction=instruction, input=input_text, output=output
        )
    else:
        text = PROMPT_TEMPLATE_NO_INPUT.format(
            instruction=instruction, output=output
        )
    return {"text": text}


def build_dataset(
    data_path: str,
    tokenizer: Optional[AutoTokenizer] = None,
    test_size: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    records = load_jsonl(data_path)
    formatted = [format_sample(r) for r in records]
    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/dataset.jsonl")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    ds = build_dataset(args.data_path)
    os.makedirs(args.output_dir, exist_ok=True)
    ds.save_to_disk(args.output_dir)
    print(f"Train: {len(ds['train'])} | Test: {len(ds['test'])}")
