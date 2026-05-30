import argparse
import json
import sys
import os
if sys.platform == "win32" and not sys.flags.utf8_mode:
    os.execv(sys.executable, [sys.executable, "-X", "utf8"] + sys.argv)

from pathlib import Path

import evaluate as hf_evaluate
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline

sys.path.insert(0, os.path.dirname(__file__))
from dataset import build_dataset


rouge_metric = hf_evaluate.load("rouge")
bleu_metric = hf_evaluate.load("sacrebleu")


def build_prompt(instruction: str, input_text: str) -> str:
    if input_text.strip():
        return f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
    return f"<s>[INST] {instruction} [/INST]"


def extract_response(generated: str) -> str:
    return generated.split("[/INST]")[-1].strip().replace("</s>", "").strip()


def generate_predictions(pipe, test_samples: list[dict], max_new_tokens: int = 256) -> tuple[list, list]:
    predictions, references = [], []

    for i, sample in enumerate(test_samples):
        text = sample["text"]
        parts = text.split("[/INST]")
        prompt = parts[0] + "[/INST]"
        reference = parts[1].strip().replace("</s>", "").strip() if len(parts) > 1 else ""

        result = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        pred = extract_response(result[0]["generated_text"])

        predictions.append(pred)
        references.append(reference)
        print(f"[{i+1}/{len(test_samples)}] generado", flush=True)

    return predictions, references


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
    bleu_score = bleu_metric.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )
    return {
        "rouge1": round(rouge_scores["rouge1"], 4),
        "rouge2": round(rouge_scores["rouge2"], 4),
        "rougeL": round(rouge_scores["rougeL"], 4),
        "bleu": round(bleu_score["score"], 4),
        "num_samples": len(predictions),
    }


def evaluate_model(model_id: str, data_path: str, output_path: str = "outputs/eval_results.json", max_samples: int = 30) -> dict:
    print(f"Cargando modelo desde: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"quantization_config": bnb, "device_map": "auto"},
    )
    print("Modelo cargado. Generando predicciones...", flush=True)

    dataset = build_dataset(data_path)
    test_samples = list(dataset["test"])[:max_samples]

    predictions, references = generate_predictions(pipe, test_samples)
    metrics = compute_metrics(predictions, references)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--data_path", default="data/dataset.jsonl")
    parser.add_argument("--output", default="outputs/eval_results.json")
    parser.add_argument("--max_samples", type=int, default=30)
    args = parser.parse_args()

    evaluate_model(args.model_id, args.data_path, args.output, args.max_samples)
