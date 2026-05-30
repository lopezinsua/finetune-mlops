import argparse
import os
import sys

# Windows: TRL lee templates Jinja con bytes > 0x7F; re-lanza con UTF-8 mode si hace falta
if sys.platform == "win32" and not sys.flags.utf8_mode:
    os.execv(sys.executable, [sys.executable, "-X", "utf8"] + sys.argv)

import torch
import yaml
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.dirname(__file__))
from dataset import build_dataset


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(cfg["bnb_4bit_compute_dtype"], torch.bfloat16)
    return BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def train(config_path: str) -> None:
    cfg = load_config(config_path)

    if not torch.cuda.is_available():
        raise RuntimeError("No se detectó GPU CUDA. Verifica la instalación de PyTorch+CUDA.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB | bf16: {torch.cuda.is_bf16_supported()}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=get_bnb_config(cfg),
        device_map="auto",
    )

    dataset = build_dataset(cfg["dataset_path"], tokenizer=tokenizer)
    print(f"Dataset — train: {len(dataset['train'])} | test: {len(dataset['test'])}")

    use_bf16 = cfg.get("bf16", True)

    sft_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        bf16=use_bf16,
        fp16=not use_bf16,
        optim=cfg.get("optim", "paged_adamw_8bit"),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=cfg.get("push_to_hub", False),
        hub_model_id=cfg.get("hub_model_id") if cfg.get("push_to_hub") else None,
        report_to=cfg.get("report_to", "none"),
        max_length=cfg["max_seq_length"],
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=get_lora_config(cfg),
        args=sft_args,
    )

    print(f"\nParametros entrenables:")
    trainer.model.print_trainable_parameters()

    print("\nIniciando entrenamiento...")
    trainer.train()

    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"\nModelo guardado en: {cfg['output_dir']}")

    if cfg.get("push_to_hub"):
        trainer.push_to_hub()
        tokenizer.push_to_hub(cfg["hub_model_id"])
        print(f"Publicado en: https://huggingface.co/{cfg['hub_model_id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()
    train(args.config)
