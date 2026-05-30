import os

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./outputs")
MAX_INPUT_CHARS = 4000

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


def review_code(code: str, max_tokens: int = 256) -> str:
    if not code.strip():
        return "Introduce código para revisar."
    if len(code) > MAX_INPUT_CHARS:
        return f"El código supera el límite de {MAX_INPUT_CHARS} caracteres. Reduce el fragmento."
    prompt = f"<s>[INST] Revisa este código Python y da feedback detallado:\n\n{code} [/INST]"
    result = pipe(prompt, max_new_tokens=int(max_tokens), do_sample=True, temperature=0.7)
    generated = result[0]["generated_text"]
    return generated.split("[/INST]")[-1].strip().replace("</s>", "").strip()


demo = gr.Interface(
    fn=review_code,
    inputs=[
        gr.Textbox(label="Código Python", lines=15, placeholder="Pega tu código aquí...", max_lines=200),
        gr.Slider(64, 512, value=256, step=64, label="Tokens máximos"),
    ],
    outputs=gr.Textbox(label="Feedback del modelo", lines=10),
    title="Code Reviewer — Mistral-7B fine-tuned",
    description=(
        "Modelo fine-tuneado con QLoRA para revisar código Python en español. "
        "Detecta errores lógicos, de indentación, naming conventions y más."
    ),
    examples=[
        ["def suma(a, b):\n  return a - b", 256],
        ["for i in range(10):\nprint(i)", 128],
        ["lista = [1, 2, 3]\nprint(lista[5])", 200],
        ["def divide(a, b):\n    return a / b\n\nresultado = divide(10, 0)", 256],
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(share=False)
