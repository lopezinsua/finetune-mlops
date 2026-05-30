import os

import gradio as gr
import torch
from transformers import AutoTokenizer, pipeline


MODEL_ID = os.getenv("MODEL_ID", "lopezinsua/mistral-7b-code-reviewer-es")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)


def review_code(code: str, max_tokens: int = 256) -> str:
    if not code.strip():
        return "Introduce código para revisar."
    prompt = f"<s>[INST] Revisa este código Python y da feedback detallado:\n\n{code} [/INST]"
    result = pipe(prompt, max_new_tokens=int(max_tokens), do_sample=True, temperature=0.7)
    generated = result[0]["generated_text"]
    return generated.split("[/INST]")[-1].strip().replace("</s>", "").strip()


demo = gr.Interface(
    fn=review_code,
    inputs=[
        gr.Textbox(label="Código Python", lines=15, placeholder="Pega tu código aquí..."),
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
