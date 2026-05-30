---
language:
- es
- code
license: apache-2.0
base_model: mistralai/Mistral-7B-Instruct-v0.3
tags:
- code-review
- qlora
- peft
- python
- spanish
datasets:
- lopezinsua/code-review-es
metrics:
- rouge
- bleu
---

# mistral-7b-code-reviewer-es

Mistral-7B-Instruct-v0.3 fine-tuneado con QLoRA para revisar código Python en español.

## Uso

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="lopezinsua/mistral-7b-code-reviewer-es")

code = """
def suma(a, b):
  return a - b
"""

prompt = f"<s>[INST] Revisa este código Python:\n\n{code} [/INST]"
result = pipe(prompt, max_new_tokens=256)[0]["generated_text"]
print(result.split("[/INST]")[-1].strip())
```

## Entrenamiento

- **Método:** QLoRA (LoRA rank 16, alpha 32)
- **Cuantización:** 4-bit NF4 + double quant
- **Epochs:** 3 | **Steps:** 51
- **Learning rate:** 2e-4 (cosine scheduler, warmup 3%)
- **Batch size efectivo:** 2 × 8 (gradient accumulation)
- **Max seq length:** 512
- **Optimizer:** paged_adamw_8bit
- **Compute:** NVIDIA RTX 5060 Laptop GPU (8GB VRAM) — ~9 min
- **Parámetros entrenables:** 6.8M / 7.254B (0.094%)

| Época | Train loss | Eval loss | Token acc |
|-------|-----------|-----------|-----------|
| 1 | 1.854 | 1.189 | 76.1% |
| 2 | 1.051 | 1.039 | 76.5% |
| 3 | 0.885 | 1.023 | 76.7% |

## Dataset

Dataset curado manualmente de 300 pares (instrucción, código Python, feedback en español). Split: 270 train / 30 test.

Tipos de errores cubiertos:
- Errores lógicos
- Errores de indentación
- Naming conventions (PEP8)
- Division by zero / index out of range
- Errores de tipo
- Bucles y condicionales incorrectos

## Métricas

| Métrica | Train | Test |
|---|---|---|
| ROUGE-1 | — | 0.3382 |
| ROUGE-2 | — | 0.1503 |
| ROUGE-L | — | **0.2712** |
| BLEU | — | **18.78** |

## Limitaciones

- Entrenado con ~400 ejemplos; puede fallar en patrones no vistos
- Optimizado para código Python simple/intermedio
- Feedback en español únicamente
