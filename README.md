# Fine-tuning MLOps — Code Reviewer en Español

Ciclo MLOps completo: fine-tuning de Mistral-7B con QLoRA para revisar código Python en español. Dataset curado a mano, entrenamiento local en 8GB VRAM, evaluación con ROUGE/BLEU, modelo publicado en HuggingFace y demo interactiva en Spaces.

---

## Qué hace

Dado un fragmento de código Python, el modelo identifica errores, explica qué está mal y sugiere cómo arreglarlo, todo en español.

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="lopezinsua/mistral-7b-code-reviewer-es")

code = """
def divide(a, b):
    return a / b

resultado = divide(10, 0)
"""

prompt = f"<s>[INST] Revisa este código Python:\n\n{code} [/INST]"
output = pipe(prompt, max_new_tokens=256)[0]["generated_text"]
print(output.split("[/INST]")[-1].strip())
# → "El código tiene un error crítico: división por cero en la línea 2..."
```

---

## Por qué es interesante

Hay miles de proyectos que llaman a la API de GPT-4 y la envuelven en una interfaz. Esto no es eso.

Aquí se fine-tunea un modelo de 7B parámetros desde cero con recursos de consumidor (8GB VRAM), usando QLoRA para hacer posible lo que normalmente requeriría decenas de GB de memoria. Solo se entrenan 6.8 millones de parámetros (el 0.09% del modelo) gracias a los adaptadores LoRA, y aun así el modelo aprende el formato y el estilo de feedback en español.

El pipeline completo corre en local. No hace falta Colab, no hace falta créditos de cloud.

---

## Pipeline

```
01. Dataset curation   →  300 pares (instrucción, código, feedback)
02. QLoRA fine-tuning  →  Mistral-7B + LoRA r=16, 3 epochs, ~9 min
03. Evaluación         →  ROUGE-L, BLEU sobre test set
04. HuggingFace Hub    →  modelo + adapter publicados
05. Demo Gradio        →  interfaz interactiva en HF Spaces
```

---

## Resultados del entrenamiento

Entrenado en RTX 5060 Laptop GPU (8GB VRAM) con cuantización 4-bit NF4.

| Epoch | Train loss | Eval loss | Token accuracy |
|-------|-----------|-----------|----------------|
| 1 | 1.854 | 1.189 | 76.1% |
| 2 | 1.051 | 1.039 | 76.5% |
| 3 | **0.885** | **1.023** | **76.7%** |

- Parámetros entrenables: **6,815,744 / 7,254,839,296 (0.094%)**
- Tiempo total: **~9 minutos**
- Sin overfitting: eval loss se estabiliza, no sube

| Métrica | Test set (n=30) |
|---------|-----------------|
| ROUGE-1 | 0.3382 |
| ROUGE-2 | 0.1503 |
| ROUGE-L | **0.2712** |
| BLEU | **18.78** |

---

## Stack

| Componente | Tecnología | Versión |
|---|---|---|
| Modelo base | Mistral-7B-Instruct-v0.3 | — |
| Fine-tuning | QLoRA via PEFT | 0.19.1 |
| Trainer | TRL SFTTrainer | 1.4.0 |
| Cuantización | bitsandbytes 4-bit | 0.49.2 |
| Framework | PyTorch + CUDA 12.8 | 2.11.0 |
| Evaluación | ROUGE, SacreBLEU | — |
| Demo | Gradio | — |

---

## Estructura

```
├── notebooks/
│   ├── 01_dataset_curation.ipynb    # cómo se construyó el dataset
│   ├── 02_finetune_qlora.ipynb      # entrenamiento paso a paso
│   ├── 03_evaluation.ipynb          # métricas y ejemplos
│   └── 04_inference_demo.ipynb      # inferencia interactiva
├── src/
│   ├── train.py        # entrenamiento por CLI
│   ├── evaluate.py     # ROUGE-L / BLEU automatizados
│   ├── dataset.py      # carga y preprocesado
│   └── app.py          # demo Gradio
├── configs/
│   └── training_config.yaml         # todos los hiperparámetros
└── data/
    └── dataset.jsonl                # no incluido en repo
```

---

## Setup

```bash
# 1. PyTorch con CUDA 12.8 (Blackwell / Ada / Ampere)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Resto de dependencias
pip install -r requirements.txt
```

---

## Uso

```bash
# Entrenar
python src/train.py --config configs/training_config.yaml

# Evaluar
python src/evaluate.py --model_id ./outputs --data_path data/dataset.jsonl

# Demo local
python src/app.py
```

---

## Modelo

[lopezinsua/mistral-7b-code-reviewer-es](https://huggingface.co/lopezinsua/mistral-7b-code-reviewer-es) en HuggingFace Hub.

[Demo interactiva](https://huggingface.co/spaces/lopezinsua/code-reviewer-es) en HuggingFace Spaces.

---

## Autor

Lopez Insua · [GitHub](https://github.com/lopezinsua) · [LinkedIn](https://www.linkedin.com/in/lopezinsua/)
