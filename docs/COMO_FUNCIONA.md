# Cómo funciona este proyecto

Esta guía explica, sin asumir conocimientos previos de ML, qué hace cada parte del código y por qué está estructurado así.

---

## La idea en una frase

Tomamos un modelo de lenguaje ya entrenado (Mistral-7B) y lo "afinamos" para que, dado un fragmento de código Python, responda con un análisis de errores en español. Para eso necesitamos: un dataset de ejemplos, una técnica de entrenamiento eficiente, y un pipeline que lo lleve desde el entrenamiento hasta una demo usable.

---

## El problema de entrenar un modelo grande

Mistral-7B tiene 7.000 millones de parámetros. Entrenar ese modelo completo requeriría decenas de gigabytes de memoria de GPU y días de cómputo. Hay dos técnicas que resuelven esto:

**QLoRA** (Quantized Low-Rank Adaptation):
- En lugar de modificar los 7B parámetros, solo se añaden pequeñas matrices de "adaptación" (LoRA) encima del modelo original. Solo esas matrices se entrenan.
- El modelo base se carga en formato comprimido de 4 bits (en lugar de los 32 bits habituales), lo que reduce la memoria a ~4 GB.
- Resultado: en vez de entrenar 7.000M parámetros, solo se entrenan **6.8M** (el 0.09%), y todo cabe en una GPU de 8GB.

---

## Los archivos, explicados

### `data/dataset.jsonl`
El dataset. Cada línea es un JSON con tres campos:
```json
{"instruction": "Revisa este código Python", "input": "def suma(a,b): return a-b", "output": "Error: la función resta en lugar de sumar..."}
```
300 ejemplos escritos a mano cubriendo errores típicos de Python.

### `src/dataset.py`
Lee el `dataset.jsonl` y lo transforma al formato que espera el trainer. Cada ejemplo se convierte en un string con la estructura `[INST] ... [/INST] ...` que es el formato de instrucción de Mistral. Divide automáticamente en 90% train / 10% test.

### `configs/training_config.yaml`
Todos los hiperparámetros del entrenamiento en un solo archivo. No hay números mágicos dispersos por el código: batch size, learning rate, configuración de LoRA, todo está aquí. Para reproducir el experimento con distintos parámetros solo hay que cambiar este archivo.

### `src/train.py`
El script principal de entrenamiento. Lo que hace, en orden:
1. Lee el `training_config.yaml`
2. Verifica que hay GPU disponible
3. Carga el tokenizer (convierte texto a números que el modelo entiende)
4. Carga Mistral-7B comprimido a 4 bits en VRAM
5. Construye el dataset
6. Configura y lanza el trainer (TRL SFTTrainer)
7. Guarda el modelo en `outputs/`

El resultado en `outputs/` no es el modelo completo (14 GB) sino solo los adaptadores LoRA entrenados (~26 MB). Para hacer inferencia se cargan el modelo base + los adaptadores.

### `src/run_eval.py`
Mide qué tan buenas son las respuestas generadas comparándolas con las respuestas de referencia del test set. Usa dos métricas:
- **ROUGE-L**: mide solapamiento de secuencias entre predicción y referencia (0 = nada en común, 1 = idéntico).
- **BLEU**: métrica clásica de traducción automática, penaliza predicciones demasiado cortas.

Los resultados se guardan en `outputs/eval_results.json`.

### `src/app.py`
Una interfaz web sencilla construida con Gradio. El usuario pega código Python, el modelo genera el feedback, y la respuesta aparece en pantalla. Carga el modelo con la misma cuantización 4-bit del entrenamiento para que funcione en la misma GPU.

### `notebooks/`
Los cuatro notebooks documentan el proceso paso a paso con celdas ejecutables. Son útiles para entender cada fase antes de usar los scripts.

---

## El flujo de datos, de principio a fin

```
dataset.jsonl
    │
    ▼
dataset.py          ← formatea ejemplos como [INST]/[/INST]
    │
    ▼
train.py            ← carga modelo 4-bit + entrena LoRA adapters
    │
    ▼
outputs/            ← adapter_model.safetensors (~26 MB)
    │
    ├──▶ run_eval.py   ← genera predicciones → ROUGE-L / BLEU
    │
    └──▶ app.py        ← demo Gradio: usuario pega código → feedback
```

---

## Por qué ROUGE-L y BLEU

Estas métricas comparan texto generado contra texto de referencia automáticamente. Sus limitaciones son conocidas: dos frases pueden significar lo mismo con palabras distintas y sacar puntuación baja. Para una evaluación más robusta se necesitaría evaluación humana. Los números que arroja este pipeline (ROUGE-L: 0.27, BLEU: 18.78) son una señal de que el modelo aprendió el formato y el tipo de respuesta esperado, no una medida de calidad absoluta.

---

## Cómo usar el modelo sin reentrenar

**Opción 1 — Sin GPU (más fácil):**
La [demo en HuggingFace Spaces](https://huggingface.co/spaces/lopezinsua/code-reviewer-es) corre en los servidores de HuggingFace. Solo hace falta un navegador.

**Opción 2 — Con GPU local (≥6GB VRAM):**

```python
from transformers import pipeline, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
pipe = pipeline(
    "text-generation",
    model="lopezinsua/mistral-7b-code-reviewer-es",
    model_kwargs={"quantization_config": bnb, "device_map": "auto"},
)

code = "def suma(a, b):\n  return a - b"
out = pipe(f"<s>[INST] Revisa este código Python:\n\n{code} [/INST]", max_new_tokens=256)
print(out[0]["generated_text"].split("[/INST]")[-1].strip())
```

La primera vez descarga ~14 GB (el modelo base desde HuggingFace) y ~26 MB de adaptadores. En ejecuciones siguientes lo carga desde la caché local.

No hace falta ninguna API key ni cuenta de pago. El modelo es público.

---

## Para reproducir el entrenamiento

```bash
# 1. Instalar dependencias (ver README)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. Colocar el dataset en data/dataset.jsonl

# 3. Entrenar
python src/train.py --config configs/training_config.yaml

# 4. Evaluar
python src/run_eval.py --model_id ./outputs

# 5. Demo local
python src/app.py
```
