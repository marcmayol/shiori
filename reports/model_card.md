---
license: apache-2.0
base_model: google/gemma-3-270m
library_name: transformers
pipeline_tag: text-generation
tags:
  - routing
  - model-routing
  - on-device
  - gguf
  - llama.cpp
  - ollama
language:
  - en
  - es
---

# Shiori 栞 — On-device model router (270M)

**Shiori** is a tiny (270M) policy that decides **which model to use for a task**.
Given a list of available models (rendered in the prompt) and a task, it outputs a
single JSON object:

```json
{"mode": "DIRECT | TOOL_CALL | PLAN", "model_id": "<the cheapest sufficient model in the list>"}
```

It runs **100% locally** (llama.cpp / Ollama / LM Studio), with **constrained
decoding** so the output is always a valid choice from the models you provide. It
is a full fine-tune of [`google/gemma-3-270m`](https://huggingface.co/google/gemma-3-270m).

- **Modes**: `DIRECT` (answerable in one shot), `TOOL_CALL` (needs tools), `PLAN`
  (needs multi-step planning → route to a capable model).
- **model_id**: the *minimal sufficient* model in the pool — it reads each model's
  tags, cost and context window to pick the cheapest one that can do the job.
- **Portable**: trained with randomized model names/costs, so it generalizes to
  pools of models it has never seen.

Code, training pipeline and reports: **https://github.com/marcmayol/shiori**

---

## 🇬🇧 English

### Intended use

A **local, zero-network router** in front of a set of LLMs. It *predicts* the best
model and mode; it does not run or verify anything — your surrounding code does the
execution (try → verify → escalate). Useful when you have a mix of cheap local
models and expensive capable ones and want to send each task to the smallest model
that suffices.

**Out of scope**: it only works with the fixed registry format below; it is not a
chat model and does not solve the tasks itself.

### How to use

**Ollama** (recommended):

```bash
# download shiori-270m-Q4_K_M.gguf and Modelfile from this repo, then:
ollama create shiori-router -f Modelfile
```

Send a prompt built as `SYSTEM + registry + task` and use structured output (JSON
schema) so the answer is always valid. The registry format is fixed:

```
Available models:
- fast-local | tags:code,general | ctx:8192 | cost:1 | loc:local | tools:no
- mid-local  | tags:code,reasoning | ctx:32768 | cost:3 | loc:local | tools:yes
- big-api    | tags:code,reasoning,planning | ctx:200000 | cost:20 | loc:api | tools:yes

Task:
Design and implement a distributed rate limiter with tests and a rollout plan.
```

→ `{"mode": "PLAN", "model_id": "big-api"}`

**Transformers**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("natzx94/shiori-router-270m")
model = AutoModelForCausalLM.from_pretrained("natzx94/shiori-router-270m")
# build "<start_of_turn>user\n{SYSTEM}\n\n{registry}\n\nTask:\n{task}<end_of_turn>\n<start_of_turn>model\n"
# generate, stop at <end_of_turn>, parse the JSON.
```

A reference `route(task, registry)` helper (talks to Ollama, builds the JSON
schema, applies the calibration prior, falls back to the most capable model on any
failure) lives in the GitHub repo.

### Evaluation

On a **frozen test set of never-seen model pools** (n = 400), constrained decoding:

| | exact | mode acc | model_id acc | DIRECT | TOOL_CALL | PLAN |
|---|------:|---------:|-------------:|-------:|----------:|-----:|
| default | 0.760 | 0.853 | 0.802 | 0.94 | 0.79 | 0.59 |
| with `plan_prior=0.25` | ~0.77 | **0.865** | — | 0.91 | 0.80 | 0.84 |
| with `plan_prior=0.50` | ~0.76 | 0.838 | — | 0.84 | 0.79 | **0.94** |

- **Invalid outputs**: 0.000 with constrained decoding (0.013 without).
- **Calibration knob** — `plan_prior` biases the mode decision toward `PLAN` at
  inference time. `PLAN` accuracy is largely a *calibration* issue, not a capacity
  one: a small prior recovers it **without retraining** (`0.25` is nearly free and
  even improves overall accuracy; `0.50` maximizes `PLAN` recall).

**Versus zero-shot baselines (same test set):**

| router | exact | mode acc | PLAN acc |
|--------|------:|---------:|---------:|
| **Shiori 270M (this model)** | **0.760** | **0.853** | 0.59 / **0.94** calibrated |
| GPT-4o zero-shot | 0.228 | 0.718 | 0.00 |
| Qwen 7B zero-shot | 0.153 | 0.367 | 0.27 |
| "always cheapest" heuristic | 0.212 | 0.585 | 0.00 |

A **trained 270M beats a frontier model zero-shot** here, because the task is about
learning the *minimal-sufficient* semantics and the fixed format, not raw capability.

**Economics (honest note)**: on verifiable code tasks, a plain cheapest-first
cascade is ~8% cheaper than routing at the same pass rate. The router's value is
**mode selection, latency (fewer failed attempts), and domains with no mechanical
verifier**, not cost savings on code.

### Latency (llama.cpp, ~320-token registry + 24-token answer)

| quant | size | CPU (8 threads) | GPU |
|-------|-----:|----------------:|----:|
| Q8_0 | 286 MB | 512 ms | 35 ms |
| Q4_K_M | 249 MB | 526 ms | 35 ms |

CPU-only is fully functional (< 1 s per decision).

### Training data

- **Sufficiency (code)**: HumanEval+ / MBPP+ (evalplus) — labels come from actually
  running models and checking their tests.
- **Mode**: Hermes function-calling v1 (Apache-2.0) and Dolly-15k (CC-BY-SA-3.0).

Datasets are not redistributed; only the learned behavior ships. Model: Apache-2.0.

### Limitations

- Raw `PLAN` recall is 0.59 (mitigate with `plan_prior`, see above).
- Hard dependency on the **fixed registry format** — it will not work outside it.
- Domains covered: code, tool-calling, general chat (English). Others are not
  represented.
- It **predicts, it does not verify** — real sufficiency depends on your wrapper.
- This is the **smallest model in a size ladder** (270M); a larger step could raise
  the `PLAN` ceiling further.

---

## 🇪🇸 Español

### Para qué sirve

Un **router local, sin red**, delante de un conjunto de LLMs. *Predice* el mejor
modelo y modo; no ejecuta ni verifica — eso lo hace el código que lo envuelve
(probar → verificar → escalar). Útil cuando tienes una mezcla de modelos locales
baratos y modelos capaces caros y quieres mandar cada tarea al **modelo más pequeño
que basta**.

**Fuera de alcance**: solo funciona con el formato de registro fijo de abajo; no es
un modelo de chat ni resuelve las tareas.

### Cómo usarlo

**Ollama** (recomendado):

```bash
# descarga shiori-270m-Q4_K_M.gguf y Modelfile de este repo, luego:
ollama create shiori-router -f Modelfile
```

El prompt es `SYSTEM + registro + tarea`, con structured output (JSON schema) para
que la salida sea siempre válida. El formato del registro es fijo:

```
Available models:
- fast-local | tags:code,general | ctx:8192 | cost:1 | loc:local | tools:no
- mid-local  | tags:code,reasoning | ctx:32768 | cost:3 | loc:local | tools:yes
- big-api    | tags:code,reasoning,planning | ctx:200000 | cost:20 | loc:api | tools:yes

Task:
Diseña e implementa un rate limiter distribuido con tests y plan de despliegue.
```

→ `{"mode": "PLAN", "model_id": "big-api"}`

### Evaluación

Sobre un **test congelado con pools de modelos nunca vistos** (n = 400), con
decoding constreñido:

| | exact | acc. modo | acc. model_id | DIRECT | TOOL_CALL | PLAN |
|---|------:|----------:|--------------:|-------:|----------:|-----:|
| por defecto | 0.760 | 0.853 | 0.802 | 0.94 | 0.79 | 0.59 |
| `plan_prior=0.25` | ~0.77 | **0.865** | — | 0.91 | 0.80 | 0.84 |
| `plan_prior=0.50` | ~0.76 | 0.838 | — | 0.84 | 0.79 | **0.94** |

- **Salidas inválidas**: 0.000 con constraint (0.013 sin él).
- **Ajuste `plan_prior`**: sesga la decisión de modo hacia `PLAN` en inferencia. La
  precisión de `PLAN` es sobre todo un problema de **calibración**, no de capacidad:
  un prior pequeño la recupera **sin reentrenar** (`0.25` casi gratis y mejora la
  exactitud global; `0.50` maximiza el recall de `PLAN`).

**Frente a baselines zero-shot (mismo test):** un **270M entrenado supera a GPT-4o
zero-shot** (0.760 vs 0.228, PLAN 0.94 vs 0.00) porque la tarea va de aprender la
semántica de "mínimo suficiente" y el formato fijo, no de capacidad bruta.

**Nota económica honesta**: en tareas de código verificable, una cascada pura
(empezar por el más barato) es ~8% más barata que el router a igual pass-rate. El
valor del router está en **decidir el modo, la latencia y los dominios sin
verificador mecánico**, no en el ahorro de coste en código.

### Latencia (llama.cpp)

| cuant. | tamaño | CPU (8 hilos) | GPU |
|--------|-------:|--------------:|----:|
| Q8_0 | 286 MB | 512 ms | 35 ms |
| Q4_K_M | 249 MB | 526 ms | 35 ms |

CPU-only es funcional (< 1 s por decisión).

### Datos de entrenamiento

- **Suficiencia (código)**: HumanEval+ / MBPP+ (evalplus) — etiquetas por ejecución
  real de modelos y verificación de sus tests.
- **Modo**: Hermes function-calling v1 (Apache-2.0) y Dolly-15k (CC-BY-SA-3.0).

No se redistribuyen los datasets; solo viaja el comportamiento aprendido. Apache-2.0.

### Límites conocidos

- Recall de `PLAN` crudo 0.59 (mitigable con `plan_prior`).
- Depende del **formato de registro fijo**; fuera de él no funciona.
- Dominios: código, tool-calling y chat general (inglés). Otros no están cubiertos.
- **Predice, no verifica** — la suficiencia real depende de tu código.
- Es el **modelo más pequeño de una escalera**; un peldaño mayor podría subir el
  techo de `PLAN`.

---

### Citation

```
@software{shiori_router_2026,
  title  = {Shiori: an on-device model-routing policy},
  author = {Mayol, Marc},
  year   = {2026},
  url    = {https://github.com/marcmayol/shiori}
}
```
